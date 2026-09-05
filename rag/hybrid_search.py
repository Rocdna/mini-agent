"""
hybrid_search — 混合检索（BM25 稀疏 + 稠密向量 → RRF 融合）

把完整的检索拼起来，回答"用户问一句，哪几个 chunk 最该被召回"：
  1. 同一个 42 chunk，BM25 建索引（稀疏），向量缓存载入（稠密）
  2. 对一条 query：BM25 给一套名次、稠密给一套名次
  3. rrf() 融合两份名次，得出最终 top-k

对比意义（对照 eval_bm25.py / dense_search.py 单边结果）：
  - 纯 BM25 在精确词上强（q1/q2/q8），但语义改写抓瞎（q3/q4 miss）
  - 纯稠密在语义上强（q3/q4 命中），但精确词场景会偏（q6 top-1 错到 refund）
  - 混合用 RRF 把两边名次并起来，理论上让"一边强 + 另一边也跟着排前"的
    chunk 胜出，补上各自盲区。

用法：
  python -m partical.rag.eval_hybrid        # 跑 10 条 query 的评估
"""

import math

from partical.rag.embedded import embed_texts
from partical.rag.retriever import build_index, tokenize
from partical.rag.rrf import rrf
from partical.rag.vector_store import build_or_load_index

# RRF 常数（论文常用 60），也可在调用处覆盖
RRF_K = 60


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：只看向量方向。与 dense_search.py 保持一致。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def _normalize(chunk: dict) -> dict:
    """把向量缓存里的扁平结构归一成 retriever 认识的嵌套 meta 结构。

    vector_store 落盘缓存用扁平 {"index","doc","text","vector"}（省一层），
    而 retriever.BM25Index.search / 本模块读 doc 都走 chunk["meta"]["doc"]。
    这里统一补上 meta 层，让两套形状一致、doc 不再取到兜底 "?"。
    """
    if "meta" in chunk and "doc" in chunk.get("meta", {}):
        return chunk  # 已是嵌套结构，原样返回
    if "doc" in chunk:
        return {
            "index": chunk.get("index"),
            "text": chunk.get("text", ""),
            "vector": chunk.get("vector"),
            "meta": {"doc": chunk["doc"]},
        }
    return chunk


class HybridIndex:
    """同一批 chunk 上同时建的 BM25 索引 + 稠密向量库。

    只建一次，之后每条 query 都用同一套数据打两套名次，保证可比。
    """

    def __init__(self, chunks: list[dict]):
        # 统一：稀疏(BM25 的 doc_idx)和稠密都用"列表位置"充当 chunk 的唯一身份。
        # 不能拿缓存里的 index 字段当身份——vector_store 存的 index 是"文档内局部编号"
        # (每篇从 0 重新数)，所有文档的 chunk 号都撞在 0~5；用它做 dict key 会互相覆盖。
        # 列表位置 0..41 是全局唯一的，BM25 和稠密在同一份列表上排位，才能对齐融合。
        self.chunks = [_normalize(c) for c in chunks]
        self.bm25 = build_index(self.chunks)
        # 稠密侧：与 self.chunks 平行的位置→向量列表（同下标即同一 chunk）。
        self.vecs = [c["vector"] for c in self.chunks]

    def _bm25_ranking(self, query: str) -> dict[int, int]:
        """BM25 给全库 chunk 打名次。返回 {位置: rank}，rank 从 1 起。

        search() 返回里的 index 即 build_index 的列表位置（全局唯一）。用 top_k=len
        取回全部 chunk；分数为 0（完全无命中词）的不在结果里 → 不进 dict，即不贡献融合分。
        """
        results = self.bm25.search(query, top_k=len(self.chunks))
        return {r["index"]: rank for rank, r in enumerate(results, 1)}

    def _dense_ranking(self, qvec: list[float]) -> dict[int, int]:
        """稠密给全库 chunk 按余弦排名次。返回 {位置: rank}，rank 从 1 起。

        和 BM25 一样对同一批 chunk 排序，只是排序键从"词频分"换成"余弦相似度"。
        用 enumerate 遍历位置列表，保证稠密名次的 key = 列表位置 = BM25 的 key，两者对齐。
        """
        scored = sorted(
            enumerate(self.vecs),
            key=lambda pv: cosine(qvec, pv[1]),
            reverse=True,
        )
        return {pos: rank for rank, (pos, _v) in enumerate(scored, 1)}

    async def search(self, query: str, top_k: int = 3, k: int | float = RRF_K) -> list[dict]:
        """对一条 query 做混合检索：BM25 名次 + 稠密名次 → RRF 融合。

        返回 [{"index", "score", "doc", "chunk_text"}]，按融合分降序，取 top_k。
        score 是 RRF 融合分（0~2 的范围，越大越靠前）。
        """
        # 1) 反面 query 只嵌入一次（唯一一次外部 API 调用），供稠密侧用
        qvec = (await embed_texts([query]))[0]

        # 2) 两份名次
        bm25_rank = self._bm25_ranking(query)
        dense_rank = self._dense_ranking(qvec)

        # 3) RRF 融合 → top_k
        fused = rrf([bm25_rank, dense_rank], k=k)[:top_k]

        results = []
        for idx, score in fused:
            chunk = self.chunks[idx]
            results.append({
                "index": idx,
                "score": round(score, 4),
                "doc": chunk.get("meta", {}).get("doc", "?"),
                "chunk_text": chunk.get("text", ""),
            })
        return results


async def build() -> HybridIndex:
    """一次性构建混合索引：载入（或首次生成）稠密向量缓存 + 建 BM25。

    稠密侧缓存命中则不调 API；miss 才嵌一遍 42 chunk（一次批量调用）。
    """
    chunks = await build_or_load_index()
    idx = HybridIndex(chunks)
    return idx