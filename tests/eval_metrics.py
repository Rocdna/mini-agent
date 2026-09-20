"""
eval_metrics — 用 recall@3 和 MRR 三列对比 BM25 / 稠密 / 混合(同一批 chunk)

背景：三个检索器单独跑时都只报 top-3 命中率（recall@3），三份都 10/10 就饱和了，
看不出"混合比稠密好在哪"。MRR 是更细的尺子——它按"正确答案排第几"计分，
第 1 得 1.0、第 2 得 0.5、第 3 得约 0.33。如果混合能"一把就排对"(MRR 高)，
而稠密常排到第 3 才出现，那个差异就是混合的真实价值。

指标定义见评测脚本头注释，三个指标的直觉：
  recall@k  : 前 k 里有没有对？（该找的找着了吗）
  MRR       : 对的排第几？越靠前分越高（排够靠前吗）
  nDCG      : 排序整体多好？（需分级相关度，当前 query 只有对/错，不适用，跳过）

用法： python -m partical.tests.eval_metrics
       （稠密缓存命中则只嵌 1 次 10 条 query）
"""

import asyncio
import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path(__file__).resolve().parent.parent / "rag" / "data"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def recall_at_k(top_docs: list[str], target: str, k: int) -> int:
    """前 k 里有没有正确答案：有=1，没有=0。等价于 hit rate。"""
    return int(target in top_docs[:k])


def mrr(top_docs: list[str], target: str) -> float:
    """第一个正确答案排名的倒数；top_k 内没排到=0。"""
    for i, d in enumerate(top_docs, start=1):
        if d == target:
            return 1.0 / i
    return 0.0


async def main():
    from partical.rag.embedded import embed_texts
    from partical.rag.retriever import build_index
    from partical.rag.vector_store import build_or_load_index

    top_k = 3
    queries = json.loads((DATA / "test_queries.json").read_text(encoding="utf-8"))
    query_texts = [q["query"] for q in queries]

    # 同一批 chunk（稠密缓存命中的那份），只建/载一次
    from partical.rag.hybrid_search import _normalize
    raw = await build_or_load_index()
    chunks = [_normalize(c) for c in raw]   # 统一成嵌套 meta 结构，三处一致取 doc
    bm25 = build_index(chunks)
    vecs = [c["vector"] for c in chunks]

    # 预嵌 10 条 query（一次 API），三个检索器复用，不重复调
    print(f"[eval_metrics] 嵌入 {len(query_texts)} 条 query ...")
    qvecs = await embed_texts(query_texts)

    # 集中三个检索器的 top-3 doc 序列（同一批 chunk、同一批 qvec）
    rows = []  # 每条 query 一个 dict
    for q, qv in zip(queries, qvecs):
        target = q["target"]

        # 1) 纯 BM25：按位置打名次取 top_k
        bm_rank = bm25.search(q["query"], top_k=len(chunks))
        bm_top = [chunks[r["index"]]["meta"]["doc"] for r in bm_rank[:top_k]]

        # 2) 纯稠密：按余弦打名次取 top_k
        dense_order = sorted(range(len(vecs)), key=lambda i: cosine(qv, vecs[i]), reverse=True)
        den_top = [chunks[i]["meta"]["doc"] for i in dense_order[:top_k]]

        # 3) 混合：RRF 融合两份名次取 top_k
        bm_rank_map = {r["index"]: rank for rank, r in enumerate(bm_rank, 1)}
        den_rank_map = {pos: rank for rank, pos in enumerate(dense_order, 1)}
        from partical.rag.rrf import rrf
        fused = [idx for idx, _score in rrf([bm_rank_map, den_rank_map])[:top_k]]
        hyp_top = [chunks[idx]["meta"]["doc"] for idx in fused]

        rows.append({
            "id": q["id"], "q": q["query"], "target": target,
            "bm": bm_top, "den": den_top, "hyp": hyp_top,
        })

    # 汇总三套指标
    def summarize(get_docs):
        r3 = sum(recall_at_k(get_docs(r), r["target"], top_k) for r in rows)
        mrrs = [mrr(get_docs(r), r["target"]) for r in rows]
        return r3, sum(mrrs) / len(rows)

    bm_r3, bm_mrr = summarize(lambda r: r["bm"])
    den_r3, den_mrr = summarize(lambda r: r["den"])
    hyp_r3, hyp_mrr = summarize(lambda r: r["hyp"])

    # 打印逐条对比表
    print(f"\n{'='*92}")
    print(f"{'query':<4} {'目标':<24} {'BM25_top3':<34} {'稠密_top3':<34} {'混合_top3':<34}")
    for r in rows:
        print(f"{r['id']:<4} {r['target']:<24} {str(r['bm']):<34} {str(r['den']):<34} {str(r['hyp']):<34}")
    print(f"{'='*92}")

    header = f"{'检索器':<8} {'recall@3':>10} {'avg MRR':>10}"
    print(header)
    print(f"{'BM25':<8} {(f'{bm_r3}/{len(rows)} = {bm_r3/len(rows)*100:.0f}%'):>10} {bm_mrr:>10.3f}")
    print(f"{'稠密':<8} {(f'{den_r3}/{len(rows)} = {den_r3/len(rows)*100:.0f}%'):>10} {den_mrr:>10.3f}")
    print(f"{'混合RRF':<8} {(f'{hyp_r3}/{len(rows)} = {hyp_r3/len(rows)*100:.0f}%'):>10} {hyp_mrr:>10.3f}")
    print(f"\n解读：recall@3 看'找没找着'，三份都高就是饱和；MRR 看'排得够不够靠前'，")
    print(f"      谁 MRR 高，谁更常一把就把正确答案排到第 1 —— 这才是混合想赢的战场。")


if __name__ == "__main__":
    asyncio.run(main())