"""
dense_search — 纯稠密检索 + 评估（与 eval_bm25.py 镜像对比）

流程：
  1. 加载向量存储（缓存命中则跳过嵌入，否则首次调方舟 API 对 42 chunk 嵌入落盘）
  2. 对 10 条测试 query 批量调一次 API 嵌入
  3. 每条 query 向量与 42 个 chunk 向量算余弦相似度 → top-k
  4. 对照 target 判定命中，打印结果

对比意义：稠密检索捕捉语义（q3/q4 这类 BM25 抓瞎的查询该命中），
但精确专名/字面查询可能被语义稀释。和 eval_bm25.py 的输出并列看。

用法： python -m partical.rag.dense_search
       （首次跑会嵌入 42 chunk + 10 query，共调约 2 次 API）
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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：只看方向，不看模长。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


async def main():
    from partical.rag.vector_store import build_or_load_index
    from partical.rag.embedded import embed_texts

    top_k = 3
    chunks = await build_or_load_index()

    queries = json.loads((DATA / "test_queries.json").read_text(encoding="utf-8"))
    query_texts = [q["query"] for q in queries]

    # 批量嵌所有 query（一次 API 调用）
    qvecs = await embed_texts(query_texts)

    print(f"\n{'='*64}")
    hits = 0
    for q, qv in zip(queries, qvecs):
        # 算该 query 与所有 chunk 的余弦，取 top_k
        scored = sorted(
            ((cosine(qv, c["vector"]), c) for c in chunks),
            key=lambda t: t[0], reverse=True,
        )[:top_k]
        top_docs = [c["doc"] for _, c in scored]
        hit = q["target"] in top_docs
        hits += int(hit)
        mark = "✓命中" if hit else "✗未命中"

        print(f"[{mark}] {q['id']} 「{q['query']}」")
        print(f"      目标: {q['target']}")
        print(f"      top-{top_k}: {top_docs}")
        if scored:
            print(f"      最高分: {scored[0][0]:.3f} ({scored[0][1]['doc']})")
        print(f"      说明: {q['note']}\n")

    total = len(queries)
    print(f"{'='*64}")
    print(f"命中率: {hits}/{total} = {hits/total*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())