"""
eval_hybrid — 混合检索（RRF）评估，与 eval_bm25.py / dense_search.py 并排看

对同一批 10 条 query 跑混合检索，对照 target 判定 top-3 命中和 top-1 干净度。
重点看两类曾被单边打漏的 query 是否能被融合救回：
  - q3/q4（语义 miss）：纯 BM25 打错，靠稠密补
  - q6（精确词偏）：纯稠密 top-1 偏到 refund，靠 BM25 补

用法： python -m partical.tests.eval_hybrid
       （稠密缓存命中则只调 1 次 embed 10 条 query）
"""

import asyncio
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = Path(__file__).resolve().parent.parent / "rag" / "data"


async def main():
    from partical.rag.hybrid_search import build

    top_k = 3
    hybrid = await build()
    queries = json.loads((DATA / "test_queries.json").read_text(encoding="utf-8"))

    print(f"\n{'='*64}")
    hits = 0
    for q in queries:
        res = await hybrid.search(q["query"], top_k=top_k)
        top_docs = [r["doc"] for r in res]
        hit = q["target"] in top_docs
        hits += int(hit)
        mark = "✓命中" if hit else "✗未命中"

        print(f"[{mark}] {q['id']} 「{q['query']}」")
        print(f"      目标: {q['target']}")
        print(f"      top-{top_k}: {top_docs}")
        if res:
            print(f"      融合最高分: {res[0]['score']:.4f} ({res[0]['doc']})")
        print(f"      说明: {q['note']}\n")

    total = len(queries)
    print(f"{'='*64}")
    print(f"命中率: {hits}/{total} = {hits/total*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())