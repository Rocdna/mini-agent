"""
eval_bm25 — 用 test_queries.json 评估 BM25 检索质量

流程：读 data/*.md → 分块(recursive) → 建倒排索引 → 对每条查询 top-k 检索 →
对照 target 判定命中，并打印命中的 chunk 来源。

预期（你要观察的核心对比）：
  - q1/q2/q6/q7/q9 等"字面关键字"查询 → BM25 应命中（稀疏检索的强项）
  - q3/q4/q8/q10 等"语义近义、字面不重叠"查询 → BM25 大概率 miss（慢稀疏的死穴）

纯本地，零 API。用法： python -m partical.rag.eval_bm25
"""

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

from partical.rag.chunker import chunk_text
from partical.rag.retriever import build_index


def main():
    top_k = 3

    # 1) 读全部文档，分块
    chunks = []
    for f in sorted(DATA.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        chunks += chunk_text(text, method="recursive", doc=f.name)
    print(f"共 {len(chunks)} 个 chunk")

    # 2) 建索引
    index = build_index(chunks)

    # 3) 读查询
    queries = json.loads((DATA / "test_queries.json").read_text(encoding="utf-8"))

    print(f"\n{'='*60}\n")
    hits = 0
    for q in queries:
        res = index.search(q["query"], top_k=top_k)
        # top-k 的文档来源
        docs = [r["doc"] for r in res]
        hit = q["target"] in docs
        hits += int(hit)
        mark = "✓命中" if hit else "✗未命中"
        print(f"[{mark}] {q['id']} 「{q['query']}」")
        print(f"      目标: {q['target']}")
        print(f"      top-{top_k}: {docs}")
        if res:
            print(f"      最高分: {res[0]['score']} ({res[0]['doc']})")
        print(f"      说明: {q['note']}\n")

    total = len(queries)
    print(f"\n{'='*60}")
    print(f"命中率: {hits}/{total} = {hits/total*100:.0f}%")


if __name__ == "__main__":
    main()