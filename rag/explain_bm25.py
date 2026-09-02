"""
explain_bm25 — 诊断脚本：把查询在某篇文档上的 BM25 打分逐词拆开，看分从哪来。

用法：
  python -m partical.rag.explain_bm25 "买了不喜欢的东西能退吗"
  # 默认分析 q3，可指定不同查询和对比的两篇文档

目的：亲眼看到稀疏检索的"死穴"——查询词没在目标文档里出现时，贡献从何而来。
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from partical.rag.chunker import chunk_text
from partical.rag.retriever import build_index, tokenize


def _chunks():
    data = Path(__file__).resolve().parent / "data"
    out = []
    for f in sorted(data.glob("*.md")):
        out += chunk_text(f.read_text(encoding="utf-8"), method="recursive", doc=f.name)
    return out


def _find_doc_idx(index, target_file):
    """找到"目标文档"里分数最高的那个 chunk 的 index，作为对比基准。"""
    best = None
    for i, c in enumerate(index.docs):
        if (c or {}).get("meta", {}).get("doc") == target_file:
            if best is None or index.doc_len[i] > index.doc_len[best]:
                best = i
    return best


def _explain(query, target, compare, top_k=3):
    chunks = _chunks()
    idx = build_index(chunks)

    print(f"\n{'='*64}\n查询: 「{query}」\n分词: {tokenize(query)}\t共 {len(idx.docs)} 个文档\n{'='*64}")

    res = idx.search(query, top_k=top_k)
    print(f"\n▶ top-{top_k} 检索结果:")
    for r in res:
        print(f"   [{r['score']:.4f}] {r['doc']}  (chunk#{r['index']})")

    # 逐词分解：对比两篇
    for label, fname in [("目标文档", target), ("顶头文档", compare)]:
        d = _find_doc_idx(idx, fname)
        if d is None:
            print(f"\n  ✗ 找不到 {fname}")
            continue
        ex = idx.explain(query, d)
        print(f"\n▶ {label}: {fname}  (chunk#{d}, 长度{ex['doc_len']}, avgdl={ex['avgdl']}, n={ex['n']})")
        print(f"   总分 = {ex['total']}")
        for p in ex["parts"]:
            if p["命中?"]:
                print(f"   词「{p['词']}」df={p['df']} IDF={p['IDF']} TF={p['TF']} "
                      f"词频饱和项={p['词频饱和项']} → 贡献 {p['贡献']}")
            else:
                print(f"   词「{p['词']}」✗ 未命中 (df={p['df']}) → 贡献 0")
    print("\n")


if __name__ == "__main__":
    # 默认诊断 q3：字面无"退款"却该命中退款政策的语义查询
    Q = sys.argv[1] if len(sys.argv) > 1 else "买了不喜欢的东西能退吗"
    _explain(Q, target="refund_policy.md", compare="invoice.md")