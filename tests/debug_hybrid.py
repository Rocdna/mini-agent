"""
debug_hybrid — 诊断混合检索：把 BM25 名次、稠密名次、RRF 融合结果三张表并排打印

用来回答"为什么混合比单边还差"：是稠密把错文档排高了，还是 BM25 污染拖累融合。
对每个问题 query 打印每篇文档在两侧的名次与融合分，一眼看清 top-1 是谁定的。

用法： python -m partical.tests.debug_hybrid
       （会 embed 一次 list of queries）
"""

import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 重点怀疑的 query：混合命中但 top-1 错的
PROBLEM_QUERIES = [
    "消费积分能当钱花吗？",       # q4 混合 top1 错到 account_security
    "买了不喜欢的东西能退吗？",   # q3 混合 top1 错到 after_sales
    "收到货大概要几天？",        # q6 混合 top1 错到 invoice
    "最近商场有满减活动吗？",    # q9 混合 top1 错到 account_security
]


async def main():
    from partical.rag.hybrid_search import build, cosine

    hybrid = await build()
    n = len(hybrid.chunks)

    print(f"共 {n} 个 chunk\n")
    for q in PROBLEM_QUERIES:
        qvec = (await __import__("partical.rag.embedded", fromlist=["embed_texts"]).embed_texts([q]))[0]
        bm_rank = hybrid._bm25_ranking(q)
        den_rank = hybrid._dense_ranking(qvec)

        # 合并所有曾被任意一侧命中的文档
        docs = {}
        for idx in set(bm_rank) | set(den_rank):
            docs[idx] = hybrid.chunks[idx]["meta"]["doc"]
        # 按融合分排序
        from partical.rag.rrf import rrf
        fused = dict(rrf([bm_rank, den_rank]))
        order = sorted(docs, key=lambda i: fused[i], reverse=True)

        print("=" * 78)
        print("Q:", q)
        print(f"{'chunk':>5}  {'doc':<24} {'bm_rank':>7} {'den_rank':>8}  fuse")
        for idx in order:
            print(f"{idx:>5}  {docs[idx]:<24} {bm_rank.get(idx,'—'):>7} {den_rank.get(idx,'—'):>8}  {fused[idx]:.4f}")
        print()
        top = max(order, key=lambda i: fused[i])
        print("  混合 top-1 =", hybrid.chunks[top]["meta"]["doc"])


if __name__ == "__main__":
    asyncio.run(main())