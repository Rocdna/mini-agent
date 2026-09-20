"""
compare_chunkers — 对比固定大小 vs 段落感知分块

对 data/ 下每篇文档用两种策略各分一次，输出：
  1. 每篇的块数 / 平均块长 / 最小最大块长
  2. 具代表性的一篇，抽样展示两种策略的块内容差异（fixed 会拦腰切句，recursive 对齐段落）

纯文本，零 API。用法：
  python -m partical.tests.compare_chunkers
  # 只看某一篇：
  python -m partical.tests.compare_chunkers --doc refund_policy.md
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent / "rag"
DATA = ROOT / "data"

from partical.rag.chunker import chunk_text


def _load_docs() -> dict[str, str]:
    docs = {}
    for f in sorted(DATA.glob("*.md")):
        docs[f.name] = f.read_text(encoding="utf-8")
    return docs


def _stats(chunks: list[dict]) -> tuple[int, float, int, int, int]:
    """>> 返回 (块数, 平均长, 最小长, 最大长, 重叠字数)"""
    if not chunks:
        return 0, 0.0, 0, 0, 0
    lens = [len(c["text"]) for c in chunks]
    # 重叠字数：把相邻块拼接后与原文对比，估算重复的字符量
    total = sum(lens)
    return (
        len(chunks),
        round(total / len(lens), 1),
        min(lens),
        max(lens),
        total - sum(len(c["text"]) for c in chunks),
    )


def _show_breaks(chunks: list[dict], label: str, max_blocks: int = 6):
    """在每一块开头的缩进处前后标记，直观展示切点。"""
    print(f"  [{label}] 共 {len(chunks)} 块：")
    for c in chunks[:max_blocks]:
        first = c["text"].splitlines()[0].strip() if c["text"] else ""
        # 用「|」代表切点边界
        preview = (first[:40] + "…" ) if len(first) > 40 else first
        print(f"    ┃ {preview}")
    if len(chunks) > max_blocks:
        print(f"    …(其余 {len(chunks) - max_blocks} 块省略)")


def _dump_blocks(chunks: list[dict], label: str):
    """完整打印每块的 dict 结构（含 index/text/meta）与全文，看数据处理后全貌。"""
    print(f"\n────────── [{label}] 共 {len(chunks)} 块 ──────────")
    for c in chunks:
        print(f"\n--- 块 index={c['index']}  meta={c['meta']}  text_len={len(c['text'])} ---")
        print(c["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="", help="只看某一篇 md（给文件名），默认全部")
    ap.add_argument("--size", type=int, default=150, help="块大小字符数")
    ap.add_argument("--overlap", type=int, default=20, help="fixed 策略的重叠字符数")
    ap.add_argument("--dump", action="store_true",
                    help="完整打印每块的 dict 结构和全文（需要 --doc 指定一篇，否则只看第一篇）")
    args = ap.parse_args()

    docs = _load_docs()
    if not docs:
        print("data/ 下没有 .md 文档")
        return

    targets = {args.doc: docs[args.doc]} if args.doc else docs

    # --dump 模式：完整打印一篇文档两种策略切出的块结构全貌
    if args.dump:
        doc_name = args.doc or next(iter(targets))
        text = docs[doc_name]
        print(f"\n══════════ 原文全文 ══════════\n{text.strip()}\n")

        fixed = chunk_text(text, method="fixed", chunk_size=args.size,
                           chunk_overlap=args.overlap, doc=doc_name)
        recur = chunk_text(text, method="recursive", chunk_size=args.size, doc=doc_name)
        _dump_blocks(fixed, "fixed")
        _dump_blocks(recur, "recursive")
        return

    for name, text in targets.items():
        total_len = len(text.strip())
        fixed = chunk_text(text, method="fixed", chunk_size=args.size,
                           chunk_overlap=args.overlap, doc=name)
        recur = chunk_text(text, method="recursive", chunk_size=args.size, doc=name)

        nf, mf, mnf, mxf, _ = _stats(fixed)
        nr, mr, mnr, mxr, _ = _stats(recur)

        print(f"\n══════ {name}  (原文 {total_len} 字符) ══════")
        print(f"  fixed      : {nf} 块 | 平均 {mf} | 最小 {mnf} | 最大 {mxf}")
        print(f"  recursive  : {nr} 块 | 平均 {mr} | 最小 {mnr} | 最大 {mxr}")

        _show_breaks(fixed, "fixed")
        _show_breaks(recur, "recursive")


if __name__ == "__main__":
    main()