"""
chunker — 文档分块（第 3 章 RAG 管道第一步）

支持两种策略（对照 book 三种里的前两种）：
  fixed      固定大小切分：按 chunk_size 字符硬切，相邻块保留 chunk_overlap 重叠
  recursive  递归/段落感知：先按空行切段落，段落超限再降级到句子，句子仍超再硬切
              （生产最常用默认，book 的 agentic-rag chunking.py 就是这套）

纯文本处理，零 API。块对象统一为 {index, text, meta}，
meta 里带 doc/strategy 标签，方便对比脚本展示来源。

刻意不在这里引入 embedding —— 语义切分(第三种策略)要额外算向量，留到后面单独做。
"""

from typing import Any


# 默认参数：块大小 256-1024 token（中文约按字符/2 估），重叠 10-20%。
# 示例语料每篇 ~1.3KB，先取 150 字符、重叠 20，够在单篇文档上看出切分差异。
DEFAULT_CHUNK_SIZE = 150
DEFAULT_CHUNK_OVERLAP = 20
MIN_CHUNK_SIZE = 50   # 太小的碎片不产出 block，避免一堆意义不大的短块


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点把文本拆成句子（含标点）。简单实现，够演示用。"""
    import re
    parts = re.split(r"([。！？!?；;]+)", text)
    sents = []
    for i in range(0, len(parts) - 1, 2):
        s = (parts[i] + parts[i + 1]).strip()
        if s:
            sents.append(s)
    tail = parts[-1].strip()
    if tail:
        sents.append(tail)
    return sents


def _make_chunk(index: int, text: str, strategy: str, doc: str) -> dict[str, Any]:
    return {"index": index, "text": text, "meta": {"doc": doc, "strategy": strategy}}


def chunk_text(
    text: str,
    *,
    method: str = "recursive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    doc: str = "",
) -> list[dict[str, Any]]:
    """对一段文本分块。

    Args:
        text: 待分块文本
        method: "fixed" | "recursive"
        chunk_size: 目标块大小（字符数，中文近似）
        chunk_overlap: 相邻块重叠字符数（fixed 生效；recursive 里作为段，不用于硬切）
        doc: 文档标识，塞进每块的 meta，便于追溯

    Returns:
        list[dict]，每块含 index / text / meta
    """
    text = text.strip()
    if not text:
        return []

    if method == "fixed":
        return _chunk_fixed(text, chunk_size, chunk_overlap, doc)
    return _chunk_recursive(text, chunk_size, doc)


def _chunk_fixed(
    text: str, size: int, overlap: int, doc: str
) -> list[dict[str, Any]]:
    """固定大小切分：按固定字符数切，相邻块重叠 overlap 字符。最简单，结果可预测。"""
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        seg = text[i : i + size]
        if len(seg) < MIN_CHUNK_SIZE:
            # 最后一段太短（剩个尾巴），并入上一块结尾或丢弃
            if chunks:
                chunks[-1]["text"] = chunks[-1]["text"] + seg
            continue
        chunks.append(_make_chunk(len(chunks), seg, "fixed", doc))
    return chunks


def _chunk_recursive(text: str, size: int, doc: str) -> list[dict[str, Any]]:
    """递归/段落感知切分：先按空行切段落，段落超限降级到句子，句子仍超再硬切。

    这是 book agentic-rag chunking.py 的思路（respect_paragraph_boundary 那一支）：
      1. 以空行分隔的段落为基本单元
      2. 累积段落直到接近 size，超过就结算成一块
      3. 单段超 size → 按句子拆；单句超 size → 按 size 硬切
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 输入没空行分段时退化为"按行段落"，避免整篇被当一个大段落
    if len(paragraphs) <= 1 and "\n" in text:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[dict[str, Any]] = []

    def flush(acc: list[str], acc_size: int):
        if acc:
            chunk_text_ = "\n\n".join(acc)
            if len(chunk_text_) >= MIN_CHUNK_SIZE:
                chunks.append(_make_chunk(len(chunks), chunk_text_, "recursive", doc))

    acc: list[str] = []   # 当前正在累积的段落
    acc_size = 0
    for para in paragraphs:
        para_size = len(para)

        # 单段本身就超上限：先把累积的结算，再单独处理这一段
        if para_size > size:
            flush(acc, acc_size)
            acc, acc_size = [], 0

            # 段内按句子拆，句子仍超再按 size 硬切 —— 递归降级
            for sent in _split_sentences(para):
                if len(sent) <= size:
                    flush([sent], len(sent))
                else:
                    for i in range(0, len(sent), size):
                        seg = sent[i : i + size]
                        if len(seg) >= MIN_CHUNK_SIZE:
                            chunks.append(_make_chunk(len(chunks), seg, "recursive", doc))
            continue

        # 加上这个段会超过目标：结算当前累积，开新块
        if acc_size + para_size > size and acc:
            flush(acc, acc_size)
            acc, acc_size = [para], para_size
        else:
            acc.append(para)
            acc_size += para_size

    flush(acc, acc_size)
    return chunks