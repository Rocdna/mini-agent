"""
vector_store — 向量存储层（稠密检索的"索引"）

干两件事：
  1. 确保 42 个 chunk 的向量已嵌入并落盘（首次调方舟 API，之后读缓存不再调用）
  2. 提供"载入全部存储"供检索器算余弦相似度

存储格式：data/indexes/embeddings.json
    {
      "model": doubao-embedding-vision,
      "dim": 2048,
      "chunks": [
        {"index": 0, "doc": "refund_policy.md", "text": "...", "vector": [0.xxx, ...]},
        ...
      ]
    }

缓存是否命中的判定：embedding 文件和语料绑定。若数据文档变了，应重建缓存。
本实现用"文件存在即视为有效"的简单策略 —— 语料改动时删掉该文件即可重建。

注意：chunk 的 sumber 用的 recursive 分块（与 eval_bm25 一致），保证 dense 和 sparse 是在同一批 chunk 上对比。
"""

import json
from pathlib import Path

from rag.chunker import chunk_text

_DEFAULT_CORPUS = Path(__file__).resolve().parent / "data"

# 记忆化每个语料的索引文件路径，避免重复构建
_index_providers: dict[str, tuple[Path, Path]] = {}


def _resolve(corpus_root: str | Path | None) -> tuple[Path, Path]:
    """返回 (md目录, 索引文件路径)。默认用 data/，可指定其他语料目录。"""
    corpus = Path(corpus_root) if corpus_root else _DEFAULT_CORPUS
    index_file = corpus / "indexes" / "embeddings.json"
    return corpus, index_file


def _load_docs(corpus_root: str | Path | None = None) -> dict[str, str]:
    """读语料目录下所有 md，返回 {文件名: 文本}。默认 data/。"""
    corpus, _ = _resolve(corpus_root)
    docs = {}
    for f in sorted(corpus.glob("*.md")):
        docs[f.name] = f.read_text(encoding="utf-8")
    return docs


def _chunks(corpus_root: str | Path | None = None):
    """与 eval_bm25 相同的分块方式：把全部文档切成一排 chunk。"""
    chunks = []
    for fname, text in _load_docs(corpus_root).items():
        chunks += chunk_text(text, method="recursive", doc=fname)
    return chunks


def _cache_exists(corpus_root: str | Path | None = None) -> bool:
    _, index_file = _resolve(corpus_root)
    return index_file.exists()


def _load_cache(corpus_root: str | Path | None = None) -> list[dict]:
    """从缓存 JSON 读回 chunk+vector。"""
    _, index_file = _resolve(corpus_root)
    data = json.loads(index_file.read_text(encoding="utf-8"))
    return data["chunks"]


async def build_or_load_index(auto_embed: bool = True,
                              corpus_root: str | Path | None = None) -> list[dict]:
    """核心入口：返回可用的一列 chunk（含 vector）。

    Args:
        auto_embed: 缓存缺失时是否自动嵌入。False 时返回空（只读/检查场景）。
        corpus_root: 语料目录，默认 data/。传其他目录可独立建索引，
                     索引文件落在 {corpus_root}/indexes/embeddings.json，互不覆盖。

    若缓存已存在 → 直接读缓存（不调 API）。
    若不存在 → 嵌入全部 chunk 并写缓存（调方舟 API，除非 auto_embed=False）。

    Returns:
        list of {"index","doc","text","vector"}
    """
    # 1) 缓存命中：直接读
    if _cache_exists(corpus_root):
        _, index_file = _resolve(corpus_root)
        print(f"[vector_store] 命中缓存 {index_file.name}, 跳过嵌入")
        return _load_cache(corpus_root)

    # 2) 缓存缺失：嵌入 + 落盘
    if not auto_embed:
        print("[vector_store] 缓存缺失且 auto_embed=False，返回空")
        return []

    chunks = _chunks(corpus_root)
    texts = [c["text"] for c in chunks]
    print(f"[vector_store] 首次嵌入 {len(texts)} 个 chunk ...")

    from rag.embedded import embed_texts, embedding_dim

    vecs = await embed_texts(texts)   # 调一次方舟 API
    dim = embedding_dim()
    print(f"[vector_store] 完成, 维度={dim}")

    records = []
    for c, v in zip(chunks, vecs):
        records.append({
            "index": c["index"],
            "doc": c["meta"]["doc"],
            "text": c["text"],
            "vector": v,
        })

    _, index_file = _resolve(corpus_root)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(
        json.dumps({"model": "doubao-embedding-vision", "dim": dim, "chunks": records},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[vector_store] 已写入 {index_file}")
    return records