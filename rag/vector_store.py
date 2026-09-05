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

from partical.rag.chunker import chunk_text

_INDEX_FILE = Path(__file__).resolve().parent / "data" / "indexes" / "embeddings.json"


def _load_docs() -> dict[str, str]:
    """读 data/ 下所有 md，返回 {文件名: 文本}。"""
    data = _INDEX_FILE.parent.parent
    docs = {}
    for f in sorted(data.glob("*.md")):
        docs[f.name] = f.read_text(encoding="utf-8")
    return docs


def _chunks():
    """与 eval_bm25 相同的分块方式：把全部文档切成一排 chunk。"""
    chunks = []
    for fname, text in _load_docs().items():
        chunks += chunk_text(text, method="recursive", doc=fname)
    return chunks


def _cache_exists() -> bool:
    return _INDEX_FILE.exists()


def _load_cache() -> list[dict]:
    """从缓存 JSON 读回 chunk+vector。"""
    data = json.loads(_INDEX_FILE.read_text(encoding="utf-8"))
    return data["chunks"]


async def build_or_load_index(auto_embed: bool = True) -> list[dict]:
    """核心入口：返回可用的一列 chunk（含 vector）。

    若缓存已存在 → 直接读缓存（不调 API）。
    若不存在 → 嵌入全部 chunk 并写缓存（调一次方舟 API，除非 auto_embed=False）。
    auto_embed=False 时若缓存缺失，则返回空（供只读/检查场景）。

    Returns:
        list of {"index","doc","text","vector"}
    """
    # 1) 缓存命中：直接读
    if _cache_exists():
        print(f"[vector_store] 命中缓存 {_INDEX_FILE.name}, 跳过嵌入")
        return _load_cache()

    # 2) 缓存缺失：嵌入 + 落盘
    if not auto_embed:
        print("[vector_store] 缓存缺失且 auto_embed=False，返回空")
        return []

    chunks = _chunks()
    texts = [c["text"] for c in chunks]
    print(f"[vector_store] 首次嵌入 {len(texts)} 个 chunk ...")

    from partical.rag.embedded import embed_texts, embedding_dim

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

    _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_FILE.write_text(
        json.dumps({"model": "doubao-embedding-vision", "dim": dim, "chunks": records},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[vector_store] 已写入 {_INDEX_FILE}")
    return records