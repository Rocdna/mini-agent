"""
embedded — 稠密嵌入（第 3 章实验 3-4 / 3-6 的稠密腿）

用豆包 doubao-embedding-vision 把文本变成向量（OpenAI 兼容协议，火山方舟）。
与 BM25（稀疏/字面）互补：embedding 捕捉语义，BM25 捕捉精确词字面。

调用方式（OpenAI 兼容）：
    POST {ARK_BASE_URL}/embeddings
    Authorization: Bearer {ARK_API_KEY}
    Body: {"model": doubao-embedding-vision, "input": ["文本1", "文本2", ...]}
    Resp: {"data": [{"embedding": [...], "index": 0}, ...], "model": ..., "usage": {...}}

配置读取 .env（load_dotenv）。此模块负责"调 API 拿向量"，是否建索引 / 查询在外部组织。
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

ARK_KEY = os.getenv("ARK_API_KEY", "")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
MODEL = os.getenv("EMBEDDING_MODEL", "doubao-embedding-vision")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {ARK_KEY}",
        "Content-Type": "application/json",
    }


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量，返回与输入等长的 list[list[float]]。

    Args:
        texts: 待嵌入的文本列表（如 N 个 chunk 的 text）

    Returns:
        list of 向量，长度与 texts 一致，顺序与输入对应。
        任一文本为空串会被跳过，因此返回长度可能 < len(texts) ——
        调用方需自行对齐（通常传非空文本）。

    网络/API 异常抛出 httpx.HTTPError，由上层兜底，不在这里吞。
    """
    if not texts:
        return []

    # 火山方舟一次最多接受 BATCH_SIZE 条 input，超出要分批调用再拼接。
    # 放这里做批次切分，调用方（存向量 / 查询）不用自己关心上限。
    BATCH_SIZE = 10
    url = f"{ARK_BASE_URL}/embeddings"
    futures: list[list[float]] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            payload = {"model": MODEL, "input": batch}

            resp = await client.post(url, headers=_headers(), json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding API {resp.status_code}: {resp.text[:300]}")

            data = resp.json()["data"]
            # 按 index 排序，保证本批返回顺序与输入一致（方舟可能乱序）
            data.sort(key=lambda d: d["index"])
            futures += [d["embedding"] for d in data]

    # 首次调用后缓存真实维度，供 embedding_dim() 查询
    global _cached_dim
    if futures and _cached_dim is None:
        _cached_dim = len(futures[0])

    return futures


# 首次调用后记录真实维度。不硬编码——以模型的第 1 次返回为准。
_cached_dim: int | None = None


def embedding_dim() -> int | None:
    """返回模型嵌入维度。

    首次调用 embed_texts()/embed_text() 后，维度会被自动记录并缓存；
    调用本函数即可查询。若尚未调用过任何嵌入，返回 None（维度未知）。
    所以要多一个"调一次就缓存"的设计：维度是模型的固定属性，查一次即可长期复用。
    """
    return _cached_dim


async def embed_text(text: str) -> list[float]:
    """单条文本嵌入（快捷，内部转成单元素列表调 embed_texts）。"""
    r = await embed_texts([text])
    return r[0]