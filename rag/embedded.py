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

import asyncio
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


# 预乘基址（可当 0.5s 起步），429/5xx 网络错误时指数退避
_RETRY_BASE = 0.5

# 429(限流)/500/502/503(服务端瞬时) 值得等重试；4xx 业务错(如 400)不该死循环
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


async def _post_with_retry(client: httpx.AsyncClient,
                           url: str,
                           payload: dict,
                           max_retries: int) -> httpx.Response | None:
    """POST 一次，429/5xx/网络错误按 Retry-After 或指数退避重试 max_retries 次。

    Returns:
        成功响应；重试耗尽返回 None 让调用方决定怎么抛。

    退避策略：若响应带 Retry-After 头优先用它；否则 2^n * 基址 指数增长。
    只对"瞬时可恢复"的状态码重试，彻底的业务错误（4xx 非 408/429）直接抛出。
    """
    attempt = 0
    while True:
        try:
            resp = await client.post(url, headers=_headers(), json=payload)
        except httpx.HTTPError:
            if attempt >= max_retries:
                return None
            attempt += 1
            await asyncio.sleep(_RETRY_BASE * (2 ** attempt))
            continue

        if resp.status_code == 200:
            return resp

        # 非 200：区分"可重试"和"业务错"
        if resp.status_code not in _RETRYABLE_STATUS:
            raise RuntimeError(f"Embedding API {resp.status_code}: {resp.text[:300]}")

        if attempt >= max_retries:
            return None

        # 计算等待时长：优先响应头的 Retry-After（秒），否则指数退避
        ra = resp.headers.get("Retry-After")
        wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() \
            else _RETRY_BASE * (2 ** attempt)
        attempt += 1
        await asyncio.sleep(wait)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量，返回与输入等长的 list[list[float]]。

    Args:
        texts: 待嵌入的文本列表（如 N 个 chunk 的 text）

    Returns:
        list of 向量，长度与 texts 一致，顺序与输入对应。
        任一文本为空串会被跳过，因此返回长度可能 < len(texts) ——
        调用方需自行对齐（通常传非空文本）。

    网络/API 异常抛出 httpx.HTTPError，由上层兜底，不在这里吞。
    对 429（限流）/5xx/网络抖动做指数退避重试——批量建索引几十上百批连发
    很容易触发账号级限流（如 AccountRateLimitExceeded），自动等再试而不是全挂。
    """
    if not texts:
        return []

    # 火山方舟一次最多接受 BATCH_SIZE 条 input，超出要分批调用再拼接。
    # 放这里做批次切分，调用方（存向量 / 查询）不用自己关心上限。
    BATCH_SIZE = 10
    RETRY_MAX = 6
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    url = f"{ARK_BASE_URL}/embeddings"
    futures: list[list[float]] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            payload = {"model": MODEL, "input": batch}

            resp = await _post_with_retry(client, url, payload, RETRY_MAX)
            if resp is None:
                raise RuntimeError(f"Embedding API 重试 {RETRY_MAX} 次仍失败，放弃。")

            data = resp.json()["data"]
            # 按 index 排序，保证本批返回顺序与输入一致（方舟可能乱序）
            data.sort(key=lambda d: d["index"])
            futures += [d["embedding"] for d in data]

            n = i // BATCH_SIZE + 1
            print(f"[embed] batch {n}/{total_batches} 完成", flush=True)  # 逐批刷新,批量嵌入时可见进度

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