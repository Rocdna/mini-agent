"""
DeepSeek API 底层调用 — 所有模块共用

维护两个入口：
  - chat_stream(messages):       流式对话（打字机效果，聊天模式用）
  - chat_with_tools(messages):   非流式 + 工具调用（ReAct 模式用）

练习路线（见 partical/chapter2-practice-tutorial.md）：
  第 0 步：实现 chat_with_tools（非流式 + 工具调用）
  第 2 关：返回值加上 cache_hit_tokens / cache_miss_tokens（观察 KV Cache 命中）
"""

import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }


async def chat_stream(messages: list[dict]):
    """流式对话，逐 token yield。

    用法：
        async for token in chat_stream(messages):
            print(token, end="", flush=True)
    """
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", DEEPSEEK_URL, headers=_headers(), json=payload) as resp:
            if resp.status_code != 200:
                yield f"\n[错误] API 返回 {resp.status_code}"
                return

            # 逐行解析 SSE：data: {...}，遇到 [DONE] 结束
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                    token = data["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue  # 心跳行 / 空行，跳过


async def chat_with_tools(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """非流式 + 工具调用（ReAct 模式用）。

    Args:
        messages: 对话历史（list[dict]）
        tools: OpenAI/DeepSeek 格式的工具定义列表

    Returns:
        {
            "content": str,          # 模型输出的文字（可能为空字符串）
            "tool_calls": list|None, # [{id,name,arguments}]，无则 None
            "error": str|None,       # 出错时的描述
            "cache_hit_tokens": int, # KV Cache 命中数（第 2 关）
            "cache_miss_tokens": int # KV Cache 未命中数
        }
    """
    payload: dict = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(DEEPSEEK_URL, headers=_headers(), json=payload)
        if resp.status_code != 200:
            return {
                "content": "", "tool_calls": None,
                "error": f"API {resp.status_code}: {resp.text[:300]}",
                "cache_hit_tokens": 0, "cache_miss_tokens": 0,
            }

        data = resp.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage", {})

        return {
            "content": message.get("content", "") or "",
            "tool_calls": _parse_tool_calls(message),
            "error": None,
            # 第 2 关：把 KV Cache 命中量暴露出来，方便在 agent_loop 里观察
            "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
            "cache_miss_tokens": usage.get("prompt_cache_miss_tokens", 0),
        }


def _parse_tool_calls(message: dict) -> list[dict] | None:
    """从 API 返回的 message 里提取工具调用列表。参数是 JSON 字符串，要 json.loads 成 dict。"""
    raw = message.get("tool_calls", [])
    if not raw:
        return None

    calls = []
    for tc in raw:
        func = tc.get("function", {})
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}  # 模型给的不是合法 JSON，当空参数处理，别让解析炸掉
        calls.append({
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "arguments": args,
        })
    return calls or None