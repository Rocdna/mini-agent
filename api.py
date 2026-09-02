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

    # 网络掉线 / 超时也当错误处理：不抛给上层，改成 yield 一条错误行（否则主循环崩掉退出）
    try:
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
    except httpx.HTTPError as exc:
        # 掉线/超时/连接被重置统一提示，别让异常一路炸到 main 退出
        yield f"\n[错误] 连接异常：{exc.__class__.__name__}"


async def chat_stream_with_tools(messages: list[dict], tools: list[dict] | None = None):
    """流式 + 工具调用，逐 token yield。

    流式协议里文本(contact)与工具调用(tool_calls)分帧到达：
      - 文本帧：choices[0].delta.content  → 立刻 yield {"type":"token", "text":...}
      - 工具帧：choices[0].delta.tool_calls 是分片累积的（index/function.arguments 分多帧
        拼起来），要攒到流结束时才完整 → 结束时 yield {"type":"tool_calls", "calls":[...]}
    DeepSeek 流式里 tool_calls 的参数是 JSON 字符串分片，必须逐帧拼接后一次 json.loads。

    用法：
        calls = None
        async for ev in chat_stream_with_tools(messages, tools):
            if ev["type"] == "token":   print(ev["text"], end="")
            elif ev["type"] == "tool_calls": calls = ev["calls"]
    """
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    # 工具调用分片累积：{"id": ...}, {"function": {...}} 每帧叠加
    _tool_id = None
    _tool_name = ""
    _args_slices: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", DEEPSEEK_URL, headers=_headers(), json=payload) as resp:
                if resp.status_code != 200:
                    yield {"type": "token", "text": f"\n[错误] API 返回 {resp.status_code}"}
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = data["choices"][0].get("delta", {})
                    # ① 文本 token
                    text = delta.get("content")
                    if text:
                        yield {"type": "token", "text": text}
                    # ② 工具调用分片（可能有多个 tool_call，这里拼第一个）
                    #    流式里 delta.tool_calls 每帧给 index + function.arguments 片段
                    for tc in delta.get("tool_calls") or []:
                        func = tc.get("function", {})
                        if tc.get("id"):
                            _tool_id = tc["id"]
                        if func.get("name"):
                            _tool_name = func["name"]
                        args_part = func.get("arguments")
                        if args_part:
                            _args_slices.append(args_part)
    except httpx.HTTPError as exc:
        yield {"type": "token", "text": f"\n[错误] 连接异常：{exc.__class__.__name__}"}
        return

    # 流结束：若有工具调用，攒齐参数后 yield
    if _tool_id is not None:
        raw_args = "".join(_args_slices)
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
        yield {"type": "tool_calls", "calls": [{"id": _tool_id, "name": _tool_name, "arguments": args}]}


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

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(DEEPSEEK_URL, headers=_headers(), json=payload)
    except httpx.HTTPError as exc:
        # 掉线/超时/连接被重置：转成 error，让 agent_loop 走已有错误渲染，不崩进程
        return {
            "content": "", "tool_calls": None,
            "error": f"连接异常：{exc.__class__.__name__}（{exc}）",
            "cache_hit_tokens": 0, "cache_miss_tokens": 0,
        }

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