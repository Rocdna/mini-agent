"""
ReAct 循环 — Agent 的核心引擎

while 循环：调 LLM → 有 tool_calls 就执行并喂回 → 没有就结束。
max_turns 兜底，防止模型无限调工具。

练习路线：
  第 0 关：不塞"强制收尾"的假 user 消息，让模型自己决定何时停止
  第 2 关：每轮打印 cache hit/miss，观察 KV Cache 命中
  第 3 关：每轮调用前注入临时状态栏消息（只进本次请求，不写进 messages）
  第 4 关：每轮调用前检查是否需要压缩（status_bar / compressor 挂进来）
"""

import copy
import json
import logging
from typing import AsyncGenerator

from partical.api import chat_with_tools

logger = logging.getLogger("agent_loop")

# 注意：这里刻意【没有】把工具结果后追加一个假 user 消息去"引导收尾"。
# 那是第 0 关要拆掉的拐杖——往轨迹里塞伪造 user 消息会让模型以为换话题、
# 清掉历史思维链。ReAct 的终止条件就一个：模型不再请求 tool_calls。见教程第 0 关。


def _build_schemas(tool_executors: dict) -> list[dict] | None:
    """把注册的工具函数转成 API 的 tools 格式。

    每个函数都有 __tool_schema__ 属性；无工具时返回 None。
    """
    schemas = []
    for name, fn in tool_executors.items():
        if hasattr(fn, "__tool_schema__"):
            schema = fn.__tool_schema__.copy()
            schema["function"]["name"] = name
            schemas.append(schema)
    return schemas or None


def _assistant_msg_with_tools(content: str, tool_calls: list[dict]) -> dict:
    """构建 assistant 消息并附上 tool_calls（API 要求 arguments 是 JSON 字符串）。"""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                },
            }
            for c in tool_calls
        ],
    }


def _tool_result_msg(tool_call_id: str, result: str) -> dict:
    """构建 tool 消息，把工具执行结果按 tool_call_id 喂回历史。"""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result,
    }


async def _execute_one_tool(name: str, args: dict, tool_executors: dict) -> str:
    """执行单个工具调用，返回结果字符串。

    未知工具 / 执行异常都不抛——错误描述字符串也是喂给模型的有效信息。
    """
    executor = tool_executors.get(name)
    if executor is None:
        return f"未知工具: {name}。可用工具: {list(tool_executors.keys())}"

    try:
        result = await executor(args)
    except Exception as e:
        return f"工具执行错误: {e}"

    # API 要求 tool 消息的 content 必须是 string
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


async def run_agent_loop(
    messages: list[dict],
    tool_executors: dict,
    max_turns: int = 5,
    verbose: bool = False,
) -> AsyncGenerator[dict, None]:
    """执行 ReAct 循环。

    yield 事件类型：
      tool_start   {"type", "name", "arguments"}
      tool_result  {"type", "name", "result"}
      token        {"type", "content"}        最终回答文本
      done         {"type", "content"}        循环正常结束
      error        {"type", "message"}
      trace        {"type", "turn", "messages", "tools"}  只有 verbose=True 时，
                   每轮调模型前把本次请求的完整 messages 快照发出来，供 /debug 渲染

    每轮：
      1. 调 LLM（带上工具定义）
      2. 无 tool_calls → 模型给了最终回答 → 结束
      3. 有 tool_calls → assistant 消息原样放回历史（模型要"看到"自己的决策）
         → 逐个执行工具 → 结果按 tool_call_id 追加 → 回到顶再调模型
    """
    tool_schemas = _build_schemas(tool_executors)
    prev_cache = None  # 上上轮请求的 KV Cache 数字，随 trace 一起展示

    for turn in range(1, max_turns + 1):
        logger.info(f"── Agent 轮次 {turn}/{max_turns} ──")

        # verbose: 把"即将发给模型"的 messages 快照发出去（深拷贝，避免打印时被后续 append 干扰）
        if verbose:
            yield {
                "type": "trace",
                "turn": turn,
                "messages": copy.deepcopy(messages),
                "tools": tool_schemas,
                "cache": prev_cache,  # 上轮请求的 hit/miss（这轮还没发出去）
            }

        response = await chat_with_tools(messages, tool_schemas)
        if response["error"]:
            yield {"type": "error", "message": response["error"]}
            return

        content = response["content"]
        tool_calls = response["tool_calls"]

        # 第 2 关：观察 KV Cache 命中——把数字带进下轮 trace 展示，替代看不见的 logger
        prev_cache = {
            "hit": response.get("cache_hit_tokens", 0),
            "miss": response.get("cache_miss_tokens", 0),
        }

        # ── 模型不再要工具 → 这就是最终回答 → 结束 ──
        if not tool_calls:
            if content:
                yield {"type": "token", "content": content}
            yield {"type": "done", "content": content}
            return

        # ── 有工具调用 → 记录 assistant 决策，逐个执行 ──
        messages.append(_assistant_msg_with_tools(content, tool_calls))

        for call in tool_calls:
            yield {"type": "tool_start", "name": call["name"], "arguments": call["arguments"]}

            result = await _execute_one_tool(call["name"], call["arguments"], tool_executors)

            yield {"type": "tool_result", "name": call["name"], "result": result}
            messages.append(_tool_result_msg(call["id"], result))

        # 回到循环顶：带上更新后的完整历史再调模型，让模型自己决定是否继续

    # max_turns 兜底（真实防线，靠模型自觉不算数）
    yield {
        "type": "error",
        "message": f"Agent 达到最大轮次 ({max_turns})，强制终止。",
    }