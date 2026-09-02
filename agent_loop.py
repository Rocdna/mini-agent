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

from partical.api import chat_stream_with_tools
from partical.compressor import estimate_tokens, should_compress, compress_tool_results
from partical.status_bar import StatusBar

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
    use_status_bar: bool = True,
    use_compressor: bool = True,
    enable_memory: bool = True,
    memory_tools: dict | None = None,
) -> AsyncGenerator[dict, None]:
    """执行 ReAct 循环。

    Args:
        use_status_bar: 第 3 关的状态栏是否注入。设 False 可做对照实验。
        use_compressor: 第 4 关的上下文压缩是否启用。设 False 可对照"不压"基线。
        enable_memory: 第 3 章 — 记忆工具是否接入本循环。True 时把 memory_tools
                       并进工具集，让 agent 能在回答时调用 add_memory 沉淀记忆；
                       False 时完全不带记忆工具（对照"无记忆"基线）。
        memory_tools:   enable_memory=True 时要并入的 {name: fn} 记忆工具字典
                        （来自 build_memory_tools）。缺省则不额外并入。

    yield 事件类型：
      tool_start   {"type", "name", "arguments"}
      tool_result  {"type", "name", "result"}
      token        {"type", "content"}        最终回答文本
      done         {"type", "content"}        循环正常结束
      error        {"type", "message"}
      compress     {"type", "before_est", "after_est", "merged"}  压缩触发时，
                   报告压前/压后估算 token 与合并条数，供 main 渲染
      trace        {"type", "turn", "messages", "tools"}  只有 verbose=True 时，
                   每轮调模型前把本次请求的完整 messages 快照发出来，供 /debug 渲染

    每轮：
      1. 历史超预算则上下文压缩（只压 tool 结果）
      2. 调 LLM（带上工具定义）
      3. 无 tool_calls → 模型给了最终回答 → 结束
      4. 有 tool_calls → assistant 消息原样放回历史（模型要"看到"自己的决策）
         → 逐个执行工具 → 结果按 tool_call_id 追加 → 回到顶再调模型
    """
    # 第 3 章：启用记忆时，把记忆工具并进工具集，让 agent 能在回答中沉淀记忆。
    # 与 state_bar/compressor 的"临时注入"呼应：记忆也是一个可开关的插件。
    if enable_memory and memory_tools:
        # 不污染调用方传入的 dict：拷贝后合并，保留原有工具
        tool_executors = {**tool_executors, **memory_tools}

    tool_schemas = _build_schemas(tool_executors)
    prev_cache = None  # 上上轮请求的 KV Cache 数字，随 trace 一起展示
    bar = StatusBar()  # 第 3 关：状态栏，用代码统计工具调用次数喂给模型

    # 第 4 关：用户的原始问题作为压缩的"任务意图"（摘要按它留什么）
    task = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")

    for turn in range(1, max_turns + 1):
        logger.info(f"── Agent 轮次 {turn}/{max_turns} ──")
        bar.on_turn()

        # ── 第 4 关：历史超预算就批量压缩（只压 tool 结果，原位替换）──
        # 压缩动的是 messages（长期历史），与状态栏的"临时追加"相反；
        # 用 before/after 数字发 compress 事件，让 main 把"压了多少"渲染出来。
        if use_compressor and should_compress(messages):
            before_est = estimate_tokens(messages)
            await compress_tool_results(messages, task)
            yield {
                "type": "compress",
                "before_est": before_est,
                "after_est": estimate_tokens(messages),
                "merged": len([m for m in messages
                               if "[COMPRESSED]" in (m.get("content") or "")]),
            }

        # ── 第 3 关：把状态栏作为一条【临时】user 消息追加到请求末尾 ──
        # 关键：只拼进 msgs_to_send（本次请求），messages 保持干净不写历史。
        # use_status_bar=False 时退化成旧行为（不注入），供对照实验用。
        if use_status_bar:
            msgs_to_send = [*messages, {"role": "user", "content": bar.render()}]
        else:
            msgs_to_send = messages

        # verbose: 把"即将发给模型"的 messages 快照发出去（含临时状态栏，深拷贝防干扰）
        if verbose:
            yield {
                "type": "trace",
                "turn": turn,
                "messages": copy.deepcopy(msgs_to_send),
                "tools": tool_schemas,
                "cache": prev_cache,  # 上轮请求的 hit/miss（这轮还没发出去）
            }

        # ── 流式调用：文字逐 token 冒出来（打字机），工具调用攒齐后返回 ──
        # 这样 ReAct 的 assistant 文本 + 最终回答都有打字机效果，记忆工具不受影响。
        content_tokens: list[str] = []
        tool_calls = None
        api_error = None
        async for ev in chat_stream_with_tools(msgs_to_send, tool_schemas):
            if ev["type"] == "token":
                text = ev["text"]
                if text.startswith("\n[错误]"):    # 流式层的网络/错误提示
                    api_error = text
                    continue
                content_tokens.append(text)
                yield {"type": "token", "content": text}   # 打字机：main 逐字渲染
            elif ev["type"] == "tool_calls":
                tool_calls = ev["calls"]

        if api_error:
            yield {"type": "error", "message": api_error}
            return

        content = "".join(content_tokens)

        # 第 2 关：观察 KV Cache 命中——把数字带进下轮 trace 展示（当前流式接口暂不回报，留 0）
        prev_cache = {
            "hit": 0,
            "miss": 0,
        }

        # ── 模型不再要工具 → 这就是最终回答 → 结束（文本已打字机流出）──
        if not tool_calls:
            yield {"type": "done", "content": content}
            return

        # ── 有工具调用 → 记录 assistant 决策，逐个执行 ──
        messages.append(_assistant_msg_with_tools(content, tool_calls))

        for call in tool_calls:
            bar.on_tool_call(call["name"])  # 第 3 关：执行前记一次，供下一轮状态栏显示

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