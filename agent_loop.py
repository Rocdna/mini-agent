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
from typing import AsyncGenerator, Awaitable, Callable, Optional

from api import chat_stream_with_tools
from compressor import estimate_tokens, should_compress, compress_tool_results
from status_bar import StatusBar

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


# 空参暖机柔推：常见工具的"正确用法"示范，供重复空转提醒时引用
_NUDGE_EXAMPLES = {
    "grep_files": "grep_files(pattern='关键词', path='.', file_pattern='*.py')",
    "glob_files": "glob_files(path='.', file_pattern='**/*.py')",
    "read_file": "read_file(file_path='main.py')",
    "execute_code": "execute_code(code='print(1+1)', lang='python', timeout=10)",
    "web_search": "web_search(query='关键词')",
}


def _nudge_msg(name: str) -> dict:
    """构造一条温和纠正的 system 消息：同工具·同参数连续空转时注入，指引落参。

    注意：这是【柔和助推】，不是硬终止——DeepSeek 这类模型往往先发空壳 {} 试探
    工具，空报错只会让它原地重试同一个 {}，但硬终止又会误杀 turn 5 那种最终带正
    成功运行。折中：识别"同一工具·相同参数"连续 ≥2 次，注入一句带正确用法的
    system 纠正，让它下一轮能落正参数。
    """
    example = _NUDGE_EXAMPLES.get(name, f"{name}(<完整 required 参数>)")
    return {
        "role": "system",
        "content": (
            f"[harness 提示] 同一工具「{name}」已连续带相同(或空)参数调用 2 次而没有进展。"
            f"一个工具调用必须填满 required 参数才有意义，请参考正确用法重试：{example}"
        ),
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
    max_turns: int = 12,
    verbose: bool = False,
    use_status_bar: bool = True,
    use_compressor: bool = True,
    enable_memory: bool = True,
    memory_tools: dict | None = None,
    use_empty_arg_nudge: bool = True,
    tool_approver: Optional[Callable[[str, dict], Awaitable[bool]]] = None,
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
        tool_approver:  审批门。给"需人工批准"的工具（打上 __requires_approval__，
                        如真实工作区的 bash）用：执行前 await tool_approver(name, args)，
                        返回 True 才跑、False 则跳过并把"用户拒绝"作为工具结果喂回，
                        让 ReAct 循环继续而不是崩掉。None 时不做审批（全自动）。

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

    # 第 4.5 关(空参暖机柔推，可开关)：识别"同一工具·相同参数"连续重复空转，
    # 到第 2 次时注入温和纠正（见正文 while 里的 use_empty_arg_nudge 分支）。
    # dup_counter: (name, args_json) -> 连续出现次数；某轮没再出现的键会清零。
    dup_counter: dict[tuple[str, str], int] = {}

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

            # ── 审批门：对打上 __requires_approval__ 的工具（如真实工作区的 bash），
            #    执行前先征求用户批准。拒绝不当报错崩掉，而是喂回"用户拒绝"结果。
            executor = tool_executors.get(call["name"])
            if tool_approver is not None and executor is not None and getattr(
                executor, "__requires_approval__", False
            ):
                approved = await tool_approver(call["name"], call["arguments"])
                if not approved:
                    denied = (
                        "❌ 用户拒绝了此工具调用，【未执行】。因此这条命令没有任何副作用。"
                        "请换一种不修改文件/不跑 shell 的思路重试，"
                        "或向用户说明你想执行什么、以及为什么需要，再去请求批准。"
                    )
                    yield {"type": "tool_result", "name": call["name"], "result": denied}
                    messages.append(_tool_result_msg(call["id"], denied))
                    continue

            result = await _execute_one_tool(call["name"], call["arguments"], tool_executors)

            yield {"type": "tool_result", "name": call["name"], "result": result}
            messages.append(_tool_result_msg(call["id"], result))

        # ── 空参暖机柔推：同工具·同参数连续 ≥2 次 → 每轮注入一次温和纠正 ──
        # 软推不硬杀：同 grant 里 use_empty_arg_nudge 注释，避免误杀"先试后正"的运行。
        # 该 system 消息会随下轮请求进模型上下文，指引它把 required 参数填正。
        if use_empty_arg_nudge:
            seen_this_turn: set[tuple[str, str]] = set()
            guided = False
            for call in tool_calls:
                key = (
                    call["name"],
                    json.dumps(call.get("arguments") or {}, sort_keys=True, ensure_ascii=False),
                )
                seen_this_turn.add(key)
                dup_counter[key] = dup_counter.get(key, 0) + 1
                if dup_counter[key] >= 2 and not guided:
                    messages.append(_nudge_msg(call["name"]))
                    guided = True
            # 本轮没再出现的键清零，避免把上一任务的巧合带进新任务误判
            dup_counter = {k: v for k, v in dup_counter.items() if k in seen_this_turn}

        # 回到循环顶：带上更新后的完整历史再调模型，让模型自己决定是否继续

    # max_turns 兜底（真实防线，靠模型自觉不算数）
    yield {
        "type": "error",
        "message": f"Agent 达到最大轮次 ({max_turns})，强制终止。",
    }