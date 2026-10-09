"""
自测脚本 — 验证 ReAct 循环调用的四种典型路径。

用法（在仓库根目录）:
    python -m selftest [全部]
    python -m selftest 2     # 只跑第 2 条

每条都会打印事件轨迹 + 统计，用来判断循环是否按预期工作。
"""

import asyncio
import sys

from agent_loop import run_agent_loop
from tools import ALL_TOOLS

SYSTEM = "你是助手，需要实时/外部信息就调用工具，需要本地时钟就调 get_current_time，NEVER 编造。"

# 四条路径：(标题, 用户问题, 期望的行为)
CASES = [
    ("1️⃣ 无需工具-纯回答", "什么是 Python 的闭包？", "直接给答案，0 次工具调用，一轮 done"),
    ("2️⃣ 单工具单次", "现在几点？", "调 1 次 get_current_time，然后 done"),
    ("3️⃣ 跨工具组合(可能并行)", "东京现在几点、气温多少度？", "调 time + temperature 各 1 次，再 done"),
    ("4️⃣ 需要联网搜索", "DeepSeek 最新模型的上下文长度是多少？", "调 web_search，基于结果回答"),
    ("5️⃣ max_turns 兜底", "告诉我 2025 年每天的天气", "搜索反复停不下来 → error 强制终止"),
]


async def run_one(index: int, title: str, question: str, expect: str, max_turns: int = 3):
    print(f"\n{'=' * 60}\n{title}\n  期望: {expect}\n{'-' * 60}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    tool_calls, events = 0, 0
    final = ""
    async for ev in run_agent_loop(msgs, ALL_TOOLS, max_turns=max_turns):
        t = ev["type"]
        events += 1
        if t == "tool_start":
            tool_calls += 1
            print(f"  [tool] {ev['name']}{ev['arguments']}")
        elif t == "tool_result":
            print(f"  [result] {ev['result'][:70]}...")
        elif t == "done":
            final = ev["content"]
            print(f"  [done] {final[:120]}")
        elif t == "error":
            print(f"  [error] {ev['message']}")

    # 统计（大致判定）：有 done 内容=正常收尾，否则是被顶格
    outcome = "error(被顶格)" if not final else "done(正常收尾)"
    print(f"  ── 统计: 工具调用 {tool_calls} 次 | 事件数 {events} | 结果 {outcome}")


async def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else ["全部"]
    if "全部" in which:
        targets = list(range(len(CASES)))
    else:
        targets = [int(w) for w in which if w.isdigit()]

    for i in targets:
        title, q, expect = CASES[i]
        await run_one(i, title, q, expect, max_turns=3)


if __name__ == "__main__":
    asyncio.run(main())