"""记忆闭环演示：两段「会话」完整走一遍 沉淀→落盘→跨会话召回→凭记忆答。

复刻 main.do_agent 的组装（AGENT_SYSTEM_PROMPT + <user_memory> 注入 + 记忆工具并入），
但用一个独立 user_id=demo_memory 的 MemoryManager，避免污染真实 default 记忆文件。

会话1：用户说出个人信息 → 期待 agent 主动调 add_memory 落盘到 demo 文件。
会话2：全新 MemoryManager(模拟重启，从磁盘读回) 注入已存记忆 → 期待 agent 凭记忆回答。

两段各自独立 run_agent_loop，每段会调 LLM。按规矩与用户确认后跑。
"""
import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from agent_loop import run_agent_loop
from memory.memory_manager import MemoryManager
from memory.tools import build_memory_tools

FORMAT = "enhanced_notes"
USER = "demo_memory"
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_tmp")

SESSION1_USER = "你好，我是王小明，在慢语科技做后端工程师，邮箱 wxm@example.com，平时喜欢研究 SQL 性能优化。"
SESSION2_USER = "还记得我吗？我叫什么、做什么的、邮箱是多少？"


def make_mgr() -> MemoryManager:
    # storage_dir 指向一块临时目录，读完即测，不碰真实 default 记忆
    return MemoryManager(user_id=USER, storage_dir=TMP_DIR, filename_key=FORMAT)


def assembly(mgr, history):
    """等价 main.do_agent：注入记忆上下文 + 并进记忆工具，返回 (loop_messages, tool_executors)。"""
    from main import AGENT_SYSTEM_PROMPT
    from tools import ALL_TOOLS

    ctx = mgr.context_string()
    memory_part = "" if ctx == "(无记忆)" else f"\n<user_memory>\n{ctx}\n</user_memory>\n"
    system_content = AGENT_SYSTEM_PROMPT + memory_part
    loop_messages = [{"role": "system", "content": system_content}, *history]
    mem_tools = build_memory_tools(mgr, FORMAT)
    return loop_messages, ALL_TOOLS, mem_tools


async def run_session(label, messages):
    print(f"\n{'='*70}\n[{label}]\n用户: {messages[-1]['content']}\n{'='*70}")
    mgr = make_mgr()  # 每个"会话"一个全新 manager（模拟跨进程/天重启）
    loop_messages, tools, mem_tools = assembly(mgr, messages)
    used = {"add_memory": False, "update_memory": False, "delete_memory": False, "other": []}
    answer = ""
    async for ev in run_agent_loop(loop_messages, tools, enable_memory=True,
                                   memory_tools=mem_tools):
        et = ev["type"]
        if et == "tool_start":
            n = ev["name"]
            if n in used:
                used[n] = True
            else:
                used["other"].append(n)
            print(f"  → 调用工具: {n} {ev['arguments']}")
        elif et == "tool_result":
            print(f"    ↳ 结果: {str(ev['result'])[:80]}")
        elif et == "done":
            answer = ev["content"]
            print(f"  [最终回答]\n{ev['content']}")
        elif et == "error":
            print(f"  ⚠ {ev['message']}")
    return used, answer, mgr


async def main():
    # ── 会话1: 沉淀 ‑‑
    used1, ans1, mgr1 = await run_session("会话1·沉淀记忆", [
        {"role": "user", "content": SESSION1_USER},
    ])
    print(f"\n> 会话1 是否触发 add_memory: {used1['add_memory']}"
          f" | 磁盘落盘 items={len(mgr1.items)} @ {os.path.basename(mgr1.memory_file)}")

    # ── 会话2: 召回(全新 manager 读回磁盘) ─‑
    print(f"\n>>> 跨会话: 新 MemoryManager 从磁盘读回 {mgr1.items and len(mgr1.items)} 条记忆")
    used2, ans2, mgr2 = await run_session("会话2·凭记忆回答", [
        {"role": "user", "content": SESSION2_USER},
    ])
    print(f"\n> 会话2 是否又 add_memory: {used2['add_memory']}")

    ok = used1["add_memory"] and not used2["add_memory"] and \
        ("王小明" in ans2 or "后端" in ans2 or "wxm" in ans2)
    print(f"\n{'#'*70}\n闭环判定: 会话1 沉淀 {'✓' if used1['add_memory'] else '✗'}，"
          f"会话2 凭记忆答出身份/职业/邮箱 {'✓' if ok else '✗（需人工看回答）'}\n{'#'*70}")


if __name__ == "__main__":
    asyncio.run(main())