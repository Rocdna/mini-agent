"""触发验证：跑一条问题，看 Agent 是否【主动】调 search_knowledge 而非 web_search，并引用规范编号。

用法（支持传任意问题）：
  python -m tests.test_rag_trig "请审查：cursor.execute('SELECT * WHERE id='+uid)"
  不带参数则用默认的 SQL 注入审查问题。

模拟 main.do_agent 的组装，但非交互：跑完就退出，打印每步工具调用/结果/最终回答。
会调 embedding(query 嵌入) + LLM(生成)，按规矩先跟用户确认再跑。
"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from agent_loop import run_agent_loop

DEFAULT_QUESTION = (
    "请审查下面这条代码有没有安全问题，依据规范给出评价。\n"
    "```python\n"
    "name = request.args.get('name')\n"
    "row = db.execute('SELECT * FROM users WHERE name = ' + name).fetchall()\n"
    "```"
)


async def real_main():
    # 从 main 拿 system prompt 会触发其模块级 MemoryManager 创建，故延迟导入仅在使用时
    from main import AGENT_SYSTEM_PROMPT
    from tools import ALL_TOOLS

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    print(f"Q: {question}\n{'='*60}")
    async for ev in run_agent_loop(messages, ALL_TOOLS, verbose=True):
        et = ev["type"]
        if et == "tool_start":
            print(f"\n🔧 调用工具: {ev['name']} args={ev['arguments']}")
        elif et == "tool_result":
            print(f"   ↳ 结果片段={str(ev['result'])[:120]!r}")
        elif et == "done":
            print(f"\n\n[最终回答]\n{ev['content']}")
        elif et == "error":
            print(f"⚠ {ev['message']}")


if __name__ == "__main__":
    asyncio.run(real_main())