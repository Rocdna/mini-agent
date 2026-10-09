"""
compressor — 上下文压缩（第 4 关）

三部件，对应教程：
  ① estimate_tokens(messages)        估算长度（练手用字符数，生产换 tiktoken）
  ② should_compress(messages)        阈值触达才压（80% 预算）；别每轮都压（压一次缓存失效一次）
  ③ compress_tool_results(messages, task)
                                     只压 role=="tool" 的老搜索结果：
                                     把多条未压缩的 tool 内容合并成一份任务感知摘要，
                                     原地替换（位置不变），其余 tool 消息打 [COMPRESSED] 占位防重。

书里冠军思路：自适应窗口化（阈值触发+批量压缩+防重复标记）+ 上下文感知（摘要带任务意图）。

挂接方式（agent_loop 每轮调用前）：
    if should_compress(messages):
        messages = await compress_tool_results(messages, 用户原始问题)
"""

import asyncio
import logging

from api import chat_with_tools

logger = logging.getLogger("compressor")

# 压缩预算（字符）：历史总长度超过 BUDGET*THRESHOLD 就触发压缩。
# 练手用字符数估算；生产上应该用 tiktoken 数精确 token。
BUDGET = 30000
THRESHOLD = 0.8

COMPRESSED_TAG = "[COMPRESSED]"  # 防重复标记：已压过的消息看到它就不再处理


def estimate_tokens(messages: list[dict]) -> int:
    """粗略估算整个 messages 的 token 数（中文字符≈1 token，取一半）。"""
    return sum(len(str(m.get("content") or "")) for m in messages) // 2


def should_compress(messages: list[dict]) -> bool:
    """历史长度是否超过预算的 80%——只有这时才值得付出一次缓存失效的代价。"""
    return estimate_tokens(messages) > BUDGET * THRESHOLD


async def _llm_summarize(contents: list[str], task: str) -> str:
    """默认摘要实现：调一次 LLM，按用户任务意图提炼。

    任务感知是关键（书策略四）：摘要 prompt 里明确交代用户要什么，
    这样压缩才不会丢掉早期决策、关键数字和失败路径。
    """
    payload = "\n\n---\n\n".join(contents)
    prompt = (
        f"用户任务：{task}\n"
        "下面是一批工具返回的原始内容。请把它们压缩成一份事实列表：\n"
        "- 保留与任务相关的数字 / 名称 / 结论 / 决策与失败的路径\n"
        "- 丢弃导航噪声、重复描述、无关细节\n"
        f"以 '{COMPRESSED_TAG}' 开头。\n\n原始内容：\n{payload[:8000]}"
    )
    resp = await chat_with_tools([{"role": "user", "content": prompt}])
    if resp["error"]:
        logger.warning(f"压缩摘要调用失败: {resp['error']}，保留原文")
        # 摘要失败不能炸掉系统：只标记已知压缩过，后续不会重复处理
        return COMPRESSED_TAG + "（摘要失败，保留原文要点较长的超链接）"
    return (resp["content"] or COMPRESSED_TAG + "（无要点）").strip()


async def compress_tool_results(
    messages: list[dict],
    task: str,
    summarize=None,
) -> list[dict]:
    """把未压缩的 tool 消息合并成一份任务感知摘要，原地替换。

    Args:
        messages: 对话历史（本函数会原地修改其中的 tool 消息 content）
        task: 用户原始问题，用来引导"按任务留什么"
        summarize: 摘要函数 summarize(contents, task) -> str；
                   默认走真实 LLM。测试可注入 stub 离线演示（不花钱）。

    Returns:
        处理后的 messages（同一个列表对象）。
    """
    if summarize is None:
        summarize = _llm_summarize

    # ① 找出所有"还没压过"的 tool 消息
    targets = [m for m in messages
               if m.get("role") == "tool"
               and COMPRESSED_TAG not in (m.get("content") or "")]
    if not targets:
        return messages  # 没有可压的，原样返回

    contents = [str(t.get("content") or "") for t in targets]

    # ② 一次 LLM 调用，把全部内容合并成一份摘要
    #    兼容同步/异步注入：测试传 lambda 也行，生产传 async 也行
    res = summarize(contents, task)
    summary = await res if asyncio.iscoroutine(res) else res

    # ③ 原地替换：位置不变！system 与以往历史不动。
    #    第一条 tool 消息放完整摘要，其余打占位标签（防重复 + 不拆结构）。
    first = targets[0]
    first["content"] = summary
    for m in targets[1:]:
        m["content"] = f"{COMPRESSED_TAG}（内容已并入上一条摘要）"

    logger.info(f"压缩完成：{len(contents)} 条 tool 消息 → 1 条摘要")
    return messages


# ── 离线演示（python -m compressor）───────────────────
if __name__ == "__main__":
    import asyncio

    async def _demo():
        from tools import ALL_TOOLS  # noqa: F401
        # 造几条"超长搜索结果"样的 tool 消息
        fake = [
            {"role": "user", "content": "对比 langchain 和 llama-index 做 RAG 谁更强？"},
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "tool_call_id": "1",
             "content": "LangChain 是通用 LLM 应用框架…" + "导航栏 首页/文档/教程 页脚 广告 " * 5000},
            {"role": "tool", "tool_call_id": "2",
             "content": "LlamaIndex 专注 RAG 数据框架…" + "相关推荐 相似文章 底部链接 " * 5000},
        ]
        before = estimate_tokens(fake)
        print(f"压缩前估算: {before} tokens | 触发压缩? {should_compress(fake)}")
        out = await compress_tool_results(
            fake, "对比两个框架做RAG", summarize=lambda c, t: f"{COMPRESSED_TAG} LangChain:通用框架; LlamaIndex:专注RAG。结论:看场景选")
        after = estimate_tokens(out)
        print(f"压缩后估算: {after} tokens ({before - after} 省)")
        print(f"第一条 tool: {out[2]['content'][:80]}...")
        print(f"第二条 tool: {out[3]['content']}")

    asyncio.run(_demo())