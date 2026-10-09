"""
rag_answer — 把 RAG 检索接进 LLM 生成（RAG 闭环的第一步）

作用：给一个问题，返回"基于指定知识库（默认《民法典》）检索 top-k 片段 + LLM 基于它回答"的答案。
这是 RAG 从"检索练习"变成"能回答问题"的那一步——之前只打印 top-3 给评估看，
现在第一次让检索结果进入 LLM 上下文、真正生成答案。
语料可切换：默认 DEFAULT_CORPUS=《民法典》(910 chunk)，也可传 corpus_root 换成别的大文档。

用法：
  python -m rag.rag_answer "退款多久到账？"

关键点：
  - 复用 hybrid_search.HybridIndex（民法典缓存命中，不重嵌知识库）
  - 每次 `search` 只嵌这条 query（一次 OpenAI 兼容 embedding API）
  - 让 LLM 只基于检索到片段回答，资料没有就明说不知道（防幻觉）
"""

import asyncio
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# 默认语料：民法典（真实大文档测试）。也可指定其他目录。
DEFAULT_CORPUS = Path(__file__).resolve().parent / "data_minfa"

# Agent 进 search_knowledge 的知识库：代码审查规范（贴合"代码审查助手"人设）。
# 与 DEFAULT_CORPUS(dist 上的 rag_answer 直接问答用民法典) 分开，互不覆盖。
REVIEW_CORPUS = Path(__file__).resolve().parent / "data_review"

# 进程级缓存：每个语料构建一次索引，后续问答复用
_built_index: dict[str, object] = {}


async def _ensure_built(corpus_root: str | Path | None = DEFAULT_CORPUS):
    """懒构建：首次提问才加载知识库索引，之后复用（按语料区分）。"""
    from rag.hybrid_search import build
    name = str(corpus_root or DEFAULT_CORPUS)
    if name not in _built_index:
        _built_index[name] = await build(corpus_root)
    return _built_index[name]


# ── 知识库问答工具（供 Agent 调用，仿 tools.py 的 __tool_schema__ 用法）──

async def search_knowledge(args: dict) -> str:
    """检索知识库（代码审查规范），返回最相关的 top-k 片段（带来源）。

    Agent 调用该工具时，把抓到的片段拼进后续请求，即可"基于知识库回答"。
    这是 RAG 接入 mini-agent 的入口（第二个里程碑）。知识库指向 REVIEW_CORPUS。
    """
    query = (args.get("query") or "").strip()
    if not query:
        return ("错误：query 参数必填且不能为空。请把【用户的原问题】原样作为 query 传入"
                "（例如 query='这条代码的 SQL 拼接安全吗？'），不要留空。")

    idx = await _ensure_built(REVIEW_CORPUS)
    hits = await idx.search(query, top_k=3)

    if not hits:
        return f"知识库未找到与「{query}」相关的内容。"

    body = "\n\n".join(f"[来源 {h['doc']}] {h['chunk_text']}" for h in hits)
    return (
        f"<external_content source='knowledge_base'>\n"
        f"关于「{query}」从知识库检索到 {len(hits)} 条相关片段：\n\n{body}\n"
        f"</external_content>"
    )


search_knowledge.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "检索『代码审查规范』知识库，返回最相关的 top-3 条规范片段（带章节编号如 R6.1）。\n"
            "当你在评审代码、需要确认风格/复杂度/安全/性能/测试等规范依据时调用，\n"
            "以便引用规范编号（如 R6.1）评价代码，而不是凭印象。返回的片段带 [来源] 文件名。\n"
            "注意：query 必填，请填入要查的具体规范问题（如'SQL注入防护规则'）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "想查询的规范问题，如：SQL注入防护规则？",
                }
            },
            "required": ["query"],
        },
    },
}


# ── 纯 RAG 问答（不依赖 Agent，直接"检索→生成"）──

async def rag_answer(question: str) -> str:
    """给定问题，检索知识库 top-3 片段，让 LLM 基于片段生成答案。"""
    from api import chat_stream  # 复用你现有的 LLM 流式封装

    # 1) 检索：RRF 混合检索 top-3
    idx = await _ensure_built()
    hits = await idx.search(question, top_k=3)
    if not hits:
        return "知识库中未找到相关内容,我无法回答。"

    # 2) 注上下文：把片段拼进 system(让 LLM 只根据资料答,不能编)
    evidence = "".join(
        f"[片段-{i}(来源 {h['doc']})] {h['chunk_text']}\n"
        for i, h in enumerate(hits, 1)
    )
    system = (
        "你是法律助手。请【只根据】下面提供的《民法典》片段回答用户问题。\n"
        "若片段里没有答案,直接说明'资料里没有',不要编造法条。\n"
        "回答时尽量引用片段来源。\n\n"
        f"===== 知识库片段 =====\n{evidence}\n===== 片段结束 =====\n"
    )

    # 3) 生成：LLM 基于注入的上下文回答
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    out: list[str] = []
    # chat_stream 直接逐 token yield 裸字符串（不是 _with_tools 的 dict 事件），
    # 所以把每个产出的 str 原样收集。掉线时它会 yield 一条 "[错误]..." 字符串。
    async for token in chat_stream(messages):
        out.append(token)
    return "".join(out)


async def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "退款多久到账？"
    print(f"\nQ: {q}\n")
    print("A:", await rag_answer(q))
    print()


if __name__ == "__main__":
    asyncio.run(main())