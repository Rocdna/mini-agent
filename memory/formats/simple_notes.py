"""
Simple Notes — 记忆格式一：每条一个最小事实（book 第95行）

设计哲学（对照 book）：
  极简主义。每条记忆是一个最小的、不可再分的事实，如 "用户邮箱：john@example.com"。
  优势 O(1) 极低开销；代价是信息关联性完全丢失——"在 TechCorp 担任高级工程师，
  负责推荐系统开发" 被拆成 3 条独立 note，内在联系被割裂。
  跨会话、消歧类查询（layer2/3）会因此失分。这正是本实验要验证的。

生成（提炼）走 LLM：把对话历史喂给 DeepSeek，让它提炼成若干条独立最小事实。
  这和 book 配套项目一致（agent.py 用 chat.completions 调 LLM 提炼，memory_manager 只负责存取）。
检索走代码：按问题关键词从 notes 里取回相关条（更快、省 API，book 配套也是这么做的）。

接口：generate 是 async（要调 LLM），retrieve 是同步（纯代码）。
"""

import re

from api import chat_with_tools
from memory.base import MemoryFormat

# 提炼最小事实的指令模板
PROMPT = """下面是用户与客服的对话历史。请把它提炼成若干条【不可再分的】最小事实（Simple Notes）。

要求：
- 每条是一句独立、可单独成立的事实，如 "用户邮箱：john@example.com"
- 事实要尽量少地粘连其他事实（同一条里的多个属性拆开）
- 只保留明确出现过的信息，不要推断、不要编造
- 每条单独一行，不要编号、不要其他解释

对话历史：
{history}"""


class SimpleNotes(MemoryFormat):
    name = "simple_notes"

    def __init__(self):
        super().__init__()
        self.notes: list[str] = []

    # ── 生成：LLM 提炼最小事实 → 存成 note ──
    # 工程约束：deepseek-v4-flash 对输入长度很敏感（越长越慢、越不稳，2000 字符会偶发超时）。
    # 实测 400~500 字符块稳定且快。所以用【小字符块 + 句子边界对齐】切分，避免硬切把事实截断。
    CHUNK = 500  # 每块目标字符数

    async def generate(self, conversations):
        """conversations: list[dict]，每个含 'messages'（role/content 消息）。
        拼成文本交给 LLM 分块提炼，返回的每行 = 一条 note。"""
        history = self._flatten(conversations)

        notes: list[str] = []
        for chunk in self._chunk_sentences(history, self.CHUNK):
            resp = await chat_with_tools([{"role": "user", "content": PROMPT.format(history=chunk)}])
            if resp["error"]:
                raise RuntimeError(f"Simple Notes 提炼失败: {resp['error']}")

            text = resp["content"] or ""
            for line in text.splitlines():
                line = self._clean(line)
                if line and not any(skip in line for skip in ("---", "```")):
                    if line not in notes:  # 跨块去重
                        notes.append(line)
        self.notes = notes
        return self.notes

    @staticmethod
    def _chunk_sentences(text: str, target: int) -> list[str]:
        """按句子边界把长文本切成 ~target 字符的块，不硬切句子避免截断事实。
        句子按换行(每轮消息) + 中英文句号切。"""
        # 先按换行切成"行"，再合并到不超过 target 的块
        lines = [ln for ln in text.split("\n")]
        chunks, buf = [], ""
        for ln in lines:
            if len(buf) + len(ln) + 1 > target and buf:
                chunks.append(buf)
                buf = ln
            else:
                buf = (buf + "\n" + ln) if buf else ln
        if buf:
            chunks.append(buf)
        return chunks

    @staticmethod
    def _flatten(conversations) -> str:
        """把多段会话压成纯文本历史：轮次 + 角色 + 内容。"""
        blocks = []
        for ci, conv in enumerate(conversations, 1):
            blocks.append(f"--- 会话 {ci} ---")
            for m in conv.get("messages", []):
                role = m.get("role", "?")
                content = str(m.get("content") or "")
                blocks.append(f"[{role}] {content}")
        return "\n".join(blocks)

    @staticmethod
    def _clean(line: str) -> str:
        """去掉 LLM 加的行首编号/项目符号/引号，保留纯事实文本。"""
        s = line.strip()
        s = re.sub(r"^[-*\d\.\)\s]+", "", s)  # 去 "- "、"1. "、"* "
        s = s.strip('"\'“”"')
        return s.strip()

    # ── 检索：按问题关键词取回相关 note（纯代码）──
    def retrieve(self, question):
        """在 self.notes 里找含问题关键词的 note，命中越多越相关，拼串返回。"""
        STOP = {"the", "a", "an", "my", "your", "what", "is", "are", "of", "for",
                "and", "it", "to", "in", "on", "we", "i", "with"}
        keys = [w for w in self._words(question) if w.lower() not in STOP and len(w) > 2]
        if not keys:
            keys = self._words(question)  # 全中文没有空格词，退化为整句

        scored = []
        for note in self.notes:
            hit = sum(1 for k in keys if k.lower() in str(note).lower())
            if hit:
                scored.append((hit, note))

        scored.sort(reverse=True)
        return "\n".join(note for _, note in scored)

    @staticmethod
    def _words(text: str) -> list[str]:
        """提取单词：英文按空格分词，中文按字符。"""
        # 保留英文/数字词
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*|\d{2,}", text)
        # 中文部分直接存整串（关键词匹配靠子串）
        cn = re.sub(r"[^，。一-鿿0-9A-Za-z]", "", text)
        result = words
        if cn and not any(k in cn for k in words):
            pass
        return result if words else [text]