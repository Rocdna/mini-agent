"""
记忆格式的统一接口 — 四种格式都实现这两件事

第三章的四种存储格式（Simple Notes / Enhanced Notes / JSON Cards / Advanced JSON Cards）
共享同一套接口，实验才能"切换格式、比对结果"。book 实验3-2 要求：
"每种模式各自提供记忆生成（分析会话、写入记忆）与记忆检索（按问题取回相关记忆）"

  生成 generate(conversations, ...)   : 看到一段/几段对话历史 → 提取并组织成某种格式的记忆
  检索 retrieve(memory, question)     : 给定一个问题 → 从该格式的记忆里返回对回答有用的串

第一阶段：纯代码规则（不调 LLM）。四格式差异只体现在"同一批事实被组织成什么结构"，
所以这里刻意把"事实抽取"抽成可复用的规则工具，让四种格式聚焦于"组织方式"而非"抽取"。
"""


class MemoryFormat:
    """四格式的公共基类。子类实现 generate / retrieve 两步。

    状态约定：实例化后 self.memory 保存这一轮生成的记忆（格式自定，见各格式）。
    """

    name: str = "base"  # 子类覆写，用于实验对比标识

    def __init__(self):
        self.memory = None  # 本格式生成的记忆（类型由子类决定）

    def generate(self, conversations):
        """从对话历史生成记忆。conversations: list[dict]，见 loader 格式。"""
        raise NotImplementedError

    def retrieve(self, question):
        """按问题从 self.memory 取回对回答有用的字符串。"""
        raise NotImplementedError