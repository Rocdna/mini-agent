"""
StatusBar — Agent 状态栏（第 3 关）

核心认知（教程原话）：上下文学习是**检索不是推理**——模型擅长在上下文里"找"，
不擅长"数"。所以要"已经搜了几次"这种事，别指望模型自己数，用**代码**算好，
渲染成一段文字塞到模型眼前。它不用算，直接读。

用法（在 agent_loop.py 里挂）：
    bar = StatusBar()
    bar.on_turn()                                  # 每开一轮
    bar.on_tool_call("web_search")                 # 每个工具执行前
    msgs_to_send = messages + [{"role": "user", "content": bar.render()}]
    # 注意：render() 结果只进本次请求，不写进 messages 历史（不污染 + 追加在末尾）
"""


class StatusBar:
    def __init__(self):
        # 每个工具被调用的次数，比如 {"web_search": 2}
        self.tool_counts = {}
        # 当前是第几轮（对应 run_agent_loop 的 for turn）
        self.turn = 0

    def on_tool_call(self, name: str):
        """某工具被调用一次 → 计数器 +1。

        用 dict.get(name, 0)+1 而不是直接 self.tool_counts[name]+=1，
        因为工具第一次出现时键还不存在，get 的默认值 0 兜住了这个情况。
        """
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1

    def on_turn(self):
        """每向模型发起一轮请求前调用 → 轮次 +1。"""
        self.turn += 1

    def render(self) -> str:
        """把当前状态渲染成一段自然语言，拼到本轮请求最末尾给模型看。

        设计原则：这是给模型"检索"的信息，所以写成它最好读的样子——
        - 用 <agent_status> … </agent_status> 画清边界（标签名自带语义）；
        - 每个工具一行，计数到 3 次及以上时亮明"已重复，考虑换策略"，
          诱导模型换关键词 / 主动收尾（这就是治"反复搜索"的关键）；
        - 只给结论性的数字，不给过程，避免噪音稀释注意力。
        """
        lines = ["<agent_status>", f"当前轮次: {self.turn}"]
        for name, n in sorted(self.tool_counts.items()):
            warn = "（已重复调用，考虑换策略）" if n >= 3 else ""
            lines.append(f"- 工具 '{name}' 已调用 {n} 次{warn}")
        lines.append("</agent_status>")
        return "\n".join(lines)


# ── 快速自测（跑 python -m partical.status_bar）────────────────
if __name__ == "__main__":
    bar = StatusBar()
    bar.on_turn()
    bar.on_tool_call("web_search")
    bar.on_turn()
    bar.on_tool_call("web_search")
    bar.on_turn()
    bar.on_tool_call("web_search")  # 第 3 次 → 触发换策略提示
    print(bar.render())