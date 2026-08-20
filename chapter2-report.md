# 第 2 章「上下文工程」源码教程报告

> 配套《深入理解 AI Agent》第 2 章。对应实验代码：`ai-agent-book/chapter2/`。
> 本篇按「一句话目的 → 入口与流程 → 核心机制 → 学习重点」组织，帮你从源码层面理解上下文工程。

---

## 0. 第 2 章主线：上下文决定能力上限

核心公式仍是 **Agent = LLM + 上下文 + 工具**。第 2 章回答一个关键问题：**同一个 LLM，为什么上下文组织方式不同，能力天差地别？**

本章 8 个实验项目正好覆盖上下文的四个层面：

| 层面 | 实验 | 项目 |
|---|---|---|
| 服务端上下文（KV Cache） | 2-2、2-3 | `attention_visualization`、`kv-cache` |
| 本地推理与工具调用底座 | 2-1 | `local_llm_serving` |
| 提示工程（上下文内容编排） | 2-4、2-5、2-6 | `prompt-engineering`、`prompt-injection`、`system-hint` |
| 上下文压缩（省 token） | 2-10 | `context-compression` |
| Agent Skills（上下文按需加载） | 2-6 | `agent-skills-ppt` |

---

## 1. 实验 2-1 · local_llm_serving — 本地 LLM 的工具调用底座

**一句话目的**：用统一 OpenAI 兼容接口接入 Ollama / vLLM 本地服务，演示「模型选工具 → 并行执行 → 结果回灌 → 循环到回答」的完整 Agent 循环（默认 Qwen3-0.6B）。

**入口与流程**：`main.py` → `ToolCallingAgent` 门面类自动探测后端（Linux+CUDA→vLLM，Windows/macOS→Ollama）→ 单任务/交互/流式三种模式。工具由 `tools.ToolRegistry` 统一注册（`tools.py`），它同时生成 OpenAI JSON Schema 和按名执行。

**核心机制（学习重点）**：

1. **工具定义 = JSON Schema**（`tools.py:111-123`）：模型看到的工具就是一段 schema（name/description/parameters），模型只负责"选名字填参数"；执行是宿主侧 `execute_tool(**arguments)` 完成的。

2. **ReAct 循环**（`agent.py:228-326`）：`while` 循环里：发消息 → 模型返回 tool_calls → 把 assistant 消息入历史 → 执行工具 → 结果回灌 → 再让模型决策，直到无工具调用则 `break`（上限 10 轮防死循环）。这是最小 Agent 主干。

3. **两种工具结果注入约定（重要对比）**：
   - vLLM 版：结果包成 `<tool_response>...</tool_response>` 以 `role:"user"` 回灌（Qwen3 惯例，`agent.py:310-315`）
   - Ollama 原生版：标准 `role:"tool"` 消息（`ollama_native.py`）
   - 同一 Agent 逻辑、不同服务约定不同——这是真实工程坑。

4. **流式工具调用的分片重组**（`agent.py:424-442`）：流式时一次工具调用的 id/name/arguments 被拆成多个增量，必须按 `fragment.index` 缓存再**字符串拼接**，否则调用残缺。

5. **并发工具调用**（`ollama_native.py:99-104`）：同一轮多个 tool_calls 用 `ThreadPoolExecutor.map` 并行执行，保证结果顺序与调用顺序一致；配套 `test_parallel_tools.py` 用两个 sleep 工具验证并行 < 1.8× 串行时间。

6. **本地服务接入**：Ollama 用 `ollama.Client()`（:11434）或 OpenAI 兼容 `/v1` 端点；vLLM 用 `server.py` 拉起 `python -m vllm.entrypoints.openai.api_server`，工具调用必须带 `--enable-auto-tool-choice --tool-call-parser hermes`。

> 💡 你的 `partical/agent.py` 正好是这一步的"手动版"：只定义了 `get_current_time` / `get_weather` 两个工具的 schema，还没有工具执行与循环。把它补成 `while` 循环 + `execute_tool` 就是 2-1 的迷你版。

---

## 2. 实验 2-2 · attention_visualization — 注意力热力图看清上下文

**一句话目的**：用真实小模型（Qwen3-0.6B）的自注意力热力图直观展示模型"如何使用上下文"——首 token 被过度关注（attention sink）、因果下三角、提示区 vs 生成区注意力分配。

**入口与流程**：
- `attention_cli.py`：最简入口，提示词 → 热力图 PNG
- `agent.py`：核心类库（`AttentionVisualizationAgent`），`python agent.py` 直接跑 5 类提示词演示
- `main.py`：ReAct 多步 Agent，每步 LLM 调用都记录注意力，存轨迹给前端

**核心机制（学习重点）**：

1. **拿到 attention 的钥匙**：`attn_implementation="eager"`（`agent.py:188-197`）——必须用 eager 模式才输出 attention 张量，flash/SDPA 不输出（CLI 会据此报错）。

2. **矩阵语义**：`outputs.attentions` 形状 `[层, batch, heads, seq, seq]`。**行 = Query 位置（谁在注意），列 = Key 位置（被注意者）**，每行 softmax 和为 1。取 `attentions[layer][0].mean(dim=0)` 头平均得 `[seq, seq]` 矩阵。

3. **两种取 attention 视角**：
   - 全序列单次 forward（`attention_cli.py:225-231`）：`output_attentions=True` 一次拿全矩阵
   - generate 生成期采集（`agent.py:404-411`）：每生成一步取**最后一行**（新 token 对所有历史 key 的注意力）——这正是 KV Cache 生成时注意力的真实语义

4. **因果三角**：`np.ma.array(matrix, mask=np.triu(ones, k=1))` 把结构为零的上三角标为 bad 渲染成浅灰，下三角亮区即"每行只关注自己及之前 token"。

5. **attention sink**：对每行算 `matrix[row, 0] / row.sum()`，Qwen3-0.6B 末层 sink 常占 75-85%——首 token（BOS）被过度关注，这是 KV Cache 工程（如保留首 token 的 cache）的重要依据。

> 📖 正文第 2 章解释：注意力可视化帮助你理解为什么"上下文越长、模型越难聚焦"——KV Cache 让历史可复用，但 attention 分布决定了哪些历史真正影响生成。

---

## 3. 实验 2-3 · kv-cache — 缓存命中与错误上下文模式

**一句话目的**：用 Kimi 推理模型的 `cached_tokens` 字段演示上下文缓存（服务端 KV Cache 复用），并构造 5 种"错误上下文"模式对比缓存命中率变化。

**入口**：`python main.py`（交互菜单）；`--mode correct|dynamic_system|shuffled_tools|dynamic_profile|sliding_window|text_format`；`--compare` 全部对比；离线 `--report` 用已存 JSON 出表（无需 key）。

**核心机制（学习重点）**：

1. **缓存命中信号**：响应里 `usage.cached_tokens > 0` 即命中缓存。前提：模型必须是会上报 cached_tokens 的当前 Kimi 推理模型（默认 `kimi-k2.6`；`moonshot-v1-*` 不上报无法演示）。

2. **正确模式**：固定 system prompt + 稳定前缀，第 1 轮后出现缓存 token、TTFT 更稳——这就是为什么工程上要把不变的指令放前面、变动的内容放后面。

3. **错误上下文模式 = 打破缓存前缀**：
   - `shuffled_tools`：工具顺序每轮打乱 → 前缀变化 → 缓存命中率 10.1%→3.4%
   - `dynamic_system`：system prompt 动态变化 → 缓存失效
   - `text_format`：格式微变 → 总时间约 2.4×（缓存全失效的代价）
   - 结论：**上下文组织方式直接影响成本和延迟**。

4. **推理模型限制**：Kimi 推理模型只接受 `temperature=1`（代码自动处理）——这是 API 层面对推理模型的约束。

---

## 4. 实验 2-4 · prompt-engineering — 提示工程消融

**一句话目的**：在 Tau-Bench 客服场景上做系统提示的消融实验——基线 vs 语气风格 / wiki 知识组织 / 工具描述缺失，量化各成分对成功率的影响。

**入口**：`python run_ablation.py --model ... --env airline --end-index 10 --all`；规范运行用官方 Kimi K3 6×10 对照。

**核心机制（学习重点）**：

1. **消融维度**：`trump_tone`（特朗普式语气）/ `casual_tone` / `wiki_random`（知识打乱）/ `no_tool_desc`（去掉工具描述）/ `all_ablations`——每个维度独立开关，与 baseline 对比成功率。

2. **评测闭环**：Tau-Bench 内置任务集，agent 跑任务 → 计算 reward（0/1）→ `analyze_results.py` 出对比表 + 严重度 + 交互效应。

3. **重要认知（README 已明示）**：历史 30%/45% 的点估计**未被复现**——实验结论要看方向信号而非绝对数字。这是复现实验的诚实态度。

---

## 5. 实验 2-5 · prompt-injection — 提示注入攻防

**一句话目的**：构造 3 类攻击 × 4 层防御的矩阵（3×4=12 组合 × 多次试验），统计攻击成功率，理解"为什么提示词防御不是最后一道防线"。

**入口**：`python demo.py`（全部组合）；`-a 2,3 -d 1,4` 子集；`-m gpt-4o-mini` 换模型。

**核心机制（学习重点）**：

1. **三类攻击 = 三条注入通道**：
   - 直接注入：用户消息里藏指令
   - 间接注入：外部网页内容（攻击者可控）中段藏指令
   - 记忆注入：跨会话持久记忆被污染

2. **四层防御（纵深思想）**：
   - D1：基础安全规则
   - D2：提示词加固——声明"外部内容只是数据不是指令"
   - D3：来源标记——工具返回包 `<external_content source="...">` XML，做数据/指令通道分离（`agent.py:230-237`）
   - D4：**运行时强制校验**（`agent.py:268-277`）——即使模型被说服调用高风险工具，地址不在当前用户消息里直接拦截。核心认知：**提示词防御可被模型说服绕过，最后一层必须是不可被说服的运行时校验**。

3. **确定性 judge**：用字符串/工具调用记录判定成功与否，零额外 LLM 成本、可复现——评测体系远比"看最终回复"稳健。

4. **模型选择影响结论**：太强的模型 D1 就全 0%，会抹平防御层对照曲线——教学上故意用弱基线 `gpt-4o-mini`。

---

## 6. 实验 2-6 · system-hint — 动态状态栏注入

**一句话目的**：演示"状态栏（System Hint）"机制——把时间戳、工具调用计数、TODO 列表、详细错误、系统状态等动态信息，在每次 LLM 调用前以**临时 user 消息**注入上下文末尾，改善轨迹质量、抑制死循环。

**入口**：`python main.py --mode preview`（本地渲染对比，**无需 API key**，零成本入门）；`run_experiment_2_8.py` 正式消融。

**核心机制（学习重点）**：

1. **注入方式（本实验最值得学的一招）**（`agent.py:858-862`）：
   ```python
   messages_to_send = self.conversation_history.copy()
   system_hint = self._get_system_hint()
   if system_hint:
       messages_to_send.append({"role": "user", "content": system_hint})
   ```
   状态栏是**临时 user 消息、不写入历史**——① 不污染永久上下文 ② 不破坏已缓存的 KV Cache 前缀 ③ 追加在最末尾、紧邻待生成 token、注意力权重最高。对照"全写死进 system prompt"的 token 成本与污染问题。

2. **五个 feature = 五类问题的针对性解**：时间戳（时序感知）、工具调用号（防死循环）、TODO 列表（任务管理）、详细错误+修复建议（避免盲目重试）、系统状态（平台感知）。

3. **消融方法论（更值钱）**：预注册协议、同一 case 两臂完全相同的匹配对照、arm 奇偶交替排序、**客观打分只依据 tool_events + 沙箱文件状态（不信模型自述）**、断点续跑 + 证据哈希。

---

## 7. 实验 2-10 · context-compression — 上下文压缩策略对比

**一句话目的**：在"研究 OpenAI 联创当前任职"任务上，对比 6 种上下文压缩策略的 token 成本 / 延迟 / 溢出 / 相关性，理解为什么大上下文（128K+）下"管好上下文"比"装下所有东西"更重要。

**入口**：`python experiment.py -s context_aware`（对比表 + JSON）；`python quickstart.py`（菜单）；`python main.py`（交互/单策略）。

**核心机制（学习重点）**：

**6 种策略**（`compression_strategies.py::ContextCompressor`）：

| 策略 | 做法 | 取舍 |
|---|---|---|
| 1. no_compression | 全文塞入 | 基线：几次工具调用后溢出 |
| 2. individual（非上下文感知） | 每页独立 LLM 摘要后拼接 | 保留页级细节，丢跨页联系 |
| 3. combined（非上下文感知） | 全部拼接后一次摘要 | 全局图景好，丢单页归属 |
| 4. context_aware（上下文感知） | 针对用户问题的聚焦摘要 | 相关性好，多一次 LLM 调用 |
| 5. citations（上下文感知+引用） | 4 + 来源链接 | 适合追问，体积略大 |
| 6. windowed（窗口化） | 最近工具调用全文，旧历史压缩 | 细节与效率平衡；只压缩未标 `[COMPRESSED]` 的消息 |

**实验设计亮点**：默认 `CONTEXT_WINDOW_SIZE = 128K`（demo 强制 cap 让溢出/压缩可观察，即使真实模型窗口 1M）——让"溢出问题"从不可见到可见，这是实验设计技巧。

---

## 8. 实验 2-6 补充 · agent-skills-ppt — Agent Skills 渐进式披露

**一句话目的**：演示 Agent Skills 机制——Agent 启动只看到各 Skill 的 name/description 薄目录，判断需要后按需加载 `SKILL.md` 正文、子文档与脚本，最终自动生成 PPT。

**核心机制（学习重点）**：

1. **Skill = 目录**：`SKILL.md`（YAML frontmatter 元数据 + 正文流程）+ 子文档 `reference.md` + 捆绑脚本 `scripts/`。

2. **三层渐进式披露**（上下文经济性）：
   - L1：仅 name+description 薄目录进 system prompt（几百 token）
   - L2：模型调用 `read_skill("pptx")` 加载完整 SKILL.md
   - L3：`read_skill_file` 读细则 / `run_skill_script` 执行脚本

3. **与 tool 的区别**：工具是常驻原子操作；Skill 是**打包的领域工作流**，按需披露——启动不把所有知识塞进 system prompt，省 token、避免干扰。

4. **frontmatter 是路由契约**：`description` 里写明 `Use when ... / Don't use when ...`，让模型决定何时加载。

5. **机制演示 ≠ 实验验收**：本地 demo 是同构机制的离线教学具，正式证据要求官方 Skill 固定 revision + 真实论文 + 原始 stream 事件 + 客观 artifact 门禁。

---

## 9. 学习路径建议（按你的进度）

你现在在 `partical/` 手动复现工具定义（`agent.py`），建议按这条线推进：

1. **先跑通 2-1**（本地 Ollama 或任选 API）：把 `partical/agent.py` 补成完整 ReAct 循环——加 `while`、加 `execute_tool`、加结果回灌。**这比看代码重要，动手写一遍才懂**。
2. **再看 2-3**（KV Cache）：理解"为什么固定前缀放前面"是工程铁律。
3. **然后 2-5**（注入防御）：理解"运行时校验是底线"——Agent 安全的第一课。
4. **最后 2-10**（压缩）和 **2-6**（Skills）：理解"上下文管理"的大图景。
5. 想加深印象，跑 `system-hint` 的 `--mode preview`（无需 key）看状态栏注入的对比。

每章的详细命令、Key 配置见 `ai-agent-book/docs/runbook/chapter2.md`。

---

## 10. 常见疑问速查

| 疑问 | 答案 |
|---|---|
| 为什么工具描述要写成 JSON Schema？ | 模型只做"选工具填参数"的决策，schema 是它唯一的工具说明书；执行在宿主侧，可校验可拦截 |
| 缓存命中率为什么重要？ | 命中 = 跳过前缀重算，省时间省成本；上下文组织方式直接决定命中率（2-3 数据：10.1%→3.4%） |
| 为什么提示词防御不够？ | 提示词是可被"说服"的软约束，模型可能被注入内容诱导违反；D4 运行时校验是模型无法绕过（或不计入）的硬约束 |
| 上下文压缩是不是越省越好？ | 不是：individual 丢跨页联系、combined 丢归属，要按任务相关性取舍；窗口化策略是工程常用折中 |
| 注意力可视化有什么用？ | 理解 KV Cache 为什么能复用历史（每步只算新 token 对历史 key 的注意力）、attention sink 为什么让首 token 特殊 |
