# mini-agent

一个手写的 mini-agent 练习项目，用于理解模型调用、工具执行、RAG、长期记忆和上下文管理，并将它们组合成终端代码审查助手。

## 已实现的功能

- ReAct 循环：模型选择工具、执行工具、结果回灌，以及最大轮次限制。
- 流式输出与调试事件：查看模型请求中的 messages、工具调用和上下文压缩。
- RAG：固定/递归分块、BM25、向量缓存、稠密检索和 RRF 混合检索。
- 长期记忆：工具驱动的事实提取，JSON 落盘、启动加载与上下文注入。
- 上下文工程：临时状态栏、工具次数统计、阈值触发压缩、重复调用提醒。
- 代码审查：检索带编号的审查规范，搜索和读取本地源码，以及执行、写入和修改文件的工具。

本项目是学习原型。已有功能与待完善项见本文最后一节。

## 安装与配置

需要 Python 3.10 或更高版本。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

编辑本地 `.env`：

```dotenv
DEEPSEEK_API_KEY=填写自己的密钥
DEEPSEEK_MODEL=填写账号可用的模型名称
```

密钥只保存在本地 `.env` 中。仓库提供的 `.env.example` 不包含真实密钥。

RAG 的稠密/混合检索以及 Agent 的 `search_knowledge` 工具还需要配置：

```dotenv
ARK_API_KEY=填写自己的密钥
ARK_BASE_URL=填写账号对应的 embedding API 基础地址
EMBEDDING_MODEL=填写可用的 embedding 模型或端点名称
```

代码会在基础地址后追加 `/embeddings`。第一次使用知识库会创建向量缓存，需要联网和可用的模型账户。

## 启动

```powershell
python main.py
```

终端命令：

| 命令 | 用途 |
| --- | --- |
| `/agent` | 切换到带工具调用的 Agent 模式 |
| `/chat` | 切换到普通聊天模式 |
| `/tool` | 查看已注册工具 |
| `/debug` | 切换模型请求快照显示 |
| `/memory` | 切换记忆注入和记忆工具 |
| `/quit` | 退出 |

示例：先输入 `/agent`，再输入 `请审查：cursor.execute("SELECT * FROM users WHERE id = " + user_id)`。

## 项目结构

```text
mini-agent/
├── main.py                  # 终端入口和代码审查提示词
├── agent_loop.py            # ReAct 循环与事件输出
├── api.py                   # DeepSeek HTTP/SSE 接口
├── tools.py                 # 工具注册与联网搜索
├── compressor.py            # 上下文压缩
├── status_bar.py            # 动态状态栏
├── code/                    # 搜索、读取、编辑、执行工具
├── memory/                  # 记忆存取、工具与提取提示词
├── rag/                     # 分块、检索、融合与知识库
└── tests/                   # 评估和演示脚本
```

`rag/data/` 是客服检索样例，`rag/data_review/` 是 Agent 使用的代码审查规范，`rag/data_minfa/` 是独立 RAG 问答示例的语料。

## 验证与评估

离线验证，不调用模型 API：

```powershell
python -m tests.eval_bm25
python -m tests.compare_chunkers
python -m status_bar
```

联网评估，需要对应的模型配置：

```powershell
python -m tests.eval_metrics
python -m tests.test_rag_trig
python -m tests.demo_memory_loop
```

这些主要是学习用的评估和演示脚本，部分结果需要人工检查。代码审查样例见 `tests/review_test_prompts.md`。

`tests.run_experiment` 的记忆实验依赖另行准备的教材评估集：`ai-agent-book/chapter3/user-memory-evaluation/test_cases/` 及其相邻的 `fixtures/gold_facts.json`。该第三方评估集没有随仓库发布。

## 当前限制与后续计划

- `execute_code` 使用普通子进程、临时目录和精简环境，并没有实现操作系统级的文件或网络隔离。仅在受信任的本地练习环境执行代码。
- 交互入口为写文件和真实工作区 shell 提供人工审批；直接复用 `run_agent_loop` 时需要显式传入审批器。
- 流式工具解析需要完善为按工具 `index` 分别重组多个调用。
- 压缩失败时的原文保留策略、短文本分块和向量缓存失效策略待修复。
- Notes 已接入主流程；JSON Cards / Advanced JSON Cards 的格式传递和上下文显示需要进一步完善。
- 跨轮工具证据保留，以及代码审查的自动评分与误报评估仍需补充。

本仓库保留了原练习工作区中涉及 `partical/` 的 12 个历史提交，并将该目录的文件提升到仓库根目录。历史提交保留原消息、作者及时间；由于目录路径与提交树变化，commit ID 已重新生成。最后一个新增提交包含根目录布局适配、发布配置和原本尚未提交的项目改动。

仓库不包含其他练习项目、Notebook、用户记忆或本地凭据。
