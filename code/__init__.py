"""
partical.code — 代码/文件系统工具子包（Coding Agent 的能力层）。

与 partical.rag、partical.memory 平级，把"agent 摸代码/读代码/跑代码"的
能力独立成包，工具实现均带 __tool_schema__，由 partical.tools 统一注册进 ALL_TOOLS。

模块划分：
  search.py      grep_files —— 在目录里递归搜文本/正则（按内容定位）
                 glob_files —— 按文件名模式递归列文件（列目录/找文件）
  read_write.py  read_file  —— 读文件内容带行号 + 截断（看清上下文）
  execute.py     execute_code —— 受限沙箱子进程里跑 Python/Shell（实证代码）

安全：search/read 只读；execute 在瘦身 env + 独立临时目录 + 超时/截断的
子进程里执行，不碰真实项目文件与用户凭据。
"""

from partical.code.execute import execute_code
from partical.code.read_write import read_file
from partical.code.search import grep_files, glob_files

__all__ = ["grep_files", "glob_files", "read_file", "execute_code"]