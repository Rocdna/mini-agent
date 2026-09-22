"""
partical.code — 代码/文件系统工具子包（Coding Agent 的能力层）。

与 partical.rag、partical.memory 平级，把"agent 摸代码/读代码/跑代码"的
能力独立成包，工具实现均带 __tool_schema__，由 partical.tools 统一注册进 ALL_TOOLS。

模块划分：
  search.py      grep_files —— 在目录里递归搜文本/正则（按内容定位）
                 glob_files —— 按文件名模式递归列文件（列目录/找文件）
  read_write.py  read_file  —— 读文件内容带行号 + 截断（看清上下文）
                 write_file —— 在工作区新建/覆盖文件（需审批）
                 edit_file  —— 在工作区做精确片段替换（需审批）
  workspace.py   有界工作区路径解析（bash/write_file/edit_file 共用落点）
  execute.py     execute_code —— 受限沙箱子进程里跑 Python/Shell（实证代码）

安全分三档：
  search/read 纯只读、免审批；
  write/edit/bash 写入【有界工作区】真实文件，走 __requires_approval__ 审批门，
                路径经 workspace.resolve_within_workspace 钳制在 _coding_workspace 内；
  execute 只读验证，跑在瘦身 env + 独立临时目录 + 超时/截断的子进程里。
"""

from partical.code.execute import execute_code
from partical.code.read_write import read_file, write_file, edit_file
from partical.code.search import grep_files, glob_files

__all__ = ["grep_files", "glob_files", "read_file", "write_file", "edit_file", "execute_code"]