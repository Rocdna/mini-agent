"""
代码/文件系统工具 · 读写(Read/Write)
────────────────────────────
read_file  —— 读取文件内容（带行号 + 截断）。纯只读、离线、安全、免审批。
write_file —— 在【有界工作区】新建/覆盖文件，需审批（__requires_approval__）。
edit_file  —— 在【有界工作区】已有文件里做精确片段替换，需审批。
写工具沿用与世界里的 bash 相同的安全口径：审批门是墙、路径圈定在工作区内。
"""

import os

from partical.code.workspace import workspace_path, resolve_within_workspace

# 读取时单次返回的最大字符数；超长文件截断，防止占爆上下文
_MAX_READ_CHARS = 6000


async def read_file(args: dict) -> str:
    """
    读取指定文件的文本内容，每行带行号；超长时截断并提示剩余行数。

    纯只读，安全；用 os.path 限制路径，不做任何写入。
    """
    file_path = args.get("file_path", "")
    if not file_path:
        return "错误：请提供 file_path 参数。"

    fpath = os.path.abspath(file_path)
    if not os.path.isfile(fpath):
        return f"错误：文件不存在或不是普通文件「{fpath}」"

    try:
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError as e:
        return f"错误：无法读取文件「{fpath}」—— {e}"

    if not lines:
        return f"文件「{fpath}」为空（0 行）。"

    # 带行号拼出来，同时做总长截断
    out = []
    used = 0
    for i, line in enumerate(lines, 1):
        num = f"{i:6d} │ "  # 行号右对齐 + 竖线分隔,一眼看清
        chunk = num + line.rstrip("\n")
        used += len(chunk)
        if used > _MAX_READ_CHARS:
            remaining = len(lines) - i + 1
            out.append(f"\n…（已截断，后续还有约 {remaining} 行未显示）")
            break
        out.append(chunk)

    head = "\n".join(out)
    return (
        f"文件「{fpath}」共 {len(lines)} 行，已显示 {len(out)} 行：\n\n"
        f"```\n{head}\n```"
    )


read_file.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "读取指定文件的文本内容，每行带行号返回；超长文件会截断并提示剩余行数。\n"
            "Use when: 需要看某个文件到底写了什么——完整逻辑、某个函数/定义的上下文。"
            "配 grep_files 用：grep 定位『在哪几行』，read_file 看清『那几行前后是什么』。"
            "Don't use when: 只想搜某段文本在哪(那用 grep_files)、或文件很大只要片段。\n"
            "Example: file_path='partical/main.py'。\n"
            "常见错误: 路径要是文件而不是目录；给出相对项目根或绝对路径均可。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "要读取的文件路径"},
            },
            "required": ["file_path"],
        },
    },
}


async def write_file(args: dict) -> str:
    """在【有界工作区】新建/覆盖一个文件。需审批（agent_loop 见 __requires_approval__）。

    只写工作区内的路径（resolve_within_workspace 钳制，越界直接拒绝）。纯 Python，
    不碰 shell —— 不像 bash 依赖 PATH/WSL/heredoc，Windows 上更稳。
    """
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "错误：请提供 path 参数（要写入的文件路径）。"

    target = resolve_within_workspace(path)
    if target is None:
        return f"错误：路径越出了工作区「{workspace_path()}」，本工具不允许写工作区之外的路径：{path}"

    try:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"错误：写入失败「{target}」—— {e}"

    return f"✅ 已写入 {target}（{len(content)} 字符）"


write_file.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "在【有界工作区】里新建或覆盖一个文件，写入文本内容，返回已写路径。\n"
            "本工具执行前需要你的批准（同 bash 一样是写操作）。\n"
            "这是什么：创建 md/脚本/配置等文件，或用新内容整体替换已有文件。"
            "它用纯 Python 写文件，不依赖 shell——比 bash 建文件更稳。\n"
            "Use when: 要『新建一个文件并写入内容』或『覆盖整个文件』。"
            "Don't use when: 只想看文件内容(用 read_file)、只想改文件里的一小段(用 edit_file)、"
            "或需要跑命令做计算(用 execute_code/bash)。\n"
            "Example: path='notes.md', content='# Title\\n正文'——写在 {workspace_path()}"
            " 工作区里（相对路径以它为基准）；不要在 path 里给出越过工作区的绝对路径。\n"
            "常见错误: path 给绝对路径时必须落在工作区内，否则会被拒绝；"
            "写后会弹 y/n 确认，输入 y 才真正写入。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "工作区内的文件路径（相对或在工作区内的绝对路径）"},
                "content": {"type": "string", "description": "要写入的完整文本内容"},
            },
            "required": ["path", "content"],
        },
    },
}
write_file.__requires_approval__ = True


async def edit_file(args: dict) -> str:
    """在【有界工作区】的已有文件里做【一次】精确片段替换。需审批。

    先按 old_string 精确定位（必须在文件里、且希望只改一处），替换成 new_string；
    没找到就拒绝修改并提示先去 read_file。防止模型编造/错位地乱改。
    """
    path = args.get("path", "")
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    if not path:
        return "错误：请提供 path 参数（要修改的文件路径）。"
    if not old:
        return "错误：请提供 old_string（要替换的原文片段）。"

    target = resolve_within_workspace(path)
    if target is None:
        return f"错误：路径越出了工作区「{workspace_path()}」，本工具不允许改工作区之外的路径：{path}"
    if not os.path.isfile(target):
        return f"错误：文件不存在「{target}」"

    try:
        with open(target, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return f"错误：读取失败「{target}」—— {e}"

    if old not in text:
        return (
            f"错误：old_string 在「{target}」（{len(text)} 字符）里没找到。"
            f"请先 read_file 确认原文，再提供与文件实际内容一致的 old_string。"
        )

    new_text = text.replace(old, new, 1)  # 只替换第一处，避免误改多处
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(new_text)
    except OSError as e:
        return f"错误：写入失败「{target}」—— {e}"

    return f"✅ 已在「{target}」替换 1 处"


edit_file.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "在【有界工作区】的已有文件里，把精确匹配的一小段（old_string）替换成新文本，"
            "并确认改到。执行前需要你的批准（写操作）。\n"
            "这是什么：精确的『查找→替换』，不会重排文件、不会动到别的部分。\n"
            "Use when: 只改文件里的一小段（改个函数体、换个词、修一行）。"
            "Don't use when: 新建/整体覆盖文件(用 write_file)、想按行号编辑或做大改动"
            "(先 read_file 再想清楚)、或只需读文件(用 read_file)。\n"
            "Example: path='notes.md', old_string='旧文案', new_string='新文案', "
            "替换在工作区里进行。\n"
            "常见错误: old_string 必须和文件【里实际存在的字】完全一致——先 read_file 抄原文；"
            "找不到会拒绝修改；一次只改第一处。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "工作区内的文件路径"},
                "old_string": {"type": "string", "description": "要替换的原文片段（必须精确匹配文件内容）"},
                "new_string": {"type": "string", "description": "替换成的新文本"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}
edit_file.__requires_approval__ = True