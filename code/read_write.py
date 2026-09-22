"""
代码/文件系统工具 · 读写(Read/Write)
────────────────────────────
read_file —— 读取指定文件的完整内容（带行号 + 输出截断）。纯只读、离线、安全。
"""

import os

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