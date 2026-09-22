"""
代码/文件系统工具 · 搜索(Search)
────────────────────────────
grep_files —— 在目录里递归搜文本/正则，返回命中的【文件:行号】。Coding Agent 的核心
检索工具，类比编辑器 Ctrl+F 但能一次扫整个目录。纯本地、离线、只读、安全。
"""

import os
import re

# 搜索时默认跳过的目录 —— 这些是噪音，不是源码
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".cache", "build", "dist"}
# glob 递归时同样跳过噪音目录
_GLOB_SKIP_DIRS = {*_SKIP_DIRS}


async def grep_files(args: dict) -> str:
    """
    在 path 目录下递归搜索 pattern（支持正则），返回命中的文件与行。

    用纯 Python（os.walk + re），不依赖系统 grep 命令，跨平台可用。
    """
    pattern = args.get("pattern", "")
    path = args.get("path", ".")
    file_pattern = args.get("file_pattern", "*")  # 如 *.py, *.md
    case_sensitive = bool(args.get("case_sensitive", False))
    max_results = int(args.get("max_results", 50))

    if not pattern:
        # 自愈(不是死路报错)：DeepSeek 这类模型常先发空壳 {} 试探工具，空报错只会
        # 让它原地重试同一个 {}。改成喂它"可能正缺的东西"——目录文件清单 + 正确
        # 用法示范——让当轮不浪费，并直接教会它下一轮怎么填 required 参数。
        root0 = os.path.abspath(path) if os.path.isdir(os.path.abspath(path)) else os.path.abspath(".")
        return (
            "pattern 为空：需要先填 required 参数 pattern，指定要搜索的文本/正则，"
            "才能搜内容。\n\n"
            "【正确示例】grep_files(pattern='要搜的词', path='partical', "
            "file_pattern='*.py', max_results=50)\n\n"
            f"{_list_files(root0)}"
        )

    # 只读路径，绝对化并确认存在
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        return f"错误：路径不存在或不是目录「{root}」"

    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        rx = re.compile(pattern, flags)
    except re.error as e:
        return f"错误：无效的正则「{pattern}」—— {e}"

    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地过滤掉噪音目录，避免整棵 .git 被扫
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for fname in filenames:
            if not _match_glob(fname, file_pattern):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if len(hits) >= max_results:
                            break
                        if rx.search(line):
                            hits.append(f"{fpath}:{lineno}: {line.rstrip()[:200]}")
                    if len(hits) >= max_results:
                        break
            except OSError:
                continue  # 跳过不可读文件
        if len(hits) >= max_results:
            break

    if not hits:
        return f"在「{root}」下没找到匹配「{pattern}」的内容。"

    # 展示计数 + 前若干条（避免刷屏），并注明可继续搜更精确
    body = "\n".join(hits)
    return (
        f"在「{root}」找到 {sum(1 for _ in hits)} 条匹配「{pattern}」：\n\n"
        f"{body}\n\n（只显示前 {max_results} 条；如需更精确，可缩小 file_pattern 或加更多关键词）"
    )


def _list_files(root: str, limit: int = 40) -> str:
    """列 root 下可见文件(相对路径，跳噪音目录)，供 grep_files 空参时自愈喂给模型。

    浅层 os.walk + 截断，够当"这个目录里有什么"的预览即可。
    """
    hits = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                if len(hits) >= limit:
                    break
                rel = os.path.relpath(os.path.join(dirpath, fname), root).replace("\\", "/")
                hits.append(rel)
            if len(hits) >= limit:
                break
    except OSError:
        pass
    if not hits:
        return f"目录「{root}」下暂无可见文件（可能是空目录或权限限制）。"
    return f"目录「{root}」下的文件预览（前 {len(hits)} 个）：\n" + "\n".join(hits)


def _match_glob(fname: str, glob: str) -> bool:
    """极简 glob 匹配：只支持 * 和 ? 通配（够用就行，不引 fnmatch 复杂度）。"""
    if glob in ("*", ""):
        return True
    return _match_glob_impl(fname, glob)


def _match_glob_impl(s: str, p: str) -> bool:
    """递归 glob 匹配，支持 *（任意长度）和 ?（单个字符）。"""
    if not p:
        return not s
    if p[0] == "*":
        return _match_glob_impl(s, p[1:]) or (bool(s) and _match_glob_impl(s[1:], p))
    if p[0] == "?":
        return bool(s) and _match_glob_impl(s[1:], p[1:])
    return bool(s) and s[0] == p[0] and _match_glob_impl(s[1:], p[1:])


grep_files.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "grep_files",
        "description": (
            "在目标目录下递归搜索某段文本或正则，返回命中的文件路径:行号 和内容。\n"
            "Use when: 想找某段代码/字符串/调用/单词在项目的哪些位置出现——"
            "例如搜 `cursor.execute` 确认 SQL 拼接、搜 `TODO`、搜某函数名在哪定义。"
            "Don't use when: 只需要读某个已知文件的完整内容（那用 read_file，不在本工具范围）。\n"
            "Example: pattern='cursor.execute', path='partical', file_pattern='*.py'。\n"
            "常见错误: pattern 是正则，搜含特殊字符（如 .）要写 `cursor\\.execute`；"
            "路径要指向项目根而不是文件；默认忽略大小写。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "要搜索的文本或正则表达式"},
                "path": {"type": "string", "description": "要搜索的目录，默认当前目录 ."},
                "file_pattern": {"type": "string", "description": "文件通配，如 *.py / *.md / *，默认 *"},
                "case_sensitive": {"type": "boolean", "description": "是否区分大小写，默认 false"},
                "max_results": {"type": "integer", "description": "最多显示多少条结果，默认 50"},
            },
            "required": ["pattern"],
        },
    },
}


async def glob_files(args: dict) -> str:
    """
    按【文件名模式】递归列出文件路径，而非按内容搜（那用 grep_files）。

    类比编辑器的"文件面板" / shell 的 `ls -R`。返回匹配的文件路径列表，
    如列出一个目录下所有 .py 文件。
    """
    path = args.get("path", ".")
    file_pattern = args.get("file_pattern", "**/*")  # 默认递归列出所有(跳过噪音目录)
    max_results = int(args.get("max_results", 200))

    if not file_pattern:
        return "错误：file_pattern 参数不能为空。"

    root = os.path.abspath(path)
    if not os.path.isdir(root):
        return f"错误：路径不存在或不是目录「{root}」"

    # 转成相对 root 的正则：把 ** 递归展开 + 其他 glob 转义
    try:
        rx = _glob_to_regex(file_pattern)
    except re.error as e:
        return f"错误：无效的 file_pattern「{file_pattern}」—— {e}"

    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 过滤噪音目录，避免整棵 .git 被递归
        dirnames[:] = [d for d in dirnames if d not in _GLOB_SKIP_DIRS]
        for fname in filenames:
            if len(matches) >= max_results:
                break
            # 相对路径用 / 分隔，好和 pattern 匹配
            rel = os.path.relpath(os.path.join(dirpath, fname), root).replace("\\", "/")
            if rx.search(rel):
                matches.append(os.path.join(dirpath, fname).replace("\\", "/"))
        if len(matches) >= max_results:
            break

    if not matches:
        return f"在「{root}」下没有匹配 file_pattern「{file_pattern}」的文件。"

    body = "\n".join(matches)
    return (
        f"在「{root}」找到 {len(matches)} 个文件匹配「{file_pattern}」（前 {max_results} 条）：\n\n"
        f"{body}\n\n（只列出文件名/路径；要看某个文件内容请用 read_file）"
    )


def _glob_to_regex(pat: str) -> re.Pattern:
    """把类 glob 模式(支持 ** /* / ?)逐字符转成正则 re.Pattern，匹配相对路径。

    - `**/`  匹配零或多层目录前缀(含其结尾斜杠)
    - `**`   匹配任意剩余路径(含斜杠)
    - `*`    匹配同层内任意字符(不含 /)
    - `?`    匹配单个字符(不含 /)
    """
    expr = []
    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        if pat.startswith("**", i):
            if i + 2 < n and pat[i + 2] == "/":
                # ** / -> 匹配零或多层目录前缀,把紧跟的斜杠一起吸收掉
                expr.append("(?:.*/)?")
                i += 2  # 跳过 **
                while i < n and pat[i] == "/":
                    i += 1  # 吃掉 /，避免后面再补一个
            else:
                # 末尾的 ** -> 匹配任意剩余路径
                expr.append(".*")
                i += 2
            continue
        if c == "*":
            expr.append("[^/]*")
            i += 1
            continue
        if c == "?":
            expr.append("[^/]")
            i += 1
            continue
        if c == "/":
            expr.append("/")
            i += 1
            continue
        expr.append(re.escape(c))
        i += 1
    return re.compile(f"^{''.join(expr)}$")


glob_files.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "glob_files",
        "description": (
            "按【文件名模式】递归列出项目里的文件路径，而非按内容搜索（那用 grep_files）。\n"
            "Use when: 想知道某目录下有哪些文件、某个模式的文件在哪、或列出一个项目的结构——"
            "例如『当前目录下有哪些 .py 文件』『partical/code 目录下都有什么』。"
            "Don't use when: 想按文件【内容】找某段代码（那用 grep_files）、"
            "或想读某个已知文件的内容（那用 read_file）。\n"
            "Example: path='partical', file_pattern='**/*.py'；"
            "path='.', file_pattern='*.md'；file_pattern='**/*' 列出全部(跳过隐藏/二进制噪音)。\n"
            "常见错误: file_pattern 是【文件名模式】不是内容关键词；"
            "默认递归列出全部（**/*），可用 '*.py' 只找顶层、'**/*.md' 跨层找注释文档。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "起始目录，默认当前目录 ."},
                "file_pattern": {
                    "type": "string",
                    "description": "文件名模式：支持 **/*.py、*.md、**/* 等；默认 **/* 列出全部",
                },
                "max_results": {"type": "integer", "description": "最多返回多少条路径，默认 200"},
            },
            "required": [],
        },
    },
}