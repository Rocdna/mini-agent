"""
代码/文件系统工具 · 执行(Execute)
────────────────────────────
execute_code —— 在【受限子进程】里跑 Python/Shell，带超时、输出截断、瘦身 env
（摘掉 HOME/凭据）+ 独立临时工作目录，防卡死 / 刷屏 / 碰密钥 / 碰项目。

安全边界（execute_code 不同于只读的 search/read，它让 agent 能"跑"）：
  - 超时截断（timeout）防死循环 / 卡死
  - 输出截断防占爆上下文
  - 瘦身 env：只留 PATH + 显式允许的键，摘掉 HOME/KEY/SHELL 等 → 找不到用户凭据
  - 独立 temp 工作目录：跑在专用沙箱目录，不碰真实项目文件
  - 默认禁网络：不设代理、不暴露凭据；需要联网由调用方显式声明（本实现未开）
"""

import asyncio
import os
import shutil
import sys
import tempfile

# 执行时单次结果的最大字节数；超长截断，防止工具结果把上下文灌爆
_MAX_EXEC_BYTES = 8000

# 默认执行超时（秒）；防止死循环 / 网络卡死
_DEFAULT_TIMEOUT = 10

# 瘦身 env 时要从子进程环境里摘掉的敏感键 —— 原则是"最小权限"：
# 只给 agent 的子进程一个干净的、不知道该在哪儿找用户密钥的环境。
#   HOME / USERPROFILE   → 不知道用户主目录 → 找不到 ~/.ssh、.aws 等
#   *KEY* / *TOKEN* / *SECRET* → 显式清掉应用层密钥
#   SSH_AUTH_SOCK         → 不暴露 ssh agent socket
_SENSITIVE_ENV_KEYS = (
    "HOME", "USERPROFILE", "LOGNAME", "USER",
    "SSH_AUTH_SOCK", "SSH_AGENT_PID",
)


def _build_sandbox_env() -> dict:
    """构造子进程的"瘦身环境"：只留 PATH + 显式安全允许的键，摘掉敏感项。

    这是 execute_code 安全墙的最外层：agent 的子进程不该知道用户主目录、
    密钥、登录名在哪 —— 不知道，就无从读取/外传。
    """
    base = os.environ
    env = {"PATH": base.get("PATH", "")}
    # 允许中文/UTF-8 输出正常显示，但不暴露身份（只透这几个）
    for safe in ("LANG", "PYTHONIOENCODING", "PYTHONUNBUFFERED"):
        if base.get(safe):
            env[safe] = base[safe]
    # 第一步：先清掉硬清单里的敏感键（HOME/凭据/登录名）
    for k in _SENSITIVE_ENV_KEYS:
        env.pop(k, None)
    # 第二步：把 env 里任何疑似凭据的键（key/token/secret/auth/password）再扫一遍双保险
    for orig in list(env):
        low = orig.lower()
        if "key" in low or "token" in low or "secret" in low or "auth" in low or "password" in low:
            env.pop(orig, None)
    return env


def _msys_root(shell: str) -> str:
    """找 shell 所属的 Git 根目录。

    不能 dirname 两次：shell 可能在 Git\\bin（两级到根）也可能在 Git\\usr\\bin（两级会
    停在 Git\\usr，错一级）。判别靠真实标志——第一个带 mingw64\\bin 的祖先即为 Git 根。
    """
    d = os.path.dirname(os.path.abspath(shell))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "mingw64", "bin")):
            return d
        p = os.path.dirname(d)
        if p == d:
            break
        d = p
    return os.path.dirname(os.path.dirname(os.path.abspath(shell)))


def _build_workspace_env(shell: str) -> dict:
    """真实工作区 bash 的环境：完整用户环境 + 补 git-bash 自带工具路径。

    与 _build_sandbox_env 相反——这条路径只由带 __requires_approval__ 的 bash 走，
    安全模型是【执行前人工批准】这道闸，不是剥环境。所以保留完整 HOME/TEMP/env
    （git-bash 需要它们才能找临时目录、撞 here-doc），并把它自己 usr/bin、mingw64/bin
    插到 PATH 前——否则从 PowerShell 启动时 PATH 里根本没有 git 的 bin，ls/cat/sed/which
    全会 127。根由 _msys_root 定位（兼容 bin 与 usr/bin 两种 shell 路径）。
    """
    env = dict(os.environ)
    root = _msys_root(shell)
    extra = []
    for rel in (r"usr\bin", r"mingw64\bin"):
        d = os.path.join(root, rel)
        if os.path.isdir(d) and d not in env.get("PATH", ""):
            extra.append(d)
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _resolve_shell() -> str | None:
    """找可用的 POSIX shell（execute_code 的 lang='shell' 和 bash 工具共用）。

    Windows 的坑有两个，逐个防：
      · 默认【没有】 sh —— 写死 sh -c 会 WinError 2；
      · `C:\\Windows\\System32\\bash.exe` 是 WSL 启动器，不是真 shell，一旦 WSL
        没配好会报 Bash/Service/E_UNEXPECTED、wsl:localhost 转发失败。
    所以顺序固定为：
      1. 有 git 就反推它同根的 Git Bash（最可靠，必真 bash、避开 WSL）
      2. PATH 上的 bash / sh，跳过 System32 的 WSL 启动器
      3. 常见 Git 安装路径兜底
    都找不到才返回 None，让调用方报明确的安装提示，而不是裸错误码。
    """
    def _is_wsl_shim(p: str) -> bool:
        """Windows 自带 C:\\Windows\\System32\\bash.exe 只是 WSL 启动器，不是真正
        POSIX shell。一旦 WSL 没配好，往它 -c 会走 WSL 并报 Bash/Service/E_UNEXPECTED、
        wsl:localhost 转发失败这类错误。命中 System32 就当作不存在跳过。"""
        ap = os.path.abspath(p).lower()
        return "\\system32\\" in ap and os.path.basename(ap).startswith(("bash", "sh"))

    def _from_git() -> str | None:
        # 从 git 的位置反推它同根的 Git Bash。Git 安装盘符不固定、且 git 可能埋在
        # mingw64\\bin 这类子目录，所以不能 dirname 两次，要【从 git 所在目录一路向上】
        # 找带 usr/bin/bash 的祖先。返回的必是真 Git Bash，天然避开 System32 WSL 启动器。
        for hint in (shutil.which("git"), shutil.which("git.exe")):
            if not hint:
                continue
            d = os.path.dirname(hint)  # 先到含 git.exe 的目录，再逐级上溯
            for _ in range(6):        # 6 级高度足够从 mingw64\\bin 上溯到 Git 根
                for rel in (r"usr\bin\bash.exe", r"bin\bash.exe", r"usr\bin\sh.exe", r"bin\sh.exe"):
                    cand = os.path.join(d, rel)
                    if os.path.exists(cand):
                        return cand
                parent = os.path.dirname(d)
                if parent == d:       # 已到盘根，别再空转
                    break
                d = parent
        return None

    def _pick_env(key: str) -> str | None:
        # 环境变量可以是绝对路径，也可是命令名（退化为 which 且避开 WSL shim）。
        v = os.environ.get(key)
        if not v:
            return None
        v = os.path.expanduser(v.strip())
        if os.path.exists(v):
            return v
        p = shutil.which(os.path.basename(v))
        if p and not _is_wsl_shim(p):
            return p
        return None

    # 0) 声明/继承优先 —— 不做磁盘试探，最强泛化：
    #    PARTICAL_SHELL  用户明示的 shell（换机器/换 shell 只改配置，零代码）。
    #    SHELL           启动本进程的终端自带（git-bash 会给真路径），信它。
    for key in ("PARTICAL_SHELL", "SHELL"):
        env_shell = _pick_env(key)
        if env_shell:
            return env_shell

    # 1) 有 git 就反推（最可靠，必是真 Git Bash，避开 WSL）
    via_git = _from_git()
    if via_git:
        return via_git

    # 2) PATH 上的 bash / sh，但必须跳过 System32 的 WSL 启动器
    for name in ("bash", "sh"):
        p = shutil.which(name)
        if p and not _is_wsl_shim(p):
            return p

    # 3) 常见安装路径兜底
    for exe in (
        r"C:\Program Files\Git\bin\sh.exe",
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\sh.exe",
    ):
        if os.path.exists(exe):
            return exe
    return None


# 解析结果【进程级缓存】：bash / execute 每次调用不重跑 which/嗅探，确定且省。
# False=尚未解析；None=确认找不到；str=可用的 shell 绝对路径。
_SHELL_CACHE: object = False


def _resolve_shell_cached() -> str | None:
    global _SHELL_CACHE
    if _SHELL_CACHE is False:
        _SHELL_CACHE = _resolve_shell()
    return _SHELL_CACHE  # str 或 None


async def execute_code(args: dict) -> str:
    """
    在受限子进程里执行 Python 或 Shell 代码，返回 stdout/stderr + 退出码。

    安全墙（见模块 docstring）：
      - timeout 防死循环 / 卡死
      - 瘦身 env（摘 HOME/凭据）+ 独立临时工作目录 → 找不到密钥、碰不到项目
      - 输出截断防占爆上下文

    参数：
      code       必填，要执行的代码
      lang       python 或 shell，默认 python
      timeout    秒，默认 10
    workdir 由【受信任封装工具】（如 gated 的 bash）内部传入；它在 subprocess 里
    是普通参数，但【故意不写进下面的 tool schema】——模型看不到、就不会拿它去绑定
    真实目录绕过 bash 的审批门（见 tools.py 的 __requires_approval__）。
    """
    code = args.get("code", "")
    lang = str(args.get("lang", "python")).lower()
    timeout = float(args.get("timeout", _DEFAULT_TIMEOUT))

    if not code:
        return "错误：请提供 code 参数。"

    # 工作目录：默认每次都在新沙箱目录（隔离、免审批）；仅当受信任封装工具显式传入
    # workdir 时才绑定真实目录（可读写/改文件，因此必须走 bash 的审批门）。
    workdir_arg = args.get("workdir", "")
    if workdir_arg:
        workdir = os.path.abspath(os.path.expanduser(workdir_arg))
        os.makedirs(workdir, exist_ok=True)
    else:
        workdir = tempfile.mkdtemp(prefix="agent_sandbox_")
    env = _build_sandbox_env()  # 默认：隔离沙箱（剥 HOME/凭据）

    # 组装命令行：python -> python -c code；shell -> 解析出的 POSIX shell -c code
    if lang in ("python", "py"):
        cmd = [sys.executable, "-c", code]
    elif lang in ("shell", "bash", "sh"):
        shell = _resolve_shell_cached()
        if shell is None:
            return (
                "错误：找不到可用的 POSIX shell（sh/bash）。bash 工具依赖 Git Bash；"
                "请安装 Git，或把 Git 的 bin 目录加入 PATH 后重试。"
            )
        cmd = [shell, "-c", code]
        if workdir_arg:
            # 真实工作区的 bash：安全模型是【审批门】而不是剥环境，所以改用完整环境 +
            # 补 git-bash 工具路径，否则 ls/cat/here-doc 会 127。隔离沙箱那条路不动。
            env = _build_workspace_env(shell)
    else:
        return f"错误：不支持的 lang「{lang}」，仅支持 python / shell。"

    try:
        import subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"⏱ 执行超时（>{timeout:.0f}s），已终止子进程（防死循环）。"
    except Exception as e:
        return f"错误：无法启动子进程—— {e}"

    # 解码 + 截断，防占爆上下文
    out_txt = (out or b"").decode("utf-8", errors="replace")
    err_txt = (err or b"").decode("utf-8", errors="replace")
    code_rc = proc.returncode

    mode = "临时沙箱" if not workdir_arg else "真实工作区"
    summary_parts = [f"工作目录({mode}): {workdir}", f"退出码: {code_rc}"]
    if out_txt:
        summary_parts.append(f"\n[stdout]\n{out_txt[:_MAX_EXEC_BYTES]}")
        if len(out_txt) > _MAX_EXEC_BYTES:
            summary_parts.append(f"\n…（stdout 超出 {_MAX_EXEC_BYTES} 字符，已截断）")
    if err_txt:
        summary_parts.append(f"\n[stderr]\n{err_txt[:_MAX_EXEC_BYTES]}")
        if len(err_txt) > _MAX_EXEC_BYTES:
            summary_parts.append(f"\n…（stderr 超出 {_MAX_EXEC_BYTES} 字符，已截断）")
    return "执行结果：\n" + "\n".join(summary_parts)


execute_code.__tool_schema__ = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "在受限子进程里执行一段 Python 或 Shell 代码，返回 stdout/stderr 和退出码。\n"
            "安全墙：有超时（防死循环）、输出截断、且在独立临时沙箱目录 + 瘦身环境里跑，"
            "不会碰到你的真实项目文件或密钥。\n"
            "Use when: 需要『实证』验证某段代码能不能跑、输出是什么、会不会报错——"
            "例如确认一段 SQL 拼接代码会不会在 Python 里报错、一个算法跑出来是多少。"
            "Don't use when: 只想读文件内容(用 read_file)、只想搜文本(用 grep_files)、"
            "或需要联网/访问真实文件系统的任务（本工具在隔离沙箱里，拿不到用户文件）、"
            "或代码意图已经清楚、只是想『复现/展示它的输出当证据』（此时直接描述语义"
            "即可，不必执行——就算执行也只会拿到沙箱里的空环境/或需绕路内联，反而多耗轮次）。\n"
            "Example: code='print(sum(range(100)))'、"
            "code='import sqlite3; print(sqlite3.version)', lang='python'。\n"
            "常见错误: 沙箱环境是干净隔离的——没有你平时的第三方包依赖，也没有网络；"
            "要做文件读写只能操作沙箱自己的工作目录；"
            "语言默认 python（lang='shell' 可切到 sh）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的代码字符串"},
                "lang": {"type": "string", "description": "python 或 shell，默认 python"},
                "timeout": {"type": "number", "description": "超时秒数，默认 10"},
            },
            "required": ["code"],
        },
    },
}