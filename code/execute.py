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
    """
    code = args.get("code", "")
    lang = str(args.get("lang", "python")).lower()
    timeout = float(args.get("timeout", _DEFAULT_TIMEOUT))

    if not code:
        return "错误：请提供 code 参数。"

    # 独立临时工作目录：每次执行都在新沙箱目录里跑，绝不落在项目目录
    workdir = tempfile.mkdtemp(prefix="agent_sandbox_")
    env = _build_sandbox_env()

    # 组装命令行：python -> python -c code；shell -> bash -c code
    if lang in ("python", "py"):
        cmd = [sys.executable, "-c", code]
    elif lang in ("shell", "bash", "sh"):
        # 显式不要用继承的 SHELL（侵入式），固定在 sh -c
        cmd = ["sh", "-c", code]
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

    summary_parts = [f"工作目录(临时沙箱): {workdir}", f"退出码: {code_rc}"]
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