"""
code.workspace — 有界工作区路径解析（bash / write_file / edit_file 共用）。

所有"能真写/真跑"的工具（都带 __requires_approval__，执行前经审批门）都写进同一个
工作区根，路径经 resolve_within_workspace 钳制在根内 —— 审批门是第一道墙，路径圈定
是第二道（连误删/越界也拦在内）。
"""

import os

# 工作区根：可用环境变量 CODING_WORKSPACE 覆盖；默认项目根下 _coding_workspace
_WORKSPACE_ENV = "CODING_WORKSPACE"
_WORKSPACE_DEFAULT = os.path.abspath("_coding_workspace")


def workspace_path() -> str:
    """返回工作区根（必要时创建）。写/跑工具都把这里当默认落点。"""
    base = os.path.abspath(os.environ.get(_WORKSPACE_ENV, _WORKSPACE_DEFAULT))
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        pass  # 解析阶段不必强建；写工具打开文件时才真建
    return base


def resolve_within_workspace(requested: str) -> str | None:
    """把用户请求的路径解析到工作区内的绝对路径；越界返回 None。

    相对路径 → 以工作区根为基准；
    绝对路径 → 必须落在工作区内，否则拒绝。
    """
    base = os.path.abspath(os.environ.get(_WORKSPACE_ENV, _WORKSPACE_DEFAULT))
    target = os.path.abspath(requested) if os.path.isabs(requested) else os.path.abspath(
        os.path.join(base, requested)
    )
    base_l, target_l = base.lower(), target.lower()
    ok = target_l == base_l or target_l.startswith(base_l + os.sep)
    return target if ok else None