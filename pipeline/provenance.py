"""实验来源信息（provenance）：这次实验是谁、用哪份数据、在哪份代码上跑出来的。

为什么单独立一层：
    一次失败或一次提升，只有在"Agent / 数据 / 代码"三样都能对上时才有解释力。
    这三样现在分散在 report.json、任务文件、Git 里，事后只能靠目录名猜
    （实际已经出现过目录名 `main-a2` 里跑的却是 shop-cold）。
    这一层只负责把它们记成一份可机读的 manifest，不参与执行、不参与判分。

三条原则，都是为了防止"看起来可追溯、实际不可信"：

1. 值必须来自运行时。
   取不到就写 None，绝不填默认值——"dirty=false" 猜错一次，
   整份实验记录就变成了伪证。

2. Git 失败不等于工作区干净。
   命令不存在、目录不是仓库、超时，全部记 commit=None、dirty=None，
   并把原始错误留在 `git.error` 里。

3. 写不出去要吵。
   本模块不吞异常：`write_manifest` 抛 `ManifestError`，由调用方决定
   打印警告还是中止。静默失败会产出"没有 manifest 但看起来正常"的目录。

刻意不做的事：不读 Agent 的内部状态、不解析模型名、不猜环境变量。
这些东西没有任何一处是权威来源，记下来只会制造新的不一致。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "1.0"

# manifest 的相对路径基准：项目根目录。写在文件里的路径一律相对它，
# 换机器 clone 之后仍然能对上，不会出现 C:\Users\xxx 这种绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_HASH_CHUNK = 1 << 20


class ManifestError(RuntimeError):
    """manifest 写不出去。调用方必须显式处理，不允许静默吞掉。"""


def now_iso() -> str:
    """带时区的本地时间，秒级。比 naive datetime 更适合跨机器对照。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel_path(path: Path | str) -> str:
    """项目内的路径转相对（统一用 /），项目外的保持绝对字符串。"""
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return str(p)


def _git(args: list[str], cwd: str) -> tuple[str | None, str]:
    """跑一条 git 命令。失败返回 (None, 错误原文)，不抛。"""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # git 没装 / 没有执行权限 / 超时 / 输出编码异常，都在这里
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return None, detail or f"git {' '.join(args)} 退出码 {proc.returncode}"
    return proc.stdout.strip(), ""


def git_info(cwd: Path | str | None = None) -> dict:
    """当前工作区的 git 状态。

    dirty 三态：True 有未提交改动、False 干净、None 取不到。
    None 和 False 必须分开：把"不知道"写成"干净"是这里最严重的错误。
    """
    base = str(cwd or PROJECT_ROOT)
    commit, commit_err = _git(["rev-parse", "HEAD"], base)
    branch, _ = _git(["rev-parse", "--abbrev-ref", "HEAD"], base)
    status, status_err = _git(["status", "--porcelain"], base)

    return {
        "commit": commit or None,
        "branch": branch or None,
        "dirty": None if status is None else bool(status),
        "error": "; ".join(x for x in (commit_err, status_err) if x),
    }


def sha256_file(path: Path | str) -> str | None:
    """文件的 SHA-256。文件不存在或读不了返回 None，不抛。"""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def file_entry(path: Path | str) -> dict:
    """一个数据文件的可追溯描述。拿不到的字段一律 None。"""
    p = Path(path)
    exists = p.is_file()
    size = None
    if exists:
        try:
            size = p.stat().st_size
        except OSError:
            size = None
    return {
        "path": rel_path(p),
        "sha256": sha256_file(p),
        "bytes": size,
        "exists": exists,
    }


def build_run_manifest(
    *,
    run_id: str,
    agent: str,
    agent_requested: str,
    goldens,
    tasks_total: int,
    fit_ids,
    holdout_ids,
    params: dict,
    store_path: Path | str | None = None,
    cwd: Path | str | None = None,
) -> dict:
    """一次实验目录的 manifest。纯函数：只读文件与 git，不写盘。"""
    # 先物化：调用方可能传生成器，len(list(x)) 加 list(x) 会把生成器取空。
    fit_list = [str(x) for x in fit_ids]
    holdout_list = [str(x) for x in holdout_ids]

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run",
        "run_id": str(run_id),
        "created_at": now_iso(),
        "agent": str(agent),
        "agent_requested": str(agent_requested),
        "python": sys.version.split()[0],
        # 数据集按文件记哈希：任务集被改过一条，哈希就变，通过率才有可比性。
        "datasets": [file_entry(g) for g in goldens],
        "splits": {
            "fit": len(fit_list),
            "holdout": len(holdout_list),
            "tasks_total": int(tasks_total),
        },
        "task_ids": {"fit": fit_list, "holdout": holdout_list},
        "params": dict(params),
        "store": rel_path(store_path) if store_path else None,
        "git": git_info(cwd),
    }


def distill_provenance(
    *,
    inputs,
    sft: int,
    failed: int,
    by_group: dict,
    agents: dict,
    splits: dict,
    code_path: Path | str,
) -> dict:
    """一次 distill 产物的来源信息。

    inputs 每项至少含 path / sha256 / bytes / traces；轨迹文件被替换过，
    哈希就对不上，这份 SFT 到底来自哪批数据不再需要靠目录名推断。
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "distill",
        "generated_at": now_iso(),
        "code": {
            "file": rel_path(code_path),
            "sha256": sha256_file(code_path),
            "git": git_info(Path(code_path).resolve().parent.parent),
        },
        "inputs": list(inputs),
        "outputs": {"sft": int(sft), "failed": int(failed)},
        "by_group": dict(by_group),
        "agents": dict(agents),
        "splits": dict(splits),
    }


def write_manifest(path: Path | str, payload: dict) -> Path:
    """写 manifest。失败抛 ManifestError，调用方必须显式处理。"""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise ManifestError(
            f"manifest 写入失败：{p}（{type(exc).__name__}: {exc}）"
        ) from exc
    return p


__all__ = [
    "SCHEMA_VERSION",
    "ManifestError",
    "PROJECT_ROOT",
    "now_iso",
    "rel_path",
    "git_info",
    "sha256_file",
    "file_entry",
    "build_run_manifest",
    "distill_provenance",
    "write_manifest",
]
