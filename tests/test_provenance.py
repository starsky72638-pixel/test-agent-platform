"""实验来源（provenance）的离线测试。

三类行为必须钉住：
1. 取不到的值写 None，不能写成"干净"——把未知当已知比缺字段更危险；
2. 哈希与条数来自真实文件，不是从参数回填的；
3. 写不出去要抛，不能静默产出一个"看起来正常但缺 manifest"的目录。

全程不联网，不依赖 Git（有仓库时额外断言 commit 形状）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.trace import Trace, Turn, save_traces
from pipeline import provenance
from scripts import distill


def test_git_info_reports_unknown_outside_a_repo(tmp_path):
    """不是仓库时 commit/dirty 都必须是 None，且留下错误原文。"""
    info = provenance.git_info(tmp_path)

    assert info["commit"] is None
    assert info["dirty"] is None
    assert info["error"], "取不到 Git 信息时必须留下原因，不能是空串"


def test_git_info_reads_this_repo_when_present():
    """在仓库里跑时必须给出 40 位 commit 和布尔 dirty。"""
    if not (provenance.PROJECT_ROOT / ".git").exists():
        pytest.skip("当前目录不是 Git 仓库")

    info = provenance.git_info(provenance.PROJECT_ROOT)

    assert isinstance(info["commit"], str) and len(info["commit"]) == 40
    assert isinstance(info["dirty"], bool)


def test_sha256_file_matches_bytes_and_missing_is_none(tmp_path):
    payload = "订单 1002 已发货".encode("utf-8")
    target = tmp_path / "traces.jsonl"
    target.write_bytes(payload)

    assert provenance.sha256_file(target) == hashlib.sha256(payload).hexdigest()
    assert provenance.sha256_file(tmp_path / "nope.jsonl") is None


def test_write_manifest_writes_json_and_raises_on_failure(tmp_path):
    target = tmp_path / "runs" / "x" / "manifest.json"

    provenance.write_manifest(target, {"schema_version": "1.0"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"schema_version": "1.0"}

    # 目标是目录，写不进去时必须抛，不能吞
    with pytest.raises(provenance.ManifestError, match="manifest 写入失败"):
        provenance.write_manifest(tmp_path, {"a": 1})


def test_build_run_manifest_records_real_values(tmp_path):
    data = tmp_path / "tasks.jsonl"
    data.write_text('{"id": "A"}\n', encoding="utf-8")

    manifest = provenance.build_run_manifest(
        run_id="20260101-000000",
        agent="wf:demo",
        agent_requested="wf:demo",
        goldens=[data],
        tasks_total=3,
        fit_ids=["A"],
        holdout_ids=["B", "C"],
        params={"top_k": 6, "min_score": None},
        store_path=tmp_path / "experiences.jsonl",
        cwd=tmp_path,
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["kind"] == "run"
    assert manifest["run_id"] == "20260101-000000"
    assert manifest["splits"] == {"fit": 1, "holdout": 2, "tasks_total": 3}
    assert manifest["task_ids"]["holdout"] == ["B", "C"]
    assert manifest["datasets"][0]["sha256"] == hashlib.sha256(
        data.read_bytes()
    ).hexdigest()
    assert manifest["params"]["min_score"] is None
    # tmp_path 不是仓库：必须写 None，而不是 False
    assert manifest["git"]["commit"] is None
    assert manifest["git"]["dirty"] is None
    assert manifest["created_at"]


def test_distill_provenance_hashes_inputs_and_counts_outputs(tmp_path):
    src = tmp_path / "traces.jsonl"
    src.write_text("{}\n", encoding="utf-8")

    block = provenance.distill_provenance(
        inputs=[{**provenance.file_entry(src), "traces": 2}],
        sft=1,
        failed=1,
        by_group={"trace": {"traces": 2, "sft": 1, "failed": 1}},
        agents={"wf:demo": 2},
        splits={"holdout": 2},
        code_path=Path(provenance.__file__),
    )

    assert block["kind"] == "distill"
    assert block["inputs"][0]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert block["inputs"][0]["traces"] == 2
    assert block["outputs"] == {"sft": 1, "failed": 1}
    assert block["agents"] == {"wf:demo": 2}
    assert block["code"]["sha256"], "distill 代码自身也要有哈希"


def test_distill_writes_manifest_with_provenance(tmp_path):
    """真跑一次 distill：产物里必须带 provenance，且哈希对得上源文件。"""
    trace = Trace(
        task_id="NOPE",
        agent="probe",
        turns=[Turn(index=0, user_input="随便问问", answer="随便答答")],
    )
    src = tmp_path / "traces.jsonl"
    save_traces([trace], src)

    out = tmp_path / "out"
    assert distill.main(["--trace", str(src), "--out", str(out)]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["provenance"]

    assert block["inputs"][0]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert block["inputs"][0]["traces"] == 1
    assert block["outputs"] == {"sft": 0, "failed": 0}
    assert block["agents"] == {"probe": 1}
    # 既有字段不能被 provenance 挤掉：旧读取方仍要能跑
    assert manifest["total"]["sft"] == 0
    assert manifest["tasks"] > 0
