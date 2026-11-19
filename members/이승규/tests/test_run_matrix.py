"""scripts/run_matrix.py: 4단계 일괄 실행 목록과 이어서 실행하기 위한 상태 판정."""

import json
import sys
from pathlib import Path

import pytest

MEMBER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBER_ROOT / "scripts"))
import yaml  # noqa: E402
from run_matrix import Run, expected_config, index_rows, plan_runs, run_status  # noqa: E402

from fedsec.experiment.config import AGGREGATORS  # noqa: E402

ENVS = ["baseline", "delay", "partial", "label_skew"]


def test_full_matrix_has_72_runs_for_three_seeds():
    runs = plan_runs(ENVS, ["normal", "attack"], list(AGGREGATORS), [0, 1, 2])

    assert len(runs) == 72
    assert len({r.name for r in runs}) == 72
    assert runs[0].name == "baseline-normal-fedavg-s0"
    assert [r.seed for r in runs[:24]] == [0] * 24  # 시드 하나씩 끝낸다


def test_check_matrix_has_12_runs():
    assert len(plan_runs(ENVS, ["attack"], list(AGGREGATORS), [0])) == 12


def test_unknown_choice_is_rejected():
    with pytest.raises(ValueError):
        plan_runs(["windy"], ["attack"], ["fedavg"], [0])


def test_command_always_evaluates_test_and_sets_attack_flag():
    attack = Run("delay", "attack", "median", 1).command("m", rounds=2)
    normal = Run("delay", "normal", "median", 1).command("m", rounds=None)

    assert "--evaluate-test" in attack and "--evaluate-test" in normal
    assert "--attack" in attack and "--no-attack" in normal
    assert attack[attack.index("--rounds") + 1] == "2" and "--rounds" not in normal
    assert attack[attack.index("--env") + 1] == "configs/env/delay.yaml"
    assert attack[attack.index("--name") + 1] == "m/delay-attack-median-s1"


def write_result(out_dir, config, summary):
    out_dir.mkdir(parents=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    (out_dir / "summary.json").write_text(summary, encoding="utf-8")


def test_run_status_and_index(tmp_path):
    runs = plan_runs(["baseline"], ["normal"], ["fedavg", "median", "clip_fedavg"], [0])
    expected = {run: expected_config(run, "m", rounds=None) for run in runs}
    write_result(tmp_path / runs[0].name, {**expected[runs[0]], "server_url": "http://127.0.0.1:1"},
                 json.dumps({"final_test_acc": 0.5}))
    (tmp_path / runs[1].name).mkdir()  # 끊긴 실행

    statuses = {run: run_status(tmp_path / run.name, expected[run])[0] for run in runs}
    rows = index_rows(tmp_path, runs, statuses)

    assert list(statuses.values()) == ["done", "incomplete", "todo"]
    assert [row["status"] for row in rows] == ["done", "incomplete", "todo"]
    assert rows[0]["final_test_acc"] == 0.5 and rows[1]["final_test_acc"] == ""


def test_short_check_result_is_not_reused_for_full_run(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["median"], [0])[0]
    write_result(tmp_path / run.name, expected_config(run, "m", rounds=2), "{}")  # 2라운드 확인 결과

    status, mismatches = run_status(tmp_path / run.name, expected_config(run, "m", rounds=None))

    assert status == "stale"
    assert any(m.startswith("rounds:") for m in mismatches)


def test_other_scenario_result_is_stale(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["median"], [0])[0]
    other = plan_runs(["delay"], ["attack"], ["median"], [0])[0]
    write_result(tmp_path / run.name, expected_config(other, "m", rounds=None), "{}")

    assert run_status(tmp_path / run.name, expected_config(run, "m", rounds=None))[0] == "stale"


def test_broken_summary_counts_as_incomplete(tmp_path):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    write_result(tmp_path / run.name, expected_config(run, "m", rounds=None), "{")

    assert run_status(tmp_path / run.name, expected_config(run, "m", rounds=None))[0] == "incomplete"
