import sys

import pytest
import yaml

from sweep import build_command, config_mismatches, parse_plan, run_status, run_sweep

# run_experiment.py 대신 쓰는 가짜: 설정과 summary.json만 쓰고, 이름에 fail이 있으면 실패한다.
FAKE_RUNNER = '''
import argparse, json, sys
from pathlib import Path
import yaml
p = argparse.ArgumentParser()
p.add_argument("--name"); p.add_argument("--seed", type=int); p.add_argument("--lr", type=float)
a = p.parse_args()
if "fail" in a.name:
    sys.exit(3)
out = Path("experiments") / a.name
out.mkdir(parents=True)
(out / "config.yaml").write_text(yaml.safe_dump({"seed": a.seed, "train": {"lr": a.lr}}))
(out / "summary.json").write_text(json.dumps({"final_val_acc": 0.5}))
with open("calls.txt", "a") as f:
    f.write(a.name + "\\n")
'''


@pytest.fixture
def root(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_experiment.py").write_text(FAKE_RUNNER, encoding="utf-8")
    return tmp_path


def calls(root):
    path = root / "calls.txt"
    return path.read_text().split() if path.exists() else []


def test_parse_plan_merges_common_values_under_each_run():
    plan = {"common": {"seed": 0, "rounds": 50}, "runs": [{"name": "a", "lr": 0.05, "rounds": 10}]}

    runs = parse_plan(plan)

    assert runs == [{"name": "a", "options": {"seed": 0, "rounds": 10, "lr": 0.05}}]


@pytest.mark.parametrize("plan", [
    {},
    {"runs": []},
    {"runs": [{"lr": 0.1}]},                                  # 이름 없음
    {"runs": [{"name": "../x"}]},                             # 폴더 밖으로 나가는 이름
    {"runs": [{"name": "a"}, {"name": "a"}]},                 # 이름 겹침
    {"runs": [{"name": "a", "learning_rate": 0.1}]},          # 모르는 옵션
    {"runs": [{"name": "a", "local_ep": 1.5}]},               # 정수 옵션에 소수
    {"runs": [{"name": "a", "lr": "0.1"}]},                   # 숫자가 아님
    {"common": {"optimizer": "adam"}, "runs": [{"name": "a"}]},
])
def test_parse_plan_rejects_invalid_plan(plan):
    with pytest.raises(ValueError):
        parse_plan(plan)


def test_build_command_uses_sweep_folder_and_dashed_flags():
    run = {"name": "wd", "options": {"weight_decay": 0.0005, "local_ep": 2}}

    cmd = build_command("py", run)

    assert cmd == ["py", "scripts/run_experiment.py", "--name", "sweep/wd",
                   "--weight-decay", "0.0005", "--local-ep", "2"]


def test_config_mismatches_compares_top_level_and_train_values():
    config = {"seed": 0, "rounds": 50, "train": {"lr": 0.05, "momentum": 0.5}}

    result = config_mismatches(config, {"seed": 0, "lr": 0.05, "momentum": 0.9})

    assert result == ["momentum: 요청 0.9, 기존 0.5"]


def test_run_status_new_when_folder_missing(tmp_path):
    assert run_status(tmp_path / "none", {}) == "new"


def test_run_status_rejects_unfinished_folder(tmp_path):
    (tmp_path / "cut").mkdir()

    with pytest.raises(ValueError, match="끊긴 실행"):
        run_status(tmp_path / "cut", {})


def test_run_sweep_runs_in_order_then_skips_finished_runs(root):
    runs = parse_plan({"common": {"seed": 0}, "runs": [{"name": "a", "lr": 0.1}, {"name": "b", "lr": 0.2}]})

    first = run_sweep(runs, root, sys.executable)
    second = run_sweep(runs, root, sys.executable)

    assert (first, second) == (0, 0)
    assert calls(root) == ["sweep/a", "sweep/b"]          # 두 번째는 모두 건너뜀


def test_run_sweep_stops_at_failed_run(root, capsys):
    runs = parse_plan({"runs": [{"name": "a", "lr": 0.1}, {"name": "fail", "lr": 0.2}, {"name": "c", "lr": 0.3}]})

    result = run_sweep(runs, root, sys.executable)

    assert result == 1
    assert calls(root) == ["sweep/a"]
    assert "[2/3] fail 실패 (종료 코드 3)" in capsys.readouterr().out


def test_run_sweep_stops_when_existing_result_has_other_settings(root, capsys):
    out = root / "experiments" / "sweep" / "a"
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"seed": 0, "train": {"lr": 0.5}}))
    (out / "summary.json").write_text("{}")

    result = run_sweep(parse_plan({"runs": [{"name": "a", "lr": 0.1}]}), root, sys.executable)

    assert result == 1
    assert calls(root) == []
    assert "다른 설정" in capsys.readouterr().out


def test_run_sweep_dry_run_executes_nothing(root):
    runs = parse_plan({"runs": [{"name": "a", "lr": 0.1}]})

    result = run_sweep(runs, root, sys.executable, dry_run=True)

    assert result == 0
    assert calls(root) == []
