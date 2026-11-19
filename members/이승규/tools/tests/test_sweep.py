import sys

import pytest
import yaml
from sweep import build_command, config_mismatches, expected_config, parse_plan, run_status, run_sweep

# run_experiment.py 대신 쓰는 가짜: base.yaml에 옵션을 덮어쓴 설정과 summary.json만 쓰고,
# 이름에 fail이 있으면 실패한다.
FAKE_RUNNER = '''
import argparse, json, sys
from pathlib import Path
import yaml
p = argparse.ArgumentParser()
p.add_argument("--name"); p.add_argument("--seed", type=int); p.add_argument("--lr", type=float)
a = p.parse_args()
if "fail" in a.name:
    sys.exit(3)
base = yaml.safe_load(Path("configs/base.yaml").read_text())
train = {k: v for k, v in base["train"].items() if k != "optimizer"}
config = {**base, "seed": a.seed if a.seed is not None else base["seed"],
          "train": {**train, "lr": a.lr if a.lr is not None else train["lr"]}, "num_clients": 10}
out = Path("experiments") / a.name
out.mkdir(parents=True)
(out / "config.yaml").write_text(yaml.safe_dump(config))
(out / "summary.json").write_text(json.dumps({"final_val_acc": 0.5}))
with open("calls.txt", "a") as f:
    f.write(a.name + "\\n")
'''


BASE = {"seed": 0, "split": "configs/splits/iid_seed0.json", "data_root": "data", "rounds": 50,
        "model": "simple_cnn", "eval_bs": 1000,
        "train": {"optimizer": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "local_ep": 1, "local_bs": 32}}
BASE_TRAIN = {k: v for k, v in BASE["train"].items() if k != "optimizer"}


def write_base(root, base):
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")


@pytest.fixture
def root(tmp_path):
    write_base(tmp_path, BASE)
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


def test_expected_config_overrides_base_and_drops_optimizer():
    result = expected_config(BASE, {"seed": 2, "lr": 0.05})

    assert result == {**{k: v for k, v in BASE.items() if k != "train"}, "seed": 2,
                      "train": {**BASE_TRAIN, "lr": 0.05}}
    assert BASE["seed"] == 0 and BASE["train"]["lr"] == 0.01      # 원본은 그대로


def test_config_mismatches_compares_all_expected_values():
    saved = {"seed": 0, "model": "simple_cnn", "num_clients": 10,
             "train": {"lr": 0.05, "momentum": 0.5, "weight_decay": 0.0}}
    expected = {"seed": 0, "model": "simple_cnn", "train": {"lr": 0.05, "momentum": 0.9, "weight_decay": 0.0005}}

    result = config_mismatches(saved, expected)

    assert result == ["train.momentum: 지금 설정 0.9, 기존 0.5", "train.weight_decay: 지금 설정 0.0005, 기존 0.0"]


def test_config_mismatches_reports_value_missing_in_saved_config():
    result = config_mismatches({"seed": 0, "train": {}}, {"seed": 0, "train": {"lr": 0.01}})

    assert result == ["train.lr: 지금 설정 0.01, 기존 None"]


def test_expected_config_converts_types_like_load_config():
    base = {**BASE, "rounds": 40.0, "train": {**BASE["train"], "lr": "1e-2", "weight_decay": "5e-4", "local_bs": "32"}}

    result = expected_config(base, {})

    assert result["rounds"] == 40 and isinstance(result["rounds"], int)
    assert result["train"] == {**BASE_TRAIN, "lr": 0.01, "weight_decay": 0.0005, "local_bs": 32}


def test_expected_config_keeps_only_keys_written_to_config_yaml():
    base = {**BASE, "note": "x", "attack": {"kind": "sign_flip"}}

    result = expected_config(base, {})

    assert "note" not in result and "attack" not in result


def test_expected_config_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="형식이 잘못"):
        expected_config({**BASE, "train": {**BASE["train"], "lr": "abc"}}, {})


@pytest.mark.parametrize("value", [1.9, "1.9"])
def test_expected_config_rejects_fractional_integer(value):
    with pytest.raises(ValueError, match="정수여야 합니다"):
        expected_config({**BASE, "train": {**BASE["train"], "local_ep": value}}, {})



@pytest.mark.parametrize("value", [9007199254740993, "9007199254740993"])
def test_expected_config_keeps_large_seed_exact(value):
    assert expected_config({**BASE, "seed": value}, {})["seed"] == 9007199254740993


def test_expected_config_rejects_bool_integer():
    with pytest.raises(ValueError, match="정수여야 합니다"):
        expected_config({**BASE, "train": {**BASE["train"], "local_ep": True}}, {})

def test_expected_config_rejects_missing_required_key():
    train = {k: v for k, v in BASE["train"].items() if k != "weight_decay"}

    with pytest.raises(ValueError, match="필수 항목이 없습니다: weight_decay"):
        expected_config({**BASE, "train": train}, {})


def test_config_mismatches_compares_ints_exactly():
    assert config_mismatches({"seed": 10**20}, {"seed": 10**20 + 1}) == [f"seed: 지금 설정 {10**20 + 1}, 기존 {10**20}"]


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


def test_run_sweep_stops_when_base_config_changed_after_run(root, capsys):
    runs = parse_plan({"runs": [{"name": "a", "lr": 0.1}]})      # weight_decay는 실행 목록에 없음
    run_sweep(runs, root, sys.executable)
    write_base(root, {**BASE, "train": {**BASE["train"], "weight_decay": 0.0005}})

    result = run_sweep(runs, root, sys.executable)

    assert result == 1
    assert calls(root) == ["sweep/a"]                              # 기존 결과를 완료로 치지 않음
    assert "train.weight_decay: 지금 설정 0.0005, 기존 0.0" in capsys.readouterr().out


def test_run_sweep_stops_when_base_config_missing(tmp_path, capsys):
    result = run_sweep(parse_plan({"runs": [{"name": "a", "lr": 0.1}]}), tmp_path, sys.executable)

    assert result == 1
    assert "기본 설정을 읽지 못했습니다" in capsys.readouterr().out


def test_run_sweep_dry_run_executes_nothing(root):
    runs = parse_plan({"runs": [{"name": "a", "lr": 0.1}]})

    result = run_sweep(runs, root, sys.executable, dry_run=True)

    assert result == 0
    assert calls(root) == []


@pytest.mark.parametrize("text", ["", "- 1\n", "train: 5\n"])
def test_run_sweep_stops_on_malformed_saved_config(root, capsys, text):
    out = root / "experiments" / "sweep" / "a"
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(text)
    (out / "summary.json").write_text("{}")

    result = run_sweep(parse_plan({"runs": [{"name": "a", "lr": 0.1}]}), root, sys.executable)

    assert result == 1
    assert "설정 묶음이 아닙니다" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["", "- 1\n", "train: 5\n", "seed: [\n"])
def test_run_sweep_stops_on_malformed_base_config(root, capsys, text):
    (root / "configs" / "base.yaml").write_text(text)

    result = run_sweep(parse_plan({"runs": [{"name": "a", "lr": 0.1}]}), root, sys.executable)

    assert result == 1
    assert calls(root) == []
    assert "기본 설정을 읽지 못했습니다" in capsys.readouterr().out


def test_run_sweep_stops_on_unsupported_optimizer_before_checking_results(root, capsys):
    runs = parse_plan({"runs": [{"name": "a", "lr": 0.1}]})
    run_sweep(runs, root, sys.executable)
    write_base(root, {**BASE, "train": {**BASE["train"], "optimizer": "adam"}})

    result = run_sweep(runs, root, sys.executable)

    assert result == 1
    assert "지원하지 않는 옵티마이저입니다: adam" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["", "{", "[]"])
def test_run_status_rejects_broken_summary(tmp_path, text):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "summary.json").write_text(text)

    with pytest.raises(ValueError, match="완료 기록이 깨졌습니다"):
        run_status(tmp_path / "a", {})


def test_run_sweep_stops_when_base_misses_required_key_before_reusing_result(root, capsys):
    runs = parse_plan({"runs": [{"name": "a", "lr": 0.1}]})
    run_sweep(runs, root, sys.executable)
    write_base(root, {**BASE, "train": {k: v for k, v in BASE["train"].items() if k != "weight_decay"}})

    result = run_sweep(runs, root, sys.executable)

    assert result == 1
    assert "필수 항목이 없습니다: weight_decay" in capsys.readouterr().out
