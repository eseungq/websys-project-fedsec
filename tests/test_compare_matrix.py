"""compare_matrix.py: runs.csv → 비교표·그래프 파일."""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_matrix import OUTPUT_FILES, accuracy_chart, build_outputs, read_runs

from fedsec.experiment.comparison import CONDITIONS, parse_runs

ACC = {("normal", "fedavg"): 0.76, ("normal", "clip_fedavg"): 0.75, ("normal", "median"): 0.77,
       ("attack", "fedavg"): 0.10, ("attack", "clip_fedavg"): 0.60, ("attack", "median"): 0.65}


def _rows(envs=("baseline", "delay"), seeds=(0, 1)) -> list[dict]:
    return [{"name": f"{env}-{attack}-{agg}-s{seed}", "env": env, "attack": attack, "aggregator": agg,
             "seed": str(seed), "status": "done", "final_test_acc": str(ACC[(attack, agg)]),
             "diverged_round": "13" if (attack, agg) == ("attack", "fedavg") else "",
             "rounds_completed": "12" if (attack, agg) == ("attack", "fedavg") else "40"}
            for env in envs for seed in seeds for attack, agg in CONDITIONS]


def _write_runs(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


ENVS, SEEDS, ROUNDS = ["baseline", "delay"], [0, 1], 40


def test_build_outputs_writes_tables_csv_and_chart():
    outputs = build_outputs(_rows(), ENVS, SEEDS, ROUNDS)

    assert set(outputs) == set(OUTPUT_FILES)
    assert "최종 테스트 정확도" in outputs["comparison.md"]
    assert "명세의 세 값" in outputs["comparison.md"]
    assert outputs["comparison.csv"].startswith("env,metric,defense,seed")
    ET.fromstring(outputs["accuracy.svg"])


def test_accuracy_chart_leaves_diverged_condition_without_bar():
    # 환경 2 × 조건 6 중 공격 FedAvg 2칸은 발산이라 막대 없이 '발산'만 적는다
    root = ET.fromstring(accuracy_chart(parse_runs(_rows(), ROUNDS), ENVS, SEEDS))

    bars = [r for r in root.iter("{http://www.w3.org/2000/svg}rect") if r.get("class") == "bar"]
    texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]

    assert len(bars) == 2 * 5
    assert texts.count("발산") == 2


def test_read_runs_reports_missing_file(tmp_path):
    with pytest.raises(ValueError, match="runs.csv"):
        read_runs(tmp_path / "runs.csv")


def test_read_runs_reads_rows(tmp_path):
    path = _write_runs(tmp_path / "runs.csv", _rows())

    assert len(read_runs(path)) == 2 * 2 * 6


def test_build_outputs_rejects_missing_environment():
    # 같은 prefix에서 --envs baseline만 돌리면 runs.csv에 기준 환경만 남는다 → 전체 표를 덮어쓰면 안 됨
    with pytest.raises(ValueError, match="delay"):
        build_outputs(_rows(envs=("baseline",)), ENVS, SEEDS, ROUNDS)


def test_build_outputs_rejects_missing_seed():
    with pytest.raises(ValueError, match="s1"):
        build_outputs(_rows(seeds=(0,)), ENVS, SEEDS, ROUNDS)


def test_build_outputs_rejects_run_outside_plan():
    with pytest.raises(ValueError, match="예정에 없는"):
        build_outputs(_rows(seeds=(0, 1, 2)), ENVS, SEEDS, ROUNDS)


def test_truncated_last_row_is_value_error(tmp_path):
    path = _write_runs(tmp_path / "runs.csv", _rows())
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: text.rstrip("\n").rfind(",")] + "\n", encoding="utf-8")   # 마지막 열 하나를 잘라냄

    with pytest.raises(ValueError):
        build_outputs(read_runs(path), ENVS, SEEDS, ROUNDS)


@pytest.mark.parametrize("envs, seeds", [(["baseline", "baseline", "delay"], SEEDS), (ENVS, [0, 0, 1])])
def test_build_outputs_rejects_duplicate_env_or_seed(envs, seeds):
    # 시드 0이 두 번 들어가면 평균이 바뀐다 (run_matrix.py처럼 거부)
    with pytest.raises(ValueError, match="겹치지 않는"):
        build_outputs(_rows(), envs, seeds, ROUNDS)


@pytest.mark.parametrize("change", ["cut", "extra"])
def test_read_runs_rejects_row_with_wrong_column_count(tmp_path, change):
    # 비교에 쓰지 않는 뒤쪽 열만 잘려도(또는 열이 남아도) 깨진 파일로 본다
    rows = [{**row, "dirty": "False"} for row in _rows()]
    path = _write_runs(tmp_path / "runs.csv", rows)
    text = path.read_text(encoding="utf-8").rstrip("\n")
    text = text[: text.rfind(",")] if change == "cut" else text + ",extra"
    path.write_text(text + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="열 수"):
        read_runs(path)
