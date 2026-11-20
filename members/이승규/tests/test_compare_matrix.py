"""compare_matrix.py: runs.csv → 비교표·그래프 파일."""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MEMBER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBER_ROOT / "scripts"))
from compare_matrix import OUTPUT_FILES, accuracy_chart, build_outputs, read_runs, run_order

from fedsec.experiment.comparison import CONDITIONS, parse_runs

ACC = {("normal", "fedavg"): 0.76, ("normal", "clip_fedavg"): 0.75, ("normal", "median"): 0.77,
       ("attack", "fedavg"): 0.10, ("attack", "clip_fedavg"): 0.60, ("attack", "median"): 0.65}


def _rows(envs=("baseline", "delay"), seeds=(0, 1)) -> list[dict]:
    return [{"name": f"{env}-{attack}-{agg}-s{seed}", "env": env, "attack": attack, "aggregator": agg,
             "seed": str(seed), "status": "done", "final_test_acc": str(ACC[(attack, agg)]),
             "diverged_round": "13" if (attack, agg) == ("attack", "fedavg") else ""}
            for env in envs for seed in seeds for attack, agg in CONDITIONS]


def _write_runs(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_run_order_keeps_first_seen_order():
    rows = _rows(envs=("label_skew", "baseline"), seeds=(2, 0))

    assert run_order(rows) == (["label_skew", "baseline"], [2, 0])


def test_build_outputs_writes_tables_csv_and_chart():
    outputs = build_outputs(_rows())

    assert set(outputs) == set(OUTPUT_FILES)
    assert "최종 테스트 정확도" in outputs["comparison.md"]
    assert "명세의 세 값" in outputs["comparison.md"]
    assert outputs["comparison.csv"].startswith("env,metric,defense,seed")
    ET.fromstring(outputs["accuracy.svg"])


def test_accuracy_chart_leaves_diverged_condition_without_bar():
    # 환경 2 × 조건 6 중 공격 FedAvg 2칸은 발산이라 막대 없이 '발산'만 적는다
    root = ET.fromstring(accuracy_chart(parse_runs(_rows()), ["baseline", "delay"], [0, 1]))

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
