"""tools/sweep.py의 expected_config가 run_experiment.py가 쓰는 config.yaml과 같은지 확인한다.
sweep.py는 torch 없이 돌아야 해서 load_config를 쓰지 못하고 같은 규칙을 따로 갖고 있다."""

import sys
from pathlib import Path

import pytest
import yaml

from fedsec.experiment.config import load_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from sweep import config_mismatches, expected_config

from .test_config import write_base


def saved_config(path, options):
    """run_experiment.py가 config.yaml에 쓰는 것과 같은 내용 (RunLog의 yaml 왕복 포함)."""
    cfg = load_config(path).with_overrides(**options)
    return yaml.safe_load(yaml.safe_dump({**cfg.to_dict(), "num_clients": 10}))


@pytest.mark.parametrize("options", [
    {},
    {"seed": 2, "rounds": 50},
    {"lr": 1, "momentum": 0, "weight_decay": 0.001, "local_ep": 2, "local_bs": 64},
])
def test_sweep_expected_config_matches_what_runner_saves(tmp_path, options):
    path = write_base(tmp_path)
    base = yaml.safe_load(path.read_text(encoding="utf-8"))

    expected = expected_config(base, options)
    saved = saved_config(path, options)

    assert config_mismatches(saved, expected) == []
    assert {k: v for k, v in saved.items() if k != "num_clients"} == expected


def test_sweep_expected_config_matches_real_base_yaml():
    path = ROOT / "configs" / "base.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config_mismatches(saved_config(path, {}), expected_config(base, {})) == []
