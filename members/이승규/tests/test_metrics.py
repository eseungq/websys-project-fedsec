"""클래스별·클라이언트별 성능과 라운드별 모델 저장.

수치 예제: 클래스 3개, 클래스별 정확도 (1.0, 0.5, 0.0)
- 분포 (1, 0, 0) → 1.0, 분포 (0.5, 0.5, 0) → 0.75, 균등 → 0.5
"""

import csv
import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from fedsec.client import program
from fedsec.client.program import checkpoint_path, client_plan, run_clients, should_save
from fedsec.data.cifar10 import TensorData
from fedsec.experiment.evaluate import Evaluation, evaluate, evaluate_detailed
from fedsec.experiment.metrics import (class_accuracy, class_rows, client_acc_rows, distribution_weighted_accuracy,
                                       label_distribution)
from fedsec.server.app import create_app
from fedsec.server.scenario import build_defense, scenario_policy
from fedsec.server.state import ServerState, initial_global_state
from tests.test_client_round import AD, CFG, CLIENTS, NUM_CLIENTS, SEED, VAL, FlaskApi, _env

RESULT = Evaluation(loss=0.0, acc=0.5, class_correct=(2, 1, 0), class_total=(2, 2, 2))


class FixedClassifier(nn.Module):
    """입력 첫 픽셀 값을 그대로 클래스로 예측한다 (정답 위치를 손으로 정함)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted = x[:, 0, 0, 0].long()
        return nn.functional.one_hot(predicted, num_classes=3).float() * 10


def test_class_accuracy_marks_missing_class_none():
    result = replace(RESULT, class_correct=(2, 1, 0), class_total=(2, 2, 0))

    assert class_accuracy(result) == (1.0, 0.5, None)


@pytest.mark.parametrize("distribution, expected", [((1, 0, 0), 1.0), ((0.5, 0.5, 0), 0.75), ((1 / 3,) * 3, 0.5)])
def test_distribution_weighted_accuracy_matches_hand_values(distribution, expected):
    assert distribution_weighted_accuracy(class_accuracy(RESULT), distribution) == pytest.approx(expected)


def test_weighted_accuracy_renormalizes_without_missing_class():
    assert distribution_weighted_accuracy((1.0, None, 0.0), (0.5, 0.25, 0.25)) == pytest.approx(1.0 * 0.5 / 0.75)


def test_weighted_accuracy_rejects_bad_inputs():
    with pytest.raises(ValueError):
        distribution_weighted_accuracy((1.0, 0.5), (1, 0, 0))
    with pytest.raises(ValueError):
        distribution_weighted_accuracy((None, 0.5, 0.0), (1, 0, 0))


def test_label_distribution_sums_to_one():
    assert label_distribution(torch.tensor([0, 0, 1, 2]), num_classes=3) == (0.5, 0.25, 0.25)
    with pytest.raises(ValueError):
        label_distribution(torch.tensor([], dtype=torch.int64))


def test_evaluate_detailed_counts_per_class_and_matches_evaluate():
    images = torch.zeros(5, 3, 32, 32)
    images[:, 0, 0, 0] = torch.tensor([0, 1, 2, 2, 0])   # 예측
    data = TensorData(images, torch.tensor([0, 1, 2, 0, 1]))  # 정답: 맞힌 것 0, 1, 2

    result = evaluate_detailed(FixedClassifier(), data, batch_size=2, num_classes=3)

    assert result.class_correct == (1, 1, 1) and result.class_total == (2, 2, 1)
    assert result.acc == pytest.approx(0.6)
    assert evaluate(FixedClassifier(), data, batch_size=2)[1] == pytest.approx(result.acc)


def test_rows_have_one_line_per_class_and_client():
    rows = class_rows(3, "val", RESULT)
    clients = client_acc_rows(3, "val", RESULT, [(1, 0, 0), (0, 0, 1)])

    assert [r["acc"] for r in rows] == [1.0, 0.5, 0.0]
    assert [(r["client"], r["weighted_acc"]) for r in clients] == [(0, 1.0), (1, 0.0)]


@pytest.mark.parametrize("round_idx, expected", [(0, True), (1, False), (5, True), (7, True), (10, True)])
def test_should_save_round_zero_multiples_and_last(round_idx, expected):
    assert should_save(round_idx, rounds=7, save_every=5) is expected


def test_save_every_zero_keeps_only_first_and_last():
    assert [r for r in range(8) if should_save(r, rounds=7, save_every=0)] == [0, 7]


def test_run_clients_writes_metrics_and_checkpoints(tmp_path, monkeypatch):
    env = _env("baseline")
    cfg = replace(CFG, rounds=2)
    policy = scenario_policy(env, build_defense(AD), SEED, NUM_CLIENTS)
    defense = {key: value for key, value in AD.to_dict().items() if not key.startswith("attack")}
    state = ServerState(initial_global_state(SEED, cfg.model), num_clients=NUM_CLIENTS, policy=policy,
                        config={"seed": SEED, "model": cfg.model, "env": env.name, "per_round": env.per_round,
                                **defense})
    api = FlaskApi(create_app(state).test_client())
    monkeypatch.setattr(program, "ServerApi", lambda url: api)
    monkeypatch.setattr(program, "_load_data", lambda cfg, root, device: (VAL, CLIENTS, VAL))
    monkeypatch.setattr(program, "pick_device", lambda: torch.device("cpu"))
    plan = client_plan("metrics", env, AD, SEED, NUM_CLIENTS)

    summary = run_clients(cfg, tmp_path, repo_root=tmp_path, server_url="unused", plan=plan,
                          evaluate_test=True, save_every=5)

    with (tmp_path / "class_acc.csv").open(encoding="utf-8") as f:
        class_lines = list(csv.DictReader(f))
    with (tmp_path / "client_acc.csv").open(encoding="utf-8") as f:
        client_lines = list(csv.DictReader(f))
    assert len(class_lines) == 3 * 10      # 검증 2라운드 + 테스트 1회, 클래스 10개
    assert len(client_lines) == 3 * NUM_CLIENTS
    assert [r["split"] for r in class_lines[::10]] == ["val", "val", "test"]
    assert sum(int(r["total"]) for r in class_lines if r["round"] == "1") == len(VAL)
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["final_model.pt", "round_000.pt",
                                                                      "round_002.pt"]
    final = torch.load(tmp_path / "models" / "final_model.pt", weights_only=True)
    last = torch.load(checkpoint_path(tmp_path, 2), weights_only=True)
    assert all(torch.equal(final[k], last[k]) for k in final)
    assert json.loads((tmp_path / "summary.json").read_text())["final_test_acc"] == summary["final_test_acc"]


def test_run_clients_rejects_negative_save_every(tmp_path):
    with pytest.raises(ValueError):
        run_clients(CFG, tmp_path, repo_root=tmp_path, server_url="unused", plan=None, save_every=-1)
