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
from fedsec.client.local_train import TrainingDiverged
from fedsec.client.program import checkpoint_path, client_plan, run_clients, should_save
from fedsec.data.cifar10 import TensorData
from fedsec.experiment.evaluate import Evaluation, evaluate, evaluate_detailed
from fedsec.experiment.metrics import (
    class_accuracy,
    class_rows,
    client_acc_rows,
    distribution_weighted_accuracy,
    label_distribution,
)
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
                        config={"seed": SEED, "model": cfg.model, "env": env.name, "per_round": env.per_round, "deadline": env.timing.deadline,
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


def _metrics_run(tmp_path, monkeypatch, rounds, diverge_at=None, attack=True, nan_val_at=None, huge_update_at=None):
    """run_clients를 Flask 테스트 클라이언트로 돌린다. diverge_at 라운드의 로컬 학습에서 발산(TrainingDiverged),
    nan_val_at 라운드 집계 뒤 검증 손실이 NaN (전역 모델이 깨진 경우)."""
    env = _env("baseline")
    cfg = replace(CFG, rounds=rounds)
    policy = scenario_policy(env, build_defense(AD), SEED, NUM_CLIENTS)
    defense = {key: value for key, value in AD.to_dict().items() if not key.startswith("attack")}
    state = ServerState(initial_global_state(SEED, cfg.model), num_clients=NUM_CLIENTS, policy=policy,
                        config={"seed": SEED, "model": cfg.model, "env": env.name, "per_round": env.per_round,
                                "deadline": env.timing.deadline, **defense})
    api = FlaskApi(create_app(state).test_client())
    monkeypatch.setattr(program, "ServerApi", lambda url: api)
    monkeypatch.setattr(program, "_load_data", lambda cfg, root, device: (VAL, CLIENTS, VAL))
    monkeypatch.setattr(program, "pick_device", lambda: torch.device("cpu"))
    original = program.local_update

    def local_update(model, state, data, train, generator):
        if api.status()["round"] == diverge_at:
            raise TrainingDiverged("로컬 학습 손실이 nan입니다 (발산).")
        result = original(model, state, data, train, generator)
        if api.status()["round"] == huge_update_at:   # 원소는 유한(1e20)하지만 노름이 float32 범위를 넘는 업데이트
            return replace(result, update={k: torch.full_like(v, 1e20) for k, v in result.update.items()})
        return result

    monkeypatch.setattr(program, "local_update", local_update)
    original_eval = program.evaluate_detailed

    def evaluate_val(model, data, batch_size):
        result = original_eval(model, data, batch_size)
        # 집계가 끝나 서버 라운드가 nan_val_at + 1이 된 뒤의 검증 평가만 망가뜨린다
        if data is VAL and api.status()["round"] == (nan_val_at or 0) + 1 and nan_val_at:
            return replace(result, loss=float("nan"))
        return result

    monkeypatch.setattr(program, "evaluate_detailed", evaluate_val)
    plan = client_plan("metrics", env, AD.with_overrides(attack=attack), SEED, NUM_CLIENTS)
    return run_clients(cfg, tmp_path, repo_root=tmp_path, server_url="unused", plan=plan, evaluate_test=True)


def test_divergence_ends_run_with_last_global_model(tmp_path, monkeypatch):
    # 공격 + FedAvg는 실제로 13라운드에 발산했다. 실행 전체가 멈추지 않고 발산 전 모델로 결과를 남긴다
    summary = _metrics_run(tmp_path, monkeypatch, rounds=3, diverge_at=2)

    assert (summary["diverged_round"], summary["rounds_completed"]) == (2, 1)
    assert 0 <= summary["final_test_acc"] <= 1
    with (tmp_path / "rounds.csv").open(encoding="utf-8") as f:
        assert [row["round"] for row in csv.DictReader(f)] == ["1"]
    final = torch.load(tmp_path / "models" / "final_model.pt", weights_only=True)
    after_round_1 = torch.load(checkpoint_path(tmp_path, 1), weights_only=True)
    assert all(torch.equal(final[k], after_round_1[k]) for k in final)
    assert json.loads((tmp_path / "summary.json").read_text())["diverged_round"] == 2


def test_run_without_divergence_records_none(tmp_path, monkeypatch):
    summary = _metrics_run(tmp_path, monkeypatch, rounds=2)

    assert (summary["diverged_round"], summary["rounds_completed"]) == (None, 2)


def test_divergence_from_initial_model_still_stops(tmp_path, monkeypatch):
    # 1라운드 로컬 학습의 발산은 초기 모델에서 생긴 것이라 공격 결과가 아니다 (설정 문제) → 그대로 멈춘다
    with pytest.raises(FloatingPointError):
        _metrics_run(tmp_path, monkeypatch, rounds=2, diverge_at=1)



def test_normal_run_divergence_stops(tmp_path, monkeypatch):
    # 발산은 공격 실험의 결과로만 인정한다. 정상 실행이 발산하면 짧은 결과가 정상 기준값으로 섞이지 않게 멈춘다
    with pytest.raises(TrainingDiverged):
        _metrics_run(tmp_path, monkeypatch, rounds=3, diverge_at=2, attack=False)


def test_broken_global_model_is_divergence_of_that_round(tmp_path, monkeypatch):
    # 로컬 손실은 유한한데 집계된 전역 모델이 깨진 경우: 그 라운드를 발산으로 보고 이전 모델로 평가한다
    summary = _metrics_run(tmp_path, monkeypatch, rounds=3, nan_val_at=2)

    assert (summary["diverged_round"], summary["rounds_completed"]) == (2, 1)
    with (tmp_path / "class_acc.csv").open(encoding="utf-8") as f:
        test_rows = [row for row in csv.DictReader(f) if row["split"] == "test"]
    assert {row["round"] for row in test_rows} == {"1"}   # 평가한 모델의 라운드



def test_attack_broken_in_first_round_records_without_accuracy(tmp_path, monkeypatch):
    # 1라운드 집계 직후 전역 모델이 깨지면(공격의 결과) 실행을 멈추지 않고, 정확도 없이 diverged_round=1로 남긴다
    summary = _metrics_run(tmp_path, monkeypatch, rounds=3, nan_val_at=1)

    assert (summary["diverged_round"], summary["rounds_completed"]) == (1, 0)
    assert summary["final_val_acc"] is None and summary["final_test_acc"] is None
    assert json.loads((tmp_path / "summary.json").read_text())["rounds_completed"] == 0



def test_update_with_overflowing_norm_is_divergence_not_crash(tmp_path, monkeypatch):
    # 서버가 non_finite로 거부해 실행 전체가 멈추던 경우: 제출 전에 서버와 같은 노름으로 확인해 발산으로 처리한다
    summary = _metrics_run(tmp_path, monkeypatch, rounds=3, huge_update_at=2)

    assert (summary["diverged_round"], summary["rounds_completed"]) == (2, 1)
