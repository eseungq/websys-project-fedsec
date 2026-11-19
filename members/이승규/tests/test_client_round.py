"""클라이언트 프로그램 라운드 1회를 서버(Flask 테스트 클라이언트)와 끝까지 돌려 본다.

이미지 8장짜리 클라이언트 10명을 CPU에서 학습한다. 가상 시간 기준 표본 수도 8장으로 맞춰
정상 1.0, 지연 3.0, 마감 1.5가 되게 한다.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fedsec.aggregation.fedavg import apply_update, weighted_average
from fedsec.aggregation.norms import update_norm
from fedsec.client.local_train import TrainConfig, local_update
from fedsec.client.program import check_decision, check_server, client_plan, fetch_snapshot, run_round
from fedsec.data.cifar10 import TensorData
from fedsec.environments.acceptance import judge_round
from fedsec.environments.config import load_env_config
from fedsec.environments.virtual_time import VirtualTiming
from fedsec.experiment.config import ExperimentConfig, load_attack_defense
from fedsec.experiment.runtime import client_generator
from fedsec.models.cnn import build_model
from fedsec.server.app import create_app
from fedsec.server.scenario import build_defense, scenario_policy
from fedsec.server.state import ServerState, initial_global_state

MEMBER_ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = MEMBER_ROOT / "configs" / "env"
AD = load_attack_defense(MEMBER_ROOT / "configs" / "attack_defense.yaml")
NUM_CLIENTS = 10
PER_CLIENT = 8
SEED = 0
CPU = torch.device("cpu")
TIMING = VirtualTiming(base_time=1.0, base_samples=PER_CLIENT, delay=2.0, deadline=1.5)
CFG = ExperimentConfig(seed=SEED, split="unused", data_root="unused", rounds=1, model="simple_cnn",
                       train=TrainConfig(lr=0.01, momentum=0.9, weight_decay=0.0, local_ep=1, local_bs=4), eval_bs=16)
CLIENT_KEYS = ("experiment", "round", "client", "malicious", "selected", "attacked", "defense", "num_samples",
               "train_loss", "update_norm", "norm_submitted", "norm_defended", "accepted", "reason",
               "virtual_finish", "train_sec", "comm_sec")
ROUND_KEYS = ("experiment", "round", "val_acc", "num_selected", "num_accepted", "num_late", "malicious_selected",
              "malicious_accepted", "overall_malicious_ratio", "accepted_malicious_ratio", "attack_applied",
              "updated", "skip_reason", "aggregator", "aggregate_norm", "virtual_elapsed", "train_sec", "total_sec")


class FlaskApi:
    """ServerApi와 같은 메서드를 Flask 테스트 클라이언트로 부른다 (실제 HTTP 없이)."""

    def __init__(self, client):
        self._client = client

    def status(self) -> dict:
        return self._client.get("/status").get_json()["data"]

    def model(self) -> dict:
        return self._client.get("/model").get_json()["data"]

    def submit(self, payload: dict) -> dict:
        return self._client.post("/update", json=payload).get_json()

    def aggregate(self) -> dict:
        return self._client.post("/aggregate", json={}).get_json()["data"]


def _data(seed: int, n: int) -> TensorData:
    g = torch.Generator().manual_seed(seed)
    return TensorData(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))


CLIENTS = [_data(100 + cid, PER_CLIENT) for cid in range(NUM_CLIENTS)]
VAL = _data(999, 16)


def _env(name: str, **changes):
    return replace(load_env_config(ENV_DIR / f"{name}.yaml"), timing=TIMING, **changes)


def _round(env, attack_defense):
    """서버를 새로 띄우고 라운드 1을 돈다. (rounds 행, clients 행들, 다음 스냅샷, 계획, 시작 상태)"""
    policy = scenario_policy(env, build_defense(attack_defense), SEED, NUM_CLIENTS)
    start = initial_global_state(SEED, CFG.model)
    api = FlaskApi(create_app(ServerState(start, num_clients=NUM_CLIENTS, policy=policy)).test_client())
    plan = client_plan("test", env, attack_defense, SEED, NUM_CLIENTS)
    snapshot = fetch_snapshot(api, CPU, expected_round=1)
    row, rows, next_snapshot, _ = run_round(api, build_model(CFG.model), snapshot, CLIENTS, VAL, CFG, plan,
                                         env.min_accepted, CPU)
    return row, rows, next_snapshot, plan, start


@pytest.mark.parametrize("env_name", ["baseline", "delay", "partial", "label_skew"])
@pytest.mark.parametrize("aggregator", ["fedavg", "clip_fedavg", "median"])
def test_every_env_and_aggregator_fills_log_columns(env_name, aggregator):
    row, rows, _, plan, _ = _round(_env(env_name), AD.with_overrides(attack=True, aggregator=aggregator))

    assert all(row[key] not in ("", None) for key in ROUND_KEYS if key not in ("skip_reason",))
    assert len(rows) == NUM_CLIENTS and all(tuple(r) == CLIENT_KEYS for r in rows)
    selected = [r for r in rows if r["selected"]]
    assert len(selected) == row["num_selected"]
    filled = [key for key in CLIENT_KEYS if key not in ("reason", "norm_defended")]
    assert all(r[key] != "" for r in selected for key in filled)
    assert all(r["norm_defended"] != "" for r in selected if r["accepted"])
    assert {r["defense"] for r in rows} == {aggregator}


def test_delay_env_excludes_three_late_and_counts_ratio():
    row, rows, _, plan, _ = _round(_env("delay"), AD.with_overrides(attack=True))

    late = sorted(r["client"] for r in rows if r["reason"] == "late")
    assert late == list(plan.delayed)
    assert not set(plan.delayed) & set(plan.malicious)
    assert (row["num_selected"], row["num_accepted"], row["num_late"]) == (10, 7, 3)
    # 악성 2명은 지연이 아니라서 모두 채택 → 2/7
    assert row["accepted_malicious_ratio"] == pytest.approx(round(2 / 7, 4))
    assert row["overall_malicious_ratio"] == 0.2
    assert row["virtual_elapsed"] == 1.5
    assert all(r["virtual_finish"] == 3.0 for r in rows if r["client"] in plan.delayed)


def test_baseline_virtual_elapsed_is_latest_finish():
    row, _, _, _, _ = _round(_env("baseline"), AD)

    assert (row["num_late"], row["virtual_elapsed"]) == (0, 1.0)


def test_partial_env_selects_five_and_marks_others():
    row, rows, _, _, _ = _round(_env("partial"), AD)

    assert row["num_selected"] == 5
    unselected = [r for r in rows if not r["selected"]]
    assert len(unselected) == 5
    assert all(r["accepted"] == "" and r["num_samples"] == "" for r in unselected)


def test_attack_submits_same_scaled_vector_for_both_malicious():
    row, rows, _, plan, _ = _round(_env("baseline"), AD.with_overrides(attack=True))

    attackers = [r for r in rows if r["attacked"]]
    assert sorted(r["client"] for r in attackers) == list(plan.malicious)
    assert row["attack_applied"] is True
    # 두 악성의 제출 노름은 같고 (같은 벡터), 정상 클라이언트는 조작 전·후가 같다
    assert attackers[0]["norm_submitted"] == attackers[1]["norm_submitted"]
    assert all(r["norm_submitted"] == r["update_norm"] for r in rows if not r["attacked"])


def test_single_malicious_submits_own_update_times_minus_five():
    # 부분 참여에서 악성이 1명만 뽑히는 상황을 고정 ID로 만든다
    env = _env("baseline", per_round=None, malicious_ids=(0,), num_malicious=1)

    _, rows, _, _, _ = _round(env, AD.with_overrides(attack=True))

    attacker = rows[0]
    assert attacker["attacked"] is True
    # 로그 노름은 소수 4자리로 반올림되어 있어 5배하면 반올림 오차도 5배 (최대 0.00025 + 0.00005)
    assert attacker["norm_submitted"] == pytest.approx(5 * attacker["update_norm"], abs=3e-4)


def test_no_attack_keeps_updates():
    row, rows, _, _, _ = _round(_env("baseline"), AD)

    assert row["attack_applied"] is False
    assert not any(r["attacked"] for r in rows)


def test_clip_fedavg_defended_norms_are_at_most_clip_norm():
    clip = 0.01  # 학습 업데이트가 모두 넘도록 아주 작게
    _, rows, _, _, _ = _round(_env("baseline"), AD.with_overrides(attack=True, aggregator="clip_fedavg", clip_norm=clip))

    accepted = [r for r in rows if r["accepted"] is True]
    assert all(r["norm_defended"] == pytest.approx(clip, abs=1e-4) for r in accepted)


def test_two_accepted_keeps_model_and_reports_ratio():
    # 지연 8명 고정 → 채택 2명(클라이언트 8, 9): 갱신 안 함, 비율은 숫자
    env = _env("delay", delayed_ids=tuple(range(8)), num_delayed=8, malicious_ids=(8, 3), delayed_avoid_malicious=False)

    row, _, next_snapshot, _, start = _round(env, AD.with_overrides(attack=True))

    assert (row["num_accepted"], row["updated"], row["skip_reason"]) == (2, False, "too_few_accepted")
    assert row["accepted_malicious_ratio"] == 0.5
    assert row["aggregate_norm"] is None
    assert all(torch.equal(next_snapshot.state[k], start[k]) for k in start)


def test_zero_accepted_gives_na():
    env = _env("delay", delayed_ids=tuple(range(10)), num_delayed=10, delayed_avoid_malicious=False)

    row, _, _, _, _ = _round(env, AD)

    assert (row["num_accepted"], row["accepted_malicious_ratio"]) == (0, "N/A")
    assert (row["updated"], row["skip_reason"]) == (False, "no_accepted")


def test_server_path_matches_plain_fedavg_for_baseline():
    # 정상·FedAvg·기준 환경: 서버를 거친 다음 모델 = 같은 업데이트를 직접 가중 평균한 모델
    _, _, next_snapshot, _, start = _round(_env("baseline"), AD)
    model = build_model(CFG.model)
    results = [local_update(model, start, data, CFG.train, client_generator(SEED, 1, cid))
               for cid, data in enumerate(CLIENTS)]

    expected = apply_update(start, weighted_average([r.update for r in results], [r.num_samples for r in results]))

    assert all(torch.equal(next_snapshot.state[k], expected[k]) for k in expected)


def test_same_seed_gives_same_plan_for_any_attack_or_aggregator():
    env = _env("delay")

    plans = [client_plan("x", env, AD.with_overrides(attack=a, aggregator=g), SEED, NUM_CLIENTS)
             for a in (False, True) for g in ("fedavg", "median")]

    assert len({(p.malicious, p.delayed) for p in plans}) == 1


def test_mismatched_decision_is_an_error():
    decision = judge_round([0, 1, 2], {}, malicious=set(), num_clients=3)

    with pytest.raises(RuntimeError):
        check_decision(decision, {"accepted_clients": [0, 1], "updated": True})
    check_decision(decision, {"accepted_clients": [0, 1, 2], "updated": True})


def test_norm_columns_match_update_norm_definition():
    _, rows, _, _, start = _round(_env("baseline"), AD)
    model = build_model(CFG.model)

    first = local_update(model, start, CLIENTS[0], CFG.train, client_generator(SEED, 1, 0))

    assert rows[0]["update_norm"] == round(update_norm(first.update), 4)


@pytest.mark.parametrize("change", [{"min_accepted": 8}, {"config_per_round": 5}])
def test_check_server_rejects_other_round_policy(change):
    env = _env("baseline")
    plan = client_plan("test", env, AD, SEED, NUM_CLIENTS)
    defense = {key: plan.defense[key] for key in plan.defense}
    status = {"round": 1, "model_version": 0, "num_clients": NUM_CLIENTS, "min_accepted": env.min_accepted,
              "config": {"seed": SEED, "model": CFG.model, "env": env.name, "per_round": env.per_round, **defense}}
    check_server(status, CFG, NUM_CLIENTS, plan)  # 같으면 통과
    if "min_accepted" in change:
        status = {**status, "min_accepted": change["min_accepted"]}
    else:
        status = {**status, "config": {**status["config"], "per_round": change["config_per_round"]}}

    with pytest.raises(RuntimeError):
        check_server(status, CFG, NUM_CLIENTS, plan)
