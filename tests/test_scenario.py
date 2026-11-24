"""서버 운영 규칙 조립(server/scenario.py)과 공격·방어 설정 파일."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fedsec.aggregation.fedavg import weighted_average
from fedsec.defenses.clipping import clipped_fedavg
from fedsec.defenses.median import coordinate_median
from fedsec.environments.config import load_env_config
from fedsec.experiment.config import load_attack_defense
from fedsec.server.app import create_app
from fedsec.server.scenario import build_defense, scenario_policy
from fedsec.server.serialization import encode_state
from fedsec.server.state import ServerState

ROOT = Path(__file__).resolve().parents[1]
AD = load_attack_defense(ROOT / "configs" / "attack_defense.yaml")
ENV_DIR = ROOT / "configs" / "env"
# 노름 5, 1, 1인 업데이트 3개 (손 계산용)
UPDATES = [{"w": torch.tensor([3.0, 4.0])}, {"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([0.0, 1.0])}]
WEIGHTS = [1.0, 1.0, 2.0]


# 공격·방어 설정 파일

def test_attack_defense_file_defaults():
    assert (AD.attack, AD.attack_scale, AD.aggregator, AD.clip_norm, AD.clip_before_median) == \
        (False, -5.0, "fedavg", 3.5, False)


def test_overrides_replace_only_given_values():
    cfg = AD.with_overrides(attack=True, aggregator="median", clip_norm=None)

    assert (cfg.attack, cfg.aggregator, cfg.clip_norm) == (True, "median", 3.5)


@pytest.mark.parametrize("overrides", [{"aggregator": "mean"}, {"clip_norm": 0.0}, {"clip_norm": -1.0}])
def test_invalid_attack_defense_is_rejected(overrides):
    with pytest.raises(ValueError):
        AD.with_overrides(**overrides)


# 집계 함수 (이름은 라운드 기록의 aggregator로 남는다)

@pytest.mark.parametrize("aggregator, clip_before_median, name, has_scale", [
    ("fedavg", False, "fedavg", False),
    ("clip_fedavg", False, "clip_fedavg", True),
    ("median", False, "median", False),
    ("median", True, "clip_median", True),
])
def test_defense_names_and_update_scale(aggregator, clip_before_median, name, has_scale):
    defense = build_defense(AD.with_overrides(aggregator=aggregator, clip_before_median=clip_before_median))

    assert defense.aggregator.__name__ == name
    assert (defense.update_scale is not None) == has_scale


def test_fedavg_is_weighted_average():
    result = build_defense(AD).aggregator(UPDATES, WEIGHTS)

    assert torch.equal(result["w"], weighted_average(UPDATES, WEIGHTS)["w"])


def test_clip_fedavg_clips_then_averages():
    # 상한 1: [3,4] → [0.6,0.8]. 가중 평균 = ([0.6,0.8] + [1,0] + 2·[0,1]) / 4 = [0.4, 0.7]
    defense = build_defense(AD.with_overrides(aggregator="clip_fedavg", clip_norm=1.0))

    result = defense.aggregator(UPDATES, WEIGHTS)

    assert torch.allclose(result["w"], torch.tensor([0.4, 0.7]))
    assert torch.equal(result["w"], clipped_fedavg(UPDATES, WEIGHTS, 1.0)["w"])
    assert [defense.update_scale(u) for u in UPDATES] == pytest.approx([0.2, 1.0, 1.0])


def test_median_ignores_weights():
    # 좌표별 중앙값: [3,1,0] → 1, [4,0,1] → 1
    result = build_defense(AD.with_overrides(aggregator="median")).aggregator(UPDATES, WEIGHTS)

    assert torch.equal(result["w"], torch.tensor([1.0, 1.0]))
    assert torch.equal(result["w"], coordinate_median(UPDATES)["w"])


def test_clip_before_median():
    # 상한 1: [0.6,0.8], [1,0], [0,1] → 중앙값 [0.6, 0.8]
    defense = build_defense(AD.with_overrides(aggregator="median", clip_before_median=True, clip_norm=1.0))

    assert torch.allclose(defense.aggregator(UPDATES, WEIGHTS)["w"], torch.tensor([0.6, 0.8]))


# 서버 운영 규칙

def test_partial_policy_gives_five_and_same_schedule_for_any_aggregator():
    env = load_env_config(ENV_DIR / "partial.yaml")
    fedavg = scenario_policy(env, build_defense(AD), seed=0, num_clients=10)
    median = scenario_policy(env, build_defense(AD.with_overrides(aggregator="median")), seed=0, num_clients=10)

    schedule = [tuple(fedavg.participants(r)) for r in range(1, 11)]

    assert all(len(p) == 5 for p in schedule)
    assert schedule == [tuple(median.participants(r)) for r in range(1, 11)]
    assert schedule == [tuple(fedavg.participants(r)) for r in range(1, 11)]  # 여러 번 불러도 같음


def test_policy_takes_deadline_and_min_accepted_from_env():
    env = replace(load_env_config(ENV_DIR / "delay.yaml"), min_accepted=4)

    policy = scenario_policy(env, build_defense(AD), seed=0, num_clients=10)

    assert policy.deadline(1) == 1.5 and policy.deadline(7) == 1.5
    assert policy.min_accepted == 4
    assert list(policy.participants(3)) == list(range(10))


def _submit(client, cid: int, update: dict, finish: float = 1.0):
    body = {"client_id": cid, "round": 1, "model_version": 0, "num_samples": 10,
            "delta_weights": encode_state(update), "virtual_finish_time": finish}
    return client.post("/update", json=body)


def test_server_records_defended_and_aggregate_norms():
    env = load_env_config(ENV_DIR / "baseline.yaml")
    policy = scenario_policy(env, build_defense(AD.with_overrides(aggregator="clip_fedavg", clip_norm=1.0)), 0, 3)
    client = create_app(ServerState({"w": torch.zeros(2)}, num_clients=3, policy=policy)).test_client()
    for cid, update in enumerate(UPDATES):
        assert _submit(client, cid, update).status_code == 200

    record = client.post("/aggregate", json={}).get_json()["data"]

    assert record["aggregator"] == "clip_fedavg"
    assert record["update_norms"] == [5.0, 1.0, 1.0]
    assert record["defended_norms"] == [1.0, 1.0, 1.0]
    # 표본 수가 같으므로 ([0.6,0.8] + [1,0] + [0,1]) / 3 → 노름 = sqrt(1.6² + 1.8²) / 3
    assert record["aggregate_norm"] == pytest.approx(round((1.6 ** 2 + 1.8 ** 2) ** 0.5 / 3, 4))


def test_late_submission_is_excluded_by_env_deadline():
    env = load_env_config(ENV_DIR / "delay.yaml")
    policy = scenario_policy(env, build_defense(AD), seed=0, num_clients=3)
    client = create_app(ServerState({"w": torch.zeros(2)}, num_clients=3, policy=policy)).test_client()

    on_time = _submit(client, 0, UPDATES[0], finish=1.0)
    late = _submit(client, 1, UPDATES[1], finish=3.0)
    record = client.post("/aggregate", json={}).get_json()["data"]

    assert on_time.status_code == 200
    assert late.get_json()["data"]["reason"] == "late"
    assert (record["updated"], record["skip_reason"], record["aggregate_norm"]) == (False, "too_few_accepted", None)
    assert record["defended_norms"] == [5.0]  # FedAvg는 업데이트별 배율이 없어 그대로


def test_defended_norms_are_rounded_without_clipping():
    env = load_env_config(ENV_DIR / "baseline.yaml")
    policy = scenario_policy(env, build_defense(AD.with_overrides(aggregator="median")), seed=0, num_clients=3)
    client = create_app(ServerState({"w": torch.zeros(2)}, num_clients=3, policy=policy)).test_client()
    for cid in range(3):
        _submit(client, cid, {"w": torch.tensor([1.0, 1.0])})  # 노름 √2 = 1.41421...

    record = client.post("/aggregate", json={}).get_json()["data"]

    assert record["defended_norms"] == record["update_norms"] == [1.4142] * 3


@pytest.mark.parametrize("clip_norm", [float("inf"), float("nan")])
def test_clip_norm_must_be_finite(clip_norm):
    with pytest.raises(ValueError):
        AD.with_overrides(aggregator="clip_fedavg", clip_norm=clip_norm)
