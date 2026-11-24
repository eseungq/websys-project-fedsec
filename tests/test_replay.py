"""재실행(scripts/replay.py)에 쓰는 엔진 기능: 시작 라운드, 저장 모델 읽기, 특정 악성의 조작 해제, 재현성."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from fedsec.client.program import (
    check_server,
    client_plan,
    fetch_snapshot,
    run_round,
    updates_to_submit,
)
from fedsec.models.cnn import build_model
from fedsec.server.app import create_app
from fedsec.server.scenario import build_defense, scenario_policy
from fedsec.server.state import ServerState, initial_global_state, load_global_state
from tests.test_client_round import (
    AD,
    CFG,
    CLIENTS,
    CPU,
    NUM_CLIENTS,
    SEED,
    VAL,
    FlaskApi,
    _env,
    _status,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay  # scripts/는 패키지가 아니라 경로로 불러온다

SOURCE_RUN = ROOT / "experiments" / "matrix" / "baseline-attack-fedavg-s0"


def _server(state: dict, env, attack_defense, start_round: int = 1) -> FlaskApi:
    policy = scenario_policy(env, build_defense(attack_defense), SEED, NUM_CLIENTS)
    server = ServerState(state, num_clients=NUM_CLIENTS, policy=policy, start_round=start_round)
    return FlaskApi(create_app(server).test_client())


def _rounds(api: FlaskApi, plan, env, start_round: int, count: int):
    """start_round부터 count라운드를 돌고 마지막 전역 모델과 rounds 행들을 돌려준다."""
    snapshot = fetch_snapshot(api, CPU, expected_round=start_round)
    rows = []
    for _ in range(count):
        row, _, snapshot, _ = run_round(api, build_model(CFG.model), snapshot, CLIENTS, VAL, CFG, plan,
                                        env.min_accepted, CPU)
        rows.append(row)
    return snapshot.state, rows


# --- 서버 시작 라운드와 저장 모델 -------------------------------------------------------------

def test_server_starts_at_given_round_with_version_zero():
    api = _server(initial_global_state(SEED, CFG.model), _env("baseline"), AD, start_round=6)

    status = api.status()

    assert (status["round"], status["model_version"]) == (6, 0)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_server_rejects_invalid_start_round(bad):
    with pytest.raises(ValueError, match="start_round"):
        ServerState(initial_global_state(SEED, CFG.model), num_clients=NUM_CLIENTS, start_round=bad)


def test_load_global_state_round_trips(tmp_path):
    reference = initial_global_state(SEED, CFG.model)
    path = tmp_path / "round_005.pt"
    torch.save(reference, path)

    loaded = load_global_state(path, reference)

    assert all(torch.equal(loaded[key], reference[key]) for key in reference)


def test_load_global_state_rejects_other_model(tmp_path):
    reference = initial_global_state(SEED, CFG.model)
    path = tmp_path / "other.pt"
    torch.save({"weight": torch.zeros(3)}, path)

    with pytest.raises(ValueError, match="파라미터 이름"):
        load_global_state(path, reference)


def test_load_global_state_rejects_wrong_shape(tmp_path):
    reference = initial_global_state(SEED, CFG.model)
    key = next(iter(reference))
    path = tmp_path / "shape.pt"
    torch.save({**reference, key: torch.zeros(1)}, path)

    with pytest.raises(ValueError, match="shape"):
        load_global_state(path, reference)


def test_load_global_state_rejects_non_finite(tmp_path):
    reference = initial_global_state(SEED, CFG.model)
    key = next(k for k, v in reference.items() if v.is_floating_point())
    broken = reference[key].clone()
    broken.view(-1)[0] = float("nan")
    path = tmp_path / "nan.pt"
    torch.save({**reference, key: broken}, path)

    with pytest.raises(ValueError, match="NaN"):
        load_global_state(path, reference)


def test_check_server_expects_start_round():
    env = _env("baseline")
    plan = client_plan("t", env, AD, SEED, NUM_CLIENTS)
    status = {**_status(env, plan), "round": 6}

    check_server(status, CFG, NUM_CLIENTS, plan, start_round=6)  # 같으면 통과
    with pytest.raises(RuntimeError, match="처음 상태"):
        check_server(status, CFG, NUM_CLIENTS, plan)


# --- 조작 해제 ---------------------------------------------------------------------------

def _normal_updates():
    g = torch.Generator().manual_seed(1)
    return {cid: {"w": torch.randn(4, generator=g)} for cid in range(NUM_CLIENTS)}


def test_unmasked_attacker_submits_own_normal_update():
    plan = client_plan("t", _env("baseline"), AD.with_overrides(attack=True), SEED, NUM_CLIENTS)
    kept, freed = plan.malicious
    normal = _normal_updates()

    submitted = updates_to_submit(normal, replace(plan, unmasked=(freed,)))

    assert submitted[freed] is normal[freed]
    # 남은 악성 1명은 자기 업데이트 × (−5) (해제한 악성은 공모 평균에 끼지 않음)
    assert torch.equal(submitted[kept]["w"], normal[kept]["w"] * plan.attack_scale)


def test_unmasking_all_attackers_submits_all_normal():
    plan = client_plan("t", _env("baseline"), AD.with_overrides(attack=True), SEED, NUM_CLIENTS)
    normal = _normal_updates()

    submitted = updates_to_submit(normal, replace(plan, unmasked=plan.malicious))

    assert all(submitted[cid] is normal[cid] for cid in normal)


def test_plan_dict_has_unmasked_only_when_set():
    plan = client_plan("t", _env("baseline"), AD.with_overrides(attack=True), SEED, NUM_CLIENTS)

    assert "unmasked" not in plan.to_dict()  # 기존 결과의 config.yaml과 비교가 어긋나지 않게
    assert replace(plan, unmasked=(plan.malicious[0],)).to_dict()["unmasked"] == [plan.malicious[0]]


# --- 재현성: 이어서 돈 결과 = 저장 모델에서 다시 돈 결과 -----------------------------------------

@pytest.mark.parametrize("env_name", ["baseline", "partial"])
def test_replay_from_saved_round_matches_continuous_run(env_name):
    env = _env(env_name)
    attack_defense = AD.with_overrides(attack=True, aggregator="median")
    plan = client_plan("t", replace(env, min_accepted=0), attack_defense, SEED, NUM_CLIENTS)
    start = initial_global_state(SEED, CFG.model)

    after_one, _ = _rounds(_server(start, env, attack_defense), plan, env, start_round=1, count=1)
    continuous, rows_continuous = _rounds(_server(start, env, attack_defense), plan, env, start_round=1, count=3)
    saved = {key: value.cpu().clone() for key, value in after_one.items()}
    replayed, rows_replayed = _rounds(_server(saved, env, attack_defense, start_round=2), plan, env,
                                      start_round=2, count=2)

    assert all(torch.equal(continuous[key], replayed[key]) for key in continuous)
    keep = ("round", "val_acc", "val_loss", "num_accepted", "malicious_accepted", "aggregate_norm")
    assert [{k: r[k] for k in keep} for r in rows_continuous[1:]] == [{k: r[k] for k in keep} for r in rows_replayed]


# --- scripts/replay.py ------------------------------------------------------------------

def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"unmask": [], "aggregator": None, "no_attack": False, "same": False, **kw})


def test_describe_and_label():
    source = AD.with_overrides(attack=True, aggregator="fedavg")

    change = replay.describe(_args(unmask=[3, 2, 3], aggregator="median"), source)

    assert change == {"unmask": [2, 3], "aggregator": {"from": "fedavg", "to": "median"}}
    assert replay.change_label(change) == "unmask2-3_median"
    assert replay.change_label({}) == "same"


def test_describe_ignores_aggregator_equal_to_source():
    source = AD.with_overrides(aggregator="median")

    assert replay.describe(_args(aggregator="median"), source) == {}


@pytest.mark.parametrize("kw, change", [
    ({}, {}),                                    # 아무것도 안 바꾸고 --same도 없음
    ({"same": True}, {"unmask": [2]}),           # --same인데 바꿈
    ({}, {"unmask": [7]}),                       # 악성이 아닌 ID
])
def test_check_change_rejects(kw, change):
    with pytest.raises(SystemExit):
        replay.check_change(change, _args(**kw), AD.with_overrides(attack=True), malicious=[2, 3])


def test_check_change_rejects_unmask_without_attack():
    with pytest.raises(SystemExit):
        replay.check_change({"unmask": [2]}, _args(), AD.with_overrides(attack=False), malicious=[2, 3])


def test_rebuild_matches_saved_run():
    saved = yaml.safe_load((SOURCE_RUN / "config.yaml").read_text(encoding="utf-8"))

    cfg, plan = replay.rebuild(saved, rounds=7, attack_defense=replay.source_attack_defense(saved))

    assert cfg.rounds == 7 and cfg.seed == saved["seed"] and cfg.split == saved["split"]
    assert list(plan.malicious) == saved["plan"]["malicious"]


def test_rebuild_stops_when_saved_setting_differs():
    saved = yaml.safe_load((SOURCE_RUN / "config.yaml").read_text(encoding="utf-8"))
    changed = {**saved, "eval_bs": saved["eval_bs"] + 1}

    with pytest.raises(SystemExit):
        replay.rebuild(changed, rounds=7, attack_defense=replay.source_attack_defense(saved))


def test_rebuild_stops_when_timing_differs():
    saved = yaml.safe_load((SOURCE_RUN / "config.yaml").read_text(encoding="utf-8"))
    plan = {**saved["plan"], "timing": {**saved["plan"]["timing"], "deadline": 9.9}}

    with pytest.raises(SystemExit):
        replay.rebuild({**saved, "plan": plan}, rounds=7, attack_defense=replay.source_attack_defense(saved))


def test_source_last_round_counts_from_start_round_of_a_replay():
    saved = yaml.safe_load((SOURCE_RUN / "config.yaml").read_text(encoding="utf-8"))
    replayed = {**saved, "rounds": 35, "server": {**saved["server"],
                                                 "config": {**saved["server"]["config"], "start_round": 6}}}

    assert replay.source_last_round(saved) == saved["rounds"]
    assert replay.source_last_round(replayed) == 40


def test_merged_unmask_keeps_attackers_unmasked_in_source_replay():
    saved = {"plan": {"unmasked": [2]}}

    assert replay.merged_unmask(saved, {"unmask": [3]}) == (2, 3)
    assert replay.merged_unmask(saved, {}) == (2,)
    assert replay.merged_unmask({"plan": {}}, {"unmask": [3]}) == (3,)
