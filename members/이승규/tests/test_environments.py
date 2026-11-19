import random
from dataclasses import replace
from pathlib import Path

import pytest

from fedsec.environments.acceptance import LATE, NOT_APPLICABLE, TOO_FEW_ACCEPTED, judge_round
from fedsec.environments.config import load_env_config
from fedsec.environments.schedule import delayed_clients, malicious_clients, round_participants
from fedsec.environments.virtual_time import (VirtualTiming, check_delay_setting, late_clients, round_elapsed,
                                              virtual_finish_times)

ENV_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"
ENV_NAMES = ("baseline", "delay", "partial", "label_skew")
NUM_CLIENTS = 10
SAMPLES = {cid: 4_500 for cid in range(NUM_CLIENTS)}
TIMING = VirtualTiming(base_time=1.0, base_samples=4_500, delay=2.0, deadline=1.5)


def _schedule(seed: int, rounds: int = 5) -> tuple:
    malicious = malicious_clients(seed, NUM_CLIENTS, 2)
    return (
        malicious,
        delayed_clients(seed, NUM_CLIENTS, 3, exclude=malicious),
        [round_participants(seed, r, NUM_CLIENTS, 5) for r in range(1, rounds + 1)],
    )


# 일정 재현성

def test_same_seed_gives_same_schedule():
    assert _schedule(seed=0) == _schedule(seed=0)


def test_different_seed_changes_schedule():
    assert _schedule(seed=0) != _schedule(seed=1)


def test_schedule_ignores_global_random_state():
    # 집계·공격 코드가 전역 난수를 써도 일정은 시드만으로 정해진다
    random.seed(123)
    first = _schedule(seed=0)
    random.seed(999)
    random.random()

    second = _schedule(seed=0)

    assert first == second


def test_fixed_ids_override_seed_choice():
    assert malicious_clients(seed=0, num_clients=10, count=2, fixed_ids=[7, 3]) == (3, 7)


@pytest.mark.parametrize("ids", [[1, 1], [0, 10], [-1, 2]])
def test_invalid_fixed_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        malicious_clients(seed=0, num_clients=10, count=2, fixed_ids=ids)


def test_too_many_malicious_are_rejected():
    with pytest.raises(ValueError):
        malicious_clients(seed=0, num_clients=10, count=11)


@pytest.mark.parametrize("seed", range(50))
def test_delayed_never_overlap_malicious(seed):
    malicious = malicious_clients(seed, NUM_CLIENTS, 2)

    delayed = delayed_clients(seed, NUM_CLIENTS, 3, exclude=malicious)

    assert len(delayed) == 3
    assert not set(delayed) & set(malicious)


def test_overlap_is_possible_when_exclusion_is_off():
    # 옵션을 끄면 전체에서 뽑으므로 어떤 시드에서는 겹친다
    overlaps = [
        set(malicious_clients(s, NUM_CLIENTS, 2)) & set(delayed_clients(s, NUM_CLIENTS, 3))
        for s in range(50)
    ]

    assert any(overlaps)


def test_too_few_candidates_for_delayed_are_rejected():
    with pytest.raises(ValueError):
        delayed_clients(seed=0, num_clients=4, count=3, exclude=(0, 1))


# 부분 참여

def test_partial_participation_picks_exactly_five_each_round():
    rounds = [round_participants(seed=0, round_idx=r, num_clients=10, per_round=5) for r in range(1, 41)]

    assert all(len(p) == 5 and len(set(p)) == 5 for p in rounds)
    assert all(0 <= cid < 10 for p in rounds for cid in p)
    assert len(set(rounds)) > 1  # 라운드마다 다시 뽑는다


def test_full_participation_when_per_round_is_none():
    assert round_participants(seed=0, round_idx=1, num_clients=10) == tuple(range(10))


# 가상 시간

def test_delayed_clients_always_miss_deadline():
    delayed = delayed_clients(seed=0, num_clients=NUM_CLIENTS, count=3)

    finish = virtual_finish_times(SAMPLES, delayed, TIMING)

    assert late_clients(finish, TIMING.deadline) == delayed
    assert finish[delayed[0]] == pytest.approx(3.0)
    check_delay_setting(SAMPLES, delayed, TIMING)  # 지연 3명만 초과하면 통과


def test_delay_too_small_is_rejected():
    short = replace(TIMING, delay=0.2)  # 1.0 + 0.2 = 1.2 <= 마감 1.5

    with pytest.raises(ValueError):
        check_delay_setting(SAMPLES, (0, 1, 2), short)


def test_finish_time_is_proportional_to_samples():
    finish = virtual_finish_times({0: 2_250, 1: 9_000}, delayed=(), timing=TIMING)

    assert finish == {0: pytest.approx(0.5), 1: pytest.approx(2.0)}


def test_round_elapsed_is_deadline_when_someone_is_late():
    assert round_elapsed({0: 1.0, 1: 3.0}, deadline=1.5) == 1.5


def test_round_elapsed_is_latest_finish_without_late():
    assert round_elapsed({0: 0.5, 1: 1.0}, deadline=1.5) == 1.0
    assert round_elapsed({}, deadline=1.5) == 0.0


# 채택 규칙

def test_accepted_malicious_ratio_counts_only_accepted():
    # 선택 0~4, 1번 지연 제외, 악성 0·7 → 채택 [0,2,3,4] 중 악성 1개
    decision = judge_round([0, 1, 2, 3, 4], {1: LATE}, malicious={0, 7}, num_clients=10)

    assert decision.accepted == (0, 2, 3, 4)
    assert decision.num_late == 1
    assert decision.malicious_selected == 1
    assert decision.overall_malicious_ratio == pytest.approx(0.2)
    assert decision.accepted_malicious_ratio == pytest.approx(0.25)
    assert decision.update_model and decision.skip_reason is None


def test_zero_accepted_gives_na_and_no_update():
    decision = judge_round([0, 1], {0: LATE, 1: LATE}, malicious={0}, num_clients=10)

    log = decision.to_log()

    assert decision.accepted_malicious_ratio is None
    assert log["accepted_malicious_ratio"] == NOT_APPLICABLE
    assert log["model_updated"] is False
    assert log["skip_reason"] == TOO_FEW_ACCEPTED


def test_fewer_than_three_accepted_skips_update():
    two = judge_round([0, 1, 2], {2: LATE}, malicious=set(), num_clients=10)
    three = judge_round([0, 1, 2], {}, malicious=set(), num_clients=10)

    assert (two.update_model, two.skip_reason) == (False, TOO_FEW_ACCEPTED)
    assert two.accepted_malicious_ratio == 0.0  # 채택이 있으면 N/A가 아니라 0
    assert (three.update_model, three.skip_reason) == (True, None)


def test_excluded_must_be_selected():
    with pytest.raises(ValueError):
        judge_round([0, 1], {5: LATE}, malicious=set(), num_clients=10)


def test_delay_environment_round_excludes_three_late():
    delayed = delayed_clients(seed=0, num_clients=NUM_CLIENTS, count=3)
    finish = virtual_finish_times(SAMPLES, delayed, TIMING)
    excluded = {cid: LATE for cid in late_clients(finish, TIMING.deadline)}

    decision = judge_round(tuple(range(NUM_CLIENTS)), excluded, malicious={0, 1}, num_clients=NUM_CLIENTS)

    assert decision.to_log()["num_late"] == 3
    assert len(decision.accepted) == 7
    assert round_elapsed(finish, TIMING.deadline) == TIMING.deadline


# 환경 설정 파일

def test_all_environment_files_load():
    configs = [load_env_config(ENV_DIR / f"{name}.yaml") for name in ENV_NAMES]

    assert [c.name for c in configs] == list(ENV_NAMES)
    assert all(c.num_malicious == 2 and c.min_accepted == 3 for c in configs)
    assert all(c.delayed_avoid_malicious for c in configs)


def test_avoid_malicious_must_be_boolean(tmp_path):
    text = (ENV_DIR / "delay.yaml").read_text(encoding="utf-8").replace("avoid_malicious: true", "avoid_malicious: 1")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_env_config(path)


def test_each_environment_changes_one_factor_from_baseline():
    base = load_env_config(ENV_DIR / "baseline.yaml")
    delay = load_env_config(ENV_DIR / "delay.yaml")
    partial = load_env_config(ENV_DIR / "partial.yaml")
    skew = load_env_config(ENV_DIR / "label_skew.yaml")

    assert (base.per_round, base.num_delayed) == (None, 0)
    assert replace(delay, name=base.name, num_delayed=0) == base
    assert replace(partial, name=base.name, per_round=None) == base
    assert replace(skew, name=base.name, split=base.split) == base


def test_delay_file_values_make_delayed_late():
    env = load_env_config(ENV_DIR / "delay.yaml")
    delayed = delayed_clients(seed=0, num_clients=NUM_CLIENTS, count=env.num_delayed)

    check_delay_setting(SAMPLES, delayed, env.timing)


def test_ids_count_mismatch_is_rejected(tmp_path):
    text = (ENV_DIR / "baseline.yaml").read_text(encoding="utf-8").replace("ids: null", "ids: [1, 2, 3]", 1)
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_env_config(path)
