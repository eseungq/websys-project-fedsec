"""비교표 계산: runs.csv → 공격 피해·방어 개선·정상 손실 (%p)."""

import pytest

from fedsec.experiment.comparison import (
    Difference,
    RunResult,
    accuracy_table,
    compare,
    difference,
    format_cell,
    parse_runs,
    to_csv,
    to_markdown,
)

CONDITIONS = [(attack, agg) for attack in ("normal", "attack") for agg in ("fedavg", "clip_fedavg", "median")]


ROUNDS = 40


def _row(env="baseline", attack="normal", aggregator="fedavg", seed=0, acc="0.76", status="done", diverged="",
         completed=None):
    if completed is None:
        completed = int(diverged) - 1 if diverged else ROUNDS
    return {"name": f"{env}-{attack}-{aggregator}-s{seed}", "env": env, "attack": attack, "aggregator": aggregator,
            "seed": str(seed), "status": status, "final_test_acc": acc, "diverged_round": diverged,
            "rounds_completed": str(completed)}


def _parse(rows):
    return parse_runs(rows, rounds=ROUNDS)


def _rows(accs: dict, env="baseline", seeds=(0,), diverged: dict | None = None) -> list[dict]:
    """accs: (attack, aggregator) → 정확도. diverged: (attack, aggregator) → 발산 라운드."""
    diverged = diverged or {}
    return [_row(env, attack, agg, seed, str(accs[(attack, agg)]), diverged=str(diverged.get((attack, agg), "")))
            for seed in seeds for attack, agg in CONDITIONS]


ACCS = {("normal", "fedavg"): 0.76, ("normal", "clip_fedavg"): 0.75, ("normal", "median"): 0.77,
        ("attack", "fedavg"): 0.30, ("attack", "clip_fedavg"): 0.60, ("attack", "median"): 0.65}


def _metric(rows, metric, defense=""):
    return [r for r in rows if r["metric"] == metric and r["defense"] == defense]


def test_compare_follows_spec_definitions_in_percentage_points():
    # Arrange
    results = _parse(_rows(ACCS))

    # Act
    rows = compare(results, envs=["baseline"], seeds=[0])

    # Assert
    assert _metric(rows, "attack_damage")[0]["value"] == pytest.approx(46.0)                 # 76 − 30
    assert _metric(rows, "defense_gain", "clip_fedavg")[0]["value"] == pytest.approx(30.0)   # 60 − 30
    assert _metric(rows, "defense_gain", "median")[0]["value"] == pytest.approx(35.0)
    assert _metric(rows, "normal_loss", "clip_fedavg")[0]["value"] == pytest.approx(1.0)     # 76 − 75
    assert _metric(rows, "normal_loss", "median")[0]["value"] == pytest.approx(-1.0)         # 음수도 그대로


def test_compare_keeps_seed_values_separate():
    rows = _rows(ACCS, seeds=(0,)) + _rows({**ACCS, ("normal", "fedavg"): 0.80}, seeds=(1,))

    damage = _metric(compare(_parse(rows), envs=["baseline"], seeds=[0, 1]), "attack_damage")

    assert [r["seed"] for r in damage] == [0, 1]
    assert [r["value"] for r in damage] == pytest.approx([46.0, 50.0])


def test_diverged_run_has_no_main_value_but_keeps_reference():
    # 발산 전 모델(예: 12라운드)과 40라운드 모델의 차이는 학습 길이 차이도 섞여 주 표에 넣지 않는다
    rows = _rows(ACCS, diverged={("attack", "fedavg"): 13})

    result = compare(_parse(rows), envs=["baseline"], seeds=[0])

    damage = _metric(result, "attack_damage")[0]
    assert damage["value"] is None
    assert damage["reference"] == pytest.approx(46.0)
    assert damage["diverged_rounds"] == (13,)
    assert _metric(result, "normal_loss", "median")[0]["value"] == pytest.approx(-1.0)  # 발산과 무관한 값은 그대로


def test_first_round_divergence_has_no_reference():
    rows = [r for r in _rows(ACCS) if (r["attack"], r["aggregator"]) != ("attack", "fedavg")]
    rows.append(_row(attack="attack", aggregator="fedavg", acc="", status="diverged", diverged="1"))

    damage = _metric(compare(_parse(rows), envs=["baseline"], seeds=[0]), "attack_damage")[0]

    assert damage["value"] is None and damage["reference"] is None
    assert damage["diverged_rounds"] == (1,)


def test_difference_of_two_finished_runs():
    assert difference(RunResult(0.5, None), RunResult(0.25, None)) == Difference(25.0, 25.0, ())


@pytest.mark.parametrize("status", ["todo", "incomplete", "stale"])
def test_parse_runs_rejects_unfinished_runs(status):
    with pytest.raises(ValueError, match="끝나지 않은 실행"):
        _parse([_row(status=status)])


@pytest.mark.parametrize("acc", ["", "abc", "1.5", "-0.1", "nan"])
def test_parse_runs_rejects_bad_accuracy_of_done_run(acc):
    with pytest.raises(ValueError, match="정확도"):
        _parse([_row(acc=acc)])


@pytest.mark.parametrize("diverged", ["abc", "0", "-3"])
def test_parse_runs_rejects_bad_diverged_round(diverged):
    with pytest.raises(ValueError, match="발산 라운드"):
        _parse([_row(attack="attack", diverged=diverged, completed=ROUNDS)])


def test_parse_runs_rejects_duplicate_run():
    with pytest.raises(ValueError, match="중복"):
        _parse([_row(), _row()])


def test_compare_names_missing_run():
    rows = [r for r in _rows(ACCS) if (r["attack"], r["aggregator"]) != ("attack", "median")]

    with pytest.raises(ValueError, match="baseline-attack-median-s0"):
        compare(_parse(rows), envs=["baseline"], seeds=[0])


def test_format_cell_shows_seed_values_and_mean():
    assert format_cell([0.18, -0.15, -0.14], [(), (), ()]) == "+0.18 / −0.15 / −0.14 (−0.04)"


def test_format_cell_writes_zero_without_sign():
    # −0.004도 반올림하면 0이므로 "−0.00"이 아니라 "0.00"
    assert format_cell([0.0, -0.004, 0.004], [(), (), ()]) == "0.00 / 0.00 / 0.00 (0.00)"


def test_format_cell_unsigned_for_accuracy():
    assert format_cell([76.16, 76.44], [(), ()], signed=False) == "76.16 / 76.44 (76.30)"


def test_format_cell_shows_divergence_rounds_instead_of_numbers():
    assert format_cell([None, None, None], [(13,), (11,), (17,)]) == "발산 (13 / 11 / 17라운드)"


def test_format_cell_without_mean_when_some_seed_diverged():
    assert format_cell([1.0, None], [(), (7,)]) == "+1.00 / 발산(7라운드) (평균 없음)"


def test_accuracy_table_shows_percent_and_divergence():
    rows = _rows(ACCS, diverged={("attack", "fedavg"): 13})

    table = accuracy_table(_parse(rows), envs=["baseline"], seeds=[0])

    assert "| baseline | 76.00 (76.00) |" in table
    assert "발산 (13라운드)" in table


def test_markdown_has_main_and_reference_tables():
    rows = compare(_parse(_rows(ACCS, diverged={("attack", "fedavg"): 13})), envs=["baseline"], seeds=[0])

    text = to_markdown(rows, envs=["baseline"])

    assert "| baseline | 발산 (13라운드) |" in text
    assert "참고: 발산 전 모델 기준" in text
    assert "+46.00 (+46.00)" in text   # 참고 표의 공격 피해


def test_markdown_omits_reference_table_without_divergence():
    text = to_markdown(compare(_parse(_rows(ACCS)), envs=["baseline"], seeds=[0]), envs=["baseline"])

    assert "참고" not in text


def test_csv_has_one_line_per_metric_and_seed():
    rows = compare(_parse(_rows(ACCS, diverged={("attack", "fedavg"): 13})), envs=["baseline"], seeds=[0])

    lines = to_csv(rows).splitlines()

    assert lines[0] == "env,metric,defense,seed,value_pp,reference_pp,diverged_rounds"
    assert len(lines) == 1 + 5   # 공격 피해 1 + 방어 개선 2 + 정상 손실 2
    assert "baseline,attack_damage,,0,,46.0,13" in lines


def test_parse_runs_rejects_run_with_other_round_count():
    # 2라운드 동작 확인 결과가 40라운드 본 실험 표에 섞이지 않게
    with pytest.raises(ValueError, match="라운드"):
        _parse([_row(completed=2)])


@pytest.mark.parametrize("diverged, completed", [("41", 40), ("13", 5)])
def test_parse_runs_rejects_inconsistent_divergence(diverged, completed):
    # 발산 라운드는 1~목표 라운드, 완료 라운드는 발산 라운드 − 1
    with pytest.raises(ValueError, match="라운드"):
        _parse([_row(attack="attack", acc="0.1", diverged=diverged, completed=completed)])


def test_parse_runs_rejects_divergence_of_normal_run():
    with pytest.raises(ValueError, match="정상 실행"):
        _parse([_row(acc="0.1", diverged="13")])


def test_parse_runs_reports_truncated_row_as_value_error():
    # 마지막 행이 잘리면 csv.DictReader가 빠진 열에 None을 넣는다
    row = {**_row(), "diverged_round": None, "rounds_completed": None}

    with pytest.raises(ValueError):
        _parse([row])


@pytest.mark.parametrize("diverged, completed, acc", [("13", 12, "0.10"), ("1", 0, "0.10"), ("13", 12, "")])
def test_parse_runs_rejects_diverged_status_other_than_first_round_without_accuracy(diverged, completed, acc):
    # diverged 상태는 1라운드 발산(평가할 모델 없음)만. 발산 전 모델로 평가한 실행은 done + diverged_round
    row = _row(attack="attack", status="diverged", acc=acc, diverged=diverged, completed=completed)

    with pytest.raises(ValueError, match="diverged 상태"):
        _parse([row])


def test_parse_runs_rejects_first_round_divergence_recorded_as_done():
    # 1라운드 발산은 평가할 모델이 없으므로 정확도가 있는 done 행은 잘못된 기록
    with pytest.raises(ValueError, match="1라운드 발산"):
        _parse([_row(attack="attack", acc="0.10", diverged="1")])


def test_parse_runs_rejects_diverged_row_with_validation_accuracy():
    row = {**_row(attack="attack", status="diverged", acc="", diverged="1"), "final_val_acc": "0.10"}

    with pytest.raises(ValueError, match="diverged 상태"):
        _parse([row])
