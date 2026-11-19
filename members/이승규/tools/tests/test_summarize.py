import json

import pytest
import yaml

from summarize import (
    best_val, last_mean, load_run, min_val_loss_round, norm_stats, plateau_round, round_times,
    summarize_run, to_csv, to_markdown, val_at_epochs,
)


def make_rounds(val_accs, val_losses=None, times=None):
    n = len(val_accs)
    val_losses = val_losses or [1.0] * n
    times = times or [1.0] * n
    return [{"round": i + 1, "val_acc": a, "val_loss": l, "total_sec": t}
            for i, (a, l, t) in enumerate(zip(val_accs, val_losses, times))]


CONFIG = {"seed": 0, "rounds": 12,
          "train": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "local_ep": 1, "local_bs": 32}}


def test_last_mean_uses_only_last_n_rounds():
    rounds = make_rounds([0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])

    result = last_mean(rounds, "val_acc", 5)

    # (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5
    assert result == pytest.approx(0.7)


def test_last_mean_uses_all_rounds_when_fewer_than_n():
    rounds = make_rounds([0.2, 0.4])

    result = last_mean(rounds, "val_acc", 5)

    assert result == pytest.approx(0.3)


def test_best_val_returns_first_round_on_tie():
    rounds = make_rounds([0.3, 0.7, 0.5, 0.7])

    result = best_val(rounds)

    assert result == (0.7, 2)


def test_plateau_round_is_first_round_within_margin_of_last10_mean():
    # 마지막 10라운드(3~12) 평균 = (0.70 + 0.74 + 0.75 × 8) / 10 = 0.744 → 기준 0.739
    rounds = make_rounds([0.30, 0.60, 0.70, 0.74] + [0.75] * 8)

    result = plateau_round(rounds)

    assert result == 4


def test_plateau_round_counts_early_round_that_already_reaches_threshold():
    # 1라운드가 이미 기준(평균 0.5 − 0.005) 이상이면 1
    rounds = make_rounds([0.5] * 12)

    result = plateau_round(rounds)

    assert result == 1


def test_min_val_loss_round_marks_overfitting_start():
    rounds = make_rounds([0.5] * 5, val_losses=[1.2, 0.9, 0.8, 0.85, 1.0])

    result = min_val_loss_round(rounds)

    assert result == 3


def test_norm_stats_median_p90_max():
    norms = [float(x) for x in range(1, 11)]   # 1 ~ 10

    median, p90, maximum = norm_stats(norms)

    # 선형 보간(numpy 기본값과 같음): 1 + 0.9 × (10 − 1) = 9.1
    assert (median, p90, maximum) == (pytest.approx(5.5), pytest.approx(9.1), 10.0)


def test_norm_stats_rejects_too_few_norms():
    with pytest.raises(ValueError):
        norm_stats([1.0])


def test_round_times_excludes_first_round_from_mean():
    rounds = make_rounds([0.5] * 3, times=[8.0, 3.0, 4.0])

    mean_after_first, total = round_times(rounds)

    assert (mean_after_first, total) == (pytest.approx(3.5), pytest.approx(15.0))


def test_round_times_with_single_round_uses_that_round():
    rounds = make_rounds([0.5], times=[8.0])

    mean_after_first, total = round_times(rounds)

    assert (mean_after_first, total) == (8.0, 8.0)


@pytest.mark.parametrize("local_ep, epochs, expected", [
    (1, 4, 0.4),     # 4라운드
    (2, 4, 0.2),     # 2라운드 × 2 epoch
    (3, 4, None),    # 나누어떨어지지 않음
    (1, 9, None),    # 라운드 부족
])
def test_val_at_epochs_matches_same_compute(local_ep, epochs, expected):
    rounds = make_rounds([0.1, 0.2, 0.3, 0.4, 0.5])

    result = val_at_epochs(rounds, local_ep, epochs)

    assert result == expected


def test_summarize_run_collects_all_columns():
    rounds = make_rounds([0.30, 0.60, 0.70, 0.74] + [0.75] * 8,
                         val_losses=[2.0, 1.0, 0.8, 0.7] + [0.75] * 8, times=[5.0] + [2.0] * 11)

    row = summarize_run(rounds, [1.0, 2.0, 3.0], CONFIG, {"final_test_acc": 0.74}, epochs=10)

    assert row["lr"] == 0.01 and row["local_bs"] == 32 and row["rounds"] == 12
    assert row["last5_val"] == pytest.approx(0.75)
    assert (row["best_val"], row["best_round"]) == (0.75, 5)
    assert row["plateau_round"] == 4
    assert row["min_val_loss_round"] == 4
    assert row["final_test"] == 0.74
    assert row["norm_max"] == 3.0
    assert (row["round_sec"], row["total_sec"]) == (pytest.approx(2.0), pytest.approx(27.0))
    assert row["val_ep10"] == 0.75


def test_summarize_run_rejects_empty_rounds():
    with pytest.raises(ValueError):
        summarize_run([], [1.0, 2.0], CONFIG, {"final_test_acc": 0.5}, epochs=50)


def write_run(run_dir, with_summary=True):
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    (run_dir / "rounds.csv").write_text(
        "round,train_loss,val_loss,val_acc,norm_min,norm_mean,norm_max,train_sec,total_sec\n"
        "1,2.0,1.5,0.40,1.0,1.5,2.0,3.0,4.0\n"
        "2,1.5,1.2,0.50,1.0,1.5,2.0,3.0,3.0\n", encoding="utf-8")
    (run_dir / "clients.csv").write_text(
        "round,client,num_samples,train_loss,update_norm\n"
        "1,0,4500,2.0,1.0\n1,1,4500,2.0,2.0\n2,0,4500,1.5,3.0\n2,1,4500,1.5,4.0\n", encoding="utf-8")
    if with_summary:
        (run_dir / "summary.json").write_text(json.dumps({"final_test_acc": 0.49}), encoding="utf-8")


def test_load_run_reads_result_folder(tmp_path):
    write_run(tmp_path / "lr0.01")

    row = load_run(tmp_path / "lr0.01", epochs=2)

    assert row["name"] == "lr0.01"
    assert row["last5_val"] == pytest.approx(0.45)
    assert row["min_val_loss_round"] == 2
    assert row["norm_max"] == 4.0
    assert row["val_ep2"] == 0.5


def test_load_run_rejects_unfinished_run(tmp_path):
    write_run(tmp_path / "cut", with_summary=False)

    with pytest.raises(ValueError, match="summary.json"):
        load_run(tmp_path / "cut", epochs=50)


def test_markdown_shows_accuracy_as_percent_and_missing_as_dash():
    rows = [{"name": "a", "last5_val": 0.75584, "norm_p90": 3.92421, "val_ep50": None}]

    result = to_markdown(rows)

    assert result.splitlines() == [
        "| name | last5_val | norm_p90 | val_ep50 |",
        "|---|---|---|---|",
        "| a | 75.58% | 3.92 | — |",
    ]


def test_csv_keeps_raw_fractions_and_blank_for_missing():
    rows = [{"name": "a", "last5_val": 0.755841234, "best_round": 45, "val_ep50": None}]

    result = to_csv(rows)

    assert result == "name,last5_val,best_round,val_ep50\na,0.755841,45,\n"
