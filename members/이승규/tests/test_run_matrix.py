"""scripts/run_matrix.py: 4단계 일괄 실행 목록과 이어서 실행하기 위한 상태 판정."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

MEMBER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBER_ROOT / "scripts"))
import yaml
from run_clients import EXPERIMENTS_DIR
from run_matrix import NUM_CLASSES, NUM_CLIENTS, Run, expected_config, index_rows, plan_runs, run_status

from fedsec.experiment.config import AGGREGATORS

ENVS = ["baseline", "delay", "partial", "label_skew"]


def test_full_matrix_has_72_runs_for_three_seeds():
    runs = plan_runs(ENVS, ["normal", "attack"], list(AGGREGATORS), [0, 1, 2])

    assert len(runs) == 72
    assert len({r.name for r in runs}) == 72
    assert runs[0].name == "baseline-normal-fedavg-s0"
    assert [r.seed for r in runs[:24]] == [0] * 24  # 시드 하나씩 끝낸다


def test_check_matrix_has_12_runs():
    assert len(plan_runs(ENVS, ["attack"], list(AGGREGATORS), [0])) == 12


def test_unknown_choice_is_rejected():
    with pytest.raises(ValueError):
        plan_runs(["windy"], ["attack"], ["fedavg"], [0])


def test_command_always_evaluates_test_and_sets_attack_flag():
    attack = Run("delay", "attack", "median", 1).command("m", rounds=2)
    normal = Run("delay", "normal", "median", 1).command("m", rounds=None)

    assert "--evaluate-test" in attack and "--evaluate-test" in normal
    assert "--attack" in attack and "--no-attack" in normal
    assert attack[attack.index("--rounds") + 1] == "2" and "--rounds" not in normal
    assert attack[attack.index("--env") + 1] == "configs/env/delay.yaml"
    assert attack[attack.index("--name") + 1] == "m/delay-attack-median-s1"


SUMMARY = json.dumps({"final_val_acc": 0.4, "final_test_acc": 0.5})


def saved_config(expected: dict, **server) -> dict:
    """run_clients가 남기는 config.yaml처럼 서버 상태(server)를 붙인다. 기본은 서버도 같은 최소 채택 수."""
    return {**expected, "server": {"min_accepted": expected["plan"]["min_accepted"], **server}}


def _acc_lines(header: str, rounds: int, per_round: int, test: bool) -> str:
    rows = [f"{r},val,{i}" for r in range(1, rounds + 1) for i in range(per_round)]
    rows += [f"{rounds},test,{i}" for i in range(per_round)] if test else []
    return header + "\n" + "".join(f"{row}\n" for row in rows)


def write_result(out_dir, config, summary, rounds_logged=None, clients_logged=None, acc_rounds=None, test_rows=True,
                 skip=()):
    """끝난 실행처럼 config.yaml, summary.json과 비교에 필요한 산출물을 만든다. skip에 적은 파일은 만들지 않는다.

    class_acc.csv·client_acc.csv는 run_clients처럼 라운드마다 검증 행(클래스 10개, 클라이언트 10명)과 마지막 테스트 행.
    """
    out_dir.mkdir(parents=True)
    n = config["rounds"] if rounds_logged is None else rounds_logged
    m = n * NUM_CLIENTS if clients_logged is None else clients_logged
    a = n if acc_rounds is None else acc_rounds
    files = {
        "config.yaml": yaml.safe_dump(config, allow_unicode=True),
        "summary.json": summary,
        "rounds.csv": "round\n" + "".join(f"{r}\n" for r in range(1, n + 1)),
        "clients.csv": "round\n" + "1\n" * m,
        "class_acc.csv": _acc_lines("round,split,class", a, NUM_CLASSES, test_rows),
        "client_acc.csv": _acc_lines("round,split,client", a, NUM_CLIENTS, test_rows),
        "env.json": json.dumps({"device": "mps", "gpu": None, "torch": "2.15.0", "commit": "abc", "dirty": False}),
    }
    for name, text in files.items():
        if name not in skip:
            (out_dir / name).write_text(text, encoding="utf-8")


def test_run_status_and_index(tmp_path):
    runs = plan_runs(["baseline"], ["normal"], ["fedavg", "median", "clip_fedavg"], [0])
    expected = {run: expected_config(run, "m", rounds=None) for run in runs}
    write_result(tmp_path / runs[0].name, {**saved_config(expected[runs[0]]), "server_url": "http://127.0.0.1:1"},
                 SUMMARY)
    (tmp_path / runs[1].name).mkdir()  # 끊긴 실행

    statuses = {run: run_status(tmp_path / run.name, expected[run])[0] for run in runs}
    rows = index_rows(tmp_path, runs, statuses)

    assert list(statuses.values()) == ["done", "incomplete", "todo"]
    assert [row["status"] for row in rows] == ["done", "incomplete", "todo"]
    assert rows[0]["final_test_acc"] == 0.5 and rows[1]["final_test_acc"] == ""


def test_short_check_result_is_not_reused_for_full_run(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["median"], [0])[0]
    write_result(tmp_path / run.name, saved_config(expected_config(run, "m", rounds=2)), SUMMARY)  # 2라운드 확인 결과

    status, mismatches = run_status(tmp_path / run.name, expected_config(run, "m", rounds=None))

    assert status == "stale"
    assert any(m.startswith("rounds:") for m in mismatches)


def test_other_scenario_result_is_stale(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["median"], [0])[0]
    other = plan_runs(["delay"], ["attack"], ["median"], [0])[0]
    write_result(tmp_path / run.name, saved_config(expected_config(other, "m", rounds=None)), SUMMARY)

    assert run_status(tmp_path / run.name, expected_config(run, "m", rounds=None))[0] == "stale"


def test_broken_summary_counts_as_incomplete(tmp_path):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    write_result(tmp_path / run.name, expected_config(run, "m", rounds=None), "{")

    assert run_status(tmp_path / run.name, expected_config(run, "m", rounds=None))[0] == "incomplete"


def test_other_min_accepted_result_is_stale(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["median"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    saved = deepcopy(expected)
    saved["plan"]["min_accepted"] = 8
    write_result(tmp_path / run.name, saved_config(saved), SUMMARY)

    status, mismatches = run_status(tmp_path / run.name, expected)

    assert status == "stale"
    assert mismatches == ["plan.min_accepted: 지금 3, 기존 8", "server.min_accepted: 지금 3, 기존 8"]


def test_server_min_accepted_differs_is_stale(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["median"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected, min_accepted=8), SUMMARY)

    status, mismatches = run_status(tmp_path / run.name, expected)

    assert status == "stale" and mismatches == ["server.min_accepted: 지금 3, 기존 8"]


def test_other_per_round_result_is_stale(tmp_path):
    run = plan_runs(["partial"], ["attack"], ["median"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    saved = deepcopy(expected)
    saved["plan"]["per_round"] = 10
    write_result(tmp_path / run.name, saved_config(saved), SUMMARY)

    assert run_status(tmp_path / run.name, expected)[0] == "stale"


def test_summary_without_test_acc_is_incomplete(tmp_path):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), json.dumps({"final_val_acc": 0.4}))

    assert run_status(tmp_path / run.name, expected) == ("incomplete", ["summary.json에 final_test_acc 없음"])


@pytest.mark.parametrize("value", ["N/A", "0.5", float("nan"), float("inf"), -0.1, 1.5, True])
def test_summary_with_bad_test_acc_is_incomplete(tmp_path, value):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), json.dumps({"final_val_acc": 0.4, "final_test_acc": value}))

    status, problems = run_status(tmp_path / run.name, expected)

    assert status == "incomplete" and problems[0].startswith("summary.json의 final_test_acc가")


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0])
def test_summary_accepts_accuracy_bounds(tmp_path, value):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), json.dumps({"final_val_acc": value, "final_test_acc": value}))

    assert run_status(tmp_path / run.name, expected) == ("done", [])


@pytest.mark.parametrize("kwargs, problem", [
    ({"rounds_logged": 39}, "rounds.csv가 39줄 (설정 40라운드)"),                       # 중간에 끊긴 로그
    ({"clients_logged": 399}, "clients.csv가 399줄 (설정 40라운드 × 클라이언트 10명)"),
    ({"skip": ("class_acc.csv",)}, "class_acc.csv: 빠진 행 410개 (예: 1라운드 val class 0)"),
    ({"acc_rounds": 1}, "class_acc.csv: 빠진 행 400개 (예: 2라운드 val class 0)"),     # 평가 로그만 중간에 잘림
    ({"test_rows": False}, "class_acc.csv: 빠진 행 10개 (예: 40라운드 test class 0)"),
    ({"skip": ("env.json",)}, "env.json을 읽을 수 없음"),
])
def test_result_without_full_outputs_is_incomplete(tmp_path, kwargs, problem):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), SUMMARY, **kwargs)

    status, problems = run_status(tmp_path / run.name, expected)

    assert status == "incomplete" and problems[0].startswith(problem)   # class_acc가 잘리면 client_acc 사유도 나옴


def test_result_without_model_file_is_done(tmp_path):
    # models/는 커밋하지 않으므로, 커밋된 결과를 받은 팀원 쪽에서도 끝난 실행으로 보여야 한다
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), SUMMARY)

    assert not (tmp_path / run.name / "models").exists()
    assert run_status(tmp_path / run.name, expected) == ("done", [])


def test_unreadable_summary_reports_which_file(tmp_path):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), SUMMARY, skip=("summary.json",))

    assert run_status(tmp_path / run.name, expected) == ("incomplete", ["summary.json 없음 또는 읽을 수 없음"])


@pytest.mark.parametrize("seeds", [[0, 0], []])
def test_duplicate_or_empty_choices_are_rejected(seeds):
    with pytest.raises(ValueError):
        plan_runs(["baseline"], ["normal"], ["fedavg"], seeds)


def test_index_records_device_and_versions(tmp_path):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), SUMMARY, skip=("env.json",))
    (tmp_path / run.name / "env.json").write_text(json.dumps({"device": "cuda", "gpu": "Tesla T4", "torch": "2.15.0",
                                                               "commit": "abc", "dirty": False}), encoding="utf-8")

    row = index_rows(tmp_path, [run], {run: "done"})[0]

    assert (row["device"], row["gpu"], row["torch"], row["commit"], row["dirty"]) == ("cuda", "Tesla T4", "2.15.0",
                                                                                     "abc", False)


def test_diverged_run_counts_rounds_until_divergence(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    summary = json.dumps({**json.loads(SUMMARY), "diverged_round": 13, "rounds_completed": 12})
    write_result(tmp_path / run.name, saved_config(expected), summary, rounds_logged=12,
                 clients_logged=12 * NUM_CLIENTS)

    assert run_status(tmp_path / run.name, expected) == ("done", [])
    row = index_rows(tmp_path, [run], {run: "done"})[0]
    assert (row["diverged_round"], row["rounds_completed"]) == (13, 12)


@pytest.mark.parametrize("value", [0, 41, "13", True])
def test_diverged_round_outside_rounds_is_incomplete(tmp_path, value):
    run = plan_runs(["baseline"], ["attack"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), json.dumps({**json.loads(SUMMARY), "diverged_round": value}))

    status, problems = run_status(tmp_path / run.name, expected)

    assert status == "incomplete" and problems[0].startswith("summary.json의 diverged_round가")


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", ".", "x/..", "dlg_run.partial"])
def test_result_folder_must_stay_under_base(tmp_path, name):
    from run_clients import inside

    with pytest.raises(SystemExit):
        inside(EXPERIMENTS_DIR, name)


def test_result_folder_under_base_is_accepted():
    from run_clients import inside

    assert inside(EXPERIMENTS_DIR, "matrix/baseline-normal-fedavg-s0") == \
        (EXPERIMENTS_DIR / "matrix" / "baseline-normal-fedavg-s0").resolve()


@pytest.mark.parametrize("text", ["0", "-1", "2.5"])
def test_rounds_option_rejects_non_positive(text):
    import argparse

    from run_matrix import positive_int

    with pytest.raises((argparse.ArgumentTypeError, ValueError)):   # argparse는 둘 다 잘못된 값으로 처리
        positive_int(text)


def test_diverged_normal_run_is_incomplete(tmp_path):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    summary = json.dumps({**json.loads(SUMMARY), "diverged_round": 13, "rounds_completed": 12})
    write_result(tmp_path / run.name, saved_config(expected), summary, rounds_logged=12,
                 clients_logged=12 * NUM_CLIENTS)

    status, problems = run_status(tmp_path / run.name, expected)

    assert status == "incomplete" and "정상 실행" in problems[0]


def test_diverged_rounds_completed_must_match(tmp_path):
    run = plan_runs(["baseline"], ["attack"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    summary = json.dumps({**json.loads(SUMMARY), "diverged_round": 13, "rounds_completed": 20})
    write_result(tmp_path / run.name, saved_config(expected), summary, rounds_logged=12,
                 clients_logged=12 * NUM_CLIENTS)

    status, problems = run_status(tmp_path / run.name, expected)

    assert status == "incomplete" and "rounds_completed" in problems[0]


def test_finished_run_with_missing_outputs_stops_matrix(tmp_path):
    # 실행 프로세스가 0으로 끝나도 산출물이 덜 쓰였으면 다음 실행으로 넘어가지 않는다
    from run_matrix import check_finished

    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), SUMMARY, skip=("class_acc.csv",))

    with pytest.raises(SystemExit):
        check_finished(tmp_path, run, expected)



def test_attack_diverged_in_first_round_is_separate_status(tmp_path):
    # 1라운드 집계 직후 전역 모델이 깨진 공격 실험: 비교표에 넣을 정확도는 없지만 끝난 실행으로 보고 다음 조합으로 넘어간다
    run = plan_runs(["baseline"], ["attack"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    summary = json.dumps({"final_val_acc": None, "final_test_acc": None, "diverged_round": 1, "rounds_completed": 0})
    write_result(tmp_path / run.name, saved_config(expected), summary, rounds_logged=0, test_rows=False)

    assert run_status(tmp_path / run.name, expected) == ("diverged", [])
    row = index_rows(tmp_path, [run], {run: "diverged"})[0]
    assert (row["status"], row["final_test_acc"], row["rounds_completed"]) == ("diverged", None, 0)


def test_normal_run_diverged_in_first_round_is_incomplete(tmp_path):
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    summary = json.dumps({"final_val_acc": None, "final_test_acc": None, "diverged_round": 1, "rounds_completed": 0})
    write_result(tmp_path / run.name, saved_config(expected), summary, rounds_logged=0, test_rows=False)

    assert run_status(tmp_path / run.name, expected)[0] == "incomplete"



@pytest.mark.parametrize("name, id_key", [("class_acc.csv", "class"), ("client_acc.csv", "client")])
def test_eval_log_with_swapped_round_is_incomplete(tmp_path, name, id_key):
    # 행 수는 같지만 40라운드 검증 행이 빠지고 39라운드 행이 두 번 들어간 로그
    run = plan_runs(["baseline"], ["normal"], ["fedavg"], [0])[0]
    expected = expected_config(run, "m", rounds=None)
    write_result(tmp_path / run.name, saved_config(expected), SUMMARY)
    path = tmp_path / run.name / name
    lines = path.read_text(encoding="utf-8").splitlines()
    swapped = [line.replace("40,val,", "39,val,", 1) if line.startswith("40,val,") else line for line in lines]
    path.write_text("\n".join(swapped) + "\n", encoding="utf-8")

    status, problems = run_status(tmp_path / run.name, expected)

    assert status == "incomplete"
    assert problems == [f"{name}: 빠진 행 10개 (예: 40라운드 val {id_key} 0), 중복 행 10개 (예: 39라운드 val {id_key} 0)"]
