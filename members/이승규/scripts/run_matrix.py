"""4단계 비교 실험 일괄 실행: 환경 4 × (정상, 공격) × (FedAvg, 클리핑, 중앙값) × 시드.

실행 하나마다 run_server_experiment.py를 띄운다 (서버를 새로 띄우므로 라운드 1·버전 0에서 시작).
최종 테스트 정확도로 공격 피해·방어 개선·정상 손실을 계산하므로 --evaluate-test를 항상 켠다.

예) python scripts/run_matrix.py --dry-run                         # 목록과 상태만 출력 (기본 시드 0 1 2 → 72회)
    python scripts/run_matrix.py --seeds 0 1 2                     # 본 실험
    python scripts/run_matrix.py --prefix matrix_check --rounds 2 --seeds 0 --attacks attack   # 동작 확인 12회
결과: experiments/<prefix>/<환경>-<normal|attack>-<집계>-s<시드>/, 실행 목록과 결과 요약 experiments/<prefix>/runs.csv
- 끝난 실행(summary.json과 비교에 필요한 로그가 모두 있음)은 건너뛴다 → 중간에 끊겨도 같은 명령으로 이어서 실행
- 단, 저장된 config.yaml이 지금 실행할 설정(base.yaml·환경·참여 수·최소 채택 수·공격·방어·라운드 수)과 다르면
  시작 전에 멈춘다 (예: --rounds 2로 확인한 결과를 40라운드 본 실험으로 착각하지 않게)
- 끊긴 실행(summary.json 없음, 검증·테스트 정확도가 빠졌거나 0~1 사이 숫자가 아님, rounds.csv·clients.csv 줄 수가
  라운드 수와 맞지 않음, class_acc.csv·client_acc.csv에 라운드·분할·ID별 행이 빠지거나 중복됨, env.json을 읽을 수 없음,
  정상 실행이 발산함)이 있으면 시작 전에 멈춘다
  → 그 폴더를 지우거나 옮긴 뒤 다시 실행
- 공격 실험이 발산하면 발산 전 전역 모델로 평가해 끝난 실행으로 친다 (summary.json의 diverged_round).
  1라운드 집계 직후 발산하면 정확도 없는 diverged 상태로 끝난 실행으로 친다 (비교표에 넣지 않음)
- 실행이 실패하거나, 끝났는데 위 판정이 done·diverged가 아니면 그 자리에서 멈춘다
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from run_clients import (
    DEFAULT_NUM_CLIENTS,
    EXPERIMENTS_DIR,
    add_experiment_args,
    add_scenario_args,
    client_setup,
    inside,
)

MEMBER_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = Path(__file__).resolve().parent / "run_server_experiment.py"
ENVS = ("baseline", "delay", "partial", "label_skew")
ATTACKS = ("normal", "attack")
AGGREGATORS = ("fedavg", "clip_fedavg", "median")   # experiment.config.AGGREGATORS와 같음
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_PREFIX = "matrix"
NUM_CLIENTS = DEFAULT_NUM_CLIENTS
NUM_CLASSES = 10    # CIFAR-10. class_acc.csv는 라운드마다 클래스 10줄
# 실행 환경 (기기가 다르면 같은 시드도 결과가 조금 다를 수 있어, 비교표에서 섞였는지 보이게 한다)
RUN_ENV_KEYS = ("device", "gpu", "torch", "commit", "dirty")
# 클래스별·클라이언트별 비교에 쓰는 로그와 라운드당 줄 수. 완료한 라운드마다 검증 행, 끝에 테스트 행이 있어야 한다
ACC_LOGS = {"class_acc.csv": ("class", NUM_CLASSES), "client_acc.csv": ("client", NUM_CLIENTS)}
SUMMARY_KEYS = ("final_val_acc", "final_test_acc", "diverged_round", "rounds_completed", "rounds_not_updated",
                "total_virtual_time", "total_sec")
# 비교표에 꼭 있어야 하는 결과. 0~1 사이 유한한 숫자가 아니면 끝난 실행으로 치지 않음
REQUIRED_SUMMARY_KEYS = ("final_val_acc", "final_test_acc")
FINISHED = ("done", "diverged")    # 이어서 실행할 때 건너뛰는 상태


@dataclass(frozen=True)
class Run:
    env: str
    attack: str      # normal | attack
    aggregator: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.env}-{self.attack}-{self.aggregator}-s{self.seed}"

    def command(self, prefix: str, rounds: int | None) -> list[str]:
        command = [sys.executable, str(RUN_SCRIPT), "--name", f"{prefix}/{self.name}",
                   "--env", f"configs/env/{self.env}.yaml", "--aggregator", self.aggregator,
                   "--attack" if self.attack == "attack" else "--no-attack",
                   "--seed", str(self.seed), "--evaluate-test"]
        return command + (["--rounds", str(rounds)] if rounds is not None else [])


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 이상이어야 합니다: {value}")
    return value


def plan_runs(envs, attacks, aggregators, seeds) -> list[Run]:
    """시드 → 환경 → 공격 → 집계 순. 시드 하나를 끝내면 그 시드의 비교표가 다 채워진다."""
    for name, chosen, allowed in (("env", envs, ENVS), ("attack", attacks, ATTACKS),
                                  ("aggregator", aggregators, AGGREGATORS), ("seed", seeds, None)):
        unknown = set(chosen) - set(allowed) if allowed else set()
        if unknown:
            raise ValueError(f"알 수 없는 {name}: {sorted(unknown)} (가능: {', '.join(allowed)})")
        if not chosen or len(set(chosen)) != len(chosen):
            raise ValueError(f"{name}는 겹치지 않는 값이 1개 이상이어야 합니다: {list(chosen)}")
    return [Run(env, attack, agg, seed) for seed in seeds for env in envs for attack in attacks for agg in aggregators]


def expected_config(run: Run, prefix: str, rounds: int | None) -> dict:
    """이 실행이 config.yaml에 남길 설정. run_clients.py와 같은 함수로 만든다.

    서버 주소처럼 실행마다 바뀌는 값은 넣지 않는다. 저장할 때와 같은 YAML 왕복으로 형을 맞춘다.
    """
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)
    add_scenario_args(parser)
    args = parser.parse_args(run.command(prefix, rounds)[2:])
    cfg, plan, _ = client_setup(args)
    return yaml.safe_load(yaml.safe_dump({**cfg.to_dict(), "plan": plan.to_dict()}, allow_unicode=True))


def config_mismatches(saved: dict, expected: dict) -> list[str]:
    """실행 설정(학습·일정·참여 수·최소 채택 수 등)과 서버가 실제로 쓴 최소 채택 수를 비교한다."""
    mismatches = _diff(saved, expected, prefix="")
    server = saved.get("server")
    server_min = server.get("min_accepted") if isinstance(server, dict) else None
    if server_min != expected["plan"]["min_accepted"]:
        mismatches.append(f"server.min_accepted: 지금 {expected['plan']['min_accepted']}, 기존 {server_min}")
    return mismatches


def _diff(saved: dict, expected: dict, prefix: str) -> list[str]:
    """다른 항목을 "train.weight_decay: 지금 0.0005, 기존 0.0"처럼 가장 안쪽 키로 보여 준다."""
    lines = []
    for key, want in expected.items():
        have = saved.get(key)
        if isinstance(want, dict) and isinstance(have, dict):
            lines += _diff(have, want, f"{prefix}{key}.")
        elif have != want:
            lines.append(f"{prefix}{key}: 지금 {want}, 기존 {have}")
    return lines


def summary_problems(summary: dict) -> list[str]:
    """비교표에 쓸 정확도가 빠졌거나 0~1 사이 유한한 숫자가 아니면 그 항목들."""
    problems = []
    for key in REQUIRED_SUMMARY_KEYS:
        value = summary.get(key)
        if value is None:
            problems.append(f"summary.json에 {key} 없음")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or not 0 <= value <= 1:
            problems.append(f"summary.json의 {key}가 0~1 사이 숫자가 아님: {value!r}")
    return problems


def output_problems(out_dir: Path, rounds: int) -> list[str]:
    """summary.json 말고도 비교에 필요한 산출물이 끝까지 쓰였는지.

    models/는 커밋하지 않으므로 보지 않는다. summary.json을 모델 저장 뒤에 쓰므로 끝까지 갔는지는 summary로 안다.
    """
    problems = []
    logged = _count_rows(out_dir / "rounds.csv")
    if logged != rounds:
        problems.append(f"rounds.csv가 {logged}줄 (설정 {rounds}라운드)")
    logged = _count_rows(out_dir / "clients.csv")
    if logged != rounds * NUM_CLIENTS:
        problems.append(f"clients.csv가 {logged}줄 (설정 {rounds}라운드 × 클라이언트 {NUM_CLIENTS}명)")
    for name, (id_key, per_round) in ACC_LOGS.items():
        problems += acc_log_problems(out_dir / name, id_key, per_round, rounds)
    if not _read_json(out_dir / "env.json"):
        problems.append("env.json을 읽을 수 없음")
    return problems


def acc_log_problems(path: Path, id_key: str, per_round: int, rounds: int) -> list[str]:
    """평가 로그에 (라운드, 분할, 클래스·클라이언트 ID)마다 정확히 한 줄씩 있는지. 행 수만 세면 라운드가 통째로
    빠지고 다른 라운드가 중복돼도 통과한다. 1라운드에 발산한 실행(rounds 0)은 평가할 모델이 없어 테스트 행도 없다."""
    expected = {(r, "val", i) for r in range(1, rounds + 1) for i in range(per_round)}
    expected |= {(rounds, "test", i) for i in range(per_round)} if rounds > 0 else set()
    counts = Counter(_acc_keys(path, id_key))
    missing = sorted(expected - counts.keys())
    duplicated = sorted(key for key, n in counts.items() if n > 1 and key in expected)
    unexpected = sorted((key for key in counts if key not in expected), key=str)
    parts = [f"{label} {len(keys)}개 (예: {_key_text(keys[0], id_key)})"
             for label, keys in (("빠진 행", missing), ("중복 행", duplicated), ("예상 밖 행", unexpected)) if keys]
    return [f"{path.name}: " + ", ".join(parts)] if parts else []


def _acc_keys(path: Path, id_key: str) -> list[tuple]:
    try:
        with path.open(encoding="utf-8") as f:
            return [_acc_key(row, id_key) for row in csv.DictReader(f)]
    except OSError:
        return []


def _acc_key(row: dict, id_key: str) -> tuple:
    try:
        return int(row["round"]), row["split"], int(row[id_key])
    except (KeyError, TypeError, ValueError):
        return row.get("round"), row.get("split"), row.get(id_key)   # 형식이 틀린 줄은 예상 밖 행으로 잡힌다


def _key_text(key: tuple, id_key: str) -> str:
    round_idx, split, item = key
    return f"{round_idx}라운드 {split} {id_key} {item}"


def _count_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except OSError:
        return 0



def diverged_problems(summary: dict, expected: dict) -> list[str]:
    """발산한 실행이 믿을 만한지. 발산은 공격 실험의 결과로만 인정한다 (정상 실행의 발산은 설정 문제)."""
    diverged, completed, rounds = summary["diverged_round"], summary.get("rounds_completed"), expected["rounds"]
    if type(diverged) is not int or not 1 <= diverged <= rounds:
        return [f"summary.json의 diverged_round가 1~{rounds} 사이 정수가 아님: {diverged!r}"]
    if not expected["plan"]["attack"]:
        return [f"정상 실행이 {diverged}라운드에 발산함 (공격 실험에서만 결과로 인정)"]
    if completed != diverged - 1:
        return [f"summary.json의 rounds_completed {completed!r}가 diverged_round - 1({diverged - 1})과 다름"]
    return []

def run_status(out_dir: Path, expected: dict) -> tuple[str, list[str]]:
    """("todo" | "incomplete" | "stale" | "done" | "diverged", 다른 설정 또는 빠진 결과 목록).

    done·diverged는 끝난 실행이다. diverged는 1라운드에 발산해 정확도가 없는 공격 실험 (비교표에서 따로 표시).
    """
    if not out_dir.exists():
        return "todo", []
    summary = _read_json(out_dir / "summary.json")
    if not summary:
        return "incomplete", ["summary.json 없음 또는 읽을 수 없음"]
    try:
        saved = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        saved = None
    if not isinstance(saved, dict):
        return "incomplete", ["config.yaml 없음 또는 읽을 수 없음"]
    diverged = summary.get("diverged_round")
    # 1라운드 집계 직후 깨진 공격 실험은 정확도가 없다 (비교표에 넣지 않는 별도 상태 diverged)
    at_start = diverged == 1 and type(diverged) is int
    problems = [] if at_start else summary_problems(summary)
    if problems:
        return "incomplete", problems
    mismatches = config_mismatches(saved, expected)
    if mismatches:
        return "stale", mismatches
    if diverged is not None:
        problems = diverged_problems(summary, expected)
        if problems:
            return "incomplete", problems
    # 발산한 실행은 발산 전 라운드까지만 로그가 있다 (program.run_clients)
    problems = output_problems(out_dir, expected["rounds"] if diverged is None else diverged - 1)
    if problems:
        return "incomplete", problems
    return ("diverged", []) if at_start else ("done", [])


def index_rows(prefix_dir: Path, runs: list[Run], statuses: dict[Run, str]) -> list[dict]:
    rows = []
    for run in runs:
        out_dir = prefix_dir / run.name
        status = statuses[run]
        finished = status in FINISHED
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) if finished else {}
        env = _read_json(out_dir / "env.json") if finished else {}
        rows.append({"name": run.name, "env": run.env, "attack": run.attack, "aggregator": run.aggregator,
                     "seed": run.seed, "status": status, **{key: summary.get(key, "") for key in SUMMARY_KEYS},
                     **{key: env.get(key, "") for key in RUN_ENV_KEYS}})
    return rows


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_index(prefix_dir: Path, rows: list[dict]) -> Path:
    prefix_dir.mkdir(parents=True, exist_ok=True)
    path = prefix_dir / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path



def check_finished(prefix_dir: Path, run: Run, expected: dict) -> None:
    """실행 프로세스가 0으로 끝나도 산출물이 덜 쓰였거나 설정이 다르면 다음 실행으로 넘어가지 않는다."""
    status, problems = run_status(prefix_dir / run.name, expected)
    if status not in FINISHED:
        sys.exit(f"실행은 끝났지만 결과가 {status}입니다: {run.name}\n  " + "\n  ".join(problems))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="experiments/ 아래 묶음 폴더 이름")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--envs", nargs="+", default=list(ENVS))
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    parser.add_argument("--aggregators", nargs="+", default=list(AGGREGATORS))
    parser.add_argument("--rounds", type=positive_int, help="라운드 수 덮어쓰기 (동작 확인용. 본 실험은 base.yaml 값)")
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 목록과 상태만 출력")
    args = parser.parse_args()

    runs = plan_runs(args.envs, args.attacks, args.aggregators, args.seeds)
    prefix_dir = inside(EXPERIMENTS_DIR, args.prefix)
    checked = {run: run_status(prefix_dir / run.name, expected_config(run, args.prefix, args.rounds)) for run in runs}
    statuses = {run: status for run, (status, _) in checked.items()}
    for i, run in enumerate(runs, 1):
        print(f"{i:>3}/{len(runs)} [{statuses[run]:>10}] {run.name}")
    incomplete = [run.name + "".join(f"\n    {reason}" for reason in checked[run][1])
                  for run in runs if statuses[run] == "incomplete"]
    if incomplete:
        sys.exit("끊긴 실행이 있습니다 (아래 사유: 빠진 산출물, 잘못된 값, 정상 실행의 발산 등). 지우거나 옮긴 뒤 다시 실행하세요:\n  "
                 + "\n  ".join(incomplete))
    stale = [f"{run.name}\n    " + "\n    ".join(checked[run][1]) for run in runs if statuses[run] == "stale"]
    if stale:
        sys.exit("같은 이름의 결과가 지금 설정과 다릅니다. --prefix를 바꾸거나 폴더를 옮긴 뒤 다시 실행하세요:\n  "
                 + "\n  ".join(stale))
    todo = [run for run in runs if statuses[run] == "todo"]
    print(f"전체 {len(runs)}회, 끝남 {len(runs) - len(todo)}회, 남음 {len(todo)}회")
    if args.dry_run:
        if todo:
            print("첫 명령: " + " ".join(todo[0].command(args.prefix, args.rounds)))
        return

    try:
        for i, run in enumerate(todo, 1):
            print(f"\n=== {i}/{len(todo)} {run.name} ===", flush=True)
            if subprocess.run(run.command(args.prefix, args.rounds), check=False, cwd=MEMBER_ROOT).returncode != 0:
                sys.exit(f"실행 실패: {run.name} (끊긴 폴더를 정리한 뒤 같은 명령으로 다시 실행하면 이 실행부터 이어서 함)")
            check_finished(prefix_dir, run, expected_config(run, args.prefix, args.rounds))
    finally:
        final = {run: run_status(prefix_dir / run.name, expected_config(run, args.prefix, args.rounds))[0] for run in runs}
        print(f"요약: {write_index(prefix_dir, index_rows(prefix_dir, runs, final)).relative_to(MEMBER_ROOT)}")


if __name__ == "__main__":
    main()
