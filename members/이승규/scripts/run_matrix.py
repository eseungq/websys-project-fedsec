"""4단계 비교 실험 일괄 실행: 환경 4 × (정상, 공격) × (FedAvg, 클리핑, 중앙값) × 시드.

실행 하나마다 run_server_experiment.py를 띄운다 (서버를 새로 띄우므로 라운드 1·버전 0에서 시작).
최종 테스트 정확도로 공격 피해·방어 개선·정상 손실을 계산하므로 --evaluate-test를 항상 켠다.

예) python scripts/run_matrix.py --dry-run                         # 목록과 상태만 출력 (기본 시드 0 1 2 → 72회)
    python scripts/run_matrix.py --seeds 0 1 2                     # 본 실험
    python scripts/run_matrix.py --prefix matrix_check --rounds 2 --seeds 0 --attacks attack   # 동작 확인 12회
결과: experiments/<prefix>/<환경>-<normal|attack>-<집계>-s<시드>/, 실행 목록과 결과 요약 experiments/<prefix>/runs.csv
- 끝난 실행(summary.json 있음)은 건너뛴다 → 중간에 끊겨도 같은 명령으로 이어서 실행
- 끊긴 실행(폴더는 있는데 summary.json 없음)이 있으면 시작 전에 멈춘다 → 그 폴더를 지우거나 옮긴 뒤 다시 실행
- 실행이 실패하면 그 자리에서 멈춘다
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MEMBER_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = Path(__file__).resolve().parent / "run_server_experiment.py"
ENVS = ("baseline", "delay", "partial", "label_skew")
ATTACKS = ("normal", "attack")
AGGREGATORS = ("fedavg", "clip_fedavg", "median")   # experiment.config.AGGREGATORS와 같음 (torch 없이 목록만 보려고 따로 둠)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_PREFIX = "matrix"
SUMMARY_KEYS = ("final_val_acc", "final_test_acc", "rounds_not_updated", "total_virtual_time", "total_sec")


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
        return command + (["--rounds", str(rounds)] if rounds else [])


def plan_runs(envs, attacks, aggregators, seeds) -> list[Run]:
    """시드 → 환경 → 공격 → 집계 순. 시드 하나를 끝내면 그 시드의 비교표가 다 채워진다."""
    for name, chosen, allowed in (("env", envs, ENVS), ("attack", attacks, ATTACKS),
                                  ("aggregator", aggregators, AGGREGATORS)):
        unknown = set(chosen) - set(allowed)
        if unknown:
            raise ValueError(f"알 수 없는 {name}: {sorted(unknown)} (가능: {', '.join(allowed)})")
    return [Run(env, attack, agg, seed) for seed in seeds for env in envs for attack in attacks for agg in aggregators]


def run_status(out_dir: Path) -> str:
    if (out_dir / "summary.json").exists():
        return "done"
    return "incomplete" if out_dir.exists() else "todo"


def index_rows(prefix_dir: Path, runs: list[Run]) -> list[dict]:
    rows = []
    for run in runs:
        out_dir = prefix_dir / run.name
        status = run_status(out_dir)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) if status == "done" else {}
        rows.append({"name": run.name, "env": run.env, "attack": run.attack, "aggregator": run.aggregator,
                     "seed": run.seed, "status": status, **{key: summary.get(key, "") for key in SUMMARY_KEYS}})
    return rows


def write_index(prefix_dir: Path, rows: list[dict]) -> Path:
    prefix_dir.mkdir(parents=True, exist_ok=True)
    path = prefix_dir / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="experiments/ 아래 묶음 폴더 이름")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--envs", nargs="+", default=list(ENVS))
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    parser.add_argument("--aggregators", nargs="+", default=list(AGGREGATORS))
    parser.add_argument("--rounds", type=int, help="라운드 수 덮어쓰기 (동작 확인용. 본 실험은 base.yaml 값)")
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 목록과 상태만 출력")
    args = parser.parse_args()

    runs = plan_runs(args.envs, args.attacks, args.aggregators, args.seeds)
    prefix_dir = MEMBER_ROOT / "experiments" / args.prefix
    statuses = {run: run_status(prefix_dir / run.name) for run in runs}
    for i, run in enumerate(runs, 1):
        print(f"{i:>3}/{len(runs)} [{statuses[run]:>10}] {run.name}")
    incomplete = [run.name for run in runs if statuses[run] == "incomplete"]
    if incomplete:
        sys.exit(f"끊긴 실행이 있습니다 (summary.json 없음). 지우거나 옮긴 뒤 다시 실행하세요: {', '.join(incomplete)}")
    todo = [run for run in runs if statuses[run] == "todo"]
    print(f"전체 {len(runs)}회, 끝남 {len(runs) - len(todo)}회, 남음 {len(todo)}회")
    if args.dry_run:
        if todo:
            print("첫 명령: " + " ".join(todo[0].command(args.prefix, args.rounds)))
        return

    try:
        for i, run in enumerate(todo, 1):
            print(f"\n=== {i}/{len(todo)} {run.name} ===", flush=True)
            if subprocess.run(run.command(args.prefix, args.rounds), cwd=MEMBER_ROOT).returncode != 0:
                sys.exit(f"실행 실패: {run.name} (끊긴 폴더를 정리한 뒤 같은 명령으로 다시 실행하면 이 실행부터 이어서 함)")
    finally:
        print(f"요약: {write_index(prefix_dir, index_rows(prefix_dir, runs)).relative_to(MEMBER_ROOT)}")


if __name__ == "__main__":
    main()
