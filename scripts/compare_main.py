"""본 실험 결과(runs.csv)로 비교표와 그래프를 만든다 (명세 5단계 ④·⑤).

예) python scripts/compare_main.py                    # experiments/4_main/runs.csv, 환경 4 × 시드 0·1·2 × 40라운드
    python scripts/compare_main.py --prefix main_check --rounds 2 --seeds 0   # run_main.py에 준 옵션과 같게
결과: experiments/<prefix>/_tables/
- comparison.md: 최종 테스트 정확도 표, 공격 피해·방어 개선·정상 손실 표 (%p, 시드별 값과 평균)
- comparison.csv: 환경 × 지표 × 시드마다 한 줄 (value_pp는 40라운드끼리의 차이, reference_pp는 발산 전 모델도 뺀 참고값)
- accuracy.svg: 환경별 최종 테스트 정확도 평균 막대그래프 (발산한 조건은 막대 없이 '발산')
예정 조합(환경 × 정상·공격 × 집계 3 × 시드, 기본은 run_main.py와 같음)이 runs.csv와 정확히 같아야 한다.
빠지거나 예정에 없는 실행, 끝나지 않은 실행, 목표 라운드 수(base.yaml, --rounds)와 다른 완료·발산 기록이 있으면
파일을 쓰지 않고 멈춘다 (예: 같은 prefix에서 --envs baseline만 돌려 runs.csv가 18회로 바뀐 경우)
"""

import argparse
import csv
import sys
from pathlib import Path

from run_clients import EXPERIMENTS_DIR, ROOT, inside
from run_main import (
    AGGREGATORS,
    ATTACKS,
    DEFAULT_PREFIX,
    DEFAULT_SEEDS,
    ENVS,
    plan_runs,
    positive_int,
)

from fedsec.experiment.bar_chart import grouped_bar_svg
from fedsec.experiment.comparison import (
    AGG_NAMES,
    ATTACK_NAMES,
    CONDITIONS,
    Key,
    RunResult,
    accuracy_table,
    compare,
    parse_runs,
    to_csv,
    to_markdown,
)
from fedsec.experiment.config import load_config

BASE_CONFIG = "configs/base.yaml"   # run_main.py가 쓰는 학습 설정 (rounds의 기본값)
TABLES_DIR = "_tables"
OUTPUT_FILES = ("comparison.md", "comparison.csv", "accuracy.svg")


def read_runs(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"runs.csv를 읽지 못했습니다 ({path}): {error}") from error
    if not rows:
        raise ValueError(f"runs.csv에 실행이 없습니다: {path}")
    # csv.DictReader는 모자란 열에 None을, 남는 값은 None 키에 넣는다 (잘리거나 깨진 행)
    broken = [line for line, row in enumerate(rows, start=2) if None in row or None in row.values()]
    if broken:
        raise ValueError(f"runs.csv의 열 수가 머리줄과 다른 행이 있습니다 (잘리거나 깨진 파일): {path} {broken[:5]}행")
    return rows


def check_plan(results: dict[Key, RunResult], envs: list[str], seeds: list[int]) -> None:
    """runs.csv의 실행이 예정 조합과 정확히 같은지. 빠진 실행은 compare()가, 예정에 없는 실행은 여기서 멈춘다."""
    planned = {(env, *condition, seed) for env in envs for condition in CONDITIONS for seed in seeds}
    extra = sorted(f"{e}-{a}-{g}-s{s}" for e, a, g, s in set(results) - planned)
    if extra:
        raise ValueError(f"runs.csv에 예정에 없는 실행이 있습니다 (--envs·--seeds를 run_main.py와 같게): {extra[:5]}")


def accuracy_chart(results: dict[Key, RunResult], envs: list[str], seeds: list[int]) -> str:
    """환경 × 6조건의 시드 평균 정확도. 한 시드라도 발산했으면 막대 없이 '발산'."""
    values, notes = [], []
    for env in envs:
        runs = [[results[(env, *condition, seed)] for seed in seeds] for condition in CONDITIONS]
        diverged = [any(r.diverged_round for r in cell) for cell in runs]
        values.append([None if d else sum(r.acc for r in cell) / len(cell) * 100 for cell, d in zip(runs, diverged)])
        notes.append(["발산" if d else "" for d in diverged])
    series = [f"{ATTACK_NAMES[a]} {AGG_NAMES[g]}" for a, g in CONDITIONS]
    return grouped_bar_svg(f"최종 테스트 정확도 (시드 {len(seeds)}개 평균, %)", envs, series, values, notes)


def build_outputs(rows: list[dict], envs: list[str], seeds: list[int], rounds: int) -> dict[str, str]:
    """파일 이름 → 내용. 예정과 다른 실행·끝나지 않은 실행·잘못된 값이 있으면 ValueError."""
    plan_runs(envs, ATTACKS, AGGREGATORS, seeds)   # 겹치거나 모르는 환경·시드는 run_main.py와 같은 기준으로 거부
    results = parse_runs(rows, rounds)
    check_plan(results, envs, seeds)
    comparison = compare(results, envs, seeds)   # 빠진 조합은 여기서 ValueError (그래프보다 먼저)
    seed_text = " / ".join(str(s) for s in seeds)
    markdown = (f"최종 테스트 정확도 (%, 시드 {seed_text}, 괄호는 평균)\n\n{accuracy_table(results, envs, seeds)}\n"
                f"{to_markdown(comparison, envs)}")
    return {"comparison.md": markdown, "comparison.csv": to_csv(comparison),
            "accuracy.svg": accuracy_chart(results, envs, seeds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="experiments/ 아래 묶음 폴더 이름 (run_main.py와 같음)")
    parser.add_argument("--envs", nargs="+", default=list(ENVS), choices=ENVS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--rounds", type=positive_int, help="목표 라운드 수 (없으면 base.yaml 값, run_main.py와 같게)")
    args = parser.parse_args()

    prefix_dir = inside(EXPERIMENTS_DIR, args.prefix)
    try:
        rounds = args.rounds or load_config(ROOT / BASE_CONFIG).rounds
        outputs = build_outputs(read_runs(prefix_dir / "runs.csv"), args.envs, args.seeds, rounds)
    except (OSError, KeyError, ValueError) as error:
        sys.exit(f"비교표를 만들지 못했습니다: {error}")
    out_dir = prefix_dir / TABLES_DIR
    out_dir.mkdir(exist_ok=True)
    for name, text in outputs.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    print(outputs["comparison.md"], end="")
    print(f"\n저장: {out_dir.relative_to(EXPERIMENTS_DIR.parent)}/ ({', '.join(OUTPUT_FILES)})")


if __name__ == "__main__":
    main()
