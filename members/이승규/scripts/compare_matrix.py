"""본 실험 결과(runs.csv)로 비교표와 그래프를 만든다 (명세 5단계 ④·⑤).

예) python scripts/compare_matrix.py                    # experiments/matrix/runs.csv
    python scripts/compare_matrix.py --prefix matrix_check
결과: experiments/<prefix>/_tables/
- comparison.md: 최종 테스트 정확도 표, 공격 피해·방어 개선·정상 손실 표 (%p, 시드별 값과 평균)
- comparison.csv: 환경 × 지표 × 시드마다 한 줄 (value_pp는 40라운드끼리의 차이, reference_pp는 발산 전 모델도 뺀 참고값)
- accuracy.svg: 환경별 최종 테스트 정확도 평균 막대그래프 (발산한 조건은 막대 없이 '발산')
끝나지 않은 실행이 있거나 조합이 빠지면 멈춘다 → run_matrix.py로 마저 실행
"""

import argparse
import csv
import sys
from pathlib import Path

from run_clients import EXPERIMENTS_DIR, inside

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

DEFAULT_PREFIX = "matrix"   # run_matrix.DEFAULT_PREFIX와 같음
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
    return rows


def run_order(rows: list[dict]) -> tuple[list[str], list[int]]:
    """runs.csv에 처음 나온 순서대로 환경과 시드."""
    envs = list(dict.fromkeys(row["env"] for row in rows))
    seeds = list(dict.fromkeys(int(row["seed"]) for row in rows))
    return envs, seeds


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


def build_outputs(rows: list[dict]) -> dict[str, str]:
    """파일 이름 → 내용. 끝나지 않은 실행·빠진 조합·잘못된 값이 있으면 ValueError."""
    envs, seeds = run_order(rows)
    results = parse_runs(rows)
    comparison = compare(results, envs, seeds)   # 빠진 조합은 여기서 ValueError (그래프보다 먼저)
    seed_text = " / ".join(str(s) for s in seeds)
    markdown = (f"최종 테스트 정확도 (%, 시드 {seed_text}, 괄호는 평균)\n\n{accuracy_table(results, envs, seeds)}\n"
                f"{to_markdown(comparison, envs)}")
    return {"comparison.md": markdown, "comparison.csv": to_csv(comparison),
            "accuracy.svg": accuracy_chart(results, envs, seeds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="experiments/ 아래 묶음 폴더 이름 (run_matrix.py와 같음)")
    args = parser.parse_args()

    prefix_dir = inside(EXPERIMENTS_DIR, args.prefix)
    try:
        outputs = build_outputs(read_runs(prefix_dir / "runs.csv"))
    except (KeyError, ValueError) as error:
        sys.exit(f"비교표를 만들지 못했습니다: {error}")
    out_dir = prefix_dir / TABLES_DIR
    out_dir.mkdir(exist_ok=True)
    for name, text in outputs.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    print(outputs["comparison.md"], end="")
    print(f"\n저장: {out_dir.relative_to(EXPERIMENTS_DIR.parent)}/ ({', '.join(OUTPUT_FILES)})")


if __name__ == "__main__":
    main()
