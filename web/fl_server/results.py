"""조회형: experiments/ 아래 결과 폴더를 찾고 읽는다 (파일을 쓰지 않음).

결과 폴더는 summary.json이 있는 폴더다 (models/, dlg_images/, *.partial 아래는 보지 않음).
재실행 도구의 결과(experiments/replay/<name>/)는 같은 파일에 replay.json이 더 있다.
"""

import os
from pathlib import Path

from fedsec.experiment.comparison import (
    AGG_NAMES,
    ATTACK_NAMES,
    CONDITIONS,
    METRIC_NAMES,
)
from web.fl_server.files import NotFound, read_csv, read_json, read_yaml, resolve_inside

SKIP_DIRS = frozenset({"models", "dlg_images", "logs"})
PARTIAL_SUFFIX = ".partial"
SUMMARY_KEYS = ("env", "attack", "aggregator", "seed", "rounds_completed", "final_val_acc", "final_test_acc",
                "diverged_round", "runner")
MATRIX_DIR = "matrix"
TABLES_DIR = "_tables"
COMPARISON_CSV = "comparison.csv"
ACCURACY_SVG = "accuracy.svg"
ENV_ORDER = ("baseline", "delay", "partial", "label_skew")   # run_matrix.ENVS와 같은 순서


def _skipped(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(PARTIAL_SUFFIX) or name.startswith(".")


def _run_folders(experiments_dir: Path) -> list[Path]:
    found = []
    for current, dirs, files in os.walk(experiments_dir):
        dirs[:] = sorted(d for d in dirs if not _skipped(d))   # 하위 폴더 탐색을 줄인다 (모델 파일 등)
        if "summary.json" in files:
            found.append(Path(current))
    return found


def _replay_info(run_dir: Path) -> dict | None:
    replay = read_json(run_dir / "replay.json")
    if replay is None:
        return None
    return {"source": replay.get("source"), "from_round": replay.get("from_round"), "change": replay.get("change")}


def _run_entry(experiments_dir: Path, run_dir: Path) -> dict:
    summary = read_json(run_dir / "summary.json") or {}
    run_id = run_dir.relative_to(experiments_dir).as_posix()
    group = run_dir.parent.relative_to(experiments_dir).as_posix()
    return {"id": run_id, "group": "" if group == "." else group, "name": run_dir.name,
            **{key: summary.get(key) for key in SUMMARY_KEYS}, "replay": _replay_info(run_dir)}


def discover_runs(experiments_dir: Path) -> list[dict]:
    """summary.json이 있는 모든 결과 폴더 (이름순)."""
    if not experiments_dir.is_dir():
        return []
    base = experiments_dir.resolve()
    return [_run_entry(base, run_dir) for run_dir in sorted(_run_folders(base))]


def run_dir_for(experiments_dir: Path, run_id: str) -> Path:
    """run_id가 experiments/ 안의 결과 폴더(summary.json 또는 rounds.csv가 있음)인지 확인한다."""
    run_dir = resolve_inside(experiments_dir, run_id)
    parts = run_dir.relative_to(experiments_dir.resolve()).parts
    if any(_skipped(part) for part in parts):
        raise NotFound("결과 폴더가 아닙니다.")
    if not run_dir.is_dir() or not ((run_dir / "summary.json").is_file() or (run_dir / "rounds.csv").is_file()):
        raise NotFound(f"결과 폴더가 없습니다: {run_id}")
    return run_dir


def _last_round(rows: list[dict]) -> list[dict]:
    """class_acc.csv·client_acc.csv의 마지막 라운드 행 (검증·테스트 모두)."""
    if not rows:
        return []
    last = max(row["round"] for row in rows if isinstance(row.get("round"), int))
    return [row for row in rows if row.get("round") == last]


def read_run_files(experiments_dir: Path, run_dir: Path) -> dict:
    """한 결과 폴더의 화면용 내용. 진행 중이면 없는 파일은 빈 값."""
    config = read_yaml(run_dir / "config.yaml") or {}
    return {
        "id": run_dir.relative_to(experiments_dir.resolve()).as_posix(),
        "summary": read_json(run_dir / "summary.json"),
        "config": config,
        "plan": config.get("plan") if isinstance(config.get("plan"), dict) else None,
        "env_info": read_json(run_dir / "env.json"),
        "replay": _replay_info(run_dir),
        "rounds": read_csv(run_dir / "rounds.csv"),
        "clients": read_csv(run_dir / "clients.csv"),
        "class_acc": _last_round(read_csv(run_dir / "class_acc.csv")),
        "client_acc": _last_round(read_csv(run_dir / "client_acc.csv")),
        "exclusions": read_csv(run_dir / "server" / "exclusions.csv"),
    }


def run_detail(experiments_dir: Path, run_id: str) -> dict:
    return read_run_files(experiments_dir, run_dir_for(experiments_dir, run_id))


def _accuracy_table(runs_rows: list[dict]) -> list[dict]:
    """runs.csv → 환경 × 조건(정상·공격 × 집계)마다 시드별 최종 테스트 정확도와 발산 라운드."""
    cells = {}
    for row in runs_rows:
        key = (row.get("env"), row.get("attack"), row.get("aggregator"))
        seed_value = {"seed": row.get("seed"), "acc": row.get("final_test_acc"),
                      "diverged_round": row.get("diverged_round"), "status": row.get("status")}
        cells = {**cells, key: [*cells.get(key, []), seed_value]}
    return [{"env": env, "attack": attack, "aggregator": agg, "seeds": seeds}
            for (env, attack, agg), seeds in cells.items()]


def _metric_groups(rows: list[dict]) -> list[dict]:
    """comparison.csv → (환경, 지표, 방어)마다 시드별 값과 평균."""
    groups = {}
    for row in rows:
        key = (row.get("env"), row.get("metric"), row.get("defense") or "")
        groups = {**groups, key: [*groups.get(key, []), row]}
    return [{"env": env, "metric": metric, "defense": defense,
             "seeds": [{"seed": r.get("seed"), "value_pp": r.get("value_pp"), "reference_pp": r.get("reference_pp"),
                        "diverged_rounds": r.get("diverged_rounds")} for r in seeds],
             "mean_value_pp": _mean([r.get("value_pp") for r in seeds]),
             "mean_reference_pp": _mean([r.get("reference_pp") for r in seeds])}
            for (env, metric, defense), seeds in groups.items()]


def _mean(values: list) -> float | None:
    """모든 시드에 값이 있을 때만 평균 (발산한 시드가 있으면 None)."""
    if not values or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
        return None
    return round(sum(values) / len(values), 2)


def comparison(experiments_dir: Path) -> dict:
    """compare_matrix.py 결과(_tables/comparison.csv)와 runs.csv의 최종 정확도."""
    matrix = experiments_dir / MATRIX_DIR
    tables = matrix / TABLES_DIR
    metrics = read_csv(tables / COMPARISON_CSV)
    return {
        "available": bool(metrics),
        "metrics": _metric_groups(metrics),
        "accuracy": _accuracy_table(read_csv(matrix / "runs.csv")),
        "has_chart": (tables / ACCURACY_SVG).is_file(),
        "env_order": list(ENV_ORDER),
        "conditions": [{"attack": a, "aggregator": g} for a, g in CONDITIONS],
        "names": {"metric": METRIC_NAMES, "attack": ATTACK_NAMES, "aggregator": AGG_NAMES},
    }


def comparison_chart(experiments_dir: Path) -> Path:
    path = experiments_dir / MATRIX_DIR / TABLES_DIR / ACCURACY_SVG
    if not path.is_file():
        raise NotFound("비교 그래프가 없습니다. scripts/compare_matrix.py를 먼저 실행하세요.")
    return path
