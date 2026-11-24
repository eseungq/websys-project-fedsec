"""조회형 DLG 탭: experiments/3_dlg/<실험>/ 의 results.csv, summary.png, dlg_images/*.png (파일을 쓰지 않음).

복원 이미지 이름은 scripts/run_dlg.py의 image_name과 같은 규칙(img{번호}_sigma{σ:g}_it{반복}.png)이다.
dlg_images/는 git에서 빠지므로 없을 수 있다 (그때는 summary.png만 보임).
"""

import re
from pathlib import Path

from web.files import (
    NotFound,
    PathRejected,
    read_csv,
    read_yaml,
    resolve_inside,
)

DLG_DIR = "3_dlg"
IMAGES_DIR = "dlg_images"
SUMMARY_PNG = "summary.png"
RESULT_KEYS = ("image_index", "label", "class", "sigma", "iterations", "iterations_done", "diverged", "mse",
               "grad_diff_over_noise", "pred_label", "label_correct", "seconds")
CONFIG_KEYS = ("client", "images", "sigmas", "iterations", "model_seed", "dummy_seed", "noise_seed")
IMAGE_NAME = re.compile(r"img[0-9]{1,6}_(original|sigma[0-9.e+-]{1,16}_it[0-9]{1,6})\.png")
RUN_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}")   # 점으로 시작하는 이름(..) 거부


def original_name(index: int) -> str:
    return f"img{index}_original.png"


def reconstructed_name(index: int, sigma: float, iterations: int) -> str:
    return f"img{index}_sigma{sigma:g}_it{iterations}.png"


def _existing(images: Path, name: str) -> str | None:
    return name if (images / name).is_file() else None


def _result_row(row: dict, images: Path) -> dict:
    index, sigma, iterations = row.get("image_index"), row.get("sigma"), row.get("iterations")
    known = isinstance(index, int) and isinstance(sigma, (int, float)) and isinstance(iterations, int)
    return {**{key: row.get(key) for key in RESULT_KEYS},
            "original": _existing(images, original_name(index)) if known else None,
            "reconstructed": _existing(images, reconstructed_name(index, float(sigma), iterations)) if known else None}


def _dlg_run(run_dir: Path) -> dict:
    config = read_yaml(run_dir / "config.yaml") or {}
    images = run_dir / IMAGES_DIR
    return {"id": run_dir.name,
            "config": {key: config.get(key) for key in CONFIG_KEYS},
            "has_summary": (run_dir / SUMMARY_PNG).is_file(),
            "results": [_result_row(row, images) for row in read_csv(run_dir / "results.csv")]}


def dlg_runs(experiments_dir: Path) -> list[dict]:
    base = experiments_dir / DLG_DIR
    if not base.is_dir():
        return []
    run_dirs = sorted(d for d in base.iterdir()
                      if d.is_dir() and RUN_NAME.fullmatch(d.name) and not d.name.endswith(".partial")
                      and (d / "results.csv").is_file())
    return [_dlg_run(d) for d in run_dirs]


def dlg_image_path(experiments_dir: Path, run_id: str, filename: str) -> Path:
    """summary.png 또는 dlg_images/의 원본·복원 PNG. 이름 규칙과 폴더 위치를 모두 확인한다."""
    if not RUN_NAME.fullmatch(run_id or ""):
        raise PathRejected("DLG 실험 이름이 올바르지 않습니다.")
    root = experiments_dir.resolve()
    run_dir = resolve_inside(root / DLG_DIR, run_id)
    if filename == SUMMARY_PNG:
        path = run_dir / SUMMARY_PNG
    elif IMAGE_NAME.fullmatch(filename or ""):
        path = run_dir / IMAGES_DIR / filename
    else:
        raise PathRejected("허용하지 않은 이미지 이름입니다.")
    resolved = resolve_inside(root, path.relative_to(root).as_posix())
    if not resolved.is_file():
        raise NotFound(f"이미지가 없습니다: {filename}")
    return resolved
