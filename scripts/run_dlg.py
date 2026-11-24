"""모델 역추적(DLG) 소규모 실험: 이미지 목록 × 노이즈 σ 목록 × 반복 횟수 목록을 CPU에서 실행한다.

연합학습 본 실험과 분리된 실험이다. 같은 이미지에서 노이즈·반복 횟수만 바꿔 복원 MSE를 비교한다.
원본 이미지와 정답 레이블은 피해자 그래디언트 계산과 평가(MSE, 레이블 일치)에만 쓰고 복원 함수에는 주지 않는다.

예) python scripts/run_dlg.py --name sweep_sigma
    python scripts/run_dlg.py --name it1000 --sigmas 0.01 --iterations 100 300 1000
결과: experiments/dlg/<name>/ 에 config.yaml, env.json, results.csv, trajectory.csv,
      summary.png, dlg_images/(복원 PNG, git 무시)
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from run_clients import PARTIAL_SUFFIX, new_out_dir
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from fedsec.attacks.dlg import Reconstruction, add_gaussian_noise, pixel_mse, reconstruct, victim_gradients
from fedsec.data.split import load_split
from fedsec.experiment.config import whole_number
from fedsec.experiment.runtime import environment_info
from fedsec.models.lenet import build_dlg_lenet

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
DEVICE = torch.device("cpu")  # 실험 규칙: DLG는 CPU만 쓴다 (MPS는 다른 워커가 사용)
TILE = 64                     # summary.png 한 칸 크기 (32px 이미지를 2배로)
CAPTION = 14                  # summary.png 글자 줄 높이


@dataclass(frozen=True)
class DLGConfig:
    split: str
    data_root: str
    client: int
    images: tuple[int, ...]
    model_seed: int
    dummy_seed: int
    noise_seed: int
    sigmas: tuple[float, ...]
    iterations: tuple[int, ...]
    record_every: int


def _validated(cfg: DLGConfig) -> DLGConfig:
    checks = [
        (len(cfg.images) > 0 and len(set(cfg.images)) == len(cfg.images), "images는 겹치지 않는 1개 이상이어야 합니다."),
        (cfg.client >= 0, "client는 0 이상이어야 합니다."),
        (len(cfg.sigmas) > 0 and len(set(cfg.sigmas)) == len(cfg.sigmas)
         and all(math.isfinite(s) and s >= 0 for s in cfg.sigmas), "sigmas는 겹치지 않는 0 이상 유한한 값 1개 이상이어야 합니다."),
        (len(cfg.iterations) > 0 and len(set(cfg.iterations)) == len(cfg.iterations) and all(n > 0 for n in cfg.iterations),
         "iterations는 겹치지 않는 1 이상 값 1개 이상이어야 합니다."),
        (cfg.record_every > 0, "record_every는 1 이상이어야 합니다."),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(f"설정 오류: {message}")
    return cfg


def load_dlg_config(path: Path) -> DLGConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        cfg = DLGConfig(
            split=str(raw["split"]), data_root=str(raw["data_root"]), client=whole_number(raw["client"]),
            images=tuple(whole_number(i) for i in raw["images"]),
            model_seed=whole_number(raw["model_seed"]), dummy_seed=whole_number(raw["dummy_seed"]),
            noise_seed=whole_number(raw["noise_seed"]),
            sigmas=tuple(float(s) for s in raw["sigmas"]),  # YAML 1.1은 1e-4를 문자열로 읽는다
            iterations=tuple(whole_number(n) for n in raw["iterations"]),
            record_every=whole_number(raw["record_every"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"설정 파일 {path}의 항목이 잘못되었습니다: {error}") from error
    return _validated(cfg)


def load_images(cfg: DLGConfig) -> tuple[list[tuple[int, int, torch.Tensor]], list[str]]:
    """(학습 이미지 인덱스, 정답 클래스, [0, 1] 텐서 (1, 3, 32, 32)) 목록과 클래스 이름을 돌려준다."""
    clients = load_split(REPO_ROOT / cfg.split)["clients"]
    if cfg.client >= len(clients):
        raise ValueError(f"client {cfg.client}가 없습니다 (클라이언트 {len(clients)}명)")
    client_indices = set(clients[cfg.client])
    outside = [i for i in cfg.images if i not in client_indices]
    if outside:
        raise ValueError(f"클라이언트 {cfg.client}의 데이터가 아닌 이미지 인덱스입니다: {outside}")
    try:
        dataset = CIFAR10(root=str(REPO_ROOT / cfg.data_root), train=True, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(f"CIFAR-10 데이터가 없습니다: {REPO_ROOT / cfg.data_root} (내려받지 않습니다)") from error
    # 원 논문 코드처럼 정규화 없이 픽셀 [0, 1] 그대로 쓴다 (MSE도 이 범위 기준)
    images = [(i, int(dataset.targets[i]),
               torch.from_numpy(dataset.data[i]).permute(2, 0, 1).float().div(255.0).unsqueeze(0))
              for i in cfg.images]
    return images, list(dataset.classes)


def squared_norm(tensors: list[torch.Tensor]) -> float:
    return float(sum((t.double() ** 2).sum() for t in tensors))


def run_case(model, clean_grads, original, label, sigma, iterations, cfg) -> tuple[Reconstruction, dict]:
    """한 이미지·σ·반복 횟수 조합을 복원하고, 원본은 평가에만 써서 결과 행을 만든다."""
    noisy = add_gaussian_noise(clean_grads, sigma, cfg.noise_seed)
    start = time.perf_counter()
    result = reconstruct(model, noisy, tuple(original.shape), iterations, cfg.dummy_seed, cfg.record_every)
    seconds = time.perf_counter() - start
    pred = int(result.label_logits.argmax(dim=-1).item())
    noise_sq_norm = squared_norm([n - c for n, c in zip(noisy, clean_grads)])
    row = {
        "iterations": iterations, "iterations_done": result.iterations, "diverged": result.diverged,
        "mse": pixel_mse(result.image, original), "mse_unclipped": pixel_mse(result.image, original, clip=False),
        # 1보다 훨씬 크면 더미 픽셀이 폭주해 시그모이드가 포화된 것 (최적화 정체)
        "max_abs_pixel": result.image.abs().max().item(),
        "grad_diff": result.grad_diff, "noise_sq_norm": noise_sq_norm,
        # 노이즈 바닥 대비 그래디언트 차이. 1 근처면 노이즈만 남기고 수렴, 훨씬 크면 최적화가 덜 됨
        "grad_diff_over_noise": result.grad_diff / noise_sq_norm if noise_sq_norm > 0 else "",
        "pred_label": pred, "label_correct": pred == label, "seconds": round(seconds, 2),
    }
    return result, row


def trajectory_rows(result: Reconstruction, original: torch.Tensor, base: dict) -> list[dict]:
    return [{**base, "iteration": s.iteration, "grad_diff": s.grad_diff,
             "mse": pixel_mse(s.image, original), "mse_unclipped": pixel_mse(s.image, original, clip=False)}
            for s in result.trajectory]


def image_name(index: int, sigma: float, iterations: int) -> str:
    return f"img{index}_sigma{sigma:g}_it{iterations}.png"


def run_all(cfg: DLGConfig, out_dir: Path) -> tuple[list[dict], list[dict]]:
    images, classes = load_images(cfg)
    model = build_dlg_lenet(cfg.model_seed)
    image_dir = out_dir / "dlg_images"
    image_dir.mkdir(parents=True)
    results, trajectory = [], []
    for index, label, original in images:
        clean_grads = victim_gradients(model, original, torch.tensor([label]))
        save_image(original, image_dir / f"img{index}_original.png")
        for iterations in cfg.iterations:
            for sigma in cfg.sigmas:
                base = {"image_index": index, "label": label, "class": classes[label], "sigma": sigma}
                result, row = run_case(model, clean_grads, original, label, sigma, iterations, cfg)
                save_image(result.image.clamp(0.0, 1.0), image_dir / image_name(index, sigma, iterations))
                results.append({**base, "grad_sq_norm": squared_norm(clean_grads), **row})
                trajectory.extend(trajectory_rows(result, original, {**base, "iterations": iterations}))
                print(f"이미지 {index}({classes[label]}) σ={sigma:g} {iterations}회: "
                      f"MSE {row['mse']:.4f}, 그래디언트 차이 {row['grad_diff']:.3g}, {row['seconds']}초", flush=True)
    return results, trajectory


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_png(cfg: DLGConfig, results: list[dict], out_dir: Path) -> None:
    """이미지별 한 줄: 원본 | σ별 복원(가장 많은 반복 횟수). 칸 아래에 MSE."""
    iterations = max(cfg.iterations)
    headers = ["original", *[f"s={s:g}" for s in cfg.sigmas]]
    mse = {(r["image_index"], r["sigma"]): r["mse"] for r in results if r["iterations"] == iterations}
    row_height = TILE + CAPTION
    canvas = Image.new("RGB", (TILE * len(headers), CAPTION + row_height * len(cfg.images)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, header in enumerate(headers):
        draw.text((col * TILE + 2, 1), header, fill="black")
    for row, index in enumerate(cfg.images):
        top = CAPTION + row * row_height
        names = [f"img{index}_original.png", *[image_name(index, s, iterations) for s in cfg.sigmas]]
        for col, name in enumerate(names):
            with Image.open(out_dir / "dlg_images" / name) as tile:
                canvas.paste(tile.convert("RGB").resize((TILE, TILE), Image.NEAREST), (col * TILE, top))
            caption = f"#{index}" if col == 0 else f"{mse[(index, cfg.sigmas[col - 1])]:.4f}"
            draw.text((col * TILE + 2, top + TILE + 1), caption, fill="black")
    canvas.save(out_dir / "summary.png", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="결과 폴더 이름 (experiments/dlg/<name>)")
    parser.add_argument("--config", default="configs/dlg.yaml", help="레포 루트 기준 상대 경로 또는 절대 경로")
    parser.add_argument("--sigmas", type=float, nargs="+", help="설정의 σ 목록 대신 사용")
    parser.add_argument("--iterations", type=int, nargs="+", help="설정의 반복 횟수 목록 대신 사용")
    parser.add_argument("--images", type=int, nargs="+", help="설정의 이미지 인덱스 목록 대신 사용")
    parser.add_argument("--dummy-seed", type=int, dest="dummy_seed", help="복원 시작 더미 초기값 시드만 바꿀 때")
    args = parser.parse_args()

    out_dir = new_out_dir(args.name, base=ROOT / "experiments" / "dlg")
    cfg = load_dlg_config(ROOT / args.config)
    overrides = {k: tuple(getattr(args, k)) for k in ("sigmas", "iterations", "images") if getattr(args, k) is not None}
    if args.dummy_seed is not None:
        overrides["dummy_seed"] = args.dummy_seed
    cfg = _validated(replace(cfg, **overrides))

    # 임시 폴더에 다 쓴 뒤 이름을 바꾼다 (중간에 실패하면 반쪽 결과 폴더가 같은 --name을 막지 않게)
    final_dir, out_dir = out_dir, out_dir.with_name(out_dir.name + PARTIAL_SUFFIX)
    if out_dir.exists():
        sys.exit(f"이전에 실패한 임시 폴더가 있습니다. 확인 후 지우세요: {out_dir.relative_to(ROOT)}")
    results, trajectory = run_all(cfg, out_dir)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump({k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
                       allow_unicode=True, sort_keys=False), encoding="utf-8")
    env = {**environment_info(DEVICE), "torch_threads": torch.get_num_threads()}
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(results, out_dir / "results.csv")
    write_csv(trajectory, out_dir / "trajectory.csv")
    write_summary_png(cfg, results, out_dir)
    out_dir.rename(final_dir)
    print(f"결과: {final_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
