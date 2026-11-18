"""시드 고정, 기기 선택, 시간 측정, 실행 환경 기록."""

import platform
import random
import subprocess
import time

import torch

# 라운드·클라이언트마다 다른 셔플 순서를 쓰되, 시드와 번호만으로 다시 만들 수 있게 한다.
ROUND_SEED_STRIDE = 10_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def client_generator(seed: int, round_idx: int, client_id: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * ROUND_SEED_STRIDE ** 2 + round_idx * ROUND_SEED_STRIDE + client_id)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """GPU 연산은 비동기라서, 시간을 재기 전에 끝날 때까지 기다린다."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_now(device: torch.device) -> float:
    synchronize(device)
    return time.perf_counter()


def git_commit() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout != ""
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def environment_info(device: torch.device) -> dict:
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    return {
        "device": device.type,
        "gpu": gpu,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        **git_commit(),
    }
