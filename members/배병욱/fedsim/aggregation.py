"""집계와 내부자 협력 방어. 모든 함수는 1차원 업데이트 벡터만 다룬다.

방어 함수는 악성 ID 정답을 받지 않는다 (입력은 업데이트와 표본 수뿐).
"""
import torch


def fedavg(updates, num_samples):
    """표본 수에 비례한 가중 평균."""
    w = torch.tensor(num_samples, dtype=torch.float64)
    w = (w / w.sum()).to(torch.float32)
    stacked = torch.stack(updates)
    return (w[:, None] * stacked).sum(0)


def clip(update, bound):
    """L2 노름이 bound를 넘으면 bound로 줄인다. (줄인 벡터, 원래 노름) 반환."""
    norm = update.norm().item()
    if norm > bound > 0:
        return update * (bound / norm), norm
    return update, norm


def coordinate_median(updates):
    """좌표별 중앙값. 짝수 개면 가운데 두 값의 평균."""
    s = torch.stack(updates).sort(dim=0).values
    n = s.shape[0]
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def aggregate(method, updates, num_samples, clip_bound=None):
    """method: fedavg | clip_fedavg | median.

    반환: (집계 벡터, 방어 후 각 업데이트의 노름 목록)
    """
    if method == "fedavg":
        return fedavg(updates, num_samples), [u.norm().item() for u in updates]
    if method == "clip_fedavg":
        if not clip_bound:
            raise ValueError("clip_fedavg에는 clip_bound가 필요합니다")
        clipped = [clip(u, clip_bound)[0] for u in updates]
        return fedavg(clipped, num_samples), [u.norm().item() for u in clipped]
    if method == "median":
        # 중앙값은 개별 업데이트를 바꾸지 않으므로 방어 후 노름은 제출 노름과 같다.
        return coordinate_median(updates), [u.norm().item() for u in updates]
    raise ValueError(f"알 수 없는 집계 방식: {method}")
