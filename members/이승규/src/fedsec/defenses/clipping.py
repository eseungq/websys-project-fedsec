"""업데이트 노름 클리핑 후 FedAvg. 서버 쪽 집계 방어이며 악성 ID는 받지 않는다."""

import math
from collections.abc import Sequence

import torch

from fedsec.aggregation.fedavg import weighted_average
from fedsec.aggregation.norms import update_norm

State = dict[str, torch.Tensor]


def clip_scale(update: State, max_norm: float) -> float:
    """업데이트에 곱할 배율 min(1, max_norm / ||u||)를 돌려준다.

    노름은 update_norm과 같은 정의(모든 파라미터를 이은 L2)다. 기록용으로 쓸 수 있다:
    클리핑 전 노름 = update_norm(u), 클리핑 후 노름 = 전 노름 × 배율, 클리핑 여부 = 배율 < 1.
    """
    if not max_norm > 0:
        raise ValueError(f"클리핑 상한은 양수여야 합니다: {max_norm}")
    norm = update_norm(update)
    if not math.isfinite(norm):
        raise ValueError("노름이 유한하지 않은 업데이트는 클리핑할 수 없습니다.")
    if norm <= max_norm:
        return 1.0
    return max_norm / norm


def clip_update(update: State, max_norm: float) -> State:
    """노름이 max_norm을 넘으면 방향은 그대로 두고 노름이 max_norm이 되도록 줄인 새 업데이트."""
    scale = clip_scale(update, max_norm)
    return {key: tensor * scale for key, tensor in update.items()}


def clipped_fedavg(updates: Sequence[State], weights: Sequence[float], max_norm: float) -> State:
    """각 업데이트를 클리핑한 뒤 weights(표본 수)로 가중 평균한다. 입력 텐서는 바꾸지 않는다."""
    if len(updates) == 0:
        raise ValueError("집계할 업데이트가 없습니다.")
    return weighted_average([clip_update(u, max_norm) for u in updates], weights)
