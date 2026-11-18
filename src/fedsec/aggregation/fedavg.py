"""FedAvg 집계. 업데이트(로컬 가중치 − 시작 전역 가중치) 목록을 받아 새 텐서를 돌려주는 순수 함수."""

from collections.abc import Sequence

import torch

State = dict[str, torch.Tensor]


def weighted_average(updates: Sequence[State], weights: Sequence[float]) -> State:
    """weights(표본 수)에 비례한 가중 평균. 입력 텐서는 바꾸지 않는다."""
    if len(updates) == 0:
        raise ValueError("집계할 업데이트가 없습니다.")
    if len(updates) != len(weights):
        raise ValueError(f"업데이트 {len(updates)}개와 가중치 {len(weights)}개의 개수가 다릅니다.")
    if any(w <= 0 for w in weights):
        raise ValueError("가중치(표본 수)는 모두 양수여야 합니다.")
    keys = updates[0].keys()
    if any(u.keys() != keys for u in updates):
        raise ValueError("업데이트들의 파라미터 이름이 서로 다릅니다.")

    total = float(sum(weights))
    return {
        key: sum(u[key] * (w / total) for u, w in zip(updates, weights))
        for key in keys
    }


def apply_update(state: State, update: State) -> State:
    """전역 가중치에 집계된 업데이트를 더한 새 상태를 만든다."""
    if state.keys() != update.keys():
        raise ValueError("전역 모델과 업데이트의 파라미터 이름이 다릅니다.")
    return {key: state[key] + update[key] for key in state}
