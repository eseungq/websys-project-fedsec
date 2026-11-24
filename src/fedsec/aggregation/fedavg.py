"""FedAvg 집계. 업데이트(로컬 가중치 − 시작 전역 가중치) 목록을 받아 새 텐서를 돌려주는 순수 함수.

구성 출처: McMahan et al., AISTATS 2017. 공개 코드 https://github.com/AshwinRJ/Federated-Learning-PyTorch
(`src/utils.py`의 `average_weights`, `src/federated_main.py`). 코드를 가져오지 않고 명세에 맞춰 새로 작성했다.
달라진 점 (자세한 표는 docs/stages/2_서버-API.md):
- 원 코드는 로컬 가중치 자체를 단순 평균한다. 여기서는 업데이트(차이)를 표본 수에 비례해 평균하고 전역 가중치에 더한다
- 입력 텐서를 바꾸지 않고 새 상태를 돌려준다 (원 코드는 첫 클라이언트 가중치 복사본에 더해 나간다)
"""

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
