"""좌표별 중앙값 집계. 서버 쪽 집계 방어이며 악성 ID와 표본 수는 쓰지 않는다.

클리핑은 하지 않는다. 중앙값 전에 클리핑할지는 호출하는 쪽에서 정한다.
"""

from collections.abc import Sequence

import torch

State = dict[str, torch.Tensor]


def coordinate_median(updates: Sequence[State]) -> State:
    """파라미터의 좌표마다 업데이트들의 중앙값을 구한 새 업데이트.

    개수가 짝수면 가운데 두 값의 평균을 쓴다. torch.median은 짝수 개에서 아래쪽 값을
    돌려주므로 쓰지 않고, 정렬한 뒤 직접 고른다. 입력 텐서는 바꾸지 않는다.
    """
    if len(updates) == 0:
        raise ValueError("집계할 업데이트가 없습니다.")
    keys = updates[0].keys()
    if any(u.keys() != keys for u in updates):
        raise ValueError("업데이트들의 파라미터 이름이 서로 다릅니다.")
    return {key: _median_of([u[key] for u in updates], key) for key in keys}


def _median_of(tensors: list[torch.Tensor], key: str) -> torch.Tensor:
    """같은 모양의 텐서들을 쌓아 0번 축으로 중앙값을 구한다."""
    shape = tensors[0].shape
    if any(t.shape != shape for t in tensors):
        raise ValueError(f"파라미터 {key}의 모양이 업데이트마다 다릅니다.")
    ordered = torch.stack(tensors).sort(dim=0).values
    count = len(tensors)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle].clone()
    return (ordered[middle - 1] + ordered[middle]) / 2
