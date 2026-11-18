"""업데이트 노름. 클리핑 상한을 정상 실험의 노름 분포로 정하기 위해 기록에도 쓴다."""

import torch

State = dict[str, torch.Tensor]


def update_norm(update: State) -> float:
    """모든 파라미터를 한 벡터로 이어 붙였을 때의 L2 노름."""
    if len(update) == 0:
        raise ValueError("빈 업데이트의 노름은 정의되지 않습니다.")
    # MPS는 float64를 지원하지 않으므로 텐서별 합은 float32로 구하고, 텐서 간 합만 Python float로 더한다.
    squared = sum(tensor.float().pow(2).sum().item() for tensor in update.values())
    return squared ** 0.5
