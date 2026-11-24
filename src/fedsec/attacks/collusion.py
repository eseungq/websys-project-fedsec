"""내부자 협력 공격. 같은 라운드에 참여한 악성 클라이언트들이 공모해 같은 조작 벡터를 제출한다.

가상 클라이언트 실행 프로그램 쪽에서 계산한다. 정상 업데이트를 모두 계산한 뒤 이 함수로
악성 클라이언트의 제출 벡터만 바꿔 끼운다.
"""

from collections.abc import Collection, Mapping

import torch

State = dict[str, torch.Tensor]

DEFAULT_SCALE = -5.0


def collusion_vectors(normal_updates: Mapping[int, State], malicious_ids: Collection[int],
                      scale: float = DEFAULT_SCALE) -> dict[int, State]:
    """참여한 악성 클라이언트에게 제출할 벡터를 돌려준다 (client_id → State).

    normal_updates는 이번 라운드에 참여한 클라이언트들의 정상 업데이트다. 그중 악성
    클라이언트들의 업데이트를 **단순 평균**(표본 수 가중 아님)하고 scale을 곱한 같은 값을
    각 악성 클라이언트에게 준다. 악성 참여자가 1명이면 자기 업데이트 × scale이 된다.

    정상 클라이언트와 참여하지 않은 악성 클라이언트는 결과에 넣지 않는다. 참여한 악성이
    0명이면 빈 dict를 돌려준다(그 라운드는 공격 없음). 입력 텐서는 바꾸지 않는다.
    """
    attackers = sorted(cid for cid in normal_updates if cid in malicious_ids)
    if not attackers:
        return {}
    chosen = [normal_updates[cid] for cid in attackers]
    keys = chosen[0].keys()
    if any(u.keys() != keys for u in chosen):
        raise ValueError("악성 클라이언트 업데이트들의 파라미터 이름이 서로 다릅니다.")

    count = len(chosen)
    shared = {key: sum(u[key] for u in chosen) * (scale / count) for key in keys}
    # 같은 값이지만 클라이언트마다 따로 텐서를 만들어, 한쪽을 바꿔도 다른 쪽에 번지지 않게 한다.
    return {cid: {key: value.clone() for key, value in shared.items()} for cid in attackers}
