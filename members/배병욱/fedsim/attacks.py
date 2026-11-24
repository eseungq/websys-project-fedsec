"""내부자 협력 공격."""
import torch


def collusion(honest_updates, scale=-5.0):
    """같은 라운드에 참여한 악성 클라이언트들의 정상 업데이트 평균 × scale을 각자 제출한다.

    honest_updates: {client_id: 정상 업데이트 벡터}
    - 2명 이상: 평균 × scale 동일 벡터
    - 1명: 자기 업데이트 × scale
    - 0명: 공격 없음 (빈 dict)
    """
    if not honest_updates:
        return {}
    mean = torch.stack(list(honest_updates.values())).mean(0)
    forged = mean * scale
    return {cid: forged.clone() for cid in honest_updates}
