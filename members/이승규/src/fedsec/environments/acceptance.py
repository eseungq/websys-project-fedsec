"""라운드 채택 규칙: 채택 수, 악성 비율, 모델 갱신 여부를 정한다.

명세: 전체 악성 비율과 실제 채택 악성 비율을 구분하고, 채택 수가 0이면 비율은 N/A,
채택이 3개 미만이면 모델을 갱신하지 않고 사유를 기록한다.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

NOT_APPLICABLE = "N/A"
LATE = "late"
TOO_FEW_ACCEPTED = "too_few_accepted"


@dataclass(frozen=True)
class RoundDecision:
    selected: tuple[int, ...]
    accepted: tuple[int, ...]
    excluded: tuple[tuple[int, str], ...]   # (클라이언트 ID, 제외 사유)
    malicious_selected: int
    malicious_accepted: int
    overall_malicious_ratio: float          # 악성 수 / 전체 클라이언트 수
    accepted_malicious_ratio: float | None  # 채택 중 악성 비율, 채택 0개면 None
    update_model: bool
    skip_reason: str | None                 # 모델을 갱신하지 않은 사유

    @property
    def num_late(self) -> int:
        return sum(1 for _, reason in self.excluded if reason == LATE)

    def to_log(self) -> dict:
        """로그용 라운드 요약. 비율이 없으면 N/A로 적는다."""
        return {
            "num_selected": len(self.selected),
            "num_accepted": len(self.accepted),
            "num_late": self.num_late,
            "malicious_selected": self.malicious_selected,
            "malicious_accepted": self.malicious_accepted,
            "overall_malicious_ratio": round(self.overall_malicious_ratio, 4),
            "accepted_malicious_ratio": ratio_text(self.accepted_malicious_ratio),
            "model_updated": self.update_model,
            "skip_reason": self.skip_reason,
        }


def ratio_text(ratio: float | None) -> float | str:
    return NOT_APPLICABLE if ratio is None else round(ratio, 4)


def judge_round(selected: Sequence[int], excluded: Mapping[int, str], malicious: Collection[int],
                num_clients: int, min_accepted: int = 3) -> RoundDecision:
    """선택된 클라이언트 중 제외되지 않은 것을 채택하고, 채택 수가 min_accepted 미만이면 갱신하지 않는다.

    malicious는 로그용 정답이다. 집계·방어 함수에는 넘기지 않는다.
    """
    if num_clients <= 0 or min_accepted < 0:
        raise ValueError("num_clients는 양수, min_accepted는 0 이상이어야 합니다.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"선택된 클라이언트 ID가 중복되었습니다: {list(selected)}")
    unknown = set(excluded) - set(selected)
    if unknown:
        raise ValueError(f"선택되지 않은 클라이언트가 제외 목록에 있습니다: {sorted(unknown)}")

    accepted = tuple(cid for cid in selected if cid not in excluded)
    malicious_accepted = sum(1 for cid in accepted if cid in malicious)
    update_model = len(accepted) >= min_accepted
    return RoundDecision(
        selected=tuple(selected),
        accepted=accepted,
        excluded=tuple(sorted(excluded.items())),
        malicious_selected=sum(1 for cid in selected if cid in malicious),
        malicious_accepted=malicious_accepted,
        overall_malicious_ratio=len(set(malicious)) / num_clients,
        accepted_malicious_ratio=malicious_accepted / len(accepted) if accepted else None,
        update_model=update_model,
        skip_reason=None if update_model else TOO_FEW_ACCEPTED,
    )
