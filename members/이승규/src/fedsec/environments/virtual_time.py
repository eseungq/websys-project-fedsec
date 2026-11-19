"""가상 시간으로 마감 초과를 판정한다. 실제 순차 실행 시간과는 따로 계산한다.

클라이언트가 한 프로그램에서 차례로 학습하므로 실제 시간은 순서에 따라 달라진다.
그래서 채택 여부는 표본 수와 설정 값만으로 정해지는 가상 완료 시각으로 판정한다.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class VirtualTiming:
    base_time: float     # 표본 base_samples개를 학습하는 데 걸리는 가상 시간
    base_samples: int
    delay: float         # 지연 클라이언트에만 더하는 시간
    deadline: float      # 이 시각보다 늦게 끝나면 제외 (사유 late)

    def __post_init__(self) -> None:
        if self.base_time <= 0 or self.base_samples <= 0:
            raise ValueError("base_time과 base_samples는 양수여야 합니다.")
        if self.delay < 0 or self.deadline <= 0:
            raise ValueError("delay는 0 이상, deadline은 양수여야 합니다.")


def virtual_finish_times(sample_counts: Mapping[int, int], delayed: Collection[int],
                         timing: VirtualTiming) -> dict[int, float]:
    """클라이언트별 가상 완료 시각 = 표본 수에 비례한 계산 시간 + 지연 값(지연 클라이언트만)."""
    return {
        cid: timing.base_time * n / timing.base_samples + (timing.delay if cid in delayed else 0.0)
        for cid, n in sample_counts.items()
    }


def late_clients(finish_times: Mapping[int, float], deadline: float) -> tuple[int, ...]:
    """마감보다 늦게 끝난 클라이언트 (마감과 같으면 채택)."""
    return tuple(sorted(cid for cid, t in finish_times.items() if t > deadline))


def round_elapsed(finish_times: Mapping[int, float], deadline: float) -> float:
    """라운드 가상 경과 시간: 늦은 제출이 있으면 마감까지 기다리므로 마감 시각, 없으면 가장 늦은 완료 시각."""
    if not finish_times:
        return 0.0
    if late_clients(finish_times, deadline):
        return deadline
    return max(finish_times.values())


def check_delay_setting(sample_counts: Mapping[int, int], delayed: Collection[int],
                        timing: VirtualTiming) -> None:
    """지연 클라이언트만 항상 마감을 넘는지 확인한다 (명세: 고정된 3개만 마감 초과)."""
    late = set(late_clients(virtual_finish_times(sample_counts, delayed, timing), timing.deadline))
    expected = set(delayed) & set(sample_counts)
    if late != expected:
        raise ValueError(
            f"마감 초과 클라이언트 {sorted(late)}가 지연 클라이언트 {sorted(expected)}와 다릅니다. "
            "base_time, delay, deadline 값을 확인하세요."
        )
