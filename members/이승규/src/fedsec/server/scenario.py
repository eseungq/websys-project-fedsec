"""환경 설정과 방어 설정으로 서버 운영 규칙(ServerPolicy)을 만든다.

서버는 참여자 일정·가상 마감·집계 방식만 알고, 악성·지연 클라이언트가 누구인지는 모른다.
집계 함수는 기록(라운드 기록의 aggregator)에 이름이 남도록 이름 있는 함수로 만든다.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fedsec.aggregation.fedavg import weighted_average
from fedsec.defenses.clipping import clip_scale, clip_update, clipped_fedavg
from fedsec.defenses.median import coordinate_median
from fedsec.environments.config import EnvConfig
from fedsec.environments.schedule import round_participants
from fedsec.experiment.config import AttackDefenseConfig
from fedsec.server.serialization import State
from fedsec.server.state import Aggregator, ServerPolicy


@dataclass(frozen=True)
class Defense:
    aggregator: Aggregator
    update_scale: Callable[[State], float] | None  # 업데이트별 클리핑 배율, None이면 업데이트를 그대로 씀


def fedavg(updates: Sequence[State], weights: Sequence[float]) -> State:
    return weighted_average(updates, weights)


def median(updates: Sequence[State], weights: Sequence[float]) -> State:
    """좌표별 중앙값. 표본 수 가중은 쓰지 않는다."""
    return coordinate_median(updates)


def _clipping(max_norm: float) -> tuple[Aggregator, Aggregator, Callable[[State], float]]:
    def clip_fedavg(updates: Sequence[State], weights: Sequence[float]) -> State:
        return clipped_fedavg(updates, weights, max_norm)

    def clip_median(updates: Sequence[State], weights: Sequence[float]) -> State:
        return coordinate_median([clip_update(u, max_norm) for u in updates])

    def clip_factor(update: State) -> float:
        return clip_scale(update, max_norm)

    return clip_fedavg, clip_median, clip_factor


def build_defense(cfg: AttackDefenseConfig) -> Defense:
    clip_fedavg, clip_median, clip_factor = _clipping(cfg.clip_norm)
    if cfg.aggregator == "fedavg":
        return Defense(fedavg, None)
    if cfg.aggregator == "clip_fedavg":
        return Defense(clip_fedavg, clip_factor)
    if cfg.aggregator == "median":
        return Defense(clip_median, clip_factor) if cfg.clip_before_median else Defense(median, None)
    raise ValueError(f"알 수 없는 집계 방식입니다: {cfg.aggregator}")


def scenario_policy(env: EnvConfig, defense: Defense, seed: int, num_clients: int) -> ServerPolicy:
    """참여자는 (시드, 라운드)로 정해지므로 한 라운드에 여러 번 불려도 같은 값이 나온다."""
    def participants(round_idx: int) -> tuple[int, ...]:
        return round_participants(seed, round_idx, num_clients, env.per_round)

    def deadline(round_idx: int) -> float:
        return env.timing.deadline

    return ServerPolicy(aggregator=defense.aggregator, min_accepted=env.min_accepted,
                        participants=participants, deadline=deadline, update_scale=defense.update_scale)
