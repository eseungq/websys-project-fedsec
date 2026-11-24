"""악성·지연 클라이언트와 라운드별 참여자를 시드만으로 정한다.

입력은 시드·라운드·클라이언트 수뿐이라서, 같은 시드면 집계 방식이나 공격 여부가 달라도
같은 일정이 나온다. 용도마다 난수 흐름을 따로 써서 한쪽 선택이 다른 쪽에 영향을 주지 않는다.
"""

import random
from collections.abc import Collection, Sequence


def _rng(purpose: str, seed: int, round_idx: int | None = None) -> random.Random:
    """문자열 시드는 SHA-512로 바뀌어 실행·기기와 상관없이 같은 난수열이 나온다."""
    key = f"{purpose}:{seed}" if round_idx is None else f"{purpose}:{seed}:{round_idx}"
    return random.Random(key)


def _checked_ids(ids: Sequence[int], num_clients: int) -> tuple[int, ...]:
    if len(set(ids)) != len(ids):
        raise ValueError(f"클라이언트 ID가 중복되었습니다: {list(ids)}")
    if any(not 0 <= cid < num_clients for cid in ids):
        raise ValueError(f"클라이언트 ID는 0 이상 {num_clients} 미만이어야 합니다: {list(ids)}")
    return tuple(sorted(ids))


def _fixed_clients(purpose: str, seed: int, num_clients: int, count: int,
                   fixed_ids: Sequence[int] | None, exclude: Collection[int] = ()) -> tuple[int, ...]:
    if fixed_ids is not None:
        return _checked_ids(fixed_ids, num_clients)
    candidates = [cid for cid in range(num_clients) if cid not in exclude]
    if not 0 <= count <= len(candidates):
        raise ValueError(f"{purpose} 수는 0 이상 후보 수 {len(candidates)} 이하여야 합니다: {count}")
    return tuple(sorted(_rng(purpose, seed).sample(candidates, count)))


def malicious_clients(seed: int, num_clients: int, count: int,
                      fixed_ids: Sequence[int] | None = None) -> tuple[int, ...]:
    """실험 전체에서 고정되는 악성 ID. fixed_ids를 주면 시드 대신 그 값을 쓴다."""
    return _fixed_clients("malicious", seed, num_clients, count, fixed_ids)


def delayed_clients(seed: int, num_clients: int, count: int,
                    fixed_ids: Sequence[int] | None = None, exclude: Collection[int] = ()) -> tuple[int, ...]:
    """실험 전체에서 고정되는 지연 ID. exclude(보통 악성 ID)를 뺀 클라이언트 중에서 뽑는다.

    악성과 지연이 겹치면 지연 효과와 공격 약화가 섞이므로 기본 설정은 악성 ID를 뺀다.
    fixed_ids를 주면 exclude와 상관없이 그 값을 쓴다.
    """
    return _fixed_clients("delayed", seed, num_clients, count, fixed_ids, exclude)


def round_participants(seed: int, round_idx: int, num_clients: int,
                       per_round: int | None = None) -> tuple[int, ...]:
    """라운드 참여자. per_round가 None이면 전원, 아니면 (시드, 라운드)로 그 수만큼 무작위 선택."""
    if per_round is None:
        return tuple(range(num_clients))
    if not 1 <= per_round <= num_clients:
        raise ValueError(f"라운드 참여자 수는 1 이상 {num_clients} 이하여야 합니다: {per_round}")
    return tuple(sorted(_rng("participants", seed, round_idx).sample(range(num_clients), per_round)))
