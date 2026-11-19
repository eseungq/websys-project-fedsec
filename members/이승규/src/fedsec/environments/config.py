"""운영 환경 설정 파일(configs/env/*.yaml)을 읽는다. 학습 설정은 configs/base.yaml에 따로 있다."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from fedsec.environments.virtual_time import VirtualTiming
from fedsec.experiment.config import whole_number


@dataclass(frozen=True)
class EnvConfig:
    name: str
    split: str                            # 레포 루트 기준 분할 파일 경로
    per_round: int | None                 # 라운드 참여자 수, None이면 전원
    num_malicious: int
    malicious_ids: tuple[int, ...] | None  # 직접 지정하면 시드 선택 대신 사용
    num_delayed: int
    delayed_ids: tuple[int, ...] | None
    delayed_avoid_malicious: bool         # True면 지연 ID를 악성이 아닌 클라이언트 중에서 뽑음
    timing: VirtualTiming
    min_accepted: int


# 설정 파일에 있어야 하는 항목. 오타(min_acepted 등)나 빠진 항목이 기본값으로 조용히 바뀌지 않게 모두 요구한다
ENV_KEYS = frozenset({"name", "split", "per_round", "malicious", "delayed", "virtual_time", "min_accepted"})
ROLE_KEYS = {"malicious": frozenset({"count", "ids"}), "delayed": frozenset({"count", "ids", "avoid_malicious"})}


def _role(raw: dict, key: str) -> tuple[int, tuple[int, ...] | None]:
    """{count, ids} 항목. ids를 주면 count는 그 길이와 같아야 한다."""
    role = raw[key]
    if not isinstance(role, dict) or "count" not in role or set(role) - ROLE_KEYS[key]:
        raise ValueError(f"{key}에는 count가 있어야 하고 {sorted(ROLE_KEYS[key])} 외의 항목은 쓸 수 없습니다: {role!r}")
    count = whole_number(role["count"])
    ids = role.get("ids")
    if ids is None:
        return count, None
    fixed = tuple(whole_number(cid) for cid in ids)
    if len(fixed) != count:
        raise ValueError(f"{key}.ids 개수({len(fixed)})가 {key}.count({count})와 다릅니다.")
    return count, fixed


def load_env_config(path: Path) -> EnvConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != ENV_KEYS:
        found = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(f"환경 설정 파일 {path}: 빠진 항목 {sorted(ENV_KEYS - found)}, 모르는 항목 {sorted(found - ENV_KEYS)}")
    try:
        num_malicious, malicious_ids = _role(raw, "malicious")
        num_delayed, delayed_ids = _role(raw, "delayed")
        avoid_malicious = raw["delayed"].get("avoid_malicious", True)
        if not isinstance(avoid_malicious, bool):
            raise ValueError(f"delayed.avoid_malicious는 true/false여야 합니다: {avoid_malicious!r}")
        timing = raw["virtual_time"]
        per_round = raw.get("per_round")
        cfg = EnvConfig(
            name=str(raw["name"]), split=str(raw["split"]),
            per_round=None if per_round is None else whole_number(per_round),
            num_malicious=num_malicious, malicious_ids=malicious_ids,
            num_delayed=num_delayed, delayed_ids=delayed_ids, delayed_avoid_malicious=avoid_malicious,
            timing=VirtualTiming(
                base_time=float(timing["base_time"]), base_samples=whole_number(timing["base_samples"]),
                delay=float(timing["delay"]), deadline=float(timing["deadline"]),
            ),
            min_accepted=whole_number(raw["min_accepted"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"환경 설정 파일 {path}의 항목이 잘못되었습니다: {error}") from error
    if cfg.per_round is not None and cfg.per_round <= 0:
        raise ValueError(f"환경 설정 오류: per_round는 1 이상이어야 합니다 ({path})")
    if cfg.num_malicious < 0 or cfg.num_delayed < 0:
        raise ValueError(f"환경 설정 오류: count는 0 이상이어야 합니다 ({path})")
    if cfg.min_accepted < 1:
        raise ValueError(f"환경 설정 오류: min_accepted는 1 이상이어야 합니다 ({path})")
    return cfg
