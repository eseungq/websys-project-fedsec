"""실험 설정 파일(YAML)을 읽고 값이 올바른지 확인한다."""

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from fedsec.client.local_train import TrainConfig


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    split: str
    data_root: str
    rounds: int
    model: str
    train: TrainConfig
    eval_bs: int

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, **train_overrides) -> "ExperimentConfig":
        """None이 아닌 값만 바꾼 새 설정을 만든다. 키는 seed, rounds 또는 TrainConfig 필드."""
        top = {k: v for k, v in train_overrides.items() if k in ("seed", "rounds") and v is not None}
        train = {k: v for k, v in train_overrides.items() if k not in top and v is not None}
        return _validated(replace(self, **top, train=replace(self.train, **train)))


def _validated(cfg: ExperimentConfig) -> ExperimentConfig:
    t = cfg.train
    checks = [
        (cfg.rounds > 0, "rounds는 1 이상이어야 합니다."),
        (cfg.eval_bs > 0, "eval_bs는 1 이상이어야 합니다."),
        (all(math.isfinite(v) for v in (t.lr, t.momentum, t.weight_decay)), "lr, momentum, weight_decay는 유한한 숫자여야 합니다."),
        (t.lr > 0, "lr은 양수여야 합니다."),
        (0 <= t.momentum < 1, "momentum은 0 이상 1 미만이어야 합니다."),
        (t.weight_decay >= 0, "weight_decay는 0 이상이어야 합니다."),
        (t.local_ep > 0, "local_ep는 1 이상이어야 합니다."),
        (t.local_bs > 0, "local_bs는 1 이상이어야 합니다."),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(f"설정 오류: {message}")
    return cfg


def whole_number(value) -> int:
    """정수 항목: 1, 1.0, "32"는 받고 1.9처럼 소수 부분이 있으면 거부한다 (int()는 조용히 버림). true/false도 거부."""
    if isinstance(value, bool):
        raise ValueError(f"정수여야 합니다: {value!r}")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"정수여야 합니다: {value!r}")
    return int(number)


# YAML 1.1은 5e-4 같은 지수 표기를 문자열로 읽으므로 숫자로 바꾼다
TRAIN_TYPES = {"lr": float, "momentum": float, "weight_decay": float, "local_ep": whole_number, "local_bs": whole_number}


AGGREGATORS = ("fedavg", "clip_fedavg", "median")


@dataclass(frozen=True)
class AttackDefenseConfig:
    """공격·방어 설정 (configs/attack_defense.yaml). 학습 설정(base.yaml)과 따로 둔다."""
    attack: bool               # 악성 클라이언트의 공모 공격 켜기
    attack_scale: float        # 악성 정상 업데이트 평균에 곱하는 값 (명세 -5)
    aggregator: str            # fedavg | clip_fedavg | median
    clip_norm: float           # clip_fedavg(와 clip_before_median)의 노름 상한
    clip_before_median: bool   # 중앙값 전에 클리핑할지

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, **overrides) -> "AttackDefenseConfig":
        """None이 아닌 값만 바꾼 새 설정 (명령줄 덮어쓰기)."""
        return _validated_attack_defense(replace(self, **{k: v for k, v in overrides.items() if v is not None}))


def _validated_attack_defense(cfg: AttackDefenseConfig) -> AttackDefenseConfig:
    if cfg.aggregator not in AGGREGATORS:
        raise ValueError(f"설정 오류: aggregator는 {', '.join(AGGREGATORS)} 중 하나여야 합니다: {cfg.aggregator}")
    if not (math.isfinite(cfg.clip_norm) and cfg.clip_norm > 0):
        raise ValueError(f"설정 오류: clip_norm은 유한한 양수여야 합니다: {cfg.clip_norm}")
    if not (math.isfinite(cfg.attack_scale) and cfg.attack_scale != 0):
        raise ValueError(f"설정 오류: attack.scale은 0이 아닌 유한한 숫자여야 합니다: {cfg.attack_scale}")
    if not isinstance(cfg.attack, bool) or not isinstance(cfg.clip_before_median, bool):
        raise ValueError("설정 오류: attack과 clip_before_median은 true/false여야 합니다.")
    return cfg


def load_attack_defense(path: Path) -> AttackDefenseConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        cfg = AttackDefenseConfig(
            attack=raw["attack"]["enabled"], attack_scale=float(raw["attack"]["scale"]),
            aggregator=str(raw["defense"]["aggregator"]), clip_norm=float(raw["defense"]["clip_norm"]),
            clip_before_median=raw["defense"]["clip_before_median"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"공격·방어 설정 파일 {path}의 항목이 잘못되었습니다: {error}") from error
    return _validated_attack_defense(cfg)


CONFIG_KEYS = frozenset({"seed", "split", "data_root", "rounds", "model", "train", "eval_bs"})


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS or not isinstance(raw["train"], dict):
        found = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(f"설정 파일 {path}: 빠진 항목 {sorted(CONFIG_KEYS - found)}, 모르는 항목 {sorted(found - CONFIG_KEYS)}"
                         " (train은 항목 묶음이어야 함)")
    train = dict(raw["train"])
    optimizer = train.pop("optimizer", "sgd")
    if optimizer != "sgd":
        raise ValueError(f"아직 지원하지 않는 옵티마이저입니다: {optimizer}")
    try:
        cfg = ExperimentConfig(
            seed=whole_number(raw["seed"]), split=str(raw["split"]), data_root=str(raw["data_root"]),
            rounds=whole_number(raw["rounds"]), model=str(raw["model"]),
            train=TrainConfig(**{k: TRAIN_TYPES[k](v) if k in TRAIN_TYPES else v for k, v in train.items()}),
            eval_bs=whole_number(raw["eval_bs"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"설정 파일 {path}의 항목이 잘못되었습니다: {error}") from error
    return _validated(cfg)
