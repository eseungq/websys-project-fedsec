"""실험 설정 파일(YAML)을 읽고 값이 올바른지 확인한다."""

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


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    train = dict(raw["train"])
    optimizer = train.pop("optimizer", "sgd")
    if optimizer != "sgd":
        raise ValueError(f"아직 지원하지 않는 옵티마이저입니다: {optimizer}")
    try:
        cfg = ExperimentConfig(
            seed=int(raw["seed"]), split=str(raw["split"]), data_root=str(raw["data_root"]),
            rounds=int(raw["rounds"]), model=str(raw["model"]),
            train=TrainConfig(**train), eval_bs=int(raw["eval_bs"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"설정 파일 {path}의 항목이 잘못되었습니다: {error}") from error
    return _validated(cfg)
