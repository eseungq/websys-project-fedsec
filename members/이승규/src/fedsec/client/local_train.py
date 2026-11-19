"""가상 클라이언트의 로컬 학습: 전역 모델에서 시작해 학습한 뒤 업데이트를 계산한다."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from fedsec.data.cifar10 import TensorData

State = dict[str, torch.Tensor]


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    momentum: float
    weight_decay: float
    local_ep: int
    local_bs: int


@dataclass(frozen=True)
class ClientResult:
    update: State      # 로컬 가중치 − 시작 전역 가중치
    num_samples: int
    train_loss: float  # 마지막 local epoch의 평균 손실



class TrainingDiverged(FloatingPointError):
    """학습이 발산했다 (손실·업데이트·전역 모델에 NaN·inf). program.run_clients가 공격 실험의 결과로 처리한다."""


def local_update(model: nn.Module, global_state: State, data: TensorData,
                 cfg: TrainConfig, generator: torch.Generator) -> ClientResult:
    """model 객체는 재사용하되, 매번 global_state를 불러와 시작점을 맞춘다."""
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = len(data)

    epoch_loss = 0.0
    for _ in range(cfg.local_ep):
        order = torch.randperm(n, generator=generator).to(data.labels.device)
        loss_sum = torch.zeros((), device=data.labels.device)
        for start in range(0, n, cfg.local_bs):
            batch = order[start:start + cfg.local_bs]
            optimizer.zero_grad()
            loss = loss_fn(model(data.images[batch]), data.labels[batch])
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(batch)
        epoch_loss = loss_sum.item() / n
    if not math.isfinite(epoch_loss):
        raise TrainingDiverged(f"로컬 학습 손실이 {epoch_loss}입니다 (발산). 학습률 등 설정을 확인하세요.")

    local_state = model.state_dict()
    update = {key: local_state[key].detach() - global_state[key] for key in global_state}
    # 마지막 스텝에서 가중치가 깨지면 손실은 유한해도 업데이트가 비유한값이 된다 (서버가 거부하기 전에 발산으로 처리)
    if not all(torch.isfinite(value).all() for value in update.values()):
        raise TrainingDiverged("로컬 업데이트에 NaN·inf가 있습니다 (발산).")
    return ClientResult(update=update, num_samples=n, train_loss=epoch_loss)
