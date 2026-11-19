from dataclasses import dataclass

import torch
from torch import nn

from fedsec.data.cifar10 import TensorData

NUM_CLASSES = 10


@dataclass(frozen=True)
class Evaluation:
    loss: float                     # 평균 손실
    acc: float                      # 정확도 0~1
    class_correct: tuple[int, ...]  # 클래스별 맞힌 수
    class_total: tuple[int, ...]    # 클래스별 이미지 수


@torch.no_grad()
def evaluate_detailed(model: nn.Module, data: TensorData, batch_size: int,
                      num_classes: int = NUM_CLASSES) -> Evaluation:
    """평균 손실, 정확도와 클래스별 맞힌 수·전체 수. 손실·정확도는 evaluate()와 같은 값이다."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = torch.zeros((), device=data.labels.device)
    correct = torch.zeros((), device=data.labels.device)
    class_correct = torch.zeros(num_classes, dtype=torch.int64, device=data.labels.device)
    for start in range(0, len(data), batch_size):
        images = data.images[start:start + batch_size]
        labels = data.labels[start:start + batch_size]
        logits = model(images)
        loss_sum += loss_fn(logits, labels)
        hits = logits.argmax(dim=1) == labels
        correct += hits.sum()
        class_correct += torch.bincount(labels[hits], minlength=num_classes)
    n = len(data)
    class_total = torch.bincount(data.labels, minlength=num_classes)
    return Evaluation(loss=loss_sum.item() / n, acc=correct.item() / n,
                      class_correct=tuple(class_correct.tolist()), class_total=tuple(class_total.tolist()))


def evaluate(model: nn.Module, data: TensorData, batch_size: int) -> tuple[float, float]:
    """(평균 손실, 정확도 0~1)을 반환한다."""
    result = evaluate_detailed(model, data, batch_size)
    return result.loss, result.acc
