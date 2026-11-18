import torch
from torch import nn

from fedsec.data.cifar10 import TensorData


@torch.no_grad()
def evaluate(model: nn.Module, data: TensorData, batch_size: int) -> tuple[float, float]:
    """(평균 손실, 정확도 0~1)을 반환한다."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = torch.zeros((), device=data.labels.device)
    correct = torch.zeros((), device=data.labels.device)
    for start in range(0, len(data), batch_size):
        images = data.images[start:start + batch_size]
        labels = data.labels[start:start + batch_size]
        logits = model(images)
        loss_sum += loss_fn(logits, labels)
        correct += (logits.argmax(dim=1) == labels).sum()
    n = len(data)
    return loss_sum.item() / n, correct.item() / n
