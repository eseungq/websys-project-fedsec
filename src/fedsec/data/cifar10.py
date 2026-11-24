"""CIFAR-10을 내려받아 정규화된 텐서로 한 번에 올린다.

이미지가 작아서(학습 50,000장 약 600MB) 통째로 기기 메모리에 올리고 인덱스로 잘라 쓴다.
DataLoader 작업자 설정이 기기마다 달라 생기는 차이와 속도 문제를 피하기 위해서다.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class TensorData:
    images: torch.Tensor  # (N, 3, 32, 32) float32, 정규화됨
    labels: torch.Tensor  # (N,) int64

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, indices: list[int]) -> "TensorData":
        index = torch.as_tensor(indices, device=self.labels.device)
        return TensorData(self.images[index], self.labels[index])


def _to_tensor_data(dataset: CIFAR10, device: torch.device) -> TensorData:
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    labels = torch.as_tensor(dataset.targets, dtype=torch.long)
    return TensorData(images.to(device), labels.to(device))


def load_cifar10(root: Path, device: torch.device) -> tuple[TensorData, TensorData]:
    """(학습 50,000장, 공식 테스트 10,000장)을 반환한다."""
    train = CIFAR10(root=str(root), train=True, download=True)
    test = CIFAR10(root=str(root), train=False, download=True)
    return _to_tensor_data(train, device), _to_tensor_data(test, device)
