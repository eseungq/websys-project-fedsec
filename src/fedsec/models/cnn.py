"""학습용 작은 CNN. BatchNorm을 쓰지 않는다.

BatchNorm의 running mean/var는 학습 파라미터가 아니라서 업데이트·클리핑·중앙값을
어떻게 적용할지 따로 정해야 한다. 기준 모델에서는 이 문제를 피한다.
"""

import torch
from torch import nn

NUM_CLASSES = 10


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 → 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 → 8
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8 → 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODELS = {"simple_cnn": SimpleCNN}


def build_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"알 수 없는 모델 '{name}'. 사용 가능: {sorted(MODELS)}")
    return MODELS[name]()
