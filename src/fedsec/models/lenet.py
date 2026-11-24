"""모델 역추적(DLG) 실험용 LeNet. 연합학습 본 실험의 SimpleCNN과는 별개다.

구성 출처: Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019) 공개 코드
https://github.com/mit-han-lab/dlg 의 models/vision.py (LeNet, weights_init).
코드는 복사하지 않고 같은 구성으로 새로 작성했다. 달라진 점:
- 출력 클래스 100(CIFAR-100) → 10(CIFAR-10)
- 초기화에 전역 시드 대신 전용 torch.Generator를 써서, 모델 시드만으로 같은 가중치를 다시 만든다

활성화가 시그모이드인 이유: DLG는 그래디언트의 그래디언트(2차 미분)를 따라 내려가므로
ReLU처럼 2차 미분이 0인 활성화에서는 복원이 잘 되지 않는다.
"""

import torch
from torch import nn

NUM_CLASSES = 10
# 원 논문 코드의 weights_init: 가중치·편향을 U(-0.5, 0.5)로. 기본 초기화로 두면 복원이 실패했다.
INIT_RANGE = 0.5


class LeNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=2, stride=2), nn.Sigmoid(),   # 32 → 16
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2), nn.Sigmoid(),  # 16 → 8
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1), nn.Sigmoid(),  # 8 → 8
        )
        self.fc = nn.Linear(12 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.body(x).flatten(1))


def build_dlg_lenet(seed: int) -> LeNet:
    """가중치·편향을 모두 U(-INIT_RANGE, INIT_RANGE)로 채운 새 LeNet을 만든다 (CPU)."""
    generator = torch.Generator().manual_seed(seed)
    model = LeNet()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.rand(param.shape, generator=generator) * 2 * INIT_RANGE - INIT_RANGE)
    return model
