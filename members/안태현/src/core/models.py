import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCifar(nn.Module):
    """연합학습 기본 분류 모델 (CIFAR-10, 3x32x32)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNetDLG(nn.Module):
    """mit-han-lab/dlg 기반 모델 역추적 실험 전용 모델 (Sigmoid + Strided Conv)"""

    def __init__(self, num_classes=10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out