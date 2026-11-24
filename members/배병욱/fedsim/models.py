"""학습용 SimpleCNN과 DLG용 LeNet."""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """합성곱 3층 + 완전연결 2층 (BatchNorm 없음, 연합 집계가 가중치만으로 끝나도록)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LeNetDLG(nn.Module):
    """DLG 공개 구현(mit-han-lab/dlg)의 LeNet. 2차 미분이 가능하도록 Sigmoid를 쓴다."""

    def __init__(self, num_classes=10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding=2, stride=2), act(),
            nn.Conv2d(12, 12, 5, padding=2, stride=2), act(),
            nn.Conv2d(12, 12, 5, padding=2, stride=1), act(),
        )
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1))


def init_dlg_weights(model, seed):
    """DLG 원본과 같이 가중치를 U(-0.5, 0.5)로 초기화한다."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.rand(p.shape, generator=g) - 0.5)
    return model


def build_model(name):
    if name == "simple_cnn":
        return SimpleCNN()
    raise ValueError(f"알 수 없는 모델: {name}")


def get_vector(model):
    return torch.cat([v.detach().reshape(-1).float().cpu() for v in model.state_dict().values()])


def set_vector(model, vec):
    sd = model.state_dict()
    new, i = {}, 0
    for k, v in sd.items():
        n = v.numel()
        new[k] = vec[i:i + n].view_as(v).to(device=v.device, dtype=v.dtype)
        i += n
    if i != vec.numel():
        raise ValueError(f"가중치 길이 불일치: 모델 {i}, 벡터 {vec.numel()}")
    model.load_state_dict(new)
