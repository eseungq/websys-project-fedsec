import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def compute_state_diff(local_state, global_state):
    """업데이트 정의: Δw = w_local - w_global (CPU 텐서로 반환)"""
    delta = {}
    for k in global_state.keys():
        delta[k] = (local_state[k].detach().cpu() - global_state[k].detach().cpu()).clone()
    return delta


def compute_l2_norm(state_dict):
    """파라미터 전체를 하나의 벡터로 본 L2 노름 계산"""
    total_sq = 0.0
    for v in state_dict.values():
        if v.is_floating_point():
            total_sq += torch.sum(v.detach().cpu().float() ** 2).item()
    return total_sq ** 0.5


def train_local_client(model, train_dataset, idxs, local_ep, local_bs, lr, momentum, device):
    """단일 클라이언트 로컬 학습 수행 후 업데이트(Δw)와 평균 손실 반환"""
    global_state = copy.deepcopy(model.state_dict())
    model.to(device)
    model.train()

    loader = DataLoader(Subset(train_dataset, idxs), batch_size=local_bs, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    for _ in range(local_ep):
        batch_losses = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / max(len(batch_losses), 1))

    delta_w = compute_state_diff(model.state_dict(), global_state)
    avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
    return delta_w, float(avg_loss)


@torch.no_grad()
def evaluate_model(model, dataset, device, batch_size=128, per_class=False):
    """검증/테스트/클라이언트별 정확도 및 클래스별 정확도 평가"""
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if per_class:
            for c in range(10):
                mask = (labels == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)

    if per_class:
        per_class_acc = {
            c: round(class_correct[c] / max(class_total[c], 1), 4) for c in range(10)
        }
        return float(acc), float(avg_loss), per_class_acc

    return float(acc), float(avg_loss)