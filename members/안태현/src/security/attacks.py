import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_insider_collusion(honest_updates, malicious_ids, multiplier=-5.0):
    """
    내부자 협력 공격 적용:
    - 라운드 내 참여 악성 클라이언트 2명 이상: 참여 악성들의 정상 업데이트 평균 * (-5) 동일 벡터 제출
    - 참여 악성 클라이언트 1명: 본인 정상 업데이트 * (-5) 제출
    - 참여 악성 클라이언트 0명: 공격 미적용
    """
    submitted_updates = {cid: copy.deepcopy(upd) for cid, upd in honest_updates.items()}
    active_malicious = [cid for cid in honest_updates.keys() if cid in malicious_ids]

    if len(active_malicious) == 0:
        return submitted_updates, active_malicious

    if len(active_malicious) >= 2:
        avg_mal_update = {}
        keys = honest_updates[active_malicious[0]].keys()
        for k in keys:
            stacked = torch.stack([honest_updates[cid][k].float() for cid in active_malicious], dim=0)
            avg_mal_update[k] = (stacked.mean(dim=0) * multiplier).to(honest_updates[active_malicious[0]][k].dtype)

        for cid in active_malicious:
            submitted_updates[cid] = copy.deepcopy(avg_mal_update)
    else:
        cid = active_malicious[0]
        for k in submitted_updates[cid].keys():
            submitted_updates[cid][k] = submitted_updates[cid][k] * multiplier

    return submitted_updates, active_malicious


def compute_single_image_gradients(model, image_tensor, label_int, device):
    """클라이언트 측: 이미지 1장에 대한 1회 손실 그래디언트 추출"""
    model.to(device)
    model.eval()
    model.zero_grad()

    x = image_tensor.unsqueeze(0).to(device)
    y = torch.tensor([label_int], dtype=torch.long, device=device)

    out = model(x)
    loss = F.cross_entropy(out, y)
    grads = torch.autograd.grad(loss, model.parameters())
    return [g.detach().cpu().clone() for g in grads]


def run_dlg_inversion(model, observed_gradients, num_classes=10, iterations=300, seed=1, device="cpu"):
    """
    mit-han-lab/dlg 기반 모델 역추적 공격:
    원본 이미지와 정답 레이블을 입력받지 않고 오직 observed_gradients만으로 복원 수행.
    """
    torch.manual_seed(seed)
    model = copy.deepcopy(model).to(device)
    model.eval()

    target_dy_dx = [g.to(device) for g in observed_gradients]

    # Sigmoid 매핑을 통해 최적화 중 픽셀이 항상 [0, 1] 구간에 머무르도록 유지
    dummy_latent = (torch.randn((1, 3, 32, 32), device=device) * 0.5).requires_grad_(True)
    dummy_label = torch.zeros((1, num_classes), device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [dummy_latent, dummy_label],
        lr=1.0,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    grad_loss_history = []

    for _ in range(iterations):
        def closure():
            optimizer.zero_grad()
            dummy_data = torch.sigmoid(dummy_latent)
            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = torch.mean(
                torch.sum(-dummy_onehot_label * F.log_softmax(dummy_pred, dim=-1), dim=-1)
            )
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0.0
            for gx, gy in zip(dummy_dy_dx, target_dy_dx):
                grad_diff = grad_diff + ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        loss_val = optimizer.step(closure)
        cur_loss = float(loss_val.item())
        grad_loss_history.append(cur_loss)

        # 그래디언트 오차가 수렴하면 불필요한 추가 반복 조기 종료
        if cur_loss < 1e-11:
            break

    with torch.no_grad():
        reconstructed = torch.sigmoid(dummy_latent).detach().cpu().squeeze(0)
        inferred_label = int(torch.argmax(dummy_label.detach().cpu(), dim=-1).item())

    return reconstructed, inferred_label, grad_loss_history


def evaluate_dlg_mse(gt_image_tensor, reconstructed_tensor):
    """복원 완료 후 평가 단계에서만 원본 이미지와 픽셀 오차(MSE) 계산"""
    gt = gt_image_tensor.detach().cpu().float()
    rec = reconstructed_tensor.detach().cpu().float()
    mse = torch.mean((gt - rec) ** 2).item()
    return float(mse)