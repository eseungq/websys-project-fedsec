import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def weights_init(m):
    """DLG 공식 초기화: 가중치/편향을 [-0.5, 0.5] 균등분포로 초기화"""
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.uniform_(-0.5, 0.5)


class LeNet(nn.Module):
    """DLG 공식 논문(Zhu et al.) 기반 Sigmoid LeNet (CIFAR-10, 10클래스용)"""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
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

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def add_gaussian_noise_to_grads(gradients, noise_std, seed=42):
    """방어 기법: 전송 전 그래디언트 텐서마다 가우시안 노이즈 N(0, noise_std^2) 추가"""
    if noise_std <= 0.0:
        return [g.detach().clone() for g in gradients]

    gen = torch.Generator(device=gradients[0].device)
    gen.manual_seed(seed)

    noisy_grads = []
    for g in gradients:
        noise = torch.randn(g.size(), generator=gen, device=g.device, dtype=g.dtype) * noise_std
        noisy_grads.append((g + noise).detach().clone())
    return noisy_grads


def run_dlg_reconstruction(net, target_dy_dx, img_shape, num_classes=10, iters=300, seed=1234, device="cpu"):
    """단일 이미지 1회 그래디언트로부터 L-BFGS 최적화를 통해 입력 이미지 복원"""
    torch.manual_seed(seed)
    dummy_data = torch.rand(img_shape, device=device, requires_grad=True)  # randn -> rand 로 변경
    dummy_label = torch.randn((1, num_classes), device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    criterion = cross_entropy_for_onehot

    snapshots = {}
    for it in range(iters):
        def closure():
            optimizer.zero_grad()
            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, target_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if it in [0, 50, 100, 200, iters - 1]:
            snapshots[it] = torch.clamp(dummy_data.detach().clone(), 0.0, 1.0)

    final_reconstructed = torch.clamp(dummy_data.detach().clone(), 0.0, 1.0)
    pred_label = torch.argmax(F.softmax(dummy_label.detach(), dim=-1), dim=-1).item()
    return final_reconstructed, pred_label, snapshots


def main():
    parser = argparse.ArgumentParser(description="DLG 모델 역추적 공격 및 가우시안 노이즈 방어 실험")
    parser.add_argument("--index", type=int, default=25, help="CIFAR-10 테스트 데이터셋 이미지 인덱스")
    parser.add_argument("--iters", type=int, default=300, help="L-BFGS 복원 반복 횟수")
    parser.add_argument("--noise_none", type=float, default=0.0, help="노이즈 없음 표준편차")
    parser.add_argument("--noise_small", type=float, default=0.001, help="노이즈 작음 표준편차")
    parser.add_argument("--noise_large", type=float, default=0.01, help="노이즈 큼 표준편차")
    parser.add_argument("--seed", type=int, default=1234, help="실험 재현용 시드")
    parser.add_argument("--save_dir", type=str, default="../save/dlg", help="결과 이미지 및 JSON 저장 경로")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"=== DLG 모델 역추적 공격 및 방어 실험 시작 (Device: {device}) ===")

    # 1. CIFAR-10 단일 이미지 로드
    data_root = "../data/cifar" if os.path.exists("../data/cifar") else "~/.torch"
    dst = datasets.CIFAR10(data_root, train=False, download=True)
    classes = dst.classes
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    gt_data = tp(dst[args.index][0]).unsqueeze(0).to(device)
    gt_label_idx = dst[args.index][1]
    gt_label = torch.tensor([gt_label_idx], dtype=torch.long, device=device)
    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

    print(f"[대상 이미지] Index: {args.index} | 실제 클래스: {classes[gt_label_idx]} ({gt_label_idx})")

    # 2. 모델 초기화 및 이미지 1장의 1회 원본 그래디언트 추출
    torch.manual_seed(args.seed)
    net = LeNet(num_classes=10).to(device)
    net.apply(weights_init)

    pred = net(gt_data)
    y = cross_entropy_for_onehot(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = [g.detach().clone() for g in dy_dx]

    # 3. 3단계 가우시안 노이즈 조건(없음 / 작음 / 큼)에서 동일 조건 복원 및 MSE 비교
    noise_configs = [
        ("노이즈 없음 (None)", args.noise_none),
        ("노이즈 작음 (Small)", args.noise_small),
        ("노이즈 큼 (Large)", args.noise_large),
    ]

    results = []
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].imshow(tt(gt_data[0].cpu()))
    axes[0].set_title(f"Original Image\n(Class: {classes[gt_label_idx]})")
    axes[0].axis("off")

    for idx, (label_name, sigma) in enumerate(noise_configs):
        print(f"\n--- [{label_name}] sigma = {sigma} 조건 복원 진행 중... ---")
        defended_grads = add_gaussian_noise_to_grads(original_dy_dx, noise_std=sigma, seed=args.seed)

        recon_img, pred_cls, _ = run_dlg_reconstruction(
            net=net,
            target_dy_dx=defended_grads,
            img_shape=gt_data.size(),
            num_classes=10,
            iters=args.iters,
            seed=args.seed,
            device=device,
        )

        # 원본 이미지(gt_data)와 복원 이미지(recon_img) 간 픽셀 MSE 계산
        mse_val = F.mse_loss(recon_img, gt_data).item()
        print(f"  -> 완료 | 복원 예측 클래스: {classes[pred_cls]} | 복원 이미지 MSE: {mse_val:.6f}")

        results.append({
            "condition": label_name,
            "noise_std": sigma,
            "target_index": args.index,
            "true_class": classes[gt_label_idx],
            "reconstructed_class": classes[pred_cls],
            "mse": round(mse_val, 6),
        })

        axes[idx + 1].imshow(tt(recon_img[0].cpu()))
        axes[idx + 1].set_title(f"{label_name}\nσ={sigma} | MSE={mse_val:.4f}")
        axes[idx + 1].axis("off")

    plt.tight_layout()
    img_path = os.path.join(args.save_dir, f"dlg_comparison_idx{args.index}.png")
    plt.savefig(img_path, dpi=200)
    plt.close()

    json_path = os.path.join(args.save_dir, f"dlg_metrics_idx{args.index}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n================ [모델 역추적 공격·방어 MSE 비교 요약] ================")
    print(f"{'방어 조건':<22} | {'노이즈 표준편차(σ)':<18} | {'복원 MSE (높을수록 방어 성공)':<28}")
    print("-" * 74)
    for r in results:
        print(f"{r['condition']:<20} | {r['noise_std']:<18} | {r['mse']:.6f}")
    print(f"\n[저장 완료] 비교 시각화 이미지: {img_path}")
    print(f"[저장 완료] 정량 수치 JSON:     {json_path}")


if __name__ == "__main__":
    main()