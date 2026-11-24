"""모델 역추적 공격(DLG)과 가우시안 노이즈 방어. 학습 성능 실험과 분리된 소규모 실험.

- 서버가 한 클라이언트의 이미지 1장 · 1회 그래디언트를 본다고 가정한다.
- 원본 이미지와 정답 레이블은 공격 입력으로 주지 않고 MSE 평가에만 쓴다.
- 같은 이미지·모델·복원 초기값에서 노이즈 σ만 바꾼다.
- 반복 부족과 노이즈 영향을 구분하려고, 한 실행에서 `check_iters`(예: 300)와 최종 `iters`(예: 600)의
  MSE를 함께 기록한다. 반복을 2배로 늘려도 MSE가 거의 그대로면 '노이즈로 정체',
  크게 떨어지면 '반복 부족(복원 미완료)'으로 판정한다.

공개 코드 출처: https://github.com/mit-han-lab/dlg (MIT) — 손실(그래디언트 차이 제곱합),
L-BFGS(lr=1), 소프트 레이블 최적화, LeNet 구조를 따랐다. 노이즈 방어·판정·로그는 새로 작성.
수정: 원본 설정(선 탐색 없음)은 이 환경에서 σ=0에서도 수십 회 만에 폭주해서(experiments/dlg_original_lbfgs),
기본값을 L-BFGS strong-Wolfe 선 탐색으로 바꿨다. --line-search none 으로 원본 동작 재현.

예 (members/배병욱 에서): python -m fedsim.dlg --images 3 10 --sigmas 0 0.01 0.1
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F

from .data import load_raw
from .models import LeNetDLG, init_dlg_weights
from .runner import ROOT, git_commit

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def client_gradient(net, x, y, sigma, noise_seed):
    """클라이언트가 전송하는 그래디언트 (+ 가우시안 노이즈 방어)."""
    loss = F.cross_entropy(net(x), y)
    grads = torch.autograd.grad(loss, net.parameters())
    g = torch.Generator().manual_seed(noise_seed)
    return [gr.detach() + sigma * torch.randn(gr.shape, generator=g) for gr in grads]


def attack(net, target_grads, shape, iters, init_seed, check_iters, record_every=10, x_eval=None,
           line_search="strong_wolfe"):
    g = torch.Generator().manual_seed(init_seed)
    dummy_x = torch.randn(shape, generator=g).requires_grad_(True)
    dummy_y = torch.randn((1, 10), generator=g).requires_grad_(True)
    opt = torch.optim.LBFGS([dummy_x, dummy_y], lr=1,
                            line_search_fn=None if line_search == "none" else line_search)
    traj, at_check = [], None

    def closure():
        opt.zero_grad()
        pred = net(dummy_x)
        loss = torch.sum(-F.softmax(dummy_y, -1) * F.log_softmax(pred, -1))
        dg = torch.autograd.grad(loss, net.parameters(), create_graph=True)
        diff = sum(((a - b) ** 2).sum() for a, b in zip(dg, target_grads))
        diff.backward()
        return diff

    status = "ok"
    for it in range(1, iters + 1):
        diff = float(opt.step(closure).detach())
        if not torch.isfinite(dummy_x).all() or diff != diff:
            status = f"nan_at_{it}"
            break
        mse = F.mse_loss(dummy_x.detach().clamp(0, 1), x_eval).item()  # 화면에 보이는 [0,1] 이미지 기준
        if it % record_every == 0 or it == 1:
            traj.append({"iter": it, "grad_diff": diff, "mse": mse,
                         "mse_raw": F.mse_loss(dummy_x.detach(), x_eval).item()})
        if it == check_iters:
            at_check = mse
    final = F.mse_loss(dummy_x.detach().clamp(0, 1), x_eval).item() if status == "ok" else float("nan")
    return dummy_x.detach(), dummy_y.detach(), final, at_check, traj, status


def verdict(mse_check, mse_final, sigma):
    if mse_check is None or mse_final != mse_final:
        return "실패(NaN)"
    if mse_final < 1e-2:
        return "복원 성공"
    if mse_final < 0.5 * mse_check:
        return "반복 부족(더 돌리면 개선)"
    if mse_final > mse_check:
        return "노이즈 쪽으로 멀어짐" if sigma > 0 else "최적화 불안정"
    return "노이즈로 정체" if sigma > 0 else "최적화 정체(노이즈 아님)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", type=int, default=[3, 10], help="CIFAR-10 테스트 인덱스")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.01, 0.1])
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--check-iters", type=int, default=300)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--line-search", default="strong_wolfe", choices=["strong_wolfe", "none"])
    ap.add_argument("--out", default="experiments/dlg")
    a = ap.parse_args()
    if not 0 < a.check_iters < a.iters:
        ap.error("0 < --check-iters < --iters 이어야 합니다")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    _, _, x_test, y_test = load_raw(os.path.join(ROOT, "data"))
    rows, traj_rows, recon = [], [], {}
    for idx in a.images:
        x = torch.from_numpy(x_test[idx:idx + 1])
        y = torch.tensor([int(y_test[idx])])
        for sigma in a.sigmas:
            net = init_dlg_weights(LeNetDLG(), a.model_seed)
            tg = client_gradient(net, x, y, sigma, noise_seed=1000 + idx)
            t = time.time()
            dx, dy, mse, mse_chk, traj, status = attack(net, tg, x.shape, a.iters, a.init_seed,
                                                        a.check_iters, x_eval=x, line_search=a.line_search)
            pred = int(dy.argmax().item())
            rows.append({"image_idx": idx, "label": CLASSES[int(y)], "sigma": sigma, "iters": a.iters,
                         "mse_at_check": None if mse_chk is None else round(mse_chk, 6),
                         "check_iters": a.check_iters, "mse_final": round(mse, 6),
                         "label_recovered": pred == int(y), "status": status,
                         "verdict": verdict(mse_chk, mse, sigma), "sec": round(time.time() - t, 1)})
            traj_rows += [{"image_idx": idx, "sigma": sigma, **r} for r in traj]
            recon[(idx, sigma)] = dx.clamp(0, 1)[0]
            print(rows[-1])

    for name, data in [("results.csv", rows), ("trajectory.csv", traj_rows)]:
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(a), "commit": git_commit(), "torch": torch.__version__}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_img, n_s = len(a.images), len(a.sigmas)
    fig, ax = plt.subplots(n_img, n_s + 2, figsize=(2.2 * (n_s + 2), 2.4 * n_img), squeeze=False)
    for i, idx in enumerate(a.images):
        ax[i][0].imshow(x_test[idx].transpose(1, 2, 0))
        ax[i][0].set_title(f"original #{idx}", fontsize=8)
        for j, s in enumerate(a.sigmas):
            ax[i][j + 1].imshow(recon[(idx, s)].permute(1, 2, 0).numpy())
            r = next(r for r in rows if r["image_idx"] == idx and r["sigma"] == s)
            ax[i][j + 1].set_title(f"σ={s}  MSE={r['mse_final']:.4f}", fontsize=8)
        for s in a.sigmas:
            tr = [r for r in traj_rows if r["image_idx"] == idx and r["sigma"] == s]
            ax[i][-1].semilogy([r["iter"] for r in tr], [max(r["mse"], 1e-8) for r in tr], label=f"σ={s}")
        ax[i][-1].axvline(a.check_iters, ls="--", c="gray", lw=0.8)
        ax[i][-1].set_title("MSE vs iter", fontsize=8)
        ax[i][-1].legend(fontsize=6)
        for k in range(n_s + 1):
            ax[i][k].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=110)
    print("saved", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
