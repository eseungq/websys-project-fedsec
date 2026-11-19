"""모델 역추적 공격(DLG)과 가우시안 노이즈 방어.

가정: 서버(공격자)가 한 클라이언트가 이미지 한 장으로 계산한 그래디언트 1회분과 모델을 본다.
역할을 함수로 나눈다.
- 피해자: `victim_gradients` (정답 레이블 사용), 방어 `add_gaussian_noise` (전송 전에 적용)
- 공격자: `reconstruct` (모델과 받은 그래디언트만 입력. 원본 이미지·정답 레이블 인자가 없다)
- 평가: `pixel_mse` (원본은 여기서만 쓴다)

구성 출처: https://github.com/mit-han-lab/dlg 의 main.py (Zhu et al., NeurIPS 2019).
같은 구성(soft label 더미, L-BFGS lr 1, 그래디언트 차이 제곱합 최소화)으로 새로 작성했다. 달라진 점:
- 노이즈 방어 추가 (논문 5장 실험에 해당, 공개 코드에는 없음). σ는 표준편차
- 시드 분리: 모델 초기화·더미 초기값·노이즈 시드를 각각 받아, 하나만 바꿔도 나머지는 같다
- 모델 파라미터의 .grad를 건드리지 않도록 backward 대신 autograd.grad로 더미에만 그래디언트를 넣는다
- 반복 중 그래디언트 차이가 NaN·무한대가 되면 멈추고 `diverged`로 표시한다
- N회마다 더미 이미지와 그래디언트 차이를 기록해, 평가 쪽에서 MSE 궤적을 계산할 수 있게 한다
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

# 원 논문 코드와 같은 L-BFGS 설정. 반복 1회 = optimizer.step 1회(내부 최대 20회 평가, PyTorch 기본값)
LBFGS_LR = 1.0


@dataclass(frozen=True)
class Snapshot:
    iteration: int          # 지금까지 끝낸 반복 수 (0 = 초기값)
    grad_diff: float        # 받은 그래디언트와 더미 그래디언트의 차이 제곱합
    image: torch.Tensor     # 그 시점의 더미 이미지 (1, 3, H, W), 자르지 않은 값


@dataclass(frozen=True)
class Reconstruction:
    image: torch.Tensor             # 마지막 더미 이미지 (1, 3, H, W), 자르지 않은 값
    label_logits: torch.Tensor      # 마지막 더미 soft label (1, C). softmax 전 값
    grad_diff: float                # 마지막 그래디언트 차이
    iterations: int                 # 실제로 끝낸 반복 수 (발산하면 요청보다 적다)
    diverged: bool
    trajectory: tuple[Snapshot, ...]


def victim_gradients(model: nn.Module, image: torch.Tensor, label: torch.Tensor) -> list[torch.Tensor]:
    """피해자 쪽: 이미지 한 장 (1, 3, H, W)과 정답 레이블 (1,)로 교차 엔트로피 그래디언트를 계산한다."""
    if image.dim() != 4 or image.shape[0] != 1 or label.shape != (1,):
        raise ValueError("이미지는 (1, 3, H, W), 레이블은 (1,) 모양이어야 합니다 (이미지 한 장).")
    loss = F.cross_entropy(model(image), label)
    return [g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))]


def add_gaussian_noise(gradients: list[torch.Tensor], sigma: float, seed: int) -> list[torch.Tensor]:
    """방어: 각 그래디언트 원소에 N(0, sigma²) 노이즈를 더한 새 목록을 돌려준다. 입력은 바꾸지 않는다."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"노이즈 σ는 0 이상의 유한한 값이어야 합니다: {sigma}")
    if sigma == 0:
        return [g.clone() for g in gradients]
    generator = torch.Generator().manual_seed(seed)
    return [g + sigma * torch.randn(g.shape, generator=generator).to(g) for g in gradients]


def _dummy_gradients(model: nn.Module, dummy_image: torch.Tensor, dummy_logits: torch.Tensor) -> tuple:
    soft_label = F.softmax(dummy_logits, dim=-1)
    loss = torch.sum(-soft_label * F.log_softmax(model(dummy_image), dim=-1))
    return torch.autograd.grad(loss, list(model.parameters()), create_graph=True)


def _grad_diff(model: nn.Module, gradients: list[torch.Tensor],
               dummy_image: torch.Tensor, dummy_logits: torch.Tensor) -> torch.Tensor:
    dummy = _dummy_gradients(model, dummy_image, dummy_logits)
    return sum(((d - g) ** 2).sum() for d, g in zip(dummy, gradients))


def reconstruct(model: nn.Module, gradients: list[torch.Tensor], image_shape: tuple[int, ...],
                iterations: int, dummy_seed: int, record_every: int = 10) -> Reconstruction:
    """공격자 쪽: 받은 그래디언트와 모델만으로 입력 이미지와 soft label을 복원한다.

    더미 이미지·레이블은 dummy_seed로 N(0, 1)에서 뽑는다. 같은 모델·그래디언트·시드면 같은 경로를 따른다.
    record_every회마다(그리고 시작과 끝에) 더미 이미지와 그래디언트 차이를 trajectory에 남긴다.
    """
    if iterations < 0 or record_every <= 0:
        raise ValueError("iterations는 0 이상, record_every는 1 이상이어야 합니다.")
    params = list(model.parameters())
    if len(gradients) != len(params) or any(g.shape != p.shape for g, p in zip(gradients, params)):
        raise ValueError("그래디언트 목록이 모델 파라미터와 개수·모양이 맞지 않습니다.")

    generator = torch.Generator().manual_seed(dummy_seed)
    num_classes = gradients[-1].shape[0]  # 마지막 층 편향의 길이 = 클래스 수
    dummy_image = torch.randn(image_shape, generator=generator).requires_grad_(True)
    dummy_logits = torch.randn((1, num_classes), generator=generator).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_image, dummy_logits], lr=LBFGS_LR)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        diff = _grad_diff(model, gradients, dummy_image, dummy_logits)
        dummy_image.grad, dummy_logits.grad = torch.autograd.grad(diff, [dummy_image, dummy_logits])
        return diff.detach()

    def snapshot(iteration: int, diff: float) -> Snapshot:
        return Snapshot(iteration, diff, dummy_image.detach().clone())

    diff = _grad_diff(model, gradients, dummy_image, dummy_logits).item()
    trajectory = [snapshot(0, diff)]
    done, diverged = 0, False
    for it in range(1, iterations + 1):
        optimizer.step(closure)
        diff = _grad_diff(model, gradients, dummy_image, dummy_logits).item()
        done = it
        if not math.isfinite(diff):
            diverged = True
            trajectory.append(snapshot(it, diff))
            break
        if it % record_every == 0 or it == iterations:
            trajectory.append(snapshot(it, diff))

    return Reconstruction(
        image=dummy_image.detach().clone(), label_logits=dummy_logits.detach().clone(),
        grad_diff=diff, iterations=done, diverged=diverged, trajectory=tuple(trajectory),
    )


def pixel_mse(reconstructed: torch.Tensor, original: torch.Tensor, clip: bool = True) -> float:
    """평가: 픽셀 [0, 1] 기준 평균 제곱 오차. clip이면 복원 이미지를 [0, 1]로 자른 뒤 계산한다."""
    if reconstructed.shape != original.shape:
        raise ValueError(f"이미지 모양이 다릅니다: {tuple(reconstructed.shape)} vs {tuple(original.shape)}")
    candidate = reconstructed.clamp(0.0, 1.0) if clip else reconstructed
    return F.mse_loss(candidate, original).item()
