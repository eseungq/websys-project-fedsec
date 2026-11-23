import copy
import torch
from src.core.local_trainer import compute_l2_norm


def add_gaussian_noise_to_gradients(gradients, noise_std, seed=1):
    """모델 역추적 방어: 전송 전 그래디언트에 가우시안 노이즈 N(0, sigma^2) 추가"""
    if noise_std <= 0.0:
        return [g.detach().cpu().clone() for g in gradients]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    noisy_grads = []
    for g in gradients:
        g_cpu = g.detach().cpu()
        noise = torch.randn(g_cpu.size(), generator=gen, dtype=g_cpu.dtype) * noise_std
        noisy_grads.append(g_cpu + noise)
    return noisy_grads


def aggregate_fedavg(accepted_updates, sample_counts):
    """채택된 업데이트를 표본 수(n_k)에 비례해 가중 평균 (FedAvg)"""
    total_samples = sum(sample_counts[cid] for cid in accepted_updates.keys())
    first_cid = next(iter(accepted_updates.keys()))
    agg_delta = {}

    for k in accepted_updates[first_cid].keys():
        weighted_sum = torch.zeros_like(accepted_updates[first_cid][k], dtype=torch.float32)
        for cid, upd in accepted_updates.items():
            weight = sample_counts[cid] / total_samples
            weighted_sum += upd[k].float() * weight
        agg_delta[k] = weighted_sum.to(accepted_updates[first_cid][k].dtype)

    return agg_delta


def clip_updates_l2(accepted_updates, clip_bound):
    """
    내부자 협력 방어 1단계: 업데이트 L2 노름 클리핑 (악성 ID 정보 미사용)
    각 업데이트의 L2 노름이 clip_bound를 초과하면 비율대로 축소.
    """
    clipped_updates = {}
    post_norms = {}

    for cid, upd in accepted_updates.items():
        norm = compute_l2_norm(upd)
        scale = min(1.0, clip_bound / (norm + 1e-12))
        clipped = {}
        for k, v in upd.items():
            if v.is_floating_point():
                clipped[k] = (v.float() * scale).to(v.dtype)
            else:
                clipped[k] = v.clone()
        clipped_updates[cid] = clipped
        post_norms[cid] = compute_l2_norm(clipped)

    return clipped_updates, post_norms


def aggregate_coordinate_wise_median(accepted_updates):
    """
    내부자 협력 방어 2: 좌표별 중앙값 집계 (악성 ID 정보 미사용)
    - 채택 개수가 홀수일 때: 정렬된 가운데 값 사용
    - 채택 개수가 짝수일 때: 정렬된 가운데 두 값의 평균 사용
    """
    cids = list(accepted_updates.keys())
    n = len(cids)
    first_cid = cids[0]
    agg_delta = {}

    for k in accepted_updates[first_cid].keys():
        stacked = torch.stack([accepted_updates[cid][k].float() for cid in cids], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if n % 2 == 1:
            med = sorted_vals[n // 2]
        else:
            med = 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
        agg_delta[k] = med.to(accepted_updates[first_cid][k].dtype)

    post_norms = {cid: compute_l2_norm(accepted_updates[cid]) for cid in cids}
    return agg_delta, post_norms


def apply_defense_and_aggregate(accepted_updates, sample_counts, defense_mode="none", clip_bound=1.5):
    """
    선택된 방어 방식에 따라 업데이트 처리 및 최종 집계 Δw 반환.
    주의: 방어 함수에는 악성 클라이언트 ID 정답을 절대 전달하지 않음.
    """
    if defense_mode == "clipping":
        clipped, post_norms = clip_updates_l2(accepted_updates, clip_bound=clip_bound)
        agg_delta = aggregate_fedavg(clipped, sample_counts)
        return agg_delta, post_norms

    if defense_mode == "median":
        agg_delta, post_norms = aggregate_coordinate_wise_median(accepted_updates)
        return agg_delta, post_norms

    # defense_mode == "none" (기본 FedAvg)
    post_norms = {cid: compute_l2_norm(upd) for cid, upd in accepted_updates.items()}
    agg_delta = aggregate_fedavg(accepted_updates, sample_counts)
    return agg_delta, post_norms