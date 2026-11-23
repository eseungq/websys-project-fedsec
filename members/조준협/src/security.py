# -*- coding: utf-8 -*-
import copy
from typing import Dict, List, Tuple, Optional
import torch


def compute_l2_norm(delta_dict: Dict[str, torch.Tensor]) -> float:
    """업데이트(state_dict 변화량) 전체 파라미터를 하나의 벡터로 본 L2 노름 계산"""
    total_sq = 0.0
    for tensor in delta_dict.values():
        t = tensor.detach().float()
        total_sq += torch.sum(t * t).item()
    return total_sq ** 0.5


def scale_delta(delta_dict: Dict[str, torch.Tensor], factor: float) -> Dict[str, torch.Tensor]:
    """업데이트 딕셔너리의 모든 텐서에 스칼라 배수 적용"""
    return {k: v.detach().clone().float() * factor for k, v in delta_dict.items()}


def apply_collusion_attack(
    client_updates: Dict[int, Dict[str, torch.Tensor]],
    malicious_ids: List[int],
    attack_scale: float = -5.0,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[dict]]:
    """
    내부자 협력 공격 (Collusion Attack):
    - 같은 라운드에 참여한 악성 클라이언트가 2명 이상이면:
      참여한 악성 클라이언트들의 정상 업데이트를 평균낸 뒤 attack_scale(-5)을 곱한 동일 벡터를 각각 제출
    - 악성 참여자가 1명이면: 자기 업데이트의 attack_scale(-5)배를 제출
    - 0명이면: 공격 미적용
    """
    manipulated_updates = {cid: copy.deepcopy(upd) for cid, upd in client_updates.items()}
    active_malicious = sorted([cid for cid in client_updates.keys() if cid in malicious_ids])
    attack_logs = []

    if len(active_malicious) == 0:
        return manipulated_updates, attack_logs

    # 참여한 악성 클라이언트들의 정상 업데이트 평균 계산 (1명이면 자기 자신과 동일)
    ref_keys = list(client_updates[active_malicious[0]].keys())
    avg_malicious_delta = {}
    for k in ref_keys:
        stacked = torch.stack([client_updates[cid][k].detach().float() for cid in active_malicious], dim=0)
        avg_malicious_delta[k] = torch.mean(stacked, dim=0)

    # 평균 벡터에 -5배(attack_scale) 적용하여 담합 벡터 생성
    colluded_vector = scale_delta(avg_malicious_delta, attack_scale)
    mode = "collusion_avg_x_scale" if len(active_malicious) >= 2 else "single_malicious_x_scale"

    for cid in active_malicious:
        norm_before = compute_l2_norm(client_updates[cid])
        manipulated_updates[cid] = copy.deepcopy(colluded_vector)
        norm_after = compute_l2_norm(manipulated_updates[cid])
        attack_logs.append({
            "client_id": cid,
            "is_malicious": True,
            "attack_applied": True,
            "attack_mode": mode,
            "active_malicious_count": len(active_malicious),
            "norm_before_attack": round(norm_before, 4),
            "norm_after_attack": round(norm_after, 4),
        })

    return manipulated_updates, attack_logs


def clip_update_norm(
    delta_dict: Dict[str, torch.Tensor],
    clip_norm: float,
) -> Tuple[Dict[str, torch.Tensor], float, float, bool]:
    """
    업데이트 L2 노름 클리핑:
    - ||delta||_2 > clip_norm 이면 delta * (clip_norm / ||delta||_2) 로 축소
    - 반환: (클리핑된 delta, 클리핑 전 노름, 클리핑 후 노름, 클리핑 발동 여부)
    """
    norm_before = compute_l2_norm(delta_dict)
    if clip_norm is None or clip_norm <= 0 or norm_before <= clip_norm:
        return copy.deepcopy(delta_dict), norm_before, norm_before, False

    ratio = clip_norm / (norm_before + 1e-12)
    clipped = scale_delta(delta_dict, ratio)
    norm_after = compute_l2_norm(clipped)
    return clipped, norm_before, norm_after, True


def aggregate_robust(
    updates_list: List[Dict[str, torch.Tensor]],
    sample_counts: List[int],
    method: str = "fedavg",
    clip_norm: Optional[float] = None,
) -> Tuple[Optional[Dict[str, torch.Tensor]], List[dict]]:
    """
    방어 및 집계 통합 함수 (주의: 악성 ID 정답을 절대 입력받지 않음):
    - method == 'fedavg': 클리핑 없이 표본 수 비례 가중 평균
    - method == 'clipped_fedavg': 노름 클리핑 적용 후 표본 수 비례 가중 평균
    - method == 'median' 또는 'clipped_median':
      노름 클리핑(설정 시) 후 좌표별 중앙값 집계 (짝수 개일 때 가운데 두 값의 평균 사용)
    """
    n = len(updates_list)
    if n < 3:
        # 명세서: 채택 수가 3개 미만이면 모델을 갱신하지 않고 사유 기록
        return None, []

    processed_updates = []
    defense_logs = []

    use_clipping = (clip_norm is not None and clip_norm > 0) and (method in ["clipped_fedavg", "median", "clipped_median"])

    for idx, upd in enumerate(updates_list):
        if use_clipping:
            clipped_upd, nb, na, was_clipped = clip_update_norm(upd, clip_norm)
        else:
            clipped_upd = copy.deepcopy(upd)
            nb = compute_l2_norm(upd)
            na = nb
            was_clipped = False

        processed_updates.append(clipped_upd)
        defense_logs.append({
            "index": idx,
            "num_samples": sample_counts[idx],
            "norm_before_defense": round(nb, 4),
            "norm_after_defense": round(na, 4),
            "was_clipped": was_clipped,
        })

    ref_keys = list(processed_updates[0].keys())
    aggregated_delta = {}

    if method in ["fedavg", "clipped_fedavg"]:
        total_samples = float(sum(sample_counts))
        for k in ref_keys:
            acc = torch.zeros_like(processed_updates[0][k], dtype=torch.float32)
            for upd, cnt in zip(processed_updates, sample_counts):
                acc += upd[k].detach().float() * (cnt / total_samples)
            aggregated_delta[k] = acc

    elif method in ["median", "clipped_median"]:
        for k in ref_keys:
            stacked = torch.stack([upd[k].detach().float() for upd in processed_updates], dim=0)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if n % 2 == 1:
                aggregated_delta[k] = sorted_vals[n // 2]
            else:
                # 짝수 개일 때 가운데 두 값의 평균 사용
                aggregated_delta[k] = 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    else:
        raise ValueError(f"지원하지 않는 집계 방식입니다: {method}")

    return aggregated_delta, defense_logs


if __name__ == "__main__":
    # 명세서 요구 산출물용: 작은 수치 예제 검증 (작동 테스트)
    print("=== 내부자 협력 공격 및 방어(클리핑 + 중앙값) 수치 검증 테스트 ===")
    u0 = {"w": torch.tensor([1.0, 2.0])}   # 악성 0
    u1 = {"w": torch.tensor([3.0, 4.0])}   # 악성 1
    u2 = {"w": torch.tensor([2.0, 2.0])}   # 정상 2
    u3 = {"w": torch.tensor([2.5, 1.5])}   # 정상 3

    updates = {0: u0, 1: u1, 2: u2, 3: u3}
    attacked, a_logs = apply_collusion_attack(updates, malicious_ids=[0, 1], attack_scale=-5.0)
    # 악성 평균 = [2.0, 3.0], -5배 = [-10.0, -15.0]
    print("공격 후 악성 0 벡터:", attacked[0]["w"].tolist(), "(기대값: [-10.0, -15.0])")
    print("공격 후 악성 1 벡터:", attacked[1]["w"].tolist(), "(기대값: [-10.0, -15.0])")

    agg_med, _ = aggregate_robust(
        updates_list=[attacked[i] for i in range(4)],
        sample_counts=[100, 100, 100, 100],
        method="clipped_median",
        clip_norm=5.0,
    )
    # 4개(짝수) 정렬 후 가운데 두 값 평균 검증
    print("클리핑(5.0)+중앙값 집계 결과:", agg_med["w"].tolist())