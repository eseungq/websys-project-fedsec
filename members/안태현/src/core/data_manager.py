import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset


def load_cifar10_splits(data_dir="./data", seed=1, label_skew=False, num_users=10):
    """
    CIFAR-10 데이터를 학습(40,000) / 검증(10,000) / 테스트(10,000)로 분리하고
    10개 가상 클라이언트에 균등(IID) 또는 2-클래스 편향(Non-IID) 배정.
    """
    rng = np.random.default_rng(seed)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # DLG 시각화를 위한 [0, 1] 정규화 없는 원본 텐서용
    transform_raw = transforms.ToTensor()

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    full_train_eval = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_eval)
    full_train_raw = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_raw)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_eval)

    # 50,000장 중 40,000장 학습 / 10,000장 검증 고정 분리 (동일 시드 시 항상 동일 풀 사용)
    all_indices = rng.permutation(len(full_train))
    train_pool_indices = all_indices[:40000]
    val_indices = all_indices[40000:]

    val_dataset = Subset(full_train_eval, val_indices.tolist())
    targets = np.array(full_train.targets)
    pool_targets = targets[train_pool_indices]

    samples_per_user = len(train_pool_indices) // num_users  # 4,000장
    user_groups = {}

    if not label_skew:
        # IID: 40,000장을 무작위로 4,000장씩 균등 배분
        shuffled_pool = rng.permutation(train_pool_indices)
        for uid in range(num_users):
            start = uid * samples_per_user
            user_groups[uid] = shuffled_pool[start:start + samples_per_user].tolist()
    else:
        # Label Skew (Non-IID): 동일한 40,000장을 클래스별로 정렬 후 20개 샤드(각 2,000장)로 나누어
        # 각 클라이언트에게 2개 샤드(2개 클래스 집중)씩 배정 -> 전체 이미지 풀 및 4,000장 수량 100% 유지
        sorted_order = np.argsort(pool_targets)
        sorted_pool_indices = train_pool_indices[sorted_order]

        num_shards = num_users * 2  # 20 shards
        shard_size = len(sorted_pool_indices) // num_shards  # 2,000
        shard_ids = rng.permutation(num_shards)

        for uid in range(num_users):
            s1, s2 = shard_ids[2 * uid], shard_ids[2 * uid + 1]
            idx1 = sorted_pool_indices[s1 * shard_size:(s1 + 1) * shard_size]
            idx2 = sorted_pool_indices[s2 * shard_size:(s2 + 1) * shard_size]
            combined = np.concatenate([idx1, idx2])
            user_groups[uid] = rng.permutation(combined).tolist()

    # 명세서 요구사항: 클라이언트별 데이터 수와 클래스 분포 기록용 통계 생성
    distribution_report = {}
    for uid, idxs in user_groups.items():
        counts = np.bincount(targets[idxs], minlength=10).tolist()
        distribution_report[uid] = {
            "num_samples": len(idxs),
            "class_counts": counts,
        }

    return {
        "train_dataset": full_train,
        "train_raw_dataset": full_train_raw,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "user_groups": user_groups,
        "distribution_report": distribution_report,
    }