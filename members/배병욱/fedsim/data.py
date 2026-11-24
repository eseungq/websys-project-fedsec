"""CIFAR-10 로드와 클라이언트 분할.

- 이미지는 공식 python 배치 파일(data_batch_1~5, test_batch)을 직접 읽는다.
  data_batch_1~5를 순서대로 이어 붙인 인덱스가 팀 공용 분할 파일의 인덱스와 같다.
- IID 분할은 팀 공용 파일(configs/splits/iid_seed0.json)을 그대로 쓴다.
- 레이블 편향 분할은 같은 이미지·같은 클라이언트별 표본 수를 유지하고 배정만 바꾼다.
"""
import json
import os
import pickle

import numpy as np
import torch

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return x, y


def load_raw(root):
    """정규화 전 [0,1] 범위 이미지. DLG 실험에서 쓴다."""
    base = os.path.join(root, "cifar-10-batches-py")
    parts = [_load_batch(os.path.join(base, f"data_batch_{i}")) for i in range(1, 6)]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test, y_test = _load_batch(os.path.join(base, "test_batch"))
    return x_train, y_train, x_test, y_test


def normalize(x):
    return (x - MEAN[None, :, None, None]) / STD[None, :, None, None]


def load_cifar10(root):
    x_train, y_train, x_test, y_test = load_raw(root)
    return (torch.from_numpy(normalize(x_train)), torch.from_numpy(y_train),
            torch.from_numpy(normalize(x_test)), torch.from_numpy(y_test))


def load_split(path):
    with open(path) as f:
        d = json.load(f)
    return d["val"], d["clients"]


def label_skew_split(clients, labels, seed, skew=0.8, shards_per_client=2):
    """IID 분할과 같은 이미지·같은 표본 수로 배정만 바꾼 레이블 편향 분할.

    각 클라이언트 표본의 `skew` 비율은 레이블로 정렬한 조각(shard) `shards_per_client`개에서,
    나머지는 무작위로 채운다. 클라이언트별 표본 수는 그대로 유지된다.
    """
    sizes = [len(c) for c in clients]
    if len(set(sizes)) != 1:
        raise ValueError("레이블 편향 분할은 클라이언트별 표본 수가 같을 때만 지원합니다")
    n_clients, size = len(clients), sizes[0]
    labels = np.asarray(labels)
    rng = np.random.default_rng([seed, 7])
    pool = np.concatenate([np.asarray(c) for c in clients])
    pool = pool[rng.permutation(len(pool))]

    n_skew = int(round(size * skew))
    if n_skew % shards_per_client:
        raise ValueError(f"편향 표본 수 {n_skew}가 shards_per_client={shards_per_client}로 나누어떨어지지 않습니다")
    skew_part, rand_part = pool[: n_skew * n_clients], pool[n_skew * n_clients:]
    skew_part = skew_part[np.argsort(labels[skew_part], kind="stable")]
    shards = np.array_split(skew_part, n_clients * shards_per_client)
    order = rng.permutation(len(shards))
    rand_chunks = np.array_split(rand_part, n_clients)

    out = []
    for k in range(n_clients):
        picked = [shards[order[k * shards_per_client + j]] for j in range(shards_per_client)]
        idx = np.concatenate(picked + [rand_chunks[k]])
        out.append(sorted(idx.tolist()))
    assert [len(c) for c in out] == sizes, "표본 수가 유지되지 않음"
    assert sorted(np.concatenate(out).tolist()) == sorted(pool.tolist()), "사용 이미지가 달라짐"
    return out


def class_distribution(clients, labels, num_classes=10):
    labels = np.asarray(labels)
    return [np.bincount(labels[np.asarray(c)], minlength=num_classes).tolist() for c in clients]
