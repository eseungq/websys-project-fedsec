"""학습 이미지 인덱스를 검증용과 클라이언트용으로 나누고 JSON으로 저장·검증한다.

시드 숫자만 같아서는 코드나 라이브러리가 다를 때 결과가 달라질 수 있으므로,
분할 결과 파일 자체를 레포에 커밋해 팀 전체가 같은 파일을 쓴다.
"""

import json
import random
from pathlib import Path

NUM_TRAIN_IMAGES = 50_000


def make_iid_split(num_clients: int, val_size: int, seed: int,
                   num_images: int = NUM_TRAIN_IMAGES) -> dict:
    """무작위로 섞은 뒤 앞쪽을 검증용, 나머지를 클라이언트에 같은 개수로 나눈다."""
    train_size = num_images - val_size
    if num_clients <= 0 or val_size < 0 or train_size <= 0:
        raise ValueError("num_clients는 양수, val_size는 0 이상이고 전체보다 작아야 합니다.")
    if train_size % num_clients != 0:
        raise ValueError(f"학습 이미지 {train_size}장을 클라이언트 {num_clients}명에게 똑같이 나눌 수 없습니다.")

    shuffled = random.Random(seed).sample(range(num_images), num_images)
    per_client = train_size // num_clients
    client_pool = shuffled[val_size:]
    clients = [sorted(client_pool[i * per_client:(i + 1) * per_client]) for i in range(num_clients)]
    return {
        "kind": "iid",
        "seed": seed,
        "num_images": num_images,
        "val": sorted(shuffled[:val_size]),
        "clients": clients,
    }


NUM_CLASSES = 10
SINKHORN_STEPS = 500
MIN_WEIGHT = 1e-12


def class_counts(split: dict, labels: list[int], num_classes: int = NUM_CLASSES) -> list[list[int]]:
    """클라이언트별 클래스 이미지 수 (행: 클라이언트, 열: 클래스)."""
    counts = [[0] * num_classes for _ in split["clients"]]
    for row, indices in zip(counts, split["clients"]):
        for i in indices:
            row[labels[i]] += 1
    return counts


def skew_score(counts: list[list[int]]) -> float:
    """클라이언트 분포가 전체 분포에서 떨어진 정도: 클라이언트별 총변동거리(0~1)의 평균."""
    totals = [sum(col) for col in zip(*counts)]
    overall = [t / sum(totals) for t in totals]
    distances = [
        0.5 * sum(abs(c / sum(row) - p) for c, p in zip(row, overall))
        for row in counts
    ]
    return sum(distances) / len(distances)


def _dirichlet(rng: random.Random, alpha: float, size: int) -> list[float]:
    # alpha가 작으면 0이 나올 수 있는데, 한 클래스 열이 모두 0이면 열 합을 맞출 수 없어 아주 작은 값을 둔다
    draws = [max(rng.gammavariate(alpha, 1.0), MIN_WEIGHT) for _ in range(size)]
    total = sum(draws)
    return [d / total for d in draws]


def _fit_margins(weights: list[list[float]], row_sums: list[int], col_sums: list[int]) -> list[list[float]]:
    """행·열 합이 목표값이 되도록 행과 열을 번갈아 늘리고 줄인다 (Sinkhorn)."""
    table = [row[:] for row in weights]
    for _ in range(SINKHORN_STEPS):
        table = [[v * target / sum(row) for v in row] for row, target in zip(table, row_sums)]
        col_totals = [sum(col) for col in zip(*table)]
        table = [[v * target / total for v, target, total in zip(row, col_sums, col_totals)] for row in table]
    return table


def _round_margins(table: list[list[float]], row_sums: list[int], col_sums: list[int]) -> list[list[int]]:
    """내림한 뒤 모자란 행·열에 소수 부분이 큰 칸부터 1씩 더해 행·열 합을 정확히 맞춘다."""
    counts = [[int(v) for v in row] for row in table]
    row_left = [target - sum(row) for row, target in zip(counts, row_sums)]
    col_left = [target - sum(col) for col, target in zip(zip(*counts), col_sums)]
    cells = sorted(((table[r][c] - counts[r][c], r, c) for r in range(len(table)) for c in range(len(col_sums))),
                   reverse=True)
    for _, r, c in cells:
        if row_left[r] > 0 and col_left[c] > 0:
            counts[r][c] += 1
            row_left[r] -= 1
            col_left[c] -= 1
    # 한 칸에 1씩만 더해서 남는 경우가 드물게 있다. 그때는 남은 행·열끼리 채운다.
    for r in range(len(counts)):
        for c in range(len(col_sums)):
            add = min(row_left[r], col_left[c])
            counts[r][c] += add
            row_left[r] -= add
            col_left[c] -= add
    return counts


def make_label_skew_split(base: dict, labels: list[int], alpha: float, seed: int,
                          num_classes: int = NUM_CLASSES) -> dict:
    """base(IID 분할)와 같은 검증 세트·이미지 풀·클라이언트별 표본 수를 두고 배정만 바꾼다.

    클라이언트별 클래스 비율을 Dirichlet(alpha)로 뽑고, 행(클라이언트 표본 수)과
    열(클래스별 이미지 수) 합을 정확히 맞춘 개수만큼 클래스별로 섞은 이미지를 나눠 준다.
    """
    validate_split(base)
    if alpha <= 0:
        raise ValueError(f"alpha는 양수여야 합니다: {alpha}")
    rng = random.Random(seed)
    pool = sorted(i for client in base["clients"] for i in client)
    by_class = [[i for i in pool if labels[i] == k] for k in range(num_classes)]
    row_sums = [len(client) for client in base["clients"]]
    col_sums = [len(images) for images in by_class]

    weights = [_dirichlet(rng, alpha, num_classes) for _ in row_sums]
    counts = _round_margins(_fit_margins(weights, row_sums, col_sums), row_sums, col_sums)

    clients = [[] for _ in row_sums]
    for k, images in enumerate(by_class):
        shuffled = rng.sample(images, len(images))
        start = 0
        for cid, row in enumerate(counts):
            clients[cid].extend(shuffled[start:start + row[k]])
            start += row[k]
    return {
        "kind": "label_skew",
        "seed": seed,
        "alpha": alpha,
        "num_images": base["num_images"],
        "val": list(base["val"]),
        "clients": [sorted(c) for c in clients],
        "class_counts": counts,
    }


def validate_split(split: dict) -> None:
    """인덱스가 범위 안에 있고, 검증·클라이언트 사이에 겹치지 않는지 확인한다."""
    for key in ("num_images", "val", "clients"):
        if key not in split:
            raise ValueError(f"분할 파일에 '{key}' 항목이 없습니다.")
    groups = [split["val"], *split["clients"]]
    all_indices = [i for group in groups for i in group]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("분할 파일에 중복된 인덱스가 있습니다.")
    if any(not 0 <= i < split["num_images"] for i in all_indices):
        raise ValueError("분할 파일에 범위를 벗어난 인덱스가 있습니다.")
    if any(len(c) == 0 for c in split["clients"]):
        raise ValueError("이미지가 없는 클라이언트가 있습니다.")


def save_split(split: dict, path: Path) -> None:
    validate_split(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split), encoding="utf-8")


def load_split(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"분할 파일이 없습니다: {path} (scripts/make_split.py로 만들 수 있습니다)")
    split = json.loads(path.read_text(encoding="utf-8"))
    validate_split(split)
    return split
