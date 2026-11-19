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
