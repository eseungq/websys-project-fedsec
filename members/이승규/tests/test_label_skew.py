from pathlib import Path

import pytest

from fedsec.data.split import class_counts, load_split, make_iid_split, make_label_skew_split, skew_score

MEMBER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMBER_ROOT.parents[1]
NUM_CLASSES = 5
# 작은 예제: 이미지 300장, 레이블은 i % 5 → 클래스마다 60장
LABELS = [i % NUM_CLASSES for i in range(300)]


def _base() -> dict:
    return make_iid_split(num_clients=4, val_size=60, seed=0, num_images=300)


def _skew(seed: int = 0, alpha: float = 0.5) -> dict:
    return make_label_skew_split(_base(), LABELS, alpha=alpha, seed=seed, num_classes=NUM_CLASSES)


def _pool(split: dict) -> list[int]:
    return sorted(i for client in split["clients"] for i in client)


def test_label_skew_keeps_val_pool_and_client_sizes():
    base = _base()

    skew = _skew()

    assert skew["val"] == base["val"]
    assert _pool(skew) == _pool(base)
    assert [len(c) for c in skew["clients"]] == [len(c) for c in base["clients"]]


@pytest.mark.parametrize("seed", range(20))
def test_row_and_column_sums_match_exactly(seed):
    base = _base()

    skew = _skew(seed=seed, alpha=0.1)  # 작은 alpha에서 반올림 보정이 많이 필요하다
    counts = class_counts(skew, LABELS, NUM_CLASSES)

    assert counts == skew["class_counts"]
    assert [sum(row) for row in counts] == [len(c) for c in base["clients"]]
    assert [sum(col) for col in zip(*counts)] == [sum(col) for col in zip(*class_counts(base, LABELS, NUM_CLASSES))]


def test_label_skew_is_more_skewed_than_iid():
    iid_score = skew_score(class_counts(_base(), LABELS, NUM_CLASSES))

    skew_score_value = skew_score(_skew()["class_counts"])

    assert skew_score_value > iid_score


def test_skew_score_numeric_example():
    # 전체 분포 (0.5, 0.5). 첫 클라이언트 (1, 0) → 거리 0.5, 둘째 (0, 1) → 0.5
    assert skew_score([[10, 0], [0, 10]]) == pytest.approx(0.5)
    assert skew_score([[5, 5], [5, 5]]) == pytest.approx(0.0)


def test_same_seed_gives_same_label_skew_split():
    assert _skew(seed=3) == _skew(seed=3)


def test_non_positive_alpha_is_rejected():
    with pytest.raises(ValueError):
        _skew(alpha=0.0)


def test_committed_label_skew_split_matches_iid_base():
    # 레이블 없이 확인할 수 있는 것: 같은 검증 세트, 같은 이미지 풀, 클라이언트별 4,500장
    iid = load_split(REPO_ROOT / "configs" / "splits" / "iid_seed0.json")
    skew = load_split(MEMBER_ROOT / "configs" / "splits" / "label_skew_a0.5_seed0.json")

    assert skew["val"] == iid["val"]
    assert _pool(skew) == _pool(iid)
    assert [sum(row) for row in skew["class_counts"]] == [4_500] * 10
