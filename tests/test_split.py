import pytest

from fedsec.data.split import make_iid_split, validate_split


def test_iid_split_sizes_and_no_overlap():
    split = make_iid_split(num_clients=10, val_size=5_000, seed=0)

    assert len(split["val"]) == 5_000
    assert [len(c) for c in split["clients"]] == [4_500] * 10
    validate_split(split)  # 중복·범위 오류가 없으면 통과


def test_same_seed_gives_same_split():
    assert make_iid_split(4, 20, seed=7, num_images=100) == make_iid_split(4, 20, seed=7, num_images=100)


def test_uneven_client_sizes_are_rejected():
    with pytest.raises(ValueError):
        make_iid_split(num_clients=3, val_size=0, seed=0, num_images=100)


def test_overlapping_indices_are_rejected():
    split = {"num_images": 10, "val": [0, 1], "clients": [[1, 2], [3, 4]]}

    with pytest.raises(ValueError):
        validate_split(split)
