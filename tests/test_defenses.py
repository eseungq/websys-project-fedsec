import pytest
import torch

from fedsec.aggregation.fedavg import weighted_average
from fedsec.aggregation.norms import update_norm
from fedsec.defenses.clipping import clip_scale, clip_update, clipped_fedavg
from fedsec.defenses.median import coordinate_median

# --- 노름 클리핑 ---

def test_update_within_limit_is_unchanged():
    update = {"w": torch.tensor([0.3, 0.4])}

    result = clip_update(update, max_norm=1.0)

    # 노름 0.5 ≤ 1 → 그대로
    assert clip_scale(update, max_norm=1.0) == 1.0
    assert torch.allclose(result["w"], torch.tensor([0.3, 0.4]))


def test_update_over_limit_is_scaled_to_limit():
    update = {"w": torch.tensor([3.0, 4.0])}

    result = clip_update(update, max_norm=1.0)

    # 노름 5 > 1 → 배율 1/5 → (0.6, 0.8), 노름 1
    assert clip_scale(update, max_norm=1.0) == pytest.approx(0.2)
    assert torch.allclose(result["w"], torch.tensor([0.6, 0.8]))
    assert update_norm(result) == pytest.approx(1.0)


def test_norm_spans_all_parameter_tensors():
    update = {"w": torch.tensor([3.0]), "b": torch.tensor([4.0])}

    result = clip_update(update, max_norm=2.5)

    # 이어 붙인 (3, 4)의 노름 5 → 배율 0.5 → w 1.5, b 2
    assert torch.allclose(result["w"], torch.tensor([1.5]))
    assert torch.allclose(result["b"], torch.tensor([2.0]))


def test_clipping_does_not_change_inputs():
    original = torch.tensor([3.0, 4.0])
    update = {"w": original.clone()}

    clip_update(update, max_norm=1.0)

    assert torch.equal(update["w"], original)


@pytest.mark.parametrize("update, max_norm", [
    ({"w": torch.tensor([1.0])}, 0.0),
    ({"w": torch.tensor([1.0])}, -1.0),
    ({"w": torch.tensor([1.0])}, float("nan")),
    ({}, 1.0),
    ({"w": torch.tensor([float("inf")])}, 1.0),
])
def test_invalid_clipping_input_raises(update, max_norm):
    with pytest.raises(ValueError):
        clip_update(update, max_norm)


def test_clipped_fedavg_clips_then_weights_by_samples():
    updates = [{"w": torch.tensor([3.0, 4.0])}, {"w": torch.tensor([0.6, 0.0])}]

    result = clipped_fedavg(updates, weights=[1, 3], max_norm=1.0)

    # 첫째 (3, 4) → (0.6, 0.8), 둘째 노름 0.6이라 그대로
    # ((0.6·1 + 0.6·3) / 4, (0.8·1 + 0·3) / 4) = (0.6, 0.2)
    assert torch.allclose(result["w"], torch.tensor([0.6, 0.2]))


def test_clipped_fedavg_equals_fedavg_when_nothing_is_clipped():
    updates = [{"w": torch.tensor([0.1])}, {"w": torch.tensor([0.5])}]

    result = clipped_fedavg(updates, weights=[1, 1], max_norm=10.0)

    # 모두 상한 이하 → 일반 FedAvg와 같음: (0.1 + 0.5) / 2 = 0.3
    assert torch.allclose(result["w"], weighted_average(updates, [1, 1])["w"])


def test_clipped_fedavg_with_no_updates_raises():
    with pytest.raises(ValueError):
        clipped_fedavg([], weights=[], max_norm=1.0)


# --- 좌표별 중앙값 ---

def test_median_of_odd_count_is_middle_value():
    updates = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([5.0])}, {"w": torch.tensor([3.0])}]

    result = coordinate_median(updates)

    # 정렬 1, 3, 5 → 가운데 3
    assert torch.allclose(result["w"], torch.tensor([3.0]))


def test_median_of_even_count_is_mean_of_two_middle_values():
    updates = [{"w": torch.tensor([v])} for v in (10.0, 1.0, 4.0, 2.0)]

    result = coordinate_median(updates)

    # 정렬 1, 2, 4, 10 → (2 + 4) / 2 = 3 (torch.median이라면 아래쪽 2)
    assert torch.allclose(result["w"], torch.tensor([3.0]))


def test_median_is_taken_per_coordinate_and_parameter():
    updates = [
        {"w": torch.tensor([[1.0, 9.0]]), "b": torch.tensor([0.0])},
        {"w": torch.tensor([[2.0, 8.0]]), "b": torch.tensor([-6.0])},
        {"w": torch.tensor([[3.0, 7.0]]), "b": torch.tensor([6.0])},
    ]

    result = coordinate_median(updates)

    # 첫 좌표 {1, 2, 3} → 2, 둘째 좌표 {9, 8, 7} → 8, b {0, −6, 6} → 0. 좌표마다 다른 업데이트에서 옴
    assert torch.allclose(result["w"], torch.tensor([[2.0, 8.0]]))
    assert torch.allclose(result["b"], torch.tensor([0.0]))


def test_single_outlier_barely_moves_median():
    updates = [{"w": torch.tensor([v])} for v in (1.0, 2.0, 3.0, 1000.0)]

    result = coordinate_median(updates)

    # 평균은 251.5지만 중앙값은 (2 + 3) / 2 = 2.5
    assert torch.allclose(result["w"], torch.tensor([2.5]))


def test_median_resists_two_colluders_among_ten():
    updates = [{"w": torch.tensor([1.0])} for _ in range(8)] + [{"w": torch.tensor([-5.0])}] * 2

    median = coordinate_median(updates)
    mean = weighted_average(updates, [1] * 10)

    # 정상 8개 1, 공모 2개 −5: 평균 (8 − 10) / 10 = −0.2, 중앙값 정렬 5·6번째 (1 + 1) / 2 = 1
    assert torch.allclose(mean["w"], torch.tensor([-0.2]))
    assert torch.allclose(median["w"], torch.tensor([1.0]))


def test_median_does_not_change_inputs():
    original = torch.tensor([3.0, 1.0])
    updates = [{"w": original.clone()}, {"w": torch.tensor([1.0, 3.0])}]

    coordinate_median(updates)

    assert torch.equal(updates[0]["w"], original)


@pytest.mark.parametrize("updates", [
    [],
    [{"w": torch.zeros(1)}, {"b": torch.zeros(1)}],
    [{"w": torch.zeros(1)}, {"w": torch.zeros(2)}],
])
def test_invalid_median_input_raises(updates):
    with pytest.raises(ValueError):
        coordinate_median(updates)
