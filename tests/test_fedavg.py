import pytest
import torch

from fedsec.aggregation.fedavg import apply_update, weighted_average


def test_weighted_average_uses_sample_counts():
    updates = [{"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([4.0, 3.0])}]

    result = weighted_average(updates, weights=[100, 300])

    # (1·100 + 4·300) / 400 = 3.25, (0·100 + 3·300) / 400 = 2.25
    assert torch.allclose(result["w"], torch.tensor([3.25, 2.25]))


def test_equal_sample_counts_match_simple_mean():
    updates = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([2.0])}, {"w": torch.tensor([6.0])}]

    result = weighted_average(updates, weights=[4500, 4500, 4500])

    assert torch.allclose(result["w"], torch.tensor([3.0]))


def test_weighted_average_does_not_change_inputs():
    original = torch.tensor([1.0, 2.0])
    updates = [{"w": original.clone()}, {"w": torch.tensor([3.0, 4.0])}]

    weighted_average(updates, weights=[1, 1])

    assert torch.equal(updates[0]["w"], original)


def test_model_average_equals_update_average_from_same_start():
    start = {"w": torch.tensor([10.0, -2.0])}
    local_models = [{"w": torch.tensor([11.0, -1.0])}, {"w": torch.tensor([7.0, 2.0])}]
    weights = [1, 3]
    updates = [{"w": m["w"] - start["w"]} for m in local_models]

    via_updates = apply_update(start, weighted_average(updates, weights))
    via_models = weighted_average(local_models, weights)

    assert torch.allclose(via_updates["w"], via_models["w"])


@pytest.mark.parametrize("updates, weights", [
    ([], []),
    ([{"w": torch.zeros(1)}], [1, 2]),
    ([{"w": torch.zeros(1)}], [0]),
    ([{"w": torch.zeros(1)}, {"b": torch.zeros(1)}], [1, 1]),
])
def test_weighted_average_rejects_invalid_input(updates, weights):
    with pytest.raises(ValueError):
        weighted_average(updates, weights)


def test_update_norm_spans_all_parameters():
    from fedsec.aggregation.norms import update_norm

    update = {"w": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0])}

    assert update_norm(update) == pytest.approx(5.0)
