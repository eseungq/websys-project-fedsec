import pytest
import torch

from fedsec.attacks.collusion import collusion_vectors


def test_no_participating_attacker_means_no_attack():
    updates = {0: {"w": torch.tensor([1.0])}, 2: {"w": torch.tensor([2.0])}}

    result = collusion_vectors(updates, malicious_ids={1, 3})

    # 참여자 {0, 2} 중 악성 없음 → 공격 없음
    assert result == {}


def test_single_attacker_submits_own_update_times_minus_five():
    updates = {0: {"w": torch.tensor([9.0])}, 3: {"w": torch.tensor([1.0, -2.0])}}

    result = collusion_vectors(updates, malicious_ids={3, 7})

    # 악성 3만 참여: (1, −2) × −5 = (−5, 10)
    assert result.keys() == {3}
    assert torch.allclose(result[3]["w"], torch.tensor([-5.0, 10.0]))


def test_two_attackers_submit_same_scaled_mean():
    updates = {
        1: {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])},
        4: {"w": torch.tensor([3.0, 6.0]), "b": torch.tensor([2.0])},
        5: {"w": torch.tensor([100.0, 100.0]), "b": torch.tensor([100.0])},
    }

    result = collusion_vectors(updates, malicious_ids=[1, 4])

    # 평균 w = (2, 4), b = 1 → × −5 = (−10, −20), −5. 둘 다 같은 값
    assert result.keys() == {1, 4}
    for cid in (1, 4):
        assert torch.allclose(result[cid]["w"], torch.tensor([-10.0, -20.0]))
        assert torch.allclose(result[cid]["b"], torch.tensor([-5.0]))


def test_mean_is_simple_not_sample_weighted():
    updates = {1: {"w": torch.tensor([0.0])}, 4: {"w": torch.tensor([4.0])}}

    result = collusion_vectors(updates, malicious_ids={1, 4})

    # 표본 수와 무관하게 (0 + 4) / 2 = 2 → × −5 = −10
    assert torch.allclose(result[1]["w"], torch.tensor([-10.0]))


def test_scale_can_be_changed():
    updates = {2: {"w": torch.tensor([1.0, 3.0])}}

    result = collusion_vectors(updates, malicious_ids={2}, scale=-2.0)

    # (1, 3) × −2 = (−2, −6)
    assert torch.allclose(result[2]["w"], torch.tensor([-2.0, -6.0]))


def test_does_not_change_inputs_and_outputs_are_separate_tensors():
    original = torch.tensor([1.0, 2.0])
    updates = {1: {"w": original.clone()}, 4: {"w": torch.tensor([3.0, 4.0])}}

    result = collusion_vectors(updates, malicious_ids={1, 4})
    result[1]["w"].zero_()

    # 입력은 그대로, 한 악성의 벡터를 바꿔도 다른 악성의 벡터는 그대로 (평균 (2, 3) × −5)
    assert torch.equal(updates[1]["w"], original)
    assert torch.allclose(result[4]["w"], torch.tensor([-10.0, -15.0]))


def test_attackers_with_different_parameter_names_raise():
    updates = {1: {"w": torch.zeros(1)}, 4: {"b": torch.zeros(1)}}

    with pytest.raises(ValueError):
        collusion_vectors(updates, malicious_ids={1, 4})
