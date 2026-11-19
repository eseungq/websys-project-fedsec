import inspect

import pytest
import torch

from fedsec.attacks.dlg import add_gaussian_noise, pixel_mse, reconstruct, victim_gradients
from fedsec.models.lenet import INIT_RANGE, build_dlg_lenet

IMAGE_SHAPE = (1, 3, 32, 32)


def _victim(seed: int = 0) -> tuple:
    """테스트용 피해자: 고정 시드의 [0, 1] 무작위 이미지 한 장과 레이블 3."""
    image = torch.rand(IMAGE_SHAPE, generator=torch.Generator().manual_seed(seed))
    model = build_dlg_lenet(seed=0)
    return model, image, victim_gradients(model, image, torch.tensor([3]))


def test_lenet_init_is_within_uniform_range():
    model = build_dlg_lenet(seed=0)

    values = torch.cat([p.detach().flatten() for p in model.parameters()])

    assert values.min() >= -INIT_RANGE and values.max() <= INIT_RANGE
    assert values.std() > 0.25  # U(-0.5, 0.5)의 표준편차 ≈ 0.289, 기본 초기화(훨씬 작음)가 아님


def test_lenet_init_is_reproducible_by_seed():
    a, b, c = build_dlg_lenet(1), build_dlg_lenet(1), build_dlg_lenet(2)

    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert not all(torch.equal(x, z) for x, z in zip(a.parameters(), c.parameters()))


def test_zero_noise_returns_same_gradients():
    grads = [torch.tensor([1.0, -2.0]), torch.tensor([[0.5]])]

    noisy = add_gaussian_noise(grads, sigma=0.0, seed=0)

    assert all(torch.equal(n, g) for n, g in zip(noisy, grads))


def test_noise_is_reproducible_and_seed_dependent():
    grads = [torch.zeros(1000)]

    first = add_gaussian_noise(grads, sigma=0.1, seed=7)[0]
    again = add_gaussian_noise(grads, sigma=0.1, seed=7)[0]
    other = add_gaussian_noise(grads, sigma=0.1, seed=8)[0]

    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert first.std().item() == pytest.approx(0.1, rel=0.1)


def test_noise_does_not_change_input():
    original = torch.tensor([1.0, 2.0, 3.0])
    grads = [original.clone()]

    add_gaussian_noise(grads, sigma=1.0, seed=0)

    assert torch.equal(grads[0], original)


@pytest.mark.parametrize("sigma", [-0.1, float("nan"), float("inf")])
def test_noise_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        add_gaussian_noise([torch.zeros(2)], sigma=sigma, seed=0)


def test_reconstruct_takes_no_original_image_or_label():
    params = set(inspect.signature(reconstruct).parameters)

    assert params == {"model", "gradients", "image_shape", "iterations", "dummy_seed", "record_every"}


def test_reconstruct_output_shapes_and_trajectory_marks():
    model, _, grads = _victim()

    result = reconstruct(model, grads, IMAGE_SHAPE, iterations=3, dummy_seed=0, record_every=2)

    assert result.image.shape == IMAGE_SHAPE
    assert result.label_logits.shape == (1, 10)
    assert [s.iteration for s in result.trajectory] == [0, 2, 3]  # 시작, 2회마다, 마지막
    assert result.iterations == 3 and not result.diverged


def test_reconstruct_does_not_change_model():
    model, _, grads = _victim()
    before = [p.detach().clone() for p in model.parameters()]

    reconstruct(model, grads, IMAGE_SHAPE, iterations=2, dummy_seed=0)

    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before))
    assert all(p.grad is None for p in model.parameters())


def test_same_dummy_seed_gives_same_start_regardless_of_noise():
    model, _, grads = _victim()
    noisy = add_gaussian_noise(grads, sigma=0.1, seed=0)

    clean_start = reconstruct(model, grads, IMAGE_SHAPE, iterations=0, dummy_seed=5).trajectory[0].image
    noisy_start = reconstruct(model, noisy, IMAGE_SHAPE, iterations=0, dummy_seed=5).trajectory[0].image

    assert torch.equal(clean_start, noisy_start)


def test_reconstruction_without_noise_reduces_mse():
    model, image, grads = _victim()

    result = reconstruct(model, grads, IMAGE_SHAPE, iterations=30, dummy_seed=0, record_every=30)

    start_mse = pixel_mse(result.trajectory[0].image, image)
    assert pixel_mse(result.image, image) < 0.5 * start_mse


def test_reconstruct_rejects_mismatched_gradients():
    model, _, grads = _victim()

    with pytest.raises(ValueError):
        reconstruct(model, grads[:-1], IMAGE_SHAPE, iterations=1, dummy_seed=0)


def test_pixel_mse_clipped_and_unclipped():
    original = torch.tensor([0.0, 0.5, 1.0, 0.2])
    reconstructed = torch.tensor([-1.0, 0.5, 3.0, 0.4])

    # 자른 값 [0, 0.5, 1, 0.4] → 오차 [0, 0, 0, 0.2] → 0.04 / 4 = 0.01
    assert pixel_mse(reconstructed, original) == pytest.approx(0.01)
    # 자르지 않으면 오차 [-1, 0, 2, 0.2] → (1 + 4 + 0.04) / 4 = 1.26
    assert pixel_mse(reconstructed, original, clip=False) == pytest.approx(1.26)


def test_pixel_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        pixel_mse(torch.zeros(3), torch.zeros(4))


def test_victim_gradients_requires_single_image():
    model = build_dlg_lenet(0)

    with pytest.raises(ValueError):
        victim_gradients(model, torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))


# scripts/run_dlg.py: 설정 검증과 "원본은 공격 입력이 아니라 평가에만"

def _run_dlg_module():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_dlg
    return run_dlg


def _dlg_config(**changes):
    from dataclasses import replace

    run_dlg = _run_dlg_module()
    base = run_dlg.DLGConfig(split="s", data_root="d", client=0, images=(1, 2), model_seed=0, dummy_seed=0,
                             noise_seed=0, sigmas=(0.0, 0.01), iterations=(5,), record_every=1)
    return replace(base, **changes)


@pytest.mark.parametrize("changes", [
    {"client": -1},                         # 예전에는 마지막 클라이언트를 조용히 골랐음
    {"sigmas": (0.01, 0.01)},               # 같은 σ면 복원 이미지 파일이 덮어써짐
    {"sigmas": (float("inf"),)},
    {"iterations": (5, 5)},
])
def test_dlg_config_rejects_bad_values(changes):
    with pytest.raises(ValueError):
        _run_dlg_module()._validated(_dlg_config(**changes))


def test_run_case_gives_attack_no_original_image_or_label(monkeypatch):
    run_dlg = _run_dlg_module()
    model, image, grads = _victim()
    seen = []

    def spy(*args, **kwargs):
        seen.extend([*args, *kwargs.values()])
        return reconstruct(*args, **kwargs)

    monkeypatch.setattr(run_dlg, "reconstruct", spy)
    run_dlg.run_case(model, grads, image, 3, 0.0, 2, _dlg_config())

    tensors = [a for a in seen if isinstance(a, torch.Tensor)]
    assert not any(t is image or (t.shape == image.shape and torch.equal(t, image)) for t in tensors)
    assert tuple(image.shape) in seen          # 이미지 크기만 넘김


def test_reconstruct_stops_and_marks_diverged_on_inf_gradients():
    model, _, grads = _victim()

    result = reconstruct(model, [torch.full_like(g, float("inf")) for g in grads], IMAGE_SHAPE, 50, dummy_seed=0)

    assert result.diverged and result.iterations < 50
