import base64
import io

import pytest
import torch

from fedsec.server.serialization import PayloadTooLarge, decode_state, encode_state


class NotAllowed:
    """weights_only=True 로드에서 거부되어야 하는 임의 객체."""


def _b64(obj) -> str:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_round_trip_is_bit_exact():
    state = {"w": torch.randn(3, 4, generator=torch.Generator().manual_seed(0)), "b": torch.tensor([1e-30, -0.0, 7.5])}

    restored = decode_state(encode_state(state))

    assert restored.keys() == state.keys()
    for key in state:
        assert restored[key].dtype == state[key].dtype
        assert torch.equal(restored[key], state[key])
    assert torch.signbit(restored["b"][1])  # -0.0의 부호까지 그대로


def test_encode_does_not_change_input():
    original = torch.tensor([1.0, 2.0])
    state = {"w": original.clone()}

    encode_state(state)

    assert torch.equal(state["w"], original)


def test_decoded_tensors_are_on_cpu():
    restored = decode_state(encode_state({"w": torch.zeros(2)}))

    assert restored["w"].device.type == "cpu"


def test_rejects_arbitrary_objects():
    text = _b64({"w": NotAllowed()})

    with pytest.raises(ValueError, match="읽을 수 없습니다"):
        decode_state(text)


@pytest.mark.parametrize("obj", [[torch.zeros(1)], {"w": 3}, {1: torch.zeros(1)}])
def test_rejects_non_state_dict(obj):
    with pytest.raises(ValueError, match="형태"):
        decode_state(_b64(obj))


def test_rejects_invalid_base64():
    with pytest.raises(ValueError, match="Base64"):
        decode_state("이건 base64가 아님")


def test_rejects_payload_over_size_limit():
    text = encode_state({"w": torch.zeros(1000)})  # 4,000 바이트 이상

    with pytest.raises(PayloadTooLarge):
        decode_state(text, max_bytes=1000)
