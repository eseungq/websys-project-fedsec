"""state_dict ↔ Base64 문자열. 서버와 클라이언트 프로그램이 HTTP JSON으로 가중치를 주고받을 때 쓴다.

받는 쪽은 torch.load(weights_only=True)로만 푼다. 텐서·기본 자료형 외의 객체는 거부되므로
제출된 파일로 임의 코드를 실행할 수 없다.
"""

import base64
import binascii
import io

import torch

State = dict[str, torch.Tensor]

# 직렬화된 state_dict 상한. simple_cnn은 약 1.3MB라서 10배 이상 여유를 둔다.
MAX_STATE_BYTES = 16 * 1024 * 1024


class PayloadTooLarge(ValueError):
    """직렬화된 가중치가 상한을 넘는다."""


def encode_state(state: State) -> str:
    """텐서를 CPU로 옮겨 torch.save → Base64(ASCII)로 만든다. 입력은 바꾸지 않는다."""
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in state.items()}, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_state(text: str, max_bytes: int = MAX_STATE_BYTES) -> State:
    """Base64 문자열을 CPU 텐서 dict로 되돌린다. 형식이 틀리면 ValueError, 너무 크면 PayloadTooLarge."""
    if not isinstance(text, str):
        raise ValueError("가중치는 Base64 문자열이어야 합니다.")
    if len(text) * 3 // 4 > max_bytes:
        raise PayloadTooLarge(f"가중치가 상한 {max_bytes} 바이트를 넘습니다.")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"Base64 형식이 아닙니다: {error}") from error
    try:
        loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:  # 손상·악성 파일은 torch 내부에서 여러 종류의 예외를 낸다
        raise ValueError(f"가중치 파일을 읽을 수 없습니다: {type(error).__name__}: {error}") from error
    if not isinstance(loaded, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in loaded.items()):
        raise ValueError("가중치는 {이름: 텐서} 형태여야 합니다.")
    return loaded
