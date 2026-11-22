import io
import base64
import copy
import torch


def serialize_weights(state_dict: dict) -> str:
    """
    PyTorch state_dict(또는 delta_weights)를 바이트 버퍼로 저장한 뒤
    JSON 전송이 가능한 Base64 ASCII 문자열로 인코딩합니다.
    """
    # GPU 텐서일 경우를 대비해 모든 텐서를 CPU로 이동 후 직렬화
    cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
    buffer = io.BytesIO()
    torch.save(cpu_state_dict, buffer)
    buffer.seek(0)
    encoded_str = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded_str


def deserialize_weights(encoded_str: str, device: str = 'cpu') -> dict:
    """
    Base64 문자열을 디코딩하여 PyTorch state_dict 형태로 복원합니다.
    """
    raw_bytes = base64.b64decode(encoded_str.encode('utf-8'))
    buffer = io.BytesIO(raw_bytes)
    state_dict = torch.load(buffer, map_location=device, weights_only=True)
    return state_dict


def compute_weight_delta(local_weights: dict, global_weights: dict) -> dict:
    """
    명세서 요구사항: 로컬 학습 후 가중치와 시작 전역 가중치의 차이(Delta W)를 계산합니다.
    Delta_W = W_local - W_global
    """
    delta = {}
    for key in global_weights.keys():
        delta[key] = local_weights[key].detach().cpu() - global_weights[key].detach().cpu()
    return delta


def validate_weight_tensor(weight_dict: dict, reference_keys: list) -> tuple[bool, str]:
    """
    제출된 업데이트(Delta W)의 무결성을 검증합니다.
    - 키(레이어 이름) 구조 일치 여부 확인
    - NaN 또는 Inf 포함 여부 확인
    """
    if set(weight_dict.keys()) != set(reference_keys):
        return False, "모델 레이어 키 구조 불일치 (Invalid model architecture)"

    for key, tensor in weight_dict.items():
        if not torch.is_tensor(tensor):
            return False, f"텐서가 아닌 데이터 포함: {key}"
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False, f"가중치에 NaN 또는 Inf 이상치 포함: {key}"

    return True, "OK"


# 단독 실행 시 정상 작동 여부 자가 테스트용 코드
if __name__ == "__main__":
    dummy_global = {"layer1.weight": torch.ones((2, 2)), "layer1.bias": torch.zeros(2)}
    dummy_local = {"layer1.weight": torch.ones((2, 2)) * 1.5, "layer1.bias": torch.ones(2) * 0.2}

    delta = compute_weight_delta(dummy_local, dummy_global)
    encoded = serialize_weights(delta)
    decoded = deserialize_weights(encoded)
    is_valid, msg = validate_weight_tensor(decoded, list(dummy_global.keys()))

    print(f"[테스트 결과] 검증 통과: {is_valid} ({msg})")
    print(f"[복원된 Delta Weight]:\n{decoded}")