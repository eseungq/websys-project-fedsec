"""POST /update 제출 검증. 통과하면 Submission, 아니면 제외 사유(Rejection)를 돌려주는 순수 함수."""

from dataclasses import dataclass

import torch

from fedsec.server.serialization import PayloadTooLarge, State, decode_state

# 제외 사유 코드 → HTTP 상태 코드
REASON_STATUS = {
    "bad_json": 400,          # 본문이 JSON 객체가 아님
    "missing_field": 400,     # 필수 항목 없음 (마감이 있는 라운드의 virtual_finish_time 포함)
    "bad_type": 400,          # 항목 자료형이 틀림
    "bad_num_samples": 400,   # 표본 수가 양의 정수가 아님
    "unknown_client": 403,    # 등록되지 않은 클라이언트 ID
    "not_selected": 403,      # 이번 라운드 참여자가 아님
    "round_mismatch": 409,    # 서버의 현재 라운드와 다름
    "version_mismatch": 409,  # 서버의 현재 모델 버전과 다름
    "duplicate": 409,         # 이번 라운드에 이미 채택된 제출이 있음
    "late": 409,              # 가상 완료 시각이 마감을 넘음
    "too_large": 413,         # 요청·가중치가 크기 상한을 넘음
    "bad_payload": 400,       # Base64·torch 파일 형식이 아님
    "bad_shape": 400,         # 파라미터 이름·모양·자료형이 전역 모델과 다름
    "non_finite": 400,        # NaN 또는 Inf 포함
}

REQUIRED_FIELDS = ("client_id", "round", "model_version", "num_samples", "delta_weights")
INT_FIELDS = ("client_id", "round", "model_version")


@dataclass(frozen=True)
class Rejection:
    reason: str
    detail: str

    @property
    def status(self) -> int:
        return REASON_STATUS[self.reason]


@dataclass(frozen=True)
class Submission:
    client_id: int
    num_samples: int
    update: State
    virtual_finish_time: float | None


@dataclass(frozen=True)
class RoundContext:
    """검증에 필요한 서버 상태의 한 시점 사본."""
    round: int
    model_version: int
    num_clients: int
    participants: frozenset[int]
    deadline: float | None
    accepted: frozenset[int]   # 이번 라운드에 이미 채택된 클라이언트
    reference: State           # 전역 모델 (이름·모양·자료형 비교용)
    max_bytes: int


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_fields(payload) -> Rejection | None:
    if not isinstance(payload, dict):
        return Rejection("bad_json", "요청 본문은 JSON 객체여야 합니다.")
    missing = [key for key in REQUIRED_FIELDS if key not in payload]
    if missing:
        return Rejection("missing_field", f"필수 항목이 없습니다: {', '.join(missing)}")
    wrong = [key for key in INT_FIELDS if not _is_int(payload[key])]
    if wrong:
        return Rejection("bad_type", f"정수여야 합니다: {', '.join(wrong)}")
    if not isinstance(payload["delta_weights"], str):
        return Rejection("bad_type", "delta_weights는 Base64 문자열이어야 합니다.")
    finish = payload.get("virtual_finish_time")
    if finish is not None and (isinstance(finish, bool) or not isinstance(finish, (int, float))):
        return Rejection("bad_type", "virtual_finish_time은 숫자여야 합니다.")
    return None


def _check_round(payload: dict, ctx: RoundContext) -> Rejection | None:
    cid = payload["client_id"]
    finish = payload.get("virtual_finish_time")
    checks = [
        (0 <= cid < ctx.num_clients, "unknown_client", f"등록되지 않은 클라이언트입니다: {cid}"),
        (payload["round"] == ctx.round, "round_mismatch", f"현재 라운드는 {ctx.round}입니다 (제출 {payload['round']})."),
        (payload["model_version"] == ctx.model_version, "version_mismatch",
         f"현재 모델 버전은 {ctx.model_version}입니다 (제출 {payload['model_version']})."),
        (cid in ctx.participants, "not_selected", f"클라이언트 {cid}는 라운드 {ctx.round}의 참여자가 아닙니다."),
        (cid not in ctx.accepted, "duplicate", f"클라이언트 {cid}의 업데이트가 이번 라운드에 이미 채택되었습니다."),
        (_is_int(payload["num_samples"]) and payload["num_samples"] > 0, "bad_num_samples",
         f"num_samples는 양의 정수여야 합니다: {payload['num_samples']!r}"),
        (ctx.deadline is None or finish is not None, "missing_field", "마감이 있는 라운드에는 virtual_finish_time이 필요합니다."),
        (ctx.deadline is None or finish is None or finish <= ctx.deadline, "late",
         f"가상 완료 시각 {finish}이 마감 {ctx.deadline}을 넘었습니다."),
    ]
    for ok, reason, detail in checks:
        if not ok:
            return Rejection(reason, detail)
    return None


def _check_update(text: str, ctx: RoundContext) -> State | Rejection:
    try:
        update = decode_state(text, ctx.max_bytes)
    except PayloadTooLarge as error:
        return Rejection("too_large", str(error))
    except ValueError as error:
        return Rejection("bad_payload", str(error))
    if update.keys() != ctx.reference.keys():
        return Rejection("bad_shape", "파라미터 이름이 전역 모델과 다릅니다.")
    for key, ref in ctx.reference.items():
        if update[key].shape != ref.shape or update[key].dtype != ref.dtype:
            return Rejection("bad_shape", f"{key}: {tuple(update[key].shape)} {update[key].dtype}, "
                                          f"전역 모델은 {tuple(ref.shape)} {ref.dtype}")
    bad = [key for key, value in update.items() if not torch.isfinite(value).all()]
    if bad:
        return Rejection("non_finite", f"NaN 또는 Inf가 있습니다: {', '.join(bad)}")
    return update


def check_submission(payload, ctx: RoundContext) -> Submission | Rejection:
    """검사 순서: 형식 → 클라이언트·라운드·버전·참여·중복·표본 수·마감 → 가중치 (무거운 복원은 마지막)."""
    rejection = _check_fields(payload) or _check_round(payload, ctx)
    if rejection:
        return rejection
    update = _check_update(payload["delta_weights"], ctx)
    if isinstance(update, Rejection):
        return update
    finish = payload.get("virtual_finish_time")
    return Submission(client_id=payload["client_id"], num_samples=payload["num_samples"], update=update,
                      virtual_finish_time=None if finish is None else float(finish))
