"""서버 상태 한 곳: 현재 라운드, 모델 버전, 전역 가중치, 이번 라운드 대기 업데이트, 라운드·제외 기록.

라운드는 POST /aggregate마다 1씩 늘고, 모델 버전은 전역 모델이 실제로 바뀔 때만 늘어난다
(채택이 min_accepted개 미만이면 모델을 그대로 두므로 버전도 그대로).
"""

import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import torch

from fedsec.aggregation.fedavg import apply_update, weighted_average
from fedsec.aggregation.norms import update_norm
from fedsec.experiment.runtime import set_seed
from fedsec.logger.server_log import ServerLog
from fedsec.models.cnn import build_model
from fedsec.server.serialization import MAX_STATE_BYTES, State, encode_state
from fedsec.server.validation import Rejection, RoundContext, Submission, check_submission

# 명세 5단계: 채택이 3개 미만이면 모델을 갱신하지 않고 사유를 기록한다
MIN_ACCEPTED = 3

# (업데이트 목록, 표본 수 목록) → 집계된 업데이트. 기본은 FedAvg, 워커 C가 클리핑·중앙값으로 바꿔 끼운다.
Aggregator = Callable[[Sequence[State], Sequence[float]], State]


@dataclass(frozen=True)
class ServerPolicy:
    """라운드 운영 규칙 (확장 지점). 방어 함수에는 악성 ID 정답을 넘기지 않는다.

    participants·deadline은 한 라운드에 여러 번 불리므로 같은 라운드에는 항상 같은 값을 돌려줘야 한다
    (무작위 선택은 시드로 미리 만든 일정표를 라운드로 찾는 식으로).
    """
    aggregator: Aggregator = weighted_average
    min_accepted: int = MIN_ACCEPTED
    participants: Callable[[int], Sequence[int]] | None = None  # 라운드 → 참여자 ID. None이면 전원
    deadline: Callable[[int], float | None] | None = None       # 라운드 → 가상 마감 시각. None이면 마감 없음
    update_scale: Callable[[State], float] | None = None         # 업데이트별 방어 배율 (클리핑). 기록용, None이면 1

    @property
    def aggregator_name(self) -> str:
        return getattr(self.aggregator, "__name__", repr(self.aggregator))


def initial_global_state(seed: int, model_name: str) -> State:
    """서버 없는 실행기(run_fedavg)와 같은 순서(set_seed → build_model)로 초기 모델을 만든다."""
    set_seed(seed)
    model = build_model(model_name)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def load_global_state(path, reference: State) -> State:
    """저장된 전역 모델(models/round_XXX.pt)을 읽는다. 재실행의 시작 모델로 쓴다.

    weights_only=True로만 풀고, 키·shape·dtype이 reference(같은 설정의 초기 모델)와 같고 값이 모두 유한해야 한다.
    """
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not all(isinstance(v, torch.Tensor) for v in loaded.values()):
        raise ValueError(f"모델 파일이 파라미터 이름 → 텐서 형식이 아닙니다: {path}")
    if loaded.keys() != reference.keys():
        raise ValueError(f"모델 파일의 파라미터 이름이 설정의 모델과 다릅니다: {path}")
    wrong = [key for key in reference
             if loaded[key].shape != reference[key].shape or loaded[key].dtype != reference[key].dtype]
    if wrong:
        raise ValueError(f"모델 파일의 shape·dtype이 설정의 모델과 다릅니다: {wrong[:3]}")
    if not all(torch.isfinite(value).all() for value in loaded.values() if value.is_floating_point()):
        raise ValueError(f"모델 파일에 NaN·inf가 있습니다 (발산한 라운드의 모델은 재실행 시작점으로 쓸 수 없음): {path}")
    return {key: value.detach().clone() for key, value in loaded.items()}


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class ServerState:
    def __init__(self, global_state: State, num_clients: int, policy: ServerPolicy = ServerPolicy(),
                 log: ServerLog | None = None, max_update_bytes: int = MAX_STATE_BYTES, config: dict | None = None,
                 start_round: int = 1):
        """config: 시드·모델 이름 등 서버를 만든 설정. /status로 공개해 클라이언트 프로그램이 대조한다.
        start_round: 재실행(scripts/replay.py)에서 저장된 라운드 다음부터 시작할 때 쓴다. 모델 버전은 0부터 센다."""
        if num_clients <= 0:
            raise ValueError("num_clients는 1 이상이어야 합니다.")
        if policy.min_accepted < 1:
            raise ValueError("min_accepted는 1 이상이어야 합니다.")
        if type(start_round) is not int or start_round < 1:
            raise ValueError("start_round는 1 이상의 정수여야 합니다.")
        self.round = start_round
        self.model_version = 0
        self.global_state = {key: value.detach().cpu().clone() for key, value in global_state.items()}
        self.num_clients = num_clients
        self.policy = policy
        self.max_update_bytes = max_update_bytes
        self.config = dict(config or {})
        self.pending: dict[int, Submission] = {}
        self.history: list[dict] = []
        self.exclusions: list[dict] = []
        self._log = log
        self._lock = threading.Lock()
        self._encoded: tuple[int, str] | None = None  # (모델 버전, Base64) 같은 라운드에는 같은 스냅샷

    def participants(self) -> list[int]:
        chosen = self.policy.participants
        return sorted(chosen(self.round)) if chosen else list(range(self.num_clients))

    def deadline(self) -> float | None:
        return self.policy.deadline(self.round) if self.policy.deadline else None

    def model_snapshot(self) -> dict:
        with self._lock:
            if self._encoded is None or self._encoded[0] != self.model_version:
                self._encoded = (self.model_version, encode_state(self.global_state))
            return {"round": self.round, "model_version": self.model_version, "weights": self._encoded[1],
                    "participants": self.participants(), "deadline": self.deadline()}

    def submit(self, payload) -> Submission | Rejection:
        with self._lock:
            ctx = RoundContext(round=self.round, model_version=self.model_version, num_clients=self.num_clients,
                               participants=frozenset(self.participants()), deadline=self.deadline(),
                               accepted=frozenset(self.pending), reference=self.global_state,
                               max_bytes=self.max_update_bytes)
            outcome = check_submission(payload, ctx)
            if isinstance(outcome, Rejection):
                self._record_exclusion(payload, outcome)
            else:
                self.pending = {**self.pending, outcome.client_id: outcome}
            return outcome

    def reject_oversized(self, detail: str) -> Rejection:
        """요청 본문이 너무 커서 읽지 못한 제출 (client_id를 알 수 없음)."""
        rejection = Rejection("too_large", detail)
        with self._lock:
            self._record_exclusion(None, rejection)
        return rejection

    def _record_exclusion(self, payload, rejection: Rejection) -> None:
        fields = payload if isinstance(payload, dict) else {}
        record = {
            "time": _now(), "round": self.round, "client_id": _int_or_none(fields.get("client_id")),
            "submitted_round": _int_or_none(fields.get("round")),
            "submitted_version": _int_or_none(fields.get("model_version")),
            "reason": rejection.reason, "detail": rejection.detail,
        }
        self.exclusions = [*self.exclusions, record]
        if self._log:
            self._log.log_exclusion(record)

    def aggregate(self) -> dict:
        """대기 업데이트를 집계해 전역 모델을 갱신하고 다음 라운드로 넘어간다. 라운드 요약을 돌려준다."""
        with self._lock:
            t_start = time.perf_counter()
            accepted = [self.pending[cid] for cid in sorted(self.pending)]  # 제출 순서와 무관하게 ID 순으로 더한다
            skip_reason = self._skip_reason(len(accepted))
            # 새 상태와 기록을 모두 계산한 뒤에 한 번에 반영한다 (중간에 실패하면 서버 상태는 그대로)
            new_state, new_version, aggregate_norm = self.global_state, self.model_version, None
            if skip_reason is None:
                aggregated = self.policy.aggregator([s.update for s in accepted], [s.num_samples for s in accepted])
                new_state = apply_update(self.global_state, aggregated)
                new_version = self.model_version + 1
                aggregate_norm = round(update_norm(aggregated), 4)
            record = {**self._round_record(accepted, skip_reason, time.perf_counter() - t_start),
                      "model_version": new_version, "aggregate_norm": aggregate_norm}
            if self._log:
                self._log.log_round(record)   # 기록에 실패하면 상태를 바꾸지 않는다 (서버 기록과 상태가 어긋나지 않게)
            self.global_state, self.model_version = new_state, new_version
            self.history = [*self.history, record]
            self.round += 1
            self.pending = {}
            return record

    def _skip_reason(self, num_accepted: int) -> str | None:
        if num_accepted == 0:
            return "no_accepted"
        if num_accepted < self.policy.min_accepted:
            return "too_few_accepted"
        return None

    def _defended_norms(self, norms: list[float], accepted: list[Submission]) -> list[float]:
        """방어(클리핑)가 업데이트마다 적용된 뒤의 노름. 배율이 없는 방어(FedAvg·중앙값)는 그대로."""
        scale = self.policy.update_scale
        factors = [1.0 if scale is None else scale(s.update) for s in accepted]
        return [round(norm * factor, 4) for norm, factor in zip(norms, factors)]

    def _round_record(self, accepted: list[Submission], skip_reason: str | None, seconds: float) -> dict:
        excluded = [e for e in self.exclusions if e["round"] == self.round]
        norms = [update_norm(s.update) for s in accepted]
        return {
            "round": self.round,
            "model_version": self.model_version,
            "updated": skip_reason is None,
            "skip_reason": skip_reason,
            "participants": self.participants(),
            "accepted_clients": [s.client_id for s in accepted],
            "num_accepted": len(accepted),
            "num_excluded": len(excluded),
            "exclusion_reasons": dict(Counter(e["reason"] for e in excluded)),
            "total_samples": sum(s.num_samples for s in accepted),
            "update_norms": [round(norm, 4) for norm in norms],
            "defended_norms": self._defended_norms(norms, accepted),
            "aggregator": self.policy.aggregator_name,
            "aggregate_sec": round(seconds, 4),
        }

    def status(self) -> dict:
        with self._lock:
            return {
                "round": self.round, "model_version": self.model_version,
                "pending": len(self.pending), "pending_clients": sorted(self.pending),
                "num_clients": self.num_clients, "min_accepted": self.policy.min_accepted,
                "aggregator": self.policy.aggregator_name,
                "config": self.config,
                "history": self.history, "exclusions": self.exclusions,
            }


def _int_or_none(value) -> int | None:
    """제외 기록에는 검증 전 값 중 정수만 남긴다 (큰 문자열·CSV 수식이 기록에 그대로 들어가지 않게)."""
    return value if type(value) is int else None
