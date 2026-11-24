"""가상 클라이언트 실행 프로그램: 한 프로세스에서 클라이언트들을 순서대로 학습시키며 서버와 REST API로 통신한다.

라운드마다: GET /model → 참여자를 ID 순으로 local_update (전원 학습)
→ 공격을 켰으면 악성 참여자의 업데이트를 공모 벡터로 바꿈 → 가상 완료 시각을 붙여 전원 POST /update
→ POST /aggregate → GET /model(갱신된 전역 모델)로 검증 정확도 평가. 이 모델이 다음 라운드의 시작점이다.
검증 평가는 전체·클래스별·클라이언트별(분포 가중) 정확도를 남기고, 전역 모델은 save_every 라운드마다 저장한다.
초기 모델은 서버가 만들고(시드 고정) 클라이언트는 받기만 한다.

악성·지연 ID와 공격 여부는 클라이언트 프로그램만 안다(ClientPlan). 서버에는 넘기지 않는다.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import torch

from fedsec.aggregation.norms import update_norm
from fedsec.attacks.collusion import collusion_vectors
from fedsec.client.local_train import ClientResult, TrainingDiverged, local_update
from fedsec.data.cifar10 import TensorData, load_cifar10
from fedsec.data.split import load_split
from fedsec.environments.acceptance import RoundDecision, judge_round
from fedsec.environments.config import EnvConfig
from fedsec.environments.schedule import delayed_clients, malicious_clients
from fedsec.environments.virtual_time import VirtualTiming, check_delay_setting, round_elapsed, virtual_finish_times
from fedsec.experiment.config import AttackDefenseConfig, ExperimentConfig
from fedsec.experiment.evaluate import Evaluation, evaluate_detailed
from fedsec.experiment.metrics import class_rows, client_acc_rows, label_distribution
from fedsec.experiment.runtime import client_generator, environment_info, pick_device, set_seed, timed_now
from fedsec.logger.run_log import RunLog
from fedsec.models.cnn import build_model
from fedsec.server.serialization import State, decode_state, encode_state

REQUEST_TIMEOUT_SEC = 120
# 실험 설계상 정상인 제외 사유. 이 밖의 사유(non_finite, version_mismatch 등)는 버그이므로 실행을 멈춘다
EXPECTED_EXCLUSIONS = frozenset({"late"})
DEFAULT_SAVE_EVERY = 5      # 전역 모델 저장 간격 (라운드). 0이면 라운드 0·마지막만
# 서버 /status의 config와 맞춰 볼 방어 설정 항목
DEFENSE_KEYS = ("aggregator", "clip_norm", "clip_before_median")
# 라운드마다 화면에 보여 줄 항목 (전체는 rounds.csv)
PRINT_KEYS = ("round", "val_acc", "val_loss", "num_selected", "num_accepted", "num_late", "malicious_accepted",
              "accepted_malicious_ratio", "attack_applied", "updated", "virtual_elapsed", "total_sec")


class ServerApi:
    """서버 REST API 호출. 응답 형식 {"ok", "data", "error"}를 풀어 data를 돌려준다."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        response = self._session.request(method, self.base_url + path, json=body, timeout=REQUEST_TIMEOUT_SEC)
        try:
            envelope = response.json()
        except ValueError as error:
            raise RuntimeError(f"{method} {path}: JSON이 아닌 응답 (HTTP {response.status_code})") from error
        if path != "/update" and not envelope.get("ok"):
            raise RuntimeError(f"{method} {path} 실패 (HTTP {response.status_code}): {envelope.get('error')}")
        return envelope

    def status(self) -> dict:
        return self._call("GET", "/status")["data"]

    def model(self) -> dict:
        return self._call("GET", "/model")["data"]

    def submit(self, payload: dict) -> dict:
        """제외되어도 예외 없이 응답 전체를 돌려준다 (ok=False, data.reason)."""
        return self._call("POST", "/update", payload)

    def aggregate(self) -> dict:
        return self._call("POST", "/aggregate", {})["data"]


@dataclass(frozen=True)
class ClientPlan:
    """클라이언트 쪽 실험 일정. 같은 환경·시드면 공격·집계 방식이 달라도 같은 악성·지연 ID."""
    experiment_id: str
    env_name: str
    malicious: tuple[int, ...]
    delayed: tuple[int, ...]
    timing: VirtualTiming
    per_round: int | None   # 서버의 라운드 참여자 수 (None이면 전원). 서버 설정과 대조하고 결과 재사용 판정에 씀
    min_accepted: int       # 서버의 모델 갱신 최소 채택 수. 위와 같음
    attack: bool
    attack_scale: float
    defense: dict           # 서버 설정과 대조할 방어 설정 (DEFENSE_KEYS)
    unmasked: tuple[int, ...] = ()  # 재실행에서 조작만 해제한 악성 ID (악성 표시는 그대로, 정상 업데이트를 제출)

    def to_dict(self) -> dict:
        """unmasked는 재실행에서 해제했을 때만 남긴다 (기존 결과의 config.yaml과 비교가 어긋나지 않게)."""
        data = {**asdict(self), "malicious": list(self.malicious), "delayed": list(self.delayed)}
        if self.unmasked:
            return {**data, "unmasked": list(self.unmasked)}
        return {key: value for key, value in data.items() if key != "unmasked"}

    @property
    def attackers(self) -> tuple[int, ...]:
        """실제로 공격 벡터를 제출하는 악성 ID (조작 해제한 ID는 뺀다)."""
        return tuple(cid for cid in self.malicious if cid not in self.unmasked)


def client_plan(experiment_id: str, env: EnvConfig, attack_defense: AttackDefenseConfig, seed: int,
                num_clients: int) -> ClientPlan:
    malicious = malicious_clients(seed, num_clients, env.num_malicious, env.malicious_ids)
    exclude = malicious if env.delayed_avoid_malicious else ()
    delayed = delayed_clients(seed, num_clients, env.num_delayed, env.delayed_ids, exclude=exclude)
    check_plan(env, attack_defense.attack, malicious, delayed, num_clients)
    ad = attack_defense.to_dict()
    return ClientPlan(experiment_id=experiment_id, env_name=env.name, malicious=malicious, delayed=delayed,
                      timing=env.timing, per_round=env.per_round, min_accepted=env.min_accepted,
                      attack=attack_defense.attack, attack_scale=attack_defense.attack_scale,
                      defense={key: ad[key] for key in DEFENSE_KEYS})


def check_plan(env: EnvConfig, attack: bool, malicious: tuple[int, ...], delayed: tuple[int, ...],
               num_clients: int) -> None:
    """설정 조합이 실험 이름과 다른 실험을 만들지 않는지 확인한다 (예: 악성 0명인 공격 실험)."""
    per_round = env.per_round or num_clients
    # 지연 클라이언트가 가장 적게 뽑힌 라운드의 채택 가능 인원 (지연은 항상 마감을 넘음: check_delay_setting)
    reachable = per_round - max(0, len(delayed) - (num_clients - per_round))
    checks = [
        (not attack or len(malicious) > 0, "공격을 켰는데 악성 클라이언트가 없습니다."),
        (not env.delayed_avoid_malicious or not set(delayed) & set(malicious),
         f"지연 {list(delayed)}와 악성 {list(malicious)}이 겹칩니다 (avoid_malicious: true). 직접 지정한 ids를 확인하세요."),
        (per_round <= num_clients, f"per_round {per_round}가 클라이언트 수 {num_clients}보다 큽니다."),
        (env.min_accepted <= per_round,
         f"min_accepted {env.min_accepted}가 라운드 참여자 수 {per_round}보다 커서 모델이 한 번도 갱신되지 않습니다."),
        (env.min_accepted <= reachable,
         (f"지연 {len(delayed)}명은 항상 late로 빠지므로 한 라운드에 많아야 {reachable}명만 채택되는데, "
          f"min_accepted가 {env.min_accepted}라 모델이 한 번도 갱신되지 않습니다.")),
    ]
    problems = [message for ok, message in checks if not ok]
    if problems:
        raise ValueError(f"{env.name} 환경 설정 오류: " + " ".join(problems))



class GlobalModelDiverged(TrainingDiverged):
    """집계한 전역 모델이 깨졌다. 1라운드에서도 공격 업데이트가 원인일 수 있다 (로컬 발산은 초기 모델에서 생기지 않음)."""


def is_attack_result(error: TrainingDiverged, rounds_done: int, attack: bool) -> bool:
    """이 발산을 공격 실험의 결과로 남길지. 정상 실행이나 초기 모델에서의 로컬 발산은 설정 문제이므로 False."""
    return attack and (rounds_done > 0 or isinstance(error, GlobalModelDiverged))

@dataclass(frozen=True)
class Snapshot:
    round: int
    model_version: int
    state: State            # 기기에 올린 전역 가중치
    participants: list[int]


@dataclass(frozen=True)
class Trained:
    result: ClientResult
    train_sec: float


@dataclass(frozen=True)
class Sent:
    update: State           # 실제로 제출한 업데이트 (공격이면 조작된 벡터)
    virtual_finish: float
    reply: dict             # POST /update 응답 전체
    comm_sec: float         # 직렬화 + HTTP

    @property
    def accepted(self) -> bool:
        return bool(self.reply["ok"])

    @property
    def reason(self) -> str:
        return "" if self.accepted else self.reply["data"]["reason"]


def fetch_snapshot(api: ServerApi, device: torch.device, expected_round: int) -> Snapshot:
    data = api.model()
    if data["round"] != expected_round:
        raise RuntimeError(f"서버 라운드 {data['round']}, 클라이언트가 기대한 라운드 {expected_round}")
    state = {key: value.to(device) for key, value in decode_state(data["weights"]).items()}
    return Snapshot(data["round"], data["model_version"], state, list(data["participants"]))


def check_server(status: dict, cfg: ExperimentConfig, num_clients: int, plan: ClientPlan, start_round: int = 1) -> None:
    """서버가 같은 시드·모델·클라이언트 수·환경·참여·채택 규칙·방어 설정으로 처음 상태(start_round, 버전 0)에서
    시작했는지 확인한다."""
    server_cfg = status["config"]
    pairs = [("seed", server_cfg.get("seed"), cfg.seed), ("model", server_cfg.get("model"), cfg.model),
             ("num_clients", status["num_clients"], num_clients), ("env", server_cfg.get("env"), plan.env_name),
             ("per_round", server_cfg.get("per_round"), plan.per_round),
             ("deadline", server_cfg.get("deadline"), plan.timing.deadline),
             ("min_accepted", status["min_accepted"], plan.min_accepted),
             *[(key, server_cfg.get(key), plan.defense[key]) for key in DEFENSE_KEYS]]
    mismatches = [f"{name}: 서버 {theirs}, 클라이언트 {ours}" for name, theirs, ours in pairs if theirs != ours]
    if status["round"] != start_round or status["model_version"] != 0:
        mismatches.append(f"서버가 처음 상태가 아닙니다 (라운드 {status['round']}, 버전 {status['model_version']})")
    if mismatches:
        raise RuntimeError("서버 설정이 다릅니다: " + "; ".join(mismatches))


def train_participants(model: torch.nn.Module, snapshot: Snapshot, clients: list[TensorData],
                       cfg: ExperimentConfig, device: torch.device) -> dict[int, Trained]:
    """참여자 전원이 같은 전역 모델에서 ID 순으로 학습한다 (제출 전에 모두 끝낸다)."""
    trained = {}
    for cid in snapshot.participants:
        t_start = timed_now(device)
        result = local_update(model, snapshot.state, clients[cid], cfg.train,
                              client_generator(cfg.seed, snapshot.round, cid))
        trained = {**trained, cid: Trained(result, timed_now(device) - t_start)}
    return trained


def updates_to_submit(normal: dict[int, State], plan: ClientPlan) -> dict[int, State]:
    """공격을 켰으면 이번 라운드에 학습한 악성 참여자(조작 해제한 ID 제외)의 정상 업데이트로 공모 벡터를 만들어
    바꿔 끼운다. 해제한 악성은 남은 악성의 공모에 끼지 않고 자기 정상 업데이트를 낸다."""
    if not plan.attack:
        return dict(normal)
    return {**normal, **collusion_vectors(normal, plan.attackers, plan.attack_scale)}



def check_submittable(updates: dict[int, State]) -> None:
    """서버가 non_finite로 거부할 업데이트(원소는 유한해도 노름이 넘침, 공격 벡터 × scale 포함)는 제출 전에 발산으로 본다.
    서버와 같은 update_norm으로 판정하고, 아무것도 보내기 전이라 서버 상태는 그대로다."""
    broken = [cid for cid, update in updates.items() if not math.isfinite(update_norm(update))]
    if broken:
        raise TrainingDiverged(f"클라이언트 {broken}의 제출할 업데이트 노름이 float32 범위를 넘습니다 (발산).")

def submit_all(api: ServerApi, snapshot: Snapshot, updates: dict[int, State], trained: dict[int, Trained],
               plan: ClientPlan, device: torch.device) -> dict[int, Sent]:
    finish = virtual_finish_times({cid: t.result.num_samples for cid, t in trained.items()}, plan.delayed, plan.timing)
    sent = {}
    for cid, update in updates.items():
        t_start = timed_now(device)
        reply = api.submit({"client_id": cid, "round": snapshot.round, "model_version": snapshot.model_version,
                            "num_samples": trained[cid].result.num_samples, "delta_weights": encode_state(update),
                            "virtual_finish_time": finish[cid]})
        check_reply(cid, reply)
        sent = {**sent, cid: Sent(update, finish[cid], reply, timed_now(device) - t_start)}
    return sent


def check_reply(cid: int, reply: dict) -> None:
    """채택 또는 예상한 제외(late)만 통과. 그 밖의 거부나 서버 오류는 결과를 믿을 수 없으므로 멈춘다."""
    if reply.get("ok"):
        return
    data = reply.get("data")
    reason = data.get("reason") if isinstance(data, dict) else None
    if reason not in EXPECTED_EXCLUSIONS:
        raise RuntimeError(f"클라이언트 {cid}의 제출이 예상하지 못한 이유로 거부되었습니다: "
                           f"{reason or '서버 오류'} ({reply.get('error')})")


def check_decision(decision: RoundDecision, record: dict) -> None:
    """클라이언트가 계산한 채택 판정이 서버 기록과 같은지 확인한다 (다르면 로그를 믿을 수 없음)."""
    if list(decision.accepted) != record["accepted_clients"] or decision.update_model != record["updated"]:
        raise RuntimeError(f"채택 판정이 서버와 다릅니다: 클라이언트 {list(decision.accepted)} 갱신 {decision.update_model}, "
                           f"서버 {record['accepted_clients']} 갱신 {record['updated']}")


def _r4(value: float) -> float:
    return round(value, 4)


def client_rows(round_idx: int, num_clients: int, trained: dict[int, Trained], sent: dict[int, Sent],
                normal: dict[int, State], record: dict, plan: ClientPlan) -> list[dict]:
    """명세 5단계 학습 로그: 라운드 × 클라이언트 (참여하지 않은 클라이언트도 한 줄, 값은 빈칸)."""
    defended = dict(zip(record["accepted_clients"], record["defended_norms"]))
    rows = []
    for cid in range(num_clients):
        base = {"experiment": plan.experiment_id, "round": round_idx, "client": cid,
                "malicious": cid in plan.malicious, "selected": cid in sent, "attacked": False,
                "defense": record["aggregator"]}
        if cid not in sent:
            rows.append({**base, "num_samples": "", "train_loss": "", "update_norm": "", "norm_submitted": "",
                         "norm_defended": "", "accepted": "", "reason": "", "virtual_finish": "",
                         "train_sec": "", "comm_sec": ""})
            continue
        s, t = sent[cid], trained[cid]
        rows.append({**base, "attacked": s.update is not normal[cid],
                     "num_samples": t.result.num_samples, "train_loss": _r4(t.result.train_loss),
                     "update_norm": _r4(update_norm(normal[cid])), "norm_submitted": _r4(update_norm(s.update)),
                     "norm_defended": defended.get(cid, ""), "accepted": s.accepted, "reason": s.reason,
                     "virtual_finish": _r4(s.virtual_finish), "train_sec": round(t.train_sec, 3),
                     "comm_sec": round(s.comm_sec, 3)})
    return rows


def round_row(snapshot: Snapshot, trained: dict[int, Trained], sent: dict[int, Sent], decision: RoundDecision,
              record: dict, val: tuple[float, float], plan: ClientPlan, seconds: dict) -> dict:
    norms = [update_norm(t.result.update) for t in trained.values()]
    log = decision.to_log()
    return {
        "experiment": plan.experiment_id,
        "round": snapshot.round,
        "train_loss": _r4(sum(t.result.train_loss for t in trained.values()) / len(trained)),
        "val_loss": _r4(val[0]),
        "val_acc": _r4(val[1]),
        "norm_min": _r4(min(norms)),
        "norm_mean": _r4(sum(norms) / len(norms)),
        "norm_max": _r4(max(norms)),
        **log,
        "num_excluded": record["num_excluded"],
        "attack_applied": any(s.update is not trained[cid].result.update for cid, s in sent.items()),
        "updated": record["updated"],
        "skip_reason": record["skip_reason"],   # 서버 사유 (no_accepted / too_few_accepted)
        "aggregator": record["aggregator"],
        "aggregate_norm": record["aggregate_norm"],
        "virtual_elapsed": _r4(round_elapsed({cid: s.virtual_finish for cid, s in sent.items()},
                                             plan.timing.deadline)),
        **seconds,
    }


def run_round(api: ServerApi, model: torch.nn.Module, snapshot: Snapshot, clients: list[TensorData],
              val: TensorData, cfg: ExperimentConfig, plan: ClientPlan, min_accepted: int,
              device: torch.device) -> tuple[dict, list[dict], Snapshot, Evaluation]:
    """라운드 1회. (rounds.csv 한 줄, clients.csv 줄들, 다음 라운드 스냅샷, 갱신된 모델의 검증 평가)."""
    t_start = timed_now(device)
    trained = train_participants(model, snapshot, clients, cfg, device)
    normal = {cid: t.result.update for cid, t in trained.items()}
    to_submit = updates_to_submit(normal, plan)
    check_submittable(to_submit)
    sent = submit_all(api, snapshot, to_submit, trained, plan, device)
    t_aggregate = timed_now(device)
    record = api.aggregate()
    next_snapshot = fetch_snapshot(api, device, snapshot.round + 1)
    t_fetched = timed_now(device)
    model.load_state_dict(next_snapshot.state)
    val_eval = evaluate_detailed(model, val, cfg.eval_bs)
    t_end = timed_now(device)
    # 업데이트는 유한해도 집계한 전역 모델이 깨질 수 있다. 이 라운드를 발산으로 보고 이전 모델로 평가하게 한다
    if not math.isfinite(val_eval.loss) or not all(torch.isfinite(v).all() for v in next_snapshot.state.values()):
        raise GlobalModelDiverged(f"라운드 {snapshot.round} 집계 뒤 전역 모델 또는 검증 손실이 NaN·inf입니다 (발산).")

    excluded = {cid: s.reason for cid, s in sent.items() if not s.accepted}
    decision = judge_round(snapshot.participants, excluded, plan.malicious, len(clients), min_accepted)
    check_decision(decision, record)
    seconds = {
        "train_sec": round(sum(t.train_sec for t in trained.values()), 3),
        "comm_sec": round(sum(s.comm_sec for s in sent.values()) + (t_fetched - t_aggregate), 3),
        "total_sec": round(t_end - t_start, 3),
    }
    row = round_row(snapshot, trained, sent, decision, record, (val_eval.loss, val_eval.acc), plan, seconds)
    return row, client_rows(snapshot.round, len(clients), trained, sent, normal, record, plan), next_snapshot, val_eval


def should_save(round_idx: int, rounds: int, save_every: int) -> bool:
    """라운드 0(초기 모델), save_every의 배수, 마지막 라운드(rounds)의 전역 모델을 저장한다.
    재실행이면 rounds에 마지막 라운드 번호(시작 라운드 − 1 + 라운드 수)를 넘긴다."""
    return round_idx in (0, rounds) or (save_every > 0 and round_idx % save_every == 0)


def save_state(state: State, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in state.items()}, path)


def checkpoint_path(out_dir: Path, round_idx: int) -> Path:
    """models/round_005.pt = 라운드 5 집계 후 전역 모델 (= 라운드 6 시작 모델)."""
    return out_dir / "models" / f"round_{round_idx:03d}.pt"


def _load_data(cfg: ExperimentConfig, repo_root: Path, device: torch.device):
    """(검증, 클라이언트별 데이터, 테스트). run_fedavg와 같은 분할 파일을 쓴다."""
    train_all, test = load_cifar10(repo_root / cfg.data_root, device)
    split = load_split(repo_root / cfg.split)
    if split["num_images"] != len(train_all):
        raise ValueError("분할 파일의 이미지 수가 CIFAR-10 학습 데이터와 다릅니다.")
    return train_all.subset(split["val"]), [train_all.subset(indices) for indices in split["clients"]], test


def run_clients(cfg: ExperimentConfig, out_dir: Path, repo_root: Path, server_url: str, plan: ClientPlan,
                evaluate_test: bool = False, save_every: int = DEFAULT_SAVE_EVERY, start_round: int = 1) -> dict:
    """결과 폴더 형식은 run_fedavg와 같고, 서버 기록(server_status.json), 클래스별(class_acc.csv)·
    클라이언트별(client_acc.csv) 정확도, 라운드별 전역 모델(models/round_XXX.pt)을 더 남긴다.

    cfg.split은 환경의 분할 파일이어야 한다 (레이블 편향 환경은 다른 분할).
    start_round > 1은 재실행: 서버가 저장된 모델로 그 라운드부터 시작해 있어야 하고, cfg.rounds만큼 더 돈다.
    """
    if save_every < 0:
        raise ValueError("save_every는 0 이상이어야 합니다.")
    set_seed(cfg.seed)
    device = pick_device()
    val, clients, test = _load_data(cfg, repo_root, device)
    check_delay_setting({cid: len(data) for cid, data in enumerate(clients)}, plan.delayed, plan.timing)
    api = ServerApi(server_url)
    status = api.status()
    check_server(status, cfg, len(clients), plan, start_round)

    model = build_model(cfg.model).to(device)
    env = environment_info(device)
    server_info = {k: status[k] for k in ("config", "min_accepted", "aggregator")}
    log = RunLog(out_dir, {**cfg.to_dict(), "num_clients": len(clients), "runner": "server",
                           "server_url": server_url, "server": server_info, "plan": plan.to_dict()}, env)
    print(f"device={env['device']} gpu={env['gpu']} commit={env['commit'][:7]} clients={len(clients)} server={server_url} "
          f"env={plan.env_name} attack={plan.attack} malicious={list(plan.malicious)} delayed={list(plan.delayed)}")

    distributions = [label_distribution(data.labels) for data in clients]
    snapshot = fetch_snapshot(api, device, expected_round=start_round)
    save_state(snapshot.state, checkpoint_path(out_dir, start_round - 1))
    last_round = start_round - 1 + cfg.rounds
    rows = []
    diverged_round = None
    for _ in range(cfg.rounds):
        round_idx = snapshot.round
        try:
            row, rows_by_client, next_snapshot, val_eval = run_round(api, model, snapshot, clients, val, cfg, plan,
                                                                     status["min_accepted"], device)
        except TrainingDiverged as error:
            if not is_attack_result(error, len(rows), plan.attack):
                raise
            # 이 라운드는 기록하지 않고, 발산 전(지난 라운드 끝) 전역 모델로 결과를 남긴다
            diverged_round = round_idx
            print(f"라운드 {round_idx}에서 발산 ({error}) → 라운드 {round_idx - 1}의 전역 모델로 평가하고 끝냄")
            save_state(snapshot.state, checkpoint_path(out_dir, round_idx - 1))
            break
        snapshot = next_snapshot
        log.log_round(row)
        log.log_clients(rows_by_client)
        log.log_class_acc(class_rows(round_idx, "val", val_eval))
        log.log_client_acc(client_acc_rows(round_idx, "val", val_eval, distributions))
        if should_save(round_idx, last_round, save_every):
            save_state(snapshot.state, checkpoint_path(out_dir, round_idx))
        rows.append(row)
        print(" ".join(f"{k}={row[k]}" for k in PRINT_KEYS))

    (out_dir / "server_status.json").write_text(json.dumps(api.status(), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"experiment": plan.experiment_id, "env": plan.env_name, "attack": plan.attack, "seed": cfg.seed,
               "aggregator": status["aggregator"], "diverged_round": diverged_round, "rounds_completed": len(rows)}
    if not rows:
        # 1라운드 집계 직후 전역 모델이 깨진 공격 실험: 비교표에 넣을 정확도가 없다 (run_matrix 상태 diverged)
        summary = {**summary, "final_val_acc": None, "final_test_acc": None}
    else:
        summary = {**_summary(rows), **summary}
    if evaluate_test and rows:
        model.load_state_dict(snapshot.state)
        test_eval = evaluate_detailed(model, test, cfg.eval_bs)
        summary = {**summary, "final_test_acc": round(test_eval.acc, 4), "final_test_loss": round(test_eval.loss, 4)}
        log.log_class_acc(class_rows(snapshot.round - 1, "test", test_eval))
        log.log_client_acc(client_acc_rows(snapshot.round - 1, "test", test_eval, distributions))
    save_state(snapshot.state, out_dir / "models" / "final_model.pt")
    log.write_summary(summary)   # 마지막에 쓴다: summary.json이 있으면 실행이 끝까지 간 것 (run_matrix 완료 판정)
    return summary


def _summary(rows: list[dict]) -> dict:
    times = [row["total_sec"] for row in rows]
    later = times[1:] or times
    comm = [row["comm_sec"] for row in rows]
    return {
        "final_val_acc": rows[-1]["val_acc"],
        "rounds_not_updated": sum(1 for row in rows if not row["updated"]),
        "total_virtual_time": round(sum(row["virtual_elapsed"] for row in rows), 4),
        "first_round_sec": times[0],
        "mean_round_sec_after_first": round(sum(later) / len(later), 3),
        "mean_comm_sec": round(sum(comm) / len(comm), 3),
        "total_sec": round(sum(times), 1),
        "runner": "server",
    }
