"""실행형: 같은 Flask 서버의 상태를 새 ServerState로 바꾸고, 가상 클라이언트 프로그램(scripts/run_clients.py)을
하위 프로세스로 띄운다. 결과는 experiments/live/<시각>-<이름>/ (git 무시). 한 번에 한 실행만.

create_app(state)는 상태 객체를 하나만 받으므로, 현재 상태에 위임하는 StateHolder를 넘기고 실행마다 바꿔 끼운다.
서버 상태는 serve.py와 같은 순서(load_config → load_env_config → scenario_policy → initial_global_state)로 만든다.
"""

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from fedsec.environments.config import load_env_config
from fedsec.experiment.config import AGGREGATORS, load_attack_defense, load_config
from fedsec.logger.server_log import ServerLog
from fedsec.server.scenario import build_defense, scenario_policy
from fedsec.server.state import ServerState, initial_global_state
from web.files import read_csv, read_json, read_yaml

ENVS = ("baseline", "delay", "partial", "label_skew")
ATTACKS = ("normal", "attack")
NUM_CLIENTS = 10                      # run_clients.py의 DEFAULT_NUM_CLIENTS와 같음
DEFAULT_ROUNDS = 3
MAX_ROUNDS = 10                       # 시연용: 라운드가 길면 화면 앞에서 기다리기 어려움
MAX_SEED = 9999
LABEL = re.compile(r"[A-Za-z0-9_-]{1,40}")
START_KEYS = frozenset({"env", "attack", "aggregator", "seed", "rounds", "name"})
LIVE_DIR = "live"
LOG_TAIL_LINES = 40
STOP_TIMEOUT_SEC = 10
CONFIG_FILE = "configs/base.yaml"
ATTACK_DEFENSE_FILE = "configs/attack_defense.yaml"


def env_file(env: str) -> str:
    return f"configs/env/{env}.yaml"


class StateHolder:
    """현재 ServerState에 모든 속성을 넘긴다. create_app에는 이 객체를 주고, 실행마다 swap으로 바꾼다."""

    def __init__(self, state: ServerState):
        self._state = state

    @property
    def current(self) -> ServerState:
        return self._state

    def swap(self, state: ServerState) -> None:
        self._state = state   # 참조 한 번 바꾸기 (진행 중인 요청은 이전 상태로 끝남)

    def __getattr__(self, name: str):
        return getattr(self._state, name)


class LazyServerLog:
    """서버 기록(server/)을 첫 기록 때 만든다. run_clients.py는 결과 폴더가 이미 있으면 멈추므로 미리 만들 수 없다."""

    def __init__(self, out_dir: Path):
        self._out_dir = out_dir
        self._log: ServerLog | None = None
        self._lock = threading.Lock()

    def _target(self) -> ServerLog:
        with self._lock:
            if self._log is None:
                self._log = ServerLog(self._out_dir)
            return self._log

    def log_round(self, record: dict) -> None:
        self._target().log_round(record)

    def log_exclusion(self, record: dict) -> None:
        self._target().log_exclusion(record)


def build_server_state(root: Path, env: str, attack: bool, aggregator: str, seed: int | None = None,
                       config_file: str = CONFIG_FILE, log=None) -> ServerState:
    """serve.py의 main과 같은 설정으로 ServerState를 만든다 (run_clients.py의 서버 설정 대조를 통과하도록)."""
    cfg = load_config(root / config_file).with_overrides(seed=seed)
    env_cfg = load_env_config(root / env_file(env))
    attack_defense = load_attack_defense(root / ATTACK_DEFENSE_FILE).with_overrides(attack=attack, aggregator=aggregator)
    policy = scenario_policy(env_cfg, build_defense(attack_defense), cfg.seed, NUM_CLIENTS)
    defense = {key: value for key, value in attack_defense.to_dict().items() if not key.startswith("attack")}
    config = {"seed": cfg.seed, "model": cfg.model, "config_file": config_file, "env": env_cfg.name,
              "env_file": env_file(env), "per_round": env_cfg.per_round, "deadline": env_cfg.timing.deadline, **defense}
    return ServerState(initial_global_state(cfg.seed, cfg.model), num_clients=NUM_CLIENTS, policy=policy, log=log,
                       config=config)


@dataclass(frozen=True)
class LiveParams:
    env: str
    attack: bool
    aggregator: str
    seed: int
    rounds: int
    label: str

    def to_dict(self) -> dict:
        return {"env": self.env, "attack": "attack" if self.attack else "normal", "aggregator": self.aggregator,
                "seed": self.seed, "rounds": self.rounds, "name": self.label}


def _choice(body: dict, key: str, allowed: tuple[str, ...], default: str) -> str:
    value = body.get(key, default)
    if value not in allowed:
        raise ValueError(f"{key}는 {', '.join(allowed)} 중 하나여야 합니다.")
    return value


def _int_in(body: dict, key: str, low: int, high: int, default: int) -> int:
    value = body.get(key, default)
    if type(value) is not int or not low <= value <= high:   # bool·실수·문자열 거부
        raise ValueError(f"{key}는 {low} 이상 {high} 이하의 정수여야 합니다.")
    return value


def parse_start(body) -> LiveParams:
    """POST /api/start 본문 검증. 객체가 아니면 TypeError, 모르는 항목·목록 밖 값·범위 밖 정수는 ValueError."""
    if not isinstance(body, dict):
        raise TypeError("요청 본문은 JSON 객체여야 합니다.")
    unknown = sorted(str(key)[:40] for key in set(body) - START_KEYS)
    if unknown:
        raise ValueError(f"모르는 항목입니다: {', '.join(unknown)}")
    env = _choice(body, "env", ENVS, "baseline")
    attack = _choice(body, "attack", ATTACKS, "normal")
    aggregator = _choice(body, "aggregator", AGGREGATORS, "fedavg")
    seed = _int_in(body, "seed", 0, MAX_SEED, 0)
    rounds = _int_in(body, "rounds", 1, MAX_ROUNDS, DEFAULT_ROUNDS)
    label = body.get("name") or f"{env}-{attack}-{aggregator}-s{seed}"
    if not isinstance(label, str) or not LABEL.fullmatch(label):
        raise ValueError("name은 영문·숫자·_·- 1~40자여야 합니다.")
    return LiveParams(env, attack == "attack", aggregator, seed, rounds, label)


class LiveBusy(RuntimeError):
    """이미 실행 중."""


@dataclass(frozen=True)
class LiveRun:
    run_id: str            # experiments/ 기준 결과 폴더 (live/<시각>-<이름>)
    params: LiveParams
    started: float
    process: object        # subprocess.Popen (시험에서는 가짜)
    log_tail: deque
    stopped: bool = False


class LiveRunner:
    def __init__(self, holder: StateHolder, root: Path, experiments_dir: Path, server_url: str,
                 config_file: str = CONFIG_FILE, popen=subprocess.Popen, state_factory=build_server_state):
        self._holder = holder
        self._root = root
        self._experiments_dir = experiments_dir
        self._server_url = server_url
        self._config_file = config_file
        self._popen = popen
        self._state_factory = state_factory
        self._run: LiveRun | None = None
        self._lock = threading.Lock()

    def _running(self) -> bool:
        return self._run is not None and self._run.process.poll() is None

    def _new_run_id(self, label: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidates = (f"{LIVE_DIR}/{stamp}-{label}" if n == 0 else f"{LIVE_DIR}/{stamp}-{label}-{n}" for n in range(100))
        return next(c for c in candidates if not (self._experiments_dir / c).exists())

    def _command(self, run_id: str, params: LiveParams) -> list[str]:
        return [sys.executable, str(self._root / "scripts" / "run_clients.py"), "--name", run_id,
                "--server-url", self._server_url, "--config", self._config_file, "--seed", str(params.seed),
                "--rounds", str(params.rounds), "--env", env_file(params.env),
                "--attack" if params.attack else "--no-attack", "--aggregator", params.aggregator]

    def _child_env(self) -> dict:
        src = str(self._root / "src")
        path = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)
        return {**os.environ, "PYTHONPATH": path, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

    def start(self, params: LiveParams) -> dict:
        with self._lock:
            if self._running():
                raise LiveBusy("이미 실행 중입니다. 끝나거나 중지한 뒤 다시 시작하세요.")
            run_id = self._new_run_id(params.label)
            log = LazyServerLog(self._experiments_dir / run_id / "server")
            state = self._state_factory(self._root, params.env, params.attack, params.aggregator, params.seed,
                                        config_file=self._config_file, log=log)
            self._holder.swap(state)
            process = self._popen(self._command(run_id, params), cwd=str(self._root), env=self._child_env(),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace")
            tail = deque(maxlen=LOG_TAIL_LINES)
            self._run = LiveRun(run_id, params, time.monotonic(), process, tail)
            threading.Thread(target=_collect_output, args=(process, tail), daemon=True).start()
            return {"run_id": run_id, "params": params.to_dict()}

    def stop(self) -> dict:
        with self._lock:
            if not self._running():
                return {"stopped": False}
            run = self._run
            self._run = replace(run, stopped=True)
            run.process.terminate()
            try:
                run.process.wait(timeout=STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                run.process.kill()
                run.process.wait()
            return {"stopped": True, "run_id": run.run_id}

    def status(self) -> dict:
        run = self._run
        if run is None:
            return {"state": "idle", "run_id": None, "max_rounds": MAX_ROUNDS}
        code = run.process.poll()
        run_dir = self._experiments_dir / run.run_id
        rounds = read_csv(run_dir / "rounds.csv")
        config = read_yaml(run_dir / "config.yaml") or {}
        return {
            "state": _state_name(code, run.stopped),
            "run_id": run.run_id,
            "params": run.params.to_dict(),
            "returncode": code,
            "elapsed_sec": round(time.monotonic() - run.started, 1),
            "rounds_total": run.params.rounds,
            "rounds_done": len(rounds),
            "rounds": rounds,
            # 실행형은 최대 MAX_ROUNDS × 클라이언트 10명이라 전부 보낸다 (화면은 조회형과 같이 라운드를 골라 그림)
            "clients": read_csv(run_dir / "clients.csv"),
            "plan": config.get("plan") if isinstance(config.get("plan"), dict) else None,
            "summary": read_json(run_dir / "summary.json"),
            "server": {key: self._holder.status()[key] for key in ("round", "model_version", "pending")},
            "log_tail": list(run.log_tail),
            "max_rounds": MAX_ROUNDS,
        }


def _state_name(code: int | None, stopped: bool) -> str:
    if code is None:
        return "running"
    if stopped:
        return "stopped"
    return "completed" if code == 0 else "error"


def _collect_output(process, tail: deque) -> None:
    """하위 프로세스 출력의 마지막 몇 줄만 기억한다 (오류 원인 표시용)."""
    stream = getattr(process, "stdout", None)
    if stream is None:
        return
    for line in stream:
        tail.append(line.rstrip()[:300])
