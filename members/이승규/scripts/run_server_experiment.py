"""서버를 하위 프로세스로 띄우고, 준비되면 가상 클라이언트 실행 프로그램을 돌린 뒤 서버를 끈다.

예) python scripts/run_server_experiment.py --name server_check --rounds 3
    python scripts/run_server_experiment.py --name d_atk_med --env configs/env/delay.yaml --attack --aggregator median
결과: experiments/<name>/ (run_clients.py와 같음) + server/ (server_rounds.csv, exclusions.csv, server.log)
실패하거나 Ctrl+C로 멈춰도 서버 프로세스는 종료된다.
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from fedsec.client.program import run_clients
from fedsec.server.app import HOST

from run_clients import (MEMBER_ROOT, REPO_ROOT, add_experiment_args, add_scenario_args, client_setup, new_out_dir,
                         server_scenario_args)

SERVE_SCRIPT = Path(__file__).resolve().parent / "serve.py"
READY_TIMEOUT_SEC = 60
READY_POLL_SEC = 0.2
STOP_TIMEOUT_SEC = 10


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def wait_until_ready(url: str, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"서버가 시작 중에 종료되었습니다 (코드 {server.returncode}). server.log를 확인하세요.")
        try:
            if requests.get(f"{url}/status", timeout=1).ok:
                return
        except requests.RequestException:
            pass  # 아직 포트를 열지 않았거나 응답 전. 서버가 죽었으면 위 poll()에서 멈춘다
        time.sleep(READY_POLL_SEC)
    raise RuntimeError(f"서버가 {READY_TIMEOUT_SEC}초 안에 준비되지 않았습니다.")


def stop(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=STOP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_experiment_args(parser)
    add_scenario_args(parser)
    parser.add_argument("--port", type=int, help="서버 포트 (없으면 빈 포트를 고름)")
    args = parser.parse_args()

    cfg, plan, attack_defense = client_setup(args)
    out_dir = new_out_dir(args.name)
    server_dir = out_dir / "server"
    server_dir.mkdir(parents=True)
    port = args.port or free_port()
    url = f"http://{HOST}:{port}"
    command = [sys.executable, str(SERVE_SCRIPT), "--config", str(MEMBER_ROOT / args.config),
               "--seed", str(cfg.seed), "--port", str(port), "--log-dir", str(server_dir),
               *server_scenario_args(args, attack_defense)]

    with (server_dir / "server.log").open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            wait_until_ready(url, server)
            summary = run_clients(cfg, out_dir, repo_root=REPO_ROOT, server_url=url, plan=plan,
                                  evaluate_test=args.evaluate_test, save_every=args.save_every)
        finally:
            stop(server)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
