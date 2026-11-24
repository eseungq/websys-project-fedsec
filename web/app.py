"""웹 대시보드와 연합학습 REST API를 한 Flask 서버로 띄운다 (127.0.0.1, 한 프로세스, 한 포트).

- http://127.0.0.1:<port>/ : 대시보드 (조회형: experiments/ 결과 보기, 실행형: 짧은 시연 실행)
- /status, /model, /update, /aggregate : serve.py와 같은 연합학습 API (가상 클라이언트 프로그램이 씀)
처음 서버 상태는 serve.py의 기본값(기준 환경, 공격 없음, FedAvg, 설정 파일의 seed)과 같다.
대시보드에서 실행을 시작하면 서버 상태를 그 설정으로 새로 만들고 scripts/run_clients.py를 하위 프로세스로 띄운다.

예) python web/app.py                 # http://127.0.0.1:8765/
    python web/app.py --port 8800
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)   # web/ 대신 레포 루트를 경로에 둔다 (web 패키지로 불러오기)

from fedsec.server.app import DEFAULT_PORT, HOST, create_app
from web.routes import create_dashboard
from web.run import CONFIG_FILE, LiveRunner, StateHolder, build_server_state

EXPERIMENTS_DIR = ROOT / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config", default=CONFIG_FILE, help="학습 설정 (레포 루트 기준, 실행형에서도 씀)")
    args = parser.parse_args()

    holder = StateHolder(build_server_state(ROOT, "baseline", attack=False, aggregator="fedavg",
                                            config_file=args.config))
    url = f"http://{HOST}:{args.port}"
    runner = LiveRunner(holder, ROOT, EXPERIMENTS_DIR, server_url=url, config_file=args.config)
    app = create_app(holder)
    app.register_blueprint(create_dashboard(EXPERIMENTS_DIR, runner))
    print(f"dashboard {url}/  (연합학습 API도 같은 주소)", flush=True)
    app.run(host=HOST, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
