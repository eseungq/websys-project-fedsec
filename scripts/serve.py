"""중앙 Flask 서버만 실행한다 (127.0.0.1). 초기 모델은 설정 파일의 seed·model로 서버가 만든다.

서버는 환경 설정에서 참여자 일정·가상 마감·최소 채택 수만, 공격·방어 설정에서 집계 방식만 쓴다.
악성·지연 클라이언트가 누구인지와 공격 여부는 모른다.

예) python scripts/serve.py                                  # 포트 8765, configs/base.yaml, 기준 환경, FedAvg
    python scripts/serve.py --port 8800 --log-dir experiments/demo_server   # 결과 폴더(demo)와 다른 이름
    python scripts/serve.py --env configs/env/partial.yaml --aggregator clip_fedavg
다른 터미널에서 python scripts/run_clients.py --name demo --server-url http://127.0.0.1:8800 (같은 환경·방어 옵션)
"""

import argparse
from dataclasses import replace
from pathlib import Path

from run_clients import DEFAULT_NUM_CLIENTS, EXPERIMENTS_DIR, add_scenario_args, inside, load_scenario

from fedsec.experiment.config import load_config
from fedsec.logger.server_log import ServerLog
from fedsec.server.app import DEFAULT_PORT, HOST, create_app
from fedsec.server.scenario import build_defense, scenario_policy
from fedsec.server.state import ServerState, initial_global_state, load_global_state

ROOT = Path(__file__).resolve().parents[1]


def shown_path(path: str) -> str:
    """기록용 경로: 레포 안이면 상대 경로로 (절대 경로의 PC 사용자 이름이 결과 파일에 남지 않게)."""
    resolved = (ROOT / path).resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml", help="레포 루트 기준 상대 경로 또는 절대 경로")
    parser.add_argument("--seed", type=int, help="초기 모델·참여 일정 시드 (없으면 설정 파일 값)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS, dest="num_clients")
    parser.add_argument("--min-accepted", type=int, dest="min_accepted",
                        help="채택이 이 수보다 적으면 모델을 갱신하지 않음 (없으면 환경 파일 값)")
    parser.add_argument("--log-dir", help="server_rounds.csv, exclusions.csv를 쓸 폴더 (레포 루트 기준)")
    parser.add_argument("--start-round", type=int, default=1, dest="start_round",
                        help="재실행: 이 라운드부터 시작 (--init-model 필요, scripts/replay.py가 씀)")
    parser.add_argument("--init-model", dest="init_model",
                        help="재실행: 시작 전역 모델 파일 (experiments/ 아래 models/round_XXX.pt)")
    add_scenario_args(parser)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config).with_overrides(seed=args.seed)
    env, attack_defense = load_scenario(args)
    if args.min_accepted is not None:
        env = replace(env, min_accepted=args.min_accepted)
    policy = scenario_policy(env, build_defense(attack_defense), cfg.seed, args.num_clients)
    # 결과 폴더와 같이 experiments/ 아래만 (절대 경로로 받아도 그 안이어야 함)
    log = ServerLog(inside(EXPERIMENTS_DIR, str((ROOT / args.log_dir).resolve()))) if args.log_dir else None
    defense = {key: value for key, value in attack_defense.to_dict().items() if not key.startswith("attack")}
    if args.start_round > 1 and not args.init_model:
        parser.error("--start-round가 1보다 크면 --init-model로 시작 모델을 줘야 합니다.")
    start = initial_global_state(cfg.seed, cfg.model)
    resume = {}
    if args.init_model:
        model_path = inside(EXPERIMENTS_DIR, str((ROOT / args.init_model).resolve()))
        start = load_global_state(model_path, reference=start)
        resume = {"start_round": args.start_round, "init_model": shown_path(str(model_path))}
    state = ServerState(start, num_clients=args.num_clients, policy=policy, log=log, start_round=args.start_round,
                        config={"seed": cfg.seed, "model": cfg.model, "config_file": shown_path(args.config),
                                "env": env.name, "env_file": shown_path(args.env), "per_round": env.per_round,
                                "deadline": env.timing.deadline, **defense, **resume})
    print(f"server http://{HOST}:{args.port} seed={cfg.seed} model={cfg.model} clients={args.num_clients} "
          f"env={env.name} per_round={env.per_round} deadline={env.timing.deadline} "
          f"aggregator={policy.aggregator_name} min_accepted={policy.min_accepted}", flush=True)
    create_app(state).run(host=HOST, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
