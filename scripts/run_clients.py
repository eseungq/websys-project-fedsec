"""가상 클라이언트 실행 프로그램만 실행한다. 서버(scripts/serve.py)가 먼저 떠 있어야 한다.

예) python scripts/run_clients.py --name demo                      # http://127.0.0.1:8765, 기준 환경, 공격 없음, FedAvg
    python scripts/run_clients.py --name demo --server-url http://127.0.0.1:8800 --rounds 3
    python scripts/run_clients.py --name d1 --env configs/env/delay.yaml --attack --aggregator median
결과: experiments/<name>/ 에 run_experiment.py와 같은 파일 + server_status.json
서버를 같은 설정(--config·--seed·--env·방어 옵션)으로 새로 띄운 뒤 실행하세요 (다르면 멈춥니다).
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from fedsec.client.program import DEFAULT_SAVE_EVERY, ClientPlan, client_plan, run_clients
from fedsec.environments.config import EnvConfig, load_env_config
from fedsec.experiment.config import (
    AGGREGATORS,
    AttackDefenseConfig,
    ExperimentConfig,
    load_attack_defense,
    load_config,
)
from fedsec.server.app import DEFAULT_PORT, HOST

# 기본 설정·결과는 레포 루트 기준
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
OVERRIDE_KEYS = ("seed", "rounds", "lr", "momentum", "weight_decay", "local_ep", "local_bs")
DEFAULT_NUM_CLIENTS = 10


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    """run_experiment.py와 같은 옵션."""
    parser.add_argument("--name", required=True, help="결과 폴더 이름 (experiments/<name>)")
    parser.add_argument("--config", default="configs/base.yaml", help="레포 루트 기준 상대 경로 또는 절대 경로")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay")
    parser.add_argument("--local-ep", type=int, dest="local_ep")
    parser.add_argument("--local-bs", type=int, dest="local_bs")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="마지막에 공식 테스트셋도 평가 (설정 확정 후에만. 설정 비교는 검증 정확도로)")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, dest="save_every",
                        help=f"전역 모델 저장 간격 (라운드, 기본 {DEFAULT_SAVE_EVERY}. 라운드 0·마지막은 항상 저장)")


def add_scenario_args(parser: argparse.ArgumentParser) -> None:
    """운영 환경·공격·방어 옵션 (serve.py와 함께 씀). 없으면 설정 파일 값."""
    parser.add_argument("--env", default="configs/env/baseline.yaml", help="환경 설정 (레포 루트 기준)")
    parser.add_argument("--attack-defense", default="configs/attack_defense.yaml", dest="attack_defense",
                        help="공격·방어 설정 (레포 루트 기준)")
    parser.add_argument("--attack", action=argparse.BooleanOptionalAction, default=None,
                        help="악성 클라이언트의 공모 공격 켜기/끄기 (클라이언트 쪽만 씀)")
    parser.add_argument("--aggregator", choices=AGGREGATORS)
    parser.add_argument("--clip-norm", type=float, dest="clip_norm")
    parser.add_argument("--clip-before-median", action=argparse.BooleanOptionalAction, default=None,
                        dest="clip_before_median")


def load_scenario(args: argparse.Namespace) -> tuple[EnvConfig, AttackDefenseConfig]:
    env = load_env_config(ROOT / args.env)
    attack_defense = load_attack_defense(ROOT / args.attack_defense).with_overrides(
        attack=args.attack, aggregator=args.aggregator, clip_norm=args.clip_norm,
        clip_before_median=args.clip_before_median)
    return env, attack_defense


def server_scenario_args(args: argparse.Namespace, attack_defense: AttackDefenseConfig) -> list[str]:
    """serve.py에 넘길 환경·방어 옵션. 덮어쓴 최종 값을 그대로 넘겨 양쪽 설정을 맞춘다."""
    clip_flag = "--clip-before-median" if attack_defense.clip_before_median else "--no-clip-before-median"
    return ["--env", str(ROOT / args.env), "--attack-defense", str(ROOT / args.attack_defense),
            "--aggregator", attack_defense.aggregator, "--clip-norm", repr(attack_defense.clip_norm), clip_flag]


def experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    overrides = {k: getattr(args, k) for k in OVERRIDE_KEYS}
    return load_config(ROOT / args.config).with_overrides(**overrides)


def client_setup(args: argparse.Namespace) -> tuple[ExperimentConfig, ClientPlan, AttackDefenseConfig]:
    """학습 설정의 분할 파일을 환경의 분할로 바꾸고, 실험 ID(결과 폴더 이름)로 클라이언트 일정을 만든다."""
    env, attack_defense = load_scenario(args)
    cfg = replace(experiment_config(args), split=env.split)
    plan = client_plan(args.name, env, attack_defense, cfg.seed, DEFAULT_NUM_CLIENTS)
    return cfg, plan, attack_defense


EXPERIMENTS_DIR = ROOT / "experiments"


PARTIAL_SUFFIX = ".partial"   # run_dlg.py가 끝나기 전에 쓰는 임시 폴더 이름


def inside(base: Path, name: str) -> Path:
    """base 아래 결과 폴더. ../나 절대 경로로 base 밖을 가리키면 멈춘다."""
    out_dir = (base / name).resolve()
    if out_dir == base.resolve() or not out_dir.is_relative_to(base.resolve()):
        sys.exit(f"결과 폴더는 {base.relative_to(ROOT)}/ 아래여야 합니다: {name}")
    if out_dir.name.endswith(PARTIAL_SUFFIX):
        sys.exit(f"결과 폴더 이름은 {PARTIAL_SUFFIX}로 끝날 수 없습니다 (DLG 임시 폴더와 겹침): {name}")
    return out_dir


def new_out_dir(name: str, base: Path = EXPERIMENTS_DIR) -> Path:
    out_dir = inside(base, name)
    if out_dir.exists():
        sys.exit(f"이미 있는 결과 폴더입니다: {out_dir.relative_to(ROOT)} (다른 --name을 쓰세요)")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_experiment_args(parser)
    add_scenario_args(parser)
    parser.add_argument("--server-url", default=f"http://{HOST}:{DEFAULT_PORT}", dest="server_url")
    args = parser.parse_args()

    cfg, plan, _ = client_setup(args)
    out_dir = new_out_dir(args.name)
    summary = run_clients(cfg, out_dir, repo_root=REPO_ROOT, server_url=args.server_url, plan=plan,
                          evaluate_test=args.evaluate_test, save_every=args.save_every)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
