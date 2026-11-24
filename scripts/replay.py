"""저장된 라운드 모델에서 조건 하나만 바꿔 다시 실행한다 (명세 5단계 재실행, 6단계 원인 분석).

원래 실행의 config.yaml에서 학습 설정·환경·시드·악성/지연 ID를 그대로 가져오고, models/round_XXX.pt(라운드 K 집계 후
전역 모델)로 서버를 라운드 K+1부터 띄운다. 참여자·셔플·가상 시간은 (시드, 라운드)로 정해지므로 바꾼 조건 말고는
원래 실행과 같다. 아무것도 바꾸지 않으면(--same) 원래 실행의 K+1라운드 이후와 같은 결과가 나와야 한다.

바꿀 수 있는 조건 (하나 이상, 또는 --same)
  --unmask ID        이 악성 클라이언트의 조작만 해제 (정상 업데이트 제출, 로그의 malicious 표시는 그대로)
  --aggregator NAME  방어(집계) 방식만 교체
  --no-attack        공격 전체 해제

예) python scripts/replay.py --run 4_main/baseline-attack-fedavg-s0 --from-round 5 --unmask 2
    python scripts/replay.py --run 4_main/baseline-attack-fedavg-s0 --from-round 5 --aggregator median
    python scripts/replay.py --run 4_main/baseline-attack-fedavg-s0 --from-round 5 --same   # 재현 확인
결과: experiments/6_replay/<name>/ (run_server_experiment.py와 같은 파일 + replay.json)

재실행 방식은 배병욱 PR #9의 fedsim.replay(체크포인트에서 조건 하나만 바꿔 재실행)를 참고해 이 엔진에 맞게 새로 작성했다.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import yaml
from run_clients import (
    DEFAULT_NUM_CLIENTS,
    EXPERIMENTS_DIR,
    REPO_ROOT,
    ROOT,
    inside,
    new_out_dir,
)
from run_server_experiment import SERVE_SCRIPT, free_port, stop, wait_until_ready

from fedsec.client.program import ClientPlan, checkpoint_path, client_plan, run_clients
from fedsec.environments.config import load_env_config
from fedsec.experiment.config import (
    AGGREGATORS,
    AttackDefenseConfig,
    ExperimentConfig,
    load_config,
)
from fedsec.server.app import HOST

REPLAY_DIR = "6_replay"
# 원래 실행과 같아야 하는 학습 설정 (config.yaml 최상위 키). rounds는 재실행 길이라 빼고, split은 환경 파일과 따로 대조
SAME_KEYS = ("seed", "data_root", "model", "train", "eval_bs")


def load_source(run_dir: Path) -> dict:
    """원래 실행의 config.yaml과 summary.json. 끝나지 않은 실행(summary.json 없음)은 쓰지 않는다."""
    config_path, summary_path = run_dir / "config.yaml", run_dir / "summary.json"
    if not config_path.exists() or not summary_path.exists():
        sys.exit(f"끝난 실행 폴더가 아닙니다 (config.yaml, summary.json 필요): {run_dir.relative_to(ROOT)}")
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if saved.get("runner") != "server":
        sys.exit("서버 경유 실행(run_server_experiment.py, run_main.py)의 결과만 재실행할 수 있습니다.")
    return {"config": saved, "summary": json.loads(summary_path.read_text(encoding="utf-8"))}


def source_attack_defense(saved: dict) -> AttackDefenseConfig:
    plan = saved["plan"]
    return AttackDefenseConfig(attack=plan["attack"], attack_scale=plan["attack_scale"], **plan["defense"])


def rebuild(saved: dict, rounds: int, attack_defense: AttackDefenseConfig) -> tuple[ExperimentConfig, ClientPlan]:
    """원래 실행의 설정 파일로 설정을 다시 만들고, 저장된 값과 같은지 확인한다 (설정 파일이 그사이 바뀌었으면 멈춤)."""
    server_cfg = saved["server"]["config"]
    env = load_env_config(ROOT / server_cfg["env_file"])
    cfg = load_config(ROOT / server_cfg["config_file"]).with_overrides(
        seed=saved["seed"], lr=saved["train"]["lr"], momentum=saved["train"]["momentum"],
        weight_decay=saved["train"]["weight_decay"], local_ep=saved["train"]["local_ep"],
        local_bs=saved["train"]["local_bs"], rounds=rounds)
    cfg = replace(cfg, split=env.split)
    now = yaml.safe_load(yaml.safe_dump(cfg.to_dict(), allow_unicode=True))
    diffs = [key for key in SAME_KEYS if now.get(key) != saved.get(key)]
    if now["split"] != saved["split"]:
        diffs.append("split")
    plan = client_plan("", env, attack_defense, cfg.seed, DEFAULT_NUM_CLIENTS)
    planned = yaml.safe_load(yaml.safe_dump(plan.to_dict(), allow_unicode=True))
    for key in ("malicious", "delayed", "per_round", "min_accepted", "timing"):
        if planned[key] != saved["plan"].get(key):
            diffs.append(f"plan.{key}")
    if diffs:
        sys.exit(f"원래 실행과 지금 설정 파일이 다릅니다: {', '.join(diffs)}. 같은 조건으로 재실행할 수 없습니다.")
    return cfg, plan


def source_last_round(saved: dict) -> int:
    """원래 실행의 마지막 라운드 번호. 원래 실행이 재실행이면 시작 라운드가 1이 아니다."""
    start = saved["server"]["config"].get("start_round", 1)
    return start - 1 + saved["rounds"]


def merged_unmask(saved: dict, change: dict) -> tuple[int, ...]:
    """원래 실행(재실행 결과일 수 있음)에서 이미 해제한 악성과 이번에 해제할 악성을 합친다."""
    return tuple(sorted(set(saved["plan"].get("unmasked", [])) | set(change.get("unmask", []))))


def describe(args: argparse.Namespace, source_ad: AttackDefenseConfig) -> dict:
    """바꾼 조건. 아무것도 안 바꿨으면 빈 dict (--same)."""
    change = {}
    if args.unmask:
        change["unmask"] = sorted(set(args.unmask))
    if args.aggregator and args.aggregator != source_ad.aggregator:
        change["aggregator"] = {"from": source_ad.aggregator, "to": args.aggregator}
    if args.no_attack and source_ad.attack:
        change["attack"] = {"from": True, "to": False}
    return change


def change_label(change: dict) -> str:
    if not change:
        return "same"
    parts = [f"unmask{'-'.join(map(str, change['unmask']))}"] if "unmask" in change else []
    parts += [change["aggregator"]["to"]] if "aggregator" in change else []
    parts += ["noattack"] if "attack" in change else []
    return "_".join(parts)


def check_change(change: dict, args: argparse.Namespace, source_ad: AttackDefenseConfig, malicious) -> None:
    if bool(change) == args.same:
        sys.exit("바꿀 조건(--unmask, --aggregator, --no-attack) 중 하나 이상을 주거나, 재현 확인이면 --same만 주세요.")
    if "unmask" in change:
        if not source_ad.attack or args.no_attack:
            sys.exit("--unmask는 공격 실험에서만 씁니다 (공격 전체 해제는 --no-attack).")
        outside = [cid for cid in change["unmask"] if cid not in malicious]
        if outside:
            sys.exit(f"악성이 아닌 ID입니다: {outside} (이 실행의 악성 {list(malicious)})")


def server_command(cfg: ExperimentConfig, saved: dict, attack_defense: AttackDefenseConfig, port: int,
                   server_dir: Path, start_round: int, model_path: Path) -> list[str]:
    server_cfg = saved["server"]["config"]
    clip_flag = "--clip-before-median" if attack_defense.clip_before_median else "--no-clip-before-median"
    return [sys.executable, str(SERVE_SCRIPT), "--config", str(ROOT / server_cfg["config_file"]),
            "--seed", str(cfg.seed), "--port", str(port), "--log-dir", str(server_dir),
            "--env", str(ROOT / server_cfg["env_file"]), "--aggregator", attack_defense.aggregator,
            "--clip-norm", repr(attack_defense.clip_norm), clip_flag,
            "--start-round", str(start_round), "--init-model", str(model_path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="원래 실행 폴더 (experiments/ 기준, 예: 4_main/baseline-attack-fedavg-s0)")
    parser.add_argument("--from-round", type=int, required=True, dest="from_round",
                        help="이 라운드 집계 후 모델(models/round_XXX.pt)에서 출발")
    parser.add_argument("--rounds", type=int, help="몇 라운드 돌릴지 (기본: 원래 실행의 마지막 라운드까지)")
    parser.add_argument("--unmask", type=int, action="append", default=[], help="조작을 해제할 악성 ID (여러 번 가능)")
    parser.add_argument("--aggregator", choices=AGGREGATORS, help="방어(집계) 방식 교체")
    parser.add_argument("--no-attack", action="store_true", dest="no_attack", help="공격 전체 해제")
    parser.add_argument("--same", action="store_true", help="아무것도 바꾸지 않음 (원래 실행 재현 확인용)")
    parser.add_argument("--name", help=f"결과 폴더 이름 (experiments/{REPLAY_DIR}/<name>, 기본: 원래 이름-from라운드-바꾼 조건)")
    parser.add_argument("--port", type=int, help="서버 포트 (없으면 빈 포트를 고름)")
    args = parser.parse_args()

    run_dir = inside(EXPERIMENTS_DIR, args.run)
    source = load_source(run_dir)
    saved = source["config"]
    source_ad = source_attack_defense(saved)
    change = describe(args, source_ad)
    check_change(change, args, source_ad, saved["plan"]["malicious"])

    model_path = checkpoint_path(run_dir, args.from_round)
    if args.from_round < 0 or not model_path.exists():
        saved_rounds = sorted(int(p.stem.split("_")[1]) for p in (run_dir / "models").glob("round_*.pt"))
        sys.exit(f"라운드 {args.from_round}의 저장된 모델이 없습니다. 저장된 라운드: {saved_rounds}")
    last_round = source_last_round(saved)
    rounds = args.rounds if args.rounds is not None else last_round - args.from_round
    if rounds < 1:
        sys.exit(f"돌릴 라운드가 없습니다 (원래 마지막 라운드 {last_round}, 출발 {args.from_round}, --rounds {args.rounds}).")

    attack_defense = source_ad.with_overrides(aggregator=args.aggregator, attack=False if args.no_attack else None)
    cfg, plan = rebuild(saved, rounds, attack_defense)
    name = args.name or f"{run_dir.name}-from{args.from_round:03d}-{change_label(change)}"
    out_dir = new_out_dir(f"{REPLAY_DIR}/{name}")
    unmasked = merged_unmask(saved, change)
    if unmasked and not attack_defense.attack:
        unmasked = ()   # 공격을 끈 재실행에서는 해제할 조작이 없다
    plan = replace(plan, experiment_id=f"{REPLAY_DIR}/{name}", unmasked=unmasked)

    server_dir = out_dir / "server"
    server_dir.mkdir(parents=True)
    start_round = args.from_round + 1
    info = {"source": run_dir.relative_to(EXPERIMENTS_DIR).as_posix(), "from_round": args.from_round,
            "start_round": start_round, "rounds": rounds,
            "checkpoint": model_path.relative_to(ROOT).as_posix(), "change": change,
            "unmasked": list(unmasked),   # 원래 실행에서 이어받은 해제 포함
            "source_summary": {key: source["summary"].get(key)
                               for key in ("final_test_acc", "final_val_acc", "diverged_round", "rounds_completed")}}
    (out_dir / "replay.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    port = args.port or free_port()
    url = f"http://{HOST}:{port}"
    command = server_command(cfg, saved, attack_defense, port, server_dir, start_round, model_path)
    with (server_dir / "server.log").open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            wait_until_ready(url, server)
            summary = run_clients(cfg, out_dir, repo_root=REPO_ROOT, server_url=url, plan=plan,
                                  evaluate_test=True, start_round=start_round)
        finally:
            stop(server)
    result = {"replay": {key: summary.get(key) for key in ("final_test_acc", "diverged_round", "rounds_completed")},
              "source": info["source_summary"]}
    print(json.dumps({"out": out_dir.relative_to(ROOT).as_posix(), "change": change, **result},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
