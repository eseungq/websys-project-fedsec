"""체크포인트 재실행 (명세 6단계).

저장된 실험의 특정 라운드 모델에서 출발해, 한 가지 조건만 바꾸고 나머지(환경·시드·참여·지연 일정)는
그대로 둔 채 다시 학습한다. 원래 실행과 결과를 비교해 원인 가설을 확인한다.

예 (members/배병욱 에서):
    # 5라운드 모델에서 출발, 악성 4번의 조작만 해제
    python -m fedsim.replay --run experiments/matrix/baseline-attack-fedavg-s0 --from-round 5 --disable-attacker 4
    # 같은 출발점에서 방어만 중앙값으로 교체
    python -m fedsim.replay --run experiments/matrix/baseline-attack-fedavg-s0 --from-round 5 --defense median
"""
import argparse
import json
import os

import torch

from .runner import ROOT, ServerProcess, run_experiment


def main():
    ap = argparse.ArgumentParser(description="체크포인트에서 조건 하나만 바꿔 재실행")
    ap.add_argument("--run", required=True, help="원래 실험 폴더")
    ap.add_argument("--from-round", type=int, required=True, help="이 라운드 끝의 모델에서 출발")
    ap.add_argument("--rounds", type=int, help="몇 라운드 더 돌릴지 (기본: 원래 끝 라운드까지)")
    ap.add_argument("--defense", choices=["fedavg", "clip_fedavg", "median"])
    ap.add_argument("--no-attack", action="store_true", help="공격 전체 해제")
    ap.add_argument("--disable-attacker", type=int, action="append", default=[], help="조작을 해제할 악성 ID")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    run_dir = os.path.join(ROOT, a.run) if not os.path.isabs(a.run) else a.run
    with open(os.path.join(run_dir, "config.json")) as f:
        orig = json.load(f)
    ckpt = torch.load(os.path.join(run_dir, "models", f"round_{a.from_round:03d}.pt"), weights_only=True)
    cfg = orig["base"]
    if orig.get("clip_bound"):
        cfg["clip_bound"] = orig["clip_bound"]
    defense = a.defense or orig["defense"]
    attack = orig["attack"] and not a.no_attack
    end = orig["start_round"] + orig["rounds"] - 1
    rounds = a.rounds or (end - a.from_round)

    change = []
    if a.defense:
        change.append(f"def-{a.defense}")
    if a.no_attack:
        change.append("noattack")
    change += [f"off{c}" for c in a.disable_attacker]
    name = f"from{a.from_round:03d}-" + ("_".join(change) or "same") + (f"-{a.tag}" if a.tag else "")
    out = os.path.join(run_dir, "replay", name)

    srv = ServerProcess()
    try:
        s = run_experiment(cfg, orig["env"], attack, defense, orig["seed"], out, srv.url, rounds=rounds,
                           start_round=a.from_round + 1, init_vec=ckpt["vector"],
                           disabled_attackers=a.disable_attacker)
    finally:
        srv.close()
    with open(os.path.join(run_dir, "summary.json")) as f:
        base = json.load(f)
    print(json.dumps({"original_test_acc": base["final_test_acc"], "replay_test_acc": s["final_test_acc"],
                      "diff_pp": round(s["final_test_acc"] - base["final_test_acc"], 4),
                      "original_diverged": base["diverged_round"], "replay_diverged": s["diverged_round"],
                      "out": os.path.relpath(out, ROOT)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
