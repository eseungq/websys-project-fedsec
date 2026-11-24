"""환경 × {정상, 공격} × {FedAvg, 클리핑, 중앙값} × 시드 일괄 실행. 같은 설정(라운드 수·기본 설정)으로 끝난 실행만 건너뛰고,
설정이 다른 기존 결과는 덮어쓰지 않고 `-r<라운드>` 폴더에 따로 저장한다.

예: python -m fedsim.run_matrix --envs baseline delay partial label_skew --seeds 0 1 2 --rounds 40
"""
import argparse
import json
import os
import time

from .runner import DEFENSES, ROOT, ServerProcess, load_config, run_experiment, run_name


def same_setting(run_dir, cfg, rounds):
    """저장된 config.json의 라운드 수·기본 설정이 요청과 같은지."""
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return False
    return saved.get("rounds") == rounds and saved.get("start_round", 1) == 1 and saved.get("base") == cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--envs", nargs="+", default=["baseline", "delay", "partial", "label_skew"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--defenses", nargs="+", default=list(DEFENSES))
    ap.add_argument("--attacks", nargs="+", default=["normal", "attack"])
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--out", default="experiments/matrix")
    a = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, a.config))
    rounds = a.rounds or cfg["rounds"]
    jobs = [(e, at == "attack", d, s) for e in a.envs for s in a.seeds for at in a.attacks for d in a.defenses]
    srv = ServerProcess()
    try:
        for i, (env, attack, defense, seed) in enumerate(jobs, 1):
            name = run_name(env, attack, defense, seed)
            out = os.path.join(ROOT, a.out, name)
            if os.path.exists(os.path.join(out, "summary.json")):
                if same_setting(out, cfg, rounds):
                    print(f"[{i}/{len(jobs)}] 건너뜀 {name} (같은 설정으로 이미 완료)")
                    continue
                # 설정이 다른 기존 결과는 덮어쓰지 않고 라운드 수를 붙인 별도 폴더에 저장
                out = os.path.join(ROOT, a.out, f"{name}-r{rounds}")
                print(f"[{i}/{len(jobs)}] 기존 {name}와 설정이 달라 {os.path.basename(out)}에 따로 저장")
                if os.path.exists(os.path.join(out, "summary.json")) and same_setting(out, cfg, rounds):
                    continue
            t = time.time()
            print(f"[{i}/{len(jobs)}] {run_name(env, attack, defense, seed)}")
            s = run_experiment(cfg, env, attack, defense, seed, out, srv.url, rounds)
            print(f"   test={s['final_test_acc']:.2f}% diverged={s['diverged_round']} ({time.time() - t:.0f}s)")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
