"""환경 × {정상, 공격} × {FedAvg, 클리핑, 중앙값} × 시드 일괄 실행. 끝난 실행(summary.json 있음)은 건너뛴다.

예: python -m fedsim.run_matrix --envs baseline delay partial label_skew --seeds 0 1 2 --rounds 40
"""
import argparse
import os
import time

from .runner import DEFENSES, ROOT, ServerProcess, load_config, run_experiment, run_name


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
    jobs = [(e, at == "attack", d, s) for e in a.envs for s in a.seeds for at in a.attacks for d in a.defenses]
    srv = ServerProcess()
    try:
        for i, (env, attack, defense, seed) in enumerate(jobs, 1):
            out = os.path.join(ROOT, a.out, run_name(env, attack, defense, seed))
            if os.path.exists(os.path.join(out, "summary.json")):
                print(f"[{i}/{len(jobs)}] 건너뜀 {run_name(env, attack, defense, seed)}")
                continue
            t = time.time()
            print(f"[{i}/{len(jobs)}] {run_name(env, attack, defense, seed)}")
            s = run_experiment(cfg, env, attack, defense, seed, out, srv.url, a.rounds)
            print(f"   test={s['final_test_acc']:.2f}% diverged={s['diverged_round']} ({time.time() - t:.0f}s)")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
