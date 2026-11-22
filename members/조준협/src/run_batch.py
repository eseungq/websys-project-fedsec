import os
import sys
import json
import time
import argparse
import itertools
import subprocess
import requests

# 하이퍼파라미터 설정부터 하고 나서 실행
# 터미널1로 server.py 실행 후, 터미널2로 run_batch.py 실행 (그전에 기본으로 돌려 클리핑 상한 확인하기)
# 실험 완료 후 evaluator.py, plot_results.py를 통해 정량 평가표 및 비교 그래프 생성

def is_experiment_completed(log_path: str, expected_epochs: int) -> bool:
    """이미 해당 조합의 실험이 목표 라운드까지 완료되었는지 확인 (중단 후 재시작 시 건너뛰기용)"""
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = data.get("round_history", [])
        return len(history) >= expected_epochs
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="24개 조합 × 다중 시드 자동 배치 실행기")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["cnn", "simple_cnn"])
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local_ep", type=int, default=1)
    parser.add_argument("--local_bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--clip_norm", type=float, default=2.5, help="정상 기준 실험에서 확정한 클리핑 상한")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="반복할 시드 목록 (쉼표 구분)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU 사용 시 번호 (예: 0)")
    parser.add_argument("--force", action="store_true", help="기존 완료 로그가 있어도 덮어쓰고 재실행")
    args = parser.parse_args()

    # 1. 서버 연결 확인
    try:
        res = requests.get(f"{args.server_url}/status", timeout=3)
        if res.status_code != 200:
            print("[오류] 서버 응답이 정상적이지 않습니다. server.py가 실행 중인지 확인하세요.")
            return
    except Exception:
        print("[오류] server.py에 연결할 수 없습니다. 먼저 터미널에서 'python server.py'를 실행해 주세요.")
        return

    # 2. 4개 환경 × 2개 공격 × 3개 방어 × N개 시드 조합 생성
    envs = ["baseline", "delay", "partial", "noniid"]
    attacks = ["none", "collusion"]
    defenses = ["fedavg", "clipped_fedavg", "clipped_median"]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    combinations = list(itertools.product(seeds, envs, attacks, defenses))
    total_runs = len(combinations)

    print("=================================================================")
    print(f"총 {total_runs}개 실험 조합 자동 실행 시작 (시드 {len(seeds)}개 × 24개 조건)")
    print(f"설정: model={args.model}, opt={args.optimizer}, lr={args.lr}, epochs={args.epochs}, clip_norm={args.clip_norm}")
    print("=================================================================")

    batch_start = time.time()
    completed_times = []

    for idx, (seed, env, attack, defense) in enumerate(combinations, start=1):
        exp_id = f"{args.model}_{env}_{attack}_{defense}_{args.optimizer}_s{seed}"
        log_path = os.path.join("../save/logs", f"log_{exp_id}.json")

        if not args.force and is_experiment_completed(log_path, args.epochs):
            print(f"[{idx:2d}/{total_runs}] ⏭️ 이미 완료됨 (건너뜀): {exp_id}")
            continue

        print(f"\n[{idx:2d}/{total_runs}] ▶️ 실행 시작: {exp_id}")
        run_start = time.time()

        cmd = [
            sys.executable, "client_runner.py",
            "--server_url", args.server_url,
            "--model", args.model,
            "--env", env,
            "--attack", attack,
            "--defense", defense,
            "--optimizer", args.optimizer,
            "--lr", str(args.lr),
            "--local_ep", str(args.local_ep),
            "--local_bs", str(args.local_bs),
            "--epochs", str(args.epochs),
            "--clip_norm", str(args.clip_norm),
            "--seed", str(seed),
        ]
        if args.gpu is not None:
            cmd.extend(["--gpu", str(args.gpu)])

        subprocess.run(cmd, check=True)

        run_elapsed = time.time() - run_start
        completed_times.append(run_elapsed)
        avg_time = sum(completed_times) / len(completed_times)
        remaining_runs = total_runs - idx
        est_rem_min = (avg_time * remaining_runs) / 60.0

        print(f"[{idx:2d}/{total_runs}] ✅ 완료 ({run_elapsed:.1f}초 소요) | 남은 예상 시간: 약 {est_rem_min:.1f}분")

    total_min = (time.time() - batch_start) / 60.0
    print("\n=================================================================")
    print(f"🎉 전체 배치 실험 완료! (총 소요 시간: {total_min:.1f}분)")
    print("=================================================================")

    # 3. 정량 평가표(CSV) 및 비교 그래프(PNG) 자동 생성
    print("\n[후처리 1/2] 정량 평가표(%p) CSV 생성 중...")
    subprocess.run([sys.executable, "evaluator.py"], check=True)

    print("[후처리 2/2] 비교 그래프(PNG) 생성 중...")
    subprocess.run([sys.executable, "plot_results.py"], check=True)
    print("\n모든 실험, 평가표(../save/tables), 그래프(../save/plots) 생성이 완료되었습니다!")


if __name__ == "__main__":
    main()