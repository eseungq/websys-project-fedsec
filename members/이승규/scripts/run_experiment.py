"""정상 FedAvg 실험 1회를 실행한다. 설정 파일 값을 명령줄로 덮어쓸 수 있다.

예) python scripts/run_experiment.py --name baseline
    python scripts/run_experiment.py --name lr0.05 --lr 0.05 --rounds 10
    python scripts/run_experiment.py --name final --evaluate-test     # 설정 확정 후에만 테스트셋 평가
결과: members/이승규/experiments/<name>/ 에 config.yaml, env.json, rounds.csv, clients.csv, summary.json
"""

import argparse
import json
from pathlib import Path

from fedsec.experiment.config import load_config
from fedsec.experiment.runner import run_fedavg
from run_clients import new_out_dir

# 기본 설정·결과는 개인 폴더(members/이승규/), 공용 분할 파일·데이터는 레포 루트 기준
MEMBER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMBER_ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="결과 폴더 이름 (experiments/<name>)")
    parser.add_argument("--config", default="configs/base.yaml", help="members/이승규/ 기준 상대 경로 또는 절대 경로")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay")
    parser.add_argument("--local-ep", type=int, dest="local_ep")
    parser.add_argument("--local-bs", type=int, dest="local_bs")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="마지막에 공식 테스트셋도 평가 (설정 확정 후에만. 설정 비교는 검증 정확도로)")
    args = parser.parse_args()

    out_dir = new_out_dir(args.name)

    overrides = {k: getattr(args, k) for k in ("seed", "rounds", "lr", "momentum", "weight_decay", "local_ep", "local_bs")}
    cfg = load_config(MEMBER_ROOT / args.config).with_overrides(**overrides)
    summary = run_fedavg(cfg, out_dir, repo_root=REPO_ROOT, evaluate_test=args.evaluate_test)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
