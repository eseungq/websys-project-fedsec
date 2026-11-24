"""분할 파일을 만들거나 클래스 분포를 출력한다. 팀에서 한 번 만들어 커밋하고, 모두 같은 파일을 쓴다.

예) python scripts/make_split.py --seed 0                        # IID (레포 루트 configs/splits/)
    python scripts/make_split.py --kind label_skew --alpha 0.5   # 레이블 편향 (이승규 폴더 configs/splits/)
    python scripts/make_split.py --show-dist configs/splits/iid_seed0.json
"""

import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10

from fedsec.data.split import class_counts, load_split, make_iid_split, make_label_skew_split, save_split, skew_score

# IID 분할은 팀 공용이라 레포 루트의 configs/splits/에 저장한다.
REPO_ROOT = Path(__file__).resolve().parents[1]
# 레이블 편향 분할은 머지 전까지 이승규 폴더에 둔다 (11/24 머지 때 옮길 후보).
ROOT = Path(__file__).resolve().parents[1]


def load_train_labels(data_root: Path) -> list[int]:
    """내려받아 둔 CIFAR-10 학습 레이블. 없으면 새로 받지 않고 오류를 낸다."""
    try:
        return list(CIFAR10(root=str(data_root), train=True, download=False).targets)
    except RuntimeError as error:
        raise FileNotFoundError(f"CIFAR-10이 {data_root}에 없습니다. scripts/run_experiment.py를 한 번 실행해 받으세요.") from error


def print_distribution(split: dict, labels: list[int]) -> None:
    counts = class_counts(split, labels)
    print("클라이언트 | 표본 수 | 클래스 0~9 이미지 수")
    for cid, row in enumerate(counts):
        print(f"{cid:>10} | {sum(row):>7} | {' '.join(f'{c:>4}' for c in row)}")
    print(f"편향 정도(클라이언트별 총변동거리 평균, 0=전체와 같음): {skew_score(counts):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--kind", choices=["iid", "label_skew"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.5, help="레이블 편향 Dirichlet 농도 (작을수록 치우침)")
    parser.add_argument("--base", default="configs/splits/iid_seed0.json",
                        help="레이블 편향의 기준 IID 분할 (레포 루트 기준). 검증 세트·이미지 풀·표본 수를 그대로 쓴다")
    parser.add_argument("--data-root", default="data", help="CIFAR-10 위치 (레포 루트 기준)")
    parser.add_argument("--show-dist", metavar="SPLIT", help="분할 파일의 클래스 분포만 출력 (레포 루트 기준 경로)")
    args = parser.parse_args()

    if args.show_dist:
        print_distribution(load_split(REPO_ROOT / args.show_dist), load_train_labels(REPO_ROOT / args.data_root))
        return

    if args.kind == "iid":
        split = make_iid_split(args.num_clients, args.val_size, args.seed)
        path = REPO_ROOT / "configs" / "splits" / f"iid_seed{args.seed}.json"
        save_split(split, path)
        sizes = [len(c) for c in split["clients"]]
        print(f"저장: {path.relative_to(REPO_ROOT)} (검증 {len(split['val'])}장, 클라이언트별 {sizes})")
        return

    labels = load_train_labels(REPO_ROOT / args.data_root)
    split = make_label_skew_split(load_split(REPO_ROOT / args.base), labels, args.alpha, args.seed)
    split = {**split, "base": args.base}
    path = ROOT / "configs" / "splits" / f"label_skew_a{args.alpha:g}_seed{args.seed}.json"
    save_split(split, path)
    print(f"저장: {path.relative_to(REPO_ROOT)} (기준 {args.base}, 검증 {len(split['val'])}장)")
    print_distribution(split, labels)


if __name__ == "__main__":
    main()
