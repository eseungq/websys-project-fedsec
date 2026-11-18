"""IID 분할 파일을 만든다. 팀에서 한 번 만들어 커밋하고, 모두 같은 파일을 쓴다.

예) python scripts/make_split.py --seed 0
"""

import argparse
from pathlib import Path

from fedsec.data.split import make_iid_split, save_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=5_000)
    args = parser.parse_args()

    split = make_iid_split(args.num_clients, args.val_size, args.seed)
    path = REPO_ROOT / "configs" / "splits" / f"iid_seed{args.seed}.json"
    save_split(split, path)
    sizes = [len(c) for c in split["clients"]]
    print(f"저장: {path.relative_to(REPO_ROOT)} (검증 {len(split['val'])}장, 클라이언트별 {sizes})")


if __name__ == "__main__":
    main()
