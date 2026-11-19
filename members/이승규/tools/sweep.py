"""실행 목록(YAML)대로 run_experiment.py를 하나씩 순서대로 실행한다. 표준 라이브러리 + PyYAML만 쓴다.

예) python tools/sweep.py tools/plans/s1.yaml
    python tools/sweep.py tools/plans/s1.yaml --dry-run      # 실행할 명령만 출력
결과: experiments/sweep/<name>/
- 끝난 실행(summary.json 있음)은 건너뛴다 → 중간에 끊겨도 같은 명령으로 이어서 실행
- 끊긴 실행(폴더는 있는데 summary.json 없음)이 있으면 멈춘다 → 그 폴더를 지우거나 옮긴 뒤 다시 실행
- 실행이 실패하면 그 자리에서 멈추고 실패한 이름을 출력한다

실행 목록 형식:
    common: {seed: 0, rounds: 50}       # 모든 실행에 공통 (선택)
    runs:
      - {name: m0.5_lr0.025, momentum: 0.5, lr: 0.025}
"""

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

MEMBER_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = "sweep"
OPTIONS = ("seed", "rounds", "lr", "momentum", "weight_decay", "local_ep", "local_bs")
INT_OPTIONS = ("seed", "rounds", "local_ep", "local_bs")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_options(options: dict, where: str) -> None:
    unknown = set(options) - set(OPTIONS)
    if unknown:
        raise ValueError(f"{where}: 모르는 옵션 {sorted(unknown)} (가능: {', '.join(OPTIONS)})")
    for key, value in options.items():
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not is_number or (key in INT_OPTIONS and not isinstance(value, int)):
            raise ValueError(f"{where}: {key} 값이 잘못됐습니다: {value!r}")


def parse_plan(plan: dict) -> list[dict]:
    """실행 목록 → [{"name": ..., "options": {...}}]. common 값 위에 각 실행 값을 덮어쓴다."""
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list) or not plan["runs"]:
        raise ValueError("실행 목록에 runs 목록이 있어야 합니다.")
    common = plan.get("common") or {}
    _check_options(common, "common")

    runs, seen = [], set()
    for i, entry in enumerate(plan["runs"], start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"runs[{i}]: 이름과 옵션의 묶음이어야 합니다.")
        name = str(entry.get("name", ""))
        if not NAME_PATTERN.match(name):
            raise ValueError(f"runs[{i}]: 이름이 비었거나 쓸 수 없는 문자가 있습니다: {name!r}")
        if name in seen:
            raise ValueError(f"runs[{i}]: 이름이 겹칩니다: {name}")
        seen.add(name)
        options = {k: v for k, v in entry.items() if k != "name"}
        _check_options(options, f"runs[{i}] {name}")
        runs.append({"name": name, "options": {**common, **options}})
    return runs


def build_command(python: str, run: dict) -> list[str]:
    cmd = [python, "scripts/run_experiment.py", "--name", f"{SWEEP_DIR}/{run['name']}"]
    for key, value in run["options"].items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    return cmd


def config_mismatches(config: dict, options: dict) -> list[str]:
    """이미 있는 결과의 config.yaml이 요청한 값과 다른 항목들."""
    actual = {"seed": config.get("seed"), "rounds": config.get("rounds"), **(config.get("train") or {})}
    return [f"{key}: 요청 {value}, 기존 {actual.get(key)}" for key, value in options.items()
            if actual.get(key) is None or not math.isclose(actual[key], value)]


def run_status(out_dir: Path, options: dict) -> str:
    """"new"(실행할 것), "done"(건너뜀). 건너뛸 수 없는 상태면 ValueError."""
    if not out_dir.exists():
        return "new"
    if not (out_dir / "summary.json").exists():
        raise ValueError(f"끊긴 실행입니다 (summary.json 없음): {out_dir}\n"
                         "  그 폴더를 지우거나 옮긴 뒤 다시 실행하세요.")
    config = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    mismatches = config_mismatches(config, options)
    if mismatches:
        raise ValueError(f"같은 이름의 결과가 다른 설정입니다: {out_dir}\n  " + "\n  ".join(mismatches))
    return "done"


def run_sweep(runs: list[dict], root: Path, python: str, dry_run: bool = False) -> int:
    """하나씩 순서대로 실행. 모두 끝나면 0, 실패하면 1."""
    total = len(runs)
    for i, run in enumerate(runs, start=1):
        label = f"[{i}/{total}] {run['name']}"
        out_dir = root / "experiments" / SWEEP_DIR / run["name"]
        try:
            status = run_status(out_dir, run["options"])
        except (OSError, ValueError) as e:
            print(f"{label} 멈춤: {e}", flush=True)
            return 1
        if status == "done":
            print(f"{label} 건너뜀 (이미 끝남)", flush=True)
            continue

        cmd = build_command(python, run)
        print(f"{label} 실행: {' '.join(cmd[1:])}", flush=True)
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0 or not (out_dir / "summary.json").exists():
            print(f"{label} 실패 (종료 코드 {result.returncode}). 여기서 멈춥니다.", flush=True)
            return 1
    print(f"끝: {total}개" + (" (dry-run, 실행하지 않음)" if dry_run else ""), flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path, help="실행 목록 YAML")
    parser.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력")
    parser.add_argument("--root", type=Path, default=MEMBER_ROOT, help="run_experiment.py가 있는 폴더 (기본: members/이승규)")
    parser.add_argument("--python", default=sys.executable, help="실험에 쓸 Python (기본: 지금 Python)")
    args = parser.parse_args()

    try:
        runs = parse_plan(yaml.safe_load(args.plan.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as e:
        sys.exit(f"실행 목록을 읽지 못했습니다: {e}")
    sys.exit(run_sweep(runs, args.root.resolve(), args.python, args.dry_run))


if __name__ == "__main__":
    main()
