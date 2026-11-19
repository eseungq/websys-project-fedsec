"""실행 목록(YAML)대로 run_experiment.py를 하나씩 순서대로 실행한다. 표준 라이브러리 + PyYAML만 쓴다.

예) python tools/sweep.py tools/plans/s1.yaml
    python tools/sweep.py tools/plans/s1.yaml --dry-run      # 실행할 명령만 출력
결과: experiments/sweep/<name>/
- 끝난 실행(summary.json 있음)은 건너뛴다 → 중간에 끊겨도 같은 명령으로 이어서 실행.
  단, configs/base.yaml에 옵션을 덮어쓴 실제 설정이 기존 config.yaml과 다르면 멈춘다
- configs/base.yaml은 시작할 때 한 번 읽는다 → 탐색 도중에 고치지 말 것
- 끊긴 실행(폴더는 있는데 summary.json 없음)이 있으면 멈춘다 → 그 폴더를 지우거나 옮긴 뒤 다시 실행
- 실행이 실패하면 그 자리에서 멈추고 실패한 이름을 출력한다

실행 목록 형식:
    common: {seed: 0, rounds: 50}       # 모든 실행에 공통 (선택)
    runs:
      - {name: m0.5_lr0.025, momentum: 0.5, lr: 0.025}
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

MEMBER_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = "sweep"
BASE_CONFIG = "configs/base.yaml"      # run_experiment.py의 --config 기본값과 같음
OPTIONS = ("seed", "rounds", "lr", "momentum", "weight_decay", "local_ep", "local_bs")
INT_OPTIONS = ("seed", "rounds", "local_ep", "local_bs")
TOP_OPTIONS = ("seed", "rounds")       # 나머지 옵션은 train 아래 값


INT_TEXT = re.compile(r"[+-]?[0-9]+")


def whole_number(value) -> int:
    """정수 항목: 1, 1.0, "32"는 받고 1.9처럼 소수 부분이 있으면 거부한다 (load_config와 같음). true/false도 거부."""
    if isinstance(value, bool):
        raise ValueError(f"정수여야 합니다: {value!r}")
    if isinstance(value, int):
        return value  # float를 거치면 2**53보다 큰 값이 바뀌므로 그대로 돌려준다
    if isinstance(value, str):
        if INT_TEXT.fullmatch(value.strip()):
            return int(value)   # int()는 "1_0"도 10으로 받으므로 숫자만 있는지 먼저 본다
        if "_" in value:   # float()도 "1_0"을 10.0으로 받는다
            raise ValueError(f"정수여야 합니다: {value!r}")
        # "1.0" 같은 문자열은 아래에서 소수 부분을 확인한다
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"정수여야 합니다: {value!r}")
    return int(number)


# config.yaml에 기록되는 키와 형 (fedsec.experiment.config.load_config와 같은 규칙. tests/test_sweep_contract.py가 확인)
TOP_TYPES = {"seed": whole_number, "split": str, "data_root": str, "rounds": whole_number, "model": str,
             "eval_bs": whole_number}
TRAIN_TYPES = {"lr": float, "momentum": float, "weight_decay": float, "local_ep": whole_number,
               "local_bs": whole_number}
SUPPORTED_OPTIMIZERS = ("sgd",)
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


def _check_mapping(config, where: str) -> None:
    if not isinstance(config, dict) or not isinstance(config.get("train") or {}, dict):
        raise ValueError(f"{where}: 설정 묶음이 아닙니다.")


def _typed(values: dict, types: dict, where: str) -> dict:
    """config.yaml에 기록되는 키만 남기고 load_config와 같은 형으로 바꾼다. 빠진 키는 load_config처럼 거부."""
    missing = [key for key in types if key not in values]
    if missing:
        raise ValueError(f"{where}: 필수 항목이 없습니다: {', '.join(missing)}")
    try:
        return {key: cast(values[key]) for key, cast in types.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: 값의 형식이 잘못됐습니다: {e}") from e


def expected_config(base: dict, options: dict) -> dict:
    """base.yaml 위에 옵션을 덮어쓴 실제 설정 (run_experiment.py와 같은 규칙)."""
    top = {**base, **{k: v for k, v in options.items() if k in TOP_OPTIONS}}
    train = {**(base.get("train") or {}), **{k: v for k, v in options.items() if k not in TOP_OPTIONS}}
    return {**_typed(top, TOP_TYPES, BASE_CONFIG), "train": _typed(train, TRAIN_TYPES, f"{BASE_CONFIG} train")}


def _flatten(config: dict) -> dict:
    train = config.get("train") or {}
    return {**{k: v for k, v in config.items() if k != "train"}, **{f"train.{k}": v for k, v in train.items()}}


def _same(a, b) -> bool:
    if all(isinstance(x, int) and not isinstance(x, bool) for x in (a, b)):
        return a == b
    numbers = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
    return math.isclose(a, b) if numbers else a == b


def config_mismatches(saved: dict, expected: dict) -> list[str]:
    """기존 config.yaml이 실제로 쓸 설정과 다른 항목들. 기존 쪽에만 있는 값(num_clients 등)은 보지 않는다."""
    actual = _flatten(saved)
    return [f"{key}: 지금 설정 {value}, 기존 {actual.get(key)}" for key, value in _flatten(expected).items()
            if key not in actual or not _same(actual[key], value)]


def run_status(out_dir: Path, expected: dict) -> str:
    """"new"(실행할 것), "done"(건너뜀). 건너뛸 수 없는 상태면 ValueError."""
    if not out_dir.exists():
        return "new"
    if not (out_dir / "summary.json").exists():
        raise ValueError(f"끊긴 실행입니다 (summary.json 없음): {out_dir}\n"
                         "  그 폴더를 지우거나 옮긴 뒤 다시 실행하세요.")
    try:
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        summary = None
    if not isinstance(summary, dict):
        raise ValueError(f"완료 기록이 깨졌습니다 (summary.json을 읽을 수 없음): {out_dir}\n"
                         "  그 폴더를 지우거나 옮긴 뒤 다시 실행하세요.")
    config = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    _check_mapping(config, str(out_dir / "config.yaml"))
    mismatches = config_mismatches(config, expected)
    if mismatches:
        raise ValueError(f"같은 이름의 결과가 다른 설정입니다: {out_dir}\n  " + "\n  ".join(mismatches))
    return "done"


def load_base(root: Path) -> dict:
    """configs/base.yaml. run_experiment.py가 거부할 설정이면 기존 결과를 보기 전에 멈춘다."""
    path = root / BASE_CONFIG
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_mapping(base, str(path))
    optimizer = (base.get("train") or {}).get("optimizer", "sgd")
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"{path}: 지원하지 않는 옵티마이저입니다: {optimizer} (가능: {', '.join(SUPPORTED_OPTIMIZERS)})")
    return base


def run_sweep(runs: list[dict], root: Path, python: str, dry_run: bool = False) -> int:
    """하나씩 순서대로 실행. 모두 끝나면 0, 실패하면 1."""
    try:
        base = load_base(root)
    except (OSError, yaml.YAMLError, ValueError) as e:
        print(f"기본 설정을 읽지 못했습니다: {e}", flush=True)
        return 1
    total = len(runs)
    for i, run in enumerate(runs, start=1):
        label = f"[{i}/{total}] {run['name']}"
        out_dir = root / "experiments" / SWEEP_DIR / run["name"]
        try:
            status = run_status(out_dir, expected_config(base, run["options"]))
        except (OSError, yaml.YAMLError, ValueError) as e:
            print(f"{label} 멈춤: {e}", flush=True)
            return 1
        if status == "done":
            print(f"{label} 건너뜀 (이미 끝남)", flush=True)
            continue

        cmd = build_command(python, run)
        print(f"{label} 실행: {' '.join(cmd[1:])}", flush=True)
        if dry_run:
            continue
        result = subprocess.run(cmd, check=False, cwd=root)
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
