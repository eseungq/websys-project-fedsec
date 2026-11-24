"""4단계 본 실험의 비교표: runs.csv → 공격 피해·방어 개선·정상 손실 (명세 5단계 ④·⑤).

같은 환경·시드의 최종 테스트 정확도 차이를 %p로 계산한다. 발산한 실행(diverged_round 있음)은 40라운드 모델이
없으므로 주 표에는 값 없이 발산 라운드만 보이고, 발산 전 모델로 뺀 값은 참고값으로 따로 둔다
(학습 길이가 달라 같은 열에 섞지 않음, docs/stages/4_운영환경.md 한계).
"""

import csv
import io
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

FINISHED = ("done", "diverged")   # run_matrix.FINISHED와 같음
DEFENSES = ("clip_fedavg", "median")
DEFENSE_NAMES = {"clip_fedavg": "클리핑", "median": "중앙값"}
CONDITIONS = tuple((attack, agg) for attack in ("normal", "attack") for agg in ("fedavg", *DEFENSES))
ATTACK_NAMES = {"normal": "정상", "attack": "공격"}
AGG_NAMES = {"fedavg": "FedAvg", **DEFENSE_NAMES}

# (지표, 방어, 빼지는 실행, 빼는 실행). 명세: 공격 피해 = 정상 FedAvg − 공격 FedAvg,
# 방어 개선 = 공격+방어 − 공격 FedAvg, 정상 손실 = 정상 FedAvg − 정상+방어
METRICS = (
    ("attack_damage", "", ("normal", "fedavg"), ("attack", "fedavg")),
    *(("defense_gain", d, ("attack", d), ("attack", "fedavg")) for d in DEFENSES),
    *(("normal_loss", d, ("normal", "fedavg"), ("normal", d)) for d in DEFENSES),
)
METRIC_NAMES = {"attack_damage": "공격 피해", "defense_gain": "방어 개선", "normal_loss": "정상 손실"}
CSV_COLUMNS = ("env", "metric", "defense", "seed", "value_pp", "reference_pp", "diverged_rounds")

Key = tuple[str, str, str, int]   # (환경, normal|attack, 집계, 시드)


@dataclass(frozen=True)
class RunResult:
    acc: float | None             # 최종 테스트 정확도 (1라운드 발산이면 None)
    diverged_round: int | None    # 발산한 라운드 (발산하지 않았으면 None)


@dataclass(frozen=True)
class Difference:
    value: float | None           # %p. 두 실행 모두 발산하지 않았을 때만 (주 표)
    reference: float | None       # 발산 전 모델도 그대로 뺀 값 (참고)
    diverged_rounds: tuple[int, ...]


def difference(minuend: RunResult, subtrahend: RunResult) -> Difference:
    rounds = tuple(r.diverged_round for r in (minuend, subtrahend) if r.diverged_round is not None)
    both = minuend.acc is not None and subtrahend.acc is not None
    reference = (minuend.acc - subtrahend.acc) * 100 if both else None
    return Difference(None if rounds else reference, reference, rounds)


def parse_runs(rows: Iterable[dict], rounds: int) -> dict[Key, RunResult]:
    """runs.csv 행 → 실행별 결과. rounds는 목표 라운드 수 (본 실험은 base.yaml의 40).

    끝나지 않았거나 값이 잘못된 실행, 목표와 다른 라운드 수, 중복 실행이 있으면 ValueError.
    """
    rows = list(rows)
    keys = [(_field(row, "env"), _field(row, "attack"), _field(row, "aggregator"), _whole(row, "seed"))
            for row in rows]
    counts = Counter(keys)
    duplicates = sorted(_field(row, "name") for row, key in zip(rows, keys) if counts[key] > 1)
    if duplicates:
        raise ValueError(f"runs.csv에 중복된 실행이 있습니다: {duplicates}")
    return {key: _parse_result(row, rounds) for key, row in zip(keys, rows)}


def _field(row: dict, key: str) -> str:
    """빈 칸과, 잘린 행에서 csv.DictReader가 넣는 None을 모두 ""로."""
    return (row.get(key) or "").strip()


def _whole(row: dict, key: str) -> int:
    text = _field(row, key)
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{key}가 정수가 아닙니다: {_field(row, 'name')} ({text!r})") from None


def _parse_result(row: dict, rounds: int) -> RunResult:
    name, status = _field(row, "name"), _field(row, "status")
    if status not in FINISHED:
        raise ValueError(f"끝나지 않은 실행입니다 ({status or '상태 없음'}): {name} → run_matrix.py로 마저 실행")
    diverged = _diverged_round(row)
    _check_rounds(row, diverged, rounds)
    # run_matrix.py는 1라운드 집계 직후 발산(평가할 모델 없음)만 diverged로 두고 정확도를 비운다.
    # 발산 전 모델로 평가한 실행(2라운드 이후 발산)은 done
    accuracies = {key: _field(row, key) for key in ("final_val_acc", "final_test_acc")}
    if status == "diverged":
        if diverged != 1 or any(accuracies.values()):
            raise ValueError(f"diverged 상태는 1라운드 발산이고 검증·테스트 정확도가 비어 있어야 합니다: {name} "
                             f"(발산 라운드 {diverged}, 정확도 {accuracies})")
        return RunResult(None, diverged)
    if diverged == 1:
        raise ValueError(f"1라운드 발산은 평가할 모델이 없으므로 diverged 상태여야 합니다: {name} (상태 done)")
    return RunResult(_accuracy(row), diverged)


def _check_rounds(row: dict, diverged: int | None, rounds: int) -> None:
    """run_matrix.diverged_problems와 같은 기준: 발산은 공격 실험만, 1~목표 라운드, 완료 라운드 = 발산 − 1."""
    name, completed = _field(row, "name"), _whole(row, "rounds_completed")
    if diverged is None:
        if completed != rounds:
            raise ValueError(f"완료 라운드 {completed}가 목표 {rounds}라운드와 다릅니다: {name}")
        return
    if _field(row, "attack") != "attack":
        raise ValueError(f"정상 실행이 발산했습니다 (설정 문제로 보고 비교하지 않음): {name}")
    if not 1 <= diverged <= rounds or completed != diverged - 1:
        raise ValueError(f"발산 라운드 {diverged}와 완료 라운드 {completed}가 목표 {rounds}라운드와 맞지 않습니다: {name}")


def _diverged_round(row: dict) -> int | None:
    text = _field(row, "diverged_round")
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"발산 라운드는 1 이상의 정수여야 합니다: {_field(row, 'name')} ({text!r})")
    return value


def _accuracy(row: dict) -> float:
    text = _field(row, "final_test_acc")
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and 0 <= value <= 1):
        raise ValueError(f"최종 테스트 정확도가 0~1 사이 숫자가 아닙니다: {_field(row, 'name')} ({text!r})")
    return value


def _run(results: dict[Key, RunResult], env: str, condition: tuple[str, str], seed: int) -> RunResult:
    key = (env, *condition, seed)
    if key not in results:
        raise ValueError(f"runs.csv에 없는 실행입니다: {env}-{condition[0]}-{condition[1]}-s{seed}")
    return results[key]


def compare(results: dict[Key, RunResult], envs: Sequence[str], seeds: Sequence[int]) -> list[dict]:
    """환경 × 지표 × 시드마다 한 행: value(주 표), reference(참고), diverged_rounds."""
    rows = []
    for env in envs:
        for metric, defense, minuend, subtrahend in METRICS:
            for seed in seeds:
                diff = difference(_run(results, env, minuend, seed), _run(results, env, subtrahend, seed))
                rows.append({"env": env, "metric": metric, "defense": defense, "seed": seed, "value": diff.value,
                             "reference": diff.reference, "diverged_rounds": diff.diverged_rounds})
    return rows


def _number(value: float, signed: bool) -> str:
    if signed and round(value, 2) == 0:
        return "0.00"   # 반올림해 0이면 부호 없이 ("−0.00" 방지)
    return f"{value:+.2f}".replace("-", "−") if signed else f"{value:.2f}"


def _rounds_text(rounds: tuple[int, ...]) -> str:
    return "·".join(str(r) for r in rounds)


def format_cell(values: Sequence[float | None], diverged: Sequence[tuple[int, ...]], signed: bool = True) -> str:
    """시드별 값과 평균: "+0.18 / −0.15 (−0.04)". 모든 시드가 발산하면 발산 라운드만, 일부면 평균 없음."""
    if all(v is None for v in values):
        return f"발산 ({' / '.join(_rounds_text(r) for r in diverged)}라운드)"
    parts = [_number(v, signed) if v is not None else f"발산({_rounds_text(r)}라운드)" for v, r in zip(values, diverged)]
    mean = "(평균 없음)" if None in values else f"({_number(sum(values) / len(values), signed)})"
    return f"{' / '.join(parts)} {mean}"


def _table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in body]) + "\n"


def _accuracy_cell(runs: Sequence[RunResult]) -> str:
    return format_cell([None if r.diverged_round else r.acc * 100 for r in runs],
                       [(r.diverged_round,) if r.diverged_round else () for r in runs], signed=False)


def accuracy_table(results: dict[Key, RunResult], envs: Sequence[str], seeds: Sequence[int]) -> str:
    """환경 × 6조건의 최종 테스트 정확도(%). 발산한 실행은 정확도 대신 발산 라운드."""
    header = ["환경", *(f"{ATTACK_NAMES[a]} {AGG_NAMES[g]}" for a, g in CONDITIONS)]
    body = [[env, *(_accuracy_cell([_run(results, env, condition, seed) for seed in seeds])
                    for condition in CONDITIONS)] for env in envs]
    return _table(header, body)


def _columns() -> list[tuple[str, str]]:
    return [(metric, defense) for metric, defense, *_ in METRICS]


def _column_name(metric: str, defense: str) -> str:
    return f"{METRIC_NAMES[metric]} ({DEFENSE_NAMES[defense]})" if defense else METRIC_NAMES[metric]


def _cells(rows: list[dict], env: str, metric: str, defense: str) -> list[dict]:
    return [r for r in rows if (r["env"], r["metric"], r["defense"]) == (env, metric, defense)]


def _main_cell(cells: list[dict]) -> str:
    return format_cell([c["value"] for c in cells], [c["diverged_rounds"] for c in cells])


def _reference_cell(cells: list[dict]) -> str:
    if not any(c["diverged_rounds"] for c in cells):
        return "—"
    return format_cell([c["reference"] for c in cells], [c["diverged_rounds"] for c in cells])


def to_markdown(rows: list[dict], envs: Sequence[str]) -> str:
    """주 표(40라운드끼리)와, 발산한 칸이 있으면 발산 전 모델 기준 참고 표."""
    header = ["환경", *(_column_name(m, d) for m, d in _columns())]
    main = [[env, *(_main_cell(_cells(rows, env, m, d)) for m, d in _columns())] for env in envs]
    text = "명세의 세 값 (%p, 시드별 값과 괄호 안 평균)\n\n" + _table(header, main)
    if not any(r["diverged_rounds"] for r in rows):
        return text
    reference = [[env, *(_reference_cell(_cells(rows, env, m, d)) for m, d in _columns())] for env in envs]
    return (text + "\n참고: 발산 전 모델 기준 (발산한 칸만. 학습 길이가 달라 위 표와 섞지 않는다)\n\n"
            + _table(header, reference))


def _csv_number(value: float | None) -> float | str:
    return "" if value is None else round(value, 4)


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    writer.writerows([r["env"], r["metric"], r["defense"], r["seed"], _csv_number(r["value"]),
                      _csv_number(r["reference"]), _rounds_text(r["diverged_rounds"])] for r in rows)
    return buf.getvalue()
