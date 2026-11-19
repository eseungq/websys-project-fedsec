"""결과 폴더 여러 개를 비교표(Markdown, CSV)로 만든다. 표준 라이브러리 + PyYAML만 쓴다.

예) python tools/summarize.py experiments/sweep/[!_]*     # _tables(결과 표 폴더)는 빼고
    python tools/summarize.py experiments/lr*_r50 --epochs 50 --md out.md --csv out.csv
행은 last5_val(선택 기준) 높은 순. 테스트 정확도는 설정 선택에 쓰지 않으므로 표에 넣지 않는다.
"""

import argparse
import csv
import io
import statistics
import sys
from pathlib import Path

import yaml

LAST_N = 5                 # last5_val: 마지막 5라운드 검증 정확도 평균
PLATEAU_WINDOW = 10        # plateau 기준: 마지막 10라운드 평균
PLATEAU_MARGIN = 0.005     # 그 평균 − 0.5%p 이상이 되는 첫 라운드
DEFAULT_EPOCHS = 50        # 같은 데이터 통과 횟수 비교: epochs_seen = round × local_ep

ACC_KEYS = ("last5_val", "best_val")


def last_mean(rounds: list[dict], key: str, n: int) -> float:
    tail = rounds[-n:]
    return sum(r[key] for r in tail) / len(tail)


def best_val(rounds: list[dict]) -> tuple[float, int]:
    """가장 높은 검증 정확도와 그 라운드 (같으면 먼저 나온 라운드)."""
    best = max(rounds, key=lambda r: r["val_acc"])
    return best["val_acc"], best["round"]


def plateau_round(rounds: list[dict]) -> int:
    threshold = last_mean(rounds, "val_acc", PLATEAU_WINDOW) - PLATEAU_MARGIN
    return next(r["round"] for r in rounds if r["val_acc"] >= threshold)


def min_val_loss_round(rounds: list[dict]) -> int:
    return min(rounds, key=lambda r: r["val_loss"])["round"]


def norm_stats(norms: list[float]) -> tuple[float, float, float]:
    """전체 클라이언트 업데이트 노름의 중앙값, 90% 지점(선형 보간), 최대."""
    if len(norms) < 2:
        raise ValueError("업데이트 노름이 2개 이상 있어야 합니다.")
    p90 = statistics.quantiles(norms, n=10, method="inclusive")[8]
    return statistics.median(norms), p90, max(norms)


def round_times(rounds: list[dict]) -> tuple[float, float]:
    """2라운드부터의 평균 라운드 시간(1라운드뿐이면 그 값), 총 시간."""
    times = [r["total_sec"] for r in rounds]
    later = times[1:] or times
    return sum(later) / len(later), sum(times)


def val_at_epochs(rounds: list[dict], local_ep: int, epochs: int) -> float | None:
    """epochs_seen = round × local_ep 가 epochs가 되는 라운드의 검증 정확도. 없으면 None."""
    if epochs % local_ep:
        return None
    target = epochs // local_ep
    return next((r["val_acc"] for r in rounds if r["round"] == target), None)


def summarize_run(rounds: list[dict], norms: list[float], config: dict, epochs: int) -> dict:
    if not rounds:
        raise ValueError("rounds.csv에 라운드가 없습니다.")
    train = config["train"]
    best, best_round = best_val(rounds)
    norm_median, norm_p90, norm_max = norm_stats(norms)
    round_sec, total_sec = round_times(rounds)
    return {
        "lr": train["lr"],
        "momentum": train["momentum"],
        "weight_decay": train["weight_decay"],
        "local_ep": train["local_ep"],
        "local_bs": train["local_bs"],
        "seed": config["seed"],
        "rounds": config["rounds"],
        "last5_val": last_mean(rounds, "val_acc", LAST_N),
        "best_val": best,
        "best_round": best_round,
        "plateau_round": plateau_round(rounds),
        "min_val_loss_round": min_val_loss_round(rounds),
        "norm_median": norm_median,
        "norm_p90": norm_p90,
        "norm_max": norm_max,
        "round_sec": round_sec,
        "total_sec": total_sec,
        f"val_ep{epochs}": val_at_epochs(rounds, train["local_ep"], epochs),
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_run(run_dir: Path, epochs: int) -> dict:
    """결과 폴더 하나 → 표의 한 행. 끝나지 않은 실행(summary.json 없음)은 오류."""
    if not run_dir.is_dir():
        raise ValueError(f"결과 폴더가 없습니다: {run_dir}")
    if not (run_dir / "summary.json").exists():
        raise ValueError(f"끝나지 않은 실행입니다 (summary.json 없음): {run_dir}")
    rounds = [{**r, "round": int(r["round"])} for r in _read_csv(run_dir / "rounds.csv")]
    norms = [r["update_norm"] for r in _read_csv(run_dir / "clients.csv")]
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return {"name": run_dir.name, **summarize_run(rounds, norms, config, epochs)}


def _format_cell(key: str, value) -> str:
    if value is None:
        return "—"
    if key in ACC_KEYS or key.startswith("val_ep"):
        return f"{value * 100:.2f}%"
    if key.startswith("norm_"):
        return f"{value:.2f}"
    if key in ("round_sec", "total_sec"):
        return f"{value:.1f}"
    return str(value)


def to_markdown(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(_format_cell(k, row[k]) for k in keys) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _csv_value(value):
    if value is None:
        return ""
    return round(value, 6) if isinstance(value, float) else value


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: _csv_value(v) for k, v in row.items()} for row in rows)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="결과 폴더 (experiments/<name>)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="같은 데이터 통과 횟수 비교 기준 epochs_seen")
    parser.add_argument("--md", type=Path, help="Markdown 표 저장 경로 (없으면 화면에만 출력)")
    parser.add_argument("--csv", type=Path, help="CSV 저장 경로")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs는 1 이상이어야 합니다.")

    try:
        rows = [load_run(d, args.epochs) for d in args.run_dirs]
    except (OSError, KeyError, ValueError) as e:
        sys.exit(f"결과를 읽지 못했습니다: {e}")
    rows = sorted(rows, key=lambda r: r["last5_val"], reverse=True)

    markdown = to_markdown(rows)
    print(markdown, end="")
    if args.md:
        args.md.write_text(markdown, encoding="utf-8")
    if args.csv:
        args.csv.write_text(to_csv(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
