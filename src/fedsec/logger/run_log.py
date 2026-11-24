"""실험 1회의 결과 폴더: config.yaml, env.json, rounds.csv, clients.csv, class_acc.csv, client_acc.csv, summary.json."""

import csv
import json
from pathlib import Path

import yaml


class RunLog:
    def __init__(self, out_dir: Path, config: dict, env: dict):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (out_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
        self._rounds_path = out_dir / "rounds.csv"
        self._clients_path = out_dir / "clients.csv"
        self._class_acc_path = out_dir / "class_acc.csv"
        self._client_acc_path = out_dir / "client_acc.csv"

    def log_round(self, row: dict) -> None:
        _append_rows(self._rounds_path, [row])

    def log_clients(self, rows: list[dict]) -> None:
        """라운드마다 클라이언트별 한 줄 (업데이트 노름, 학습 손실 등)."""
        _append_rows(self._clients_path, rows)

    def log_class_acc(self, rows: list[dict]) -> None:
        """라운드(·평가 데이터) × 클래스 정확도."""
        _append_rows(self._class_acc_path, rows)

    def log_client_acc(self, rows: list[dict]) -> None:
        """라운드(·평가 데이터) × 클라이언트의 분포 가중 정확도."""
        _append_rows(self._client_acc_path, rows)

    def write_summary(self, summary: dict) -> None:
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_rows(path: Path, rows: list[dict]) -> None:
    """한 줄씩 바로 써서, 중간에 끊겨도 그때까지의 기록이 남게 한다."""
    if not rows:
        return
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
