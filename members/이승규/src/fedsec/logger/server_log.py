"""서버 기록 파일: server_rounds.csv(라운드별 집계 요약), exclusions.csv(제외된 제출과 사유)."""

import json
from pathlib import Path

from fedsec.logger.run_log import _append_rows

FORMULA_PREFIXES = ("=", "+", "-", "@")


def _cell(value):
    """목록·사전은 CSV 한 칸에 들어가도록 JSON 문자열로 바꾸고, 수식으로 시작하는 문자열은 앞에 '를 붙인다."""
    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str) and value[:1] in FORMULA_PREFIXES:
        return "'" + value   # 표 프로그램이 수식으로 실행하지 않게
    return value


class ServerLog:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._rounds_path = out_dir / "server_rounds.csv"
        self._exclusions_path = out_dir / "exclusions.csv"
        existing = [p.name for p in (self._rounds_path, self._exclusions_path) if p.exists()]
        if existing:  # 이어 쓰면 이전 실행의 라운드 1과 섞인다
            raise FileExistsError(f"서버 기록 폴더 {out_dir}에 이전 기록이 있습니다: {', '.join(existing)}. 다른 --log-dir을 쓰세요.")

    def log_round(self, record: dict) -> None:
        _append_rows(self._rounds_path, [{k: _cell(v) for k, v in record.items()}])

    def log_exclusion(self, record: dict) -> None:
        _append_rows(self._exclusions_path, [{k: _cell(v) for k, v in record.items()}])
