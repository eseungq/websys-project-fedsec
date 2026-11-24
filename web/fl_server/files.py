"""결과 폴더 파일을 읽기만 하는 도구: experiments/ 밖을 가리키는 경로 거부, CSV·JSON·YAML 읽기.

화면에 그대로 보내므로 NaN·Infinity는 null로 바꾼다 (브라우저의 JSON.parse가 받지 못함).
"""

import csv
import json
import math
import re
from pathlib import Path

import yaml

# 한 파일 상한: 결과 폴더의 CSV는 수십 KB이므로 이보다 크면 잘못된 파일로 본다
MAX_FILE_BYTES = 8 * 1024 * 1024
INT_TEXT = re.compile(r"[+-]?[0-9]+")


class PathRejected(ValueError):
    """experiments/ 밖이거나 허용하지 않은 경로."""


class NotFound(LookupError):
    """요청한 결과 폴더·파일이 없음."""


def resolve_inside(base: Path, relative: str) -> Path:
    """base 아래 경로. ../, 절대 경로, 심볼릭 링크로 base 밖을 가리키면 PathRejected."""
    if not isinstance(relative, str) or not relative or "\x00" in relative or "\\" in relative:
        raise PathRejected("경로가 올바르지 않습니다.")
    root = base.resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise PathRejected("experiments/ 밖의 경로는 읽을 수 없습니다.")
    return target


def finite(value):
    """NaN·Infinity를 None으로 바꾼 새 값 (사전·목록은 안쪽까지)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def parse_cell(text: str | None):
    """CSV 한 칸: 빈칸은 None, True/False는 bool, 숫자는 int·float, 나머지는 문자열."""
    if text is None or text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    if INT_TEXT.fullmatch(text):
        return int(text)
    try:
        number = float(text)
    except ValueError:
        return text
    return number if math.isfinite(number) else None


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"파일이 너무 큽니다: {path.name}")
    return path.read_text(encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    """없는 파일은 빈 목록. 진행 중인 실행은 마지막 줄이 쓰이는 중일 수 있어 열 수가 맞지 않는 줄은 버린다."""
    if not path.is_file():
        return []
    reader = csv.DictReader(_read_text(path).splitlines())
    return [{key: parse_cell(value) for key, value in row.items()}
            for row in reader if None not in row and None not in row.values()]


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError:
        return None   # 쓰는 중이거나 깨진 파일: 없는 것으로 본다
    return finite(data) if isinstance(data, dict) else None


def read_yaml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(_read_text(path))
    except yaml.YAMLError:
        return None
    return finite(data) if isinstance(data, dict) else None
