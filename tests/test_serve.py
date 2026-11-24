"""serve.py가 /status·config.yaml에 남기는 설정 파일 경로."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from serve import shown_path


def test_shown_path_makes_absolute_member_path_relative():
    # run_server_experiment.py는 절대 경로를 넘기므로, 그대로 기록하면 PC 사용자 이름이 결과 파일에 남는다
    assert shown_path(str(ROOT / "configs" / "base.yaml")) == "configs/base.yaml"


def test_shown_path_keeps_relative_path():
    assert shown_path("configs/env/delay.yaml") == "configs/env/delay.yaml"


def test_shown_path_keeps_path_outside_member_folder():
    assert shown_path("/tmp/other.yaml") == "/tmp/other.yaml"
