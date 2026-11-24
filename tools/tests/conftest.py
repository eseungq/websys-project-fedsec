"""tools/ 의 스크립트를 모듈로 불러올 수 있게 경로를 더한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
