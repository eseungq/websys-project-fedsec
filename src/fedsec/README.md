# src/fedsec — 실험 엔진

웹과 독립적으로 동작하는 순수 Python 패키지입니다. 웹(`web/`)은 이 패키지를 호출만 합니다.

| 모듈 | 역할 |
|---|---|
| `data/` | CIFAR-10 로드, 분할 파일 읽기, 레이블 편향 분할 (클라이언트별 표본 수 유지) |
| `models/` | 학습용 CNN(`cnn.py`), DLG용 LeNet(`lenet.py`) |
| `client/` | 로컬 학습과 업데이트(가중치 차이) 계산(`local_train.py`), 가상 클라이언트 실행 프로그램(`program.py`) |
| `server/` | Flask REST API(`app.py`), 서버 상태·집계(`state.py`), 제출 검증(`validation.py`), 환경별 운영 규칙(`scenario.py`) |
| `aggregation/` | FedAvg 가중 평균, 업데이트 노름 |
| `defenses/` | 노름 클리핑(`clipping.py`), 좌표별 중앙값(`median.py`) |
| `attacks/` | 내부자 협력 공격 −5배(`collusion.py`), DLG 복원과 가우시안 노이즈 방어(`dlg.py`) |
| `environments/` | 악성·지연·참여 일정, 가상 시간, 채택 판정 |
| `experiment/` | 설정 읽기, 시드, 평가, 클래스별·클라이언트별 성능, 서버 없는 실행기, 비교표(`comparison.py`)·막대그래프(`bar_chart.py`) |
| `logger/` | 학습 로그(`run_log.py`), 서버 기록(`server_log.py`) |

특정 라운드부터 다시 실행하는 재실행 도구는 `scripts/replay.py`에 있습니다.

> 규칙: 집계·방어 함수(`aggregation/`, `defenses/`)와 서버는 악성 ID 정답을 인자로 받지 않습니다.
