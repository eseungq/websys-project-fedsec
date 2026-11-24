# 설치·실행 설명서

명령은 모두 **레포 루트**에서 실행합니다.

## 1. 준비

| 항목 | 값 |
|---|---|
| Python | 3.10 이상 (실험은 3.13) |
| PyTorch | 2.10.0 이상. Apple Silicon은 MPS, NVIDIA는 CUDA, 없으면 CPU를 자동으로 고릅니다 |
| 디스크 | CIFAR-10 약 170MB, 본 실험 72회의 모델 파일 약 860MB |

```bash
git clone https://github.com/eseungq/websys-project-fedsec.git
cd websys-project-fedsec
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest -q                     # 설치 확인 (집계·방어 수치 예제 포함)
```

macOS에서 한글 폴더 이름이 깨지면 한 번만 설정합니다.
```bash
git config --global core.precomposeunicode true
git config --global core.quotepath false
```

**데이터:** 처음 학습할 때 CIFAR-10을 `data/`에 자동으로 내려받습니다. 클라이언트별 분할은 `configs/splits/`의 파일(커밋됨)을 그대로 씁니다.

## 2. 실행

### 웹 대시보드 (발표 시연)
```bash
python web/app.py                            # 브라우저에서 http://127.0.0.1:8765/
```
- 연합학습 서버와 대시보드가 한 서버·한 포트에서 뜹니다.
- **조회형:** 끝난 실험(본 실험 72회, 재실행), 비교표, DLG 결과를 봅니다.
- **실행형:** 환경·공격·집계·시드·라운드 수(최대 10)를 골라 짧은 실험을 돌립니다. 결과는 `experiments/live/`에 남습니다.

### 실험 1회 (서버를 띄우고 → 가상 클라이언트 10명 실행 → 서버 종료)
```bash
python scripts/run_server_experiment.py --name demo --rounds 3
python scripts/run_server_experiment.py --name d_atk_med --env configs/env/delay.yaml --attack --aggregator median --evaluate-test
```
서버와 클라이언트를 터미널 두 개로 따로 띄우는 방법은 [`docs/stages/2_서버-API.md`](../stages/2_서버-API.md#실행)에 있습니다.

### 본 실험 72회와 비교표
```bash
python scripts/run_main.py --dry-run       # 상태만 확인 (이미 끝난 실행은 건너뜀)
python scripts/run_main.py                 # 남은 실행을 이어서 (중단해도 다시 실행하면 이어짐)
python scripts/compare_main.py             # experiments/4_main/_tables/ 에 비교표·그래프
```
MPS 기준 약 2시간 35분입니다.

### 재실행 (원인 분석)
```bash
python scripts/replay.py --run 4_main/baseline-attack-fedavg-s0 --from-round 5 --unmask 2         # 악성 2번 조작만 해제
python scripts/replay.py --run 4_main/baseline-attack-fedavg-s0 --from-round 5 --aggregator median  # 방어만 교체
python scripts/replay.py --run 4_main/baseline-attack-fedavg-s0 --from-round 5 --same               # 재현 확인
```
저장된 모델 파일(`experiments/<실행>/models/round_XXX.pt`)이 있어야 합니다. 레포에는 원인 분석에 쓴 출발 모델만 있습니다: `baseline-attack-fedavg-s0~2`, `delay-attack-fedavg-s0~2`의 `round_000.pt`·`round_005.pt`.

다른 실행·라운드의 모델이 필요하면 그 실행을 다시 돌려야 합니다. `run_main.py`는 `summary.json`과 로그로 완료를 판정해 끝난 실행을 건너뛰므로, 실행 폴더를 `experiments/` 밖으로 옮긴 뒤 그 실행만 지정해 돌립니다 (`experiments/` 안에 두면 대시보드 목록에 같은 실행이 두 번 나옵니다).

같은 시드라도 같은 모델이 나오는 것은 기기·torch 버전이 원래 실행과 같을 때입니다. 본 실험은 Apple MPS에서 돌렸고 기기·torch 버전은 실행마다 `env.json`에 적혀 있습니다. 다른 기기(CPU·CUDA)나 버전에서는 값이 조금 달라질 수 있습니다. 원래 체크포인트와 같은 모델이 필요하면 같은 환경에서 돌리세요.
```bash
mv experiments/4_main/baseline-normal-fedavg-s0 ../baseline-normal-fedavg-s0.bak   # 레포 밖으로
python scripts/run_main.py --envs baseline --attacks normal --aggregators fedavg --seeds 0
```

### 모델 역추적(DLG)
```bash
python scripts/run_dlg.py --name my_dlg                              # configs/dlg.yaml (σ 0, 1e-2, 1e-1)
python scripts/run_dlg.py --name it --sigmas 0 0.01 --iterations 50 1000
```
CPU에서 돌고, 이미지 5장 × σ 3개에 몇 분 걸립니다.

## 3. 결과 보는 법
- 실행 1회의 파일 구성과 열 설명: [`scripts/README.md`](../../scripts/README.md)
- 결과 폴더 구성: [`experiments/README.md`](../../experiments/README.md)
- 단계별 구현 내용과 결과: [`docs/stages/`](../stages/README.md)

## 4. 자주 생기는 문제
| 증상 | 해결 |
|---|---|
| `이미 있는 결과 폴더입니다` | 같은 `--name`이 이미 있음. 다른 이름을 쓰거나 기존 폴더를 지웁니다 |
| `서버 설정이 다릅니다` | `serve.py`와 `run_clients.py`에 다른 환경·방어 옵션을 줌. 같은 값을 줍니다 |
| `라운드 K의 저장된 모델이 없습니다` | 재실행할 실행의 `models/`에 그 라운드가 없음. 메시지에 나온 저장된 라운드 중에서 고릅니다 |
| macOS에서 포트 5000 충돌 | AirPlay 수신이 5000을 씀. 기본 포트는 8765이고 `--port`로 바꿀 수 있습니다 |

## 5. 로컬에만 생기는 파일

커밋되지 않는 파일(`.gitignore`)은 ✱로 표시했습니다. 실험을 돌린 PC에만 있습니다.

```
websys-project-fedsec/
├── .venv/                          ✱ 가상환경 (pip install -e ".[test]")
├── data/
│   └── cifar-10-batches-py/        ✱ CIFAR-10 원본 (data/README.md 참고)
├── docs/00_spec/Project 5.pdf      ✱ 과제 명세 원본
├── experiments/
│   ├── 4_main/<실행>/
│   │   ├── config.yaml  env.json  summary.json  rounds.csv  clients.csv  class_acc.csv  client_acc.csv
│   │   ├── server/server_rounds.csv  server/exclusions.csv
│   │   ├── server/server.log  server_status.json   ✱ 서버 원시 로그, 상태 덤프
│   │   └── models/round_XXX.pt     ✱ 라운드별 전역 모델 (재실행의 시작점, 원인 분석 출발 모델 12개만 커밋)
│   ├── 6_replay/<실행>/models/       ✱
│   ├── 3_dlg/<실험>/dlg_images/      ✱ 원본·복원 이미지
│   └── live/                       ✱ 대시보드 실행형으로 돌린 시연 결과
```
