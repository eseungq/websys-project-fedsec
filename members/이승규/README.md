# 이승규 — 후보 코드

11/24 대표 코드 선정 전까지 쓰는 개인 폴더입니다. 선정되면 레포 루트 `src/fedsec/` 등으로 머지됩니다.

## 구현 범위
- 명세 1단계 일부: 정상 FedAvg, IID 분할, 클라이언트 전원 참여
- 라운드 × 클라이언트별 업데이트 노름 기록 (클리핑 상한 정할 때 사용)
- 없음: 서버(웹), 공격, 방어, DLG

## 실행 방법
명령은 모두 이 폴더(`members/이승규/`)에서 실행합니다. 가상환경은 레포 루트의 `.venv/`를 씁니다.

```bash
cd members/이승규
pip install -e ".[test]"                              # 설치 (torch, torchvision, PyYAML + pytest)
python -m pytest -q                                   # 테스트 (13개)
python scripts/run_experiment.py --name baseline      # 실험 1회 (configs/base.yaml 그대로)
python scripts/run_experiment.py --name lr0.02 --lr 0.02 --rounds 10   # 옵션으로 값 바꾸기
python scripts/run_experiment.py --name final --evaluate-test         # 설정 확정 후에만 테스트셋 평가
```
- 바꿀 수 있는 값: `--seed --rounds --lr --momentum --weight-decay --local-ep --local-bs`
- 공식 테스트셋은 `--evaluate-test`를 줄 때만 평가합니다. 설정 비교는 검증 정확도로 합니다.
- 결과: `experiments/<name>/` (`--name sweep/abc`처럼 하위 폴더도 가능)
- 데이터(`data/`)와 공용 분할 파일(`configs/splits/iid_seed0.json`)은 레포 루트 것을 씁니다.
- Kaggle 실행, 결과 파일 설명: [`scripts/README.md`](scripts/README.md)

### Flask 서버를 거쳐 실행 (명세 2단계)
중앙 서버(`127.0.0.1`)와 가상 클라이언트 10개를 순서대로 돌리는 실행 프로그램이 별도 프로세스로 REST API(`GET /status`, `GET /model`, `POST /update`, `POST /aggregate`)를 주고받습니다. 학습 옵션은 `run_experiment.py`와 같습니다.

```bash
python scripts/run_server_experiment.py --name server_demo --rounds 3   # 서버를 띄우고 → 클라이언트 실행 → 서버 종료

# 서버와 클라이언트를 터미널 두 개로 따로 띄울 때
python scripts/serve.py --port 8765 --log-dir experiments/demo_server  # 터미널 1 (서버 기록 폴더는 결과 폴더와 다른 이름)
python scripts/run_clients.py --name demo --server-url http://127.0.0.1:8765   # 터미널 2
```
- 결과: `run_experiment.py`와 같은 파일 + `server_status.json`(라운드·제외 기록), `server/`(`server_rounds.csv`, `exclusions.csv`, `server.log`)
- 서버를 거친 결과는 서버 없는 실행기와 같습니다 (3라운드 최종 모델이 비트 단위로 같음, 라운드당 약 0.5초 더 걸림)

### 운영 환경·공격·방어 (명세 3·4단계)
환경은 `configs/env/`(기준·지연·부분 참여·레이블 편향), 공격·방어 기본값은 `configs/attack_defense.yaml`에 있습니다. 위 세 스크립트에 아래 옵션을 함께 줍니다 (`serve.py`와 `run_clients.py`를 따로 띄울 때는 두 쪽에 같은 값).

```bash
python scripts/run_server_experiment.py --name d_atk_med --env configs/env/delay.yaml --attack --aggregator median
python scripts/run_server_experiment.py --name p_atk_clip --env configs/env/partial.yaml --attack --aggregator clip_fedavg --clip-norm 3.5

python scripts/run_matrix.py --dry-run                 # 본 실험 목록: 환경 4 × (정상, 공격) × (FedAvg, 클리핑, 중앙값) × 시드 0 1 2 = 72회
python scripts/run_matrix.py --seeds 0 1 2             # 본 실험 (끊기면 같은 명령으로 이어서, 테스트셋 평가 포함)
python scripts/run_matrix.py --prefix matrix_check --rounds 2 --seeds 0 --attacks attack   # 동작 확인 12회

python scripts/make_split.py --show-dist configs/splits/iid_seed0.json   # 분할의 클라이언트별 클래스 분포 (레포 루트 기준 경로)
```
- 옵션: `--env`, `--attack / --no-attack`, `--aggregator {fedavg,clip_fedavg,median}`, `--clip-norm`, `--clip-before-median`, `--attack-defense`(설정 파일)
- 같은 환경·시드면 공격·집계 방식이 달라도 악성 ID·지연 ID·라운드별 참여자가 같습니다. 서버는 참여자·가상 마감·집계 방식만 알고 악성 ID는 모릅니다.
- 로그: `rounds.csv`(선택·채택·지연 제외 수, 전체·실제 채택 악성 비율(채택 0개면 `N/A`), 모델 갱신 여부와 사유, 가상 경과 시간, 실제 시간), `clients.csv`(라운드 × 클라이언트: 악성·선택·공격 여부, 조작 전·후·방어 후 노름, 채택·제외 사유, 가상 완료 시각, 실제 시간)
- `run_matrix.py`의 결과 요약: `experiments/<prefix>/runs.csv`

## 현재 설정 (`configs/base.yaml`)
| 항목 | 값 |
|---|---|
| 모델 | `simple_cnn` (BatchNorm 없음) |
| 라운드 | 40 |
| 옵티마이저 | SGD, lr 0.01, momentum 0.9, weight_decay 0 |
| local_ep / local_bs | 1 / 32 |
| 시드 | 0 |
| 분할 | `configs/splits/iid_seed0.json` (레포 루트) |
| eval_bs | 1000 |

## 결과 요약
**[학습 설정 결과](docs/학습설정-결과.md)** (11/18 \~ 11/19, 실험 7개)

- 추천: momentum 0.9, lr 0.01, local_ep 1, local_bs 32, 40라운드 + **weight_decay 5e-4** (제안 단계, `configs/base.yaml`에는 아직 반영하지 않음)
- 시드 0\~2 평균 검증 정확도(마지막 5라운드) 75.44% → **76.40%** (+0.96%p)
- 실험 1회(40라운드) 로컬 MPS 약 2.5분, Kaggle T4 약 2.2분
- 결과 폴더 [`experiments/`](experiments/), 비교표 [`experiments/sweep/_tables/`](experiments/sweep/_tables/)

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `src/fedsec/` | 실험 엔진 패키지 (데이터, 모델, 로컬 학습, 집계, 로그, 실행) |
| `scripts/` | 실행 스크립트 (`run_experiment.py`, `make_split.py`) |
| `tests/` | 집계·분할 수치 예제 테스트 |
| `configs/` | 기준 설정 `base.yaml` |
| `experiments/` | 실험 결과 (`models/`는 커밋 안 함) |
| `docs/` | 학습 설정 결과 문서 (`학습설정-결과.md`) |
