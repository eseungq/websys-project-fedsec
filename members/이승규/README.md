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
- 40라운드 기준 검증 정확도(36\~40라운드 평균, 시드 0\~2 평균) 75.92% → **76.38%** (+0.47%p)
- 50라운드로 돌린 비교 실험의 마지막 5라운드(46\~50) 평균은 75.44% → 76.40% (+0.96%p)
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
