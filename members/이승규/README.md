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
```
- 바꿀 수 있는 값: `--seed --rounds --lr --momentum --weight-decay --local-ep --local-bs`
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
[`docs/001-baseline-setup.md`](docs/001-baseline-setup.md)
- 학습률 비교(0.005 / 0.01 / 0.02, 50라운드) → lr 0.01, 40라운드 제안
- 시드 반복(시드 0~2, Kaggle T4 포함) → 40라운드 검증 75.8~76.2%
- 업데이트 노름 분포 (클리핑 상한 참고)

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `src/fedsec/` | 실험 엔진 패키지 (데이터, 모델, 로컬 학습, 집계, 로그, 실행) |
| `scripts/` | 실행 스크립트 (`run_experiment.py`, `make_split.py`) |
| `tests/` | 집계·분할 수치 예제 테스트 |
| `configs/` | 기준 설정 `base.yaml` |
| `experiments/` | 실험 결과 (`models/`는 커밋 안 함) |
| `docs/` | 설정 결정 기록 |
