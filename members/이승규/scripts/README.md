# scripts — 실행 스크립트

모든 명령은 `members/이승규/`에서 실행합니다(`cd members/이승규`). 결과는 `members/이승규/experiments/`에, 공용 분할 파일은 레포 루트 `configs/splits/`에 있습니다.

| 파일 | 상태 | 역할 |
|---|---|---|
| `make_split.py` | 있음 | 분할 파일 생성 (레포 루트 `configs/splits/iid_seed0.json`, 이미 커밋됨 → 다시 만들 필요 없음) |
| `run_experiment.py` | 있음 | 서버 없이 정상 FedAvg 1회 실행 (학습 설정 탐색용) |
| `serve.py`, `run_clients.py` | 있음 | Flask 서버, 가상 클라이언트 실행 프로그램을 따로 실행 |
| `run_server_experiment.py` | 있음 | 서버를 띄우고 클라이언트를 실행한 뒤 서버 종료 (환경·공격·방어 선택) |
| `run_matrix.py` | 있음 (본 실험 11/20) | 환경 4 × (정상, 공격) × 집계 3 × 시드 일괄 실행, `--dry-run` |
| `replay_round.py` | 예정 | 저장된 상태에서 특정 라운드부터 조건 하나를 바꿔 재실행 |
| `run_dlg.py` | 있음 | DLG 복원 실험 (CPU) |
| `compare_matrix.py` | 있음 | `runs.csv` → 공격 피해·방어 개선·정상 손실 표, 정확도 막대그래프 (`experiments/<prefix>/_tables/`) |

## 하이퍼파라미터 비교 실행
```bash
cd members/이승규
python scripts/run_experiment.py --name baseline                 # configs/base.yaml 그대로
python scripts/run_experiment.py --name lr0.05 --lr 0.05         # 학습률만 변경
python scripts/run_experiment.py --name ep2 --local-ep 2 --rounds 10
python scripts/run_experiment.py --name final --evaluate-test  # 설정 확정 후에만 테스트셋 평가
```
바꿀 수 있는 값: `--seed --rounds --lr --momentum --weight-decay --local-ep --local-bs`

결과 `members/이승규/experiments/<name>/`
- `config.yaml` 실제로 쓴 전체 설정, `env.json` 기기·버전·**커밋 ID**
- `rounds.csv` 라운드별 학습 손실, 검증 손실·정확도, 업데이트 노름(최소·평균·최대), 학습 시간(`train_sec`), 검증 포함 시간(`total_sec`)
- `clients.csv` 라운드 × 클라이언트별 표본 수, 학습 손실, 업데이트 노름 (클리핑 상한 정할 때 사용)
- `summary.json` 최종 검증 정확도, 첫 라운드 시간, 이후 평균 시간. `--evaluate-test`를 준 실행에만 테스트 정확도·손실(`final_test_acc`, `final_test_loss`)
- 서버를 거친 실행(`run_server_experiment.py`, `run_clients.py`, `run_matrix.py`)만: `class_acc.csv` 라운드 × 클래스 검증 정확도, `client_acc.csv` 라운드 × 클라이언트의 분포 가중 정확도(Σ 클래스 비율 × 클래스 정확도), `models/round_XXX.pt` 라운드 0·5의 배수·마지막 라운드의 전역 모델(간격 `--save-every`). `--evaluate-test`면 두 CSV에 테스트셋 줄(`split=test`) 추가
- 발산: 로컬 학습 손실·업데이트, 제출할 업데이트의 노름(공격 벡터 포함, float32 범위 초과), 집계 뒤 전역 모델·검증 손실에 NaN·inf가 생기면 그 라운드는 기록하지 않고 끝낸다. 공격 실험이면 발산 전 전역 모델로 최종 정확도를 계산해 `models/round_XXX.pt`로도 저장하고, `summary.json`에 `diverged_round`·`rounds_completed`를 남긴다. 1라운드 집계 직후 전역 모델이 깨지면 정확도 없이 `diverged_round` 1, `rounds_completed` 0으로 남긴다(비교표에 넣지 않음). 정상 실행의 발산과 1라운드 로컬 학습의 발산(초기 모델에서 생기므로 공격과 무관)은 설정 문제로 보고 멈춘다
- `summary.json`은 모델 저장 뒤 **마지막에** 쓴다. 이 파일이 있으면 실행이 끝까지 간 것
- DLG(`run_dlg.py`)는 `<name>.partial/`에 쓰고 끝나면 `<name>/`으로 이름을 바꾼다. 실패해서 `.partial`이 남으면 같은 `--name`으로 다시 실행되지 않으니 지우고 다시 실행
- 결과 폴더 이름(`--name`, `--prefix`, 서버 `--log-dir`)은 `experiments/` 아래만 된다(`../`, 절대 경로, `.partial`로 끝나는 이름 거부). 서버 기록 폴더에 이전 기록이 있으면 섞이지 않게 멈춘다

**`run_matrix.py`의 완료 판정과 `runs.csv`** (`experiments/<prefix>/runs.csv`)
- 끝난 실행: `summary.json`의 검증·테스트 정확도가 0\~1, `rounds_completed`가 설정 라운드 수(발산했으면 발산 라운드 − 1), 저장된 설정이 지금 설정과 같음, 완료한 라운드는 설정 라운드 수(발산했으면 발산 전까지), 로그마다 키 조합이 정확히 한 줄씩: `rounds.csv`는 라운드, `clients.csv`는 라운드 × 클라이언트, `class_acc.csv`·`client_acc.csv`는 (라운드, 분할, 클래스·클라이언트 ID)로 완료한 라운드마다 검증 행과 마지막 테스트 행. 빠지거나 중복되거나 예상 밖인 행, 읽을 수 없는 파일이 있으면 미완료, `env.json`이 읽힘 발산했으면 공격 실험이어야 한다. `models/`는 커밋하지 않으므로 보지 않는다
- 1라운드에 발산한 공격 실험은 `diverged`(정확도 없음)로 표시하고 끝난 실행으로 친다
- 이어서 실행할 때 끝난 실행의 기기·GPU 모델명·torch 버전이 지금과 다르면 `stale`로 멈춘다(커밋만 다르면 경고). MPS는 GPU 모델명이 비어 있어 같은 Mac에서 재개하는 것이 전제
- 설정이 다르면 `stale`, 산출물이 덜 쓰였거나 정상 실행이 발산했으면 `incomplete`로 표시하고 시작 전에 멈춘다(사유 출력). 실행 하나가 끝날 때마다도 같은 판정을 해서 `done`이 아니면 멈춘다
- 열: 실행 이름·환경·공격·집계·시드·상태, `final_val_acc`, `final_test_acc`, `diverged_round`, `rounds_completed`, `rounds_not_updated`, `total_virtual_time`, `total_sec`, 실행 환경 `device`·`gpu`(GPU 모델명)·`torch`·`commit`·`dirty`(추적 중인 파일이 수정됐는지)

> 공식 테스트셋은 기본으로 평가하지 않습니다. 설정 비교는 **검증 정확도**로 하고, 설정을 확정한 뒤에만 `--evaluate-test`로 한 번 평가하세요.

## Kaggle에서 실행
노트북 설정: Accelerator **GPU T4 x2**(또는 P100), Internet **On**(휴대폰 인증 필요).
```python
!git clone https://github.com/eseungq/websys-project-fedsec.git
%cd websys-project-fedsec
!git checkout <커밋 ID>
%cd members/이승규
!pip install -q -e .
!python scripts/run_experiment.py --name baseline_kaggle
```
끝나면 `members/이승규/experiments/baseline_kaggle/`의 `summary.json`, `rounds.csv`를 내려받아 공유합니다.
(Kaggle은 세션이 끝나면 파일이 사라집니다. 오른쪽 Output 패널에서 내려받거나 노트북을 Save Version 하세요.)
