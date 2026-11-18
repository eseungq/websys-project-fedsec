# scripts — 실행 스크립트

| 파일 | 상태 | 역할 |
|---|---|---|
| `make_split.py` | 있음 | 분할 파일 생성 (`configs/splits/iid_seed0.json`, 이미 커밋됨 → 다시 만들 필요 없음) |
| `run_experiment.py` | 있음 (정상 FedAvg만) | 설정 하나로 실험 1회 실행 |
| `run_matrix.py` | 예정 | 환경 4 × (정상, 공격) × 집계 3 × 시드 일괄 실행 |
| `replay_round.py` | 예정 | 저장된 상태에서 특정 라운드부터 조건 하나를 바꿔 재실행 |
| `run_dlg.py` | 예정 | DLG 복원 실험 |
| `make_figures.py` | 예정 | 평가표·비교 그래프 생성 |

## 하이퍼파라미터 비교 실행
```bash
python scripts/run_experiment.py --name baseline                 # configs/base.yaml 그대로
python scripts/run_experiment.py --name lr0.05 --lr 0.05         # 학습률만 변경
python scripts/run_experiment.py --name ep2 --local-ep 2 --rounds 10
```
바꿀 수 있는 값: `--seed --rounds --lr --momentum --weight-decay --local-ep --local-bs`

결과 `experiments/<name>/`
- `config.yaml` 실제로 쓴 전체 설정, `env.json` 기기·버전·**커밋 ID**
- `rounds.csv` 라운드별 학습 손실, 검증 손실·정확도, 업데이트 노름(최소·평균·최대), 학습 시간(`train_sec`), 검증 포함 시간(`total_sec`)
- `clients.csv` 라운드 × 클라이언트별 표본 수, 학습 손실, 업데이트 노름 (클리핑 상한 정할 때 사용)
- `summary.json` 최종 검증·테스트 정확도, 첫 라운드 시간, 이후 평균 시간

> 공식 테스트셋은 마지막에 한 번만 평가합니다. 설정 비교는 **검증 정확도**로 하세요.

## Kaggle에서 실행
노트북 설정: Accelerator **GPU T4 x2**(또는 P100), Internet **On**(휴대폰 인증 필요).
```python
!git clone https://github.com/eseungq/websys-project-fedsec.git
%cd websys-project-fedsec
!git checkout <공유받은 커밋 ID>
!pip install -q -e .
!python scripts/run_experiment.py --name baseline_kaggle
```
끝나면 `experiments/baseline_kaggle/`의 `summary.json`, `rounds.csv`를 내려받아 공유합니다.
(Kaggle은 세션이 끝나면 파일이 사라집니다. 오른쪽 Output 패널에서 내려받거나 노트북을 Save Version 하세요.)
