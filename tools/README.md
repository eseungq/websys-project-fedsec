# tools — 학습 설정 탐색 도구

표준 라이브러리 + PyYAML만 씁니다(torch 불필요). 모든 명령은 레포 루트에서 실행합니다.

| 파일 | 역할 |
|---|---|
| `sweep.py` | 실행 목록(YAML)대로 `scripts/run_experiment.py`를 하나씩 순서대로 실행 |
| `summarize.py` | 결과 폴더 여러 개 → 비교표(Markdown, CSV) |
| `plans/` | 탐색 단계별 실행 목록 |
| `tests/` | 도구 테스트 (가짜 CSV·가짜 실행 스크립트 사용) |

## sweep.py
```bash
python tools/sweep.py tools/plans/s1_momentum_lr.yaml --dry-run   # 실행할 명령만 확인
python tools/sweep.py tools/plans/s1_momentum_lr.yaml             # 실행
```
실행 목록 형식:
```yaml
common: {seed: 0, rounds: 50}          # 모든 실행에 공통 (선택)
runs:
  - {name: m0.5_lr0.025, momentum: 0.5, lr: 0.025}
```
- 바꿀 수 있는 값: `seed rounds lr momentum weight_decay local_ep local_bs`. 나머지는 `configs/base.yaml`.
- 결과: `experiments/1_setup/sweep/<name>/`
- 끝난 실행(`summary.json` 있음)은 건너뜀 → 끊겨도 같은 명령으로 이어서 실행. 단, `configs/base.yaml`에 옵션을 덮어쓴 실제 설정이 기존 `config.yaml`과 다르면 멈춤(실행 목록에 없는 값이 `base.yaml`에서 바뀐 경우 포함).
- 끊긴 실행(폴더만 있고 `summary.json` 없음)이 있으면 멈춤 → 그 폴더를 지우거나 옮긴 뒤 다시 실행.
- `configs/base.yaml`은 시작할 때 한 번 읽고 확인합니다(옵티마이저는 `sgd`만). 탐색 도중에는 고치지 마세요.
- 비교하는 것은 `config.yaml`에 기록되는 값뿐입니다. 분할 파일 내용, 데이터, 코드가 바뀐 것은 잡지 못합니다.
- 실행이 실패하면 그 자리에서 멈추고 `[2/6] <이름> 실패 (종료 코드 N)`을 출력.

## summarize.py
```bash
python tools/summarize.py experiments/lr*_r50 experiments/1_setup/sweep/m0.5_*
python tools/summarize.py experiments/1_setup/sweep/[!_]* --epochs 50 --md all.md --csv all.csv   # _tables 폴더는 빼고
```
행은 `last5_val` 높은 순. Markdown은 정확도를 %로, CSV는 원래 값(0\~1)으로 씁니다. 인자에는 실행 결과 폴더만 넘깁니다(`summary.json` 없는 폴더는 오류).

테스트 정확도는 설정 선택에 쓰지 않으므로 표에 넣지 않습니다. (`experiments/1_setup/sweep/_tables/`의 표는 이 변경 전에 만들어 `final_test` 열이 남아 있습니다.)

| 열 | 뜻 |
|---|---|
| `lr momentum weight_decay local_ep local_bs seed rounds` | `config.yaml`의 실제 설정 |
| `last5_val` | 마지막 5라운드 검증 정확도 평균. **선택 기준** |
| `best_val`, `best_round` | 최고 검증 정확도와 그 라운드(같으면 먼저 나온 라운드) |
| `plateau_round` | 검증 정확도가 "마지막 10라운드 평균 − 0.5%p" 이상이 되는 첫 라운드 |
| `min_val_loss_round` | 검증 손실이 가장 낮은 라운드 (과적합 시작 신호) |
| `norm_median`, `norm_p90`, `norm_max` | 전체 라운드 × 클라이언트 업데이트 노름의 중앙값, 90%(선형 보간), 최대 |
| `round_sec`, `total_sec` | 2라운드부터 평균 라운드 시간, 총 시간 (`rounds.csv`의 `total_sec`, 검증 포함) |
| `val_ep<N>` | `epochs_seen = round × local_ep`가 N(`--epochs`, 기본 50)인 라운드의 검증 정확도. 그 라운드가 없으면 `—`. 같은 데이터 통과 횟수 비교이며, local_bs가 다르면 SGD 스텝 수는 다름 |

## 테스트
```bash
.venv/bin/python -m pytest -q tools/tests
```
