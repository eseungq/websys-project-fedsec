# experiments — 실험 결과

실험 1회마다 폴더 하나를 만듭니다. 명령과 파일 설명은 [scripts/README.md](../scripts/README.md)에 있습니다.

| 폴더 | 내용 |
|---|---|
| `matrix/` | 본 실험 72회 (환경 4 × 정상·공격 × 집계 3 × 시드 3, 40라운드), `runs.csv`, 비교표·그래프 `_tables/` |
| `replay/` | 재실행 결과 (`scripts/replay.py`, `replay.json`에 원래 실행·출발 라운드·바꾼 조건) |
| `dlg/` | 모델 역추적(DLG) 실험 |
| `sweep/`, `lr*_r50/`, `baseline_mps/` | 학습 설정 탐색 |
| `norm_check/` | 클리핑 상한을 정하려고 잰 정상 업데이트 노름 |
| `matrix_repeat/` | 같은 조합을 다시 돌린 재현성 확인 |
| `live/` | 대시보드 실행형 시연 결과 (커밋 안 함) |

```
experiments/<묶음>/<실행>/
├── config.yaml  env.json  summary.json          # 실제로 쓴 설정, 기기·버전·커밋, 요약 (커밋)
├── rounds.csv  clients.csv  class_acc.csv  client_acc.csv   # 라운드·클라이언트·클래스 로그 (커밋)
├── server/server_rounds.csv  server/exclusions.csv          # 서버 집계·제외 기록 (커밋)
├── server/server.log  server_status.json        # 서버 원시 로그, 상태 덤프 (커밋 안 함)
└── models/round_XXX.pt                           # 라운드별 전역 모델, 재실행의 시작점 (원인 분석 출발 모델 12개만 커밋)
```
