# API 설명서

서버는 `127.0.0.1`에서만 받습니다 (기본 포트 8765). 연합학습 API와 대시보드 API가 한 서버에 있습니다.

## 공통 규칙
- JSON API의 응답은 `{"ok": true/false, "data": ..., "error": 문자열 또는 null}` 형식입니다. 대시보드 화면(`GET /`, HTML)과 파일 응답(그래프 SVG, DLG 이미지 PNG)은 이 형식이 아니라 파일 그대로 돌려줍니다.
- Host 헤더가 `127.0.0.1`·`localhost`가 아니면 403입니다.
- POST는 `Content-Type: application/json`만 받습니다. 아니면 415입니다 (본문이 없는 `/aggregate`, `/api/stop`도 `{}`를 보냅니다).

## 1. 연합학습 API (가상 클라이언트 실행 프로그램이 씀)

| API | 역할 |
|---|---|
| `GET /status` | 서버 상태: 라운드, 모델 버전, 대기 중인 제출, 최소 채택 수, 집계 방식, 서버 설정, 라운드·제외 기록 |
| `GET /model` | 이번 라운드의 전역 모델(Base64), 참여자, 가상 마감 시각 |
| `POST /update` | 업데이트(로컬 가중치 − 시작 전역 가중치) 제출. 라운드·모델 버전·클라이언트 ID·형식·NaN·중복·마감을 검사해 채택하거나 사유와 함께 제외 |
| `POST /aggregate` | 라운드를 닫음. 채택이 3개 이상이면 집계해 전역 모델을 바꾸고, 아니면 갱신하지 않고 사유를 기록 |

요청·응답 예시와 제외 사유 14가지는 [`docs/stages/2_서버-API.md`](../stages/2_서버-API.md#api)에 있습니다.

간단한 확인:
```bash
python scripts/serve.py --port 8765 &
curl -s http://127.0.0.1:8765/status
```

## 2. 대시보드 API (브라우저가 씀)

| API | 요청 | 응답 `data` |
|---|---|---|
| `GET /` | | 대시보드 화면 |
| `GET /api/runs` | | 결과 폴더 목록: `id, group, env, attack, aggregator, seed, rounds_completed, final_val_acc, final_test_acc, diverged_round, replay` |
| `GET /api/runs/<실행>` | `<실행>`은 `experiments/` 기준 경로 (예: `4_main/baseline-attack-fedavg-s0`) | `summary, config, plan, replay, rounds[], clients[], class_acc[], client_acc[], exclusions[]` |
| `GET /api/comparison` | | 본 실험 비교표: 공격 피해·방어 개선·정상 손실(시드별·평균, %p), 조건별 정확도 |
| `GET /api/comparison/accuracy.svg` | | 정확도 막대그래프 |
| `GET /api/dlg` | | DLG 실험 목록과 결과 (σ, 반복, MSE, 레이블 복원, 이미지 파일 이름) |
| `GET /api/dlg/<실험>/images/<파일>` | | 원본·복원 이미지 PNG |
| `POST /api/start` | `{"env", "attack", "aggregator", "seed", "rounds", "name"}` (모두 선택) | `{"run_id": "live/<시각>-<이름>", "params"}` |
| `GET /api/status` | | 실행형 진행 상황: `state`(idle, running, completed, stopped, error), 끝난 라운드, 라운드 기록, 모든 라운드의 클라이언트 기록, 서버 라운드 |
| `POST /api/stop` | `{}` | `{"stopped": true, "run_id"}` 또는 `{"stopped": false}` |

`/api/start` 입력:
| 항목 | 값 | 기본 |
|---|---|---|
| `env` | `baseline`, `delay`, `partial`, `label_skew` | `baseline` |
| `attack` | `normal`, `attack` | `normal` |
| `aggregator` | `fedavg`, `clip_fedavg`, `median` | `fedavg` |
| `seed` | 0\~9999 정수 | 0 |
| `rounds` | 1\~10 정수 | 3 |
| `name` | 영문·숫자·`_`·`-` 1\~40자 | `<env>-<attack>-<aggregator>-s<seed>` |

- 모르는 항목, 범위 밖 값, 정수 자리의 문자열·실수·참거짓은 400입니다.
- 실행 중에 다시 시작하면 409입니다. 한 번에 하나만 돕니다.
- 경로가 `experiments/` 밖을 가리키면 400, 없는 실행은 404입니다.
