# 연합학습 환경: Flask 서버와 API (명세 2단계)

- 작성: 2026-11-19, 이승규
- 상태: 구현·검증 끝 (테스트, 서버 없는 실행기와 결과 비교)
- 명세 요구 산출물 "API·입력 파일 설명서"를 겸한다

---

## 한눈에 보기

| 명세 항목 | 구현 내용 | 결과 |
|---|---|---|
| 중앙 Flask 서버 1개와 하나의 실행 프로그램에서 순차 학습하는 가상 클라이언트 10개 구성 | 서버 `scripts/serve.py`, 클라이언트 실행 프로그램 `scripts/run_clients.py`(10명을 ID 순으로 학습). 둘을 한 번에 띄우는 `run_server_experiment.py` | 40라운드 1회 약 2.8분 (MPS) |
| REST API로 모델 다운로드·업데이트 제출 구현. 같은 라운드의 모든 클라이언트는 같은 전역 모델에서 학습 | `GET /model`로 받은 전역 모델에서 참여자 전원이 학습을 끝낸 뒤 `POST /update`로 제출 | 같은 라운드 안에서는 모델 버전이 바뀌지 않음 (테스트) |
| 업데이트는 로컬 학습 후 가중치와 시작 전역 가중치의 차이로 정의. 채택 업데이트를 표본 수에 비례해 평균하는 FedAvg 연결 | `delta = 로컬 학습 후 가중치 − 시작 전역 가중치`. `POST /aggregate`에서 채택된 업데이트만 표본 수 가중 평균 | 서버를 거친 3라운드 최종 모델이 서버 없는 실행기와 **비트 단위로 같음**. 라운드당 약 0.5초 더 걸림 |
| 라운드·모델 버전·클라이언트 ID를 확인하고, 중복 제출이나 잘못된 업데이트는 제외 사유와 함께 기록 | 제출마다 아래 "제외 사유" 14가지를 검사. 제외되면 사유 코드와 HTTP 상태로 응답하고 서버 기록에 남김 | 사유별로 테스트 |

---

## 구조
```
클라이언트 실행 프로그램 (한 프로세스)                 Flask 서버 (127.0.0.1:8765)
  라운드마다                                           ServerState (라운드, 모델 버전, 전역 가중치, 대기 업데이트)
  1. GET /model ───────────────────────────────────▶  전역 모델 + 참여자 + 가상 마감
  2. 참여자를 ID 순으로 로컬 학습 (같은 전역 모델에서)
  3. (공격을 켰으면 악성 업데이트를 조작)
  4. POST /update × 참여자 수 ─────────────────────▶  검증 → 채택 또는 제외(사유 기록)
  5. POST /aggregate ──────────────────────────────▶  채택 3개 이상이면 집계·갱신, 라운드 +1
  6. GET /model → 검증 정확도 평가, 로그 기록
```
- 초기 모델은 서버가 시드로 만든다. 클라이언트는 받기만 한다.
- 악성·지연 ID와 공격 여부는 클라이언트 프로그램만 안다. 서버는 참여자, 가상 마감, 집계 방식만 안다.
- 클라이언트는 시작할 때 서버 `/status`의 시드·모델·환경·참여자 수·최소 채택 수·가상 마감·방어 설정을 자기 설정과 대조하고, 다르면 멈춘다.
- 설정 파일은 정해진 항목만 받는다. 빠진 항목, 모르는 항목(오타), NaN·inf, 불가능한 조합(악성 0명인 공격 실험, 악성과 겹치는 지연 ID, 참여자 수보다 큰 최소 채택 수)은 실행 전에 거부한다.
- 클라이언트는 제외 사유가 `late`(설계상 정상)가 아니거나 서버 오류이면 멈춘다. 학습 손실이 NaN·inf가 되어도 멈춘다. 결과가 "완료"로 남아 비교표에 섞이지 않게 하려는 것이다.

### 실행
```bash
# 한 번에 (서버를 띄우고 → 클라이언트 실행 → 서버 종료)
python scripts/run_server_experiment.py --name server_demo --rounds 3

# 터미널 두 개로 따로 (환경·방어 옵션은 두 쪽에 같은 값을 준다)
python scripts/serve.py --port 8765 --log-dir experiments/demo_server          # 터미널 1 (서버 기록 폴더는 결과 폴더와 다른 이름)
python scripts/run_clients.py --name demo --server-url http://127.0.0.1:8765   # 터미널 2
```
- 결과 폴더에는 클라이언트 쪽 로그와 `server_status.json`(라운드·제외 기록 전체, 커밋 안 함), 서버 쪽 `server/`(`server_rounds.csv`, `exclusions.csv`, `server.log`)가 남는다

| 코드 | 역할 |
|---|---|
| `src/fedsec/server/app.py` | Flask 앱과 API 4개 |
| `src/fedsec/server/state.py` | 서버 상태, 제출 받기, 집계, 라운드 기록 |
| `src/fedsec/server/validation.py` | 제출 검증과 제외 사유 |
| `src/fedsec/server/serialization.py` | 가중치 ↔ 문자열 (`torch.save` + Base64, 받을 때 `weights_only=True`) |
| `src/fedsec/server/scenario.py` | 환경·방어 설정 → 서버 정책 (참여자, 마감, 집계 방식) |
| `src/fedsec/client/program.py` | 클라이언트 실행 프로그램 |

---

## API
모든 응답은 `{"ok": true/false, "data": ..., "error": 문자열 또는 null}` 형식이다.
- Host 헤더가 `127.0.0.1`·`localhost`가 아니면 403. POST는 `Content-Type: application/json`만 받는다(`/aggregate`도 빈 객체 `{}`를 보냄). 같은 PC의 브라우저를 통한 요청 위조를 막기 위해서다
- 제외 사유 문장(`error`)은 200자에서 자른다

### `GET /status`
서버 상태. 클라이언트가 시작할 때 설정을 대조하는 데 쓴다.
```json
{"round": 1, "model_version": 0, "pending": 0, "pending_clients": [], "num_clients": 10, "min_accepted": 3,
 "aggregator": "fedavg",
 "config": {"seed": 0, "model": "simple_cnn", "env": "baseline", "per_round": null, "aggregator": "fedavg", "clip_norm": 3.5, "...": "..."},
 "history": ["라운드 기록..."], "exclusions": ["제외 기록..."]}
```

### `GET /model`
지금 라운드의 전역 모델.
```json
{"round": 3, "model_version": 2, "weights": "<Base64>", "participants": [0, 1, 2], "deadline": 1.5}
```
- `weights`: `torch.save`한 state dict를 Base64로 바꾼 문자열
- `participants`: 이번 라운드 참여자. `deadline`: 가상 마감 시각 (없으면 null)

### `POST /update`
업데이트 제출.
```json
{"client_id": 4, "round": 3, "model_version": 2, "num_samples": 4500,
 "delta_weights": "<Base64>", "virtual_finish_time": 1.0}
```
- 채택: `200`, `{"ok": true, "data": {"client_id": 4, "round": 3, "accepted": true}}`
- 제외: 4xx, `{"ok": false, "data": {"reason": "late"}, "error": "가상 완료 시각 3.0이 마감 1.5를 넘었습니다."}`

### `POST /aggregate`
이번 라운드를 닫는다. 채택이 3개 이상이면 집계해 전역 모델을 바꾸고 모델 버전을 1 올린다. 라운드는 항상 1 오른다.
```json
{"round": 3, "model_version": 3, "updated": true, "skip_reason": null, "participants": [0, 1, 2],
 "accepted_clients": [0, 1, 2], "num_accepted": 7, "num_excluded": 3, "exclusion_reasons": {"late": 3},
 "total_samples": 31500, "update_norms": [1.8], "defended_norms": [1.8], "aggregator": "fedavg", "aggregate_sec": 0.01}
```
- `skip_reason`: 갱신하지 않은 이유. `no_accepted`(채택 0개), `too_few_accepted`(3개 미만)
- `defended_norms`: 방어(클리핑) 뒤 노름. FedAvg·중앙값은 방어 전과 같음

### 제외 사유
| 사유 | HTTP | 뜻 |
|---|---|---|
| `bad_json` | 400 | 본문이 JSON 객체가 아님 |
| `missing_field` | 400 | 필수 항목 없음 (마감이 있는 라운드의 `virtual_finish_time` 포함) |
| `bad_type` | 400 | 항목 자료형이 틀림 (`virtual_finish_time`은 0 이상의 유한한 숫자) |
| `bad_num_samples` | 400 | 표본 수가 1\~50,000 사이 정수가 아님 (상한이 없으면 한 제출이 FedAvg 가중치를 독점할 수 있음) |
| `unknown_client` | 403 | 등록되지 않은 클라이언트 ID |
| `not_selected` | 403 | 이번 라운드 참여자가 아님 |
| `round_mismatch` | 409 | 서버의 현재 라운드와 다름 |
| `version_mismatch` | 409 | 서버의 현재 모델 버전과 다름 |
| `duplicate` | 409 | 이번 라운드에 이미 채택된 제출이 있음 |
| `late` | 409 | 가상 완료 시각이 마감을 넘음 |
| `too_large` | 413 | 요청·가중치가 크기 상한(16MB)을 넘음 |
| `bad_payload` | 400 | Base64·torch 파일 형식이 아님 |
| `bad_shape` | 400 | 파라미터 이름·모양·자료형·저장 형식(sparse 등)이 전역 모델과 다름 |
| `non_finite` | 400 | NaN 또는 Inf 포함, 또는 업데이트 노름이 float32 범위를 넘음 |

---

## 입력 파일
경로는 `members/이승규/` 기준이다. 분할 파일과 데이터만 레포 루트 기준이다.

| 파일 | 내용 | 바꾸는 방법 |
|---|---|---|
| `configs/base.yaml` | 학습 설정: 시드, 라운드, 모델, SGD 값, local_ep·bs, eval_bs, 분할 파일 | `--config` 또는 `--seed --rounds --lr --momentum --weight-decay --local-ep --local-bs` |
| `configs/env/*.yaml` | 운영 환경: 분할 파일, 라운드 참여자 수, 악성·지연 수(또는 ID), 가상 시간·마감, 최소 채택 수 | `--env` |
| `configs/attack_defense.yaml` | 공격 켬/끔, 배수(−5), 집계 방식, 클리핑 상한, 중앙값 전 클리핑 | `--attack-defense` 또는 `--attack/--no-attack --aggregator --clip-norm --clip-before-median` |
| `configs/splits/label_skew_a0.5_seed0.json` | 레이블 편향 분할 (IID 분할은 레포 루트 `configs/splits/iid_seed0.json`) | `scripts/make_split.py`로 만듦 |
| `configs/dlg.yaml` | DLG 실험: 이미지, σ, 반복 횟수, 시드 | `run_dlg.py --sigmas --iterations --images --dummy-seed` |

결과 파일 설명은 [`scripts/README.md`](../scripts/README.md), 운영 환경 설정 값은 [4_운영환경.md](4_운영환경.md)에 있다.

---

## 검증
- **서버 없는 실행기와 비교:** 같은 설정·시드로 3라운드를 돌려 최종 모델 가중치가 비트 단위로 같았다. 서버를 거치는 비용은 라운드당 약 0.5초(직렬화 + HTTP)
- **테스트:** `tests/test_server.py`(API·제외 사유·집계), `tests/test_serialization.py`(가중치 변환·크기 상한·악성 파일 거부), `tests/test_client_round.py`(서버와 클라이언트를 Flask 테스트 클라이언트로 끝까지 돌림)
- **안전:** 서버는 `127.0.0.1`에서만 받고, `/aggregate`가 실패하면 서버 상태(라운드·모델·대기 업데이트)를 바꾸지 않는다. 가중치는 `torch.load(weights_only=True)`로만 풀어 텐서 외 객체를 거부한다
