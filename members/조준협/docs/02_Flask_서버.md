# 02. Flask 서버 (2단계)

원문: 조준협 작업 기록(Google Docs)의 "2단계" 부분. 내용은 바꾸지 않고 형식만 정리했습니다. 원문의 실행 화면은 "원문 문서의 실행 화면 참고"로, 깨진 수식은 "(원문 수식 생략)"으로 표시했습니다.

---

## 구현 방법
1. **1단계**: PyTorch 모델 가중치를 문자열(Base64)로 압축하고 다시 복원하는 직렬화 유틸리티(`serialization.py`) 작성 및 테스트
2. **2단계**: 모델 배포(`/model`), 업데이트 검증 및 로그 기록(`/update`), FedAvg 집계(`/aggregate`) 기능을 갖춘 Flask 중앙 서버(`server.py`) 작성
3. **3단계**: 10명의 가상 클라이언트가 순차적으로 서버와 통신하며 학습을 수행하는 클라이언트 실행기(`client_runner.py`) 작성 및 연동 테스트

---

## 1. 분산 연합학습(FedAvg) 시스템 아키텍처 개요
기존 단일 프로세스 기반 시뮬레이션 코드를 확장하여, 중앙 파라미터 서버(Flask REST API Server)와 독립된 다중 클라이언트 실행기(Virtual Client Runner)가 HTTP 통신을 통해 모델 가중치와 업데이트(원문 수식 생략)를 주고받는 실제 분산 환경 구조를 구현하였다.

- **통신 프로토콜 및 데이터 포맷**: HTTP/1.1 REST API (JSON 페이로드)
- **가중치 전송 방식**: PyTorch `state_dict` 텐서를 바이너리 직렬화(`torch.save`) 후 Base64 문자열로 인코딩하여 송수신하며, 클라이언트는 전체 가중치가 아닌 학습 전후 변화량(원문 수식 생략)만을 전송하여 통신 무결성 검증을 용이하게 설계함
- **실시간 관제 기능**: 중앙 서버 내장 웹 대시보드(`http://127.0.0.1:5000`) 및 비동기 JSON 폴링(`/status`, 1.5초 주기)을 통해 라운드별 정확도 추이와 비정상 업데이트 차단 로그를 실시간으로 모니터링

---

## 2. 핵심 구현 모듈(.py)별 역할 및 내부 구조

### ① 가중치 직렬화 및 무결성 검증 모듈 (`src/serialization.py`)
네트워크를 통해 PyTorch 모델의 다차원 텐서 딕셔너리(`state_dict`)를 손실 없이 JSON으로 변환하고, 비정상 데이터의 유입을 사전 차단하는 유틸리티 모듈이다.

| 주요 함수명 | 세부 역할 및 동작 원리 |
|---|---|
| `serialize_weights(state_dict)` | PyTorch 텐서를 CPU로 이동시킨 뒤 메모리 버퍼(`io.BytesIO`)에 `torch.save`로 직렬화하고, HTTP JSON 전송이 가능하도록 Base64 ASCII 문자열로 인코딩한다. |
| `deserialize_weights(encoded_str)` | 수신된 Base64 문자열을 디코딩한 뒤 `torch.load`로 역직렬화하여 원본과 동일한 형상(Shape) 및 자료형(Dtype)의 PyTorch `state_dict`로 복원한다. |
| `compute_weight_delta(local_w, global_w)` | 로컬 학습 완료 후 가중치(원문 수식 생략)에서 초기 전역 가중치(원문 수식 생략)를 레이어별로 차감하여 가중치 변화량(원문 수식 생략)을 산출한다. |
| `validate_weight_tensor(delta_dict, ref_keys)` | 수신된 업데이트 텐서의 레이어 키 일치 여부를 검사하고, `torch.isnan()` 및 `torch.isinf()` 검사를 통해 발산하거나 오염된 텐서를 즉시 탐지한다. |

### ② 중앙 파라미터 서버 및 실시간 모니터링 모듈 (`src/server.py`)
전역 모델(`CNNCifar`)을 관리하고, 클라이언트로부터 수신된 업데이트를 엄격히 검증한 뒤 표본 수 비례 가중 평균(FedAvg)으로 집계하는 중앙 서버이다.

**A. 내부 상태 관리 클래스 (`FederatedServerState`)**
- `aggregate_fedavg()`: 채택된 클라이언트들의 표본 수(`num_samples`) 합계를 기준으로 각 클라이언트의 가중 비중(원문 수식 생략)을 계산하여 (원문 수식 생략)를 가중 평균하고, 전역 모델에 반영(원문 수식 생략)한 뒤 CIFAR-10 테스트 데이터셋(10,000장)에 대한 전역 정확도(Test Acc)와 손실(Test Loss)을 평가한다.
- `log_exclusion(...)`: 검증에 실패한 업데이트의 발생 시각, 서버 라운드, 클라이언트 ID, 제출 메타데이터 및 구체적인 거절 사유를 서버 내 `excluded_logs` 리스트에 기록한다.

**B. REST API 엔드포인트 구성**

| HTTP 메서드 및 경로 | 역할 및 처리 로직 |
|---|---|
| `GET /` | 모니터링 웹 대시보드 HTML을 렌더링한다. |
| `GET /status` | 현재 라운드, 모델 버전, 대기 중인 업데이트 수, 라운드별 성능 이력(`round_history`), 제외 로그(`excluded_logs`)를 JSON 형태로 반환한다. |
| `GET /model` | 클라이언트에게 현재 라운드 번호(`round`), 모델 버전(`model_version`), Base64 직렬화된 현재 전역 모델 가중치(`weights`)를 배포한다. 동일 라운드 내 모든 클라이언트는 동일한 스냅샷을 수신한다. |
| `POST /update` | 클라이언트가 제출한 로컬 업데이트(원문 수식 생략)에 대해 5단계 문지기 검증을 수행한다. ① 필수 메타데이터 누락 확인 ② 현재 서버 라운드 일치 여부 ③ 모델 버전 일치 여부 ④ 동일 라운드 내 동일 `client_id` 중복 제출 여부 ⑤ 텐서 역직렬화 및 NaN/Inf 무결성 검증. 모두 통과 시 HTTP 200으로 채택하고, 하나라도 위배 시 HTTP 400으로 거절 및 제외 사유를 기록한다. |
| `POST /aggregate` | 라운드 내 채택된 업데이트들을 `aggregate_fedavg()`로 집계하여 전역 모델을 갱신하고, 다음 라운드로 상태를 전환(`current_round += 1`)한다. |

### ③ 가상 클라이언트 실행기 및 예외 방어 검증 모듈 (`src/client_runner.py`)
다수의 가상 클라이언트를 순차 시뮬레이션하며 서버와 HTTP API로 통신하고, 로컬 모델 학습 및 예외 차단 테스트를 수행하는 실행 모듈이다.

| 주요 함수 및 구성 요소 | 세부 역할 및 동작 원리 |
|---|---|
| `DatasetSplit(Dataset)` | 전체 CIFAR-10 학습 데이터셋(50,000장) 중 특정 클라이언트에게 할당된 인덱스(`user_idxs`)만 독립적으로 추출하는 PyTorch Dataset 래퍼 클래스이다. |
| `train_local_client(...)` | 서버로부터 받은 전역 모델 사본에 대해 지정된 하이퍼파라미터(optimizer=adam, lr=0.001, local_ep, local_bs)로 로컬 학습을 수행하고 갱신된 `state_dict`를 반환한다. |
| `run_exception_tests(...)` | `--test_exceptions` 옵션 활성화 시 실행되는 보안·무결성 자동 검증 스위트이다. 의도적으로 ① 라운드 불일치(Round +99), ② 잘못된 모델 버전, ③ NaN 오염 텐서, ④ 동일 클라이언트 중복 제출 4가지 비정상 요청을 서버로 전송하여 서버가 모두 HTTP 400으로 차단하고 로그에 기록하는지 검증한다. |
| `main()` | 글로벌 라운드 루프를 제어한다. 매 라운드마다 `GET /model`로 전역 가중치를 다운로드한 뒤, 참여 클라이언트(frac)별 로컬 학습 및 (원문 수식 생략) 산출 (원문 수식 생략) `POST /update` 제출 (원문 수식 생략) `POST /aggregate` 집계 요청을 순차적으로 수행한다. |

---

## 3. 시스템 실행 및 검증 절차
로컬 환경(Windows PowerShell, Python .venv 가상환경)에서 중앙 서버와 가상 클라이언트를 2개의 독립된 터미널로 구동하여 분산 통신을 실행한다.

**1단계: 직렬화 모듈 단위 테스트**

(원문 문서의 실행 화면 참고)

```powershell
cd Federated-Learning-PyTorch\src
python serialization.py
```

**2단계: 중앙 Flask 서버 구동 (터미널 1)**

(원문 문서의 실행 화면 참고)

```powershell
cd Federated-Learning-PyTorch\src
python server.py --port 5000 --num_clients 10
```

**3단계: 가상 클라이언트 실행 및 예외 차단 통합 검증 (터미널 2)**

(원문 문서의 실행 화면 참고)

```powershell
cd Federated-Learning-PyTorch\src
python client_runner.py --epochs 2 --local_ep 1 --test_exceptions
```

실행 직후 4가지 비정상 제출(라운드 불일치, 버전 불일치, NaN 이상치, 중복 제출)이 서버에서 모두 HTTP 400으로 거절되어 대시보드 하단 '제외된 업데이트 및 사유 로그'에 기록되며, 정상 클라이언트 10명의 업데이트만 집계되어 라운드별 정확도가 갱신됨을 검증하였다.

**실행화면**: 1단계, 2단계, 3단계 (원문 문서의 실행 화면 참고)

---

## 명세서에서 구현이 다 되었는지 여부

### 1. "중앙 Flask 서버 1개와 하나의 실행 프로그램에서 순차 학습하는 가상 클라이언트 10개 구성"
- **중앙 Flask 서버 1개** (`src/server.py`): `Flask(__name__)` 기반의 중앙 서버 앱(`app.run(host="127.0.0.1", port=5000)`)과 전역 상태 관리 객체 `FederatedServerState(num_clients=10)`로 구현되어 있습니다.
- **순차 학습하는 가상 클라이언트 10개** (`src/client_runner.py`): `main()` 함수 내에서 `num_users=10` 설정으로 10명의 로컬 데이터 인덱스(`user_groups`)를 분할한 뒤, 매 라운드마다 `for cid in selected_users:` 반복문을 통해 단일 프로세스 내에서 순차적으로 로컬 학습과 서버 제출을 수행합니다.

### 2. "REST API로 모델 다운로드·업데이트 제출 구현. 같은 라운드의 모든 클라이언트는 같은 전역 모델에서 학습"
- **REST API 모델 다운로드** (`src/server.py` (원문 수식 생략) `GET /model`): 서버의 `download_model()` 함수가 현재 라운드(`current_round`), 모델 버전(`model_version`), Base64 직렬화된 현재 전역 가중치(`global_model.state_dict()`)를 JSON으로 반환합니다.
- **REST API 업데이트 제출** (`src/server.py` (원문 수식 생략) `POST /update` & `src/client_runner.py`): 클라이언트가 `requests.post(f"{server_url}/update", json=payload)`로 학습 결과를 전송하고 서버의 `submit_update()`가 수신합니다.
- **동일 전역 모델 스냅샷 보장** (`src/client_runner.py` (원문 수식 생략) `main()`): 라운드 시작 시점에 `GET /model`로 받아온 `global_weights`를 고정해 두고, 각 클라이언트(`cid`)가 학습할 때마다 `local_model.load_state_dict(copy.deepcopy(global_weights))`로 원본을 깊은 복사(Deep Copy)하여 사용하므로, 같은 라운드의 모든 클라이언트가 100% 동일한 초기 전역 모델에서 출발합니다.

### 3. "업데이트는 로컬 학습 후 가중치와 시작 전역 가중치의 차이로 정의. 채택 업데이트를 표본 수에 비례해 평균하는 FedAvg 연결"
- **가중치 변화량(원문 수식 생략) 정의** (`src/serialization.py` (원문 수식 생략) `compute_weight_delta()`): 로컬 학습 후 모델 가중치(`updated_weights`)에서 시작 전역 가중치(`global_weights`)를 레이어별로 차감(원문 수식 생략)하여 전송합니다.
- **표본 수 비례 FedAvg 집계** (`src/server.py` (원문 수식 생략) `aggregate_fedavg()`): `POST /aggregate` 호출 시 서버가 채택된 업데이트들의 전체 표본 수(`total_samples = sum(num_samples)`)를 구한 뒤, 각 클라이언트의 비중(`weight_ratio = num_samples / total_samples`)을 곱해 `avg_delta`를 산출하고 `new_state_dict[key] += avg_delta[key]`로 전역 모델에 반영합니다.

### 4. "라운드·모델 버전·클라이언트 ID를 확인하고, 중복 제출이나 잘못된 업데이트는 제외 사유와 함께 기록"
- **5단계 검증 및 차단** (`src/server.py` (원문 수식 생략) `submit_update()`):
  1. 필수 메타데이터(`client_id`, `round`, `model_version`, `delta_weights`) 누락 확인
  2. `req_round != server_state.current_round` (라운드 번호 일치 확인)
  3. `req_version != server_state.model_version` (모델 버전 일치 확인)
  4. `client_id in server_state.accepted_updates` (동일 라운드 중복 제출 확인)
  5. `validate_weight_tensor()` (`src/serialization.py`에서 레이어 키 구조 불일치 및 NaN/Inf 이상치 검출)
- **제외 사유 기록 및 대시보드 시각화** (`src/server.py` (원문 수식 생략) `log_exclusion()`): 검증 실패 시 HTTP 400으로 거절함과 동시에 `server_state.excluded_logs`에 시각, 라운드, `client_id`, 제출 정보, 구체적 제외 사유를 저장하며, `http://127.0.0.1:5000` 웹 대시보드 하단 표에서 실시간으로 확인할 수 있습니다. (또한 `client_runner.py --test_exceptions`로 4가지 차단 동작 검증까지 완료했습니다.)
