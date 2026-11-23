# 🛡️ 연합학습(FedAvg) 보안 시뮬레이션 및 공격·방어 원인 분석 시스템

CIFAR-10 데이터셋 기반의 **중앙 Flask REST API 파라미터 서버**와 **10개 가상 클라이언트 순차 실행기**를 통합하여, **내부자 협력 공격(Insider Collusion)** 및 **모델 역추적 공격(Deep Leakage from Gradients, DLG)**의 피해와 방어 기법(L2 노름 클리핑, 좌표별 중앙값, 가우시안 노이즈)의 효과를 정량 분석하는 화이트톤 웹 대시보드 기반 시뮬레이터입니다.

---

## 📁 프로젝트 디렉터리 구조

```text
project5/
├── README.md                            # 프로젝트 개요, 명세서 구현 매핑표, 실행 및 API 설명서
├── configs/
│   └── default_config.json              # 기본 실험 하이퍼파라미터 및 환경·공격·방어 설정 파일
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py                    # 연합학습용 CNNCifar 및 역추적용 LeNetDLG 모델 정의
│   │   ├── data_manager.py              # CIFAR-10 40k/10k/10k 분할 (IID 및 2-클래스 레이블 편향)
│   │   └── local_trainer.py             # 로컬 학습, 업데이트(Δw) 산출, 정확도·L2 노름 평가
│   ├── security/
│   │   ├── __init__.py
│   │   ├── attacks.py                   # 내부자 협력 공격(-5배) 및 mit-han-lab/dlg 역추적 공격
│   │   └── defenses.py                  # L2 노름 클리핑, 좌표별 중앙값(짝수 평균), 가우시안 노이즈
│   ├── comm/
│   │   ├── __init__.py
│   │   ├── server.py                    # 중앙 Flask 서버 (웹 대시보드 서빙 + REST API + subprocess 실행)
│   │   └── client_runner.py             # 10개 가상 클라이언트 순차 생성 및 라운드 학습 실행기
│   └── logging_utils/
│       ├── __init__.py
│       └── audit_logger.py              # 감사 CSV 로그, 라운드 체크포인트, DLG 이미지, %p 지표 산출
├── templates/
│   └── index.html                       # 화이트톤 웹 대시보드 메인 HTML
├── static/
│   ├── css/
│   │   └── style.css                    # 화이트톤 대시보드 스타일시트
│   └── js/
│       └── app.js                       # 실시간 폴링, 10개 클라이언트 카드, SVG 차트 및 탭 제어
├── data/                                # [자동 생성] CIFAR-10 원본 데이터셋 다운로드 경로
└── outputs/                             # [자동 생성] 실험 로그(CSV), 체크포인트(.pt), DLG 복원 이미지 출력 경로
    ├── logs/
    ├── checkpoints/
    └── dlg_images/
```

---

## ⚙️ 실행 환경 및 설치 방법

### 1. 권장 실행 환경
* **OS**: Windows 10 / 11 (또는 Linux / macOS)
* **Python**: 3.10 이상
* **주요 라이브러리**: `torch`, `torchvision`, `flask`, `requests`, `pillow`, `numpy`

### 2. 가상환경 생성 및 패키지 설치
```powershell
# 1) 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) 필수 패키지 설치
pip install torch torchvision flask requests pillow numpy
```

---

## 🚀 실행 방법 (단계별 시연 및 실험 가이드)

### 1. 통합 웹 대시보드 서버 실행 (필수)
프로젝트 루트 디렉터리에서 중앙 Flask 서버를 실행합니다.
```powershell
python src/comm/server.py
```
* 브라우저에서 **`http://127.0.0.1:5000`** 접속 시 실시간 클라이언트 모니터와 4개 분석 탭으로 구성된 화이트톤 대시보드가 열립니다.
  * **좌측 제어 패널**: 4가지 운영 환경(`정상` · `네트워크 지연` · `부분 참여` · `레이블 편향`), 3가지 공격(`정상` · `모델 역추적(DLG)` · `내부자 협력 공격`), 공격 유형에 연동되는 방어 기법, `Seed`·`Rounds` 설정 후 **`[▶ 시뮬레이션 실행]`** 클릭
  * **상단 실시간 모니터**: `Client #0` ~ `Client #9` 10개 박스가 순차 활성화되며 채택·지연제외·미참여 상태를 실시간 표시
  * **탭 1 (`학습·참여 추이`)**: 라운드별 검증 정확도·손실 SVG 곡선 및 최종 클래스별(`0`~`9`) 정확도 조회
  * **탭 2 (`모델 역추적 (DLG)`)**: 단일 이미지 1회 그래디언트 복원 결과(원본 vs 복원 이미지) 및 픽셀 오차(`MSE`) 비교
  * **탭 3 (`환경·방어 비교표 (%p)`)**: 동일 환경·시드 기준 `공격 피해 · 방어 개선 · 정상 손실(%p)` 자동 산출 및 누적 기록 비교
  * **탭 4 (`특정 라운드 재실행`)**: 저장된 라운드 체크포인트(`round_<N>.pt`)를 불러와 방어 기법 변경 후 재집계 결과 비교

### 2. 외부 설정 파일(`--config`)을 이용한 단독 클라이언트 실행기 구동
`server.py`가 실행 중이며 초기화가 완료된 상태에서 별도 터미널로 실행할 수 있습니다.
```powershell
python src/comm/client_runner.py --config outputs/checkpoints/<exp_id>/runtime_config.json --server [http://127.0.0.1:5000](http://127.0.0.1:5000)
```

---

## 📋 프로젝트 명세서 요구사항 구현 매핑표

| 명세서 단계 | 세부 요구사항 | 구현 모듈 및 처리 방식 |
| :--- | :--- | :--- |
| **1. 기본 학습 및 데이터** | CIFAR-10 학습(40k)·검증(10k)·테스트(10k) 분리, 시드/초기모델 통일 | `src/core/models.py` (`CNNCifar`), `src/core/data_manager.py` (`load_cifar10_splits`) |
| **2. 연합학습 환경** | 중앙 Flask 서버 + 순차 가상 클라이언트 10개, $\Delta w_k = w_k^{\text{local}} - w_t$ 전송 | `src/comm/server.py`, `src/comm/client_runner.py`, `src/core/local_trainer.py` |
| **2. 예외 업데이트 검증** | 라운드·모델 버전·클라이언트 ID·중복 제출·가상 마감 시각 초과 검사 | `src/comm/server.py` (`submit_client_update`) |
| **3. 모델 역추적(DLG)** | 단일 이미지 1회 그래디언트 복원, 가우시안 노이즈(없음/작음/큼) MSE 비교 | `src/security/attacks.py` (`run_dlg_inversion`), `src/security/defenses.py` |
| **3. 내부자 협력 공격** | 악성 2명(`#3, #7`) 동시 참여 시 평균 후 `-5배` 동일 벡터 제출, 1명 `-5배`, 0명 미적용 | `src/security/attacks.py` (`apply_insider_collusion`) |
| **3. 내부자 협력 방어** | L2 노름 클리핑 후 FedAvg / 좌표별 중앙값(짝수 시 가운데 두 값 평균, 악성 ID 미전달) | `src/security/defenses.py` (`clip_updates_l2`, `aggregate_coordinate_wise_median`) |
| **4. 4단계 운영 환경** | 정상(`normal`), 지연(`delay`, `#2,#5,#8` 마감초과), 부분참여(`partial`, 5/10), 편향(`label_skew`) | `src/comm/client_runner.py`, `src/core/data_manager.py` (가상 시각과 실제 시간 분리) |
| **5. 로그 및 평가 지표** | 조작·방어 전후 노름, 전체 vs 채택 악성 비율(`0`이면 `N/A`), `< 3`개 시 갱신 보류 | `src/comm/server.py` (`_finalize_round`), `src/logging_utils/audit_logger.py` |
| **6. 원인 분석 재실행** | 저장된 라운드 체크포인트(`.pt`) 및 설정 로드 후 방어 조건 변경 재실행 | `src/comm/server.py` (`/api/rerun_round`), `src/logging_utils/audit_logger.py` |

---

## 🌐 Flask REST API 명세서

| Endpoint | Method | 설명 | 주요 요청(Request) / 응답(Response) 필드 |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | 통합 화이트톤 웹 대시보드 HTML UI 반환 | 실시간 클라이언트 모니터, 차트, DLG, 비교표, 감사 로그 UI |
| `/api/start` | `POST` | 실험 초기화 및 `subprocess.Popen` 자식 프로세스 실행 | `environment`, `attack`, `defense`, `dlg_noise_mode`, `seed`, `epochs` |
| `/api/stop` | `POST` | 실행 중인 `client_runner.py` 자식 프로세스 강제 종료 | `status` (`stopped`) |
| `/api/status` | `GET` | 현재 진행 상태, 10개 클라이언트 카드, 차트·로그 조회 | `status`, `current_round`, `client_cards`, `val_acc_history`, `audit_logs`, `comparison` |
| `/api/model` | `GET` | 현재 라운드 전역 모델 가중치 배포 | `exp_id`, `round`, `model_version`, `weights` (Base64 직렬화 텐서) |
| `/api/round_plan` | `POST` | 라운드 참여 선정 클라이언트 명단 등록 | `selected_clients` |
| `/api/client_start` | `POST` | 특정 클라이언트(`#0`~`#9`) 학습 시작 실시간 알림 | `client_id`, `mode` (`training` \| `dlg`) |
| `/api/submit_update` | `POST` | 클라이언트 업데이트($\Delta w_k$) 제출 및 검증·집계 | `client_id`, `round`, `model_version`, `delta_w`, `pre_attack_norm`, `post_attack_norm`, `virtual_time_sec` |
| `/api/dlg_submit` | `POST` | 단일 이미지 그래디언트 제출 및 서버 측 DLG 역추적 수행 | `client_id`, `sample_index`, `noise_mode`, `noise_std`, `gradients` |
| `/api/rerun_round` | `POST` | 저장된 라운드 체크포인트(`.pt`) 기반 방어 기법 변경 재집계 | `exp_id`, `round`, `defense`, `unmask_client_ids` $\rightarrow$ `before_round_val_acc`, `after_rerun_val_acc`, `diff_pp` |

---

## 📚 공개 코드 출처 및 주요 수정 사항

1. **FedAvg 공개 구현 (`Federated-Learning-PyTorch`)**
   * **활용 범위**: `CNNCifar` 합성곱 신경망 구조(`src/core/models.py`), IID 및 샤드 기반 Non-IID 분할 개념(`src/core/data_manager.py`)
   * **수정 및 확장 사항**: 단일 스크립트 로컬 루프 구조를 중앙 Flask REST API 서버(`src/comm/server.py`)와 `subprocess.Popen` 기반 순차 클라이언트 실행기(`src/comm/client_runner.py`)로 분리하고, CIFAR-10을 학습(40k)·검증(10k)·테스트(10k)로 고정 분할하여 동일한 40k 풀에서 IID와 2-클래스 레이블 편향을 비교하도록 개선했습니다.
2. **Deep Leakage from Gradients (`mit-han-lab/dlg`)**
   * **활용 범위**: `LeNetDLG` Strided Conv + Sigmoid 신경망 구조(`src/core/models.py`) 및 단일 이미지 그래디언트 매칭 기반 L-BFGS 역추적 최적화(`src/security/attacks.py`)
   * **수정 및 확장 사항**: 최적화 과정에서 픽셀 값이 발산해 원색 포화되는 현상을 방지하기 위해 `Sigmoid` 잠재 변수 매핑(`torch.sigmoid(dummy_latent)`)과 `LBFGS(line_search_fn="strong_wolfe")`를 적용했으며, 전송 전 가우시안 노이즈 방어(`없음 0.0` / `작음 0.01` / `큼 0.1`)와 원본·복원 이미지 비교 및 `MSE` 자동 기록 기능을 추가했습니다.

---

## ✅ 코드 구현 완료 현황 및 향후 실험·분석 로드맵

### 1. 코드 구현 완료 점검
* **연합학습 통신 및 예외 차단 (`src/comm/server.py`, `src/comm/client_runner.py`)**
  * 중앙 Flask 서버 + 순차 가상 클라이언트 10개 구조 및 Base64 텐서 직렬화 구현
  * 라운드·모델 버전·클라이언트 ID 불일치, 중복 제출, 가상 마감 시각(`10.0s`) 초과 자동 차단 및 제외 사유 기록
* **공격 및 방어 알고리즘 (`src/security/attacks.py`, `src/security/defenses.py`)**
  * **내부자 협력 공격**: 악성 2명(`#3, #7`) 동시 참여 시 정상 업데이트 평균 후 `-5배` 동일 벡터 제출, 1명 참여 시 `-5배`, 0명 참여 시 미적용
  * **내부자 협력 방어**: L2 노름 클리핑 후 FedAvg 및 좌표별 중앙값 집계(짝수 개일 때 가운데 두 값 평균 적용, 방어 함수에 악성 ID 미전달)
  * **모델 역추적(DLG) 공격·방어**: 단일 CIFAR-10 이미지의 1회 그래디언트 기반 L-BFGS 복원 및 가우시안 노이즈 3단계(`없음` · `작음` · `큼`) MSE 비교
* **4가지 운영 환경 및 재현성 (`src/core/data_manager.py`, `src/logging_utils/audit_logger.py`)**
  * 정상(`normal`), 네트워크 지연(`delay`, 고정 3명 `#2, #5, #8` 가상 마감 초과), 부분 참여(`partial`, 라운드당 5/10 무작위 선택), 레이블 편향(`label_skew`, 동일 40k 풀·표본 수 유지 및 2-클래스 집중 배정) 지원
  * 채택 업데이트 수 `< 3`일 때 전역 모델 갱신 보류 및 사유 기록, 전체 악성 비율과 실제 채택 악성 비율(`0`명일 때 `N/A`) 구분 기록
  * 라운드별 모델 체크포인트(`round_<N>.pt`) 및 실험 메타데이터(`experiment_meta.json`) 자동 저장과 조건 변경 재실행(`POST /api/rerun_round`) 지원

---

### 2. 실험 수행 및 결과 보고서 작성

#### setp 1. 다중 라운드 연합학습 실험표 

| 공격 상황 \ 운영 환경 | 정상 (`normal`) | 네트워크 지연 (`delay`) | 부분 참여 (`partial`) | 레이블 편향 (`label_skew`) |
| :--- | :---: | :---: | :---: | :---: |
| **정상 (`normal`)** | O | O | X | X |
| **내부자 협력 (`collusion`)** | O | x | X | X |
| **모델 역추적 (`dlg`)** | O | X | X | X |

* O 항목은 실험을 웹 상에서 진행한 것이고, X 부분은 추후 실험을 진행해야 합니다.

#### setp 2. 실행 화면