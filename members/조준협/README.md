# 🛡️ 연합학습(FedAvg) 보안 시뮬레이션 및 공격·방어 원인 분석 시스템

CIFAR-10 데이터셋 기반의 **중앙 Flask REST API 파라미터 서버**와 **10개 가상 클라이언트 시뮬레이터**를 통합하여, **내부자 협력 공격(Collusion Attack)** 및 **모델 역추적 공격(DLG)**의 피해와 방어 기법(노름 클리핑, 좌표별 중앙값, 가우시안 노이즈)의 효과를 정량 분석하는 프로젝트입니다.

---

## 📁 프로젝트 디렉터리 구조

```text
Federated-Learning-PyTorch/
├── README.md               # 전체 프로젝트 개요, 명세서 구현 매핑표, 실행 및 API 설명서 (현재 파일)
├── requirments.txt         # 파이썬 패키지 의존성 목록
├── configs/                # 외부 실험 설정 파일(JSON) 모음 (configs/README.md 참고)
├── data/                   # CIFAR-10 데이터셋 저장 경로 (data/README.md 참고)
├── save/                   # 실험 로그, 체크포인트(.pt), 분할 정보, 평가표, 그래프 출력 경로 (save/README.md 참고)
└── src/                    # 학습·공격·방어·통신·로그·분석 모듈 소스 코드 (src/README.md 참고)
```

---

## ⚙️ 실행 환경 및 설치 방법

### 1. 권장 실행 환경
* **OS**: Windows 10 / 11 (또는 Linux / Kaggle GPU 환경)
* **Python**: 3.10 이상
* **주요 라이브러리**: `torch`, `torchvision`, `flask`, `requests`, `numpy`, `matplotlib`

### 2. 가상환경 생성 및 패키지 설치
```powershell
# 1) 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) 필수 패키지 설치
pip install torch torchvision flask requests numpy matplotlib
```

---

## 🚀 실행 방법 (단계별 시연 및 실험 가이드)

모든 실행 명령어는 `src` 디렉터리 내부에서 수행합니다.
```powershell
cd Federated-Learning-PyTorch\src
```

### 1. 통합 웹 대시보드 서버 실행 (필수)
```powershell
python server.py
```
* 브라우저에서 **`http://127.0.0.1:5000`** 접속 시 3개 탭으로 구성된 통합 제어 대시보드가 열립니다.
  * **탭 1**: 4개 운영 환경·공격·방어·모델(`CNNCifar` / `SimpleCNN`) 선택 실행 및 라운드/클라이언트/차단 로그 실시간 조회
  * **탭 2**: 단일 이미지 DLG 역추적 공격 및 가우시안 노이즈 단계별 복원 이미지·MSE 조회/실행
  * **탭 3**: `공격 피해 · 방어 개선 · 정상 손실(%p)` 자동 비교표 및 저장된 체크포인트(`.pt`) 기반 원인 분석 재실행

### 2. 단위 수치 검증 테스트 (집계·방어 수식 검증용)
```powershell
python security.py
```
* 악성 클라이언트 2명 담합(`평균 × -5`) 및 짝수 개 업데이트의 좌표별 중앙값(가운데 두 값 평균) 산출을 작은 수치 예제로 검증합니다[cite: 24, 25, 26].

### 3. 외부 설정 파일(`--config`)을 이용한 단일 실험 실행
```powershell
# server.py가 켜진 상태에서 별도 터미널 실행
python client_runner.py --config ../configs/sample_config.json
```

### 4. 전체 비교 실험(24개 조합 × 다중 시드) 일괄 자동 실행
```powershell
# server.py가 켜진 상태에서 별도 터미널 실행 (완료된 로그는 자동 건너뜀)
python run_batch.py --model simple_cnn --optimizer sgd --lr 0.01 --local_ep 1 --local_bs 32 --epochs 40 --clip_norm 2.5 --seeds 1,2,3
```

### 5. 정량 평가표(`CSV`) 및 비교 그래프(`PNG`) 단독 생성
```powershell
python evaluator.py      # ../save/tables/security_evaluation_summary.csv 생성
python plot_results.py   # ../save/plots/*.png 생성
```

---

## 📋 프로젝트 명세서 요구사항 구현 매핑표

| 명세서 단계 | 세부 요구사항 | 구현 모듈 및 처리 방식 |
| :--- | :--- | :--- |
| **1. 기본 학습 및 데이터** | CIFAR-10 10분할, 시드/초기모델 통일, `CNNCifar` 및 `SimpleCNN` 지원 | `utils.py`, `models.py`, `client_runner.py` (`split_*.json` 저장) |
| **2. 연합학습 환경** | 중앙 Flask 서버 + 가상 클라이언트 10개, $\Delta W = W_{local} - W_{global}$ 전송 | `server.py`, `client_runner.py`, `serialization.py` |
| **2. 예외 업데이트 검증** | 라운드·모델 버전·중복 제출·`NaN/Inf` 손상 텐서 검사 및 제외 사유 기록 | `server.py` (`/update`), `serialization.py` (`validate_weight_tensor`) |
| **3. 모델 역추적(DLG)** | 이미지 1장 1회 그래디언트 복원, 가우시안 노이즈(없음/작음/큼) MSE 비교 | `dlg_attack_defense.py`, `server.py` (대시보드 탭 2) |
| **3. 내부자 협력 공격** | 악성 2명 동시 참여 시 평균 후 `-5배` 동일 벡터 제출, 1명 `-5배`, 0명 미적용 | `security.py` (`apply_collusion_attack`) |
| **3. 내부자 협력 방어** | L2 노름 클리핑 후 FedAvg / 좌표별 중앙값(짝수 시 가운데 두 값 평균, 악성 ID 미전달) | `security.py` (`clip_update_norm`, `aggregate_robust`) |
| **4. 4단계 운영 환경**| 기준(`baseline`), 지연(`delay`, 3명 마감초과), 부분참여(`partial`, 5/10), 편향(`noniid`) | `client_runner.py`, `server.py` (가상 시각과 실제 실행 시간 분리 판정) |
| **5. 로그 및 평가 지표** | 조작·방어 전후 노름, 전체 vs 채택 악성 비율(`0`이면 `N/A`), `< 3`개 시 갱신 생략 | `server.py` (`aggregate_round`), `evaluator.py` (`%p` 자동 산출) |
| **6. 원인 분석 재실행** | 저장된 라운드 체크포인트(`.pt`) 및 데이터 분할(`.json`) 로드 후 조건 변경 재실행 | `server.py` (`/reset`, 대시보드 탭 3), `client_runner.py` (`--checkpoint_path`) |

---

## 🌐 Flask REST API 명세서

| Endpoint | Method | 설명 | 주요 요청(Request) / 응답(Response) 필드 |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | 통합 웹 대시보드 HTML UI 반환 | 실시간 라운드·클라이언트·차단 로그, DLG, 평가표 조회 |
| `/status` | `GET` | 현재 서버 상태 및 누적 로그 조회 | `is_running`, `current_round`, `round_history`, `client_logs`, `excluded_logs` |
| `/reset` | `POST` | 새 실험(또는 체크포인트 재실행) 초기화 | `model`, `env`, `attack`, `defense`, `clip_norm`, `seed`, `checkpoint_path` |
| `/model` | `GET` | 현재 라운드 전역 모델 가중치 배포 | `round`, `model_version`, `weights` (Base64 직렬화 텐서) |
| `/update` | `POST` | 클라이언트 업데이트($\Delta W$) 제출 및 검증 | `client_id`, `round`, `model_version`, `delta_weights`, `virtual_time` |
| `/aggregate` | `POST` | 라운드 마감 후 강건 집계 및 전역 모델 갱신 | `selected_count`, `delayed_count`, `virtual_elapsed`, `is_final_round` |
| `/api/run_fl` | `POST` | 대시보드 UI 설정으로 학습 프로세스 백그라운드 실행 | `model`, `env`, `attack`, `defense`, `optimizer`, `lr`, `epochs`, `seed` |
| `/api/run_dlg` | `POST` | 대시보드 UI 설정으로 DLG 역추적 공격·방어 실행 | `index`, `iters`, `noise_none`, `noise_small`, `noise_large` |
| `/api/summary_metrics` | `GET` | 누적된 로그로부터 `%p` 정량 비교표 반환 | `attack_damage`, `defense_gain`, `normal_loss` |

---

## 📚 공개 코드 출처 및 주요 수정 사항

1. **FedAvg 공개 구현 (`Federated-Learning-PyTorch`)**
   * **활용 범위**: `models.py` (`CNNCifar`), `sampling.py` (`cifar_iid`, `cifar_noniid`), `utils.py` (`get_dataset`)
   * **수정 및 확장 사항**: 단일 스크립트 로컬 루프 구조를 중앙 Flask REST API 서버(`server.py`)와 클라이언트 실행기(`client_runner.py`)로 분리하고, BatchNorm 통계 모호성을 배제한 3층 합성곱 모델 `SimpleCNN`을 `models.py`에 추가했습니다.
2. **Deep Leakage from Gradients (`DLG`) 공개 구현**
   * **활용 범위**: `LeNet` 기반 단일 이미지 L-BFGS 역추적 최적화 구조
   * **수정 및 확장 사항**: 초기 더미 입력 분포를 Sigmoid 포화 방지를 위해 균등분포(`torch.rand`)로 개선하고, 전송 전 그래디언트 가우시안 노이즈 방어(`없음/작음/큼`)와 원본·복원 비교 시각화 및 MSE JSON 저장 기능을 `dlg_attack_defense.py`로 모듈화했습니다.

   ---

## ✅ 코드 구현 완료 현황 및 향후 실험·분석 로드맵

### 1. 코드 구현 완료 점검 (100% 완료)
* **연합학습 통신 및 예외 차단 (`server.py`, `serialization.py`)**
  * 중앙 Flask 서버 + 가상 클라이언트 10개 구조 및 Base64 텐서 직렬화 구현
  * 라운드·모델 버전·클라이언트 ID 불일치, 중복 제출, 가상 마감 시간 초과, `NaN/Inf` 손상 텐서 자동 차단 및 제외 사유 기록
* **공격 및 방어 알고리즘 (`security.py`, `dlg_attack_defense.py`)**
  * **내부자 협력 공격**: 악성 2명 동시 참여 시 평균 후 `-5배` 동일 벡터 제출, 1명 참여 시 `-5배`, 0명 참여 시 미적용
  * **내부자 협력 방어**: L2 노름 클리핑 후 FedAvg 및 좌표별 중앙값 집계(짝수 개일 때 가운데 두 값 평균 적용, 방어 함수에 악성 ID 미전달)
  * **모델 역추적(DLG) 공격·방어**: 단일 CIFAR-10 이미지의 1회 그래디언트 기반 L-BFGS 복원 및 가우시안 노이즈 3단계(`없음`·`작음`·`큼`) MSE 비교
* **4가지 운영 환경 및 재현성 (`client_runner.py`, `server.py`)**
  * 기본(`baseline`), 네트워크 지연(`delay`, 고정 3개 클라이언트 가상 마감 초과), 부분 참여(`partial`, 라운드당 5/10 무작위 선택), 레이블 편향(`noniid`, 동일 이미지·표본 수 유지 및 클래스 배정만 변경) 지원
  * 채택 업데이트 수 `< 3`일 때 전역 모델 갱신 생략 및 사유 기록, 전체 악성 비율과 실제 채택 악성 비율(`0`명일 때 `N/A`) 구분 기록
* **로그 · 재실행 · 자동화 · 평가 및 시각화 (`evaluator.py`, `plot_results.py`, `run_batch.py`, `configs/sample_config.json`)**
  * 라운드별 모델 체크포인트(`.pt`) 및 클라이언트 데이터 분할(`split_*.json`) 자동 저장과 조건 변경 부분 재실행(Re-run) 기능
  * `공격 피해 · 방어 개선 · 정상 손실(%p)` 정량 평가표(`CSV`) 및 라운드별 정확도 곡선·조건별 비교 막대 그래프(`PNG`) 자동 생성
  * 24개 조합(4개 환경 × 2개 공격 × 3개 방어) × 다중 시드 일괄 자동 실행기(`run_batch.py`) 및 외부 설정 파일 교체 실행(`--config`) 지원

---

### 향후 실험 수행 및 산출물 작성 로드맵 (확인을 위해 추가한 내용임)

#### STEP 1. 기준선 확정 및 전체 비교 실험 수행 (명세서 4단계)
1. **하이퍼파라미터 및 클리핑 상한(`Clip Norm`) 고정**
   * 팀과 합의한 기준 설정(예: `SimpleCNN`, `SGD(lr=0.01, momentum=0.9)`, `local_ep=1`, `local_bs=32`, `epochs=40`)으로 `기본 환경 + 정상 + FedAvg`를 먼저 실행해 정상 평균 L2 노름을 확인하고, 비교 실험 전체에서 사용할 고정 `Clip Norm` 값을 확정합니다.
2. **24개 조합 × 다중 시드(72회) 일괄 비교 실험**
   * `run_batch.py`를 통해 4개 환경(`baseline`, `delay`, `partial`, `noniid`) × 2개 공격(`none`, `collusion`) × 3개 방어(`fedavg`, `clipped_fedavg`, `clipped_median`)를 3개 시드(`seed=1, 2, 3`)로 실행하여 로그·평가표·비교 그래프를 확보합니다.
3. **DLG 모델 역추적 실험 기록**
   * 대표 CIFAR-10 이미지 인덱스들에 대해 가우시안 노이즈 3단계(`없음` / `작음` / `큼`) 및 L-BFGS 반복 횟수(`100회` vs `300회`)별 복원 이미지와 픽셀 MSE를 기록합니다.

#### STEP 2. 로그 기반 원인 분석 사례 3건 및 체크포인트 재실험 (명세서 6단계)
아래 3개 영역에서 각각 1건씩 사례를 선정하고, **「관찰된 현상 → 원인 가설 → 조건을 바꾼 재실험 결과 → 한계」** 순서로 분석을 수행합니다.
1. **내부자 공격·방어 사례**: 특정 라운드에서 악성 클라이언트 2명이 동시 참여해 정확도가 급락한 현상을 로그로 포착한 뒤, 저장된 체크포인트(`.pt`)로 해당 라운드만 `클리핑+중앙값`으로 변경하거나 악성 1명의 조작을 해제하여 재실행한 결과 비교
2. **운영 환경 변화 사례**: `레이블 편향(noniid)` 또는 `부분 참여(partial)` 환경에서 좌표별 중앙값 방어의 정상 손실이나 방어 효율이 기준 환경과 달라진 원인을 분석하고 동일 시작 모델·분할 조건에서 재실험
3. **모델 역추적(DLG) 사례**: 동일 이미지·모델·복원 초기값에서 가우시안 노이즈 크기(sigma)만 바꾼 경우와 L-BFGS 반복 횟수(`iters`)만 줄인 경우를 비교하여 노이즈 방어 효과와 최적화 미수렴 현상을 구분 해석

#### STEP 3. 최종 요구 산출물 정리 및 발표 준비
1. **분석 보고서 작성**: `evaluator.py`의 정량 평가표(`%p`), `plot_results.py`의 비교 그래프, 원인 분석 사례 3건, 실험 결과에 근거한 **상황별 방어 권고 2건** 및 각 기법의 한계 기술
2. **기술 문서 보완**: `security.py` 단위 실행을 통한 **집계·방어 계산 작은 수치 예제** 결과 첨부 및 실행 환경·API·입력 설정 파일 안내 최종 점검
3. **발표 자료(2~3쪽) 및 실시간 시연 준비**: 통합 웹 대시보드(탭 1~3)를 활용한 실제 학습·차단 로그·DLG 복원·체크포인트 재실행 시연 리허설