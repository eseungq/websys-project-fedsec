# 💻 `src/` 소스 코드 모듈 안내

본 디렉터리는 명세서 요구사항에 따라 **통신 · 직렬화 · 공격/방어 · 모델 · 데이터 · 실행 · 평가 · 시각화** 기능이 독립된 모듈로 분리되어 있습니다.

## 1. 신규 구현 및 확장 핵심 모듈

| 파일명 | 역할 분류 | 상세 설명 |
| :--- | :--- | :--- |
| **`server.py`** | 중앙 서버 및 통합 대시보드 | Flask REST API 서버(`/model`, `/update`, `/aggregate`, `/reset`), 업데이트 유효성 검증(라운드·버전·중복·지연·`NaN/Inf`), 체크포인트(`.pt`) 저장, 3개 탭 통합 웹 UI 제공 |
| **`client_runner.py`** | 가상 클라이언트 시뮬레이터 | 10개 가상 클라이언트의 순차 로컬 학습, $\Delta W$ 산출, 4가지 운영 환경(`baseline`, `delay`, `partial`, `noniid`) 시뮬레이션, `--config` 및 체크포인트 재실행 지원 |
| **`security.py`** | 공격 및 방어 알고리즘 | 내부자 협력 공격(`apply_collusion_attack`, 악성 평균 × `-5`), L2 노름 클리핑(`clip_update_norm`), 좌표별 중앙값 집계(`aggregate_robust`, 악성 ID 미전달) 및 단위 수치 테스트 포함 |
| **`serialization.py`** | 텐서 직렬화 및 무결성 검증 | PyTorch `state_dict`의 Base64 직렬화/역직렬화, 가중치 변화량($\Delta W$) 계산, 키 일치 및 `NaN/Inf` 손상 텐서 필터링 |
| **`dlg_attack_defense.py`** | 모델 역추적 공격·방어 | 단일 CIFAR-10 이미지의 1회 그래디언트 기반 L-BFGS 복원 공격 및 가우시안 노이즈($\sigma$) 방어, 원본·복원 비교 PNG 및 MSE JSON 출력 |
| **`evaluator.py`** | 정량 평가 지표 산출기 | `../save/logs/log_*.json`을 자동 취합하여 `공격 피해`, `방어 개선`, `정상 손실(%p)` 및 시드별 평균을 계산하고 CSV로 저장 |
| **`plot_results.py`** | 비교 그래프 시각화기 | 저장된 실험 로그를 바탕으로 환경별 라운드 정확도 변화 곡선(`curve_*.png`)과 조건별 최종 정확도 막대 그래프(`final_accuracy_comparison.png`) 생성 |
| **`run_batch.py`** | 일괄 실험 자동화 스크립트 | 4개 환경 × 2개 공격 × 3개 방어 × 다중 시드 조합을 순차 자동 실행(완료 로그 자동 스킵) 후 `evaluator.py`와 `plot_results.py`까지 일괄 수행 |

---

## 2. 기반 오픈소스(`Federated-Learning-PyTorch`) 활용 모듈

| 파일명 | 역할 | 비고 |
| :--- | :--- | :--- |
| **`models.py`** | 신경망 모델 정의 | 기존 `CNNCifar`(합성곱 2층) 외에 BatchNorm 없는 3층 합성곱 구조인 `SimpleCNN` 클래스 추가 |
| **`utils.py`** | 데이터셋 로더 | CIFAR-10 학습/테스트 데이터 로드 및 IID/Non-IID 분할 호출 (`get_dataset`) |
| **`sampling.py`** | 클라이언트 데이터 분할 | 10개 가상 클라이언트에 동일 표본 수(5,000장)로 균등 배분(`cifar_iid`) 또는 클래스 편향 배분(`cifar_noniid`) 수행 |
| **`update.py` / `options.py`** | 기존 로컬 학습 및 옵션 | 기반 저장소의 참조용 학습 루프 및 CLI 파서 정의 |
| **`baseline_main.py` / `federated_main.py`** | 기존 단일 실행 스크립트 | 기반 저장소의 원본 중앙집중식/연합학습 예제 코드 (참조용 보존) |