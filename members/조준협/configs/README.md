# ⚙️ `configs/` 실험 설정 파일 안내

이 폴더는 소스 코드를 수정하지 않고 **설정 파일만 교체하여 다양한 연합학습 보안 실험을 재현**하기 위한 JSON 설정 파일들을 보관합니다.

## 1. 파일 목록
* **`sample_config.json`**: `SimpleCNN`, `SGD(lr=0.01)`, 내부자 협력 공격(`-5배`), 노름 클리핑 + 좌표별 중앙값 방어가 적용된 표준 실험 설정 예시

## 2. 설정 파일 파라미터 명세

| 키(Key) | 타입 | 예시 값 | 설명 |
| :--- | :--- | :--- | :--- |
| `model` | `string` | `"simple_cnn"` | 모델 구조 선택 (`"cnn"` 또는 `"simple_cnn"`) |
| `env` | `string` | `"baseline"` | 운영 환경 (`"baseline"`, `"delay"`, `"partial"`, `"noniid"`)|
| `attack` | `string` | `"collusion"` | 공격 유형 (`"none"`, `"collusion"`) |
| `defense` | `string` | `"clipped_median"` | 방어·집계 방식 (`"fedavg"`, `"clipped_fedavg"`, `"clipped_median"`) |
| `optimizer` | `string` | `"sgd"` | 로컬 옵티마이저 (`"adam"`, `"sgd"`) |
| `lr` | `float` | `0.01` | 로컬 학습률 (Learning Rate) |
| `local_ep` | `int` | `1` | 라운드당 클라이언트 로컬 에폭 수 |
| `local_bs` | `int` | `32` | 로컬 미니배치 크기 |
| `epochs` | `int` | `40` | 총 글로벌 통신 라운드 수 |
| `clip_norm` | `float` | `2.5` | L2 노름 클리핑 상한선 |
| `attack_scale` | `float` | `-5.0` | 내부자 협력 공격 악성 업데이트 배수 |
| `malicious_ids` | `string` | `"0,1"` | 전체 10명 중 지정된 악성 클라이언트 ID 2개 |
| `delayed_ids` | `string` | `"0,1,2"` | `delay` 환경에서 가상 마감 시각을 초과하는 고정 클라이언트 3개 |
| `virtual_deadline` | `float` | `10.0` | 서버의 라운드별 가상 마감 시간(초) |
| `seed` | `int` | `1` | 데이터 분할·모델 초기화·클라이언트 샘플링 재현용 랜덤 시드 |

## 3. 실행 예시
```powershell
cd ..\src
python client_runner.py --config ..\configs\sample_config.json
```