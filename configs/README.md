# configs — 실험 설정

조합해서 하나의 실험을 만듭니다. 설정 파일만 바꿔서 다른 실험에 적용할 수 있어야 합니다.

| 폴더 | 예시 파일 |
|---|---|
| `base.yaml` | 모델, 라운드 수, 학습률, 시드 목록 등 공통 설정 |
| `env/` | `baseline.yaml`, `delay.yaml`, `partial.yaml`, `label_skew.yaml` |
| `attack/` | `none.yaml`, `collusion.yaml` |
| `defense/` | `fedavg.yaml`, `clip.yaml`, `median.yaml` |
| `dlg/` | 노이즈 수준별 복원 실험 설정 |
