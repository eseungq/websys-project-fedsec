# experiments — 실험 결과 (머지 자리)

11/24 대표 코드 선정 후 머지된 코드의 결과가 쌓이는 자리입니다. 그 전 결과는 각자 `members/<이름>/experiments/`에 있습니다.

실험 1회마다 폴더 하나를 만듭니다.

```
experiments/<실험ID>/
├── config.yaml     # 실제로 쓴 설정 전체 (커밋)
├── env.json        # PC, GPU, 라이브러리 버전, git commit (커밋)
├── summary.csv     # 최종 정확도 등 요약 (커밋)
├── logs/           # 원시 로그 (커밋 안 함)
└── models/         # 모델 상태 (커밋 안 함)
```
