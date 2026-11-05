# experiments — 실험 결과

실험 1회마다 폴더 하나를 만듭니다.

```
experiments/<실험ID>/
├── config.yaml     # 실제로 쓴 설정 전체 (커밋)
├── env.json        # PC, GPU, 라이브러리 버전, git commit (커밋)
├── summary.csv     # 최종 정확도 등 요약 (커밋)
├── logs/           # 원시 로그 (커밋 안 함)
└── models/         # 모델 상태 (커밋 안 함)
```

- 같은 환경·시드의 비교 묶음(정상, 공격, 공격+방어)은 같은 PC에서 실행합니다.
