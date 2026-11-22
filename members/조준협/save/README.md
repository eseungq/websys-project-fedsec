# 📊 `save/` 실험 산출물 및 로그 저장소 안내

시뮬레이션 실행 중 생성되는 모든 로그, 재실행용 모델 상태, 평가표, 시각화 그래프는 아래 하위 디렉터리에 체계적으로 저장됩니다.

```text
save/
├── logs/          # 실험별 전체 라운드·클라이언트·제외사유·클래스별 성능 통합 로그 (log_<실험ID>.json)
├── checkpoints/   # 원인 분석(6단계) 재실행을 위한 라운드별 전역 모델 가중치 (round_<N>_start.pt)
├── splits/        # 시드 및 분포별 클라이언트 데이터 분할 인덱스 (split_iid_s1.json 등)
├── dlg/           # DLG 역추적 원본·복원 비교 이미지(dlg_comparison_idx*.png) 및 MSE JSON
├── tables/        # evaluator.py가 생성한 공격 피해·방어 개선·정상 손실(%p) CSV 요약표
└── plots/         # plot_results.py가 생성한 라운드 정확도 곡선 및 비교 막대 그래프(.png)
```

## 재실행(Re-run) 활용 방법
1. `logs/log_<실험ID>.json`에서 분석하고자 하는 특정 라운드와 클라이언트 기록을 확인합니다.
2. 웹 대시보드 **3번 탭(`정량 평가표 · 체크포인트 재실행`)**에 `checkpoints/<실험ID>/round_<N>_start.pt` 경로와 `splits/split_*.json` 경로를 입력한 뒤, 공격 또는 방어 조건만 변경하여 해당 라운드를 재실행합니다.