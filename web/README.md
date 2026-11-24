# web — 웹 시스템

| 폴더 | 역할 |
|---|---|
| `fl_server/` | 웹 대시보드 Blueprint (조회형: `experiments/` 결과·비교표·DLG 보기, 실행형: 짧은 시연 실행). `python scripts/serve_web.py`로 연합학습 REST API와 한 서버·한 포트에 함께 띄웁니다. |

- 모든 통신은 같은 PC의 localhost에서 이뤄집니다.
- 대시보드 화면은 안태현 PR #8 대시보드(MIT, `fl_server/LICENSE-dashboard`)를 바탕으로 이 엔진에 맞게 다시 작성했습니다.
