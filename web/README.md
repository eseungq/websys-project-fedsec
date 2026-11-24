# web — Flask 웹 서버

`python web/app.py`로 실행하면 http://127.0.0.1:8765/ 에서 대시보드와 연합학습 REST API가 한 서버·한 포트로 함께 뜹니다.

| 파일 | 역할 |
|---|---|
| `app.py` | 서버 실행. 연합학습 API 앱(`src/fedsec/server/app.py`)에 대시보드(`routes.py`)를 붙여 띄움 |
| `routes.py` | 대시보드 주소: `/` 화면, `/api/runs`·`/api/comparison`·`/api/dlg` (조회형), `/api/start`·`/api/status`·`/api/stop` (실행형) |
| `results.py` | `experiments/` 실험 결과·비교표 읽기 |
| `dlg.py` | `experiments/3_dlg/` DLG 결과 읽기 |
| `run.py` | 실행형: 서버 상태를 새로 만들고 가상 클라이언트 프로그램을 하위 프로세스로 실행 |
| `files.py` | 파일 읽기 도우미 (`experiments/` 밖 경로 거부) |
| `templates/index.html` | 대시보드 화면 |
| `static/` | `style.css`, `app.js`(화면 동작), `views.js`·`charts.js`(그리기), `dom.js`(DOM 도우미) |

- 모든 통신은 같은 PC의 localhost에서 이뤄집니다.
- 대시보드 화면은 안태현 PR #8 대시보드(MIT, `LICENSE-dashboard`)를 바탕으로 이 엔진에 맞게 다시 작성했습니다.
