"""웹 대시보드 Blueprint. 연합학습 REST API 앱(create_app)에 붙여 한 서버·한 포트로 함께 띄운다.

안태현 PR #8 대시보드(MIT)를 바탕으로 이 엔진에 맞게 다시 작성 (라이선스: LICENSE-dashboard).

- 조회형: experiments/ 아래 결과 폴더·비교표·DLG 결과를 읽기만 한다 (/api/runs, /api/comparison, /api/dlg)
- 실행형: /api/start → 서버 상태를 새로 만들고 run_clients.py를 하위 프로세스로 실행, /api/status, /api/stop
모든 응답은 연합학습 API와 같은 {"ok", "data", "error"} 형식이다. Host 검사와 POST의 JSON 강제는 앱의
before_request가 이 Blueprint에도 그대로 적용한다.
"""

from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file

from web.fl_server.dlg import dlg_image_path, dlg_runs
from web.fl_server.files import NotFound, PathRejected
from web.fl_server.live import MAX_ROUNDS, LiveBusy, LiveRunner, parse_start
from web.fl_server.results import (
    comparison,
    comparison_chart,
    discover_runs,
    run_detail,
)

# 페이지와 정적 파일은 같은 출처에서만 불러온다 (외부 CDN 없음, 인라인 스크립트 없음)
PAGE_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
# 결과 SVG를 주소창으로 직접 열어도 그 안의 스크립트는 실행되지 않게
SVG_CSP = "default-src 'none'; style-src 'unsafe-inline'"


def _reply(data=None, error: str | None = None, status: int = 200):
    return jsonify({"ok": error is None, "data": data, "error": error}), status


def create_dashboard(experiments_dir: Path, live: LiveRunner) -> Blueprint:
    bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static",
                   static_url_path="/dashboard-static")

    @bp.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers.setdefault("Content-Security-Policy", PAGE_CSP)
        response.headers["Cache-Control"] = "no-store"
        return response

    @bp.errorhandler(PathRejected)
    def path_rejected(error):
        return _reply(None, str(error), 400)

    @bp.errorhandler(NotFound)
    def not_found(error):
        return _reply(None, str(error.args[0]) if error.args else "없습니다.", 404)

    @bp.get("/")
    def index():
        return render_template("dashboard/index.html", max_rounds=MAX_ROUNDS)

    @bp.get("/api/runs")
    def runs():
        return _reply(discover_runs(experiments_dir))

    @bp.get("/api/runs/<path:run_id>")
    def run(run_id: str):
        return _reply(run_detail(experiments_dir, run_id))

    @bp.get("/api/comparison")
    def comparison_data():
        return _reply(comparison(experiments_dir))

    @bp.get("/api/comparison/accuracy.svg")
    def comparison_svg():
        response = send_file(comparison_chart(experiments_dir), mimetype="image/svg+xml")
        response.headers["Content-Security-Policy"] = SVG_CSP
        return response

    @bp.get("/api/dlg")
    def dlg():
        return _reply(dlg_runs(experiments_dir))

    @bp.get("/api/dlg/<run_id>/images/<filename>")
    def dlg_image(run_id: str, filename: str):
        return send_file(dlg_image_path(experiments_dir, run_id, filename), mimetype="image/png")

    @bp.post("/api/start")
    def start():
        try:
            params = parse_start(request.get_json(silent=True))
        except (TypeError, ValueError) as error:
            return _reply(None, str(error), 400)
        try:
            return _reply(live.start(params))
        except LiveBusy as error:
            return _reply(None, str(error), 409)
        except (ValueError, OSError) as error:   # 설정 파일 오류, 프로세스 시작 실패
            return _reply(None, f"실행을 시작하지 못했습니다: {error}", 500)

    @bp.get("/api/status")
    def status():
        return _reply(live.status())

    @bp.post("/api/stop")
    def stop():
        return _reply(live.stop())

    return bp
