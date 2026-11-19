"""Flask REST API. 상태는 ServerState 한 객체에 두고, 앱은 create_app(state) 팩토리로 만든다.

모든 응답은 {"ok": bool, "data": ..., "error": str | null} 형식이다.
제외된 제출은 4xx와 함께 data.reason에 사유 코드(validation.REASON_STATUS)를 담는다.
"""

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from fedsec.server.state import ServerState
from fedsec.server.validation import Rejection

# 이 PC 안에서만 접속을 받는다. macOS는 5000번(AirPlay 수신)을 쓰므로 다른 포트를 기본으로 한다.
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# Host 헤더가 이 이름이 아니면 거부한다 (같은 PC의 브라우저를 통한 DNS rebinding 차단)
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})

# JSON 본문 상한: Base64는 원본의 4/3배이고, 나머지 항목을 위한 여유를 더한다
REQUEST_OVERHEAD_BYTES = 64 * 1024


def _reply(data=None, error: str | None = None, status: int = 200):
    return jsonify({"ok": error is None, "data": data, "error": error}), status


def _rejection_reply(rejection: Rejection):
    return _reply({"reason": rejection.reason}, rejection.detail, rejection.status)


def create_app(state: ServerState) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False  # 한국어 사유 메시지를 그대로 보이게
    app.config["MAX_CONTENT_LENGTH"] = state.max_update_bytes * 4 // 3 + REQUEST_OVERHEAD_BYTES

    @app.before_request
    def local_only():
        host = request.host.rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS:
            return _reply(None, f"허용되지 않은 Host입니다: {request.host[:100]}", 403)
        # POST는 JSON만 받는다. 브라우저가 preflight 없이 보내는 단순 요청(form, text/plain)을 막는다
        if request.method == "POST" and not request.is_json:
            if request.path == "/update":
                return _rejection_reply(state.reject_bad_json("Content-Type이 application/json이 아닙니다."))
            return _reply(None, "Content-Type은 application/json이어야 합니다.", 415)
        return None

    @app.get("/status")
    def status():
        return _reply(state.status())

    @app.get("/model")
    def model():
        return _reply(state.model_snapshot())

    @app.post("/update")
    def update():
        payload = request.get_json(silent=True)
        outcome = state.submit(payload)
        if isinstance(outcome, Rejection):
            return _rejection_reply(outcome)
        return _reply({"client_id": outcome.client_id, "round": payload["round"], "accepted": True})

    @app.post("/aggregate")
    def aggregate():
        return _reply(state.aggregate())

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return _rejection_reply(state.reject_oversized(f"요청 본문이 상한 {app.config['MAX_CONTENT_LENGTH']} 바이트를 넘습니다."))

    @app.errorhandler(HTTPException)
    def http_error(error):
        return _reply(None, f"{error.code} {error.name}", error.code)

    @app.errorhandler(Exception)
    def server_error(error):
        app.logger.exception("요청 처리 중 오류")
        return _reply(None, f"서버 오류: {type(error).__name__}: {error}", 500)

    return app
