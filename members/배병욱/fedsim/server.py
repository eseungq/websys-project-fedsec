"""중앙 Flask 서버.

API
- POST /reset      실험 시작: 초기 전역 가중치(본문)와 집계·참여 설정(X-Meta 헤더)
- GET  /status     현재 라운드·모델 버전·제출 현황
- GET  /model      전역 가중치(본문) + X-Round, X-Version, X-Participants, X-Deadline 헤더
- POST /update     업데이트 제출(본문) + X-Meta {client_id, round, version, num_samples, virtual_finish}
- POST /aggregate  이번 라운드 채택분 집계 → 전역 모델 갱신 여부와 사유

서버는 악성 ID를 모른다. 참여자 선택과 마감 판정에만 일정을 쓴다.
"""
import argparse
import json
import math
import time

import torch
from flask import Flask, Response, jsonify, request

from . import aggregation
from .environment import Schedule
from .serialization import from_bytes, to_bytes

# 제외 사유 코드
REASONS = {
    "bad_meta": "X-Meta 헤더가 없거나 JSON이 아님",
    "unknown_client": "등록되지 않은 클라이언트 ID",
    "wrong_round": "현재 라운드와 다른 라운드 번호",
    "stale_version": "현재 전역 모델 버전과 다름",
    "not_selected": "이번 라운드 참여자가 아님",
    "duplicate": "이번 라운드에 이미 제출함",
    "bad_num_samples": "표본 수가 양의 정수가 아님",
    "bad_payload": "본문을 텐서로 읽을 수 없음",
    "bad_shape": "업데이트 길이가 전역 모델과 다름",
    "non_finite": "NaN 또는 Inf 포함",
    "bad_virtual_finish": "virtual_finish가 없거나 유한한 숫자가 아님",
    "late": "가상 마감 시각 초과",
}


def create_app():
    app = Flask(__name__)
    S = {"ready": False}

    def _fresh_round():
        S["accepted"] = {}   # cid -> (vec, num_samples)
        S["seen"] = set()    # 이번 라운드에 신원 검사를 통과해 제출한 cid
        S["rejected"] = []   # (cid, reason)

    @app.post("/reset")
    def reset():
        meta = json.loads(request.headers["X-Meta"])
        vec = from_bytes(request.get_data()).float()
        S.clear()
        S.update(ready=True, global_vec=vec, version=0, round=meta.get("start_round", 1),
                 method=meta["aggregation"], clip_bound=meta.get("clip_bound"),
                 min_accept=meta.get("min_accept", 3), num_clients=meta["num_clients"],
                 schedule=Schedule(**meta["schedule"]), events=[])
        _fresh_round()
        return jsonify(ok=True, dim=vec.numel(), round=S["round"])

    @app.get("/status")
    def status():
        if not S.get("ready"):
            return jsonify(ready=False)
        return jsonify(ready=True, round=S["round"], version=S["version"],
                       accepted=sorted(S["accepted"]), rejected=S["rejected"])

    @app.get("/model")
    def model():
        sch = S["schedule"]
        headers = {"X-Round": str(S["round"]), "X-Version": str(S["version"]),
                   "X-Participants": json.dumps(sch.participants(S["round"])),
                   "X-Deadline": str(sch.deadline)}
        return Response(to_bytes(S["global_vec"]), mimetype="application/octet-stream", headers=headers)

    def _validate(meta, raw):
        cid = meta.get("client_id")
        if type(cid) is not int or not 0 <= cid < S["num_clients"]:  # bool 거부
            return "unknown_client", None
        if meta.get("round") != S["round"]:
            return "wrong_round", None
        if meta.get("version") != S["version"]:
            return "stale_version", None
        if cid not in S["schedule"].participants(S["round"]):
            return "not_selected", None
        if cid in S["seen"]:
            return "duplicate", None
        S["seen"].add(cid)
        ns = meta.get("num_samples")
        if type(ns) is not int or ns <= 0:
            return "bad_num_samples", None
        try:
            vec = from_bytes(raw).float()
        except Exception:
            return "bad_payload", None
        if vec.shape != S["global_vec"].shape:
            return "bad_shape", None
        if not torch.isfinite(vec).all():
            return "non_finite", None
        vf = meta.get("virtual_finish")
        if type(vf) not in (int, float) or not math.isfinite(vf):
            return "bad_virtual_finish", None
        if vf > S["schedule"].deadline:
            return "late", vec
        return None, vec

    @app.post("/update")
    def update():
        try:
            meta = json.loads(request.headers.get("X-Meta", ""))
        except ValueError:
            meta = None
        if not isinstance(meta, dict):
            S["rejected"].append((None, "bad_meta"))
            return jsonify(accepted=False, reason="bad_meta"), 400
        reason, vec = _validate(meta, request.get_data())
        cid = meta.get("client_id")
        norm = vec.norm().item() if vec is not None else None
        if reason:
            S["rejected"].append((cid, reason))
            S["events"].append({"round": S["round"], "client_id": cid, "accepted": False,
                                "reason": reason, "norm": norm, "time": time.time()})
            return jsonify(accepted=False, reason=reason, norm=norm), 409
        S["accepted"][cid] = (vec, meta["num_samples"])
        S["events"].append({"round": S["round"], "client_id": cid, "accepted": True,
                            "reason": "", "norm": norm, "time": time.time()})
        return jsonify(accepted=True, reason="", norm=norm)

    @app.post("/aggregate")
    def aggregate_round():
        rnd, acc = S["round"], S["accepted"]
        cids = sorted(acc)
        info = {"round": rnd, "version_before": S["version"],
                "selected": S["schedule"].participants(rnd), "accepted": cids,
                "rejected": S["rejected"], "updated": False, "reason": "", "after_defense_norm": {}}
        if len(cids) < S["min_accept"]:
            info["reason"] = f"too_few_accepted(<{S['min_accept']})"
        else:
            ups = [acc[c][0] for c in cids]
            ns = [acc[c][1] for c in cids]
            agg, after = aggregation.aggregate(S["method"], ups, ns, S["clip_bound"])
            new = S["global_vec"] + agg
            info["after_defense_norm"] = dict(zip(cids, after))
            info["aggregate_norm"] = agg.norm().item()
            # 원인 분석용 진단: 각 업데이트가 '다수의 방향'(좌표별 중앙값)과 얼마나 다른지,
            # 클리핑이 실제로 잘랐는지. 악성 ID 없이 서버가 볼 수 있는 값만 쓴다.
            ref = aggregation.coordinate_median(ups)
            info["cosine_to_median"] = {
                c: torch.nn.functional.cosine_similarity(u, ref, dim=0).item() for c, u in zip(cids, ups)}
            info["clipped"] = {c: bool(S["method"] == "clip_fedavg" and u.norm().item() > S["clip_bound"])
                               for c, u in zip(cids, ups)}
            if not torch.isfinite(new).all():
                info["reason"] = "non_finite_aggregate"
            else:
                S["global_vec"] = new
                S["version"] += 1
                info["updated"] = True
        info["version_after"] = S["version"]
        S["round"] += 1
        _fresh_round()
        return jsonify(info)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5055)
    args = ap.parse_args()
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    create_app().run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
