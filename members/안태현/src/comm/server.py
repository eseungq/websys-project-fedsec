import os
import sys
import io
import copy
import json
import time
import base64
import threading
import subprocess
from datetime import datetime

import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.core.models import CNNCifar, LeNetDLG
from src.core.data_manager import load_cifar10_splits
from src.core.local_trainer import evaluate_model, compute_l2_norm
from src.security.attacks import run_dlg_inversion, evaluate_dlg_mse
from src.security.defenses import apply_defense_and_aggregate
from src.logging_utils.audit_logger import SimulationLogger, compute_comparison_metrics

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static"),
)

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "default_config.json")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
COMPLETED_RUNS_PATH = os.path.join(OUTPUTS_DIR, "logs", "completed_runs.json")

state_lock = threading.Lock()
sim_state = {
    "status": "idle",          # idle | running | completed | stopped | error
    "error_message": "",
    "exp_id": None,
    "config": None,
    "runner_proc": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "global_model": None,
    "model_version": 0,
    "current_round": 0,
    "total_rounds": 10,
    "data_bundle": None,
    "client_cards": [],        # 10개 클라이언트 실시간 UI 상태
    "selected_clients": [],
    "round_submissions": {},   # {client_id: submission_dict}
    "round_start_real_ts": 0.0,
    "virtual_elapsed_sec": 0.0,
    "real_elapsed_sec": 0.0,
    "val_acc_history": [],
    "val_loss_history": [],
    "test_acc": None,
    "test_loss": None,
    "per_class_acc": {},
    "per_client_acc": {},
    "latest_kpis": {
        "selected_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "total_malicious_ratio": "0.0%",
        "accepted_malicious_ratio": "N/A",
    },
    "audit_logs": [],          # 최근 감사 로그 (웹 테이블 표시용)
    "dlg_result": None,        # DLG 복원 결과
    "completed_runs": [],      # 비교표 및 %p 산출용 누적 실험 기록
}

logger = SimulationLogger(base_dir=OUTPUTS_DIR)


def load_completed_runs():
    if os.path.exists(COMPLETED_RUNS_PATH):
        try:
            with open(COMPLETED_RUNS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_completed_runs(runs):
    os.makedirs(os.path.dirname(COMPLETED_RUNS_PATH), exist_ok=True)
    with open(COMPLETED_RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)


sim_state["completed_runs"] = load_completed_runs()


def serialize_tensor_obj(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def deserialize_tensor_obj(b64_str):
    raw = base64.b64decode(b64_str.encode("utf-8"))
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu")


def build_initial_client_cards(cfg):
    malicious_ids = set(cfg.get("malicious_client_ids", []))
    delayed_ids = set(cfg.get("delayed_client_ids", []))
    env = cfg.get("environment", "normal")
    attack = cfg.get("attack", "normal")

    cards = []
    for cid in range(cfg["num_users"]):
        cards.append({
            "client_id": cid,
            "status": "waiting",       # waiting | training | accepted | delayed | unselected
            "is_malicious": (attack == "collusion" and cid in malicious_ids),
            "is_delay_target": (env == "delay" and cid in delayed_ids),
            "sub_text": "대기 중",
            "local_loss": None,
            "virtual_time": None,
        })
    return cards


def compute_dashboard_comparison():
    """현재 선택된 (environment, seed) 기준으로 정상/공격/방어 실험 간 %p 지표 산출"""
    cfg = sim_state["config"]
    runs = sim_state["completed_runs"]
    if not cfg or not runs:
        return {"attack_damage_pp": None, "defense_gain_pp": None, "normal_loss_pp": None, "rows": runs}

    env = cfg["environment"]
    seed = cfg["seed"]
    matched = [r for r in runs if r["environment"] == env and r["seed"] == seed and r["attack"] != "dlg"]

    def latest_acc(atk, dfn):
        candidates = [r["test_acc"] for r in matched if r["attack"] == atk and r["defense"] == dfn and r["test_acc"] is not None]
        return candidates[-1] if candidates else None

    norm_fedavg = latest_acc("normal", "none")
    atk_fedavg = latest_acc("collusion", "none")
    atk_def = latest_acc("collusion", cfg["defense"]) if cfg["defense"] != "none" else None
    norm_def = latest_acc("normal", cfg["defense"]) if cfg["defense"] != "none" else None

    metrics = {}
    if norm_fedavg is not None:
        metrics = compute_comparison_metrics(norm_fedavg, atk_fedavg, atk_def, norm_def)
    else:
        metrics = {"attack_damage_pp": None, "defense_gain_pp": None, "normal_loss_pp": None}

    metrics["rows"] = runs[-20:]
    return metrics


# ==============================================================================
# 웹 페이지 및 정적 파일 라우트
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/outputs/dlg_images/<path:filename>", methods=["GET"])
def serve_dlg_image(filename):
    return send_from_directory(os.path.join(OUTPUTS_DIR, "dlg_images"), filename)


@app.route("/outputs/logs/<path:filename>", methods=["GET"])
def serve_log_file(filename):
    return send_from_directory(os.path.join(OUTPUTS_DIR, "logs"), filename, as_attachment=True)


# ==============================================================================
# 시뮬레이션 제어 및 상태 조회 API
# ==============================================================================

@app.route("/api/start", methods=["POST"])
def start_simulation():
    payload = request.get_json(silent=True) or {}

    with state_lock:
        # 기존 실행 중인 자식 프로세스가 있다면 정리
        old_proc = sim_state.get("runner_proc")
        if old_proc is not None and old_proc.poll() is None:
            old_proc.terminate()

        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["environment"] = payload.get("environment", cfg["environment"])
        cfg["attack"] = payload.get("attack", cfg["attack"])
        cfg["defense"] = payload.get("defense", cfg["defense"])
        cfg["dlg_noise_mode"] = payload.get("dlg_noise_mode", cfg["dlg_noise_mode"])
        cfg["seed"] = int(payload.get("seed", cfg["seed"]))
        cfg["epochs"] = int(payload.get("epochs", cfg["epochs"]))

        seed = cfg["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)

        exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg['environment']}_{cfg['attack']}_{cfg['defense']}_s{seed}"
        label_skew = (cfg["environment"] == "label_skew")

        data_bundle = load_cifar10_splits(
            data_dir=os.path.join(ROOT_DIR, "data"),
            seed=seed,
            label_skew=label_skew,
            num_users=cfg["num_users"],
        )

        device = sim_state["device"]
        if cfg["attack"] == "dlg":
            model = LeNetDLG(num_classes=cfg["num_classes"]).to(device)
        else:
            model = CNNCifar(num_classes=cfg["num_classes"]).to(device)

        logger.init_experiment_logs(exp_id, cfg, data_bundle["distribution_report"])

        sim_state.update({
            "status": "running",
            "error_message": "",
            "exp_id": exp_id,
            "config": cfg,
            "model_version": 0,
            "global_model": model,
            "current_round": 0,
            "total_rounds": 1 if cfg["attack"] == "dlg" else cfg["epochs"],
            "data_bundle": data_bundle,
            "client_cards": build_initial_client_cards(cfg),
            "selected_clients": [],
            "round_submissions": {},
            "round_start_real_ts": time.time(),
            "virtual_elapsed_sec": 0.0,
            "real_elapsed_sec": 0.0,
            "val_acc_history": [],
            "val_loss_history": [],
            "test_acc": None,
            "test_loss": None,
            "per_class_acc": {},
            "per_client_acc": {},
            "latest_kpis": {
                "selected_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "total_malicious_ratio": f"{len(cfg['malicious_client_ids']) / cfg['num_users'] * 100:.1f}%",
                "accepted_malicious_ratio": "N/A",
            },
            "audit_logs": [],
            "dlg_result": None,
        })

        # 자식 프로세스용 런타임 설정 저장 후 subprocess.Popen 구동
        runtime_cfg_path = os.path.join(OUTPUTS_DIR, "checkpoints", exp_id, "runtime_config.json")
        with open(runtime_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        runner_script = os.path.join(ROOT_DIR, "src", "comm", "client_runner.py")
        proc = subprocess.Popen(
            [sys.executable, runner_script, "--config", runtime_cfg_path, "--server", "http://127.0.0.1:5000"],
            cwd=ROOT_DIR,
        )
        sim_state["runner_proc"] = proc

    return jsonify({"status": "started", "exp_id": exp_id})


@app.route("/api/stop", methods=["POST"])
def stop_simulation():
    with state_lock:
        proc = sim_state.get("runner_proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if sim_state["status"] == "running":
            sim_state["status"] = "stopped"
    return jsonify({"status": "stopped"})


@app.route("/api/status", methods=["GET"])
def get_status():
    with state_lock:
        # 자식 프로세스가 예기치 않게 종료되었는지 확인
        proc = sim_state.get("runner_proc")
        if sim_state["status"] == "running" and proc is not None and proc.poll() is not None:
            if proc.returncode != 0:
                sim_state["status"] = "error"
                sim_state["error_message"] = f"클라이언트 실행기 비정상 종료 (code={proc.returncode})"

        return jsonify({
            "status": sim_state["status"],
            "error_message": sim_state["error_message"],
            "exp_id": sim_state["exp_id"],
            "config": sim_state["config"],
            "current_round": sim_state["current_round"],
            "total_rounds": sim_state["total_rounds"],
            "virtual_elapsed_sec": round(sim_state["virtual_elapsed_sec"], 1),
            "real_elapsed_sec": round(sim_state["real_elapsed_sec"], 1),
            "client_cards": sim_state["client_cards"],
            "val_acc_history": sim_state["val_acc_history"],
            "val_loss_history": sim_state["val_loss_history"],
            "test_acc": sim_state["test_acc"],
            "test_loss": sim_state["test_loss"],
            "per_class_acc": sim_state["per_class_acc"],
            "per_client_acc": sim_state["per_client_acc"],
            "latest_kpis": sim_state["latest_kpis"],
            "audit_logs": sim_state["audit_logs"][-50:],
            "dlg_result": sim_state["dlg_result"],
            "comparison": compute_dashboard_comparison(),
        })


# ==============================================================================
# 연합학습 클라이언트 REST API
# ==============================================================================

@app.route("/api/model", methods=["GET"])
def get_global_model():
    with state_lock:
        if sim_state["global_model"] is None:
            return jsonify({"error": "초기화된 모델이 없습니다."}), 400

        weights_b64 = serialize_tensor_obj(sim_state["global_model"].state_dict())
        return jsonify({
            "exp_id": sim_state["exp_id"],
            "round": sim_state["current_round"],
            "model_version": sim_state["model_version"],
            "status": sim_state["status"],
            "weights": weights_b64,
        })


@app.route("/api/round_plan", methods=["POST"])
def register_round_plan():
    """라운드 시작 시 참여 선정된 클라이언트 목록을 받아 UI 10개 카드 상태 초기화"""
    data = request.get_json()
    selected = set(int(x) for x in data["selected_clients"])

    with state_lock:
        cfg = sim_state["config"]
        sim_state["selected_clients"] = sorted(list(selected))
        sim_state["round_submissions"] = {}
        sim_state["round_start_real_ts"] = time.time()

        cards = build_initial_client_cards(cfg)
        for c in cards:
            if cfg["attack"] == "dlg":
                if c["client_id"] == cfg["dlg_target_client_id"]:
                    c["status"] = "waiting"
                    c["sub_text"] = "DLG 복원 대상"
                else:
                    c["status"] = "unselected"
                    c["sub_text"] = "비활성 (단일 실험)"
            else:
                if c["client_id"] in selected:
                    c["status"] = "waiting"
                    c["sub_text"] = "대기 중"
                else:
                    c["status"] = "unselected"
                    c["sub_text"] = "미참여 (미선택)"
        sim_state["client_cards"] = cards

    return jsonify({"status": "ok"})


@app.route("/api/client_start", methods=["POST"])
def notify_client_start():
    """특정 클라이언트가 학습을 시작하는 시점에 호출되어 해당 카드만 실시간 활성화"""
    data = request.get_json()
    cid = int(data["client_id"])
    mode = data.get("mode", "training")

    with state_lock:
        for c in sim_state["client_cards"]:
            if c["client_id"] == cid:
                c["status"] = "training"
                c["sub_text"] = "그래디언트 복원 중..." if mode == "dlg" else "학습 중..."
                break
    return jsonify({"status": "ok"})


@app.route("/api/submit_update", methods=["POST"])
def submit_client_update():
    data = request.get_json()
    cid = int(data["client_id"])
    sub_round = int(data["round"])
    sub_version = int(data["model_version"])
    num_samples = int(data["num_samples"])
    pre_norm = float(data["pre_attack_norm"])
    post_norm = float(data["post_attack_norm"])
    virtual_time = float(data["virtual_time_sec"])
    real_time = float(data["real_time_sec"])
    local_loss = float(data["local_loss"])
    delta_w = deserialize_tensor_obj(data["delta_w"])

    with state_lock:
        cfg = sim_state["config"]
        cur_round = sim_state["current_round"]
        cur_ver = sim_state["model_version"]
        deadline = float(cfg["virtual_deadline_sec"])

        # 검증 단계: 라운드, 모델 버전, 클라이언트 ID, 중복 제출, 가상 마감 시각
        decision = "채택"
        reason = "정상 검증 통과"

        if cid < 0 or cid >= cfg["num_users"]:
            decision = "제외"
            reason = f"유효하지 않은 클라이언트 ID ({cid})"
        elif sub_round != cur_round:
            decision = "제외"
            reason = f"라운드 불일치 (제출={sub_round}, 서버={cur_round})"
        elif sub_version != cur_ver:
            decision = "제외"
            reason = f"모델 버전 불일치 (제출={sub_version}, 서버={cur_ver})"
        elif cid in sim_state["round_submissions"]:
            decision = "제외"
            reason = "중복 제출"
        elif virtual_time > deadline:
            decision = "제외"
            reason = f"가상 마감 시각 초과 ({virtual_time:.1f}s > {deadline:.1f}s)"

        sim_state["round_submissions"][cid] = {
            "client_id": cid,
            "num_samples": num_samples,
            "delta_w": delta_w,
            "pre_attack_norm": pre_norm,
            "post_attack_norm": post_norm,
            "virtual_time_sec": virtual_time,
            "real_time_sec": real_time,
            "local_loss": local_loss,
            "decision": decision,
            "reason": reason,
        }

        # UI 카드 상태 즉시 갱신
        for c in sim_state["client_cards"]:
            if c["client_id"] == cid:
                c["local_loss"] = round(local_loss, 4)
                c["virtual_time"] = round(virtual_time, 1)
                if decision == "채택":
                    c["status"] = "accepted"
                    c["sub_text"] = f"채택 완료 (Loss {local_loss:.2f})"
                else:
                    c["status"] = "delayed"
                    c["sub_text"] = f"지연 제외 ({virtual_time:.1f}s)"
                break

        # 선정된 모든 클라이언트가 제출을 마쳤으면 라운드 집계 수행
        expected_total = len(sim_state["selected_clients"])
        if len(sim_state["round_submissions"]) >= expected_total:
            _finalize_round()

    return jsonify({"status": "ok"})


def _finalize_round():
    """라운드 마감 후 방어 적용·집계·검증 평가·체크포인트 및 로그 기록"""
    cfg = sim_state["config"]
    exp_id = sim_state["exp_id"]
    cur_round = sim_state["current_round"]
    device = sim_state["device"]
    global_model = sim_state["global_model"]
    malicious_ids = set(cfg["malicious_client_ids"]) if cfg["attack"] == "collusion" else set()

    start_global_state = copy.deepcopy(global_model.state_dict())
    submissions = sim_state["round_submissions"]

    accepted_updates = {}
    sample_counts = {}
    for cid, sub in submissions.items():
        if sub["decision"] == "채택":
            accepted_updates[cid] = sub["delta_w"]
            sample_counts[cid] = sub["num_samples"]

    selected_count = len(sim_state["selected_clients"])
    accepted_count = len(accepted_updates)
    rejected_count = selected_count - accepted_count

    total_mal_ratio = f"{len(malicious_ids) / cfg['num_users'] * 100:.1f}%"
    if accepted_count == 0:
        accepted_mal_ratio = "N/A"
    else:
        accepted_mal_cnt = sum(1 for cid in accepted_updates.keys() if cid in malicious_ids)
        accepted_mal_ratio = f"{accepted_mal_cnt / accepted_count * 100:.1f}%"

    # 채택 업데이트가 3개 미만이면 모델을 갱신하지 않고 사유 기록
    min_req = int(cfg["min_accepted_clients"])
    if accepted_count < min_req:
        model_updated = False
        update_reason = f"채택 수 3개 미만({accepted_count}개)으로 모델 갱신 보류"
        post_norms = {cid: None for cid in submissions.keys()}
    else:
        model_updated = True
        update_reason = f"정상 집계 완료 (방어: {cfg['defense']})"
        # 방어 함수에는 악성 클라이언트 ID 정답을 전달하지 않음
        agg_delta, post_norms = apply_defense_and_aggregate(
            accepted_updates=accepted_updates,
            sample_counts=sample_counts,
            defense_mode=cfg["defense"],
            clip_bound=float(cfg["clip_bound"]),
        )
        new_state = {}
        for k, v in start_global_state.items():
            if v.is_floating_point():
                new_state[k] = v.cpu() + agg_delta[k].cpu()
            else:
                new_state[k] = v.cpu()
        global_model.load_state_dict({k: v.to(device) for k, v in new_state.items()})
        sim_state["model_version"] += 1

    # 검증 데이터셋 평가
    val_acc, val_loss = evaluate_model(global_model, sim_state["data_bundle"]["val_dataset"], device)
    sim_state["val_acc_history"].append(round(val_acc * 100, 2))
    sim_state["val_loss_history"].append(round(val_loss, 4))

    # 가상 시간 및 실제 경과 시간 갱신
    round_virtual_sec = max((s["virtual_time_sec"] for s in submissions.values()), default=0.0)
    round_real_sec = time.time() - sim_state["round_start_real_ts"]
    sim_state["virtual_elapsed_sec"] += round_virtual_sec
    sim_state["real_elapsed_sec"] += round_real_sec

    # 클라이언트별 감사 로그 생성 및 CSV 기록
    client_rows = []
    for cid in range(cfg["num_users"]):
        is_sel = cid in submissions
        is_mal = cid in malicious_ids
        if is_sel:
            sub = submissions[cid]
            row = {
                "round": cur_round + 1,
                "client_id": cid,
                "environment": cfg["environment"],
                "attack": cfg["attack"],
                "defense": cfg["defense"],
                "num_samples": sub["num_samples"],
                "is_selected": True,
                "is_malicious_role": is_mal,
                "pre_attack_norm": sub["pre_attack_norm"],
                "post_attack_norm": sub["post_attack_norm"],
                "post_defense_norm": post_norms.get(cid),
                "decision": sub["decision"],
                "reason": sub["reason"] if model_updated else f"{sub['reason']} / {update_reason}",
                "virtual_time_sec": sub["virtual_time_sec"],
                "real_time_sec": sub["real_time_sec"],
                "local_loss": sub["local_loss"],
            }
        else:
            row = {
                "round": cur_round + 1,
                "client_id": cid,
                "environment": cfg["environment"],
                "attack": cfg["attack"],
                "defense": cfg["defense"],
                "num_samples": len(sim_state["data_bundle"]["user_groups"][cid]),
                "is_selected": False,
                "is_malicious_role": is_mal,
                "pre_attack_norm": None,
                "post_attack_norm": None,
                "post_defense_norm": None,
                "decision": "미선택",
                "reason": "부분 참여 미선정",
                "virtual_time_sec": 0.0,
                "real_time_sec": 0.0,
                "local_loss": None,
            }
        client_rows.append(row)
        sim_state["audit_logs"].append(row)

    logger.log_client_rows(exp_id, client_rows)

    summary_row = {
        "round": cur_round + 1,
        "selected_count": selected_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_malicious_ratio": total_mal_ratio,
        "accepted_malicious_ratio": accepted_mal_ratio,
        "model_updated": model_updated,
        "update_reason": update_reason,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "virtual_elapsed_sec": sim_state["virtual_elapsed_sec"],
        "real_elapsed_sec": sim_state["real_elapsed_sec"],
    }
    logger.log_round_summary(exp_id, summary_row)

    # 특정 라운드 재실행을 위한 체크포인트 저장
    logger.save_round_checkpoint(exp_id, cur_round + 1, {
        "round": cur_round + 1,
        "config": cfg,
        "start_global_state": start_global_state,
        "submissions": submissions,
    })

    sim_state["latest_kpis"] = {
        "selected_count": selected_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_malicious_ratio": total_mal_ratio,
        "accepted_malicious_ratio": accepted_mal_ratio,
    }

    sim_state["current_round"] += 1

    # 최종 라운드 완료 시 테스트셋 및 클라이언트별/클래스별 최종 성능 산출
    if sim_state["current_round"] >= sim_state["total_rounds"]:
        test_acc, test_loss, per_class_acc = evaluate_model(
            global_model, sim_state["data_bundle"]["test_dataset"], device, per_class=True
        )
        sim_state["test_acc"] = round(test_acc * 100, 2)
        sim_state["test_loss"] = round(test_loss, 4)
        sim_state["per_class_acc"] = {str(k): round(v * 100, 1) for k, v in per_class_acc.items()}

        from torch.utils.data import Subset
        client_accs = {}
        for cid in range(cfg["num_users"]):
            c_ds = Subset(sim_state["data_bundle"]["train_dataset"], sim_state["data_bundle"]["user_groups"][cid])
            c_acc, _ = evaluate_model(global_model, c_ds, device)
            client_accs[str(cid)] = round(c_acc * 100, 1)
        sim_state["per_client_acc"] = client_accs
        sim_state["status"] = "completed"

        run_record = {
            "exp_id": exp_id,
            "environment": cfg["environment"],
            "attack": cfg["attack"],
            "defense": cfg["defense"],
            "seed": cfg["seed"],
            "rounds": cfg["epochs"],
            "test_acc": test_acc,
            "final_val_acc": val_acc,
            "accepted_malicious_ratio": accepted_mal_ratio,
        }
        sim_state["completed_runs"].append(run_record)
        save_completed_runs(sim_state["completed_runs"])


# ==============================================================================
# 모델 역추적(DLG) 전용 API
# ==============================================================================

@app.route("/api/dlg_submit", methods=["POST"])
def submit_dlg_gradients():
    data = request.get_json()
    cid = int(data["client_id"])
    sample_index = int(data["sample_index"])
    noise_mode = data["noise_mode"]
    noise_std = float(data["noise_std"])
    observed_grads = deserialize_tensor_obj(data["gradients"])

    with state_lock:
        cfg = sim_state["config"]
        exp_id = sim_state["exp_id"]
        device = sim_state["device"]
        model = sim_state["global_model"]

        # 공격 함수에는 원본 이미지나 정답 레이블을 절대 입력하지 않음
        rec_tensor, inferred_label, loss_hist = run_dlg_inversion(
            model=model,
            observed_gradients=observed_grads,
            num_classes=cfg["num_classes"],
            iterations=int(cfg["dlg_iterations"]),
            seed=cfg["seed"],
            device=device,
        )

        # 복원 완료 후 평가 단계에서만 원본 이미지와 비교해 MSE 계산
        raw_ds = sim_state["data_bundle"]["train_raw_dataset"]
        target_global_idx = sim_state["data_bundle"]["user_groups"][cid][sample_index]
        gt_tensor, gt_label = raw_ds[target_global_idx]
        mse = evaluate_dlg_mse(gt_tensor, rec_tensor)

        gt_path, rec_path = logger.save_dlg_result(
            run_id=exp_id,
            sample_index=target_global_idx,
            noise_mode=noise_mode,
            noise_std=noise_std,
            iterations=int(cfg["dlg_iterations"]),
            mse=mse,
            gt_tensor=gt_tensor,
            rec_tensor=rec_tensor,
        )

        for c in sim_state["client_cards"]:
            if c["client_id"] == cid:
                c["status"] = "accepted"
                c["sub_text"] = f"복원 완료 (MSE: {mse:.4f})"

        sim_state["current_round"] = 1
        sim_state["status"] = "completed"
        sim_state["dlg_result"] = {
            "exp_id": exp_id,
            "client_id": cid,
            "sample_index": int(target_global_idx),
            "gt_label": int(gt_label),
            "inferred_label": inferred_label,
            "noise_mode": noise_mode,
            "noise_std": noise_std,
            "iterations": int(cfg["dlg_iterations"]),
            "mse": round(mse, 6),
            "gt_url": f"/outputs/dlg_images/{os.path.basename(gt_path)}",
            "rec_url": f"/outputs/dlg_images/{os.path.basename(rec_path)}",
            "grad_loss_history": [round(x, 6) for x in loss_hist[::10]],
        }

    return jsonify({"status": "ok", "mse": mse})


# ==============================================================================
# 특정 라운드 재실행(원인 분석용 Re-run) API
# ==============================================================================

@app.route("/api/rerun_round", methods=["POST"])
def rerun_round():
    """
    저장된 라운드 체크포인트를 불러와 동일 시작 모델·참여 조건에서
    방어 방식을 바꾸거나 특정 클라이언트의 악성 조작을 해제해 결과 변화 확인.
    """
    data = request.get_json()
    exp_id = data.get("exp_id") or sim_state["exp_id"]
    round_num = int(data.get("round", 1))
    new_defense = data.get("defense", "median")
    unmask_cids = set(int(x) for x in data.get("unmask_client_ids", []))

    if not exp_id:
        return jsonify({"error": "재실행할 실험 기록이 없습니다."}), 400

    ckpt = logger.load_round_checkpoint(exp_id, round_num)
    cfg = ckpt["config"]
    start_state = ckpt["start_global_state"]
    submissions = ckpt["submissions"]
    device = sim_state["device"]

    # 기존 모델 평가
    orig_model = CNNCifar(num_classes=cfg["num_classes"]).to(device)
    orig_model.load_state_dict({k: v.to(device) for k, v in start_state.items()})
    base_val_acc, _ = evaluate_model(orig_model, sim_state["data_bundle"]["val_dataset"], device)

    # 조작 해제 옵션 반영 후 채택 업데이트 수집
    accepted_updates = {}
    sample_counts = {}
    for cid, sub in submissions.items():
        if sub["decision"] == "채택":
            if cid in unmask_cids and "honest_delta_w" in sub:
                accepted_updates[cid] = sub["honest_delta_w"]
            else:
                accepted_updates[cid] = sub["delta_w"]
            sample_counts[cid] = sub["num_samples"]

    if len(accepted_updates) < int(cfg["min_accepted_clients"]):
        return jsonify({"error": "채택 클라이언트 수가 3개 미만이라 재집계할 수 없습니다."}), 400

    agg_delta, _ = apply_defense_and_aggregate(
        accepted_updates=accepted_updates,
        sample_counts=sample_counts,
        defense_mode=new_defense,
        clip_bound=float(cfg["clip_bound"]),
    )

    rerun_model = CNNCifar(num_classes=cfg["num_classes"]).to(device)
    new_state = {
        k: (v.cpu() + agg_delta[k].cpu() if v.is_floating_point() else v.cpu())
        for k, v in start_state.items()
    }
    rerun_model.load_state_dict({k: v.to(device) for k, v in new_state.items()})
    rerun_val_acc, rerun_val_loss = evaluate_model(rerun_model, sim_state["data_bundle"]["val_dataset"], device)

    return jsonify({
        "exp_id": exp_id,
        "round": round_num,
        "new_defense": new_defense,
        "unmasked_clients": sorted(list(unmask_cids)),
        "before_round_val_acc": round(base_val_acc * 100, 2),
        "after_rerun_val_acc": round(rerun_val_acc * 100, 2),
        "diff_pp": round((rerun_val_acc - base_val_acc) * 100, 2),
        "rerun_val_loss": round(rerun_val_loss, 4),
    })


if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)