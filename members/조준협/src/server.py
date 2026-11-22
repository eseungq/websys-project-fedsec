import os
import sys
import json
import copy
import base64
import argparse
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import CNNCifar, SimpleCNN
from utils import get_dataset
from serialization import (
    serialize_weights,
    deserialize_weights,
    validate_weight_tensor,
)
from security import aggregate_robust

app = Flask(__name__)

CIFAR10_CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class FederatedServerState:
    """중앙 파라미터 서버 상태, 강건 집계, 체크포인트 저장 및 통합 로그 관리"""

    def __init__(self, args):
        self.args = args
        self.device = "cuda" if (args.gpu is not None and torch.cuda.is_available()) else "cpu"
        self.num_clients = args.num_clients
        self.current_round = 1
        self.model_version = "v1.0-round1"
        self.exp_id = "default_exp"

        self.exp_config = {
            "model": "cnn",
            "env": "baseline",
            "attack": "none",
            "defense": "fedavg",
            "optimizer": "adam",
            "lr": 0.001,
            "local_ep": 3,
            "local_bs": 64,
            "clip_norm": 3.0,
            "seed": 1,
            "virtual_deadline": 10.0,
        }

        self.accepted_updates = {}
        self.round_history = []
        self.excluded_logs = []
        self.client_logs = []
        self.class_metrics = []
        self.active_process = None

        _, self.test_dataset, _ = get_dataset(args)
        self.testloader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

        torch.manual_seed(args.seed)
        self.global_model = (
            SimpleCNN(args=args).to(self.device)
            if getattr(args, "model", "cnn") == "simple_cnn"
            else CNNCifar(args=args).to(self.device)
        )
        self.reference_keys = list(self.global_model.state_dict().keys())

    def reset_experiment(self, config: dict):
        """새 실험 조건에 맞춰 서버 초기화 (체크포인트 재실행 시 해당 모델 가중치 로드)"""
        self.exp_config.update(config)
        seed = int(self.exp_config.get("seed", 1))
        torch.manual_seed(seed)

        model_name = self.exp_config.get("model", "cnn")
        self.global_model = (
            SimpleCNN(args=self.args).to(self.device)
            if model_name == "simple_cnn"
            else CNNCifar(args=self.args).to(self.device)
        )
        self.reference_keys = list(self.global_model.state_dict().keys())

        ckpt_path = config.get("checkpoint_path", "")
        start_round = int(config.get("start_round", 1))

        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            self.global_model.load_state_dict(state)

        self.current_round = start_round
        self.model_version = f"v1.0-round{self.current_round}"
        self.exp_id = (
            f"{model_name}_{self.exp_config['env']}_{self.exp_config['attack']}_"
            f"{self.exp_config['defense']}_{self.exp_config['optimizer']}_s{seed}"
        )
        self.accepted_updates = {}
        self.round_history = []
        self.excluded_logs = []
        self.client_logs = []
        self.class_metrics = []

        # 초기(시작) 라운드 모델 상태 저장
        ckpt_dir = os.path.join("../save/checkpoints", self.exp_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.global_model.state_dict(), os.path.join(ckpt_dir, f"round_{self.current_round}_start.pt"))

    def evaluate_global_model(self, detailed_classes=False):
        """전역 모델 테스트 정확도(%)·손실 및 선택적 클래스별(10종) 성능 계산"""
        self.global_model.eval()
        criterion = nn.NLLLoss().to(self.device)
        loss_sum, correct, total = 0.0, 0, 0

        num_classes = 10
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss_sum += criterion(outputs, labels).item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if detailed_classes:
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        ti, pi = int(t.item()), int(p.item())
                        class_total[ti] += 1
                        if ti == pi:
                            tp[ti] += 1
                        else:
                            fp[pi] += 1
                            fn[ti] += 1

        if detailed_classes:
            metrics = []
            for c in range(num_classes):
                prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
                rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                metrics.append({
                    "class_id": c,
                    "class_name": CIFAR10_CLASSES[c],
                    "accuracy": round(rec * 100.0, 2),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "support": class_total[c],
                })
            self.class_metrics = metrics

        return round(100.0 * correct / total, 2), round(loss_sum / total, 4)

    def log_exclusion(self, client_id, submitted_round, submitted_version, reason, virtual_time=0.0):
        self.excluded_logs.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "server_round": self.current_round,
            "client_id": client_id,
            "submitted_round": submitted_round,
            "submitted_version": submitted_version,
            "virtual_time": virtual_time,
            "reason": reason,
        })

    def aggregate_round(self, round_meta: dict):
        accepted_list = list(self.accepted_updates.values())
        accepted_count = len(accepted_list)
        selected_count = round_meta.get("selected_count", accepted_count)
        delayed_count = round_meta.get("delayed_count", 0)
        total_malicious_ratio = round_meta.get("total_malicious_ratio", "0.0%")
        virtual_elapsed = round_meta.get("virtual_elapsed", 0.0)
        actual_elapsed = round_meta.get("actual_elapsed", 0.0)
        is_final_round = bool(round_meta.get("is_final_round", False))

        if accepted_count == 0:
            accepted_malicious_ratio = "N/A"
        else:
            mal_in_accepted = sum(1 for item in accepted_list if item.get("is_malicious", False))
            accepted_malicious_ratio = f"{(mal_in_accepted / accepted_count) * 100:.1f}% ({mal_in_accepted}/{accepted_count})"

        # 채택 수 < 3 이면 모델을 갱신하지 않고 사유 기록
        if accepted_count < 3:
            test_acc, test_loss = self.evaluate_global_model(detailed_classes=is_final_round)
            skip_reason = f"채택 업데이트 부족 ({accepted_count}개 < 최소 3개) - 모델 갱신 생략"
            self.log_exclusion("SERVER", self.current_round, self.model_version, skip_reason)
            record = {
                "round": self.current_round,
                "status": "SKIPPED (<3)",
                "selected_count": selected_count,
                "accepted_count": accepted_count,
                "delayed_count": delayed_count,
                "total_malicious_ratio": total_malicious_ratio,
                "accepted_malicious_ratio": accepted_malicious_ratio,
                "avg_norm_before": 0.0,
                "avg_norm_after": 0.0,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "virtual_elapsed": round(virtual_elapsed, 2),
                "actual_elapsed": round(actual_elapsed, 2),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            self.round_history.append(record)
            self._save_logs_and_checkpoint()
            self.accepted_updates = {}
            self.current_round += 1
            self.model_version = f"v1.0-round{self.current_round}"
            return record

        defense_method = self.exp_config.get("defense", "fedavg")
        clip_norm = float(self.exp_config.get("clip_norm", 3.0))

        # 방어 함수에는 악성 ID 정답을 절대 전달하지 않음
        updates_tensors = [item["delta_weights"] for item in accepted_list]
        sample_counts = [item["num_samples"] for item in accepted_list]

        agg_delta, defense_logs = aggregate_robust(
            updates_list=updates_tensors,
            sample_counts=sample_counts,
            method=defense_method,
            clip_norm=clip_norm,
        )

        # 클라이언트별 상세 로그 기록 (조작 전/후 노름 + 방어 전/후 노름 + 로컬 정확도)
        for item, dlog in zip(accepted_list, defense_logs):
            self.client_logs.append({
                "round": self.current_round,
                "client_id": item["client_id"],
                "num_samples": item["num_samples"],
                "local_acc": item["local_acc"],
                "is_malicious": item["is_malicious"],
                "norm_before_attack": item["norm_before_attack"],
                "norm_after_attack": dlog["norm_before_defense"],
                "norm_after_defense": dlog["norm_after_defense"],
                "was_clipped": dlog["was_clipped"],
                "virtual_time": item["virtual_time"],
                "status": "ACCEPTED",
            })

        # 전역 모델 갱신
        new_state = copy.deepcopy(self.global_model.state_dict())
        for k in self.reference_keys:
            new_state[k] = new_state[k].float() + agg_delta[k].to(self.device)
        self.global_model.load_state_dict(new_state)

        test_acc, test_loss = self.evaluate_global_model(detailed_classes=is_final_round)
        avg_nb = round(sum(d["norm_before_defense"] for d in defense_logs) / len(defense_logs), 4)
        avg_na = round(sum(d["norm_after_defense"] for d in defense_logs) / len(defense_logs), 4)

        record = {
            "round": self.current_round,
            "status": "UPDATED",
            "selected_count": selected_count,
            "accepted_count": accepted_count,
            "delayed_count": delayed_count,
            "total_malicious_ratio": total_malicious_ratio,
            "accepted_malicious_ratio": accepted_malicious_ratio,
            "avg_norm_before": avg_nb,
            "avg_norm_after": avg_na,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "virtual_elapsed": round(virtual_elapsed, 2),
            "actual_elapsed": round(actual_elapsed, 2),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.round_history.append(record)
        self._save_logs_and_checkpoint()

        self.accepted_updates = {}
        self.current_round += 1
        self.model_version = f"v1.0-round{self.current_round}"
        return record

    def _save_logs_and_checkpoint(self):
        os.makedirs("../save/logs", exist_ok=True)
        ckpt_dir = os.path.join("../save/checkpoints", self.exp_id)
        os.makedirs(ckpt_dir, exist_ok=True)

        # 다음 라운드 재실행용 체크포인트 저장
        torch.save(
            self.global_model.state_dict(),
            os.path.join(ckpt_dir, f"round_{self.current_round + 1}_start.pt"),
        )

        # 통합 JSON 로그 저장
        log_path = os.path.join("../save/logs", f"log_{self.exp_id}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "exp_id": self.exp_id,
                "config": self.exp_config,
                "final_test_acc": self.round_history[-1]["test_acc"] if self.round_history else 0.0,
                "round_history": self.round_history,
                "client_logs": self.client_logs,
                "excluded_logs": self.excluded_logs,
                "class_metrics": self.class_metrics,
            }, f, ensure_ascii=False, indent=2)


server_state = None


@app.route("/status", methods=["GET"])
def get_status():
    is_running = (server_state.active_process is not None and server_state.active_process.poll() is None)
    return jsonify({
        "is_running": is_running,
        "exp_id": server_state.exp_id,
        "exp_config": server_state.exp_config,
        "current_round": server_state.current_round,
        "accepted_count": len(server_state.accepted_updates),
        "round_history": server_state.round_history,
        "client_logs": server_state.client_logs[-40:],
        "excluded_count": len(server_state.excluded_logs),
        "excluded_logs": server_state.excluded_logs[-30:],
        "class_metrics": server_state.class_metrics,
    })


@app.route("/reset", methods=["POST"])
def reset_server():
    data = request.get_json(silent=True) or {}
    server_state.reset_experiment(data)
    return jsonify({"status": "reset_ok", "exp_id": server_state.exp_id}), 200


@app.route("/model", methods=["GET"])
def download_model():
    encoded = serialize_weights(server_state.global_model.state_dict())
    return jsonify({
        "round": server_state.current_round,
        "model_version": server_state.model_version,
        "weights": encoded,
    }), 200


@app.route("/update", methods=["POST"])
def submit_update():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "rejected", "reason": "Empty JSON"}), 400

    client_id = data.get("client_id")
    req_round = data.get("round")
    req_version = data.get("model_version")
    num_samples = data.get("num_samples", 0)
    encoded_delta = data.get("delta_weights")
    virtual_time = float(data.get("virtual_time", 0.0))
    deadline = float(server_state.exp_config.get("virtual_deadline", 10.0))

    if req_round != server_state.current_round:
        reason = f"라운드 불일치 (서버: {server_state.current_round}, 제출: {req_round})"
        server_state.log_exclusion(client_id, req_round, req_version, reason, virtual_time)
        return jsonify({"status": "rejected", "reason": reason}), 400

    if req_version != server_state.model_version:
        reason = f"모델 버전 불일치 (서버: {server_state.model_version}, 제출: {req_version})"
        server_state.log_exclusion(client_id, req_round, req_version, reason, virtual_time)
        return jsonify({"status": "rejected", "reason": reason}), 400

    if client_id in server_state.accepted_updates:
        reason = f"동일 라운드(Round {req_round}) 내 중복 제출"
        server_state.log_exclusion(client_id, req_round, req_version, reason, virtual_time)
        return jsonify({"status": "rejected", "reason": reason}), 400

    if virtual_time > deadline:
        reason = f"가상 마감 시각 초과 (소요: {virtual_time:.1f}s > 마감: {deadline:.1f}s)"
        server_state.log_exclusion(client_id, req_round, req_version, reason, virtual_time)
        return jsonify({"status": "rejected", "reason": reason}), 400

    try:
        delta_dict = deserialize_weights(encoded_delta, device="cpu")
    except Exception as e:
        reason = f"역직렬화 실패: {str(e)}"
        server_state.log_exclusion(client_id, req_round, req_version, reason, virtual_time)
        return jsonify({"status": "rejected", "reason": reason}), 400

    is_valid, err_msg = validate_weight_tensor(delta_dict, server_state.reference_keys)
    if not is_valid:
        reason = f"손상 텐서 차단: {err_msg}"
        server_state.log_exclusion(client_id, req_round, req_version, reason, virtual_time)
        return jsonify({"status": "rejected", "reason": reason}), 400

    server_state.accepted_updates[client_id] = {
        "client_id": client_id,
        "num_samples": int(num_samples),
        "local_acc": float(data.get("local_acc", 0.0)),
        "norm_before_attack": float(data.get("norm_before_attack", 0.0)),
        "delta_weights": delta_dict,
        "virtual_time": virtual_time,
        "is_malicious": bool(data.get("is_malicious", False)),
    }
    return jsonify({"status": "accepted", "round": server_state.current_round}), 200


@app.route("/aggregate", methods=["POST"])
def trigger_aggregate():
    meta = request.get_json(silent=True) or {}
    result = server_state.aggregate_round(meta)
    return jsonify({"status": "aggregated", "result": result}), 200


@app.route("/api/run_fl", methods=["POST"])
def api_run_fl():
    """대시보드 UI 설정(또는 체크포인트 재실행 옵션)으로 client_runner.py 실행"""
    if server_state.active_process is not None and server_state.active_process.poll() is None:
        return jsonify({"status": "busy", "message": "이미 실험이 실행 중입니다."}), 409

    cfg = request.get_json(silent=True) or {}
    cmd = [
        sys.executable, "client_runner.py",
        "--model", str(cfg.get("model", "cnn")),
        "--env", str(cfg.get("env", "baseline")),
        "--attack", str(cfg.get("attack", "none")),
        "--defense", str(cfg.get("defense", "fedavg")),
        "--optimizer", str(cfg.get("optimizer", "adam")),
        "--lr", str(cfg.get("lr", 0.001)),
        "--local_ep", str(cfg.get("local_ep", 3)),
        "--local_bs", str(cfg.get("local_bs", 64)),
        "--clip_norm", str(cfg.get("clip_norm", 3.0)),
        "--epochs", str(cfg.get("epochs", 10)),
        "--seed", str(cfg.get("seed", 1)),
    ]
    if cfg.get("checkpoint_path"):
        cmd.extend(["--checkpoint_path", str(cfg["checkpoint_path"])])
        cmd.extend(["--start_round", str(cfg.get("start_round", 1))])
    if cfg.get("split_path"):
        cmd.extend(["--split_path", str(cfg["split_path"])])
    if server_state.args.gpu is not None:
        cmd.extend(["--gpu", str(server_state.args.gpu)])

    server_state.active_process = subprocess.Popen(cmd)
    return jsonify({"status": "started", "cmd": " ".join(cmd)}), 200


@app.route("/api/summary_metrics", methods=["GET"])
def api_summary_metrics():
    """저장된 모든 실험 로그로부터 공격 피해 · 방어 개선 · 정상 손실(%p) 자동 산출"""
    from evaluator import compute_security_comparison_table
    rows, raw_records = compute_security_comparison_table("../save/logs")
    return jsonify({"comparison_rows": rows, "raw_records": raw_records}), 200


@app.route("/api/run_dlg", methods=["POST"])
def api_run_dlg():
    cfg = request.get_json(silent=True) or {}
    img_idx = int(cfg.get("index", 25))
    cmd = [
        sys.executable, "dlg_attack_defense.py",
        "--index", str(img_idx),
        "--iters", str(cfg.get("iters", 300)),
        "--noise_none", str(cfg.get("noise_none", 0.0)),
        "--noise_small", str(cfg.get("noise_small", 0.001)),
        "--noise_large", str(cfg.get("noise_large", 0.01)),
    ]
    subprocess.run(cmd, check=True)
    return get_dlg_result(img_idx)


@app.route("/api/dlg_result/<int:img_idx>", methods=["GET"])
def get_dlg_result(img_idx):
    json_path = f"../save/dlg/dlg_metrics_idx{img_idx}.json"
    img_path = f"../save/dlg/dlg_comparison_idx{img_idx}.png"
    if not os.path.exists(json_path):
        return jsonify({"exists": False}), 200

    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    img_b64 = ""
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return jsonify({"exists": True, "metrics": metrics, "image_b64": img_b64}), 200


@app.route("/", methods=["GET"])
def dashboard():
    html = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>연합학습 보안 시뮬레이터 통합 대시보드</title>
        <style>
            body { font-family: 'Malgun Gothic', sans-serif; margin: 24px; background: #f4f7fa; color: #1a202c; }
            .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
            .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
            .tab-btn { padding: 10px 18px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; background: #e2e8f0; color: #2d3748; font-size: 14px; }
            .tab-btn.active { background: #2b6cb0; color: #fff; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .card { background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); margin-bottom: 20px; }
            .grid-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 10px; align-items: end; }
            label { display: block; font-size: 12px; font-weight: bold; color: #4a5568; margin-bottom: 4px; }
            select, input { width: 100%; padding: 7px; border: 1px solid #cbd5e0; border-radius: 6px; box-sizing: border-box; font-size: 13px; }
            .btn-run { padding: 9px 14px; background: #3182ce; color: #fff; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; width: 100%; }
            .btn-run:disabled { background: #a0aec0; cursor: not-allowed; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12.5px; }
            th, td { border: 1px solid #e2e8f0; padding: 7px; text-align: center; }
            th { background: #edf2f7; }
            .badge { padding: 4px 10px; border-radius: 12px; font-weight: bold; color: #fff; background: #3182ce; font-size: 13px; }
            .warn { color: #c53030; font-weight: bold; text-align: left; }
        </style>
    </head>
    <body>
        <div class="top-bar">
            <h1>🛡️ 연합학습(FedAvg) 보안 시뮬레이션 통합 시스템</h1>
            <span class="badge">● 실시간 모니터링 중</span>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('fl-tab', this)">1. 연합학습 실행 · 라운드/클라이언트 로그</button>
            <button class="tab-btn" onclick="switchTab('dlg-tab', this)">2. 모델 역추적(DLG) 공격 · 방어</button>
            <button class="tab-btn" onclick="switchTab('eval-tab', this)">3. 정량 평가표(%p) · 체크포인트 재실행</button>
        </div>

        <!-- 탭 1: 연합학습 본 실험 -->
        <div id="fl-tab" class="tab-content active">
            <div class="card">
                <h3>⚙️ 실험 조건 및 하이퍼파라미터 유동 제어</h3>
                <div class="grid-controls">
                    <div>
                        <label>모델 구조</label>
                        <select id="cfg-model">
                            <option value="cnn">CNNCifar (합성곱 2층)</option>
                            <option value="simple_cnn">SimpleCNN (합성곱 3층)</option>
                        </select>
                    </div>
                    <div>
                        <label>운영 환경 (4종)</label>
                        <select id="cfg-env">
                            <option value="baseline">1. 기본 환경 (IID, 100%)</option>
                            <option value="delay">2. 네트워크 지연 (3개 초과)</option>
                            <option value="partial">3. 부분 참여 (5/10 선택)</option>
                            <option value="noniid">4. 레이블 편향 (Non-IID)</option>
                        </select>
                    </div>
                    <div>
                        <label>공격 조건</label>
                        <select id="cfg-attack">
                            <option value="none">정상 환경 (공격 없음)</option>
                            <option value="collusion">내부자 협력 공격 (-5배)</option>
                        </select>
                    </div>
                    <div>
                        <label>방어 및 집계 (3종)</label>
                        <select id="cfg-defense">
                            <option value="fedavg">FedAvg (방어 없음)</option>
                            <option value="clipped_fedavg">노름 클리핑 + FedAvg</option>
                            <option value="clipped_median">노름 클리핑 + 좌표별 중앙값</option>
                        </select>
                    </div>
                    <div>
                        <label>옵티마이저</label>
                        <select id="cfg-opt">
                            <option value="adam">Adam</option>
                            <option value="sgd">SGD (momentum=0.9)</option>
                        </select>
                    </div>
                    <div><label>학습률 (LR)</label><input type="number" id="cfg-lr" value="0.001" step="0.0005"></div>
                    <div><label>클리핑 상한</label><input type="number" id="cfg-clip" value="3.0" step="0.1"></div>
                    <div><label>로컬 에폭</label><input type="number" id="cfg-local-ep" value="3" min="1"></div>
                    <div><label>글로벌 라운드</label><input type="number" id="cfg-epochs" value="10" min="1"></div>
                    <div><label>시드 (Seed)</label><input type="number" id="cfg-seed" value="1"></div>
                    <div><button class="btn-run" id="btn-start-fl" onclick="startFLExperiment()">▶ 실험 시작</button></div>
                </div>
                <p style="margin-top:12px; font-size:13px;">
                    <strong>진행:</strong> <span class="badge" id="current-round">Round 1</span> |
                    <strong>대기 업데이트:</strong> <span id="accepted-count">0</span> |
                    <strong>실험 ID:</strong> <code id="active-exp-id">-</code>
                </p>
            </div>

            <div class="card">
                <h3>📊 라운드별 전역 모델 집계 · 노름 · 정확도 추적</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Round</th><th>상태</th><th>선택/채택/지연</th><th>전체 악성</th><th>채택 악성</th>
                            <th>방어전 노름</th><th>방어후 노름</th><th>Test Acc (%)</th><th>Test Loss</th><th>가상/실제 시간</th>
                        </tr>
                    </thead>
                    <tbody id="history-body"><tr><td colspan="10">데이터 대기 중...</td></tr></tbody>
                </table>
            </div>

            <div class="card">
                <h3>👥 클라이언트별 상세 로그 (최근 40건: 표본수 · 로컬정확도 · 조작/방어 전후 노름)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Round</th><th>Client ID</th><th>유형</th><th>표본 수</th><th>로컬 정확도</th>
                            <th>조작 전 노름</th><th>방어 전(조작 후) 노름</th><th>방어 후 노름</th><th>클리핑 여부</th><th>가상 시간</th>
                        </tr>
                    </thead>
                    <tbody id="client-body"><tr><td colspan="10">클라이언트 로그 대기 중...</td></tr></tbody>
                </table>
            </div>

            <div class="card">
                <h3>🚫 제외된 업데이트 및 지연/이상치 차단 로그 (<span id="excluded-count">0</span>건)</h3>
                <table>
                    <thead>
                        <tr><th>시각</th><th>서버 라운드</th><th>Client ID</th><th>제출 라운드</th><th>가상 시간</th><th>제외 사유</th></tr>
                    </thead>
                    <tbody id="excluded-body"><tr><td colspan="6">제외 기록 없음</td></tr></tbody>
                </table>
            </div>

            <div class="card">
                <h3>🎯 최종 라운드 클래스별 상세 성능 지표 (CIFAR-10 10개 클래스)</h3>
                <table>
                    <thead>
                        <tr><th>클래스</th><th>정확도(Recall %)</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>평가 표본 수</th></tr>
                    </thead>
                    <tbody id="class-body"><tr><td colspan="6">최종 라운드 완료 후 자동 표시됩니다.</td></tr></tbody>
                </table>
            </div>
        </div>

        <!-- 탭 2: DLG 역추적 -->
        <div id="dlg-tab" class="tab-content">
            <div class="card">
                <h3>🔍 DLG 단일 이미지 역추적 공격 및 가우시안 노이즈 방어</h3>
                <div class="grid-controls">
                    <div><label>이미지 Index</label><input type="number" id="dlg-idx" value="25"></div>
                    <div><label>L-BFGS 반복</label><input type="number" id="dlg-iters" value="300"></div>
                    <div><label>노이즈 없음 (σ)</label><input type="number" id="dlg-n0" value="0.0" step="0.001"></div>
                    <div><label>노이즈 작음 (σ)</label><input type="number" id="dlg-n1" value="0.001" step="0.001"></div>
                    <div><label>노이즈 큼 (σ)</label><input type="number" id="dlg-n2" value="0.01" step="0.001"></div>
                    <div><button class="btn-run" id="btn-start-dlg" onclick="runDLGExperiment()">▶ DLG 복원 실행</button></div>
                </div>
            </div>
            <div class="card">
                <h3>🖼️ 원본 vs 노이즈 단계별 복원 이미지 및 픽셀 오차(MSE)</h3>
                <div id="dlg-image-box" style="text-align:center; margin:15px 0;">결과를 불러오는 중...</div>
                <table>
                    <thead><tr><th>방어 조건</th><th>노이즈(σ)</th><th>실제 클래스</th><th>복원 예측 클래스</th><th>픽셀 MSE</th></tr></thead>
                    <tbody id="dlg-table-body"></tbody>
                </table>
            </div>
        </div>

        <!-- 탭 3: 정량 평가표(%p) & 체크포인트 재실행 -->
        <div id="eval-tab" class="tab-content">
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3>📈 공격 피해 · 방어 개선 · 정상 손실 (%p) 자동 비교표</h3>
                    <button class="btn-run" style="width:180px;" onclick="loadSummaryMetrics()">🔄 로그 새로고침</button>
                </div>
                <p style="font-size:12px; color:#4a5568;">
                    계산식: <strong>공격 피해</strong> = 정상 FedAvg - 공격 FedAvg |
                    <strong>방어 개선</strong> = 공격+방어 - 공격 FedAvg |
                    <strong>정상 손실</strong> = 정상 FedAvg - 정상+방어
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>환경</th><th>방어 방식</th><th>시드</th>
                            <th>정상 FedAvg</th><th>공격 FedAvg</th><th>공격+방어</th><th>정상+방어</th>
                            <th>공격 피해(%p)</th><th>방어 개선(%p)</th><th>정상 손실(%p)</th>
                        </tr>
                    </thead>
                    <tbody id="eval-summary-body"><tr><td colspan="10">데이터 불러오는 중...</td></tr></tbody>
                </table>
            </div>

            <div class="card">
                <h3>🔁 원인 분석용 저장 라운드 체크포인트 재실행 (Re-run)</h3>
                <p style="font-size:12px; color:#4a5568;">저장된 특정 라운드의 모델 상태(.pt)와 데이터 분할(data_split.json)을 그대로 불러와 공격/방어 조건만 바꿔 재실행합니다.</p>
                <div class="grid-controls">
                    <div><label>체크포인트 경로 (.pt)</label><input type="text" id="re-ckpt" placeholder="../save/checkpoints/.../round_5_start.pt"></div>
                    <div><label>데이터 분할 경로 (.json)</label><input type="text" id="re-split" placeholder="../save/splits/split_baseline_s1.json"></div>
                    <div><label>시작 라운드 번호</label><input type="number" id="re-round" value="5"></div>
                    <div>
                        <label>변경할 공격 조건</label>
                        <select id="re-attack"><option value="none">공격 해제 (정상)</option><option value="collusion">내부자 협력 공격</option></select>
                    </div>
                    <div>
                        <label>변경할 방어 방식</label>
                        <select id="re-defense"><option value="fedavg">FedAvg</option><option value="clipped_fedavg">클리핑+FedAvg</option><option value="clipped_median">클리핑+중앙값</option></select>
                    </div>
                    <div><button class="btn-run" onclick="rerunCheckpoint()">▶ 해당 라운드 재실행</button></div>
                </div>
            </div>
        </div>

        <script>
            function switchTab(tabId, btn) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
                btn.classList.add('active');
                if (tabId === 'dlg-tab') loadDLGResult();
                if (tabId === 'eval-tab') loadSummaryMetrics();
            }

            async function startFLExperiment() {
                const payload = {
                    model: document.getElementById('cfg-model').value,
                    env: document.getElementById('cfg-env').value,
                    attack: document.getElementById('cfg-attack').value,
                    defense: document.getElementById('cfg-defense').value,
                    optimizer: document.getElementById('cfg-opt').value,
                    lr: parseFloat(document.getElementById('cfg-lr').value),
                    clip_norm: parseFloat(document.getElementById('cfg-clip').value),
                    local_ep: parseInt(document.getElementById('cfg-local-ep').value),
                    epochs: parseInt(document.getElementById('cfg-epochs').value),
                    seed: parseInt(document.getElementById('cfg-seed').value),
                };
                const res = await fetch('/api/run_fl', {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status !== 'started') alert(data.message);
            }

            async function rerunCheckpoint() {
                const payload = {
                    model: document.getElementById('cfg-model').value,
                    env: document.getElementById('cfg-env').value,
                    attack: document.getElementById('re-attack').value,
                    defense: document.getElementById('re-defense').value,
                    optimizer: document.getElementById('cfg-opt').value,
                    lr: parseFloat(document.getElementById('cfg-lr').value),
                    clip_norm: parseFloat(document.getElementById('cfg-clip').value),
                    epochs: 1,
                    start_round: parseInt(document.getElementById('re-round').value),
                    checkpoint_path: document.getElementById('re-ckpt').value,
                    split_path: document.getElementById('re-split').value,
                };
                const res = await fetch('/api/run_fl', {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status === 'started') {
                    alert('재실행이 시작되었습니다. 1번 탭에서 결과를 확인하세요.');
                } else {
                    alert(data.message);
                }
            }

            async function loadSummaryMetrics() {
                const res = await fetch('/api/summary_metrics');
                const data = await res.json();
                const tbody = document.getElementById('eval-summary-body');
                if (!data.comparison_rows || data.comparison_rows.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="10">저장된 비교 실험 로그가 부족합니다.</td></tr>';
                    return;
                }
                tbody.innerHTML = data.comparison_rows.map(r => `
                    <tr>
                        <td>${r.env}</td><td>${r.defense}</td><td>${r.seed}</td>
                        <td>${r.normal_fedavg ?? '-'}</td><td>${r.attack_fedavg ?? '-'}</td>
                        <td>${r.attack_defense ?? '-'}</td><td>${r.normal_defense ?? '-'}</td>
                        <td><strong>${r.attack_damage ?? '-'}</strong></td>
                        <td><strong>${r.defense_gain ?? '-'}</strong></td>
                        <td><strong>${r.normal_loss ?? '-'}</strong></td>
                    </tr>
                `).join('');
            }

            async function runDLGExperiment() {
                const btn = document.getElementById('btn-start-dlg');
                btn.disabled = true; btn.textContent = '복원 연산 중...';
                const payload = {
                    index: parseInt(document.getElementById('dlg-idx').value),
                    iters: parseInt(document.getElementById('dlg-iters').value),
                    noise_none: parseFloat(document.getElementById('dlg-n0').value),
                    noise_small: parseFloat(document.getElementById('dlg-n1').value),
                    noise_large: parseFloat(document.getElementById('dlg-n2').value),
                };
                const res = await fetch('/api/run_dlg', {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
                });
                renderDLG(await res.json());
                btn.disabled = false; btn.textContent = '▶ DLG 복원 실행';
            }

            async function loadDLGResult() {
                const idx = document.getElementById('dlg-idx').value;
                const res = await fetch(`/api/dlg_result/${idx}`);
                renderDLG(await res.json());
            }

            function renderDLG(data) {
                if (!data.exists) {
                    document.getElementById('dlg-image-box').innerHTML = '<p>저장된 DLG 결과가 없습니다.</p>';
                    return;
                }
                document.getElementById('dlg-image-box').innerHTML = `<img src="data:image/png;base64,${data.image_b64}" style="max-width:95%; border:1px solid #cbd5e0; border-radius:8px;">`;
                document.getElementById('dlg-table-body').innerHTML = data.metrics.map(m => `
                    <tr><td><strong>${m.condition}</strong></td><td>${m.noise_std}</td><td>${m.true_class}</td><td>${m.reconstructed_class}</td><td><strong>${m.mse.toFixed(6)}</strong></td></tr>
                `).join('');
            }

            async function refreshDashboard() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    const btn = document.getElementById('btn-start-fl');
                    btn.disabled = data.is_running;
                    btn.textContent = data.is_running ? '⏳ 학습 진행 중...' : '▶ 실험 시작';

                    document.getElementById('current-round').textContent = `Round ${data.current_round}`;
                    document.getElementById('accepted-count').textContent = data.accepted_count;
                    document.getElementById('active-exp-id').textContent = data.exp_id;

                    const hBody = document.getElementById('history-body');
                    hBody.innerHTML = data.round_history.length === 0
                        ? '<tr><td colspan="10">완료된 라운드가 없습니다.</td></tr>'
                        : data.round_history.map(r => `
                            <tr>
                                <td>${r.round}</td><td><strong>${r.status}</strong></td>
                                <td>${r.selected_count} / ${r.accepted_count} / ${r.delayed_count}</td>
                                <td>${r.total_malicious_ratio}</td><td>${r.accepted_malicious_ratio}</td>
                                <td>${r.avg_norm_before}</td><td>${r.avg_norm_after}</td>
                                <td><strong>${r.test_acc}%</strong></td><td>${r.test_loss}</td>
                                <td>${r.virtual_elapsed}s / ${r.actual_elapsed}s</td>
                            </tr>
                        `).join('');

                    const cBody = document.getElementById('client-body');
                    cBody.innerHTML = data.client_logs.length === 0
                        ? '<tr><td colspan="10">클라이언트 로그 대기 중...</td></tr>'
                        : data.client_logs.map(c => `
                            <tr>
                                <td>${c.round}</td><td>Client ${c.client_id}</td>
                                <td>${c.is_malicious ? '<span style="color:#c53030;font-weight:bold;">악성</span>' : '정상'}</td>
                                <td>${c.num_samples}</td><td>${c.local_acc}%</td>
                                <td>${c.norm_before_attack}</td><td>${c.norm_after_attack}</td>
                                <td>${c.norm_after_defense}</td><td>${c.was_clipped ? 'O' : '-'}</td>
                                <td>${c.virtual_time}s</td>
                            </tr>
                        `).join('');

                    document.getElementById('excluded-count').textContent = data.excluded_count;
                    const eBody = document.getElementById('excluded-body');
                    eBody.innerHTML = data.excluded_logs.length === 0
                        ? '<tr><td colspan="6">제외된 업데이트가 없습니다.</td></tr>'
                        : data.excluded_logs.map(l => `
                            <tr><td>${l.timestamp}</td><td>${l.server_round}</td><td>${l.client_id}</td><td>${l.submitted_round}</td><td>${l.virtual_time}s</td><td class="warn">${l.reason}</td></tr>
                        `).join('');

                    const clsBody = document.getElementById('class-body');
                    clsBody.innerHTML = (!data.class_metrics || data.class_metrics.length === 0)
                        ? '<tr><td colspan="6">최종 라운드 완료 후 자동 표시됩니다.</td></tr>'
                        : data.class_metrics.map(m => `
                            <tr><td><strong>${m.class_name}</strong></td><td>${m.accuracy}%</td><td>${m.precision}</td><td>${m.recall}</td><td>${m.f1}</td><td>${m.support}</td></tr>
                        `).join('');
                } catch (e) {}
            }

            refreshDashboard();
            setInterval(refreshDashboard, 1500);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--iid", type=int, default=1)
    parser.add_argument("--unequal", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    args.num_users = args.num_clients

    server_state = FederatedServerState(args)
    print(f"=== 통합 연합학습 보안 서버 실행: http://127.0.0.1:{args.port} (Device: {server_state.device}) ===")
    app.run(host="127.0.0.1", port=args.port, debug=False)