import os
import sys
import io
import copy
import json
import time
import base64
import argparse

import numpy as np
import torch
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.core.models import CNNCifar, LeNetDLG
from src.core.data_manager import load_cifar10_splits
from src.core.local_trainer import train_local_client, compute_l2_norm
from src.security.attacks import apply_insider_collusion, compute_single_image_gradients
from src.security.defenses import add_gaussian_noise_to_gradients


def serialize_tensor_obj(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def deserialize_tensor_obj(b64_str):
    raw = base64.b64decode(b64_str.encode("utf-8"))
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu")


class VirtualClient:
    """개별 가상 클라이언트 인스턴스 (순차 생성 및 로컬 학습 수행)"""

    def __init__(self, client_id, idxs, cfg, device):
        self.client_id = client_id
        self.idxs = idxs
        self.cfg = cfg
        self.device = device
        self.model = CNNCifar(num_classes=cfg["num_classes"])

    def load_global_weights(self, global_state_dict):
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def execute_local_training(self, train_dataset):
        delta_w, avg_loss = train_local_client(
            model=self.model,
            train_dataset=train_dataset,
            idxs=self.idxs,
            local_ep=int(self.cfg["local_ep"]),
            local_bs=int(self.cfg["local_bs"]),
            lr=float(self.cfg["lr"]),
            momentum=float(self.cfg["momentum"]),
            device=self.device,
        )
        return delta_w, avg_loss


def run_dlg_experiment(cfg, data_bundle, server_url, device):
    """모델 역추적(DLG) 단일 이미지 1회 그래디언트 추출 및 전송 트랙"""
    target_cid = int(cfg["dlg_target_client_id"])
    sample_idx = int(cfg["dlg_sample_index"])
    noise_mode = cfg["dlg_noise_mode"]
    noise_std = float(cfg["dlg_noise_levels"][noise_mode])

    requests.post(f"{server_url}/api/round_plan", json={"selected_clients": [target_cid]}, timeout=15)
    requests.post(f"{server_url}/api/client_start", json={"client_id": target_cid, "mode": "dlg"}, timeout=15)

    resp = requests.get(f"{server_url}/api/model", timeout=15).json()
    global_state = deserialize_tensor_obj(resp["weights"])

    model = LeNetDLG(num_classes=cfg["num_classes"])
    model.load_state_dict(global_state)

    raw_ds = data_bundle["train_raw_dataset"]
    global_img_idx = data_bundle["user_groups"][target_cid][sample_idx]
    img_tensor, label_int = raw_ds[global_img_idx]

    grads = compute_single_image_gradients(model, img_tensor, int(label_int), device=device)
    noisy_grads = add_gaussian_noise_to_gradients(grads, noise_std=noise_std, seed=cfg["seed"])

    requests.post(
        f"{server_url}/api/dlg_submit",
        json={
            "client_id": target_cid,
            "sample_index": sample_idx,
            "noise_mode": noise_mode,
            "noise_std": noise_std,
            "gradients": serialize_tensor_obj(noisy_grads),
        },
        timeout=120,
    )


def run_federated_simulation(cfg, server_url):
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_skew = (cfg["environment"] == "label_skew")

    data_bundle = load_cifar10_splits(
        data_dir=os.path.join(ROOT_DIR, "data"),
        seed=seed,
        label_skew=label_skew,
        num_users=int(cfg["num_users"]),
    )

    if cfg["attack"] == "dlg":
        run_dlg_experiment(cfg, data_bundle, server_url, device)
        return

    num_users = int(cfg["num_users"])
    total_rounds = int(cfg["epochs"])
    delayed_ids = set(cfg.get("delayed_client_ids", []))
    malicious_ids = set(cfg.get("malicious_client_ids", []))

    for r in range(total_rounds):
        model_resp = requests.get(f"{server_url}/api/model", timeout=30).json()
        if model_resp.get("status") in ("stopped", "completed"):
            break

        server_round = int(model_resp["round"])
        model_version = int(model_resp["model_version"])
        global_state = deserialize_tensor_obj(model_resp["weights"])

        # 운영 환경에 따른 참여 클라이언트 결정
        if cfg["environment"] == "partial":
            k = int(cfg.get("partial_participation_count", 5))
            selected_clients = sorted(rng.choice(num_users, size=k, replace=False).tolist())
        else:
            selected_clients = list(range(num_users))

        requests.post(
            f"{server_url}/api/round_plan",
            json={"selected_clients": selected_clients},
            timeout=15,
        )

        # 1단계: 참여 클라이언트들을 순차 생성하여 정상 로컬 학습 수행
        honest_updates = {}
        local_losses = {}
        real_times = {}
        virtual_times = {}

        for cid in selected_clients:
            requests.post(
                f"{server_url}/api/client_start",
                json={"client_id": cid, "mode": "training"},
                timeout=15,
            )

            t0 = time.time()
            client = VirtualClient(
                client_id=cid,
                idxs=data_bundle["user_groups"][cid],
                cfg=cfg,
                device=device,
            )
            client.load_global_weights(global_state)
            delta_w, loss_val = client.execute_local_training(data_bundle["train_dataset"])
            real_elapsed = time.time() - t0

            # 가상 지연 시간 생성 (실제 순차 실행 시간과 분리)
            if cfg["environment"] == "delay" and cid in delayed_ids:
                low, high = cfg["delayed_virtual_time_range"]
            else:
                low, high = cfg["normal_virtual_time_range"]
            v_time = float(rng.uniform(low, high))

            honest_updates[cid] = delta_w
            local_losses[cid] = loss_val
            real_times[cid] = real_elapsed
            virtual_times[cid] = v_time

        # 2단계: 내부자 협력 공격 활성화 시 참여 악성 수(0/1/2명)에 따라 업데이트 조작
        if cfg["attack"] == "collusion":
            final_updates, _ = apply_insider_collusion(
                honest_updates=honest_updates,
                malicious_ids=malicious_ids,
                multiplier=float(cfg.get("collusion_multiplier", -5.0)),
            )
        else:
            final_updates = honest_updates

        # 3단계: 0번부터 순서대로 서버 REST API로 업데이트 제출
        for cid in selected_clients:
            pre_norm = compute_l2_norm(honest_updates[cid])
            post_norm = compute_l2_norm(final_updates[cid])

            requests.post(
                f"{server_url}/api/submit_update",
                json={
                    "client_id": cid,
                    "round": server_round,
                    "model_version": model_version,
                    "num_samples": len(data_bundle["user_groups"][cid]),
                    "pre_attack_norm": pre_norm,
                    "post_attack_norm": post_norm,
                    "virtual_time_sec": virtual_times[cid],
                    "real_time_sec": real_times[cid],
                    "local_loss": local_losses[cid],
                    "delta_w": serialize_tensor_obj(final_updates[cid]),
                },
                timeout=30,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--server", type=str, default="http://127.0.0.1:5000")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        runtime_cfg = json.load(f)

    run_federated_simulation(runtime_cfg, args.server)