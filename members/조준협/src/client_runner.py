import os
import json
import copy
import time
import argparse
import requests
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import CNNCifar, SimpleCNN
from utils import get_dataset
from serialization import (
    serialize_weights,
    deserialize_weights,
    compute_weight_delta,
)
from security import apply_collusion_attack, compute_l2_norm


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label, dtype=torch.long)


def train_and_eval_local_client(model, train_dataset, user_idxs, args, device):
    model.train()
    model.to(device)
    criterion = nn.NLLLoss().to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    trainloader = DataLoader(
        DatasetSplit(train_dataset, user_idxs),
        batch_size=args.local_bs,
        shuffle=True,
    )

    for _ in range(args.local_ep):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

    # 로컬 데이터셋 기준 학습 정확도 산출
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    local_acc = round(100.0 * correct / total, 2) if total > 0 else 0.0
    return copy.deepcopy(model.state_dict()), local_acc


def main():
    parser = argparse.ArgumentParser(description="연합학습 운영환경 및 공격 시뮬레이션 클라이언트 실행기")
    parser.add_argument("--config", type=str, default="", help="외부 설정 JSON 파일 경로")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--env", type=str, default="baseline", choices=["baseline", "delay", "partial", "noniid"])
    parser.add_argument("--attack", type=str, default="none", choices=["none", "collusion"])
    parser.add_argument("--defense", type=str, default="fedavg", choices=["fedavg", "clipped_fedavg", "clipped_median"])
    parser.add_argument("--clip_norm", type=float, default=3.0)
    parser.add_argument("--attack_scale", type=float, default=-5.0)
    parser.add_argument("--malicious_ids", type=str, default="0,1")
    parser.add_argument("--delayed_ids", type=str, default="0,1,2")
    parser.add_argument("--virtual_deadline", type=float, default=10.0)
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--local_ep", type=int, default=3)
    parser.add_argument("--local_bs", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--unequal", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="", help="재실행용 시작 모델 체크포인트(.pt)")
    parser.add_argument("--split_path", type=str, default="", help="재실행용 데이터 분할 파일(.json)")
    parser.add_argument("--start_round", type=int, default=1)
    args = parser.parse_args()

    # 외부 설정 파일(--config) 지정 시 덮어쓰기
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        for k, v in cfg_data.items():
            if hasattr(args, k):
                setattr(args, k, v)

    args.iid = 0 if args.env == "noniid" else 1
    frac = 0.5 if args.env == "partial" else 1.0
    malicious_ids = [int(x) for x in str(args.malicious_ids).split(",") if str(x).strip()] if args.attack == "collusion" else []
    delayed_ids = [int(x) for x in str(args.delayed_ids).split(",") if str(x).strip()] if args.env == "delay" else []

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if (args.gpu is not None and torch.cuda.is_available()) else "cpu"

    # 1. 서버 초기화
    requests.post(f"{args.server_url}/reset", json={
        "model": args.model,
        "env": args.env,
        "attack": args.attack,
        "defense": args.defense,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "local_ep": args.local_ep,
        "local_bs": args.local_bs,
        "clip_norm": args.clip_norm,
        "seed": args.seed,
        "virtual_deadline": args.virtual_deadline,
        "checkpoint_path": args.checkpoint_path,
        "start_round": args.start_round,
    })

    # 2. 데이터 분할 로드 또는 저장 (재실행 재현성 보장)
    train_dataset, _, generated_groups = get_dataset(args)
    if args.split_path and os.path.exists(args.split_path):
        with open(args.split_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        user_groups = {int(k): v for k, v in loaded.items()}
    else:
        user_groups = {int(k): [int(i) for i in v] for k, v in generated_groups.items()}
        os.makedirs("../save/splits", exist_ok=True)
        split_file = f"../save/splits/split_{'noniid' if args.iid == 0 else 'iid'}_s{args.seed}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in user_groups.items()}, f)

    print(f"=== 실험 시작 | Env={args.env} | Attack={args.attack} | Defense={args.defense} | Opt={args.optimizer}(lr={args.lr}) ===")

    for r in range(args.epochs):
        round_start_actual = time.time()
        model_res = requests.get(f"{args.server_url}/model").json()
        server_round = model_res["round"]
        model_version = model_res["model_version"]
        global_weights = deserialize_weights(model_res["weights"], device="cpu")

        m = max(int(frac * args.num_users), 1)
        selected_users = sorted(np.random.choice(range(args.num_users), m, replace=False).tolist())

        raw_updates = {}
        raw_norms = {}
        local_accs = {}
        virtual_times = {}

        for cid in selected_users:
            local_model = SimpleCNN(args=args) if args.model == "simple_cnn" else CNNCifar(args=args)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            updated_w, l_acc = train_and_eval_local_client(local_model, train_dataset, user_groups[cid], args, device)
            delta = compute_weight_delta(updated_w, global_weights)

            raw_updates[cid] = delta
            raw_norms[cid] = round(compute_l2_norm(delta), 4)
            local_accs[cid] = l_acc

            if cid in delayed_ids:
                virtual_times[cid] = args.virtual_deadline + 5.0
            else:
                virtual_times[cid] = round(float(np.random.uniform(2.0, 6.0)), 2)

        if args.attack == "collusion":
            final_updates, _ = apply_collusion_attack(
                client_updates=raw_updates,
                malicious_ids=malicious_ids,
                attack_scale=args.attack_scale,
            )
        else:
            final_updates = raw_updates

        delayed_count = 0
        for cid in selected_users:
            payload = {
                "client_id": int(cid),
                "round": server_round,
                "model_version": model_version,
                "num_samples": int(len(user_groups[cid])),
                "local_acc": local_accs[cid],
                "norm_before_attack": raw_norms[cid],
                "delta_weights": serialize_weights(final_updates[cid]),
                "virtual_time": virtual_times[cid],
                "is_malicious": (cid in malicious_ids),
            }
            res = requests.post(f"{args.server_url}/update", json=payload)
            if res.status_code != 200 and "마감 시각 초과" in res.json().get("reason", ""):
                delayed_count += 1

        actual_elapsed = time.time() - round_start_actual
        virtual_elapsed = max(virtual_times.values()) if virtual_times else 0.0
        total_mal_ratio = f"{(len(malicious_ids) / args.num_users) * 100:.1f}%"

        agg_res = requests.post(f"{args.server_url}/aggregate", json={
            "selected_count": len(selected_users),
            "delayed_count": delayed_count,
            "total_malicious_ratio": total_mal_ratio,
            "virtual_elapsed": virtual_elapsed,
            "actual_elapsed": actual_elapsed,
            "is_final_round": (r == args.epochs - 1),
        }).json()["result"]

        print(
            f"[Round {agg_res['round']:2d}] 상태={agg_res['status']} | "
            f"선택/채택/지연={agg_res['selected_count']}/{agg_res['accepted_count']}/{agg_res['delayed_count']} | "
            f"Test Acc={agg_res['test_acc']}% | Loss={agg_res['test_loss']}"
        )


if __name__ == "__main__":
    main()