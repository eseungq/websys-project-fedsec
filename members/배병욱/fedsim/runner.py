"""클라이언트 실행 프로그램: 가상 클라이언트 10개를 한 프로그램에서 순차 학습한다.

한 라운드: GET /model → 참여자 순서대로 로컬 학습(업데이트 = 로컬 가중치 − 시작 전역 가중치)
→ 악성이면 협력 공격 → POST /update → POST /aggregate → 검증 평가·로그.

사용 예 (members/배병욱 에서):
    python -m fedsim.runner --env baseline --attack --defense median --seed 0
"""
import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import requests
import torch
import torch.nn.functional as F

from . import data as D
from .attacks import collusion
from .environment import Schedule
from .models import build_model, get_vector, set_vector
from .serialization import from_bytes, to_bytes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFENSES = ("fedavg", "clip_fedavg", "median")
_CACHE = {}


# ---------------------------------------------------------------- 공통 도구
def load_config(path):
    with open(path) as f:
        return json.load(f)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerProcess:
    """Flask 서버를 별도 프로세스로 띄운다 (클라이언트와 REST로만 통신)."""

    def __init__(self, port=None):
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = subprocess.Popen([sys.executable, "-m", "fedsim.server", "--port", str(self.port)],
                                     cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                requests.get(self.url + "/status", timeout=0.5)
                return
            except requests.RequestException:
                time.sleep(0.2)
        self.close()
        raise RuntimeError("서버가 시작되지 않음")

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)


def get_data(cfg, device):
    key = (cfg["data_root"], str(device))
    if key not in _CACHE:
        xtr, ytr, xte, yte = D.load_cifar10(os.path.join(ROOT, cfg["data_root"]))
        _CACHE[key] = (xtr.to(device), ytr.to(device), xte.to(device), yte.to(device), ytr.numpy())
    return _CACHE[key]


def client_split(cfg, env, seed, labels_np):
    val, clients = D.load_split(os.path.join(ROOT, cfg["split"]))
    if env == "label_skew":
        sk = cfg["env"]["label_skew"]
        clients = D.label_skew_split(clients, labels_np, seed, sk["skew"], sk["shards_per_client"])
    return val, clients


def make_schedule(cfg, env, seed):
    e = cfg["env"]
    kw = dict(env=env, seed=seed, num_clients=cfg["num_clients"], num_malicious=cfg["num_malicious"],
              num_delayed=e["delay"]["num_delayed"], participants=e["partial"]["participants"],
              base_time=e["delay"]["base_time"], jitter=e["delay"]["jitter"],
              delay=e["delay"]["delay"], deadline=e["delay"]["deadline"])
    return Schedule(**kw), kw


@torch.no_grad()
def evaluate(model, x, y, bs=1000, num_classes=10):
    model.eval()
    correct, loss_sum = 0, 0.0
    per_class = torch.zeros(num_classes, device=x.device)
    count = torch.zeros(num_classes, device=x.device)
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        yy = y[i:i + bs]
        loss_sum += F.cross_entropy(out, yy, reduction="sum").item()
        hit = (out.argmax(1) == yy).float()
        correct += hit.sum().item()
        per_class.index_add_(0, yy, hit)
        count.index_add_(0, yy, torch.ones_like(hit))
    cls = (per_class / count.clamp(min=1) * 100).tolist()
    return correct / len(x) * 100, loss_sum / len(x), cls


def local_train(model, x, y, idx, cfg, gen_seed):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(gen_seed)
    n, bs = len(idx), cfg["local_bs"]
    for _ in range(cfg["local_ep"]):
        perm = idx[torch.randperm(n, generator=g).to(idx.device)]
        for b in range(0, n, bs):
            bi = perm[b:b + bs]
            opt.zero_grad()
            F.cross_entropy(model(x[bi]), y[bi]).backward()
            opt.step()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- 실험 1회
def run_experiment(cfg, env, attack, defense, seed, out_dir, server_url, rounds=None,
                   start_round=1, init_vec=None, disabled_attackers=(), log=print):
    """실험 1회. start_round/init_vec는 체크포인트 재실행(replay)에 쓴다."""
    device = pick_device(cfg.get("device", "auto"))
    rounds = rounds or cfg["rounds"]
    os.makedirs(out_dir, exist_ok=True)
    xtr, ytr, xte, yte, labels_np = get_data(cfg, device)
    val_idx, clients = client_split(cfg, env, seed, labels_np)
    val_t = torch.tensor(val_idx, device=device)
    client_t = [torch.tensor(c, device=device) for c in clients]
    sched, sched_kw = make_schedule(cfg, env, seed)
    attackers = [m for m in sched.malicious if m not in disabled_attackers] if attack else []

    torch.manual_seed(seed)
    model = build_model(cfg["model"]).to(device)
    if init_vec is not None:
        set_vector(model, init_vec)
    vec0 = get_vector(model)

    clip_bound = cfg["clip_bound"] if defense == "clip_fedavg" else None
    run_cfg = {"env": env, "attack": attack, "defense": defense, "seed": seed, "rounds": rounds,
               "start_round": start_round, "disabled_attackers": list(disabled_attackers),
               "clip_bound": clip_bound, "schedule": sched.to_dict(),
               "client_sizes": [len(c) for c in clients],
               "class_distribution": D.class_distribution(clients, labels_np), "base": cfg}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "env.json"), "w") as f:
        json.dump({"commit": git_commit(), "python": platform.python_version(), "torch": torch.__version__,
                   "device": str(device), "platform": platform.platform()}, f, indent=1)

    http = requests.Session()
    meta = {"aggregation": defense, "clip_bound": clip_bound, "min_accept": cfg["min_accept"],
            "num_clients": cfg["num_clients"], "schedule": sched_kw, "start_round": start_round}
    http.post(server_url + "/reset", data=to_bytes(vec0), headers={"X-Meta": json.dumps(meta)}).raise_for_status()

    ckpt_dir = os.path.join(out_dir, "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    round_rows, client_rows = [], []
    virtual_elapsed, t_start, diverged_round = 0.0, time.time(), None

    for rnd in range(start_round, start_round + rounds):
        t_round = time.time()
        r = http.get(server_url + "/model")
        r.raise_for_status()
        gvec = from_bytes(r.content)
        version = int(r.headers["X-Version"])
        participants = json.loads(r.headers["X-Participants"])
        assert int(r.headers["X-Round"]) == rnd

        honest, train_sec = {}, {}
        for cid in participants:  # 같은 라운드의 모든 클라이언트는 같은 전역 모델에서 시작
            set_vector(model, gvec)
            t0 = time.time()
            local_train(model, xtr, ytr, client_t[cid], cfg, gen_seed=seed * 100003 + rnd * 101 + cid)
            train_sec[cid] = time.time() - t0
            honest[cid] = get_vector(model) - gvec

        forged = collusion({c: honest[c] for c in participants if c in attackers}, cfg["attack_scale"])
        submitted = {c: forged.get(c, honest[c]) for c in participants}

        responses = {}
        for cid in participants:
            vf = sched.virtual_finish(cid, rnd)
            m = {"client_id": cid, "round": rnd, "version": version, "num_samples": len(clients[cid]),
                 "virtual_finish": vf}
            resp = http.post(server_url + "/update", data=to_bytes(submitted[cid]), headers={"X-Meta": json.dumps(m)})
            responses[cid] = (resp.json(), vf)

        agg = http.post(server_url + "/aggregate").json()
        r = http.get(server_url + "/model")
        set_vector(model, from_bytes(r.content))
        val_acc, val_loss, _ = evaluate(model, xtr[val_t], ytr[val_t])

        accepted = agg["accepted"]
        n_late = sum(1 for c in participants if responses[c][0]["reason"] == "late")
        mal_sel = [c for c in participants if c in sched.malicious]
        mal_acc = [c for c in accepted if c in sched.malicious]
        vf_all = [responses[c][1] for c in participants]
        round_virtual = sched.deadline if n_late else max(vf_all)
        virtual_elapsed += round_virtual
        round_rows.append({
            "round": rnd, "selected": len(participants), "accepted": len(accepted), "late_excluded": n_late,
            "other_excluded": len(participants) - len(accepted) - n_late,
            "malicious_selected": len(mal_sel), "malicious_accepted": len(mal_acc),
            "overall_malicious_ratio": round(len(sched.malicious) / cfg["num_clients"], 4),
            "accepted_malicious_ratio": round(len(mal_acc) / len(accepted), 4) if accepted else "N/A",
            "updated": agg["updated"], "reason": agg["reason"], "aggregate_norm": agg.get("aggregate_norm", ""),
            "val_acc": round(val_acc, 4), "val_loss": round(val_loss, 6),
            "virtual_round_time": round(round_virtual, 4), "virtual_elapsed": round(virtual_elapsed, 4),
            "real_round_sec": round(time.time() - t_round, 3),
        })
        for cid in participants:
            res, vf = responses[cid]
            client_rows.append({
                "round": rnd, "client_id": cid, "malicious": cid in sched.malicious,
                "attacked": cid in forged, "num_samples": len(clients[cid]),
                "honest_norm": round(honest[cid].norm().item(), 6),
                "submitted_norm": round(submitted[cid].norm().item(), 6),
                "after_defense_norm": round(agg["after_defense_norm"].get(str(cid), float("nan")), 6)
                if str(cid) in agg["after_defense_norm"] else "",
                "cosine_to_median": round(agg.get("cosine_to_median", {}).get(str(cid), float("nan")), 6)
                if str(cid) in agg.get("cosine_to_median", {}) else "",
                "clipped": agg.get("clipped", {}).get(str(cid), ""),
                "accepted": res["accepted"], "reason": res["reason"],
                "virtual_finish": round(vf, 4), "real_train_sec": round(train_sec[cid], 3),
            })
        log(f"  r{rnd:03d} acc={len(accepted)}/{len(participants)} upd={agg['updated']} "
            f"val={val_acc:.2f}% loss={val_loss:.3f} {agg['reason']}")

        if rnd % cfg["checkpoint_every"] == 0 or rnd == start_round + rounds - 1:
            torch.save({"round": rnd, "vector": get_vector(model)}, os.path.join(ckpt_dir, f"round_{rnd:03d}.pt"))

        # 모든 제출이 NaN/Inf로 제외되거나 검증 손실이 NaN이면 발산으로 보고 중단
        reasons = [responses[c][0]["reason"] for c in participants]
        if not np.isfinite(val_loss) or (reasons and all(x == "non_finite" for x in reasons)):
            diverged_round = rnd
            log(f"  발산: {rnd}라운드에서 중단")
            break

    test_acc, test_loss, test_cls = evaluate(model, xte, yte)
    client_acc = []
    for cid, idx in enumerate(client_t):
        a, _, _ = evaluate(model, xtr[idx], ytr[idx])
        client_acc.append({"client_id": cid, "malicious": cid in sched.malicious, "train_acc": round(a, 4)})

    write_csv(os.path.join(out_dir, "rounds.csv"), round_rows)
    write_csv(os.path.join(out_dir, "clients.csv"), client_rows)
    write_csv(os.path.join(out_dir, "class_acc.csv"),
              [{"class": i, "test_acc": round(a, 4)} for i, a in enumerate(test_cls)])
    write_csv(os.path.join(out_dir, "client_acc.csv"), client_acc)
    summary = {"env": env, "attack": attack, "defense": defense, "seed": seed,
               "rounds_planned": rounds, "rounds_done": len(round_rows), "diverged_round": diverged_round,
               "final_val_acc": round_rows[-1]["val_acc"] if round_rows else None,
               "final_test_acc": round(test_acc, 4), "final_test_loss": test_loss,
               "malicious": sched.malicious, "delayed": sched.delayed,
               "virtual_elapsed": round(virtual_elapsed, 3), "real_sec": round(time.time() - t_start, 2)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def run_name(env, attack, defense, seed):
    return f"{env}-{'attack' if attack else 'normal'}-{defense}-s{seed}"


def main():
    ap = argparse.ArgumentParser(description="연합학습 실험 1회 실행")
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--env", default="baseline", choices=["baseline", "delay", "partial", "label_skew"])
    ap.add_argument("--attack", action="store_true")
    ap.add_argument("--defense", default="fedavg", choices=DEFENSES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--out", default="experiments/runs")
    ap.add_argument("--server", help="이미 떠 있는 서버 URL (없으면 새로 띄움)")
    a = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, a.config))
    srv = None if a.server else ServerProcess()
    url = a.server or srv.url
    try:
        out = os.path.join(ROOT, a.out, run_name(a.env, a.attack, a.defense, a.seed))
        print(json.dumps(run_experiment(cfg, a.env, a.attack, a.defense, a.seed, out, url, a.rounds),
                         ensure_ascii=False, indent=1))
    finally:
        if srv:
            srv.close()


if __name__ == "__main__":
    main()
