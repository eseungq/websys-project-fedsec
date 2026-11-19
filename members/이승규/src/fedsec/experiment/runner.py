"""정상 FedAvg 실험 1회: 클라이언트 전원 참여, 공격·방어 없음 (공통 기준 실행)."""

from pathlib import Path

import torch

from fedsec.aggregation.fedavg import apply_update, weighted_average
from fedsec.aggregation.norms import update_norm
from fedsec.client.local_train import local_update
from fedsec.data.cifar10 import load_cifar10
from fedsec.data.split import load_split
from fedsec.experiment.config import ExperimentConfig
from fedsec.experiment.evaluate import evaluate
from fedsec.experiment.runtime import client_generator, environment_info, pick_device, set_seed, timed_now
from fedsec.logger.run_log import RunLog
from fedsec.models.cnn import build_model


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def run_fedavg(cfg: ExperimentConfig, out_dir: Path, repo_root: Path, evaluate_test: bool = False) -> dict:
    """공식 테스트셋은 evaluate_test=True일 때만 평가한다 (설정을 확정한 뒤 한 번)."""
    set_seed(cfg.seed)
    device = pick_device()
    train_all, test = load_cifar10(repo_root / cfg.data_root, device)
    split = load_split(repo_root / cfg.split)
    if split["num_images"] != len(train_all):
        raise ValueError("분할 파일의 이미지 수가 CIFAR-10 학습 데이터와 다릅니다.")
    val = train_all.subset(split["val"])
    clients = [train_all.subset(indices) for indices in split["clients"]]

    model = build_model(cfg.model).to(device)
    global_state = _clone_state(model)
    env = environment_info(device)
    log = RunLog(out_dir, {**cfg.to_dict(), "num_clients": len(clients)}, env)
    print(f"device={env['device']} gpu={env['gpu']} commit={env['commit'][:7]} clients={len(clients)}")

    round_times = []
    for round_idx in range(1, cfg.rounds + 1):
        t_start = timed_now(device)
        results = [
            local_update(model, global_state, data, cfg.train, client_generator(cfg.seed, round_idx, cid))
            for cid, data in enumerate(clients)
        ]
        t_trained = timed_now(device)
        norms = [update_norm(r.update) for r in results]

        aggregated = weighted_average([r.update for r in results], [r.num_samples for r in results])
        global_state = apply_update(global_state, aggregated)
        model.load_state_dict(global_state)
        val_loss, val_acc = evaluate(model, val, cfg.eval_bs)
        t_end = timed_now(device)

        row = {
            "round": round_idx,
            "train_loss": round(sum(r.train_loss for r in results) / len(results), 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "norm_min": round(min(norms), 4),
            "norm_mean": round(sum(norms) / len(norms), 4),
            "norm_max": round(max(norms), 4),
            "train_sec": round(t_trained - t_start, 3),
            "total_sec": round(t_end - t_start, 3),
        }
        log.log_round(row)
        log.log_clients([
            {"round": round_idx, "client": cid, "num_samples": r.num_samples,
             "train_loss": round(r.train_loss, 4), "update_norm": round(norm, 4)}
            for cid, (r, norm) in enumerate(zip(results, norms))
        ])
        round_times.append(row["total_sec"])
        print(" ".join(f"{k}={v}" for k, v in row.items()))

    later = round_times[1:] or round_times
    summary = {
        "final_val_acc": row["val_acc"],
        "first_round_sec": round_times[0],
        "mean_round_sec_after_first": round(sum(later) / len(later), 3),
        "total_sec": round(sum(round_times), 1),
    }
    if evaluate_test:
        model.load_state_dict(global_state)
        test_loss, test_acc = evaluate(model, test, cfg.eval_bs)
        summary = {**summary, "final_test_acc": round(test_acc, 4), "final_test_loss": round(test_loss, 4)}
    log.write_summary(summary)
    (out_dir / "models").mkdir(exist_ok=True)
    torch.save(global_state, out_dir / "models" / "final_model.pt")
    return summary
