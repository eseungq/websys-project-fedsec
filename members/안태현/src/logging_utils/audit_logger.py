import os
import csv
import json
import torch
from PIL import Image
import numpy as np


class SimulationLogger:
    def __init__(self, base_dir="./outputs"):
        self.base_dir = base_dir
        self.logs_dir = os.path.join(base_dir, "logs")
        self.ckpt_dir = os.path.join(base_dir, "checkpoints")
        self.dlg_dir = os.path.join(base_dir, "dlg_images")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.dlg_dir, exist_ok=True)

    def init_experiment_logs(self, exp_id, config, distribution_report):
        exp_ckpt_dir = os.path.join(self.ckpt_dir, exp_id)
        os.makedirs(exp_ckpt_dir, exist_ok=True)

        # 실험 설정 및 데이터 분포 저장 (재현성 보장)
        meta = {"exp_id": exp_id, "config": config, "data_distribution": distribution_report}
        with open(os.path.join(exp_ckpt_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        client_csv = os.path.join(self.logs_dir, f"{exp_id}_client_audit.csv")
        with open(client_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "exp_id", "round", "client_id", "environment", "attack", "defense",
                "num_samples", "is_selected", "is_malicious_role",
                "pre_attack_norm", "post_attack_norm", "post_defense_norm",
                "decision", "reason", "virtual_time_sec", "real_time_sec", "local_loss"
            ])

        round_csv = os.path.join(self.logs_dir, f"{exp_id}_round_summary.csv")
        with open(round_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "exp_id", "round", "selected_count", "accepted_count", "rejected_count",
                "total_malicious_ratio", "accepted_malicious_ratio",
                "model_updated", "update_reason", "val_accuracy", "val_loss",
                "virtual_elapsed_sec", "real_elapsed_sec"
            ])

    def log_client_rows(self, exp_id, rows):
        client_csv = os.path.join(self.logs_dir, f"{exp_id}_client_audit.csv")
        with open(client_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow([
                    exp_id, r["round"], r["client_id"], r["environment"], r["attack"], r["defense"],
                    r["num_samples"], r["is_selected"], r["is_malicious_role"],
                    round(r["pre_attack_norm"], 4) if r["pre_attack_norm"] is not None else "-",
                    round(r["post_attack_norm"], 4) if r["post_attack_norm"] is not None else "-",
                    round(r["post_defense_norm"], 4) if r["post_defense_norm"] is not None else "-",
                    r["decision"], r["reason"],
                    round(r["virtual_time_sec"], 2), round(r["real_time_sec"], 3),
                    round(r["local_loss"], 4) if r["local_loss"] is not None else "-"
                ])

    def log_round_summary(self, exp_id, summary):
        round_csv = os.path.join(self.logs_dir, f"{exp_id}_round_summary.csv")
        with open(round_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                exp_id, summary["round"], summary["selected_count"],
                summary["accepted_count"], summary["rejected_count"],
                summary["total_malicious_ratio"], summary["accepted_malicious_ratio"],
                summary["model_updated"], summary["update_reason"],
                round(summary["val_accuracy"], 4), round(summary["val_loss"], 4),
                round(summary["virtual_elapsed_sec"], 2), round(summary["real_elapsed_sec"], 3)
            ])

    def save_round_checkpoint(self, exp_id, round_idx, payload):
        """특정 라운드 재실행(Re-run)을 위해 시작 전역 모델 및 클라이언트 업데이트 상태 저장"""
        path = os.path.join(self.ckpt_dir, exp_id, f"round_{round_idx}.pt")
        torch.save(payload, path)

    def load_round_checkpoint(self, exp_id, round_idx):
        path = os.path.join(self.ckpt_dir, exp_id, f"round_{round_idx}.pt")
        return torch.load(path, map_location="cpu")

    def save_dlg_result(self, run_id, sample_index, noise_mode, noise_std, iterations, mse, gt_tensor, rec_tensor):
        """원본·복원 이미지 PNG 저장 및 DLG 복원 로그 기록"""
        gt_path = os.path.join(self.dlg_dir, f"{run_id}_gt.png")
        rec_path = os.path.join(self.dlg_dir, f"{run_id}_rec.png")

        self._tensor_to_png(gt_tensor, gt_path)
        self._tensor_to_png(rec_tensor, rec_path)

        dlg_csv = os.path.join(self.logs_dir, "dlg_reconstruction_log.csv")
        file_exists = os.path.exists(dlg_csv)
        with open(dlg_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["run_id", "sample_index", "noise_mode", "noise_std", "iterations", "mse", "gt_image", "rec_image"])
            writer.writerow([run_id, sample_index, noise_mode, noise_std, iterations, round(mse, 6), gt_path, rec_path])

        return gt_path, rec_path

    @staticmethod
    def _tensor_to_png(tensor_3chw, save_path):
        arr = (tensor_3chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).resize((128, 128), Image.NEAREST).save(save_path)


def compute_comparison_metrics(normal_fedavg_acc, attack_fedavg_acc=None, attack_defense_acc=None, normal_defense_acc=None):
    """
    동일 환경·시드 기준 최종 테스트 정확도 차이(%p) 계산 (음수 결과도 그대로 기록):
    - 공격 피해 = 정상 FedAvg - 공격 FedAvg
    - 방어 개선 = 공격+방어 - 공격 FedAvg
    - 정상 손실 = 정상 FedAvg - 정상+방어
    """
    result = {
        "attack_damage_pp": round((normal_fedavg_acc - attack_fedavg_acc) * 100, 2) if attack_fedavg_acc is not None else None,
        "defense_gain_pp": round((attack_defense_acc - attack_fedavg_acc) * 100, 2) if (attack_defense_acc is not None and attack_fedavg_acc is not None) else None,
        "normal_loss_pp": round((normal_fedavg_acc - normal_defense_acc) * 100, 2) if normal_defense_acc is not None else None,
    }
    return result