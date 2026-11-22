import os
import glob
import json
import csv
from collections import defaultdict


def compute_security_comparison_table(log_dir="../save/logs"):
    """
    저장된 로그 파일들로부터 환경·시드별 정량 지표(%p) 자동 계산:
    - 공격 피해 = 정상 FedAvg - 공격 FedAvg
    - 방어 개선 = 공격+방어 - 공격 FedAvg
    - 정상 손실 = 정상 FedAvg - 정상+방어
    """
    files = sorted(glob.glob(os.path.join(log_dir, "log_*.json")))
    raw_records = []
    # lookup[(env, seed, attack, defense)] = final_test_acc
    lookup = {}

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        env = cfg.get("env", "baseline")
        attack = cfg.get("attack", "none")
        defense = cfg.get("defense", "fedavg")
        seed = int(cfg.get("seed", 1))
        acc = float(data.get("final_test_acc", 0.0))

        lookup[(env, seed, attack, defense)] = acc
        raw_records.append({
            "exp_id": data.get("exp_id"),
            "env": env,
            "attack": attack,
            "defense": defense,
            "seed": seed,
            "final_test_acc": acc,
        })

    envs = sorted(list(set(k[0] for k in lookup.keys())))
    seeds = sorted(list(set(k[1] for k in lookup.keys())))
    defenses = ["clipped_fedavg", "clipped_median"]

    comparison_rows = []
    for env in envs:
        for defense in defenses:
            seed_rows = []
            for seed in seeds:
                norm_fed = lookup.get((env, seed, "none", "fedavg"))
                att_fed = lookup.get((env, seed, "collusion", "fedavg"))
                att_def = lookup.get((env, seed, "collusion", defense))
                norm_def = lookup.get((env, seed, "none", defense))

                if all(v is None for v in [norm_fed, att_fed, att_def, norm_def]):
                    continue

                damage = round(norm_fed - att_fed, 2) if (norm_fed is not None and att_fed is not None) else None
                gain = round(att_def - att_fed, 2) if (att_def is not None and att_fed is not None) else None
                loss = round(norm_fed - norm_def, 2) if (norm_fed is not None and norm_def is not None) else None

                row = {
                    "env": env,
                    "defense": defense,
                    "seed": str(seed),
                    "normal_fedavg": norm_fed,
                    "attack_fedavg": att_fed,
                    "attack_defense": att_def,
                    "normal_defense": norm_def,
                    "attack_damage": damage,
                    "defense_gain": gain,
                    "normal_loss": loss,
                }
                comparison_rows.append(row)
                seed_rows.append(row)

            # 시드 2개 이상일 때 평균 행 자동 추가
            if len(seed_rows) >= 2:
                def avg_field(key):
                    vals = [r[key] for r in seed_rows if r[key] is not None]
                    return round(sum(vals) / len(vals), 2) if vals else None

                comparison_rows.append({
                    "env": env,
                    "defense": defense,
                    "seed": "평균(Mean)",
                    "normal_fedavg": avg_field("normal_fedavg"),
                    "attack_fedavg": avg_field("attack_fedavg"),
                    "attack_defense": avg_field("attack_defense"),
                    "normal_defense": avg_field("normal_defense"),
                    "attack_damage": avg_field("attack_damage"),
                    "defense_gain": avg_field("defense_gain"),
                    "normal_loss": avg_field("normal_loss"),
                })

    return comparison_rows, raw_records


if __name__ == "__main__":
    rows, _ = compute_security_comparison_table("../save/logs")
    os.makedirs("../save/tables", exist_ok=True)
    csv_path = "../save/tables/security_evaluation_summary.csv"

    fieldnames = [
        "env", "defense", "seed", "normal_fedavg", "attack_fedavg",
        "attack_defense", "normal_defense", "attack_damage", "defense_gain", "normal_loss"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== 정량 평가표 생성 완료 ({len(rows)}행): {csv_path} ===")
    for r in rows:
        print(r)