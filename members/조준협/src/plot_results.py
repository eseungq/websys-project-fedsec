import os
import glob
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

# 한글 폰트 깨짐 방지 (Windows 맑은 고딕 우선 적용)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


CONDITION_LABELS = {
    ("none", "fedavg"): "정상 + FedAvg (기준)",
    ("none", "clipped_fedavg"): "정상 + 클리핑 FedAvg",
    ("none", "clipped_median"): "정상 + 클리핑 중앙값",
    ("collusion", "fedavg"): "공격 + FedAvg (무방어)",
    ("collusion", "clipped_fedavg"): "공격 + 클리핑 FedAvg",
    ("collusion", "clipped_median"): "공격 + 클리핑 중앙값",
}

CONDITION_COLORS = {
    ("none", "fedavg"): "#2b6cb0",
    ("none", "clipped_fedavg"): "#4299e1",
    ("none", "clipped_median"): "#319795",
    ("collusion", "fedavg"): "#e53e3e",
    ("collusion", "clipped_fedavg"): "#dd6b20",
    ("collusion", "clipped_median"): "#38a169",
}


def load_all_logs(log_dir="../save/logs", target_model=None, target_optimizer=None, target_epochs=None):
    files = sorted(glob.glob(os.path.join(log_dir, "log_*.json")))
    loaded_items = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        num_rounds = len(data.get("round_history", []))
        if num_rounds == 0:
            continue

        m_name = cfg.get("model", "cnn")
        o_name = cfg.get("optimizer", "adam")

        if target_model and m_name != target_model:
            continue
        if target_optimizer and o_name != target_optimizer:
            continue
        if target_epochs is not None and num_rounds != int(target_epochs):
            continue

        loaded_items.append((data, m_name, o_name, num_rounds))

    if not loaded_items:
        return [], None

    # 필터 미지정 시 가장 많은 실험이 수행된 단일 설정 그룹만 선택해 혼합 방지
    if target_model is None and target_optimizer is None and target_epochs is None:
        counts = defaultdict(int)
        for _, m_name, o_name, n_rnd in loaded_items:
            counts[(m_name, o_name, n_rnd)] += 1
        selected_setting = max(counts.items(), key=lambda x: x[1])[0]
        records = [item[0] for item in loaded_items if (item[1], item[2], item[3]) == selected_setting]
        return records, selected_setting

    first = loaded_items[0]
    return [item[0] for item in loaded_items], (first[1], first[2], first[3])


def plot_round_curves(records, setting_info=None, save_dir="../save/plots"):
    """운영 환경(env)별로 라운드에 따른 Test Accuracy 변화 곡선 그래프 생성"""
    os.makedirs(save_dir, exist_ok=True)
    by_env = defaultdict(list)
    for rec in records:
        env = rec.get("config", {}).get("env", "baseline")
        by_env[env].append(rec)

    subtitle = f" ({setting_info[0]}, {setting_info[1].upper()}, {setting_info[2]}R)" if setting_info else ""

    for env, env_records in by_env.items():
        cond_rounds = defaultdict(lambda: defaultdict(list))
        for rec in env_records:
            cfg = rec.get("config", {})
            key = (cfg.get("attack", "none"), cfg.get("defense", "fedavg"))
            for r_item in rec.get("round_history", []):
                cond_rounds[key][r_item["round"]].append(r_item["test_acc"])

        if not cond_rounds:
            continue

        plt.figure(figsize=(9, 5.5))
        for key, round_dict in sorted(cond_rounds.items()):
            rounds = sorted(round_dict.keys())
            mean_accs = [sum(round_dict[r]) / len(round_dict[r]) for r in rounds]
            label = CONDITION_LABELS.get(key, f"{key[0]} + {key[1]}")
            color = CONDITION_COLORS.get(key, None)
            linestyle = "--" if key[0] == "collusion" and key[1] == "fedavg" else "-"
            plt.plot(rounds, mean_accs, marker="o", markersize=4, linewidth=2, linestyle=linestyle, color=color, label=label)

        plt.title(f"[{env.upper()}] 환경 라운드별 테스트 정확도 비교{subtitle}", fontsize=13, fontweight="bold")
        plt.xlabel("Global Round", fontsize=11)
        plt.ylabel("Test Accuracy (%)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", fontsize=9.5)
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"curve_{env}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[저장 완료] 라운드 정확도 곡선: {out_path}")


def plot_final_bar_chart(records, setting_info=None, save_dir="../save/plots"):
    """환경 및 공격·방어 조합별 최종 테스트 정확도 비교 막대 그래프 생성"""
    os.makedirs(save_dir, exist_ok=True)
    grouped = defaultdict(list)
    for rec in records:
        cfg = rec.get("config", {})
        env = cfg.get("env", "baseline")
        cond = (cfg.get("attack", "none"), cfg.get("defense", "fedavg"))
        grouped[(env, cond)].append(float(rec.get("final_test_acc", 0.0)))

    if not grouped:
        return

    subtitle = f" ({setting_info[0]}, {setting_info[1].upper()}, {setting_info[2]}R)" if setting_info else ""
    envs = sorted(list(set(k[0] for k in grouped.keys())))
    conds = [c for c in CONDITION_LABELS.keys() if any((e, c) in grouped for e in envs)]

    x = list(range(len(envs)))
    width = 0.8 / max(len(conds), 1)

    plt.figure(figsize=(10, 5.5))
    for idx, cond in enumerate(conds):
        vals = []
        for e in envs:
            scores = grouped.get((e, cond), [])
            vals.append(sum(scores) / len(scores) if scores else 0.0)
        offsets = [xi + (idx - len(conds) / 2 + 0.5) * width for xi in x]
        bars = plt.bar(offsets, vals, width=width, label=CONDITION_LABELS[cond], color=CONDITION_COLORS.get(cond))
        for bar, v in zip(bars, vals):
            if v > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, [e.upper() for e in envs], fontsize=11, fontweight="bold")
    plt.ylabel("Final Test Accuracy (%)", fontsize=11)
    plt.title(f"운영 환경 및 공격·방어 조건별 최종 테스트 정확도 비교{subtitle}", fontsize=13, fontweight="bold")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "final_accuracy_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[저장 완료] 최종 정확도 비교 막대 그래프: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="연합학습 실험 결과 시각화 그래프 생성기")
    parser.add_argument("--log_dir", type=str, default="../save/logs")
    parser.add_argument("--save_dir", type=str, default="../save/plots")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    all_records, setting = load_all_logs(
        log_dir=args.log_dir,
        target_model=args.model,
        target_optimizer=args.optimizer,
        target_epochs=args.epochs,
    )
    if not all_records:
        print("저장된 로그 파일(../save/logs/log_*.json)이 없습니다. 먼저 실험을 실행해 주세요.")
    else:
        print(f"설정 {setting} 기준 총 {len(all_records)}개의 실험 로그를 불러왔습니다.")
        plot_round_curves(all_records, setting, args.save_dir)
        plot_final_bar_chart(all_records, setting, args.save_dir)