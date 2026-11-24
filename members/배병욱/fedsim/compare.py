"""비교표: 공격 피해 · 방어 개선 · 정상 손실(%p)과 방어 진단 요약.

- 공격 피해 = 정상 FedAvg − 공격 FedAvg
- 방어 개선 = 공격+방어 − 공격 FedAvg
- 정상 손실 = 정상 FedAvg − 정상+방어
같은 환경·시드의 최종 테스트 정확도로 계산하고 시드별 값과 평균을 함께 적는다. 음수도 그대로.

방어 진단(이 코드에서 추가): 공격 실행의 clients.csv에서
- 클리핑이 실제로 자른 비율을 정상/악성으로 나눠 집계 (상한이 악성만 잡는지)
- 좌표별 중앙값과의 코사인 유사도 평균을 정상/악성으로 나눠 집계 (서버가 볼 수 있는 이상 신호)

예: python -m fedsim.compare --dir experiments/matrix
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean

from .runner import ROOT


def load(dir_):
    runs = {}
    for p in glob.glob(os.path.join(dir_, "*", "summary.json")):
        with open(p) as f:
            s = json.load(f)
        runs[(s["env"], s["attack"], s["defense"], s["seed"])] = (s, os.path.dirname(p))
    return runs


def diagnostics(run_dir):
    path = os.path.join(run_dir, "clients.csv")
    if not os.path.exists(path):
        return {}
    clip = defaultdict(list)
    cos = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("accepted") != "True":
                continue
            who = "malicious" if r["malicious"] == "True" else "honest"
            if r.get("clipped") in ("True", "False"):
                clip[who].append(r["clipped"] == "True")
            if r.get("cosine_to_median") not in (None, ""):
                cos[who].append(float(r["cosine_to_median"]))
    out = {}
    for who in ("honest", "malicious"):
        if clip[who]:
            out[f"clip_rate_{who}"] = round(100 * sum(clip[who]) / len(clip[who]), 1)
        if cos[who]:
            out[f"cos_{who}"] = round(mean(cos[who]), 3)
    return out


def fmt(v):
    return "–" if v is None else f"{v:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/matrix")
    a = ap.parse_args()
    d = os.path.join(ROOT, a.dir)
    runs = load(d)
    acc = {k: v[0]["final_test_acc"] for k, v in runs.items()}
    envs = sorted({k[0] for k in runs})
    seeds = sorted({k[3] for k in runs})

    lines = ["# 비교표 (최종 테스트 정확도 기준, %p)", "",
             "| 환경 | 시드 | 정상 FedAvg | 공격 FedAvg | 공격 피해 | 방어 | 공격+방어 | 방어 개선 | 정상+방어 | 정상 손실 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows, agg = [], defaultdict(list)
    for e in envs:
        for s in seeds:
            nf, af = acc.get((e, False, "fedavg", s)), acc.get((e, True, "fedavg", s))
            dmg = None if nf is None or af is None else nf - af
            for dfn in ("clip_fedavg", "median"):
                ad, nd = acc.get((e, True, dfn, s)), acc.get((e, False, dfn, s))
                imp = None if ad is None or af is None else ad - af
                loss = None if nf is None or nd is None else nf - nd
                lines.append(f"| {e} | {s} | {nf} | {af} | {fmt(dmg)} | {dfn} | {ad} | {fmt(imp)} | {nd} | {fmt(loss)} |")
                rows.append({"env": e, "seed": s, "defense": dfn, "normal_fedavg": nf, "attack_fedavg": af,
                             "attack_damage_pp": dmg, "attack_defense": ad, "defense_gain_pp": imp,
                             "normal_defense": nd, "normal_loss_pp": loss})
                for k, v in (("dmg", dmg), ("imp", imp), ("loss", loss)):
                    if v is not None:
                        agg[(e, dfn, k)].append(v)
    lines += ["", "## 시드 평균", "", "| 환경 | 방어 | 공격 피해 | 방어 개선 | 정상 손실 | 시드 수 |", "|---|---|---|---|---|---|"]
    for e in envs:
        for dfn in ("clip_fedavg", "median"):
            m = {k: (mean(agg[(e, dfn, k)]) if agg[(e, dfn, k)] else None) for k in ("dmg", "imp", "loss")}
            n = len(agg[(e, dfn, "imp")])
            lines.append(f"| {e} | {dfn} | {fmt(m['dmg'])} | {fmt(m['imp'])} | {fmt(m['loss'])} | {n} |")

    lines += ["", "## 발산", ""]
    for k, (s, _) in sorted(runs.items()):
        if s["diverged_round"]:
            lines.append(f"- {k[0]} / {'공격' if k[1] else '정상'} / {k[2]} / 시드 {k[3]}: "
                         f"{s['diverged_round']}라운드 발산 (최종 정확도는 발산 직전 모델 기준)")

    lines += ["", "## 방어 진단 (공격 실행, 채택된 업데이트 기준)", "",
              "| 실행 | 클리핑 비율 정상 | 클리핑 비율 악성 | 중앙값 코사인 정상 | 중앙값 코사인 악성 |", "|---|---|---|---|---|"]
    for k, (s, rd) in sorted(runs.items()):
        if not k[1]:
            continue
        dg = diagnostics(rd)
        lines.append(f"| {os.path.basename(rd)} | {dg.get('clip_rate_honest', '–')} | {dg.get('clip_rate_malicious', '–')} "
                     f"| {dg.get('cos_honest', '–')} | {dg.get('cos_malicious', '–')} |")

    os.makedirs(os.path.join(d, "_tables"), exist_ok=True)
    with open(os.path.join(d, "_tables", "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if rows:
        with open(os.path.join(d, "_tables", "comparison.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
