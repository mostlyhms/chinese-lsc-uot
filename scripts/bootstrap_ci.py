#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap_ci.py — 给 table3 / table4 的 word-level Spearman 加自助法置信区间。

为什么需要：ChiWUG 只有 40 个目标词。`table3.py` 报的是 100 次交叉验证的均值，
但每次验证的测试集只有 8 个词，而且 100 次划分之间高度重叠——
那个均值很稳，却**不能**告诉你「换一批词还会不会是这个结论」。

这里做的是对 40 个词做 bootstrap 重采样（B 次，默认 2000），
给出每个方法的 Spearman 分布，以及**配对差值**（method - 基准）的分布。
配对差值才是回答「A 真的比 B 好吗」的东西：
如果差值的 95% CI 跨过 0，那结论就是「在 ChiWUG 上区分不开」。

超参数固定用 `table3.py` 交叉验证选出来的 most common param，
避免每次重采样都重新调参（那会把选择偏差混进 CI 里）。

用法：
    UOT_DATASET=dwug_zh python3 scripts/bootstrap_ci.py
    UOT_DATASET=dwug_zh python3 scripts/bootstrap_ci.py --level g   # table4 的语义广度指标
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_config import DATA_DIR, EMB_NAME  # noqa: E402
import metrics  # noqa: E402

# (显示名, 函数, kwargs) —— 超参用 table3/table4 的 most common param
METHODS_F = [
    ("f_SUS",   metrics.f_SUS,   {"reg_m": 1000}),
    ("f_OT",    metrics.f_OT,    {}),
    ("f_APD",   metrics.f_APD,   {}),
    ("f_PRT",   metrics.f_PRT,   {}),
    ("f_APDP",  metrics.f_APDP,  {"damping": 0.9}),
    ("f_LDR",   metrics.f_LDR,   {}),
    ("f_WiDiD", metrics.f_WiDiD, {"damping": 0.9}),
]
METHODS_G = [
    ("g_SUS",   metrics.g_SUS,   {"reg_m": 10}),
    ("g_vMF",   metrics.g_vMF,   {}),
    ("g_LDR",   metrics.g_LDR,   {}),
    ("g_WiDiD", metrics.g_WiDiD, {"damping": 0.8}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["f", "g"], default="f",
                    help="f = 变化幅度(table3) / g = 语义广度(table4)")
    ap.add_argument("-B", type=int, default=2000, help="bootstrap 次数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_table(f"{DATA_DIR}/stats/opt/stats_groupings.csv", quoting=csv.QUOTE_NONE)
    with open(f"embeddings/{EMB_NAME}", "rb") as f:
        src, tgt = pickle.load(f)

    words = [w for w in df["lemma"] if w in src]
    if args.level == "f":
        gold = np.array([dict(zip(df["lemma"], df["change_graded"]))[w] for w in words])
        methods = METHODS_F
        title = "f_* 变化幅度 (对应 table3, gold = change_graded)"
    else:
        def ent(s):
            p = np.array(eval(s)); p = p[p > 0]
            return float(-(p * np.log(p)).sum())
        d = {r["lemma"]: ent(r["cluster_prob_dist2"]) - ent(r["cluster_prob_dist1"])
             for _, r in df.iterrows()}
        gold = np.array([d[w] for w in words])
        methods = METHODS_G
        title = "g_* 语义广度 (对应 table4, gold = 熵差)"

    print(f"\n{title}   n={len(words)} 词, B={args.B}\n")

    preds = {}
    for name, fn, kw in methods:
        preds[name] = np.array([fn(np.array(src[w]), np.array(tgt[w]), **kw) for w in words])

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(words), size=(args.B, len(words)))
    boot = {}
    for name in preds:
        rs = []
        for i in range(args.B):
            g, p = gold[idx[i]], preds[name][idx[i]]
            if len(np.unique(g)) < 3 or len(np.unique(p)) < 3:
                continue
            r = spearmanr(g, p).correlation
            if not np.isnan(r):
                rs.append(r)
        boot[name] = np.array(rs)

    print(f"{'方法':<10} {'点估计':>8} {'95% CI':>18} {'CI 宽度':>9}")
    print("-" * 50)
    point = {}
    for name in preds:
        point[name] = spearmanr(gold, preds[name]).correlation
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        print(f"{name:<10} {point[name]:>8.3f}   [{lo:>6.3f}, {hi:>6.3f}] {hi-lo:>9.3f}")

    base = methods[0][0]
    print(f"\n配对差值：其他方法 − {base}（CI 跨 0 = 在 ChiWUG 上区分不开）")
    print("-" * 62)
    print(f"{'对比':<20} {'Δ':>7} {'95% CI':>18} {'结论':>12}")
    print("-" * 62)
    for name in preds:
        if name == base:
            continue
        d = boot[name] - boot[base]
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "区分不开" if lo <= 0 <= hi else ("显著更好" if lo > 0 else "显著更差")
        print(f"{name+' − '+base:<20} {point[name]-point[base]:>7.3f}   "
              f"[{lo:>6.3f}, {hi:>6.3f}] {verdict:>12}")
    print()


if __name__ == "__main__":
    main()
