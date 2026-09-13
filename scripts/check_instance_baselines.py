#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_instance_baselines.py — 给 table2 的两个指标做「平凡基线」体检。

任何评测指标在用之前都该问一句：**一个什么都不懂的基线能拿多少分？**
如果平凡基线就能接近满分，那这个指标测的不是你以为的那个东西。

table2.py 报两个数：
    instance-level  所有用例上，gold tau 与方法 tau 的 Spearman
    sense-level     按 gold tau 分组、组内取方法 tau 均值后，再与 gold tau 求 Spearman

gold tau = 该用例所属义项的 log(p2/p1)：这个义项在后期比前期频繁多少。

**问题**：一个义项「变多了」，它的用例自然大多来自后期。所以「这条用例属于哪一期」
这个与语义完全无关的信号，本身就和 gold tau 高度相关。指标必须能把它排除掉才有意义。

本脚本比较四个基线与真实方法：
    ① 时期指示器   前期一律 −1、后期一律 +1。零语义信息
    ② 纯随机
    ③ 期内随机 + 期间偏移   ①的加噪版
    ④/⑤ 真 SUS

用法：
    export UOT_DATASET=dwug_zh
    python3 scripts/check_instance_baselines.py
    python3 scripts/check_instance_baselines.py --emb dwug_zh_bert-base-chinese_L5_embeddings.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_config import EMB_NAME  # noqa: E402
import metrics  # noqa: E402
from run_regm_ablation import load_instance_gold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--emb", default=None)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb = args.emb or EMB_NAME
    df, inst2cluster, inst2gold = load_instance_gold()
    with open(ROOT / "embeddings" / emb, "rb") as f:
        src, tgt = pickle.load(f)
    lemmas = [w for w in df["lemma"] if w in src]
    rng = np.random.default_rng(args.seed)

    def build(fn):
        """按 I1 逐词算分，再按位置删掉 cluster == -1。"""
        d = {}
        for w in lemmas:
            p = np.asarray(fn(w), dtype=float)
            keep = [i for i, c in enumerate(inst2cluster[w]) if c != -1]
            d[w] = p[keep]
        return d

    preds = {
        "① 时期指示器（零语义）": build(lambda w: np.r_[-np.ones(len(src[w])), np.ones(len(tgt[w]))]),
        "② 纯随机":              build(lambda w: rng.normal(size=len(src[w]) + len(tgt[w]))),
        "③ 期内随机+期间偏移":    build(lambda w: np.r_[rng.normal(-1, 1, len(src[w])),
                                                      rng.normal(1, 1, len(tgt[w]))]),
        "④ SUS (reg_m=1000)":   build(lambda w: np.append(
            *metrics.tau_SUS(np.array(src[w]), np.array(tgt[w]), reg_m=1000))),
        "⑤ SUS (reg_m=20)":     build(lambda w: np.append(
            *metrics.tau_SUS(np.array(src[w]), np.array(tgt[w]), reg_m=20))),
    }

    def score(ls, d, level):
        g = np.concatenate([inst2gold[w] for w in ls])
        m = np.concatenate([d[w] for w in ls])
        if level == "instance":
            return spearmanr(g, m).correlation
        t = pd.DataFrame({"g": g, "m": m}).groupby("g")["m"].mean()
        return spearmanr(t.values, t.index.values).correlation if len(t) > 2 else np.nan

    print(f"[info] {emb}   {len(lemmas)} 词\n")
    print(f"{'预测器':<26}{'instance ρ':>12}{'sense ρ':>10}")
    print("-" * 48)
    for name, d in preds.items():
        print(f"{name:<26}{score(lemmas, d, 'instance'):>12.3f}{score(lemmas, d, 'sense'):>10.3f}")

    # 配对检验：平凡基线 vs 真 SUS
    bidx = rng.integers(0, len(lemmas), size=(args.B, len(lemmas)))
    base, real = preds["① 时期指示器（零语义）"], preds["④ SUS (reg_m=1000)"]
    print(f"\n配对差值：① 时期指示器 − ④ SUS   (B={args.B}，按词重采样)")
    print("-" * 66)
    bad = False
    for level in ("instance", "sense"):
        a = np.array([score([lemmas[j] for j in ix], base, level) for ix in bidx])
        b = np.array([score([lemmas[j] for j in ix], real, level) for ix in bidx])
        d = a - b
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = score(lemmas, base, level) - score(lemmas, real, level)
        if lo > 0:
            v = "⚠ 平凡基线显著更高 —— 该指标不可用"
            bad = True
        elif hi < 0:
            v = "✓ SUS 显著更高 —— 指标有效"
        else:
            v = "⚠ 区分不开 —— 指标分辨力不足"
            bad = True
        print(f"  {level:<9} Δ={pt:>+7.3f}  [{lo:>+6.3f}, {hi:>+6.3f}]   {v}")
    print()
    if bad:
        print("结论：至少一个指标没能把「用例属于哪一期」这个混杂因素排除掉。")
        print("      在这样的指标上调超参，会奖励**退化**而不是奖励语义归因能力。")


if __name__ == "__main__":
    main()
