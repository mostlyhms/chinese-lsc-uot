#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_regm_ablation.py — UOT 的边际松弛系数 reg_m 该按哪一层的表现来选？

背景：`reg_m` 是 UOT 允许「沙搬不完 / 坑填不满」的罚金力度。罚金越大越接近平衡 OT。
上游 `table3.py`（word-level）和 `table2.py`（instance-level）**各自独立**地
在 [10,20,50,100,200,500,1000] 上做交叉验证，两边都只看自己那一层的 Spearman。

问题在于：word-level 的 ρ 在 reg_m 跨两个数量级时几乎是平的
（xl-lexeme：reg_m 10→2000，ρ 只在 0.774–0.786 之间晃），
于是 CV 会在一个平台上**近乎随机地**挑一个值，而它挑中的往往是高值。
高 reg_m 下未搬运质量只剩 0.2%——**SUS 的归因信号几乎被掐灭了**，
而那恰恰是 UOT 相对普通 OT 的全部卖点。

这个脚本把两层放在同一个 reg_m 轴上扫，看两者的最优点是否分离。

诊断列：
    未搬运质量  = 1 − T.sum()，UOT 真正「没搬」的质量占比。趋近 0 就等于退化成平衡 OT
    退化实例占比 = |SUS| < 1e-6 的用例比例。这些用例的 SUS 没有任何区分度

用法：
    export UOT_DATASET=dwug_zh
    python3 scripts/run_regm_ablation.py
    python3 scripts/run_regm_ablation.py --emb dwug_zh_bert-base-chinese_L5_embeddings.pkl
"""
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import ot
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import dataset_config  # noqa: E402
from dataset_config import DATASET, DATA_DIR, EMB_NAME  # noqa: E402
import metrics  # noqa: E402
from _eval_common import ci  # noqa: E402

GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--emb", default=None, help="embeddings 文件名（默认用 EMB_NAME）")
    ap.add_argument("--grid", default=None, help="逗号分隔的 reg_m 网格")
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def str2list(s):
    # table2.py 用的是 eval()，这里换成 literal_eval：同样结果，但不执行任意代码
    import ast
    return np.array(ast.literal_eval(s), dtype=float)


def load_instance_gold():
    """完全照搬 table2.py 的 gold 构造：每个义项的 log(p2/p1)，用例继承所属义项的值。"""
    df = pd.read_table(f"{DATA_DIR}/stats/opt/stats_groupings.csv", quoting=csv.QUOTE_NONE)
    df["cluster_prob_dist1"] = df["cluster_prob_dist1"].apply(str2list)
    df["cluster_prob_dist2"] = df["cluster_prob_dist2"].apply(str2list)

    word_cluster2tau, allv = {}, []
    for _, row in df.iterrows():
        with np.errstate(divide="ignore", invalid="ignore"):
            taus = np.log(row["cluster_prob_dist2"] / row["cluster_prob_dist1"])
        allv.extend(taus)
        word_cluster2tau[row["lemma"]] = {c: t for c, t in enumerate(taus)}
    fin = np.array(allv)[~np.isinf(allv)]
    hi, lo = np.max(fin), np.min(fin)
    for w, d in word_cluster2tau.items():
        for c, v in d.items():
            if v == np.inf:
                d[c] = hi
            elif v == -np.inf:
                d[c] = lo

    inst2cluster, inst2gold = {}, {}
    for lemma in word_cluster2tau:
        cl = pd.read_table(f"{DATA_DIR}/clusters/opt/{lemma}.csv")["cluster"].tolist()
        inst2cluster[lemma] = cl
        inst2gold[lemma] = np.array([word_cluster2tau[lemma][c] for c in cl if c != -1])
    return df, inst2cluster, inst2gold


def sus_full(u, v, reg_m):
    """返回 (每条用例的 SUS, 未搬运质量占比)。与 utils.calc_sus 同式，只是多回传诊断量。"""
    u = np.asarray(u, float); v = np.asarray(v, float)
    a = np.ones(len(u)) / len(u); b = np.ones(len(v)) / len(v)
    un = u / np.linalg.norm(u, axis=1, keepdims=True)
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    C = 1 - un @ vn.T
    T = ot.unbalanced.mm_unbalanced(a, b, C, reg_m=reg_m, div="l2")
    us = -(a - T.sum(1)) / a
    vs = (b - T.sum(0)) / b
    return np.append(us, vs), 1.0 - T.sum()


def main():
    args = parse_args()
    grid = [float(x) for x in args.grid.split(",")] if args.grid else GRID
    emb = args.emb or EMB_NAME
    df, inst2cluster, inst2gold = load_instance_gold()
    with open(ROOT / "embeddings" / emb, "rb") as f:
        src, tgt = pickle.load(f)

    lemmas = [w for w in df["lemma"] if w in src]
    word_gold = np.array([dict(zip(df["lemma"], df["change_graded"]))[w] for w in lemmas])
    print(f"[info] {emb}\n[info] {len(lemmas)} 词, B={args.B}\n")

    # 对齐硬校验（I1）：clusters 行数必须等于 len(u)+len(v)
    for w in lemmas:
        n = len(src[w]) + len(tgt[w])
        if len(inst2cluster[w]) != n:
            sys.exit(f"[FATAL] {w}: clusters {len(inst2cluster[w])} 行 vs 向量 {n} 条。I1 已破坏。")

    rng = np.random.default_rng(args.seed)
    bidx = rng.integers(0, len(lemmas), size=(args.B, len(lemmas)))   # 按**词**重采样

    rows, boots = [], {}
    print(f"{'reg_m':>7}{'word ρ':>9}{'instance ρ':>12}{'sense ρ':>9}"
          f"{'未搬运质量':>11}{'退化实例':>10}")
    print("-" * 60)
    for rm in grid:
        per_lemma_sus, per_lemma_word, unmatched, degen = {}, [], [], []
        for w in lemmas:
            taus, um = sus_full(src[w], tgt[w], rm)
            unmatched.append(um)
            degen.append(np.mean(np.abs(taus) < 1e-6))
            # word-level：f_SUS = |mean(前期) − mean(后期)|
            n_u = len(src[w])
            per_lemma_word.append(abs(taus[:n_u].mean() - taus[n_u:].mean()))
            # instance-level：按位置删掉 cluster == -1（I1）
            dele = [i for i, c in enumerate(inst2cluster[w]) if c == -1]
            per_lemma_sus[w] = np.delete(taus, dele)

        wr = spearmanr(word_gold, np.array(per_lemma_word)).correlation

        def pooled(ls):
            g = np.concatenate([inst2gold[w] for w in ls])
            m = np.concatenate([per_lemma_sus[w] for w in ls])
            inst = spearmanr(g, m).correlation
            t = pd.DataFrame({"g": g, "m": m}).groupby("g")["m"].mean()
            sen = spearmanr(t.values, t.index.values).correlation
            return inst, sen

        n_groups = len(pd.unique(np.concatenate([inst2gold[w] for w in lemmas])))

        ir, sr = pooled(lemmas)
        if rm == grid[0]:
            print(f"[info] sense-level 的分组数（不同 gold tau 值）= {n_groups}\n")
        bw = np.array([spearmanr(word_gold[ix], np.array(per_lemma_word)[ix]).correlation
                       for ix in bidx])
        bi, bs = np.empty(len(bidx)), np.empty(len(bidx))
        for k, ix in enumerate(bidx):
            bi[k], bs[k] = pooled([lemmas[j] for j in ix])
        boots[("word", rm)], boots[("instance", rm)], boots[("sense", rm)] = bw, bi, bs
        for lv, val in (("word", wr), ("instance", ir), ("sense", sr)):
            rows.append({"reg_m": rm, "level": lv, "rho": round(float(val), 4),
                         "unmatched_mass": round(float(np.mean(unmatched)), 4),
                         "degenerate_frac": round(float(np.mean(degen)), 4)})
        print(f"{rm:>7.0f}{wr:>9.3f}{ir:>12.3f}{sr:>9.3f}"
              f"{np.mean(unmatched):>10.1%}{np.mean(degen):>10.1%}")

    out = pd.DataFrame(rows)
    for r in rows:
        k = (r["level"], r["reg_m"])
        if k in boots:
            lo, hi = ci(boots[k])
            r["ci_lo"], r["ci_hi"] = round(float(lo), 4), round(float(hi), 4)
    out = pd.DataFrame(rows)
    out_dir = Path(args.output_dir); out_dir.mkdir(exist_ok=True, parents=True)
    out.to_csv(out_dir / "regm_ablation.csv", index=False)
    print(f"\n[OK] 写入 {out_dir / 'regm_ablation.csv'}")

    # ── 两层的最优点是否分离 ──────────────────────────────────
    print("\n" + "═" * 60)
    best = {}
    for lv in ("word", "instance", "sense"):
        s = out[out.level == lv]
        b = s.loc[s["rho"].idxmax()]
        best[lv] = b
        print(f"{lv:<9} 最优 reg_m = {b['reg_m']:>6.0f}   ρ = {b['rho']:.3f}   "
              f"未搬运质量 {b['unmatched_mass']:.1%}")
    print("═" * 60)

    bw_ = best["word"]["reg_m"]
    print(f"\n按 word-level 选出的 reg_m={bw_:.0f}，对另外两层的代价（配对，同一套重采样）：")
    for lv in ("instance", "sense"):
        b_ = best[lv]["reg_m"]
        if b_ == bw_:
            print(f"  {lv:<9} 最优点重合，无代价")
            continue
        d = boots[(lv, b_)] - boots[(lv, bw_)]
        lo, hi = ci(d)
        at_w = out[(out.level == lv) & (out.reg_m == bw_)]["rho"].iloc[0]
        print(f"  {lv:<9} {at_w:.3f} → {best[lv]['rho']:.3f}   "
              f"Δ={best[lv]['rho'] - at_w:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{'**显著**' if not (lo <= 0 <= hi) else '区分不开'}")

    bi_ = best["instance"]["reg_m"]
    if False:
        # 在 word 最优点上，instance 损失多少？（配对，同一套重采样下标）
        d = boots[("instance", bi_)] - boots[("instance", bw_)]
        lo, hi = ci(d)
        i_at_w = out[(out.level == "instance") & (out.reg_m == bw_)]["rho"].iloc[0]
        print(f"\n两层最优点**分离**：word 选 {bw_:.0f}，instance 选 {bi_:.0f}")
        print(f"  按 word-level 选 reg_m={bw_:.0f} 时，instance ρ = {i_at_w:.3f}")
        print(f"  按 instance-level 选 reg_m={bi_:.0f} 时，instance ρ = {best['instance']['rho']:.3f}")
        print(f"  配对差值 Δ = {best['instance']['rho'] - i_at_w:+.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}]  "
              f"{'显著' if not (lo <= 0 <= hi) else '区分不开'}")
    else:
        print("\n两层最优点**重合**，不存在选参冲突。")

    if not args.no_plot:
        make_plot(out, out_dir, emb)


def make_plot(out, out_dir, emb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dataset_config.setup_cjk_font()

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = {"word": "#2E4A8E", "instance": "#AE5136", "sense": "#2A6B4E"}
    for lv in ("word", "instance", "sense"):
        s = out[out.level == lv].sort_values("reg_m")
        ax.plot(s["reg_m"], s["rho"], "-o", ms=4, color=colors[lv],
                label={"word": "word-level (table3)", "instance": "instance-level (table2)",
                       "sense": "sense-level (table2)"}[lv])
        b = s.loc[s["rho"].idxmax()]
        ax.plot(b["reg_m"], b["rho"], "*", ms=16, color=colors[lv], zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("reg_m（边际松弛罚金，越大越接近平衡 OT）")
    ax.set_ylabel("Spearman ρ")
    ax.grid(alpha=.3)

    ax2 = ax.twinx()
    s = out[out.level == "word"].sort_values("reg_m")
    ax2.plot(s["reg_m"], s["unmatched_mass"], "--", color="grey", lw=1.2)
    ax2.set_ylabel("未搬运质量占比（灰虚线）", color="grey")
    ax2.tick_params(axis="y", colors="grey")

    ax.axvspan(10, 1000, alpha=.07, color="black")
    ax.text(100, ax.get_ylim()[0], " 上游 CV 的搜索范围", fontsize=8, color="grey", va="bottom",
            ha="center")
    ax.set_title(f"reg_m 该按哪一层选？（{DATASET}，★ = 各层最优）", fontsize=11)
    ax.legend(fontsize=9, loc="center left")
    fig.tight_layout()
    p = out_dir / "regm_ablation.png"
    fig.savefig(p, dpi=200)
    print(f"[OK] 图写入 {p}")


if __name__ == "__main__":
    main()
