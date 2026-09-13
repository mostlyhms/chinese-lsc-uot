#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_layer_sweep.py — 逐层扫描：每个编码器的哪一层最适合做词义演变检测。

为什么必须做这个（CLAUDE.md T6）：
`outputs/encoder_comparison.csv` 里所有数字都取自**模型末层**。
但未经微调的 MLM 模型，末层通常过拟合到预训练目标（预测被遮蔽的词），
**中间层反而更适合做语义相似度**。所以「xlm-roberta-large 显著差于 xl-lexeme」
这个结论存在一个明确的替代解释：它可能只是「末层差」，而不是「模型差」。

不扫层，那个结论能被一句话反驳掉。扫完才知道它站不站得住。

用法：
    export UOT_DATASET=dwug_zh
    python3 scripts/calc_embeddings_bert.py --model bert-base-chinese --layers all
    python3 scripts/run_layer_sweep.py

    python3 scripts/run_layer_sweep.py --methods f_SUS f_APD --encoders xlm-roberta-large

输出：outputs/layer_sweep.csv（长表）+ outputs/layer_sweep.png（ρ 随层数变化）

────────────────────────────────────────────────────────────────────────
「最佳层」这个数字**不能**直接拿去和别人比
────────────────────────────────────────────────────────────────────────
扫完之后取每个编码器表现最好的那一层，是在**同一批 40 个词**上事后挑最大值。
25 个层里挑最大，等于做了 25 次比较只报最好的那次——这是标准的选择偏差，
会系统性高估。两种比法各偏一边，都不诚实：

    只看末层     → 偏向**低估**未微调模型（末层恰好是它最差的地方）
    取最佳层     → 偏向**高估**（事后挑最大值）

`--nested-cv` 给出无偏的那个版本：把**层**和方法的超参一起，在每次交叉验证的
**验证集**上选，再到**测试集**上评。选层的代价因此被计入，报出来的数字才可比。
论文里要引的应该是这个数，不是「最佳层」那一列。

注意：xl-lexeme 无法扫层——它是 sentence-transformer，只输出一个成品句向量，
没有可选的中间层。它在图里画成一条水平参考线。
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
import dataset_config  # noqa: E402
from dataset_config import DATASET  # noqa: E402
import metrics  # noqa: E402
import table3  # noqa: E402
from _eval_common import load_gold, score_words, boot_spearman, ci, paired_diff  # noqa: E402

# (显示名, HF 名, Transformer 层数)
ENCODERS = [
    ("bert-base-chinese",       "bert-base-chinese",           12),
    ("chinese-roberta-wwm-ext", "hfl/chinese-roberta-wwm-ext", 12),
    ("chinese-macbert-base",    "hfl/chinese-macbert-base",    12),
    ("xlm-roberta-large",       "xlm-roberta-large",           24),
]
REF_ENCODER = ("xl-lexeme", f"{DATASET}_embeddings.pkl")
METHODS = ["f_SUS", "f_OT", "f_APD", "f_PRT", "f_APDP", "f_LDR", "f_WiDiD"]


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--nested-cv", action="store_true",
                    help="把**层**也当超参数放进交叉验证里选（在验证集上选，测试集上评）。"
                         "这是唯一没有选择偏差的层比较方式，见文件头")
    return ap.parse_args()


def nested_cv_over_layers(method, fn, layer_pkls, df, id2word, word2gold, words,
                          n_trials=100, train_ratio=0.8, random_seed=42):
    """把「层」当成和 reg_m/damping 同级的超参数，在验证集上一起选。

    与 table3.cross_validate_method 的唯一区别是候选集从 {param} 变成 {(layer, param)}。
    这样「挑了个好层」的代价被算进去了，得到的 ρ 才能和 xl-lexeme（无层可挑）公平比。
    """
    import itertools
    grid = table3.get_parameter_grid(fn)
    # (layer, param) -> {word: score}
    cell = {}
    for L, path in layer_pkls:
        with open(path, "rb") as f:
            src, tgt = pickle.load(f)
        for param in grid:
            cell[(L, param)] = dict(zip(words, score_words(fn, param, src, tgt, words)))

    np.random.seed(random_seed)
    perfs, picks = [], []
    n = len(df)
    for _ in range(n_trials):
        valid_ids = np.random.choice(range(n), int(n * train_ratio), replace=False)
        test_ids = [i for i in range(n) if i not in valid_ids]
        vg = [word2gold[id2word[i]] for i in valid_ids]
        tg = [word2gold[id2word[i]] for i in test_ids]

        best, best_key = -np.inf, None
        for key, w2s in cell.items():
            vp = [w2s[id2word[i]] for i in valid_ids]
            r = spearmanr(vg, vp)[0]
            if r > best:
                best, best_key = r, key
        picks.append(best_key)
        tp = [cell[best_key][id2word[i]] for i in test_ids]
        perfs.append(spearmanr(tg, tp)[0])

    from collections import Counter
    top = Counter(picks).most_common(1)[0]
    return float(np.mean(perfs)), top[0], top[1]


def main():
    args = parse_args()
    encoders = [e for e in ENCODERS if args.encoders is None or e[0] in args.encoders]
    methods = args.methods or METHODS
    pool_tag = "" if args.pool == "mean" else f"_{args.pool}"
    df, id2word, word2gold = load_gold("f")

    rng = np.random.default_rng(args.seed)
    boot_idx, gold, words = None, None, None
    rows, boots, points = [], {}, {}

    # ── 参考线：xl-lexeme（无层可扫）────────────────────────────
    ref_path = ROOT / "embeddings" / REF_ENCODER[1]
    ref = {}
    if ref_path.exists():
        with open(ref_path, "rb") as f:
            src, tgt = pickle.load(f)
        words = [w for w in df["lemma"] if w in src]
        gold = np.array([word2gold[w] for w in words])
        boot_idx = rng.integers(0, len(words), size=(args.B, len(words)))
        print(f"[info] {len(words)} 词, B={args.B}, 全局共用一套重采样下标")
        print(f"\n=== 参考线 xl-lexeme（sentence-transformer，无中间层） ===")
        for m in methods:
            fn = getattr(metrics, m)
            cv = table3.cross_validate_method(m, fn, df, id2word, word2gold, src, tgt,
                                              n_trials=args.n_trials, train_ratio=0.8,
                                              random_seed=42)
            pred = score_words(fn, cv["most_common_param"][0], src, tgt, words)
            pt = spearmanr(gold, pred).correlation
            ref[m] = pt
            boots[("xl-lexeme", None, m)] = boot_spearman(gold, pred, boot_idx)
            points[("xl-lexeme", None, m)] = pt
            print(f"  {m:<8} {pt:.3f}")
    else:
        print(f"[warn] 缺 {ref_path.name}，图上不画 xl-lexeme 参考线")

    # ── 逐编码器逐层 ───────────────────────────────────────────
    for enc_name, hf, n_layer in encoders:
        tag = hf.replace("/", "-")
        avail = []
        for L in range(n_layer + 1):
            p = ROOT / "embeddings" / f"{DATASET}_{tag}_L{L}{pool_tag}_embeddings.pkl"
            if p.exists():
                avail.append((L, p))
        if not avail:
            print(f"\n[skip] {enc_name} 一层都没有。先跑："
                  f"\n       python3 scripts/calc_embeddings_bert.py --model {hf} --layers all")
            continue

        print(f"\n=== {enc_name} —— {len(avail)} 层 ===")
        hdr = "  层  " + "".join(f"{m:>9}" for m in methods)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for L, p in avail:
            with open(p, "rb") as f:
                src, tgt = pickle.load(f)
            if words is None:
                words = [w for w in df["lemma"] if w in src]
                gold = np.array([word2gold[w] for w in words])
                boot_idx = rng.integers(0, len(words), size=(args.B, len(words)))
            line = f"  {L:>2}  "
            for m in methods:
                fn = getattr(metrics, m)
                cv = table3.cross_validate_method(m, fn, df, id2word, word2gold, src, tgt,
                                                  n_trials=args.n_trials, train_ratio=0.8,
                                                  random_seed=42)
                param = cv["most_common_param"][0]
                pred = score_words(fn, param, src, tgt, words)
                pt = spearmanr(gold, pred).correlation
                b = boot_spearman(gold, pred, boot_idx)
                lo, hi = ci(b)
                boots[(enc_name, L, m)] = b
                points[(enc_name, L, m)] = pt
                rows.append({"encoder": enc_name, "layer": L, "method": m,
                             "param": table3.format_parameter(param),
                             "cv_mean": round(float(cv["mean_spearman"]), 4),
                             "point": round(float(pt), 4),
                             "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
                line += f"{pt:>9.3f}"
            print(line)

    if not rows:
        sys.exit("[FATAL] 没有任何层的 pkl。先用 --layers all 生成。")

    # ── 每层 vs xl-lexeme 的配对差值 ──────────────────────────
    for r in rows:
        k, base = (r["encoder"], r["layer"], r["method"]), ("xl-lexeme", None, r["method"])
        if base in boots:
            r["d_vs_xllex"], r["d_lo"], r["d_hi"], r["d_verdict"] = paired_diff(
                boots[k], boots[base], points[k], points[base])
        else:
            r["d_vs_xllex"] = r["d_lo"] = r["d_hi"] = np.nan
            r["d_verdict"] = "n/a"

    out = pd.DataFrame(rows)
    out_dir = Path(args.output_dir); out_dir.mkdir(exist_ok=True, parents=True)
    out.to_csv(out_dir / "layer_sweep.csv", index=False)
    print(f"\n[OK] 长表写入 {out_dir / 'layer_sweep.csv'}  ({len(out)} 行)")

    # ── 最佳层 + 「末层是不是被冤枉了」 ───────────────────────
    print("\n" + "═" * 78)
    print("每个编码器的最佳层 vs 末层（这才是 T6 要回答的问题）")
    print("═" * 78)
    print(f"{'编码器':<26}{'方法':<9}{'末层':>7}{'最佳层':>8}{'最佳值':>8}{'提升':>8}   vs xl-lexeme")
    print("-" * 78)
    for enc_name, hf, n_layer in encoders:
        sub = out[out.encoder == enc_name]
        if not len(sub):
            continue
        for m in methods:
            s = sub[sub.method == m]
            if not len(s):
                continue
            last = s[s.layer == n_layer]
            best = s.loc[s["point"].idxmax()]
            lastv = float(last["point"].iloc[0]) if len(last) else np.nan
            print(f"{enc_name:<26}{m:<9}{lastv:>7.3f}{int(best['layer']):>8}"
                  f"{best['point']:>8.3f}{best['point'] - lastv:>+8.3f}   "
                  f"Δ={best['d_vs_xllex']:+.3f} [{best['d_lo']:+.3f}, {best['d_hi']:+.3f}] {best['d_verdict']}")
        print()

    if args.nested_cv:
        print("\n" + "═" * 78)
        print("无偏版本：把层当超参数，在验证集上选（选层的代价已计入）")
        print("═" * 78)
        print(f"{'编码器':<26}{'方法':<9}{'CV 均值':>9}{'最常选中的 (层, 超参)':>26}")
        print("-" * 78)
        nrows = []
        for enc_name, hf, n_layer in encoders:
            tag = hf.replace("/", "-")
            lp = [(L, ROOT / "embeddings" / f"{DATASET}_{tag}_L{L}{pool_tag}_embeddings.pkl")
                  for L in range(n_layer + 1)]
            lp = [(L, p) for L, p in lp if p.exists()]
            if not lp:
                continue
            for m in methods:
                fn = getattr(metrics, m)
                cvm, key, cnt = nested_cv_over_layers(
                    m, fn, lp, df, id2word, word2gold, words,
                    n_trials=args.n_trials, train_ratio=0.8, random_seed=42)
                L, param = key
                nrows.append({"encoder": enc_name, "method": m, "nested_cv_mean": round(cvm, 4),
                              "modal_layer": L, "modal_param": table3.format_parameter(param),
                              "modal_count": cnt})
                print(f"{enc_name:<26}{m:<9}{cvm:>9.3f}"
                      f"{f'层{L}, {table3.format_parameter(param)} ({cnt}/{args.n_trials})':>26}")
            print()
        # xl-lexeme 参考：它没有层可选，直接用 table3 的 CV
        if ref_path.exists():
            with open(ref_path, "rb") as f:
                src, tgt = pickle.load(f)
            print(f"{'xl-lexeme（无层可选）':<26}")
            for m in methods:
                cv = table3.cross_validate_method(m, getattr(metrics, m), df, id2word,
                                                  word2gold, src, tgt, n_trials=args.n_trials,
                                                  train_ratio=0.8, random_seed=42)
                nrows.append({"encoder": "xl-lexeme", "method": m,
                              "nested_cv_mean": round(float(cv["mean_spearman"]), 4),
                              "modal_layer": -1,
                              "modal_param": table3.format_parameter(cv["most_common_param"][0]),
                              "modal_count": cv["most_common_param"][1]})
                print(f"{'':<26}{m:<9}{cv['mean_spearman']:>9.3f}")
        nd = pd.DataFrame(nrows)
        nd.to_csv(out_dir / "layer_nested_cv.csv", index=False)
        print(f"\n[OK] 写入 {out_dir / 'layer_nested_cv.csv'}")
        print("\n无偏 CV 均值矩阵（行=编码器，列=方法）")
        print(nd.pivot(index="encoder", columns="method", values="nested_cv_mean")
                .reindex(columns=[m for m in methods if m in set(nd.method)])
                .to_string(float_format=lambda x: f"{x:.3f}"))

    if not args.no_plot:
        make_plot(out, methods, encoders, ref, out_dir)


def make_plot(out, methods, encoders, ref, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dataset_config.setup_cjk_font()

    encs = [e for e in encoders if e[0] in set(out.encoder)]
    methods = [m for m in methods if m in set(out.method)]
    fig, axs = plt.subplots(1, len(encs), figsize=(4.2 * len(encs), 4.4), sharey=True)
    if len(encs) == 1:
        axs = [axs]
    cmap = plt.cm.viridis(np.linspace(0, .92, len(methods)))

    for ax, (enc_name, hf, n_layer) in zip(axs, encs):
        sub = out[out.encoder == enc_name]
        for c, m in zip(cmap, methods):
            s = sub[sub.method == m].sort_values("layer")
            if not len(s):
                continue
            ax.plot(s["layer"], s["point"], "-o", ms=3, lw=1.5, color=c, label=m)
        if ref:
            # xl-lexeme 在最强方法上的成绩，作为一条横向标尺
            ax.axhline(max(ref.values()), ls="--", lw=1.2, color="crimson", alpha=.85)
            ax.text(0.98, max(ref.values()), " xl-lexeme 最好成绩", ha="right", va="bottom",
                    fontsize=8, color="crimson", transform=ax.get_yaxis_transform())
        ax.axvline(n_layer, ls=":", lw=1, color="grey")
        ax.text(n_layer, ax.get_ylim()[0], " 末层", fontsize=8, color="grey", va="bottom")
        ax.set_title(enc_name, fontsize=11)
        ax.set_xlabel("hidden_states 层号（0 = embedding 层）")
        ax.grid(alpha=.3)
    axs[0].set_ylabel("Spearman ρ vs change_graded")
    axs[-1].legend(fontsize=8, loc="lower left", ncol=2)
    fig.suptitle(f"逐层扫描（{DATASET}，n=40 词，点估计）", fontsize=12)
    fig.tight_layout()
    p = out_dir / "layer_sweep.png"
    fig.savefig(p, dpi=200)
    print(f"[OK] 图写入 {p}")


if __name__ == "__main__":
    main()
