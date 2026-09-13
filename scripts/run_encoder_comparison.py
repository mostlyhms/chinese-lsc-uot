#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_encoder_comparison.py — 编码器 × 方法 的完整对比评测。

对每个 (编码器, 方法) 组合跑：
  1. `table3.py` 的交叉验证（100 次划分，每次在验证集上选超参）→ CV 均值 + most common param
  2. 用那个超参在全部 40 个词上算点估计 Spearman
  3. bootstrap（B=2000）→ 95% CI
  4. 两条**配对差值**（见下）

结果写 `outputs/encoder_comparison.csv`（长表），并画 `outputs/encoder_comparison.png`（带误差棒）。

用法：
    export UOT_DATASET=dwug_zh
    python3 scripts/run_encoder_comparison.py                 # 用已有的 pkl，缺的报错
    python3 scripts/run_encoder_comparison.py --auto-encode   # 缺的自动调 calc_embeddings_bert.py 生成
    python3 scripts/run_encoder_comparison.py --encoders bert-base-chinese xlm-roberta-large

────────────────────────────────────────────────────────────────────────
为什么要两条配对差值
────────────────────────────────────────────────────────────────────────
ChiWUG 只有 40 个词，CI 宽度普遍 0.3–0.5（CLAUDE.md §8）。
比点估计的大小是没有意义的——必须看配对差值的 CI 是否跨 0。这里给两条，回答两个不同的问题：

  d_method  = 同一编码器内，method − f_SUS
              「在这个编码器上，别的方法比 UOT 差吗」

  d_encoder = 同一方法下，encoder − xl-lexeme
              「换编码器有区别吗」——**这条才是本脚本的主问题**

尤其是 `xlm-roberta-large` 那一格：XL-LEXEME 正是 XLM-R-large 在 WiC 上微调来的。
上游只测了 `xlm-roberta-base`，large 这一格在已发表工作里是空的。
把它填上，`d_encoder` 才能把「WiC 微调的贡献」和「单纯模型更大」分开。

────────────────────────────────────────────────────────────────────────
配对是靠共用一套重采样下标实现的
────────────────────────────────────────────────────────────────────────
所有编码器、所有方法共用同一个 (B, 40) 的下标矩阵。
如果各自独立重采样，差值的方差会被高估，CI 白白变宽、什么都区分不开。
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dataset_config  # noqa: E402
from dataset_config import DATASET  # noqa: E402
import metrics  # noqa: E402
import table3  # noqa: E402  —— 直接复用它的交叉验证，避免两份实现漂移
from _eval_common import (load_gold, score_words, boot_spearman, ci,  # noqa: E402
                          verdict, paired_diff)

# 编码器 → embeddings 文件名。xl-lexeme 是既有基线（src/calc_embeddings.py 的产物，文件名没有模型标签）。
ENCODERS = [
    ("xl-lexeme",                  f"{DATASET}_embeddings.pkl",                          None),
    ("bert-base-chinese",          f"{DATASET}_bert-base-chinese_L12_embeddings.pkl",     "bert-base-chinese"),
    ("chinese-roberta-wwm-ext",    f"{DATASET}_hfl-chinese-roberta-wwm-ext_L12_embeddings.pkl", "hfl/chinese-roberta-wwm-ext"),
    ("chinese-macbert-base",       f"{DATASET}_hfl-chinese-macbert-base_L12_embeddings.pkl",    "hfl/chinese-macbert-base"),
    ("xlm-roberta-large",          f"{DATASET}_xlm-roberta-large_L24_embeddings.pkl",     "xlm-roberta-large"),
]
METHODS = ["f_SUS", "f_OT", "f_APD", "f_PRT", "f_APDP", "f_LDR", "f_WiDiD"]
BASE_METHOD = "f_SUS"
BASE_ENCODER = "xl-lexeme"


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=None, help="只跑这些编码器（默认全部）")
    ap.add_argument("--methods", nargs="*", default=None, help="只跑这些方法（默认全部）")
    ap.add_argument("--auto-encode", action="store_true",
                    help="缺 pkl 时自动调 scripts/calc_embeddings_bert.py 生成")
    ap.add_argument("--n_trials", type=int, default=100, help="交叉验证次数（同 table3.py）")
    ap.add_argument("-B", type=int, default=2000, help="bootstrap 次数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args()


def ensure_embeddings(path: Path, hf_model, auto):
    if path.exists():
        return True
    if hf_model is None:
        print(f"[skip] 缺 {path.name}，且它不是本脚本能生成的（跑 src/calc_embeddings.py）")
        return False
    if not auto:
        print(f"[skip] 缺 {path.name}。加 --auto-encode 让脚本自己生成，或手动跑：\n"
              f"       python3 scripts/calc_embeddings_bert.py --model {hf_model}")
        return False
    print(f"[run] 生成 {path.name} …")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "calc_embeddings_bert.py"),
                        "--model", hf_model], cwd=ROOT)
    if r.returncode != 0 or not path.exists():
        print(f"[FAIL] {hf_model} 编码失败，跳过")
        return False
    return True


def main():
    args = parse_args()
    encoders = [e for e in ENCODERS if args.encoders is None or e[0] in args.encoders]
    methods = args.methods or METHODS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    df, id2word, word2gold = load_gold("f")

    rng = np.random.default_rng(args.seed)
    boot_idx = None          # 全局共用，见文件头
    gold = None
    rows, boots, points = [], {}, {}

    for enc_name, pkl, hf_model in encoders:
        path = ROOT / "embeddings" / pkl
        if not ensure_embeddings(path, hf_model, args.auto_encode):
            continue
        with open(path, "rb") as f:
            src, tgt = pickle.load(f)

        words = [w for w in df["lemma"] if w in src]
        if gold is None:
            gold = np.array([word2gold[w] for w in words])
            boot_idx = rng.integers(0, len(words), size=(args.B, len(words)))
            print(f"[info] {len(words)} 词, B={args.B}, 全局共用一套重采样下标")
        elif len(words) != len(gold):
            print(f"[skip] {enc_name}: 词数 {len(words)} 与基准不一致，无法配对")
            continue

        print(f"\n=== {enc_name} ({pkl}) ===")
        for m in methods:
            fn = getattr(metrics, m)
            cv = table3.cross_validate_method(
                m, fn, df, id2word, word2gold, src, tgt,
                n_trials=args.n_trials, train_ratio=0.8, random_seed=42,
            )
            param = cv["most_common_param"][0]
            pred = score_words(fn, param, src, tgt, words)
            pt = spearmanr(gold, pred).correlation
            b = boot_spearman(gold, pred, boot_idx)
            lo, hi = ci(b)
            boots[(enc_name, m)] = b
            points[(enc_name, m)] = pt
            rows.append({
                "encoder": enc_name, "method": m,
                "param": table3.format_parameter(param),
                "cv_mean": round(float(cv["mean_spearman"]), 4),
                "point": round(float(pt), 4),
                "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                "ci_width": round(float(hi - lo), 4),
            })
            print(f"  {m:<8} CV={cv['mean_spearman']:.2f}  点估计={pt:.3f}  "
                  f"[{lo:.3f}, {hi:.3f}]  param={table3.format_parameter(param)}")

    if not rows:
        sys.exit("[FATAL] 一个组合都没跑成。检查 embeddings/ 下有没有 pkl，或加 --auto-encode。")

    # 配对差值
    for r in rows:
        e, m = r["encoder"], r["method"]
        for label, key in (("method", (e, BASE_METHOD)), ("encoder", (BASE_ENCODER, m))):
            if key == (e, m) or key not in boots:
                r[f"d_{label}"] = r[f"d_{label}_lo"] = r[f"d_{label}_hi"] = np.nan
                r[f"d_{label}_verdict"] = "基准" if key == (e, m) else "n/a"
                continue
            d = boots[(e, m)] - boots[key]
            lo, hi = ci(d)
            r[f"d_{label}"] = round(float(points[(e, m)] - points[key]), 4)
            r[f"d_{label}_lo"], r[f"d_{label}_hi"] = round(float(lo), 4), round(float(hi), 4)
            r[f"d_{label}_verdict"] = verdict(lo, hi)

    out = pd.DataFrame(rows)
    csv_path = out_dir / "encoder_comparison.csv"
    out.to_csv(csv_path, index=False)
    print(f"\n[OK] 长表写入 {csv_path}  ({len(out)} 行)")

    print(f"\n点估计矩阵（行=编码器，列=方法）")
    print(out.pivot(index="encoder", columns="method", values="point")
             .reindex(index=[e[0] for e in encoders if e[0] in set(out.encoder)], columns=methods)
             .to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\n配对差值：编码器 − {BASE_ENCODER}（同一方法内比较；CI 跨 0 = 区分不开）")
    sub = out[out.encoder != BASE_ENCODER]
    if len(sub):
        for _, r in sub.iterrows():
            print(f"  {r['encoder']:<24} {r['method']:<8} Δ={r['d_encoder']:>7.3f}  "
                  f"[{r['d_encoder_lo']:>6.3f}, {r['d_encoder_hi']:>6.3f}]  {r['d_encoder_verdict']}")

    if not args.no_plot:
        make_plot(out, methods, [e[0] for e in encoders if e[0] in set(out.encoder)], out_dir)


def make_plot(out, methods, encoders, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dataset_config.setup_cjk_font()

    methods = [m for m in methods if m in set(out.method)]
    fig, ax = plt.subplots(figsize=(1.7 * len(methods) + 3, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    width = 0.8 / max(len(encoders), 1)

    for k, enc in enumerate(encoders):
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(methods):
            r = out[(out.encoder == enc) & (out.method == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            xs.append(i - 0.4 + width * (k + 0.5))
            ys.append(r["point"])
            lo.append(r["point"] - r["ci_lo"])
            hi.append(r["ci_hi"] - r["point"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3, markersize=6,
                    color=colors[k % 10], label=enc, linestyle="none", elinewidth=1.4)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("Spearman ρ vs change_graded")
    ax.set_title(f"编码器 × 方法（{DATASET}，n=40 词，误差棒 = 95% bootstrap CI）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = out_dir / "encoder_comparison.png"
    fig.savefig(p, dpi=200)
    print(f"[OK] 图写入 {p}")


if __name__ == "__main__":
    main()
