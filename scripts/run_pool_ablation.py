#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pool_ablation.py — 子词池化方式（mean / sum / first）的消融。

ChiWUG 的目标词是**词**（下海、灰色、飞越），而中文 BERT 按**字**切，
一个目标词横跨多个 token，必须把这几个子词向量合成一个。合成方式有三种，
`calc_embeddings_bert.py --pool` 控制。这个脚本回答：换池化方式，结论会不会变。

────────────────────────────────────────────────────────────────────────
sum 与 mean 什么时候相等：取决于分词，不只取决于方法
────────────────────────────────────────────────────────────────────────
sum 与 mean 只差一条用例自己的子词个数 n_i：   v_sum,i = n_i · v_mean,i

**第一层**：凡是「先把每条向量各自 L2 归一化、之后再也不用模长」的方法，
n_i 在归一化时被整除掉，无论 n_i 怎么变都**逐位相同**。查过 src/ 确认：

    f_SUS   utils.calc_sus()  开头 u/‖u‖                    → 恒不变
    f_LDR   utils.calc_ldr()  开头 u/‖u‖                    → 恒不变
    f_OT    自己 u/‖u‖ 再算余弦代价矩阵                      → 恒不变
    f_APD   自己 u/‖u‖ 再平均                                → 恒不变
    f_WiDiD AP 聚类 affinity='cosine'，余弦尺度无关          → 恒不变

**第二层**（这一层最初被我漏掉了，是跑出来才发现的）：另外两个方法吃模长——

    f_PRT   对**未归一化**向量求平均（这正是它区别于 APD 之处）
    f_APDP  在**未归一化**的簇质心上算 Canberra 距离

但它们也不是必然会变。关键在于 **n_i 在同一个目标词内部是否恒定**：
如果该词的每一条用例都切成同样多的子词，那 n_i 就是个**常数**，
sum 相当于把整团点云乘以同一个标量——余弦不受影响，Canberra 的分子分母
同时被约掉也不受影响，于是这两个方法**同样逐位相同**。

而 n_i 恒不恒定，完全由分词器决定：

    字级中文 BERT（bert-base-chinese / roberta-wwm-ext / macbert）
        目标词是固定字符串、逐字切 → 40 个词**全部**恒定 → 所有方法都不变
    xlm-roberta-large（SentencePiece BPE）
        「包装」随上下文有时切 1 个 token 有时 2 个 → 40 个词里 **16 个**不恒定
        → 只有 f_PRT / f_APDP 会变

所以脚本不写死预测，而是**自己去测**每个编码器的 n_i 恒定性，据此形成预测，
再拿实测差异去验。预测不中就 [FAIL]，说明对代码或分词的理解有问题。

用法：
    export UOT_DATASET=dwug_zh
    python3 scripts/run_pool_ablation.py --auto-encode
    python3 scripts/run_pool_ablation.py --encoders bert-base-chinese
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_config import DATASET  # noqa: E402
import metrics  # noqa: E402
import table3  # noqa: E402
from _eval_common import load_gold, score_words, boot_spearman, ci, paired_diff  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

# (显示名, HF 名, 末层下标)
ENCODERS = [
    ("bert-base-chinese",       "bert-base-chinese",            12),
    ("chinese-roberta-wwm-ext", "hfl/chinese-roberta-wwm-ext",  12),
    ("chinese-macbert-base",    "hfl/chinese-macbert-base",     12),
    ("xlm-roberta-large",       "xlm-roberta-large",            24),
]
POOLS = ["mean", "sum", "first"]
METHODS = ["f_SUS", "f_OT", "f_APD", "f_PRT", "f_APDP", "f_LDR", "f_WiDiD"]
# 逐条 L2 归一化，n_i 怎么变都不变
INVARIANT_ALWAYS = {"f_SUS", "f_OT", "f_APD", "f_LDR", "f_WiDiD"}
# 吃模长：只有当 n_i 在词内恒定（= 全局缩放）时才不变
SCALE_SENSITIVE = {"f_PRT", "f_APDP"}
BASE_POOL = "mean"


def subword_counts_constant(hf):
    """该分词器下，每个目标词的子词个数在其所有用例中是否恒定。

    返回 (是否全部恒定, 不恒定的词数, 总词数)。这决定 f_PRT / f_APDP 会不会受 pool 影响。
    """
    from transformers import AutoTokenizer
    from calc_embeddings_bert import load_lemma_rows, crop_around_target
    tok = AutoTokenizer.from_pretrained(hf, use_fast=True)
    dirs = sorted(d for d in (ROOT / "data" / DATASET / "data").iterdir()
                  if d.is_dir() and not d.name.startswith("."))
    n_vary = 0
    for d in dirs:
        rows, _ = load_lemma_rows(d)
        counts = set()
        for gid in (1, 2):
            for text, L, R in rows[gid]:
                tx, l, r = crop_around_target(tok, text, L, R, 512)
                enc = tok(tx, return_offsets_mapping=True, truncation=True, max_length=512)
                counts.add(len([k for k, (a, b) in enumerate(enc["offset_mapping"])
                                if a < r and b > l and a != b]))
        n_vary += len(counts) > 1
    return n_vary == 0, n_vary, len(dirs)


def pkl_name(hf, layer, pool):
    tag = hf.replace("/", "-")
    suffix = "" if pool == "mean" else f"_{pool}"
    return f"{DATASET}_{tag}_L{layer}{suffix}_embeddings.pkl"


def ensure(path, hf, layer, pool, auto):
    if path.exists():
        return True
    if not auto:
        print(f"[skip] 缺 {path.name}，加 --auto-encode 或手动跑："
              f"\n       python3 scripts/calc_embeddings_bert.py --model {hf} --layer {layer} --pool {pool}")
        return False
    print(f"[run] 生成 {path.name} …")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "calc_embeddings_bert.py"),
                        "--model", hf, "--layer", str(layer), "--pool", pool], cwd=ROOT)
    return r.returncode == 0 and path.exists()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--auto-encode", action="store_true")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()

    encoders = [e for e in ENCODERS if args.encoders is None or e[0] in args.encoders]
    methods = args.methods or METHODS
    df, id2word, word2gold = load_gold("f")

    rng = np.random.default_rng(args.seed)
    boot_idx, gold, words = None, None, None
    rows, boots, points, raw = [], {}, {}, {}

    for enc_name, hf, layer in encoders:
        for pool in POOLS:
            path = ROOT / "embeddings" / pkl_name(hf, layer, pool)
            if not ensure(path, hf, layer, pool, args.auto_encode):
                continue
            with open(path, "rb") as f:
                src, tgt = pickle.load(f)
            if words is None:
                words = [w for w in df["lemma"] if w in src]
                gold = np.array([word2gold[w] for w in words])
                boot_idx = rng.integers(0, len(words), size=(args.B, len(words)))
                print(f"[info] {len(words)} 词, B={args.B}, 全局共用一套重采样下标\n")

            print(f"=== {enc_name} · pool={pool} ===")
            for m in methods:
                fn = getattr(metrics, m)
                cv = table3.cross_validate_method(
                    m, fn, df, id2word, word2gold, src, tgt,
                    n_trials=args.n_trials, train_ratio=0.8, random_seed=42)
                param = cv["most_common_param"][0]
                pred = score_words(fn, param, src, tgt, words)
                pt = spearmanr(gold, pred).correlation
                b = boot_spearman(gold, pred, boot_idx)
                lo, hi = ci(b)
                key = (enc_name, pool, m)
                boots[key], points[key], raw[key] = b, pt, pred
                rows.append({"encoder": enc_name, "pool": pool, "method": m,
                             "param": table3.format_parameter(param),
                             "cv_mean": round(float(cv["mean_spearman"]), 4),
                             "point": round(float(pt), 4),
                             "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
                print(f"  {m:<8} 点估计={pt:.3f}  [{lo:.3f}, {hi:.3f}]")
            print()

    if not rows:
        sys.exit("[FATAL] 一个组合都没跑成。加 --auto-encode。")

    # ── 配对差值 vs mean ──────────────────────────────────────
    for r in rows:
        k, base = (r["encoder"], r["pool"], r["method"]), (r["encoder"], BASE_POOL, r["method"])
        if r["pool"] == BASE_POOL or base not in boots:
            r["d_vs_mean"] = r["d_lo"] = r["d_hi"] = np.nan
            r["d_verdict"] = "基准" if r["pool"] == BASE_POOL else "n/a"
        else:
            r["d_vs_mean"], r["d_lo"], r["d_hi"], r["d_verdict"] = paired_diff(
                boots[k], boots[base], points[k], points[base])

    out = pd.DataFrame(rows)
    out_dir = Path(args.output_dir); out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / "pool_ablation.csv"
    out.to_csv(csv_path, index=False)
    print(f"[OK] 长表写入 {csv_path}  ({len(out)} 行)\n")

    # ── 验预测：sum 对 mean，不变的必须真不变 ─────────────────
    print("═" * 66)
    print("预测校验：sum vs mean")
    print("═" * 66)
    n_fail = 0
    for enc_name, hf, layer in encoders:
        if (enc_name, "sum", methods[0]) not in raw:
            continue
        const, n_vary, n_tot = subword_counts_constant(hf)
        print(f"\n{enc_name} —— 子词数在词内{'恒定' if const else '不恒定'}"
              f"（{n_vary}/{n_tot} 个词的子词数会随上下文变）")
        print(f"  {'方法':<9}{'预测':<8}{'绝对差':>11}{'相对差':>11}  {'实测':<9}结论")
        print("  " + "-" * 56)
        for m in methods:
            a, b = (enc_name, "sum", m), (enc_name, BASE_POOL, m)
            if a not in raw or b not in raw:
                continue
            d = float(np.abs(raw[a] - raw[b]).max())
            scale = float(np.abs(raw[b]).max()) or 1.0
            rel = d / scale
            # 三档：逐位相同 / 浮点级（向量是 float32，归一化只能恢复到浮点精度；
            # f_LDR 的 vMF 浓度 κ = ls(D-ls²)/(1-ls²) 在高维下 1-ls² 极小，
            # 会把 1e-8 的舍入放大成 1e-3，所以必须按**相对**尺度判，不能用绝对阈值）
            cat = "逐位相同" if d == 0 else ("浮点级" if rel < 1e-5 else "真的不同")
            pred_inv = (m in INVARIANT_ALWAYS) or (m in SCALE_SENSITIVE and const)
            ok = pred_inv == (cat != "真的不同")
            n_fail += not ok
            print(f"  {m:<9}{'不变' if pred_inv else '会变':<8}{d:>11.2e}{rel:>11.1e}  "
                  f"{cat:<9}{'✓' if ok else '✗ 预测错了'}")
    print("\n" + ("[OK] 预测全部命中" if n_fail == 0 else
                   f"[FAIL] {n_fail} 处与预测不符——去查编码脚本或对代码的理解") + "\n")

    # ── first 的影响 ──────────────────────────────────────────
    print("═" * 66)
    print("first（只取第一个子词）对 mean 的配对差值")
    print("═" * 66)
    sub = out[out["pool"] == "first"]
    for _, r in sub.iterrows():
        print(f"  {r['encoder']:<26}{r['method']:<9}Δ={r['d_vs_mean']:>7.3f}  "
              f"[{r['d_lo']:>6.3f}, {r['d_hi']:>6.3f}]  {r['d_verdict']}")

    print("\n点估计矩阵（行=编码器×池化，列=方法）")
    piv = out.pivot_table(index=["encoder", "pool"], columns="method", values="point")
    print(piv.reindex(columns=[m for m in methods if m in set(out.method)])
             .to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
