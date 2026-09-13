#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dryrun_check.py — 不装 torch / 不下载 XL-LEXEME，就先验证数据格式和**对齐**是否正确。

做法：用 clusters/opt 里的 gold 簇标签合成假向量（同簇 = 同一个方向 + 噪声），
写成和 calc_embeddings.py 完全一样结构的 pkl。然后：

  * 如果 uses.csv 顺序 与 clusters/opt 顺序 是对齐的，
    SUS 算出来的 instance-level 分数应当和 gold tau 高度相关（Spearman > 0.8）；
  * 如果错位了，相关系数会掉到 0 附近。

也就是说这个脚本能在花几十分钟跑真模型之前，先把“不报错但结果全错”的坑排掉。

用法：
    UOT_DATASET=dwug_zh python3 scripts/dryrun_check.py

    # 可选：额外写出假 embeddings，用来确认 src/table*.py、src/fig*.py 不会崩
    # （跑完记得删掉 embeddings/dwug_zh_embeddings.pkl 和 embeddings/tsne/）
    UOT_DATASET=dwug_zh python3 scripts/dryrun_check.py --write-emb
    UOT_DATASET=dwug_zh python3 src/table3.py
"""
import argparse
import ast
import csv
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ot  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--write-emb", action="store_true",
                 help="额外写出假 embeddings，用于 smoke-test src/table*.py（数字无意义）")
WRITE_EMB = _ap.parse_args().write_emb

DATASET = os.environ.get("UOT_DATASET", "dwug_zh")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / DATASET
DIM = 64
RNG = np.random.default_rng(0)


def calc_sus(u, v, reg_m=100):
    a = np.ones(len(u)) / len(u)
    b = np.ones(len(v)) / len(v)
    un = u / np.linalg.norm(u, axis=1, keepdims=True)
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    C = 1 - un @ vn.T
    T = ot.unbalanced.mm_unbalanced(a, b, C, reg_m=reg_m, div="l2")
    return -(a - T.sum(axis=1)) / a, (b - T.sum(axis=0)) / b


def main():
    lemma_dirs = sorted(d for d in (DATA / "data").iterdir()
                        if d.is_dir() and not d.name.startswith("."))
    print(f"[info] dataset={DATASET}  lemmas={len(lemma_dirs)}")

    # 1) 复现 calc_embeddings.py 的读取与顺序
    order = {}
    for d in lemma_dirs:
        df = pd.read_table(d / "uses.csv", quoting=csv.QUOTE_NONE)
        ids = (df[df["grouping"] == 1]["identifier"].tolist()
               + df[df["grouping"] == 2]["identifier"].tolist())
        n1 = int((df["grouping"] == 1).sum())
        assert len(ids) == len(df), f"{d.name}: grouping 里有 1/2 之外的值"
        order[d.name] = (ids, n1)

    # 2) 与 clusters/opt 逐位比对
    mism = []
    lemma2clusters = {}
    for lemma, (ids, n1) in order.items():
        dc = pd.read_table(DATA / "clusters" / "opt" / f"{lemma}.csv", quoting=csv.QUOTE_NONE)
        cids = dc["identifier"].astype(str).tolist()
        if cids != [str(i) for i in ids]:
            mism.append(lemma)
        lemma2clusters[lemma] = dc["cluster"].tolist()
    if mism:
        print(f"[FAIL] {len(mism)} 个词的 clusters 行序与 uses 不一致: {mism[:5]}")
        sys.exit(1)
    print("[OK] 全部 40 个词：clusters/opt 行序 == uses.csv(grouping排序) 行序")

    # 3) 合成假向量
    src_vecs, tgt_vecs = defaultdict(list), defaultdict(list)
    for lemma, (ids, n1) in order.items():
        cl = lemma2clusters[lemma]
        k = max([c for c in cl if c != -1] + [0]) + 1
        centers = RNG.normal(size=(k + 1, DIM))
        for pos, c in enumerate(cl):
            base = centers[c if c != -1 else k]
            vec = base + 0.15 * RNG.normal(size=DIM)
            (src_vecs if pos < n1 else tgt_vecs)[lemma].append(vec)
        assert len(src_vecs[lemma]) == n1
    if WRITE_EMB:
        emb_dir = ROOT / "embeddings"
        emb_dir.mkdir(exist_ok=True)
        out = emb_dir / f"{DATASET}_embeddings.pkl"
        with open(out, "wb") as f:
            pickle.dump((dict(src_vecs), dict(tgt_vecs)), f)
        print("!" * 70)
        print(f"[WARN] 已写入**假** embeddings -> {out.relative_to(ROOT)}")
        print("[WARN] 它只能用来确认 table*/fig* 不崩，数字没有任何意义。")
        print("[WARN] 跑真模型前必须删掉它，以及 embeddings/tsne/ 下的 t-SNE 缓存！")
        print("!" * 70)

    # 4) 复现 table2.py 的 gold tau，并检查对齐
    df = pd.read_table(DATA / "stats" / "opt" / "stats_groupings.csv", quoting=csv.QUOTE_NONE)
    word_cluster2tau = {}
    allt = []
    for _, row in df.iterrows():
        d1 = np.array(ast.literal_eval(row["cluster_prob_dist1"]))
        d2 = np.array(ast.literal_eval(row["cluster_prob_dist2"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            taus = np.log(d2 / d1)
        allt.extend(taus)
        word_cluster2tau[row["lemma"]] = dict(enumerate(taus))
    fin = np.array(allt)[np.isfinite(allt)]
    for w, sub in word_cluster2tau.items():
        for k2, v in sub.items():
            if v == np.inf:
                sub[k2] = fin.max()
            elif v == -np.inf:
                sub[k2] = fin.min()
            elif np.isnan(v):
                print(f"[FAIL] {w} 簇 {k2} 的 gold tau 是 nan")
                sys.exit(1)

    rhos = []
    for lemma, (ids, n1) in order.items():
        cl = lemma2clusters[lemma]
        gold = [word_cluster2tau[lemma][c] for c in cl if c != -1]
        u = np.array(src_vecs[lemma]); v = np.array(tgt_vecs[lemma])
        su, sv = calc_sus(u, v)
        pred = np.delete(np.append(su, sv), [i for i, c in enumerate(cl) if c == -1])
        if len(set(gold)) < 2:
            continue
        r = spearmanr(gold, pred).correlation
        if not np.isnan(r):
            rhos.append((r, lemma))
    rhos.sort()
    mean = float(np.mean([r for r, _ in rhos]))
    print(f"[info] 合成数据上 instance-level Spearman: mean={mean:.3f}  "
          f"min={rhos[0][0]:.3f}({rhos[0][1]})  n={len(rhos)}")
    if mean > 0.6:
        print("[OK] 对齐正确（合成数据上 gold 与 SUS 强相关）。")
    else:
        print("[FAIL] 相关性过低 —— uses/clusters 很可能仍然错位。")
        sys.exit(1)


if __name__ == "__main__":
    main()
