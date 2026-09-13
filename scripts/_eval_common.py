# -*- coding: utf-8 -*-
"""评测的公共部件：被 run_encoder_comparison.py / run_pool_ablation.py / run_layer_sweep.py 共用。

单独拆出来只有一个理由：**别让三份 bootstrap 实现慢慢漂移**。
配对差值的正确性完全依赖「所有比较对象共用同一套重采样下标」，
这个约定一旦在某个脚本里被复制走样，出来的 CI 会假性变宽，而且不会报错。
"""
import csv
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset_config import DATA_DIR  # noqa: E402


def load_gold(level="f"):
    """返回 (df, id2word, word2gold)。level='f' 用 change_graded，'g' 用簇分布熵差。"""
    df = pd.read_table(f"{DATA_DIR}/stats/opt/stats_groupings.csv", quoting=csv.QUOTE_NONE)
    id2word = dict(zip(df.index, df["lemma"]))
    if level == "f":
        word2gold = dict(zip(df["lemma"], df["change_graded"]))
    else:
        def ent(s):
            p = np.array(eval(s)); p = p[p > 0]
            return float(-(p * np.log(p)).sum())
        word2gold = {r["lemma"]: ent(r["cluster_prob_dist2"]) - ent(r["cluster_prob_dist1"])
                     for _, r in df.iterrows()}
    return df, id2word, word2gold


def score_words(fn, param, src, tgt, words):
    """用固定超参在每个词上算一个分数。param 的形状同 table3.get_parameter_grid 的产物。"""
    names = inspect.signature(fn).parameters
    if param is None:
        kw = {}
    elif isinstance(param, tuple):
        kw = {"reg_m": param[0], "theta": param[1]}
    elif "reg_m" in names:
        kw = {"reg_m": param}
    elif "theta" in names:
        kw = {"theta": param}
    else:
        kw = {"damping": param}
    return np.array([fn(np.array(src[w]), np.array(tgt[w]), **kw) for w in words])


def boot_spearman(gold, pred, idx):
    """按给定的重采样下标算 Spearman 序列。

    退化的重采样（并列太多算不出秩相关）用 nan 占位而不是丢弃——
    **位置必须保留**，否则两个方法的 boot 数组错位，配对差值就成了乱配。
    """
    out = np.full(len(idx), np.nan)
    for i, ix in enumerate(idx):
        g, p = gold[ix], pred[ix]
        if len(np.unique(g)) < 3 or len(np.unique(p)) < 3:
            continue
        r = spearmanr(g, p).correlation
        if not np.isnan(r):
            out[i] = r
    return out


def ci(a, lo=2.5, hi=97.5):
    a = a[~np.isnan(a)]
    if len(a) < 100:
        return np.nan, np.nan
    return tuple(np.percentile(a, [lo, hi]))


def verdict(lo, hi):
    if np.isnan(lo):
        return "n/a"
    if lo <= 0 <= hi:
        return "区分不开"
    return "显著更好" if lo > 0 else "显著更差"


def paired_diff(boot_a, boot_b, point_a, point_b):
    """配对差值 a − b。两个 boot 数组必须来自同一套下标，否则结果无意义。"""
    d = boot_a - boot_b
    lo, hi = ci(d)
    return round(float(point_a - point_b), 4), round(float(lo), 4), round(float(hi), 4), verdict(lo, hi)
