#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_embeddings_bert.py — 用任意 HuggingFace 编码器产出符合本仓库契约的 embeddings。

上游 `src/calc_embeddings.py` 写死了 XL-LEXEME（WordTransformer）。本脚本是它的通用替代：
给 `--model bert-base-chinese` 之类，产出结构完全相同的 pkl，
下游 `metrics.py` / `table*.py` / `fig*.py` / `bootstrap_ci.py` 一行都不用改。

用法：
    export UOT_DATASET=dwug_zh
    export HF_ENDPOINT=https://hf-mirror.com          # 国内网络
    python3 scripts/calc_embeddings_bert.py --model bert-base-chinese
    python3 scripts/calc_embeddings_bert.py --model hfl/chinese-roberta-wwm-ext --layer 12
    python3 scripts/calc_embeddings_bert.py --model xlm-roberta-large --pool sum

输出：embeddings/{DATASET}_{model}_L{layer}[_{pool}]_embeddings.pkl
（pool=mean 时省略 pool 后缀，于是默认名就是 dwug_zh_bert-base-chinese_L12_embeddings.pkl）

────────────────────────────────────────────────────────────────────────
两个必须守住的东西（细节见 CLAUDE.md §2）
────────────────────────────────────────────────────────────────────────
I1 用例顺序：按 uses.csv 里 grouping==1 的行（原序）接 grouping==2 的行（原序）遍历，
   **一条都不跳**——包括 clusters/opt 里 cluster == -1 的噪声节点。
   下游 table2/fig6 是按**行号**删这些噪声的，提前删会让所有下标错位，
   而且不会抛任何异常，只会静默算错。
   （`CSSDetection/src/embs.py` 的 processing() 就提前删了，所以它的产物不能喂给本仓库。）
I2 文件契约：pickle.dump((source_token2vecs, target_token2vecs))，
   各是 {lemma: [np.ndarray(dim,), ...]}，list 顺序即上面的规范顺序。

────────────────────────────────────────────────────────────────────────
目标词定位：为什么不抄 CSSDetection
────────────────────────────────────────────────────────────────────────
`CSSDetection/src/utils.py` 的做法是把 token 用空格拼成一个字符串、在里面正则搜索目标词的
token 串，再用 `abs(current_pos - start)` 挑离原文字符偏移最近的那个匹配。
拼接串的位置尺度和原文字符偏移**根本不是一回事**：中文一字一 token、每个 token 后面补一个
空格，拼接串的下标约等于原文下标的 2 倍，所以"最近匹配"这个启发式在目标词多次出现时会选错。

这里改用 fast tokenizer 的 `return_offsets_mapping=True`，把字符区间 [L, R) 精确映射到 token：

    ids = [i for i, (a, b) in enumerate(offsets) if a < R and b > L and a != b]

`a != b` 顺带滤掉 [CLS]/[SEP]/<pad> 这些偏移为 (0, 0) 的特殊 token。

────────────────────────────────────────────────────────────────────────
--pool 的选择会影响结果，不是无关紧要的旋钮
────────────────────────────────────────────────────────────────────────
ChiWUG 的目标词是**词**（下海、灰色、飞越），而 bert-base-chinese 按**字**切，
一个目标词横跨多个 token，必须池化。

  mean（默认）：子词向量取平均
  sum        ：取和，CSSDetection 用的就是这个（注释说 mean 可能产生 nan）
  first      ：只取第一个子词

注意 sum 与 mean 只差一个每条用例各不相同的正标量（子词个数）：
  * f_APD / g_vMF 先逐条 L2 归一化 → 两者**结果完全一致**
  * f_PRT 对未归一化向量求平均 → 两者**不一致**（sum 让长词权重更大）
  * f_SUS / f_OT / f_LDR 的代价矩阵是余弦 → 一致；但 WiDiD 的 AP 聚类用 cosine affinity，也一致
换句话说，只有 PRT（和任何吃模长的量）会受 pool 影响。

────────────────────────────────────────────────────────────────────────
--layer
────────────────────────────────────────────────────────────────────────
`hidden_states` 是 (num_layers + 1) 个张量，下标 0 是 embedding 层，1..N 是各 Transformer 层。
`--layer` 默认 -1（最后一层），会在文件名里解析成实际层号（bert-base → L12，
xlm-roberta-large → L24）。上游对比里层数是重要变量，做层扫描时显式传 `--layer`。
"""
import argparse
import csv
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset_config import DATASET, DATA_DIR  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description="用任意 HF 编码器产出符合 I1/I2 契约的 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="HF 模型名，如 bert-base-chinese / hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--input_dir", default=DATA_DIR, help="DWUG 格式数据目录")
    ap.add_argument("--output_dir", default="embeddings")
    ap.add_argument("--out", default=None, help="显式指定输出文件名（默认按模型/层/池化自动生成）")
    ap.add_argument("--layer", type=int, default=-1,
                    help="hidden_states 下标：0=embedding 层，1..N=各层，-1=最后一层（默认）")
    ap.add_argument("--layers", default=None,
                    help="一次导出多层，每层一个 pkl，共用一次前向。"
                         "写法：all | 1,6,12 | 1-12 | 0-24:4。给了它就忽略 --layer")
    ap.add_argument("--pool", choices=["mean", "sum", "first"], default="mean",
                    help="目标词跨多个子词时怎么合成一个向量（默认 mean）")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    return ap.parse_args()


def pick_device(spec):
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def crop_around_target(tok, text, L, R, max_length):
    """上下文超长时，以目标词为中心截断（XL-LEXEME 的 center_sentence 也是这么做的）。

    直接 truncation=True 会从右边砍，目标词在句尾就被砍没了。这里保证 [L, R) 一定留在窗口里。
    返回 (新文本, 新 L, 新 R)。没超长就原样返回。
    """
    n_tok = len(tok(text, add_special_tokens=True)["input_ids"])
    if n_tok <= max_length:
        return text, L, R

    budget = max_length - tok.num_special_tokens_to_add(pair=False) - 2  # 留一点余量
    for _ in range(12):
        # 按当前 token/字符 比例估计能留多少字符
        est_chars = max(R - L, int(len(text) * budget / max(n_tok, 1)))
        half = max(0, (est_chars - (R - L)) // 2)
        a = max(0, L - half)
        b = min(len(text), R + half)
        cropped = text[a:b]
        n = len(tok(cropped, add_special_tokens=True)["input_ids"])
        if n <= max_length:
            return cropped, L - a, R - a
        # 还是太长：收紧预算再来
        budget = int(budget * 0.8)
        text, L, R, n_tok = cropped, L - a, R - a, n
    # 兜底：只留目标词本身
    return text[L:R], 0, R - L


def load_lemma_rows(lemma_dir):
    """按 I1 的规范顺序返回 (context, L, R) 列表，外加 uses.csv 的总行数用于校验。"""
    df = pd.read_table(lemma_dir / "uses.csv", quoting=csv.QUOTE_NONE)
    out = {}
    for gid in (1, 2):
        sub = df[df["grouping"] == gid]          # pandas 的布尔筛选保持原文件行序
        rows = []
        for ctx, idx in zip(sub["context"], sub["indexes_target_token"]):
            L, R = map(int, str(idx).split(":"))
            rows.append((str(ctx), L, R))
        out[gid] = rows
    n_total = len(out[1]) + len(out[2])
    if n_total != len(df):
        raise SystemExit(
            f"[FATAL] {lemma_dir.name}: grouping 只认 1/2，但 uses.csv 有 {len(df)} 行、"
            f"其中 grouping∈{{1,2}} 的只有 {n_total} 行。I1 会被破坏，拒绝继续。"
        )
    return out, len(df)


@torch.no_grad()
def encode_rows(rows, tok, model, device, layers, pool, batch_size, max_length, tag):
    """rows = [(text, L, R), ...] → {layer: [np.ndarray(dim,), ...]}，一一对应、不跳过任何一条。

    多层是在**同一次前向**里取的。扫层时这很重要：模型只跑一遍，
    `output_hidden_states=True` 本来就把每层都算出来了，
    为每一层重跑一次模型纯属浪费（xlm-roberta-large 扫 24 层就是 24 倍）。
    """
    out = {L: [] for L in layers}
    n_cropped = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts, spans = [], []
        for text, L, R in batch:
            tx, l, r = crop_around_target(tok, text, L, R, max_length)
            if tx is not text:
                n_cropped += 1
            texts.append(tx)
            spans.append((l, r))

        enc = tok(texts, return_offsets_mapping=True, padding=True,
                  truncation=True, max_length=max_length, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).hidden_states          # tuple, 每个 (B, T, H)

        # 目标词的 token 下标只依赖分词，和层无关，所以先算一次、各层共用
        token_ids = []
        for j, (l, r) in enumerate(spans):
            om = offsets[j].tolist()
            ids = [k for k, (a, b) in enumerate(om) if a < r and b > l and a != b]
            if not ids:
                raise SystemExit(
                    f"[FATAL] {tag} 第 {i + j} 条用例定位不到目标词 [{l}:{r}]："
                    f"{texts[j][:60]!r}。跳过它会破坏 I1，拒绝继续。"
                )
            token_ids.append(ids)

        for L in layers:
            hs = hidden[L]
            for j, ids in enumerate(token_ids):
                sub = hs[j, ids]                     # (n_sub, H)
                if pool == "mean":
                    vec = sub.mean(0)
                elif pool == "sum":
                    vec = sub.sum(0)
                else:
                    vec = sub[0]
                out[L].append(vec.float().cpu().numpy())
    return out, n_cropped


def parse_layers(spec, n_hidden):
    """--layers 的写法：'all' | '12' | '1,6,12' | '1-12' | '0-24:4'（步长）。

    统一解析成 hidden_states 的下标列表（0 = embedding 层，1..N = 各 Transformer 层）。
    负数按 Python 惯例从后数，-1 = 最后一层。
    """
    if spec.strip() == "all":
        return list(range(n_hidden))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, s = part.split(":", 1)
            step = int(s)
        if "-" in part.lstrip("-"):                  # 区间（但别把开头的负号当区间）
            lo, hi = part.rsplit("-", 1)
            out.extend(range(int(lo), int(hi) + 1, step))
        else:
            out.append(int(part))
    norm = []
    for L in out:
        L = L if L >= 0 else n_hidden + L
        if not 0 <= L < n_hidden:
            sys.exit(f"[FATAL] 层 {L} 越界：这个模型只有 {n_hidden} 个 hidden_states（0..{n_hidden - 1}）")
        if L not in norm:
            norm.append(L)
    return sorted(norm)


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[calc_embeddings_bert] dataset={DATASET} model={args.model} device={device}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tok.is_fast:
        sys.exit(f"[FATAL] {args.model} 没有 fast tokenizer，拿不到 offset_mapping。"
                 f"本脚本的目标词定位依赖它，换个模型或自行实现慢速映射。")
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True)
    model.eval().to(device)

    n_hidden = model.config.num_hidden_layers + 1        # +1 是 embedding 层
    if args.layers:
        layers = parse_layers(args.layers, n_hidden)
    else:
        layer = args.layer if args.layer >= 0 else n_hidden + args.layer
        if not 0 <= layer < n_hidden:
            sys.exit(f"[FATAL] --layer {args.layer} 越界：{args.model} 只有 {n_hidden} 个 hidden_states（0..{n_hidden - 1}）")
        layers = [layer]
    print(f"[calc_embeddings_bert] 导出层 {layers}（共 {n_hidden} 个 hidden_states）")

    input_dir = Path(args.input_dir)
    lemma_dirs = sorted(d for d in (input_dir / "data").iterdir()
                        if d.is_dir() and not d.name.startswith("."))
    if not lemma_dirs:
        sys.exit(f"[FATAL] {input_dir}/data 下没有目标词目录。忘了 export UOT_DATASET 吗？")

    # 每层各一份 {lemma: [vec, ...]}
    src_by_layer = {L: {} for L in layers}
    tgt_by_layer = {L: {} for L in layers}
    total_cropped = 0
    for lemma_dir in tqdm(lemma_dirs, desc="lemmas"):
        lemma = lemma_dir.stem
        rows, n_uses = load_lemma_rows(lemma_dir)
        for gid, store in ((1, src_by_layer), (2, tgt_by_layer)):
            per_layer, nc = encode_rows(rows[gid], tok, model, device, layers, args.pool,
                                        args.batch_size, args.max_length, f"{lemma}/g{gid}")
            for L in layers:
                store[L][lemma] = per_layer[L]
            total_cropped += nc
        # I1 的硬校验：一条都不能少（每层都查，层之间也必须一致）
        for L in layers:
            got = len(src_by_layer[L][lemma]) + len(tgt_by_layer[L][lemma])
            if got != n_uses:
                sys.exit(f"[FATAL] {lemma} 层{L}: 编码了 {got} 条，uses.csv 有 {n_uses} 行。I1 已破坏。")

    # 收尾校验 + 逐层写盘
    tag = args.model.replace("/", "-")
    pool_tag = "" if args.pool == "mean" else f"_{args.pool}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    written = []
    for L in layers:
        src, tgt = src_by_layer[L], tgt_by_layer[L]
        dims, n_vec = set(), 0
        for d in (src, tgt):
            for lemma, vs in d.items():
                for v in vs:
                    if not np.isfinite(v).all():
                        sys.exit(f"[FATAL] {lemma} 层{L} 有 NaN/Inf 向量。"
                                 f"{'试试 --pool sum' if args.pool == 'mean' else ''}")
                    dims.add(v.shape)
                    n_vec += 1
        if len(dims) != 1:
            sys.exit(f"[FATAL] 层{L} 向量维度不一致：{dims}")

        name = args.out if (args.out and len(layers) == 1) else \
            f"{DATASET}_{tag}_L{L}{pool_tag}_embeddings.pkl"
        with open(out_dir / name, "wb") as f:
            pickle.dump((dict(src), dict(tgt)), f)
        written.append((L, name, n_vec, dims.pop()[0]))

    print(f"\n[OK] {len(lemma_dirs)} 词 / 池化 {args.pool} / 居中截断 {total_cropped} 条 / 无 NaN")
    for L, name, n_vec, dim in written:
        print(f"     层 {L:>2}  {n_vec} 向量 × {dim} 维  →  {name}")

    if len(written) == 1:
        print(f"\n下一步：")
        print(f"    UOT_DATASET={DATASET} UOT_EMB_NAME={written[0][1]} python3 src/table3.py")
        print(f"    UOT_DATASET={DATASET} UOT_EMB_NAME={written[0][1]} python3 scripts/bootstrap_ci.py")
    else:
        print(f"\n下一步：python3 scripts/run_layer_sweep.py")


if __name__ == "__main__":
    main()
