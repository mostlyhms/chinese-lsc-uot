#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_chiwug.py — 把 ChiWUG 转成 Semantic-Shift-via-UOT 流水线可直接读取的 DWUG 目录。

背景 / 为什么需要这一步
----------------------------------------------------------------------
UOT 仓库对 DWUG 数据做了一个**没有写在文档里的隐含假设**：

  src/calc_embeddings.py 按 uses.csv 的“先 grouping==1 的行、再 grouping==2 的行”
  顺序逐条编码，得到向量列表 u（源期）和 v（目标期）；
  src/table2.py / fig6.py 则把 clusters/opt/<lemma>.csv 的**行序**当成
  np.append(u_sus, v_sus) 的下标来用：

      taus = np.append(source_tau, target_tau)
      del_ids = [i for i, c in enumerate(word_instance2cluster[lemma]) if c == -1]
      taus = np.delete(taus, del_ids)

  也就是说 clusters/opt/<lemma>.csv 的第 i 行，必须对应 uses.csv 里
  “grouping 排序后”的第 i 条用例。全程只用位置，从不按 identifier 对齐。

ChiWUG 原始数据不满足这个假设：identifier 集合完全相同，但 clusters/opt/*.csv
的行序是乱的（40 个词全部不一致）。直接跑会“不报错但结果全错”——
instance-level 的 gold 标签和方法预测会错位，Table 2 / Fig. 6 的相关系数变成噪声。

本脚本做的事
----------------------------------------------------------------------
1. 以 “uses.csv 中 grouping==1 的行（保持原文件顺序），再接 grouping==2 的行”
   为**规范顺序**，重写 uses.csv（让文件顺序 == 编码顺序，消除歧义）；
2. 按同一顺序重写 clusters/opt/<lemma>.csv；
3. 跳过 .DS_Store 等隐藏文件（否则 calc_embeddings.py 会去读 .DS_Store/uses.csv 崩掉）；
4. 逐条校验：identifier 集合一致、context[L:R] == lemma、
   簇编号上界 == len(cluster_prob_dist)-1、freq_dist 与实际计数一致；
5. 原样复制 judgments.csv / stats/ / 说明文件。

所有 CSV 都是 TSV（\t 分隔、无引号），脚本按**原始行**重排，不重新序列化，
保证除行序外与 ChiWUG 原始字节完全一致。
"""

import argparse
import ast
import collections
import csv
import io
import os
import shutil
import sys
from pathlib import Path

SRC_GID = "1"
TGT_GID = "2"


def read_tsv_lines(path):
    """返回 (header_line, [data_line, ...])，均为不含换行符的原始字符串。"""
    with io.open(path, encoding="utf-8", newline="") as f:
        raw = f.read()
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [ln.rstrip("\r") for ln in lines]
    return lines[0], lines[1:]


def parse_tsv(path):
    with io.open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def write_tsv_lines(path, header, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for ln in lines:
            f.write(ln + "\n")


def main():
    ap = argparse.ArgumentParser(description="Convert ChiWUG into a UOT-pipeline-ready DWUG directory")
    ap.add_argument("--chiwug", required=True, help="ChiWUG 根目录（内含 data/ clusters/ stats/）")
    ap.add_argument("--out", required=True, help="输出目录，例如 data/dwug_zh")
    ap.add_argument("--strict", action="store_true", help="任何校验失败即退出（默认只警告）")
    args = ap.parse_args()

    src = Path(args.chiwug).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not (src / "data").is_dir():
        sys.exit(f"[FATAL] 找不到 {src/'data'}")

    lemmas = sorted(
        d.name for d in (src / "data").iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    print(f"[info] 目标词 {len(lemmas)} 个")

    # stats_groupings：用于校验簇编号上界
    sg_path = src / "stats" / "opt" / "stats_groupings.csv"
    lemma2dists = {}
    if sg_path.is_file():
        for r in parse_tsv(sg_path):
            lemma2dists[r["lemma"]] = (
                ast.literal_eval(r["cluster_freq_dist"]),
                ast.literal_eval(r["cluster_prob_dist1"]),
                ast.literal_eval(r["cluster_prob_dist2"]),
            )

    problems = []
    reordered_uses = 0
    reordered_clusters = 0
    n_uses_total = 0

    for lemma in lemmas:
        uses_path = src / "data" / lemma / "uses.csv"
        clus_path = src / "clusters" / "opt" / f"{lemma}.csv"
        if not uses_path.is_file():
            problems.append(f"{lemma}: 缺少 uses.csv")
            continue
        if not clus_path.is_file():
            problems.append(f"{lemma}: 缺少 clusters/opt/{lemma}.csv")
            continue

        u_header, u_lines = read_tsv_lines(uses_path)
        u_rows = parse_tsv(uses_path)
        if len(u_rows) != len(u_lines):
            problems.append(f"{lemma}: uses.csv 存在跨行字段（行数 {len(u_lines)} != 记录数 {len(u_rows)}）")
            if args.strict:
                sys.exit(1)
            continue

        # --- 规范顺序 ---
        idx_src = [i for i, r in enumerate(u_rows) if r["grouping"] == SRC_GID]
        idx_tgt = [i for i, r in enumerate(u_rows) if r["grouping"] == TGT_GID]
        other = [i for i, r in enumerate(u_rows) if r["grouping"] not in (SRC_GID, TGT_GID)]
        if other:
            problems.append(f"{lemma}: 出现 grouping 不属于 {{1,2}} 的 {len(other)} 行，已丢弃")
        canon_idx = idx_src + idx_tgt
        canon_ids = [u_rows[i]["identifier"] for i in canon_idx]

        if canon_idx != list(range(len(u_rows))):
            reordered_uses += 1

        # --- 索引与 lemma 校验 ---
        for i in canon_idx:
            r = u_rows[i]
            try:
                L, R = map(int, r["indexes_target_token"].split(":"))
            except Exception:
                problems.append(f"{lemma}/{r['identifier']}: indexes_target_token 无法解析")
                continue
            if r["context"][L:R] != r["lemma"]:
                problems.append(
                    f"{lemma}/{r['identifier']}: context[{L}:{R}]="
                    f"{r['context'][L:R]!r} != lemma"
                )

        write_tsv_lines(out / "data" / lemma / "uses.csv", u_header, [u_lines[i] for i in canon_idx])
        n_uses_total += len(canon_idx)

        # --- clusters 重排 ---
        c_header, c_lines = read_tsv_lines(clus_path)
        c_rows = parse_tsv(clus_path)
        if len(c_rows) != len(c_lines):
            problems.append(f"{lemma}: clusters csv 行数与记录数不一致")
            continue
        id2line = {}
        id2cluster = {}
        for r, ln in zip(c_rows, c_lines):
            id2line[r["identifier"]] = ln
            id2cluster[r["identifier"]] = int(r["cluster"])

        missing = [i for i in canon_ids if i not in id2line]
        extra = [i for i in id2line if i not in set(canon_ids)]
        if missing or extra:
            problems.append(f"{lemma}: clusters 与 uses 的 identifier 不匹配 (缺 {len(missing)} / 多 {len(extra)})")
            if args.strict:
                sys.exit(1)
            continue
        if [r["identifier"] for r in c_rows] != canon_ids:
            reordered_clusters += 1
        write_tsv_lines(out / "clusters" / "opt" / f"{lemma}.csv", c_header, [id2line[i] for i in canon_ids])

        # --- 簇编号 vs. stats_groupings 校验 ---
        if lemma in lemma2dists:
            fd, d1, d2 = lemma2dists[lemma]
            ids = [id2cluster[i] for i in canon_ids]
            valid = [c for c in ids if c != -1]
            if valid and max(valid) != len(d1) - 1:
                problems.append(
                    f"{lemma}: 最大簇编号 {max(valid)} 与 cluster_prob_dist 长度 {len(d1)} 不匹配"
                    f"（table2.py 会 KeyError）"
                )
            if len(d1) != len(d2) or len(fd) != len(d1):
                problems.append(f"{lemma}: cluster_*_dist 长度不一致")
            cnt = collections.Counter(valid)
            got = [cnt.get(k, 0) for k in range(len(fd))]
            if got != list(fd):
                problems.append(f"{lemma}: 簇频次 {got} 与 stats_groupings 的 {fd} 不一致")
            if any(a == 0 and b == 0 for a, b in zip(d1, d2)):
                problems.append(f"{lemma}: 存在两期概率均为 0 的簇（log 比值会是 nan）")

        # --- judgments 原样复制 ---
        j = src / "data" / lemma / "judgments.csv"
        if j.is_file():
            (out / "data" / lemma).mkdir(parents=True, exist_ok=True)
            shutil.copy2(j, out / "data" / lemma / "judgments.csv")

    # --- stats 与说明文件原样复制 ---
    for rel in ["stats/opt/stats.csv", "stats/opt/stats_groupings.csv",
                "stats/stats_agreement.csv", "annotators.csv", "README.html"]:
        p = src / rel
        if p.is_file():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

    print(f"[info] 输出 -> {out}")
    print(f"[info] 用例总数 {n_uses_total}")
    print(f"[info] 重排了 uses.csv 的词: {reordered_uses}/{len(lemmas)}")
    print(f"[info] 重排了 clusters/opt/*.csv 的词: {reordered_clusters}/{len(lemmas)}")
    if problems:
        print(f"[WARN] {len(problems)} 条校验问题：")
        for p in problems[:50]:
            print("   -", p)
        if len(problems) > 50:
            print(f"   ... 其余 {len(problems)-50} 条略")
        if args.strict:
            sys.exit(1)
    else:
        print("[OK] 全部校验通过：identifier 对齐、目标词索引正确、簇编号自洽。")


if __name__ == "__main__":
    main()
