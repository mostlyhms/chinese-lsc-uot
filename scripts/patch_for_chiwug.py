#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_for_chiwug.py — 让 Semantic-Shift-via-UOT 的 src/*.py 支持切换到中文数据集。

原仓库为英文 DWUG 写死了三类东西：
  1. 路径：data/dwug_en/... 和 embeddings/dwug_en_embeddings.pkl（7 个脚本）
  2. 图例上的时期标签："1810―1860" / "1960―2010"（utils.py, fig4.py, fig5b.py）
     —— ChiWUG 的两期是 1954―1978 / 1979―2003，不改的话图全是错的
  3. calc_embeddings.py 遍历 data/ 时不过滤隐藏文件（macOS 的 .DS_Store 会让它崩），
     且只认 CUDA（Mac 上白跑 CPU）

本脚本把这三类都改掉，改成从 src/dataset_config.py 读取，用环境变量切换：
    export UOT_DATASET=dwug_zh   # ChiWUG
    export UOT_DATASET=dwug_en   # 原论文英文 DWUG（默认）

幂等：重复执行安全。首次执行时把原文件备份为 *.orig；
带 --restore 可以从 .orig 还原后重新打补丁。
"""
import argparse
import ast
import io
import re
import shutil
import sys
from pathlib import Path

MARKER = "# --- chiwug patch: dataset switch ---"
IMPORT_LINE = "from dataset_config import DATASET, DATA_DIR, EMB_NAME, PERIOD_SRC, PERIOD_TGT"

CONFIG_SRC = '''# -*- coding: utf-8 -*-
"""数据集切换开关（由 scripts/patch_for_chiwug.py 生成）。

    export UOT_DATASET=dwug_zh   # ChiWUG（中文，人民日报 1954-2003）
    export UOT_DATASET=dwug_en   # 原论文英文 DWUG（默认）

新增数据集时在 _PERIODS 里补一行即可。
"""
import os

DATASET = os.environ.get("UOT_DATASET", "dwug_en")
DATA_DIR = f"data/{DATASET}"

# embeddings 文件名。默认 {DATASET}_embeddings.pkl（XL-LEXEME 基线）。
# 换编码器时不用改代码，用环境变量指过去即可：
#   UOT_EMB_NAME=dwug_zh_bert-base-chinese_L12_embeddings.pkl python3 src/table3.py
EMB_NAME = os.environ.get("UOT_EMB_NAME") or f"{DATASET}_embeddings.pkl"

# 图例里两个时期的标签（grouping 1 / grouping 2）
_PERIODS = {
    "dwug_en": ("1810\\u20151860", "1960\\u20152010"),
    "dwug_zh": ("1954\\u20151978", "1979\\u20152003"),   # ChiWUG
}
PERIOD_SRC, PERIOD_TGT = _PERIODS.get(DATASET, ("group 1", "group 2"))


# 字体必须真的有这些码位，光看名字不够（见下）
_REQUIRED_CP = (
    0x4E0B,   # 下 —— CJK，图标题/图例里的目标词
    0x2015,   # ― —— PERIOD_SRC/TGT 里的那根横线（U+2015 HORIZONTAL BAR）
)
_PREFERRED_CP = (0x2212,)   # − 数学减号，缺了不致命


def _font_covers(path, codepoints):
    """查字体文件的 cmap。fontTools 装了才查，没装就返回 None（表示不知道）。"""
    try:
        from fontTools.ttLib import TTFont
    except Exception:
        return None
    try:
        f = TTFont(path, fontNumber=0) if path.lower().endswith((".ttc", ".otc")) else TTFont(path)
        tables = f["cmap"].tables
        return all(any(cp in tb.cmap for tb in tables) for cp in codepoints)
    except Exception:
        return None


def setup_cjk_font():
    """给 matplotlib 挑一个能显示中文的字体，否则中文标题/图例是豆腐块。

    这里不只按名字挑，还要查 cmap 确认字体真的有需要的码位。原因：
    macOS 上 PingFang SC 是 .ttc，matplotlib 的 ttflist 常常根本不收录它，
    于是名单会一路掉到 **Songti SC**——那是个**衬线**字体，而且**没有 U+2015**。
    结果就是图例里 "1954―1978" 的那根横线直接消失（渲染成空白），
    正文数字还莫名其妙变成衬线体。名字对、字体在、图是错的，属于不报错的坑。

    候选按「无衬线 CJK 优先」排；Songti SC 是衬线，压到最后当兜底。
    """
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return None
    candidates = [
        "PingFang SC", "PingFang HK", "Heiti SC", "Heiti TC", "STHeiti",   # macOS 无衬线
        "Hiragino Sans GB", "Lantinghei SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",       # Linux
        "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei",                   # Windows
        "Songti SC",                                                        # 衬线，兜底
    ]
    name2file = {}
    for f in font_manager.fontManager.ttflist:
        name2file.setdefault(f.name, f.fname)

    def apply(name):
        matplotlib.rcParams["font.family"] = ["sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = [name] + list(
            matplotlib.rcParams.get("font.sans-serif", [])
        )
        matplotlib.rcParams["axes.unicode_minus"] = False
        return name

    fallback = None
    for name in candidates:
        path = name2file.get(name)
        if path is None:
            continue
        ok = _font_covers(path, _REQUIRED_CP)
        if ok is None:          # 没装 fontTools，查不了，只能信名字
            return apply(name)
        if ok:
            if _font_covers(path, _PREFERRED_CP):
                return apply(name)
            fallback = fallback or name      # 缺减号而已，先记着
    if fallback:
        return apply(fallback)
    return None


if DATASET.endswith("_zh") or DATASET == "chiwug":
    _picked = setup_cjk_font()
    if _picked is None:
        import warnings
        warnings.warn(
            "[dataset_config] 没找到可用的中文字体，图里的中文会显示成方块。"
            "macOS 上通常自带 PingFang SC；若仍失败，装一个 Noto Sans CJK SC。"
        )
'''

REPLACEMENTS = [
    ('f"data/dwug_en/', 'f"{DATA_DIR}/'),
    ('"data/dwug_en/',  'f"{DATA_DIR}/'),
    ('"data/dwug_en"',  'DATA_DIR'),
    ('"embeddings/dwug_en_embeddings.pkl"', 'f"embeddings/{EMB_NAME}"'),
    ('"dwug_en_embeddings.pkl"', 'EMB_NAME'),
    ('r"1810―1860"', 'PERIOD_SRC'),
    ('r"1960―2010"', 'PERIOD_TGT'),
    ('"1810―1860"', 'PERIOD_SRC'),
    ('"1960―2010"', 'PERIOD_TGT'),
]

TARGETS = ["calc_embeddings.py", "utils.py", "fig1.py", "fig3.py", "fig4.py",
           "fig5a.py", "fig5b.py", "fig6.py", "table2.py", "table3.py", "table4.py"]

OLD_LOOP = 'for lemma_dir in tqdm(sorted((input_dir / "data").iterdir())):'
NEW_LOOP = (
    '# 过滤 .DS_Store 等隐藏项，否则会尝试读取 ".DS_Store/uses.csv"\n'
    '    lemma_dirs = sorted(\n'
    '        d for d in (input_dir / "data").iterdir()\n'
    '        if d.is_dir() and not d.name.startswith(".")\n'
    '    )\n'
    '    for lemma_dir in tqdm(lemma_dirs):'
)
OLD_DEV = 'device = "cuda:0" if torch.cuda.is_available() else "cpu"'
NEW_DEV = (
    'if args.device != "auto":\n'
    '        device = args.device\n'
    '    elif torch.cuda.is_available():\n'
    '        device = "cuda:0"\n'
    '    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():\n'
    '        device = "mps"\n'
    '    else:\n'
    '        device = "cpu"\n'
    '    print(f"[calc_embeddings] device={device}")'
)
# table3.py 的方法列表里补上 f_PRT。
# PRT 是 Periti & Tahmasebi (NAACL 2024) 在 ChiWUG 上字级 BERT 的最佳搭配（.712 vs APD .656），
# 上游 UOT 仓库只有 APD，缺 PRT 会系统性低估 bert-base-chinese 这类编码器。
OLD_METHODS3 = 'methods = ["f_SUS", "f_OT", "f_APD", "f_LDR", "f_WiDiD", "f_APDP"]'
NEW_METHODS3 = 'methods = ["f_SUS", "f_OT", "f_APD", "f_PRT", "f_LDR", "f_WiDiD", "f_APDP"]'

# ── utils.py: t-SNE 散点图被压扁的问题（CLAUDE.md T4）────────────────────────
# `scatter_plot()` 结尾写死 ax.set_aspect('equal')。t-SNE 展布宽（下海那张 x 跨 13、
# y 只跨 1.6）时，等比例会把整个绘图区压成一条，连 colorbar 的标题都被裁掉。
#
# 直接改成 'auto' 会填满画布，但那等于把 x/y 拉伸成不同尺度——t-SNE 的两个轴本来是
# 同一个空间的同一种距离，拉伸会让人误判簇间距离。所以默认给第三种：'pad'——
# **保持等比例**（距离仍然可读），但把窄的那根轴对称补白到画布的长宽比。
# 换来的是留白，而不是失真。
OLD_ASPECT_SIG = ("def scatter_plot(u, v, u_sus, v_sus, fig, ax, legend=True, "
                  "max_abs_sus=None, cmap=None, ldr=False):")
NEW_ASPECT_SIG = ("def set_plot_aspect(ax, mode=\"pad\", ratio=None, fig=None):\n"
                  "    \"\"\"t-SNE 散点的长宽比。mode: pad(默认) | equal | auto\n"
                  "\n"
                  "    pad  —— 等比例 + 把窄轴对称补白到 ratio，图能填满画布且距离不失真\n"
                  "    equal—— 上游原行为，展布悬殊时会被压成一条\n"
                  "    auto —— 填满画布但 x/y 尺度不同，t-SNE 上会误导，慎用\n"
                  "    \"\"\"\n"
                  "    if mode == \"auto\":\n"
                  "        ax.set_aspect(\"auto\")\n"
                  "        return\n"
                  "    if mode == \"pad\":\n"
                  "        if ratio is None:\n"
                  "            # 目标长宽比取绘图区的物理长宽比，这样补白后正好填满\n"
                  "            f = fig if fig is not None else ax.get_figure()\n"
                  "            pos, (fw, fh) = ax.get_position(), f.get_size_inches()\n"
                  "            ratio = (pos.width * fw) / (pos.height * fh)\n"
                  "        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()\n"
                  "        dx, dy = x1 - x0, y1 - y0\n"
                  "        if dx <= 0 or dy <= 0:\n"
                  "            ax.set_aspect(\"equal\")\n"
                  "            return\n"
                  "        if dx / dy > ratio:      # 太扁：把 y 补高\n"
                  "            pad = (dx / ratio - dy) / 2\n"
                  "            ax.set_ylim(y0 - pad, y1 + pad)\n"
                  "        else:                    # 太窄：把 x 补宽\n"
                  "            pad = (dy * ratio - dx) / 2\n"
                  "            ax.set_xlim(x0 - pad, x1 + pad)\n"
                  "    ax.set_aspect(\"equal\")\n"
                  "\n"
                  "def scatter_plot(u, v, u_sus, v_sus, fig, ax, legend=True, "
                  "max_abs_sus=None, cmap=None, ldr=False, aspect=\"pad\", aspect_ratio=None):")
OLD_ASPECT = "    ax.set_aspect('equal')\n    return cax"
NEW_ASPECT = "    set_plot_aspect(ax, aspect, aspect_ratio, fig)\n    return cax"
# multiple_scatter_plot 里共享的 xlim/ylim 是在 scatter_plot 返回**之后**才设的，
# 所以补白必须在那之后重做一遍，否则会被覆盖回去。
OLD_MULTI_SIG = ("def multiple_scatter_plot(u_list, v_list, u_sus_list, v_sus_list, "
                 "fig, axs, tgt_words, legend=True):")
NEW_MULTI_SIG = ("def multiple_scatter_plot(u_list, v_list, u_sus_list, v_sus_list, "
                 "fig, axs, tgt_words, legend=True, aspect=\"pad\", aspect_ratio=None):")
OLD_MULTI_LIM = ("        ax.set_xlim(min_x, max_x)\n"
                 "        ax.set_ylim(min_y, max_y)\n"
                 "        ax.set_title(tgt_words[i], fontsize=20)")
NEW_MULTI_LIM = ("        ax.set_xlim(min_x, max_x)\n"
                 "        ax.set_ylim(min_y, max_y)\n"
                 "        set_plot_aspect(ax, aspect, aspect_ratio, fig)\n"
                 "        ax.set_title(tgt_words[i], fontsize=20)")

OLD_ARG = '    return parser.parse_args()'

NEW_ARG = ('    parser.add_argument("--device", type=str, default="auto",\n'
           '                        help="auto | cpu | mps | cuda:0")\n'
           '    return parser.parse_args()')


def insert_import(text: str) -> str:
    if MARKER in text:
        return re.sub(r"^from dataset_config import .*$", IMPORT_LINE, text, flags=re.M)
    lines = text.split("\n")
    last_import = 0
    for i, ln in enumerate(lines[:80]):
        if re.match(r"^(import |from )\S", ln):
            last_import = i
    lines.insert(last_import + 1, f"{MARKER}\n{IMPORT_LINE}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Semantic-Shift-via-UOT 仓库根目录")
    ap.add_argument("--restore", action="store_true", help="先从 *.orig 还原再重新打补丁")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    src = repo / "src"
    if not src.is_dir():
        sys.exit(f"[FATAL] 找不到 {src}")

    io.open(src / "dataset_config.py", "w", encoding="utf-8").write(CONFIG_SRC)
    print("[ok] 写入 src/dataset_config.py")

    for name in TARGETS:
        p = src / name
        if not p.is_file():
            print(f"[skip] 缺少 {name}")
            continue
        backup = p.with_suffix(p.suffix + ".orig")
        if args.restore and backup.exists():
            shutil.copy2(backup, p)
        if not backup.exists():
            shutil.copy2(p, backup)

        text = orig = io.open(p, encoding="utf-8").read()
        n = 0
        for old, new in REPLACEMENTS:
            if old in text:
                n += text.count(old)
                text = text.replace(old, new)

        if name == "calc_embeddings.py":
            for old, new in ((OLD_LOOP, NEW_LOOP), (OLD_DEV, NEW_DEV)):
                if old in text:
                    text = text.replace(old, new)
                    n += 1
            if '"--device"' not in text and OLD_ARG in text:
                text = text.replace(OLD_ARG, NEW_ARG, 1)
                n += 1

        if name == "table3.py" and OLD_METHODS3 in text:
            text = text.replace(OLD_METHODS3, NEW_METHODS3, 1)
            n += 1

        if name == "utils.py":
            for old, new in ((OLD_ASPECT_SIG, NEW_ASPECT_SIG), (OLD_ASPECT, NEW_ASPECT),
                             (OLD_MULTI_SIG, NEW_MULTI_SIG), (OLD_MULTI_LIM, NEW_MULTI_LIM)):
                if old in text:
                    text = text.replace(old, new, 1)
                    n += 1

        if n or MARKER in text:
            text = insert_import(text)
        if text != orig:
            io.open(p, "w", encoding="utf-8").write(text)
            print(f"[ok] {name}: {n} 处替换")
        else:
            print(f"[--] {name}: 无需改动")

    bad = []
    for name in TARGETS + ["dataset_config.py"]:
        p = src / name
        if p.is_file():
            try:
                ast.parse(io.open(p, encoding="utf-8").read())
            except SyntaxError as e:
                bad.append(f"{name}: {e}")
    if bad:
        print("[FATAL] 补丁后语法错误：")
        for b in bad:
            print("   -", b)
        sys.exit(1)
    print("[OK] 语法检查通过。下一步：export UOT_DATASET=dwug_zh")


if __name__ == "__main__":
    main()
