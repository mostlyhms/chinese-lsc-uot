# -*- coding: utf-8 -*-
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
    "dwug_en": ("1810\u20151860", "1960\u20152010"),
    "dwug_zh": ("1954\u20151978", "1979\u20152003"),   # ChiWUG
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
