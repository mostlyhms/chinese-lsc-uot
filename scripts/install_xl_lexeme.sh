#!/usr/bin/env bash
# 安装 XL-LEXEME（绕开 xl-lexeme 自带 setup.py 的 bug）。
#
# 为什么原版装不上：
#   xl-lexeme 的 setup.py 里写着
#       from pip._internal.network.session import PipSession
#   而现代 pip 默认在**隔离的构建环境**里跑 setup.py，那个环境里根本没有 pip，
#   于是在 "getting requirements to build wheel" 这一步就 ModuleNotFoundError，
#   报出来的就是 `ERROR: Failed to build ... when getting requirements to build wheel`。
#   （这跟你的网络、Python 版本都没关系，是上游的 bug。）
#
# 本脚本把 setup.py 换成一个不碰 pip 私有 API 的版本，再用 --no-deps 安装
#（--no-deps 是必须的：原版 install_requires 里 sentence-transformers 没钉版本，
#  会把你刚装好的 3.1.0 升到最新，然后又崩）。
#
# 用法（先 source 好 venv）：
#   bash scripts/install_xl_lexeme.sh
set -euo pipefail

DEST="${1:-third_party/xl-lexeme}"

if ! python -c "import sentence_transformers" 2>/dev/null; then
  echo "[FATAL] 当前环境没有 sentence-transformers。先跑：pip install -r requirements-zh.txt" >&2
  exit 1
fi

if [ ! -d "$DEST" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 https://github.com/pierluigic/xl-lexeme.git "$DEST"
fi

cat > "$DEST/setup.py" <<'PY'
# 由 scripts/install_xl_lexeme.sh 重写：原版依赖 pip._internal，
# 在 pip 的隔离构建环境里必然失败。
import setuptools

setuptools.setup(
    name="WordTransformer",
    version="0.0.1",
    author="Pierluigi Cassotti",
    description="WiC Pretrained Model for Cross-Lingual LEXical sEMantic changE",
    url="https://github.com/pierluigic/xl-lexeme",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=["sentence-transformers"],
)
PY

pip install --no-deps "$DEST"

python - <<'PY'
from WordTransformer import WordTransformer, InputExample
import transformers, sentence_transformers, huggingface_hub, torch, sys
print(f"[OK] WordTransformer 导入成功")
print(f"     python={sys.version.split()[0]} torch={torch.__version__} "
      f"transformers={transformers.__version__} "
      f"sentence-transformers={sentence_transformers.__version__} "
      f"huggingface_hub={huggingface_hub.__version__}")
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 46), \
    "transformers 必须 < 4.46（4.46 起删了 transformers.AdamW）"
print("[OK] 版本约束检查通过。下一步：python src/calc_embeddings.py")
PY
