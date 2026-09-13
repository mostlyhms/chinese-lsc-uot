import csv
import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from WordTransformer import WordTransformer
from WordTransformer import InputExample
import torch
import argparse
# --- chiwug patch: dataset switch ---
from dataset_config import DATASET, DATA_DIR, EMB_NAME, PERIOD_SRC, PERIOD_TGT

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate embeddings for DWUG English dataset")
    parser.add_argument("--input_dir", type=str, default=DATA_DIR, help="Input directory containing DWUG data")
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory for embeddings")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | mps | cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[calc_embeddings] device={device}")
    model = WordTransformer("pierluigic/xl-lexeme", device=device)

    src_gid = 1
    tgt_gid = 2
    source_token2vecs = defaultdict(list)
    target_token2vecs = defaultdict(list)

    # 过滤 .DS_Store 等隐藏项，否则会尝试读取 ".DS_Store/uses.csv"
    lemma_dirs = sorted(
        d for d in (input_dir / "data").iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    for lemma_dir in tqdm(lemma_dirs):
        lemma = lemma_dir.stem
        csv_path = lemma_dir / "uses.csv"
        df = pd.read_table(csv_path, quoting=csv.QUOTE_NONE)
        
        for gid in [src_gid, tgt_gid]:
            if gid == src_gid:
                token2vecs = source_token2vecs
            else:
                token2vecs = target_token2vecs
            
            sentences = df[df["grouping"]==gid]["context"].tolist()
            indexes = df[df["grouping"]==gid]["indexes_target_token"].tolist()
            for sentence, index in tqdm(list(zip(sentences, indexes))):
                L, R = map(int, index.split(":"))
                input_example = InputExample(texts=sentence, positions=[L, R])
                vec = model.encode(input_example)
                token2vecs[lemma].append(vec)

    with open(output_dir/EMB_NAME, "wb") as f:
        pickle.dump((source_token2vecs, target_token2vecs), f)

if __name__ == "__main__":
    main()