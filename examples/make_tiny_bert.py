#!/usr/bin/env python
"""
Create a tiny, randomly initialised BERT (vocabulary built from a text file)
for offline smoke tests of the pipeline. It has no clinical knowledge; use
it only to check that the scripts run (for example on a machine without
internet access or before NeonatalBERT weights are available).
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="CSV with a note_text or text column")
    ap.add_argument("--out_dir", default="examples/tiny_bert")
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)
    col = "note_text" if "note_text" in df else "text"
    words = Counter(w for t in df[col].astype(str) for w in re.findall(r"\w+|[^\w\s]", t.lower()))
    vocab = SPECIAL + sorted(words)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")

    try:  # transformers >= 5 takes the vocabulary as a dict
        tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)
        assert len(tok.get_vocab()) == len(vocab)
    except (TypeError, AssertionError):  # transformers 4.x
        tok = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True)
    assert len(tok.get_vocab()) == len(vocab), "tokenizer vocabulary was not loaded"
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=512)
    BertForPreTraining(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    print(f"Tiny BERT with {len(vocab)} tokens saved to {out}")


if __name__ == "__main__":
    main()
