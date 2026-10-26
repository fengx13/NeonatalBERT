#!/usr/bin/env python
"""
Step 1 of NeonatalBERT pretraining: convert clinical notes into a plain-text
corpus for BERT pretraining (one sentence per line, a blank line between
documents).

Adapted from the ClinicalBERT preprocessing workflow
(https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/lm_pretraining/format_mimic_for_BERT.py).
If ClinicalBERT's `heuristic_tokenize.py` is on the Python path, its
`sent_tokenize_rules` is used for sentence splitting; otherwise a built-in
rule-based splitter is used.

In the paper, the pretraining corpus consisted of notes written in the first
3 months after birth for patients in the development (training) set only.
Restrict your input accordingly to avoid leakage into the test set.

Usage
-----
python model_pretraining/prepare_notes.py \
    --input_path notes.pkl --output_path corpus/neonatal_notes.txt \
    [--text_col note_text] [--min_sentence_chars 3]
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import read_table  # noqa: E402

try:  # optional: ClinicalBERT heuristic sentence splitter
    from heuristic_tokenize import sent_tokenize_rules  # type: ignore
except ImportError:  # pragma: no cover
    sent_tokenize_rules = None

logger = logging.getLogger("neonatalbert.prepare_notes")

_ABBREV = {"dr.", "mr.", "ms.", "mrs.", "vs.", "approx.", "wk.", "wks.", "hr.", "hrs.", "min.",
           "mo.", "yo.", "pt.", "no.", "st.", "e.g.", "i.e.", "q.", "b.i.d.", "t.i.d."}
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
_HEADER = re.compile(r"^\s*[A-Z][A-Za-z /&()-]{1,40}:\s*$")


def _split_on_sentence_ends(segment: str):
    pieces, out = _SENT_END.split(segment), []
    for piece in pieces:
        # Re-join pieces that were split after a common abbreviation.
        if out and out[-1].split()[-1].lower() in _ABBREV:
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def builtin_sentence_split(text: str):
    """Split on blank lines, section headers, list items and sentence ends."""
    sentences = []
    text = text.replace("\r", "\n")
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        buf = []
        for line in lines:
            # Headers, bullets and numbered items start a new segment.
            if _HEADER.match(line) or re.match(r"^([-*•]|\d+[.)])\s+", line):
                if buf:
                    sentences.append(" ".join(buf))
                    buf = []
            buf.append(line)
        if buf:
            sentences.append(" ".join(buf))
    out = []
    for seg in sentences:
        out.extend(s.strip() for s in _split_on_sentence_ends(seg) if s.strip())
    return out


def split_sentences(text: str):
    if sent_tokenize_rules is not None:
        return [s.strip() for s in sent_tokenize_rules(text) if s.strip()]
    return builtin_sentence_split(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_path", required=True, help=".pkl/.csv/.parquet with a text column")
    ap.add_argument("--output_path", required=True, help="Output .txt corpus")
    ap.add_argument("--text_col", default=None, help="Text column (default: 'text' or 'note_text')")
    ap.add_argument("--min_sentence_chars", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    df = read_table(args.input_path)
    col = args.text_col or ("text" if "text" in df else "note_text")
    logger.info("Read %d notes from %s (column '%s')", len(df), args.input_path, col)
    logger.info("Sentence splitter: %s", "ClinicalBERT heuristic_tokenize" if sent_tokenize_rules else "built-in rules")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_sents = 0
    with open(args.output_path, "w") as f:
        for text in tqdm(df[col].dropna().astype(str), total=len(df)):
            sents = [re.sub(r"\s+", " ", s) for s in split_sentences(text)]
            sents = [s for s in sents if len(s) >= args.min_sentence_chars]
            if len(sents) < 2:  # next-sentence prediction needs >= 2 sentences
                continue
            f.write("\n".join(sents) + "\n\n")
            n_docs += 1
            n_sents += len(sents)
    logger.info("Wrote %d documents / %d sentences to %s", n_docs, n_sents, args.output_path)


if __name__ == "__main__":
    main()
