#!/usr/bin/env python
"""
Zero-shot LLM baseline (e.g. Meta Llama-3.1-8B-Instruct) for neonatal outcome
prediction from first-day clinical notes, as compared with NeonatalBERT in
Xie et al., Lancet Digit Health 2025.

For every patient and outcome, the model receives the first-day notes and one
yes/no question, and answers with "1" (yes) or "0" (no). Instead of sampling
text, the script reads the next-token probabilities of "1" and "0":

    p(yes) = P("1") / (P("1") + P("0"))

This gives a deterministic answer (p >= 0.5 -> yes) for sensitivity and
specificity, as well as a continuous score for AUROC/AUPRC.

The same cohort construction and train/test split as finetune.py are used,
and only test patients are scored, so the results are directly comparable.

Usage
-----
python baseline_evaluation/llm_baseline.py \
    --notes data/notes.csv --labels data/labels.csv \
    --split column --split_col split --test_value test \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir results/llama31_8b
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data_utils import build_cohort, split_cohort  # noqa: E402
from helpers import bootstrap_metrics, parse_list, save_json, summarise_results  # noqa: E402

logger = logging.getLogger("neonatalbert.llm_baseline")

OUTCOME_NAMES = {
    "rds": "respiratory distress syndrome (RDS)",
    "bpd": "bronchopulmonary dysplasia (BPD)",
    "pulmonary_haemorrhage": "pulmonary haemorrhage",
    "pulmonary_hypertension": "pulmonary hypertension",
    "atelectasis": "atelectasis",
    "aspiration": "aspiration syndrome",
    "ivh": "intraventricular haemorrhage (IVH)",
    "pvl": "periventricular leukomalacia (PVL)",
    "seizures": "neonatal seizures",
    "cns_other": "another central nervous system disorder",
    "pda": "patent ductus arteriosus (PDA)",
    "cv_instability": "cardiovascular instability",
    "sepsis": "sepsis",
    "candidiasis": "candidiasis",
    "anaemia": "anaemia",
    "jaundice": "neonatal jaundice",
    "nec": "necrotising enterocolitis (NEC)",
    "rop": "retinopathy of prematurity (ROP)",
}

SYSTEM_PROMPT = (
    "You are an experienced neonatologist. Based only on the clinical notes "
    "written during the first 24 hours after birth, predict the newborn's outcome. "
    "Answer with a single digit: 1 for yes, 0 for no."
)


def question_for(outcome: str) -> str:
    if outcome in ("death", "mortality"):
        return "Will this newborn die during the hospital stay?"
    return f"Will this newborn develop {OUTCOME_NAMES.get(outcome, outcome.replace('_', ' '))}?"


def build_prompt(tokenizer, notes: str, outcome: str) -> str:
    user = f"Clinical notes:\n{notes}\n\nQuestion: {question_for(outcome)}\nAnswer 1 (yes) or 0 (no)."
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user}\nAnswer:"


def truncate_notes(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def answer_token_ids(tokenizer):
    ids = {}
    for ans in ["1", "0"]:
        toks = tokenizer(ans, add_special_tokens=False)["input_ids"]
        ids[ans] = toks[-1]
    if ids["1"] == ids["0"]:
        raise ValueError("Tokenizer maps '1' and '0' to the same token id")
    return ids["1"], ids["0"]


@torch.no_grad()
def score_prompts(model, tokenizer, prompts, batch_size, yes_id, no_id):
    probs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring", disable=not sys.stderr.isatty()):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :]  # left padding -> last position is the next token
        two = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1).float()
        probs.append(torch.softmax(two, dim=-1)[:, 0].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outcomes", default=None)
    ap.add_argument("--max_hours", type=float, default=24)
    ap.add_argument("--split", choices=["random", "column", "none"], default="random")
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--test_value", default="test")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max_note_tokens", type=int, default=6000, help="Truncate notes to fit the context window")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--n_bootstraps", type=int, default=1000)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    split_col = args.split_col if args.split == "column" else None
    cohort, outcomes = build_cohort(args.notes, args.labels, parse_list(args.outcomes), args.max_hours, None, split_col)
    _, _, test_df = split_cohort(cohort, args.split, split_col, args.test_value, args.test_size, args.val_size, args.seed)
    logger.info("Scoring %d test patients x %d outcomes with %s", len(test_df), len(outcomes), args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=getattr(torch, args.dtype) if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    yes_id, no_id = answer_token_ids(tokenizer)

    notes = [truncate_notes(tokenizer, t, args.max_note_tokens) for t in test_df["text"]]
    preds, results = [], {}
    for o in outcomes:
        prompts = [build_prompt(tokenizer, n, o) for n in notes]
        p_yes = score_prompts(model, tokenizer, prompts, args.batch_size, yes_id, no_id)
        y = test_df[o].to_numpy()
        preds.append(pd.DataFrame({"patient_id": test_df["patient_id"], "outcome": o, "y_true": y,
                                   "p_yes": p_yes, "answer": (p_yes >= 0.5).astype(int)}))
        if len(np.unique(y)) < 2:
            logger.warning("Skipping metrics for %s: only one class in the test set", o)
            continue
        # threshold 0.5 = the model's own yes/no answer
        results[o] = bootstrap_metrics(y, p_yes, n_bootstraps=args.n_bootstraps, random_seed=args.seed, threshold=0.5)
        logger.info("%s | AUROC %.3f | sensitivity %.3f | specificity %.3f", o,
                    results[o]["auroc"]["estimate"], results[o]["sensitivity"]["estimate"], results[o]["specificity"]["estimate"])

    pd.concat(preds).to_csv(out / "test_predictions.csv", index=False)
    save_json(results, out / "test_metrics.json")
    summary = summarise_results(results)
    summary.to_csv(out / "test_metrics_summary.csv", index=False)
    print(summary[[c for c in ["outcome", "auroc", "auprc", "sensitivity", "specificity", "f1"] if c in summary]].to_string(index=False))


if __name__ == "__main__":
    main()
