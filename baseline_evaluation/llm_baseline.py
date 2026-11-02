#!/usr/bin/env python
"""
Zero-shot LLM baseline (Meta Llama-3.1-8B-Instruct) for neonatal outcome
prediction, following the appendix of Xie et al., Lancet Digit Health 2025
("Implementation of Llama-3.1-8B"):

  * one question per outcome, asked after the patient's notes, in the form
        "<notes>\n\nWill this neonatal patient develop sepsis? Answer 1 if yes,
         0 if no. Answer with only the number."
  * the model is used zero-shot, and its output is constrained to "1" or "0"
    (the paper used the Outlines package for constrained generation)
  * sensitivity and specificity are computed from the model's answers
  * the appendix example uses neonatal H&P notes (--note_types "History and physical")

Constrained choice between the single tokens "1" and "0" is implemented directly:
the next-token probabilities of the two tokens are renormalised,

    p(yes) = P("1") / (P("1") + P("0")),

and the answer is sampled from p(yes) (--decoding sample, as with Outlines'
multinomial sampler in the original analysis) or taken as p(yes) >= 0.5
(--decoding greedy, deterministic). p(yes) is also used to report AUROC/AUPRC,
which the paper did not report for this baseline.

The same cohort construction and test split as finetune.py are used, so the
results are directly comparable.

Usage
-----
python baseline_evaluation/llm_baseline.py \
    --notes data/notes.csv --labels data/labels.csv --outcomes rds,bpd,nec,sepsis,death \
    --note_types "History and physical" \
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
    "rds": "Respiratory Distress Syndrome (RDS)",
    "bpd": "Bronchopulmonary Dysplasia (BPD)",
    "pulmonary_haemorrhage": "pulmonary hemorrhage",
    "pulmonary_hemorrhage": "pulmonary hemorrhage",
    "pulmonary_hypertension": "pulmonary hypertension",
    "atelectasis": "atelectasis",
    "aspiration": "meconium aspiration syndrome",
    "mas": "meconium aspiration syndrome",
    "ivh": "Intraventricular Hemorrhage (IVH)",
    "pvl": "Periventricular Leukomalacia (PVL)",
    "seizures": "neonatal seizures",
    "cns_other": "other central nervous system disorders",
    "pda": "Patent Ductus Arteriosus (PDA)",
    "cv_instability": "cardiovascular instability",
    "sepsis": "sepsis",
    "candidiasis": "Candidiasis",
    "anaemia": "anemia",
    "anemia": "anemia",
    "jaundice": "jaundice",
    "nec": "Necrotizing Enterocolitis (NEC)",
    "rop": "Retinopathy of Prematurity (ROP)",
}
ANSWER_INSTRUCTION = "Answer 1 if yes, 0 if no. Answer with only the number."


def question_for(outcome: str) -> str:
    if outcome in ("death", "mortality"):
        return "Is the neonatal patient at risk of mortality?"
    return f"Will this neonatal patient develop {OUTCOME_NAMES.get(outcome, outcome.replace('_', ' '))}?"


def has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_prompt(tokenizer, notes: str, outcome: str) -> str:
    user = f"{notes}\n\n{question_for(outcome)} {ANSWER_INSTRUCTION}"
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return f"{user}\n"


def truncate_notes(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def answer_token_ids(tokenizer):
    ids = {}
    for ans in ["1", "0"]:
        ids[ans] = tokenizer(ans, add_special_tokens=False)["input_ids"][-1]
    if ids["1"] == ids["0"]:
        raise ValueError("Tokenizer maps '1' and '0' to the same token id")
    return ids["1"], ids["0"]


@torch.no_grad()
def score_prompts(model, tokenizer, prompts, batch_size, yes_id, no_id):
    probs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring", disable=not sys.stderr.isatty()):
        batch = prompts[i : i + batch_size]
        # Chat templates already contain the BOS token; plain prompts need it added.
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=not has_chat_template(tokenizer)).to(model.device)
        try:  # only compute logits for the last position (saves GPU memory on long prompts)
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        except TypeError:  # model class without logits_to_keep
            logits = model(**enc).logits[:, -1, :]  # left padding -> last position is the next token
        two = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1).float()
        probs.append(torch.softmax(two, dim=-1)[:, 0].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outcomes", required=True, help="Comma-separated binary outcome columns")
    ap.add_argument("--max_hours", type=float, default=24)
    ap.add_argument("--note_types", default=None, help='Note types to include, e.g. "History and physical" (appendix example)')
    ap.add_argument("--split", choices=["random", "column", "none"], default="random")
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--test_value", default="test")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--decoding", choices=["sample", "greedy"], default="sample",
                    help="sample: draw the 0/1 answer from p(yes) (original analysis); greedy: answer = p(yes) >= 0.5")
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
    cohort, outcomes = build_cohort(args.notes, args.labels, parse_list(args.outcomes), args.max_hours,
                                    parse_list(args.note_types), split_col)
    _, _, test_df = split_cohort(cohort, args.split, split_col, args.test_value, args.test_size, args.val_size, args.seed)
    logger.info("Scoring %d test patients x %d outcomes with %s (%s decoding)", len(test_df), len(outcomes),
                args.model_name, args.decoding)

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
    rng = np.random.RandomState(args.seed)

    notes = np.asarray([truncate_notes(tokenizer, t, args.max_note_tokens) for t in test_df["text"]], dtype=object)
    preds, results = [], {}
    for o in outcomes:
        m = test_df[o].notna().to_numpy()  # patients excluded for this outcome are not scored
        y = test_df.loc[m, o].to_numpy().astype(int)
        p_yes = score_prompts(model, tokenizer, [build_prompt(tokenizer, n, o) for n in notes[m]], args.batch_size, yes_id, no_id)
        answer = (rng.rand(len(p_yes)) < p_yes).astype(int) if args.decoding == "sample" else (p_yes >= 0.5).astype(int)
        preds.append(pd.DataFrame({"patient_id": test_df.loc[m, "patient_id"], "outcome": o, "y_true": y,
                                   "p_yes": p_yes, "answer": answer}))
        if len(np.unique(y)) < 2:
            logger.warning("Skipping metrics for %s: only one class in the test set", o)
            continue
        # Sensitivity/specificity/F1 from the model's 0/1 answers (as in the paper) ...
        res = bootstrap_metrics(y, answer.astype(float), n_bootstraps=args.n_bootstraps, random_seed=args.seed, threshold=0.5)
        # ... and threshold-free discrimination from p(yes)
        res_p = bootstrap_metrics(y, p_yes, n_bootstraps=args.n_bootstraps, random_seed=args.seed, threshold=0.5)
        for k in ["auroc", "auprc", "auprc_relative"]:
            res[k] = res_p[k]
        results[o] = res
        logger.info("%s | sensitivity %.3f | specificity %.3f | AUROC(p_yes) %.3f", o,
                    res["sensitivity"]["estimate"], res["specificity"]["estimate"], res["auroc"]["estimate"])

    pd.concat(preds).to_csv(out / "test_predictions.csv", index=False)
    save_json(results, out / "test_metrics.json")
    summary = summarise_results(results)
    summary.to_csv(out / "test_metrics_summary.csv", index=False)
    print(summary[[c for c in ["outcome", "sensitivity", "specificity", "f1", "auroc", "auprc"] if c in summary]].to_string(index=False))


if __name__ == "__main__":
    main()
