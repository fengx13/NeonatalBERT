#!/usr/bin/env python
"""
Fine-tune and evaluate NeonatalBERT (or any BERT-style encoder) for neonatal
outcome prediction from first-day clinical notes.

This script implements the fine-tuning and evaluation protocol of
Xie et al., "Development and validation of a pre-trained language model for
neonatal morbidities", Lancet Digit Health 2025 (doi:10.1016/j.landig.2025.100926):

  * input   : all notes written within the first 24 h after birth, concatenated
              per patient in chronological order
  * model   : pretrained encoder + a logistic (single-logit) classification
              layer for each outcome
  * tuning  : hyperparameters selected on a validation subset of the training
              data (here: a small learning-rate grid + early stopping on AUPRC)
  * splits  : temporal split for the primary cohort (2014-2020 train, 2021 test)
              or a random 80/20 patient-level split (external MIMIC-III cohort)
  * metrics : AUROC, AUPRC (absolute and relative to prevalence), sensitivity,
              specificity, F1, with 95% bootstrap confidence intervals

Two modes are available:
  --mode finetune      end-to-end fine-tuning of encoder + classification head
  --mode embedding_lr  frozen encoder; [CLS] embeddings + logistic regression
                       tuned with GridSearchCV (fast, CPU friendly)

Examples
--------
# Re-fit and evaluate at a new site with a random 80/20 split
python model_fine_tune/finetune.py \
    --notes data/notes.csv --labels data/labels.csv \
    --model_name_or_path /path/to/NeonatalBERT \
    --split random --test_size 0.2 \
    --output_dir results/site_x

# Temporal split defined by a column in the labels file
python model_fine_tune/finetune.py ... --split column --split_col split --test_value test

# Apply previously fine-tuned models to a new cohort without re-fitting
python model_fine_tune/finetune.py --notes ... --labels ... \
    --eval_only --checkpoint_dir results/site_x --output_dir results/site_y_transfer
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_utils import build_cohort, split_cohort  # noqa: E402
from helpers import (  # noqa: E402
    PlotROCCurve,
    binary_metrics,
    bootstrap_metrics,
    parse_list,
    plot_calibration,
    save_json,
    summarise_results,
    youden_closest_threshold,
)

logger = logging.getLogger("neonatalbert.finetune")
META_FILE = "neonatalbert_meta.json"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_documents(tokenizer, texts: List[str], max_length: int, max_chunks: int, stride: int):
    """
    Tokenise patient documents into windows of `max_length` tokens.
    Long documents are split into up to `max_chunks` overlapping windows;
    max_chunks=1 is plain truncation. Returns tensors plus the document index of
    every window so that window predictions can be pooled per patient.
    """
    enc = tokenizer(
        list(texts),
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=max_chunks > 1,
        stride=stride if max_chunks > 1 else 0,
    )
    if max_chunks > 1:
        doc_idx = np.asarray(enc["overflow_to_sample_mapping"])
    else:
        doc_idx = np.arange(len(texts))

    # Keep only the first `max_chunks` windows of each document.
    keep, seen = [], {}
    for i, d in enumerate(doc_idx):
        seen[d] = seen.get(d, 0) + 1
        if seen[d] <= max_chunks:
            keep.append(i)
    keep = np.asarray(keep)
    input_ids = torch.tensor(np.asarray(enc["input_ids"])[keep], dtype=torch.long)
    attention_mask = torch.tensor(np.asarray(enc["attention_mask"])[keep], dtype=torch.long)
    return input_ids, attention_mask, doc_idx[keep]


def pool_by_document(window_values: np.ndarray, doc_idx: np.ndarray, n_docs: int, how: str) -> np.ndarray:
    df = pd.DataFrame(window_values)
    df["_doc"] = doc_idx
    g = df.groupby("_doc")
    pooled = g.mean() if how == "mean" else g.max()
    return pooled.reindex(range(n_docs)).to_numpy()


def get_device(no_cuda: bool = False):
    return torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")


# ---------------------------------------------------------------------------
# End-to-end fine-tuning
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, input_ids, attention_mask, doc_idx, n_docs, batch_size, device, pool, fp16=False):
    model.eval()
    loader = DataLoader(TensorDataset(input_ids, attention_mask), batch_size=batch_size)
    out = []
    for ids, mask in loader:
        with torch.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda"):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    window_probs = np.concatenate(out, axis=0)
    return pool_by_document(window_probs, doc_idx, n_docs, pool)


def mean_auprc(y: np.ndarray, p: np.ndarray) -> float:
    scores = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) == 2:
            scores.append(binary_metrics(y[:, j], p[:, j])["auprc"])
    return float(np.mean(scores)) if scores else float("nan")


def finetune_target(args, tokenizer, target: List[str], train_df, val_df, device):
    """Train one model for the outcome(s) in `target`; return best model + info."""
    tr_ids, tr_mask, tr_doc = encode_documents(tokenizer, train_df["text"], args.max_length, args.max_chunks, args.stride)
    y_train_doc = train_df[target].to_numpy(dtype=np.float32)
    tr_labels = torch.tensor(y_train_doc[tr_doc], dtype=torch.float32)
    train_ds = TensorDataset(tr_ids, tr_mask, tr_labels)

    has_val = len(val_df) > 0
    if has_val:
        va_ids, va_mask, va_doc = encode_documents(tokenizer, val_df["text"], args.max_length, args.max_chunks, args.stride)
        y_val = val_df[target].to_numpy()

    pos_weight = None
    if args.pos_weight == "auto":
        pos = y_train_doc.sum(0)
        pos_weight = torch.tensor((len(y_train_doc) - pos) / np.clip(pos, 1, None), dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = {"score": -np.inf, "state": None, "lr": None, "epoch": None}
    history = []
    for lr in args.lr_grid:
        set_seed(args.seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, num_labels=len(target)
        ).to(device)
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=lr)
        total_steps = len(loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

        bad_epochs = 0
        best_this_lr = -np.inf
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for ids, mask, yb in tqdm(loader, desc=f"{'+'.join(target)} lr={lr:g} epoch {epoch}", leave=False, disable=not sys.stderr.isatty()):
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=args.fp16 and device.type == "cuda"):
                    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = loss_fn(logits.float(), yb.to(device))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += loss.item()
            train_loss = running / max(len(loader), 1)

            if has_val:
                p_val = predict_proba(model, va_ids, va_mask, va_doc, len(val_df), args.eval_batch_size, device, args.chunk_pool, args.fp16)
                score = mean_auprc(y_val, p_val)
                if np.isnan(score):
                    score = -train_loss  # validation set has no positives; fall back to loss
            else:
                score = -train_loss
            history.append({"lr": lr, "epoch": epoch, "train_loss": train_loss, "val_score": score})
            logger.info("%s | lr=%g epoch=%d train_loss=%.4f val_mean_AUPRC=%.4f", "+".join(target), lr, epoch, train_loss, score)

            if score > best["score"]:
                best.update(score=score, lr=lr, epoch=epoch, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if score > best_this_lr:
                best_this_lr, bad_epochs = score, 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=len(target))
    model.load_state_dict(best["state"])
    model.to(device)
    info = {"outcomes": target, "best_lr": best["lr"], "best_epoch": best["epoch"], "best_val_score": best["score"], "history": history}
    return model, info


def run_finetune(args, cohort_splits, outcomes, device):
    train_df, val_df, test_df = cohort_splits
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    targets = [[o] for o in outcomes] if args.head == "per_outcome" else [outcomes]

    preds = {}
    val_preds = {}
    for target in targets:
        logger.info("=== Fine-tuning for %s ===", ", ".join(target))
        model, info = finetune_target(args, tokenizer, target, train_df, val_df, device)

        te_ids, te_mask, te_doc = encode_documents(tokenizer, test_df["text"], args.max_length, args.max_chunks, args.stride)
        p_test = predict_proba(model, te_ids, te_mask, te_doc, len(test_df), args.eval_batch_size, device, args.chunk_pool, args.fp16)
        for j, o in enumerate(target):
            preds[o] = p_test[:, j]
        if len(val_df):
            va_ids, va_mask, va_doc = encode_documents(tokenizer, val_df["text"], args.max_length, args.max_chunks, args.stride)
            p_val = predict_proba(model, va_ids, va_mask, va_doc, len(val_df), args.eval_batch_size, device, args.chunk_pool, args.fp16)
            for j, o in enumerate(target):
                val_preds[o] = p_val[:, j]

        name = "__".join(target) if len(target) <= 3 else "multilabel"
        save_json(info, Path(args.output_dir) / "training_logs" / f"{name}.json")
        if args.save_models:
            mdir = Path(args.output_dir) / "models" / name
            model.save_pretrained(mdir)
            tokenizer.save_pretrained(mdir)
            save_json(
                {"mode": "finetune", "outcomes": target, "max_length": args.max_length, "max_chunks": args.max_chunks,
                 "stride": args.stride, "chunk_pool": args.chunk_pool, "best_lr": info["best_lr"], "best_epoch": info["best_epoch"]},
                mdir / META_FILE,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return preds, val_preds


# ---------------------------------------------------------------------------
# Frozen encoder + logistic regression
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_documents(args, model_path, texts, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    encoder = AutoModel.from_pretrained(model_path).to(device).eval()
    ids, mask, doc = encode_documents(tokenizer, texts, args.max_length, args.max_chunks, args.stride)
    loader = DataLoader(TensorDataset(ids, mask), batch_size=args.eval_batch_size)
    out = []
    for b_ids, b_mask in tqdm(loader, desc="Embedding", leave=False, disable=not sys.stderr.isatty()):
        hidden = encoder(input_ids=b_ids.to(device), attention_mask=b_mask.to(device)).last_hidden_state
        out.append(hidden[:, 0, :].float().cpu().numpy())  # [CLS]
    return pool_by_document(np.concatenate(out), doc, len(texts), "mean")


def run_embedding_lr(args, cohort_splits, outcomes, device):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_df, val_df, test_df = cohort_splits
    dev_df = pd.concat([train_df, val_df], ignore_index=True)  # CV replaces the validation split
    X_dev = embed_documents(args, args.model_name_or_path, dev_df["text"], device)
    X_test = embed_documents(args, args.model_name_or_path, test_df["text"], device)

    preds = {}
    for o in outcomes:
        y = dev_df[o].to_numpy()
        n_splits = int(min(5, y.sum(), len(y) - y.sum()))
        if n_splits < 2:
            logger.warning("Skipping %s: too few positives/negatives for cross-validation", o)
            continue
        grid = GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10], "logisticregression__class_weight": [None, "balanced"]},
            scoring="average_precision",
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed),
            n_jobs=-1,
        )
        grid.fit(X_dev, y)
        logger.info("%s | best params %s | CV AUPRC %.4f", o, grid.best_params_, grid.best_score_)
        preds[o] = grid.predict_proba(X_test)[:, 1]
        if args.save_models:
            mdir = Path(args.output_dir) / "models" / o
            mdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(grid.best_estimator_, mdir / "logistic_regression.joblib")
            save_json(
                {"mode": "embedding_lr", "outcomes": [o], "encoder": str(args.model_name_or_path), "max_length": args.max_length,
                 "max_chunks": args.max_chunks, "stride": args.stride, "best_params": grid.best_params_},
                mdir / META_FILE,
            )
    return preds, {}


# ---------------------------------------------------------------------------
# Evaluation of saved models on a new cohort (no re-fitting)
# ---------------------------------------------------------------------------
def run_eval_only(args, test_df, device):
    model_root = Path(args.checkpoint_dir) / "models"
    preds = {}
    for meta_path in sorted(model_root.glob(f"*/{META_FILE}")):
        meta = json.loads(meta_path.read_text())
        mdir = meta_path.parent
        for k in ["max_length", "max_chunks", "stride"]:
            setattr(args, k, meta[k])
        missing = [o for o in meta["outcomes"] if o not in test_df]
        if missing:
            logger.warning("Skipping %s: outcome columns %s not in labels file", mdir.name, missing)
            continue
        if meta["mode"] == "finetune":
            tokenizer = AutoTokenizer.from_pretrained(mdir, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(mdir).to(device)
            ids, mask, doc = encode_documents(tokenizer, test_df["text"], args.max_length, args.max_chunks, args.stride)
            p = predict_proba(model, ids, mask, doc, len(test_df), args.eval_batch_size, device, meta["chunk_pool"], args.fp16)
        else:
            import joblib

            X = embed_documents(args, meta["encoder"], test_df["text"], device)
            p = joblib.load(mdir / "logistic_regression.joblib").predict_proba(X)[:, [1]]
        for j, o in enumerate(meta["outcomes"]):
            preds[o] = p[:, j]
    if not preds:
        raise SystemExit(f"No usable models found under {model_root}")
    return preds


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def evaluate_and_save(args, test_df, preds: Dict[str, np.ndarray], val_df=None, val_preds=None):
    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for o, p in preds.items():
        rows.append(pd.DataFrame({"patient_id": test_df["patient_id"].values, "outcome": o, "y_true": test_df[o].values, "y_prob": p}))
    pd.concat(rows).to_csv(out / "test_predictions.csv", index=False)

    results = {}
    for o, p in preds.items():
        y = test_df[o].to_numpy()
        if len(np.unique(y)) < 2:
            logger.warning("Skipping metrics for %s: only one class in the test set", o)
            continue
        thr = None
        if args.threshold_from == "val" and val_preds and o in val_preds and len(np.unique(val_df[o])) == 2:
            thr = youden_closest_threshold(val_df[o].to_numpy(), val_preds[o])
        results[o] = bootstrap_metrics(y, p, n_bootstraps=args.n_bootstraps, ci=95, random_seed=args.seed, threshold=thr)
        PlotROCCurve(p, y, ci=95, random_seed=args.seed, n_bootstraps=50, title=f"ROC: {o}", save_path=out / "figures" / f"roc_{o}.png")
        plot_calibration(y, p, title=f"Calibration: {o}", save_path=out / "figures" / f"calibration_{o}.png")

    save_json(results, out / "test_metrics.json")
    summary = summarise_results(results)
    summary.to_csv(out / "test_metrics_summary.csv", index=False)
    cols = [c for c in ["outcome", "n_pos", "auroc", "auroc_lower", "auroc_upper", "auprc", "auprc_lower", "auprc_upper", "auprc_relative", "sensitivity", "specificity", "f1"] if c in summary]
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary[cols].to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--notes", required=True, help="Note-level table (patient_id, note_text, [hours_since_birth], [note_type])")
    d.add_argument("--labels", required=True, help="Patient-level table (patient_id, outcome columns, [split column])")
    d.add_argument("--outcomes", default=None, help="Comma-separated outcome columns (default: all non-id columns)")
    d.add_argument("--max_hours", type=float, default=24, help="Use notes written within this many hours after birth (paper: 24)")
    d.add_argument("--note_types", default=None, help="Comma-separated note types to keep (default: all)")
    d.add_argument("--split", choices=["random", "column", "none"], default="random")
    d.add_argument("--split_col", default=None, help="Column in the labels file defining the test set (for --split column)")
    d.add_argument("--test_value", default="test", help="Value of --split_col that marks test patients")
    d.add_argument("--test_size", type=float, default=0.2)
    d.add_argument("--val_size", type=float, default=0.1, help="Fraction of the development set used for tuning")

    m = p.add_argument_group("model")
    m.add_argument("--model_name_or_path", default=None, help="Local NeonatalBERT directory or a Hugging Face model id")
    m.add_argument("--mode", choices=["finetune", "embedding_lr"], default="finetune")
    m.add_argument("--head", choices=["per_outcome", "multilabel"], default="per_outcome",
                   help="per_outcome: one model per outcome; multilabel: one shared encoder with one logit per outcome")
    m.add_argument("--max_length", type=int, default=512)
    m.add_argument("--max_chunks", type=int, default=4, help="Max 512-token windows per patient (1 = truncation)")
    m.add_argument("--stride", type=int, default=64, help="Token overlap between consecutive windows")
    m.add_argument("--chunk_pool", choices=["mean", "max"], default="mean", help="How window probabilities are pooled per patient")

    t = p.add_argument_group("training")
    t.add_argument("--lr_grid", default="2e-5,3e-5,5e-5", help="Comma-separated learning rates to try")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--patience", type=int, default=2)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--eval_batch_size", type=int, default=64)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--warmup_ratio", type=float, default=0.1)
    t.add_argument("--max_grad_norm", type=float, default=1.0)
    t.add_argument("--pos_weight", choices=["none", "auto"], default="none", help="'auto' up-weights positives by neg/pos ratio")
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--no_cuda", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save_models", action="store_true")

    e = p.add_argument_group("evaluation")
    e.add_argument("--n_bootstraps", type=int, default=1000)
    e.add_argument("--threshold_from", choices=["test", "val"], default="test",
                   help="Where the sensitivity/specificity threshold is chosen (ROC point closest to (0,1))")
    e.add_argument("--eval_only", action="store_true", help="Evaluate saved models from --checkpoint_dir on all patients")
    e.add_argument("--checkpoint_dir", default=None)
    e.add_argument("--output_dir", required=True)

    args = p.parse_args(argv)
    args.lr_grid = [float(x) for x in parse_list(args.lr_grid)]
    args.outcomes = parse_list(args.outcomes)
    args.note_types = parse_list(args.note_types)
    if args.eval_only and not args.checkpoint_dir:
        p.error("--eval_only requires --checkpoint_dir")
    if not args.eval_only and not args.model_name_or_path:
        p.error("--model_name_or_path is required unless --eval_only is set")
    return args


def main(argv=None):
    args = parse_args(argv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(Path(args.output_dir) / "run.log")],
    )
    set_seed(args.seed)
    device = get_device(args.no_cuda)
    logger.info("Device: %s", device)
    save_json({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, Path(args.output_dir) / "run_config.json")

    split_col = args.split_col if args.split == "column" else None
    cohort, outcomes = build_cohort(args.notes, args.labels, args.outcomes, args.max_hours, args.note_types, split_col)
    logger.info("Cohort: %d patients, outcomes: %s", len(cohort), outcomes)
    prev = cohort[outcomes].mean().sort_values()
    logger.info("Outcome prevalence:\n%s", prev.to_string())

    t0 = time.time()
    if args.eval_only:
        _, _, test_df = split_cohort(cohort, split="none")
        preds, val_df, val_preds = run_eval_only(args, test_df, device), None, None
    else:
        splits = split_cohort(cohort, args.split, split_col, args.test_value, args.test_size, args.val_size, args.seed)
        _, val_df, test_df = splits
        if args.mode == "finetune":
            preds, val_preds = run_finetune(args, splits, outcomes, device)
        else:
            preds, val_preds = run_embedding_lr(args, splits, outcomes, device)
    logger.info("Finished modelling in %.1f min", (time.time() - t0) / 60)
    evaluate_and_save(args, test_df, preds, val_df, val_preds)
    logger.info("Results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
