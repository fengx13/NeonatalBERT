#!/usr/bin/env python
"""
Fine-tune and evaluate NeonatalBERT for neonatal outcome prediction from
clinical notes written on the delivery day (first 24 h after birth).

Default protocol, following the paper and its appendix (eMethods)
(Xie et al., Lancet Digit Health 2025, doi:10.1016/j.landig.2025.100926):

  * input        : notes written within the prediction window (default 24 h);
                   cases already diagnosed within the window are excluded per
                   outcome (labels column <outcome>_dx_hours)
  * long notes   : truncated to the first 512 tokens (default); alternatively
                   split into 512-token sections and mean- or max-pooled
                   (appendix eTable 9: --long_note mean|max)
  * note -> patient: NeonatalBERT is applied to each note separately; the final
                   hidden state ([CLS]) is the note embedding, and a patient's
                   note embeddings are averaged into one patient representation
  * classifier   : a classification layer with one output per outcome, trained
                   with a class-weighted loss on top of the frozen encoder
  * tuning       : grid search over learning rate, batch size, weight decay and
                   class weights on a validation subset of the training data,
                   with early stopping; the best setting is chosen per outcome
  * evaluation   : AUROC, AUPRC (absolute and relative to prevalence), F1,
                   sensitivity, specificity with 95% bootstrap CIs

Modes
  --mode frozen        (default) frozen encoder + trained classification layer
  --mode embedding_lr  frozen encoder + scikit-learn logistic regression per
                       outcome (GridSearchCV); a lighter alternative
  --mode finetune      end-to-end: encoder updated together with the
                       classification layer (alternative, needs a GPU)

Examples
--------
# Random 80/20 patient-level split (as for the external MIMIC-III cohort)
python model_fine_tune/finetune.py \
    --notes data/notes.csv --labels data/labels.csv --outcomes rds,bpd,nec,death \
    --model_name_or_path /path/to/NeonatalBERT \
    --split random --test_size 0.2 --output_dir results/site_x

# Temporal split defined by a column in the labels file
python model_fine_tune/finetune.py ... --split column --split_col split --test_value test

# Apply saved models to a new cohort without re-fitting
python model_fine_tune/finetune.py --notes ... --labels ... --outcomes ... \
    --eval_only --checkpoint_dir results/site_x --output_dir results/site_y_transfer
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

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
SHOW_PROGRESS = sys.stderr.isatty()

MODE_DEFAULTS = {  # used when the corresponding option is not given
    "frozen": {"lr_grid": "1e-3,1e-4", "epochs": 50, "patience": 5},
    "embedding_lr": {"lr_grid": "1e-3", "epochs": 1, "patience": 1},
    "finetune": {"lr_grid": "2e-5,5e-5", "epochs": 4, "patience": 2},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(no_cuda: bool = False):
    return torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")


def labels_matrix(df: pd.DataFrame, outcomes: List[str]) -> np.ndarray:
    """Outcome matrix with NaN where a patient is excluded for that outcome."""
    return df[outcomes].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Notes -> 512-token segments
# ---------------------------------------------------------------------------
@dataclass
class Segments:
    input_ids: np.ndarray      # (n_segments, max_length) int32, right-padded
    lengths: np.ndarray        # (n_segments,) number of real tokens
    seg_note: np.ndarray       # (n_segments,) note index of each segment
    note_patient: np.ndarray   # (n_notes,) patient index of each note
    n_patients: int
    pad_id: int

    @property
    def n_notes(self):
        return len(self.note_patient)


def segment_notes(tokenizer, notes_per_patient, max_length=512, long_note="truncate", max_sections=8,
                  max_notes=None, batch_size=256) -> Segments:
    """
    Tokenise every note separately.
      long_note="truncate": keep the first max_length tokens (incl. [CLS]/[SEP]) of each note
      long_note="mean"/"max": split the note into consecutive sections of max_length tokens
                              (up to max_sections) whose embeddings are later mean/max pooled
    """
    body = max_length - 2
    cap = body * (1 if long_note == "truncate" else max_sections)
    char_cap = cap * 10  # generous bound on characters per token; avoids tokenising huge texts
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    flat, note_patient = [], []
    for p, notes in enumerate(notes_per_patient):
        notes = list(notes)[:max_notes] if max_notes else list(notes)
        for t in notes:
            flat.append(str(t)[:char_cap])
            note_patient.append(p)

    id_blocks, len_blocks, note_blocks = [], [], []
    for start in range(0, len(flat), batch_size):
        batch = tokenizer(flat[start:start + batch_size], add_special_tokens=False, truncation=False)["input_ids"]
        rows, lens, owners = [], [], []
        for k, ids in enumerate(batch):
            ids = ids[:cap]
            pieces = [ids[i:i + body] for i in range(0, len(ids), body)] or [[]]
            for piece in pieces:
                rows.append([cls_id] + piece + [sep_id])
                lens.append(len(piece) + 2)
                owners.append(start + k)
        block = np.full((len(rows), max_length), pad_id, dtype=np.int32)
        for i, r in enumerate(rows):
            block[i, : len(r)] = r
        id_blocks.append(block)
        len_blocks.append(np.asarray(lens, dtype=np.int32))
        note_blocks.append(np.asarray(owners, dtype=np.int64))

    if not id_blocks:
        raise ValueError("No notes to encode")
    return Segments(np.concatenate(id_blocks), np.concatenate(len_blocks), np.concatenate(note_blocks),
                    np.asarray(note_patient, dtype=np.int64), len(notes_per_patient), pad_id)


def pool_segments(seg_emb, seg_note, note_patient, n_notes, n_patients, within="mean"):
    """Segments -> notes (mean or max over a note's sections) -> patients (mean over notes)."""
    H = seg_emb.size(1)
    note = seg_emb.new_zeros(n_notes, H).scatter_reduce(
        0, seg_note.view(-1, 1).expand(-1, H), seg_emb, reduce="amax" if within == "max" else "mean", include_self=False)
    return seg_emb.new_zeros(n_patients, H).scatter_reduce(
        0, note_patient.view(-1, 1).expand(-1, H), note, reduce="mean", include_self=False)


def encode_batch(segs: Segments, seg_idx: np.ndarray, device):
    L = int(segs.lengths[seg_idx].max())
    ids = torch.from_numpy(segs.input_ids[seg_idx, :L].astype(np.int64)).to(device)
    mask = (torch.arange(L)[None, :] < torch.from_numpy(segs.lengths[seg_idx])[:, None]).long().to(device)
    return ids, mask


# ---------------------------------------------------------------------------
# Frozen encoder: patient embeddings
# ---------------------------------------------------------------------------
@torch.no_grad()
def patient_embeddings(args, model_path, notes_per_patient, device, desc=""):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    encoder = AutoModel.from_pretrained(model_path).to(device).eval()
    segs = segment_notes(tokenizer, notes_per_patient, args.max_length, args.long_note, args.max_sections, args.max_notes)
    logger.info("%s: %d patients, %d notes, %d segments", desc, segs.n_patients, segs.n_notes, len(segs.lengths))
    order = np.argsort(segs.lengths)  # length-sorted batches = less padding
    out = np.zeros((len(order), encoder.config.hidden_size), dtype=np.float32)
    for b in tqdm(range(0, len(order), args.eval_batch_size), desc=f"Embedding {desc}", disable=not SHOW_PROGRESS):
        idx = order[b:b + args.eval_batch_size]
        ids, mask = encode_batch(segs, idx, device)
        with torch.autocast(device_type=device.type, enabled=args.fp16 and device.type == "cuda"):
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        out[idx] = hidden[:, 0, :].float().cpu().numpy()  # final hidden state of [CLS]
    pooled = pool_segments(torch.from_numpy(out), torch.from_numpy(segs.seg_note), torch.from_numpy(segs.note_patient),
                           segs.n_notes, segs.n_patients, args.long_note if args.long_note == "max" else "mean")
    return pooled.numpy()


def masked_bce(logits, y, pos_weight=None):
    mask = ~torch.isnan(y)
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def class_weights(y: np.ndarray, scheme: str) -> np.ndarray:
    """Positive-class weight per outcome: none = 1, balanced = neg/pos, sqrt = sqrt(neg/pos)."""
    pos = np.nansum(y, axis=0)
    neg = np.sum(~np.isnan(y), axis=0) - pos
    ratio = neg / np.clip(pos, 1, None)
    return {"none": np.ones_like(ratio), "balanced": ratio, "sqrt": np.sqrt(ratio)}[scheme].astype(np.float32)


def selection_scores(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-outcome validation score: AUPRC, or negative log-loss when only one class is present."""
    scores = np.full(y.shape[1], -np.inf)
    for k in range(y.shape[1]):
        m = ~np.isnan(y[:, k])
        if m.sum() == 0:
            continue
        yk, pk = y[m, k], np.clip(p[m, k], 1e-7, 1 - 1e-7)
        if len(np.unique(yk)) == 2:
            scores[k] = binary_metrics(yk, pk)["auprc"]
        else:
            scores[k] = -1.0 - float(np.mean(-(yk * np.log(pk) + (1 - yk) * np.log(1 - pk))))  # always below any AUPRC
    return scores


def train_frozen_head(args, X_tr, Y_tr, X_va, Y_va, outcomes, device):
    """Grid search over lr x batch size x weight decay x class weights; best setting chosen per outcome."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Xtr = torch.tensor((X_tr - mu) / sd, device=device)
    Ytr = torch.tensor(Y_tr, device=device)
    has_val = len(X_va) > 0
    if not has_val:
        logger.warning("No validation set (--val_size 0): model selection uses the training data (optimistic).")
    Xva = torch.tensor((X_va - mu) / sd, device=device) if has_val else Xtr
    Yva = Y_va if has_val else Y_tr
    single = [o for k, o in enumerate(outcomes) if len(np.unique(Yva[~np.isnan(Yva[:, k]), k])) < 2]
    if single:
        logger.warning("Validation set has a single class for %s: selection uses validation log-loss for these. "
                       "Consider a larger --val_size.", ", ".join(single))

    K, H = Y_tr.shape[1], X_tr.shape[1]
    best = {"score": np.full(K, -np.inf), "W": np.zeros((K, H), np.float32), "b": np.zeros(K, np.float32),
            "config": [None] * K, "epoch": [None] * K}
    grid = list(itertools.product(args.lr_grid, args.batch_size_grid, args.weight_decay_grid, args.class_weight_grid))
    logger.info("Classification layer: %d hyperparameter settings x up to %d epochs", len(grid), args.epochs)
    history = []
    for lr, bs, wd, cw in grid:
        set_seed(args.seed)
        layer = nn.Linear(H, K).to(device)
        opt = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=wd)
        pw = torch.tensor(class_weights(Y_tr, cw), device=device)
        cfg = {"lr": lr, "batch_size": bs, "weight_decay": wd, "class_weight": cw}
        cfg_best = np.full(K, -np.inf)
        bad = 0
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        for epoch in range(1, args.epochs + 1):
            layer.train()
            perm = torch.randperm(len(Xtr), generator=g).to(device)
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = masked_bce(layer(Xtr[idx]), Ytr[idx], pw)
                loss.backward()
                opt.step()
            layer.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(layer(Xva)).cpu().numpy()
            scores = selection_scores(Yva, p_va)
            history.append({**cfg, "epoch": epoch, "val_scores": dict(zip(outcomes, map(float, scores)))})
            improved = scores > cfg_best + 1e-6
            cfg_best = np.maximum(cfg_best, scores)
            for k in np.where(scores > best["score"])[0]:
                best["score"][k] = scores[k]
                best["W"][k] = layer.weight[k].detach().cpu().numpy()
                best["b"][k] = layer.bias[k].detach().cpu().numpy()
                best["config"][k], best["epoch"][k] = cfg, epoch
            bad = 0 if improved.any() else bad + 1
            if bad >= args.patience:
                break
        valid = cfg_best[cfg_best > -1]
        logger.info("lr=%g bs=%d wd=%g cw=%s | epochs=%d | mean val AUPRC %.4f", lr, bs, wd, cw, epoch,
                    float(np.mean(valid)) if len(valid) else float("nan"))
    head = {"mu": mu, "sd": sd, "W": best["W"], "b": best["b"], "outcomes": outcomes,
            "selected": {o: {"config": best["config"][k], "epoch": best["epoch"][k], "val_score": float(best["score"][k])}
                         for k, o in enumerate(outcomes)}}
    return head, history


def predict_frozen_head(head, X):
    z = ((X - head["mu"]) / head["sd"]) @ head["W"].T + head["b"]
    return 1 / (1 + np.exp(-z))


def run_frozen(args, splits, outcomes, device):
    train_df, val_df, test_df = splits
    emb = lambda df, d: patient_embeddings(args, args.model_name_or_path, df["notes"].tolist(), device, d)  # noqa: E731
    X_tr, X_te = emb(train_df, "train"), emb(test_df, "test")
    X_va = emb(val_df, "validation") if len(val_df) else np.zeros((0, X_tr.shape[1]), np.float32)
    Y_tr, Y_va = labels_matrix(train_df, outcomes), labels_matrix(val_df, outcomes)

    head, history = train_frozen_head(args, X_tr, Y_tr, X_va, Y_va, outcomes, device)
    save_json({"selected": head["selected"], "history": history}, Path(args.output_dir) / "training_logs" / "frozen_head.json")
    for o, sel in head["selected"].items():
        logger.info("%s: selected %s (epoch %s, val score %.4f)", o, sel["config"], sel["epoch"], sel["val_score"])

    P_te = predict_frozen_head(head, X_te)
    preds = {o: P_te[:, k] for k, o in enumerate(outcomes)}
    thr_data = {}
    if len(val_df):
        P_va = predict_frozen_head(head, X_va)
        thr_data = {o: (Y_va[:, k], P_va[:, k]) for k, o in enumerate(outcomes)}
    if args.save_models:
        mdir = Path(args.output_dir) / "models" / "frozen_head"
        mdir.mkdir(parents=True, exist_ok=True)
        np.savez(mdir / "head.npz", mu=head["mu"], sd=head["sd"], W=head["W"], b=head["b"])
        save_json({"mode": "frozen", "outcomes": outcomes, "encoder": str(args.model_name_or_path),
                   **segmentation_meta(args), "selected": head["selected"]}, mdir / META_FILE)
    return preds, thr_data


# ---------------------------------------------------------------------------
# Frozen encoder + scikit-learn logistic regression
# ---------------------------------------------------------------------------
def run_embedding_lr(args, splits, outcomes, device):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_df, val_df, test_df = splits
    dev_df = pd.concat([train_df, val_df], ignore_index=True)  # cross-validation replaces the validation split
    X_dev = patient_embeddings(args, args.model_name_or_path, dev_df["notes"].tolist(), device, "development")
    X_te = patient_embeddings(args, args.model_name_or_path, test_df["notes"].tolist(), device, "test")

    preds, thr_data = {}, {}
    for o in outcomes:
        m = dev_df[o].notna().to_numpy()
        y = dev_df.loc[m, o].to_numpy().astype(int)
        n_splits = int(min(5, y.sum(), len(y) - y.sum()))
        if n_splits < 2:
            logger.warning("Skipping %s: too few positives/negatives for cross-validation", o)
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        grid = GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10], "logisticregression__class_weight": [None, "balanced"]},
            scoring="average_precision", cv=cv, n_jobs=-1,
        )
        grid.fit(X_dev[m], y)
        logger.info("%s | best params %s | CV AUPRC %.4f", o, grid.best_params_, grid.best_score_)
        preds[o] = grid.predict_proba(X_te)[:, 1]
        # out-of-fold development-set predictions, used to choose the threshold without the test set
        thr_data[o] = (y, cross_val_predict(grid.best_estimator_, X_dev[m], y, cv=cv, method="predict_proba")[:, 1])
        if args.save_models:
            mdir = Path(args.output_dir) / "models" / f"lr_{o}"
            mdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(grid.best_estimator_, mdir / "logistic_regression.joblib")
            save_json({"mode": "embedding_lr", "outcomes": [o], "encoder": str(args.model_name_or_path),
                       **segmentation_meta(args), "best_params": grid.best_params_}, mdir / META_FILE)
    return preds, thr_data


# ---------------------------------------------------------------------------
# End-to-end fine-tuning (alternative)
# ---------------------------------------------------------------------------
class PatientClassifier(nn.Module):
    """Encoder applied to every note segment -> [CLS] -> pooled per patient -> one logit per outcome."""

    def __init__(self, model_path, n_outputs, within="mean", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_outputs)
        self.within = within

    def forward(self, ids, mask, seg_note, note_patient, n_notes, n_patients):
        cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(self.dropout(pool_segments(cls, seg_note, note_patient, n_notes, n_patients, self.within)))


def patient_batches(segs: Segments, patient_idx: np.ndarray, device):
    """Gather the segments of the given patients and re-index notes/patients locally."""
    notes = np.where(np.isin(segs.note_patient, patient_idx))[0]
    local_patient = {p: i for i, p in enumerate(patient_idx)}
    note_local = {n: i for i, n in enumerate(notes)}
    seg_idx = np.where(np.isin(segs.seg_note, notes))[0]
    ids, mask = encode_batch(segs, seg_idx, device)
    seg_note = torch.tensor([note_local[n] for n in segs.seg_note[seg_idx]], device=device)
    note_patient = torch.tensor([local_patient[p] for p in segs.note_patient[notes]], device=device)
    return ids, mask, seg_note, note_patient, len(notes), len(patient_idx)


@torch.no_grad()
def predict_patients(model, segs, device, batch_size, fp16):
    model.eval()
    out = []
    for i in range(0, segs.n_patients, batch_size):
        pidx = np.arange(i, min(i + batch_size, segs.n_patients))
        with torch.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda"):
            logits = model(*patient_batches(segs, pidx, device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def finetune_target(args, target, Y_tr, Y_va, segs_tr, segs_va, device):
    within = "max" if args.long_note == "max" else "mean"
    best = {"score": -np.inf, "state": None, "config": None}
    history = []
    for lr, cw in itertools.product(args.lr_grid, args.class_weight_grid):
        set_seed(args.seed)
        model = PatientClassifier(args.model_name_or_path, len(target), within).to(device)
        pw = torch.tensor(class_weights(Y_tr, cw), device=device)
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay_grid[0]},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=lr)
        steps = int(np.ceil(segs_tr.n_patients / args.batch_size)) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
        rng = np.random.RandomState(args.seed)
        bad, cfg_best = 0, -np.inf
        for epoch in range(1, args.epochs + 1):
            model.train()
            running, nb = 0.0, 0
            order = rng.permutation(segs_tr.n_patients)
            for i in tqdm(range(0, len(order), args.batch_size), desc=f"lr={lr:g} cw={cw} epoch {epoch}", disable=not SHOW_PROGRESS):
                pidx = order[i:i + args.batch_size]
                opt.zero_grad()
                with torch.autocast(device_type=device.type, enabled=args.fp16 and device.type == "cuda"):
                    logits = model(*patient_batches(segs_tr, pidx, device))
                loss = masked_bce(logits.float(), torch.tensor(Y_tr[pidx], device=device), pw)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running, nb = running + loss.item(), nb + 1
            if segs_va is not None:
                scores = selection_scores(Y_va, predict_patients(model, segs_va, device, args.batch_size, args.fp16))
                score = float(np.mean(scores))
            else:
                score = -running / max(nb, 1)
            history.append({"lr": lr, "class_weight": cw, "epoch": epoch, "train_loss": running / max(nb, 1), "val_score": score})
            logger.info("%s | lr=%g cw=%s epoch=%d train_loss=%.4f val_score=%.4f", "+".join(target), lr, cw, epoch, running / max(nb, 1), score)
            if score > best["score"]:
                best.update(score=score, config={"lr": lr, "class_weight": cw, "epoch": epoch},
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if score > cfg_best:
                cfg_best, bad = score, 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model = PatientClassifier(args.model_name_or_path, len(target), within)
    model.load_state_dict(best["state"])
    return model.to(device), {"outcomes": target, "selected": best["config"], "val_score": best["score"], "history": history}


def run_finetune(args, splits, outcomes, device):
    train_df, val_df, test_df = splits
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    seg = lambda df: segment_notes(tokenizer, df["notes"].tolist(), args.max_length, args.long_note, args.max_sections, args.max_notes)  # noqa: E731
    segs_tr, segs_te = seg(train_df), seg(test_df)
    segs_va = seg(val_df) if len(val_df) else None
    if segs_va is None:
        logger.warning("No validation set (--val_size 0): model selection uses training loss.")
    targets = [[o] for o in outcomes] if args.head == "per_outcome" else [outcomes]

    preds, thr_data = {}, {}
    for target in targets:
        logger.info("=== End-to-end fine-tuning for %s ===", ", ".join(target))
        Y_tr, Y_va = labels_matrix(train_df, target), labels_matrix(val_df, target)
        model, info = finetune_target(args, target, Y_tr, Y_va, segs_tr, segs_va, device)
        P_te = predict_patients(model, segs_te, device, args.batch_size, args.fp16)
        for k, o in enumerate(target):
            preds[o] = P_te[:, k]
        if segs_va is not None:
            P_va = predict_patients(model, segs_va, device, args.batch_size, args.fp16)
            for k, o in enumerate(target):
                thr_data[o] = (Y_va[:, k], P_va[:, k])
        name = target[0] if len(target) == 1 else "multilabel"
        save_json(info, Path(args.output_dir) / "training_logs" / f"finetune_{name}.json")
        if args.save_models:
            mdir = Path(args.output_dir) / "models" / f"finetune_{name}"
            model.encoder.save_pretrained(mdir / "encoder")
            tokenizer.save_pretrained(mdir / "encoder")
            torch.save(model.head.state_dict(), mdir / "head.pt")
            save_json({"mode": "finetune", "outcomes": target, **segmentation_meta(args), "selected": info["selected"]}, mdir / META_FILE)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return preds, thr_data


# ---------------------------------------------------------------------------
# Evaluation of saved models on a new cohort (no re-fitting)
# ---------------------------------------------------------------------------
def segmentation_meta(args):
    return {k: getattr(args, k) for k in ["max_length", "long_note", "max_sections", "max_notes", "max_hours"]}


def run_eval_only(args, test_df, device):
    import joblib

    preds = {}
    metas = sorted((Path(args.checkpoint_dir) / "models").glob(f"*/{META_FILE}"))
    for meta_path in metas:
        meta, mdir = json.loads(meta_path.read_text()), meta_path.parent
        for k in ["max_length", "long_note", "max_sections", "max_notes"]:
            setattr(args, k, meta[k])
        wanted = [o for o in meta["outcomes"] if o in test_df]
        if not wanted:
            continue
        if meta["mode"] in ("frozen", "embedding_lr"):
            encoder = args.model_name_or_path or meta["encoder"]
            X = patient_embeddings(args, encoder, test_df["notes"].tolist(), device, "evaluation")
            if meta["mode"] == "frozen":
                h = np.load(mdir / "head.npz")
                P = predict_frozen_head({k: h[k] for k in h.files}, X)
            else:
                P = joblib.load(mdir / "logistic_regression.joblib").predict_proba(X)[:, [1]]
        else:
            tokenizer = AutoTokenizer.from_pretrained(mdir / "encoder", use_fast=True)
            model = PatientClassifier(mdir / "encoder", len(meta["outcomes"]), "max" if meta["long_note"] == "max" else "mean")
            model.head.load_state_dict(torch.load(mdir / "head.pt", map_location="cpu"))
            segs = segment_notes(tokenizer, test_df["notes"].tolist(), args.max_length, args.long_note, args.max_sections, args.max_notes)
            P = predict_patients(model.to(device), segs, device, args.batch_size, args.fp16)
        for k, o in enumerate(meta["outcomes"]):
            if o in wanted:
                preds[o] = P[:, k]
    if not preds:
        raise SystemExit(f"No usable models for the requested outcomes under {args.checkpoint_dir}/models")
    return preds


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def evaluate_and_save(args, test_df, preds: Dict[str, np.ndarray], thr_data: Optional[Dict] = None):
    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for o, p in preds.items():
        m = test_df[o].notna().to_numpy()
        rows.append(pd.DataFrame({"patient_id": test_df["patient_id"].values[m], "outcome": o,
                                  "y_true": test_df[o].values[m].astype(int), "y_prob": p[m]}))
    pd.concat(rows).to_csv(out / "test_predictions.csv", index=False)

    results = {}
    for o, p in preds.items():
        m = test_df[o].notna().to_numpy()
        y, p = test_df[o].to_numpy()[m].astype(int), p[m]
        if len(np.unique(y)) < 2:
            logger.warning("Skipping metrics for %s: only one class in the test set", o)
            continue
        thr = None
        if args.threshold_from == "val":
            if thr_data and o in thr_data:
                yv, pv = thr_data[o]
                mv = ~np.isnan(yv)
                if len(np.unique(yv[mv])) == 2:
                    thr = youden_closest_threshold(yv[mv].astype(int), pv[mv])
            if thr is None:
                logger.warning("%s: no usable validation predictions; threshold chosen on the TEST set "
                               "(sensitivity/specificity/F1 may be optimistic)", o)
        results[o] = bootstrap_metrics(y, p, n_bootstraps=args.n_bootstraps, ci=95, random_seed=args.seed, threshold=thr)
        PlotROCCurve(p, y, ci=95, random_seed=args.seed, n_bootstraps=50, title=f"ROC: {o}", save_path=out / "figures" / f"roc_{o}.png")
        plot_calibration(y, p, title=f"Calibration: {o}", save_path=out / "figures" / f"calibration_{o}.png")

    save_json(results, out / "test_metrics.json")
    summary = summarise_results(results)
    summary.to_csv(out / "test_metrics_summary.csv", index=False)
    cols = [c for c in ["outcome", "n_pos", "auroc", "auroc_lower", "auroc_upper", "auprc", "auprc_lower", "auprc_upper",
                        "auprc_relative", "sensitivity", "specificity", "f1"] if c in summary]
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.float_format", "{:.3f}".format):
        print(summary[cols].to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--notes", required=True, help="Note-level table (patient_id, note_text, [hours_since_birth], [note_type])")
    d.add_argument("--labels", required=True, help="Patient-level table (patient_id, outcome columns, [<outcome>_dx_hours], [split column])")
    d.add_argument("--outcomes", required=True, help="Comma-separated binary outcome columns in --labels, e.g. rds,bpd,nec,death")
    d.add_argument("--max_hours", type=float, default=24, help="Prediction window: notes written within this many hours after birth (paper: 24; days 1-7 in eTable 6)")
    d.add_argument("--note_types", default=None, help="Comma-separated note types to keep (default: all, as in the paper)")
    d.add_argument("--note_type_header", action="store_true", help="Prefix each note with its [note_type]")
    d.add_argument("--split", choices=["random", "column", "none"], default="random")
    d.add_argument("--split_col", default=None, help="Column in the labels file defining the test set (for --split column)")
    d.add_argument("--test_value", default="test", help="Value of --split_col that marks test patients")
    d.add_argument("--test_size", type=float, default=0.2)
    d.add_argument("--val_size", type=float, default=0.1, help="Fraction of the development set used for hyperparameter tuning")

    m = p.add_argument_group("model")
    m.add_argument("--model_name_or_path", default=None, help="Local NeonatalBERT directory or a Hugging Face model id")
    m.add_argument("--mode", choices=["frozen", "embedding_lr", "finetune"], default="frozen")
    m.add_argument("--max_length", type=int, default=512)
    m.add_argument("--long_note", choices=["truncate", "mean", "max"], default="truncate",
                   help="Notes longer than --max_length: keep the first tokens (paper default) or pool 512-token sections (eTable 9)")
    m.add_argument("--max_sections", type=int, default=8, help="Max sections per note for --long_note mean|max")
    m.add_argument("--max_notes", type=int, default=None, help="Optional cap on notes per patient (earliest kept)")
    m.add_argument("--head", choices=["multilabel", "per_outcome"], default="multilabel",
                   help="--mode finetune only: one shared model with one output per outcome, or one model per outcome")

    t = p.add_argument_group("training / hyperparameter grid")
    t.add_argument("--lr_grid", default=None, help="Learning rates (default: frozen 1e-3,1e-4; finetune 2e-5,5e-5)")
    t.add_argument("--batch_size_grid", default="32,128", help="--mode frozen: batch sizes to try")
    t.add_argument("--weight_decay_grid", default="0,0.01", help="Weight decay values to try (finetune uses the first)")
    t.add_argument("--class_weight_grid", default="none,sqrt,balanced",
                   help="Positive-class weights to try: none, sqrt (sqrt(neg/pos)), balanced (neg/pos)")
    t.add_argument("--epochs", type=int, default=None, help="Max epochs (default: frozen 50, finetune 4)")
    t.add_argument("--patience", type=int, default=None, help="Early-stopping patience (default: frozen 5, finetune 2)")
    t.add_argument("--batch_size", type=int, default=8, help="--mode finetune: patients per batch")
    t.add_argument("--eval_batch_size", type=int, default=64, help="Note segments per forward pass when embedding")
    t.add_argument("--warmup_ratio", type=float, default=0.1)
    t.add_argument("--max_grad_norm", type=float, default=1.0)
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--no_cuda", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save_models", action="store_true")

    e = p.add_argument_group("evaluation")
    e.add_argument("--n_bootstraps", type=int, default=1000)
    e.add_argument("--threshold_from", choices=["val", "test"], default="val",
                   help="Data used to choose the sensitivity/specificity/F1 threshold (ROC point closest to (0,1)): "
                        "'val' = validation set (out-of-fold predictions in embedding_lr mode); 'test' = test set (optimistic)")
    e.add_argument("--eval_only", action="store_true", help="Evaluate saved models from --checkpoint_dir on all patients")
    e.add_argument("--checkpoint_dir", default=None)
    e.add_argument("--output_dir", required=True)

    args = p.parse_args(argv)
    for k, v in MODE_DEFAULTS[args.mode].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.lr_grid = [float(x) for x in parse_list(args.lr_grid)]
    args.batch_size_grid = [int(x) for x in parse_list(args.batch_size_grid)]
    args.weight_decay_grid = [float(x) for x in parse_list(args.weight_decay_grid)]
    args.class_weight_grid = parse_list(args.class_weight_grid)
    bad_cw = set(args.class_weight_grid) - {"none", "sqrt", "balanced"}
    if bad_cw:
        p.error(f"Unknown --class_weight_grid values: {bad_cw}")
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
    logger.info("Device: %s | mode: %s", device, args.mode)
    save_json({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, Path(args.output_dir) / "run_config.json")

    split_col = args.split_col if args.split == "column" else None
    cohort, outcomes = build_cohort(args.notes, args.labels, args.outcomes, args.max_hours, args.note_types, split_col,
                                    include_note_type_header=args.note_type_header)
    logger.info("Cohort: %d patients", len(cohort))
    logger.info("Outcome prevalence (n positive / n eligible):\n%s", "\n".join(
        f"  {o}: {int(np.nansum(cohort[o]))} / {int(cohort[o].notna().sum())}" for o in outcomes))

    t0 = time.time()
    thr_data = None
    if args.eval_only:
        test_df = split_cohort(cohort, split="none")[2]
        preds = run_eval_only(args, test_df, device)
        if args.threshold_from == "val":
            logger.info("--eval_only: no validation data at this site, so thresholds are chosen on the evaluation data")
            args.threshold_from = "test"
    else:
        splits = split_cohort(cohort, args.split, split_col, args.test_value, args.test_size, args.val_size, args.seed)
        test_df = splits[2]
        for name, df in zip(["train", "validation", "test"], splits):
            logger.info("%s positives: %s", name, ", ".join(f"{o}={int(np.nansum(df[o]))}" for o in outcomes))
        runner = {"frozen": run_frozen, "embedding_lr": run_embedding_lr, "finetune": run_finetune}[args.mode]
        preds, thr_data = runner(args, splits, outcomes, device)
    logger.info("Finished modelling in %.1f min", (time.time() - t0) / 60)
    evaluate_and_save(args, test_df, preds, thr_data)
    logger.info("Results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
