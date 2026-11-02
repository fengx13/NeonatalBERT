#!/usr/bin/env python
"""
One-command domain-adaptive pretraining at your own institution.

Because the original NeonatalBERT weights cannot be shared (they were trained
on Stanford clinical notes), sites can re-create a NeonatalBERT-style model on
their own neonatal notes with the same recipe as the paper:

    Bio_ClinicalBERT  --(MLM + NSP on local neonatal notes)-->  local NeonatalBERT

This script runs the full pipeline:

  0. defines the held-out TEST patients and removes all of their notes from the
     pretraining corpus, so pretraining never sees test-set text
  1. keeps notes from the first --pretrain_max_days days of life (paper: ~3 months)
  2. prepare_notes.py           -> sentence-per-line corpus
  3. create_pretraining_data.py -> MLM + NSP instances
  4. pretrain.py                -> pretrained model (Hugging Face format)

and prints the matching fine-tuning command. Steps whose outputs already exist
are skipped (use --overwrite to redo them), so an interrupted run can be resumed.

Test-set definition (must match what you use for fine-tuning):
  --split column : test patients are rows of --labels where --split_col == --test_value
  --split random : a random patient-level split is drawn here (same procedure as
                   finetune.py) and saved as a new column `neonatalbert_split` in
                   <work_dir>/labels_with_split.csv; fine-tune with that file and
                   `--split column --split_col neonatalbert_split`
  --split none   : no labels given / no held-out set (all notes are used)

Example
-------
python model_pretraining/pretrain_local.py \
    --notes data/notes.csv --labels data/labels.csv \
    --split random --test_size 0.2 \
    --base_model emilyalsentzer/Bio_ClinicalBERT \
    --work_dir pretraining_runs/site_x --fp16

All intermediate files contain note text: keep --work_dir inside your
institution's secure computing environment.
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from data_utils import build_cohort, read_table, split_cohort  # noqa: E402
from helpers import parse_list, save_json  # noqa: E402

logger = logging.getLogger("neonatalbert.pretrain_local")


def run_step(name, cmd, done_marker: Path, overwrite: bool):
    if done_marker.exists() and not overwrite:
        logger.info("[skip] %s: %s already exists", name, done_marker)
        return
    logger.info("[run] %s\n  %s", name, " ".join(str(c) for c in cmd))
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True)
    logger.info("[done] %s in %.1f min", name, (time.time() - t0) / 60)


def define_test_patients(args, work_dir: Path):
    """Return (set of test patient ids, fine-tuning split arguments)."""
    if args.split == "none" or args.labels is None:
        logger.warning("No held-out test set defined: ALL notes will be used for pretraining. "
                       "Do not evaluate the resulting model on any of these patients.")
        return set(), None

    labels = read_table(args.labels)
    notes_ids = set(read_table(args.notes)["patient_id"])
    if not notes_ids & set(labels["patient_id"]):
        raise SystemExit("No patient_id values are shared between --notes and --labels; check the id format "
                         "(e.g. leading zeros). Test patients could not be excluded, so pretraining was not started.")
    if args.split == "column":
        if not args.split_col:
            raise SystemExit("--split column requires --split_col")
        test_ids = set(labels.loc[labels[args.split_col].astype(str) == str(args.test_value), "patient_id"])
        ft_args = ["--labels", args.labels, "--split", "column", "--split_col", args.split_col, "--test_value", args.test_value]
    else:  # random: draw the split here and save it so fine-tuning uses the identical test set
        cohort, _ = build_cohort(args.notes, args.labels, parse_list(args.outcomes) or [], args.max_hours)
        _, _, test_df = split_cohort(cohort, "random", test_size=args.test_size, seed=args.seed)
        test_ids = set(test_df["patient_id"])
        labels["neonatalbert_split"] = labels["patient_id"].isin(test_ids).map({True: "test", False: "train"})
        out = work_dir / "labels_with_split.csv"
        labels.to_csv(out, index=False)
        logger.info("Saved the train/test assignment to %s", out)
        ft_args = ["--labels", str(out), "--split", "column", "--split_col", "neonatalbert_split", "--test_value", "test"]
    logger.info("Held-out test patients excluded from pretraining: %d", len(test_ids))
    return test_ids, ft_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--notes", required=True, help="Note-level table (patient_id, note_text, [hours_since_birth])")
    d.add_argument("--labels", default=None, help="Patient-level labels table (needed to hold out test patients)")
    d.add_argument("--split", choices=["random", "column", "none"], default="random")
    d.add_argument("--split_col", default=None)
    d.add_argument("--test_value", default="test")
    d.add_argument("--test_size", type=float, default=0.2)
    d.add_argument("--outcomes", default=None, help="Outcome columns; patients missing any of them are not used to draw the random split")
    d.add_argument("--max_hours", type=float, default=24, help="Prediction window used for fine-tuning (defines the cohort)")
    d.add_argument("--pretrain_max_days", type=float, default=90, help="Use notes from the first N days of life (paper: 3 months)")
    d.add_argument("--seed", type=int, default=0)

    m = ap.add_argument_group("pretraining")
    m.add_argument("--base_model", default="emilyalsentzer/Bio_ClinicalBERT")
    m.add_argument("--work_dir", required=True, type=Path)
    m.add_argument("--max_seq_len", type=int, default=128)
    m.add_argument("--epochs_to_generate", type=int, default=3)
    m.add_argument("--epochs", type=int, default=3)
    m.add_argument("--train_batch_size", type=int, default=32)
    m.add_argument("--gradient_accumulation_steps", type=int, default=1)
    m.add_argument("--learning_rate", type=float, default=5e-5)
    m.add_argument("--no_whole_word_mask", action="store_true")
    m.add_argument("--reduce_memory", action="store_true", help="Keep the document store on disk (large corpora)")
    m.add_argument("--num_workers", type=int, default=2)
    m.add_argument("--fp16", action="store_true")
    m.add_argument("--overwrite", action="store_true",
                   help="Redo steps whose outputs already exist (required after changing the split or note filters)")
    args = ap.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO,
                        handlers=[logging.StreamHandler(), logging.FileHandler(work / "pretrain_local.log")])

    # 0-1. Select pretraining notes ------------------------------------------
    test_ids, ft_split_args = define_test_patients(args, work)

    # Guard against resuming a run whose held-out set was different (would leak test notes).
    fingerprint = hashlib.sha256(json.dumps({
        "test_ids": sorted(map(str, test_ids)), "notes": str(args.notes), "pretrain_max_days": args.pretrain_max_days,
        "base_model": args.base_model, "max_seq_len": args.max_seq_len, "seed": args.seed,
    }).encode()).hexdigest()
    fp_file = work / "run_fingerprint.txt"
    if fp_file.exists() and fp_file.read_text().strip() != fingerprint and not args.overwrite:
        raise SystemExit(f"{work} contains a previous run with a different test set or settings. "
                         "Use a new --work_dir, or add --overwrite to rebuild everything.")
    fp_file.write_text(fingerprint)
    notes = read_table(args.notes)
    n_all = len(notes)
    notes = notes[~notes["patient_id"].isin(test_ids)]
    n_after_test = len(notes)
    if "hours_since_birth" in notes:
        notes = notes[(notes["hours_since_birth"] >= 0) & (notes["hours_since_birth"] <= args.pretrain_max_days * 24)]
    else:
        logger.warning("No hours_since_birth column: --pretrain_max_days is not applied")
    notes = notes.dropna(subset=["note_text"]).drop_duplicates(subset=["note_text"])
    corpus_notes = work / "pretraining_notes.pkl"
    notes[["patient_id", "note_text"]].to_pickle(corpus_notes)
    logger.info("Pretraining notes: %d of %d (removed %d from test patients, then kept first %g days, deduplicated); %d patients",
                len(notes), n_all, n_all - n_after_test, args.pretrain_max_days, notes["patient_id"].nunique())

    # 2-4. Run the three pretraining steps -----------------------------------
    corpus_txt = work / "corpus.txt"
    pregen = work / "pregenerated"
    model_dir = work / "model"
    py = sys.executable

    run_step("prepare notes",
             [py, HERE / "prepare_notes.py", "--input_path", corpus_notes, "--output_path", corpus_txt],
             corpus_txt, args.overwrite)

    gen_cmd = [py, HERE / "create_pretraining_data.py", "--train_corpus", corpus_txt, "--bert_model", args.base_model,
               "--output_dir", pregen, "--epochs_to_generate", args.epochs_to_generate, "--max_seq_len", args.max_seq_len,
               "--seed", args.seed]
    if not args.no_whole_word_mask:
        gen_cmd.append("--do_whole_word_mask")
    if args.reduce_memory:
        gen_cmd.append("--reduce_memory")
    run_step("create pretraining data", gen_cmd, pregen / f"epoch_{args.epochs_to_generate - 1}_metrics.json", args.overwrite)

    pre_cmd = [py, HERE / "pretrain.py", "--pregenerated_data", pregen, "--bert_model", args.base_model,
               "--output_dir", model_dir, "--epochs", args.epochs, "--train_batch_size", args.train_batch_size,
               "--gradient_accumulation_steps", args.gradient_accumulation_steps, "--learning_rate", args.learning_rate,
               "--num_workers", args.num_workers, "--seed", args.seed]
    if args.fp16:
        pre_cmd.append("--fp16")
    run_step("pretrain", pre_cmd, model_dir / "config.json", args.overwrite)

    # 5. Record what was done and print the next step -------------------------
    info = {
        "base_model": args.base_model,
        "n_pretraining_notes": int(len(notes)),
        "n_pretraining_patients": int(notes["patient_id"].nunique()),
        "n_test_patients_excluded": len(test_ids),
        "pretrain_max_days": args.pretrain_max_days,
        "split": args.split,
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    save_json(info, work / "pretraining_info.json")

    split_args = ft_split_args or ["--labels", "<labels>", "--split", "random"]
    pairs = [("--notes", args.notes)] + list(zip(split_args[::2], split_args[1::2])) + [
        ("--model_name_or_path", model_dir), ("--max_hours", f"{args.max_hours:g}"),
        ("--output_dir", work / "finetune_results"), ("--save_models", "")]
    if args.fp16:
        pairs.append(("--fp16", ""))
    lines = ["python model_fine_tune/finetune.py"] + [f"{k} {v}".strip() for k, v in pairs]
    print("\nPretraining finished. Model saved to:", model_dir)
    print("Next step (fine-tune and evaluate on the held-out test patients):\n")
    print("  " + " \\\n      ".join(lines) + "\n")


if __name__ == "__main__":
    main()
