"""
Data loading utilities shared by the NeonatalBERT scripts.

Expected inputs (see README "Input data format"):

notes table (one row per note)
    patient_id          identifier of the newborn
    note_text           free text of the note
    hours_since_birth   (optional) note time relative to birth, in hours
    note_type           (optional) e.g. "History and physical", "Progress note"

labels table (one row per patient)
    patient_id
    <outcome columns>   binary 0/1, one column per outcome
    <split column>      (optional) e.g. split = "train" / "test"

Tables may be .csv, .csv.gz, .parquet or .pkl.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".pkl") or suffixes.endswith(".pickle"):
        return pd.read_pickle(path)
    if ".csv" in suffixes:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def aggregate_notes(
    notes: pd.DataFrame,
    max_hours: Optional[float] = 24,
    note_types: Optional[Sequence[str]] = None,
    id_col: str = "patient_id",
    text_col: str = "note_text",
    time_col: str = "hours_since_birth",
    type_col: str = "note_type",
    include_note_type_header: bool = True,
) -> pd.DataFrame:
    """
    Keep notes written within `max_hours` of birth (the paper uses the first
    24 h), drop exact duplicate texts, and concatenate each patient's notes in
    chronological order into a single document.
    """
    df = notes.copy()
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]

    if note_types is not None and type_col in df:
        df = df[df[type_col].isin(note_types)]

    if max_hours is not None:
        if time_col in df:
            df = df[(df[time_col] >= 0) & (df[time_col] <= max_hours)]
        else:
            logger.warning(
                "No '%s' column found; assuming notes were already restricted to the first %s h.",
                time_col,
                max_hours,
            )

    df = df.drop_duplicates(subset=[id_col, text_col])
    sort_cols = [id_col] + ([time_col] if time_col in df else [])
    df = df.sort_values(sort_cols)

    def _fmt(row):
        txt = str(row[text_col]).strip()
        if include_note_type_header and type_col in row and pd.notna(row[type_col]):
            return f"[{row[type_col]}] {txt}"
        return txt

    df["_formatted"] = df.apply(_fmt, axis=1)
    agg = df.groupby(id_col, sort=False)["_formatted"].apply(lambda s: "\n\n".join(s)).reset_index()
    agg = agg.rename(columns={"_formatted": "text"})
    logger.info("Aggregated %d notes into %d patient documents", len(df), len(agg))
    return agg


def build_cohort(
    notes_path,
    labels_path,
    outcomes: Optional[List[str]] = None,
    max_hours: Optional[float] = 24,
    note_types: Optional[Sequence[str]] = None,
    split_col: Optional[str] = None,
    id_col: str = "patient_id",
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge aggregated first-day notes with patient-level labels."""
    notes = read_table(notes_path)
    labels = read_table(labels_path)

    docs = aggregate_notes(notes, max_hours=max_hours, note_types=note_types, id_col=id_col)
    reserved = {id_col, "text"} | ({split_col} if split_col else set())
    if outcomes is None:
        # Default: every binary (0/1) column other than the id / split columns.
        outcomes = [
            c for c in labels.columns
            if c not in reserved and set(pd.unique(labels[c].dropna())) <= {0, 1}
        ]
        logger.info("No --outcomes given; using binary columns: %s", outcomes)
    missing = [o for o in outcomes if o not in labels.columns]
    if missing:
        raise ValueError(f"Outcome columns not found in labels file: {missing}")

    keep = [id_col] + list(outcomes) + ([split_col] if split_col else [])
    cohort = docs.merge(labels[keep], on=id_col, how="inner")
    n_no_notes = labels[id_col].nunique() - cohort[id_col].nunique()
    if n_no_notes:
        logger.info("Excluded %d patients without eligible notes", n_no_notes)

    # Patients with missing outcome data are excluded (as in the paper).
    before = len(cohort)
    cohort = cohort.dropna(subset=list(outcomes))
    if before - len(cohort):
        logger.info("Excluded %d patients with missing outcome data", before - len(cohort))
    for o in outcomes:
        cohort[o] = cohort[o].astype(int)
    return cohort.reset_index(drop=True), list(outcomes)


def split_cohort(
    cohort: pd.DataFrame,
    split: str = "random",
    split_col: Optional[str] = None,
    test_value: str = "test",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level train / validation / test split.

    split="column": use `split_col` (rows equal to `test_value` -> test, the
        rest -> development). Use this for the temporal split in the paper
        (births 2014-2020 for development, 2021 for testing).
    split="random": random patient-level split (the paper's external MIMIC-III
        cohort used 80% / 20%).
    split="none": every patient goes to test (evaluation of a fixed model).

    The validation set is a random subset of the development set, used for
    hyperparameter selection and early stopping.
    """
    rng = np.random.RandomState(seed)
    if split == "none":
        empty = cohort.iloc[0:0]
        return empty, empty, cohort
    if split == "column":
        if not split_col:
            raise ValueError("--split column requires --split_col")
        is_test = cohort[split_col].astype(str) == str(test_value)
        dev, test = cohort[~is_test], cohort[is_test]
    elif split == "random":
        idx = rng.permutation(len(cohort))
        n_test = int(round(test_size * len(cohort)))
        test, dev = cohort.iloc[idx[:n_test]], cohort.iloc[idx[n_test:]]
    else:
        raise ValueError(f"Unknown split: {split}")

    idx = rng.permutation(len(dev))
    n_val = int(round(val_size * len(dev)))
    val, train = dev.iloc[idx[:n_val]], dev.iloc[idx[n_val:]]
    logger.info("Split sizes: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
