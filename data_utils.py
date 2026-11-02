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
    <outcome>_dx_hours  (optional) hours from birth to diagnosis, used to exclude
                        cases already diagnosed within the prediction window
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


def read_table(path, id_col: str = "patient_id") -> pd.DataFrame:
    """Read a table; patient ids are always handled as stripped strings so that
    notes and labels match regardless of how each file stored them."""
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif suffixes.endswith(".pkl") or suffixes.endswith(".pickle"):
        df = pd.read_pickle(path)
    elif ".csv" in suffixes:
        df = pd.read_csv(path, dtype={id_col: str})
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if id_col in df:
        if df[id_col].isna().any():
            raise ValueError(f"{path}: {df[id_col].isna().sum()} rows have a missing {id_col}")
        df[id_col] = df[id_col].astype(str).str.strip()
    return df


def check_id_overlap(notes: pd.DataFrame, labels: pd.DataFrame, id_col: str = "patient_id") -> None:
    """Fail loudly when notes and labels do not share patient ids (e.g. '000123' vs '123')."""
    n_lab = labels[id_col].nunique()
    shared = len(set(labels[id_col]) & set(notes[id_col]))
    if shared == 0:
        raise ValueError(
            f"No {id_col} values are shared between the notes and labels files. Check that both use the same "
            f"identifier format (e.g. leading zeros). Examples - notes: {notes[id_col].head(3).tolist()}, "
            f"labels: {labels[id_col].head(3).tolist()}"
        )
    if shared < 0.5 * n_lab:
        logger.warning("Only %d of %d labelled patients have any notes; check the %s format.", shared, n_lab, id_col)


def check_unique_ids(labels: pd.DataFrame, id_col: str = "patient_id") -> None:
    dup = labels[id_col].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"The labels file must have one row per patient, but {labels.loc[dup, id_col].nunique()} {id_col} values "
            f"appear more than once (e.g. {labels.loc[dup, id_col].head(3).tolist()}). Duplicates could place the same "
            f"patient in both the training and test sets."
        )


def aggregate_notes(
    notes: pd.DataFrame,
    max_hours: Optional[float] = 24,
    note_types: Optional[Sequence[str]] = None,
    id_col: str = "patient_id",
    text_col: str = "note_text",
    time_col: str = "hours_since_birth",
    type_col: str = "note_type",
    include_note_type_header: bool = False,
) -> pd.DataFrame:
    """
    Keep notes written within `max_hours` of birth (the paper uses the first
    24 h), drop exact duplicate texts, and collect each patient's notes in
    chronological order.

    Returns one row per patient with
        notes : list of note texts (used for note-level embeddings, as in the paper)
        text  : the same notes joined into one document (used by the LLM baseline)
    """
    df = notes.copy()
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]

    if note_types is not None:
        if type_col not in df:
            raise ValueError(f"--note_types was given but the notes table has no '{type_col}' column")
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
    df = df.sort_values(sort_cols, kind="stable")

    def _fmt(row):
        txt = str(row[text_col]).strip()
        if include_note_type_header and type_col in row and pd.notna(row[type_col]):
            return f"[{row[type_col]}] {txt}"
        return txt

    df["_formatted"] = df.apply(_fmt, axis=1) if len(df) else pd.Series(dtype=str)
    agg = df.groupby(id_col, sort=False)["_formatted"].apply(list).reset_index().rename(columns={"_formatted": "notes"})
    agg["text"] = agg["notes"].apply(lambda xs: "\n\n".join(xs))
    logger.info("Collected %d notes for %d patients (mean %.1f notes per patient)",
                len(df), len(agg), len(df) / max(len(agg), 1))
    return agg


def build_cohort(
    notes_path,
    labels_path,
    outcomes: Optional[List[str]] = None,
    max_hours: Optional[float] = 24,
    note_types: Optional[Sequence[str]] = None,
    split_col: Optional[str] = None,
    id_col: str = "patient_id",
    include_note_type_header: bool = False,
    dx_suffix: str = "_dx_hours",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge first-day notes with patient-level labels.

    Exclusions (as in the paper):
      * patients without eligible notes
      * patients with missing outcome data
      * per outcome: patients already diagnosed within the prediction window.
        If the labels table has a column `<outcome>_dx_hours` (hours from birth
        to diagnosis), positive patients with <outcome>_dx_hours <= max_hours
        are set to missing for that outcome only, so they are excluded from
        its training and evaluation.
    """
    notes = read_table(notes_path, id_col)
    labels = read_table(labels_path, id_col)
    check_unique_ids(labels, id_col)
    check_id_overlap(notes, labels, id_col)

    docs = aggregate_notes(notes, max_hours=max_hours, note_types=note_types, id_col=id_col,
                           include_note_type_header=include_note_type_header)
    if outcomes is None:
        raise ValueError("Specify the outcome columns explicitly (--outcomes), e.g. --outcomes rds,bpd,nec,death")
    missing = [o for o in outcomes if o not in labels.columns]
    if missing:
        raise ValueError(f"Outcome columns not found in labels file: {missing}")
    bad = [o for o in outcomes if not set(pd.unique(labels[o].dropna())) <= {0, 1}]
    if bad:
        raise ValueError(f"Outcome columns must be binary 0/1: {bad}")

    dx_cols = [f"{o}{dx_suffix}" for o in outcomes if f"{o}{dx_suffix}" in labels]
    keep = [id_col] + list(outcomes) + dx_cols + ([split_col] if split_col else [])
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
        cohort[o] = cohort[o].astype(float)

    # Per-outcome exclusion of cases diagnosed within the prediction window.
    if max_hours is not None:
        for o in outcomes:
            col = f"{o}{dx_suffix}"
            if col in cohort:
                early = (cohort[o] == 1) & (cohort[col] <= max_hours)
                if early.any():
                    cohort.loc[early, o] = np.nan
                    logger.info("%s: excluded %d cases diagnosed within the first %g h", o, int(early.sum()), max_hours)
        if outcomes and not dx_cols:
            logger.warning("No <outcome>%s columns found: cases diagnosed within the prediction window cannot be "
                           "excluded (the paper excluded them). See README 'Input data format'.", dx_suffix)
        cohort = cohort.drop(columns=dx_cols)
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
