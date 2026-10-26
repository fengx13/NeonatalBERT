#!/usr/bin/env python
"""
Build the external (BIDMC) newborn cohort from MIMIC-III v1.4 in the input
format used by model_fine_tune/finetune.py.

Cohort (following Xie et al., Lancet Digit Health 2025):
  * first NEWBORN admission per subject
  * birth time approximated by ADMITTIME of the newborn admission
  * notes with CHARTTIME within --max_hours (default 24 h) of birth
  * exclude: no eligible notes, death within 24 h of birth
  * outcomes from DIAGNOSES_ICD (ICD-9 prefixes in configs/mimic3_icd9_outcomes.json)
    and in-hospital death (HOSPITAL_EXPIRE_FLAG)

Discharge summaries are never used as model input (they describe the whole
stay and would leak outcomes).

Usage
-----
python data_preparation/build_mimic3_cohort.py \
    --mimic_dir /path/to/mimic-iii-clinical-database-1.4 \
    --out_dir data/mimic3_newborn

MIMIC-III requires credentialed PhysioNet access and a data use agreement:
https://physionet.org/content/mimiciii/1.4/
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("neonatalbert.mimic3")

DEFAULT_CATEGORIES = ["Physician", "General", "Nursing/other", "Nursing", "Consult", "Respiratory"]


def find_table(mimic_dir: Path, name: str) -> Path:
    for cand in [f"{name}.csv.gz", f"{name}.csv", f"{name.lower()}.csv.gz", f"{name.lower()}.csv"]:
        if (mimic_dir / cand).exists():
            return mimic_dir / cand
    raise FileNotFoundError(f"{name}.csv(.gz) not found in {mimic_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--outcome_codes", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "mimic3_icd9_outcomes.json")
    ap.add_argument("--max_hours", type=float, default=24)
    ap.add_argument("--note_categories", default=",".join(DEFAULT_CATEGORIES),
                    help="Comma-separated NOTEEVENTS categories to keep")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Newborn admissions -------------------------------------------------
    adm = pd.read_csv(find_table(args.mimic_dir, "ADMISSIONS"))
    for col in ["ADMITTIME", "DEATHTIME"]:
        adm[col] = pd.to_datetime(adm[col], format="mixed")
    adm = adm[adm["ADMISSION_TYPE"] == "NEWBORN"].sort_values("ADMITTIME")
    adm = adm.drop_duplicates("SUBJECT_ID", keep="first")
    logger.info("Newborn admissions (first per subject): %d", len(adm))

    early_death = (adm["DEATHTIME"] - adm["ADMITTIME"]).dt.total_seconds() / 3600 < 24
    adm = adm[~early_death.fillna(False)]
    logger.info("After excluding death within 24 h: %d", len(adm))
    hadm_ids = set(adm["HADM_ID"])
    admit_time = adm.set_index("HADM_ID")["ADMITTIME"]

    # 2. First-day notes -----------------------------------------------------
    keep_cats = {c.strip() for c in args.note_categories.split(",")}
    usecols = ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "DESCRIPTION", "TEXT", "ISERROR"]
    parts = []
    for chunk in pd.read_csv(find_table(args.mimic_dir, "NOTEEVENTS"), usecols=usecols, chunksize=args.chunksize, low_memory=False):
        chunk = chunk[chunk["HADM_ID"].isin(hadm_ids)]
        chunk = chunk[chunk["ISERROR"].isna() | (chunk["ISERROR"] != 1)]
        chunk["CATEGORY"] = chunk["CATEGORY"].str.strip()
        chunk = chunk[chunk["CATEGORY"].isin(keep_cats) & chunk["CHARTTIME"].notna()]
        parts.append(chunk)
    notes = pd.concat(parts, ignore_index=True)
    notes["CHARTTIME"] = pd.to_datetime(notes["CHARTTIME"], format="mixed")
    notes["hours_since_birth"] = (notes["CHARTTIME"] - notes["HADM_ID"].map(admit_time)).dt.total_seconds() / 3600
    notes = notes[(notes["hours_since_birth"] >= 0) & (notes["hours_since_birth"] <= args.max_hours)]
    notes_out = pd.DataFrame({
        "patient_id": notes["SUBJECT_ID"],
        "note_text": notes["TEXT"],
        "hours_since_birth": notes["hours_since_birth"].round(2),
        "note_type": notes["CATEGORY"] + " - " + notes["DESCRIPTION"].fillna(""),
    })
    logger.info("Eligible notes: %d for %d patients", len(notes_out), notes_out["patient_id"].nunique())

    # 3. Outcomes --------------------------------------------------------------
    codes = {k: v for k, v in json.loads(args.outcome_codes.read_text()).items() if not k.startswith("_")}
    dx = pd.read_csv(find_table(args.mimic_dir, "DIAGNOSES_ICD"), dtype={"ICD9_CODE": str})
    dx = dx[dx["HADM_ID"].isin(hadm_ids)].dropna(subset=["ICD9_CODE"])
    labels = adm[["SUBJECT_ID", "HADM_ID", "HOSPITAL_EXPIRE_FLAG"]].rename(columns={"SUBJECT_ID": "patient_id"})
    for outcome, prefixes in codes.items():
        hit = dx[dx["ICD9_CODE"].str.startswith(tuple(prefixes))]["HADM_ID"].unique()
        labels[outcome] = labels["HADM_ID"].isin(hit).astype(int)
    labels["death"] = labels["HOSPITAL_EXPIRE_FLAG"].astype(int)
    labels = labels.drop(columns=["HADM_ID", "HOSPITAL_EXPIRE_FLAG"])
    labels = labels[labels["patient_id"].isin(notes_out["patient_id"])]

    notes_out.to_csv(args.out_dir / "notes.csv.gz", index=False)
    labels.to_csv(args.out_dir / "labels.csv", index=False)
    logger.info("Cohort: %d patients", len(labels))
    logger.info("Outcome prevalence:\n%s", labels.drop(columns="patient_id").mean().round(4).to_string())
    logger.info("Wrote %s", args.out_dir)


if __name__ == "__main__":
    main()
