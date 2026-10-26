#!/usr/bin/env python
"""
Generate a small synthetic NICU dataset in the input format expected by the
NeonatalBERT scripts. The notes are random template text with weak signal for
each outcome. They contain no real patient information and are only meant to
check that the pipeline runs end to end on your machine.

Outputs (in --out_dir):
    notes.csv        patient_id, note_text, hours_since_birth, note_type
    labels.csv       patient_id, <outcomes>, birth_year, split
    pretrain_corpus.csv  note table for the pretraining example
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOME_CUES = {
    "rds": ["grunting", "retractions", "surfactant given", "CPAP started"],
    "bpd": ["prolonged oxygen requirement", "extremely preterm", "mechanical ventilation"],
    "ivh": ["head ultrasound pending", "bulging fontanelle", "very low birth weight"],
    "nec": ["abdominal distension", "bilious aspirates", "feeds held"],
    "sepsis": ["maternal fever", "prolonged rupture of membranes", "blood culture sent", "ampicillin and gentamicin"],
    "jaundice": ["bilirubin elevated", "phototherapy", "ABO incompatibility"],
    "death": ["comfort care discussed", "severe hypotension", "multiple pressors"],
}

FILLER = [
    "Infant born via {mode} at {ga} weeks gestation.",
    "Birth weight {bw} grams.",
    "Apgar scores {a1} and {a2}.",
    "Mother is a {age} year old G{g}P{p}.",
    "Infant admitted to NICU for {reason}.",
    "Vital signs stable on exam.",
    "Plan to monitor glucose and temperature.",
    "Parents updated at bedside.",
    "Lungs clear bilaterally.",
    "Skin pink and well perfused.",
]
NOTE_TYPES = ["History and physical", "Progress note", "Procedure note", "Consult"]


def make_note(rng, positives):
    parts = [
        f.format(
            mode=rng.choice(["vaginal delivery", "cesarean section"]),
            ga=rng.randint(23, 41),
            bw=rng.randint(500, 4200),
            a1=rng.randint(1, 9),
            a2=rng.randint(5, 10),
            age=rng.randint(18, 44),
            g=rng.randint(1, 5),
            p=rng.randint(0, 4),
            reason=rng.choice(["prematurity", "respiratory distress", "sepsis evaluation", "observation"]),
        )
        for f in rng.choice(FILLER, size=rng.randint(4, 9), replace=False)
    ]
    for outcome, cues in OUTCOME_CUES.items():
        p_cue = 0.7 if outcome in positives else 0.05
        if rng.rand() < p_cue:
            parts.append(f"Notable for {rng.choice(cues)}.")
    rng.shuffle(parts)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="examples/synthetic")
    ap.add_argument("--n_patients", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prevalence = {"rds": 0.25, "bpd": 0.08, "ivh": 0.06, "nec": 0.05, "sepsis": 0.10, "jaundice": 0.30, "death": 0.04}

    notes, labels = [], []
    for i in range(args.n_patients):
        pid = f"P{i:05d}"
        y = {o: int(rng.rand() < p) for o, p in prevalence.items()}
        positives = {o for o, v in y.items() if v}
        year = int(rng.choice(range(2014, 2022)))
        labels.append({"patient_id": pid, **y, "birth_year": year, "split": "test" if year == 2021 else "train"})
        for _ in range(rng.randint(1, 5)):
            notes.append({
                "patient_id": pid,
                "note_text": make_note(rng, positives),
                "hours_since_birth": round(float(rng.uniform(0, 96)), 1),
                "note_type": rng.choice(NOTE_TYPES),
            })
    notes = pd.DataFrame(notes)
    notes.to_csv(out / "notes.csv", index=False)
    pd.DataFrame(labels).to_csv(out / "labels.csv", index=False)
    notes.rename(columns={"note_text": "text"}).assign(note_id=range(len(notes)))[["note_id", "text"]].to_csv(out / "pretrain_corpus.csv", index=False)
    print(f"Wrote {len(notes)} notes for {args.n_patients} patients to {out}")


if __name__ == "__main__":
    main()
