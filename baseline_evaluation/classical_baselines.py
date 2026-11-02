#!/usr/bin/env python
"""
Classical baselines from Xie et al., Lancet Digit Health 2025, trained and
evaluated on the same patients and splits as finetune.py:

  --baseline tabular_lr      logistic regression on first-day structured EHR
                             features (appendix eTable 1: maternal age, sex,
                             gestational age, birth weight, vital signs and
                             laboratory values); missing values imputed with
                             training-set medians
  --baseline tfidf_svm       TF-IDF of the first-day notes + linear SVM (eTable 8)
  --baseline tfidf_xgboost   TF-IDF of the first-day notes + XGBoost (eTable 8;
                             requires `pip install xgboost`)

Hyperparameters are chosen per outcome by cross-validation on the development
set (training + validation patients), scored by AUPRC. Thresholds for
sensitivity/specificity/F1 come from out-of-fold development-set predictions.

Usage
-----
python baseline_evaluation/classical_baselines.py --baseline tabular_lr \
    --notes data/notes.csv --labels data/labels.csv --features data/tabular.csv \
    --outcomes rds,bpd,nec,death --split column --split_col split --test_value test \
    --output_dir results/tabular_lr
"""

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model_fine_tune"))
from data_utils import build_cohort, read_table, split_cohort  # noqa: E402
from finetune import evaluate_and_save  # noqa: E402
from helpers import parse_list, save_json  # noqa: E402

logger = logging.getLogger("neonatalbert.classical_baselines")


def make_estimator(baseline: str, seed: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if baseline == "tabular_lr":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000))])
        grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}
    elif baseline == "tfidf_svm":
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
                         ("clf", CalibratedClassifierCV(LinearSVC(), cv=3))])
        grid = {"clf__estimator__C": [0.01, 0.1, 1], "clf__estimator__class_weight": [None, "balanced"]}
    elif baseline == "tfidf_xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise SystemExit("tfidf_xgboost requires xgboost: pip install xgboost") from e
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
                         ("clf", XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1))])
        grid = {"clf__n_estimators": [200, 500], "clf__max_depth": [3, 6], "clf__learning_rate": [0.05, 0.1]}
    else:
        raise ValueError(baseline)
    return pipe, grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", choices=["tabular_lr", "tfidf_svm", "tfidf_xgboost"], required=True)
    ap.add_argument("--notes", required=True, help="Notes table (defines the cohort, and the text for TF-IDF)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--features", default=None, help="tabular_lr: patient_id + numeric first-day features")
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--max_hours", type=float, default=24)
    ap.add_argument("--note_types", default=None)
    ap.add_argument("--split", choices=["random", "column"], default="random")
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--test_value", default="test")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_bootstraps", type=int, default=1000)
    ap.add_argument("--threshold_from", choices=["val", "test"], default="val")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO,
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

    split_col = args.split_col if args.split == "column" else None
    cohort, outcomes = build_cohort(args.notes, args.labels, parse_list(args.outcomes), args.max_hours,
                                    parse_list(args.note_types), split_col)
    train_df, val_df, test_df = split_cohort(cohort, args.split, split_col, args.test_value, args.test_size, args.val_size, args.seed)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    if args.baseline == "tabular_lr":
        if not args.features:
            ap.error("--baseline tabular_lr requires --features")
        feats = read_table(args.features)
        cols = [c for c in feats.columns if c != "patient_id"]
        non_num = [c for c in cols if not pd.api.types.is_numeric_dtype(feats[c])]
        if non_num:
            raise SystemExit(f"Features must be numeric (encode categorical variables first): {non_num}")
        feats = feats.drop_duplicates("patient_id").set_index("patient_id")
        X_dev = feats.reindex(dev_df["patient_id"]).to_numpy(dtype=float)
        X_te = feats.reindex(test_df["patient_id"]).to_numpy(dtype=float)
        logger.info("%d features; %.1f%% of development values missing (median-imputed from training data)",
                    len(cols), 100 * np.isnan(X_dev).mean())
    else:
        X_dev, X_te = dev_df["text"].to_numpy(), test_df["text"].to_numpy()

    preds, thr_data = {}, {}
    for o in outcomes:
        m = dev_df[o].notna().to_numpy()
        y = dev_df.loc[m, o].to_numpy().astype(int)
        n_splits = int(min(5, y.sum(), len(y) - y.sum()))
        if n_splits < 2:
            logger.warning("Skipping %s: too few positives/negatives for cross-validation", o)
            continue
        pipe, grid = make_estimator(args.baseline, args.seed)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        search.fit(X_dev[m], y)
        logger.info("%s | best %s | CV AUPRC %.4f", o, search.best_params_, search.best_score_)
        preds[o] = search.predict_proba(X_te)[:, 1]
        thr_data[o] = (y.astype(float), cross_val_predict(search.best_estimator_, X_dev[m], y, cv=cv, method="predict_proba")[:, 1])

    save_json(vars(args), out / "run_config.json")
    evaluate_and_save(SimpleNamespace(output_dir=args.output_dir, n_bootstraps=args.n_bootstraps, seed=args.seed,
                                      threshold_from=args.threshold_from), test_df, preds, thr_data)


if __name__ == "__main__":
    main()
