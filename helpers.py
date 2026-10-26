"""
Evaluation helpers shared by the NeonatalBERT fine-tuning and baseline scripts.

Metrics follow the NeonatalBERT paper (Xie et al., Lancet Digit Health 2025):
AUROC, AUPRC (also relative to a no-skill classifier), F1, sensitivity and
specificity, each with a 95% bootstrap confidence interval.

The operating threshold for the threshold-dependent metrics (sensitivity,
specificity, F1) is the ROC point closest to the top-left corner (0, 1),
chosen on the data being evaluated unless a fixed threshold is passed in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib

matplotlib.use("Agg")  # headless servers
from matplotlib import pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------
def youden_closest_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold of the ROC point closest to (FPR=0, TPR=1)."""
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    idx = np.sqrt(fpr ** 2 + (1 - tpr) ** 2).argmin()
    thr = thresholds[idx]
    # roc_curve returns +inf as the first threshold; fall back to max prob.
    return float(np.max(y_prob) if not np.isfinite(thr) else thr)


def binary_metrics(y_true, y_prob, threshold: Optional[float] = None) -> Dict[str, float]:
    """AUROC, AUPRC, relative AUPRC, sensitivity, specificity, F1, PPV, NPV."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if threshold is None:
        threshold = youden_closest_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prevalence = y_true.mean()
    auprc = metrics.average_precision_score(y_true, y_prob)
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "prevalence": float(prevalence),
        "auroc": float(metrics.roc_auc_score(y_true, y_prob)),
        "auprc": float(auprc),
        "auprc_relative": float(auprc / prevalence) if prevalence > 0 else float("nan"),
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_metrics(
    y_true,
    y_prob,
    n_bootstraps: int = 1000,
    ci: float = 95,
    random_seed: int = 0,
    threshold: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Point estimate plus percentile bootstrap CI for every metric in
    `binary_metrics`. The threshold is fixed from the full sample so that the
    CI reflects sampling variability of the test set, not threshold search.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    point = binary_metrics(y_true, y_prob, threshold)
    thr = point["threshold"]

    rng = np.random.RandomState(random_seed)
    keys = ["auroc", "auprc", "auprc_relative", "sensitivity", "specificity", "ppv", "npv", "f1"]
    samples = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # AUROC undefined without both classes
        m = binary_metrics(y_true[idx], y_prob[idx], thr)
        for k in keys:
            samples[k].append(m[k])

    alpha = (100 - ci) / 2
    out = {}
    for k, v in point.items():
        entry = {"estimate": v}
        if k in samples and samples[k]:
            arr = np.asarray(samples[k], dtype=float)
            entry.update(
                lower=float(np.nanpercentile(arr, alpha)),
                upper=float(np.nanpercentile(arr, 100 - alpha)),
                std=float(np.nanstd(arr)),
            )
        out[k] = entry
    return out


# ---------------------------------------------------------------------------
# Backwards-compatible plotting helper used in the original scripts
# ---------------------------------------------------------------------------
def PlotROCCurve(
    probs,
    y_test_roc,
    ci: float = 95,
    random_seed: int = 0,
    n_bootstraps: int = 1000,
    title: str = "Receiver Operating Characteristic",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot the ROC curve and return metrics with bootstrap CIs.

    Returns a list in the order
    [auroc, auprc, sensitivity, specificity, threshold,
     auroc_lower, auroc_upper, auroc_std,
     auprc_lower, auprc_upper, auprc_std,
     sensitivity_lower, sensitivity_upper, sensitivity_std,
     specificity_lower, specificity_upper, specificity_std]
    """
    y = np.asarray(y_test_roc).astype(int)
    p = np.asarray(probs).astype(float)
    res = bootstrap_metrics(y, p, n_bootstraps=n_bootstraps, ci=ci, random_seed=random_seed)

    fpr, tpr, _ = metrics.roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUROC = {res['auroc']['estimate']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)

    out = [
        res["auroc"]["estimate"],
        res["auprc"]["estimate"],
        res["sensitivity"]["estimate"],
        res["specificity"]["estimate"],
        res["threshold"]["estimate"],
    ]
    for k in ["auroc", "auprc", "sensitivity", "specificity"]:
        out += [res[k].get("lower", np.nan), res[k].get("upper", np.nan), res[k].get("std", np.nan)]
    return out


def plot_calibration(y_true, y_prob, n_bins: int = 10, title: str = "Calibration", save_path: Optional[str] = None):
    """Reliability diagram using quantile bins."""
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o")
    lim = max(float(np.max(mean_pred)), float(np.max(frac_pos)), 1e-3)
    ax.plot([0, lim], [0, lim], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarise_results(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """Flatten {outcome: bootstrap_metrics(...)} into one row per outcome."""
    rows = []
    for outcome, res in results.items():
        row = {"outcome": outcome}
        for k, v in res.items():
            row[k] = v["estimate"]
            if "lower" in v:
                row[f"{k}_lower"] = v["lower"]
                row[f"{k}_upper"] = v["upper"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) > 1:
        mean_row = {"outcome": "MEAN"}
        for col in ["auroc", "auprc", "auprc_relative", "sensitivity", "specificity", "f1"]:
            if col in df:
                mean_row[col] = df[col].mean()
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_list(arg: Optional[str]) -> Optional[Iterable[str]]:
    """'a,b,c' -> ['a', 'b', 'c']"""
    if arg is None:
        return None
    return [x.strip() for x in arg.split(",") if x.strip()]
