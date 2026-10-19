"""
Metric helpers shared by the verification report and the model comparison.

Snapshots of one node are strongly autocorrelated, so confidence intervals resample
whole nodes (all snapshots of a node in a run go in or out together). Calibration uses
equal-mass bins, because with a 0.3% failure rate fixed-width bins put almost every
prediction in the first bin and say nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import design

BOOTSTRAP = 200


def f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 (candidates: distinct score quantiles)."""
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 400)))
    best_t, best_f1 = float(cand[-1]), 0.0
    pos = y.sum()
    for t in cand:
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + (pos - tp)) if tp > 0 else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over equal-mass bins (fixed-width bins put almost every
    prediction of a rare event in the first bin and say nothing)."""
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(total)


def classification(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(y)}


def group_index(df: pd.DataFrame) -> list[np.ndarray]:
    keys = df.groupby(["scenario_id", "seed", "node_name"]).indices
    return list(keys.values())


def bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: list[np.ndarray], metric,
                 n: int = BOOTSTRAP, seed: int = 11) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric(y[idx], p[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def early_warning(df: pd.DataFrame, p: np.ndarray, t: float) -> dict:
    """Inside each failing node's last horizon: was it flagged, and how early?"""
    alarms = p >= t
    df = df.assign(_p=p, _alarm=alarms)
    failing = df.dropna(subset=["failure_time"])
    detected, leads, total = 0, [], 0
    for _, g in failing.groupby(["scenario_id", "seed", "node_name"]):
        ft = float(g.failure_time.iloc[0])
        window = g[(g.epoch > ft - design.HORIZON) & (g.epoch < ft)]
        if window.empty:
            continue
        total += 1
        hit = window[window._alarm]
        if len(hit):
            detected += 1
            leads.append(ft - float(hit.epoch.min()))
    negatives = (df.ground_truth_fail == 0)
    false_alarms = int((df._alarm & negatives).sum())
    return {"failing_nodes": total, "detected": detected,
            "detection_rate": detected / total if total else 0.0,
            "mean_lead": float(np.mean(leads)) if leads else float("nan"),
            "false_alarms_per_1000": 1000.0 * false_alarms / max(1, int(negatives.sum()))}


def metric_block(y, p, calibrated: bool) -> dict:
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)}
    if calibrated:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out.update({"log_loss": log_loss(y, pc), "brier": brier_score_loss(y, pc), "ece": ece(y, pc)})
    return out


