"""Single-metric threshold baseline."""
from __future__ import annotations
import numpy as np

class CPUThreshold:
    name = "CPU threshold"
    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def fit(self, rows, y=None):
        if y is None:
            y = np.array([int(r["ground_truth_fail"]) for r in rows])
        cpu = np.array([float(r.get("cpu_util") or 0.0) for r in rows])
        best = (-1.0, self.threshold)
        for threshold in np.linspace(0.01, 0.99, 99):
            pred = (cpu >= threshold).astype(int)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
            if f1 > best[0]:
                best = (f1, float(threshold))
        self.threshold = best[1]
        return self

    def predict_proba(self, rows):
        cpu = np.array([float(r.get("cpu_util") or 0.0) for r in rows])
        # A monotone score is required for ROC/PR comparison; this is not a calibrated
        # probability and is labelled as such in the comparison report.
        return cpu

    def predict(self, rows):
        return (self.predict_proba(rows) >= self.threshold).astype(int)
