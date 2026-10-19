"""Threshold baselines: a single metric, and an FTAPA-style composite health index."""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Composite health-index weights (Gupta and Singh's FTAPA monitors "node health"
#: against a fault threshold; this is a fixed weighted mix of normalised utilisation,
#: backlog, energy depletion and temperature).
HI_WEIGHTS = {"cpu_util": 0.40, "queue_depth": 0.30, "residual_energy": 0.15, "temperature": 0.15}
QUEUE_SATURATION = 120.0
TEMP_SAFE, TEMP_SPAN = 45.0, 40.0


def health_risk(df: pd.DataFrame) -> np.ndarray:
    """1 - HI, with missing channels dropped and the remaining weights renormalised."""
    parts = {
        "cpu_util": df["cpu_util"].to_numpy(dtype=float).clip(0, 1),
        "queue_depth": (df["queue_depth"].to_numpy(dtype=float) / QUEUE_SATURATION).clip(0, 1),
        "residual_energy": (1.0 - df["residual_energy"].to_numpy(dtype=float)).clip(0, 1),
        "temperature": ((df["temperature"].to_numpy(dtype=float) - TEMP_SAFE) / TEMP_SPAN).clip(0, 1),
    }
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for k, w in HI_WEIGHTS.items():
        ok = ~np.isnan(parts[k])
        num[ok] += w * parts[k][ok]
        den[ok] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.where(den > 0, num / den, 0.0)
    return risk


class HealthIndexThreshold:
    """FTAPA-style: migrate when the composite health index crosses a tuned threshold.

    Its score is a ranking, not a calibrated probability, so calibration metrics are not
    reported for it. When every contributing channel is missing it cannot fire at all.
    """

    name = "Health-index threshold (FTAPA)"
    family = "threshold rule"
    explains = False
    handles_missing = "drops the channel; silent if all are missing"
    calibrated = False

    def fit(self, df: pd.DataFrame) -> "HealthIndexThreshold":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return health_risk(df)


class CpuThreshold:
    """Single-metric rule on CPU utilisation (Rajab and Younis trigger on CPU)."""

    name = "CPU threshold"
    family = "threshold rule"
    explains = False
    handles_missing = "silent when CPU is missing"
    calibrated = False

    def fit(self, df: pd.DataFrame) -> "CpuThreshold":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(df["cpu_util"].to_numpy(dtype=float), nan=0.0)
