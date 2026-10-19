"""
The feature matrix every classical model sees, made explicit.

All models in the comparison receive the same information as the BBN: the thirteen
telemetry channels (eight measurements, five agent-side trends) and the device class.
They never see identifiers, time, the scenario, the seed or ground truth (``design``).

A classical model needs a value in every slot, so a channel that did not report is
**imputed** with its training median and flagged with a missing-value indicator -- the
standard, strongest generic treatment, and exactly the behaviour the BBN's
marginalisation is compared against. The transformation is written out by hand rather
than hidden in a scikit-learn pipeline so that ``org.fog.ft.MlPredictor`` can reproduce
it exactly inside the simulator (``export_models.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUMERIC = (
    "cpu_util", "ram_util", "queue_depth", "uplink_latency", "jitter_pktloss",
    "handover_rate", "residual_energy", "temperature",
    "cpu_trend", "queue_trend", "latency_trend", "temp_trend", "energy_drain",
)
CLASSES = ("cloud", "fog", "edge")


@dataclass
class FeatureSpec:
    numeric: tuple[str, ...] = NUMERIC
    classes: tuple[str, ...] = CLASSES
    medians: np.ndarray | None = None
    scale_mean: np.ndarray | None = None
    scale_std: np.ndarray | None = None
    standardise: bool = False
    names: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeatureSpec":
        raw = df[list(self.numeric)].to_numpy(dtype=float)
        med = np.nanmedian(raw, axis=0)
        self.medians = np.where(np.isnan(med), 0.0, med)
        self.names = (list(self.numeric) + [f"{c}_missing" for c in self.numeric]
                      + [f"class_{c}" for c in self.classes])
        if self.standardise:
            x = self._unscaled(df)
            self.scale_mean = x.mean(axis=0)
            std = x.std(axis=0)
            self.scale_std = np.where(std == 0, 1.0, std)
        return self

    def _unscaled(self, df: pd.DataFrame) -> np.ndarray:
        raw = df[list(self.numeric)].to_numpy(dtype=float)
        missing = np.isnan(raw)
        imputed = np.where(missing, self.medians, raw)
        onehot = np.stack([(df["device_class"] == c).to_numpy(dtype=float) for c in self.classes], axis=1)
        return np.concatenate([imputed, missing.astype(float), onehot], axis=1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._unscaled(df)
        if self.standardise:
            x = (x - self.scale_mean) / self.scale_std
        return x

    def to_dict(self) -> dict:
        return {
            "numeric": list(self.numeric),
            "medians": [float(v) for v in self.medians],
            "classes": list(self.classes),
            "scale_mean": None if self.scale_mean is None else [float(v) for v in self.scale_mean],
            "scale_std": None if self.scale_std is None else [float(v) for v in self.scale_std],
        }
