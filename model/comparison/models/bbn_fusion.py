"""Hybrid ablation: Bayesian log-pool of the BBN posterior and a random forest.

This was the "proposed" model of the earlier phase (30% BBN / 70% random forest). It is
kept only as a reported ablation: it shows how much a black-box component adds on top
of the BBN, and it is not the proposed method. Its weight is chosen on out-of-fold
predictions of the training scenarios.
"""
from __future__ import annotations

import numpy as np


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def pool(p_bbn, p_rf, bbn_weight: float):
    return sigmoid(bbn_weight * logit(p_bbn) + (1.0 - bbn_weight) * logit(p_rf))
