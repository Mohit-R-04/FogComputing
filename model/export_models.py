"""
Exports fitted comparison models so the Java controller can run them inside iFogSim.

The closed-loop experiment drives the same fault-tolerance controller with different
predictors. For that, the best classical model is written as JSON that
``org.fog.ft.MlPredictor`` evaluates exactly (``test_java_parity.py`` checks Java against
scikit-learn), and the FTAPA health-index threshold is written for
``org.fog.ft.HealthIndexPredictor``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features import FeatureSpec


def _tree_arrays(tree, leaf_values: np.ndarray) -> dict:
    left = tree.children_left.astype(int)
    right = tree.children_right.astype(int)
    return {
        "feature": [int(f) if l >= 0 else 0 for f, l in zip(tree.feature, left)],
        "threshold": [float(t) if l >= 0 else 0.0 for t, l in zip(tree.threshold, left)],
        "left": [int(v) for v in left],
        "right": [int(v) for v in right],
        "value": [float(v) for v in leaf_values],
    }


def export_sklearn(name: str, kind: str, model, spec: FeatureSpec, path: str | Path) -> None:
    doc = {"format": "fog-ml/1", "name": name, "kind": kind, "features": spec.to_dict()}
    if kind == "forest":
        trees = []
        for est in model.estimators_:
            t = est.tree_
            v = t.value[:, 0, :]
            p1 = v[:, 1] / np.maximum(v.sum(axis=1), 1e-12)
            trees.append(_tree_arrays(t, p1))
        doc["trees"] = trees
    elif kind == "gbdt":
        trees = []
        for predictors in model._predictors:
            nodes = predictors[0].nodes
            left = np.where(nodes["is_leaf"], -1, nodes["left"].astype(np.int64))
            right = np.where(nodes["is_leaf"], -1, nodes["right"].astype(np.int64))
            trees.append({
                "feature": [int(f) for f in nodes["feature_idx"]],
                "threshold": [float(t) for t in nodes["num_threshold"]],
                "left": [int(v) for v in left],
                "right": [int(v) for v in right],
                "value": [float(v) for v in nodes["value"]],
            })
        doc["trees"] = trees
        doc["baseline"] = float(np.ravel(model._baseline_prediction)[0])
    elif kind == "linear":
        doc["coef"] = [float(c) for c in model.coef_[0]]
        doc["intercept"] = float(model.intercept_[0])
    elif kind == "mlp":
        doc["layers"] = [{"weights": w.T.tolist(), "bias": b.tolist()}
                         for w, b in zip(model.coefs_, model.intercepts_)]
        doc["activation"] = model.activation
    else:
        raise ValueError(f"cannot export model kind {kind!r}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc))


def export_ftapa(threshold_on_risk: float, path: str | Path, weights: dict, norm: dict) -> None:
    """FTAPA migrates when HI < theta; the comparison tunes a threshold on risk = 1 - HI."""
    doc = {"format": "fog-ftapa/1", "weights": weights, "norm": norm,
           "threshold": float(1.0 - threshold_on_risk),
           "note": "migrate when health index < threshold; threshold chosen by max F1 on "
                   "out-of-fold predictions of the training scenarios"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1))
