"""Classical machine-learning baselines on the explicit feature matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from features import FeatureSpec

SEED = 999


class SklearnModel:
    """A scikit-learn estimator behind the explicit FeatureSpec."""

    family = "machine learning"
    explains = False
    handles_missing = "imputes (median + indicator)"

    def __init__(self, name: str, factory, standardise: bool = False, kind: str = "forest"):
        self.name = name
        self.factory = factory
        self.standardise = standardise
        self.kind = kind

    def fit(self, df: pd.DataFrame) -> "SklearnModel":
        self.spec = FeatureSpec(standardise=self.standardise).fit(df)
        self.model = self.factory()
        self.model.fit(self.spec.transform(df), df["ground_truth_fail"].to_numpy(dtype=int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.spec.transform(df))[:, 1]


def logistic(class_weight=None):
    return SklearnModel(
        "Logistic regression",
        lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight, random_state=SEED),
        standardise=True, kind="linear")


def random_forest(class_weight=None):
    return SklearnModel(
        "Random forest",
        lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features="sqrt",
                                       class_weight=class_weight, n_jobs=-1, random_state=SEED),
        kind="forest")


def gradient_boosting(class_weight=None):
    return SklearnModel(
        "Gradient boosting",
        lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                               l2_regularization=1.0, class_weight=class_weight,
                                               early_stopping=False, random_state=SEED),
        kind="gbdt")


def mlp():
    return SklearnModel(
        "Neural network (MLP)",
        lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                              learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                              n_iter_no_change=6, random_state=SEED),
        standardise=True, kind="mlp")
