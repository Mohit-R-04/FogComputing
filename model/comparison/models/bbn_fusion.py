"""BBN-driven fusion of causal posterior and continuous telemetry risk.

The causal BBN remains the explanation path. A nonlinear telemetry risk head is
used as an additional probabilistic evidence source because ordinal three-state
encoding discards continuous magnitude. Fusion is performed in log-odds space;
this is a Bayesian log-pool, not a majority vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .classical_models import CATEGORICAL, NUMERIC, matrix


class CausalBBNFusion:
    name = "BBN fusion (proposed)"

    def __init__(self, bbn, cuts, bbn_weight: float = 0.40):
        self.bbn = bbn
        self.cuts = cuts
        self.bbn_weight = bbn_weight
        numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        self.risk_head = Pipeline([
            ("features", ColumnTransformer([
                ("numeric", numeric, NUMERIC),
                ("categorical", categorical, CATEGORICAL),
            ])),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=4, random_state=999, n_jobs=-1,
            )),
        ])

    @staticmethod
    def _weights(y):
        observed = float(np.mean(y))
        w = np.where(y == 1, 0.40 / observed, 0.60 / (1.0 - observed))
        return w / w.mean()

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

    def fit(self, rows):
        y = np.array([int(r["ground_truth_fail"]) for r in rows])
        self.risk_head.fit(matrix(rows), y, model__sample_weight=self._weights(y))
        return self

    def bbn_probability(self, rows):
        import verify
        return verify.score_rows(self.bbn, rows, self.cuts)

    def risk_probability(self, rows):
        return self.risk_head.predict_proba(matrix(rows))[:, 1]

    def predict_proba(self, rows):
        causal = self.bbn_probability(rows)
        telemetry = self.risk_probability(rows)
        return self._sigmoid(self.bbn_weight * self._logit(causal)
                             + (1.0 - self.bbn_weight) * self._logit(telemetry))

    def component_probabilities(self, rows):
        return self.bbn_probability(rows), self.risk_probability(rows)
