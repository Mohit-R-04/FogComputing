"""Classical scikit-learn baselines for raw iFogSim telemetry."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "cpu_util", "ram_util", "queue_depth", "uplink_latency",
    "jitter_pktloss", "handover_rate", "residual_energy",
]
CATEGORICAL = ["device_class"]


def matrix(rows):
    return pd.DataFrame([
        {k: (None if row.get(k, "") == "" else row.get(k)) for k in NUMERIC + CATEGORICAL}
        for row in rows
    ])


def preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])


def models():
    return {
        "Logistic regression": Pipeline([
            ("features", preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=999)),
        ]),
        "Random forest": Pipeline([
            ("features", preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=4,
                                               class_weight=None,
                                               random_state=999, n_jobs=-1)),
        ]),
        "Histogram gradient boosting": Pipeline([
            ("features", preprocessor()),
            ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                                       max_leaf_nodes=15, l2_regularization=1.0,
                                                       random_state=999)),
        ]),
    }
