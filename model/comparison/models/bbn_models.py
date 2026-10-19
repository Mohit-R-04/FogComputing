"""Bayesian-network models: the proposed BBN, the zero-data BBN, and Naive Bayes."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bbn_model
import train_bbn
from bbn_fast import CompiledBBN
from discretise import RELIABILITY_STATE_BY_CLASS, build_cuts, state_index


class ProposedBBN:
    """The paper's causal BBN as selected by internal validation and trained with MAP-EM."""

    name = "BBN (proposed)"
    family = "Bayesian network"
    explains = True
    handles_missing = "marginalises"

    def __init__(self, config: dict):
        self.config = config

    def fit(self, df: pd.DataFrame) -> "ProposedBBN":
        self.cuts = train_bbn.cuts_for(df, self.config)
        self.structure = train_bbn.structure_for(self.config, self.cuts)
        self.net, self.history, self.meta = train_bbn.fit(
            df, self.structure, self.cuts, self.config["concentration"],
            max_iter=int(self.config["iterations"]), tol=1e-12, lam=float(self.config.get("lam", 0.05)))
        self.compiled = CompiledBBN(self.net)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.compiled.predict_frame(df, self.cuts)


class ElicitedBBN:
    """The paper's Fig. 2 with expert-elicited tables and tertile cuts: no failure data at all.

    This is where the framework starts on day one of a deployment, before a single
    failure has been observed -- the case the Review documents argue a classifier cannot
    handle. Only unlabelled telemetry is used, to place the tertile cut points.
    """

    name = "BBN (elicited, no failure data)"
    family = "Bayesian network"
    explains = True
    handles_missing = "marginalises"

    def fit(self, df: pd.DataFrame) -> "ElicitedBBN":
        st = bbn_model.PAPER
        self.cuts = build_cuts({v: df[v].to_numpy(dtype=float) for v in st.evidence}, None, "tertile")
        self.compiled = CompiledBBN(bbn_model.build_network(structure=st))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.compiled.predict_frame(df, self.cuts)


class NaiveBayes:
    """Naive Bayes over the same discretised evidence as the BBN: the simplest Bayesian
    network.

    Every channel is conditionally independent given the outcome (within each device
    class), so there is no latent tier and no causal structure. It receives exactly the
    BBN's per-class supervised bins, and like the BBN it skips a channel that did not
    report (its likelihood term is simply left out), so the comparison isolates the value
    of the paper's structure rather than of the discretisation or of probabilistic
    handling of missing data.
    """

    name = "Naive Bayes"
    family = "Bayesian network"
    explains = False
    handles_missing = "marginalises"

    def __init__(self, config: dict | None = None):
        self.config = dict(config or train_bbn.load_config())
        self.vars = bbn_model.make_structure(trends=True).evidence

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        classes = df["device_class"].to_numpy()
        cols = []
        for v in self.vars:
            values = df[v].to_numpy(dtype=float)
            col = np.full(len(df), -1, dtype=np.int8)
            for cls_ in np.unique(classes):
                m = classes == cls_
                col[m] = state_index(values[m], self.cuts.cuts_for(v, cls_))
            cols.append(col)
        cls = df["device_class"].map({c: i for i, c in enumerate(RELIABILITY_STATE_BY_CLASS)})
        cols.append(cls.fillna(-1).to_numpy(dtype=np.int8))
        return np.stack(cols, axis=1)

    def fit(self, df: pd.DataFrame) -> "NaiveBayes":
        y = df["ground_truth_fail"].to_numpy(dtype=int)
        self.cuts = train_bbn.cuts_for(df, {**self.config, "trends": True})
        X = self._encode(df)
        cls = X[:, -1]
        # Outcome prior and per-channel likelihoods within each device class.
        self.prior = np.ones((3, 2))
        np.add.at(self.prior, (np.maximum(cls, 0), y), 1.0)
        self.prior /= self.prior.sum(axis=1, keepdims=True)
        self.lik = []
        for j in range(X.shape[1] - 1):
            k = int(X[:, j].max()) + 1 if (X[:, j] >= 0).any() else 1
            table = np.ones((3, 2, max(k, 1)))
            obs = (X[:, j] >= 0) & (cls >= 0)
            np.add.at(table, (cls[obs], y[obs], X[obs, j].astype(int)), 1.0)
            self.lik.append(table / table.sum(axis=2, keepdims=True))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._encode(df)
        cls = np.maximum(X[:, -1], 0)
        logp = np.log(self.prior[cls])
        for j, table in enumerate(self.lik):
            k = table.shape[2]
            obs = (X[:, j] >= 0) & (X[:, j] < k)
            logp[obs] += np.log(table[cls[obs], :, X[obs, j].astype(int)])
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p[:, 1] / p.sum(axis=1)
