"""
The predictors inside iFogSim must compute exactly what the Python models compute.

    pytest test_java_parity.py -q

The closed-loop experiment runs the BBN (``org.fog.ft.BbnEngine``) and the best
classical model (``org.fog.ft.MlPredictor``) in Java. This test scores the same 2,000
held-out rows in both languages and compares: the BBN posterior to 1e-9, the ML
probability to 1e-6, and the BBN's named cause on a subset against the reference
variable-elimination engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import design
import javarun
from bbn_fast import CompiledBBN
from bbn_io import load_model
from predict import explain

SAMPLE = design.MODELS / "parity_sample.csv"
BBN = design.DATA / "bbn_trained.json"
ML = design.MODELS / "ml_best.json"


def _prerequisites():
    missing = [p for p in (SAMPLE, BBN, ML) if not p.exists()]
    if missing:
        pytest.skip(f"run the pipeline first (missing {', '.join(p.name for p in missing)})")
    javarun.ensure_built()


def _java_scores(tmp_path) -> pd.DataFrame:
    out = tmp_path / "java_scores.csv"
    res = javarun.java("ParityCli", [javarun.relpath_from_ifogsim(BBN), javarun.relpath_from_ifogsim(ML),
                                     javarun.relpath_from_ifogsim(SAMPLE), javarun.relpath_from_ifogsim(out)])
    assert res.returncode == 0, res.stdout
    return pd.read_csv(out)


def test_java_bbn_matches_python(tmp_path):
    _prerequisites()
    sample = pd.read_csv(SAMPLE)
    java = _java_scores(tmp_path)
    net, cuts, _ = load_model(BBN)
    p_python = CompiledBBN(net).predict_frame(sample, cuts)
    assert np.max(np.abs(p_python - java.p_bbn_java.to_numpy())) < 1e-9
    # The comparison suite's BBN and train_bbn.py's BBN are the same deterministic fit.
    assert np.max(np.abs(p_python - sample.p_bbn.to_numpy())) < 1e-9


def test_java_ml_matches_scikit_learn(tmp_path):
    _prerequisites()
    sample = pd.read_csv(SAMPLE)
    java = _java_scores(tmp_path)
    assert np.max(np.abs(sample.p_ml.to_numpy() - java.p_ml_java.to_numpy())) < 1e-6


def test_java_names_the_same_cause_as_python(tmp_path):
    """Same principal cause in Java as in the Python engine (predict.explain), which
    test_pipeline checks against variable elimination."""
    _prerequisites()
    sample = pd.read_csv(SAMPLE)
    java = _java_scores(tmp_path)
    net, cuts, _ = load_model(BBN)
    comp = CompiledBBN(net)
    E, P = comp.encode(sample, cuts)
    cause, _ = explain(comp, E, P)
    assert (java.cause_java.astype(str).to_numpy() == cause).all()
