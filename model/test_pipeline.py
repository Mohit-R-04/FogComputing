"""
Tests of the Phase II pipeline: the vectorised engine, training, persistence,
discretisation, the comparison models, and the experimental design.

    pytest test_pipeline.py -q

``test_inference.py`` covers the paper graph and the reference inference engine; this
file covers everything built on top of them. Tests that need generated data or a built
simulator skip cleanly when those are absent.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bbn_model
import design
import inference
from bbn_fast import CompiledBBN, aggregate_patterns
from bbn_io import load_model, save_model
from discretise import QuantileCuts, build_cuts, state_index, supervised_cuts, to_state

sys.path.insert(0, str(Path(__file__).resolve().parent / "comparison"))

STRUCTURES = [bbn_model.make_structure(t, c) for t, c in itertools.product([False, True], [False, True])]


def random_network(structure, seed=0):
    rng = np.random.default_rng(seed)
    net = bbn_model.build_network(structure=structure)
    for v in structure.variables:
        for k in net.cpts[v]:
            net.cpts[v][k] = list(rng.dirichlet(np.ones(len(net.cpts[v][k]))))
    return net


# ---- vectorised engine ---------------------------------------------------------------------

@pytest.mark.parametrize("structure", STRUCTURES, ids=lambda s: s.name)
def test_fast_engine_matches_variable_elimination(structure):
    net = random_network(structure, seed=hash(structure.name) % 1000)
    comp = CompiledBBN(net)
    rng = np.random.default_rng(1)
    E = rng.integers(-1, 3, size=(80, len(structure.evidence))).astype(np.int8)
    P = rng.integers(-1, 3, size=80)
    fast = comp.posterior(E, P)
    for i in range(80):
        ev = {structure.evidence[j]: structure.states(structure.evidence[j])[E[i, j]]
              for j in range(len(structure.evidence)) if E[i, j] >= 0}
        if P[i] >= 0:
            ev[structure.prior] = structure.states(structure.prior)[P[i]]
        assert fast[i] == pytest.approx(inference.variable_elimination(net, ev)[1], abs=1e-12)


def test_fast_explanation_matches_the_reference_engine():
    """predict.explain (vectorised, used by the reports and mirrored by the Java engine)
    and inference.infer (variable elimination) name the same cause with the same lift --
    on a network whose latent conditions depend on the device class, where the prior the
    lift is measured against must be the one for the node's own class."""
    from predict import explain
    structure = bbn_model.make_structure(trends=True, class_conditional=True, class_latent=True)
    net = random_network(structure, seed=11)
    comp = CompiledBBN(net)
    rng = np.random.default_rng(5)
    E = rng.integers(-1, 3, size=(30, len(structure.evidence))).astype(np.int8)
    P = rng.integers(0, 3, size=30)
    cause, lift = explain(comp, E, P)
    for i in range(30):
        ev = {structure.evidence[j]: structure.states(structure.evidence[j])[E[i, j]]
              for j in range(len(structure.evidence)) if E[i, j] >= 0}
        ev[structure.prior] = structure.states(structure.prior)[P[i]]
        ref = inference.infer(net, ev)
        assert cause[i] == ref.top_cause
        if ref.top_cause != "none":
            assert lift[i] == pytest.approx(ref.top_cause_lift, abs=1e-9)


def test_missing_channel_is_marginalised_not_imputed_in_bulk():
    """With the target unobserved, a missing root channel is d-separated from the other
    evidence given the device class, so the posterior must be the mixture
    sum_s P(channel = s | class) * P(fail | e, channel = s) -- and differ from plugging in
    any single state."""
    structure = bbn_model.make_structure(True, True)
    net = random_network(structure, seed=3)
    comp = CompiledBBN(net)
    E = np.ones((1, len(structure.evidence)), dtype=np.int8)
    E[0, 0] = -1
    P = np.array([1])  # reliability prior "medium"
    marginal = comp.posterior(E, P)[0]
    weights = np.array(net.cpts[structure.evidence[0]][("medium",)])
    plugged = []
    for s in range(3):
        E2 = E.copy()
        E2[0, 0] = s
        plugged.append(comp.posterior(E2, P)[0])
    assert marginal == pytest.approx(float(np.dot(weights, plugged)), abs=1e-12)
    assert all(abs(marginal - v) > 1e-6 for v in plugged)


def test_pattern_aggregation_preserves_counts():
    rng = np.random.default_rng(4)
    E = rng.integers(-1, 3, size=(500, 4)).astype(np.int8)
    P = rng.integers(0, 3, size=500)
    Y = rng.integers(0, 2, size=500)
    Eu, Pu, Yu, Wu = aggregate_patterns(E, P, Y)
    assert Wu.sum() == 500
    assert len(Eu) <= 500


# ---- training ----------------------------------------------------------------------------

def synthetic_frame(n=6000, seed=5):
    rng = np.random.default_rng(seed)
    stress = rng.random(n)
    df = pd.DataFrame({
        "device_class": rng.choice(["fog", "edge"], n),
        "cpu_util": np.clip(stress + rng.normal(0, 0.1, n), 0, 1),
        "ram_util": rng.random(n),
        "queue_depth": 100 * np.clip(stress + rng.normal(0, 0.2, n), 0, 1),
        "uplink_latency": 50 + 20 * rng.random(n),
        "jitter_pktloss": rng.random(n),
        "handover_rate": rng.integers(0, 5, n).astype(float),
        "residual_energy": rng.random(n),
        "temperature": 30 + 40 * stress,
    })
    df["ground_truth_fail"] = (rng.random(n) < 0.02 + 0.3 * (stress > 0.85)).astype(int)
    return df


def test_em_improves_the_likelihood_and_keeps_valid_tables():
    import train_bbn

    df = synthetic_frame()
    st = bbn_model.PAPER
    cuts = train_bbn.cuts_for(df, st, "supervised")
    net, history, meta = train_bbn.fit(df, st, cuts, concentration=5.0, max_iter=25, tol=0)
    net.validate()
    assert history[-1] > history[0]
    p = CompiledBBN(net).predict_frame(df, cuts)
    stressed = df.cpu_util > 0.9
    assert p[stressed].mean() > 2 * p[~stressed].mean()


def test_model_round_trip(tmp_path):
    df = synthetic_frame(2000)
    st = bbn_model.make_structure(True, False)
    df = df.assign(cpu_trend=0.0, queue_trend=0.0, latency_trend=0.0, temp_trend=0.0, energy_drain=0.01)
    cuts = build_cuts({v: df[v].to_numpy(dtype=float) for v in st.evidence},
                      df.ground_truth_fail.to_numpy(), "supervised")
    net = random_network(st, seed=9)
    path = tmp_path / "m.json"
    save_model(net, cuts, path, {"note": "test"})
    net2, cuts2, doc = load_model(path)
    assert doc["structure"] == st.name
    for v in st.variables:
        for k, row in net.cpts[v].items():
            assert net2.cpts[v][k] == pytest.approx(row)
    assert cuts2.global_cuts == pytest.approx(cuts.global_cuts) or cuts2.global_cuts == cuts.global_cuts
    a = CompiledBBN(net).predict_frame(df, cuts)
    b = CompiledBBN(net2).predict_frame(df, cuts2)
    assert np.allclose(a, b)


# ---- discretisation ----------------------------------------------------------------------

def test_vectorised_states_match_scalar_rule():
    rng = np.random.default_rng(6)
    values = rng.random(300)
    values[::17] = np.nan
    for cuts in [(0.3, 0.6), (0.5, 0.5), (0.0, 0.0)]:
        vec = state_index(values, cuts)
        for v, s in zip(values, vec):
            if np.isnan(v):
                assert s == -1
            else:
                assert ("low", "medium", "high")[s] == to_state(v, cuts)


def test_supervised_cuts_find_the_risky_tail():
    """Risk jumps at 0.9: whatever the number of bins, one cut point must land on the jump."""
    rng = np.random.default_rng(7)
    v = rng.random(20000)
    y = (rng.random(20000) < np.where(v > 0.9, 0.5, 0.01)).astype(int)
    for k in (2, 3, 5, 8):
        cuts = supervised_cuts(v, y, k)
        assert len(cuts) <= k - 1
        assert any(0.88 <= c <= 0.92 for c in cuts), (k, cuts)


def test_supervised_partition_maximises_mutual_information():
    """The dynamic programme must match brute force over every 2-cut partition."""
    from discretise import _candidates

    rng = np.random.default_rng(8)
    v = rng.gamma(2.0, 1.0, 3000)
    y = (rng.random(3000) < 0.02 + 0.2 * (v > 4)).astype(int)
    best = supervised_cuts(v, y, 3, min_fraction=0.0)

    def mi(cuts):
        s = np.searchsorted(np.asarray(cuts), v, side="right")
        t = np.zeros((len(cuts) + 1, 2))
        np.add.at(t, (s, y), 1.0)
        t += 0.5
        j = t / len(v)
        pb = (t.sum(1, keepdims=True)) / len(v)
        py = np.array([1 - y.mean(), y.mean()])[None]
        return float((j * np.log(j / (pb * py))).sum())

    cand = _candidates(v, y)
    brute = max(mi((a, b)) for i, a in enumerate(cand) for b in cand[i + 1:])
    assert mi(best) >= brute - 1e-9


def test_cut_points_round_trip(tmp_path):
    cuts = QuantileCuts(global_cuts={"cpu_util": (0.2, 0.9)}, method="supervised")
    cuts.save(tmp_path / "c.json")
    back = QuantileCuts.load(tmp_path / "c.json")
    assert back.global_cuts == cuts.global_cuts and back.method == "supervised"


# ---- comparison models ---------------------------------------------------------------------

def test_naive_bayes_skips_a_missing_channel():
    from models.bbn_models import NaiveBayes

    df = synthetic_frame(3000)
    df = df.assign(cpu_trend=0.0, queue_trend=0.0, latency_trend=0.0, temp_trend=0.0, energy_drain=0.0)
    nb = NaiveBayes().fit(df)
    one = df.head(5).copy()
    one["temperature"] = np.nan
    p = nb.predict_proba(one)
    assert np.all((p > 0) & (p < 1))


def test_feature_spec_imputes_and_flags():
    from features import FeatureSpec

    df = synthetic_frame(500)
    for c in ("cpu_trend", "queue_trend", "latency_trend", "temp_trend", "energy_drain"):
        df[c] = 0.0
    df.loc[0, "temperature"] = np.nan
    spec = FeatureSpec().fit(df)
    x = spec.transform(df)
    t = spec.numeric.index("temperature")
    assert x[0, t] == pytest.approx(spec.medians[t])
    assert x[0, len(spec.numeric) + t] == 1.0
    assert x.shape[1] == 2 * len(spec.numeric) + 3


# ---- experimental design and data ------------------------------------------------------------

def test_design_holds_out_scenarios_and_seeds():
    assert not set(design.TRAIN_SCENARIOS) & set(design.TEST_SCENARIOS)
    assert not set(design.TRAIN_SEEDS) & set(design.TEST_SEEDS)
    for col in ("ground_truth_fail", "failure_time", "true_health", "node_name", "seed"):
        assert col in design.NON_FEATURES


def test_held_out_failures_do_not_repeat_training_draws():
    manifest = design.DATA / "dataset_manifest.json"
    if not manifest.exists():
        pytest.skip("dataset not built")
    audit = json.loads(manifest.read_text())["leak_audit"]
    assert audit["same_epoch_share"] < 0.10, audit


def test_no_model_beats_the_hidden_health_oracle_on_fog_nodes():
    """A leak detector. A fog node fails either from wear, which its hidden health tracks
    (ranking by 1 - health orders wear-out failures almost perfectly), or suddenly, which
    has no precursor; so no telemetry model may rank fog failures clearly better than
    knowing the hidden health. Edge devices are left out: their batteries run flat, which
    the observed residual energy foresees and hidden health does not."""
    results = Path(__file__).resolve().parent / "comparison" / "results"
    test_pool = design.DATA / "telemetry_test_pool.csv"
    if not (results / "test_predictions.npz").exists() or not test_pool.exists():
        pytest.skip("comparison not run")
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(test_pool, usecols=["device_class", "true_health", "ground_truth_fail"])
    df = df[df.device_class != "cloud"].reset_index(drop=True)
    z = np.load(results / "test_predictions.npz")
    assert np.array_equal(z["y"], df.ground_truth_fail.to_numpy()), "predictions not aligned with the pool"
    fog = (df.device_class == "fog").to_numpy()
    y = df.ground_truth_fail.to_numpy()[fog]
    oracle = roc_auc_score(y, 1 - df.true_health.to_numpy()[fog])
    for i, name in enumerate(z["names"]):
        auc = roc_auc_score(y, z[f"p{i}"][fog])
        assert auc <= oracle + 0.01, (str(name), auc, oracle)