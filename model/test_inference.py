"""
Correctness tests for the inference engine and the network it runs on.

The engine is the part of this work that has to be right before any number it
produces means anything, so it is checked three ways: against a direct
transcription of the published equation, against hand-computable cases, and
against the probability axioms.

    pytest test_inference.py
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import bbn_model
import inference
from bbn_model import (
    ALL_VARS,
    EVIDENCE_VARS,
    LATENT_VARS,
    PARENTS,
    PRIOR_VAR,
    TARGET_VAR,
    BayesianNetwork,
    states_of,
    topological_order,
)
from cpt_tables import STATE_POINTS, truncated_normal_row
from discretise import QuantileCuts, to_state


@pytest.fixture(scope="module")
def net():
    return bbn_model.build_network()


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_network_has_fourteen_variables():
    """The paper counts fourteen: eight evidence, four latent, the prior, the target."""
    assert len(ALL_VARS) == 14
    assert len(EVIDENCE_VARS) == 8
    assert len(LATENT_VARS) == 4


def test_graph_is_acyclic():
    order = topological_order()
    position = {v: i for i, v in enumerate(order)}
    for var in ALL_VARS:
        for parent in PARENTS[var]:
            assert position[parent] < position[var], f"{parent}->{var} breaks the ordering"


def test_edges_match_the_paper():
    assert PARENTS["computational_stress"] == ("cpu_util", "ram_util", "queue_depth")
    assert PARENTS["network_degradation"] == ("uplink_latency", "jitter_pktloss", "handover_rate")
    assert PARENTS["energy_stress"] == ("residual_energy",)
    assert PARENTS["thermal_stress"] == ("temperature",)
    assert set(PARENTS[TARGET_VAR]) == set(LATENT_VARS) | {PRIOR_VAR}


def test_cpts_are_well_formed(net):
    net.validate()
    assert len(net.cpt(TARGET_VAR)) == 3 ** 5          # 243 parent configurations
    assert len(net.cpt("computational_stress")) == 3 ** 3


# --------------------------------------------------------------------------
# the engine against the published equation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("evidence", [
    {},
    {"cpu_util": "high"},
    {"cpu_util": "high", "ram_util": "high", "reliability_prior": "low"},
    {"cpu_util": "low", "ram_util": "low", "queue_depth": "low",
     "uplink_latency": "low", "jitter_pktloss": "low", "handover_rate": "low",
     "residual_energy": "high", "reliability_prior": "high"},
    {"temperature": "high", "cpu_util": "medium"},
])
def test_elimination_matches_enumeration(net, evidence):
    """Variable elimination must agree with a literal reading of Eq. (1)."""
    fast = inference.variable_elimination(net, evidence)
    slow = inference.enumerate_posterior(net, evidence)
    assert np.allclose(fast, slow, atol=1e-12), f"{fast} != {slow}"


def test_posteriors_sum_to_one(net):
    for evidence in [{}, {"cpu_util": "high"}, {"residual_energy": "low"}]:
        for var in ALL_VARS:
            if var in evidence:
                continue
            posterior = inference.variable_elimination(net, evidence, var)
            assert abs(posterior.sum() - 1.0) < 1e-12
            assert (posterior >= 0).all()


def test_marginalising_an_unobserved_variable_changes_nothing(net):
    """Observing nothing about temperature is not the same as observing a value.

    The posterior with temperature absent must equal the average of the posteriors
    over its three states, weighted by its prior. That identity is what
    "marginalised out rather than imputed" means, and it is the property the paper
    rests its partial-evidence claim on.
    """
    base = {"cpu_util": "high", "ram_util": "medium"}
    absent = inference.variable_elimination(net, base)

    prior = net.cpt("temperature")[()]
    mixture = np.zeros(2)
    for p, state in zip(prior, states_of("temperature")):
        mixture += p * inference.variable_elimination(net, {**base, "temperature": state})

    assert np.allclose(absent, mixture, atol=1e-12)


def test_imputing_a_missing_value_is_not_equivalent(net):
    """Filling a gap with a middle value gives a different answer, as it should.

    Kept as a test because it is the mistake the discretiser must never make: the
    two are not interchangeable, so a default value is a fabricated observation.
    """
    base = {"cpu_util": "high", "ram_util": "high"}
    marginalised = inference.variable_elimination(net, base)
    imputed = inference.variable_elimination(net, {**base, "temperature": "medium"})
    assert not np.allclose(marginalised, imputed, atol=1e-6)


# --------------------------------------------------------------------------
# hand-computable cases
# --------------------------------------------------------------------------

def test_three_node_chain_against_pen_and_paper():
    """A chain small enough to solve by hand, run through the real engine.

    Uses the production Factor and elimination code on a stand-in network, so a
    mistake in the machinery shows up against arithmetic anyone can check.

        P(A=a1) = 0.3
        P(B=b1|a1) = 0.8,  P(B=b1|a2) = 0.1
        P(C=c1|b1) = 0.9,  P(C=c1|b2) = 0.2

        P(C=c1) = P(a1)[P(b1|a1)P(c1|b1) + P(b2|a1)P(c1|b2)]
                + P(a2)[P(b1|a2)P(c1|b1) + P(b2|a2)P(c1|b2)]
                = 0.3(0.8*0.9 + 0.2*0.2) + 0.7(0.1*0.9 + 0.9*0.2)
                = 0.3(0.76) + 0.7(0.27) = 0.228 + 0.189 = 0.417
    """
    f_a = inference.Factor(("A",), np.array([0.3, 0.7]))
    f_b = inference.Factor(("A", "B"), np.array([[0.8, 0.2], [0.1, 0.9]]))
    f_c = inference.Factor(("B", "C"), np.array([[0.9, 0.1], [0.2, 0.8]]))

    joint = f_a.multiply(f_b).multiply(f_c)
    p_c = joint.sum_out("A").sum_out("B")
    assert abs(float(p_c.values[0]) - 0.417) < 1e-12


def test_factor_multiply_is_commutative():
    f = inference.Factor(("A",), np.array([0.3, 0.7]))
    g = inference.Factor(("A", "B"), np.array([[0.8, 0.2], [0.1, 0.9]]))
    left = f.multiply(g)
    right = g.multiply(f)
    # Same scope, possibly different axis order; compare on a common ordering.
    assert set(left.variables) == set(right.variables)
    perm = [right.variables.index(v) for v in left.variables]
    assert np.allclose(left.values, np.transpose(right.values, perm))


# --------------------------------------------------------------------------
# monotonicity: the model must move in the right direction
# --------------------------------------------------------------------------

def test_more_stress_never_lowers_failure_probability(net):
    """Raising any single stress indicator must not reduce P(fail).

    Guards the polarity of the inverted variables. Wiring residual energy or the
    reliability prior the wrong way round produces a model that is internally
    consistent, passes every structural check, and is exactly backwards.
    """
    base = {v: "low" for v in EVIDENCE_VARS if v != "temperature"}
    base["residual_energy"] = "high"      # full battery
    base[PRIOR_VAR] = "high"              # dependable hardware

    p_base = inference.variable_elimination(net, base)[1]

    for var in ("cpu_util", "ram_util", "queue_depth",
                "uplink_latency", "jitter_pktloss", "handover_rate"):
        worse = dict(base)
        worse[var] = "high"
        assert inference.variable_elimination(net, worse)[1] >= p_base - 1e-12, var

    drained = dict(base, residual_energy="low")
    assert inference.variable_elimination(net, drained)[1] > p_base

    fragile = dict(base)
    fragile[PRIOR_VAR] = "low"
    assert inference.variable_elimination(net, fragile)[1] > p_base


def test_healthy_node_is_safer_than_stressed_node(net):
    healthy = {v: "low" for v in EVIDENCE_VARS if v != "temperature"}
    healthy["residual_energy"] = "high"
    healthy[PRIOR_VAR] = "high"

    stressed = {v: "high" for v in EVIDENCE_VARS if v != "temperature"}
    stressed["residual_energy"] = "low"
    stressed[PRIOR_VAR] = "low"

    assert (inference.variable_elimination(net, stressed)[1]
            > 3 * inference.variable_elimination(net, healthy)[1])


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------

def test_unobserved_condition_is_never_named_as_the_cause(net):
    """Thermal stress has no sensor, so it can never be the principal cause.

    Ranking causes by raw posterior did exactly this: on a healthy node the flat
    one-third prior on thermal stress outranked the other three conditions, which
    the evidence had pushed down, and the model blamed the temperature of a node
    with nothing wrong with it.
    """
    healthy = {v: "low" for v in EVIDENCE_VARS if v != "temperature"}
    healthy["residual_energy"] = "high"
    healthy[PRIOR_VAR] = "high"

    result = inference.infer(net, healthy)
    assert result.top_cause != "thermal_stress"
    assert result.top_cause == "none"
    assert "temperature" in result.marginalised


def test_cause_attribution_follows_the_evidence(net):
    base = {v: "low" for v in EVIDENCE_VARS if v != "temperature"}
    base["residual_energy"] = "high"
    base[PRIOR_VAR] = "high"

    compute_bound = dict(base, cpu_util="high", ram_util="high", queue_depth="high")
    assert inference.infer(net, compute_bound).top_cause == "computational_stress"

    network_bound = dict(base, uplink_latency="high", jitter_pktloss="high",
                         handover_rate="high")
    assert inference.infer(net, network_bound).top_cause == "network_degradation"

    battery_bound = dict(base, residual_energy="low")
    assert inference.infer(net, battery_bound).top_cause == "energy_stress"


def test_marginalised_lists_only_missing_evidence(net):
    result = inference.infer(net, {"cpu_util": "high"})
    assert set(result.marginalised) <= set(EVIDENCE_VARS)
    assert "cpu_util" not in result.marginalised
    # Latent conditions are unobserved by design; they are not "missing telemetry".
    for latent in LATENT_VARS:
        assert latent not in result.marginalised


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def test_rejects_unknown_variable(net):
    with pytest.raises(KeyError):
        inference.variable_elimination(net, {"disk_temperature": "high"})


def test_rejects_bad_state(net):
    with pytest.raises(ValueError):
        inference.variable_elimination(net, {"cpu_util": "critical"})


def test_rejects_evidence_on_the_target(net):
    with pytest.raises(ValueError):
        inference.variable_elimination(net, {TARGET_VAR: "fail"})


def test_malformed_cpt_is_caught(net):
    broken = BayesianNetwork(cpts={k: {c: list(r) for c, r in t.items()}
                                   for k, t in net.cpts.items()})
    broken.cpts["cpu_util"][()] = [0.5, 0.4, 0.4]     # sums to 1.3
    with pytest.raises(ValueError, match="sums to"):
        broken.validate()


# --------------------------------------------------------------------------
# CPT generation and discretisation
# --------------------------------------------------------------------------

def test_truncated_normal_rows_are_distributions():
    for mean in (0.0, 0.1, 0.5, 0.9, 1.0):
        for sigma in (0.05, 0.2, 0.5):
            row = truncated_normal_row(mean, sigma)
            assert abs(sum(row) - 1.0) < 1e-12
            assert all(p >= 0 for p in row)


def test_truncated_normal_tracks_its_mean():
    low = truncated_normal_row(STATE_POINTS["low"], 0.2)
    high = truncated_normal_row(STATE_POINTS["high"], 0.2)
    assert low[0] > low[2]
    assert high[2] > high[0]


def test_to_state_places_values_in_the_right_tertile():
    cuts = (0.4, 0.7)
    assert to_state(0.1, cuts) == "low"
    assert to_state(0.4, cuts) == "medium"
    assert to_state(0.55, cuts) == "medium"
    assert to_state(0.7, cuts) == "high"
    assert to_state(0.99, cuts) == "high"


def test_to_state_handles_a_constant_variable():
    """Handover rate on fixed infrastructure is always zero and always lowest state."""
    assert to_state(0.0, (0.0, 0.0)) == "low"
    assert to_state(1.0, (0.0, 0.0)) == "high"


def test_missing_column_is_omitted_not_imputed():
    from discretise import row_to_evidence

    cuts = QuantileCuts(
        global_cuts={v: (0.3, 0.6) for v in EVIDENCE_VARS},
        by_class={},
    )
    row = {"device_class": "fog", "cpu_util": "0.9", "ram_util": "",
           "temperature": ""}
    evidence = row_to_evidence(row, cuts)
    assert evidence["cpu_util"] == "high"
    assert "ram_util" not in evidence
    assert "temperature" not in evidence


def test_reliability_prior_comes_from_the_device_class():
    """It is a declared attribute, not a quantised measurement."""
    from discretise import row_to_evidence

    cuts = QuantileCuts(global_cuts={}, by_class={})
    assert row_to_evidence({"device_class": "edge"}, cuts)[PRIOR_VAR] == "low"
    assert row_to_evidence({"device_class": "fog"}, cuts)[PRIOR_VAR] == "medium"
    assert row_to_evidence({"device_class": "cloud"}, cuts)[PRIOR_VAR] == "high"
