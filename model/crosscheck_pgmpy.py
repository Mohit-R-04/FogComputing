#!/usr/bin/env python3
"""
Cross-checks the hand-written inference engine against pgmpy.

`test_inference.py` already checks variable elimination against a direct
transcription of Eq. (1), but both of those are this project's own code, and a
shared misreading of the semantics would pass both. This script rebuilds the same
fourteen-variable network in pgmpy -- an independent, widely used implementation --
and asserts that the posteriors agree to nine decimal places over a range of
evidence sets, including ones with channels deliberately withheld.

Skips cleanly, rather than failing, when pgmpy is not installed.

    python crosscheck_pgmpy.py
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

import bbn_model
import inference
from bbn_model import ALL_VARS, EVIDENCE_VARS, PARENTS, PRIOR_VAR, TARGET_VAR, states_of

TOLERANCE = 1e-9


def build_pgmpy_model(net):
    """Rebuilds the network as a pgmpy DiscreteBayesianNetwork.

    pgmpy stores a CPD as a matrix with one column per parent configuration, the
    columns ordered with the *last* listed parent varying fastest. `evidence` is
    given in the same order as `PARENTS`, and `itertools.product` varies its last
    element fastest, so the column order lines up directly.
    """
    try:
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.models import DiscreteBayesianNetwork
    except ImportError:
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork

    edges = [(p, v) for v in ALL_VARS for p in PARENTS[v]]
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(ALL_VARS)

    for var in ALL_VARS:
        parents = PARENTS[var]
        card = len(states_of(var))

        if parents:
            configs = list(itertools.product(*[states_of(p) for p in parents]))
            table = np.array([[net.cpt(var)[c][s] for c in configs] for s in range(card)])
            cpd = TabularCPD(
                variable=var,
                variable_card=card,
                values=table,
                evidence=list(parents),
                evidence_card=[len(states_of(p)) for p in parents],
                state_names={var: list(states_of(var)),
                             **{p: list(states_of(p)) for p in parents}},
            )
        else:
            column = np.array([[p] for p in net.cpt(var)[()]])
            cpd = TabularCPD(variable=var, variable_card=card, values=column,
                             state_names={var: list(states_of(var))})
        model.add_cpds(cpd)

    assert model.check_model()
    return model


def evidence_cases():
    """Evidence sets to compare on, from complete telemetry down to almost none."""
    full = {
        "cpu_util": "high", "ram_util": "high", "queue_depth": "medium",
        "uplink_latency": "medium", "jitter_pktloss": "high", "handover_rate": "low",
        "residual_energy": "low", PRIOR_VAR: "low",
    }
    healthy = {
        "cpu_util": "low", "ram_util": "low", "queue_depth": "low",
        "uplink_latency": "low", "jitter_pktloss": "low", "handover_rate": "low",
        "residual_energy": "high", PRIOR_VAR: "high",
    }
    cases = [
        ("no evidence at all", {}),
        ("full telemetry, stressed", full),
        ("full telemetry, healthy", healthy),
        ("temperature observed", {**full, "temperature": "high"}),
        ("single channel", {"cpu_util": "high"}),
        ("device class only", {PRIOR_VAR: "low"}),
    ]
    # Progressive withholding: the partial-evidence path, which is the one most
    # likely to diverge if either implementation mishandles marginalisation.
    droppable = [v for v in EVIDENCE_VARS if v != "temperature"]
    for k in (1, 3, 5):
        reduced = {kk: vv for kk, vv in full.items() if kk not in droppable[:k]}
        cases.append((f"{k} channel(s) withheld", reduced))
    return cases


def main() -> int:
    try:
        import pgmpy  # noqa: F401
        from pgmpy.inference import VariableElimination
    except ImportError:
        print("pgmpy is not installed; skipping the cross-check.")
        print("  install it with:  solution/python/.venv/bin/pip install pgmpy")
        return 0

    print(f"cross-checking against pgmpy {getattr(pgmpy, '__version__', '?')}")

    net = bbn_model.build_network()
    model = build_pgmpy_model(net)
    engine = VariableElimination(model)

    worst = 0.0
    failures = 0

    print(f"\n{'evidence':30s} {'ours P(fail)':>14s} {'pgmpy P(fail)':>14s} {'|diff|':>10s}")
    for label, evidence in evidence_cases():
        ours = float(inference.variable_elimination(net, evidence)[
            states_of(TARGET_VAR).index("fail")])

        result = engine.query(variables=[TARGET_VAR], evidence=evidence or None,
                              show_progress=False)
        theirs = float(result.get_value(**{TARGET_VAR: "fail"}))

        diff = abs(ours - theirs)
        worst = max(worst, diff)
        flag = "" if diff <= TOLERANCE else "   <-- MISMATCH"
        if diff > TOLERANCE:
            failures += 1
        print(f"{label:30s} {ours:14.9f} {theirs:14.9f} {diff:10.2e}{flag}")

    # Also compare the latent marginals, not just the target.
    print("\nlatent marginals, full telemetry:")
    _, full = evidence_cases()[1]
    for latent in ("computational_stress", "network_degradation",
                   "energy_stress", "thermal_stress"):
        ours = inference.variable_elimination(net, full, latent)
        theirs_factor = engine.query(variables=[latent], evidence=full, show_progress=False)
        theirs = np.array([theirs_factor.get_value(**{latent: s})
                           for s in states_of(latent)])
        diff = float(np.abs(ours - theirs).max())
        worst = max(worst, diff)
        if diff > TOLERANCE:
            failures += 1
        print(f"  {latent:22s} max |diff| = {diff:.2e}")

    print(f"\nlargest disagreement anywhere: {worst:.3e}  (tolerance {TOLERANCE:.0e})")
    if failures:
        print(f"FAILED: {failures} comparison(s) outside tolerance")
        return 1
    print("PASS: the hand-written engine agrees with pgmpy on every case.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
