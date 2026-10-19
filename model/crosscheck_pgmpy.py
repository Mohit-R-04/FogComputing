#!/usr/bin/env python3
"""
Cross-checks the hand-written inference engine against pgmpy.

`test_inference.py` already checks variable elimination against a direct
transcription of Eq. (1), but both of those are this project's own code, and a
shared misreading of the semantics would pass both. This script rebuilds the
networks in pgmpy -- an independent, widely used implementation --
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
from bbn_model import EVIDENCE_VARS, PRIOR_VAR, TARGET_VAR, states_of

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

    st = net.structure
    edges = [(p, v) for v in st.variables for p in st.parents[v]]
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(st.variables)

    for var in st.variables:
        parents = st.parents[var]
        card = len(st.states(var))

        if parents:
            configs = list(itertools.product(*[st.states(p) for p in parents]))
            table = np.array([[net.cpt(var)[c][s] for c in configs] for s in range(card)])
            cpd = TabularCPD(
                variable=var,
                variable_card=card,
                values=table,
                evidence=list(parents),
                evidence_card=[len(st.states(p)) for p in parents],
                state_names={var: list(st.states(var)),
                             **{p: list(st.states(p)) for p in parents}},
            )
        else:
            column = np.array([[p] for p in net.cpt(var)[()]])
            cpd = TabularCPD(variable=var, variable_card=card, values=column,
                             state_names={var: list(st.states(var))})
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


def real_cases(net, cuts, n=40):
    """Evidence from real held-out snapshots, with their natural missing channels."""
    import pandas as pd

    import design
    from discretise import row_to_evidence

    path = design.DATA / "telemetry_test_pool.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, nrows=60000)
    df = pd.concat([df[df.ground_truth_fail == 1].head(n // 2), df[df.ground_truth_fail == 0].sample(n // 2, random_state=1)])
    return [(f"held-out {r.node_name}@{r.epoch:.0f}", row_to_evidence(r.to_dict(), cuts, structure=net.structure))
            for _, r in df.iterrows()]


def compare(label_net, net, cases, VariableElimination, latent_case=None) -> tuple[float, int]:
    model = build_pgmpy_model(net)
    engine = VariableElimination(model)
    worst, failures = 0.0, 0
    print(f"\n[{label_net}]  {len(net.structure.variables)} variables")
    print(f"{'evidence':34s} {'ours P(fail)':>14s} {'pgmpy P(fail)':>14s} {'|diff|':>10s}")
    for label, evidence in cases:
        ours = float(inference.variable_elimination(net, evidence)[states_of(TARGET_VAR).index("fail")])
        result = engine.query(variables=[TARGET_VAR], evidence=evidence or None, show_progress=False)
        theirs = float(result.get_value(**{TARGET_VAR: "fail"}))
        diff = abs(ours - theirs)
        worst = max(worst, diff)
        if diff > TOLERANCE:
            failures += 1
        print(f"{label[:34]:34s} {ours:14.9f} {theirs:14.9f} {diff:10.2e}{'   <-- MISMATCH' if diff > TOLERANCE else ''}")
    _, full = latent_case or cases[1]
    for latent in net.structure.latent:
        ours = inference.variable_elimination(net, full, latent)
        theirs_factor = engine.query(variables=[latent], evidence=full, show_progress=False)
        theirs = np.array([theirs_factor.get_value(**{latent: s}) for s in net.structure.states(latent)])
        diff = float(np.abs(ours - theirs).max())
        worst = max(worst, diff)
        failures += diff > TOLERANCE
        print(f"  latent {latent:22s} max |diff| = {diff:.2e}")
    return worst, failures


def main() -> int:
    try:
        import pgmpy  # noqa: F401
        from pgmpy.inference import VariableElimination
    except ImportError:
        print("pgmpy is not installed; skipping the cross-check.")
        print("  install it with:  model/.venv/bin/pip install pgmpy")
        return 0

    print(f"cross-checking against pgmpy {getattr(pgmpy, '__version__', '?')}")
    worst, failures = compare("elicited network, paper Fig. 2", bbn_model.build_network(), evidence_cases(),
                              VariableElimination)

    import design
    trained_path = design.DATA / "bbn_trained.json"
    if trained_path.exists():
        from bbn_io import load_model

        net, cuts, doc = load_model(trained_path)
        # The hand-written cases use the paper's three states; the trained network's
        # evidence has up to eight, so it is checked on real held-out snapshots.
        real = real_cases(net, cuts)
        cases = [("no evidence at all", {}), ("device class only", {PRIOR_VAR: "medium"})] + real
        richest = max(real, key=lambda c: len(c[1])) if real else None
        w, f = compare(f"trained network, {doc['structure']}", net, cases, VariableElimination, richest)
        worst, failures = max(worst, w), failures + f

    print(f"\nlargest disagreement anywhere: {worst:.3e}  (tolerance {TOLERANCE:.0e})")
    if failures:
        print(f"FAILED: {failures} comparison(s) outside tolerance")
        return 1
    print("PASS: the project's inference engine agrees with pgmpy on every case.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
