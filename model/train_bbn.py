#!/usr/bin/env python3
"""
Trains the Bayesian Belief Network on the telemetry iFogSim generated.

    python train_bbn.py                 # the configuration chosen by select_bbn.py
    python train_bbn.py --structure paper --cuts tertile --concentration 5

The network's *structure* is fixed before training (the paper's Fig. 2, or the
validated extension recorded in ``data/models/bbn_config.json``). What is learned are
the conditional probability tables.

Three things make this less trivial than counting occurrences.

**The latent conditions are never observed.** Computational stress has no sensor; it is
inferred. So the expected sufficient statistics come from expectation-maximisation: on
each pass, the posterior over the latent conditions given both the telemetry and the
outcome that followed is computed, fractional counts are accumulated, and the tables are
re-estimated.

**The elicited tables are a fixed prior.** For the target table the M-step is
``(concentration * elicited + expected counts)``, renormalised -- a MAP estimate whose
prior is the expert-elicited ranked-node table, not the previous iterate. (The earlier
implementation re-centred the prior on the previous iterate every pass, which quietly
drifts towards the unsmoothed maximum-likelihood estimate as iterations accumulate.)
The latent tables of the selected model are *learned ranked nodes* (``ranked_node.py``):
their points are re-fitted each M-step under a Gaussian prior centred on the elicited
points, which is what allows the evidence tier to be discretised finely.

**Training and evaluation come from different scenarios and seeds** (``design.py``).

Training uses the natural failure prevalence (no class re-weighting), because the
controller multiplies the posterior by a cost: it must be a calibrated probability.
Identical discretised patterns are aggregated, which leaves the sufficient statistics
unchanged and makes the full-pool fit take seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

import bbn_model
import design
from bbn_fast import CompiledBBN, aggregate_patterns
from bbn_io import save_model
from discretise import QuantileCuts, build_cuts

CONFIG_PATH = design.MODELS / "bbn_config.json"
MODEL_PATH = design.DATA / "bbn_trained.json"

DEFAULT_CONFIG = {"family": "ranked", "trends": True, "class_conditional": True, "class_latent": True,
                  "cuts": "supervised", "states": 8, "per_class_cuts": True,
                  "concentration": 5.0, "iterations": 60, "lam": 0.05}


def evidence_for(config: dict) -> tuple[str, ...]:
    return bbn_model.make_structure(trends=bool(config["trends"])).evidence


def cuts_for(df: pd.DataFrame, config_or_structure, method: str | None = None) -> QuantileCuts:
    """Cut points from training rows only (accepts a config dict, or a structure + method)."""
    if isinstance(config_or_structure, dict):
        config = config_or_structure
        evidence = evidence_for(config)
        method, states, per_class = config["cuts"], int(config.get("states", 3)), bool(config.get("per_class_cuts", False))
    else:
        evidence, states, per_class = config_or_structure.evidence, 3, False
    cols = {v: df[v].to_numpy(dtype=float) for v in evidence}
    labels = df["ground_truth_fail"].to_numpy(dtype=int)
    return build_cuts(cols, labels, method, device_class=df["device_class"].to_numpy(),
                      states=states, per_class=per_class)


def structure_for(config: dict, cuts: QuantileCuts | None = None) -> bbn_model.Structure:
    counts = {v: cuts.states_of(v) for v in evidence_for(config)} if cuts is not None else None
    return bbn_model.make_structure(trends=bool(config["trends"]),
                                    class_conditional=bool(config["class_conditional"]),
                                    class_latent=bool(config.get("class_latent", False)),
                                    state_counts=counts,
                                    latent_form=config.get("family", "table"))


def fit(df: pd.DataFrame, structure: bbn_model.Structure, cuts: QuantileCuts,
        concentration: float = 5.0, max_iter: int = 60, tol: float = 1e-7,
        verbose: bool = False, lam: float = 0.05, callback=None):
    """MAP-EM on the rows of ``df``. Returns (network, log-likelihood history, meta)."""
    elicited = bbn_model.build_network(structure=structure)
    comp = CompiledBBN(elicited)
    if structure.latent_form == "ranked":
        comp.init_ranked()
    E, P = comp.encode(df, cuts)
    Y = df["ground_truth_fail"].to_numpy(dtype=np.int64)
    Eu, Pu, Yu, Wu = aggregate_patterns(E, P, Y)
    comp.fit_roots(Eu, Pu, Wu)
    prior = {v: comp.arr[v].copy() for v in structure.latent + (structure.target,)}

    history = []
    total = float(Wu.sum())
    for it in range(1, max_iter + 1):
        counts_l, counts_t, loglik = comp.expected_counts(Eu, Pu, Yu, Wu)
        comp.maximise(counts_l, counts_t, prior, concentration, lam)
        history.append(loglik / total)
        if verbose:
            print(f"  iteration {it:3d}  mean log-likelihood {history[-1]:+.6f}")
        if callback:
            callback(it, history[-1])
        if it > 1 and abs(history[-1] - history[-2]) < tol:
            break
    net = comp.to_network()
    meta = {"rows": int(len(df)), "positives": int(Y.sum()), "patterns": int(len(Eu)),
            "iterations": len(history), "final_loglik": history[-1],
            "concentration": concentration, "lam": lam, "structure": structure.name,
            "cuts": cuts.method,
            "ranked_nodes": {l: rn.to_dict() for l, rn in comp.ranked.items()}}
    return net, history, meta


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        saved = json.loads(CONFIG_PATH.read_text())
        cfg.update({k: saved[k] for k in DEFAULT_CONFIG if k in saved})
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure", choices=["paper", "trends", "paper+class", "trends+class"])
    ap.add_argument("--family", choices=["table", "ranked"])
    ap.add_argument("--states", type=int, help="maximum evidence states for supervised cuts")
    ap.add_argument("--cuts", choices=["tertile", "supervised"])
    ap.add_argument("--concentration", type=float)
    ap.add_argument("--iterations", type=int, help="EM iterations (default: as selected)")
    ap.add_argument("--train", default=str(design.DATA / "telemetry_train_pool.csv"))
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    config = load_config()
    if args.structure:
        config["trends"] = args.structure.startswith("trends")
        config["class_conditional"] = args.structure.endswith("+class")
    if args.cuts:
        config["cuts"] = args.cuts
    if args.family:
        config["family"] = args.family
    if args.states:
        config["states"] = args.states
    if args.concentration is not None:
        config["concentration"] = args.concentration
    if args.iterations is not None:
        config["iterations"] = args.iterations

    df = pd.read_csv(args.train)
    start = time.time()
    cuts = cuts_for(df, config)
    structure = structure_for(config, cuts)
    print(f"training on {len(df)} snapshots ({int(df.ground_truth_fail.sum())} positive) "
          f"from {df.groupby(['scenario_id', 'seed']).ngroups} iFogSim runs")
    print(structure.describe())
    print(f"  discretisation: {cuts.method}; latent tables: {structure.latent_form}; "
          f"Dirichlet concentration {config['concentration']}; {config['iterations']} EM iterations")
    net, history, meta = fit(df, structure, cuts, config["concentration"],
                             max_iter=int(config["iterations"]), tol=1e-12, verbose=not args.quiet,
                             lam=float(config.get("lam", 0.05)))
    meta["config"] = config
    meta["seconds"] = round(time.time() - start, 2)
    save_model(net, cuts, args.out, meta)
    print(f"  {meta['patterns']} distinct discretised patterns; converged in "
          f"{meta['iterations']} iterations ({meta['seconds']} s)")
    print(f"  trained network written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
