#!/usr/bin/env python3
"""
Writes the BBN's posterior and explanation for every held-out fog and edge snapshot.

    python predict.py                      # trained network, held-out pool
    python predict.py --elicited           # paper network, elicited tables only

Output: model/results/predictions_test.csv (or predictions_elicited.csv) with, per
snapshot, P(fail within horizon), the principal latent cause and its lift above the
prior, how many channels were marginalised, and the ground truth for scoring.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import bbn_model
import design
from bbn_fast import CompiledBBN
from bbn_io import load_model
from cpt_tables import TARGET_WEIGHTS
from discretise import QuantileCuts


def explain(comp: CompiledBBN, E: np.ndarray, P: np.ndarray):
    """Principal cause per row: largest weighted lift of P(latent = high) over its prior."""
    high = comp.latent_high(E, P)
    none = np.full_like(E, -1)
    prior_high = comp.latent_high(none, P)
    weights = np.array([TARGET_WEIGHTS[l] for l in comp.latent])
    score = (high - prior_high) * weights
    idx = score.argmax(axis=1)
    lift = (high - prior_high)[np.arange(len(E)), idx]
    cause = np.where(score.max(axis=1) > 1e-6, np.array(comp.latent)[idx], "none")
    return cause, lift


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elicited", action="store_true")
    ap.add_argument("--data", default=str(design.DATA / "telemetry_test_pool.csv"))
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    df = df[df.device_class != "cloud"].reset_index(drop=True)
    if args.elicited:
        net = bbn_model.build_network()
        cuts = QuantileCuts.load(design.DATA / "discretisation.json")
        out = design.RESULTS / "predictions_elicited.csv"
    else:
        net, cuts, _ = load_model(design.DATA / "bbn_trained.json")
        out = design.RESULTS / "predictions_test.csv"
    comp = CompiledBBN(net)
    E, P = comp.encode(df, cuts)
    p = comp.posterior(E, P)
    cause, lift = explain(comp, E, P)
    result = df[["scenario_id", "seed", "node_name", "device_class", "epoch"]].copy()
    result["p_fail"] = p
    result["top_cause"] = cause
    result["top_cause_lift"] = lift
    result["n_marginalised"] = (E < 0).sum(axis=1)
    result["ground_truth_fail"] = df.ground_truth_fail
    result["failure_cause"] = df.failure_cause.fillna("")
    design.RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, float_format="%.6g")
    print(f"scored {len(result):,} snapshots with the {'elicited' if args.elicited else 'trained'} network "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
