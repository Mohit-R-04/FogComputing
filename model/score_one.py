#!/usr/bin/env python3
"""
Scores a single node you describe on the command line.

Useful for seeing the model work on one concrete case, for trying "what if"
questions, and for demonstrating the partial-evidence behaviour live: simply omit
any option and that channel is marginalised out instead of imputed.

Examples
--------
A loaded fog node::

    python score_one.py --device-class fog --cpu 0.93 --ram 0.88 \
                        --queue 40 --latency 12 --jitter 0.30 --energy 0.95

An edge device with a nearly flat battery, nothing else reported::

    python score_one.py --device-class edge --energy 0.12

Everything dark except the device class::

    python score_one.py --device-class edge

Add --show-work to print the ordinal state each raw number was mapped to, the
posterior over each latent condition, and which channels were marginalised.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bbn_model
import inference
from bbn_model import EVIDENCE_VARS, LATENT_VARS, PRIOR_VAR
from discretise import QuantileCuts, RELIABILITY_STATE_BY_CLASS, row_to_evidence

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

#: Command-line option -> telemetry column.
OPTIONS = {
    "cpu": "cpu_util",
    "ram": "ram_util",
    "queue": "queue_depth",
    "latency": "uplink_latency",
    "jitter": "jitter_pktloss",
    "handovers": "handover_rate",
    "energy": "residual_energy",
    "temperature": "temperature",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-class", required=True,
                    choices=sorted(RELIABILITY_STATE_BY_CLASS),
                    help="sets the node-reliability prior")
    ap.add_argument("--cpu", type=float, help="CPU utilisation, 0-1")
    ap.add_argument("--ram", type=float, help="RAM utilisation, 0-1")
    ap.add_argument("--queue", type=float, help="pending tuples in the window")
    ap.add_argument("--latency", type=float, help="uplink latency, ms")
    ap.add_argument("--jitter", type=float, help="jitter and packet loss, 0-1")
    ap.add_argument("--handovers", type=float, help="handovers in the window")
    ap.add_argument("--energy", type=float, help="residual energy, 0-1 (1 = full)")
    ap.add_argument("--temperature", type=float,
                    help="degrees C; normally unavailable, so normally omitted")
    ap.add_argument("--quantiles", default=str(DATA / "discretisation_quantiles.json"))
    ap.add_argument("--learned", default=None,
                    help="trained CPTs from train_bbn.py, e.g. ../data/cpts_trained.json")
    ap.add_argument("--show-work", action="store_true",
                    help="print the discretisation and the latent posteriors")
    args = ap.parse_args(argv)

    # Build a telemetry row exactly like one line of telemetry_cases.csv. An option
    # left off the command line becomes an empty field, which the discretiser reads
    # as "this channel did not report" and leaves out of the evidence set.
    row = {"device_class": args.device_class}
    for option, column in OPTIONS.items():
        value = getattr(args, option.replace("-", "_"))
        row[column] = "" if value is None else str(value)

    cuts = QuantileCuts.load(args.quantiles)
    evidence = row_to_evidence(row, cuts)

    if args.learned:
        from train_bbn import load_cpts
        net = bbn_model.BayesianNetwork(cpts=load_cpts(Path(args.learned)))
        net.validate()
        model_name = f"refined ({Path(args.learned).name})"
    else:
        net = bbn_model.build_network()
        model_name = "elicited"

    result = inference.infer(net, evidence)

    print(f"\nmodel        : {model_name}")
    print(f"device class : {args.device_class} "
          f"(reliability prior = {evidence[PRIOR_VAR]})")

    print("\nevidence supplied:")
    print(f"  {'channel':18s} {'value':>10s}      state     cut points (low | medium | high)")
    for var in EVIDENCE_VARS:
        raw = row[var]
        if var in evidence:
            cut = cuts.cuts_for(var, args.device_class)
            band = f"< {cut[0]:.4g} | < {cut[1]:.4g} | >=" if cut else "n/a"
            print(f"  {var:18s} {raw:>10s}  ->  {evidence[var]:<8s}  {band}")
        else:
            print(f"  {var:18s} {'—':>10s}      not reported, marginalised out")

    print("\nStates are tertiles of the baseline run, so they mean 'high for this")
    print("topology' rather than 'high' on any absolute scale. Note in particular the")
    print("residual_energy is interpreted relative to the three tiers: cloud, fog,")
    print("and edge. A full edge-device battery and a lightly loaded fog node are")
    print("different physical situations even when their normalized values are similar.")

    print(f"\nP(fail within horizon) = {result.p_fail:.4f}")
    print(f"principal cause        = {result.top_cause}", end="")
    if result.top_cause != "none":
        print(f"  (P(high)={result.top_cause_strength:.3f}, "
              f"lift {result.top_cause_lift:+.3f} above its prior)")
    else:
        print("  (no condition raised above its prior)")

    if args.show_work:
        print("\nposterior over each latent condition:")
        for latent in LATENT_VARS:
            dist = result.latent_posteriors[latent]
            parts = "  ".join(f"{s}={dist[s]:.3f}" for s in ("low", "medium", "high"))
            print(f"  {latent:22s} {parts}")
        print(f"\nchannels marginalised out: "
              f"{', '.join(result.marginalised) if result.marginalised else 'none'}")
        print("\nThose channels are summed over, not filled in with a default. That is "
              "\nwhy a prediction is still produced when telemetry is missing.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
