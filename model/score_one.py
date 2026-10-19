#!/usr/bin/env python3
"""
Scores a single node you describe on the command line.

Useful for seeing the model work on one concrete case, for trying "what if" questions,
and for demonstrating the partial-evidence behaviour live: omit any option and that
channel is marginalised out instead of imputed.

Examples
--------
A fog node as it typically looks in the two epochs before a wear-out failure: saturated,
a long queue, a slow link, hot and heating, drawing extra power::

    python score_one.py --device-class fog --cpu 1.0 --queue 380 --latency 161 \\
                        --temperature 81 --temp-trend 5 --energy-drain 0.0032 --handovers 16

The same node with most of its channels dark (still well above a healthy node)::

    python score_one.py --device-class fog --cpu 1.0 --temperature 81 --energy-drain 0.0032

A healthy fog node, and an edge device whose battery is nearly flat::

    python score_one.py --device-class fog --cpu 0.0 --queue 0 --latency 105 --temperature 31
    python score_one.py --device-class edge --energy 0.01 --energy-drain 0.011

Add --show-work to print the state each number was mapped to, the posterior over each
latent condition, and which channels were marginalised. ``--elicited`` scores with the
paper's network and the expert-elicited tables only (no failure data).
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
from discretise import RELIABILITY_STATE_BY_CLASS, QuantileCuts, row_to_evidence
from predict import explain

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
    "cpu_trend": "cpu_trend",
    "queue_trend": "queue_trend",
    "latency_trend": "latency_trend",
    "temp_trend": "temp_trend",
    "energy_drain": "energy_drain",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-class", required=True, choices=sorted(RELIABILITY_STATE_BY_CLASS),
                    help="sets the node-reliability prior")
    ap.add_argument("--cpu", type=float, help="CPU utilisation, 0-1")
    ap.add_argument("--ram", type=float, help="RAM utilisation, 0-1")
    ap.add_argument("--queue", type=float, help="pending tuples in the window")
    ap.add_argument("--latency", type=float, help="uplink latency, ms")
    ap.add_argument("--jitter", type=float, help="arrival jitter, 0-1")
    ap.add_argument("--handovers", type=float, help="handovers in the window")
    ap.add_argument("--energy", type=float, help="residual energy, 0-1 (1 = full)")
    ap.add_argument("--temperature", type=float, help="degrees C (fog and cloud report it)")
    ap.add_argument("--cpu-trend", type=float, help="CPU change against recent readings")
    ap.add_argument("--queue-trend", type=float, help="relative queue change")
    ap.add_argument("--latency-trend", type=float, help="relative latency change")
    ap.add_argument("--temp-trend", type=float, help="temperature change, degrees C")
    ap.add_argument("--energy-drain", type=float, help="residual energy used per epoch")
    ap.add_argument("--model", default=str(design.DATA / "bbn_trained.json"))
    ap.add_argument("--elicited", action="store_true", help="paper network with elicited tables only")
    ap.add_argument("--show-work", action="store_true")
    args = ap.parse_args(argv)

    row = {"device_class": args.device_class}
    for option, column in OPTIONS.items():
        value = getattr(args, option)
        row[column] = "" if value is None else str(value)

    if args.elicited:
        net = bbn_model.build_network()
        cuts = QuantileCuts.load(design.DATA / "discretisation.json")
        model_name = "elicited (paper Fig. 2, no failure data)"
    else:
        net, cuts, doc = load_model(args.model)
        model_name = f"trained ({doc['structure']}, {doc['cut_method']} cuts)"
    structure = net.structure
    evidence = row_to_evidence(row, cuts, structure=structure)

    # The vectorised exact engine (the one verify.py and the comparison use): the same
    # posterior as variable elimination, fast enough for the trained network's tables.
    comp = CompiledBBN(net)
    frame = pd.DataFrame([{c: (np.nan if row.get(c, "") == "" else float(row[c])) for c in OPTIONS.values()}
                          | {"device_class": args.device_class}])
    E, P = comp.encode(frame, cuts)
    p_fail = float(comp.posterior(E, P)[0])
    cause, lift = explain(comp, E, P)
    cause, lift = str(cause[0]), float(lift[0])
    high = comp.latent_high(E, P)[0]

    print(f"\nmodel        : {model_name}")
    print(f"device class : {args.device_class} (reliability prior = {evidence.get(structure.prior)})")
    print("\nevidence supplied:")
    node = f"{'an' if args.device_class[0] in 'aeiou' else 'a'} {args.device_class} node"
    print(f"  {'channel':16s} {'value':>10s}      state    bin edges for {node}")
    for var in structure.evidence:
        raw = row.get(var, "")
        if var in evidence:
            c = cuts.cuts_for(var, args.device_class)
            edges = ", ".join(f"{x:.4g}" for x in c)
            print(f"  {var:16s} {raw:>10s}  ->  {evidence[var]:<7s}  [{edges}]")
        else:
            print(f"  {var:16s} {'-':>10s}      not reported: marginalised out")

    print(f"\nP(fail within horizon) = {p_fail:.4f}")
    print(f"principal cause        = {cause}", end="")
    if cause != "none":
        strength = float(high[list(comp.latent).index(cause)])
        print(f"  (P(high)={strength:.3f}, lift {lift:+.3f} above its prior for {node})")
    else:
        print("  (no condition raised above its prior)")
    if args.show_work:
        print("\nposterior over each latent condition:")
        for latent, dist in zip(comp.latent, comp.latent_marginals(E, P)):
            states = structure.states(latent)
            print(f"  {latent:22s} " + "  ".join(f"{st}={p:.3f}" for st, p in zip(states, dist[0])))
        missing = [v for v in structure.evidence if v not in evidence]
        print(f"\nchannels marginalised out: {', '.join(missing) or 'none'}")
        print("They are summed over, not filled in with a default -- which is why a prediction is "
              "still produced when telemetry is missing.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
