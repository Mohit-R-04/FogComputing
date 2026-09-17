#!/usr/bin/env python3
"""
Runs the BBN over the generated cases and writes one posterior per case.

    python predict.py [--cases FILE] [--quantiles FILE] [--out FILE] [--learned FILE]

Pass ``--learned`` to score with conditional probability tables refined by
`train_bbn.py` instead of the elicited ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import bbn_model
import inference
from discretise import QuantileCuts, row_to_evidence

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE / "results"


def load_cases(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def score_cases(net, cases: list[dict], cuts: QuantileCuts,
                drop: frozenset[str] = frozenset()) -> list[dict]:
    """Scores every case, returning one result row each."""
    out = []
    for row in cases:
        evidence = row_to_evidence(row, cuts, drop=drop)
        result = inference.infer(net, evidence)
        out.append({
            "case_id": row.get("case_id", ""),
            "node_name": row.get("node_name", ""),
            "device_class": row.get("device_class", ""),
            "epoch": row.get("epoch", ""),
            "p_fail": round(result.p_fail, 6),
            "top_cause": result.top_cause,
            "top_cause_lift": round(result.top_cause_lift, 6),
            "n_evidence": len(evidence),
            "marginalised": "|".join(result.marginalised),
            "ground_truth_fail": row.get("ground_truth_fail", ""),
            "explanation": result.explain(),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=str(DATA / "telemetry_cases.csv"))
    ap.add_argument("--quantiles", default=str(DATA / "discretisation_quantiles.json"))
    ap.add_argument("--out", default=str(RESULTS / "predictions.csv"))
    ap.add_argument("--learned", default=None,
                    help="JSON of trained CPTs from train_bbn.py")
    args = ap.parse_args(argv)

    cases = load_cases(Path(args.cases))
    cuts = QuantileCuts.load(args.quantiles)

    if args.learned:
        from train_bbn import load_cpts
        net = bbn_model.BayesianNetwork(cpts=load_cpts(Path(args.learned)))
        net.validate()
        source = f"trained CPTs from {Path(args.learned).name}"
    else:
        net = bbn_model.build_network()
        source = "elicited CPTs"

    rows = score_cases(net, cases, cuts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(1 for r in rows if r["ground_truth_fail"] == "1")
    mean_p = sum(r["p_fail"] for r in rows) / len(rows)
    causes: dict[str, int] = {}
    for r in rows:
        causes[r["top_cause"]] = causes.get(r["top_cause"], 0) + 1

    print(f"scored {len(rows)} cases using {source}")
    print(f"  quantiles from {Path(args.quantiles).name} "
          f"({cuts.n_snapshots} baseline snapshots)")
    print(f"  ground-truth positives : {positives}")
    print(f"  mean P(fail)           : {mean_p:.4f}")
    print("  principal causes       :")
    for cause, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"    {cause:22s} {n:4d}")
    print(f"  written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
