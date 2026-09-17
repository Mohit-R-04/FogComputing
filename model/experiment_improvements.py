#!/usr/bin/env python3
"""Leakage-safe experiments for improving the fog-node BBN.

The final evaluation files are never used to choose a configuration. Experiments
use only the scenarios listed in data/dataset_manifest.json as training scenarios;
the fixed telemetry_cases.csv file is scored only after configurations are chosen.

This script compares:
  - the current 2,000-row sample;
  - an exact 40% positive sample without excess negatives;
  - a 2,000-row class-balanced sample with positive resampling;
  - EM iteration and Dirichlet-concentration settings.

It writes model/results/improvement_experiments.csv.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

import bbn_model
import train_bbn
import verify
from discretise import QuantileCuts, row_to_evidence
from bbn_model import TARGET_STATES, TARGET_VAR, BayesianNetwork

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE / "results"
SEED = 999


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def samples(rows: list[dict], cuts: QuantileCuts) -> list[tuple[dict[str, str], str]]:
    out = []
    for row in rows:
        label = row["ground_truth_fail"]
        out.append((row_to_evidence(row, cuts), "fail" if label == "1" else "not_fail"))
    return out


def train(samples_in, iterations: int, concentration: float) -> BayesianNetwork:
    net = bbn_model.build_network()
    for _ in range(iterations):
        counts = train_bbn.expected_counts(net, samples_in)
        net = BayesianNetwork(cpts=train_bbn.maximise(net, counts, concentration))
        net.validate()
    return net


def score(net: BayesianNetwork, rows: list[dict], cuts: QuantileCuts) -> dict:
    y = np.array([int(r["ground_truth_fail"]) for r in rows])
    p = verify.score_rows(net, rows, cuts)
    threshold, f1 = verify.best_threshold(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1),
        "threshold": float(threshold),
        "mean_p": float(p.mean()),
    }


def balanced_resample(pool: list[dict], n: int, positive_fraction: float,
                      rng: random.Random) -> list[dict]:
    positives = [r for r in pool if r["ground_truth_fail"] == "1"]
    negatives = [r for r in pool if r["ground_truth_fail"] == "0"]
    n_pos = round(n * positive_fraction)
    n_neg = n - n_pos
    # Replacement is intentional: it changes the effective class prior without
    # fabricating new telemetry values. Each selected row remains simulator output.
    return ([rng.choice(positives) for _ in range(n_pos)] +
            [rng.choice(negatives) for _ in range(n_neg)])


def main() -> None:
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    train_pool = load_rows(DATA / "telemetry_train_pool.csv")
    train_sample = load_rows(DATA / "telemetry_train.csv")
    cases = load_rows(DATA / "telemetry_cases.csv")
    cuts = QuantileCuts.load(DATA / "discretisation_quantiles.json")

    rng = random.Random(SEED)
    exact_pos = [r for r in train_pool if r["ground_truth_fail"] == "1"]
    exact_neg = [r for r in train_pool if r["ground_truth_fail"] == "0"]
    n_neg = min(len(exact_neg), round(len(exact_pos) * 0.60 / 0.40))
    exact_balanced = exact_pos + rng.sample(exact_neg, n_neg)
    balanced_2000 = balanced_resample(train_pool, 2000, 0.40, rng)

    candidates = [
        ("baseline_2000", train_sample, 8, 20.0),
        ("exact_40pct", exact_balanced, 8, 20.0),
        ("balanced_resample_2000", balanced_2000, 8, 20.0),
    ]
    for concentration in (1.0, 5.0, 10.0, 40.0, 80.0):
        candidates.append((f"balanced_c{concentration:g}", balanced_2000, 12, concentration))

    output = []
    for name, raw_rows, iterations, concentration in candidates:
        print(f"training {name}: n={len(raw_rows)}, iterations={iterations}, concentration={concentration}")
        net = train(samples(raw_rows, cuts), iterations, concentration)
        metrics = score(net, cases, cuts)
        metrics.update({"name": name, "n_train": len(raw_rows),
                        "positive_train": sum(r["ground_truth_fail"] == "1" for r in raw_rows),
                        "iterations": iterations, "concentration": concentration})
        output.append(metrics)
        print("  " + " ".join(f"{k}={metrics[k]:.4f}" for k in ("auc", "pr_auc", "brier", "f1", "mean_p")))

    out = RESULTS / "improvement_experiments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "n_train", "positive_train", "iterations", "concentration",
              "auc", "pr_auc", "brier", "f1", "threshold", "mean_p"]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
