#!/usr/bin/env python3
"""
Turns the raw iFogSim runs into the training pool, the test pool, and the default
discretisation.

    python build_dataset.py

Inputs: data/raw/scenario_SS_seed_K.csv, one per (scenario, seed) of ``design.py``.

Outputs:

    data/telemetry_train_pool.csv   every row of the training runs
                                    (training scenarios x training seeds)
    data/telemetry_test_pool.csv    every row of the held-out runs
                                    (held-out scenarios x held-out seeds)
    data/discretisation.json        tertile cut points from training rows only
    data/dataset_manifest.json      split, counts, positives, causes, channel coverage

Nothing is sampled away. The evaluation uses the whole held-out pool at its natural
failure prevalence; the old balanced 200-case sample (40% positives, 40 of its 120
negatives trivial cloud rows) is gone. Whole runs are held out rather than random rows,
because consecutive snapshots of one node are nearly identical and a random row split
would test the model on near-copies of its training data.

The script also audits the split for the defect that invalidated the earlier
comparison: it counts held-out failures that repeat a training failure at the same node
and epoch. With the fault process keyed on (seed, scenario, node, epoch) that count is
at chance level, and the manifest records it.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import design
from discretise import build_cuts

REQUIRED = [
    "scenario_id", "seed", "node_name", "device_class", "epoch",
    "cpu_util", "ram_util", "queue_depth", "uplink_latency", "jitter_pktloss",
    "handover_rate", "residual_energy", "temperature",
    "cpu_trend", "queue_trend", "latency_trend", "temp_trend", "energy_drain",
    "reliability_prior", "true_health", "ground_truth_fail", "failure_time", "failure_cause",
]

CHANNELS = [
    "cpu_util", "ram_util", "queue_depth", "uplink_latency", "jitter_pktloss",
    "handover_rate", "residual_energy", "temperature",
    "cpu_trend", "queue_trend", "latency_trend", "temp_trend", "energy_drain",
]

OBSOLETE = ["telemetry_cases.csv", "telemetry_cases_pool.csv", "telemetry_train.csv",
            "cpts_trained.json", "discretisation_quantiles.json"]


def load_runs(split: str) -> pd.DataFrame:
    frames = []
    for scenario, seed, s in design.all_runs():
        if s != split:
            continue
        path = design.raw_path(scenario, seed)
        if not path.exists():
            sys.exit(f"missing {path}; run generate_telemetry.py first")
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            sys.exit(f"{path.name} lacks columns {missing}; regenerate the telemetry")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def failures(df: pd.DataFrame) -> pd.DataFrame:
    f = df.dropna(subset=["failure_time"]).groupby(["scenario_id", "seed", "node_name"]).agg(
        device_class=("device_class", "first"), failure_time=("failure_time", "first"),
        cause=("failure_cause", "first")).reset_index()
    f["epoch_index"] = (f.failure_time // design.EPOCH).astype(int)
    return f


def repeat_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """How often a node that fails in a held-out run fails at the same epoch in training.

    In the earlier data every scenario reused one seed and the draws ignored the
    scenario, so 92.1% of such pairs coincided exactly and a model could memorise
    "node 57 fails at epoch 31". With independent draws only the genuine temporal
    clustering of wear-out on heavily loaded nodes remains.
    """
    tr = failures(train)
    te = failures(test)
    train_epochs = tr.groupby("node_name").epoch_index.apply(set).to_dict()
    pairs = same = 0
    for node, epoch in zip(te.node_name, te.epoch_index):
        for e in train_epochs.get(node, ()):
            pairs += 1
            same += int(e == epoch)
    return {"held_out_failures": int(len(te)),
            "node_pairs_failing_in_test_and_train": pairs,
            "same_epoch_share": round(same / pairs, 4) if pairs else 0.0,
            "same_epoch_share_before_fix": 0.921}


def coverage(df: pd.DataFrame) -> dict:
    out = {}
    for c in CHANNELS:
        out[c] = {"coverage": round(float(df[c].notna().mean()), 4),
                  "distinct": int(df[c].nunique())}
    return out


def summary(df: pd.DataFrame) -> dict:
    by_class = {}
    for cls, g in df.groupby("device_class"):
        by_class[cls] = {"rows": int(len(g)), "positives": int(g.ground_truth_fail.sum())}
    f = failures(df)
    return {
        "rows": int(len(df)),
        "positives": int(df.ground_truth_fail.sum()),
        "prevalence": round(float(df.ground_truth_fail.mean()), 5),
        "runs": int(df.groupby(["scenario_id", "seed"]).ngroups),
        "by_class": by_class,
        "failed_nodes": int(len(f)),
        "failure_causes": {k: int(v) for k, v in f.cause.value_counts().items()},
    }


def main(argv=None) -> int:
    train = load_runs("train")
    test = load_runs("test")

    for name in OBSOLETE:
        p = design.DATA / name
        if p.exists():
            p.unlink()

    train.to_csv(design.DATA / "telemetry_train_pool.csv", index=False)
    test.to_csv(design.DATA / "telemetry_test_pool.csv", index=False)

    cuts = build_cuts({c: train[c].to_numpy(dtype=float) for c in CHANNELS}, None, "tertile",
                      device_class=train.device_class.to_numpy())
    cuts.save(design.DATA / "discretisation.json")

    audit = repeat_audit(train, test)
    manifest = {
        "design": {
            "train_scenarios": list(design.TRAIN_SCENARIOS),
            "test_scenarios": list(design.TEST_SCENARIOS),
            "train_seeds": list(design.TRAIN_SEEDS),
            "test_seeds": list(design.TEST_SEEDS),
            "sim_time": design.SIM_TIME,
            "horizon": design.HORIZON,
        },
        "train": summary(train),
        "test": summary(test),
        "channels_train": coverage(train),
        "leak_audit": audit,
        "excluded_from_features": list(design.NON_FEATURES),
    }
    (design.DATA / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1))

    print("dataset built")
    for split in ("train", "test"):
        s = manifest[split]
        cls = ", ".join(f"{k} {v['rows']}/{v['positives']}" for k, v in s["by_class"].items())
        print(f"  {split:5s}: {s['rows']:7d} rows, {s['positives']:4d} positive "
              f"({100 * s['prevalence']:.2f}%) from {s['runs']} runs   [{cls}]")
        print(f"         failed nodes {s['failed_nodes']}, causes {s['failure_causes']}")
    print(f"  leak audit: of {audit['node_pairs_failing_in_test_and_train']} (held-out, training) "
          f"failures of the same node, {100 * audit['same_epoch_share']:.1f}% fall in the same "
          f"epoch (92.1% before the fix)")
    print("  wrote telemetry_train_pool.csv, telemetry_test_pool.csv, discretisation.json, "
          "dataset_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
