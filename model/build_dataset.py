#!/usr/bin/env python3
"""
Turns the raw scenario files into a training set and evaluation cases for the cloud/fog/edge topology.

Run after `solution/run_sweep.sh`:

    python build_dataset.py

What it does, in order:

1.  Merges every `solution/data/raw/scenario_*.csv`.
2.  **Reports the spread of every channel and warns about any that is constant.**
    This is the number to watch. A channel with one distinct value carries no
    information, and the previous version of this harness hid exactly that behind a
    synthetic overlay. If something is flat, the fix is a wider scenario knob, not
    manufactured data.
3.  Splits the *scenarios* into training and held-out. Whole configurations are held
    out rather than random rows: rows from one run are heavily autocorrelated -- the
    same node minutes apart -- so a random split would let the model see almost
    exactly its test rows during training and report a flattering number.
4.  Computes the discretisation cut points **from the training scenarios only**, so
    the held-out distribution never influences how evidence is encoded.
5.  Samples 100 evaluation cases from the held-out scenarios, stratified by device
    class and outcome.

Outputs, all in `solution/data/`:

    discretisation_quantiles.json   cut points, learned from training scenarios
    telemetry_train.csv             training rows, stratified to match the cases
    telemetry_train_pool.csv        every training row, unstratified
    telemetry_cases.csv             the 100 evaluation cases
    telemetry_cases_pool.csv        every held-out row, for lead-time analysis
    dataset_manifest.json           which scenarios went where, and the spread report
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"

#: Continuous channels written by the simulator.
CHANNELS = [
    "cpu_util", "ram_util", "queue_depth", "uplink_latency",
    "jitter_pktloss", "handover_rate", "residual_energy", "temperature",
    "reliability_prior",
]

#: Channels expected to be absent or constant, so not worth warning about.
#: `temperature` is never reported at all -- no simulated node has a thermal sensor.
#: `handover_rate` is zero on every mains-powered tier, because fixed infrastructure
#: does not roam; that is the correct reading, not a missing measurement.
EXPECTED_SPARSE = {"temperature"}
EXPECTED_FLAT_FOR_CLASS = {
    "handover_rate": {"cloud", "fog"},
    # Constant within a class by definition -- it is a declared attribute of the
    # device class, not a measurement. It varies across tiers, which is the only
    # place it needs to.
    "reliability_prior": {"cloud", "fog", "edge"},
    # iFogSim allocates a module's memory once at placement, and never varies it.
    "ram_util": {"cloud", "fog", "edge"},
}


def load_raw(raw_dir: Path) -> list[dict]:
    files = sorted(raw_dir.glob("scenario_*.csv"))
    if not files:
        sys.exit(
            f"No scenario files in {raw_dir}.\n"
            f"Generate them first:  ./solution/run_sweep.sh"
        )
    rows: list[dict] = []
    for f in files:
        with open(f, newline="") as fh:
            n = 0
            for row in csv.DictReader(fh):
                rows.append(row)
                n += 1
        print(f"  {f.name:22s} {n:7d} rows")
    return rows


def numeric(rows: list[dict], column: str) -> list[float]:
    out = []
    for r in rows:
        raw = (r.get(column) or "").strip()
        if raw:
            try:
                out.append(float(raw))
            except ValueError:
                pass
    return out


def spread_report(rows: list[dict]) -> dict:
    """Prints the distinct-value count per channel and flags anything constant."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["device_class"]].append(r)

    print(f"\n{'channel':18s} {'reported':>10s} {'distinct':>9s} {'min':>10s} {'max':>12s}")
    print("-" * 64)

    report: dict = {}
    warnings: list[str] = []

    for ch in CHANNELS:
        values = numeric(rows, ch)
        reported = len(values)
        distinct = len(set(values))
        coverage = reported / len(rows) if rows else 0.0
        report[ch] = {
            "reported": reported,
            "coverage": round(coverage, 4),
            "distinct": distinct,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
        if values:
            print(f"{ch:18s} {coverage:9.0%} {distinct:9d} {min(values):10.3f} {max(values):12.3f}")
        else:
            print(f"{ch:18s} {coverage:9.0%} {distinct:9d} {'—':>10s} {'—':>12s}")

        if ch in EXPECTED_SPARSE:
            continue
        if distinct <= 1:
            warnings.append(
                f"{ch} is CONSTANT across the whole dataset — it carries no "
                f"information and the BBN cannot learn anything from it"
            )
        elif distinct <= 3:
            warnings.append(
                f"{ch} takes only {distinct} distinct values — enough for a "
                f"three-state encoding, but only just"
            )

    # Per class, because a channel can look healthy overall while being flat where
    # it matters.
    per_class: dict = {}
    for cls, subset in sorted(by_class.items()):
        per_class[cls] = {}
        for ch in CHANNELS:
            values = numeric(subset, ch)
            per_class[cls][ch] = {"distinct": len(set(values)), "reported": len(values)}
            if ch in EXPECTED_SPARSE:
                continue
            if cls in EXPECTED_FLAT_FOR_CLASS.get(ch, set()):
                continue
            if values and len(set(values)) <= 1:
                warnings.append(f"{ch} is constant for every {cls} node")

    if warnings:
        print("\n" + "!" * 64)
        print("SPREAD WARNINGS")
        print("!" * 64)
        for w in warnings:
            print(f"  - {w}")
        print(
            "\n  Widen the scenario knobs in ScenarioSweep.java rather than adding\n"
            "  synthetic variation: host MIPS and RAM ranges, sensor interval, module\n"
            "  sizing, and the busy/idle ratio of the power model are the levers."
        )
    else:
        print("\nEvery channel varies. No warnings.")

    return {"overall": report, "per_class": per_class, "warnings": warnings}


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def cut_block(rows: list[dict]) -> dict:
    block = {}
    for ch in CHANNELS:
        if ch == "temperature":
            continue          # never observed, so nothing to quantise against
        values = sorted(numeric(rows, ch))
        if not values:
            continue
        block[ch] = {
            "q33": round(percentile(values, 100.0 / 3.0), 6),
            "q67": round(percentile(values, 200.0 / 3.0), 6),
        }
    return block


def build_quantiles(train_rows: list[dict]) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        by_class[r["device_class"]].append(r)
    return {
        "_comment": (
            "Tertile cut points. Computed from the TRAINING scenarios only, so the "
            "held-out data never influences how evidence is encoded. A value below "
            "q33 is the lowest state, at or above q67 the highest. residual_energy "
            "and reliability_prior are inverted downstream: a low numeric value "
            "means high stress."
        ),
        "n_snapshots": len(train_rows),
        "global": cut_block(train_rows),
        "by_device_class": {cls: cut_block(rs) for cls, rs in sorted(by_class.items())},
    }


def stratified_sample(rows: list[dict], n: int, rng: random.Random,
                      positive_fraction: float = 0.40) -> list[dict]:
    """Draws n rows, balancing outcome and spreading across device classes.

    The positive rate is capped rather than forced to 50%: a fog deployment in which
    half of all observations precede a failure would be unrealistic. It is lifted
    well above the natural rate all the same, because at the true prevalence of a
    couple of percent a hundred rows would contain two positives and could not
    support any measurement at all.
    """
    positives = [r for r in rows if r["ground_truth_fail"] == "1"]
    negatives = [r for r in rows if r["ground_truth_fail"] == "0"]

    want_pos = min(int(round(n * positive_fraction)), len(positives))
    want_neg = n - want_pos
    if want_neg > len(negatives):
        want_neg = len(negatives)
        want_pos = min(n - want_neg, len(positives))

    def by_class(pool: list[dict], want: int) -> list[dict]:
        if want <= 0 or not pool:
            return []
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            buckets[r["device_class"]].append(r)
        for b in buckets.values():
            rng.shuffle(b)
        out, cursor = [], 0
        classes = sorted(buckets)
        while len(out) < want:
            took = False
            for cls in classes:
                if len(out) >= want:
                    break
                if cursor < len(buckets[cls]):
                    out.append(buckets[cls][cursor])
                    took = True
            if not took:
                break
            cursor += 1
        return out

    sample = by_class(positives, want_pos) + by_class(negatives, want_neg)
    rng.shuffle(sample)
    return sample


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out-dir", default=str(DATA),
                    help="where to write the dataset (default solution/data)")
    ap.add_argument("--cases", type=int, default=100,
                    help="number of evaluation cases (default 100)")
    ap.add_argument("--holdout", type=int, default=4,
                    help="how many scenarios to hold out for evaluation (default 4)")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--positive-fraction", type=float, default=0.40,
                    help="positive rate in both the training set and the cases")
    ap.add_argument("--train-size", type=int, default=2000,
                    help="rows in the stratified training set (default 2000)")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reading raw scenario files")
    rows = load_raw(Path(args.raw))
    fieldnames = list(rows[0].keys())
    scenarios = sorted({int(r["scenario_id"]) for r in rows})
    print(f"\n{len(rows)} rows from {len(scenarios)} scenarios: {scenarios}")

    print("\n" + "=" * 64)
    print("CHANNEL SPREAD — the thing to check before trusting anything downstream")
    print("=" * 64)
    report = spread_report(rows)

    if len(scenarios) <= args.holdout:
        sys.exit(f"\nNeed more than {args.holdout} scenarios to hold {args.holdout} out; "
                 f"found {len(scenarios)}. Run the full sweep.")

    held_out = sorted(rng.sample(scenarios, args.holdout))
    train_ids = [s for s in scenarios if s not in held_out]
    print(f"\nscenario split — train on {train_ids}, evaluate on {held_out}")

    train_rows = [r for r in rows if int(r["scenario_id"]) in train_ids]
    test_rows = [r for r in rows if int(r["scenario_id"]) in held_out]
    print(f"  training rows {len(train_rows)}  "
          f"({sum(1 for r in train_rows if r['ground_truth_fail'] == '1')} positive)")
    print(f"  held-out rows {len(test_rows)}  "
          f"({sum(1 for r in test_rows if r['ground_truth_fail'] == '1')} positive)")

    quantiles = build_quantiles(train_rows)
    (out_dir / "discretisation_quantiles.json").write_text(json.dumps(quantiles, indent=2))

    cases = stratified_sample(test_rows, args.cases, rng, args.positive_fraction)
    for i, row in enumerate(cases, start=1):
        row["case_id"] = str(i)

    # The training set is stratified to the SAME positive fraction as the
    # evaluation cases, and that matters more than it looks.
    #
    # The natural prevalence here is under 1%, because failures are rare events. The
    # evaluation set has to be lifted to ~40% or a hundred rows would contain a
    # single positive and could measure nothing. Train on the raw pool and score on
    # the lifted set and the model learns the wrong base rate: measured on this data
    # it ranked perfectly well, AUC 0.717, while predicting a mean of 0.014 against
    # an actual 0.40 prevalence, and its Brier score went from 0.19 to 0.385. A
    # posterior that miscalibrated is useless to the expected-loss migration rule the
    # paper builds on it, which compares P(fail) x C_fail against C_migrate.
    #
    # The full pool is written alongside for anyone who wants to train at natural
    # prevalence and apply their own prior correction instead.
    train_sample = stratified_sample(train_rows, args.train_size, rng,
                                     args.positive_fraction)
    write_csv(out_dir / "telemetry_train.csv", train_sample, fieldnames)
    write_csv(out_dir / "telemetry_train_pool.csv", train_rows, fieldnames)
    write_csv(out_dir / "telemetry_cases.csv", cases, fieldnames)
    write_csv(out_dir / "telemetry_cases_pool.csv", test_rows, fieldnames)

    manifest = {
        "seed": args.seed,
        "scenarios_all": scenarios,
        "scenarios_train": train_ids,
        "scenarios_held_out": held_out,
        "rows_train_pool": len(train_rows),
        "rows_train_sampled": len(train_sample),
        "rows_held_out": len(test_rows),
        "cases": len(cases),
        "cases_positive": sum(1 for r in cases if r["ground_truth_fail"] == "1"),
        "channel_report": report,
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    positives = manifest["cases_positive"]
    print(f"\nwrote into {out_dir}:")
    print(f"  telemetry_train.csv         {len(train_sample)} rows "
          f"({sum(1 for r in train_sample if r['ground_truth_fail'] == '1')} positive)")
    print(f"  telemetry_train_pool.csv    {len(train_rows)} rows (unstratified)")
    print(f"  telemetry_cases.csv         {len(cases)} cases ({positives} positive)")
    print(f"  telemetry_cases_pool.csv    {len(test_rows)} rows")
    print(f"  discretisation_quantiles.json")
    print(f"  dataset_manifest.json")

    if positives < 15 or len(cases) - positives < 15:
        print("\nWARNING: the evaluation set is badly unbalanced. Raise --cases, or "
              "\nhold out scenarios with more failures.")

    print("\nNext:  .venv/bin/python train_bbn.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
