#!/usr/bin/env python3
"""
Generates the labelled telemetry of the whole design with iFogSim.

    python generate_telemetry.py            # every run of design.all_runs()
    python generate_telemetry.py --only 0   # just scenario 0 (all its seeds)

Each run simulates one operating regime with one seed under policy NO_FT -- the system
as deployed without fault tolerance, which is what the predictor learns from -- and
writes data/raw/scenario_SS_seed_K.csv. Runs execute in parallel, one JVM each.
"""

from __future__ import annotations

import argparse
import sys
import time

import design
import javarun


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=int, nargs="*", help="restrict to these scenarios")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    javarun.ensure_built()
    design.RAW.mkdir(parents=True, exist_ok=True)
    runs = [r for r in design.all_runs() if not args.only or r[0] in args.only]
    if not args.only:
        for old in design.RAW.glob("scenario_*.csv"):
            old.unlink()

    jobs = [("ScenarioSweep", [str(s), str(k), javarun.relpath_from_ifogsim(design.raw_path(s, k)),
                               str(design.SIM_TIME)]) for s, k, _ in runs]
    start = time.time()

    def progress(done, total, res):
        if not args.quiet:
            head = next((l for l in res.stdout.splitlines() if l.startswith("=== scenario")), "")
            fails = next((l for l in res.stdout.splitlines() if l.startswith("failures")), "")
            print(f"  [{done:2d}/{total}] {head.strip('= ').split('  policy')[0]:30s} {fails[21:]}")

    javarun.run_many(jobs, workers=args.workers, on_done=progress)
    print(f"generated {len(jobs)} runs in {time.time() - start:.1f} s -> {design.RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
