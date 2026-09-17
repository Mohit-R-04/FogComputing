#!/usr/bin/env python3
"""
Trains the Bayesian Belief Network on the telemetry iFogSim generated.

    python train_bbn.py
    python train_bbn.py --train ../data/telemetry_train_pool.csv

The network's *structure* is not learned: it is fixed at the causal graph of the
Review 2 paper's Fig. 2. What is learned here are the conditional probability
tables -- every number the network uses to turn evidence into a posterior.

The default fit uses the complete training pool, not only the 2,000-row report
sample. Rare failures receive normalized class weights targeting the 40% prevalence
used by the deliberately balanced evaluation cases. This changes only sufficient
statistics; it does not duplicate or synthesize telemetry rows.

Three things make this less trivial than counting occurrences.

**The latent conditions are never observed.** Computational stress has no sensor; it
is inferred. So the expected sufficient statistics have to be filled in by
expectation-maximisation: on each pass, compute the posterior over the four latent
conditions given both the telemetry and the outcome that actually followed,
accumulate fractional counts weighted by that posterior, then renormalise.

**Counts are Dirichlet-smoothed rather than used raw.** The elicited tables from
`cpt_tables.py` enter as the prior, with a concentration saying how much evidence it
takes to overturn them. The target's table has 243 parent configurations and the
training data will not visit most of them; unsmoothed counting would fill it with
zeros and ones, and any configuration never seen would collapse to a degenerate
estimate instead of keeping a sensible default.

**Training and evaluation come from different scenarios.** `build_dataset.py` holds
out whole configurations, so what is learned here is tested on operating regimes the
model has never seen. Training on the rows you then score would make every number in
the verification report meaningless.

Watch the log-likelihood: it should rise on each iteration and then flatten. If it is
still climbing steeply at the last iteration, raise `--iterations`.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import bbn_model
import inference
from bbn_model import (
    EVIDENCE_VARS,
    LATENT_VARS,
    PARENTS,
    PRIOR_VAR,
    TARGET_STATES,
    TARGET_VAR,
    BayesianNetwork,
    states_of,
)
from discretise import QuantileCuts, row_to_evidence

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE / "results"

#: Strength of the elicited tables as a Dirichlet prior, in pseudo-observations.
#: Large enough that a handful of samples cannot overturn considered judgement,
#: small enough that a few hundred can move it.
DEFAULT_CONCENTRATION = 20.0

#: Tables that are learned. The evidence priors are left alone: tertile
#: discretisation puts a third of the mass in each state by construction, so there
#: is nothing there to estimate.
LEARNED_VARS = tuple(LATENT_VARS) + (TARGET_VAR,)


def _evidence_signature(evidence: dict[str, str], label: str) -> tuple:
    return (tuple(sorted(evidence.items())), label)


def expected_counts(net: BayesianNetwork,
                   samples: list[tuple[dict[str, str], str]],
                   sample_weights: list[float] | None = None):
    """E-step: fractional counts for every learned CPT entry.

    For each observation the posterior over the four latent conditions is computed
    given the telemetry *and* the outcome that followed, and each family's counts are
    incremented by that posterior mass. ``sample_weights`` optionally corrects the
    class imbalance without duplicating telemetry rows; it is used only as a
    training-time weighting and never changes the simulator data.
    """
    if sample_weights is None:
        sample_weights = [1.0] * len(samples)
    if len(sample_weights) != len(samples):
        raise ValueError("sample_weights must have one value per sample")

    counts: dict[str, dict[tuple[str, ...], list[float]]] = {
        var: {config: [0.0] * len(states_of(var))
              for config in itertools.product(*[states_of(p) for p in PARENTS[var]])}
        for var in LEARNED_VARS
    }

    latent_space = list(itertools.product(*[states_of(v) for v in LATENT_VARS]))
    cache: dict[tuple, list[float]] = {}

    for (evidence, label), sample_weight in zip(samples, sample_weights):
        key = _evidence_signature(evidence, label)
        weights = cache.get(key)

        if weights is None:
            # P(latents | e) factorises: each latent's parents are all evidence
            # variables, so given e they are independent of one another. Any parent
            # that did not report is marginalised out by variable elimination.
            marginals = {
                latent: inference.variable_elimination(net, evidence, latent)
                for latent in LATENT_VARS
            }
            prior_state = evidence.get(PRIOR_VAR)

            raw = []
            for assignment in latent_space:
                w = 1.0
                for latent, state in zip(LATENT_VARS, assignment):
                    w *= float(marginals[latent][states_of(latent).index(state)])
                if w == 0.0:
                    raw.append(0.0)
                    continue
                if prior_state is not None:
                    w *= net.probability(TARGET_VAR, label, assignment + (prior_state,))
                else:
                    # Reliability prior missing: average over it under its own prior.
                    acc = 0.0
                    prior_row = net.cpt(PRIOR_VAR)[()]
                    for pi, pstate in enumerate(states_of(PRIOR_VAR)):
                        acc += prior_row[pi] * net.probability(
                            TARGET_VAR, label, assignment + (pstate,))
                    w *= acc
                raw.append(w)

            total = sum(raw)
            weights = [w / total for w in raw] if total > 0 else [1.0 / len(raw)] * len(raw)
            cache[key] = weights

        prior_state = evidence.get(PRIOR_VAR)
        for assignment, weight in zip(latent_space, weights):
            if weight <= 0.0:
                continue
            latent_by_name = dict(zip(LATENT_VARS, assignment))

            # Latent families: parents are evidence, child is the latent.
            for latent, state in latent_by_name.items():
                config = tuple(evidence.get(p) for p in PARENTS[latent])
                if any(c is None for c in config):
                    continue  # a parent did not report; that family learns nothing
                counts[latent][config][states_of(latent).index(state)] += sample_weight * weight

            # Target family: parents are the four latents plus the reliability prior.
            if prior_state is not None:
                config = assignment + (prior_state,)
                counts[TARGET_VAR][config][TARGET_STATES.index(label)] += sample_weight * weight

    return counts


def maximise(net: BayesianNetwork, counts, concentration: float):
    """M-step: Dirichlet-smoothed renormalisation of the counted tables."""
    new_cpts = {var: {k: list(v) for k, v in table.items()}
                for var, table in net.cpts.items()}

    for var in LEARNED_VARS:
        for config, observed in counts[var].items():
            prior_row = net.cpt(var)[config]
            posterior = [concentration * p + c for p, c in zip(prior_row, observed)]
            total = sum(posterior)
            new_cpts[var][config] = [p / total for p in posterior]

    return new_cpts


def load_samples(train_csv: Path, quantiles: Path) -> list[tuple[dict[str, str], str]]:
    cuts = QuantileCuts.load(quantiles)
    samples = []
    with open(train_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            label_raw = (row.get("ground_truth_fail") or "").strip()
            if label_raw not in ("0", "1"):
                continue
            evidence = row_to_evidence(row, cuts)
            samples.append((evidence, "fail" if label_raw == "1" else "not_fail"))
    return samples


def aggregate_samples(samples: list[tuple[dict[str, str], str]],
                      weights: list[float]) -> tuple[list[tuple[dict[str, str], str]], list[float]]:
    """Collapse duplicate discretised observations without changing sufficient counts."""
    grouped: dict[tuple, float] = {}
    evidence_by_key: dict[tuple, dict[str, str]] = {}
    label_by_key: dict[tuple, str] = {}
    for (evidence, label), weight in zip(samples, weights):
        key = _evidence_signature(evidence, label)
        grouped[key] = grouped.get(key, 0.0) + weight
        evidence_by_key[key] = evidence
        label_by_key[key] = label
    keys = list(grouped)
    return ([(evidence_by_key[k], label_by_key[k]) for k in keys],
            [grouped[k] for k in keys])


def class_weights(samples: list[tuple[dict[str, str], str]],
                  target_positive_fraction: float | None) -> list[float]:
    """Return inverse-prevalence weights for rare-event training.

    The target fraction is a training objective, not a claim about deployment
    prevalence. Scaling both classes so their average weight is one keeps the
    Dirichlet concentration comparable across runs.
    """
    if target_positive_fraction is None:
        return [1.0] * len(samples)
    positive = sum(label == "fail" for _, label in samples)
    negative = len(samples) - positive
    if positive == 0 or negative == 0:
        return [1.0] * len(samples)
    observed = positive / len(samples)
    pos_weight = target_positive_fraction / observed
    neg_weight = (1.0 - target_positive_fraction) / (1.0 - observed)
    mean_weight = observed * pos_weight + (1.0 - observed) * neg_weight
    pos_weight /= mean_weight
    neg_weight /= mean_weight
    return [pos_weight if label == "fail" else neg_weight for _, label in samples]


def save_cpts(cpts, path: Path) -> None:
    serialisable = {
        var: [{"config": list(config), "row": row} for config, row in table.items()]
        for var, table in cpts.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=1)


def load_cpts(path: Path):
    with open(path) as fh:
        raw = json.load(fh)
    return {
        var: {tuple(entry["config"]): list(entry["row"]) for entry in entries}
        for var, entries in raw.items()
    }


def log_likelihood(net: BayesianNetwork, samples) -> float:
    """Mean log-likelihood of the observed outcomes, for monitoring convergence.

    Results are cached on the evidence pattern: a few thousand rows share only a few
    hundred distinct patterns, so this is far cheaper than it looks.
    """
    cache: dict[tuple, float] = {}
    total = 0.0
    for evidence, label in samples:
        key = tuple(sorted(evidence.items()))
        p_fail = cache.get(key)
        if p_fail is None:
            posterior = inference.variable_elimination(net, evidence, TARGET_VAR)
            p_fail = float(posterior[TARGET_STATES.index("fail")])
            cache[key] = p_fail
        prob = p_fail if label == "fail" else 1.0 - p_fail
        total += math.log(max(prob, 1e-12))
    return total / len(samples)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(DATA / "telemetry_train_pool.csv"),
                    help="training rows; default uses the complete scenario-level training pool")
    ap.add_argument("--quantiles", default=str(DATA / "discretisation_quantiles.json"))
    ap.add_argument("--out", default=str(DATA / "cpts_trained.json"))
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--concentration", type=float, default=5.0)
    ap.add_argument("--target-positive-fraction", type=float, default=0.40,
                    help="weighted training prevalence; use 0 to disable weighting")
    args = ap.parse_args(argv)

    train_path = Path(args.train)
    if not train_path.exists():
        sys.exit(f"{train_path} does not exist.\n"
                 f"Build the dataset first:  .venv/bin/python build_dataset.py")
    samples = load_samples(train_path, Path(args.quantiles))
    positives = sum(1 for _, label in samples if label == "fail")
    print(f"training on {len(samples)} observations from {Path(args.train).name} "
          f"({positives} positives)")
    target_fraction = args.target_positive_fraction if args.target_positive_fraction > 0 else None
    weights = class_weights(samples, target_fraction)
    samples, weights = aggregate_samples(samples, weights)
    weighted_positive = sum(w for (_, label), w in zip(samples, weights) if label == "fail")
    weighted_total = sum(weights)
    print(f"  unique discretised patterns: {len(samples)}")
    print(f"  Dirichlet concentration on the elicited tables: {args.concentration}")
    print(f"  weighted target prevalence: "
          f"{weighted_positive / weighted_total:.3f}"
          if target_fraction is not None else "  class weighting: disabled")

    net = bbn_model.build_network()
    print(f"  iteration 0  mean log-likelihood {log_likelihood(net, samples):+.5f}  (elicited)")

    for i in range(1, args.iterations + 1):
        counts = expected_counts(net, samples, weights)
        net = BayesianNetwork(cpts=maximise(net, counts, args.concentration))
        net.validate()
        print(f"  iteration {i}  mean log-likelihood {log_likelihood(net, samples):+.5f}")

    save_cpts(net.cpts, Path(args.out))
    print(f"  trained CPTs written to {args.out}")
    print()
    print("Next:  .venv/bin/python predict.py --learned ../data/cpts_trained.json")
    print("       .venv/bin/python verify.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
