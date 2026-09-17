"""
Conditional probability tables, generated from elicited weights.

Everything a reader needs to argue with is at the top of this file: about twenty
numbers. The tables themselves (27 rows for each three-parent aggregator, 243 for
the target) are derived from those numbers by the ranked-node method, so there are
no unexplained magic entries anywhere in the network.

Ranked nodes in one paragraph. Fenton, Neil and Caballero's method maps each
ordinal state to a point on [0, 1] -- low to 1/6, medium to 1/2, high to 5/6, the
midpoints of the three equal intervals. For a given parent configuration it takes a
weighted mean of the parents' points, places a normal distribution of standard
deviation sigma at that mean truncated to [0, 1], and integrates it over the three
intervals to get the child's probability row. A small sigma makes the child track
its parents tightly; a large one admits that the aggregation is uncertain. That is
the whole method, and it is the standard way to fill an ordinal CPT when no failure
trace exists to learn one from.

Sources for the weights. Relative ordering follows the fog fault-tolerance
literature the paper surveys: Rajab and Younis trigger on CPU and memory, so
compute stress dominates; Gupta and Singh's FTAPA scores a composite in which
network health matters but less; battery state is the distinctive failure mode of
the user tier. Absolute values are elicited, and `train_bbn.py` retrains them from
observed outcomes afterwards.
"""

from __future__ import annotations

import itertools
import math

from bbn_model import (
    ALL_VARS,
    EVIDENCE_VARS,
    INVERTED_VARS,
    ORDINAL_STATES,
    PARENTS,
    PRIOR_VAR,
    TARGET_STATES,
    TARGET_VAR,
    states_of,
)

# --------------------------------------------------------------------------
# Elicited parameters -- the whole of the model's prior knowledge
# --------------------------------------------------------------------------

#: Relative influence of each parent on its latent aggregator.
AGGREGATOR_WEIGHTS: dict[str, dict[str, float]] = {
    "computational_stress": {
        # Saturated compute is the most direct precursor of a node being unable to
        # host its modules; memory pressure follows; backlog is a consequence of
        # both as much as a cause, so it carries the least weight.
        "cpu_util": 0.45,
        "ram_util": 0.35,
        "queue_depth": 0.20,
    },
    "network_degradation": {
        # Jitter and loss say more about a link that is about to become unusable
        # than raw latency does, which may simply reflect distance.
        "uplink_latency": 0.30,
        "jitter_pktloss": 0.45,
        "handover_rate": 0.25,
    },
    "energy_stress": {"residual_energy": 1.0},
    "thermal_stress": {"temperature": 1.0},
}

#: Dispersion of each aggregator. Single-parent nodes are near-deterministic
#: restatements of their one indicator, so they get a tight sigma; the
#: three-parent nodes are genuine judgements and get a looser one.
AGGREGATOR_SIGMA: dict[str, float] = {
    "computational_stress": 0.22,
    "network_degradation": 0.24,
    "energy_stress": 0.12,
    "thermal_stress": 0.12,
}

#: Contribution of each parent of the target to overall failure risk.
TARGET_WEIGHTS: dict[str, float] = {
    "computational_stress": 0.30,
    "network_degradation": 0.20,
    "energy_stress": 0.25,
    "thermal_stress": 0.10,
    # Enters inverted: an unreliable device class raises risk.
    PRIOR_VAR: 0.15,
}

#: Failure probability of a node under no stress at all on a reliable device.
#: Non-zero because hardware fails without warning.
BASE_FAILURE_PROB = 0.02

#: Failure probability of a node with every indicator at its worst.
MAX_FAILURE_PROB = 0.88

#: Curvature of the risk response. Above 1 the risk stays low through the
#: mid-range and climbs steeply near the top, which is how degradation actually
#: behaves: a node at 60% utilisation is not twice as likely to fail as one at 30%.
RISK_EXPONENT = 2.4

#: Marginal distribution of each root evidence variable. Discretisation is by
#: tertiles of the baseline run, so each state holds a third of the mass by
#: construction -- this is a consequence of the design, not a guess.
EVIDENCE_PRIOR = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

#: Reliability prior of a node drawn at random from a fog topology. Skewed toward
#: the reliable end because most deployed capacity is infrastructure rather than
#: battery-powered user hardware. Always observed in practice -- the device class
#: is known -- so this row has little influence on any posterior.
RELIABILITY_PRIOR_MARGINAL = [0.25, 0.30, 0.45]


# --------------------------------------------------------------------------
# Ranked-node machinery
# --------------------------------------------------------------------------

#: Midpoints of the three equal sub-intervals of [0, 1].
STATE_POINTS: dict[str, float] = {"low": 1.0 / 6.0, "medium": 0.5, "high": 5.0 / 6.0}

#: Boundaries of those sub-intervals.
STATE_BOUNDS: dict[str, tuple[float, float]] = {
    "low": (0.0, 1.0 / 3.0),
    "medium": (1.0 / 3.0, 2.0 / 3.0),
    "high": (2.0 / 3.0, 1.0),
}


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def truncated_normal_row(mean: float, sigma: float) -> list[float]:
    """Probability of each ordinal state under TNormal(mean, sigma) on [0, 1]."""
    lo = _normal_cdf(0.0, mean, sigma)
    hi = _normal_cdf(1.0, mean, sigma)
    mass = hi - lo
    if mass <= 1e-12:
        # Degenerate: the distribution sits entirely outside [0,1]. Fall back to
        # putting everything on the nearest state.
        row = [0.0, 0.0, 0.0]
        row[0 if mean < 0.5 else 2] = 1.0
        return row

    row = []
    for state in ORDINAL_STATES:
        a, b = STATE_BOUNDS[state]
        row.append((_normal_cdf(b, mean, sigma) - _normal_cdf(a, mean, sigma)) / mass)

    total = sum(row)
    return [p / total for p in row]


def contribution_point(parent: str, value: str) -> float:
    """The point on [0, 1] that `parent` in state `value` contributes to its child.

    Variables in ``INVERTED_VARS`` are flipped here and only here: a full battery
    (``residual_energy = high``) contributes a *low* point to energy stress, and
    dependable hardware (``reliability_prior = high``) contributes a low point to
    failure risk.
    """
    point = STATE_POINTS[value]
    return 1.0 - point if parent in INVERTED_VARS else point


def weighted_mean(parent_values: tuple[str, ...], parents: tuple[str, ...],
                  weights: dict[str, float]) -> float:
    """Weighted mean of the parents' contribution points."""
    total_w = sum(weights[p] for p in parents)
    return sum(weights[p] * contribution_point(p, v)
               for p, v in zip(parents, parent_values)) / total_w


def build_aggregator_cpt(var: str, weights: dict[str, dict[str, float]],
                         sigmas: dict[str, float]) -> dict[tuple[str, ...], list[float]]:
    """CPT of one latent aggregator over its parents."""
    parents = PARENTS[var]
    w = weights[var]
    sigma = sigmas[var]
    table: dict[tuple[str, ...], list[float]] = {}
    for config in itertools.product(*[states_of(p) for p in parents]):
        table[config] = truncated_normal_row(weighted_mean(config, parents, w), sigma)
    return table


def build_target_cpt(weights: dict[str, float],
                     base: float = BASE_FAILURE_PROB,
                     top: float = MAX_FAILURE_PROB,
                     exponent: float = RISK_EXPONENT) -> dict[tuple[str, ...], list[float]]:
    """CPT of the target over the four latent conditions and the reliability prior.

    The risk score is a weighted mean of the parents' ordinal points, with the
    reliability prior flipped so that a *less* reliable class pushes the score up.
    That score is then mapped through a convex response onto [base, top].

    The mean is rescaled by the largest point an ordinal state can take (5/6, the
    midpoint of the "high" interval). Without that rescaling the worst possible
    configuration scores only 0.83 and the elicited ``top`` is never reached, so the
    parameter would not mean what it says and the model would be systematically
    under-confident in exactly the configurations that matter most.
    """
    parents = PARENTS[TARGET_VAR]
    total_w = sum(weights[p] for p in parents)
    max_point = max(STATE_POINTS.values())

    table: dict[tuple[str, ...], list[float]] = {}
    for config in itertools.product(*[states_of(p) for p in parents]):
        score = sum(weights[parent] * contribution_point(parent, value)
                    for parent, value in zip(parents, config))
        score /= total_w * max_point

        p_fail = base + (top - base) * (score ** exponent)
        p_fail = min(max(p_fail, 1e-6), 1.0 - 1e-6)
        # TARGET_STATES is ("not_fail", "fail"), so order the row to match.
        table[config] = [1.0 - p_fail, p_fail]
    return table


def build_all_cpts(weights=None) -> dict[str, dict[tuple[str, ...], list[float]]]:
    """Builds every CPT in the network.

    ``weights`` optionally overrides the elicited parameters; ``train_bbn.py``
    uses it to hand back tables refined from observed outcomes.
    """
    if weights is not None:
        return weights

    cpts: dict[str, dict[tuple[str, ...], list[float]]] = {}

    for var in EVIDENCE_VARS:
        cpts[var] = {(): list(EVIDENCE_PRIOR)}
    cpts[PRIOR_VAR] = {(): list(RELIABILITY_PRIOR_MARGINAL)}

    for var in AGGREGATOR_WEIGHTS:
        cpts[var] = build_aggregator_cpt(var, AGGREGATOR_WEIGHTS, AGGREGATOR_SIGMA)

    cpts[TARGET_VAR] = build_target_cpt(TARGET_WEIGHTS)

    missing = set(ALL_VARS) - set(cpts)
    if missing:
        raise AssertionError(f"CPTs not generated for {sorted(missing)}")
    return cpts


def describe() -> str:
    """Human-readable dump of the elicited parameters, for the report."""
    lines = ["Elicited BBN parameters", "=" * 40, "", "Aggregator weights:"]
    for var, w in AGGREGATOR_WEIGHTS.items():
        parts = ", ".join(f"{k}={v:.2f}" for k, v in w.items())
        lines.append(f"  {var:22s} sigma={AGGREGATOR_SIGMA[var]:.2f}  {parts}")
    lines += ["", "Target weights:"]
    for k, v in TARGET_WEIGHTS.items():
        lines.append(f"  {k:22s} {v:.2f}")
    lines += [
        "",
        f"Base failure probability : {BASE_FAILURE_PROB}",
        f"Max failure probability  : {MAX_FAILURE_PROB}",
        f"Risk exponent            : {RISK_EXPONENT}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
