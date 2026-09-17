"""
Exact inference over the fog-node BBN.

This implements Eq. (1) of the Review 2 paper,

    P(F = fail | e)  =  [ sum_u prod_V P(V | pa(V)) ]  /  P(e)

in the two ways that equation can be read.

`enumerate_posterior` reads it literally: walk every joint assignment of the
unobserved variables, multiply the fourteen local conditional probabilities, add up
the results. It is transparently the equation on the page and correspondingly slow.

`variable_elimination` computes the same quantity by pushing each summation as far
inside the product as it will go. The graph is sparse -- no node has more than five
parents -- so this finishes in milliseconds.

The engine used in anger is variable elimination; enumeration exists so that the
fast path can be checked against a direct transcription of the published equation.
`test_inference.py` asserts they agree.

Missing evidence needs no special handling in either. A telemetry channel that did
not report is simply not in `evidence`, so it stays in the summation and is
marginalised out -- which is the whole reason the paper chose a Bayesian network
over a threshold rule. No imputation happens anywhere in this file.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from bbn_model import (
    ALL_VARS,
    EVIDENCE_VARS,
    PARENTS,
    TARGET_VAR,
    BayesianNetwork,
    states_of,
    topological_order,
)


class Factor:
    """A function from an assignment of `variables` to a non-negative number.

    Values are held in an ndarray whose axes follow `variables` in order.

    The factor carries its own state names rather than looking them up from the
    network. That keeps it a general-purpose object: the correctness tests build
    small hand-checkable factors over variables that are not part of the fog
    network at all, and a factor that consulted a global state table could not
    represent them.
    """

    __slots__ = ("variables", "values", "state_names")

    def __init__(self, variables: tuple[str, ...], values: np.ndarray,
                 state_names: dict[str, tuple[str, ...]] | None = None):
        if values.ndim != len(variables):
            raise ValueError(
                f"factor over {variables} needs {len(variables)} axes, got {values.ndim}")

        names: dict[str, tuple[str, ...]] = {}
        for axis, var in enumerate(variables):
            if state_names and var in state_names:
                names[var] = state_names[var]
            elif var in ALL_VARS:
                names[var] = states_of(var)
            else:
                names[var] = tuple(str(i) for i in range(values.shape[axis]))
            if len(names[var]) != values.shape[axis]:
                raise ValueError(
                    f"{var} has {len(names[var])} states but axis {axis} "
                    f"has length {values.shape[axis]}")

        self.variables = variables
        self.values = values
        self.state_names = names

    def states(self, var: str) -> tuple[str, ...]:
        return self.state_names[var]

    def _merged_names(self, other: "Factor") -> dict[str, tuple[str, ...]]:
        merged = dict(self.state_names)
        merged.update(other.state_names)
        return merged

    @classmethod
    def from_cpt(cls, net: BayesianNetwork, var: str) -> "Factor":
        """The factor phi(var, parents(var)) = P(var | parents(var))."""
        parents = PARENTS[var]
        variables = parents + (var,)
        shape = tuple(len(states_of(v)) for v in variables)
        values = np.zeros(shape)

        parent_spaces = [states_of(p) for p in parents]
        for config in itertools.product(*parent_spaces) if parents else [()]:
            idx = tuple(states_of(p).index(v) for p, v in zip(parents, config))
            values[idx] = net.cpt(var)[config]
        return cls(variables, values)

    def restrict(self, evidence: dict[str, str]) -> "Factor":
        """Slices out the observed values of any evidence variables in scope."""
        keep: list[str] = []
        index: list[object] = []
        for v in self.variables:
            if v in evidence:
                index.append(self.states(v).index(evidence[v]))
            else:
                index.append(slice(None))
                keep.append(v)
        return Factor(tuple(keep), self.values[tuple(index)], self.state_names)

    def multiply(self, other: "Factor") -> "Factor":
        """Pointwise product, broadcasting over the union of the two scopes."""
        merged = list(self.variables)
        for v in other.variables:
            if v not in merged:
                merged.append(v)
        merged_t = tuple(merged)
        names = self._merged_names(other)

        a = self._broadcast_to(merged_t, names)
        b = other._broadcast_to(merged_t, names)
        return Factor(merged_t, a * b, names)

    def _broadcast_to(self, target: tuple[str, ...],
                      names: dict[str, tuple[str, ...]]) -> np.ndarray:
        """Reshapes this factor's array so it broadcasts against `target`'s axes."""
        source_axis = {v: i for i, v in enumerate(self.variables)}
        perm = [source_axis[v] for v in target if v in source_axis]
        arr = np.transpose(self.values, perm) if perm else self.values
        shape = [len(names[v]) if v in source_axis else 1 for v in target]
        return arr.reshape(shape)

    def sum_out(self, var: str) -> "Factor":
        """Marginalises `var` out of this factor."""
        if var not in self.variables:
            return self
        axis = self.variables.index(var)
        keep = tuple(v for v in self.variables if v != var)
        return Factor(keep, self.values.sum(axis=axis), self.state_names)

    def normalised(self) -> np.ndarray:
        total = self.values.sum()
        if total <= 0:
            raise ZeroDivisionError("evidence has zero probability under this network")
        return self.values / total


@dataclass
class InferenceResult:
    """Posterior over the target, plus the explanation that goes with it."""

    p_fail: float
    #: Posterior over each latent condition, for the causal chain.
    latent_posteriors: dict[str, dict[str, float]]
    #: Latent condition the evidence most implicates, or "none" when no condition
    #: is raised above its prior.
    top_cause: str
    #: Its posterior probability of being in the "high" state.
    top_cause_strength: float
    #: How far that posterior rose above the same condition's prior. Zero means the
    #: evidence said nothing about it.
    top_cause_lift: float
    #: Evidence channels that did not report and were marginalised out. The latent
    #: conditions are unobserved by design and are not listed here -- only genuinely
    #: missing telemetry is, since that is what the paper's partial-evidence claim
    #: is about.
    marginalised: tuple[str, ...]

    def explain(self) -> str:
        """One sentence of the form the paper asks for."""
        if self.top_cause == "none":
            head = f"P(fail)={self.p_fail:.3f}; no condition raised above its prior"
        else:
            head = (
                f"P(fail)={self.p_fail:.3f}; principal cause {self.top_cause} "
                f"(P(high)={self.top_cause_strength:.3f}, "
                f"lift {self.top_cause_lift:+.3f})"
            )
        tail = f"; marginalised {', '.join(self.marginalised)}" if self.marginalised else ""
        return head + tail


def _validate_evidence(evidence: dict[str, str]) -> None:
    for var, value in evidence.items():
        if var not in ALL_VARS:
            raise KeyError(f"{var} is not a variable of this network")
        if var == TARGET_VAR:
            raise ValueError("the target cannot be given as evidence")
        if value not in states_of(var):
            raise ValueError(f"{value!r} is not a state of {var}; expected {states_of(var)}")


def variable_elimination(net: BayesianNetwork, evidence: dict[str, str],
                         query: str = TARGET_VAR) -> np.ndarray:
    """Posterior over `query` given `evidence`, by variable elimination.

    Returns a probability vector over ``states_of(query)``.
    """
    _validate_evidence(evidence)

    factors = [Factor.from_cpt(net, v).restrict(evidence) for v in ALL_VARS]

    # Eliminate everything that is neither queried nor observed. Ordering by the
    # size of the factor a variable appears in is a cheap, effective heuristic;
    # on a graph this small any valid order is fast, but a sensible one keeps the
    # intermediate arrays small.
    to_eliminate = [v for v in topological_order()
                    if v != query and v not in evidence]
    to_eliminate.sort(key=lambda v: len(PARENTS[v]))

    for var in to_eliminate:
        involved = [f for f in factors if var in f.variables]
        if not involved:
            continue
        factors = [f for f in factors if var not in f.variables]
        product = involved[0]
        for f in involved[1:]:
            product = product.multiply(f)
        factors.append(product.sum_out(var))

    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)

    # Any leftover axes are singleton; squeeze down to the query axis.
    if result.variables != (query,):
        for v in result.variables:
            if v != query:
                result = result.sum_out(v)
    return result.normalised()


def enumerate_posterior(net: BayesianNetwork, evidence: dict[str, str],
                        query: str = TARGET_VAR) -> np.ndarray:
    """Posterior over `query`, computed by direct enumeration of Eq. (1).

    Kept as the correctness oracle for `variable_elimination`, and as a literal
    transcription of the equation as the paper writes it. Exponential in the number
    of unobserved variables, so it is not the production path.
    """
    _validate_evidence(evidence)

    hidden = [v for v in ALL_VARS if v != query and v not in evidence]
    totals = np.zeros(len(states_of(query)))

    for qi, q_value in enumerate(states_of(query)):
        acc = 0.0
        for assignment in itertools.product(*[states_of(v) for v in hidden]):
            world = dict(evidence)
            world[query] = q_value
            world.update(dict(zip(hidden, assignment)))

            # prod_V P(V | pa(V)) over all fourteen variables
            joint = 1.0
            for var in ALL_VARS:
                parent_values = tuple(world[p] for p in PARENTS[var])
                joint *= net.probability(var, world[var], parent_values)
                if joint == 0.0:
                    break
            acc += joint
        totals[qi] = acc

    total = totals.sum()
    if total <= 0:
        raise ZeroDivisionError("evidence has zero probability under this network")
    return totals / total


def infer(net: BayesianNetwork, evidence: dict[str, str]) -> InferenceResult:
    """Full prediction for one node: posterior, causal explanation, and what was missing.

    The explanation is diagnostic reasoning, not a post-hoc rationalisation. Each
    latent condition's posterior is recomputed under the same evidence and compared
    with that condition's prior; the one whose posterior the evidence has raised
    furthest, weighted by how much the target CPT lets it move the risk, is named.

    Ranking by raw posterior instead would be wrong, and visibly so. Temperature is
    never observed, so thermal stress always sits at its flat prior of one third --
    which on a perfectly healthy node is higher than the other three conditions, and
    a bare-posterior ranking duly named thermal stress as the principal cause of a
    node with nothing wrong with it. Ranking by lift above the prior cannot do that:
    an unobserved condition has zero lift and is never named.
    """
    from cpt_tables import TARGET_WEIGHTS

    posterior = variable_elimination(net, evidence, TARGET_VAR)
    p_fail = float(posterior[states_of(TARGET_VAR).index("fail")])

    # A lift has to clear floating-point noise to count as the evidence saying
    # something; otherwise an unobserved condition wins on a 1e-17 rounding error.
    lift_epsilon = 1e-6

    latent_posteriors: dict[str, dict[str, float]] = {}
    best_cause, best_score, best_strength, best_lift = "none", lift_epsilon, 0.0, 0.0

    for latent in ("computational_stress", "network_degradation",
                   "energy_stress", "thermal_stress"):
        dist = variable_elimination(net, evidence, latent)
        latent_posteriors[latent] = {
            state: float(p) for state, p in zip(states_of(latent), dist)
        }
        prior = variable_elimination(net, {}, latent)
        prior_high = float(prior[states_of(latent).index("high")])

        p_high = latent_posteriors[latent]["high"]
        lift = p_high - prior_high
        score = lift * TARGET_WEIGHTS[latent]
        if score > best_score:
            best_cause, best_score = latent, score
            best_strength, best_lift = p_high, lift

    marginalised = tuple(v for v in EVIDENCE_VARS if v not in evidence)

    return InferenceResult(
        p_fail=p_fail,
        latent_posteriors=latent_posteriors,
        top_cause=best_cause,
        top_cause_strength=best_strength,
        top_cause_lift=best_lift,
        marginalised=marginalised,
    )
