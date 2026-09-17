"""
The Bayesian Belief Network over fog-node condition.

Structure is taken from Section III.B of the Review 2 paper, not invented here.
Fourteen variables in four tiers:

    evidence (8)   cpu_util, ram_util, queue_depth,
                   uplink_latency, jitter_pktloss, handover_rate,
                   residual_energy, temperature
    latent (4)     computational_stress, network_degradation,
                   energy_stress, thermal_stress
    prior (1)      reliability_prior
    target (1)     node_failure

and the edges the paper's Fig. 2 shows:

    cpu_util, ram_util, queue_depth            -> computational_stress
    uplink_latency, jitter_pktloss,
                        handover_rate          -> network_degradation
    residual_energy                            -> energy_stress
    temperature                                -> thermal_stress
    the four latent nodes + reliability_prior  -> node_failure

Every variable is ordinal on {low, medium, high} except the binary target.

Note on edge direction. The evidence tier points *into* the latent tier, which is
the opposite of the usual generative arrangement where a hidden cause produces its
symptoms. That is what the paper specifies, and it is a deliberate choice: it makes
each latent node a ranked aggregator of its indicators, which is far easier to
elicit without failure data than a generative likelihood would be. Marginalisation
over missing evidence still works exactly as Eq. (1) requires.

CPTs are not hand-written. A conditional table over three parents needs 27 rows and
the target's needs 243; tables that size cannot be elicited by hand or audited by a
reader. They are generated instead by the ranked-node method of Fenton, Neil and
Caballero -- a weighted mean of the parents' ordinal positions, dispersed by a
truncated normal -- which is the standard technique for ordinal nodes in exactly
this situation, and whose originator the paper already cites. The weights are the
elicited part, and there are only a handful of them; they are all in
`cpt_tables.py` where they can be read and argued with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ordinal state space shared by every non-target variable.
ORDINAL_STATES = ("low", "medium", "high")
TARGET_STATES = ("not_fail", "fail")

EVIDENCE_VARS = (
    "cpu_util",
    "ram_util",
    "queue_depth",
    "uplink_latency",
    "jitter_pktloss",
    "handover_rate",
    "residual_energy",
    "temperature",
)

LATENT_VARS = (
    "computational_stress",
    "network_degradation",
    "energy_stress",
    "thermal_stress",
)

PRIOR_VAR = "reliability_prior"
TARGET_VAR = "node_failure"

ALL_VARS = EVIDENCE_VARS + LATENT_VARS + (PRIOR_VAR, TARGET_VAR)

#: Parent sets, exactly as Fig. 2 of the paper draws them.
PARENTS: dict[str, tuple[str, ...]] = {
    # evidence tier is root
    "cpu_util": (),
    "ram_util": (),
    "queue_depth": (),
    "uplink_latency": (),
    "jitter_pktloss": (),
    "handover_rate": (),
    "residual_energy": (),
    "temperature": (),
    PRIOR_VAR: (),
    # latent tier
    "computational_stress": ("cpu_util", "ram_util", "queue_depth"),
    "network_degradation": ("uplink_latency", "jitter_pktloss", "handover_rate"),
    "energy_stress": ("residual_energy",),
    "thermal_stress": ("temperature",),
    # target
    TARGET_VAR: (
        "computational_stress",
        "network_degradation",
        "energy_stress",
        "thermal_stress",
        PRIOR_VAR,
    ),
}

#: Variables that contribute *inversely* to stress and risk.
#:
#: Both keep their natural reading everywhere they are seen by a human: a high
#: ``residual_energy`` means a full battery, a high ``reliability_prior`` means
#: dependable hardware. What is inverted is only their contribution to the child
#: they feed -- a full battery lowers energy stress, dependable hardware lowers
#: failure risk. That flip happens in exactly one place, when the CPTs are built in
#: ``cpt_tables.py``. The discretiser does not flip anything, so a number in the
#: CSV, the ordinal state derived from it, and the word printed in a report all
#: agree. Doing the flip in both places, which is the obvious mistake here, cancels
#: out and silently inverts the model's treatment of battery and device class.
INVERTED_VARS = frozenset({"residual_energy", PRIOR_VAR})

#: Never observed on simulated nodes. Present in the network so that it is
#: marginalised out by Eq. (1) rather than imputed -- the partial-evidence case the
#: paper builds the model to absorb.
UNOBSERVABLE_VARS = frozenset({"temperature"})


def states_of(var: str) -> tuple[str, ...]:
    """State space of a variable."""
    return TARGET_STATES if var == TARGET_VAR else ORDINAL_STATES


def topological_order() -> list[str]:
    """Variables ordered so every node follows its parents."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(v: str) -> None:
        if v in seen:
            return
        for p in PARENTS[v]:
            visit(p)
        seen.add(v)
        order.append(v)

    for v in ALL_VARS:
        visit(v)
    return order


@dataclass
class BayesianNetwork:
    """A discrete Bayesian network: a DAG plus one CPT per variable.

    ``cpts[v]`` maps a tuple of parent states (in ``PARENTS[v]`` order) to a
    probability vector over ``states_of(v)``. Root variables key on ``()``.
    """

    cpts: dict[str, dict[tuple[str, ...], list[float]]] = field(default_factory=dict)

    def parents(self, var: str) -> tuple[str, ...]:
        return PARENTS[var]

    def states(self, var: str) -> tuple[str, ...]:
        return states_of(var)

    def cpt(self, var: str) -> dict[tuple[str, ...], list[float]]:
        return self.cpts[var]

    def probability(self, var: str, value: str, parent_values: tuple[str, ...]) -> float:
        """P(var = value | parents = parent_values)."""
        row = self.cpts[var][parent_values]
        return row[self.states(var).index(value)]

    def validate(self) -> None:
        """Raises if the network is malformed. Cheap, so callers run it freely."""
        import itertools

        for var in ALL_VARS:
            if var not in self.cpts:
                raise ValueError(f"no CPT for {var}")
            parent_space = [states_of(p) for p in PARENTS[var]]
            expected = set(itertools.product(*parent_space)) if parent_space else {()}
            actual = set(self.cpts[var])
            if actual != expected:
                missing = expected - actual
                extra = actual - expected
                raise ValueError(
                    f"CPT for {var} has wrong parent configurations "
                    f"(missing {len(missing)}, unexpected {len(extra)})"
                )
            width = len(states_of(var))
            for config, row in self.cpts[var].items():
                if len(row) != width:
                    raise ValueError(f"CPT row {var}{config} has {len(row)} entries, want {width}")
                if any(p < 0 for p in row):
                    raise ValueError(f"CPT row {var}{config} has a negative probability")
                if abs(sum(row) - 1.0) > 1e-9:
                    raise ValueError(f"CPT row {var}{config} sums to {sum(row)}, want 1")


def build_network(weights=None) -> BayesianNetwork:
    """Builds the network with CPTs generated from the elicited weights."""
    from cpt_tables import build_all_cpts

    net = BayesianNetwork(cpts=build_all_cpts(weights))
    net.validate()
    return net
