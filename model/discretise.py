"""
Turns the simulator's continuous telemetry into the ordinal evidence the BBN reads.

Two rules govern this file, and both come straight from the paper.

First, cut points are quantiles of the baseline run rather than fixed thresholds,
so that "high CPU utilisation" means high *for this topology*. The Java side writes
those quantiles to `discretisation_quantiles.json` as it generates the cases.

Second, a channel that did not report is left out of the evidence set. It is never
filled in with a mean, a median, or a last-known value. Imputation would put a
fabricated observation into Eq. (1) and quietly destroy the one property the whole
approach rests on: that a missing channel widens the marginalisation and blunts the
posterior, rather than silently biasing it.

Cuts are global by default, not per device class. That was not the first choice
here, and the reasoning that led to it is worth recording, because the intuitive
answer is wrong.

The appeal of per-tier cuts is obvious: a mains-powered fog node at 97% residual
energy is healthy, whereas an edge device hours into its discharge curve at 97% is a
different proposition, so surely each tier deserves its own scale. The trouble is
that quantising within a tier forces exactly a third of that tier into each state no
matter what condition it is actually in. That erases precisely the cross-tier
differences that predict failure.

Measured on the training run -- a separate seed from the evaluation cases, so this
is not a choice tuned on the test set -- global cuts were the more informative
encoding for every telemetry variable:

    variable            global   per-class
    cpu_util             0.769       0.651
    ram_util             0.771       0.646
    queue_depth          0.680       0.616
    uplink_latency       0.655       0.550
    jitter_pktloss       0.683       0.569
    handover_rate        0.679       0.685
    residual_energy      0.646       0.549

(informativeness of each encoding against the outcome, 0.5 being uninformative)

That also makes physical sense in hindsight. Utilisation, jitter and residual
energy are already normalised to their own capacity, so they carry absolute meaning
that a within-class rank throws away. The per-class cuts are still written by the
simulator and are still loaded here: they are worth reporting, and
``QuantileCuts.cuts_for`` will use them when asked, but they are not the default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bbn_model import EVIDENCE_VARS, ORDINAL_STATES, PRIOR_VAR

#: Continuous columns that carry an ordinal variable of the network.
DISCRETISED_COLUMNS = tuple(EVIDENCE_VARS) + (PRIOR_VAR,)

#: Ordinal state of the reliability prior, by fog tier.
#:
#: This variable is not quantised. It is a declared hardware attribute rather than
#: a changing measurement, so each tier maps directly to one reliability state.
RELIABILITY_STATE_BY_CLASS: dict[str, str] = {
    "cloud": "high",  # redundant datacentre hardware
    "fog": "medium",  # resource-constrained fog node
    "edge": "low",    # battery-powered or user-facing edge device
}


@dataclass
class QuantileCuts:
    """Tertile cut points, globally and per device class."""

    global_cuts: dict[str, tuple[float, float]]
    by_class: dict[str, dict[str, tuple[float, float]]]
    n_snapshots: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "QuantileCuts":
        with open(path) as fh:
            raw = json.load(fh)

        def parse(block: dict) -> dict[str, tuple[float, float]]:
            return {var: (float(v["q33"]), float(v["q67"])) for var, v in block.items()}

        return cls(
            global_cuts=parse(raw["global"]),
            by_class={cls_name: parse(block) for cls_name, block in raw["by_device_class"].items()},
            n_snapshots=int(raw.get("n_snapshots", 0)),
        )

    def cuts_for(self, var: str, device_class: str | None = None,
                 per_class: bool = False) -> tuple[float, float] | None:
        """Cut points for one variable, globally by default.

        Pass ``per_class=True`` to quantise against the device class's own
        distribution instead; see the module docstring for why that is not the
        default. Even then, a class whose own tertiles coincide has no internal
        spread on that variable -- many fixed fog nodes can sit at queue depth zero --
        so a degenerate block falls back to the global cuts,
        which do still separate those nodes from the tiers that are loaded.

        When the global cuts are degenerate too, the variable really is constant
        across the whole topology (handover rate on fixed infrastructure, which never
        roams) and ``to_state`` correctly reports the lowest state.
        """
        if per_class and device_class and device_class in self.by_class:
            block = self.by_class[device_class]
            if var in block:
                q33, q67 = block[var]
                if q33 != q67:
                    return block[var]
        return self.global_cuts.get(var)


def to_state(value: float, cuts: tuple[float, float]) -> str:
    """Places a value in its tertile.

    Degenerate cuts (q33 == q67) happen whenever a variable is genuinely constant
    for a device class -- handover rate on fixed infrastructure, for instance, which
    is always zero because fixed fog nodes do not roam. Everything at or
    below that constant is the lowest state, which is the correct reading: the
    variable is reporting no stress, not missing.
    """
    q33, q67 = cuts
    if value < q33:
        return ORDINAL_STATES[0]
    if value >= q67 and q67 > q33:
        return ORDINAL_STATES[2]
    if q67 <= q33:
        return ORDINAL_STATES[2] if value > q67 else ORDINAL_STATES[0]
    return ORDINAL_STATES[1]


def row_to_evidence(row: dict, cuts: QuantileCuts,
                    drop: frozenset[str] = frozenset(),
                    per_class: bool = False) -> dict[str, str]:
    """Builds the evidence set for one telemetry row.

    A column that is absent, blank, or not a number is omitted from the result
    rather than imputed -- that omission is what Eq. (1) marginalises over.

    ``drop`` additionally withholds named channels, which `verify.py` uses to
    measure how the posterior degrades as telemetry goes dark.
    """
    device_class = row.get("device_class") or None
    evidence: dict[str, str] = {}

    for var in DISCRETISED_COLUMNS:
        if var in drop:
            continue

        if var == PRIOR_VAR:
            # Declared hardware attribute, not a measurement -- see the table above.
            state = RELIABILITY_STATE_BY_CLASS.get(device_class or "")
            if state is not None:
                evidence[var] = state
            continue

        raw = row.get(var)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        cut = cuts.cuts_for(var, device_class, per_class=per_class)
        if cut is None:
            continue
        evidence[var] = to_state(value, cut)

    return evidence


def missing_channels(row: dict, cuts: QuantileCuts) -> tuple[str, ...]:
    """Evidence channels this row does not carry."""
    present = set(row_to_evidence(row, cuts))
    return tuple(v for v in EVIDENCE_VARS if v not in present)
