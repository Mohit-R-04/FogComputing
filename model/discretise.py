"""
Turns the simulator's continuous telemetry into the ordinal evidence the BBN reads.

Cut points come in two kinds, chosen by internal validation (``select_bbn.py``):

* **tertile** -- the 33rd and 67th percentiles of the training telemetry, the paper's
  original rule ("binned into low, medium and high using quantiles of the baseline
  runs");
* **supervised** -- for each channel, the k-bin partition of a fine, tail-aware grid of
  candidate cut points that carries the most mutual information about the failure label
  (every bin holding at least 0.1% of the rows). Mutual information is a sum over bins,
  so the optimal partition is found exactly by dynamic programming. Failures live in
  the tails, so this places resolution where risk actually rises -- Fenton et al.'s
  "refine the bins in the region of interest", the upgrade path the Review 1 brief
  names. With ``per_class`` the partition is chosen separately for each device class
  (a mains-powered gateway and a battery-powered handset draw energy on scales four
  times apart). Always computed from training scenarios only.

Two rules govern this file, and both come straight from the paper.

First, cut points are learned from the training telemetry rather than fixed
thresholds, so that "high CPU utilisation" means high *for this topology*.
``build_dataset.py`` writes the tertiles to ``data/discretisation.json``; supervised
cut points are stored inside the trained model, ``data/bbn_trained.json``, so the Java
controller bins telemetry exactly as training did.

Second, a channel that did not report is left out of the evidence set. It is never
filled in with a mean, a median, or a last-known value. Imputation would put a
fabricated observation into Eq. (1) and quietly destroy the one property the whole
approach rests on: that a missing channel widens the marginalisation and blunts the
posterior, rather than silently biasing it.

Global or per device class? For tertiles, per-class cuts are worse: quantising within
a tier forces a third of that tier into each state whatever its condition, which
erases the cross-tier differences that predict failure (Phase I measured global
tertiles as the more informative encoding for every channel). Supervised cuts do not
have that defect -- they sit where the failure rate changes, not at equal frequencies --
and per class they won clearly in internal validation on the training scenarios
(cross-validated PR-AUC 0.759 against 0.653 with global supervised cuts,
``select_bbn.py``), because a mains-powered gateway and a battery-powered handset
drain energy and heat up on very different scales. ``QuantileCuts.cuts_for`` returns
the per-class cuts when the model was built with them and the global cuts otherwise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bbn_model import PAPER, Structure

#: Ordinal state of the reliability prior, by device class. Not quantised: it is a
#: declared hardware attribute rather than a changing measurement.
RELIABILITY_STATE_BY_CLASS: dict[str, str] = {
    "cloud": "high",  # redundant datacentre hardware
    "fog": "medium",  # resource-constrained fog node
    "edge": "low",    # battery-powered or user-facing edge device
}



@dataclass
class QuantileCuts:
    """Two cut points per channel, globally and (for reporting) per device class."""

    global_cuts: dict[str, tuple[float, ...]]
    by_class: dict[str, dict[str, tuple[float, ...]]] = field(default_factory=dict)
    n_snapshots: int = 0
    method: str = "tertile"
    #: Use each device class's own cut points (supervised per-class discretisation).
    per_class: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "QuantileCuts":
        with open(path) as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "QuantileCuts":
        def parse(block: dict) -> dict[str, tuple[float, ...]]:
            out = {}
            for var, v in block.items():
                if isinstance(v, dict) and "cuts" in v:
                    out[var] = tuple(float(c) for c in v["cuts"])
                elif isinstance(v, dict):
                    out[var] = (float(v["q33"]), float(v["q67"]))
                else:
                    out[var] = tuple(float(c) for c in v)
            return out

        return cls(
            global_cuts=parse(raw.get("global", {})),
            by_class={c: parse(b) for c, b in raw.get("by_device_class", {}).items()},
            n_snapshots=int(raw.get("n_snapshots", 0)),
            method=raw.get("method", "tertile"),
            per_class=bool(raw.get("per_class", False)),
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "per_class": self.per_class,
            "n_snapshots": self.n_snapshots,
            "global": {v: {"cuts": list(c)} for v, c in self.global_cuts.items()},
            "by_device_class": {cls_: {v: {"cuts": list(c)} for v, c in block.items()}
                                for cls_, block in self.by_class.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    def cuts_for(self, var: str, device_class: str | None = None,
                 per_class: bool | None = None) -> tuple[float, ...] | None:
        """Cut points for one variable, globally by default (see module docstring)."""
        if per_class is None:
            per_class = self.per_class
        if per_class and device_class and device_class in self.by_class:
            block = self.by_class[device_class]
            if var in block:
                cuts = block[var]
                if len(set(cuts)) > 1 or self.per_class:
                    return cuts
        return self.global_cuts.get(var)

    def states_of(self, var: str) -> int:
        """Number of bins a channel can fall in (largest over the classes in use)."""
        sizes = [len(self.global_cuts.get(var, (0.0, 0.0))) + 1]
        if self.per_class:
            sizes += [len(b[var]) + 1 for b in self.by_class.values() if var in b]
        return max(sizes)


def bin_index(value: float, cuts: tuple[float, ...]) -> int:
    """Bin of a value: the number of cut points at or below it.

    When every cut point is the same value (a channel constant over almost all of the
    data, such as handover rate on fixed infrastructure) everything at or below that
    constant is the lowest bin and anything above it the highest: the channel is
    reporting no stress, not missing.
    """
    if len(set(cuts)) == 1 and len(cuts) > 0:
        return len(cuts) if value > cuts[0] else 0
    return int(np.searchsorted(np.asarray(cuts), value, side="right"))


def state_name(index: int, n_states: int) -> str:
    from bbn_model import ordinal_states
    return ordinal_states(n_states)[index]


def to_state(value: float, cuts: tuple[float, ...]) -> str:
    """Places a value in its bin and returns the ordinal state name."""
    return state_name(bin_index(value, cuts), len(cuts) + 1)


def state_index(values: np.ndarray, cuts: tuple[float, ...]) -> np.ndarray:
    """Vectorised :func:`bin_index`, with -1 for NaN."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, -1, dtype=np.int8)
    ok = ~np.isnan(v)
    c = np.asarray(cuts, dtype=float)
    if len(c) and len(set(c.tolist())) == 1:
        out[ok] = np.where(v[ok] > c[0], len(c), 0).astype(np.int8)
    else:
        out[ok] = np.searchsorted(c, v[ok], side="right").astype(np.int8)
    return out


def row_to_evidence(row: dict, cuts: QuantileCuts,
                    drop: frozenset[str] = frozenset(),
                    per_class: bool = False,
                    structure: Structure = PAPER) -> dict[str, str]:
    """Builds the evidence set of one telemetry row for a given structure.

    A column that is absent, blank, or not a number is omitted from the result rather
    than imputed -- that omission is what Eq. (1) marginalises over. ``drop``
    additionally withholds named channels (robustness experiments).
    """
    device_class = row.get("device_class") or None
    evidence: dict[str, str] = {}
    state = RELIABILITY_STATE_BY_CLASS.get(device_class or "")
    if state is not None and structure.prior not in drop:
        evidence[structure.prior] = state

    for var in structure.evidence:
        if var in drop:
            continue
        raw = row.get(var)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        cut = cuts.cuts_for(var, device_class, per_class=per_class or None)
        if cut is None:
            continue
        evidence[var] = structure.states(var)[bin_index(value, cut)]
    return evidence


def missing_channels(row: dict, cuts: QuantileCuts, structure: Structure = PAPER) -> tuple[str, ...]:
    """Evidence channels this row does not carry."""
    present = set(row_to_evidence(row, cuts, structure=structure))
    return tuple(v for v in structure.evidence if v not in present)


# ---- building cut points ------------------------------------------------------------------

def tertile_cuts(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (0.0, 0.0)
    return (float(np.percentile(v, 100 / 3)), float(np.percentile(v, 200 / 3)))


def _candidates(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Candidate cut points: a fine grid of all-row quantiles, the failing rows' own
    quantiles (resolution where failures happen) and the extreme tail."""
    parts = [np.quantile(v, np.linspace(0.02, 0.98, 49)), np.quantile(v, [0.99, 0.995, 0.999])]
    if y.sum() >= 10:
        parts.append(np.quantile(v[y == 1], np.linspace(0.05, 0.95, 19)))
    return np.unique(np.concatenate(parts))


def supervised_cuts(values: np.ndarray, labels: np.ndarray, k: int = 3,
                    min_fraction: float = 0.001) -> tuple[float, ...]:
    """The partition into at most ``k`` bins with maximal mutual information on the label.

    Mutual information decomposes over bins, so dynamic programming over the candidate
    cut points finds the optimum exactly. Returns strictly increasing cut points.
    """
    v = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    ok = ~np.isnan(v)
    v, y = v[ok], y[ok]
    if len(v) == 0 or y.sum() == 0 or y.sum() == len(y):
        return tertile_cuts(v)
    cand = _candidates(v, y)
    fine = np.searchsorted(cand, v, side="right")
    n = len(cand) + 1
    neg = np.concatenate([[0.0], np.cumsum(np.bincount(fine[y == 0], minlength=n))])
    pos = np.concatenate([[0.0], np.cumsum(np.bincount(fine[y == 1], minlength=n))])
    total = float(len(v))
    p1 = y.mean()
    p0 = 1.0 - p1

    def gain(i: int, j: int) -> float:
        n0 = neg[j] - neg[i]
        n1 = pos[j] - pos[i]
        if n0 + n1 < min_fraction * total:
            return -np.inf
        a, b = n0 + 0.5, n1 + 0.5          # Jeffreys smoothing
        m = (a + b) / total
        return (a / total) * np.log((a / total) / (m * p0)) + (b / total) * np.log((b / total) / (m * p1))

    best = np.full((k + 1, n + 1), -np.inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for bins in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(bins - 1, j):
                if best[bins - 1, i] == -np.inf:
                    continue
                g = best[bins - 1, i] + gain(i, j)
                if g > best[bins, j]:
                    best[bins, j], back[bins, j] = g, i
    usable = [b for b in range(1, k + 1) if best[b, n] > -np.inf]
    if not usable:
        return tertile_cuts(v)
    bins = max(usable, key=lambda b: best[b, n])
    cuts, j = [], n
    for b in range(bins, 0, -1):
        i = back[b, j]
        if i > 0:
            cuts.append(float(cand[i - 1]))
        j = i
    return tuple(sorted(cuts))


def build_cuts(columns: dict[str, np.ndarray], labels: np.ndarray | None,
               method: str, device_class: np.ndarray | None = None,
               states: int = 3, per_class: bool = False,
               min_class_positives: int = 20) -> QuantileCuts:
    """Cut points for every channel, from training rows only.

    ``per_class`` chooses a supervised partition separately for each device class that
    has at least ``min_class_positives`` failures; other classes (the cloud never fails)
    use the global partition.
    """
    if method not in ("tertile", "supervised"):
        raise ValueError(f"unknown discretisation method {method!r}")
    global_cuts = {}
    for var, values in columns.items():
        global_cuts[var] = (supervised_cuts(values, labels, states) if method == "supervised"
                            else tertile_cuts(values))
    by_class: dict[str, dict[str, tuple[float, ...]]] = {}
    if device_class is not None:
        for cls_ in sorted(set(device_class)):
            mask = device_class == cls_
            block = {}
            for var, values in columns.items():
                if (per_class and method == "supervised" and labels is not None
                        and labels[mask].sum() >= min_class_positives):
                    block[var] = supervised_cuts(values[mask], labels[mask], states)
                else:
                    block[var] = global_cuts[var] if per_class else tertile_cuts(values[mask])
            by_class[cls_] = block
    n = len(next(iter(columns.values()))) if columns else 0
    return QuantileCuts(global_cuts=global_cuts, by_class=by_class, n_snapshots=n,
                        method=method + ("-per-class" if per_class else ""), per_class=per_class)
