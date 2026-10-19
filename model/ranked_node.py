"""
Learned ranked nodes: the paper's ranked-node CPTs with parameters learned by EM.

Fenton, Neil and Caballero's ranked-node method is how the project elicits the latent
tables (``cpt_tables.py``): each parent state maps to a point on [0, 1], the child's
score is the weighted sum of its parents' points, and a normal distribution around the
score, cut at the child's state boundaries, gives the row. Elicited, the points are
fixed midpoints and only a handful of weights are chosen by hand.

Here the same form is *learned*: every (device class, parent, parent state) has its own
point, initialised at the elicited value and re-estimated in each EM M-step by maximum a
posteriori fitting to the expected counts, with a Gaussian prior that pulls the points
back towards the elicited ones (the parametric counterpart of the Dirichlet prior on the
target table).

Why it matters. A free table over five parents with eight states each needs 32,768 rows
per device class; almost none would ever be visited by a failing node, so a finely
discretised evidence tier was unlearnable and the evidence had to stay at three coarse
states -- which turned out to discard most of the information in the telemetry (a
gradient-boosting model given only those three states did no better than the BBN). A
ranked node needs one number per parent state, so the evidence tier can be fine while
the latent conditions keep the paper's {low, medium, high} and the network keeps its
structure and its explanations. This is the "compact CPT parameterisation, linear rather
than exponential in the parents" that the Review 1 brief names as the remedy for table
growth.

    P(L = l | x, c) = Phi((b_{l+1} - s) / sigma) - Phi((b_l - s) / sigma),
    s = sum_i theta[c, i, x_i]

with fixed boundaries b (1/3 and 2/3 for three states) and sigma from the elicitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from bbn_model import INVERTED_VARS, Structure
from cpt_tables import AGGREGATOR_SIGMA, aggregator_weights


@dataclass
class RankedNode:
    var: str
    parents: tuple[str, ...]        # evidence parents, in the structure's order
    cards: tuple[int, ...]
    n_classes: int                  # 3 if the device class is a parent, else 1
    sigma: float
    bounds: np.ndarray              # interior state boundaries of the child
    theta: np.ndarray               # (n_classes, sum(cards))
    theta0: np.ndarray              # elicited points (prior mean)

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.cards)])

    @classmethod
    def elicited(cls, var: str, structure: Structure) -> "RankedNode":
        parents = tuple(p for p in structure.parents[var] if p != structure.prior)
        cards = tuple(structure.card(p) for p in parents)
        w = aggregator_weights(var, structure)
        total = sum(w[p] for p in parents)
        pts = []
        for p, k in zip(parents, cards):
            mid = (np.arange(k) + 0.5) / k
            if p in INVERTED_VARS:
                mid = 1.0 - mid
            pts.append(w[p] / total * mid)
        base = np.concatenate(pts)
        n_classes = 3 if structure.prior in structure.parents[var] else 1
        k_child = structure.card(var)
        bounds = np.arange(1, k_child) / k_child
        theta = np.tile(base, (n_classes, 1))
        return cls(var, parents, cards, n_classes, AGGREGATOR_SIGMA[var], bounds, theta.copy(), theta.copy())

    # ---- table -------------------------------------------------------------------------

    def _scores(self, c: int) -> np.ndarray:
        s = np.zeros(self.cards)
        offs = self.offsets
        for i, k in enumerate(self.cards):
            shape = [1] * len(self.cards)
            shape[i] = k
            s = s + self.theta[c, offs[i]:offs[i + 1]].reshape(shape)
        return s

    def _rows(self, s: np.ndarray) -> np.ndarray:
        edges = np.concatenate([[-np.inf], self.bounds, [np.inf]])
        z = (edges[None, :] - s.reshape(-1, 1)) / self.sigma
        cdf = norm.cdf(z)
        p = np.clip(cdf[:, 1:] - cdf[:, :-1], 1e-12, None)
        p /= p.sum(axis=1, keepdims=True)
        return p.reshape(s.shape + (len(self.bounds) + 1,))

    def table(self) -> np.ndarray:
        """CPT array: evidence parent axes, then the device-class axis if any, then the child."""
        per_class = [self._rows(self._scores(c)) for c in range(self.n_classes)]
        if self.n_classes == 1:
            return per_class[0]
        return np.stack(per_class, axis=len(self.cards))

    # ---- M-step ------------------------------------------------------------------------

    def fit(self, counts: np.ndarray, lam: float, maxiter: int = 60) -> None:
        """MAP update of the points from expected counts shaped like :meth:`table`."""
        edges = np.concatenate([[-np.inf], self.bounds, [np.inf]])
        offs = self.offsets
        k_child = len(self.bounds) + 1
        for c in range(self.n_classes):
            cnt = counts if self.n_classes == 1 else np.take(counts, c, axis=len(self.cards))
            flat = cnt.reshape(-1, k_child)
            nz = flat.sum(axis=1) > 1e-12
            if not nz.any():
                continue
            idx = np.array(np.unravel_index(np.nonzero(nz)[0], self.cards)).T
            C = flat[nz]
            prior = self.theta0[c]
            reg = lam * C.sum() / len(prior)

            def objective(x):
                s = np.zeros(len(idx))
                for i in range(len(self.cards)):
                    s += x[offs[i] + idx[:, i]]
                z = (edges[None, :] - s[:, None]) / self.sigma
                cdf = norm.cdf(z)
                pdf = norm.pdf(z)
                pdf[~np.isfinite(z)] = 0.0
                pr = np.clip(cdf[:, 1:] - cdf[:, :-1], 1e-12, 1.0)
                ll = float((C * np.log(pr)).sum())
                dlds = (C * (-(pdf[:, 1:] - pdf[:, :-1]) / self.sigma) / pr).sum(axis=1)
                grad = np.zeros_like(x)
                for i in range(len(self.cards)):
                    np.add.at(grad, offs[i] + idx[:, i], dlds)
                penalty = 0.5 * reg * float(((x - prior) ** 2).sum())
                return -(ll - penalty), -(grad - reg * (x - prior))

            result = minimize(objective, self.theta[c], jac=True, method="L-BFGS-B",
                              options={"maxiter": maxiter})
            self.theta[c] = result.x

    # ---- persistence ---------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"parents": list(self.parents), "cards": list(self.cards), "n_classes": self.n_classes,
                "sigma": self.sigma, "bounds": self.bounds.tolist(), "theta": self.theta.tolist()}
