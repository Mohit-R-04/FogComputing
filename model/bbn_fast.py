"""
Vectorised exact inference and EM for the layered fog-node BBN.

``inference.py`` is the reference implementation of Eq. (1): general variable
elimination plus a literal enumeration oracle, one evidence set at a time. Training on
the complete iFogSim pool and scoring every held-out snapshot needs the same numbers for
hundreds of thousands of rows, so this module computes them in bulk for the structure
the paper specifies -- evidence variables (roots, or children of the reliability prior),
a latent tier whose parents are evidence variables, and a target whose parents are the
latent variables and the prior.

In that structure each evidence variable feeds one latent variable (the latent variables
may also depend on the device class), so the latent variables are independent given the
evidence and the device class, and Eq. (1) factorises:

    P(L_i | e, p) = sum over L_i's unreported parents m  of  P(L_i | pa) prod P(m | p)
    P(F | e, p)   = sum over latent configurations l      of  prod_i P(l_i | e, p) P(F | l, p)

A channel that did not report is summed out, never imputed. ``test_inference.py``
checks every number here against variable elimination.

The same factorisation gives the E-step of expectation-maximisation: the posterior over
latent configurations given the evidence *and* the observed outcome, from which the
expected sufficient statistics of every learned table follow. The M-step is a Dirichlet
(MAP) update centred on the elicited tables -- the paper's "Bayesian parameter learning
with Dirichlet priors".
"""

from __future__ import annotations

import itertools

import numpy as np

from bbn_model import BayesianNetwork
from discretise import RELIABILITY_STATE_BY_CLASS, QuantileCuts, state_index
from ranked_node import RankedNode

#: Rows scored per block, to bound the memory of the latent joint (N x 3^k).
CHUNK = 40000


class CompiledBBN:
    """A layered BBN as numpy arrays, for bulk inference and EM."""

    def __init__(self, net: BayesianNetwork):
        self.net = net
        st = net.structure
        self.st = st
        self.evidence = list(st.evidence)
        self.latent = list(st.latent)
        self.ev_pos = {e: i for i, e in enumerate(self.evidence)}
        self.card = {v: len(st.states(v)) for v in st.variables}
        self._check_layered()
        self.arr = {v: self._array(v) for v in st.variables}

        tp = st.parents[st.target]
        self.prior_axis = tp.index(st.prior)
        self.target_latents = [p for p in tp if p != st.prior]
        self.fail = st.states(st.target).index("fail")
        prior_states = st.states(st.prior)
        self.prior_code = {c: prior_states.index(s) for c, s in RELIABILITY_STATE_BY_CLASS.items()}
        self.ranked: dict[str, RankedNode] = {}

    def init_ranked(self) -> None:
        """Ranked-node latent tables, initialised at the elicited points (training only)."""
        for latent in self.latent:
            rn = RankedNode.elicited(latent, self.st)
            self.ranked[latent] = rn
            self.arr[latent] = rn.table()

    # ---- structure ---------------------------------------------------------------------

    def _check_layered(self) -> None:
        st = self.st
        used = set()
        for e in st.evidence:
            if any(p != st.prior for p in st.parents[e]):
                raise ValueError(f"{e} may only depend on the reliability prior")
        for latent in st.latent:
            for p in st.parents[latent]:
                if p == st.prior:
                    continue
                if p not in self.ev_pos:
                    raise ValueError(f"{latent} has a non-evidence parent {p}")
                if p in used:
                    raise ValueError(f"{p} feeds more than one latent variable")
                used.add(p)
        for p in st.parents[st.target]:
            if p not in st.latent and p != st.prior:
                raise ValueError(f"target parent {p} is neither latent nor the prior")

    def _array(self, var: str) -> np.ndarray:
        parents = self.st.parents[var]
        shape = [self.card[p] for p in parents] + [self.card[var]]
        a = np.zeros(shape)
        for config in itertools.product(*[self.st.states(p) for p in parents]) if parents else [()]:
            idx = tuple(self.st.states(p).index(c) for p, c in zip(parents, config))
            a[idx] = self.net.cpts[var][config]
        return a

    @property
    def _T(self) -> np.ndarray:
        """Target table as (prior, latents in target-parent order..., target)."""
        return np.moveaxis(self.arr[self.st.target], self.prior_axis, 0)

    def to_network(self) -> BayesianNetwork:
        """The current arrays as a validated :class:`BayesianNetwork`."""
        cpts = {}
        for var in self.st.variables:
            parents = self.st.parents[var]
            table = {}
            for config in itertools.product(*[self.st.states(p) for p in parents]) if parents else [()]:
                idx = tuple(self.st.states(p).index(c) for p, c in zip(parents, config))
                row = self.arr[var][idx]
                table[config] = [float(x) for x in row / row.sum()]
            cpts[var] = table
        net = BayesianNetwork(cpts=cpts, structure=self.st)
        net.validate()
        return net

    # ---- evidence ------------------------------------------------------------------------

    def encode(self, df, cuts: QuantileCuts, drop: frozenset[str] = frozenset()):
        """Evidence states (N x evidence, -1 = did not report) and prior states (N,)."""
        n = len(df)
        E = np.full((n, len(self.evidence)), -1, dtype=np.int8)
        classes = df["device_class"].to_numpy()
        for j, e in enumerate(self.evidence):
            if e in drop or e not in df.columns or e not in cuts.global_cuts:
                continue
            values = df[e].to_numpy(dtype=float)
            if cuts.per_class and cuts.by_class:
                for cls_ in np.unique(classes):
                    m = classes == cls_
                    E[m, j] = state_index(values[m], cuts.cuts_for(e, cls_))
            else:
                E[:, j] = state_index(values, cuts.global_cuts[e])
        P = df["device_class"].map(self.prior_code).fillna(-1).to_numpy(dtype=np.int64)
        return E, P

    def _root_weights(self, e: str, p: int) -> np.ndarray:
        a = self.arr[e]
        if a.ndim == 1:
            return a
        return a[p]

    # ---- inference -------------------------------------------------------------------------

    def _evidence_parents(self, latent: str) -> tuple[str, ...]:
        return tuple(p for p in self.st.parents[latent] if p != self.st.prior)

    def latent_marginals(self, E: np.ndarray, P: np.ndarray) -> list[np.ndarray]:
        """P(L_i | e, p) for every latent variable, as N x states arrays."""
        n = len(E)
        out = []
        for latent in self.latent:
            pa = self._evidence_parents(latent)
            class_axis = len(pa) if self.st.prior in self.st.parents[latent] else None
            uses_p = self.st.class_conditional or class_axis is not None
            sub = E[:, [self.ev_pos[x] for x in pa]].astype(np.int64)
            miss = sub < 0
            code = (miss * (1 << np.arange(len(pa)))).sum(axis=1)
            key = code * 4 + ((P + 1) if uses_p else 0)
            res = np.empty((n, self.card[latent]))
            for k in np.unique(key):
                rows = np.nonzero(key == k)[0]
                mask = int(k // 4)
                p = int(k % 4) - 1 if uses_p else -1
                A = self.arr[latent]
                if class_axis is not None:
                    A = np.take(A, p, axis=class_axis)
                missing_axes = [j for j in range(len(pa)) if (mask >> j) & 1]
                for j in sorted(missing_axes, reverse=True):
                    A = np.tensordot(A, self._root_weights(pa[j], p), axes=([j], [0]))
                obs_axes = [j for j in range(len(pa)) if not (mask >> j) & 1]
                if obs_axes:
                    res[rows] = A[tuple(sub[rows, j] for j in obs_axes)]
                else:
                    res[rows] = A
            out.append(res)
        return out

    def _joint(self, M: list[np.ndarray]) -> np.ndarray:
        """Joint over latent configurations (N x 3 x ... x 3) in target-parent order."""
        ordered = [M[self.latent.index(l)] for l in self.target_latents]
        n = len(ordered[0])
        J = ordered[0]
        for m in ordered[1:]:
            J = J[..., None] * m.reshape((n,) + (1,) * (J.ndim - 1) + (m.shape[1],))
        return J

    def _prior_weights(self, E: np.ndarray) -> np.ndarray:
        """P(prior state | evidence) for rows whose device class is unknown."""
        w = np.tile(self.arr[self.st.prior], (len(E), 1))
        if self.st.class_conditional:
            for j, e in enumerate(self.evidence):
                obs = E[:, j] >= 0
                if obs.any():
                    w[obs] *= self.arr[e][:, E[obs, j]].T
        return w / w.sum(axis=1, keepdims=True)

    def _p_fail_known_prior(self, E: np.ndarray, P: np.ndarray) -> np.ndarray:
        J = self._joint(self.latent_marginals(E, P))
        T = self._T[..., self.fail][P]
        return (J * T).reshape(len(E), -1).sum(axis=1)

    def posterior(self, E: np.ndarray, P: np.ndarray) -> np.ndarray:
        """P(node_failure = fail | evidence) for every row."""
        E = np.asarray(E)
        P = np.asarray(P, dtype=np.int64)
        out = np.empty(len(E))
        for lo in range(0, len(E), CHUNK):
            hi = min(lo + CHUNK, len(E))
            e, p = E[lo:hi], P[lo:hi]
            known = p >= 0
            res = np.empty(hi - lo)
            if known.any():
                res[known] = self._p_fail_known_prior(e[known], p[known])
            if (~known).any():
                eu = e[~known]
                w = self._prior_weights(eu)
                acc = np.zeros(len(eu))
                for s in range(self.card[self.st.prior]):
                    acc += w[:, s] * self._p_fail_known_prior(eu, np.full(len(eu), s))
                res[~known] = acc
            out[lo:hi] = res
        return out

    def predict_frame(self, df, cuts: QuantileCuts, drop: frozenset[str] = frozenset()) -> np.ndarray:
        E, P = self.encode(df, cuts, drop)
        return self.posterior(E, P)

    # ---- learning ------------------------------------------------------------------------

    def fit_roots(self, E: np.ndarray, P: np.ndarray, W: np.ndarray, smoothing: float = 1.0) -> None:
        """Evidence marginals by counting (they are observed whenever they report)."""
        for j, e in enumerate(self.evidence):
            obs = E[:, j] >= 0
            if self.arr[e].ndim == 1:
                c = np.zeros(self.card[e])
                np.add.at(c, E[obs, j].astype(np.int64), W[obs])
            else:
                c = np.zeros(self.arr[e].shape)
                ok = obs & (P >= 0)
                np.add.at(c, (P[ok], E[ok, j].astype(np.int64)), W[ok])
            c = c + smoothing
            self.arr[e] = c / c.sum(axis=-1, keepdims=True)

    def expected_counts(self, E: np.ndarray, P: np.ndarray, Y: np.ndarray, W: np.ndarray):
        """E-step: expected counts of the latent and target tables, and log-likelihood."""
        T = self._T
        counts_T = np.zeros(T.shape)
        counts_L = {l: np.zeros(self.arr[l].shape) for l in self.latent}
        loglik = 0.0
        for lo in range(0, len(E), CHUNK):
            hi = min(lo + CHUNK, len(E))
            e, p, y, w = E[lo:hi], P[lo:hi], Y[lo:hi], W[lo:hi]
            ok = p >= 0
            e, p, y, w = e[ok], p[ok], y[ok], w[ok]
            n = len(e)
            if n == 0:
                continue
            J = self._joint(self.latent_marginals(e, p))
            Tsel = T[p]                                        # (n, l..., 2)
            lik = np.take_along_axis(Tsel, y.reshape((n,) + (1,) * (Tsel.ndim - 1)), axis=-1)[..., 0]
            post = J * lik
            z = post.reshape(n, -1).sum(axis=1)
            loglik += float(np.sum(w * np.log(np.maximum(z, 1e-300))))
            post /= z.reshape((n,) + (1,) * (post.ndim - 1))
            for ps in range(T.shape[0]):
                for ys in range(T.shape[-1]):
                    sel = (p == ps) & (y == ys)
                    if sel.any():
                        counts_T[ps, ..., ys] += np.tensordot(w[sel], post[sel], axes=(0, 0))
            k = len(self.target_latents)
            for i, latent in enumerate(self.target_latents):
                marg = post.sum(axis=tuple(1 + j for j in range(k) if j != i))
                pa = self._evidence_parents(latent)
                sub = e[:, [self.ev_pos[x] for x in pa]].astype(np.int64)
                complete = (sub >= 0).all(axis=1)
                if complete.any():
                    idx = tuple(sub[complete, j] for j in range(len(pa)))
                    if self.st.prior in self.st.parents[latent]:
                        idx = idx + (p[complete],)
                    np.add.at(counts_L[latent], idx, w[complete, None] * marg[complete])
        return counts_L, counts_T, loglik

    def maximise(self, counts_L, counts_T, prior_arr: dict[str, np.ndarray], concentration: float,
                 lam: float = 0.1) -> None:
        """M-step: ranked latents by MAP fitting of their points (prior strength ``lam``);
        free tables and the target by a Dirichlet posterior around the elicited tables."""
        for latent in self.latent:
            if latent in self.ranked:
                self.ranked[latent].fit(counts_L[latent], lam)
                self.arr[latent] = self.ranked[latent].table()
                continue
            new = concentration * prior_arr[latent] + counts_L[latent]
            self.arr[latent] = new / new.sum(axis=-1, keepdims=True)
        prior_T = np.moveaxis(prior_arr[self.st.target], self.prior_axis, 0)
        new_T = concentration * prior_T + counts_T
        new_T = new_T / new_T.sum(axis=-1, keepdims=True)
        self.arr[self.st.target] = np.moveaxis(new_T, 0, self.prior_axis)

    def latent_high(self, E: np.ndarray, P: np.ndarray) -> np.ndarray:
        """P(L_i = high | evidence), N x latent, for explanations."""
        M = self.latent_marginals(E, np.where(P < 0, 1, P))
        return np.stack([m[:, -1] for m in M], axis=1)


def aggregate_patterns(E: np.ndarray, P: np.ndarray, Y: np.ndarray, W: np.ndarray | None = None):
    """Collapses identical (evidence, prior, label) rows into weighted unique patterns.

    The sufficient statistics are unchanged, so EM gives exactly the same tables as on
    every row; the pool simply shrinks to the number of distinct discretised patterns.
    """
    W = np.ones(len(E)) if W is None else np.asarray(W, dtype=float)
    key = np.concatenate([E.astype(np.int16), P.reshape(-1, 1).astype(np.int16),
                          Y.reshape(-1, 1).astype(np.int16)], axis=1)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    weights = np.bincount(inverse.ravel(), weights=W, minlength=len(uniq))
    k = E.shape[1]
    return (uniq[:, :k].astype(np.int8), uniq[:, k].astype(np.int64),
            uniq[:, k + 1].astype(np.int64), weights)
