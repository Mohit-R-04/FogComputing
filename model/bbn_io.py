"""
Saving and loading a trained BBN: structure, every CPT, and the discretisation.

One file, ``data/bbn_trained.json``, is read by every Python script and by the Java
controller inside iFogSim (``org.fog.ft.BbnEngine``), so the network evaluated offline
is byte-for-byte the network that drives fault tolerance in the simulation. CPT rows
are stored in ``itertools.product`` order of the parents' states (last parent fastest),
which is also the row order the Java engine computes.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from bbn_model import BayesianNetwork, Structure
from cpt_tables import TARGET_WEIGHTS
from discretise import RELIABILITY_STATE_BY_CLASS, QuantileCuts

FORMAT = "fog-bbn/1"
#: Pseudo-counts per CPT row anchoring online refinement to the offline tables.
ONLINE_CONCENTRATION = 400.0


def save_model(net: BayesianNetwork, cuts: QuantileCuts, path: str | Path, meta: dict | None = None) -> None:
    st = net.structure
    doc = {
        "format": FORMAT,
        "structure": st.name,
        "target": st.target,
        "prior": st.prior,
        "evidence": list(st.evidence),
        "latent": list(st.latent),
        "states": {v: list(st.states(v)) for v in st.variables},
        "parents": {v: list(st.parents[v]) for v in st.variables},
        "inverted": sorted(st.inverted),
        "cpts": {
            v: [list(net.cpts[v][c]) for c in
                (itertools.product(*[st.states(p) for p in st.parents[v]]) if st.parents[v] else [()])]
            for v in st.variables
        },
        "cuts": {v: list(cuts.global_cuts[v]) for v in st.evidence if v in cuts.global_cuts},
        "cuts_by_class": ({cls_: {v: list(block[v]) for v in st.evidence if v in block}
                           for cls_, block in cuts.by_class.items()} if cuts.per_class else {}),
        "cut_method": cuts.method,
        "latent_form": st.latent_form,
        "reliability_state_by_class": dict(RELIABILITY_STATE_BY_CLASS),
        "cause_weights": {k: v for k, v in TARGET_WEIGHTS.items() if k in st.latent},
        "online": {"concentration": ONLINE_CONCENTRATION},
        "training": meta or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))


def load_model(path: str | Path) -> tuple[BayesianNetwork, QuantileCuts, dict]:
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} model")
    counts = {v: len(s) for v, s in doc["states"].items() if v in doc["evidence"] and len(s) != 3}
    st = Structure(
        name=doc["structure"],
        evidence=tuple(doc["evidence"]),
        latent=tuple(doc["latent"]),
        parents={v: tuple(p) for v, p in doc["parents"].items()},
        prior=doc["prior"],
        target=doc["target"],
        inverted=frozenset(doc["inverted"]),
        state_counts=counts,
        latent_form=doc.get("latent_form", "table"),
    )
    cpts = {}
    for v, rows in doc["cpts"].items():
        configs = (itertools.product(*[st.states(p) for p in st.parents[v]])
                   if st.parents[v] else [()])
        cpts[v] = {c: list(r) for c, r in zip(configs, rows)}
    net = BayesianNetwork(cpts=cpts, structure=st)
    net.validate()
    by_class = {cls_: {v: tuple(c) for v, c in block.items()}
                for cls_, block in doc.get("cuts_by_class", {}).items()}
    cuts = QuantileCuts(global_cuts={v: tuple(c) for v, c in doc["cuts"].items()},
                        by_class=by_class, method=doc.get("cut_method", "tertile"),
                        per_class=bool(by_class))
    return net, cuts, doc
