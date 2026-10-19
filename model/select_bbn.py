#!/usr/bin/env python3
"""
Chooses the BBN configuration by internal validation on the training scenarios only.

    python select_bbn.py

The held-out test pool is never read here. Training runs are split into four folds by
scenario (every seed of a scenario stays in one fold), so each candidate is scored on
operating regimes it did not see -- the same kind of generalisation the final
evaluation asks for. Cut points are recomputed inside every fold from that fold's
training rows only.

Candidates are searched in stages:

  A. the paper's form -- free latent tables, three evidence states -- from Fig. 2 with
     tertile cut points through supervised cut points, trend evidence and
     class-conditional evidence (documents what each step buys);
  B. learned ranked-node latent tables (``ranked_node.py``) with class-specific latent
     conditions: global or per-class supervised cut points x evidence states (6, 8, 10)
     x prior strength on the ranked-node points;
  C. for the two best of stage B: Dirichlet concentration x EM iterations;
  D. tie-break: every configuration within 1% of the best log-loss is re-scored with
     30% of telemetry channels withheld at random in the validation folds, and the most
     robust one is chosen -- missing telemetry is the condition the BBN exists for.

The criterion is the pooled out-of-fold log-loss (lower is better): a proper scoring
rule that rewards both ranking and calibration, which the expected-loss controller
needs. ROC-AUC, PR-AUC and Brier score are reported alongside.

Outputs: ``model/results/bbn_selection.csv`` and ``data/models/bbn_config.json``.
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import design
import train_bbn
from bbn_fast import CompiledBBN

N_FOLDS = 4

DROPOUT_RATE = 0.30
CHANNELS = ("cpu_util", "ram_util", "queue_depth", "uplink_latency", "jitter_pktloss",
            "handover_rate", "residual_energy", "temperature", "cpu_trend", "queue_trend",
            "latency_trend", "temp_trend", "energy_drain")


def with_dropout(df: pd.DataFrame, rate: float, seed: int = 7) -> pd.DataFrame:
    """Withholds each channel of each row independently with probability ``rate``."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in CHANNELS:
        mask = rng.random(len(out)) < rate
        out.loc[mask, c] = np.nan
    return out


def describe(c: dict) -> str:
    if c["family"] == "table":
        graph = ("fig2+trends" if c["trends"] else "paper-fig2") + ("+class-ev" if c["class_conditional"] else "")
        return f"table  {graph:22s} {c['cuts']:10s} k=3"
    cuts = "per-class" if c["per_class_cuts"] else "global"
    return f"ranked {cuts:9s} k={c['states']:<2d} lam={c['lam']:<5g}"


def evaluate(df: pd.DataFrame, folds, config: dict, dropout: float = 0.0) -> dict:
    oof = np.full(len(df), np.nan)
    for tr_idx, va_idx in folds:
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        cuts = train_bbn.cuts_for(tr, config)
        structure = train_bbn.structure_for(config, cuts)
        net, _, _ = train_bbn.fit(tr, structure, cuts, config["concentration"],
                                  max_iter=int(config["iterations"]), tol=1e-12,
                                  lam=float(config.get("lam", 0.05)))
        if dropout > 0:
            va = with_dropout(va, dropout)
        oof[va_idx] = CompiledBBN(net).predict_frame(va, cuts)
    keep = (df.device_class != "cloud").to_numpy()
    y = df.ground_truth_fail.to_numpy()[keep]
    p = np.clip(oof[keep], 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def table_config(trends, class_conditional, cuts):
    return {"family": "table", "trends": trends, "class_conditional": class_conditional, "class_latent": False,
            "cuts": cuts, "states": 3, "per_class_cuts": False, "concentration": 5.0, "iterations": 100, "lam": 0.0}


def ranked_config(per_class, states, lam, concentration=5.0, iterations=60):
    return {"family": "ranked", "trends": True, "class_conditional": True, "class_latent": True,
            "cuts": "supervised", "states": states, "per_class_cuts": per_class,
            "concentration": concentration, "iterations": iterations, "lam": lam}


def main(argv=None) -> int:
    df = pd.read_csv(design.DATA / "telemetry_train_pool.csv")
    folds = list(GroupKFold(n_splits=N_FOLDS).split(df, groups=df.scenario_id))
    print(f"internal validation: {len(df)} training rows, {N_FOLDS} folds by scenario "
          f"{[sorted(df.iloc[v].scenario_id.unique().tolist()) for _, v in folds]}", flush=True)
    rows = []

    def run(config: dict, stage: str) -> dict:
        t = time.time()
        m = evaluate(df, folds, config)
        row = {"stage": stage, **config, **m, "seconds": round(time.time() - t, 1)}
        rows.append(row)
        print(f"  [{stage}] {describe(config)} c={config['concentration']:<4g} it={config['iterations']:<4d} "
              f"log-loss={m['log_loss']:.5f} AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
              f"Brier={m['brier']:.5f}", flush=True)
        return row

    print("stage A: the paper's form (free latent tables, three evidence states)", flush=True)
    run(table_config(False, False, "tertile"), "A")
    run(table_config(False, False, "supervised"), "A")
    run(table_config(True, False, "supervised"), "A")
    run(table_config(True, True, "supervised"), "A")

    print("stage B: learned ranked-node latent tables, class-specific latent conditions", flush=True)
    for per_class, states, lam in itertools.product((False, True), (6, 8, 10), (0.02, 0.1)):
        run(ranked_config(per_class, states, lam), "B")

    stage_b = sorted([r for r in rows if r["stage"] == "B"], key=lambda r: r["log_loss"])
    print("stage C: concentration x EM iterations for the two best", flush=True)
    for best in stage_b[:2]:
        for conc, its in ((5.0, 30), (5.0, 120), (1.0, 60), (20.0, 60)):
            run(ranked_config(best["per_class_cuts"], best["states"], best["lam"], conc, its), "C")

    table = pd.DataFrame(rows).sort_values("log_loss")
    floor = table.log_loss.min() * 1.01
    near = table[table.log_loss <= floor]
    print(f"stage D: robustness tie-break among {len(near)} configurations within 1% "
          f"of the best log-loss ({int(DROPOUT_RATE * 100)}% of channels withheld)", flush=True)
    robust = {}
    keys = ("family", "trends", "class_conditional", "class_latent", "cuts", "states", "per_class_cuts",
            "concentration", "iterations", "lam")
    for i, r in near.iterrows():
        cfg = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in keys}
        m = evaluate(df, folds, cfg, dropout=DROPOUT_RATE)
        robust[i] = m["log_loss"]
        print(f"  [D] {describe(cfg)} c={cfg['concentration']:<4g} it={cfg['iterations']:<4d} "
              f"log-loss with dropout={m['log_loss']:.5f} AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f}",
              flush=True)
    table["dropout_log_loss"] = pd.Series(robust)
    design.RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(design.RESULTS / "bbn_selection.csv", index=False)
    best = table.loc[min(robust, key=robust.get)]
    chosen = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in keys}
    chosen.update({
        "structure_name": describe(chosen),
        "cv_log_loss": float(best["log_loss"]),
        "cv_roc_auc": float(best["roc_auc"]),
        "cv_pr_auc": float(best["pr_auc"]),
        "cv_brier": float(best["brier"]),
        "cv_dropout_log_loss": float(best["dropout_log_loss"]),
        "selected_by": "pooled out-of-fold log-loss over 4 scenario folds of the training runs; "
                       "ties within 1% broken by log-loss with 30% of channels withheld",
    })
    design.MODELS.mkdir(parents=True, exist_ok=True)
    (design.MODELS / "bbn_config.json").write_text(json.dumps(chosen, indent=1))
    paper = table[(table.family == "table") & (table.cuts == "tertile")].iloc[0]
    print(f"\nselected: {chosen['structure_name']}  concentration {chosen['concentration']:g}, "
          f"{chosen['iterations']} iterations")
    print(f"  CV log-loss {best['log_loss']:.5f}, AUC {best['roc_auc']:.4f}, PR-AUC {best['pr_auc']:.4f}"
          f"   (paper graph with tertiles: AUC {paper['roc_auc']:.4f}, PR-AUC {paper['pr_auc']:.4f})")
    print(f"  wrote {design.RESULTS / 'bbn_selection.csv'} and {design.MODELS / 'bbn_config.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
