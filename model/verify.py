#!/usr/bin/env python3
"""
Verification report of the trained BBN on the held-out iFogSim runs.

    python verify.py

Analyses (all on fog and edge snapshots of the held-out runs, natural prevalence):

  1. discrimination   trained vs elicited network, ROC-AUC and PR-AUC with 95% intervals
                      from a bootstrap over nodes
  2. alarms           confusion counts, precision, recall and F1 across thresholds
  3. calibration      Brier score, log-loss, equal-mass ECE, reliability diagram
  4. early warning    per failing node (grouped by scenario, seed AND node -- the earlier
                      version grouped by node name across scenarios and merged the
                      histories of different machines)
  5. explanation      does the cause the network names match how the node actually
                      failed? (battery failures should be blamed on energy stress, ...)
  6. partial evidence each channel withheld in turn, and k random channels withheld

Writes model/results/verification_report.md and its figures.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import bbn_model
import design
import vizstyle
from bbn_fast import CompiledBBN
from bbn_io import load_model
from discretise import QuantileCuts
from metrics import bootstrap_ci, classification, early_warning, ece, group_index
from predict import explain

OUT = design.RESULTS
THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def comparison_threshold() -> float:
    path = design.ROOT / "model" / "comparison" / "results" / "comparison_summary.json"
    if path.exists():
        for m in json.loads(path.read_text())["models"]:
            if m["model"] == "BBN (proposed)":
                return float(m["threshold"])
    return 0.10


def main(argv=None) -> int:
    df = pd.read_csv(design.DATA / "telemetry_test_pool.csv")
    df = df[df.device_class != "cloud"].reset_index(drop=True)
    y = df.ground_truth_fail.to_numpy(dtype=int)
    groups = group_index(df)

    net, cuts, doc = load_model(design.DATA / "bbn_trained.json")
    trained = CompiledBBN(net)
    elicited = CompiledBBN(bbn_model.build_network())
    tertiles = QuantileCuts.load(design.DATA / "discretisation.json")
    E, P = trained.encode(df, cuts)
    p = trained.posterior(E, P)
    p_el = elicited.predict_frame(df, tertiles)

    # 1. discrimination
    disc = {}
    for name, score in (("trained", p), ("elicited", p_el)):
        disc[name] = {
            "roc_auc": roc_auc_score(y, score), "roc_ci": bootstrap_ci(y, score, groups, roc_auc_score),
            "pr_auc": average_precision_score(y, score),
            "pr_ci": bootstrap_ci(y, score, groups, average_precision_score),
        }

    # 2. alarms
    sweep = [classification(y, p, t) for t in THRESHOLDS]
    t_op = comparison_threshold()
    op = classification(y, p, t_op)

    # 3. calibration
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    calib = {"brier": brier_score_loss(y, pc), "log_loss": log_loss(y, pc), "ece": ece(y, pc),
             "mean_p": float(p.mean()), "prevalence": float(y.mean())}

    # 4. early warning
    warn = early_warning(df, p, t_op)

    # 5. explanation vs how the node failed
    cause, lift = explain(trained, E, P)
    pos = (df.ground_truth_fail == 1).to_numpy()
    # Plain arrays: crosstab aligns Series on their index, and the two would not match.
    actual = df.failure_cause.fillna("none").to_numpy()[pos]
    xtab = pd.crosstab(pd.Series(actual, name="actual"), pd.Series(cause[pos], name="named"))

    # 6. partial evidence
    channels = [v for v in net.structure.evidence]
    single = []
    base_pr = average_precision_score(y, p)
    for ch in channels:
        pd_ = trained.predict_frame(df, cuts, drop=frozenset({ch}))
        single.append({"channel": ch, "pr_auc": average_precision_score(y, pd_),
                       "roc_auc": roc_auc_score(y, pd_), "delta_pr": average_precision_score(y, pd_) - base_pr})
    single = pd.DataFrame(single).sort_values("delta_pr")
    rng = np.random.default_rng(3)
    k_rows = []
    for k in range(0, 7):
        prs, aucs = [], []
        for _ in range(10 if k else 1):
            drop = frozenset(rng.choice(channels, k, replace=False)) if k else frozenset()
            pk = trained.predict_frame(df, cuts, drop=drop)
            prs.append(average_precision_score(y, pk))
            aucs.append(roc_auc_score(y, pk))
        k_rows.append({"withheld": k, "pr_auc": float(np.mean(prs)), "pr_sd": float(np.std(prs)),
                       "roc_auc": float(np.mean(aucs)), "predictions_made": len(df)})
    kdf = pd.DataFrame(k_rows)

    figs = plots(y, p, p_el, single, kdf)
    write_report(doc, disc, sweep, op, t_op, calib, warn, xtab, single, kdf, figs, len(df), int(y.sum()))
    print(f"verification: AUC {disc['trained']['roc_auc']:.4f}, PR-AUC {disc['trained']['pr_auc']:.4f}, "
          f"Brier {calib['brier']:.5f}, detection {warn['detection_rate']:.2f} -> "
          f"{OUT / 'verification_report.md'}")
    return 0


def plots(y, p, p_el, single, kdf) -> list[str]:
    import matplotlib.pyplot as plt
    vizstyle.apply()
    names = []
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for score, label, color in ((p, "BBN (trained)", vizstyle.SLOTS[0]),
                                (p_el, "BBN (elicited, no failure data)", vizstyle.MODEL_COLORS["BBN (elicited, no failure data)"])):
        s = np.clip(score, 1e-5, 1)
        order = np.argsort(s)
        xs = [s[c].mean() for c in np.array_split(order, 15)]
        ys = [max(y[c].mean(), 1e-5) for c in np.array_split(order, 15)]
        ax.plot(xs, ys, marker="o", ms=4, color=color, label=label)
    ax.plot([1e-5, 1], [1e-5, 1], color=vizstyle.MUTED, lw=1)
    ax.set(xscale="log", yscale="log", xlim=(1e-5, 1), ylim=(1e-5, 1), title="Calibration on held-out runs",
           xlabel="Predicted P(failure within horizon)", ylabel="Observed frequency")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=170)
    plt.close(fig)
    names.append("calibration.png")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    s = single.sort_values("delta_pr")
    a.barh(np.arange(len(s)), s.delta_pr, height=0.55, color=vizstyle.SLOTS[0], zorder=2)
    a.set_yticks(np.arange(len(s)), s.channel)
    a.set(title="PR-AUC change when one channel is withheld", xlabel="Change in PR-AUC")
    a.grid(axis="y", visible=False)
    b.plot(kdf.withheld, kdf.pr_auc, marker="o", color=vizstyle.SLOTS[0], label="PR-AUC")
    b.fill_between(kdf.withheld, kdf.pr_auc - kdf.pr_sd, kdf.pr_auc + kdf.pr_sd, color=vizstyle.SLOTS[0],
                   alpha=0.15, lw=0)
    b.set(title="k random channels withheld (10 draws each)", xlabel="Channels withheld", ylabel="PR-AUC",
          ylim=(0, max(0.05, kdf.pr_auc.max() * 1.25)))
    fig.tight_layout()
    fig.savefig(OUT / "partial_evidence.png", dpi=170)
    plt.close(fig)
    names.append("partial_evidence.png")
    return names


def write_report(doc, disc, sweep, op, t_op, calib, warn, xtab, single, kdf, figs, n, npos) -> None:
    tr = doc["training"]
    L = ["# BBN failure prediction -- verification report", "",
         f"- Held-out pool: **{n:,} fog and edge snapshots, {npos} positive** "
         f"({100 * npos / n:.2f}%) from runs whose scenarios and seeds were never used in training.",
         f"- Network: `{doc['structure']}` with `{doc['cut_method']}` cut points; trained by MAP-EM on "
         f"{tr.get('rows', 0):,} snapshots ({tr.get('positives', 0)} positive, {tr.get('patterns', 0):,} distinct "
         f"patterns), Dirichlet concentration {tr.get('concentration')}, {tr.get('iterations')} iterations.",
         "- Look-ahead horizon: 200 time units (two decision epochs).", "",
         "## 1. Discrimination", "",
         "| Network | ROC-AUC | 95% CI | PR-AUC | 95% CI |", "|---|---:|---|---:|---|"]
    for k, label in (("trained", "trained (proposed)"), ("elicited", "elicited only, paper Fig. 2, no failure data")):
        d = disc[k]
        L.append(f"| {label} | {d['roc_auc']:.4f} | [{d['roc_ci'][0]:.4f}, {d['roc_ci'][1]:.4f}] | "
                 f"{d['pr_auc']:.4f} | [{d['pr_ci'][0]:.4f}, {d['pr_ci'][1]:.4f}] |")
    L += ["", f"Chance level for PR-AUC is the prevalence, {npos / n:.4f}. Intervals resample whole nodes.", "",
          "## 2. Alarms across thresholds", "",
          "| Threshold | TP | FP | FN | Precision | Recall | F1 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sweep:
        L.append(f"| {c['threshold']:.2f} | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']:.3f} | "
                 f"{c['recall']:.3f} | {c['f1']:.3f} |")
    L += ["", f"At the operating threshold chosen on training folds ({t_op:.4f}): precision {op['precision']:.3f}, "
          f"recall {op['recall']:.3f}, F1 {op['f1']:.3f}. The fault-tolerance controller does not use a fixed "
          "threshold at all: it migrates when P(fail) x C_fail > C_mig, so each session gets its own.", "",
          "## 3. Calibration", "",
          f"Brier score {calib['brier']:.5f}, log-loss {calib['log_loss']:.5f}, ECE {calib['ece']:.5f} "
          f"(15 equal-mass bins). Mean predicted probability {calib['mean_p']:.4f} against an observed "
          f"failure rate of {calib['prevalence']:.4f}.", "", "![Calibration](calibration.png)", "",
          "## 4. Early warning", "",
          f"Of {warn['failing_nodes']} failing nodes, **{warn['detected']} ({100 * warn['detection_rate']:.0f}%)** "
          f"were flagged inside the two epochs before they failed, with a mean lead of "
          f"{warn['mean_lead']:.0f} time units; {warn['false_alarms_per_1000']:.1f} alarms per 1,000 "
          "healthy snapshots. A live migration needs 5-30 time units, so a flag raised within the "
          "horizon leaves time to act. Nodes are grouped by scenario, seed and node.", "",
          "## 5. Does the named cause match how the node failed?", "",
          "Rows: how positive (failing-within-horizon) snapshots actually failed. Columns: the latent "
          "condition the BBN named as the principal cause (largest lift of P(high) over its prior; "
          "*none* when no condition rose). Share of the row, with the count in brackets.", "",
          "| Actual cause | Snapshots | " + " | ".join(xtab.columns) + " |",
          "|---|---:|" + "---:|" * len(xtab.columns)]
    for idx, r in xtab.iterrows():
        total = int(r.sum())
        L.append(f"| {idx} | {total} | " + " | ".join(f"{v / total:.2f} ({v})" for v in r.values) + " |")
    L += ["", "Battery failures should be blamed on energy stress. Sudden failures have no precursor by "
          "construction, so the network should mostly raise nothing for them. For wear-out, energy "
          "stress is the expected answer in this simulator: a fog node's wear rate rises with its "
          "energy depletion as well as its load and heat, and a worn node draws extra power, so the "
          "drain rate is the earliest symptom of wear (it is also the channel whose loss costs the "
          "most PR-AUC in section 6); computational and thermal symptoms grow later, with the square "
          "and the 1.5th power of the damage.", "",
          "## 6. Partial evidence", "",
          "| Channel withheld | PR-AUC | ROC-AUC | Change in PR-AUC |", "|---|---:|---:|---:|"]
    for _, r in single.iterrows():
        L.append(f"| {r.channel} | {r.pr_auc:.4f} | {r.roc_auc:.4f} | {r.delta_pr:+.4f} |")
    L += ["", "| Channels withheld (random) | PR-AUC | s.d. | ROC-AUC | Predictions made |", "|---:|---:|---:|---:|---:|"]
    for _, r in kdf.iterrows():
        L.append(f"| {int(r.withheld)} | {r.pr_auc:.4f} | {r.pr_sd:.4f} | {r.roc_auc:.4f} | {int(r.predictions_made):,} |")
    L += ["", "A prediction is produced for every snapshot however many channels are dark: missing "
          "evidence is summed out of Eq. (1), never imputed.", "", "![Partial evidence](partial_evidence.png)"]
    (OUT / "verification_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
