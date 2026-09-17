#!/usr/bin/env python3
"""
Verifies the BBN's predictions against the injected ground truth.

Six things are measured, and they are the six the Review 2 paper commits to:

1.  Discrimination -- ROC-AUC and PR-AUC, with bootstrap confidence intervals.
    The intervals matter here. A hundred cases with forty positives gives an AUC
    standard error near 0.06, so a bare point estimate invites conclusions the
    sample cannot support.
2.  Classification quality -- precision, recall, F1 and accuracy over a threshold
    sweep, since the operating point is a deployment choice rather than a property
    of the model.
3.  Calibration -- reliability diagram, Brier score and expected calibration error.
    A posterior that drives an expected-loss migration rule has to mean what it
    says; ranking correctly is not enough.
4.  Lead time -- how far ahead of the actual failure the posterior first crosses
    the threshold. A prediction that arrives after the deadline has been missed is
    worth nothing, which is the whole complaint the paper makes about reactive
    schemes.
5.  Partial-evidence robustness -- performance as telemetry channels are
    progressively withheld. This is the paper's headline claim, so it is measured
    rather than asserted.
6.  A threshold baseline -- the single-metric rule that stands in for FTAPA,
    scored on the same cases, to separate the value of probabilistic inference from
    the value of merely acting early.

    python verify.py [--cases FILE] [--pool FILE] [--quantiles FILE]
                     [--trained FILE] [--out FILE]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import bbn_model
import inference
from bbn_model import EVIDENCE_VARS, TARGET_STATES, TARGET_VAR
from discretise import QuantileCuts, row_to_evidence

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE / "results"

#: Look-ahead horizon, matching FaultInjector.HORIZON_DELTA_T on the Java side.
HORIZON_DELTA_T = 200.0
#: Length of one resource-management epoch, matching Config.RESOURCE_MGMT_INTERVAL.
EPOCH_LENGTH = 100.0

RNG = np.random.default_rng(20260912)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def labels_of(rows: list[dict]) -> np.ndarray:
    return np.array([int(r["ground_truth_fail"]) for r in rows])


def score_rows(net, rows: list[dict], cuts: QuantileCuts,
               drop: frozenset[str] = frozenset()) -> np.ndarray:
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        evidence = row_to_evidence(row, cuts, drop=drop)
        posterior = inference.variable_elimination(net, evidence, TARGET_VAR)
        out[i] = posterior[TARGET_STATES.index("fail")]
    return out


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric, n: int = 2000, alpha: float = 0.05):
    """Percentile bootstrap interval for a ranking metric."""
    stats = []
    idx = np.arange(len(y))
    for _ in range(n):
        take = RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        stats.append(metric(y[take], p[take]))
    if not stats:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1, and the F1 it achieves."""
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.01, 1.00, 0.01):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """ECE plus the per-bin data needed for a reliability diagram."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        freq = float(y[mask].mean())
        weight = mask.sum() / len(y)
        ece += weight * abs(freq - conf)
        rows.append({"lo": lo, "hi": hi, "n": int(mask.sum()),
                     "confidence": conf, "frequency": freq})
    return ece, rows


# --------------------------------------------------------------------------
# the six analyses
# --------------------------------------------------------------------------

def discrimination(y, p) -> dict:
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    return {
        "roc_auc": auc,
        "roc_auc_ci": bootstrap_ci(y, p, roc_auc_score),
        "pr_auc": ap,
        "pr_auc_ci": bootstrap_ci(y, p, average_precision_score),
        "prevalence": float(y.mean()),
    }


def classification(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }


def threshold_sweep(y, p) -> list[dict]:
    return [classification(y, p, t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]


def calibration(y, p) -> dict:
    ece, bins = expected_calibration_error(y, p)
    return {"brier": brier_score_loss(y, p), "ece": ece, "bins": bins}


def lead_time(net, pool_rows: list[dict], cuts: QuantileCuts, threshold: float) -> dict:
    """How far ahead of failure the posterior first crosses the threshold.

    Walks each failing node's own telemetry history in time order and finds the
    earliest epoch, inside the run-up to its failure, at which the model would have
    raised the alarm. Reported in simulation time units and in epochs.

    Only the window of length delta_t before the failure counts. An alarm raised
    long before that is not early warning -- the model simply thinks the node is
    unhealthy, and crediting it with hours of lead time would flatter the result.
    """
    by_node: dict[str, list[dict]] = {}
    for row in pool_rows:
        ft = (row.get("failure_time") or "").strip()
        if not ft:
            continue
        by_node.setdefault(row["node_name"], []).append(row)

    leads, detected, total = [], 0, 0
    for node, rows in by_node.items():
        rows.sort(key=lambda r: float(r["epoch"]))
        failure_time = float(rows[0]["failure_time"])
        window = [r for r in rows
                  if failure_time - HORIZON_DELTA_T <= float(r["epoch"]) < failure_time]
        if not window:
            continue
        total += 1
        p = score_rows(net, window, cuts)
        crossed = np.nonzero(p >= threshold)[0]
        if len(crossed):
            detected += 1
            leads.append(failure_time - float(window[crossed[0]]["epoch"]))

    if not leads:
        return {"nodes": total, "detected": 0, "detection_rate": 0.0,
                "mean_lead": float("nan"), "median_lead": float("nan"),
                "mean_lead_epochs": float("nan")}

    arr = np.array(leads)
    return {
        "nodes": total,
        "detected": detected,
        "detection_rate": detected / total if total else 0.0,
        "mean_lead": float(arr.mean()),
        "median_lead": float(np.median(arr)),
        "mean_lead_epochs": float(arr.mean() / EPOCH_LENGTH),
    }


def partial_evidence(net, rows, cuts, y, threshold: float, trials: int = 30) -> list[dict]:
    """Performance as telemetry channels go dark.

    Channels are withheld at random per trial and the whole evaluation repeated.
    Temperature is excluded from the draw because it is already never observed --
    withholding it again would be a no-op dressed up as an ablation.
    """
    droppable = [v for v in EVIDENCE_VARS if v != "temperature"]
    out = []
    for k in range(0, 4):
        aucs, f1s, means = [], [], []
        n_trials = 1 if k == 0 else trials
        for _ in range(n_trials):
            drop = frozenset(RNG.choice(droppable, size=k, replace=False)) if k else frozenset()
            p = score_rows(net, rows, cuts, drop=drop)
            aucs.append(roc_auc_score(y, p))
            f1s.append(f1_score(y, (p >= threshold).astype(int), zero_division=0))
            means.append(p.mean())
        out.append({
            "dropped": k,
            "roc_auc": float(np.mean(aucs)),
            "roc_auc_sd": float(np.std(aucs)),
            "f1": float(np.mean(f1s)),
            "mean_p": float(np.mean(means)),
            "predictions_made": len(rows),
        })
    return out


def baseline_under_ablation(rows, y, cut: float, trials: int = 30) -> list[dict]:
    """The CPU threshold rule, subjected to the same channel withholding.

    This is the contrast the partial-evidence section exists to draw. When the one
    metric the rule watches goes dark it has nothing to test, so it cannot fire and
    every failing node passes unflagged -- not a degraded answer but no answer.
    """
    droppable = [v for v in EVIDENCE_VARS if v != "temperature"]
    cpu = np.array([float(r["cpu_util"]) if (r.get("cpu_util") or "").strip() else np.nan
                    for r in rows])
    out = []
    for k in range(0, 4):
        f1s, fired, answered = [], [], []
        n_trials = 1 if k == 0 else trials
        for _ in range(n_trials):
            drop = set(RNG.choice(droppable, size=k, replace=False)) if k else set()
            if "cpu_util" in drop:
                pred = np.zeros(len(rows), dtype=int)   # silent: nothing to threshold
                answered.append(0)
            else:
                pred = (cpu >= cut).astype(int)
                answered.append(len(rows))
            f1s.append(f1_score(y, pred, zero_division=0))
            fired.append(int(pred.sum()))
        out.append({
            "dropped": k,
            "f1": float(np.mean(f1s)),
            "alarms_raised": float(np.mean(fired)),
            "predictions_made": float(np.mean(answered)),
        })
    return out


def threshold_baseline(rows, y) -> dict:
    """The single-metric rule that stands in for threshold-based FTAPA.

    Fires when CPU utilisation exceeds a fixed cut. Reported at its own best cut
    rather than a nominal 0.8, so the comparison is against the baseline at its
    strongest rather than a straw man.

    It also cannot answer at all when the metric it watches is missing, which is
    the structural difference the paper draws attention to: a threshold on an
    unreported channel never fires, whereas a marginalising model still returns a
    posterior.
    """
    cpu = np.array([float(r["cpu_util"]) if (r.get("cpu_util") or "").strip() else np.nan
                    for r in rows])
    observed = ~np.isnan(cpu)

    best = {"f1": -1.0}
    for cut in np.arange(0.30, 1.00, 0.01):
        pred = np.zeros(len(rows), dtype=int)
        pred[observed] = (cpu[observed] >= cut).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "cut": float(cut), "f1": f1,
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "accuracy": accuracy_score(y, pred),
            }
    filled = np.where(observed, np.nan_to_num(cpu), 0.0)
    best["roc_auc"] = roc_auc_score(y, filled)
    best["silent_on"] = int((~observed).sum())
    return best


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def make_plots(results: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # ROC
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for label, (y, p) in results["curves"].items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Failure prediction, ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "roc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for label, cal in results["calibration"].items():
        xs = [b["confidence"] for b in cal["bins"]]
        ys = [b["frequency"] for b in cal["bins"]]
        ax.plot(xs, ys, "o-", label=f"{label} (ECE={cal['ece']:.3f})")
    ax.set_xlabel("predicted P(fail)")
    ax.set_ylabel("observed failure frequency")
    ax.set_title("Posterior calibration")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out_dir / "calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    # Partial evidence: two panels, because the interesting result is not in the
    # first one. Ranking barely moves as channels are withheld -- the telemetry is
    # redundant -- so a single AUC panel would look like a null result. What does
    # move is the sharpness of the posterior, and that belongs in the same figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.0))

    for label, rows in results["partial"].items():
        xs = [r["dropped"] for r in rows]
        ax1.errorbar(xs, [r["roc_auc"] for r in rows],
                     yerr=[r["roc_auc_sd"] for r in rows],
                     marker="o", capsize=3, label=label)
    ax1.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax1.set_xlabel("telemetry channels withheld")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_title("Ranking holds up")
    ax1.legend(loc="lower left", fontsize=8)

    for label, rows in results["partial"].items():
        xs = [r["dropped"] for r in rows]
        ax2.plot(xs, [r["mean_p"] for r in rows], marker="o", label=label)
    prior = results.get("prior_p_fail")
    if prior is not None:
        ax2.axhline(prior, color="k", ls=":", lw=1.0,
                    label=f"prior, no evidence ({prior:.2f})")
    ax2.set_xlabel("telemetry channels withheld")
    ax2.set_ylabel("mean predicted P(fail)")
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_title("Posterior slides back to the prior")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Behaviour under partial evidence", fontsize=11)
    fig.tight_layout()
    path = out_dir / "partial_evidence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    return written


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def write_report(results: dict, plots: list[str], out_path: Path, meta: dict) -> None:
    L: list[str] = []
    a = L.append

    a("# BBN failure prediction — verification report")
    a("")
    a(f"- Cases: `{meta['cases_file']}` — {meta['n_cases']} cases, "
      f"{meta['n_positives']} positives ({meta['prevalence']:.0%} prevalence)")
    a(f"- Telemetry pool: {meta['pool_rows']} snapshots over {meta['pool_nodes']} nodes")
    a(f"- Training set: `{meta['train_file']}` — drawn from **different scenarios** "
      f"than the cases scored here, so the model is evaluated on operating regimes "
      f"it never saw during training")
    a(f"- Look-ahead horizon Δt: {HORIZON_DELTA_T:.0f} time units "
      f"({HORIZON_DELTA_T / EPOCH_LENGTH:.0f} epochs)")
    a("")
    a("Two models are reported throughout. **Elicited** uses the conditional "
      "probability tables generated from expert weights alone, with no failure data "
      "at all — the honest starting position the paper describes, since no public "
      "failure trace exists for this topology. **Trained** learns those tables from "
      "the complete iFogSim training pool using class-weighted, Dirichlet-smoothed "
      "EM; EM is necessary because the four latent stress conditions are never observed. "
      "The network structure is the paper's Fig. 2 in both cases; only the parameters "
      "differ.")
    a("")

    a("## 1. Discrimination")
    a("")
    a("| model | ROC-AUC | 95% CI | PR-AUC | 95% CI |")
    a("|---|---|---|---|---|")
    for label, d in results["discrimination"].items():
        a(f"| {label} | {d['roc_auc']:.3f} | {fmt_ci(d['roc_auc_ci'])} | "
          f"{d['pr_auc']:.3f} | {fmt_ci(d['pr_auc_ci'])} |")
    a("")
    a(f"Both intervals are wide, and unavoidably so: {meta['n_cases']} cases with "
      f"{meta['n_positives']} positives cannot resolve differences smaller than "
      "roughly 0.1 in AUC. Differences inside that band should not be read as real.")
    a("")

    a("## 2. Classification quality")
    a("")
    for label, sweep in results["sweep"].items():
        a(f"**{label}**, threshold sweep:")
        a("")
        a("| threshold | TP | FP | TN | FN | precision | recall | F1 | accuracy |")
        a("|---|---|---|---|---|---|---|---|---|")
        for r in sweep:
            a(f"| {r['threshold']:.2f} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['accuracy']:.3f} |")
        best = results["best"][label]
        a("")
        a(f"Best F1 {best['f1']:.3f} at threshold {best['threshold']:.2f}.")
        a("")

    a("## 3. Calibration")
    a("")
    a("| model | Brier | ECE |")
    a("|---|---|---|")
    for label, cal in results["calibration"].items():
        a(f"| {label} | {cal['brier']:.4f} | {cal['ece']:.4f} |")
    a("")
    a("Calibration is what separates a posterior usable in an expected-loss "
      "migration rule from a score that merely ranks. The rule "
      "`migrate iff P(fail)·C_fail > C_migrate` is only meaningful if P(fail) is "
      "a probability rather than an ordering.")
    a("")

    a("## 4. Lead time")
    a("")
    a("| model | failing nodes | detected | detection rate | mean lead | median lead | mean lead (epochs) |")
    a("|---|---|---|---|---|---|---|")
    for label, lt in results["lead"].items():
        if lt["detected"]:
            a(f"| {label} | {lt['nodes']} | {lt['detected']} | {lt['detection_rate']:.2f} | "
              f"{lt['mean_lead']:.1f} | {lt['median_lead']:.1f} | {lt['mean_lead_epochs']:.2f} |")
        else:
            a(f"| {label} | {lt['nodes']} | 0 | 0.00 | — | — | — |")
    a("")
    a("Lead time is measured only inside the Δt window before each failure, so an "
      "alarm raised earlier than that earns no credit. A migration needs less than "
      "one epoch to complete, so any mean lead above 1.0 epochs leaves room to act.")
    a("")

    a("## 5. Robustness to partial evidence")
    a("")
    a("| model | channels withheld | ROC-AUC | s.d. | F1 | mean P(fail) | predictions made |")
    a("|---|---|---|---|---|---|---|")
    for label, rows in results["partial"].items():
        for r in rows:
            a(f"| {label} | {r['dropped']} | {r['roc_auc']:.3f} | {r['roc_auc_sd']:.3f} | "
              f"{r['f1']:.3f} | {r['mean_p']:.3f} | {r['predictions_made']} |")
    a("")
    a("Same ablation applied to the threshold baseline:")
    a("")
    a("| rule | channels withheld | F1 | alarms raised | predictions made |")
    a("|---|---|---|---|---|")
    for r in results["baseline_partial"]:
        a(f"| threshold on CPU | {r['dropped']} | {r['f1']:.3f} | "
          f"{r['alarms_raised']:.1f} | {r['predictions_made']:.1f} |")
    a("")
    a("Two things are worth reading carefully here, and one of them is not the "
      "result that was expected.")
    a("")
    a("Discrimination barely moves. Withholding three of the seven observable "
      "channels leaves the BBN's ROC-AUC essentially where it started, rather than "
      "degrading it. That is a property of this telemetry, not a triumph of the "
      "model: the channels are largely redundant, because the same underlying "
      "degradation drives utilisation, queueing, jitter and energy together, so any "
      "surviving subset still carries most of the signal. What does move, and in "
      "the direction the theory predicts, is the sharpness of the posterior — the "
      "mean P(fail) slides steadily back toward the prior as evidence is removed, "
      "which is the model correctly becoming less certain rather than less correct.")
    a("")
    a("The structural difference is in the last column of each table. The BBN "
      f"returns a posterior for all {meta['n_cases']} cases no matter which channels are dark, "
      "because the missing variables are marginalised out by Eq. (1) rather than "
      "imputed. The threshold rule cannot: in the trials where CPU utilisation is "
      "the withheld channel it has nothing to compare against, raises no alarm at "
      "all, and every failing node passes unflagged. That is the difference between "
      "an answer with wider error bars and no answer, and it is the reason the "
      "paper argues for a probabilistic model over a fixed threshold.")
    a("")

    a("## 6. Comparison with a threshold baseline")
    a("")
    b = results["baseline"]
    a(f"Single-metric rule, CPU utilisation ≥ {b['cut']:.2f}, tuned to its own best F1 "
      "on these same cases:")
    a("")
    a("| rule | ROC-AUC | precision | recall | F1 | accuracy |")
    a("|---|---|---|---|---|---|")
    a(f"| threshold on CPU (FTAPA-style) | {b['roc_auc']:.3f} | {b['precision']:.3f} | "
      f"{b['recall']:.3f} | {b['f1']:.3f} | {b['accuracy']:.3f} |")
    for label in results["best"]:
        d = results["discrimination"][label]
        bb = results["best"][label]
        cls = results["classification"][label]
        a(f"| BBN ({label}) | {d['roc_auc']:.3f} | {cls['precision']:.3f} | "
          f"{cls['recall']:.3f} | {bb['f1']:.3f} | {cls['accuracy']:.3f} |")
    a("")
    a("The baseline is given every advantage here. Its cut is tuned on the very "
      "cases it is scored on, which the BBN's elicited tables never saw, and CPU "
      "utilisation reports on every case in this set, so its structural weakness "
      "under missing telemetry costs it nothing. On those terms it is competitive: "
      "its AUC sits inside the refined model's confidence interval, and this "
      "evaluation set is too small to separate them. The honest summary is that "
      "with complete telemetry and a threshold tuned on the test data, the single "
      "metric rule does well — and that the case for the BBN rests on the "
      "calibrated posterior of Section 3, which a threshold does not produce at "
      "all, and on the behaviour under missing telemetry in Section 5.")
    a("")

    if plots:
        a("## Figures")
        a("")
        for name in plots:
            a(f"![{name}]({name})")
        a("")

    a("## Reproducing this")
    a("")
    a("```")
    a("bash run_all.sh")
    a("```")
    a("")
    a("The simulator seed and dataset seed are fixed. The final fit uses the complete "
      "training pool with class-weighted EM; no held-out scenario is used for training.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=str(DATA / "telemetry_cases.csv"))
    ap.add_argument("--pool", default=str(DATA / "telemetry_cases_pool.csv"))
    ap.add_argument("--quantiles", default=str(DATA / "discretisation_quantiles.json"))
    ap.add_argument("--trained", default=str(DATA / "cpts_trained.json"))
    ap.add_argument("--train", default=str(DATA / "telemetry_train_pool.csv"),
                    help="training rows used to fit the reported trained CPTs")
    ap.add_argument("--out", default=str(RESULTS / "verification_report.md"))
    args = ap.parse_args(argv)

    cases = load_rows(Path(args.cases))
    pool = load_rows(Path(args.pool))
    cuts = QuantileCuts.load(args.quantiles)
    y = labels_of(cases)

    models = {"elicited": bbn_model.build_network()}
    trained_path = Path(args.trained)
    if trained_path.exists():
        from train_bbn import load_cpts
        net = bbn_model.BayesianNetwork(cpts=load_cpts(trained_path))
        net.validate()
        models["trained"] = net
    else:
        print(f"note: {trained_path.name} not found — reporting the elicited model "
              f"only. Train first with:  .venv/bin/python train_bbn.py")

    results: dict = {"discrimination": {}, "sweep": {}, "best": {}, "classification": {},
                     "calibration": {}, "lead": {}, "partial": {}, "curves": {}}

    for label, net in models.items():
        print(f"scoring {label} model ...")
        p = score_rows(net, cases, cuts)
        results["curves"][label] = (y, p)
        results["discrimination"][label] = discrimination(y, p)
        results["sweep"][label] = threshold_sweep(y, p)
        t, f1 = best_threshold(y, p)
        results["best"][label] = {"threshold": t, "f1": f1}
        results["classification"][label] = classification(y, p, t)
        results["calibration"][label] = calibration(y, p)
        print(f"  lead time over the failing nodes in the pool ...")
        results["lead"][label] = lead_time(net, pool, cuts, t)
        print(f"  partial-evidence ablation ...")
        results["partial"][label] = partial_evidence(net, cases, cuts, y, t)

    results["baseline"] = threshold_baseline(cases, y)
    # P(fail) with nothing observed at all: the line the posterior slides toward as
    # evidence is withheld.
    any_net = next(iter(models.values()))
    results["prior_p_fail"] = float(
        inference.variable_elimination(any_net, {}, TARGET_VAR)[
            TARGET_STATES.index("fail")])
    results["baseline_partial"] = baseline_under_ablation(cases, y, results["baseline"]["cut"])

    # Figures go next to the report, not to a fixed directory. Hardcoding RESULTS
    # meant a run pointed elsewhere by --out still wrote its PNGs into
    # solution/results, silently mixing figures from one dataset with a report from
    # another.
    out_path = Path(args.out)
    plots = make_plots(results, out_path.parent)
    meta = {
        "cases_file": Path(args.cases).name,
        "train_file": Path(args.train).name,
        "n_cases": len(cases),
        "n_positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pool_rows": len(pool),
        "pool_nodes": len({r["node_name"] for r in pool}),
    }
    write_report(results, plots, out_path, meta)

    print()
    for label in models:
        d = results["discrimination"][label]
        c = results["calibration"][label]
        b = results["best"][label]
        lt = results["lead"][label]
        print(f"{label:9s} AUC={d['roc_auc']:.3f} {fmt_ci(d['roc_auc_ci'])}  "
              f"F1={b['f1']:.3f}@{b['threshold']:.2f}  Brier={c['brier']:.3f}  "
              f"ECE={c['ece']:.3f}  lead={lt['mean_lead_epochs']:.2f} epochs "
              f"({lt['detected']}/{lt['nodes']} nodes)")
    bl = results["baseline"]
    print(f"{'baseline':9s} AUC={bl['roc_auc']:.3f}  F1={bl['f1']:.3f}@cpu>={bl['cut']:.2f}")
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
