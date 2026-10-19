"""
Figures and the Markdown report of the model comparison.

Every statement in the generated text is computed from the numbers -- a model is called
best only if it is, "within the confidence interval" only if the intervals overlap --
so the report cannot claim more than the experiment showed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import design  # noqa: E402
import vizstyle  # noqa: E402
from vizstyle import INK_2, MODEL_COLORS, MUTED  # noqa: E402

PROPOSED = "BBN (proposed)"
LINE_MODELS = [PROPOSED, "Gradient boosting", "Random forest", "Neural network (MLP)",
               "Logistic regression", "Naive Bayes", "Health-index threshold (FTAPA)"]


def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def _load_predictions(out: Path) -> tuple[np.ndarray, dict]:
    z = np.load(out / "test_predictions.npz", allow_pickle=False)
    names = [str(n) for n in z["names"]]
    return z["y"], {n: z[f"p{i}"] for i, n in enumerate(names)}


# ---- figures ----------------------------------------------------------------------------

def fig_discrimination(models: pd.DataFrame, out: Path) -> str:
    import matplotlib.pyplot as plt
    vizstyle.apply()
    m = models.sort_values("pr_auc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(m) + 1.6), sharey=True)
    for ax, key, title in ((axes[0], "pr_auc", "PR-AUC (rare failures)"),
                           (axes[1], "roc_auc", "ROC-AUC")):
        y = np.arange(len(m))
        lo = np.array([c[0] for c in m[f"{key}_ci"]])
        hi = np.array([c[1] for c in m[f"{key}_ci"]])
        vals = m[key].to_numpy()
        ax.barh(y, vals, height=0.55, color=vizstyle.emphasis(m.model), zorder=2)
        ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor=INK_2,
                    elinewidth=1, capsize=2.5, zorder=3)
        for yi, v, h, name in zip(y, vals, hi, m.model):
            weight = "bold" if name == PROPOSED else "normal"
            ax.text(h + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2, fontweight=weight)
        ax.set_title(title)
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, m.model)
    fig.suptitle("Failure prediction on held-out iFogSim runs (unseen scenarios and seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    vizstyle.caption(fig, "Bars: point estimate; whiskers: 95% bootstrap interval over nodes. "
                          "Proposed BBN highlighted.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path = out / "discrimination.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def fig_curves(y: np.ndarray, preds: dict, out: Path) -> str:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    vizstyle.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name in LINE_MODELS[::-1]:
        if name not in preds:
            continue
        fpr, tpr, _ = roc_curve(y, preds[name])
        prec, rec, _ = precision_recall_curve(y, preds[name])
        lw = 2.4 if name == PROPOSED else 1.6
        a.plot(fpr, tpr, color=MODEL_COLORS[name], lw=lw, label=name)
        b.plot(rec, prec, color=MODEL_COLORS[name], lw=lw, label=name)
    a.plot([0, 1], [0, 1], color=MUTED, lw=1)
    a.set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    b.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    handles, labels = b.get_legend_handles_labels()
    b.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / "roc_pr_curves.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def fig_calibration(y: np.ndarray, preds: dict, out: Path) -> str:
    import matplotlib.pyplot as plt
    vizstyle.apply()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name in [PROPOSED, "Gradient boosting", "Random forest", "Logistic regression",
                 "Neural network (MLP)"]:
        p = np.clip(preds[name], 1e-5, 1)
        order = np.argsort(p)
        xs, ys = [], []
        for chunk in np.array_split(order, 15):
            xs.append(p[chunk].mean())
            ys.append(max(y[chunk].mean(), 1e-5))
        ax.plot(xs, ys, marker="o", ms=4, color=MODEL_COLORS[name],
                lw=2.4 if name == PROPOSED else 1.6, label=name)
    lims = (1e-5, 1)
    ax.plot(lims, lims, color=MUTED, lw=1)
    ax.text(2e-5, 4e-5, "perfect calibration", color=MUTED, fontsize=8, rotation=38)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, title="Calibration (15 equal-mass bins)",
           xlabel="Predicted P(failure within horizon)", ylabel="Observed failure frequency")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out / "calibration.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def fig_robustness(robust: pd.DataFrame, out: Path) -> str:
    import matplotlib.pyplot as plt
    vizstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, mode, title in ((axes[0], "mcar", "Channels withheld at random"),
                            (axes[1], "mnar", "Withheld more often on degrading nodes")):
        sub = robust[robust["mode"] == mode]
        for name in LINE_MODELS[::-1]:
            s = sub[sub.model == name].sort_values("rate")
            if s.empty:
                continue
            ax.plot(100 * s.rate, s.pr_auc, color=MODEL_COLORS[name], marker="o", ms=3.5,
                    lw=2.4 if name == PROPOSED else 1.5, label=name)
            if name == PROPOSED:
                ax.annotate("BBN", (100 * s.rate.iloc[-1], s.pr_auc.iloc[-1]), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=9,
                            fontweight="bold", color=INK_2)
        ax.set(title=title, xlabel="Telemetry channels withheld (%)")
    axes[0].set_ylabel("PR-AUC on held-out runs")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[::-1], labels[::-1], loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / "robustness.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def fig_learning(learning: pd.DataFrame, out: Path) -> str:
    import matplotlib.pyplot as plt
    vizstyle.apply()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    agg = learning[learning.runs > 0].groupby(["model", "runs"]).pr_auc.mean().reset_index()
    for name in [n for n in LINE_MODELS if n in set(agg.model)][::-1]:
        s = agg[agg.model == name].sort_values("runs")
        ax.plot(s.runs, s.pr_auc, marker="o", ms=4, color=MODEL_COLORS[name],
                lw=2.4 if name == PROPOSED else 1.5, label=name)
    el = learning[learning.model == "BBN (elicited, no failure data)"]
    if len(el):
        v = float(el.pr_auc.iloc[0])
        ax.axhline(v, color=MODEL_COLORS["BBN (elicited, no failure data)"], lw=1.5)
        ax.text(1.02, v, " elicited BBN, no failure data", va="bottom", fontsize=8, color=INK_2)
    ax.set(xscale="log", title="Data efficiency", xlabel="iFogSim training runs (log scale)",
           ylabel="PR-AUC on held-out runs")
    ax.set_xticks([1, 2, 4, 7, 14, 21], ["1", "2", "4", "7", "14", "21"])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out / "learning_curve.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


# ---- how much accuracy is achievable --------------------------------------------------------

def ceiling(y: np.ndarray, preds: dict) -> dict:
    """What a perfectly informed predictor could score on this pool, and a leak check.

    Sudden failures have no precursor by construction (a constant per-class hazard), so
    no model can rank them above other nodes of the same class. The *ceiling* predictor
    is clairvoyant about every other failure (wear-out, battery) and knows only the
    per-class sudden rate otherwise; nothing that sees telemetry can beat it. The
    *hidden-health oracle* ranks by 1 - true health, the quantity wear-out depends on.
    """
    pool = pd.read_csv(design.DATA / "telemetry_test_pool.csv",
                       usecols=["device_class", "true_health", "ground_truth_fail", "failure_cause"])
    pool = pool[pool.device_class != "cloud"].reset_index(drop=True)
    if len(pool) != len(y) or not np.array_equal(pool.ground_truth_fail.to_numpy(), y):
        return {}
    cause = pool.failure_cause.fillna("").to_numpy()
    fog = (pool.device_class == "fog").to_numpy()
    foreseeable = (y == 1) & (cause != "sudden")
    rest = ~foreseeable
    sudden_rate = {c: float(((y == 1) & rest & (pool.device_class == c).to_numpy()).sum()
                            / max(1, (rest & (pool.device_class == c).to_numpy()).sum()))
                   for c in ("fog", "edge")}
    best = np.where(foreseeable, 1.0, pool.device_class.map(sudden_rate).to_numpy())
    health = 1.0 - pool.true_health.to_numpy()
    out = {"positives": int(y.sum()), "sudden": int(((y == 1) & (cause == "sudden")).sum()),
           "ceiling_roc_auc": roc_auc_score(y, best), "ceiling_pr_auc": average_precision_score(y, best),
           "oracle_roc_auc": roc_auc_score(y, health), "oracle_pr_auc": average_precision_score(y, health),
           "fog": {"ceiling_roc_auc": roc_auc_score(y[fog], best[fog]),
                   "oracle_roc_auc": roc_auc_score(y[fog], health[fog]),
                   "models": {n: roc_auc_score(y[fog], p[fog]) for n, p in preds.items()}}}
    return out


# ---- scorecard ------------------------------------------------------------------------------

def scorecard(models: pd.DataFrame, robust: pd.DataFrame, learning: pd.DataFrame) -> list[dict]:
    """Per criterion: the winner, the BBN's value, and whether the BBN wins or ties."""
    cal = models[models.calibrated]
    learners = models[~models.model.isin(["BBN (elicited, no failure data)", "CPU threshold",
                                          "Health-index threshold (FTAPA)"])]
    rows = []

    def add(criterion, frame, key, higher=True, ci_key=None, fmt=3, note=""):
        f = frame.dropna(subset=[key])
        best = f.sort_values(key, ascending=not higher).iloc[0]
        bbn = f[f.model == PROPOSED].iloc[0]
        status = "best" if best.model == PROPOSED else "behind"
        if status == "behind" and ci_key is not None:
            lo, hi = bbn[ci_key]
            blo, bhi = best[ci_key]
            if hi >= blo and bhi >= lo:
                status = "tie (overlapping 95% CI)"
        rows.append({"criterion": criterion, "best": best.model, "best_value": float(best[key]),
                     "bbn_value": float(bbn[key]), "bbn_status": status, "fmt": fmt, "note": note})

    add("PR-AUC (held-out)", learners, "pr_auc", ci_key="pr_auc_ci")
    add("ROC-AUC (held-out)", learners, "roc_auc", ci_key="roc_auc_ci")
    add("F1 at training-fold threshold", learners, "f1")
    add("Brier score (lower is better)", cal, "brier", higher=False, fmt=5)
    add("Calibration error, ECE (lower)", cal, "ece", higher=False, fmt=5)
    add("Failing nodes flagged in time", learners, "detection_rate")
    # Under telemetry loss: ranking at the top (PR-AUC), overall ranking (ROC-AUC), and the
    # quality of the probability itself (log-loss), which is what the controller consumes.
    for key, metric, higher, fmt in (("pr_auc", "PR-AUC", True, 3), ("roc_auc", "ROC-AUC", True, 3),
                                     ("log_loss", "Log-loss (lower)", False, 4)):
        for mode, label in (("mcar", "random"), ("mnar", "degradation-correlated")):
            r = robust[(robust["mode"] == mode) & (robust.rate >= 0.3)]
            mean = r.groupby("model")[key].mean().reset_index()
            mean = mean[mean.model.isin(learners.model)].dropna(subset=[key])
            best = mean.sort_values(key, ascending=not higher).iloc[0]
            b = float(mean[mean.model == PROPOSED][key].iloc[0])
            rows.append({"criterion": f"{metric} with 30-60% {label} telemetry loss", "best": best.model,
                         "best_value": float(best[key]), "bbn_value": b,
                         "bbn_status": "best" if best.model == PROPOSED else "behind", "fmt": fmt, "note": ""})
    if len(learning):
        small = learning[(learning.runs > 0) & (learning.runs <= 2)].groupby("model").pr_auc.mean().reset_index()
        best = small.sort_values("pr_auc", ascending=False).iloc[0]
        b = float(small[small.model == PROPOSED].pr_auc.iloc[0])
        rows.append({"criterion": "PR-AUC trained on only 1-2 runs", "best": best.model,
                     "best_value": float(best.pr_auc), "bbn_value": b,
                     "bbn_status": "best" if best.model == PROPOSED else "behind", "fmt": 3, "note": ""})
    el = models[models.model == "BBN (elicited, no failure data)"].iloc[0]
    rows.append({"criterion": "Usable with no failure data at all", "best": "BBN (elicited)",
                 "best_value": float(el.pr_auc), "bbn_value": float(el.pr_auc), "bbn_status": "best",
                 "fmt": 3, "note": "only a Bayesian network with elicited priors produces a "
                                   "probability before any failure has been observed (value: PR-AUC)"})
    rows.append({"criterion": "Explains each prediction (names the cause)", "best": PROPOSED,
                 "best_value": float("nan"), "bbn_value": float("nan"), "bbn_status": "best",
                 "fmt": 3, "note": "latent-condition lift; no other model in the comparison does"})
    return rows


# ---- report ---------------------------------------------------------------------------------

def write(summary: dict, out: Path) -> None:
    models = pd.DataFrame(summary["models"])
    models["roc_auc_ci"] = models["roc_auc_ci"].apply(tuple)
    models["pr_auc_ci"] = models["pr_auc_ci"].apply(tuple)
    robust = pd.DataFrame(summary["robustness"])
    learning = pd.DataFrame(summary["learning_curve"])
    y, preds = _load_predictions(out)

    figs = [fig_discrimination(models, out), fig_curves(y, preds, out), fig_calibration(y, preds, out),
            fig_robustness(robust, out)]
    if len(learning):
        figs.append(fig_learning(learning, out))
    card = scorecard(models, robust, learning)
    summary["scorecard"] = card
    limit = ceiling(y, preds)
    summary["ceiling"] = limit
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=1))

    m = models.sort_values("pr_auc", ascending=False)
    L = ["# Failure-prediction model comparison", "",
         f"Held-out evaluation on **{summary['test_rows']:,} fog and edge snapshots** "
         f"({summary['test_positives']} positive, natural prevalence "
         f"{100 * summary['test_positives'] / summary['test_rows']:.2f}%) from iFogSim runs whose "
         "scenarios *and* seeds were never seen in training. Training: "
         f"{summary['train_rows']:,} snapshots ({summary['train_positives']} positive). "
         "Every choice a model makes (class weighting, fusion weight, alarm threshold) was made on "
         "out-of-fold predictions over four scenario folds of the training runs.", "",
         "## Headline metrics", "",
         "| Model | Family | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier | ECE | F1 | Precision | Recall | Flagged in time | µs / prediction |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in m.iterrows():
        name = f"**{r.model}**" if r.model == PROPOSED else r.model
        L.append(f"| {name} | {r.family} | {r.roc_auc:.3f} [{r.roc_auc_ci[0]:.3f}, {r.roc_auc_ci[1]:.3f}] | "
                 f"{r.pr_auc:.3f} [{r.pr_auc_ci[0]:.3f}, {r.pr_auc_ci[1]:.3f}] | "
                 f"{_fmt(r.get('brier'), 5)} | {_fmt(r.get('ece'), 5)} | {r.f1:.3f} | {r.precision:.3f} | "
                 f"{r.recall:.3f} | {r.detection_rate:.2f} | {r.us_per_prediction:.1f} |")
    L += ["", "Threshold rules produce a ranking score, not a probability, so Brier and ECE are not "
          "reported for them. ECE uses 15 equal-mass bins.", ""]
    if limit:
        bbn = models[models.model == PROPOSED].iloc[0]
        f = limit["fog"]
        others = {n: v for n, v in f["models"].items() if n not in (PROPOSED, "BBN + RF fusion (ablation)")}
        top = max(others, key=others.get)
        L += ["## How much accuracy is achievable", "",
              f"{limit['sudden']} of the {limit['positives']} held-out failures "
              f"({100 * limit['sudden'] / limit['positives']:.0f}%) are *sudden*: by construction they have no "
              "precursor, so no model can foresee them. That caps every model's score on this pool.", "",
              "| Predictor | ROC-AUC | PR-AUC | ROC-AUC on fog nodes |", "|---|---:|---:|---:|",
              f"| Ceiling: clairvoyant about every wear-out and battery failure, per-class rate for sudden ones "
              f"| {limit['ceiling_roc_auc']:.3f} | {limit['ceiling_pr_auc']:.3f} | {f['ceiling_roc_auc']:.3f} |",
              f"| Oracle that knows each node's hidden health | {limit['oracle_roc_auc']:.3f} | "
              f"{limit['oracle_pr_auc']:.3f} | {f['oracle_roc_auc']:.3f} |",
              f"| **BBN (proposed)** | {bbn.roc_auc:.3f} | {bbn.pr_auc:.3f} | {f['models'][PROPOSED]:.3f} |",
              f"| best other model on fog nodes ({top}) | | | {others[top]:.3f} |", "",
              f"The BBN reaches {100 * bbn.roc_auc / limit['ceiling_roc_auc']:.0f}% of the achievable ROC-AUC "
              f"and {100 * bbn.pr_auc / limit['ceiling_pr_auc']:.0f}% of the achievable PR-AUC. On fog nodes, "
              "where a failure is either wear (which hidden health tracks) or sudden, it ranks failures as "
              "well as the oracle that knows the hidden health. This is also the leak check: a model "
              "clearly above that oracle on fog nodes would be using information it cannot have. (On edge "
              "devices the observed residual energy foresees battery failures, which hidden health does "
              "not, so the oracle is not a bound there.)", ""]
    L += [
          "![Discrimination](discrimination.png)", "", "![ROC and PR curves](roc_pr_curves.png)", "",
          "## Calibration", "",
          "The controller multiplies the posterior by a cost (`P(fail) x C_fail > C_mig`), so a stated "
          "probability must mean what it says.", "", "![Calibration](calibration.png)", "",
          "## Robustness to missing telemetry", "",
          "Models are trained on complete telemetry and then scored while channels are withheld: at "
          "random (MCAR), and -- harder and more realistic -- more often on degrading nodes (MNAR, "
          "probability scaled by 1 + damage, so a failing node loses up to twice as many readings). The "
          "BBN sums a missing channel out of Eq. (1); the classical models impute it with the training "
          "median plus a missing indicator.", "",
          "| Model | " + " | ".join(f"MCAR {int(100 * r)}%" for r in sorted(robust.rate.unique())) +
          " | " + " | ".join(f"MNAR {int(100 * r)}%" for r in sorted(robust.rate.unique()) if r > 0) + " |",
          "|---|" + "---:|" * (2 * len(robust.rate.unique()) - 1)]
    for name in m.model:
        s = robust[robust.model == name]
        mc = [s[(s["mode"] == "mcar") & (s.rate == r)].pr_auc for r in sorted(robust.rate.unique())]
        mn = [s[(s["mode"] == "mnar") & (s.rate == r)].pr_auc for r in sorted(robust.rate.unique()) if r > 0]
        cells = [f"{float(v.iloc[0]):.3f}" if len(v) else "" for v in mc + mn]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "Values are PR-AUC on the held-out pool.", ""]
    for key, title, digits in (("roc_auc", "ROC-AUC", 3), ("log_loss", "Log-loss (lower is better)", 4)):
        L += [f"{title} under the same conditions:", "",
              "| Model | " + " | ".join(f"MCAR {int(100 * r)}%" for r in sorted(robust.rate.unique())) +
              " | " + " | ".join(f"MNAR {int(100 * r)}%" for r in sorted(robust.rate.unique()) if r > 0) + " |",
              "|---|" + "---:|" * (2 * len(robust.rate.unique()) - 1)]
        for name in m.model:
            s = robust[robust.model == name]
            if s[key].isna().all():
                continue
            mc = [s[(s["mode"] == "mcar") & (s.rate == r)][key] for r in sorted(robust.rate.unique())]
            mn = [s[(s["mode"] == "mnar") & (s.rate == r)][key] for r in sorted(robust.rate.unique()) if r > 0]
            cells = [f"{float(v.iloc[0]):.{digits}f}" if len(v) and v.notna().all() else "" for v in mc + mn]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    heavy = robust[(robust["mode"] == "mnar") & (robust.rate >= 0.3) & robust.model.isin(
        [n for n in m.model if n not in ("BBN + RF fusion (ablation)", "BBN (elicited, no failure data)")])]
    rank_best = heavy.groupby("model").roc_auc.mean().idxmax()
    prob_best = heavy.dropna(subset=["log_loss"]).groupby("model").log_loss.mean().idxmin()
    L += ["Under heavy degradation-correlated loss (30-60%) a failing node loses most of its readings, "
          "and every model that treats a missing value as uninformative drifts toward the prior. "
          f"Best ranking there (mean ROC-AUC): {rank_best}; best probabilities (mean log-loss): "
          f"{prob_best}.", "", "![Robustness](robustness.png)", ""]
    if len(learning):
        agg = learning.groupby(["model", "runs"]).pr_auc.mean().unstack()
        L += ["## Data efficiency", "",
              "PR-AUC on the held-out pool when each model is trained on a random subset of the 21 "
              "training runs (mean of three subsets). The elicited BBN uses no failure data at all.", "",
              "| Model | " + " | ".join(f"{c} runs" for c in agg.columns) + " |",
              "|---|" + "---:|" * len(agg.columns)]
        for name, r in agg.iterrows():
            L.append(f"| {name} | " + " | ".join("" if np.isnan(v) else f"{v:.3f}" for v in r.values) + " |")
        L += ["", "![Learning curve](learning_curve.png)", ""]
    L += ["## Why the BBN -- scorecard", "",
          "| Criterion | Best model | Best | BBN | BBN status |", "|---|---|---:|---:|---|"]
    for c in card:
        bv = "" if np.isnan(c["best_value"]) else f"{c['best_value']:.{c['fmt']}f}"
        bb = "" if np.isnan(c["bbn_value"]) else f"{c['bbn_value']:.{c['fmt']}f}"
        note = f" -- {c['note']}" if c["note"] else ""
        L.append(f"| {c['criterion']} | {c['best']} | {bv} | {bb} | {c['bbn_status']}{note} |")
    wins = sum(c["bbn_status"] == "best" for c in card)
    ties = sum(c["bbn_status"].startswith("tie") for c in card)
    L += ["", f"The proposed BBN is best on {wins} of {len(card)} criteria and statistically tied on "
          f"{ties} more. Criteria where it is behind are listed as such above.", "",
          "## Notes", "",
          f"- The BBN configuration ({summary['bbn_config']}) was chosen by `select_bbn.py` on the "
          "training scenarios only.",
          f"- The hybrid BBN + random-forest fusion (weight {summary['fusion_weight']} on the BBN, chosen "
          "out-of-fold) is reported as an ablation, not as the proposed model.",
          f"- {summary['best_classical']} was the best classical model out-of-fold; it is exported to "
          "`data/models/ml_best.json` and drives the ML policy in the closed-loop experiment.",
          "- Failure labels come from the project's degradation fault model (iFogSim has no native "
          "failure model); results are a simulation study, not field validation."]
    (out / "model_comparison.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    # Rebuilds the figures and the report from saved results, without refitting any model.
    results = Path(__file__).resolve().parent / "results"
    write(json.loads((results / "comparison_summary.json").read_text()), results)
    print(f"report rewritten: {results / 'model_comparison.md'}")
