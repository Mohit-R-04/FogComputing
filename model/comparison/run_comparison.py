#!/usr/bin/env python3
"""
Compares the proposed BBN with the alternatives on held-out iFogSim telemetry.

    python comparison/run_comparison.py            # full comparison (~15 min)
    python comparison/run_comparison.py --fast     # skip the learning curve

Protocol (the same for every model):

* Training data: every row of the training runs (7 scenarios x 3 seeds).
* Choices a model is allowed to make -- class weighting, the fusion weight, the alarm
  threshold -- are made on out-of-fold predictions over four scenario folds of the
  training runs. The held-out pool is used once, for the final numbers.
* Test data: every fog and edge snapshot of the held-out runs (5 unseen scenarios x 2
  unseen seeds) at its natural failure prevalence. Cloud rows never fail and are left
  out of the metrics so they cannot inflate them.
* Confidence intervals: bootstrap over nodes (a node's snapshots are resampled
  together, because consecutive snapshots are not independent).

What is measured:

  discrimination      ROC-AUC, PR-AUC (the honest metric for rare failures)
  calibration         Brier score, log-loss, ECE (equal-mass bins)
  alarm quality       precision, recall, F1 at the threshold chosen on training folds
  early warning       failing nodes detected inside the horizon, lead time, false alarms
  robustness          the same metrics as channels are withheld at random (MCAR) and,
                      harder, more often on degrading nodes (MNAR)
  data efficiency     PR-AUC when trained on 1 to 21 runs (and on none, for the BBN)
  cost                inference time per prediction
  explanation         whether the model names the condition behind a prediction

Outputs in comparison/results/: model_comparison.csv/.md, comparison_summary.json,
robustness.csv, learning_curve.csv and the figures. The best classical model and the
FTAPA threshold are exported to data/models/ for the closed-loop experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(HERE))

import design  # noqa: E402
import export_models  # noqa: E402
import train_bbn  # noqa: E402
import vizstyle  # noqa: E402
from models import bbn_fusion  # noqa: E402
from models.bbn_models import ElicitedBBN, NaiveBayes, ProposedBBN  # noqa: E402
from models.classical_models import gradient_boosting, logistic, mlp, random_forest  # noqa: E402
from models.threshold_model import (HI_WEIGHTS, QUEUE_SATURATION, TEMP_SAFE, TEMP_SPAN,  # noqa: E402
                                    CpuThreshold, HealthIndexThreshold)
from metrics import (bootstrap_ci, classification, early_warning, ece, f1_threshold,  # noqa: E402
                     group_index, metric_block)

OUT = HERE / "results"
N_FOLDS = 4
DROPOUT_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
LEARNING_RUNS = (1, 2, 4, 7, 14)
LEARNING_REPEATS = 3
CHANNELS = ("cpu_util", "ram_util", "queue_depth", "uplink_latency", "jitter_pktloss",
            "handover_rate", "residual_energy", "temperature", "cpu_trend", "queue_trend",
            "latency_trend", "temp_trend", "energy_drain")
PROPOSED = "BBN (proposed)"
FUSION = "BBN + RF fusion (ablation)"
CLASSICAL = ("Logistic regression", "Random forest", "Gradient boosting", "Neural network (MLP)")


# =========================================================================================
# models and their allowed variants
# =========================================================================================

def model_variants(bbn_config: dict) -> dict:
    """name -> [(variant label, factory)]; the variant is chosen on training folds."""
    return {
        PROPOSED: [("selected configuration", lambda: ProposedBBN(bbn_config))],
        "BBN (elicited, no failure data)": [("elicited tables", ElicitedBBN)],
        "Naive Bayes": [("BBN's per-class bins", lambda: NaiveBayes(bbn_config))],
        "Logistic regression": [("unweighted", lambda: logistic(None)),
                                ("class-balanced", lambda: logistic("balanced"))],
        "Random forest": [("unweighted", lambda: random_forest(None)),
                          ("class-balanced", lambda: random_forest("balanced_subsample"))],
        "Gradient boosting": [("unweighted", lambda: gradient_boosting(None)),
                              ("class-balanced", lambda: gradient_boosting("balanced"))],
        "Neural network (MLP)": [("default", mlp)],
        "Health-index threshold (FTAPA)": [("fixed weights", HealthIndexThreshold)],
        "CPU threshold": [("cpu only", CpuThreshold)],
    }


def is_calibrated(model) -> bool:
    return getattr(model, "calibrated", True)


# =========================================================================================
# metrics
# =========================================================================================

# =========================================================================================
# missing telemetry
# =========================================================================================

def withhold(df: pd.DataFrame, rate: float, mode: str, seed: int = 21) -> pd.DataFrame:
    """MCAR: each channel missing with probability ``rate``. MNAR: the rate scaled by
    (1 + damage), so a degrading node's agent drops up to twice as many readings."""
    if rate <= 0:
        return df
    rng = np.random.default_rng(seed)
    out = df.copy()
    if mode == "mnar":
        r = np.minimum(0.95, rate * (1.0 + (1.0 - df["true_health"].to_numpy(dtype=float))))
    else:
        r = np.full(len(df), rate)
    for c in CHANNELS:
        mask = rng.random(len(out)) < r
        out.loc[mask, c] = np.nan
    return out


# =========================================================================================
# the comparison
# =========================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip the learning curve")
    args = ap.parse_args(argv)
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    design.MODELS.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(design.DATA / "telemetry_train_pool.csv")
    test_all = pd.read_csv(design.DATA / "telemetry_test_pool.csv")
    test = test_all[test_all.device_class != "cloud"].reset_index(drop=True)
    y_test = test.ground_truth_fail.to_numpy(dtype=int)
    groups = group_index(test)
    bbn_config = train_bbn.load_config()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, groups=train.scenario_id))
    train_eval = (train.device_class != "cloud").to_numpy()
    y_train = train.ground_truth_fail.to_numpy(dtype=int)
    print(f"comparison: {len(train)} training rows ({y_train.sum()} positive), "
          f"{len(test)} held-out fog/edge rows ({y_test.sum()} positive)")

    # ---- 1. variants and thresholds from out-of-fold predictions ------------------------
    chosen, oof = {}, {}
    for name, variants in model_variants(bbn_config).items():
        best = None
        for label, factory in variants:
            p = np.full(len(train), np.nan)
            for tr, va in folds:
                p[va] = factory().fit(train.iloc[tr]).predict_proba(train.iloc[va])
            probe = factory()
            if is_calibrated(probe):
                score = log_loss(y_train[train_eval], np.clip(p[train_eval], 1e-6, 1 - 1e-6))
            else:
                score = -average_precision_score(y_train[train_eval], p[train_eval])
            print(f"  out-of-fold  {name:34s} {label:22s} "
                  f"{'log-loss' if is_calibrated(probe) else '-PR-AUC'} {score:.5f}")
            if best is None or score < best[0]:
                best = (score, label, factory, p)
        chosen[name] = (best[1], best[2])
        oof[name] = best[3]

    thresholds = {n: f1_threshold(y_train[train_eval], oof[n][train_eval])[0] for n in oof}

    # Fusion ablation: weight chosen on the same out-of-fold predictions.
    weights = np.round(np.linspace(0, 1, 11), 2)
    fusion_scores = [log_loss(y_train[train_eval], np.clip(bbn_fusion.pool(
        oof[PROPOSED][train_eval], oof["Random forest"][train_eval], w), 1e-6, 1 - 1e-6)) for w in weights]
    fusion_w = float(weights[int(np.argmin(fusion_scores))])
    oof[FUSION] = bbn_fusion.pool(oof[PROPOSED], oof["Random forest"], fusion_w)
    thresholds[FUSION] = f1_threshold(y_train[train_eval], oof[FUSION][train_eval])[0]
    print(f"  fusion weight on the BBN chosen out-of-fold: {fusion_w}")

    # ---- 2. final fits on all training runs; held-out predictions -----------------------
    fitted, pred, fit_seconds = {}, {}, {}
    for name, (label, factory) in chosen.items():
        t = time.time()
        fitted[name] = factory().fit(train)
        fit_seconds[name] = time.time() - t
        pred[name] = fitted[name].predict_proba(test)
    pred[FUSION] = bbn_fusion.pool(pred[PROPOSED], pred["Random forest"], fusion_w)
    np.savez_compressed(OUT / "test_predictions.npz", y=y_test,
                        names=np.array(list(pred)), **{f"p{i}": pred[n] for i, n in enumerate(pred)})

    # ---- 3. held-out metrics ---------------------------------------------------------------
    names = list(chosen) + [FUSION]
    rows = []
    for name in names:
        calibrated = name == FUSION or is_calibrated(fitted[name])
        m = metric_block(y_test, pred[name], calibrated)
        m["roc_auc_ci"] = bootstrap_ci(y_test, pred[name], groups, roc_auc_score)
        m["pr_auc_ci"] = bootstrap_ci(y_test, pred[name], groups, average_precision_score)
        cls = classification(y_test, pred[name], thresholds[name])
        warn = early_warning(test, pred[name], thresholds[name])
        model = fitted.get(name)
        rows.append({
            "model": name,
            "variant": chosen[name][0] if name in chosen else f"BBN weight {fusion_w}",
            "family": getattr(model, "family", "hybrid"),
            "calibrated": calibrated,
            **m, **{k: cls[k] for k in ("threshold", "precision", "recall", "f1", "accuracy")},
            "detection_rate": warn["detection_rate"], "mean_lead": warn["mean_lead"],
            "false_alarms_per_1000": warn["false_alarms_per_1000"],
            "explains": bool(getattr(model, "explains", False)),
            "missing_data": getattr(model, "handles_missing", "imputes (median + indicator)"),
            "fit_seconds": fit_seconds.get(name, float("nan")),
        })
        print(f"  held-out     {name:34s} AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  "
              f"F1 {cls['f1']:.3f}  Brier {m.get('brier', float('nan')):.5f}  "
              f"detect {warn['detection_rate']:.2f}")

    # ---- 4. inference latency ----------------------------------------------------------------
    sample = test.sample(min(20000, len(test)), random_state=3)
    for r in rows:
        name = r["model"]
        if name == FUSION:
            r["us_per_prediction"] = next(x for x in rows if x["model"] == PROPOSED)["us_per_prediction"] + \
                next(x for x in rows if x["model"] == "Random forest")["us_per_prediction"]
            continue
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            fitted[name].predict_proba(sample)
            best = min(best, time.perf_counter() - t)
        r["us_per_prediction"] = 1e6 * best / len(sample)

    # ---- 5. robustness to missing telemetry -------------------------------------------------
    robust_rows = []
    for mode in ("mcar", "mnar"):
        for rate in DROPOUT_LEVELS:
            perturbed = withhold(test, rate, mode)
            preds = {n: fitted[n].predict_proba(perturbed) for n in chosen}
            preds[FUSION] = bbn_fusion.pool(preds[PROPOSED], preds["Random forest"], fusion_w)
            for n in names:
                calibrated = n == FUSION or is_calibrated(fitted[n])
                m = metric_block(y_test, preds[n], calibrated)
                cls = classification(y_test, preds[n], thresholds[n])
                robust_rows.append({"mode": mode, "rate": rate, "model": n, **m,
                                    "recall": cls["recall"], "precision": cls["precision"]})
            print(f"  robustness   {mode} {rate:.1f}: BBN PR-AUC "
                  f"{robust_rows[-len(names)]['pr_auc']:.4f}")
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(OUT / "robustness.csv", index=False)

    # ---- 6. data efficiency -------------------------------------------------------------------
    learning = pd.DataFrame()
    if not args.fast:
        runs = train[["scenario_id", "seed"]].drop_duplicates().to_numpy()
        rng = np.random.default_rng(5)
        lrows = []
        learn_models = [PROPOSED, "Naive Bayes", *CLASSICAL]
        for n_runs in LEARNING_RUNS:
            for rep in range(LEARNING_REPEATS):
                pick = runs[rng.choice(len(runs), n_runs, replace=False)]
                keys = set(map(tuple, pick))
                sub = train[[tuple(k) in keys for k in train[["scenario_id", "seed"]].to_numpy()]]
                if sub.ground_truth_fail.sum() < 3:
                    continue
                for name in learn_models:
                    try:
                        p = chosen[name][1]().fit(sub).predict_proba(test)
                        lrows.append({"runs": n_runs, "repeat": rep, "model": name,
                                      "rows": len(sub), "positives": int(sub.ground_truth_fail.sum()),
                                      "roc_auc": roc_auc_score(y_test, p),
                                      "pr_auc": average_precision_score(y_test, p)})
                    except Exception as exc:  # a learner that cannot fit tiny data scores nothing
                        print(f"    {name} could not fit {n_runs} run(s): {exc}")
            print(f"  learning curve: {n_runs} run(s) done")
        full = next(r for r in rows if r["model"] == PROPOSED)
        for name in learn_models:
            r = next(x for x in rows if x["model"] == name)
            lrows.append({"runs": len(runs), "repeat": 0, "model": name, "rows": len(train),
                          "positives": int(y_train.sum()), "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"]})
        el = next(x for x in rows if x["model"] == "BBN (elicited, no failure data)")
        lrows.append({"runs": 0, "repeat": 0, "model": "BBN (elicited, no failure data)", "rows": 0,
                      "positives": 0, "roc_auc": el["roc_auc"], "pr_auc": el["pr_auc"]})
        learning = pd.DataFrame(lrows)
        learning.to_csv(OUT / "learning_curve.csv", index=False)

    # ---- 7. exports for the closed-loop experiment ----------------------------------------------
    table = pd.DataFrame(rows)
    oof_ll = {n: log_loss(y_train[train_eval], np.clip(oof[n][train_eval], 1e-6, 1 - 1e-6)) for n in CLASSICAL}
    best_ml = min(oof_ll, key=oof_ll.get)
    m = fitted[best_ml]
    export_models.export_sklearn(best_ml, m.kind, m.model, m.spec, design.MODELS / "ml_best.json")
    export_models.export_ftapa(thresholds["Health-index threshold (FTAPA)"], design.MODELS / "ftapa.json",
                               {k: v for k, v in HI_WEIGHTS.items()},
                               {"queue_saturation": QUEUE_SATURATION, "temp_safe": TEMP_SAFE,
                                "temp_span": TEMP_SPAN})
    parity = test.sample(2000, random_state=17)
    parity_out = parity[["device_class", *CHANNELS]].copy()
    parity_out["p_ml"] = m.predict_proba(parity)
    parity_out["p_bbn"] = fitted[PROPOSED].predict_proba(parity)
    parity_out.to_csv(design.MODELS / "parity_sample.csv", index=False)
    print(f"  exported {best_ml} (best classical model out-of-fold) and the FTAPA threshold to "
          f"{design.MODELS}")

    # ---- 8. report ----------------------------------------------------------------------------
    summary = {
        "test_rows": int(len(test)), "test_positives": int(y_test.sum()),
        "train_rows": int(len(train)), "train_positives": int(y_train.sum()),
        "bbn_config": bbn_config, "fusion_weight": fusion_w, "best_classical": best_ml,
        "models": json.loads(table.to_json(orient="records")),
        "robustness": json.loads(robust.to_json(orient="records")),
        "learning_curve": json.loads(learning.to_json(orient="records")) if len(learning) else [],
        "seconds": round(time.time() - t_start, 1),
    }
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=1))
    table.drop(columns=["roc_auc_ci", "pr_auc_ci"]).assign(
        roc_auc_lo=[c[0] for c in table.roc_auc_ci], roc_auc_hi=[c[1] for c in table.roc_auc_ci],
        pr_auc_lo=[c[0] for c in table.pr_auc_ci], pr_auc_hi=[c[1] for c in table.pr_auc_ci],
    ).to_csv(OUT / "model_comparison.csv", index=False)

    import report_comparison
    report_comparison.write(summary, OUT)
    print(f"comparison finished in {summary['seconds']} s; report: {OUT / 'model_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
