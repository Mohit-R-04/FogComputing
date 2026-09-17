#!/usr/bin/env python3
"""Compare the proposed BBN with representative fault-prediction baselines."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
ROOT = MODEL.parent
DATA = ROOT / "data"
OUT = HERE / "results"
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(HERE / "models"))

import bbn_model
import verify
from discretise import QuantileCuts
from models.classical_models import matrix, models as classical_models
from models.threshold_model import CPUThreshold

SEED = 999
VALIDATION_SCENARIOS = {5, 11}
TARGET_POSITIVE_FRACTION = 0.40


def load_rows(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def labels(rows):
    return np.array([int(r["ground_truth_fail"]) for r in rows], dtype=int)


def sample_weights(y):
    observed = float(y.mean())
    pos = TARGET_POSITIVE_FRACTION / observed
    neg = (1.0 - TARGET_POSITIVE_FRACTION) / (1.0 - observed)
    w = np.where(y == 1, pos, neg)
    return w / w.mean()


def best_threshold(y, p):
    best = (-1.0, 0.5)
    for t in np.arange(0.01, 1.00, 0.01):
        score = f1_score(y, p >= t, zero_division=0)
        if score > best[0]:
            best = (float(score), float(t))
    return best[1], best[0]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = ((p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1
                else (p >= edges[i]) & (p <= edges[i + 1]))
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def bbn_scores(rows, net, cuts):
    return verify.score_rows(net, rows, cuts)


def model_metrics(name, y, p, threshold, calibrated=True, note=""):
    result = {
        "model": name,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(threshold),
        "f1": float(f1_score(y, p >= threshold, zero_division=0)),
        "note": note,
    }
    if calibrated:
        result["brier"] = float(brier_score_loss(y, p))
        result["ece"] = ece(y, p)
    else:
        result["brier"] = "n/a"
        result["ece"] = "n/a"
    return result


def fit_threshold_on_internal(model, train_rows, validation_rows, bbn=False, cuts=None):
    y_train = labels(train_rows)
    y_val = labels(validation_rows)
    if bbn:
        p_val = bbn_scores(validation_rows, model, cuts)
    else:
        if isinstance(model, CPUThreshold):
            model.fit(train_rows, y_train)
            p_val = model.predict_proba(validation_rows)
        else:
            model.fit(matrix(train_rows), y_train, model__sample_weight=sample_weights(y_train))
            p_val = model.predict_proba(matrix(validation_rows))[:, 1]
    threshold, _ = best_threshold(y_val, p_val)
    return threshold


def fit_final(model, train_rows, test_rows, bbn=False, cuts=None):
    y_train = labels(train_rows)
    if bbn:
        return bbn_scores(test_rows, model, cuts)
    if isinstance(model, CPUThreshold):
        model.fit(train_rows, y_train)
        return model.predict_proba(test_rows)
    model.fit(matrix(train_rows), y_train, model__sample_weight=sample_weights(y_train))
    return model.predict_proba(matrix(test_rows))[:, 1]


def make_plots(results, cases):
    labels_y = labels(cases)
    names = [r["model"] for r in results]
    aucs = [r["roc_auc"] for r in results]
    f1s = [r["f1"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, aucs, width, label="ROC-AUC", color="#2f6f8f")
    ax.bar(x + width / 2, f1s, width, label="F1 at validation threshold", color="#d18b3d")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Cloud/fog/edge fault-prediction model comparison")
    ax.set_xticks(x, names, rotation=22, ha="right")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "comparison_metrics.png", dpi=180)
    plt.close(fig)

    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for result in results:
        p = result["_p"]
        fpr, tpr, _ = roc_curve(labels_y, p)
        precision, recall, _ = precision_recall_curve(labels_y, p)
        roc_ax.plot(fpr, tpr, label=f'{result["model"]} ({result["roc_auc"]:.3f})')
        pr_ax.plot(recall, precision, label=f'{result["model"]} ({result["pr_auc"]:.3f})')
    roc_ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    roc_ax.set_title("ROC curves")
    roc_ax.set_xlabel("False-positive rate")
    roc_ax.set_ylabel("True-positive rate")
    pr_ax.set_title("Precision-recall curves")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    for ax in (roc_ax, pr_ax):
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Same held-out iFogSim cases; thresholds chosen on internal scenarios")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_roc_pr.png", dpi=180)
    plt.close(fig)


def write_report(results, manifest):
    fields = ["model", "roc_auc", "pr_auc", "brier", "ece", "threshold", "f1", "note"]
    with open(OUT / "model_comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in results])

    lines = [
        "# Model comparison on iFogSim cloud/fog/edge telemetry", "",
        "The proposed BBN is compared with a single-metric threshold and three "
        "classical machine-learning baselines.", "",
        "## Evaluation protocol", "",
        f"- Training pool: `telemetry_train_pool.csv` ({manifest['rows_train_pool']} rows).",
        f"- Final cases: `telemetry_cases.csv` ({manifest['cases']} cases, "
        f"{manifest['cases_positive']} positives).",
        "- Training scenarios and final scenarios are disjoint.",
        "- F1 thresholds were selected on internal training scenarios 5 and 11 only.",
        "- Numeric features use training-only median imputation and missing indicators.",
        "- Identifiers and scenario labels are excluded from classical-model features.", "",
        "## Results", "",
        "| Model | ROC-AUC | PR-AUC | Brier | ECE | F1 | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        def fmt(v): return v if isinstance(v, str) else f"{v:.3f}"
        lines.append(f"| {r['model']} | {fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} | "
                     f"{fmt(r['brier'])} | {fmt(r['ece'])} | {fmt(r['f1'])} | "
                     f"{r['threshold']:.2f} |")
    lines += [
        "", "## Model notes", "",
        "- **BBN:** causal latent-stress explanation, partial-evidence marginalisation "
        "and calibrated posterior intended for expected-loss migration.",
        "- **CPU threshold:** reactive single-metric score; Brier/ECE are not reported "
        "because CPU is a ranking score, not a calibrated probability.",
        "- **Logistic regression:** linear discriminative reference model.",
        "- **Random forest:** nonlinear bagged-tree reference model.",
        "- **Histogram gradient boosting:** nonlinear boosted-tree reference model.",
        "", "## Figures", "", "![Metric comparison](comparison_metrics.png)", "",
        "![ROC and PR curves](comparison_roc_pr.png)", "",
        "These results are simulation-study comparisons. The failure labels are produced "
        "by the project's seeded fault injector because iFogSim has no native failure "
        "model. The final cases are balanced for evaluation and do not represent natural "
        "deployment prevalence.",
    ]
    (OUT / "model_comparison.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(DATA / "telemetry_train_pool.csv")
    cases = load_rows(DATA / "telemetry_cases.csv")
    validation_rows = [r for r in train_rows if int(r["scenario_id"]) in VALIDATION_SCENARIOS]
    fit_rows = [r for r in train_rows if int(r["scenario_id"]) not in VALIDATION_SCENARIOS]
    y_cases = labels(cases)
    cuts = QuantileCuts.load(DATA / "discretisation_quantiles.json")

    results = []
    trained_cpts = MODEL.parent / "data" / "cpts_trained.json"
    from train_bbn import load_cpts
    bbn = bbn_model.BayesianNetwork(cpts=load_cpts(trained_cpts))
    bbn.validate()
    bbn_threshold = fit_threshold_on_internal(bbn, fit_rows, validation_rows, bbn=True, cuts=cuts)
    bbn_p = fit_final(bbn, train_rows, cases, bbn=True, cuts=cuts)
    results.append(model_metrics("BBN (proposed)", y_cases, bbn_p, bbn_threshold,
                                 note="weighted EM CPTs; causal explanation"))

    threshold = CPUThreshold().fit(fit_rows, labels(fit_rows))
    threshold_value = fit_threshold_on_internal(threshold, fit_rows, validation_rows)
    threshold.threshold = threshold_value
    threshold_p = threshold.predict_proba(cases)
    results.append(model_metrics("CPU threshold", y_cases, threshold_p, threshold_value,
                                 calibrated=False, note="single-metric reactive baseline"))

    for name, model in classical_models().items():
        threshold_value = fit_threshold_on_internal(model, fit_rows, validation_rows)
        p = fit_final(model, train_rows, cases)
        results.append(model_metrics(name, y_cases, p, threshold_value,
                                     note="raw telemetry; training-only imputation"))

    for result, p in zip(results, [bbn_p, threshold_p] + [fit_final(m, train_rows, cases) for m in classical_models().values()]):
        result["_p"] = p
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    make_plots(results, cases)
    write_report(results, manifest)
    print("wrote model/comparison/results/model_comparison.csv")
    print("wrote model/comparison/results/model_comparison.md")
    for result in results:
        print(f"{result['model']}: AUC={result['roc_auc']:.4f} PR-AUC={result['pr_auc']:.4f} "
              f"F1={result['f1']:.4f}")


if __name__ == "__main__":
    main()
