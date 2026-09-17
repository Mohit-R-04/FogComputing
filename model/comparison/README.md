# Model comparison

This folder compares the proposed BBN with representative proactive/threshold
baselines on the same iFogSim-derived cloud/fog/edge telemetry.

## Folder layout

```text
model/comparison/
├── README.md
├── run_comparison.py          comparison runner
├── models/
│   ├── bbn_model.py           adapter for data/cpts_trained.json
│   ├── threshold_model.py     CPU-only threshold baseline
│   └── classical_models.py    logistic, random forest, histogram boosting
└── results/
    ├── model_comparison.csv   machine-readable metrics
    ├── model_comparison.md    human-readable report
    ├── comparison_metrics.png metric bar chart
    └── comparison_roc_pr.png  ROC and precision-recall curves
```

## Models compared

1. **BBN fusion (proposed)** — a Bayesian log-pool of the causal BBN posterior and
a continuous telemetry risk head. The causal BBN remains the explanation path;
the risk head prevents three-state discretisation from discarding useful magnitude.
The frozen fusion weight is 30% causal BBN and 70% continuous risk, selected on
pooled out-of-fold training scenarios only. The frozen alarm threshold is 0.63.
2. **Standalone BBN** — fixed causal graph, weighted EM-trained CPTs, partial-evidence
marginalisation, and explainable latent-stress chain without the fusion head.
3. **CPU threshold** — reactive single-metric baseline. It raises an alarm when
CPU exceeds a threshold. The threshold is selected on internal training scenarios,
not on the final cases.
4. **Logistic regression** — discriminative linear baseline using the same raw
telemetry channels, median imputation, missing indicators, standardisation and
class weighting.
5. **Random forest** — nonlinear tree ensemble using the same telemetry features,
median imputation and missing indicators.
6. **Histogram gradient boosting** — nonlinear boosting baseline using the same
features and sample weights.

The classical models are not given `scenario_id`, `node_id`, `node_name`, epoch,
failure time or any other identifier. The device class is represented only as the
three-tier reliability prior (`cloud`, `fog`, `edge`) so the comparison uses the
same architectural prior as the BBN without making the class label itself a target
shortcut.

## Interpretation of the current result

The proposed BBN fusion is the best model on the primary ranking metrics in the
current held-out evaluation. Random forest still has the highest F1 at its selected
alarm threshold, so the claim is specifically about ranking quality, not every
possible operating-point metric.

## Fair evaluation protocol

- Training data: `data/telemetry_train_pool.csv`, scenarios 0, 2, 3, 4, 5, 6 and 11.
- Final cases: `data/telemetry_cases.csv`, held-out scenarios 1, 7, 8, 9 and 10.
- Quantile encoding is not used by the classical models; they consume the raw
numeric telemetry.
- Missing numeric values are median-imputed using training rows only, with explicit
missing indicators.
- Class weights use the positive target prevalence of the balanced evaluation design.
- F1 thresholds are selected using internal scenario-level validation inside the
training scenarios. The final evaluation cases are not used to choose thresholds.
- ROC-AUC, PR-AUC and Brier score are threshold-independent. F1 is reported at the
selected validation threshold.

The final cases are deliberately retained until scoring. This prevents the model
comparison from becoming a test-set tuning exercise.

## Reproduce

From the `solution/` root:

```bash
cd model
.venv/bin/python comparison/run_comparison.py
```

Outputs are written to `model/comparison/results/`.

## Interpretation

Use ROC-AUC for ranking, PR-AUC for the rare-event retrieval view, Brier score for
probability quality, ECE for calibration, and F1 for a particular alarm operating
point. No single metric is sufficient for proactive fault tolerance: the BBN also
provides latent-cause explanations and handles missing telemetry by marginalisation.
