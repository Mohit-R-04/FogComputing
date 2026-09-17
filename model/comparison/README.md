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

1. **BBN (proposed)** — fixed causal graph, weighted EM-trained CPTs, partial-
evidence marginalisation, and an explainable latent-stress chain.
2. **CPU threshold** — reactive single-metric baseline. It raises an alarm when
CPU exceeds a threshold. The threshold is selected on internal training scenarios,
not on the final cases.
3. **Logistic regression** — discriminative linear baseline using the same raw
telemetry channels, median imputation, missing indicators, standardisation and
class weighting.
4. **Random forest** — nonlinear tree ensemble using the same telemetry features,
median imputation and missing indicators.
5. **Histogram gradient boosting** — nonlinear boosting baseline using the same
features and sample weights.

The classical models are not given `scenario_id`, `node_id`, `node_name`, epoch,
failure time or any other identifier. The device class is represented only as the
three-tier reliability prior (`cloud`, `fog`, `edge`) so the comparison uses the
same architectural prior as the BBN without making the class label itself a target
shortcut.

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
