# Model comparison on iFogSim cloud/fog/edge telemetry

The proposed BBN fusion is compared with its standalone causal BBN, a single-metric threshold, and three classical machine-learning baselines.

## Evaluation protocol

- Training pool: `telemetry_train_pool.csv` (50991 rows).
- Final cases: `telemetry_cases.csv` (200 cases, 80 positives).
- Training scenarios and final scenarios are disjoint.
- Fusion weight 0.30 and proposed threshold 0.63 were selected from pooled out-of-fold predictions on internal training scenarios only.
- Classical-model thresholds were also selected on internal scenarios.
- Numeric features use training-only median imputation and missing indicators.
- Identifiers and scenario labels are excluded from classical-model features.

## Results

| Model | ROC-AUC | PR-AUC | Brier | ECE | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Standalone BBN | 0.767 | 0.679 | 0.187 | 0.099 | 0.202 | 0.73 |
| BBN fusion (proposed) | 0.881 | 0.855 | 0.181 | 0.185 | 0.351 | 0.63 |
| CPU threshold | 0.622 | 0.600 | n/a | n/a | 0.182 | 0.61 |
| Logistic regression | 0.758 | 0.671 | 0.198 | 0.118 | 0.140 | 0.81 |
| Random forest | 0.873 | 0.845 | 0.195 | 0.203 | 0.447 | 0.54 |
| Histogram gradient boosting | 0.795 | 0.728 | 0.208 | 0.178 | 0.271 | 0.71 |

## Model notes

- **BBN fusion (proposed):** a 30% causal-BBN/70% continuous-risk Bayesian log-pool; the causal BBN remains the explanation path and the risk head recovers continuous telemetry magnitude.
- **Standalone BBN:** the causal latent-stress model without the fusion head; retained to show the value added by the proposed fusion.
- **CPU threshold:** reactive single-metric score; Brier/ECE are not reported because CPU is a ranking score, not a calibrated probability.
- **Logistic regression:** linear discriminative reference model.
- **Random forest:** nonlinear bagged-tree reference model.
- **Histogram gradient boosting:** nonlinear boosted-tree reference model.

## Figures

![Metric comparison](comparison_metrics.png)

![ROC and PR curves](comparison_roc_pr.png)

These results are simulation-study comparisons. The failure labels are produced by the project's seeded fault injector because iFogSim has no native failure model. The final cases are balanced for evaluation and do not represent natural deployment prevalence.
