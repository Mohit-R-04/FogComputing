# Model comparison

Why a Bayesian Belief Network? This folder answers that with an experiment: the proposed
BBN against eight alternatives on the same iFogSim telemetry, scored on runs whose
operating regimes **and** random seeds were never seen in training.

```bash
cd model
.venv/bin/python comparison/run_comparison.py          # about 9 minutes
.venv/bin/python comparison/run_comparison.py --fast   # skip the learning curve
.venv/bin/python comparison/report_comparison.py       # rebuild report and figures only
```

The report is `results/model_comparison.md`.

## Layout

```text
model/comparison/
├── run_comparison.py        protocol: out-of-fold choices, final fits, held-out scoring,
│                            robustness, learning curve, exports for the Java controller
├── report_comparison.py     figures, scorecard and the Markdown report (every claim in the
│                            text is computed from the numbers)
├── models/
│   ├── bbn_models.py        proposed BBN, elicited BBN (no failure data), Naive Bayes
│   ├── classical_models.py  logistic regression, random forest, gradient boosting, MLP
│   ├── threshold_model.py   CPU threshold, FTAPA health index (Gupta and Singh 2025)
│   └── bbn_fusion.py        BBN + random forest log-pool (ablation only)
└── results/                 report, CSVs, figures, test_predictions.npz
```

## Models

| Model | What it is |
|---|---|
| **BBN (proposed)** | the trained network of `data/bbn_trained.json`: paper Fig. 2 graph with trend evidence and device-class conditioning, learned ranked-node latent tables, up to 8 supervised bins per channel and device class, MAP-EM |
| BBN (elicited, no failure data) | the paper's Fig. 2 network with expert-elicited tables and tertile bins; never sees a failure label |
| Naive Bayes | the simplest Bayesian network, on the same per-class bins as the BBN |
| Logistic regression, random forest, gradient boosting, neural network (MLP) | scikit-learn, on the explicit feature matrix of `features.py` |
| Health-index threshold (FTAPA) | fixed-weight mix of CPU, queue, energy and temperature against one tuned threshold |
| CPU threshold | single metric |
| BBN + RF fusion | Bayesian log-pool of the BBN and the random forest; reported as an ablation |

The classical models see the 13 telemetry channels, a missing indicator per channel
(missing values imputed with the training median) and the device class. None of them
sees identifiers, epochs, failure times or the hidden health.

## Protocol

1. **Choices made on training folds only.** Four folds by scenario over the 21 training
   runs. For each model the variant with the best out-of-fold log-loss is kept
   (unweighted vs class-balanced); thresholds are F1-optimal out-of-fold; the fusion
   weight is chosen out-of-fold.
2. **Final fits** on all training runs.
3. **One scoring** on the held-out pool: 159,813 fog and edge snapshots, 422 positive
   (0.26%, natural prevalence). 95% intervals come from a bootstrap over nodes.
4. **Robustness:** channels withheld at random (MCAR) and more often on degrading
   nodes (MNAR, probability scaled by 1 + damage), 0-60%. The BBN sums a missing
   channel out; the classical models impute it.
5. **Data efficiency:** each model retrained on 1, 2, 4, 7, 14 and 21 training runs.
6. **Exports:** the best classical model out-of-fold (`data/models/ml_best.json`) and
   the FTAPA weights (`data/models/ftapa.json`) for the closed-loop experiment, plus
   `parity_sample.csv` for the Java/Python parity test.

## Result in one paragraph

On held-out runs the BBN has the best ROC-AUC (0.866), the best Brier score (0.00128)
and the best calibration (ECE 0.0005), and is statistically tied with the random forest
on PR-AUC (0.639 vs 0.651, overlapping intervals); the random forest has a higher F1 and
flags slightly more failing nodes at its threshold. 29% of held-out failures are sudden
by design, which caps any model at ROC-AUC 0.872 and PR-AUC 0.712 on this pool; the BBN
reaches 99% and 90% of that, and on fog nodes it ranks failures as well as an oracle that
knows each node's hidden health. With telemetry missing it keeps the best probabilities
(log-loss) of all models; Naive Bayes keeps the best ranking under heavy
degradation-correlated loss. Only a Bayesian network gives a usable probability before
any failure has been observed (elicited tables: ROC-AUC 0.750), and only the BBN names the
condition behind each alarm. The full scorecard, including the criteria where the BBN is
behind, is at the end of the report.
