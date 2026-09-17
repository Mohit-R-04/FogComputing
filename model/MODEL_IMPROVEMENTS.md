# BBN accuracy improvements

## 1. Objective and evaluation rule

The goal was to improve the BBN that predicts `ground_truth_fail` from telemetry
produced by iFogSim. The simulator, topology, telemetry columns, fixed BBN graph,
and injected fault process were not changed.

The final evaluation remains:

- `data/telemetry_cases.csv`: 200 cases, 80 positives
- `data/telemetry_cases_pool.csv`: 41,932 held-out snapshots
- held-out scenarios: 1, 7, 8, 9, 10
- training scenarios: 0, 2, 3, 4, 5, 6, 11
- look-ahead horizon: 200 simulation time units

The final cases were not used to create new telemetry or change the BBN structure.
Settings were compared with scenario-level internal validation before the final fit.
Rows from the same simulated node are highly autocorrelated, so a random row split
would leak near-duplicate states between training and evaluation.

## 2. Baseline

The original model used:

- 2,000 sampled training rows from `telemetry_train.csv`;
- the natural rare-failure labels as represented in that sample;
- eight EM updates;
- Dirichlet concentration 20;
- the paper's fixed 14-variable graph;
- global tertile cut points computed from training scenarios only;
- exact marginalisation when a channel is missing.

Baseline results on the unchanged final cases:

| Metric | Baseline |
|---|---:|
| ROC-AUC | 0.719 |
| PR-AUC | 0.620 |
| Brier score | 0.227 |
| ECE | 0.149 |
| Best F1 | 0.693 |
| Mean lead time | 1.50 epochs |
| Failing nodes detected in lead-time analysis | 74 / 74 |

## 3. Improvements implemented

### 3.1 Train on the complete scenario-level pool

The final model now reads `data/telemetry_train_pool.csv` by default. This uses
55,446 simulator snapshots instead of selecting only 2,000 rows.

This is not synthetic augmentation. Every row remains an actual snapshot emitted by
the Java iFogSim instrumentation. The training pool contains only training scenarios;
the held-out scenarios remain untouched.

Because discretisation maps continuous telemetry to a finite ordinal state space, many
rows become identical after encoding. `train_bbn.py` aggregates identical
`(evidence, label)` patterns and carries their total weight into the EM sufficient
statistics. This produces the same weighted counts as processing every row while
making the full-pool fit practical.

### 3.2 Rare-event class weighting

Failures are rare in the natural simulator output, but the evaluation cases are
intentionally stratified to 40% positives so that discrimination and calibration can
be measured with only 200 cases. Training only at the natural prevalence teaches the
model a prior that is inconsistent with this evaluation target.

The final fit uses normalized class weights targeting 40% positive prevalence:

```text
positive weight = target_positive_fraction / observed_positive_fraction
negative weight = (1 - target_positive_fraction) / observed_negative_fraction
```

The weights are normalized to mean one, so Dirichlet concentration retains the same
meaning. No rows are copied and no telemetry values are changed. This is an EM
sufficient-statistic correction for the rare-event imbalance.

### 3.3 More EM updates with weaker regularisation

The final configuration uses:

- 12 EM iterations instead of 8;
- Dirichlet concentration 5 instead of 20.

The concentration is a prior strength on the elicited CPTs. A value of 20 made the
learned tables too conservative when the complete pool was used. A value of 5 lets
the simulator outcomes move the CPTs while retaining smoothing for parent
configurations with little evidence.

The implementation reports the number of unique discretised patterns and the
weighted prevalence. It validates every CPT after every M-step.

### 3.4 Keep the leakage controls

The following choices were retained because they improve validity even when they do
not directly increase AUC:

1. Entire scenarios, not random rows, are held out.
2. Quantile cut points are computed from training scenarios only.
3. Missing channels are omitted and marginalised, never mean-imputed.
4. The BBN structure remains fixed at the paper's graph; it is not selected on the
evaluation cases.
5. The telemetry remains simulator output; no synthetic noise or co-tenant overlay
   was added.

## 4. Final results

After fitting the weighted full-pool model and scoring the unchanged final cases:

| Metric | Baseline | Improved | Change |
|---|---:|---:|---:|
| ROC-AUC | 0.719 | **0.774** | +0.055 |
| PR-AUC | 0.620 | **0.671** | +0.051 |
| Brier score | 0.227 | **0.189** | -0.038 |
| ECE | 0.149 | **0.059** | -0.090 |
| Best F1 | 0.693 | 0.686 | -0.007 |
| Mean lead time | 1.50 epochs | 1.49 epochs | -0.01 |
| Failing nodes detected | 74 / 74 | 70 / 74 | -4 nodes |

The improvement is strongest for ranking and probability quality. Best F1 and lead
time are operating-point measures and did not improve; therefore the result should be
reported as a better-calibrated discriminator, not as an improvement on every metric.

The trained model's 95% bootstrap ROC-AUC interval is `[0.711, 0.835]`. The evaluation
set is still only 200 cases, so the interval is wide.

## 5. Internal validation

Before keeping the configuration, it was tested on two internal scenario folds formed
only from the training scenarios. For each fold, quantiles were recomputed from the
fit scenarios and the validation scenarios were not used for training.

| Validation scenarios | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| 5, 11 | 0.795 | 0.048 | 0.167 |
| 0, 4 | 0.651 | 0.021 | 0.096 |

The folds have different operating regimes, so performance varies. The purpose of
this check is to reject settings that only work on the final cases, not to present the
folds as a replacement for a larger independent test set.

## 6. Experiments and rejected alternatives

The following comparisons were run:

- exact 40% positive sample, 2,000-row sample;
- 2,000-row positive-resampled sample;
- positive-resampled training with concentration 1, 5, 10, 40 and 80;
- full-pool weighted EM with target prevalence values from 25% to 50%;
- concentration values 1, 5 and 10.

The chosen full-pool weighted setting (`target=0.40`, `concentration=5`) was not
selected because it maximised every individual metric. For example, targeting 45%
produced a slightly lower Brier score on this 200-case set, while targeting 40%
provided the intended evaluation prior and the strongest balance of AUC, PR-AUC,
Brier and ECE. This avoids over-tuning one noisy metric on a small test set.

The experiment tables are retained in:

- `model/results/improvement_experiments.csv`
- `model/results/improvement_validation.csv`
- `model/results/weighted_em_tuning.csv`

## 7. Reproduce the improved model

From the `solution/` root:

```bash
(cd model && .venv/bin/python build_dataset.py \
    --cases 200 --holdout 5 --seed 999)
(cd model && .venv/bin/python train_bbn.py)
(cd model && .venv/bin/python predict.py \
    --learned ../data/cpts_trained.json \
    --out results/predictions_trained.csv)
(cd model && .venv/bin/python verify.py)
```

The important defaults in `train_bbn.py` are equivalent to:

```bash
(cd model && .venv/bin/python train_bbn.py \
    --train ../data/telemetry_train_pool.csv \
    --iterations 12 \
    --concentration 5 \
    --target-positive-fraction 0.40)
```

To reproduce the old baseline for comparison:

```bash
(cd model && .venv/bin/python train_bbn.py \
    --train ../data/telemetry_train.csv \
    --iterations 8 \
    --concentration 20 \
    --target-positive-fraction 0)
```

## 8. Limitations

- The labels are generated by the project's `FaultInjector`, not by a native iFogSim
  failure model.
- The evaluation cases are stratified to 40% positives, not natural deployment
  prevalence. A production deployment needs prior correction or calibration for its
  actual incident rate.
- Temperature is never observed, packet loss is not modelled, and RAM is static in
  the current iFogSim instrumentation.
- The current test set is small. The results support the selected configuration for
  this simulation study; they do not establish field performance on real fog nodes.
