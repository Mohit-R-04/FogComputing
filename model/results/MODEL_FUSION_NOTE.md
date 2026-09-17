# Proposed BBN fusion model

## Why fusion was necessary

The standalone causal BBN discretises continuous telemetry into three states:
`low`, `medium`, and `high`. That is excellent for explainability and exact
partial-evidence inference, but it loses magnitude information that a tree model
can use. In the current comparison, standalone BBN ROC-AUC is 0.767 while random
forest ROC-AUC is 0.873.

Directly adding many observed telemetry parents to the BBN target was tested and
rejected: the resulting target CPT was sparse and reduced internal-fold ranking.
A five-state ordinal BBN improved PR-AUC but did not exceed the production model
reliably.

## Final design

The proposed model is a Bayesian logarithmic pool of two probability sources:

1. **Causal BBN posterior**
   - trained CPTs from `data/cpts_trained.json`;
   - latent computational, network, energy and thermal stress nodes;
   - reliability prior from cloud/fog/edge tier;
   - exact variable elimination;
   - missing telemetry marginalised;
   - causal explanation retained.

2. **Continuous telemetry risk head**
   - random-forest probability model over raw telemetry;
   - no scenario ID, node ID, node name, epoch, failure time, or target-derived
     feature;
   - training-only median imputation and missing indicators;
   - class weights targeting the 40% balanced evaluation design.

The fusion is performed in log-odds space:

```text
logit(P_fusion) = 0.30 * logit(P_causal_bbn)
                + 0.70 * logit(P_continuous_risk)
```

The 0.30/0.70 weight was selected from pooled out-of-fold predictions on internal
training scenarios only. The final `telemetry_cases.csv` was not used to choose it.
The proposed alarm threshold is frozen at `0.63`, selected on the same internal
out-of-fold predictions.

This remains BBN-driven: the BBN is one probability source, the causal explanation
path, and the missing-evidence component. The continuous risk head is an explicit
sensor-fusion component, not hidden inside the BBN CPTs.

## Current held-out result

| Model | ROC-AUC | PR-AUC | Brier | ECE | F1 |
|---|---:|---:|---:|---:|---:|
| BBN fusion | 0.881 | 0.855 | 0.181 | 0.185 | 0.351 |
| Standalone BBN | 0.767 | 0.679 | 0.187 | 0.099 | 0.202 |
| Random forest | 0.873 | 0.845 | 0.195 | 0.203 | 0.447 |

The fusion is best on ROC-AUC and PR-AUC. Random forest remains best on F1 at its
selected threshold, so the claim is not that the fusion wins every metric.

## Files

- `model/comparison/models/bbn_fusion.py` — fusion class;
- `model/comparison/run_comparison.py` — frozen weight and fair evaluation;
- `model/comparison/results/model_comparison.csv` — generated metrics;
- `model/comparison/results/model_comparison.md` — generated report;
- `model/comparison/results/comparison_metrics.png` — metric diagram;
- `model/comparison/results/comparison_roc_pr.png` — curve diagram.

## Reproduce

```bash
cd ~/Desktop/solution/model
.venv/bin/python comparison/run_comparison.py
```
