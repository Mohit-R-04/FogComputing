# Data guide

## One-line flow

```text
iFogSim → raw scenario CSVs → train/test files → trained BBN CPTs
```

## 1. Raw simulator data

Folder:

```text
data/raw/
```

Files:

```text
scenario_00.csv ... scenario_11.csv
```

Each scenario is one complete iFogSim experiment with different workload,
capacity, bandwidth, or device settings.

One row means:

```text
one cloud/fog/edge device measured at one simulation time
```

The raw files contain telemetry such as:

```text
CPU, RAM, queue, latency, jitter, handovers, energy
```

They also contain:

```text
ground_truth_fail
failure_time
```

These two failure fields are created by the project's `FaultInjector.java`, because
iFogSim does not have a native failure model.

Generate raw files:

```bash
cd ~/Desktop/solution/ifogsim
bash build.sh
bash run_sweep.sh
```

## 2. Training and held-out scenarios

A scenario is one complete simulation run. The project keeps whole scenarios
separate instead of randomly mixing rows.

| Group | Scenarios | Purpose |
|---|---|---|
| Training | `0, 2, 3, 4, 5, 6, 11` | The BBN learns from these |
| Held out | `1, 7, 8, 9, 10` | The BBN is tested on these later |

**Held out** means kept separate from training. The model does not learn from
held-out rows, but their labels are used afterward to measure accuracy.

Think of it as:

```text
training scenarios = exam papers used for study
held-out scenarios = new exam papers used for testing
```

## 3. `telemetry_train_pool.csv`

Created by `model/build_dataset.py`.

```text
50,991 rows
scenarios: 0, 2, 3, 4, 5, 6, 11
```

This is the complete training data used by the improved BBN. It still contains
continuous values such as:

```text
cpu_util = 0.73
queue_depth = 42
```

## 4. `telemetry_cases_pool.csv`

Created by `model/build_dataset.py`.

```text
38,484 rows
scenarios: 1, 7, 8, 9, 10
```

This is the complete unseen test history. It is not used to train the BBN. It is
used for testing and lead-time analysis.

The BBN receives telemetry, predicts failure, and is then compared with:

```text
ground_truth_fail
```

The label is the answer key, not an input feature.

## 5. `telemetry_train.csv`

A smaller 2,000-row training sample from `telemetry_train_pool.csv`. It is kept
for comparison with the old model. The improved model uses the full pool instead.

## 6. `telemetry_cases.csv`

The final evaluation sample:

```text
200 cases
80 positive cases
```

It is selected only from held-out scenarios and is used by `predict.py`, `verify.py`,
and the model-comparison script.

## 7. `discretisation_quantiles.json`

The simulator produces continuous values. The standalone BBN converts them to:

```text
low / medium / high
```

This file stores the conversion cut points. The cut points are calculated from
training data only.

Blank measurements are left blank and marginalised by the BBN; they are not
replaced with fake values.

## 8. `dataset_manifest.json`

The dataset record. It stores:

```text
seed
training scenarios
held-out scenarios
row counts
positive-case count
channel coverage
```

## 9. `cpts_trained.json`

The trained BBN parameters. `CPT` means **conditional probability table**.

Created by:

```bash
cd ~/Desktop/solution/model
.venv/bin/python train_bbn.py
```

Training flow:

```text
telemetry_train_pool.csv
        ↓
low/medium/high encoding
        ↓
weighted EM training
        ↓
Dirichlet smoothing
        ↓
cpts_trained.json
```

## Rebuild the data and BBN

```bash
cd ~/Desktop/solution
(cd ifogsim && bash build.sh)
(cd ifogsim && bash run_sweep.sh)
(cd model && .venv/bin/python build_dataset.py --cases 200 --holdout 5 --seed 999)
(cd model && .venv/bin/python train_bbn.py)
```

## Short summary

```text
raw/                         = original iFogSim output
telemetry_train_pool.csv    = data used to train
telemetry_cases_pool.csv    = unseen data used to test
telemetry_cases.csv          = final test sample
dataset_manifest.json        = dataset record
cpts_trained.json            = learned BBN probabilities
```
