# Understanding the `data/` folder

This document explains every file in `solution/data/`, where it comes from, and how it is used.

## The complete pipeline in one picture

```text
ifogsim/src/org/fog/ft/ScenarioSweep.java
        |
        | selects one of 12 simulator configurations
        v
ifogsim/run_sweep.sh
        |
        | runs iFogSim once per scenario
        v
ifogsim/src/org/fog/ft/MelbCBDFaultToleranceSim.java
        |
        | builds topology, sensors, workload, mobility and placement
        v
ifogsim/src/org/fog/ft/InstrumentedFogDevice.java
        |
        | reads simulator state every 100 simulation units
        v
ifogsim/src/org/fog/ft/TelemetryCollector.java
        |
        | applies the injected failure label and writes CSV
        v
data/raw/scenario_00.csv ... scenario_11.csv
        |
        | model/build_dataset.py merges and splits by whole scenario
        v
training pool + held-out pool
        |
        | quantiles, sampling and manifest are written
        v
data/telemetry_train.csv
 data/telemetry_train_pool.csv
 data/telemetry_cases.csv
 data/telemetry_cases_pool.csv
 data/discretisation_quantiles.json
 data/dataset_manifest.json
        |
        | model/train_bbn.py learns the CPTs
        v
data/cpts_trained.json
```

There are two different stages:

1. **Simulation stage:** Java/iFogSim creates the raw telemetry.
2. **Dataset/model stage:** Python reorganises that telemetry and trains the BBN.

The Python stage does not create new simulator measurements.

---

## 1. `data/raw/`

### What this folder contains

There are 12 files:

```text
scenario_00.csv  baseline
scenario_01.csv  idle-fleet
scenario_02.csv  chatty-sensors
scenario_03.csv  thin-fog
scenario_04.csv  starved-fog
scenario_05.csv  fat-fog
scenario_06.csv  weak-edge
scenario_07.csv  dense-edge
scenario_08.csv  slow-backhaul
scenario_09.csv  fog-heavy
scenario_10.csv  edge-heavy
scenario_11.csv  saturated
```

Together they contain 89,475 telemetry rows. The visible line count is one larger
per file because the first line is the CSV header.

Each file is one complete iFogSim run. The scenario changes simulator settings such
as host MIPS/RAM, sensor rate, number of users, module size, power model and link
bandwidth/latency. The simulator then produces different queues, execution, latency,
jitter, energy and mobility behaviour.

These are not 12 copies of the same data. They are 12 operating regimes.

### How the files are created

Run:

```bash
cd ~/Desktop/solution/ifogsim
bash build.sh
bash run_sweep.sh
```

`run_sweep.sh` loops over scenario IDs. For each ID it runs:

```text
org.fog.ft.ScenarioSweep <scenario_id> <seed> <output_csv>
```

`ScenarioSweep.java` selects the `ScenarioSpec`. `MelbCBDFaultToleranceSim.java`
then:

1. starts CloudSim/iFogSim;
2. creates the cardiac-monitoring application;
3. creates the three-tier Melbourne CBD topology: cloud, fog nodes and edge devices;
4. creates edge sensors and actuators;
5. places modules edgeward;
6. runs the event simulation until time 5,000;
7. lets the collector write the raw CSV.

A run records snapshots at iFogSim's resource-management interval: 100 simulation
time units. A snapshot is one node at one simulation time.

### What creates one raw row

`InstrumentedFogDevice.java` subclasses iFogSim's `FogDevice`. It overrides
resource-management and tuple callbacks:

- `processTupleArrival` records incoming tuples, MI, bytes and arrival times;
- `executeTuple` records MI dispatched to the node CPU;
- `setParentId` counts mobility handovers;
- `manageResources` first lets iFogSim update its own state and energy, then creates
a `NodeTelemetry` snapshot.

`TelemetryCollector.java` stores those snapshots, sorts them by node and time,
labels them, and writes the file.

### Raw CSV columns

| Column | Meaning | How it is obtained |
|---|---|---|
| `case_id` | Row number inside that scenario file | Assigned by the collector when writing; not a global ID |
| `scenario_id` | Scenario that produced the row | From `ScenarioSpec.id` |
| `node_id` | iFogSim node ID | `FogDevice.getId()` |
| `node_name` | iFogSim node name | For example `cloud`, `fog-14`, `edge-3` |
| `level` | Topology level | 0 cloud, 1 fog, 2 edge |
| `device_class` | Device tier | `cloud`, `fog`, `edge` |
| `epoch` | Simulation clock time | `CloudSim.clock()` |
| `cpu_util` | CPU work fraction during the window | Executed MI divided by host MIPS × window length |
| `ram_util` | Fraction of host RAM allocated | iFogSim RAM provisioner total minus available RAM, divided by total |
| `queue_depth` | Pending/arriving work | Tuple queues + scheduler lists + arrivals/backlog |
| `uplink_latency` | Effective uplink delay | Configured link latency plus observed bytes divided by uplink bandwidth |
| `jitter_pktloss` | Arrival timing variability | Coefficient of variation of real tuple arrival intervals; blank if not measurable |
| `handover_rate` | Recent edge-to-fog parent changes | `setParentId` changes over a trailing window |
| `residual_energy` | Remaining fraction of declared energy budget | `1 - iFogSim cumulative energy consumption / declared budget` |
| `temperature` | Thermal reading | Always blank; iFogSim has no thermal sensor here |
| `reliability_prior` | Tier reliability prior | Declared value: cloud .99, fog .85, edge .70 |
| `ground_truth_fail` | Target label | 1 if injected failure occurs within the next 200 simulation units, otherwise 0 |
| `failure_time` | Injected failure timestamp | Used for verification/lead time; not used as BBN evidence |

### Which values are truly from iFogSim?

The telemetry measurements are read from live simulator state. The following are
not native iFogSim failure outputs:

- `ground_truth_fail` and `failure_time` come from the project's `FaultInjector`;
- `reliability_prior` is a declared class attribute;
- the energy consumption is from iFogSim, but the energy budget is declared because
iFogSim has no battery-capacity concept;
- `temperature` is not measured at all.

The failure injector uses a seeded proportional-hazards process. It computes a
hazard from CPU, RAM, queue, latency, energy depletion, jitter and handovers, then
samples whether the node fails during the current epoch. A row is positive if the
failure is in the next 200 simulation units.

---

## 2. `telemetry_train_pool.csv`

This is the complete training pool created by `model/build_dataset.py`.

Current contents:

- 50,991 data rows;
- scenarios 0, 2, 3, 4, 5, 6 and 11;
- no rows from held-out scenarios 1, 7, 8, 9 and 10.

It is called a pool because it contains every eligible training row. The improved
BBN uses this file by default, with class weighting during EM.

It still has the raw simulator columns. No discretisation is stored in this CSV.
Discretisation happens when Python reads the rows for the BBN.

Generated here:

```python
train_rows = [
    row for row in all_rows
    if row["scenario_id"] is one of the training scenario IDs
]
write_csv("telemetry_train_pool.csv", train_rows)
```

---

## 3. `telemetry_cases_pool.csv`

This is the complete held-out pool.

Current contents:

- 38,484 data rows;
- scenarios 1, 7, 8, 9 and 10;
- no rows from the training scenarios.

It is used for lead-time analysis. The verifier follows each failing node through
its held-out history and measures how early the model raises an alarm.

It is not used to train the CPTs.

---

## 4. `telemetry_train.csv`

This is a smaller stratified training sample made from
`telemetry_train_pool.csv`.

Current contents:

- 2,000 data rows;
- sampled to approximately 40% positive labels;
- same training scenarios as `telemetry_train_pool.csv`.

It was the original training input. It remains in the folder for comparison and for
reproducing the old baseline, but the improved `train_bbn.py` now defaults to the
complete `telemetry_train_pool.csv`.

To train the old baseline explicitly:

```bash
cd ~/Desktop/solution/model
.venv/bin/python train_bbn.py \
  --train ../data/telemetry_train.csv \
  --iterations 8 \
  --concentration 20 \
  --target-positive-fraction 0
```

---

## 5. `telemetry_cases.csv`

This is the final evaluation sample.

Current contents:

- 200 data rows;
- 80 positives;
- approximately 40% positive prevalence;
- sampled only from held-out scenarios 1, 7, 8, 9 and 10.

It is used by `predict.py` and `verify.py` to produce the final reported metrics.

The 40% positive rate is deliberate. Natural failures are rare, and a 200-row test
set at natural prevalence would contain too few positive examples to estimate AUC,
F1 or calibration usefully. The report must therefore state that these are balanced
evaluation cases, not natural deployment prevalence.

---

## 6. `discretisation_quantiles.json`

This file contains the cut points used to convert continuous telemetry into the BBN's
three ordinal states:

```text
low, medium, high
```

For example, the current global queue cut points are:

```text
below 0      -> low
0 to below 37 -> medium
37 or above  -> high
```

The exact conversion is handled by `model/discretise.py`.

### Why the file exists

The BBN is discrete, but the simulator emits continuous values. The Python pipeline
computes tertile cut points from training scenarios only. The held-out scenarios are
never used to calculate these thresholds.

That prevents test-set leakage.

The file contains:

- `_comment`: explanation of the encoding;
- `n_snapshots`: number of training-pool rows used for the cuts;
- `global`: cut points for the whole training pool;
- `by_device_class`: optional class-specific cut points.

The default BBN encoding uses global cuts. The device-class reliability prior is
handled from the known device class rather than quantised numerically, because it is
a declared hardware attribute, not a changing measurement.

Blank values are not replaced with averages. The corresponding BBN variable is left
out and marginalised during inference.

---

## 7. `dataset_manifest.json`

This is the bookkeeping and audit file for the dataset build.

It records:

- the dataset seed: `999`;
- all 12 scenario IDs;
- training scenario IDs;
- held-out scenario IDs;
- training-pool row count;
- sampled-training row count;
- held-out-pool row count;
- number of evaluation cases;
- number of positive evaluation cases;
- channel spread report.

The channel report answers: "Did a channel actually vary?"

For example, current overall spread includes:

- `cpu_util`: 247 distinct values;
- `queue_depth`: 711 distinct values;
- `uplink_latency`: 3,119 distinct values;
- `jitter_pktloss`: 2,483 reported distinct values;
- `temperature`: 0 reported values, intentionally;
- `ram_util`: 10 values, because memory is allocated once at placement;
- `reliability_prior`: 3 values, one per tier.

The manifest is written by `model/build_dataset.py` after the raw files are merged,
checked and split.

### Important counting detail

The manifest stores **data rows**, excluding headers. A shell command such as
`wc -l data/raw/scenario_00.csv` counts the header too, so it reports one more line
than the manifest's row count.

---

## 8. `cpts_trained.json`

This is not telemetry. It is the trained BBN model parameters.

CPT means **conditional probability table**. The file stores the probability rows that
the inference engine uses.

It contains 14 variables:

```text
8 evidence roots:
  cpu_util, ram_util, queue_depth, uplink_latency,
  jitter_pktloss, handover_rate, residual_energy, temperature

4 latent stress variables:
  computational_stress, network_degradation,
  energy_stress, thermal_stress

1 reliability prior:
  reliability_prior

1 target:
  node_failure
```

The table sizes show the structure:

- root variables: 1 row each;
- three-parent latent variables: 27 rows each;
- one-parent latent variables: 3 rows each;
- target variable: 243 rows because it has five three-state parents.

`model/train_bbn.py` creates this file. The improved fit:

1. reads `telemetry_train_pool.csv`;
2. converts each row to ordinal evidence using `discretisation_quantiles.json`;
3. applies normalized rare-event class weights;
4. aggregates identical discretised patterns;
5. runs 12 EM iterations;
6. applies Dirichlet smoothing with concentration 5;
7. validates every CPT row sums to one;
8. writes `cpts_trained.json`.

`model/predict.py` loads this file to calculate `P(node_failure = fail)`.

---

## What is and is not used for training

### Used by the BBN

- `cpu_util`
- `ram_util`
- `queue_depth`
- `uplink_latency`
- `jitter_pktloss` when present
- `handover_rate`
- `residual_energy`
- known device-class reliability prior
- `ground_truth_fail` as the training target

### Not used as input evidence

- `case_id`
- `scenario_id`
- `node_id`
- `node_name`
- `level`
- `device_class` as a direct feature
- `epoch`
- `failure_time`

`device_class` is used only to map the declared reliability prior and to identify
nodes during evaluation. `failure_time` is used only for lead-time verification.

`temperature` is present in the network but normally absent from evidence, so the
inference engine marginalises it rather than inventing a temperature.

---

## The simplest way to recreate all data files

From the project root:

```bash
cd ~/Desktop/solution

# Generate the 12 raw iFogSim CSV files.
(cd ifogsim && bash build.sh)
(cd ifogsim && bash run_sweep.sh)

# Merge, check, split and sample them.
(cd model && .venv/bin/python build_dataset.py \
    --cases 200 --holdout 5 --seed 999)

# Train the model parameters.
(cd model && .venv/bin/python train_bbn.py)
```

After these commands:

- `data/raw/` contains simulator output;
- the four `telemetry_*.csv` files contain dataset views;
- `discretisation_quantiles.json` contains encoding thresholds;
- `dataset_manifest.json` records the split and audit information;
- `cpts_trained.json` contains learned BBN probabilities.

The model's predictions and verification report are written under
`model/results/`, not under `data/`.
