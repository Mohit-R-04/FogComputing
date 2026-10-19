# Data guide

Every data file in the project, where it comes from, and what one row means.

```text
iFogSim (ScenarioSweep)           data/raw/scenario_SS_seed_K.csv     31 runs, one JVM each
        |  build_dataset.py
        v
data/telemetry_train_pool.csv     training scenarios x training seeds  (21 runs)
data/telemetry_test_pool.csv      held-out scenarios x held-out seeds  (10 runs)
data/discretisation.json          tertile cut points (the paper's elicited network)
data/dataset_manifest.json        split, counts, failure causes, coverage, leak audit
        |  select_bbn.py -> data/models/bbn_config.json
        |  train_bbn.py
        v
data/bbn_trained.json             the trained BBN, read by Python and by the Java controller
data/models/ml_best.json          best classical model, exported for the ML policy
data/models/ftapa.json            FTAPA health-index weights and threshold
```

## 1. Raw telemetry: `data/raw/scenario_SS_seed_K.csv`

Produced by `ifogsim/src/org/fog/ft/ScenarioSweep.java` (called by
`model/generate_telemetry.py`, or by `ifogsim/run_sweep.sh`). One file is one complete
iFogSim simulation of 10,000 time units for one operating regime (scenario `SS`) and one
random seed (`K`).

**One row is one device (cloud, fog node or edge device) reporting at one epoch.** An
epoch is 100 time units, so every live device reports 100 times per run. A node that has
failed stops reporting, as a dead machine would.

| Column | Meaning | Source |
|---|---|---|
| `case_id`, `scenario_id`, `seed` | row number, regime, seed | run settings |
| `node_id`, `node_name`, `level`, `device_class` | identity; level 0 cloud, 1 fog, 2 edge | iFogSim topology |
| `epoch` | simulation time of the snapshot | CloudSim clock |
| `cpu_util` | work executed in the window / work the host could execute, 0-1 | iFogSim execution |
| `ram_util` | allocated RAM / host RAM, 0-1 | iFogSim RAM provisioner |
| `queue_depth` | tuples resident in link and scheduler queues plus the window's backlog | iFogSim queues |
| `uplink_latency` | link delay plus transmission load toward the parent, ms | iFogSim links |
| `jitter_pktloss` | coefficient of variation of tuple inter-arrival times, 0-1 | tuple arrivals |
| `handover_rate` | topology changes over the last 5 epochs (own re-attachments for an edge device; users attaching/leaving for a fog node) | mobility driver |
| `residual_energy` | 1 - energy used / declared energy capacity, 0-1 | iFogSim power model |
| `temperature` | lumped thermal model driven by sustained utilisation, plus 0.8 C sensor noise | agent model |
| `cpu_trend`, `queue_trend`, `latency_trend`, `temp_trend` | change against the mean of the agent's last 3 readings (relative for queue and latency) | agent |
| `energy_drain` | residual energy used per epoch since the last reading | agent |
| `reliability_prior` | prior reliability of the device class | design input |
| `true_health` | hidden health in [0, 1] | fault model, **never a feature** |
| `ground_truth_fail` | 1 if the node fails within the next 200 time units (two epochs) | fault model, **the label** |
| `failure_time`, `failure_cause` | when and how the node failed (wear-out, sudden, battery); empty if it never failed | fault model, **never a feature** |

An empty field means "did not report". That happens for `temperature` on edge devices
(no thermal sensor), for `jitter_pktloss` when fewer than three tuples arrived in the
window, for trends before the agent has history, and in the missing-telemetry
experiments. Every loader keeps it missing; nothing is imputed for the BBN.

### What iFogSim produces and what the project adds

iFogSim has no failure model. `FaultInjector.java` adds one: each fog node and edge
device carries a hidden health that wears faster under sustained load, backlog, heat,
energy depletion and poor links, with random shocks, about one failure in five being
sudden (no precursor), and batteries that run flat. Wear shows up inside the simulator
as fail-slow symptoms (more work per tuple, a slower network interface, extra power draw,
more heat), so the telemetry columns above are measured from iFogSim's own state after
those symptoms took effect. When health reaches zero the node fails, drops every tuple it
receives and stops reporting.

All random draws are keyed on (seed, scenario, node, epoch, stream). Two consequences:
runs with different scenarios or seeds have independent failure luck (no leak between
training and held-out runs), and two policies run on the same scenario and seed see the
same luck (a fair comparison).

## 2. Split: which runs train and which test

| | Scenarios (operating regimes) | Seeds | Runs |
|---|---|---|---:|
| Training | 0, 2, 3, 4, 5, 6, 11 | 101, 102, 103 | 21 |
| Held out | 1, 7, 8, 9, 10 | 201, 202 | 10 |

Defined in `model/design.py`. Held-out runs combine regimes **and** seeds that training
never saw. Whole runs are held out, never random rows: consecutive snapshots of one node
are nearly identical, so a random row split would test on near-copies of training data.

## 3. Pools: `telemetry_train_pool.csv`, `telemetry_test_pool.csv`

Written by `model/build_dataset.py`: every row of the training runs and every row of the
held-out runs, same columns as the raw files. Nothing is sampled away; evaluation uses the
held-out pool at its natural failure prevalence.

| | Rows | Positive rows | Prevalence | Failed nodes (wear-out / sudden / battery) |
|---|---:|---:|---:|---|
| Training pool | 314,323 | 1,508 | 0.48% | 754 (486 / 125 / 143) |
| Held-out pool | 160,823 | 422 | 0.26% | 211 (139 / 61 / 11) |

Cloud rows are kept in the pools but excluded from scoring (the datacentre has a very low
hazard by design and failed in none of the 31 runs), which leaves 159,813 held-out fog and
edge snapshots.

## 4. `dataset_manifest.json`

The split, counts per device class, failure causes, channel coverage on training rows,
the columns excluded from features, and the **leak audit**: of the node pairs that fail in
both a held-out and a training run, 2.0% fail at the same epoch (chance level). The
earlier fault process replayed the same random draws in every scenario and scored 92.1%,
which let a model "recognise" held-out failures; that defect is fixed.

## 5. Discretisation

The BBN reasons over discrete states. Two sets of cut points exist:

- `data/discretisation.json`: tertiles of the training rows (low / medium / high). Used
  only by the paper's elicited network (`predict.py --elicited`, `score_one.py --elicited`).
- Inside `data/bbn_trained.json`: the trained network's cut points, chosen on training
  rows per channel and per device class (up to 8 ordered bins placed by exact dynamic
  programming to carry the most information about failure; `model/discretise.py`).

## 6. Model files

| File | Written by | Read by |
|---|---|---|
| `data/models/bbn_config.json` | `select_bbn.py` (internal validation on training scenarios) | `train_bbn.py`, `demo.py` |
| `data/bbn_trained.json` | `train_bbn.py` | every Python script; `BbnEngine.java` in iFogSim |
| `data/models/ml_best.json` | `comparison/run_comparison.py` (`export_models.py`) | `MlPredictor.java` (ML policy) |
| `data/models/ftapa.json` | `comparison/run_comparison.py` | `HealthIndexPredictor.java` (FTAPA policy) |

`bbn_trained.json` (format `fog-bbn/1`) holds the states, parents and every CPT row (in
`itertools.product` order of the parents' states, last parent fastest), the cut points,
the reliability state per device class, the cause weights and the training metadata. The
same file drives offline evaluation and the controller inside the simulator; the Java and
Python posteriors agree to 1e-9 (`test_java_parity.py`).

## 7. Results

| Path | Contents |
|---|---|
| `model/results/verification_report.md` | the BBN on held-out runs: discrimination, alarms, calibration, early warning, causes, partial evidence |
| `model/results/predictions_test.csv` | P(fail), principal cause and its lift for every held-out snapshot |
| `model/results/bbn_selection.csv` | every configuration tried during model selection |
| `model/comparison/results/model_comparison.md` | BBN against eight alternatives, with figures |
| `model/results/ft/policy_comparison.md` | closed-loop fault tolerance in iFogSim, seven policies |
| `model/results/ft/runs/*.json` | one summary per simulation run (450: 7 policies plus the degraded, sensitivity and ablation variants, on 25 held-out runs) |
| `model/results/ft/live_run.json`, `model/results/ft/race/*.json` | the demo's live BBN run and its seven-policy race on scenario 7, seed 201 |
| `model/results/bbn_retrained_in_demo.json` | the network the demo retrains live; identical to `data/bbn_trained.json` |
