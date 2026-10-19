# Bayesian Belief Network-Driven Proactive Fault Tolerance for Microservice Placement in Fog Computing Environments

UG final-year project (Phase II), Department of Information Technology, SSN College of
Engineering. Rahul Akshath Rajendran and Mohit R, guided by Dr Sabireen H.

Fog nodes are small, crowded and fragile: when one dies, every service it hosts loses its
state. This project predicts failures *before* they happen and acts on the prediction.

- **iFogSim devices generate the data.** A cardiac-monitoring application runs on the
  Melbourne CBD topology (1 cloud, 118 fog gateways, 20-80 patients' devices). Every
  device reports telemetry each epoch while a degradation fault model wears nodes out.
- **A Bayesian Belief Network learns from it.** The network of the paper's Fig. 2,
  extended with trend evidence and device-class conditioning and trained by MAP-EM,
  outputs P(node fails within two epochs | telemetry) and names the condition behind
  the risk.
- **A controller inside iFogSim acts on it.** Every epoch it migrates sessions whose
  expected loss exceeds the cost of moving (N2), places users by a risk-aware utility
  (N2) and gives each session the protection its risk warrants (N3), while the network
  keeps learning online (N1).

![System block diagram](details/Project_Block_Diagram.png)

New to the project, or preparing for the review? Read
[PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md): the changes and why, how to run everything, how each
part works, and where the important code is, explained from the ground up.

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Quick start](#2-quick-start)
3. [How it works](#3-how-it-works)
4. [Repository layout](#4-repository-layout)
5. [Verification and tests](#5-verification-and-tests)
6. [Limitations](#6-limitations)
7. [Troubleshooting](#7-troubleshooting)
8. [What changed in Phase II](#8-what-changed-in-phase-ii)

---

## 1. Results at a glance

All numbers come from held-out runs: operating regimes **and** random seeds that training
never saw. Full reports: [`model/comparison/results/model_comparison.md`](model/comparison/results/model_comparison.md),
[`model/results/ft/policy_comparison.md`](model/results/ft/policy_comparison.md),
[`model/results/verification_report.md`](model/results/verification_report.md).

### Failure prediction

159,813 fog and edge snapshots, 422 failing within the horizon (natural prevalence 0.26%).

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier | ECE | F1 |
|---|---:|---:|---:|---:|---:|
| **BBN (proposed)** | **0.866** [0.829, 0.895] | 0.639 [0.585, 0.695] | **0.00128** | **0.0005** | 0.658 |
| Random forest | 0.846 [0.802, 0.886] | 0.651 [0.592, 0.708] | 0.00157 | 0.0084 | 0.687 |
| Gradient boosting | 0.851 [0.809, 0.886] | 0.629 [0.569, 0.690] | 0.00134 | 0.0006 | 0.645 |
| Naive Bayes | 0.862 [0.822, 0.895] | 0.522 [0.465, 0.578] | 0.01510 | 0.0169 | 0.549 |
| Logistic regression | 0.840 [0.803, 0.876] | 0.469 [0.419, 0.521] | 0.00180 | 0.0008 | 0.491 |
| FTAPA health index (Gupta and Singh 2025) | 0.860 [0.822, 0.888] | 0.376 [0.324, 0.433] | n/a | n/a | 0.466 |
| Neural network (MLP) | 0.837 [0.799, 0.870] | 0.239 [0.210, 0.282] | 0.00328 | 0.0034 | 0.332 |
| BBN, elicited tables only (no failure data) | 0.750 [0.717, 0.777] | 0.015 [0.010, 0.021] | 0.0998 | 0.2938 | 0.082 |

29% of the held-out failures are *sudden* (no precursor, by design), so no model can
exceed ROC-AUC 0.872 or PR-AUC 0.712 on this pool. The BBN reaches 99% and 90% of that
ceiling, and on fog nodes it ranks failures as well as an oracle that knows each node's
hidden health (0.898 against 0.896).

### Closed-loop fault tolerance inside iFogSim

25 held-out runs per policy (5 unseen regimes x 5 unseen seeds), with identical failure luck
for every policy. *Failures + migrations* is the downtime fault tolerance controls; *all
downtime* adds mobility handovers (users walking between gateways), which no policy causes
or prevents.

| Policy | Failures + migrations / user (ms) | All downtime / user (ms) | Users hit per run | Warm standby (% of sessions) | Loop p95 (ms) | Critical deadline misses (%) |
|---|---:|---:|---:|---:|---:|---:|
| No fault tolerance | 517.2 | 603.4 | 78.5 | 0.0 | 116 | 3.82 |
| Cloud-only placement | 2.3 | 2.3 | 2.0 | 0.0 | 310 | 100.00 |
| Reactive checkpoint + backup (Rajab and Younis 2021) | 110.9 | 201.2 | 89.2 | 0.0 | 122 | 4.39 |
| Threshold migration, FTAPA (Gupta and Singh 2025) | 37.7 | 132.4 | 54.1 | 100.0 | 148 | 5.46 |
| Expected-loss controller + ML (gradient boosting) | 18.8 | 86.2 | 3.8 | 0.3 | 35 | 0.22 |
| **Proposed BBN framework** | **20.0** | **87.6** | **4.5** | **0.2** | **35** | **0.13** |
| Oracle predictor (upper bound) | 14.5 | 81.5 | 3.4 | 0.4 | 36 | 0.12 |

Paired Wilcoxon tests over the 25 runs:

- Against no fault tolerance, reactive checkpointing and FTAPA threshold migration, the BBN
  framework cuts fault-related downtime by 97%, 85% and 55% (p < 0.002) and critical
  patients' downtime by 82%, 58% and 39%, while holding a warm standby for 0.2% of sessions
  (FTAPA: every session).
- Cloud-only placement has the least downtime but its loop p95 of 310 ms misses the 150 ms
  deadline on every critical alarm.
- Against the same controller driven by gradient boosting: no significant difference with
  complete telemetry (+18.5%, p = 0.12). With 30% of the channels missing, more often on
  degrading nodes, the BBN framework has 29% less fault-related downtime (p = 0.006).
- Against a perfect predictor (the oracle knows the hidden health): the BBN framework achieves
  98.9% of the reduction the oracle achieves over no fault tolerance (20.0 against 14.5 ms
  per user).
- Ablation: without risk-aware placement fault-related downtime rises 71% (p = 0.004), without
  selective replication 58% (p = 0.02). Removing online learning (+11%) or proactive
  migration (-7%) makes no significant difference: once placement and replication are
  risk-aware, a migration's pause costs about as much as the failures it averts.
- Sensitivity: fault-related downtime stays between 20 and 23 ms per user while every
  migration cost varies 16-fold (migrations fall from 104 to 11 per run).
- With degraded telemetry, FTAPA's warm standby for every session makes it the most robust
  on fault-related downtime (33.9 against 65.9 ms, p = 0.004); total downtime is equal.

### Why a BBN

The evidence, including where the BBN is *not* ahead:

1. **Its probabilities mean what they say.** Best Brier score (0.00128) and calibration
   (ECE 0.0005) of all models, and the best log-loss at every level of telemetry loss
   tested. The controller multiplies P(fail) by costs (Eqs. 3 and 4), so this is the
   property that matters most for decisions.
2. **As accurate as the best black box, near the ceiling.** Best ROC-AUC (0.866); PR-AUC
   0.639 against the random forest's 0.651 with overlapping intervals; 99% and 90% of what
   any predictor could reach on this data.
3. **Missing telemetry is summed out, never imputed.** With 30-60% of the channels missing,
   more often on degrading nodes, the BBN keeps the best probabilities (log-loss 0.0171
   against 0.0189 for gradient boosting and 0.0217 for the random forest). For ranking,
   Naive Bayes, also a Bayesian network, holds up best (ROC-AUC 0.821; BBN 0.700, random
   forest 0.572). In the closed loop with 30% of the channels missing that way, the
   BBN-driven controller has 29% less fault-related downtime than the ML-driven one
   (p = 0.006).
4. **It works before any failure has been seen.** With the expert-elicited tables alone it
   reaches ROC-AUC 0.750; a classifier cannot be trained without failure labels.
5. **It explains every alarm.** Each P(fail) comes with the latent condition driving it
   (shown live in the demo): battery failures are blamed on energy stress (86%), and for
   half of the sudden failures, which have no precursor, no condition is raised at all.
6. **Where it is not ahead.** The random forest has a slightly higher PR-AUC and F1 (0.687
   against 0.658) and flags more failing nodes in time (68% against 63%). With complete
   telemetry the ML-driven controller's fault-related downtime is slightly lower (18.8
   against 20.0 ms, not significant). A perfect predictor would save another 5.5 ms per
   user.

---

## 2. Quick start

**Requirements:** macOS or Linux, Java 17 or newer (developed with OpenJDK 21), Python
3.10 or newer (developed with 3.12). The Python environment is created on first use
(`model/setup_venv.sh`).

```bash
bash run_demo.sh                    # the terminal demonstration, under a minute
bash run_demo.sh --no-pause         # the same, with the live run played without pacing
bash run_all.sh                     # regenerate everything from scratch, about 45 minutes
bash run_all.sh --skip-selection    # same, reusing the selected BBN configuration
```

`run_demo.sh` runs the full pipeline once if no results exist yet, then shows nine stages:
the build, the simulated system, the telemetry, the dataset and leak audit, **live BBN
training**, the prediction comparison, **the trained BBN running fault tolerance inside
iFogSim live** (every migration, failure and recovery as it happens), every policy on the
same held-out run, and the verdict.

`run_all.sh` does, in order:

| Step | Command (from `model/`) | Output |
|---|---|---|
| 1 compile the iFogSim extension | `bash ../ifogsim/build.sh` | `ifogsim/build/classes` |
| 2 generate telemetry (31 runs) | `python generate_telemetry.py` | `data/raw/scenario_SS_seed_K.csv` |
| 3 build the dataset | `python build_dataset.py` | pools, manifest, tertiles |
| 4 select the BBN configuration | `python select_bbn.py` | `data/models/bbn_config.json`, `results/bbn_selection.csv` |
| 5 train the BBN | `python train_bbn.py` | `data/bbn_trained.json` |
| 6 verify it on held-out runs | `python predict.py; python verify.py` | `results/verification_report.md` |
| 7 compare with eight alternatives | `python comparison/run_comparison.py` | `comparison/results/model_comparison.md`, `data/models/ml_best.json`, `ftapa.json` |
| 8 closed-loop fault tolerance | `python run_ft_experiments.py` | `results/ft/policy_comparison.md` |
| 9 tests | `pytest test_inference.py test_pipeline.py test_java_parity.py; python crosscheck_pgmpy.py` | |

Other useful commands (from `model/`, with `.venv/bin/python`):

```bash
python score_one.py --device-class fog --cpu 1.0 --queue 380 --latency 161 --temperature 81 \
                    --temp-trend 5 --energy-drain 0.0032 --handovers 16 --show-work   # worn node: P = 0.78
python score_one.py --device-class fog --cpu 1.0 --temperature 81 --energy-drain 0.0032  # most channels dark
python score_one.py --device-class fog --cpu 0.0 --queue 0 --latency 105 --temperature 31  # healthy: P = 0.0007
python predict.py --elicited                                           # paper network, no data
python run_ft_experiments.py --quick                                   # one run per policy
python comparison/report_comparison.py                                 # rebuild the report only
```

One simulation by hand (from `ifogsim/`):

```bash
CP="build/classes:jars/*:jars/commons-math3-3.5/*"
java -cp "$CP" org.fog.ft.FtExperiment --policy bbn --scenario 7 --seed 201 --log pretty \
     --bbn ../data/bbn_trained.json --out /tmp/run.json
java -cp "$CP" org.fog.ft.ScenarioSweep 7 201 ../data/raw/example.csv     # telemetry only
```

`FtExperiment` options: `--policy no-ft|cloud-only|reactive|threshold|ml|bbn|oracle`,
`--scenario 0-11`, `--seed N`, `--out run.json`, `--log quiet|pretty|json`,
`--dropout none|mcar|mnar --rate 0.3`, `--cost-scale 1.0`, `--no-online`,
`--no-risk-placement`, `--no-migration`, `--no-replication`, `--telemetry rows.csv`,
`--bbn/--ml/--ftapa model.json`, `--sim-time 10000`.

---

## 3. How it works

### 3.1 The simulation (Java, `ifogsim/src/org/fog/ft/`)

The extension is a new package on top of an unmodified iFogSim 2 checkout. Each run is one
JVM (CloudSim keeps its clock and entities in statics).

**Topology.** Three tiers built from the Melbourne CBD data shipped with iFogSim
(`ifogsim/dataset/`): one cloud datacentre, 118 fog gateways at their real locations, and
20-80 user devices moving along the Melbourne CBD mobility traces shipped with iFogSim and
handing over between gateways.
Every node is an `InstrumentedFogDevice`, a `FogDevice` subclass.

**Application.** Cardiac monitoring: vitals flow from a client module on the user's device
to an `ecg_analyser` on the serving fog node, which sends an alarm back to the device's
actuator (the critical loop, deadline 150 ms) and a summary to the cloud for archival.
Patients are 25% critical (ICU), 45% standard and 30% best-effort (`Criticality.java`); the
class sets the weight of a failure, the state size and the placement weights.

**Operating regimes** (`ScenarioSweep.java`). Twelve regimes vary capacity, load, sensors,
backhaul and power. Seven train the model and five are held out:

| Id | Regime | What differs | Split |
|---:|---|---|---|
| 0 | baseline | 40 users, 2,800 MIPS fog nodes | train |
| 1 | idle-fleet | 20 users, one reading per 100 units | **held out** |
| 2 | chatty-sensors | two sensors per user | train |
| 3 | thin-fog | half-size fog nodes | train |
| 4 | starved-fog | quarter-size fog nodes | train |
| 5 | fat-fog | double-size fog nodes, two sensors | train |
| 6 | weak-edge | slow, hot user devices | train |
| 7 | dense-edge | 80 users | **held out** |
| 8 | slow-backhaul | 250 ms fog uplinks with a fifth of the bandwidth | **held out** |
| 9 | fog-heavy | 60 users, heavier analysis | **held out** |
| 10 | edge-heavy | heavier client module, hot devices | **held out** |
| 11 | saturated | thin fog, weak edge, slow backhaul | train |

**Fault model** (`FaultInjector.java`). iFogSim has no failures, so one is injected. Each
fog node and user device has a hidden health in [0, 1] (older hardware starts lower) that
loses, every epoch, `W(class) * exp(s * S) * LogNormal(0, 0.35)`, where the stress `S`
weighs utilisation, backlog, heat, energy depletion, link latency, jitter and handovers.
Rare shocks remove 0.1-0.3 at once; about one failure in five is *sudden* (no precursor);
user devices also die when their battery runs flat. When health reaches zero the node
fails: it drops every tuple and stops reporting.

**Fail-slow symptoms.** Wear shows up *inside* the simulator before the failure: a worn
node needs more CPU work per tuple (queues and loop delays grow), its network interface
slows, it draws extra power and runs hotter. The telemetry measures those effects from
iFogSim's own state; the network never sees the health.

**Telemetry** (`InstrumentedFogDevice`, `NodeTelemetry`). Every 100 time units each device
reports CPU and RAM utilisation, queue depth, uplink latency, arrival jitter, handovers,
residual energy and temperature (fog and cloud only), plus five trends computed by its agent
(CPU, queue, latency, temperature, energy drain). Channels can be missing; nothing is
imputed. The label is whether the node fails within the next 200 time units (two epochs).

**Common random numbers.** Every random draw is keyed on (seed, scenario, node, epoch,
stream) (`FtRandom`). Different scenarios and seeds therefore have independent failure
luck (no leak between training and held-out runs), and two policies run on the same
scenario and seed face the same luck (a fair comparison).

### 3.2 The Bayesian Belief Network (Python, `model/`)

**Structure** (`bbn_model.py`). The paper's Fig. 2: eight measurement channels feed four
latent conditions (computational stress, network degradation, energy stress, thermal
stress), which with the device-class reliability prior feed `node_failure`. Phase II adds,
after validating each on training folds:

- the five trend channels as evidence, each feeding its own condition;
- the device class as a parent of every evidence channel and every latent condition.

| Latent condition | Evidence |
|---|---|
| computational stress | cpu_util, ram_util, queue_depth, cpu_trend, queue_trend |
| network degradation | uplink_latency, jitter_pktloss, handover_rate, latency_trend |
| energy stress | residual_energy, energy_drain |
| thermal stress | temperature, temp_trend |

**Discretisation** (`discretise.py`). Each channel is cut into up to 8 ordered bins,
separately for each device class, placed by exact dynamic programming where they carry the most
information about failure (training runs only). The paper's tertiles remain for the
elicited network.

**Tables.** Latent tables are *ranked nodes* (Fenton, Neil and Caballero): each parent state
maps to a point, and the condition is a normal around the weighted score cut at its state
boundaries. The paper elicits those points by hand; here they are learned
(`ranked_node.py`), one per device class and parent state, with a Gaussian prior that pulls
them toward the elicited values. With 8 bins a free table for computational stress (five
evidence parents) would need 32,768 rows per device class; a ranked node needs 40 numbers
per class. The `node_failure` table is learned with the elicited table as a Dirichlet
prior.

**Training** (`train_bbn.py`, `bbn_fast.py`). The latent conditions are never observed,
so training is MAP expectation-maximisation over 314,323 training snapshots (43,374
distinct discretised patterns), about 10 seconds. The configuration (latent-table form,
number of bins, global or per-class bins, prior strengths, iterations) was chosen by
`select_bbn.py` with four scenario folds of the training runs; see
[`model/MODEL_IMPROVEMENTS.md`](model/MODEL_IMPROVEMENTS.md) for what each change bought.

**Inference** (`bbn_fast.py`, `inference.py`, `BbnEngine.java`). Exact. A channel that did
not report is summed out of Eq. (1), never imputed, so a prediction exists however many
channels are dark. The principal cause of an alarm is the latent condition whose
P(high) rose most above its prior for that device class (a condition the telemetry says
nothing about is therefore never named). The same JSON file (`data/bbn_trained.json`) drives the
Python evaluation and the Java engine inside iFogSim; their posteriors agree to within
1e-15 (the test requires 1e-9).

### 3.3 The fault-tolerance controller (Java, `FaultToleranceController.java`)

A CloudSim entity that runs the paper's Fig. 3 loop once per epoch, just after every
device has reported:

1. **N1, inference.** Read each fog node's report; compute P(fail) with the BBN.
2. **N2, expected-loss migration (Eq. 3).** Migrate a session when
   `P(fail) x C_fail > C_mig`, i.e. above the session's own threshold `P* = C_mig / C_fail`.
   `C_fail` is the patient's criticality times the outage a failure would cause at the
   session's protection tier; `C_mig` adds the stop-and-copy pause, the transfer, a fixed
   control cost and any latency given up at the new node.
3. **N2, risk-aware placement (Eq. 2).** New users, handovers and migration targets go to
   the candidate maximising `U = a(1 - L) + b(1 - E) + g(1 - P)`: predicted loop latency
   against the deadline (an M/M/1 estimate including the node's load), radio energy
   (access distance) and failure risk, weighted by criticality.
4. **N3, selective replication (Eq. 4).** Each session gets the tier minimising overhead
   plus expected loss: none, periodic checkpoint, or warm standby, with at most 20% of
   sessions holding a warm standby.
5. **Online learning.** Once a report's horizon has passed, its outcome updates the
   `node_failure` table as a Dirichlet posterior around the offline one (every 5 epochs);
   reports from nodes the controller acted on are censored.

Between epochs it reacts: a failure is detected after a 30 ms heartbeat timeout (50 ms by
link loss without heartbeats) and each affected session recovers as its tier allows
(warm standby 35 ms, checkpoint about 70 ms, cold restart 330 ms, all including detection).

**Policies compared** (`FtPolicy.java`); all run through this one class, differing only in
the predictor and in switches:

| Policy | What it does |
|---|---|
| No fault tolerance | nearest-gateway placement; a dead node's users reconnect and cold-start |
| Cloud-only placement | every analyser in the cloud: no fog state to lose, every alarm crosses the WAN |
| Reactive checkpoint + backup (Rajab and Younis 2021) | heartbeat detection, checkpoint recovery for every session |
| Threshold migration, FTAPA (Gupta and Singh 2025) | migrate when a fixed-weight health index crosses a tuned threshold; a warm backup for every session |
| Expected-loss controller + ML | the proposed controller driven by the best classical model (gradient boosting) instead of the BBN |
| **Proposed BBN framework** | the controller above driven by the BBN, with online learning |
| Oracle predictor (upper bound) | the same controller driven by the best possible predictor: it knows each node's hidden health and the wear law (not future random draws) |

Cost constants are in `FtCostModel.java` (state 50-300 KB moved at 12.5 KB per ms,
stop-and-copy pause 1 ms + 10% of the transfer, warm-standby budget 20% of sessions); the
sensitivity experiment scales every migration cost by 0.25-4.

### 3.4 Evaluation

**Prediction** (`comparison/run_comparison.py`). Nine models on the same data. Every
choice a model makes (class weighting, alarm threshold, fusion weight) is made on
out-of-fold predictions over four scenario folds of the training runs; the held-out pool is
scored once. Reported: ROC-AUC and PR-AUC with bootstrap intervals over nodes, Brier score,
ECE, F1, failing nodes flagged in time, robustness as channels go missing at random (MCAR)
or more often on degrading nodes (MNAR), learning curves and inference cost.

**Closed loop** (`run_ft_experiments.py`). Seven policies on the same 25 held-out runs
(5 unseen scenarios x 5 unseen seeds) with identical randomness; downtime split into
failures + migrations (what fault tolerance controls) and handovers (mobility), users hit
and evacuated, standby overhead, loop delay and deadline misses; paired Wilcoxon tests;
the telemetry-driven policies again with 30% of channels missing (MNAR); the BBN policy
across cost ratios and with each component removed.

---

## 4. Repository layout

```text
FogComputing/
├── README.md, DATA_GUIDE.md, dictionary.md      this guide, every data file, terminology
├── PROJECT_EXPLAINED.md                          beginner-friendly walkthrough for the review
├── run_demo.sh, run_all.sh                       the demonstration, the whole pipeline
├── details/                                      project documents, papers, reviews,
│   ├── Project_Block_Diagram.png (.drawio)       the system block diagram (black and white)
│   ├── BBN_Structure.png (.drawio)               the structure of the trained BBN (black and white)
│   ├── *_colour.png (.drawio)                    colour versions of the two figures above
│   ├── Project_Architecture_Detailed.png         a detailed architecture drawing
│   └── Project_Explained.docx (.pdf)             changes, how to run, how it works, the code
├── ifogsim/                                      iFogSim 2 (upstream, unmodified) + build.sh, run_sweep.sh
│   └── src/org/fog/ft/                           the project's Java package
├── data/                                         generated telemetry, pools, trained models
└── model/                                        Python: BBN, experiments, reports, tests
    ├── comparison/                               the model comparison
    └── results/                                  verification report, FT experiment results
```

### Java package `org.fog.ft`

| File | Role |
|---|---|
| `MelbCBDFaultToleranceSim.java` | builds and runs one simulation (scenario, seed, policy) |
| `FtExperiment.java` | command line of one closed-loop run |
| `ScenarioSweep.java`, `ScenarioSpec.java` | the twelve regimes; telemetry-only runs |
| `InstrumentedFogDevice.java` | a `FogDevice` that reports telemetry, degrades and genuinely fails |
| `InstrumentedActuator.java` | records the end-to-end loop delay of every alarm |
| `FaultInjector.java`, `FtRandom.java` | degradation fault model; keyed random draws |
| `NodeTelemetry.java`, `TelemetryCollector.java` | one snapshot; the labelled CSV writer |
| `MobilityDriver.java`, `Topology.java` | users walking the traces; candidate nodes and re-association |
| `ServiceSession.java`, `SessionRegistry.java` | one user's analysis session; the run-wide registry |
| `Criticality.java` | patient classes: weight, state size, placement weights |
| `FaultToleranceController.java` | the controller: N1-N3, detection, recovery, metrics |
| `FtPolicy.java`, `FtCostModel.java` | the seven policies; time and cost constants |
| `RiskPredictor.java` | predictor interface |
| `BbnEngine.java`, `BbnPredictor.java` | exact BBN inference from `bbn_trained.json`; online refinement |
| `MlPredictor.java`, `HealthIndexPredictor.java`, `OraclePredictor.java` | exported ML model; FTAPA health index; best-possible predictor |
| `EventLog.java`, `FtEvents.java` | event stream for the live demo; CloudSim event tags |
| `ParityCli.java` | scores rows with the Java engines for the parity test |
| `QuietController.java`, `RunConfig.java`, `RunContext.java`, `SeededFractionalSelectivity.java` | quiet run ending, settings, run-wide state, reproducible selectivity |

### Python `model/`

| File | Role |
|---|---|
| `design.py` | the experimental design: splits, seeds, horizon, paths |
| `generate_telemetry.py`, `javarun.py` | run iFogSim from Python, in parallel JVMs |
| `build_dataset.py` | pools, tertiles, manifest, leak audit |
| `bbn_model.py`, `cpt_tables.py` | the network's structure; elicited tables |
| `discretise.py` | tertile and supervised (per-class, exact DP) cut points |
| `ranked_node.py` | learned ranked-node tables |
| `bbn_fast.py`, `inference.py` | vectorised exact inference and EM; reference variable elimination |
| `train_bbn.py`, `select_bbn.py`, `bbn_io.py` | training; configuration selection; saving and loading |
| `predict.py`, `verify.py`, `score_one.py` | held-out predictions; verification report; one node by hand |
| `features.py`, `export_models.py`, `metrics.py`, `vizstyle.py` | classical feature matrix; exports for Java; metrics; figure style |
| `comparison/` | `run_comparison.py`, `report_comparison.py`, `models/` |
| `run_ft_experiments.py` | the closed-loop experiment |
| `demo.py` | the terminal demonstration |
| `test_inference.py`, `test_pipeline.py`, `test_java_parity.py`, `crosscheck_pgmpy.py` | tests |

Every data file is described in [`DATA_GUIDE.md`](DATA_GUIDE.md).

---

## 5. Verification and tests

```bash
cd model
.venv/bin/python -m pytest -q test_inference.py test_pipeline.py test_java_parity.py
.venv/bin/python crosscheck_pgmpy.py
```

| Check | What it guarantees |
|---|---|
| `test_inference.py` (29 tests) | the paper's network (14 variables, its edges, valid tables, risk never falls as stress rises); variable elimination equals enumeration and a pen-and-paper example; a missing channel is marginalised, which is not the same as imputing it; the named cause follows the evidence and is never an unobserved condition; bad input is rejected |
| `test_pipeline.py` (18 tests) | the vectorised engine equals variable elimination, also with channels missing, and names the same cause; pattern aggregation keeps the counts; EM raises the likelihood and keeps valid tables; save/load round trip; the supervised cut points maximise mutual information (checked against brute force); the split holds out scenarios and seeds; the leak audit; no model ranks fog failures better than knowing the hidden health |
| `test_java_parity.py` (3 tests) | the BBN, its named cause and the exported ML model compute the same numbers in Java (inside iFogSim) as in Python |
| `crosscheck_pgmpy.py` | the project's inference engine matches pgmpy, an independent implementation, on the elicited and the trained network |
| leak audit (`data/dataset_manifest.json`) | 2.0% of the nodes failing in both a held-out and a training run fail at the same epoch: chance level (92.1% in the Phase I data) |

---

## 6. Limitations

1. **Simulation, not field data.** Failures come from the project's degradation model,
   because iFogSim has none. The model is plausible (wear under stress, fail-slow
   symptoms, sudden failures, batteries) but the results are a simulation study.
2. **The fault model shapes what can be learned.** For example energy drain is the
   earliest symptom of wear in this simulator, so the network names energy stress for most
   wear-out failures. Other hardware would have other precursors and would need retraining
   (the online learning helps but only adjusts the failure table).
3. **Sudden failures cannot be predicted.** 29% of the held-out failures have no
   precursor; only redundancy protects against them.
4. **Accelerated life.** One epoch stands for one hour of operation so that wear-out
   happens within a run; absolute lifetimes are not meaningful, orderings are.
5. **Heavy telemetry loss on degrading nodes defeats every predictor.** With half the
   channels missing, more often on failing nodes, all models fall toward the prior; the
   BBN keeps the best probabilities, Naive Bayes the best ranking.
6. **Costs are in downtime-equivalent units.** Only their ratios matter; the sensitivity
   experiment varies the migration cost 16-fold.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Not compiled yet. Run ./build.sh first.` | `cd ifogsim && bash build.sh` (needs `javac`, JDK 17+) |
| `dataset/edgeResources-melbCBD.csv not found` | run Java from inside `ifogsim/`; the scripts do this themselves |
| `missing data/models/ml_best.json` | run `comparison/run_comparison.py` before `run_ft_experiments.py` |
| `externally-managed-environment` from pip | use the virtual environment: `bash model/setup_venv.sh` |
| `crosscheck_pgmpy.py` skipped | pgmpy is optional: `model/.venv/bin/pip install pgmpy` |
| results differ after changing a seed | expected: seeds change the failure luck; the split and every choice stay fixed in `design.py` |
| a test is killed (exit 137) | memory pressure: do not run the tests while the 8-JVM experiments are running |

---

## 8. What changed in Phase II

- **The data were made valid.** The Phase I fault injector reused the same random draws in
  every scenario (held-out failures repeated training failures) and its failures had no
  warning signs. It was replaced by the degradation model with keyed draws; the leak audit
  is now at chance level.
- **Failures take effect.** Failed nodes drop tuples and stop reporting; users lose service
  and recover according to the policy.
- **The BBN was improved and validated** on training folds only: supervised per-class bins,
  trend evidence, learned ranked-node tables (cross-validated PR-AUC 0.041 -> 0.759).
- **The controller was built inside iFogSim** with the three novelties and six comparison
  policies, and the BBN runs in Java from the same model file.
- **Evaluation:** whole held-out runs at natural prevalence, intervals over nodes, an
  accuracy ceiling, missing-telemetry robustness, closed-loop experiments with paired tests,
  sensitivity and ablation.

A snapshot of the project as it was before Phase II is kept in `_backup/` (ignored by git).
