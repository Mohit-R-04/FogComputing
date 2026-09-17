# BBN failure prediction for fog nodes, trained on iFogSim data

For a file-by-file explanation of `data/`, read [DATA_GUIDE.md](DATA_GUIDE.md).

Three folders, nothing else:

```
solution/
├── ifogsim/   the simulator + the code that makes it emit data
├── data/      the data the simulator produced
└── model/     the Bayesian Belief Network
```

The project runs iFogSim over the Melbourne CBD topology across 12
configurations with exactly three tiers — edge devices, fog nodes and cloud —
records one telemetry snapshot per node per epoch, and trains a 14-variable
Bayesian Belief Network to predict node failure ahead of time.

## The honest split (read this first)

- **The telemetry is iFogSim's own output.** `cpu_util`, `ram_util`,
  `queue_depth`, `uplink_latency`, `jitter_pktloss`, `handover_rate` and the
  consumption behind `residual_energy` are all read out of live simulator state.
- **The failure label is injected.** iFogSim2 has no failure model, so
  `ground_truth_fail` comes from a proportional-hazards model in
  `FaultInjector.java`. It is the ground truth the BBN is tested against, not
  simulator output.

| Column in `data/raw/*.csv` | Source |
|---|---|
| `cpu_util` | MI executed ÷ (host MIPS × window), from `executeTuple` |
| `ram_util` | RAM provisioner available vs total |
| `queue_depth` | tuple queues + `CloudletScheduler` lists + arrivals |
| `uplink_latency` | `getUplinkLatency()` + arriving bytes ÷ bandwidth |
| `jitter_pktloss` | coefficient of variation of tuple inter-arrival times |
| `handover_rate` | `setParentId` as users walk the melbCBD mobility traces |
| `residual_energy` | `1 − getEnergyConsumption() / budget` (budget is declared) |
| `temperature` | never observed — no thermal sensor |
| `reliability_prior` | declared MTBF per device class |
| `ground_truth_fail` | injected hazard model (not the simulator) |

## Folders

**`ifogsim/`** — a clone of <https://github.com/Cloudslab/iFogSim.git> plus one
new package, `src/org/fog/ft/` (9 files). The project extension instantiates only
cloud, fog-node and edge-device tiers; proxy resources from the upstream Melbourne
CSV are not instantiated. Nothing upstream source is modified.

- `build.sh` — compiles the extension into `build/classes`
- `run_sweep.sh` — runs the 12 scenarios, writing `../data/raw/scenario_NN.csv`

**`data/`** — everything the simulation produced.

- `raw/scenario_00..11.csv` — one file per scenario, every labelled snapshot; rows contain only `cloud`, `fog` and `edge`
- `telemetry_cases.csv`, `telemetry_train.csv` — evaluation and training rows
- `cpts_trained.json` — the learned network
- `dataset_manifest.json` — the train/eval split and the channel spread report

**`model/`** — the BBN. Structure is fixed at the paper's Fig. 2; every
conditional probability table is learned from the telemetry by weighted,
Dirichlet-smoothed expectation-maximisation.

- `bbn_model.py`, `inference.py`, `cpt_tables.py`, `discretise.py` — the network
- `build_dataset.py` — merge raw files, split by scenario, discretise
- `train_bbn.py` — full-pool weighted EM training
- `predict.py`, `score_one.py` — scoring
- `verify.py` — the verification suite → `results/verification_report.md`
- `test_inference.py`, `crosscheck_pgmpy.py` — engine checks
- `MODEL_IMPROVEMENTS.md` — methods, experiments, leakage controls and before/after metrics
- `comparison/` — organized comparison with CPU threshold, logistic regression, random forest and histogram gradient boosting; see `comparison/results/model_comparison.md`

## Run

Everything runs from the `solution/` root.

```bash
# 1. compile the extension
( cd ifogsim && bash build.sh )

# 2. generate the telemetry (12 scenarios; the last one takes a few minutes)
( cd ifogsim && bash run_sweep.sh )

# 3. Python environment (once)
( cd model && bash setup_venv.sh )

# 4. dataset -> train -> predict -> verify
( cd model && .venv/bin/python build_dataset.py --cases 200 --holdout 5 --seed 999 )
( cd model && .venv/bin/python train_bbn.py )  # full pool + weighted EM
( cd model && .venv/bin/python predict.py --learned ../data/cpts_trained.json )
( cd model && .venv/bin/python verify.py )
```

Or all of it in one go:

```bash
bash run_all.sh
```

Results land in `model/results/` — see `verification_report.md` for the BBN verification metrics. For comparisons with other models, see `model/comparison/results/model_comparison.md` and its diagrams.

## Known limitations

- **Packet loss is not modelled.** iFogSim has no loss model, so only the jitter
  half of "jitter with packet loss" is measurable.
- **Energy has no battery.** `getEnergyConsumption()` only rises; a budget is
  declared in the scenario so "residual" is meaningful.
- **RAM is static.** iFogSim allocates memory once at placement, so `ram_util`
  distinguishes configurations rather than moments.

# FogComputing
