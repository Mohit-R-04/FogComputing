"""
The experimental design, in one place.

Every script that generates, splits, trains or evaluates reads these constants, so the
split between what a model may learn from and what it is judged on cannot drift.

Two independent kinds of hold-out protect the evaluation:

* **Scenario hold-out.** Five of the twelve operating regimes are never seen in
  training: the model is judged on topologies, loads and link conditions it never saw.
* **Seed hold-out.** Test runs use different seeds from training runs. Since the fault
  process is keyed on (seed, scenario, node, epoch), the failures in a test run share no
  random draw with any training run -- the defect that let the earlier comparison's
  random forest memorise "node 57 fails at epoch 31".
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IFOGSIM = ROOT / "ifogsim"
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = DATA / "models"
RESULTS = ROOT / "model" / "results"

TRAIN_SCENARIOS = (0, 2, 3, 4, 5, 6, 11)
TEST_SCENARIOS = (1, 7, 8, 9, 10)
TRAIN_SEEDS = (101, 102, 103)
TEST_SEEDS = (201, 202)
#: Seeds of the closed-loop fault-tolerance experiment on the held-out scenarios: the two
#: held-out telemetry seeds plus three that never produced any telemetry, so that the
#: paired policy comparisons rest on 25 runs per policy.
FT_SEEDS = (201, 202, 203, 204, 205)

#: One hundred decision epochs of 100 time units per run.
SIM_TIME = 10000
EPOCH = 100.0
#: Look-ahead horizon of the target variable, in simulation time units (two epochs).
HORIZON = 200.0

#: Columns never used as model inputs: identifiers, time, and ground truth.
NON_FEATURES = (
    "case_id", "scenario_id", "seed", "node_id", "node_name", "level", "epoch",
    "true_health", "ground_truth_fail", "failure_time", "failure_cause",
)


def raw_path(scenario: int, seed: int) -> Path:
    return RAW / f"scenario_{scenario:02d}_seed_{seed}.csv"


def all_runs() -> list[tuple[int, int, str]]:
    """Every (scenario, seed, split) telemetry run of the design."""
    runs = [(s, k, "train") for s in TRAIN_SCENARIOS for k in TRAIN_SEEDS]
    runs += [(s, k, "test") for s in TEST_SCENARIOS for k in TEST_SEEDS]
    return runs
