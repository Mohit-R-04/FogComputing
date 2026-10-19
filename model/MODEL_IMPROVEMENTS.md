# How the BBN's accuracy was improved

This note records what was changed in the Bayesian Belief Network during Phase II, why,
and what each change bought. Every number in sections 2 and 3 is a cross-validated
score on the **training scenarios only** (four folds by scenario, `select_bbn.py`,
`results/bbn_selection.csv`); the held-out runs were scored once, at the end
(section 4).

## 1. Before the model: making the evaluation valid

Two defects in the Phase I data had to be fixed before any accuracy figure meant
anything.

1. **A leak between training and held-out runs.** The old fault injector drew its
   random numbers from (seed, node, epoch) with one seed for every scenario, so the
   same node failed at the same epoch in every scenario: 92.1% of the nodes failing in
   both a held-out and a training run failed at the same epoch. A model could
   "recognise" held-out failures instead of predicting them. Draws are now keyed on
   (seed, scenario, node, epoch, stream); the share is 2.0%, chance level
   (`data/dataset_manifest.json`, `test_pipeline.py`).
2. **Failures without warning signs.** The old process was a memoryless hazard: a
   node's chance of failing did not depend on its history, so even an oracle could
   barely predict it (ROC-AUC about 0.69). It was replaced by a degradation model:
   hidden health that wears faster under stress, fail-slow symptoms that iFogSim then
   measures, about one sudden failure in five, and batteries that run flat
   (`FaultInjector.java`). Proactive fault tolerance assumes failures announce
   themselves; the new model makes that true for most failures and false for the rest.

The evaluation also changed: whole held-out runs at their natural failure prevalence
(0.26%), not a balanced 200-row sample; confidence intervals from a bootstrap over
nodes; thresholds chosen on training folds.

## 2. The improvements, in order

| Step | Change | CV PR-AUC | CV log-loss |
|---|---|---:|---:|
| 0 | paper's Fig. 2 graph, tertile bins, MAP-EM | 0.041 | 0.0255 |
| 1 | supervised cut points (3 bins per channel) | 0.434 | 0.0154 |
| 2 | + five trend channels as evidence | 0.562 | 0.0138 |
| 3 | + learned ranked-node latent tables with device-class specific points, 8 bins per channel | 0.653 | 0.0131 |
| 4 | + bins chosen separately for each device class | 0.753 | 0.0114 |
| 5 | Dirichlet prior strength 5 -> 1 (selected configuration) | **0.759** | **0.0106** |

Prevalence (chance-level PR-AUC) is about 0.005 on the training runs.

**Step 1: supervised discretisation** (`discretise.py`). Tertiles put two cut points at
the 33rd and 67th percentiles, but failures live in the extreme tail of a channel:
nearly every pre-failure reading lands in the same "high" bin as a third of all healthy
readings. Supervised cuts are chosen, per channel, to carry the most mutual information
about the failure label. Mutual information is a sum over bins, so the best partition
of a fine, tail-aware candidate grid is found exactly by dynamic programming (a test
checks it against brute force).

**Step 2: trend evidence** (`bbn_model.py`, `InstrumentedFogDevice.java`). A degrading
node shows drift before it shows an extreme value. The telemetry agent reports five
trends (CPU, queue, latency, temperature, energy drain against its last three
readings), and each feeds the latent condition it belongs to. The Fig. 2 structure is
otherwise unchanged.

**Step 3: learned ranked nodes** (`ranked_node.py`, `bbn_fast.py`). With three bins per
channel the evidence still discarded most of the information (a gradient-boosting
model given only those bins did no better than the BBN). Finer bins were unlearnable
with free tables: a latent node with five parents of eight states needs 32,768 rows.
The paper already elicits latent tables with Fenton's ranked-node method (each parent
state maps to a point, the child is a normal around the weighted score); here those
points are *learned* in every EM M-step, with a Gaussian prior pulling them back to the
elicited values. That needs one number per parent state, so the evidence can be fine
while the latent conditions keep {low, medium, high} and the network keeps its
structure and its explanations. The points are learned separately for each device
class, and the evidence priors (used when a channel is missing) are class-specific too.

**Step 4: bins per device class.** A mains-powered fog gateway and a battery-powered
handset drain energy and heat up on different scales, so "high energy drain" must mean
different things. The supervised bins are chosen separately for each device class.
This was the largest single gain.

**Step 5: prior strength.** The target table's Dirichlet prior (the elicited table as
pseudo-counts) was weakened from 5 to 1 once the evidence tier was rich enough to
support it.

The final choice between 6 and 8 bins (equal within noise) was broken on robustness:
log-loss with 30% of channels withheld at random, 0.0158 for 8 bins against 0.0159.

## 3. Tried and not kept

Compared with the step-4 configuration (8 bins, Dirichlet strength 5, ranked-node prior
0.02, 60 iterations: PR-AUC 0.753, log-loss 0.0114):

| Variant | CV PR-AUC | CV log-loss | Outcome |
|---|---:|---:|---|
| class-specific evidence priors with free tables and 3 bins | 0.561 | 0.0139 | no gain on their own (step 2: 0.562); kept, because per-class bins need them |
| 10 bins per channel | 0.751 | 0.0115 | no better than 8, slower |
| ranked-node prior strength 0.1 | 0.753 | 0.0116 | slightly worse log-loss |
| Dirichlet prior strength 20 | 0.742 | 0.0132 | over-regularised |
| 120 EM iterations | 0.759 | 0.0118 | equal PR-AUC, worse log-loss |
| BBN + random-forest fusion | -- | -- | the out-of-fold optimum put all weight (1.0) on the BBN; reported as an ablation |

Class-weighted training was not used for the BBN: it distorts the probabilities, and the
cost-based controller multiplies them by costs.

## 4. Held-out result (scored once)

159,813 fog and edge snapshots from runs whose scenarios and seeds were never seen:

| | Phase I graph, elicited tables | Final BBN |
|---|---:|---:|
| ROC-AUC | 0.750 | **0.866** [0.829, 0.895] |
| PR-AUC | 0.015 | **0.639** [0.585, 0.695] |
| Brier score | 0.0998 | **0.00128** |
| ECE | 0.294 | **0.0005** |

**Is there room left?** 122 of the 422 held-out failures (29%) are sudden: by
construction they have no precursor. Even a predictor that knew every wear-out and
battery failure in advance would score at most ROC-AUC 0.872 and PR-AUC 0.712 on this
pool. The BBN reaches 99% and 90% of that ceiling. On fog nodes, where a failure is
either wear or sudden, it ranks failures as well as an oracle that knows each node's
hidden health (ROC-AUC 0.898 against 0.896). The remaining headroom is small, and it
lies mostly at the top of the ranking (PR-AUC), where the random forest is marginally
ahead (0.651, intervals overlapping).

Full reports: `results/verification_report.md` (the BBN alone) and
`comparison/results/model_comparison.md` (against eight alternatives).
