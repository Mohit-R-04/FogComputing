package org.fog.ft;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.utils.NetworkUsageMonitor;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

/**
 * The Proactive Fault-Tolerance Controller of the paper's Fig. 1, as a CloudSim entity.
 *
 * <p>Once per decision epoch (just after every device has reported) it runs the loop of
 * the paper's Fig. 3:
 * <ol>
 *   <li>read each fog node's latest telemetry;</li>
 *   <li>compute its failure risk with the policy's predictor (the BBN for the proposed
 *       policy), marginalising whatever did not report;</li>
 *   <li>migrate sessions whose expected loss from staying exceeds the cost of moving,
 *       {@code P(fail) * C_fail > C_mig} (Eq. 3), to the node of highest placement
 *       utility (Eq. 2);</li>
 *   <li>give each session the redundancy tier that minimises its expected loss plus
 *       overhead, {@code R = P(fail) * kappa} (Eq. 4), within a warm-standby budget;</li>
 *   <li>feed matured outcomes back to the predictor (online CPT refinement).</li>
 * </ol>
 * Between epochs it reacts to events: a failure is detected after a heartbeat timeout
 * (or, without heartbeats, after link loss) and each affected session is recovered in the
 * way its protection tier allows. Mobility handovers are routed through
 * {@link #chooseServingNode}, so risk-aware placement also governs where users go as
 * they move.
 *
 * <p>Every policy of {@link FtPolicy} runs through this one class; they differ only in
 * the predictor and in the switches {@link FtPolicy#proactive()},
 * {@link FtPolicy#expectedLoss()} and so on. Baselines are therefore implemented with
 * exactly the same machinery as the proposed method.
 */
public class FaultToleranceController extends SimEntity {

    /** Offset after the devices' epoch boundary, so every report of the epoch is in. */
    private static final double EPOCH_OFFSET = 1.0;
    /** Candidate serving nodes considered for one user. */
    private static final int K_CANDIDATES = 4;
    /** A session is not migrated again within this many time units. */
    private static final double MIGRATION_COOLDOWN = 2 * FtCostModel.EPOCH;
    /** Minimum utility gain before a utility-placed user is moved without a risk reason. */
    private static final double UTILITY_HYSTERESIS = 0.10;
    /** Access distance at which the energy term of Eq. (2) saturates (CBD scale). */
    private static final double ENERGY_REFERENCE_KM = 5.0;

    private final FtPolicy policy;
    private final RiskPredictor predictor;
    private final HealthIndexPredictor healthIndex;
    private final Topology topology;
    private final ScenarioSpec spec;
    private final double simTime;
    /** Analysis work one user generates, in MI per time unit. */
    private final double userMiRate;

    private final SessionRegistry registry = SessionRegistry.get();
    private final Map<Integer, Double> risk = new HashMap<Integer, Double>();
    private final Map<Integer, String> cause = new HashMap<Integer, String>();
    private final Set<Integer> draining = new HashSet<Integer>();
    private final Map<Integer, Integer> calmEpochs = new HashMap<Integer, Integer>();
    private final Map<Integer, Double> actualFailureTime = new HashMap<Integer, Double>();
    private final Map<Integer, Double> detectedAt = new HashMap<Integer, Double>();
    /** Completed proactive evacuations per node: switchover times. */
    private final Map<Integer, List<Double>> evacuations = new HashMap<Integer, List<Double>>();
    /** First proactive decision per node (for lead time). */
    private final Map<Integer, List<Double>> decisionsByNode = new HashMap<Integer, List<Double>>();
    /** Reports waiting for their horizon to elapse (online learning). */
    private final Map<Integer, ArrayDeque<NodeTelemetry>> pending =
            new HashMap<Integer, ArrayDeque<NodeTelemetry>>();

    // ---- metrics ------------------------------------------------------------------------
    private int epochs = 0;
    private long migrationsStarted = 0;
    private long migrationsCompleted = 0;
    private long migrationsAborted = 0;
    private long handovers = 0;
    private int fogFailures = 0;
    private int fogFailuresWithUsers = 0;
    private int sessionsHit = 0;
    private int sessionsEvacuated = 0;
    private int edgeFailures = 0;
    private double standbyEpochs = 0.0;
    private double activeSessionEpochs = 0.0;
    private double replicationKb = 0.0;
    private long controllerNanos = 0;
    private long predictions = 0;
    private final List<Double> leadTimes = new ArrayList<Double>();
    private final Map<String, Integer> recoveries = new LinkedHashMap<String, Integer>();
    /** Sessions on a fog node when it failed, by how the node failed. */
    private final Map<String, Integer> hitByCause = new LinkedHashMap<String, Integer>();
    private final List<double[]> migrationDecisions = new ArrayList<double[]>();
    private final JSONArray timeline = new JSONArray();
    private int failuresThisEpoch = 0;
    private int migrationsThisEpoch = 0;
    /** True health of the source node at each migration decision (analysis only). */
    private final List<Double> decisionHealth = new ArrayList<Double>();
    /** A node this worn shows clear fail-slow symptoms (work inflation >= 1.5x). */
    private static final double DEGRADED_HEALTH = 0.42;

    private boolean riskPlacement = true;
    private boolean proactiveMigration = true;
    private boolean selectiveReplication = true;

    public FaultToleranceController(String name, FtPolicy policy, RiskPredictor predictor,
                                    HealthIndexPredictor healthIndex, Topology topology,
                                    ScenarioSpec spec, double simTime) {
        super(name);
        this.policy = policy;
        this.predictor = predictor;
        this.healthIndex = healthIndex;
        this.topology = topology;
        this.spec = spec;
        this.simTime = simTime;
        this.userMiRate = spec.sensorsPerEdge * (1.0 / spec.sensorInterval) * 0.9
                * spec.analysisTupleMi * MelbCBDFaultToleranceSim.ANALYSIS_COST_SCALE;
        registry.setController(this);
    }

    public FtPolicy policy() {
        return policy;
    }

    /** Ablation switches; they only affect the expected-loss policies. */
    public void setAblation(boolean riskPlacement, boolean proactiveMigration, boolean selectiveReplication) {
        this.riskPlacement = riskPlacement;
        this.proactiveMigration = proactiveMigration;
        this.selectiveReplication = selectiveReplication;
    }

    // =====================================================================================
    // CloudSim entity
    // =====================================================================================

    @Override
    public void startEntity() {
        send(getId(), EPOCH_OFFSET, FtEvents.FT_EPOCH);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == FtEvents.FT_EPOCH) {
            epoch();
        } else if (ev.getTag() == FtEvents.FT_DETECT) {
            onDetect((Integer) ev.getData());
        } else if (ev.getTag() == FtEvents.FT_SWITCHOVER) {
            onSwitchover((Integer) ev.getData());
        }
    }

    @Override
    public void shutdownEntity() {
    }

    // =====================================================================================
    // the decision epoch
    // =====================================================================================

    private void epoch() {
        long t0 = System.nanoTime();
        double now = CloudSim.clock();
        epochs++;

        // 1-2. Telemetry and risk for every fog node the control plane believes alive.
        for (InstrumentedFogDevice n : topology.fogNodes()) {
            if (registry.isKnownFailed(n.getId())) continue;
            NodeTelemetry t = freshReport(n, now);
            if (predictor != null) {
                RiskPredictor.Prediction p = predictor.predict(n, t, now);
                risk.put(n.getId(), p.pFail);
                cause.put(n.getId(), p.cause);
                predictions++;
            }
            if (t != null && predictor != null) queue(n.getId(), t);
        }
        // Edge devices' reports also teach the network (they fail too).
        for (InstrumentedFogDevice e : topology.edges()) {
            NodeTelemetry t = freshReport(e, now);
            if (t != null && predictor != null) queue(e.getId(), t);
        }

        // 5 (Fig. 3, step 8). Outcomes whose horizon has elapsed refine the predictor.
        learnFromMaturedOutcomes(now);
        if (predictor != null) predictor.endEpoch();

        // 3. Proactive migration.
        if (policy.proactive() && (proactiveMigration || !policy.expectedLoss())) {
            updateDraining(now);
            proactiveStep(now);
        }

        // 4. Redundancy tiers and their overhead.
        replicationStep(now);

        controllerNanos += System.nanoTime() - t0;
        recordTimeline(now);
        if (now + FtCostModel.EPOCH < simTime) {
            send(getId(), FtCostModel.EPOCH, FtEvents.FT_EPOCH);
        }
    }

    /** The node's report for this epoch, or null if it sent none (dead or silent). */
    private static NodeTelemetry freshReport(InstrumentedFogDevice n, double now) {
        NodeTelemetry t = n.latestTelemetry();
        if (t == null || t.epoch < now - FtCostModel.EPOCH / 2.0) return null;
        return t;
    }

    private void queue(int nodeId, NodeTelemetry t) {
        ArrayDeque<NodeTelemetry> q = pending.get(nodeId);
        if (q == null) {
            q = new ArrayDeque<NodeTelemetry>();
            pending.put(nodeId, q);
        }
        q.addLast(t);
    }

    /**
     * A report's outcome is known once its horizon plus the detection delay has elapsed.
     * Reports from nodes the controller evacuated inside that horizon are censored: the
     * intervention changed what would have happened, so they say nothing about the
     * model.
     */
    private void learnFromMaturedOutcomes(double now) {
        if (predictor == null) return;
        double settle = FaultInjector.HORIZON_DELTA_T + FtCostModel.HEARTBEAT_TIMEOUT
                + FtCostModel.LINK_LOSS_TIMEOUT;
        for (Map.Entry<Integer, ArrayDeque<NodeTelemetry>> e : pending.entrySet()) {
            ArrayDeque<NodeTelemetry> q = e.getValue();
            int nodeId = e.getKey();
            while (!q.isEmpty() && q.peekFirst().epoch + settle <= now) {
                NodeTelemetry t = q.pollFirst();
                if (actedOnWithin(nodeId, t.epoch, t.epoch + FaultInjector.HORIZON_DELTA_T)) continue;
                Double ft = detectedFailureTime(nodeId);
                boolean failed = ft != null && ft > t.epoch && ft <= t.epoch + FaultInjector.HORIZON_DELTA_T;
                predictor.observeOutcome(t, failed);
            }
        }
    }

    private Double detectedFailureTime(int nodeId) {
        if (!detectedAt.containsKey(nodeId)) {
            InstrumentedFogDevice d = topology.device(nodeId);
            // Edge failures are known at once: the user's session ends.
            if (d != null && d.getLevel() == 2 && actualFailureTime.containsKey(nodeId)) {
                return actualFailureTime.get(nodeId);
            }
            return null;
        }
        return actualFailureTime.get(nodeId);
    }

    private boolean actedOnWithin(int nodeId, double from, double to) {
        List<Double> d = decisionsByNode.get(nodeId);
        if (d == null) return false;
        for (double t : d) if (t > from - 1e-9 && t <= to) return true;
        return false;
    }

    // =====================================================================================
    // proactive migration (N2)
    // =====================================================================================

    /** Node-level trigger of the threshold policy (FTAPA), with hysteresis. */
    private void updateDraining(double now) {
        if (policy != FtPolicy.THRESHOLD) return;
        for (InstrumentedFogDevice n : topology.fogNodes()) {
            int id = n.getId();
            if (registry.isKnownFailed(id)) {
                draining.remove(id);
                continue;
            }
            double hi = healthIndex.healthIndex(freshReport(n, now));
            boolean trigger = !Double.isNaN(hi) && hi < healthIndex.threshold();
            if (trigger) {
                draining.add(id);
                calmEpochs.put(id, 0);
            } else if (draining.contains(id)) {
                int calm = calmEpochs.containsKey(id) ? calmEpochs.get(id) + 1 : 1;
                calmEpochs.put(id, calm);
                if (calm >= 2) draining.remove(id);
            }
        }
    }

    private void proactiveStep(double now) {
        for (ServiceSession s : registry.sessions()) {
            if (s.terminated || s.migrating) continue;
            InstrumentedFogDevice n = topology.device(s.servingNodeId);
            if (n == null || n == topology.cloud() || registry.isKnownFailed(n.getId())) continue;
            if (now - s.lastMigrationTime < MIGRATION_COOLDOWN) continue;

            InstrumentedFogDevice target = null;
            double p = riskOf(n);
            double pStar = Double.NaN;
            boolean go;
            switch (policy) {
                case THRESHOLD:
                    go = draining.contains(n.getId());
                    if (go) target = backupFor(s, n);
                    break;
                default: { // ML, BBN and the oracle predictor: Eq. (3)
                    target = bestAlternative(s, n);
                    if (target == null) {
                        go = false;
                        break;
                    }
                    double addedLatency = topology.accessLatency(s.edgeId, target)
                            - topology.accessLatency(s.edgeId, n);
                    double cFail = FtCostModel.failureCost(s, s.tier);
                    double cMig = FtCostModel.migrationCost(s, addedLatency);
                    pStar = cMig / cFail;
                    go = p * cFail > cMig && riskOf(target) < p;
                }
            }
            if (go && target != null && target != n) {
                startMigration(s, n, target, p, pStar, now);
            }
        }
    }

    private void startMigration(ServiceSession s, InstrumentedFogDevice from,
                                InstrumentedFogDevice to, double p, double pStar, double now) {
        s.migrating = true;
        s.migrationSource = from.getId();
        s.migrationTarget = to.getId();
        s.migrationStart = now;
        double transfer = FtCostModel.transferTime(s);
        send(getId(), transfer, FtEvents.FT_SWITCHOVER, s.edgeId);
        migrationsStarted++;
        migrationsThisEpoch++;
        List<Double> d = decisionsByNode.get(from.getId());
        if (d == null) {
            d = new ArrayList<Double>();
            decisionsByNode.put(from.getId(), d);
        }
        d.add(now);
        migrationDecisions.add(new double[]{from.getId(), now});
        decisionHealth.add(RunContext.injector().health(from.getName()));
        EventLog.emit("MIGRATE", now, "user", s.edgeName, "class", s.criticality.label,
                "from", from.getName(), "to", to.getName(), "p", p, "pstar", pStar,
                "cause", cause.containsKey(from.getId()) ? cause.get(from.getId()) : "-",
                "transfer", transfer);
    }

    private void onSwitchover(int edgeId) {
        ServiceSession s = registry.session(edgeId);
        if (s == null || !s.migrating) return;
        double now = CloudSim.clock();
        InstrumentedFogDevice from = topology.device(s.migrationSource);
        InstrumentedFogDevice to = topology.device(s.migrationTarget);
        s.migrating = false;
        if (from == null || from.isFailed() || to == null || to.isFailed()
                || registry.isKnownFailed(to.getId()) || s.terminated) {
            migrationsAborted++;
            return;
        }
        InstrumentedFogDevice edge = topology.device(edgeId);
        topology.reassociate(edge, to);
        s.servingNodeId = to.getId();
        s.addOutage(now, now + FtCostModel.stopAndCopy(s), ServiceSession.Cause.MIGRATION);
        s.lastMigrationTime = now;
        s.proactiveMigrations++;
        migrationsCompleted++;
        List<Double> ev = evacuations.get(from.getId());
        if (ev == null) {
            ev = new ArrayList<Double>();
            evacuations.put(from.getId(), ev);
        }
        ev.add(now);
    }

    // =====================================================================================
    // placement (N2, Eq. 2)
    // =====================================================================================

    private double riskOf(InstrumentedFogDevice n) {
        Double r = risk.get(n.getId());
        return r == null ? 0.0 : r;
    }

    /** Risk as the placement utility sees it (zero when that term is ablated). */
    private double placementRisk(InstrumentedFogDevice n) {
        return riskPlacement ? riskOf(n) : 0.0;
    }

    /** P(fail) * C_fail > C_mig if the session sat on this node now. */
    private boolean wouldMigrate(ServiceSession s, InstrumentedFogDevice n) {
        if (s == null || !policy.expectedLoss() || !riskPlacement) return false;
        return riskOf(n) * FtCostModel.failureCost(s, s.tier) > FtCostModel.migrationCost(s, 0.0);
    }

    /** Projected utilisation of a node if one more user were added. */
    private double projectedUtil(InstrumentedFogDevice n) {
        double share = userMiRate / Math.max(1.0, n.getHost().getTotalMips());
        double byUsers = (n.getChildrenIds().size() + 1) * share;
        NodeTelemetry t = n.latestTelemetry();
        double reported = (t == null || Double.isNaN(t.cpuUtil)) ? 0.0 : t.cpuUtil + share;
        return Math.max(byUsers, reported);
    }

    /** Predicted loop latency at a candidate: access delay plus an M/M/1 processing delay. */
    private double predictedLatency(ServiceSession s, Topology.Candidate c) {
        double service = spec.analysisTupleMi * MelbCBDFaultToleranceSim.ANALYSIS_COST_SCALE
                / Math.max(1.0, c.node.getHost().getTotalMips());
        double rho = Math.min(0.95, projectedUtil(c.node));
        return c.accessLatency + service / (1.0 - rho);
    }

    /**
     * Placement utility of Eq. (2). Latency is normalised by the critical loop deadline
     * (the share of the deadline the predicted loop would consume) and energy by access
     * distance (radio energy grows with it), so a utility difference means the same
     * thing for every user. Min-max normalisation across the candidates was tried first
     * and rejected: it stretched a 2 ms difference into a full 0-to-1 swing and made
     * users hop between near-identical nodes, paying a migration pause each time.
     */
    private Topology.Candidate argmaxUtility(ServiceSession s, List<Topology.Candidate> cands,
                                             Map<Integer, Double> utilities) {
        if (cands.isEmpty()) return null;
        Criticality k = s == null ? Criticality.STANDARD : s.criticality;
        Topology.Candidate best = null;
        double bestU = -Double.MAX_VALUE;
        for (Topology.Candidate c : cands) {
            double ln = Math.min(1.0, predictedLatency(s, c) / MelbCBDFaultToleranceSim.CRITICAL_LOOP_DEADLINE);
            double en = Math.min(1.0, c.km / ENERGY_REFERENCE_KM);
            double u = k.alpha * (1.0 - ln) + k.beta * (1.0 - en) + k.gamma * (1.0 - placementRisk(c.node));
            if (utilities != null) utilities.put(c.node.getId(), u);
            if (u > bestU) {
                bestU = u;
                best = c;
            }
        }
        return best;
    }

    /** Nodes a policy will not place a user on right now. */
    private Set<Integer> exclusionsFor(ServiceSession s) {
        Set<Integer> ex = new HashSet<Integer>(draining);
        if (policy.expectedLoss() && s != null) {
            for (InstrumentedFogDevice n : topology.fogNodes()) {
                if (wouldMigrate(s, n)) ex.add(n.getId());
            }
        }
        return ex;
    }

    /**
     * The serving node a user should have. Baselines: nearest node not known dead (and,
     * for the threshold policy, not draining). Expected-loss policies: the
     * candidate of highest placement utility, keeping the current node unless it has
     * become risky, left the candidate set, or a clearly better node exists.
     */
    public InstrumentedFogDevice chooseServingNode(InstrumentedFogDevice edge, boolean initial) {
        ServiceSession s = registry.session(edge.getId());
        if (s != null && (s.migrating || s.terminated)) return null;
        Set<Integer> ex = exclusionsFor(s);
        List<Topology.Candidate> cands = topology.nearestCandidates(edge.getId(), K_CANDIDATES, ex);
        if (cands.isEmpty()) cands = topology.nearestCandidates(edge.getId(), K_CANDIDATES, null);
        if (cands.isEmpty()) return topology.cloud();
        if (!policy.expectedLoss()) return cands.get(0).node;

        Map<Integer, Double> utilities = new HashMap<Integer, Double>();
        Topology.Candidate best = argmaxUtility(s, cands, utilities);
        InstrumentedFogDevice current = topology.device(edge.getParentId());
        if (!initial && current != null && utilities.containsKey(current.getId())) {
            if (utilities.get(best.node.getId()) - utilities.get(current.getId()) < UTILITY_HYSTERESIS) {
                return current;
            }
        }
        return best.node;
    }

    /** Best migration / standby target other than the current node. */
    private InstrumentedFogDevice bestAlternative(ServiceSession s, InstrumentedFogDevice current) {
        Set<Integer> ex = exclusionsFor(s);
        ex.add(current.getId());
        List<Topology.Candidate> cands = topology.nearestCandidates(s.edgeId, K_CANDIDATES, ex);
        if (cands.isEmpty()) return null;
        if (!policy.expectedLoss()) return cands.get(0).node;
        Topology.Candidate best = argmaxUtility(s, cands, null);
        return best == null ? null : best.node;
    }

    /** FTAPA's pre-selected backup: least active connections among nearby nodes (DCLB). */
    private InstrumentedFogDevice backupFor(ServiceSession s, InstrumentedFogDevice current) {
        InstrumentedFogDevice standby = topology.device(s.standbyNodeId);
        if (standby != null && standby != current && !standby.isFailed()
                && !registry.isKnownFailed(standby.getId()) && !draining.contains(standby.getId())) {
            return standby;
        }
        Set<Integer> ex = new HashSet<Integer>(draining);
        ex.add(current.getId());
        List<Topology.Candidate> cands = topology.nearestCandidates(s.edgeId, K_CANDIDATES, ex);
        InstrumentedFogDevice best = null;
        int fewest = Integer.MAX_VALUE;
        for (Topology.Candidate c : cands) {
            int load = c.node.getChildrenIds().size();
            if (load < fewest) {
                fewest = load;
                best = c.node;
            }
        }
        return best;
    }

    // =====================================================================================
    // redundancy (N3, Eq. 4)
    // =====================================================================================

    private void replicationStep(double now) {
        List<ServiceSession> active = new ArrayList<ServiceSession>();
        for (ServiceSession s : registry.sessions()) {
            if (!s.terminated) active.add(s);
        }
        activeSessionEpochs += active.size();

        if (policy.expectedLoss() && selectiveReplication) {
            // Choose the tier minimising overhead + expected loss for each session.
            final Map<ServiceSession, Double> r = new HashMap<ServiceSession, Double>();
            List<ServiceSession> wantWarm = new ArrayList<ServiceSession>();
            for (ServiceSession s : active) {
                InstrumentedFogDevice n = topology.device(s.servingNodeId);
                double p = n == null ? 0.0 : riskOf(n);
                double kappa = s.criticality.kappa;
                double none = p * kappa * FtCostModel.outageIfFails(s, ServiceSession.Tier.NONE,
                        FtCostModel.HEARTBEAT_TIMEOUT);
                double ckpt = FtCostModel.checkpointCostPerEpoch(s) + p * kappa
                        * FtCostModel.outageIfFails(s, ServiceSession.Tier.CHECKPOINT, FtCostModel.HEARTBEAT_TIMEOUT);
                double warm = FtCostModel.warmCostPerEpoch(s) + p * kappa
                        * FtCostModel.outageIfFails(s, ServiceSession.Tier.WARM_STANDBY, FtCostModel.HEARTBEAT_TIMEOUT);
                ServiceSession.Tier tier = ServiceSession.Tier.NONE;
                double best = none;
                if (ckpt < best) { best = ckpt; tier = ServiceSession.Tier.CHECKPOINT; }
                if (warm < best) { tier = ServiceSession.Tier.WARM_STANDBY; }
                s.tier = tier;
                r.put(s, p * kappa);
                if (tier == ServiceSession.Tier.WARM_STANDBY) wantWarm.add(s);
            }
            int budget = (int) Math.ceil(FtCostModel.WARM_BUDGET_FRACTION * active.size());
            if (wantWarm.size() > budget) {
                Collections.sort(wantWarm, (a, b) -> Double.compare(r.get(b), r.get(a)));
                for (int i = budget; i < wantWarm.size(); i++) {
                    wantWarm.get(i).tier = ServiceSession.Tier.CHECKPOINT;
                }
            }
        } else {
            ServiceSession.Tier fixed = policy.expectedLoss() ? ServiceSession.Tier.NONE
                    : policy == FtPolicy.REACTIVE ? ServiceSession.Tier.CHECKPOINT
                    : policy == FtPolicy.THRESHOLD ? ServiceSession.Tier.WARM_STANDBY
                    : ServiceSession.Tier.NONE;
            for (ServiceSession s : active) s.tier = fixed;
        }

        // Standby placement and overhead accounting.
        for (ServiceSession s : active) {
            InstrumentedFogDevice n = topology.device(s.servingNodeId);
            if (s.tier == ServiceSession.Tier.NONE || n == null) {
                s.standbyNodeId = -1;
                continue;
            }
            InstrumentedFogDevice standby = topology.device(s.standbyNodeId);
            if (standby == null || standby == n || standby.isFailed()
                    || registry.isKnownFailed(standby.getId()) || policy.expectedLoss()) {
                InstrumentedFogDevice alt = policy == FtPolicy.THRESHOLD ? backupFor(s, n) : bestAlternative(s, n);
                s.standbyNodeId = alt == null ? -1 : alt.getId();
            }
            if (s.tier == ServiceSession.Tier.WARM_STANDBY) {
                standbyEpochs += 1.0;
                replicationKb += FtCostModel.checkpointKbPerEpoch(s, FtCostModel.CHECKPOINT_INTERVAL);
            } else {
                NodeTelemetry t = n.latestTelemetry();
                boolean loaded = t != null && !Double.isNaN(t.cpuUtil) && t.cpuUtil > 0.2;
                double interval = (policy == FtPolicy.REACTIVE && !loaded)
                        ? FtCostModel.CHECKPOINT_INTERVAL_IDLE : FtCostModel.CHECKPOINT_INTERVAL;
                replicationKb += FtCostModel.checkpointKbPerEpoch(s, interval);
            }
        }
    }

    // =====================================================================================
    // failures, detection, recovery
    // =====================================================================================

    /** Called synchronously at the instant a device dies. */
    public void onDeviceFailed(InstrumentedFogDevice device, double t) {
        actualFailureTime.put(device.getId(), t);
        if (device.getLevel() == 2) {
            ServiceSession s = registry.session(device.getId());
            if (s != null && !s.terminated) {
                s.terminated = true;
                s.terminatedAt = t;
                edgeFailures++;
            }
            EventLog.emit("EDGE_FAIL", t, "user", device.getName(),
                    "cause", RunContext.injector().failureCauseOf(device.getName()));
            return;
        }
        fogFailures++;
        failuresThisEpoch++;
        int hit = 0;
        for (ServiceSession s : registry.sessions()) {
            if (!s.terminated && s.servingNodeId == device.getId()) {
                s.failuresHit++;
                hit++;
            }
        }
        sessionsHit += hit;
        if (hit > 0) fogFailuresWithUsers++;
        String failCause = RunContext.injector().failureCauseOf(device.getName());
        Integer hc = hitByCause.get(failCause);
        hitByCause.put(failCause, (hc == null ? 0 : hc) + hit);

        int evacuated = 0;
        List<Double> ev = evacuations.get(device.getId());
        if (ev != null) {
            for (double when : ev) {
                if (when >= t - FaultInjector.HORIZON_DELTA_T - FtCostModel.EPOCH && when <= t) evacuated++;
            }
        }
        sessionsEvacuated += evacuated;
        double lead = Double.NaN;
        List<Double> d = decisionsByNode.get(device.getId());
        if (d != null) {
            for (double when : d) {
                if (when >= t - FaultInjector.HORIZON_DELTA_T - FtCostModel.EPOCH && when <= t) {
                    lead = Double.isNaN(lead) ? t - when : Math.max(lead, t - when);
                }
            }
        }
        if (!Double.isNaN(lead)) leadTimes.add(lead);

        double detect = policy.heartbeat() ? FtCostModel.HEARTBEAT_TIMEOUT : FtCostModel.LINK_LOSS_TIMEOUT;
        send(getId(), detect, FtEvents.FT_DETECT, device.getId());
        EventLog.emit("FAIL", t, "node", device.getName(), "users_hit", hit, "evacuated", evacuated,
                "lead", lead, "cause", RunContext.injector().failureCauseOf(device.getName()),
                "p_before", riskOf(device));
    }

    private void onDetect(int nodeId) {
        double now = CloudSim.clock();
        registry.markKnownFailed(nodeId);
        detectedAt.put(nodeId, now);
        draining.remove(nodeId);
        Double tf = actualFailureTime.get(nodeId);
        double failedAt = tf == null ? now : tf;
        for (ServiceSession s : registry.sessions()) {
            if (!s.terminated && s.servingNodeId == nodeId) {
                recover(s, failedAt, now);
            }
        }
    }

    private double recoveryTime(ServiceSession s) {
        if (policy.cloudHosted()) return 0.0;
        switch (s.tier) {
            case WARM_STANDBY: return FtCostModel.WARM_SWITCHOVER;
            case CHECKPOINT: return FtCostModel.checkpointRestore(s);
            default: return FtCostModel.COLD_START;
        }
    }

    private void recover(ServiceSession s, double failedAt, double now) {
        s.migrating = false;
        InstrumentedFogDevice edge = topology.device(s.edgeId);
        InstrumentedFogDevice target = null;
        InstrumentedFogDevice standby = topology.device(s.standbyNodeId);
        if (s.tier != ServiceSession.Tier.NONE && standby != null && !standby.isFailed()
                && !registry.isKnownFailed(standby.getId())) {
            target = standby;
        }
        if (target == null) target = chooseServingNode(edge, false);
        if (target == null || target.getId() == s.servingNodeId) target = topology.cloud();
        double end = now + recoveryTime(s);
        String mode = policy.cloudHosted() ? "reconnect" : s.tier.name().toLowerCase(Locale.ROOT);
        topology.reassociate(edge, target);
        s.servingNodeId = target.getId();
        s.addOutage(failedAt, end, ServiceSession.Cause.FAILURE);
        Integer c = recoveries.get(mode);
        recoveries.put(mode, c == null ? 1 : c + 1);
        EventLog.emit("RECOVER", now, "user", s.edgeName, "class", s.criticality.label,
                "mode", mode, "to", target.getName(), "downtime", end - failedAt);
    }

    // =====================================================================================
    // association callbacks from the mobility driver
    // =====================================================================================

    public void onInitialAssociation(InstrumentedFogDevice edge, InstrumentedFogDevice target) {
        ServiceSession s = registry.session(edge.getId());
        if (s != null) s.servingNodeId = target.getId();
    }

    /** A user moved and its serving node changes; returns true if the handover happened. */
    public boolean onMobilityHandover(InstrumentedFogDevice edge, InstrumentedFogDevice target) {
        ServiceSession s = registry.session(edge.getId());
        InstrumentedFogDevice from = topology.device(edge.getParentId());
        if (s != null && (s.migrating || s.terminated)) return false;
        double now = CloudSim.clock();
        topology.reassociate(edge, target);
        handovers++;
        if (s == null) return true;
        s.servingNodeId = target.getId();
        if (policy.cloudHosted()) return true;
        if (from != null && from.isFailed()) {
            // The user walked away from a node that had died unnoticed: its state is lost.
            Double tf = actualFailureTime.get(from.getId());
            s.addOutage(tf == null ? now : tf, now + recoveryTime(s), ServiceSession.Cause.FAILURE);
        } else {
            s.addOutage(now, now + FtCostModel.stopAndCopy(s), ServiceSession.Cause.HANDOVER);
        }
        return true;
    }

    // =====================================================================================
    // reporting
    // =====================================================================================

    private void recordTimeline(double now) {
        int alive = 0;
        int atRisk = 0;
        double meanRisk = 0.0;
        for (InstrumentedFogDevice n : topology.fogNodes()) {
            if (registry.isKnownFailed(n.getId())) continue;
            alive++;
            double r = riskOf(n);
            meanRisk += r;
            if (r >= 0.10) atRisk++;
        }
        meanRisk = alive == 0 ? 0.0 : meanRisk / alive;
        double downtime = 0.0;
        int warm = 0;
        int ckpt = 0;
        for (ServiceSession s : registry.sessions()) {
            downtime += s.downtime;
            if (!s.terminated && s.tier == ServiceSession.Tier.WARM_STANDBY) warm++;
            if (!s.terminated && s.tier == ServiceSession.Tier.CHECKPOINT) ckpt++;
        }
        JSONObject o = new JSONObject();
        o.put("t", now);
        o.put("alive_fog", alive);
        o.put("at_risk", atRisk);
        o.put("mean_risk", meanRisk);
        o.put("downtime", downtime);
        o.put("migrations", migrationsStarted);
        o.put("fog_failures", fogFailures);
        o.put("warm", warm);
        o.put("checkpoint", ckpt);
        timeline.add(o);
        EventLog.emit("EPOCH", now, "alive_fog", alive, "at_risk", atRisk, "mean_risk", meanRisk,
                "migrations", migrationsThisEpoch, "failures", failuresThisEpoch,
                "downtime", downtime, "warm", warm, "checkpoint", ckpt,
                "learning", predictor == null ? "" : predictor.status());
        failuresThisEpoch = 0;
        migrationsThisEpoch = 0;
    }

    private static double percentile(List<Double> v, double q) {
        if (v.isEmpty()) return Double.NaN;
        List<Double> s = new ArrayList<Double>(v);
        Collections.sort(s);
        double pos = q * (s.size() - 1);
        int lo = (int) Math.floor(pos);
        int hi = (int) Math.ceil(pos);
        return s.get(lo) + (s.get(hi) - s.get(lo)) * (pos - lo);
    }

    private static Object num(double v) {
        return Double.isNaN(v) || Double.isInfinite(v) ? null : v;
    }

    /** Summary of the run, written as JSON. */
    @SuppressWarnings("unchecked")
    public JSONObject summary(double end, String predictorName) {
        double downtime = 0, downFailure = 0, downMigration = 0, downHandover = 0;
        double activeTime = 0;
        long emitted = 0, delivered = 0, droppedFailed = 0, droppedUnavailable = 0;
        long critDelivered = 0, critMisses = 0;
        double delaySum = 0;
        List<Double> delays = new ArrayList<Double>();
        Map<String, double[]> byClass = new LinkedHashMap<String, double[]>();
        for (Criticality c : Criticality.values()) byClass.put(c.label, new double[3]);
        for (ServiceSession s : registry.sessions()) {
            s.closeAt(end);
            double alive = s.terminated ? Math.min(end, s.terminatedAt) : end;
            activeTime += alive;
            downtime += s.downtime;
            downFailure += s.downtimeFailure;
            downMigration += s.downtimeMigration;
            downHandover += s.downtimeHandover;
            emitted += s.analysesEmitted;
            delivered += s.alarmsDelivered;
            droppedFailed += s.analysesDroppedFailedNode;
            droppedUnavailable += s.analysesDroppedUnavailable;
            delaySum += s.loopDelaySum;
            for (int i = 0; i < s.loopDelays.size(); i++) delays.add(s.loopDelays.get(i));
            if (s.criticality == Criticality.CRITICAL) {
                critDelivered += s.alarmsDelivered;
                critMisses += s.deadlineMisses;
            }
            double[] bc = byClass.get(s.criticality.label);
            bc[0] += s.downtime;
            bc[1] += alive;
            bc[2] += 1;
        }
        int unnecessary = 0;
        for (double[] d : migrationDecisions) {
            Double ft = actualFailureTime.get((int) d[0]);
            boolean justified = ft != null && ft > d[1] && ft <= d[1] + FaultInjector.HORIZON_DELTA_T + FtCostModel.EPOCH;
            if (!justified) unnecessary++;
        }
        double energyCloud = 0, energyFog = 0, energyEdge = 0;
        energyCloud += topology.cloud().getEnergyConsumption();
        for (InstrumentedFogDevice n : topology.fogNodes()) energyFog += n.getEnergyConsumption();
        for (InstrumentedFogDevice n : topology.edges()) energyEdge += n.getEnergyConsumption();

        JSONObject o = new JSONObject();
        o.put("policy", policy.key);
        o.put("policy_label", policy.label);
        o.put("predictor", predictorName);
        o.put("scenario", RunContext.scenarioId());
        o.put("seed", RunContext.seed());
        o.put("dropout", RunContext.dropout().name().toLowerCase(Locale.ROOT));
        o.put("dropout_rate", RunContext.dropoutRate());
        o.put("sim_time", end);
        o.put("users", registry.sessions().size());
        o.put("fog_nodes", topology.fogNodes().size());
        o.put("fog_failures", fogFailures);
        o.put("fog_failures_with_users", fogFailuresWithUsers);
        o.put("sessions_hit", sessionsHit);
        o.put("sessions_evacuated", sessionsEvacuated);
        o.put("edge_failures", edgeFailures);
        o.put("failure_causes", new JSONObject(RunContext.injector().failureCauseCounts()));
        o.put("downtime_total", downtime);
        o.put("downtime_failure", downFailure);
        o.put("downtime_migration", downMigration);
        o.put("downtime_handover", downHandover);
        o.put("availability", activeTime > 0 ? 1.0 - downtime / activeTime : 1.0);
        JSONObject cls = new JSONObject();
        for (Map.Entry<String, double[]> e : byClass.entrySet()) {
            JSONObject c = new JSONObject();
            c.put("downtime", e.getValue()[0]);
            c.put("availability", e.getValue()[1] > 0 ? 1.0 - e.getValue()[0] / e.getValue()[1] : 1.0);
            c.put("users", (int) e.getValue()[2]);
            cls.put(e.getKey(), c);
        }
        o.put("by_class", cls);
        o.put("analyses_emitted", emitted);
        o.put("alarms_delivered", delivered);
        o.put("dropped_failed_node", droppedFailed);
        o.put("dropped_unavailable", droppedUnavailable);
        o.put("delivery_ratio", emitted > 0 ? (double) delivered / emitted : null);
        o.put("loop_delay_mean", delivered > 0 ? delaySum / delivered : null);
        o.put("loop_delay_p95", num(percentile(delays, 0.95)));
        o.put("critical_deadline_miss_ratio", critDelivered > 0 ? (double) critMisses / critDelivered : null);
        o.put("migrations_started", migrationsStarted);
        o.put("migrations_completed", migrationsCompleted);
        o.put("migrations_aborted", migrationsAborted);
        o.put("migrations_unnecessary", unnecessary);
        int degraded = 0;
        for (double h : decisionHealth) if (h < DEGRADED_HEALTH) degraded++;
        o.put("migrations_from_degraded", degraded);
        o.put("ablation", (riskPlacement ? "" : "-risk-placement") + (proactiveMigration ? "" : "-migration")
                + (selectiveReplication ? "" : "-replication"));
        o.put("handovers", handovers);
        o.put("standby_ratio", activeSessionEpochs > 0 ? standbyEpochs / activeSessionEpochs : 0.0);
        o.put("replication_mb", replicationKb / 1024.0);
        o.put("recoveries", new JSONObject(recoveries));
        o.put("sessions_hit_by_cause", new JSONObject(hitByCause));
        double meanLead = 0;
        for (double l : leadTimes) meanLead += l;
        o.put("lead_time_mean", leadTimes.isEmpty() ? null : meanLead / leadTimes.size());
        o.put("energy_cloud", energyCloud);
        o.put("energy_fog", energyFog);
        o.put("energy_edge", energyEdge);
        o.put("network_usage", NetworkUsageMonitor.getNetworkUsage() / Math.max(1.0, end));
        o.put("controller_ms_per_epoch", epochs > 0 ? controllerNanos / 1e6 / epochs : 0.0);
        o.put("predictions", predictions);
        o.put("predictor_status", predictor == null ? "" : predictor.status());
        o.put("timeline", timeline);
        return o;
    }

    public void writeSummary(String path, double end, String predictorName) throws Exception {
        File f = new File(path);
        if (f.getParentFile() != null) f.getParentFile().mkdirs();
        FileWriter w = new FileWriter(f);
        try {
            w.write(summary(end, predictorName).toJSONString());
        } finally {
            w.close();
        }
    }
}
