package org.fog.ft;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.special.Erf;

/**
 * Ground-truth failure process for the simulated fog topology: a stress-driven
 * degradation (wear-out) model with fail-slow symptoms.
 *
 * <p>iFogSim has no native failure model, so one is injected here. This is the
 * fault-injection harness Section IV of the Review 2 paper promises. Its design follows
 * the premise the project rests on (Review slides: "nodes degrade first: utilisation
 * rises, queues grow, links and energy decay") and the field evidence on fail-slow
 * hardware: components usually lose performance for a while before they stop.
 *
 * <h2>Hidden health</h2>
 * Every node carries a health value {@code h} in [0, 1] that is never observed by any
 * predictor. It starts below 1 for older hardware ({@code h0 = 1 - 0.45 u^3}, so most
 * nodes are near-new and roughly one in ten is noticeably worn) and loses, at every
 * epoch,
 * <pre>
 *   wear = W(class) * exp( s(class) * S ) * LogNormal(0, 0.35)    (+ rare shocks)
 *   S    = 1.6 cpu + 0.8 queue + 1.0 thermal + 1.4 depletion
 *        + 0.4 latency + 0.3 jitter + 0.2 handovers               (all normalised to [0,1])
 * </pre>
 * computed from the node's <i>true</i> state, before any telemetry dropout. Cumulative
 * damage under sustained load, heat and battery drain is the standard wear-out picture
 * (Miner's rule, Arrhenius-type acceleration).
 *
 * <h2>Failure</h2>
 * A node fails during the next epoch when (a) its health is exhausted (wear-out, or a
 * shock that exhausts it), (b) an edge device's battery is empty, or (c) a sudden
 * failure occurs. Sudden failures are deliberately memoryless and symptom-free: they
 * are the share of failures no predictor can anticipate, and they keep the evaluation
 * honest about the limits of prediction.
 *
 * <h2>Symptoms</h2>
 * As {@code h} falls the node becomes slower ({@link #slowdown}), its network interface
 * retransmits more ({@link #latencyFactor}), it draws extra power ({@link
 * #extraPowerFraction}) and runs hotter ({@link #heatRise}). These factors are applied
 * <i>inside</i> iFogSim by {@link InstrumentedFogDevice}, so the observable precursors
 * -- higher CPU utilisation, longer queues and loop delays, higher latency, faster
 * energy drain, higher temperature -- emerge from the simulator's own mechanics rather
 * than being written into the telemetry. They also feed back into the stress, so
 * degradation accelerates near the end of life.
 *
 * <h2>Not circular</h2>
 * The predictor is a discrete Bayesian network over discretised telemetry. The ground
 * truth is a continuous hidden-state process with noise, shocks, feedback and
 * memoryless sudden failures. The network never sees health, the stress formula or the
 * random draws; it sees sampled, discretised and sometimes missing telemetry.
 *
 * <p>All randomness comes from {@link FtRandom}, keyed by seed, scenario, node name and
 * epoch, so draws are independent across scenarios and seeds and identical across
 * fault-tolerance policies (common random numbers).
 */
public class FaultInjector {

    /**
     * One epoch of simulation time is treated as one hour of node operation: an
     * accelerated-life compression so that wear-out happens within a run of feasible
     * length. Uniform compression preserves the ordering of device-class reliability.
     */
    public static final double HOURS_PER_EPOCH = 1.0;

    /** Look-ahead horizon in simulation time units; must exceed migration time. */
    public static final double HORIZON_DELTA_T = 200.0;

    // ---- stress weights on the normalised stressors ------------------------------
    static final double W_CPU = 1.6;
    static final double W_QUEUE = 0.8;
    static final double W_THERMAL = 1.0;
    static final double W_ENERGY = 1.4;
    static final double W_LATENCY = 0.4;
    static final double W_JITTER = 0.3;
    static final double W_HANDOVER = 0.2;

    /** Normalisation points: the value treated as fully saturated. */
    static final double QUEUE_SATURATION = 120.0;
    static final double LATENCY_SATURATION = 150.0;
    static final double HANDOVER_SATURATION = 8.0;
    /** Temperature below which heat causes no extra wear, and the span to "critical". */
    static final double THERMAL_SAFE_C = 45.0;
    static final double THERMAL_SPAN_C = 40.0;

    // ---- per-class wear parameters ------------------------------------------------
    private static final Map<String, Double> BASE_WEAR = new HashMap<String, Double>();
    private static final Map<String, Double> SENSITIVITY = new HashMap<String, Double>();
    private static final Map<String, Double> SUDDEN_PER_EPOCH = new HashMap<String, Double>();
    private static final Map<String, Double> RELIABILITY_PRIOR = new HashMap<String, Double>();
    private static final Map<String, Double> MTBF_HOURS = new HashMap<String, Double>();
    static {
        // Datacentre hardware: redundant power and active cooling, so load barely ages it.
        BASE_WEAR.put("cloud", 0.0002);
        SENSITIVITY.put("cloud", 0.3);
        SUDDEN_PER_EPOCH.put("cloud", 0.00002);
        // Fog gateways: commodity hardware, passive cooling, exposed enclosures.
        BASE_WEAR.put("fog", 0.0010);
        SENSITIVITY.put("fog", 1.0);
        SUDDEN_PER_EPOCH.put("fog", 0.0004);
        // Edge devices: battery powered, the battery being their dominant failure mode.
        BASE_WEAR.put("edge", 0.0008);
        SENSITIVITY.put("edge", 1.0);
        SUDDEN_PER_EPOCH.put("edge", 0.0006);

        RELIABILITY_PRIOR.put("cloud", 0.99);
        RELIABILITY_PRIOR.put("fog", 0.85);
        RELIABILITY_PRIOR.put("edge", 0.70);
        MTBF_HOURS.put("cloud", 87600.0);
        MTBF_HOURS.put("fog", 8760.0);
        MTBF_HOURS.put("edge", 8760.0);
    }

    /**
     * {@code -Dft.noFailures=true} disables degradation and failure entirely: the
     * "baseline run with no failures, for normal loop delay and energy" of the Review 2
     * evaluation plan, and a check that the service pipeline itself loses nothing.
     */
    public static final boolean FAILURES_ENABLED = !Boolean.getBoolean("ft.noFailures");

    static final double WEAR_NOISE_SIGMA = 0.35;
    static final double SHOCK_PROBABILITY = 0.0005;
    static final double SHOCK_MIN = 0.10;
    static final double SHOCK_MAX = 0.30;
    static final double INITIAL_AGE_SPREAD = 0.45;
    static final double BATTERY_EMPTY = 0.01;

    // ---- symptom gains --------------------------------------------------------------
    static final double SLOWDOWN_GAIN = 1.5;   // work inflation at h = 0 is 2.5x
    static final double LATENCY_GAIN = 0.6;    // link delay at h = 0 is 1.6x
    static final double POWER_GAIN = 0.30;     // extra draw at h = 0 is 30% of busy power
    static final double HEAT_GAIN_C = 18.0;    // extra temperature at h = 0

    /** State of one node. */
    static final class Node {
        final String name;
        final String deviceClass;
        final long hash;
        final double initialHealth;
        double health;
        /** Stress S of the latest epoch (what the next epoch's wear will most likely see). */
        double lastStress = 0.0;
        double failureTime = Double.NaN;
        String cause = "";

        Node(String name, String deviceClass, long hash, double initialHealth) {
            this.name = name;
            this.deviceClass = deviceClass;
            this.hash = hash;
            this.initialHealth = initialHealth;
            this.health = initialHealth;
        }
    }

    private final long seed;
    private final int scenario;
    private final Map<String, Node> nodes = new HashMap<String, Node>();

    public FaultInjector(long seed, int scenario) {
        this.seed = seed;
        this.scenario = scenario;
    }

    public long seed() {
        return seed;
    }

    public int scenario() {
        return scenario;
    }

    /** Uniform draw for one coordinate of the experiment. */
    double uniform(long nodeHash, long epochIndex, long stream) {
        return FtRandom.uniform(FtRandom.key(seed, scenario, nodeHash, epochIndex, stream));
    }

    double normal(long nodeHash, long epochIndex, long stream) {
        return FtRandom.normal(FtRandom.key(seed, scenario, nodeHash, epochIndex, stream));
    }

    /** Registers a node on first sight; its initial wear is drawn here. */
    Node node(String name, String deviceClass) {
        Node n = nodes.get(name);
        if (n == null) {
            long h = FtRandom.hash(name);
            double u = uniform(h, 0, FtRandom.S_INITIAL_HEALTH);
            double h0 = ("cloud".equals(deviceClass) || !FAILURES_ENABLED) ? 1.0
                    : 1.0 - INITIAL_AGE_SPREAD * u * u * u;
            n = new Node(name, deviceClass, h, h0);
            nodes.put(name, n);
        }
        return n;
    }

    public void register(String name, String deviceClass) {
        node(name, deviceClass);
    }

    /** Current hidden health of a node, 1.0 if it has not been seen yet. */
    public double health(String name) {
        Node n = nodes.get(name);
        return n == null ? 1.0 : Math.max(0.0, Math.min(1.0, n.health));
    }

    private double damage(String name) {
        return 1.0 - health(name);
    }

    /** Multiplier on the CPU work a degraded node needs per tuple (>= 1). */
    public double slowdown(String name) {
        double d = damage(name);
        return 1.0 + SLOWDOWN_GAIN * d * d;
    }

    /** Multiplier on the node's link delay (>= 1). */
    public double latencyFactor(String name) {
        double d = damage(name);
        return 1.0 + LATENCY_GAIN * d * d;
    }

    /** Extra power drawn by a degraded node, as a fraction of its busy power. */
    public double extraPowerFraction(String name) {
        double d = damage(name);
        return POWER_GAIN * Math.pow(d, 1.5);
    }

    /** Extra steady-state temperature of a degraded node, in degrees Celsius. */
    public double heatRise(String name) {
        double d = damage(name);
        return HEAT_GAIN_C * Math.pow(d, 1.5);
    }

    public static double reliabilityPriorFor(String deviceClass) {
        Double p = RELIABILITY_PRIOR.get(deviceClass);
        return p == null ? 0.85 : p;
    }

    public static double mtbfHoursFor(String deviceClass) {
        Double m = MTBF_HOURS.get(deviceClass);
        return m == null ? 8760.0 : m;
    }

    private static double clamp01(double v) {
        if (Double.isNaN(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    private static double param(Map<String, Double> m, String cls, double fallback) {
        Double v = m.get(cls);
        return v == null ? fallback : v;
    }

    /** Normalised stress of one true-state snapshot. */
    static double stress(NodeTelemetry t) {
        double thermal = Double.isNaN(t.temperature) ? 0.0
                : clamp01((t.temperature - THERMAL_SAFE_C) / THERMAL_SPAN_C);
        double depletion = Double.isNaN(t.residualEnergy) ? 0.0 : clamp01(1.0 - t.residualEnergy);
        return W_CPU * clamp01(t.cpuUtil)
             + W_QUEUE * clamp01(t.queueDepth / QUEUE_SATURATION)
             + W_THERMAL * thermal
             + W_ENERGY * depletion
             + W_LATENCY * clamp01(t.uplinkLatency / LATENCY_SATURATION)
             + W_JITTER * clamp01(t.jitterPktLoss)
             + W_HANDOVER * clamp01(t.handoverRate / HANDOVER_SATURATION);
    }

    /**
     * Advances one node by one epoch from its true state, and decides whether it fails.
     *
     * @return the absolute failure time if a failure was scheduled at this step, else NaN
     */
    public double step(NodeTelemetry trueState, double epochLength) {
        Node n = node(trueState.nodeName, trueState.deviceClass);
        if (!Double.isNaN(n.failureTime) || !FAILURES_ENABLED) {
            return Double.NaN;
        }
        long epochIndex = Math.round(trueState.epoch / Math.max(1.0, epochLength));
        String cls = trueState.deviceClass;

        double s = stress(trueState);
        n.lastStress = s;
        double noise = Math.exp(WEAR_NOISE_SIGMA * normal(n.hash, epochIndex, FtRandom.S_WEAR_NOISE)
                - 0.5 * WEAR_NOISE_SIGMA * WEAR_NOISE_SIGMA);
        double wear = param(BASE_WEAR, cls, 0.0010) * HOURS_PER_EPOCH
                * Math.exp(param(SENSITIVITY, cls, 1.0) * s) * noise;

        boolean shock = uniform(n.hash, epochIndex, FtRandom.S_SHOCK) < SHOCK_PROBABILITY;
        if (shock) {
            wear += SHOCK_MIN + (SHOCK_MAX - SHOCK_MIN)
                    * uniform(n.hash, epochIndex, FtRandom.S_SHOCK_SIZE);
        }
        n.health -= wear;

        String cause = null;
        if (n.health <= 0.0) {
            cause = shock ? "shock" : "wear-out";
            n.health = 0.0;
        } else if ("edge".equals(cls) && !Double.isNaN(trueState.residualEnergy)
                && trueState.residualEnergy <= BATTERY_EMPTY) {
            cause = "battery";
        } else if (uniform(n.hash, epochIndex, FtRandom.S_SUDDEN) < param(SUDDEN_PER_EPOCH, cls, 0.0004)) {
            cause = "sudden";
        }
        if (cause == null) {
            return Double.NaN;
        }
        // The node dies at a uniformly random instant inside the coming epoch.
        n.failureTime = trueState.epoch
                + uniform(n.hash, epochIndex, FtRandom.S_FAIL_OFFSET) * epochLength;
        n.cause = cause;
        return n.failureTime;
    }

    /**
     * The probability that a node fails within the horizon of a report taken now, given
     * its hidden state: what a predictor with a perfect health sensor and the exact wear
     * law would output (the oracle policy). It knows the damage already done -- a node
     * whose health reached zero at this step, by wear or by a shock, is certain to fail --
     * but not the random draws still to come, so a sudden failure surprises it as it
     * surprises every other predictor.
     *
     * <p>A report at epoch t is labelled by a failure in (t, t + horizon]. Every later
     * step whose failure window still falls inside the horizon (one, for a two-epoch
     * horizon) kills the node if its wear at the current stress exhausts the remaining
     * health, if a shock does, or if a sudden failure is drawn. The sum of the steps'
     * lognormal wear is matched by a lognormal (Fenton-Wilkinson, exact for one step);
     * the only approximation is that the next epoch's stress equals this one's. Battery
     * exhaustion of edge devices is not included: the controller asks about fog nodes.
     */
    public double failureProbability(String name, double now, double horizon) {
        Node n = nodes.get(name);
        if (n == null || !FAILURES_ENABLED) return 0.0;
        if (n.health <= 0.0) return 1.0;
        int steps = Math.max(0, (int) Math.round(horizon / FtCostModel.EPOCH) - 1);
        if (steps == 0) return 0.0;
        String cls = n.deviceClass;
        double h = Math.max(0.0, n.health);
        double perStep = param(BASE_WEAR, cls, 0.0010) * HOURS_PER_EPOCH
                * Math.exp(param(SENSITIVITY, cls, 1.0) * n.lastStress);
        double pWear;
        if (h <= 0.0) {
            pWear = 1.0;
        } else {
            // Each step's noise factor has mean 1 and variance exp(sigma^2) - 1.
            double mean = steps * perStep;
            double var = steps * perStep * perStep
                    * (Math.exp(WEAR_NOISE_SIGMA * WEAR_NOISE_SIGMA) - 1.0);
            double s2 = Math.log(1.0 + var / (mean * mean));
            double mu = Math.log(mean) - 0.5 * s2;
            pWear = 1.0 - normalCdf((Math.log(h) - mu) / Math.sqrt(s2));
        }
        double fatalShock = clamp01((SHOCK_MAX - (h - perStep)) / (SHOCK_MAX - SHOCK_MIN));
        double pShock = 1.0 - Math.pow(1.0 - SHOCK_PROBABILITY * fatalShock, steps);
        double pSudden = 1.0 - Math.pow(1.0 - param(SUDDEN_PER_EPOCH, cls, 0.0004), steps);
        return 1.0 - (1.0 - pWear) * (1.0 - pShock) * (1.0 - pSudden);
    }

    private static double normalCdf(double z) {
        return 0.5 * (1.0 + Erf.erf(z / Math.sqrt(2.0)));
    }

    /** Absolute failure time of a node, or NaN if no failure is scheduled. */
    public double failureTimeOf(String name) {
        Node n = nodes.get(name);
        return n == null ? Double.NaN : n.failureTime;
    }

    public String failureCauseOf(String name) {
        Node n = nodes.get(name);
        return n == null ? "" : n.cause;
    }

    public boolean hasFailureScheduled(String name) {
        return !Double.isNaN(failureTimeOf(name));
    }

    /** True when a failure is scheduled in (now, now + horizon]: the clairvoyant oracle. */
    public boolean failsWithin(String name, double now, double horizon) {
        double ft = failureTimeOf(name);
        return !Double.isNaN(ft) && ft > now && ft <= now + horizon;
    }

    public int failedNodeCount() {
        int c = 0;
        for (Node n : nodes.values()) {
            if (!Double.isNaN(n.failureTime)) c++;
        }
        return c;
    }

    public Map<String, Integer> failureCauseCounts() {
        Map<String, Integer> out = new HashMap<String, Integer>();
        for (Node n : nodes.values()) {
            if (!Double.isNaN(n.failureTime)) {
                Integer c = out.get(n.cause);
                out.put(n.cause, c == null ? 1 : c + 1);
            }
        }
        return out;
    }

    /**
     * Ground-truth label: does the node fail within (t, t + delta_t]? This is exactly
     * the event the BBN's target variable F is defined to predict.
     */
    public int labelFor(String name, double epoch) {
        double ft = failureTimeOf(name);
        if (Double.isNaN(ft)) return 0;
        return (ft > epoch && ft <= epoch + HORIZON_DELTA_T) ? 1 : 0;
    }
}
