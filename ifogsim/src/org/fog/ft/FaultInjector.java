package org.fog.ft;

import java.util.HashMap;
import java.util.Map;

/**
 * Ground-truth failure process for the simulated fog topology.
 *
 * <p>iFogSim2 has no native failure model, so one is injected here. This is the
 * fault-injection harness Section IV of the Review 2 paper promises.
 *
 * <p><b>Why this model and not the BBN's own.</b> The hazard below is a continuous
 * log-linear proportional-hazards model. The predictor being evaluated is a discrete
 * multi-level Bayesian network with elicited conditional probability tables. The two
 * have deliberately different functional forms, so the BBN is not being graded
 * against its own generative assumptions -- it has to recover a continuous hazard
 * relationship through a three-state discretisation and a latent aggregation tier.
 *
 * <p>The instantaneous hazard for node <i>n</i> at epoch <i>t</i> is
 *
 * <pre>
 *   lambda(n,t) = lambda_0(class) * A * exp( a*cpu + b*ram + c*queue_norm
 *                                          + d*latency_norm + e*(1-energy)
 *                                          + f*jitter + g*handover_norm )
 * </pre>
 *
 * where lambda_0 comes from the device class MTBF and A is an explicit acceleration
 * factor. Accelerated-life testing is standard practice: without it a one-year-MTBF
 * gateway essentially never fails inside a simulation of feasible length, and no
 * positive cases would exist to verify against. The factor is declared here rather
 * than buried so a reviewer can see exactly how much compression was applied.
 */
public class FaultInjector {

    /** One epoch of simulated time is treated as one hour of node operation. */
    public static final double HOURS_PER_EPOCH = 1.0;

    /**
     * Accelerated-life compression. Raises every base hazard uniformly so that a
     * usable number of failures occurs within the simulation horizon. Uniform
     * scaling preserves the relative ordering of device-class reliability.
     */
    public static final double ACCELERATION_FACTOR = 8.0;

    /** Look-ahead horizon in simulation time units; must exceed migration time. */
    public static final double HORIZON_DELTA_T = 200.0;

    // Hazard coefficients on the normalised stressors.
    private static final double A_CPU       = 1.8;
    private static final double B_RAM       = 1.0;
    private static final double C_QUEUE     = 0.8;
    private static final double D_LATENCY   = 0.6;
    private static final double E_ENERGY    = 1.5;
    private static final double F_JITTER    = 0.7;
    private static final double G_HANDOVER  = 0.4;

    /** Mean time between failures in hours, by device class. */
    private static final Map<String, Double> MTBF_HOURS = new HashMap<String, Double>();
    /** Prior reliability in [0,1] exposed to the BBN, by device class. */
    private static final Map<String, Double> RELIABILITY_PRIOR = new HashMap<String, Double>();
    static {
        MTBF_HOURS.put("cloud",   87600.0);  // ~10 years, redundant datacentre hardware
        MTBF_HOURS.put("fog",     8760.0);  // ~1 year, resource-constrained fog node
        MTBF_HOURS.put("edge",    8760.0);  // ~1 year; battery exposure is represented
                                           // by the energy term rather than a depressed MTBF

        RELIABILITY_PRIOR.put("cloud", 0.99);
        RELIABILITY_PRIOR.put("fog",   0.85);
        RELIABILITY_PRIOR.put("edge",  0.70);
    }

    /**
     * Per-class sensitivity of the hazard to stress.
     *
     * <p>MTBF alone does not capture how hardware responds to being pushed. A
     * datacentre host has redundant power and active cooling, so high utilisation
     * translates weakly into failure risk; a passively-cooled handset running on
     * battery has no such margin and degrades far faster under the same relative
     * load. Without this term a single cloud node driven to full utilisation failed
     * within four epochs, which is not a defensible ground truth for hardware with a
     * ten-year MTBF.
     */
    private static final Map<String, Double> STRESS_SENSITIVITY = new HashMap<String, Double>();
    static {
        STRESS_SENSITIVITY.put("cloud", 0.25);
        STRESS_SENSITIVITY.put("fog",   1.00);
        STRESS_SENSITIVITY.put("edge",  1.25);
    }

    /** Queue depth treated as fully saturated, for normalisation. */
    private static final double QUEUE_SATURATION = 120.0;
    /** Uplink latency treated as fully degraded, in ms, for normalisation. */
    private static final double LATENCY_SATURATION = 150.0;
    /** Handovers per window treated as fully degraded, for normalisation. */
    private static final double HANDOVER_SATURATION = 8.0;

    private final long seed;
    /** Absolute simulation time at which each node fails; absent means still alive. */
    private final Map<Integer, Double> failureTimes = new HashMap<Integer, Double>();

    public FaultInjector(long seed) {
        this.seed = seed;
    }

    /**
     * Deterministic uniform draw for one (node, epoch, stream) triple.
     *
     * <p>Deliberately not a shared {@link java.util.Random} sequence. A single
     * sequence makes every draw depend on how many draws came before it, and
     * therefore on the order in which CloudSim happens to deliver the
     * resource-management events -- which is not stable between runs. Seeding the
     * generator did not make the simulation reproducible: three runs at seed
     * 20260912 retained 6326, 6184 and 6355 snapshots and produced three different
     * case files.
     *
     * <p>Hashing the coordinates of the draw instead makes each one independent of
     * every other, so the failure process is identical run to run no matter what
     * order the events arrive in. The mixing function is splitmix64.
     */
    private double uniform(int nodeId, long epochIndex, long stream) {
        long h = seed;
        h += 0x9E3779B97F4A7C15L * (nodeId + 1);
        h += 0xBF58476D1CE4E5B9L * (epochIndex + 1);
        h += 0x94D049BB133111EBL * (stream + 1);

        h ^= (h >>> 30);
        h *= 0xBF58476D1CE4E5B9L;
        h ^= (h >>> 27);
        h *= 0x94D049BB133111EBL;
        h ^= (h >>> 31);

        // Top 53 bits to a double in [0, 1).
        return (h >>> 11) * 0x1.0p-53;
    }

    public static double reliabilityPriorFor(String deviceClass) {
        Double p = RELIABILITY_PRIOR.get(deviceClass);
        return p == null ? 0.85 : p;
    }

    public static double mtbfHoursFor(String deviceClass) {
        Double m = MTBF_HOURS.get(deviceClass);
        return m == null ? 8760.0 : m;
    }

    /** Base per-hour hazard implied by the device class MTBF, before acceleration. */
    private static double baseHazard(String deviceClass) {
        return 1.0 / mtbfHoursFor(deviceClass);
    }

    /**
     * Clamps to [0,1], treating an unmeasurable reading as no stress.
     *
     * <p>Jitter is now genuinely absent whenever too few tuples reached a node to
     * measure the variability of their arrival timing, so NaN reaches here. Left
     * unhandled it would poison the exponent and make the hazard NaN, which would
     * silently stop every node in the run from ever failing.
     */
    private static double clamp01(double v) {
        if (Double.isNaN(v)) return 0.0;
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    /**
     * Computes the stress-accelerated hazard for one snapshot.
     * Temperature is excluded: it is unobserved on simulated nodes, and a ground-truth
     * process must not depend on a channel the predictor can never see.
     */
    private double hazard(NodeTelemetry t) {
        double queueNorm    = clamp01(t.queueDepth / QUEUE_SATURATION);
        double latencyNorm  = clamp01(t.uplinkLatency / LATENCY_SATURATION);
        double handoverNorm = clamp01(t.handoverRate / HANDOVER_SATURATION);
        double depletion    = clamp01(1.0 - t.residualEnergy);

        double exponent = A_CPU      * clamp01(t.cpuUtil)
                        + B_RAM      * clamp01(t.ramUtil)
                        + C_QUEUE    * queueNorm
                        + D_LATENCY  * latencyNorm
                        + E_ENERGY   * depletion
                        + F_JITTER   * clamp01(t.jitterPktLoss)
                        + G_HANDOVER * handoverNorm;

        Double sensitivity = STRESS_SENSITIVITY.get(t.deviceClass);
        double s = (sensitivity == null) ? 1.0 : sensitivity;

        return baseHazard(t.deviceClass) * ACCELERATION_FACTOR * Math.exp(s * exponent);
    }

    /**
     * Advances the failure process for one node by one epoch, given the telemetry
     * observed at that epoch. Returns true if the node is already dead.
     *
     * <p>Failure is sampled from an exponential survival model over the epoch:
     * P(fail in this epoch) = 1 - exp(-lambda * hours_per_epoch). Once a node fails
     * it stays failed.
     */
    public boolean stepAndSample(NodeTelemetry t, double epochLengthInTimeUnits) {
        if (failureTimes.containsKey(t.nodeId)) {
            return true;
        }
        long epochIndex = Math.round(t.epoch / Math.max(1.0, epochLengthInTimeUnits));

        double lambda = hazard(t);
        double pFail = 1.0 - Math.exp(-lambda * HOURS_PER_EPOCH);
        if (uniform(t.nodeId, epochIndex, 0L) < pFail) {
            // Place the failure at a uniformly random instant inside this epoch,
            // drawn from an independent stream so it does not disturb the next
            // node's survival draw.
            double offset = uniform(t.nodeId, epochIndex, 1L) * epochLengthInTimeUnits;
            failureTimes.put(t.nodeId, t.epoch + offset);
            return false; // it was alive when this snapshot was taken
        }
        return false;
    }

    /** Absolute failure time of a node, or NaN if it survived the simulation. */
    public double failureTimeOf(int nodeId) {
        Double f = failureTimes.get(nodeId);
        return f == null ? Double.NaN : f;
    }

    public boolean hasFailed(int nodeId) {
        return failureTimes.containsKey(nodeId);
    }

    public int failedNodeCount() {
        return failureTimes.size();
    }

    /**
     * Ground-truth label for a snapshot: does the node fail within (t, t + delta_t]?
     * This is exactly the event the BBN's target variable F is defined to predict.
     */
    public int labelFor(NodeTelemetry t) {
        double ft = failureTimeOf(t.nodeId);
        if (Double.isNaN(ft)) return 0;
        return (ft > t.epoch && ft <= t.epoch + HORIZON_DELTA_T) ? 1 : 0;
    }
}
