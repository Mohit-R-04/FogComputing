package org.fog.ft;

/**
 * One telemetry snapshot taken from a node at one resource-management epoch.
 *
 * <p>The eight measurement channels correspond one-for-one to the evidence tier of the
 * Bayesian Belief Network (Section III.B of the Review 2 paper). The five trend
 * channels are what a monitoring agent reports alongside the raw readings -- the rate
 * of change over its last few samples -- because a node that is degrading shows it as
 * a drift before it shows it as an extreme value. Values are kept continuous here;
 * discretisation into {low, medium, high} happens on the Python side.
 *
 * <p>Any channel can be absent and is written as an empty CSV field, which every
 * loader treats as "did not report". Absent channels arise from:
 * <ul>
 *   <li>{@code temperature} on edge devices, which expose no thermal sensor;</li>
 *   <li>{@code jitterPktLoss} when too few tuples reached the node to measure it;</li>
 *   <li>trend channels before the agent has a history to compare against;</li>
 *   <li>injected telemetry dropout in the robustness experiments.</li>
 * </ul>
 *
 * <p>{@code trueHealth} and {@code failureCause} are ground truth from the fault
 * model. They are written for analysis and for the oracle baseline and are never
 * given to any predictor.
 */
public class NodeTelemetry {

    /** Which scenario of the sweep and which seed produced this row. */
    public final int scenarioId;
    public final long seed;

    /** Simulation-level identity of the node. */
    public final int nodeId;
    public final String nodeName;
    /** 0 = cloud, 1 = fog node, 2 = edge device. */
    public final int level;
    /** Device class label, used for the node-reliability prior. */
    public final String deviceClass;
    /** Simulation clock at which this snapshot was taken. */
    public final double epoch;

    // ---- evidence tier: measurements ----------------------------------------
    /** Work executed against capacity over the window, in [0, 1]. */
    public final double cpuUtil;
    /** Fraction of host RAM allocated, in [0, 1]. */
    public final double ramUtil;
    /** Pending and arriving tuples for the window. */
    public final double queueDepth;
    /** Effective uplink latency in ms: link delay plus transmission load. */
    public final double uplinkLatency;
    /** Arrival-time jitter in [0, 1]. */
    public final double jitterPktLoss;
    /** Topology changes over the trailing window (own parent changes for edge
     *  devices; user attach/detach events for fog nodes). */
    public final double handoverRate;
    /** Remaining energy as a fraction of declared capacity, in [0, 1]. */
    public final double residualEnergy;
    /** Die/enclosure temperature in degrees Celsius. */
    public final double temperature;

    // ---- evidence tier: agent-side trends -----------------------------------
    /** cpu_util minus the mean of the agent's previous readings. */
    public final double cpuTrend;
    /** Relative change of queue depth against the previous readings. */
    public final double queueTrend;
    /** Relative change of uplink latency against the previous readings. */
    public final double latencyTrend;
    /** Temperature change against the previous readings, degrees Celsius. */
    public final double tempTrend;
    /** Residual energy consumed since the previous reading. */
    public final double energyDrain;

    // ---- reliability prior ---------------------------------------------------
    /** Prior reliability of the device class, in [0, 1]; higher is more reliable. */
    public final double reliabilityPrior;

    // ---- ground truth (never shown to a predictor) ---------------------------
    public final double trueHealth;
    /** 1 if this node failed within the look-ahead horizon, else 0. */
    public int groundTruthFail;
    /** Absolute time of the failure, or NaN if the node never failed. */
    public double failureTime;
    /** wear-out, shock, sudden, battery, or empty. */
    public String failureCause = "";

    public NodeTelemetry(int scenarioId, long seed, int nodeId, String nodeName, int level,
                         String deviceClass, double epoch,
                         double cpuUtil, double ramUtil, double queueDepth,
                         double uplinkLatency, double jitterPktLoss, double handoverRate,
                         double residualEnergy, double temperature,
                         double cpuTrend, double queueTrend, double latencyTrend,
                         double tempTrend, double energyDrain,
                         double reliabilityPrior, double trueHealth) {
        this.scenarioId = scenarioId;
        this.seed = seed;
        this.nodeId = nodeId;
        this.nodeName = nodeName;
        this.level = level;
        this.deviceClass = deviceClass;
        this.epoch = epoch;
        this.cpuUtil = cpuUtil;
        this.ramUtil = ramUtil;
        this.queueDepth = queueDepth;
        this.uplinkLatency = uplinkLatency;
        this.jitterPktLoss = jitterPktLoss;
        this.handoverRate = handoverRate;
        this.residualEnergy = residualEnergy;
        this.temperature = temperature;
        this.cpuTrend = cpuTrend;
        this.queueTrend = queueTrend;
        this.latencyTrend = latencyTrend;
        this.tempTrend = tempTrend;
        this.energyDrain = energyDrain;
        this.reliabilityPrior = reliabilityPrior;
        this.trueHealth = trueHealth;
        this.groundTruthFail = 0;
        this.failureTime = Double.NaN;
    }

    /** Channel names in a fixed order, shared with the dropout model and predictors. */
    public static final String[] CHANNELS = {
        "cpu_util", "ram_util", "queue_depth", "uplink_latency", "jitter_pktloss",
        "handover_rate", "residual_energy", "temperature",
        "cpu_trend", "queue_trend", "latency_trend", "temp_trend", "energy_drain",
    };

    /** Value of a channel by name; NaN when the channel did not report. */
    public double value(String channel) {
        switch (channel) {
            case "cpu_util": return cpuUtil;
            case "ram_util": return ramUtil;
            case "queue_depth": return queueDepth;
            case "uplink_latency": return uplinkLatency;
            case "jitter_pktloss": return jitterPktLoss;
            case "handover_rate": return handoverRate;
            case "residual_energy": return residualEnergy;
            case "temperature": return temperature;
            case "cpu_trend": return cpuTrend;
            case "queue_trend": return queueTrend;
            case "latency_trend": return latencyTrend;
            case "temp_trend": return tempTrend;
            case "energy_drain": return energyDrain;
            default: throw new IllegalArgumentException("unknown channel " + channel);
        }
    }

    /** Header matching {@link #toCsvRow(int)}. */
    public static String csvHeader() {
        return "case_id,scenario_id,seed,node_id,node_name,level,device_class,epoch,"
             + "cpu_util,ram_util,queue_depth,uplink_latency,jitter_pktloss,"
             + "handover_rate,residual_energy,temperature,"
             + "cpu_trend,queue_trend,latency_trend,temp_trend,energy_drain,"
             + "reliability_prior,true_health,ground_truth_fail,failure_time,failure_cause";
    }

    /** Formats a possibly-absent measurement: empty means "did not report". */
    private static String optional(double value, String format) {
        return Double.isNaN(value) ? "" : String.format(java.util.Locale.ROOT, format, value);
    }

    /** Renders the snapshot as a CSV row; absent channels become empty fields. */
    public String toCsvRow(int caseId) {
        return String.format(java.util.Locale.ROOT,
                "%d,%d,%d,%d,%s,%d,%s,%.1f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%.4f,%d,%s,%s",
                caseId, scenarioId, seed, nodeId, nodeName, level, deviceClass, epoch,
                optional(cpuUtil, "%.4f"),
                optional(ramUtil, "%.4f"),
                optional(queueDepth, "%.1f"),
                optional(uplinkLatency, "%.3f"),
                optional(jitterPktLoss, "%.4f"),
                optional(handoverRate, "%.0f"),
                optional(residualEnergy, "%.4f"),
                optional(temperature, "%.2f"),
                optional(cpuTrend, "%.4f"),
                optional(queueTrend, "%.4f"),
                optional(latencyTrend, "%.4f"),
                optional(tempTrend, "%.3f"),
                optional(energyDrain, "%.5f"),
                reliabilityPrior, trueHealth, groundTruthFail,
                optional(failureTime, "%.1f"),
                failureCause == null ? "" : failureCause);
    }
}
