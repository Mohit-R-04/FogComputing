package org.fog.ft;

/**
 * One telemetry snapshot taken from a fog node at one resource-management epoch.
 *
 * <p>The fields correspond one-for-one to the evidence tier of the Bayesian Belief
 * Network described in Section III.B of the Review 2 paper. Values are kept
 * continuous here; discretisation into {low, medium, high} happens on the Python
 * side against quantiles of the generated data, so that a state such as "high CPU
 * utilisation" is defined relative to the observed distribution rather than by an
 * arbitrary cut-off.
 *
 * <p>Two channels can be absent, and both are written as an empty CSV field so the
 * Python loader treats them as "did not report" and marginalises them out rather
 * than imputing anything:
 *
 * <ul>
 *   <li>{@code temperature} -- always, since no simulated node exposes a thermal
 *       sensor;</li>
 *   <li>{@code jitterPktLoss} -- whenever too few tuples reached the node in the
 *       window to measure the variability of their arrival times.</li>
 * </ul>
 */
public class NodeTelemetry {

    /** Which scenario of the sweep produced this row. */
    public final int scenarioId;

    /** Simulation-level identity of the node. */
    public final int nodeId;
    public final String nodeName;
    /** 0 = cloud, 1 = fog node, 2 = edge device. */
    public final int level;
    /** Device class label, used for the node-reliability prior. */
    public final String deviceClass;
    /** Simulation clock at which this snapshot was taken. */
    public final double epoch;

    // ---- evidence tier (8 variables) ----------------------------------------
    /** Fraction of host MIPS requested, in [0, 1]. */
    public final double cpuUtil;
    /** Fraction of host RAM allocated, in [0, 1]. */
    public final double ramUtil;
    /** Pending and arriving tuples for the window. */
    public final int queueDepth;
    /** Effective uplink latency in ms: configured link delay plus queueing delay. */
    public final double uplinkLatency;
    /** Arrival-time jitter in [0, 1], or NaN when too few arrivals to measure. */
    public final double jitterPktLoss;
    /** Parent reassignments observed over the trailing window. */
    public final int handoverRate;
    /** Remaining energy as a fraction of declared capacity, in [0, 1]. */
    public final double residualEnergy;
    /** Die/ambient temperature in degrees Celsius, or NaN when unobserved. */
    public final double temperature;

    // ---- reliability prior ---------------------------------------------------
    /** Prior reliability of the device class, in [0, 1]; higher is more reliable. */
    public final double reliabilityPrior;

    // ---- ground truth (hidden from the BBN until verification) ---------------
    /** 1 if this node actually failed within the look-ahead horizon, else 0. */
    public int groundTruthFail;
    /** Absolute time of the injected failure, or NaN if the node never failed. */
    public double failureTime;
    /** Set by the collector once the scenario is known. */
    public int scenario;

    public NodeTelemetry(int nodeId, String nodeName, int level, String deviceClass,
                         double epoch, double cpuUtil, double ramUtil, int queueDepth,
                         double uplinkLatency, double jitterPktLoss, int handoverRate,
                         double residualEnergy, double temperature,
                         double reliabilityPrior) {
        this(-1, nodeId, nodeName, level, deviceClass, epoch, cpuUtil, ramUtil,
             queueDepth, uplinkLatency, jitterPktLoss, handoverRate, residualEnergy,
             temperature, reliabilityPrior);
    }

    public NodeTelemetry(int scenarioId, int nodeId, String nodeName, int level,
                         String deviceClass, double epoch, double cpuUtil,
                         double ramUtil, int queueDepth, double uplinkLatency,
                         double jitterPktLoss, int handoverRate, double residualEnergy,
                         double temperature, double reliabilityPrior) {
        this.scenarioId = scenarioId;
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
        this.reliabilityPrior = reliabilityPrior;
        this.groundTruthFail = 0;
        this.failureTime = Double.NaN;
    }

    /** Returns a copy of this snapshot tagged with a scenario id. */
    public NodeTelemetry withScenario(int id) {
        NodeTelemetry t = new NodeTelemetry(id, nodeId, nodeName, level, deviceClass,
                epoch, cpuUtil, ramUtil, queueDepth, uplinkLatency, jitterPktLoss,
                handoverRate, residualEnergy, temperature, reliabilityPrior);
        t.groundTruthFail = groundTruthFail;
        t.failureTime = failureTime;
        return t;
    }

    /** Header matching {@link #toCsvRow(int)}. */
    public static String csvHeader() {
        return "case_id,scenario_id,node_id,node_name,level,device_class,epoch,"
             + "cpu_util,ram_util,queue_depth,uplink_latency,jitter_pktloss,"
             + "handover_rate,residual_energy,temperature,reliability_prior,"
             + "ground_truth_fail,failure_time";
    }

    /** Formats a possibly-absent measurement: empty means "did not report". */
    private static String optional(double value, String format) {
        return Double.isNaN(value) ? "" : String.format(format, value);
    }

    /**
     * Renders the snapshot as a CSV row. Unobserved channels are written as empty
     * fields, which the Python loader reads as "absent from the evidence set".
     */
    public String toCsvRow(int caseId) {
        return String.format(
                "%d,%d,%d,%s,%d,%s,%.1f,%.4f,%.4f,%d,%.3f,%s,%d,%.4f,%s,%.3f,%d,%s",
                caseId, scenarioId, nodeId, nodeName, level, deviceClass, epoch,
                cpuUtil, ramUtil, queueDepth, uplinkLatency,
                optional(jitterPktLoss, "%.4f"),
                handoverRate, residualEnergy,
                optional(temperature, "%.2f"),
                reliabilityPrior, groundTruthFail,
                optional(failureTime, "%.1f"));
    }
}
