package org.fog.ft;

import org.fog.utils.Config;

/**
 * Time and cost constants of the fault-tolerance model, all in simulation time units
 * (iFogSim's unit, read as milliseconds for network timing).
 *
 * <p>Recovery times follow the usual ordering of the literature: a cold restart (which
 * must rebuild the analyser's signal context) is far slower than restoring a
 * checkpoint, which is slower than promoting a synchronised warm standby, which is
 * slower than the stop-and-copy pause of a live migration.
 *
 * <p>The decision costs are expressed in "downtime-equivalent" units so that the
 * expected-loss comparisons of Eqs. (3) and (4) compare like with like; only their
 * ratios matter. {@link #migrationCostScale} multiplies every migration cost and is the
 * knob of the sensitivity analysis over C_mig / C_fail promised in Review 1.
 */
public final class FtCostModel {

    private FtCostModel() {
    }

    public static final double EPOCH = Config.RESOURCE_MGMT_INTERVAL;

    // ---- detection and recovery ---------------------------------------------------
    /** Three missed heartbeats at a 10-unit interval. */
    public static final double HEARTBEAT_TIMEOUT = 30.0;
    /** Without application heartbeats a device notices only that its link is gone. */
    public static final double LINK_LOSS_TIMEOUT = 50.0;
    /** Instantiate the analyser and rebuild its ECG context window from scratch. */
    public static final double COLD_START = 300.0;
    /** Promote a synchronised warm standby. */
    public static final double WARM_SWITCHOVER = 5.0;
    /** Checkpoint period under load (Rajab and Younis checkpoint when CPU > 0.2). */
    public static final double CHECKPOINT_INTERVAL = 50.0;
    /** Checkpoint period on a lightly loaded node. */
    public static final double CHECKPOINT_INTERVAL_IDLE = 200.0;
    /** Fog-to-fog state transfer rate: 100 Mbps is 12.5 KB per ms. */
    public static final double FOG_LINK_KB_PER_UNIT = 12.5;
    /** Live (pre-copy) migration pauses the service for a fraction of the transfer. */
    public static final double STOP_AND_COPY_FRACTION = 0.10;
    public static final double STOP_AND_COPY_FIXED = 1.0;

    // ---- decision costs (downtime-equivalent units) ---------------------------------
    /** Network cost per unit of state-transfer time. */
    public static final double C_NET_PER_UNIT = 0.05;
    /** Control-plane overhead of one migration. */
    public static final double C_MIGRATION_FIXED = 2.0;
    /** Penalty per ms of extra access latency at the new node. */
    public static final double C_LATENCY_PER_MS = 0.5;
    /** Reserving compute for a warm standby for one epoch (on top of its sync traffic). */
    public static final double C_RESERVE_PER_EPOCH = 3.0;
    /** Budget B: at most this share of sessions may hold a warm standby at once. */
    public static final double WARM_BUDGET_FRACTION = 0.20;

    /** Sensitivity-analysis multiplier on every migration cost (default 1). */
    public static double migrationCostScale = 1.0;

    public static double transferTime(ServiceSession s) {
        return s.criticality.stateKb / FOG_LINK_KB_PER_UNIT;
    }

    public static double stopAndCopy(ServiceSession s) {
        return STOP_AND_COPY_FIXED + STOP_AND_COPY_FRACTION * transferTime(s);
    }

    public static double checkpointRestore(ServiceSession s) {
        return transferTime(s) + CHECKPOINT_INTERVAL / 2.0;
    }

    /** Outage a session suffers if its node fails while it holds {@code tier}. */
    public static double outageIfFails(ServiceSession s, ServiceSession.Tier tier, double detect) {
        switch (tier) {
            case WARM_STANDBY: return detect + WARM_SWITCHOVER;
            case CHECKPOINT: return detect + checkpointRestore(s);
            default: return detect + COLD_START;
        }
    }

    /** C_fail(m) of Eq. (3): criticality times the outage a failure would cause. */
    public static double failureCost(ServiceSession s, ServiceSession.Tier tier) {
        return s.criticality.kappa * outageIfFails(s, tier, HEARTBEAT_TIMEOUT);
    }

    /** C_mig(m, n -> n') of Eq. (3): pause, network, control plane and latency given up. */
    public static double migrationCost(ServiceSession s, double addedLatencyMs) {
        double kappa = s.criticality.kappa;
        return migrationCostScale * (kappa * stopAndCopy(s)
                + C_NET_PER_UNIT * transferTime(s)
                + C_MIGRATION_FIXED
                + kappa * C_LATENCY_PER_MS * Math.max(0.0, addedLatencyMs));
    }

    /** Checkpoint traffic cost for one epoch. */
    public static double checkpointCostPerEpoch(ServiceSession s) {
        return C_NET_PER_UNIT * transferTime(s) * (EPOCH / CHECKPOINT_INTERVAL);
    }

    /** Warm standby cost for one epoch: continuous state sync plus reserved compute. */
    public static double warmCostPerEpoch(ServiceSession s) {
        return checkpointCostPerEpoch(s) + C_RESERVE_PER_EPOCH;
    }

    /** Checkpoint traffic in KB for one epoch at a given checkpoint period. */
    public static double checkpointKbPerEpoch(ServiceSession s, double interval) {
        return s.criticality.stateKb * (EPOCH / interval);
    }
}
