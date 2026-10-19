package org.fog.ft;

/**
 * One user's analysis session: the ecg_analyser state serving one edge device.
 *
 * <p>The session is what fault tolerance protects. It is hosted by the user's serving
 * node (the edge device's parent), it is unavailable while that node is dead or while
 * its state is being restored or transferred, and every such interval is charged to the
 * session as downtime. Outages are kept as a union, so overlapping causes (a node that
 * dies in the middle of a migration, say) are never counted twice.
 *
 * <p>The same availability window gates the real iFogSim tuple flow: an analyser host
 * drops this user's tuples while {@link #isAvailable} is false, so the measured loop
 * delays and delivered alarms agree with the downtime accounting.
 */
public final class ServiceSession {

    /** Protection tier of Novelty 3. */
    public enum Tier { NONE, CHECKPOINT, WARM_STANDBY }

    public final int edgeId;
    public final String edgeName;
    public final Criticality criticality;

    /** Node currently hosting the session (the edge device's parent). */
    public int servingNodeId = -1;
    /** Analyses for this user are dropped until this time. */
    public double unavailableUntil = 0.0;
    /** The user's own device has failed; nothing on the fog side can help. */
    public boolean terminated = false;
    public double terminatedAt = Double.NaN;

    public Tier tier = Tier.NONE;
    public int standbyNodeId = -1;
    public double lastMigrationTime = -1e18;

    /** Proactive migration in flight. */
    public boolean migrating = false;
    public int migrationSource = -1;
    public int migrationTarget = -1;
    public double migrationStart = Double.NaN;

    // ---- accounting -------------------------------------------------------------
    private double outageEnd = 0.0;
    public double downtime = 0.0;
    public double downtimeFailure = 0.0;
    public double downtimeMigration = 0.0;
    public double downtimeHandover = 0.0;

    public long analysesEmitted = 0;
    public long analysesDroppedFailedNode = 0;
    public long analysesDroppedUnavailable = 0;
    public long alarmsDelivered = 0;
    public long deadlineMisses = 0;
    public double loopDelaySum = 0.0;
    public final Reservoir loopDelays = new Reservoir(4096);

    public int failuresHit = 0;
    public int proactiveMigrations = 0;

    public ServiceSession(int edgeId, String edgeName, Criticality criticality) {
        this.edgeId = edgeId;
        this.edgeName = edgeName;
        this.criticality = criticality;
    }

    public boolean isAvailable(double now) {
        return !terminated && now >= unavailableUntil;
    }

    /** Kinds of outage, for the breakdown in the run summary. */
    public enum Cause { FAILURE, MIGRATION, HANDOVER }

    /**
     * Charges the interval [start, end) as an outage, counting only the part not
     * already covered by an earlier outage, and extends the availability gate.
     */
    public void addOutage(double start, double end, Cause cause) {
        if (end <= start) {
            return;
        }
        double from = Math.max(start, outageEnd);
        double added = Math.max(0.0, end - from);
        if (added > 0) {
            downtime += added;
            switch (cause) {
                case FAILURE: downtimeFailure += added; break;
                case MIGRATION: downtimeMigration += added; break;
                default: downtimeHandover += added; break;
            }
        }
        outageEnd = Math.max(outageEnd, end);
        unavailableUntil = Math.max(unavailableUntil, end);
    }

    /** Clamps any outage still open at the end of the run to the run length. */
    public void closeAt(double simEnd) {
        if (outageEnd > simEnd) {
            double over = outageEnd - simEnd;
            downtime -= over;
            downtimeFailure -= Math.min(over, downtimeFailure);
            outageEnd = simEnd;
        }
    }

    public void recordLoopDelay(double delay, double deadline) {
        alarmsDelivered++;
        loopDelaySum += delay;
        loopDelays.add(delay);
        if (delay > deadline) {
            deadlineMisses++;
        }
    }

    public double meanLoopDelay() {
        return alarmsDelivered == 0 ? Double.NaN : loopDelaySum / alarmsDelivered;
    }

    /** Fixed-size uniform reservoir sample, for tail percentiles without unbounded memory. */
    public static final class Reservoir {
        private final double[] values;
        private long seen = 0;
        private long state = 0x9E3779B97F4A7C15L;

        public Reservoir(int capacity) {
            values = new double[capacity];
        }

        public void add(double v) {
            if (seen < values.length) {
                values[(int) seen] = v;
            } else {
                state = FtRandom.mix(state + seen);
                long j = Long.remainderUnsigned(state, seen + 1);
                if (j < values.length) {
                    values[(int) j] = v;
                }
            }
            seen++;
        }

        public int size() {
            return (int) Math.min(seen, values.length);
        }

        public double get(int i) {
            return values[i];
        }
    }
}
