package org.fog.ft;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Accumulates telemetry from every instrumented node, drives the ground-truth
 * failure process, and writes one raw CSV per scenario.
 *
 * <p>This class deliberately does not sample, split or discretise. Its whole job is
 * to emit everything the simulation observed, labelled, so that
 * {@code build_dataset.py} can decide what to do with it without anyone having to
 * recompile Java to change a sampling rule.
 *
 * <p>Singleton because iFogSim device callbacks have no natural place to thread a
 * collector reference through. {@link #reset(long, int)} must be called before each
 * run.
 */
public class TelemetryCollector {

    private static TelemetryCollector instance;

    private final List<NodeTelemetry> snapshots = new ArrayList<NodeTelemetry>();
    private FaultInjector faultInjector;
    private double epochLength = 100.0;
    private int scenarioId = -1;
    private boolean labelled = false;

    private TelemetryCollector() {
    }

    public static synchronized TelemetryCollector getInstance() {
        if (instance == null) {
            instance = new TelemetryCollector();
        }
        return instance;
    }

    /** Clears all state and seeds a fresh run. Call before every simulation. */
    public static synchronized void reset(long seed, int scenarioId) {
        instance = new TelemetryCollector();
        instance.faultInjector = new FaultInjector(seed);
        instance.scenarioId = scenarioId;
    }

    public void setEpochLength(double epochLength) {
        this.epochLength = epochLength;
    }

    /**
     * Records one snapshot and advances that node's failure process by one epoch.
     * Snapshots taken after a node has already died are discarded: a dead node
     * reports no telemetry, and keeping such rows would leak the label.
     */
    public void record(NodeTelemetry t) {
        if (faultInjector.stepAndSample(t, epochLength)) {
            return;
        }
        snapshots.add(t.withScenario(scenarioId));
    }

    public int snapshotCount() {
        return snapshots.size();
    }

    public int failedNodeCount() {
        return faultInjector.failedNodeCount();
    }

    /**
     * Puts the snapshots in a canonical order, by node then by time.
     *
     * <p>They arrive in whatever order CloudSim delivered the resource-management
     * events. Sorting here means the output file does not depend on that order.
     */
    private void sortCanonically() {
        Collections.sort(snapshots, new Comparator<NodeTelemetry>() {
            @Override
            public int compare(NodeTelemetry a, NodeTelemetry b) {
                if (a.nodeId != b.nodeId) return Integer.compare(a.nodeId, b.nodeId);
                return Double.compare(a.epoch, b.epoch);
            }
        });
    }

    /** Applies the ground-truth label to every retained snapshot. */
    private void labelAll() {
        if (labelled) return;
        sortCanonically();
        for (NodeTelemetry t : snapshots) {
            t.groundTruthFail = faultInjector.labelFor(t);
            t.failureTime = faultInjector.failureTimeOf(t.nodeId);
        }
        labelled = true;
    }

    public int positiveCount() {
        labelAll();
        int n = 0;
        for (NodeTelemetry t : snapshots) {
            if (t.groundTruthFail == 1) n++;
        }
        return n;
    }

    /** Writes every retained snapshot, labelled, for this scenario. */
    public void write(String path) throws IOException {
        labelAll();
        File f = new File(path);
        if (f.getParentFile() != null) f.getParentFile().mkdirs();
        PrintWriter w = new PrintWriter(new FileWriter(f));
        try {
            w.println(NodeTelemetry.csvHeader());
            int id = 1;
            for (NodeTelemetry t : snapshots) {
                w.println(t.toCsvRow(id++));
            }
        } finally {
            w.close();
        }
    }

    /** Per-level snapshot and positive counts, for the run summary. */
    public Map<Integer, int[]> levelBreakdown() {
        labelAll();
        Map<Integer, int[]> out = new HashMap<Integer, int[]>();
        for (NodeTelemetry t : snapshots) {
            int[] c = out.get(t.level);
            if (c == null) {
                c = new int[2];
                out.put(t.level, c);
            }
            c[0]++;
            if (t.groundTruthFail == 1) c[1]++;
        }
        return out;
    }

    /**
     * How many rows actually carry a reading for each optional channel.
     *
     * <p>Printed after every run so a channel that silently stops reporting is
     * visible immediately rather than discovered later in the Python.
     */
    public Map<String, int[]> channelCoverage() {
        Map<String, int[]> out = new HashMap<String, int[]>();
        out.put("jitter_pktloss", new int[]{0, snapshots.size()});
        out.put("temperature", new int[]{0, snapshots.size()});
        for (NodeTelemetry t : snapshots) {
            if (!Double.isNaN(t.jitterPktLoss)) out.get("jitter_pktloss")[0]++;
            if (!Double.isNaN(t.temperature)) out.get("temperature")[0]++;
        }
        return out;
    }
}
