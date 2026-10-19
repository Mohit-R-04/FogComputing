package org.fog.ft;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Accumulates the reported telemetry of every node and writes one labelled CSV per run.
 *
 * <p>This class deliberately does not sample, split or discretise. Its whole job is to
 * emit everything the control plane observed, labelled with the ground truth from the
 * {@link FaultInjector}, so that {@code build_dataset.py} can decide what to do with it
 * without anyone having to recompile Java to change a sampling rule. A dead node
 * reports nothing, so no row is ever emitted for a node after its failure.
 *
 * <p>Singleton because iFogSim device callbacks have no natural place to thread a
 * collector reference through. {@link #reset()} must be called before each run.
 */
public class TelemetryCollector {

    private static TelemetryCollector instance = new TelemetryCollector();

    private final List<NodeTelemetry> snapshots = new ArrayList<NodeTelemetry>();
    private boolean keepRows = true;
    private boolean labelled = false;

    private TelemetryCollector() {
    }

    public static synchronized TelemetryCollector getInstance() {
        return instance;
    }

    /** Clears all state. Call before every simulation. */
    public static synchronized void reset(boolean keepRows) {
        instance = new TelemetryCollector();
        instance.keepRows = keepRows;
    }

    public void record(NodeTelemetry t) {
        if (keepRows) {
            snapshots.add(t);
            labelled = false;
        }
    }

    public int snapshotCount() {
        return snapshots.size();
    }

    /** Canonical order (node name, then time), independent of CloudSim event order. */
    private void sortCanonically() {
        Collections.sort(snapshots, new Comparator<NodeTelemetry>() {
            @Override
            public int compare(NodeTelemetry a, NodeTelemetry b) {
                if (a.level != b.level) return Integer.compare(a.level, b.level);
                int byName = a.nodeName.compareTo(b.nodeName);
                if (byName != 0) return byName;
                return Double.compare(a.epoch, b.epoch);
            }
        });
    }

    /** Applies the ground-truth label to every retained snapshot. */
    private void labelAll() {
        if (labelled) return;
        sortCanonically();
        FaultInjector injector = RunContext.injector();
        for (NodeTelemetry t : snapshots) {
            t.groundTruthFail = injector.labelFor(t.nodeName, t.epoch);
            t.failureTime = injector.failureTimeOf(t.nodeName);
            t.failureCause = injector.failureCauseOf(t.nodeName);
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

    /** Writes every retained snapshot, labelled. */
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

    /** Per-class snapshot and positive counts, for the run summary. */
    public Map<String, int[]> classBreakdown() {
        labelAll();
        Map<String, int[]> out = new TreeMap<String, int[]>();
        for (NodeTelemetry t : snapshots) {
            int[] c = out.get(t.deviceClass);
            if (c == null) {
                c = new int[2];
                out.put(t.deviceClass, c);
            }
            c[0]++;
            if (t.groundTruthFail == 1) c[1]++;
        }
        return out;
    }

    /** How many rows carry a reading for each channel, as a percentage. */
    public Map<String, Integer> channelCoverage() {
        Map<String, Integer> out = new LinkedHashMap<String, Integer>();
        if (snapshots.isEmpty()) return out;
        for (String ch : NodeTelemetry.CHANNELS) {
            int present = 0;
            for (NodeTelemetry t : snapshots) {
                if (!Double.isNaN(t.value(ch))) present++;
            }
            out.put(ch, (int) Math.round(100.0 * present / snapshots.size()));
        }
        return out;
    }
}
