package org.fog.ft;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * Exact inference over the trained fog-node Bayesian Belief Network, inside the
 * simulator.
 *
 * <p>The network is trained in Python ({@code model/train_bbn.py}) and exported to
 * {@code data/bbn_trained.json}: structure, state spaces, every conditional probability
 * table and the discretisation cut points. This class evaluates Eq. (1) of the paper,
 * <pre>  P(F = fail | e) = sum_u prod_V P(V | pa(V)) / P(e)  </pre>
 * exactly, for the layered structure the paper specifies: evidence variables (roots, or
 * children of the reliability prior), a latent tier whose parents are evidence variables
 * (and optionally the device class), and the target whose parents are the latent
 * variables and the prior. Because each evidence variable feeds one latent variable, the
 * latent variables are independent given the evidence and the device class, so the sum
 * over unobserved variables factorises:
 * <pre>
 *   P(L_i | e, p) = sum over L_i's unobserved parents m of P(L_i | pa) prod P(m | p)
 *   P(F | e, p)   = sum over latent configurations l of prod_i P(l_i | e, p) P(F | l, p)
 * </pre>
 * A telemetry channel that did not report is summed out, never imputed. The result is
 * checked against the Python variable-elimination engine to 1e-9 by
 * {@code model/test_java_parity.py}.
 *
 * <p>Evidence channels may have any number of ordered states, with cut points chosen per
 * device class.
 *
 * <p>The engine also refines its target table online (Novelty 1, workflow step 8): once a
 * report's horizon has elapsed, its outcome contributes expected counts (the same E-step
 * as offline training) and the table is periodically re-estimated as a Dirichlet
 * posterior centred on the offline table. The latent tables are learned ranked nodes with
 * tens of thousands of rows; they are left as trained.
 */
public final class BbnEngine {

    /** Inference result for one node. */
    public static final class Result {
        public final double pFail;
        public final String cause;
        public final double causeLift;
        public final int missing;

        Result(double pFail, String cause, double causeLift, int missing) {
            this.pFail = pFail;
            this.cause = cause;
            this.causeLift = causeLift;
            this.missing = missing;
        }
    }

    private String target;
    private String prior;
    private String[] evidence;
    private String[] latent;
    private final Map<String, String[]> states = new HashMap<String, String[]>();
    private final Map<String, String[]> parents = new HashMap<String, String[]>();
    private final Map<String, double[][]> cpt = new HashMap<String, double[][]>();
    private final Map<String, double[][]> baseCpt = new HashMap<String, double[][]>();
    private final Map<String, double[]> cuts = new HashMap<String, double[]>();
    private final Map<String, Map<String, double[]>> cutsByClass = new HashMap<String, Map<String, double[]>>();
    private final Map<String, String> reliabilityByClass = new HashMap<String, String>();
    private final Map<String, Double> causeWeights = new HashMap<String, Double>();
    /** P(L_i = high) with no telemetry: per reliability-prior state, and for an unknown class. */
    private double[][] latentPriorHighByPrior;
    private double[] latentPriorHigh;
    private int failIndex;

    // ---- online learning -------------------------------------------------------------
    private double onlineConcentration = 400.0;
    private final Map<String, double[][]> counts = new HashMap<String, double[][]>();
    private long observations = 0;
    private int refreshes = 0;
    private double lastMaxChange = 0.0;

    private BbnEngine() {
    }

    // =====================================================================================
    // loading
    // =====================================================================================

    public static BbnEngine load(String path) throws Exception {
        JSONObject root;
        FileReader reader = new FileReader(path);
        try {
            root = (JSONObject) new JSONParser().parse(reader);
        } finally {
            reader.close();
        }
        BbnEngine e = new BbnEngine();
        e.target = (String) root.get("target");
        e.prior = (String) root.get("prior");
        e.evidence = strings((JSONArray) root.get("evidence"));
        e.latent = strings((JSONArray) root.get("latent"));

        JSONObject st = (JSONObject) root.get("states");
        for (Object k : st.keySet()) {
            e.states.put((String) k, strings((JSONArray) st.get(k)));
        }
        JSONObject pa = (JSONObject) root.get("parents");
        for (Object k : pa.keySet()) {
            e.parents.put((String) k, strings((JSONArray) pa.get(k)));
        }
        JSONObject cp = (JSONObject) root.get("cpts");
        for (Object k : cp.keySet()) {
            JSONArray rows = (JSONArray) cp.get(k);
            double[][] table = new double[rows.size()][];
            for (int i = 0; i < rows.size(); i++) {
                JSONArray row = (JSONArray) rows.get(i);
                table[i] = new double[row.size()];
                for (int j = 0; j < row.size(); j++) {
                    table[i][j] = ((Number) row.get(j)).doubleValue();
                }
            }
            e.cpt.put((String) k, table);
            e.baseCpt.put((String) k, copy(table));
        }
        JSONObject cu = (JSONObject) root.get("cuts");
        for (Object k : cu.keySet()) {
            e.cuts.put((String) k, doubles((JSONArray) cu.get(k)));
        }
        JSONObject cbc = (JSONObject) root.get("cuts_by_class");
        if (cbc != null) {
            for (Object cls : cbc.keySet()) {
                JSONObject block = (JSONObject) cbc.get(cls);
                Map<String, double[]> m = new HashMap<String, double[]>();
                for (Object k : block.keySet()) m.put((String) k, doubles((JSONArray) block.get(k)));
                e.cutsByClass.put((String) cls, m);
            }
        }
        JSONObject rel = (JSONObject) root.get("reliability_state_by_class");
        for (Object k : rel.keySet()) {
            e.reliabilityByClass.put((String) k, (String) rel.get(k));
        }
        JSONObject cw = (JSONObject) root.get("cause_weights");
        if (cw != null) {
            for (Object k : cw.keySet()) {
                e.causeWeights.put((String) k, ((Number) cw.get(k)).doubleValue());
            }
        }
        JSONObject online = (JSONObject) root.get("online");
        if (online != null && online.get("concentration") != null) {
            e.onlineConcentration = ((Number) online.get("concentration")).doubleValue();
        }
        e.failIndex = indexOf(e.states.get(e.target), "fail");
        e.validate();

        int[] none = new int[e.evidence.length];
        java.util.Arrays.fill(none, -1);
        e.latentPriorHigh = e.latentHigh(none, -1);
        int nPrior = e.states.get(e.prior).length;
        e.latentPriorHighByPrior = new double[nPrior][];
        for (int s = 0; s < nPrior; s++) e.latentPriorHighByPrior[s] = e.latentHigh(none, s);
        return e;
    }

    private static String[] strings(JSONArray a) {
        String[] out = new String[a.size()];
        for (int i = 0; i < a.size(); i++) out[i] = (String) a.get(i);
        return out;
    }

    private static double[] doubles(JSONArray a) {
        double[] out = new double[a.size()];
        for (int i = 0; i < a.size(); i++) out[i] = ((Number) a.get(i)).doubleValue();
        return out;
    }

    private static int indexOf(String[] arr, String v) {
        for (int i = 0; i < arr.length; i++) if (arr[i].equals(v)) return i;
        return -1;
    }

    private static double[][] copy(double[][] t) {
        double[][] c = new double[t.length][];
        for (int i = 0; i < t.length; i++) c[i] = t[i].clone();
        return c;
    }

    /** Refuses any structure the factorised inference above would get wrong. */
    private void validate() {
        java.util.Set<String> ev = new java.util.HashSet<String>(java.util.Arrays.asList(evidence));
        java.util.Set<String> lat = new java.util.HashSet<String>(java.util.Arrays.asList(latent));
        java.util.Set<String> used = new java.util.HashSet<String>();
        for (String v : evidence) {
            for (String p : parents.get(v)) {
                if (!p.equals(prior)) throw new IllegalStateException(v + " may only depend on the prior");
            }
        }
        for (String l : latent) {
            for (String p : parents.get(l)) {
                if (p.equals(prior)) continue;
                if (!ev.contains(p)) throw new IllegalStateException(l + " has non-evidence parent " + p);
                if (!used.add(p)) throw new IllegalStateException(p + " feeds two latent variables");
            }
        }
        for (String p : parents.get(target)) {
            if (!lat.contains(p) && !p.equals(prior)) {
                throw new IllegalStateException("target parent " + p + " is not latent or prior");
            }
        }
        if (parents.get(prior).length != 0) throw new IllegalStateException("prior must be a root");
        for (String v : cpt.keySet()) {
            int expected = 1;
            for (String p : parents.get(v)) expected *= states.get(p).length;
            if (cpt.get(v).length != expected) {
                throw new IllegalStateException("CPT of " + v + " has " + cpt.get(v).length
                        + " rows, expected " + expected);
            }
        }
        if (failIndex < 0) throw new IllegalStateException("target has no 'fail' state");
    }

    // =====================================================================================
    // evidence
    // =====================================================================================

    /**
     * Bin of a value: the number of cut points at or below it; when every cut point is the
     * same value, everything at or below it is the lowest bin. Identical to
     * discretise.bin_index.
     */
    static int binIndex(double value, double[] c) {
        boolean degenerate = c.length > 0;
        for (int i = 1; i < c.length; i++) {
            if (c[i] != c[0]) {
                degenerate = false;
                break;
            }
        }
        if (degenerate) return value > c[0] ? c.length : 0;
        int n = 0;
        while (n < c.length && c[n] <= value) n++;
        return n;
    }

    /** Cut points of a channel for a device class (per-class if trained that way). */
    private double[] cutsFor(String var, String deviceClass) {
        if (deviceClass != null) {
            Map<String, double[]> block = cutsByClass.get(deviceClass);
            if (block != null && block.containsKey(var)) return block.get(var);
        }
        return cuts.get(var);
    }

    /** Evidence state per evidence variable (-1 = did not report). */
    public int[] evidenceFrom(NodeTelemetry t, String deviceClass) {
        int[] ev = new int[evidence.length];
        for (int i = 0; i < evidence.length; i++) {
            ev[i] = -1;
            if (t == null) continue;
            double[] c = cutsFor(evidence[i], deviceClass);
            if (c == null) continue;
            double v;
            try {
                v = t.value(evidence[i]);
            } catch (IllegalArgumentException unknown) {
                continue;
            }
            if (Double.isNaN(v)) continue;
            ev[i] = binIndex(v, c);
        }
        return ev;
    }

    public int[] evidenceFrom(NodeTelemetry t) {
        return evidenceFrom(t, t == null ? null : t.deviceClass);
    }

    /** Evidence from a CSV row given as column -> raw string (used by the parity tool). */
    public int[] evidenceFromRow(Map<String, String> row) {
        String cls = row.get("device_class");
        int[] ev = new int[evidence.length];
        for (int i = 0; i < evidence.length; i++) {
            ev[i] = -1;
            String raw = row.get(evidence[i]);
            double[] c = cutsFor(evidence[i], cls);
            if (raw == null || raw.trim().isEmpty() || c == null) continue;
            try {
                ev[i] = binIndex(Double.parseDouble(raw.trim()), c);
            } catch (NumberFormatException ignored) {
                // not a number: did not report
            }
        }
        return ev;
    }

    public int priorStateFor(String deviceClass) {
        String s = deviceClass == null ? null : reliabilityByClass.get(deviceClass);
        return s == null ? -1 : indexOf(states.get(prior), s);
    }

    // =====================================================================================
    // inference
    // =====================================================================================

    private int evidenceIndex(String var) {
        return indexOf(evidence, var);
    }

    /** P(root evidence variable m = state | prior p); p may be -1 only for parentless m. */
    private double evidencePrior(String m, int state, int p) {
        double[][] t = cpt.get(m);
        if (parents.get(m).length == 0) return t[0][state];
        return t[p][state];
    }

    /** Distribution of one latent variable given the evidence and a prior state. */
    private double[] latentDistribution(String l, int[] ev, int p) {
        String[] pa = parents.get(l);
        int k = pa.length;
        int[] card = new int[k];
        int[] fixed = new int[k];
        List<Integer> missing = new ArrayList<Integer>();
        for (int j = 0; j < k; j++) {
            card[j] = states.get(pa[j]).length;
            if (pa[j].equals(prior)) {
                fixed[j] = p;                 // the device class is observed
                continue;
            }
            int ei = evidenceIndex(pa[j]);
            fixed[j] = ei >= 0 ? ev[ei] : -1;
            if (fixed[j] < 0) missing.add(j);
        }
        double[][] table = cpt.get(l);
        double[] dist = new double[states.get(l).length];
        int combos = 1;
        for (int j : missing) combos *= card[j];
        int[] assign = fixed.clone();
        for (int c = 0; c < combos; c++) {
            int rem = c;
            double w = 1.0;
            for (int idx = missing.size() - 1; idx >= 0; idx--) {
                int j = missing.get(idx);
                assign[j] = rem % card[j];
                rem /= card[j];
                w *= evidencePrior(pa[j], assign[j], p);
            }
            if (w == 0.0) continue;
            int row = 0;
            for (int j = 0; j < k; j++) row = row * card[j] + assign[j];
            double[] r = table[row];
            for (int s = 0; s < dist.length; s++) dist[s] += w * r[s];
        }
        return dist;
    }

    /** P(F = fail | e, p) for an observed prior state. */
    private double pFailGivenPrior(int[] ev, int p, double[][] dists) {
        String[] tp = parents.get(target);
        int k = tp.length;
        int[] card = new int[k];
        int[] latentPos = new int[k];
        for (int j = 0; j < k; j++) {
            card[j] = states.get(tp[j]).length;
            latentPos[j] = indexOf(latent, tp[j]);
        }
        double[][] table = cpt.get(target);
        int combos = 1;
        for (int j = 0; j < k; j++) if (latentPos[j] >= 0) combos *= card[j];
        double total = 0.0;
        int[] assign = new int[k];
        for (int c = 0; c < combos; c++) {
            int rem = c;
            double w = 1.0;
            for (int j = k - 1; j >= 0; j--) {
                if (latentPos[j] < 0) {
                    assign[j] = p; // the prior
                    continue;
                }
                assign[j] = rem % card[j];
                rem /= card[j];
                w *= dists[latentPos[j]][assign[j]];
            }
            if (w == 0.0) continue;
            int row = 0;
            for (int j = 0; j < k; j++) row = row * card[j] + assign[j];
            total += w * table[row][failIndex];
        }
        return total;
    }

    /** Posterior weights of the prior given the evidence (used only if it is missing). */
    private double[] priorWeights(int[] ev, int p) {
        int n = states.get(prior).length;
        double[] w = new double[n];
        if (p >= 0) {
            w[p] = 1.0;
            return w;
        }
        double sum = 0.0;
        for (int s = 0; s < n; s++) {
            double v = cpt.get(prior)[0][s];
            for (int i = 0; i < evidence.length; i++) {
                if (ev[i] >= 0 && parents.get(evidence[i]).length > 0) {
                    v *= cpt.get(evidence[i])[s][ev[i]];
                }
            }
            w[s] = v;
            sum += v;
        }
        for (int s = 0; s < n; s++) w[s] = sum > 0 ? w[s] / sum : 1.0 / n;
        return w;
    }

    private double[][] latentDistributions(int[] ev, int p) {
        double[][] d = new double[latent.length][];
        for (int i = 0; i < latent.length; i++) d[i] = latentDistribution(latent[i], ev, p);
        return d;
    }

    /** P(L_i = high | e) for every latent variable. */
    private double[] latentHigh(int[] ev, int p) {
        double[] w = priorWeights(ev, p);
        double[] out = new double[latent.length];
        for (int s = 0; s < w.length; s++) {
            if (w[s] == 0.0) continue;
            double[][] d = latentDistributions(ev, s);
            for (int i = 0; i < latent.length; i++) {
                int hi = indexOf(states.get(latent[i]), "high");
                out[i] += w[s] * d[i][hi >= 0 ? hi : d[i].length - 1];
            }
        }
        return out;
    }

    /** P(F = fail | e). */
    public double pFail(int[] ev, int p) {
        double[] w = priorWeights(ev, p);
        double total = 0.0;
        for (int s = 0; s < w.length; s++) {
            if (w[s] == 0.0) continue;
            total += w[s] * pFailGivenPrior(ev, s, latentDistributions(ev, s));
        }
        return total;
    }

    /**
     * Posterior plus its causal explanation: the latent condition whose P(high) the
     * telemetry raised most above its prior *for this device class*. Against a prior
     * pooled over classes, a condition whose class-specific prior is simply higher would
     * show a lift with no telemetry at all -- e.g. thermal stress on a device that has no
     * thermal sensor.
     */
    public Result infer(int[] ev, int p) {
        double pf = pFail(ev, p);
        double[] high = latentHigh(ev, p);
        double[] base = p >= 0 ? latentPriorHighByPrior[p] : latentPriorHigh;
        String best = "none";
        double bestScore = 1e-6;
        double bestLift = 0.0;
        for (int i = 0; i < latent.length; i++) {
            double lift = high[i] - base[i];
            Double weight = causeWeights.get(latent[i]);
            double score = lift * (weight == null ? 1.0 : weight);
            if (score > bestScore) {
                bestScore = score;
                best = latent[i];
                bestLift = lift;
            }
        }
        int missing = 0;
        for (int v : ev) if (v < 0) missing++;
        return new Result(pf, best, bestLift, missing);
    }

    public Result infer(NodeTelemetry t, String deviceClass) {
        return infer(evidenceFrom(t, deviceClass), priorStateFor(deviceClass));
    }

    // =====================================================================================
    // online learning
    // =====================================================================================

    /** E-step for one matured report: fractional counts for the target table. */
    public void observe(int[] ev, int p, boolean failed) {
        if (p < 0) return;
        double[][] d = latentDistributions(ev, p);
        String[] tp = parents.get(target);
        int k = tp.length;
        int[] card = new int[k];
        int[] latentPos = new int[k];
        for (int j = 0; j < k; j++) {
            card[j] = states.get(tp[j]).length;
            latentPos[j] = indexOf(latent, tp[j]);
        }
        int label = failed ? failIndex : 1 - failIndex;
        double[][] table = cpt.get(target);
        int combos = 1;
        for (int j = 0; j < k; j++) if (latentPos[j] >= 0) combos *= card[j];

        double[] weights = new double[combos];
        int[] rows = new int[combos];
        double total = 0.0;
        for (int c = 0; c < combos; c++) {
            int rem = c;
            double w = 1.0;
            int[] assign = new int[k];
            for (int j = k - 1; j >= 0; j--) {
                if (latentPos[j] < 0) {
                    assign[j] = p;
                    continue;
                }
                assign[j] = rem % card[j];
                rem /= card[j];
                w *= d[latentPos[j]][assign[j]];
            }
            int row = 0;
            for (int j = 0; j < k; j++) row = row * card[j] + assign[j];
            w *= table[row][label];
            weights[c] = w;
            rows[c] = row;
            total += w;
        }
        if (total <= 0) return;

        double[][] tc = counts(target);
        for (int c = 0; c < combos; c++) {
            double w = weights[c] / total;
            if (w != 0.0) tc[rows[c]][label] += w;
        }
        observations++;
    }

    private double[][] counts(String var) {
        double[][] c = counts.get(var);
        if (c == null) {
            double[][] t = cpt.get(var);
            c = new double[t.length][t[0].length];
            counts.put(var, c);
        }
        return c;
    }

    /** M-step: Dirichlet posterior centred on the offline tables. */
    public void refresh() {
        double maxChange = 0.0;
        for (Map.Entry<String, double[][]> e : counts.entrySet()) {
            double[][] base = baseCpt.get(e.getKey());
            double[][] cur = cpt.get(e.getKey());
            double[][] c = e.getValue();
            for (int r = 0; r < base.length; r++) {
                double norm = onlineConcentration;
                for (double v : c[r]) norm += v;
                for (int s = 0; s < base[r].length; s++) {
                    double updated = (onlineConcentration * base[r][s] + c[r][s]) / norm;
                    maxChange = Math.max(maxChange, Math.abs(updated - cur[r][s]));
                    cur[r][s] = updated;
                }
            }
        }
        lastMaxChange = maxChange;
        refreshes++;
    }

    public long observations() {
        return observations;
    }

    public int refreshes() {
        return refreshes;
    }

    public double lastMaxChange() {
        return lastMaxChange;
    }

    public String[] evidenceVariables() {
        return evidence.clone();
    }
}
