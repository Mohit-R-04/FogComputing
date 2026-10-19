package org.fog.ft;

import java.io.FileReader;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * FTAPA-style composite health index (Gupta and Singh 2025): a fixed weighted mix of
 * normalised CPU utilisation, queue depth, energy depletion and temperature, compared
 * against a single tuned fault threshold.
 *
 * <p>The returned "probability" is {@code 1 - HI}; it is a score, not a calibrated
 * probability, and the THRESHOLD policy only compares it with {@code 1 - theta}. When a
 * channel does not report, its weight is dropped and the rest renormalised -- the most
 * charitable treatment available to a threshold rule. The threshold is chosen in Python
 * on training scenarios only (max F1), giving the baseline its best possible setting.
 */
public final class HealthIndexPredictor implements RiskPredictor {

    private double wCpu = 0.40;
    private double wQueue = 0.30;
    private double wEnergy = 0.15;
    private double wTemp = 0.15;
    private double queueSaturation = 120.0;
    private double tempSafe = 45.0;
    private double tempSpan = 40.0;
    private double threshold = 0.5;

    public static HealthIndexPredictor load(String path) throws Exception {
        JSONObject root;
        FileReader r = new FileReader(path);
        try {
            root = (JSONObject) new JSONParser().parse(r);
        } finally {
            r.close();
        }
        HealthIndexPredictor h = new HealthIndexPredictor();
        JSONObject w = (JSONObject) root.get("weights");
        h.wCpu = num(w, "cpu_util", h.wCpu);
        h.wQueue = num(w, "queue_depth", h.wQueue);
        h.wEnergy = num(w, "residual_energy", h.wEnergy);
        h.wTemp = num(w, "temperature", h.wTemp);
        JSONObject n = (JSONObject) root.get("norm");
        h.queueSaturation = num(n, "queue_saturation", h.queueSaturation);
        h.tempSafe = num(n, "temp_safe", h.tempSafe);
        h.tempSpan = num(n, "temp_span", h.tempSpan);
        h.threshold = ((Number) root.get("threshold")).doubleValue();
        return h;
    }

    private static double num(JSONObject o, String k, double fallback) {
        Object v = o == null ? null : o.get(k);
        return v == null ? fallback : ((Number) v).doubleValue();
    }

    private static double clamp01(double v) {
        return v < 0 ? 0 : (v > 1 ? 1 : v);
    }

    /** Health index in [0, 1]; NaN if no contributing channel reported. */
    public double healthIndex(NodeTelemetry t) {
        if (t == null) return Double.NaN;
        double num = 0.0;
        double den = 0.0;
        if (!Double.isNaN(t.cpuUtil)) { num += wCpu * clamp01(t.cpuUtil); den += wCpu; }
        if (!Double.isNaN(t.queueDepth)) { num += wQueue * clamp01(t.queueDepth / queueSaturation); den += wQueue; }
        if (!Double.isNaN(t.residualEnergy)) { num += wEnergy * clamp01(1.0 - t.residualEnergy); den += wEnergy; }
        if (!Double.isNaN(t.temperature)) { num += wTemp * clamp01((t.temperature - tempSafe) / tempSpan); den += wTemp; }
        return den == 0.0 ? Double.NaN : 1.0 - num / den;
    }

    /** The tuned fault threshold theta: migrate when HI < theta. */
    public double threshold() {
        return threshold;
    }

    @Override
    public String name() {
        return "Health-index threshold";
    }

    @Override
    public Prediction predict(InstrumentedFogDevice node, NodeTelemetry t, double now) {
        double hi = healthIndex(t);
        return new Prediction(Double.isNaN(hi) ? 0.0 : 1.0 - hi, "-", 0);
    }
}
