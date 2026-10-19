package org.fog.ft;

import java.io.FileReader;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * A classical machine-learning model exported from scikit-learn, evaluated in Java.
 *
 * <p>Supports the three families of the comparison: random forests (mean of tree leaf
 * probabilities), gradient-boosted trees (sigmoid of baseline plus leaf values) and
 * logistic regression. Preprocessing is the explicit feature specification of
 * {@code model/features.py}: median imputation, a missing-value indicator per numeric
 * feature, a one-hot device class, and (for linear models) standardisation. A missing
 * channel is therefore <i>imputed</i>, which is the behaviour the BBN is compared against.
 * Parity with scikit-learn's {@code predict_proba} is checked by
 * {@code model/test_java_parity.py}.
 */
public final class MlPredictor implements RiskPredictor {

    private String name;
    private String kind;
    private String[] numeric;
    private double[] medians;
    private String[] classes;
    private double[] scaleMean;
    private double[] scaleStd;
    // trees
    private int[][] feature;
    private double[][] threshold;
    private int[][] left;
    private int[][] right;
    private double[][] value;
    private double baseline;
    // linear
    private double[] coef;
    private double intercept;
    // multilayer perceptron: weights[layer][out][in], bias[layer][out]
    private double[][][] mlpWeights;
    private double[][] mlpBias;
    private String activation = "relu";

    private MlPredictor() {
    }

    public static MlPredictor load(String path) throws Exception {
        JSONObject root;
        FileReader r = new FileReader(path);
        try {
            root = (JSONObject) new JSONParser().parse(r);
        } finally {
            r.close();
        }
        MlPredictor m = new MlPredictor();
        m.name = (String) root.get("name");
        m.kind = (String) root.get("kind");
        JSONObject f = (JSONObject) root.get("features");
        m.numeric = strings((JSONArray) f.get("numeric"));
        m.medians = doubles((JSONArray) f.get("medians"));
        m.classes = strings((JSONArray) f.get("classes"));
        if (f.get("scale_mean") != null) {
            m.scaleMean = doubles((JSONArray) f.get("scale_mean"));
            m.scaleStd = doubles((JSONArray) f.get("scale_std"));
        }
        if ("linear".equals(m.kind)) {
            m.coef = doubles((JSONArray) root.get("coef"));
            m.intercept = ((Number) root.get("intercept")).doubleValue();
        } else if ("mlp".equals(m.kind)) {
            JSONArray layers = (JSONArray) root.get("layers");
            m.mlpWeights = new double[layers.size()][][];
            m.mlpBias = new double[layers.size()][];
            for (int l = 0; l < layers.size(); l++) {
                JSONObject layer = (JSONObject) layers.get(l);
                JSONArray w = (JSONArray) layer.get("weights");
                m.mlpWeights[l] = new double[w.size()][];
                for (int o = 0; o < w.size(); o++) m.mlpWeights[l][o] = doubles((JSONArray) w.get(o));
                m.mlpBias[l] = doubles((JSONArray) layer.get("bias"));
            }
            if (root.get("activation") != null) m.activation = (String) root.get("activation");
        } else {
            JSONArray trees = (JSONArray) root.get("trees");
            int n = trees.size();
            m.feature = new int[n][];
            m.threshold = new double[n][];
            m.left = new int[n][];
            m.right = new int[n][];
            m.value = new double[n][];
            for (int i = 0; i < n; i++) {
                JSONObject t = (JSONObject) trees.get(i);
                m.feature[i] = ints((JSONArray) t.get("feature"));
                m.threshold[i] = doubles((JSONArray) t.get("threshold"));
                m.left[i] = ints((JSONArray) t.get("left"));
                m.right[i] = ints((JSONArray) t.get("right"));
                m.value[i] = doubles((JSONArray) t.get("value"));
            }
            Object b = root.get("baseline");
            m.baseline = b == null ? 0.0 : ((Number) b).doubleValue();
        }
        return m;
    }

    private static String[] strings(JSONArray a) {
        String[] o = new String[a.size()];
        for (int i = 0; i < o.length; i++) o[i] = (String) a.get(i);
        return o;
    }

    private static double[] doubles(JSONArray a) {
        double[] o = new double[a.size()];
        for (int i = 0; i < o.length; i++) o[i] = ((Number) a.get(i)).doubleValue();
        return o;
    }

    private static int[] ints(JSONArray a) {
        int[] o = new int[a.size()];
        for (int i = 0; i < o.length; i++) o[i] = ((Number) a.get(i)).intValue();
        return o;
    }

    /** Feature vector exactly as features.FeatureSpec.transform builds it. */
    public double[] features(double[] raw, String deviceClass) {
        int n = numeric.length;
        double[] x = new double[2 * n + classes.length];
        for (int i = 0; i < n; i++) {
            boolean missing = Double.isNaN(raw[i]);
            x[i] = missing ? medians[i] : raw[i];
            x[n + i] = missing ? 1.0 : 0.0;
        }
        for (int c = 0; c < classes.length; c++) {
            x[2 * n + c] = classes[c].equals(deviceClass) ? 1.0 : 0.0;
        }
        if (scaleMean != null) {
            for (int i = 0; i < x.length; i++) {
                x[i] = (x[i] - scaleMean[i]) / (scaleStd[i] == 0.0 ? 1.0 : scaleStd[i]);
            }
        }
        return x;
    }

    public double predictRaw(double[] raw, String deviceClass) {
        double[] x = features(raw, deviceClass);
        if ("linear".equals(kind)) {
            double z = intercept;
            for (int i = 0; i < coef.length; i++) z += coef[i] * x[i];
            return 1.0 / (1.0 + Math.exp(-z));
        }
        if ("mlp".equals(kind)) {
            double[] h = x;
            for (int l = 0; l < mlpWeights.length; l++) {
                double[] next = new double[mlpWeights[l].length];
                for (int o = 0; o < next.length; o++) {
                    double z = mlpBias[l][o];
                    for (int i = 0; i < h.length; i++) z += mlpWeights[l][o][i] * h[i];
                    boolean output = l == mlpWeights.length - 1;
                    next[o] = output ? z : activate(z);
                }
                h = next;
            }
            return 1.0 / (1.0 + Math.exp(-h[0]));
        }
        double sum = 0.0;
        for (int t = 0; t < feature.length; t++) {
            int node = 0;
            while (left[t][node] >= 0) {
                node = x[feature[t][node]] <= threshold[t][node] ? left[t][node] : right[t][node];
            }
            sum += value[t][node];
        }
        if ("forest".equals(kind)) {
            return sum / feature.length;
        }
        double z = baseline + sum;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private double activate(double z) {
        switch (activation) {
            case "tanh": return Math.tanh(z);
            case "logistic": return 1.0 / (1.0 + Math.exp(-z));
            case "identity": return z;
            default: return Math.max(0.0, z);
        }
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public Prediction predict(InstrumentedFogDevice node, NodeTelemetry t, double now) {
        double[] raw = new double[numeric.length];
        int missing = 0;
        for (int i = 0; i < numeric.length; i++) {
            raw[i] = t == null ? Double.NaN : t.value(numeric[i]);
            if (Double.isNaN(raw[i])) missing++;
        }
        return new Prediction(predictRaw(raw, node.getDeviceClass()), "-", missing);
    }

    public String[] numericFeatures() {
        return numeric.clone();
    }
}
