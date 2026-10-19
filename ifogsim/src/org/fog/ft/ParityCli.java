package org.fog.ft;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Scores rows of a CSV with the Java BBN engine and the exported ML model, so
 * {@code model/test_java_parity.py} can check that the simulator's predictors compute
 * exactly what the Python models compute.
 *
 * <pre>  ParityCli &lt;bbn.json&gt; &lt;ml.json&gt; &lt;in.csv&gt; &lt;out.csv&gt;  </pre>
 */
public final class ParityCli {

    private ParityCli() {
    }

    public static void main(String[] args) throws Exception {
        BbnEngine bbn = BbnEngine.load(args[0]);
        MlPredictor ml = MlPredictor.load(args[1]);
        String[] mlFeatures = ml.numericFeatures();
        BufferedReader in = new BufferedReader(new FileReader(args[2]));
        PrintWriter out = new PrintWriter(new FileWriter(args[3]));
        try {
            String[] header = in.readLine().split(",", -1);
            out.println("p_bbn_java,p_ml_java,cause_java");
            String line;
            while ((line = in.readLine()) != null) {
                String[] cells = line.split(",", -1);
                Map<String, String> row = new HashMap<String, String>();
                for (int i = 0; i < header.length && i < cells.length; i++) row.put(header[i], cells[i]);
                String cls = row.get("device_class");
                BbnEngine.Result r = bbn.infer(bbn.evidenceFromRow(row), bbn.priorStateFor(cls));
                double[] raw = new double[mlFeatures.length];
                for (int i = 0; i < mlFeatures.length; i++) {
                    String v = row.get(mlFeatures[i]);
                    raw[i] = (v == null || v.trim().isEmpty()) ? Double.NaN : Double.parseDouble(v);
                }
                out.println(String.format(Locale.ROOT, "%.15g,%.15g,%s", r.pFail,
                        ml.predictRaw(raw, cls), r.cause));
            }
        } finally {
            in.close();
            out.close();
        }
    }
}
