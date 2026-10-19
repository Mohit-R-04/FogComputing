package org.fog.ft;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

import org.json.simple.JSONObject;

/**
 * Event stream of the fault-tolerance controller.
 *
 * <p>{@link Mode#JSON} prints one {@code @FT {...}} line per event, which the terminal
 * demonstration ({@code model/demo.py}) parses and renders; {@link Mode#PRETTY} prints
 * the same events for a human when the simulator is run on its own; {@link Mode#QUIET}
 * prints nothing (batch experiments).
 */
public final class EventLog {

    public enum Mode { QUIET, JSON, PRETTY }

    private static Mode mode = Mode.QUIET;

    private EventLog() {
    }

    public static void setMode(Mode m) {
        mode = m;
    }

    public static boolean enabled() {
        return mode != Mode.QUIET;
    }

    /** Emits an event; {@code kv} alternates keys and values. */
    @SuppressWarnings("unchecked")
    public static void emit(String type, double t, Object... kv) {
        if (mode == Mode.QUIET) return;
        Map<String, Object> fields = new LinkedHashMap<String, Object>();
        for (int i = 0; i + 1 < kv.length; i += 2) {
            fields.put(String.valueOf(kv[i]), kv[i + 1]);
        }
        if (mode == Mode.JSON) {
            JSONObject o = new JSONObject();
            o.put("type", type);
            o.put("t", Math.round(t * 10.0) / 10.0);
            for (Map.Entry<String, Object> e : fields.entrySet()) {
                Object v = e.getValue();
                if (v instanceof Double) {
                    double d = (Double) v;
                    v = Double.isNaN(d) || Double.isInfinite(d) ? null : Math.round(d * 1e5) / 1e5;
                }
                o.put(e.getKey(), v);
            }
            System.out.println("@FT " + o.toJSONString());
        } else {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(Locale.ROOT, "[t=%7.1f] %-9s", t, type));
            for (Map.Entry<String, Object> e : fields.entrySet()) {
                Object v = e.getValue();
                String s = v instanceof Double ? String.format(Locale.ROOT, "%.4g", (Double) v)
                        : String.valueOf(v);
                sb.append(' ').append(e.getKey()).append('=').append(s);
            }
            System.out.println(sb);
        }
        System.out.flush();
    }
}
