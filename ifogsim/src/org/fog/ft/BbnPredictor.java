package org.fog.ft;

/**
 * Novelty 1 in the control loop: the BBN posterior for each node, with online CPT
 * refinement from outcomes the controller has actually observed.
 */
public final class BbnPredictor implements RiskPredictor {

    /** Re-estimate the tables every this many epochs. */
    private static final int REFRESH_EVERY_EPOCHS = 5;

    private final BbnEngine engine;
    private final boolean online;
    private int epochs = 0;

    public BbnPredictor(BbnEngine engine, boolean online) {
        this.engine = engine;
        this.online = online;
    }

    public BbnEngine engine() {
        return engine;
    }

    @Override
    public String name() {
        return online ? "BBN (online)" : "BBN";
    }

    @Override
    public Prediction predict(InstrumentedFogDevice node, NodeTelemetry t, double now) {
        BbnEngine.Result r = engine.infer(t, node.getDeviceClass());
        return new Prediction(r.pFail, r.cause, r.missing);
    }

    @Override
    public void observeOutcome(NodeTelemetry t, boolean failedWithinHorizon) {
        if (!online || t == null) return;
        engine.observe(engine.evidenceFrom(t), engine.priorStateFor(t.deviceClass), failedWithinHorizon);
    }

    @Override
    public void endEpoch() {
        if (!online) return;
        epochs++;
        if (epochs % REFRESH_EVERY_EPOCHS == 0 && engine.observations() > 0) {
            engine.refresh();
        }
    }

    @Override
    public String status() {
        if (!online) return "";
        return String.format(java.util.Locale.ROOT, "online CPT updates: %d obs, %d refreshes, max |dP| %.4f",
                engine.observations(), engine.refreshes(), engine.lastMaxChange());
    }
}
