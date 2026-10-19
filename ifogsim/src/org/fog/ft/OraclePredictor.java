package org.fog.ft;

/**
 * The best possible predictor, as an upper bound: it outputs the probability that the
 * node fails within the horizon given what no real predictor can see -- the node's
 * hidden health and the exact wear law ({@link FaultInjector#failureProbability}). It
 * is not clairvoyant: sudden failures are random draws it cannot know in advance. It
 * drives the same expected-loss controller as the BBN and ML policies, so the gap
 * between them and this policy is the most a better predictor could still gain.
 */
public final class OraclePredictor implements RiskPredictor {

    @Override
    public String name() {
        return "Oracle (hidden state)";
    }

    @Override
    public Prediction predict(InstrumentedFogDevice node, NodeTelemetry t, double now) {
        double p = RunContext.injector().failureProbability(node.getName(), now,
                FaultInjector.HORIZON_DELTA_T);
        return new Prediction(p, "-", 0);
    }
}
