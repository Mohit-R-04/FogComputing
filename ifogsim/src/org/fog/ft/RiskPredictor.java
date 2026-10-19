package org.fog.ft;

/**
 * Source of P(node fails within the horizon | what the control plane observed).
 *
 * <p>Every policy that acts on risk is given one of these; the controller logic is the
 * same whichever predictor is plugged in, so a comparison between predictors isolates
 * the predictor.
 */
public interface RiskPredictor {

    /** Prediction for one node at one decision epoch. */
    final class Prediction {
        public final double pFail;
        /** Principal latent cause (BBN only), or "-". */
        public final String cause;
        /** Telemetry channels that did not report and were marginalised or imputed. */
        public final int missingChannels;

        public Prediction(double pFail, String cause, int missingChannels) {
            this.pFail = pFail;
            this.cause = cause;
            this.missingChannels = missingChannels;
        }
    }

    String name();

    /**
     * @param node the node
     * @param t    its latest report, or null if it reported nothing this epoch
     * @param now  current simulation time
     */
    Prediction predict(InstrumentedFogDevice node, NodeTelemetry t, double now);

    /** Outcome feedback once a report's horizon has elapsed (online learning hook). */
    default void observeOutcome(NodeTelemetry t, boolean failedWithinHorizon) {
    }

    /** Called once per epoch after outcomes were observed. */
    default void endEpoch() {
    }

    /** Short status for the event log, e.g. online-learning progress. */
    default String status() {
        return "";
    }
}
