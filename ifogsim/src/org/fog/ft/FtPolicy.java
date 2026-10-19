package org.fog.ft;

/**
 * Fault-tolerance policies compared in the closed-loop experiment.
 *
 * <p>The four baselines are the ones the Review 1 and Review 2 documents commit to, plus
 * an ML-driven controller and a clairvoyant oracle:
 * <ul>
 *   <li>{@link #NO_FT}: iFogSim edgeward placement with no fault tolerance -- a user
 *       whose node dies reconnects after link loss and its analyser cold-starts;</li>
 *   <li>{@link #CLOUD_ONLY}: every analyser in the cloud -- immune to state loss, but
 *       every loop crosses the WAN;</li>
 *   <li>{@link #REACTIVE}: heartbeat detection plus checkpoint/primary-backup recovery
 *       for every session (Rajab and Younis 2021);</li>
 *   <li>{@link #THRESHOLD}: proactive migration when a composite health index crosses a
 *       fixed, tuned threshold, with a pre-selected backup node for every session
 *       (FTAPA, Gupta and Singh 2025) -- the baseline that separates probabilistic
 *       inference from merely acting early;</li>
 *   <li>{@link #ML}: the proposed controller (Eqs. 2-4) driven by the best classical
 *       machine-learning model instead of the BBN -- isolates the predictor;</li>
 *   <li>{@link #BBN}: the proposed framework -- BBN posterior (N1), risk-aware placement
 *       and expected-loss migration (N2), criticality-aware selective replication (N3),
 *       online CPT adaptation;</li>
 *   <li>{@link #ORACLE}: the proposed controller driven by the best possible predictor,
 *       which knows each node's hidden health and the wear law (but not future random
 *       draws) -- the most any predictor could achieve with this controller.</li>
 * </ul>
 */
public enum FtPolicy {
    NO_FT("no-ft", "No fault tolerance"),
    CLOUD_ONLY("cloud-only", "Cloud-only placement"),
    REACTIVE("reactive", "Reactive checkpoint + backup (Rajab 2021)"),
    THRESHOLD("threshold", "Threshold migration, FTAPA (Gupta 2025)"),
    ML("ml", "Expected-loss controller + ML predictor"),
    BBN("bbn", "Proposed BBN framework"),
    ORACLE("oracle", "Oracle predictor (upper bound)");

    public final String key;
    public final String label;

    FtPolicy(String key, String label) {
        this.key = key;
        this.label = label;
    }

    public static FtPolicy parse(String s) {
        for (FtPolicy p : values()) {
            if (p.key.equalsIgnoreCase(s) || p.name().equalsIgnoreCase(s)) return p;
        }
        throw new IllegalArgumentException("unknown policy '" + s + "'");
    }

    /** Acts before failure. */
    public boolean proactive() {
        return this == THRESHOLD || this == ML || this == BBN || this == ORACLE;
    }

    /** Uses the decision-theoretic rules of Eqs. (2)-(4). */
    public boolean expectedLoss() {
        return this == ML || this == BBN || this == ORACLE;
    }

    /** Detects failures with application heartbeats (otherwise only by link loss). */
    public boolean heartbeat() {
        return this != NO_FT && this != CLOUD_ONLY;
    }

    /** Analysis runs in the cloud; fog nodes only forward. */
    public boolean cloudHosted() {
        return this == CLOUD_ONLY;
    }
}
