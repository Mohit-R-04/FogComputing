package org.fog.ft;

/**
 * Service classes of the users (patients) of the cardiac-monitoring application.
 *
 * <p>The Review documents set criticality per application class ("health monitoring
 * highest ... best-effort lowest"). Within the single cardiac application simulated
 * here, the same idea appears as triage levels: an ICU patient's arrhythmia alarm is
 * worth far more than a wellness tracker's. Each class carries the criticality weight
 * kappa used by the expected-loss rules (Eqs. 3 and 4 of the paper), the size of the
 * analyser state that a migration or a restore has to move, and the placement-utility
 * weights (alpha latency, beta energy, gamma risk) of Eq. 2.
 */
public enum Criticality {
    CRITICAL("critical", 1.00, 300.0, 0.30, 0.10, 0.60),
    STANDARD("standard", 0.50, 150.0, 0.45, 0.15, 0.40),
    BEST_EFFORT("best-effort", 0.15, 50.0, 0.60, 0.25, 0.15);

    public final String label;
    /** Criticality weight kappa in [0, 1]. */
    public final double kappa;
    /** Analyser state that must be transferred on migration or restore, in KB. */
    public final double stateKb;
    /** Placement-utility weights of Eq. (2). */
    public final double alpha;
    public final double beta;
    public final double gamma;

    Criticality(String label, double kappa, double stateKb,
                double alpha, double beta, double gamma) {
        this.label = label;
        this.kappa = kappa;
        this.stateKb = stateKb;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    /** Deterministic class assignment: 25% critical, 45% standard, 30% best-effort. */
    public static Criticality assign(long seed, String userName) {
        double u = FtRandom.uniform(FtRandom.key(seed, 0, FtRandom.hash(userName), 0,
                FtRandom.S_CRITICALITY));
        if (u < 0.25) return CRITICAL;
        if (u < 0.70) return STANDARD;
        return BEST_EFFORT;
    }
}
