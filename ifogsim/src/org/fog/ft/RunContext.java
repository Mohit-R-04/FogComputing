package org.fog.ft;

/**
 * Settings and shared state of the current simulation run.
 *
 * <p>CloudSim and iFogSim keep their own state in statics and every run uses its own
 * JVM, so a static context is the honest way to reach run settings from device
 * callbacks. {@link #reset} is called once at the start of each run.
 */
public final class RunContext {

    /** How telemetry channels go missing in the robustness experiments. */
    public enum Dropout {
        /** Only naturally unavailable channels are missing. */
        NONE,
        /** Each channel independently missing with the configured rate. */
        MCAR,
        /** Missing more often on degrading nodes: "a channel falls silent exactly when
         *  its node is in trouble" (Review 1 brief): the rate is scaled by 1 + damage. */
        MNAR
    }

    private static long seed = 20260912L;
    private static int scenarioId = 0;
    private static FaultInjector injector = new FaultInjector(seed, scenarioId);
    private static Dropout dropout = Dropout.NONE;
    private static double dropoutRate = 0.0;
    private static double simTime = 10000;

    private RunContext() {
    }

    public static void reset(long runSeed, int scenario, Dropout mode, double rate, double runTime) {
        seed = runSeed;
        scenarioId = scenario;
        injector = new FaultInjector(runSeed, scenario);
        dropout = mode;
        dropoutRate = rate;
        simTime = runTime;
    }

    public static long seed() {
        return seed;
    }

    public static int scenarioId() {
        return scenarioId;
    }

    public static FaultInjector injector() {
        return injector;
    }

    public static Dropout dropout() {
        return dropout;
    }

    public static double dropoutRate() {
        return dropoutRate;
    }

    public static double simTime() {
        return simTime;
    }
}
