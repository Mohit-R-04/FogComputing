package org.fog.ft;

/** Everything one simulation run needs. */
public final class RunConfig {
    public ScenarioSpec spec;
    public long seed = 20260912L;
    public FtPolicy policy = FtPolicy.NO_FT;
    public double simTime = ScenarioSweep.SIM_TIME;
    /** Labelled telemetry output (null: do not keep rows). */
    public String telemetryCsv;
    /** Run summary output (null: do not write). */
    public String summaryJson;
    public RunContext.Dropout dropout = RunContext.Dropout.NONE;
    public double dropoutRate = 0.0;
    public String bbnModel = "../data/bbn_trained.json";
    public String mlModel = "../data/models/ml_best.json";
    public String healthIndexModel = "../data/models/ftapa.json";
    public boolean onlineLearning = true;
    public EventLog.Mode log = EventLog.Mode.QUIET;
    public double migrationCostScale = 1.0;
    /** Print the human-readable run summary to stdout. */
    public boolean printSummary = true;

    // ---- ablation switches (expected-loss policies only) ---------------------------
    /** Risk term of the placement utility and risk-based exclusions (Novelty 2). */
    public boolean riskPlacement = true;
    /** Expected-loss proactive migration (Novelty 2). */
    public boolean proactiveMigration = true;
    /** Criticality-aware selective replication (Novelty 3). */
    public boolean selectiveReplication = true;
}
