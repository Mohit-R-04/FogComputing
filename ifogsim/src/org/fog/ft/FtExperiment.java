package org.fog.ft;

/**
 * Command line for one closed-loop fault-tolerance run.
 *
 * <pre>
 *   FtExperiment --policy bbn --scenario 7 --seed 4101 --out run.json
 *                [--telemetry rows.csv] [--dropout none|mcar|mnar] [--rate 0.3]
 *                [--log quiet|json|pretty] [--sim-time 10000] [--no-online]
 *                [--bbn ../data/bbn_trained.json] [--ml ../data/models/ml_best.json]
 *                [--ftapa ../data/models/ftapa.json] [--cost-scale 1.0] [--no-summary]
 *                [--no-risk-placement] [--no-migration] [--no-replication]   (ablations)
 * </pre>
 */
public final class FtExperiment {

    private FtExperiment() {
    }

    public static void main(String[] args) throws Exception {
        RunConfig cfg = new RunConfig();
        cfg.spec = ScenarioSweep.get(0);
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--policy": cfg.policy = FtPolicy.parse(args[++i]); break;
                case "--scenario": cfg.spec = ScenarioSweep.get(Integer.parseInt(args[++i])); break;
                case "--seed": cfg.seed = Long.parseLong(args[++i]); break;
                case "--out": cfg.summaryJson = args[++i]; break;
                case "--telemetry": cfg.telemetryCsv = args[++i]; break;
                case "--dropout": cfg.dropout = RunContext.Dropout.valueOf(args[++i].toUpperCase()); break;
                case "--rate": cfg.dropoutRate = Double.parseDouble(args[++i]); break;
                case "--log": cfg.log = EventLog.Mode.valueOf(args[++i].toUpperCase()); break;
                case "--sim-time": cfg.simTime = Double.parseDouble(args[++i]); break;
                case "--no-online": cfg.onlineLearning = false; break;
                case "--bbn": cfg.bbnModel = args[++i]; break;
                case "--ml": cfg.mlModel = args[++i]; break;
                case "--ftapa": cfg.healthIndexModel = args[++i]; break;
                case "--cost-scale": cfg.migrationCostScale = Double.parseDouble(args[++i]); break;
                case "--no-summary": cfg.printSummary = false; break;
                case "--no-risk-placement": cfg.riskPlacement = false; break;
                case "--no-migration": cfg.proactiveMigration = false; break;
                case "--no-replication": cfg.selectiveReplication = false; break;
                default:
                    System.err.println("unknown argument " + a);
                    System.exit(2);
            }
        }
        MelbCBDFaultToleranceSim.run(cfg);
    }
}
