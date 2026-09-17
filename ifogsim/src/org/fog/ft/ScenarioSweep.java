package org.fog.ft;

import java.util.ArrayList;
import java.util.List;

/** Twelve deterministic three-tier operating regimes for the iFogSim sweep. */
public class ScenarioSweep {
    private static final int SIM_TIME = 5000;

    private static final double[] CLOUD_POWER = {16 * 103.0, 16 * 83.25};
    private static final double[] FOG_POWER_FLAT = {107.339, 83.4333};
    private static final double[] FOG_POWER_WIDE = {120.0, 45.0};
    private static final double[] EDGE_POWER_STD = {3000.0, 300.0};
    private static final double[] EDGE_POWER_HOT = {4200.0, 350.0};

    private static List<ScenarioSpec> scenarios;

    public static synchronized List<ScenarioSpec> all() {
        if (scenarios != null) return scenarios;
        List<ScenarioSpec> s = new ArrayList<ScenarioSpec>();
        // id label              cloudMips cloudRam fogMips fogRam edgeMips edgeRam
        // power                         edges sensors interval clientMips clientRam
        // analyserMips analyserRam tupleMI fogLatency fogBw simTime
        s.add(new ScenarioSpec(0, "baseline",       44800,40000,2800,4000,1000,1000,
                CLOUD_POWER,FOG_POWER_FLAT,EDGE_POWER_STD,40,1,10,300,10,1500,1200,15000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(1, "idle-fleet",     44800,40000,2800,4000,1000,1000,
                CLOUD_POWER,FOG_POWER_FLAT,EDGE_POWER_STD,20,1,100,300,10,1500,1200,15000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(2, "chatty-sensors", 44800,40000,2800,4000,1000,1000,
                CLOUD_POWER,FOG_POWER_FLAT,EDGE_POWER_STD,40,2,10,300,10,1500,1200,15000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(3, "thin-fog",        44800,40000,1400,2000,1000,1000,
                CLOUD_POWER,FOG_POWER_WIDE,EDGE_POWER_STD,40,1,10,300,10,1200,1000,15000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(4, "starved-fog",     44800,40000,700,1000,1000,1000,
                CLOUD_POWER,FOG_POWER_WIDE,EDGE_POWER_STD,40,1,10,300,10,600,800,12000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(5, "fat-fog",         44800,40000,5600,8000,1000,1000,
                CLOUD_POWER,FOG_POWER_WIDE,EDGE_POWER_STD,40,2,10,300,10,3000,2400,20000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(6, "weak-edge",       44800,40000,2800,4000,500,600,
                CLOUD_POWER,FOG_POWER_FLAT,EDGE_POWER_HOT,40,1,10,200,8,1500,1200,15000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(7, "dense-edge",      44800,40000,2800,4000,1000,1000,
                CLOUD_POWER,FOG_POWER_FLAT,EDGE_POWER_STD,80,1,20,300,10,1500,1200,15000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(8, "slow-backhaul",   44800,40000,2800,4000,1000,1000,
                CLOUD_POWER,FOG_POWER_FLAT,EDGE_POWER_STD,40,1,10,300,10,1500,1200,15000,250,2000,SIM_TIME));
        s.add(new ScenarioSpec(9, "fog-heavy",       44800,40000,2800,4000,1000,1000,
                CLOUD_POWER,FOG_POWER_WIDE,EDGE_POWER_STD,60,2,15,300,10,2200,1800,18000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(10,"edge-heavy",      44800,40000,2800,4000,1000,1000,
                CLOUD_POWER,FOG_POWER_FLAT,EDGE_POWER_HOT,40,1,10,800,400,1500,1200,15000,100,10000,SIM_TIME));
        s.add(new ScenarioSpec(11,"saturated",       44800,40000,1400,2000,700,800,
                CLOUD_POWER,FOG_POWER_WIDE,EDGE_POWER_HOT,40,1,20,300,15,1000,900,14000,150,5000,SIM_TIME));
        scenarios = s;
        return scenarios;
    }

    public static ScenarioSpec get(int index) { return all().get(index); }
    public static int count() { return all().size(); }

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("--list")) {
            System.out.println("scenarios in the sweep: " + count());
            for (ScenarioSpec spec : all()) System.out.println("  " + spec);
            return;
        }
        if (args.length < 3) {
            System.err.println("usage: ScenarioSweep <scenarioIndex> <seed> <outputCsv>");
            System.exit(2);
        }
        MelbCBDFaultToleranceSim.runScenario(
                get(Integer.parseInt(args[0])), Long.parseLong(args[1]), args[2]);
    }
}
