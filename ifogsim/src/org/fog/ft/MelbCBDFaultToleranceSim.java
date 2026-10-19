package org.fog.ft;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.cloudbus.cloudsim.sdn.overbooking.BwProvisionerOverbooking;
import org.cloudbus.cloudsim.sdn.overbooking.PeProvisionerOverbooking;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.Application;
import org.fog.entities.Actuator;
import org.fog.entities.FogBroker;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.mobilitydata.DataParser;
import org.fog.mobilitydata.Location;
import org.fog.placement.ModuleMapping;
import org.fog.placement.ModulePlacementMapping;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.Config;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;

/**
 * Builds and runs one simulation: one scenario, one seed, one fault-tolerance policy.
 *
 * <p>The topology has exactly three tiers. The Melbourne CBD edge-resource set supplies
 * one cloud and the 118 gateway resources, which are the fog nodes (proxy resources are
 * not instantiated); battery-powered user devices are the edge tier. Every node is an
 * {@link InstrumentedFogDevice}. The workload is a cardiac-monitoring application, the
 * latency-critical use case the paper cites: vitals are filtered on the user's device,
 * analysed by an ecg_analyser on the user's serving node, and the verdict raises the
 * alarm locally; summaries are archived in the cloud.
 *
 * <p>Every fog node (and the cloud) hosts an ecg_analyser instance, so wherever a user
 * is served -- after a handover, a migration or a recovery -- its analysis keeps working.
 * Under {@link FtPolicy#CLOUD_ONLY} the analyser exists only in the cloud.
 *
 * <p>The same code path produces the training telemetry ({@link FtPolicy#NO_FT}, with
 * the CSV kept) and every closed-loop experiment, so the model is evaluated on exactly
 * the system it was trained on.
 *
 * <p><b>Run from {@code ifogsim/}.</b> {@link DataParser} resolves {@code ./dataset/...}
 * relative to the working directory.
 */
public class MelbCBDFaultToleranceSim {

    private static final String APP_ID = "cardiac_monitoring";
    private static final double SENSOR_LATENCY = 1.0;
    /** End-to-end deadline of a critical user's alarm loop, in ms. */
    public static final double CRITICAL_LOOP_DEADLINE = 150.0;
    /** Temperature rise at full utilisation: passive fog enclosures vs a cooled cloud. */
    private static final double FOG_THERMAL_RISE_C = 45.0;
    private static final double CLOUD_THERMAL_RISE_C = 12.0;
    /**
     * Scale on the analyser's CPU cost per tuple. At the scenario table's raw values one
     * user needed 48% of a gateway in the baseline and 154% in "starved-fog", and forty
     * users needed 120% of the cloud: every queue grew without bound and even a run with
     * no failures delivered under half of its alarms. Scaled, a gateway serves several
     * users as it would in deployment, hot spots still saturate, and the twelve regimes
     * keep their relative loads.
     */
    public static final double ANALYSIS_COST_SCALE = 0.25;

    private MelbCBDFaultToleranceSim() {
    }

    /** The objects of one built topology. */
    private static final class Built {
        InstrumentedFogDevice cloud;
        final List<InstrumentedFogDevice> fogs = new ArrayList<InstrumentedFogDevice>();
        final List<InstrumentedFogDevice> edges = new ArrayList<InstrumentedFogDevice>();
        final List<FogDevice> all = new ArrayList<FogDevice>();
        final List<Sensor> sensors = new ArrayList<Sensor>();
        final List<Actuator> actuators = new ArrayList<Actuator>();
        final Map<Integer, Location> fogLocations = new HashMap<Integer, Location>();
    }

    /** Runs one configuration end to end; returns the controller for its metrics. */
    public static FaultToleranceController run(RunConfig cfg) throws Exception {
        ScenarioSpec spec = cfg.spec;
        RunContext.reset(cfg.seed, spec.id, cfg.dropout, cfg.dropoutRate, cfg.simTime);
        SessionRegistry.reset();
        TelemetryCollector.reset(cfg.telemetryCsv != null);
        EventLog.setMode(cfg.log);
        FtCostModel.migrationCostScale = cfg.migrationCostScale;

        Log.disable();
        Config.MAX_SIMULATION_TIME = (int) Math.round(cfg.simTime);
        CloudSim.init(1, Calendar.getInstance(), false);

        FogBroker broker = new FogBroker("broker");
        Application application = createApplication(APP_ID, broker.getId(), spec, cfg.seed);
        application.setUserId(broker.getId());

        Built b = createTopology(broker.getId(), APP_ID, spec, cfg.seed);
        RiskPredictor predictor = buildPredictor(cfg);
        HealthIndexPredictor healthIndex = cfg.policy == FtPolicy.THRESHOLD
                ? HealthIndexPredictor.load(cfg.healthIndexModel) : null;

        QuietController master = new QuietController("master-controller", b.all, b.sensors, b.actuators);
        Topology topology = new Topology(b.cloud, b.fogs, b.edges, b.fogLocations,
                spec.fogUplinkLatency + 20.0);
        MobilityDriver mobility = new MobilityDriver("mobility-driver", b.edges, topology, cfg.seed);
        FaultToleranceController ft = new FaultToleranceController("ft-controller", cfg.policy,
                predictor, healthIndex, topology, spec, cfg.simTime);
        ft.setAblation(cfg.riskPlacement, cfg.proactiveMigration, cfg.selectiveReplication);
        mobility.setController(ft);
        mobility.associateInitially();
        for (InstrumentedFogDevice d : allInstrumented(b)) d.markTopologyBuilt();

        ModuleMapping mapping = ModuleMapping.createModuleMapping();
        mapping.addModuleToDevice("archival_store", "cloud");
        mapping.addModuleToDevice("ecg_analyser", "cloud");
        if (!cfg.policy.cloudHosted()) {
            for (InstrumentedFogDevice f : b.fogs) mapping.addModuleToDevice("ecg_analyser", f.getName());
        }
        for (InstrumentedFogDevice e : b.edges) mapping.addModuleToDevice("client", e.getName());
        master.submitApplication(application, 0, new ModulePlacementMapping(b.all, application, mapping));

        TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());
        long wallStart = System.currentTimeMillis();
        EventLog.emit("START", 0.0, "scenario", spec.id, "label", spec.label, "policy", cfg.policy.key,
                "predictor", predictor == null ? "-" : predictor.name(), "seed", cfg.seed,
                "fog", b.fogs.size(), "edge", b.edges.size(), "sim_time", cfg.simTime);

        CloudSim.startSimulation();

        double end = QuietController.endTime();
        if (Double.isNaN(end)) end = cfg.simTime;
        double wall = (System.currentTimeMillis() - wallStart) / 1000.0;

        TelemetryCollector collector = TelemetryCollector.getInstance();
        if (cfg.telemetryCsv != null) collector.write(cfg.telemetryCsv);
        String predictorName = predictor == null ? "-" : predictor.name();
        if (cfg.summaryJson != null) ft.writeSummary(cfg.summaryJson, end, predictorName);
        if (cfg.printSummary) printSummary(cfg, b, ft, end, wall, predictorName);
        EventLog.emit("END", end, "summary", cfg.summaryJson == null ? "" : cfg.summaryJson,
                "wall_seconds", wall);
        return ft;
    }

    private static List<InstrumentedFogDevice> allInstrumented(Built b) {
        List<InstrumentedFogDevice> out = new ArrayList<InstrumentedFogDevice>();
        out.add(b.cloud);
        out.addAll(b.fogs);
        out.addAll(b.edges);
        return out;
    }

    private static RiskPredictor buildPredictor(RunConfig cfg) throws Exception {
        switch (cfg.policy) {
            case BBN: return new BbnPredictor(BbnEngine.load(cfg.bbnModel), cfg.onlineLearning);
            case ML: return MlPredictor.load(cfg.mlModel);
            case ORACLE: return new OraclePredictor();
            case THRESHOLD: return HealthIndexPredictor.load(cfg.healthIndexModel);
            default: return null;
        }
    }

    private static void printSummary(RunConfig cfg, Built b, FaultToleranceController ft,
                                     double end, double wall, String predictorName) {
        org.json.simple.JSONObject s = ft.summary(end, predictorName);
        TelemetryCollector c = TelemetryCollector.getInstance();
        System.out.println(String.format(Locale.ROOT,
                "=== scenario %02d (%s)  policy=%s  seed=%d  %s ===",
                cfg.spec.id, cfg.spec.label, cfg.policy.key, cfg.seed,
                cfg.dropout == RunContext.Dropout.NONE ? "" : "dropout=" + cfg.dropout + "@" + cfg.dropoutRate));
        System.out.println(String.format(Locale.ROOT,
                "devices            : 1 cloud, %d fog, %d edge; sim time %.0f; wall %.1f s",
                b.fogs.size(), b.edges.size(), end, wall));
        System.out.println(String.format(Locale.ROOT,
                "failures           : fog %s (%s with users), edge %s; causes %s",
                s.get("fog_failures"), s.get("fog_failures_with_users"), s.get("edge_failures"),
                s.get("failure_causes")));
        System.out.println(String.format(Locale.ROOT,
                "sessions           : hit %s, evacuated before failure %s",
                s.get("sessions_hit"), s.get("sessions_evacuated")));
        System.out.println(String.format(Locale.ROOT,
                "service            : downtime %.1f, availability %.5f, delivered %s",
                ((Number) s.get("downtime_total")).doubleValue(),
                ((Number) s.get("availability")).doubleValue(), s.get("delivery_ratio")));
        System.out.println(String.format(Locale.ROOT,
                "latency            : loop mean %s, p95 %s, critical deadline misses %s",
                s.get("loop_delay_mean"), s.get("loop_delay_p95"), s.get("critical_deadline_miss_ratio")));
        System.out.println(String.format(Locale.ROOT,
                "actions            : migrations %s (unnecessary %s), standby ratio %.3f, replication %.2f MB",
                s.get("migrations_started"), s.get("migrations_unnecessary"),
                ((Number) s.get("standby_ratio")).doubleValue(),
                ((Number) s.get("replication_mb")).doubleValue()));
        if (cfg.telemetryCsv != null) {
            System.out.println("telemetry          : " + c.snapshotCount() + " rows, "
                    + c.positiveCount() + " positive -> " + cfg.telemetryCsv);
            Map<String, int[]> bd = c.classBreakdown();
            for (Map.Entry<String, int[]> e : bd.entrySet()) {
                System.out.println("  " + e.getKey() + " : " + e.getValue()[0] + " rows / "
                        + e.getValue()[1] + " positive");
            }
            System.out.println("channel coverage % : " + c.channelCoverage());
        }
        if (cfg.summaryJson != null) System.out.println("summary            : " + cfg.summaryJson);
    }

    // ---- topology --------------------------------------------------------------------------

    private static Built createTopology(int userId, String appId, ScenarioSpec spec, long seed)
            throws Exception {
        DataParser parser = new DataParser();
        parser.parseResourceData();
        Built b = new Built();

        // Level 0: the cloud datacentre.
        for (String dataId : parser.levelwiseResources.get(0)) {
            b.cloud = createDevice("cloud", 0, spec);
            b.cloud.setParentId(-1);
            b.all.add(b.cloud);
            break;
        }

        // Level 1: fog nodes, from the dataset's gateway resources.
        for (String dataId : parser.levelwiseResources.get(2)) {
            InstrumentedFogDevice fog = createDevice("fog-" + dataId.substring(dataId.indexOf('_') + 1), 1, spec);
            fog.setParentId(b.cloud.getId());
            fog.setUplinkLatency(spec.fogUplinkLatency);
            b.all.add(fog);
            b.fogs.add(fog);
            b.fogLocations.put(fog.getId(), parser.resourceLocationData.get(dataId));
        }

        // Level 2: edge devices (users), each with its sensors and alarm actuator.
        int sensorIndex = 0;
        for (int i = 0; i < spec.edgeCount; i++) {
            InstrumentedFogDevice edge = createDevice("edge-" + i, 2, spec);
            edge.setParentId(b.fogs.get(i % b.fogs.size()).getId());
            edge.setUplinkLatency(2);
            b.all.add(edge);
            b.edges.add(edge);
            SessionRegistry.get().addSession(
                    new ServiceSession(edge.getId(), edge.getName(), Criticality.assign(seed, edge.getName())));

            for (int k = 0; k < spec.sensorsPerEdge; k++) {
                Sensor vitals = new Sensor("s-" + sensorIndex, "VITALS", userId, appId,
                        new DeterministicDistribution(spec.sensorInterval));
                vitals.setGatewayDeviceId(edge.getId());
                vitals.setLatency(SENSOR_LATENCY);
                b.sensors.add(vitals);
                sensorIndex++;
            }
            InstrumentedActuator alarm = new InstrumentedActuator("a-" + i, userId, appId, "ALARM",
                    edge.getId(), CRITICAL_LOOP_DEADLINE);
            alarm.setGatewayDeviceId(edge.getId());
            alarm.setLatency(SENSOR_LATENCY);
            b.actuators.add(alarm);
        }
        return b;
    }

    /** One instrumented device sized by the scenario. */
    private static InstrumentedFogDevice createDevice(String name, int level, ScenarioSpec spec)
            throws Exception {
        long mips = spec.mipsFor(level);
        int ram = spec.ramFor(level);
        double[] power = spec.powerFor(level);

        String deviceClass;
        long upBw;
        long downBw;
        switch (level) {
            case 0: deviceClass = "cloud"; upBw = 100; downBw = 10000; break;
            case 1: deviceClass = "fog"; upBw = spec.fogUplinkBw; downBw = 10000; break;
            default: deviceClass = "edge"; upBw = 10000; downBw = 270; break;
        }

        List<Pe> peList = new ArrayList<Pe>();
        peList.add(new Pe(0, new PeProvisionerOverbooking(mips)));
        PowerHost host = new PowerHost(
                FogUtils.generateEntityId(),
                new RamProvisionerSimple(ram),
                new BwProvisionerOverbooking(10000),
                1000000,
                peList,
                new StreamOperatorScheduler(peList),
                new FogLinearPowerModel(power[0], power[1]));
        List<Host> hostList = new ArrayList<Host>();
        hostList.add(host);

        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(
                Config.FOG_DEVICE_ARCH, Config.FOG_DEVICE_OS, Config.FOG_DEVICE_VMM,
                host, Config.FOG_DEVICE_TIMEZONE, Config.FOG_DEVICE_COST,
                Config.FOG_DEVICE_COST_PER_MEMORY, Config.FOG_DEVICE_COST_PER_STORAGE,
                Config.FOG_DEVICE_COST_PER_BW);

        boolean thermalSensor = level != 2;
        double thermalRise = level == 0 ? CLOUD_THERMAL_RISE_C : FOG_THERMAL_RISE_C;
        InstrumentedFogDevice device = new InstrumentedFogDevice(
                name, characteristics, new AppModuleAllocationPolicy(hostList),
                new LinkedList<Storage>(), 10, upBw, downBw, 0,
                (level == 0) ? 0.01 : 0.0, deviceClass,
                spec.energyBudgetFor(level), power[0], thermalSensor, thermalRise);
        device.setLevel(level);
        return device;
    }

    // ---- application ------------------------------------------------------------------------

    /**
     * Cardiac monitoring: vitals are filtered on the user device (client), analysed by an
     * ecg_analyser on the serving node, and the verdict raises the alarm locally; patient
     * summaries are archived in the cloud. Module sizes and the analysis tuple's CPU cost
     * come from the scenario.
     */
    private static Application createApplication(String appId, int userId, ScenarioSpec spec, long seed) {
        Application application = Application.createApplication(appId, userId);

        application.addAppModule("client", spec.clientRam, spec.clientMips, 10000);
        application.addAppModule("ecg_analyser", spec.analyserRam, spec.analyserMips, 10000);
        application.addAppModule("archival_store", 100, 1000, 10000);

        application.addAppEdge("VITALS", "client", 1000, 500, "VITALS", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("client", "ecg_analyser", spec.analysisTupleMi * ANALYSIS_COST_SCALE, 500,
                "FILTERED_VITALS", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("ecg_analyser", "archival_store", 100, 1000, 1000,
                "PATIENT_SUMMARY", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("ecg_analyser", "client", 14, 500,
                "ARRHYTHMIA_VERDICT", Tuple.DOWN, AppEdge.MODULE);
        application.addAppEdge("client", "ALARM", 100, 50, "ALARM_SIGNAL", Tuple.DOWN, AppEdge.ACTUATOR);

        // Seeded selectivity: iFogSim's FractionalSelectivity uses unseeded Math.random().
        application.addTupleMapping("client", "VITALS", "FILTERED_VITALS",
                new SeededFractionalSelectivity(0.9, seed ^ 0x11L));
        application.addTupleMapping("ecg_analyser", "FILTERED_VITALS", "ARRHYTHMIA_VERDICT",
                new SeededFractionalSelectivity(1.0, seed ^ 0x22L));
        application.addTupleMapping("client", "ARRHYTHMIA_VERDICT", "ALARM_SIGNAL",
                new SeededFractionalSelectivity(1.0, seed ^ 0x33L));

        final AppLoop loop = new AppLoop(new ArrayList<String>() {{
            add("VITALS"); add("client"); add("ecg_analyser"); add("client"); add("ALARM");
        }});
        List<AppLoop> loops = new ArrayList<AppLoop>() {{ add(loop); }};
        application.setLoops(loops);
        return application;
    }
}
