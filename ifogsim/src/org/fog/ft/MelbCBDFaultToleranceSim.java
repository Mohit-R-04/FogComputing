package org.fog.ft;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

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
import org.fog.placement.Controller;
import org.fog.placement.ModuleMapping;
import org.fog.placement.ModulePlacementEdgewards;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.Config;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;

/**
 * Runs one scenario of the sweep and writes its labelled telemetry.
 *
 * <p>The topology has exactly three tiers: cloud, fog nodes and edge devices.
 * The Melbourne CBD edge-resource set supplies one cloud and the original gateway
 * resources, which are used as fog nodes; proxy resources are intentionally excluded.
 * Battery-powered user devices are added as edge devices. Every node is an
 * {@link InstrumentedFogDevice}, so telemetry is emitted at each resource-management
 * epoch.
 *
 * <p>The workload is a cardiac-monitoring application -- the latency-critical use
 * case the paper cites as motivation -- placed edgeward.
 *
 * <p>Every capacity, power model, module size and emission rate comes from the
 * {@link ScenarioSpec} handed in, which is how the sweep produces genuinely
 * different simulator behaviour from one run to the next.
 *
 * <p><b>Run from the project root.</b> {@link DataParser} resolves
 * {@code ./dataset/edgeResources-melbCBD.csv} relative to the working directory.
 *
 * <p>Entry point is {@link ScenarioSweep}, not this class.
 */
public class MelbCBDFaultToleranceSim {

    private static final String APP_ID = "cardiac_monitoring";
    private static final double SENSOR_LATENCY = 1.0;

    private static final List<FogDevice> fogDevices = new ArrayList<FogDevice>();
    private static final List<Sensor> sensors = new ArrayList<Sensor>();
    private static final List<Actuator> actuators = new ArrayList<Actuator>();
    private static final List<FogDevice> fogDevicesForMobility = new ArrayList<FogDevice>();
    private static final List<FogDevice> edgeDevices = new ArrayList<FogDevice>();
    private static final Map<Integer, Location> fogLocations =
            new HashMap<Integer, Location>();

    private static MobilityDriver mobilityDriver;
    private static ScenarioSpec spec;
    private static final AtomicBoolean WRITTEN = new AtomicBoolean(false);

    /**
     * Runs one scenario end to end and writes its telemetry to {@code outCsv}.
     */
    public static void runScenario(ScenarioSpec scenario, long seed, String outCsv) {
        spec = scenario;

        System.out.println("=== iFogSim telemetry generation ===");
        System.out.println(scenario);
        System.out.println("seed=" + seed + "  output=" + outCsv);

        fogDevices.clear();
        sensors.clear();
        actuators.clear();
        fogDevicesForMobility.clear();
        edgeDevices.clear();
        fogLocations.clear();
        WRITTEN.set(false);

        try {
            Log.disable();
            Config.MAX_SIMULATION_TIME = scenario.simulationTime;

            TelemetryCollector.reset(seed, scenario.id);
            TelemetryCollector.getInstance().setEpochLength(Config.RESOURCE_MGMT_INTERVAL);

            CloudSim.init(1, Calendar.getInstance(), false);

            FogBroker broker = new FogBroker("broker");
            Application application = createApplication(APP_ID, broker.getId(), seed);
            application.setUserId(broker.getId());

            createTopology(broker.getId(), APP_ID, seed);

            // Users walk the real melbCBD traces and re-associate with their nearest
            // gateway, which is what gives handover_rate and uplink_latency their
            // variation.
            mobilityDriver = new MobilityDriver("mobility-driver", edgeDevices,
                    fogDevicesForMobility, fogLocations, seed);

            for (FogDevice d : fogDevices) {
                if (d instanceof InstrumentedFogDevice) {
                    ((InstrumentedFogDevice) d).markTopologyBuilt();
                }
            }

            ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();
            moduleMapping.addModuleToDevice("archival_store", "cloud");
            for (FogDevice device : fogDevices) {
                if (device.getName().startsWith("edge-")) {
                    moduleMapping.addModuleToDevice("client", device.getName());
                }
            }

            Controller controller = new Controller("master-controller", fogDevices,
                    sensors, actuators);
            controller.submitApplication(application, 0,
                    new ModulePlacementEdgewards(fogDevices, sensors, actuators,
                            application, moduleMapping));

            TimeKeeper.getInstance().setSimulationStartTime(
                    Calendar.getInstance().getTimeInMillis());

            System.out.println("devices=" + fogDevices.size()
                    + "  sensors=" + sensors.size()
                    + "  actuators=" + actuators.size());
            System.out.println("running to t=" + Config.MAX_SIMULATION_TIME + " ...");

            // iFogSim's Controller calls System.exit(0) when it handles
            // STOP_SIMULATION, so nothing placed after startSimulation() would ever
            // run. A shutdown hook writes the output without patching any existing
            // iFogSim source file.
            installOutputHook(outCsv);

            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            writeOutput(outCsv);   // only reached if iFogSim ever stops exiting

        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("scenario " + scenario.id + " failed");
            System.exit(1);
        }
    }

    private static void installOutputHook(final String outCsv) {
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                try {
                    writeOutput(outCsv);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private static void writeOutput(String outCsv) throws Exception {
        if (!WRITTEN.compareAndSet(false, true)) {
            return;
        }
        TelemetryCollector c = TelemetryCollector.getInstance();
        c.write(outCsv);

        System.out.println();
        System.out.println("=== scenario " + spec.id + " (" + spec.label + ") ===");
        System.out.println("snapshots written  : " + c.snapshotCount());
        System.out.println("nodes that failed  : " + c.failedNodeCount());
        System.out.println("positive snapshots : " + c.positiveCount());
        System.out.println("handovers observed : "
                + (mobilityDriver == null ? 0 : mobilityDriver.getHandoverCount()));
        System.out.println("per level (snapshots / positives):");
        Map<Integer, int[]> bd = c.levelBreakdown();
        for (int level = 0; level <= 2; level++) {
            int[] v = bd.get(level);
            if (v != null) {
                System.out.println("  level " + level + " : " + v[0] + " / " + v[1]);
            }
        }
        System.out.println("optional channel coverage:");
        for (Map.Entry<String, int[]> e : c.channelCoverage().entrySet()) {
            int[] v = e.getValue();
            int pct = v[1] == 0 ? 0 : (100 * v[0] / v[1]);
            System.out.println("  " + e.getKey() + " : " + v[0] + "/" + v[1]
                    + " rows reported (" + pct + "%)");
        }
        System.out.println("written to " + outCsv);
    }

    // ---- topology ------------------------------------------------------------

    private static void createTopology(int userId, String appId, long seed) throws Exception {
        DataParser parser = new DataParser();
        parser.parseResourceData();

        Map<String, Integer> dataIdToEntityId = new HashMap<String, Integer>();
        Map<String, String[]> rawRows = readRawRows();

        // Level 0: the cloud datacentre.
        for (String dataId : parser.levelwiseResources.get(0)) {
            InstrumentedFogDevice cloud = createDevice("cloud", 0);
            cloud.setParentId(-1);
            fogDevices.add(cloud);
            dataIdToEntityId.put(dataId, cloud.getId());
        }

        // Level 1: fog nodes. The source CSV's gateway resources are the fog tier.
        // Proxy resources are deliberately not instantiated, so the generated data
        // contains only cloud, fog and edge device classes.
        List<InstrumentedFogDevice> gateways = new ArrayList<InstrumentedFogDevice>();
        for (String dataId : parser.levelwiseResources.get(2)) {
            InstrumentedFogDevice fog = createDevice("fog-" + idOf(dataId), 1);
            fog.setParentId(fogDevices.get(0).getId());
            fog.setUplinkLatency(spec.fogUplinkLatency);
            fogDevices.add(fog);
            gateways.add(fog);
            fogDevicesForMobility.add(fog);
            fogLocations.put(fog.getId(), parser.resourceLocationData.get(dataId));
            dataIdToEntityId.put(dataId, fog.getId());
        }

        // Level 2: edge devices, attached to fog nodes.
        int sensorIndex = 0;
        for (int i = 0; i < spec.edgeCount; i++) {
            InstrumentedFogDevice edge = createDevice("edge-" + i, 2);
            InstrumentedFogDevice fog = gateways.get(i % gateways.size());
            edge.setParentId(fog.getId());
            edge.setUplinkLatency(2);
            fogDevices.add(edge);
            edgeDevices.add(edge);

            for (int k = 0; k < spec.sensorsPerEdge; k++) {
                Sensor vitals = new Sensor("s-" + sensorIndex, "VITALS", userId, appId,
                        new DeterministicDistribution(spec.sensorInterval));
                vitals.setGatewayDeviceId(edge.getId());
                vitals.setLatency(SENSOR_LATENCY);
                sensors.add(vitals);
                sensorIndex++;
            }

            Actuator alarm = new Actuator("a-" + i, userId, appId, "ALARM");
            alarm.setGatewayDeviceId(edge.getId());
            alarm.setLatency(SENSOR_LATENCY);
            actuators.add(alarm);
        }
    }

    private static String idOf(String dataId) {
        return dataId.substring(dataId.indexOf('_') + 1);
    }

    /** Re-reads the resource CSV for the parent column, which DataParser discards. */
    private static Map<String, String[]> readRawRows() throws Exception {
        Map<String, String[]> rows = new HashMap<String, String[]>();
        BufferedReader r = new BufferedReader(
                new FileReader("./dataset/edgeResources-melbCBD.csv"));
        try {
            String line;
            while ((line = r.readLine()) != null) {
                String[] data = line.split(",");
                if (data.length > 6 && data[6].equals("VIC")) {
                    rows.put("res_" + data[0], data);
                }
            }
        } finally {
            r.close();
        }
        return rows;
    }

    /**
     * Creates one instrumented device sized by the scenario.
     *
     * <p>Capacity, memory and power model all come from {@link ScenarioSpec}, so a
     * gateway in the "starved-gateways" scenario is a materially different piece of
     * hardware from the same gateway in "fat-gateways" -- which is the mechanism by
     * which the sweep produces varied telemetry.
     */
    private static InstrumentedFogDevice createDevice(String name, int level) throws Exception {
        long mips = spec.mipsFor(level);
        int ram = spec.ramFor(level);
        double[] power = spec.powerFor(level);

        String deviceClass;
        long upBw, downBw;
        switch (level) {
            case 0:  deviceClass = "cloud"; upBw = 100; downBw = 10000; break;
            case 1:  deviceClass = "fog";   upBw = spec.fogUplinkBw; downBw = 10000; break;
            default: deviceClass = "edge";  upBw = 10000; downBw = 270; break;
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

        InstrumentedFogDevice device = new InstrumentedFogDevice(
                name, characteristics, new AppModuleAllocationPolicy(hostList),
                new LinkedList<Storage>(), 10, upBw, downBw, 0,
                (level == 0) ? 0.01 : 0.0, deviceClass,
                spec.energyBudgetFor(level));
        device.setLevel(level);
        return device;
    }

    // ---- application ---------------------------------------------------------

    /**
     * A cardiac-monitoring pipeline: vitals are filtered on the user device, analysed
     * at the edge, an alarm is raised locally, and summaries are archived in the cloud.
     *
     * <p>Module sizes and the analysis tuple's CPU cost come from the scenario.
     * {@code ModulePlacementEdgewards} decides placement on rate x tupleCpuLength
     * rather than on declared RAM or MIPS, so the tuple cost is what actually
     * determines whether the analyser lands on a handset or is shifted north onto a
     * gateway -- and therefore which tier carries the load in this scenario.
     */
    private static Application createApplication(String appId, int userId, long seed) {
        Application application = Application.createApplication(appId, userId);

        application.addAppModule("client", spec.clientRam, spec.clientMips, 10000);
        application.addAppModule("ecg_analyser", spec.analyserRam, spec.analyserMips, 10000);
        application.addAppModule("archival_store", 100, 1000, 10000);

        application.addAppEdge("VITALS", "client", 1000, 500, "VITALS",
                Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("client", "ecg_analyser", spec.analysisTupleMi, 500,
                "FILTERED_VITALS", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("ecg_analyser", "archival_store", 100, 1000, 1000,
                "PATIENT_SUMMARY", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("ecg_analyser", "client", 14, 500,
                "ARRHYTHMIA_VERDICT", Tuple.DOWN, AppEdge.MODULE);
        application.addAppEdge("client", "ALARM", 100, 50, "ALARM_SIGNAL",
                Tuple.DOWN, AppEdge.ACTUATOR);

        // SeededFractionalSelectivity, not iFogSim's FractionalSelectivity: the
        // latter decides with unseeded Math.random(), which alone made the whole
        // harness non-reproducible. See that class for the measurements.
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
