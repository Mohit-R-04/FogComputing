package org.fog.ft;

import java.util.ArrayDeque;
import java.util.List;

import org.cloudbus.cloudsim.CloudletScheduler;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.application.AppModule;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Tuple;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.utils.Config;
import org.fog.utils.FogEvents;
import org.fog.utils.NetworkUsageMonitor;

/**
 * A {@link FogDevice} that reports telemetry every resource-management epoch, degrades
 * according to the {@link FaultInjector}, and genuinely stops when it fails.
 *
 * <p>Everything here is a subclass override; no upstream iFogSim file is modified.
 *
 * <h2>Telemetry</h2>
 * The hook is {@link FogDevice#manageResources(SimEvent)}, which iFogSim fires at t=0 and
 * every {@code Config.RESOURCE_MGMT_INTERVAL} afterwards. Each snapshot is measured from
 * live simulator state: work executed against capacity (CPU), the RAM provisioner, tuple
 * queues and arrivals, link delay plus transmission load, arrival-time jitter, topology
 * churn, iFogSim's cumulative energy against a declared budget, and a lumped thermal
 * model driven by the measured utilisation. The agent also reports the trend of each
 * reading over its last three samples.
 *
 * <h2>Degradation inside the simulator</h2>
 * A worn node needs more CPU work per tuple (work inflation), its interface adds delay,
 * it draws extra power and it runs hotter. iFogSim's own scheduler, links and energy
 * model turn those into longer queues, higher utilisation, longer loop delays and
 * faster battery drain -- which is what the telemetry then observes.
 *
 * <h2>Failure</h2>
 * At its failure instant the device drops every tuple it receives, clears its queues and
 * stops reporting. Users attached to it lose service until the active fault-tolerance
 * policy recovers them.
 *
 * <h2>Routing</h2>
 * Upstream iFogSim broadcasts a downward tuple to every child; a verdict computed in the
 * cloud reached all 118 fog nodes and every user. Downward tuples are routed here only
 * towards the edge device whose {@code client} instance they belong to.
 */
public class InstrumentedFogDevice extends FogDevice {

    private final String deviceClass;
    private final double reliabilityPrior;
    /** Declared energy capacity, in the simulator's power x time units. */
    private final double energyBudget;
    /** Busy power of the device, used to scale degradation's extra draw. */
    private final double busyPower;
    /** Whether the device exposes a temperature sensor (fog and cloud do; edge does not). */
    private final boolean hasThermalSensor;
    /** Temperature rise at full utilisation, degrees Celsius. */
    private final double thermalRiseC;

    private static final int HANDOVER_WINDOW_EPOCHS = 5;
    private static final int TREND_WINDOW = 3;
    private static final double THERMAL_TAU_EPOCHS = 2.5;
    private static final double THERMAL_NOISE_C = 0.8;
    private static final int MAX_ARRIVAL_TIMES = 512;

    private final ArrayDeque<Integer> handoverWindow = new ArrayDeque<Integer>();
    private int handoversThisEpoch = 0;
    private boolean topologyBuilt = false;

    private int arrivalsThisWindow = 0;
    private double arrivedMiThisWindow = 0.0;
    private double arrivedBytesThisWindow = 0.0;
    private final ArrayDeque<Double> arrivalTimes = new ArrayDeque<Double>();
    private double executedMiThisWindow = 0.0;

    private double ambientC = 25.0;
    private double modelTemperature = Double.NaN;

    private final ArrayDeque<Double> cpuHist = new ArrayDeque<Double>();
    private final ArrayDeque<Double> queueHist = new ArrayDeque<Double>();
    private final ArrayDeque<Double> latencyHist = new ArrayDeque<Double>();
    private final ArrayDeque<Double> tempHist = new ArrayDeque<Double>();
    private double lastReportedEnergy = Double.NaN;
    private double lastReportedEnergyTime = Double.NaN;

    private boolean failed = false;
    private double failedAt = Double.NaN;
    private NodeTelemetry latest;

    public InstrumentedFogDevice(String name,
                                 FogDeviceCharacteristics characteristics,
                                 AppModuleAllocationPolicy allocationPolicy,
                                 List<Storage> storageList,
                                 double schedulingInterval,
                                 double uplinkBandwidth,
                                 double downlinkBandwidth,
                                 double uplinkLatency,
                                 double ratePerMips,
                                 String deviceClass,
                                 double energyBudget,
                                 double busyPower,
                                 boolean hasThermalSensor,
                                 double thermalRiseC) throws Exception {
        super(name, characteristics, allocationPolicy, storageList, schedulingInterval,
              uplinkBandwidth, downlinkBandwidth, uplinkLatency, ratePerMips);
        this.deviceClass = deviceClass;
        this.reliabilityPrior = FaultInjector.reliabilityPriorFor(deviceClass);
        this.energyBudget = energyBudget;
        this.busyPower = busyPower;
        this.hasThermalSensor = hasThermalSensor;
        this.thermalRiseC = thermalRiseC;
        RunContext.injector().register(name, deviceClass);
        SessionRegistry.get().registerDevice(this);
    }

    /** Call once the topology is wired, so setup-time parent assignment is not counted. */
    public void markTopologyBuilt() {
        this.topologyBuilt = true;
        if (hasThermalSensor) {
            double u = FtRandom.uniform(FtRandom.key(RunContext.seed(), RunContext.scenarioId(),
                    FtRandom.hash(getName()), 0, FtRandom.S_AMBIENT));
            // Climate-controlled datacentre versus street cabinets in the Melbourne CBD.
            ambientC = "cloud".equals(deviceClass) ? 20.0 : 22.0 + 12.0 * u;
            modelTemperature = ambientC;
        }
    }

    // ---- topology churn -------------------------------------------------------------

    @Override
    public void setParentId(int parentId) {
        int previous = getParentId();
        super.setParentId(parentId);
        if (topologyBuilt && previous != parentId && previous != -1) {
            handoversThisEpoch++;
        }
    }

    @Override
    public void addChild(int childId) {
        boolean had = getChildrenIds().contains(childId);
        super.addChild(childId);
        if (topologyBuilt && !had && getChildrenIds().contains(childId)) {
            handoversThisEpoch++;
        }
    }

    @Override
    public void removeChild(int childId) {
        boolean had = getChildrenIds().contains(childId);
        super.removeChild(childId);
        if (topologyBuilt && had) {
            handoversThisEpoch++;
        }
    }

    // ---- failure ----------------------------------------------------------------------

    @Override
    protected void processOtherEvent(SimEvent ev) {
        if (ev.getTag() == FtEvents.NODE_FAIL) {
            markFailed();
            return;
        }
        super.processOtherEvent(ev);
    }

    private void markFailed() {
        if (failed) {
            return;
        }
        failed = true;
        failedAt = CloudSim.clock();
        getNorthTupleQueue().clear();
        getSouthTupleQueue().clear();
        SessionRegistry.get().onDeviceFailed(this, failedAt);
    }

    public boolean isFailed() {
        return failed;
    }

    public double failedAt() {
        return failedAt;
    }

    // ---- module bookkeeping -------------------------------------------------------------

    @Override
    protected void processModuleArrival(SimEvent ev) {
        super.processModuleArrival(ev);
        Object data = ev.getData();
        if (data instanceof AppModule && "client".equals(((AppModule) data).getName())) {
            SessionRegistry.get().registerClientVm(((AppModule) data).getId(), getId());
        }
    }

    public boolean hostsModule(String appId, String module) {
        List<String> modules = appToModulesMap.get(appId);
        return modules != null && modules.contains(module);
    }

    // ---- tuple flow ---------------------------------------------------------------------

    @Override
    protected void processTupleArrival(SimEvent ev) {
        Object data = ev.getData();
        if (!(data instanceof Tuple)) {
            super.processTupleArrival(ev);
            return;
        }
        Tuple tuple = (Tuple) data;
        SessionRegistry registry = SessionRegistry.get();

        if (failed) {
            if ("ecg_analyser".equals(tuple.getDestModuleName())) {
                ServiceSession s = registry.sessionForTuple(tuple);
                if (s != null) s.analysesDroppedFailedNode++;
            }
            return;
        }

        arrivalsThisWindow++;
        arrivedMiThisWindow += tuple.getCloudletLength();
        arrivedBytesThisWindow += tuple.getCloudletFileSize();
        if (arrivalTimes.size() < MAX_ARRIVAL_TIMES) {
            arrivalTimes.addLast(CloudSim.clock());
        }

        String dest = tuple.getDestModuleName();
        boolean hostsDest = dest != null && hostsModule(tuple.getAppId(), dest);

        // A session that is down (being restored, migrated or cold-started) cannot
        // analyse: its tuples are lost, exactly as they would be in deployment.
        if (hostsDest && "ecg_analyser".equals(dest) && tuple.getDirection() == Tuple.UP) {
            ServiceSession s = registry.sessionForTuple(tuple);
            if (s != null && !s.isAvailable(CloudSim.clock())) {
                s.analysesDroppedUnavailable++;
                return;
            }
        }

        if (!hostsDest && dest != null && tuple.getDirection() == Tuple.DOWN) {
            int target = registry.edgeForTuple(tuple);
            if (target >= 0 && target != getId()) {
                int child = getChildrenIds().contains(target) ? target : getChildIdWithRouteTo(target);
                if (child != -1) {
                    sendDown(tuple, child);
                } else if (!forwardToCurrentGateway(tuple, target)) {
                    ServiceSession s = registry.session(target);
                    if (s != null) s.analysesDroppedUnavailable++;
                }
                return;
            }
        }
        super.processTupleArrival(ev);
    }

    /** Backhaul delay of forwarding a tuple to a user's new gateway, in ms. */
    private static final double FORWARD_LATENCY = 5.0;
    private static final int MAX_FORWARDS = 2;

    /**
     * The user handed over while its result was in flight. Like mobile IP, the old
     * gateway forwards the tuple once over the backhaul to the user's current gateway,
     * instead of dropping it because iFogSim's routing only walks the tree. Bounded, so
     * a user moving repeatedly cannot make a tuple circulate.
     */
    private boolean forwardToCurrentGateway(Tuple tuple, int target) {
        FogDevice edge = SessionRegistry.get().device(target);
        if (edge == null) return false;
        FogDevice gateway = SessionRegistry.get().device(edge.getParentId());
        if (!(gateway instanceof InstrumentedFogDevice) || gateway == this
                || ((InstrumentedFogDevice) gateway).isFailed()) {
            return false;
        }
        if (SessionRegistry.get().countForward(tuple.getCloudletId()) > MAX_FORWARDS) return false;
        send(gateway.getId(), FORWARD_LATENCY, FogEvents.TUPLE_ARRIVAL, tuple);
        NetworkUsageMonitor.sendingTuple(FORWARD_LATENCY, tuple.getCloudletFileSize());
        return true;
    }

    @Override
    protected void executeTuple(SimEvent ev, String moduleName) {
        Object data = ev.getData();
        if (data instanceof Tuple) {
            Tuple tuple = (Tuple) data;
            double factor = RunContext.injector().slowdown(getName());
            if (factor > 1.0 + 1e-9) {
                tuple.setCloudletLength((long) Math.ceil(tuple.getCloudletLength() * factor));
            }
            executedMiThisWindow += tuple.getCloudletLength();
        }
        super.executeTuple(ev, moduleName);
    }

    @Override
    protected void sendUp(Tuple tuple) {
        if (getLevel() == 2 && "ecg_analyser".equals(tuple.getDestModuleName())) {
            ServiceSession s = SessionRegistry.get().session(getId());
            if (s != null) s.analysesEmitted++;
        }
        if (failed) return;
        super.sendUp(tuple);
    }

    @Override
    protected void sendDown(Tuple tuple, int childId) {
        if (failed) return;
        super.sendDown(tuple, childId);
    }

    @Override
    protected void sendTupleToActuator(Tuple tuple) {
        if (failed) return;
        super.sendTupleToActuator(tuple);
    }

    @Override
    protected void updateNorthTupleQueue() {
        if (failed) {
            getNorthTupleQueue().clear();
            setNorthLinkBusy(false);
            return;
        }
        super.updateNorthTupleQueue();
    }

    @Override
    protected void updateSouthTupleQueue() {
        if (failed) {
            getSouthTupleQueue().clear();
            setSouthLinkBusy(false);
            return;
        }
        super.updateSouthTupleQueue();
    }

    /** Upstream logic with this node's degradation applied to its interface delay. */
    @Override
    protected void sendDownFreeLink(Tuple tuple, int childId) {
        double networkDelay = tuple.getCloudletFileSize() / getDownlinkBandwidth();
        setSouthLinkBusy(true);
        Double base = getChildToLatencyMap().get(childId);
        double latency = (base == null ? 0.0 : base) * RunContext.injector().latencyFactor(getName());
        send(getId(), networkDelay, FogEvents.UPDATE_SOUTH_TUPLE_QUEUE);
        send(childId, networkDelay + latency, FogEvents.TUPLE_ARRIVAL, tuple);
        NetworkUsageMonitor.sendingTuple(latency, tuple.getCloudletFileSize());
    }

    /** Link delay including this node's interface degradation. */
    @Override
    public double getUplinkLatency() {
        return super.getUplinkLatency() * RunContext.injector().latencyFactor(getName());
    }

    /** Configured link delay without degradation (geometry only). */
    public double getBaseUplinkLatency() {
        return super.getUplinkLatency();
    }

    // ---- epoch: degrade, measure, report ------------------------------------------------

    @Override
    protected void manageResources(SimEvent ev) {
        if (failed) {
            return; // a dead node neither accounts energy nor reports
        }
        super.manageResources(ev);
        addDegradationEnergy();

        NodeTelemetry truth = measure();
        FaultInjector injector = RunContext.injector();
        double failureTime = injector.step(truth, Config.RESOURCE_MGMT_INTERVAL);
        if (!Double.isNaN(failureTime)) {
            send(getId(), Math.max(0.0, failureTime - CloudSim.clock()), FtEvents.NODE_FAIL);
        }
        latest = report(truth, injector.health(getName()));
        TelemetryCollector.getInstance().record(latest);
    }

    /** Most recent reported snapshot (what the control plane sees). */
    public NodeTelemetry latestTelemetry() {
        return latest;
    }

    private void addDegradationEnergy() {
        double extra = RunContext.injector().extraPowerFraction(getName())
                * busyPower * Config.RESOURCE_MGMT_INTERVAL;
        if (extra > 0) {
            setEnergyConsumption(getEnergyConsumption() + extra);
        }
    }

    /** True state of the node this epoch (before any telemetry dropout). */
    private NodeTelemetry measure() {
        double cpu = cpuUtilisation();
        double ram = ramUtilisation();
        // Order matters: queueDepth() consumes the arrival counters, so jitter and
        // latency read them first.
        double jitter = arrivalJitter();
        double latency = effectiveUplinkLatency();
        double queue = queueDepth();
        double energy = residualEnergy();
        double handovers = handoverWindowSum();
        double temperature = updateThermalModel(cpu);
        return new NodeTelemetry(RunContext.scenarioId(), RunContext.seed(), getId(), getName(),
                getLevel(), deviceClass, CloudSim.clock(),
                cpu, ram, queue, latency, jitter, handovers, energy, temperature,
                Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                reliabilityPrior, RunContext.injector().health(getName()));
    }

    /** The agent's report: sensor noise, dropout, and trends over reported history. */
    private NodeTelemetry report(NodeTelemetry t, double healthAfterStep) {
        long nodeHash = FtRandom.hash(getName());
        long epochIndex = Math.round(t.epoch / Config.RESOURCE_MGMT_INTERVAL);
        long seed = RunContext.seed();
        int scenario = RunContext.scenarioId();

        double temperature = t.temperature;
        if (!Double.isNaN(temperature)) {
            temperature += THERMAL_NOISE_C * FtRandom.normal(FtRandom.key(seed, scenario, nodeHash,
                    epochIndex, FtRandom.S_THERMAL_NOISE));
        }
        double[] v = {t.cpuUtil, t.ramUtil, t.queueDepth, t.uplinkLatency, t.jitterPktLoss,
                      t.handoverRate, t.residualEnergy, temperature};

        RunContext.Dropout mode = RunContext.dropout();
        if (mode != RunContext.Dropout.NONE && RunContext.dropoutRate() > 0) {
            double rate = RunContext.dropoutRate();
            if (mode == RunContext.Dropout.MNAR) {
                rate = Math.min(0.95, rate * (1.0 + (1.0 - healthAfterStep)));
            }
            for (int ch = 0; ch < v.length; ch++) {
                double u = FtRandom.uniform(FtRandom.key(seed, scenario, nodeHash, epochIndex,
                        FtRandom.S_DROPOUT + ch));
                if (u < rate) {
                    v[ch] = Double.NaN;
                }
            }
        }

        double cpuTrend = trend(v[0], cpuHist, false);
        double queueTrend = trend(v[2], queueHist, true);
        double latencyTrend = trend(v[3], latencyHist, true);
        double tempTrend = trend(v[7], tempHist, false);
        double energyDrain = Double.NaN;
        if (!Double.isNaN(v[6])) {
            if (!Double.isNaN(lastReportedEnergy)) {
                double epochs = Math.max(1.0, (t.epoch - lastReportedEnergyTime)
                        / Config.RESOURCE_MGMT_INTERVAL);
                energyDrain = (lastReportedEnergy - v[6]) / epochs;
            }
            lastReportedEnergy = v[6];
            lastReportedEnergyTime = t.epoch;
        }

        return new NodeTelemetry(t.scenarioId, t.seed, t.nodeId, t.nodeName, t.level,
                t.deviceClass, t.epoch, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                cpuTrend, queueTrend, latencyTrend, tempTrend, energyDrain,
                t.reliabilityPrior, healthAfterStep);
    }

    /** Change of a reading against the mean of the agent's previous readings. */
    private static double trend(double value, ArrayDeque<Double> history, boolean relative) {
        double out = Double.NaN;
        if (!Double.isNaN(value) && !history.isEmpty()) {
            double sum = 0.0;
            for (Double d : history) sum += d;
            double mean = sum / history.size();
            out = relative ? (value - mean) / Math.max(1.0, Math.abs(mean)) : value - mean;
        }
        if (!Double.isNaN(value)) {
            history.addLast(value);
            while (history.size() > TREND_WINDOW) history.removeFirst();
        }
        return out;
    }

    // ---- the measurement channels -------------------------------------------------------

    /** Work executed during the window against the work the host could execute. */
    private double cpuUtilisation() {
        double capacityMi = getHost().getTotalMips() * Config.RESOURCE_MGMT_INTERVAL;
        double executed = executedMiThisWindow;
        executedMiThisWindow = 0.0;
        if (capacityMi <= 0) return 0.0;
        return Math.min(1.0, executed / capacityMi);
    }

    /** Allocated RAM against capacity; iFogSim allocates module memory once at placement. */
    private double ramUtilisation() {
        Host host = getHost();
        int totalRam = host.getRam();
        if (totalRam <= 0) return 0.0;
        int available = host.getRamProvisioner().getAvailableRam();
        return Math.min(1.0, Math.max(0.0, (totalRam - available) / (double) totalRam));
    }

    /**
     * Pending work for the window: tuples resident in link and scheduler queues, tuples
     * that arrived, and the excess of arriving work over what the host could execute. A
     * degraded node needs more work per tuple, so its effective arriving work is scaled.
     */
    private double queueDepth() {
        int resident = 0;
        if (getNorthTupleQueue() != null) resident += getNorthTupleQueue().size();
        if (getSouthTupleQueue() != null) resident += getSouthTupleQueue().size();
        if (getClusterTupleQueue() != null) resident += getClusterTupleQueue().size();
        for (Object vmObj : getHost().getVmList()) {
            CloudletScheduler scheduler = ((Vm) vmObj).getCloudletScheduler();
            if (scheduler == null) continue;
            resident += scheduler.getCloudletExecList().size();
            resident += scheduler.getCloudletWaitingList().size();
        }
        int arrived = arrivalsThisWindow;
        double arrivedMi = arrivedMiThisWindow * RunContext.injector().slowdown(getName());
        arrivalsThisWindow = 0;
        arrivedMiThisWindow = 0.0;

        double capacityMi = getHost().getTotalMips() * Config.RESOURCE_MGMT_INTERVAL;
        int backlog = 0;
        if (arrived > 0 && arrivedMi > capacityMi) {
            double meanMi = arrivedMi / arrived;
            backlog = (int) Math.round((arrivedMi - capacityMi) / Math.max(1.0, meanMi));
        }
        return resident + arrived + backlog;
    }

    /** Coefficient of variation of tuple inter-arrival times; NaN with < 3 arrivals. */
    private double arrivalJitter() {
        if (arrivalTimes.size() < 3) {
            arrivalTimes.clear();
            return Double.NaN;
        }
        Double[] times = arrivalTimes.toArray(new Double[0]);
        arrivalTimes.clear();
        int n = times.length - 1;
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += times[i + 1] - times[i];
        double mean = sum / n;
        if (mean <= 0.0) return 0.0;
        double variance = 0.0;
        for (int i = 0; i < n; i++) {
            double d = (times[i + 1] - times[i]) - mean;
            variance += d * d;
        }
        return Math.min(1.0, Math.sqrt(variance / n) / mean);
    }

    /** Residual energy as a fraction of the declared capacity. */
    private double residualEnergy() {
        if (energyBudget <= 0) return 1.0;
        return Math.min(1.0, Math.max(0.0, 1.0 - getEnergyConsumption() / energyBudget));
    }

    /** Link delay (with degradation) plus the time the window's traffic takes to clear. */
    private double effectiveUplinkLatency() {
        double base = getUplinkLatency();
        double bandwidth = getUplinkBandwidth();
        double bytes = arrivedBytesThisWindow;
        arrivedBytesThisWindow = 0.0;
        double transmission = (bandwidth > 0) ? (bytes / bandwidth) : 0.0;
        double busyPenalty = isNorthLinkBusy() ? base * 0.5 : 0.0;
        return base + transmission + busyPenalty;
    }

    private double handoverWindowSum() {
        handoverWindow.addLast(handoversThisEpoch);
        while (handoverWindow.size() > HANDOVER_WINDOW_EPOCHS) handoverWindow.removeFirst();
        int total = 0;
        for (Integer h : handoverWindow) total += h;
        handoversThisEpoch = 0;
        return total;
    }

    /**
     * First-order (lumped RC) thermal model: the enclosure relaxes towards ambient plus a
     * rise proportional to utilisation plus the extra heat of degraded components, with a
     * time constant of 2.5 epochs. Returns NaN on devices without a sensor.
     */
    private double updateThermalModel(double cpu) {
        if (!hasThermalSensor) return Double.NaN;
        double steady = ambientC + thermalRiseC * cpu + RunContext.injector().heatRise(getName());
        if (Double.isNaN(modelTemperature)) modelTemperature = ambientC;
        modelTemperature += (steady - modelTemperature) * (1.0 - Math.exp(-1.0 / THERMAL_TAU_EPOCHS));
        return modelTemperature;
    }

    public String getDeviceClass() {
        return deviceClass;
    }

    public double getEnergyBudget() {
        return energyBudget;
    }

    public double getBusyPower() {
        return busyPower;
    }
}
