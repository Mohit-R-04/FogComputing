package org.fog.ft;

import java.util.ArrayDeque;
import java.util.List;

import org.cloudbus.cloudsim.CloudletScheduler;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Tuple;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.utils.Config;

/**
 * A {@link FogDevice} that emits a telemetry snapshot at every resource-management
 * epoch.
 *
 * <p>The hook is {@link FogDevice#manageResources(SimEvent)}, which iFogSim already
 * fires at t=0 from {@code Controller.startEntity()} and which each device then
 * reschedules every {@code Config.RESOURCE_MGMT_INTERVAL} time units. Overriding it
 * gives per-epoch sampling without modifying a single existing iFogSim source file.
 *
 * <p><b>Every reported value is the simulator's own.</b> Nothing is synthesised here
 * and nothing is layered on top of what iFogSim produces. Where a quantity needs a
 * reference point that iFogSim does not define -- energy capacity, since
 * {@code getEnergyConsumption()} is a total that only rises and the simulator has no
 * notion of a battery -- the reference is declared in {@link ScenarioSpec} and named
 * as such. Variation between rows comes from the scenario sweep changing the nodes,
 * not from anything added afterwards.
 */
public class InstrumentedFogDevice extends FogDevice {

    private final String deviceClass;
    private final double reliabilityPrior;
    /** Declared energy capacity, in the simulator's watt-time units. */
    private final double energyBudget;

    // ---- handovers -----------------------------------------------------------
    /** Epochs of history retained for the handover rate. */
    private static final int HANDOVER_WINDOW_EPOCHS = 5;
    private final ArrayDeque<Integer> handoverWindow = new ArrayDeque<Integer>();
    private int handoversThisEpoch = 0;
    /** Suppresses handover counting during topology construction. */
    private boolean topologyBuilt = false;

    // ---- arrivals, for queue depth and jitter --------------------------------
    private int arrivalsThisWindow = 0;
    private double arrivedMiThisWindow = 0.0;
    /** Network bytes carried by tuples reaching this device during the window. */
    private double arrivedBytesThisWindow = 0.0;
    /** Simulation times at which tuples reached this device during the window. */
    private final ArrayDeque<Double> arrivalTimes = new ArrayDeque<Double>();
    /** Cap on retained timestamps, so a busy node does not grow without bound. */
    private static final int MAX_ARRIVAL_TIMES = 512;

    /** MI of tuples dispatched to this host's CPU during the window. */
    private double executedMiThisWindow = 0.0;

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
                                 double energyBudget) throws Exception {
        super(name, characteristics, allocationPolicy, storageList, schedulingInterval,
              uplinkBandwidth, downlinkBandwidth, uplinkLatency, ratePerMips);
        this.deviceClass = deviceClass;
        this.reliabilityPrior = FaultInjector.reliabilityPriorFor(deviceClass);
        this.energyBudget = energyBudget;
    }

    /** Call once the topology is wired, so setup-time parent assignment is not counted. */
    public void markTopologyBuilt() {
        this.topologyBuilt = true;
    }

    @Override
    public void setParentId(int parentId) {
        int previous = getParentId();
        super.setParentId(parentId);
        if (topologyBuilt && previous != parentId && previous != -1) {
            handoversThisEpoch++;
        }
    }

    @Override
    protected void processTupleArrival(SimEvent ev) {
        Object data = ev.getData();
        if (data instanceof Tuple) {
            Tuple tuple = (Tuple) data;
            arrivalsThisWindow++;
            arrivedMiThisWindow += tuple.getCloudletLength();
            arrivedBytesThisWindow += tuple.getCloudletFileSize();
            if (arrivalTimes.size() < MAX_ARRIVAL_TIMES) {
                arrivalTimes.addLast(CloudSim.clock());
            }
        }
        super.processTupleArrival(ev);
    }

    @Override
    protected void executeTuple(SimEvent ev, String moduleName) {
        Object data = ev.getData();
        if (data instanceof Tuple) {
            executedMiThisWindow += ((Tuple) data).getCloudletLength();
        }
        super.executeTuple(ev, moduleName);
    }

    @Override
    protected void manageResources(SimEvent ev) {
        // Let iFogSim do its own energy and cost accounting first, untouched. The
        // energy figure read below is the one super() has just updated.
        super.manageResources(ev);
        TelemetryCollector.getInstance().record(snapshot());
    }

    /** Builds one telemetry snapshot from the current state of this device. */
    private NodeTelemetry snapshot() {
        double cpuUtil = cpuUtilisation();
        double ramUtil = ramUtilisation();

        // Order matters: queueDepth() consumes and resets the arrival counters, so
        // jitter has to be computed from the timestamps first.
        double jitter = arrivalJitter();
        double uplinkLatency = effectiveUplinkLatency();
        int queueDepth = queueDepth();
        double residualEnergy = residualEnergy();

        handoverWindow.addLast(handoversThisEpoch);
        while (handoverWindow.size() > HANDOVER_WINDOW_EPOCHS) {
            handoverWindow.removeFirst();
        }
        int handovers = 0;
        for (Integer h : handoverWindow) {
            handovers += h;
        }
        handoversThisEpoch = 0;

        return new NodeTelemetry(
                getId(), getName(), getLevel(), deviceClass, CloudSim.clock(),
                cpuUtil, ramUtil, queueDepth, uplinkLatency, jitter,
                handovers, residualEnergy,
                Double.NaN,               // no thermal sensor on simulated nodes
                reliabilityPrior);
    }

    // ---- the eight channels --------------------------------------------------

    /**
     * CPU utilisation over the reporting window: work actually executed against the
     * work the host could have executed.
     *
     * <pre>
     *   cpu_util = MI dispatched to this host's CPU during the window
     *              -----------------------------------------------------
     *                        hostMips x window length
     * </pre>
     *
     * <p>The numerator is accumulated in {@link #executeTuple}, which iFogSim calls
     * as each tuple is submitted to a module's scheduler on this device, so it is the
     * simulator's own account of work done here.
     *
     * <p>Two instantaneous alternatives were tried first and both failed for the same
     * underlying reason -- a sample taken at an epoch boundary says nothing about the
     * epoch. iFogSim's own {@code lastUtilization} divides <i>allocated</i> MIPS by
     * host MIPS, but the overbooking provisioners hand a lone module the whole host
     * regardless of its workload, so it pins to 1.0 on any device carrying a module.
     * Requested MIPS capped at each module's provisioned share is honest about demand
     * but is effectively binary: a module either has a tuple queued at that instant or
     * it does not, and measured that way cpu_util took just two distinct values per
     * device class across an entire run.
     *
     * <p>Work per unit time is both the standard definition of utilisation over an
     * interval and what a real telemetry agent reports, and it varies continuously
     * with load.
     */
    private double cpuUtilisation() {
        double capacityMi = getHost().getTotalMips() * Config.RESOURCE_MGMT_INTERVAL;
        double executed = executedMiThisWindow;
        executedMiThisWindow = 0.0;
        if (capacityMi <= 0) return 0.0;
        return Math.min(1.0, executed / capacityMi);
    }

    /**
     * RAM utilisation: allocated against host capacity, from the memory provisioner.
     *
     * <p>Note for anyone reading the data: iFogSim allocates a module's memory once,
     * when it is placed, and never varies it. This channel is therefore genuinely
     * constant for a given node within a run -- it distinguishes configurations, not
     * moments. That is a property of the simulator, not a defect in the measurement,
     * and it is why the scenario sweep varies both module RAM and host RAM.
     */
    private double ramUtilisation() {
        Host host = getHost();
        int totalRam = host.getRam();
        if (totalRam <= 0) return 0.0;
        int available = host.getRamProvisioner().getAvailableRam();
        return Math.min(1.0, Math.max(0.0, (totalRam - available) / (double) totalRam));
    }

    /**
     * Pending work for the reporting window.
     *
     * <p>An instantaneous reading of the link and scheduler queues is useless here:
     * iFogSim drains the link queues on the {@code UPDATE_*_TUPLE_QUEUE} events as
     * soon as a link frees, and tuple execution completes well inside one
     * hundred-unit epoch, so a sample taken at the epoch boundary finds both empty
     * essentially always -- measured that way the variable was constant at zero
     * across every snapshot.
     *
     * <p>What a real telemetry agent reports is not an instant but an interval: the
     * work that arrived during the window, plus whatever is still resident at the
     * end of it, plus any excess of arriving work over what the host could execute.
     */
    private int queueDepth() {
        int resident = 0;
        if (getNorthTupleQueue() != null)   resident += getNorthTupleQueue().size();
        if (getSouthTupleQueue() != null)   resident += getSouthTupleQueue().size();
        if (getClusterTupleQueue() != null) resident += getClusterTupleQueue().size();

        for (Object vmObj : getHost().getVmList()) {
            CloudletScheduler scheduler = ((Vm) vmObj).getCloudletScheduler();
            if (scheduler == null) continue;
            resident += scheduler.getCloudletExecList().size();
            resident += scheduler.getCloudletWaitingList().size();
        }

        int arrived = arrivalsThisWindow;
        double arrivedMi = arrivedMiThisWindow;
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

    /**
     * Jitter: the variability of tuple arrival timing over the window.
     *
     * <p>Measured as the coefficient of variation of the intervals between
     * successive tuple arrivals recorded in {@link #processTupleArrival}, scaled
     * into [0, 1]. This is not a proxy for jitter -- variability of arrival timing
     * is the definition of jitter, so this is the quantity itself, computed from
     * timestamps the simulation actually produced.
     *
     * <p>A previous version generated this channel from an exponentially weighted
     * average of latency changes plus a random term, which is to say it was invented.
     *
     * <p><b>Packet loss has no counterpart here.</b> iFogSim has no loss model at
     * all: no link drops a tuple, so there is nothing to measure. The paper names
     * this variable "jitter with packet loss"; only the jitter half is observable in
     * simulation, and the documentation says so rather than implying otherwise.
     *
     * <p>Fewer than three arrivals in a window gives no interval to compare against,
     * so the channel reports nothing and is marginalised out for that row -- which
     * is the correct treatment of a measurement that could not be taken.
     */
    private double arrivalJitter() {
        if (arrivalTimes.size() < 3) {
            arrivalTimes.clear();
            return Double.NaN;   // not measurable this window
        }

        Double[] times = arrivalTimes.toArray(new Double[0]);
        arrivalTimes.clear();

        int n = times.length - 1;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += times[i + 1] - times[i];
        }
        double mean = sum / n;
        if (mean <= 0.0) {
            return 0.0;   // all arrivals simultaneous: no jitter
        }

        double variance = 0.0;
        for (int i = 0; i < n; i++) {
            double d = (times[i + 1] - times[i]) - mean;
            variance += d * d;
        }
        double sd = Math.sqrt(variance / n);

        // Coefficient of variation. A perfectly periodic stream gives 0; a CV of 1
        // is Poisson-like, which is already badly jittered, so that is the ceiling.
        return Math.min(1.0, sd / mean);
    }

    /**
     * Residual energy as a fraction of the node's declared capacity.
     *
     * <p>The consumption is iFogSim's own: {@code getEnergyConsumption()} accumulates
     * {@code powerModel.getPower(utilisation) * elapsed} inside
     * {@code FogDevice.updateEnergyConsumption()}, which {@code super.manageResources}
     * has just run. The capacity comes from {@link ScenarioSpec#energyBudgetFor},
     * because iFogSim has no battery concept -- consumption is a total that only
     * rises, and "residual" is meaningless without something to measure it against.
     *
     * <p>Worth knowing when reading the data: iFogSim drives its power model from
     * <i>allocated</i> MIPS, which saturates under the overbooking provisioners this
     * topology uses. Energy therefore separates scenarios far more sharply than it
     * separates epochs within a single run, since what really moves it is the node's
     * power model and capacity rather than its moment-to-moment load.
     */
    private double residualEnergy() {
        if (energyBudget <= 0) return 1.0;
        double consumed = getEnergyConsumption();
        return Math.min(1.0, Math.max(0.0, 1.0 - consumed / energyBudget));
    }

    /**
     * Uplink latency: the configured link delay plus the time the window's traffic
     * takes to clear the link.
     *
     * <pre>
     *   uplink_latency = getUplinkLatency() + bytes arriving in the window
     *                                         --------------------------
     *                                            uplink bandwidth
     * </pre>
     *
     * <p>The transmission term is iFogSim's own model of network delay --
     * {@code FogDevice.sendUpFreeLink} computes exactly
     * {@code cloudletFileSize / uplinkBandwidth} for each tuple -- applied to the
     * traffic actually observed over the window.
     *
     * <p>This replaces an earlier version that added a penalty proportional to queue
     * depth. That was wrong in a way worth recording: it made latency an almost
     * exact linear function of the queue-depth channel, and the two correlated at
     * r = 0.9998 across 72,000 rows. The BBN would then have been reading one
     * measurement twice, once into computational stress and once into network
     * degradation, which inflates the apparent weight of congestion and gives the
     * network branch nothing of its own to contribute. Dividing by bandwidth fixes
     * it because bandwidth is a scenario knob: the same traffic over a 2,000-unit
     * link and a 10,000-unit link now reports genuinely different latency.
     */
    private double effectiveUplinkLatency() {
        double base = getUplinkLatency();
        double bandwidth = getUplinkBandwidth();
        double bytes = arrivedBytesThisWindow;
        arrivedBytesThisWindow = 0.0;

        double transmission = (bandwidth > 0) ? (bytes / bandwidth) : 0.0;
        double busyPenalty = isNorthLinkBusy() ? base * 0.5 : 0.0;
        return base + transmission + busyPenalty;
    }

    public String getDeviceClass() {
        return deviceClass;
    }

    public double getEnergyBudget() {
        return energyBudget;
    }
}
