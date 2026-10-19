package org.fog.ft;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.mobilitydata.Location;
import org.fog.utils.Config;
import org.fog.utils.FogEvents;

/**
 * Walks the user devices along the Melbourne CBD mobility traces and asks the
 * fault-tolerance controller which fog node should serve each of them.
 *
 * <p>The trajectories are the real traces shipped with iFogSim2
 * ({@code dataset/random_usersLocation-melbCBD_*.csv}, 100 waypoints each) and the fog
 * coordinates come from {@code dataset/edgeResources-melbCBD.csv}. Without movement no
 * device ever changes parent, so handover rate is identically zero and uplink latency
 * never departs from its configured constant.
 *
 * <p>Which node serves a user is a policy decision, so it is delegated to
 * {@link FaultToleranceController#chooseServingNode}: the baselines use the nearest node
 * that is not known to be dead, while the proposed policy uses the risk-aware placement
 * utility of Eq. (2). Every change of serving node is a genuine iFogSim topology change
 * and, for policies that keep state at the fog, a session migration.
 */
public class MobilityDriver extends SimEntity {

    /** How often users move, in simulation time units. */
    public static final double MOBILITY_INTERVAL = 100.0;

    private final List<InstrumentedFogDevice> edgeDevices;
    private final Topology topology;
    private final Map<Integer, List<Location>> traces = new HashMap<Integer, List<Location>>();
    private final Map<Integer, Integer> traceCursor = new HashMap<Integer, Integer>();
    private FaultToleranceController controller;
    private int handoverCount = 0;

    public MobilityDriver(String name, List<InstrumentedFogDevice> edgeDevices, Topology topology,
                          long seed) throws Exception {
        super(name);
        this.edgeDevices = edgeDevices;
        this.topology = topology;
        loadTraces(seed);
    }

    public void setController(FaultToleranceController controller) {
        this.controller = controller;
    }

    /** Each device follows one of the five traces, entered at a seeded random offset. */
    private void loadTraces(long seed) throws Exception {
        Random rng = new Random(seed);
        List<List<Location>> pool = new ArrayList<List<Location>>();
        for (int i = 1; i <= 5; i++) {
            List<Location> trace = readTrace("./dataset/random_usersLocation-melbCBD_" + i + ".csv");
            if (!trace.isEmpty()) pool.add(trace);
        }
        if (pool.isEmpty()) {
            throw new IllegalStateException("no mobility traces found under ./dataset/");
        }
        for (int i = 0; i < edgeDevices.size(); i++) {
            InstrumentedFogDevice d = edgeDevices.get(i);
            List<Location> trace = pool.get(i % pool.size());
            traces.put(d.getId(), trace);
            int cursor = rng.nextInt(trace.size());
            traceCursor.put(d.getId(), cursor);
            topology.setUserLocation(d.getId(), trace.get(cursor));
        }
    }

    private static List<Location> readTrace(String path) throws Exception {
        List<Location> out = new ArrayList<Location>();
        BufferedReader r = new BufferedReader(new FileReader(path));
        try {
            String line;
            while ((line = r.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 2) continue;
                try {
                    out.add(new Location(Double.parseDouble(parts[0]), Double.parseDouble(parts[1]), -1));
                } catch (NumberFormatException ignored) {
                    // header row
                }
            }
        } finally {
            r.close();
        }
        return out;
    }

    /** Initial association, before the first telemetry epoch. Called during setup. */
    public void associateInitially() {
        for (InstrumentedFogDevice edge : edgeDevices) {
            InstrumentedFogDevice target = controller.chooseServingNode(edge, true);
            if (target != null) {
                topology.reassociate(edge, target);
                controller.onInitialAssociation(edge, target);
            }
        }
    }

    @Override
    public void startEntity() {
        send(getId(), MOBILITY_INTERVAL, FogEvents.MOBILITY_MANAGEMENT);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == FogEvents.MOBILITY_MANAGEMENT) {
            step();
            if (CloudSim.clock() + MOBILITY_INTERVAL < Config.MAX_SIMULATION_TIME) {
                send(getId(), MOBILITY_INTERVAL, FogEvents.MOBILITY_MANAGEMENT);
            }
        }
    }

    /** Advances every user one waypoint and lets the controller re-decide its node. */
    private void step() {
        for (InstrumentedFogDevice edge : edgeDevices) {
            if (edge.isFailed()) continue;
            List<Location> trace = traces.get(edge.getId());
            if (trace == null || trace.isEmpty()) continue;
            int cursor = (traceCursor.get(edge.getId()) + 1) % trace.size();
            traceCursor.put(edge.getId(), cursor);
            topology.setUserLocation(edge.getId(), trace.get(cursor));

            InstrumentedFogDevice target = controller.chooseServingNode(edge, false);
            if (target == null || target.getId() == edge.getParentId()) {
                topology.refreshLatency(edge);
                continue;
            }
            if (controller.onMobilityHandover(edge, target)) {
                handoverCount++;
            }
        }
    }

    public int getHandoverCount() {
        return handoverCount;
    }

    @Override
    public void shutdownEntity() {
    }
}
