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
import org.fog.entities.FogDevice;
import org.fog.mobilitydata.Location;
import org.fog.placement.LocationHandler;
import org.fog.utils.FogEvents;

/**
 * Walks the user devices along the Melbourne CBD mobility traces and re-associates
 * each one with its nearest fog node.
 *
 * <p>This is what makes two of the BBN's evidence variables non-degenerate. Without
 * it no device ever changes parent, so {@code handover_rate} is identically zero and
 * {@code uplink_latency} never departs from its configured constant -- two of eight
 * evidence channels carrying no information at all.
 *
 * <p>The trajectories are the real traces shipped with iFogSim2
 * ({@code dataset/random_usersLocation-melbCBD_*.csv}, 100 waypoints each) and the
 * fog-node coordinates are the real ones from
 * {@code dataset/edgeResources-melbCBD.csv}. Association is nearest-fog by
 * haversine distance, reusing {@link LocationHandler#calculateDistance}. Every
 * re-association is a genuine {@code setParentId} call on the simulated topology,
 * which {@link InstrumentedFogDevice} counts as a handover.
 */
public class MobilityDriver extends SimEntity {

    /** How often users move and re-associate, in simulation time units. */
    public static final double MOBILITY_INTERVAL = 100.0;

    /** Access-link latency floor in ms. */
    private static final double BASE_ACCESS_LATENCY = 1.5;
    /** Additional ms per kilometre of separation from the serving fog node. */
    private static final double LATENCY_PER_KM = 3.2;
    /** Cap so a badly-placed user cannot produce an absurd link delay. */
    private static final double MAX_ACCESS_LATENCY = 60.0;

    private final List<FogDevice> edgeDevices;
    private final List<FogDevice> fogNodes;
    private final Map<Integer, Location> fogLocations;
    private final Map<Integer, List<Location>> traces = new HashMap<Integer, List<Location>>();
    private final Map<Integer, Integer> traceCursor = new HashMap<Integer, Integer>();
    private final Map<Integer, FogDevice> deviceById = new HashMap<Integer, FogDevice>();

    private int handoverCount = 0;

    public MobilityDriver(String name,
                          List<FogDevice> edgeDevices,
                          List<FogDevice> fogNodes,
                          Map<Integer, Location> fogLocations,
                          long seed) throws Exception {
        super(name);
        this.edgeDevices = edgeDevices;
        this.fogNodes = fogNodes;
        this.fogLocations = fogLocations;

        for (FogDevice d : fogNodes) {
            deviceById.put(d.getId(), d);
        }
        for (FogDevice d : edgeDevices) {
            deviceById.put(d.getId(), d);
        }

        loadTraces(seed);
    }

    /**
     * Assigns each user device one of the five shipped traces, entered at a random
     * offset so that forty devices follow forty distinct trajectories.
     */
    private void loadTraces(long seed) throws Exception {
        Random rng = new Random(seed);
        List<List<Location>> pool = new ArrayList<List<Location>>();
        for (int i = 1; i <= 5; i++) {
            List<Location> trace = readTrace("./dataset/random_usersLocation-melbCBD_" + i + ".csv");
            if (!trace.isEmpty()) {
                pool.add(trace);
            }
        }
        if (pool.isEmpty()) {
            throw new IllegalStateException("no mobility traces found under ./dataset/");
        }
        for (int i = 0; i < edgeDevices.size(); i++) {
            FogDevice d = edgeDevices.get(i);
            List<Location> trace = pool.get(i % pool.size());
            traces.put(d.getId(), trace);
            traceCursor.put(d.getId(), rng.nextInt(trace.size()));
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
                    out.add(new Location(Double.parseDouble(parts[0]),
                                         Double.parseDouble(parts[1]), -1));
                } catch (NumberFormatException ignored) {
                    // header row
                }
            }
        } finally {
            r.close();
        }
        return out;
    }

    @Override
    public void startEntity() {
        // Associate everyone before the first telemetry epoch, then step periodically.
        step(false);
        send(getId(), MOBILITY_INTERVAL, FogEvents.MOBILITY_MANAGEMENT);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == FogEvents.MOBILITY_MANAGEMENT) {
            step(true);
            if (CloudSim.clock() < org.fog.utils.Config.MAX_SIMULATION_TIME) {
                send(getId(), MOBILITY_INTERVAL, FogEvents.MOBILITY_MANAGEMENT);
            }
        }
    }

    /**
     * Advances every edge device one waypoint and re-associates it if a nearer fog node is
     * now available.
     *
     * @param countHandovers false during start-up, so initial association is not
     *                       recorded as a handover
     */
    private void step(boolean countHandovers) {
        for (FogDevice edge : edgeDevices) {
            List<Location> trace = traces.get(edge.getId());
            if (trace == null || trace.isEmpty()) continue;

            int cursor = (traceCursor.get(edge.getId()) + 1) % trace.size();
            traceCursor.put(edge.getId(), cursor);
            Location here = trace.get(cursor);

            FogDevice nearest = null;
            double nearestKm = Double.MAX_VALUE;
            for (FogDevice gw : fogNodes) {
                Location gwLoc = fogLocations.get(gw.getId());
                if (gwLoc == null) continue;
                double km = LocationHandler.calculateDistance(here, gwLoc);
                if (km < nearestKm) {
                    nearestKm = km;
                    nearest = gw;
                }
            }
            if (nearest == null) continue;

            double latency = Math.min(MAX_ACCESS_LATENCY,
                    BASE_ACCESS_LATENCY + LATENCY_PER_KM * nearestKm);
            edge.setUplinkLatency(latency);

            int currentParent = edge.getParentId();
            if (currentParent != nearest.getId()) {
                reassociate(edge, currentParent, nearest, latency);
                if (countHandovers) {
                    handoverCount++;
                }
            } else {
                // Same fog node, but distance changed: keep the latency map current.
                nearest.getChildToLatencyMap().put(edge.getId(), latency);
            }
        }
    }

    /**
     * Moves a user device between fogNodes, keeping both parents' bookkeeping right.
     *
     * <p>The old parent's entry in {@code childToLatencyMap} is deliberately left in
     * place. {@code FogDevice.sendDown} checks {@code childrenIds} before queueing a
     * tuple, but {@code updateSouthTupleQueue} drains a queue that was filled before
     * the handover and looks the latency up again at send time -- so dropping the
     * entry strands any tuple already queued for the old parent and throws a
     * NullPointerException inside {@code sendDownFreeLink}. Removing the child from
     * {@code childrenIds} is enough to stop new traffic being routed there; the stale
     * latency entry simply lets in-flight tuples land.
     */
    private void reassociate(FogDevice edge, int oldParentId, FogDevice newParent, double latency) {
        FogDevice oldParent = deviceById.get(oldParentId);
        if (oldParent != null) {
            oldParent.removeChild(edge.getId());
        }
        edge.setParentId(newParent.getId());
        newParent.addChild(edge.getId());
        newParent.getChildToLatencyMap().put(edge.getId(), latency);
    }

    public int getHandoverCount() {
        return handoverCount;
    }

    @Override
    public void shutdownEntity() {
    }
}
