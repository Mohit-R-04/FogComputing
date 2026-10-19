package org.fog.ft;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.fog.mobilitydata.Location;
import org.fog.placement.LocationHandler;

/**
 * The cloud/fog/edge topology as the control plane sees it: where users are, how far
 * each fog node is, what a link to it costs in latency, and how a user is moved from
 * one serving node to another.
 *
 * <p>Re-association is a genuine iFogSim topology change ({@code setParentId},
 * {@code addChild}, {@code removeChild} and the parent's latency map), shared by
 * mobility handovers, proactive migrations and failure recovery.
 */
public final class Topology {

    /** Access-link latency floor in ms. */
    public static final double BASE_ACCESS_LATENCY = 1.5;
    /** Additional ms per kilometre of separation from the serving fog node. */
    public static final double LATENCY_PER_KM = 3.2;
    /** Cap so a badly placed user cannot produce an absurd link delay. */
    public static final double MAX_ACCESS_LATENCY = 60.0;

    /** A candidate serving node for one user. */
    public static final class Candidate {
        public final InstrumentedFogDevice node;
        public final double km;
        public final double accessLatency;

        Candidate(InstrumentedFogDevice node, double km, double accessLatency) {
            this.node = node;
            this.km = km;
            this.accessLatency = accessLatency;
        }
    }

    private final InstrumentedFogDevice cloud;
    private final List<InstrumentedFogDevice> fogNodes;
    private final List<InstrumentedFogDevice> edges;
    private final Map<Integer, Location> fogLocations;
    private final Map<Integer, Location> userLocations = new HashMap<Integer, Location>();
    private final Map<Integer, InstrumentedFogDevice> byId = new HashMap<Integer, InstrumentedFogDevice>();
    /** Latency of a user that bypasses the fog layer and talks to the cloud directly. */
    private final double cloudAccessLatency;

    public Topology(InstrumentedFogDevice cloud, List<InstrumentedFogDevice> fogNodes,
                    List<InstrumentedFogDevice> edges, Map<Integer, Location> fogLocations,
                    double cloudAccessLatency) {
        this.cloud = cloud;
        this.fogNodes = fogNodes;
        this.edges = edges;
        this.fogLocations = fogLocations;
        this.cloudAccessLatency = cloudAccessLatency;
        byId.put(cloud.getId(), cloud);
        for (InstrumentedFogDevice d : fogNodes) byId.put(d.getId(), d);
        for (InstrumentedFogDevice d : edges) byId.put(d.getId(), d);
    }

    public InstrumentedFogDevice cloud() {
        return cloud;
    }

    public List<InstrumentedFogDevice> fogNodes() {
        return fogNodes;
    }

    public List<InstrumentedFogDevice> edges() {
        return edges;
    }

    public InstrumentedFogDevice device(int id) {
        return byId.get(id);
    }

    public void setUserLocation(int edgeId, Location here) {
        userLocations.put(edgeId, here);
    }

    public Location userLocation(int edgeId) {
        return userLocations.get(edgeId);
    }

    public static double accessLatencyForKm(double km) {
        return Math.min(MAX_ACCESS_LATENCY, BASE_ACCESS_LATENCY + LATENCY_PER_KM * km);
    }

    public double distanceKm(int edgeId, InstrumentedFogDevice fog) {
        Location here = userLocations.get(edgeId);
        Location there = fogLocations.get(fog.getId());
        if (here == null || there == null) return Double.MAX_VALUE;
        return LocationHandler.calculateDistance(here, there);
    }

    /** Access latency of a user to a candidate parent (fog node or cloud). */
    public double accessLatency(int edgeId, InstrumentedFogDevice parent) {
        if (parent == cloud) return cloudAccessLatency;
        return accessLatencyForKm(distanceKm(edgeId, parent));
    }

    /**
     * The {@code k} nearest fog nodes the control plane may use for a user: not known
     * to be dead and not excluded. A node that has died but whose failure has not been
     * detected yet is still a candidate -- the control plane cannot know.
     */
    public List<Candidate> nearestCandidates(int edgeId, int k, java.util.Set<Integer> exclude) {
        List<Candidate> all = new ArrayList<Candidate>();
        SessionRegistry registry = SessionRegistry.get();
        for (InstrumentedFogDevice fog : fogNodes) {
            if (registry.isKnownFailed(fog.getId())) continue;
            if (exclude != null && exclude.contains(fog.getId())) continue;
            double km = distanceKm(edgeId, fog);
            if (km == Double.MAX_VALUE) continue;
            all.add(new Candidate(fog, km, accessLatencyForKm(km)));
        }
        Collections.sort(all, new Comparator<Candidate>() {
            @Override
            public int compare(Candidate a, Candidate b) {
                int c = Double.compare(a.km, b.km);
                return c != 0 ? c : a.node.getName().compareTo(b.node.getName());
            }
        });
        return all.size() > k ? new ArrayList<Candidate>(all.subList(0, k)) : all;
    }

    /** Moves a user's edge device under a new parent, keeping both parents' books right. */
    public void reassociate(InstrumentedFogDevice edge, InstrumentedFogDevice newParent) {
        int oldParentId = edge.getParentId();
        if (oldParentId == newParent.getId()) {
            refreshLatency(edge);
            return;
        }
        InstrumentedFogDevice oldParent = byId.get(oldParentId);
        if (oldParent != null) {
            // The old parent's latency entry is deliberately kept: tuples already queued
            // for this child still need it to land (see MobilityDriver history).
            oldParent.removeChild(edge.getId());
        }
        double latency = accessLatency(edge.getId(), newParent);
        edge.setParentId(newParent.getId());
        edge.setUplinkLatency(latency);
        newParent.addChild(edge.getId());
        newParent.getChildToLatencyMap().put(edge.getId(), latency);
    }

    /** Same parent, but the user moved: keep the link latency current. */
    public void refreshLatency(InstrumentedFogDevice edge) {
        InstrumentedFogDevice parent = byId.get(edge.getParentId());
        if (parent == null) return;
        double latency = accessLatency(edge.getId(), parent);
        edge.setUplinkLatency(latency);
        parent.getChildToLatencyMap().put(edge.getId(), latency);
    }
}
