package org.fog.ft;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;

/**
 * Run-wide registry of user sessions and of the module instances that identify them.
 *
 * <p>iFogSim tuples carry no user identity, but every result tuple carries the copy map
 * of the module instances it passed through. The VM id of the {@code client} instance
 * identifies the edge device -- and so the user -- a tuple belongs to. This registry
 * keeps that mapping, the sessions themselves, and the set of nodes the control plane
 * knows to be dead (which is not the same as the set of dead nodes: a failure is only
 * known once it has been detected).
 *
 * <p>Singleton because iFogSim's device callbacks have no natural place to thread a
 * reference through; {@link #reset} must be called before each run.
 */
public final class SessionRegistry {

    private static SessionRegistry instance = new SessionRegistry();

    private final Map<Integer, Integer> clientVmToEdge = new HashMap<Integer, Integer>();
    private final Map<Integer, ServiceSession> sessionsByEdge =
            new LinkedHashMap<Integer, ServiceSession>();
    private final Map<Integer, FogDevice> devices = new HashMap<Integer, FogDevice>();
    private final Set<Integer> knownFailed = new HashSet<Integer>();
    private FaultToleranceController controller;
    private final Map<Integer, Integer> forwards = new HashMap<Integer, Integer>();

    private SessionRegistry() {
    }

    public static SessionRegistry get() {
        return instance;
    }

    public static void reset() {
        instance = new SessionRegistry();
    }

    public void registerDevice(FogDevice d) {
        devices.put(d.getId(), d);
    }

    public FogDevice device(int id) {
        return devices.get(id);
    }

    public Collection<FogDevice> devices() {
        return devices.values();
    }

    public void registerClientVm(int vmId, int edgeId) {
        clientVmToEdge.put(vmId, edgeId);
    }

    public ServiceSession addSession(ServiceSession s) {
        sessionsByEdge.put(s.edgeId, s);
        return s;
    }

    public ServiceSession session(int edgeId) {
        return sessionsByEdge.get(edgeId);
    }

    public List<ServiceSession> sessions() {
        return new ArrayList<ServiceSession>(sessionsByEdge.values());
    }

    /** Edge device a tuple belongs to, from the client instance in its copy map. */
    public int edgeForTuple(Tuple tuple) {
        Integer vm = tuple.getModuleCopyMap() == null ? null
                : tuple.getModuleCopyMap().get("client");
        if (vm != null) {
            Integer edge = clientVmToEdge.get(vm);
            if (edge != null) {
                return edge;
            }
        }
        // Results of the client carry its host as the source device.
        if (sessionsByEdge.containsKey(tuple.getSourceDeviceId())) {
            return tuple.getSourceDeviceId();
        }
        return -1;
    }

    public ServiceSession sessionForTuple(Tuple tuple) {
        int edge = edgeForTuple(tuple);
        return edge < 0 ? null : sessionsByEdge.get(edge);
    }

    /** Counts how often a tuple was forwarded after a handover; returns the new count. */
    public int countForward(int tupleId) {
        Integer c = forwards.get(tupleId);
        int n = c == null ? 1 : c + 1;
        forwards.put(tupleId, n);
        return n;
    }

    public void setController(FaultToleranceController c) {
        this.controller = c;
    }

    public FaultToleranceController controller() {
        return controller;
    }

    /** Called by a device at the instant it dies. */
    public void onDeviceFailed(InstrumentedFogDevice device, double time) {
        if (controller != null) {
            controller.onDeviceFailed(device, time);
        }
    }

    public boolean isKnownFailed(int deviceId) {
        return knownFailed.contains(deviceId);
    }

    public void markKnownFailed(int deviceId) {
        knownFailed.add(deviceId);
    }
}
