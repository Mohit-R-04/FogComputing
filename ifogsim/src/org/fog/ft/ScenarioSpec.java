package org.fog.ft;

/** One three-tier simulation configuration: cloud, fog nodes and edge devices. */
public class ScenarioSpec {
    public final int id;
    public final String label;

    // Tier capacity: level 0 = cloud, 1 = fog, 2 = edge.
    public final long cloudMips;
    public final int cloudRam;
    public final long fogMips;
    public final int fogRam;
    public final long edgeMips;
    public final int edgeRam;

    // Power model per tier: {busy, idle} watts.
    public final double[] cloudPower;
    public final double[] fogPower;
    public final double[] edgePower;

    // Edge workload.
    public final int edgeCount;
    public final int sensorsPerEdge;
    public final double sensorInterval;

    // Application module sizing.
    public final int clientMips;
    public final int clientRam;
    public final int analyserMips;
    public final int analyserRam;
    public final double analysisTupleMi;

    // Cloud/fog link settings.
    public final double fogUplinkLatency;
    public final long fogUplinkBw;
    public final int simulationTime;

    public ScenarioSpec(int id, String label,
                        long cloudMips, int cloudRam,
                        long fogMips, int fogRam,
                        long edgeMips, int edgeRam,
                        double[] cloudPower, double[] fogPower, double[] edgePower,
                        int edgeCount, int sensorsPerEdge, double sensorInterval,
                        int clientMips, int clientRam,
                        int analyserMips, int analyserRam, double analysisTupleMi,
                        double fogUplinkLatency, long fogUplinkBw, int simulationTime) {
        this.id = id;
        this.label = label;
        this.cloudMips = cloudMips;
        this.cloudRam = cloudRam;
        this.fogMips = fogMips;
        this.fogRam = fogRam;
        this.edgeMips = edgeMips;
        this.edgeRam = edgeRam;
        this.cloudPower = cloudPower;
        this.fogPower = fogPower;
        this.edgePower = edgePower;
        this.edgeCount = edgeCount;
        this.sensorsPerEdge = sensorsPerEdge;
        this.sensorInterval = sensorInterval;
        this.clientMips = clientMips;
        this.clientRam = clientRam;
        this.analyserMips = analyserMips;
        this.analyserRam = analyserRam;
        this.analysisTupleMi = analysisTupleMi;
        this.fogUplinkLatency = fogUplinkLatency;
        this.fogUplinkBw = fogUplinkBw;
        this.simulationTime = simulationTime;
    }

    public long mipsFor(int level) {
        switch (level) {
            case 0: return cloudMips;
            case 1: return fogMips;
            default: return edgeMips;
        }
    }

    public int ramFor(int level) {
        switch (level) {
            case 0: return cloudRam;
            case 1: return fogRam;
            default: return edgeRam;
        }
    }

    public double[] powerFor(int level) {
        switch (level) {
            case 0: return cloudPower;
            case 1: return fogPower;
            default: return edgePower;
        }
    }

    /** Declared capacity; consumption itself comes from iFogSim. */
    public double energyBudgetFor(int level) {
        double[] power = powerFor(level);
        double headroom = level == 2 ? 1.05 : 4.0;
        return power[0] * simulationTime * headroom;
    }

    @Override
    public String toString() {
        return String.format(
                "scenario %02d [%s]  fogMips=%d fogRam=%d edgeMips=%d edges=%d "
                + "sensors/edge=%d interval=%.0fms analyserMips=%d tupleMI=%.0f "
                + "fogPower=%.0f/%.0f edgePower=%.0f/%.0f",
                id, label, fogMips, fogRam, edgeMips, edgeCount,
                sensorsPerEdge, sensorInterval, analyserMips, analysisTupleMi,
                fogPower[0], fogPower[1], edgePower[0], edgePower[1]);
    }
}
