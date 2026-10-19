package org.fog.ft;

import java.util.List;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.Sensor;
import org.fog.placement.Controller;
import org.fog.utils.FogEvents;

/**
 * iFogSim's master {@link Controller}, minus its ending.
 *
 * <p>Upstream, STOP_SIMULATION prints some 170 lines of per-device statistics and then
 * calls {@code System.exit(0)}, so nothing after {@code CloudSim.startSimulation()} ever
 * runs. Here it ends the event loop instead, records the end time, and lets the caller
 * write its own outputs. Everything else is inherited unchanged.
 */
public class QuietController extends Controller {

    private static double endTime = Double.NaN;

    public QuietController(String name, List<FogDevice> fogDevices, List<Sensor> sensors,
                           List<Actuator> actuators) {
        super(name, fogDevices, sensors, actuators);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == FogEvents.STOP_SIMULATION) {
            endTime = CloudSim.clock();
            CloudSim.abruptallyTerminate();
            return;
        }
        super.processEvent(ev);
    }

    public static double endTime() {
        return endTime;
    }
}
