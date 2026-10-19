package org.fog.ft;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.entities.Actuator;
import org.fog.entities.Tuple;
import org.fog.utils.FogEvents;
import org.fog.utils.TimeKeeper;

/**
 * An {@link Actuator} that records the end-to-end loop delay of every alarm it receives,
 * per user.
 *
 * <p>iFogSim keeps only a running average per application loop. Fault tolerance is
 * judged on tails and on deadline violations of individual users, so the delay of each
 * VITALS -> client -> ecg_analyser -> client -> ALARM loop is read here, from the same
 * emit-time table iFogSim uses, just before the upstream actuator consumes the entry.
 */
public class InstrumentedActuator extends Actuator {

    private final int userEdgeId;
    private final double deadline;

    public InstrumentedActuator(String name, int userId, String appId, String actuatorType,
                                int userEdgeId, double deadline) {
        super(name, userId, appId, actuatorType);
        this.userEdgeId = userEdgeId;
        this.deadline = deadline;
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == FogEvents.TUPLE_ARRIVAL && ev.getData() instanceof Tuple) {
            Tuple tuple = (Tuple) ev.getData();
            Double start = TimeKeeper.getInstance().getEmitTimes().get(tuple.getActualTupleId());
            if (start != null) {
                ServiceSession s = SessionRegistry.get().session(userEdgeId);
                if (s != null) {
                    s.recordLoopDelay(CloudSim.clock() - start, deadline);
                }
            }
        }
        super.processEvent(ev);
    }
}
