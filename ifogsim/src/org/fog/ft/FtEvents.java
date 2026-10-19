package org.fog.ft;

import org.cloudbus.cloudsim.core.CloudSimTags;

/** Event tags of the fault-tolerance extension. Nothing upstream is modified. */
public enum FtEvents implements CloudSimTags {
    /** A device's scheduled failure instant has arrived (device to itself). */
    NODE_FAIL,
    /** The controller's per-epoch decision loop. */
    FT_EPOCH,
    /** Failure detection: heartbeat timeout (or link loss) for a failed node. */
    FT_DETECT,
    /** A proactive migration finishes its state transfer and switches over. */
    FT_SWITCHOVER
}
