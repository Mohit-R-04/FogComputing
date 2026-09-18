# Fog Computing Dictionary

## Architecture

```text
Cloud
  ↑ north / upstream
Fog node
  ↑ north / upstream
Edge device
```

| Term | Meaning |
|---|---|
| **Parent** | The device one level closer to the cloud. Example: `edge-3` parent is `fog-12`. |
| **Child** | The device one level closer to the edge. Example: `fog-12` child is `edge-3`. |
| **North** | Direction toward the cloud; in this project, north means upstream. |
| **South** | Direction toward the edge; in this project, south means downstream. |
| **Upstream** | Data moving from edge to fog to cloud: `edge → fog → cloud`. |
| **Downstream** | Data moving from cloud to fog to edge: `cloud → fog → edge`. |
| **Uplink** | A network connection from a device to its parent, such as `edge → fog`. |
| **Downlink** | A network connection from a parent to its child, such as `fog → edge`. |
| **North link** | The connection from a device toward its parent/cloud. |
| **North queue** | Tuples waiting to move toward the parent or cloud. |
| **South queue** | Tuples waiting to move toward a child or edge device. |
| **Handover** | When an edge device changes its serving fog node, for example `edge-3: fog-12 → fog-15`. |
| **`setParentId()`** | The iFogSim method used to assign or change a device's parent. |
| **`getUplinkLatency()`** | The configured delay for communication toward the parent. |
| **`getUplinkBandwidth()`** | The available bandwidth for communication toward the parent. |
| **`isNorthLinkBusy()`** | Checks whether the upward link toward the parent is busy. |

There is no geographic north or south here. They are only names for directions in the hierarchical fog topology.

## Tuple

A **tuple** is one simulated IoT message or computational task moving through the system. For example, a heart-rate sensor creates a tuple, which travels from an edge device to a fog node and possibly to the cloud.

A tuple consumes CPU, memory, network bandwidth, queue space, transmission time, and energy. `InstrumentedFogDevice` measures the effect of many tuples and stores the result as one telemetry row; a tuple is not the same as a CSV row.

| Tuple term | Meaning |
|---|---|
| **`Tuple`** | iFogSim's simulated data message or processing task (`org.fog.entities.Tuple`). |
| **Tuple arrival** | A tuple reaches a device and is added to its processing/network flow. |
| **Tuple execution** | The device's CPU processes the tuple. |
| **`getCloudletLength()`** | The tuple's computational workload in MI (Million Instructions). |
| **`getCloudletFileSize()`** | The tuple's data size used to calculate transmission time. |
| **Tuple path** | Usually `sensor → edge → fog → cloud`, depending on application placement. |
| **Telemetry row** | A measurement of a node during an epoch, summarising the effect of multiple tuples. |

## Simple data flow

```text
Sensor creates tuple
        ↓
Edge device receives tuple
        ↓
Fog node processes or forwards tuple
        ↓
Cloud receives tuple or response
        ↓
InstrumentedFogDevice records CPU, queue, latency, jitter, and energy
```
