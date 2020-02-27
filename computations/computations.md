# Computations

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com)                 |
| **Sponsor**   |                                                      |
| **Updated**   | YYYY-MM-DD                                           |

## Objective

- Front load what everyone involved in a computation is expected to do. This can for instance be used to assure a data owner that their dataset will only be used for encrypted training (and not extracted).

- Offer concrete object for analysis.

- Computations contain all logic needed for a set of workers to choreograph an execution.

## Motivation

## Design Proposal

### Still valid

- Computations orchestrates the execution of functions on encrypted data
- Python eDSL is functional but computations are DAGs
- Nodes in computations may call functions (potentially only by name)
- In logical computations functions must be assigned to a single (potentially composite) device; also makes 'with' Python context handler simpler since it makes sense for both execution and destination
- Policy as code; can use eg secrecy property of values and owner properties of devices
- functional paradigm (eg no loops) is nice for MPC: via DatasetSpec with finite length we can define offline phase; how to deal with unbounded?
- concrete computations can be queried with corruption scenario, revealing whether security is breached
- all state lives outside computations, ie created and owned by a device
- we don't want dependent types (for sequences), but rather specs, to avoid complex type checking in the former; enough with syntactic equality for equality?
- computations can be run on sequence; training on a dataset (sequence) of batches is a reduce; same as IterativeProcess in TFF
- A computation may put requirements on the device through DeviceSpec
- networking can be either push or pull, as determined by runtime
- all devices have an attached owner (potentially composite for eg Pond) with strong identity (digital signature)
- networked computations may not use composite devices
- values have a classification which may be enforced both at compile- and run-time
- devices must prove player ownership (via identity signature)
- computations may reference external values (eg models weights) and functionality (eg functions and data connectors)


- a computation is submitted to run in a union/setup, is it *not* launched by a single device; this means that device annotations are simply used to specify some of the players involved
- a driver may choose to create some devices at runtime (eg EnclaveDevice)

### Computation objects

Computations are DAGs stored as protobuf files. Every node has an assigned operation and an assigned device, and every edge represents either a tensor or a sequence (of tensors). Computations are purely functional and without side-effects, and any assignment to variables must happen elsewhere; this ensures that it is safe to call unknown sub-computations on your data?

```python
@dataclass
class Node:
  op: Operation
  device: Device
  inputs: ...
```

Computations are effectively parameterized by the devices and players contained within them, both typically captured through _specifications_ to stick to TensorFlow terminology.

### Devices

Every device must have an associated device type.

```python
@dataclass
class Device:
  device_type: DeviceType
  owner: ...
```

We expect new device types to be introduced frequently, e.g. as part of introducing new schemes or as an abstraction mechanism for new analysis methods. Some device types are intended to be used only as part of the compilation process, which we call _virtual_ (e.g. `PondDevice`). Others are instead intended to be used during execution, with one or more executors be able to execute the operations assigned to them; we call these _concrete_ (e.g. `EnclaveDevice`). During the compilation process we typically replace virtual devices with concrete devices, with the final graph containing only the latter.

Note that virtual devices are a form of _ideal functionalities_ as used in cryptographic protocol theory to assist modular analysis of complex protocols. Note also that executors for virtual devices may exist, yet these are typically intended only for (insecure) development and testing.

## Detailed Design

## Questions and Discussion Topics
