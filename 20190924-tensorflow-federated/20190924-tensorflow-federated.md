# Integrating TF Encrypted and TensorFlow Federated

| Status        | Proposed |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com) |
| **Sponsor**   | |
| **Updated**   | 2019-10-15 |

## Objective

This document describes an integration between TF Encrypted (TFE) and TensorFlow Federated (TFF), with the former providing secure computation services for the latter. The use of TFE for secure aggregation in real-world scenarios is the driving motivation for this proposal. All integration code is proposed to be located in the `tfe.integrations.federated` Python module.

This proposal and its understanding of TFF is based on [release 0.9.0](https://github.com/tensorflow/federated/releases/tag/v0.9.0). Please see [Inside TensorFlow Federated](./inside-tensorflow-federated.md) for background information.

## Motivation

We wish to support the following scenarios for secure aggregation:

- performed by distinct set of aggregators  
- performed by clients themselves
- using full suite of secure computation including enclaves, MPC, HE, and GC
- using specialized protocols such as SDA and Google
- using real-world infrastructure such as bulletin boards

## Design Proposal

There are at least two different ways of adding support for secure aggregation to TFF:

1. by expressing secure aggregation protocols directly as federated algorithms in the language of TFF;
2. by re-implementing certain intrinsic functions to use secure computation.

It is not clear that the former approach fits with the [overall aim](https://github.com/tensorflow/federated/blob/v0.9.0/docs/tutorials/custom_federated_algorithms_1.ipynb#design-overview) of TFF at a conceptual level since aggregation algorithms are all currently implemented as intrinsic functions and secure computation protocols inherently need to express behavior at a more fine-grained per-player level. At the technical level, the currently implementation also lacks for instance types to represent encrypted data and computations, which would hence have to be expressed in raw forms. One advantage of the approach is that secure aggregation is a highly integrated part of federated algorithms, allowing developers to stay in the same framework if they have to develop custom protocols.

The potential disadvantages of the second approach is that secure aggregation happens at a lower level and remain opaque to the federated algorithms expressed in TFF. This requires knowledge of another framework (TFE) when custom protocols have to be developed, and care must be taken to prevent accidentally leaking too much information through other channels; concretely, care much be taken when specifying the executor stack, as using the wrong executor could effectively disable security. One advantage of this approach is the extra control over how secure computations are specified and executed. Another is the modularity that comes from keeping TFF operating at a higher level of abstraction, ideally making it easy to test algorithms independently of the concrete aggregation method used. [Hints in the code](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/proto/v0/computation.proto#L625) seem to suggest this approach.

The second approach can be seen as using secure aggregation as a service, and may be implemented using custom executors on potentially both server and clients. It can be done:

- internally, where the secure computation is fully embedded in TFF and performed using its computation, communication, and orchestration mechanisms only;
- externally, where TFF may e.g. be used only for orchestrating input and output to an external service run by a set of players opaque to the federated algorithms.

Note that:

- the internal strategy is not the same as the approach of expressing secure aggregation as federated algorithms since an executor on e.g. the server knows the exact players at runtime and may schedule TF computations independently on each;
- both strategies may require additional functionality in e.g. the EagerExecutor, such as setting up distributed computations or performing encryption;
- the external strategy does not necessarily imply a distinct set of players, as the same set may e.g. run both the TFF service and the TFE service.

The external strategy may turn out to be the best fit for employing certain robust or large-scale aggregation systems such as SDA and Google's protocol involving e.g. bulletin boards. One advantage of the internal strategy is that everything is contained within TFF and no other services have to be understood. One disadvantage is that secure computations have to be compiled down to the TFF computation format, which in the external strategy is only the case for logic dealing with input and output; this may make the internal strategy more suitable for computations with relatively simple communication patterns (e.g. based on HE) that can be completely implemented inside the executor, whereas the external strategy may be more suitable for complex computations compiled by TFE; in an ideal world TFE could automatically compile down to TFF.

In the internal strategy, players may communicate sensitive values through the server using e.g. public-key encryption exposed as a primitive through TFE. In the external strategy, this can also happen directly between the players by e.g. using TFE to configure and run the TF distributed engine with a cluster specification.

## Detailed Design Proposal

We here detail several examples of concrete secure aggregation protocols via custom executors implementing certain intrinsic functions.

### Internal Secure Aggregation Service using Homomorphic Encryption

This example consists of the following players: a server, a set of clients, and an aggregator holding the decryption key. Note that the aggregator does not have a formal placement but is e.g. an EagerExecutor referenced through a RemoteExecutor by the custom executor on the server.

The custom executor performs the following steps to compute an aggregation supported by the HE scheme:

1. Ask the aggregator to generate an encryption keypair; this can be done using a TF computation calling a TFE primitive.
2. Collect the public key and broadcast to clients; this can be done using a move and a plaintext implementation of `federated_broadcast`.
3. Encrypt the values to aggregate on the clients; this can be done using a plaintext implementation of `federated_map` with a TF computation calling a TFE primitive.
4. Collect the encrypted values and perform aggregation; this can done using moves and a TF computation calling a local TFE protocol.
5. Send aggregated encrypted value to aggregator and decrypt; this can be done using a move and a TF computation calling a TFE primitive.
6. Collect aggregated value; this can be done using a move.

```python
decryptor = tfe.Player()
aggregator = tfe.Player()

# keypair generation on decryptor

@tff.tf_computation
def setup():
  keypair = tfe.protocols.paillier.generate_key()
  return keypair.as_string_tensor()

setup_graph = setup.get_concrete_function().graph_def

# graph for exporting public key

@tf.function
def export_public_key():


# NOTE: to be run on each client
@tf.function
def encrypt_input(x):
  x_encrypted = tfe.protocols.paillier.
  


```

```python
decryptor = tfe.Player()
aggregator = tfe.Player()

# keypair generation on decryptor

@tf.function
def setup():
  keypair = tfe.protocols.paillier.generate_key()
  return keypair.as_string_tensor()

setup_graph = setup.get_concrete_function().graph_def

# graph for exporting public key

@tf.function
def export_public_key():


# NOTE: to be run on each client
@tf.function
def encrypt_input(x):
  x_encrypted = tfe.protocols.paillier.
  


```


```python
decryptor = tfe.Player()
aggregator = tfe.Player()

@tfe.computation
def encrypt_input(x):
  return tfe.encrypt(x)

@tfe.encrypted_computation(aggregator)
def secure_mean(xs):
  return tfe.add_n(xs) / len(xs)

@tfe.computation
def decrypt_output(y):
  return tfe.decrypt(y)

protocol = tfe.protocols.Paillier(decryptor)

# generate keypair on decryptor;
# this is a tfe.Computation that can be serialized as a TF
# graph returning a variant tensor representing the keypair;
# this may have to be a string tensor to work with TFF
protocol.setup()

@tfe.tf_computation(decryptor)
def setup():
  keypair = tfe.protocols.paillier.generate_key()



```

Possible issues that need to be addressed:

- Will any part of TFF complain about the aggregator not having a formal placement?
- How can the aggregator verify the identity of the clients?
- How can the clients verify the identity of the aggregator?
- How are ground truth about identities defined and distributed?
- How can TFE be used to specify the HE aggregation function?

### External Secure Aggregation Service using Secret Sharing with Direct Communication and Queues

This example consists of the following players: a server, a set of clients, and a set of aggregators. Note that the aggregators does not have a formal placement and only need to run a TFE service.

The server executor performs the following steps to compute an aggregation supported by the secret sharing scheme:

1. Use TFE to set up a secure computation graph on the aggregators, using distinct queues for inputs and outputs.
2. Send TF computations to the clients that compute and send shares to the queues using the TF distributed engine.
3. Perform a blocking poll on output queues until shares of result are available.

Possible issues that need to be addressed:

- How is TFE used to set up the secure computation?
- Can we have a TF computation with side-effects and no return value?
- How do we set up secure channels between clients and aggregators?
- EagerExecutors must be configured to use the distributed engine to talk to queues.
- Enqueue synchronization when several clients push inputs concurrently; relevant when using composite tensors such as in secret sharing.
- Access control on the graph nodes, in particular the queues; this requires some notion of strong player identity:
  - The aggregators can verify the computation/graph being constructed on them.
  - No-one can push additional nodes to the graph, including the aggregators.
  - No-one besides the aggregators can read from the input queues.
  - Only clients can push to the input queues.
  - Each client can only push once to the input queues.
  - Only the server can read from the output queue.
- New executors are created for every call by the current ExecutionContext; secure computation service must be enable to handle this one way or another.

## Questions and Discussion Topics

See issues inlined in the [detailed design proposal](#detailed-design-proposal).
