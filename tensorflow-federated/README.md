# Integrating TF Encrypted and TensorFlow Federated

| Status        | Draft |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl |
| **Sponsor**   | |
| **Updated**   | 2020-03-10 |

## Objective

This document describes an integration between TF Encrypted (TFE) and TensorFlow Federated (TFF), with the former providing secure computation functionality for the latter. It is being developed for both cross-silo and cross-device federated learning (FL) with the following additional design goals in mind:

- Suitable for both easy experimentation and practical production deployment.
- Support the use of a variety of secure computation techniques, including MPC and HE.
- Support external actors without TFF placements, such as partially-trusted servers or enclaves.

The proposal is based on [TFF 0.12.0](https://github.com/tensorflow/federated/releases/tag/v0.12.0).

## Motivation

Secure aggregation has been part of federated learning since early on (see e.g. [BIK+17](https://eprint.iacr.org/2017/281)), yet remains an open area for experimentation as no single solution is a perfect fit for all environments. For example, secure aggregation protocols optimized for cross-device FL (e.g. large group of volatile mobile devices) are from a cryptographic perspective vastly different than protocols optimized for cross-silo FL (e.g. small set of reliable servers running in a cluster).

The design and analysis of such protocols require expertise and tools that are mostly orthogonal to what is currently available in TFF, and suggests a modular approach where federated algorithms developed and executed with TFF are built on top of secure aggregation protocols developed and executed with TFE.

## Design Proposal

This design introduces federating executors similar to the built-in `FederatingExecutor` but implementing encrypted versions of the supported intrinsic functions such as `tff.federated_secure_sum`, `tff.federated_sum`, and `tff.federated_mean`. In the former case the specific secure aggregation protocol remains opaque from the perspective of the TFF glue language, and in the latter cases it remains opaque that secure aggregation is even used at all.

```python
import tensorflow_federated as tff

# create secure executor stack
executor_factory = tff.framework.create_secure_executor_factory(
    protocol="BIK+17", ...)
tff.framework.set_default_executor(executor_factory)

# develop federated algorithm using TFF glue language
@tff.federated_computation
def my_computation(xs):
  return tff.federated_secure_sum(xs)

# evaluate computations in the normal way
my_computation([1., 2., 3., 4., 5.]))
```

In [phase 1](#roadmap) users can choose between a fixed set of protocols built using TFE primitives, and in [phase 2](#roadmap) we open up for custom protocols expressed via the high-level TFE language; see below for more details.

```python
# develop encrypted computation using TFE high-level language
@tfe.computation
def my_secure_mean(xs):
  cs = [tfe.classify(x, {x.owner}) for x in xs]

  with aggregation_device:
    d = tfe.add_n(cs) / len(cs)
    e = tfe.classify(d, {server_device.owner})

  with server_device:
    y = tfe.declassify(e)

  return y

# create executor based on the encrypted computation
executor_fn = tff.framework.create_secure_executor_factory(
    federated_mean=my_secure_mean, ...)
tff.framework.set_default_executor(executor_fn)
```

As illustrated above, the integration code lives in TFF but relies on cryptographic primitives and tools from TFE. For this reason we add TFE as a dependency to TFF, and hence assume a one-time installation of TFE on all involved parties. Planned work on a TFE core module will make this viable on e.g. non-Python platform such as mobile devices.

### Execution Strategy

For now we embed all protocol steps and their cryptographic operations into [TensorFlow computations](https://github.com/tensorflow/federated/blob/v0.12.0/tensorflow_federated/proto/v0/computation.proto#L99-L103) and use the built-in `EagerTFExecutor` for execution.

This means that no changes to the underlying computational model (`computation.proto`, `executor.proto`, and `Executor`) are needed. On the other hand, it also seems to limit interesting security features such as allowing clients to enforce individual policies on values and (whole) computations, and for the server to move away from a honest-but-curious security model.

We plan on revisiting this strategy in the future once TFF and TF have matured and their directions are clearer. In particular, the TF runtime seems to be undergoing significant changes, and the closed-source version of TFF may already have a suitable design in place given that they likely already implemented e.g. BIK+17.

### Communication Channels

Cryptographic protocols typically require different communication patterns than those found in the TFF glue language (such as `tff.federated_broadcast` and `tff.federated_collect`). In particular, protocols typically dictate that a specific client must send a message to another specific client through an authenticated or secure channel.

For now we only focus on supporting such patterns by encrypting and relaying all messages through the server. This is not an unrealistic assumption in cross-device FL, where direct communication between e.g. mobile devices can be impractical. We use [libsodium](https://github.com/jedisct1/libsodium) primitives exposed by TFE to implement such *indirect* authenticated or secure channels.

Future work may consider alternatives, especially with respect to efficient cross-silo FL and e.g. using gRPC over mTLS.

### Roadmap

We propose the following implementation phases:

1. Implement secure federating executors each supporting a single fixed protocol.
2. Implement a secure federating executor parameterized by protocols expressed as TFE computations.
3. (Optional) Implement a client executor with the ability to enforce security policies.

Phase 1 also requires implementing sub-components such as secure channels. Phase 2 has a dependency on other TFE projects, including high-level language and compiler infrastructure.

### Phase 1: Specific Federating Executors

For the first phase of the integration we propose several specific executors that each implement a fixed secure aggregation protocol. Besides being of practical use, this also allows us to develop the right abstractions and sub-components for the generic executor introduced in the next phase.

#### Trusted Executor

This federating executor computes aggregations using a distinct third-party executor referenced through a `RemoteExecutor` with no formal placement. To compute an aggregation it collections values from the clients, sends them to the third-party executor, instructs it to aggregate them, and finally pulls back the result. All of this is done over indirect secure channels, yet the aggregation itself is performed on plaintext data by the third-party executor.

While useful on its own, this executor also functions as an important step towards a concrete template to build on, in addition to being a testbed for implementing indirect secure channels and the cryptographic setup needed by them.

This work depends on secure channel primitives being available in TFE.

#### Paillier Executor

This federating executor is a natural evolution of the trusted executor, following essentially the same pattern but where the aggregation done by the third-party is instead performed on data encrypted with the Paillier homomorphic encrypted scheme under a key pair owned by the server. This removes some trust in the third-party since it is now unable to see any of the plaintext values.

In addition to secure and authenticated channels, this work depends on Paillier primitives being available in TFE.

#### Keyed-PRG Executor

This federating executor removes the need for a third-party aggregator by instead instructing the clients to generate and use correlated randomness generated by a keyed PRG. In particular, during a setup process all clients share keys between each other that during the aggregation process can be used to generate zero-sum masks for the values to be aggregated. Once masked by the clients, the federating executor can simply collect and sum. Besides the initial cost of setup, this executor allows secure aggregation without overhead.

This executor is particularly useful in a cross-silo setting, where the set of participating clients remain fixed for a longer period of time. It also poses relevant challenges wrt. longer-term setup material.

In addition to secure channels, this work depends only on a small extension of the secure randomness primitives available in TFE.

#### BIK+'17 Executor

This federating executor implements [Google's secure sum protocol](https://eprint.iacr.org/2017/281) optimized for volatile clients. Although a highly relevant protocol, it is also not clear at this point whether TFF will ship with an built-in implementation in the near future.

#### DPP'17 Executor

This federating executor implements the [SDA secure sum protocol](https://eprint.iacr.org/2017/643) optimized for volatile clients.

It depends on primitives for both Paillier and Shamir (packed) secret sharing.

#### Enclave Executor (Optional)

This federating executor is almost identical to the [trusted executor](#trusted-executor) except that the third-party executor is running inside an enclave. Assuming that enclave availability improves this may be an interesting alternative to the cryptographic solutions. An interesting aspect here is how attestation can be performed through indirect channels.

In addition to secure channels, this work must figure out how to run an executor inside an enclave, perhaps using TF Trusted and TF Lite.

### Phase 2: Generic Server Executor

The specific executors from the first phase are not intended for customization, and their implementation will likely require intimate knowledge of both cryptography and the TFF platform. As such, we do not imagine they represent a viable approach for experimenting with secure aggregation protocols. Instead, as a second phase we propose the implementation of a programmable federating executor parameterized by encrypted computations expressed in the high-level language of TFE.

Note that the goal is not necessarily to capture all possible secure aggregation protocols nor scenarios, and we may likely encounter protocols outside its scope. For this reason it may still be relevant to maintain the specific executors from phase 1.

```python
@tfe.computation
def secure_aggregation(aggregation_device, server_device, aggregation_fn, xs):
  cs = [tfe.classify(x, {x.owner}) for x in xs]

  with aggregation_device:
    d = aggregation_fn(cs)
    e = tfe.classify(d, {server_device.owner})

  with server_device:
    y = tfe.declassify(e)

  return y
```

```python
# define how we want to run the encrypted computations, in
# particular which cryptographic scheme we want for each device

aggregation_device = tfe.DeviceSpec(
    scheme=tfe.schemes.Pond(...)
)

server_device = tfe.DeviceSpec(
    scheme=tfe.schemes.Native(...)
)

# define encrypted computations lazily since the exact input
# signature is not known until runtime and may vary across runs

secure_mean_comp_fn = functools.partial(
    secure_aggregation.get_concrete_computation,
    aggregation_device=aggregation_device,
    server_device=server_device,
    aggregation_fn=lambda xs: tfe.add_n(xs) / len(xs),
)

secure_sum_comp_fn = functools.partial(
    secure_aggregation.get_concrete_computation,
    aggregation_device=aggregation_device,
    server_device=server_device,
    aggregation_fn=lambda xs: tfe.add_n(xs),
)
```

```python
from tensorflow_federated.python.core.impl.compiler import FEDERATED_MEAN
from tensorflow_federated.python.core.impl.compiler import FEDERATED_SUM

executor_fn = tff.framework.create_secure_executor_factory(
    supported_aggregations={
        FEDERATED_MEAN: secure_mean_comp_fn,
        FEDERATED_SUM: secure_sum_comp_fn,
    }
)
tff.framework.set_default_executor(executor_fn)
```

The exact details of this approach remains a work in progress, and in part depends on learnings from phase 1.

### Phase 3: Client Executor

*(work in progress)*

## Detailed Design Proposal

*(work in progress)*

<!--
We here go into further details regarding the implementation of the custom executors.

### Implementing the Paillier Executor

This example consists of the following parties:

- a set of clients providing inputs to the aggregation;
- an aggregator combining ciphertexts using homomorphic properties;
- a key holder offering decryption services;
- a server receiving the aggregation output.

Some of these may not have a formal TFF placement but can be an `EagerExecutor` referenced through a `RemoteExecutor` similar to the `None` executor used by `FederatedExecutor`. Secure channels from the clients to the aggregator are implemented using the bulletin board strategy by routing all messages through the server. Note that the server may play the part of the aggregator (at least in the case of passive security).

The protocol consists of the following steps split into two phases:

- Secure channel setup:

    1. Generate libsodium keypairs on the aggregator, the key holder, and the server; this can be done with a TFE primitive. Copy the encryption key of the aggregator to the clients, the encryption key of the key holder to the aggregator, and the encryption key of the server to the key holder.

- Aggregation; this is repeatable for the same channel setup but care must be taken to increment nonce values accordingly:

    1. (Session setup) Generate a Paillier keypair on the key holder; this can be done with a TFE primitive. Copy the Paillier encryption key to the clients.

    2. Encrypt each client's input using the Paillier encryption key; this can be done using `federated_map` and a TFE primitive.

    3. Send encrypted inputs to the aggregator over secure channel.

    4. Aggregate encrypted inputs on the aggregator; this can be done with a TFE primitive.

    5. Send encrypted result to the server over secure channel.
  
    6. Mask the encrypted result on the server; this can be done with a TFE primitive.

    7. Send the encrypted masked result to the key holder over secure channel.

    8. Decrypt the masked result on the key holder; this can be done with a TFE primitive.

    9. Send the masked result to the server over secure channel.

    10. Unmask the result on the server; this can be done with a TFE primitive.

Each send over secure channel can be implemented as follows:

1. Encrypt the sending party's value using the libsodium encryption key of the recipient; this can be done using a TFE primitive.

2. Copy the encrypted value through the server to the recipient.

3. Decrypt each encrypted value on the recipient; this can be done using a TFE primitive.

See [`paillier_federated_executor.py`](./paillier_federated_executor.py) for full details.

-->

<!--

```python
import tensorflow_federated as tff
from tf_encrypted.protocols import paillier
from tf_encrypted.primitives import libsodium

@tff.tf_computation
def setup():
  keypair = paillier.generate_keypair()
  encryption_key = keypair.encryption_key()
  raw_encryption_key = encryption_key.into_raw_tensor()
  raw_keypair = keypair.into_raw_tensor()
  return raw_keypair, raw_encryption_key  

@tff.tf_computation
def encrypt_input(raw_encryption_key, raw_x):
  encryption_key = paillier.EncryptionKey.from_raw_tensor(raw_encryption_key)
  x = paillier.PlaintextTensor.from_raw_tensor(raw_x)
  c = paillier.encrypt(encryption_key, x)
  raw_c = c.into_raw_tensor()
  return raw_c

@tff.tf_computation
def aggregate(raw_encryption_key, raw_cs):
  encryption_key = paillier.EncryptionKey.from_raw_tensor(raw_encryption_key)
  cs = [paillier.EncryptedTensor.from_raw_tensor(c) for c in raw_cs]
  c = paillier.add_n(encryption_key, cs) / len(cs)
  raw_c = c.to_raw_tensor()
  return raw_c

@tff.tf_computation
def decrypt_output(raw_keypair, raw_c):
  keypair = paillier.Keypair.from_raw_tensor(raw_keypair)
  c = paillier.EncryptedTensor.from_raw_tensor(raw_c)
  decryption_key = keypair.decryption_key()
  y = paillier.decrypt(decryption_key, c)
  raw_y = y.to_raw_tensor()
  return raw_y
```

-->

<!--
### Implementing the Enclave Executor

This example consists of the following parties:

- a set of clients providing inputs to the aggregation;
- an external host offering enclave services;
- a server receiving the aggregated output.

The protocol consists of the following steps:

1. (Session setup) Launch enclave on external host, passing in the specific agggregation computation to be performed, and obtaining gRPC connection details in return; this can be done using a TFE primitive.

2. Store each client's input in the enclave and receive reference in return; this can be done using a TFE primitive.

3. Collect all client references on the server.

4. Ask enclave to run the aggregation computation on referenced inputs and receive reference to result in return; this can be done using a TFE primitive.

5. Retrieve referenced result from enclave onto the server; this can be done using a TFE primitive.

Note that the TFE primitives used here are custom ops wrapping a gRPC client. Note also that the clients can decide which computations are allowed to be performed on their values, and the enclave can prevent the server from retriving client inputs through the secrecy policy attached to them.

### Implementing the Keyed PRGs Executor

*(work in progress)*

-->

<!--
- Setup phase (for keyed PRGs):

  1. Generate a sealed boxes keypair on the aggregator;  
  this can be done using a TF computation calling a TFE primitive.

  2. Broadcast the public encryption key to the clients;  
  this can be done using a move and `federated_broadcast`.

- Aggregation phase (repeatable for same setup):

  1. Generate a Paillier keypair on the server;  
  this can be done using a TF computation calling a TFE primitive.

  2. Broadcast the public encryption key to the clients;  
  this can be done using `federated_broadcast`.

  3. Encrypt each client's value using the public Paillier key;  
  this can be done using `federated_map` with a TF computation calling a TFE primitive.

  4. Encrypt each client's encrypted value using the public key for sealed boxes;  
  this can be done using `federated_map` with a TF computation calling a TFE primitive.

  5. Move the doubly-encrypted values through the server to the aggregator;  
  this can done using `federated_collect` (??) and a move.

  6. Unbox each doubly-encrypted value;  
  this can done along the lines of `sequence_map` with a TF computation calling a TFE primitive.

  7. Aggregate all singly-encrypted values in the sequence;  
  this can be done along the lines of `sequence_reduce` and a TF computation calling a TFE primitive.

  8. Move the resulting singly-encrypted value to the server;  
  this can be done using a move.

  9. Decrypt the singly-encrypted value on the server;  
  this can be done using a TF computation calling TFE primitives.

-->

<!--
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

-->

<!--
### Implementing the Generic Executor

*(work in progress)*

We here give more details on how the custom executor may be implemented. Given (abstract) computations it maps players to known executors and orchestrates TensorFlow computations on them. We concretely focus on `compute_federated_mean`. Note that `create_custom_executor` is responsible for setting up auxiliary executors needed by the encrypted computations when creating the executor stack, and passing these to `CustomExecutor` below.

```python
from tf_encrypted.networking import BulletinBoardStrategy

class CustomExecutor(Executor):

  # ...

  def compute_federated_mean(self, xs, client_executors):
    # assume match between `xs` and `client_executors`

    # create tfe.Player for each input as part of signature
    input_players = [tfe.Player('client-%d' % i)
                     for i, _ in enumerate(xs)]

    input_signature = self._compile_input_signature(
        xs, input_players)

    # instantiate protocol; for Paillier this will create a
    # sequence of program steps that generates a new key pair
    # TODO this needs to be scheduled as well
    new_session_comp = self.protocol.new_session
    new_session = self._run_concrete_computation(new_session_comp)

    # specify where the aggregation will happen; in this
    # particular case the effect is the same as if we
    # hadn't specified a device in the computation
    aggregation_device = tfe.Device('aggregation_device')
    output_receiver = tfe.Player('output_receiver')
    output_receiver_device = output_receiver.default_device
    device_replacements = {
        aggregation_device: output_receiver_device
    }

    # we can finally derive a concrete computation

    # derive concrete computation for these inputs; this is
    # essentially a graph similar to the graphs produces by
    # tf.function but with operations annotated with players,
    # protocols, secrecy requirements, etc.
    aggregation_comp = self._secure_mean_fn(
        input_signature=input_signature,
        protocol=protocol,
        replacements=replacements)

    self._run_concrete_computation(aggregation_comp, new_session)

    # add external executors based on input provided
    # by user along side encrypted computation; con-
    # cretely, this is where the key holder is added
    device_executor_map.update(self._secure_mean_player_executor_map)

    # map the output receiver to the server's local executor
    device_executor_map[output_receiver_device] = self._local_executor

    # TODO here we have to be careful not to map several
    # players to the same executor, since we then loose
    # TFE's ability to check secrecy properties; instead
    # we can replace one player with another before exe-
    # cuting the computations

    # make sure all players are accounted for
    assert concrete_comp.players <= player_executor_map.keys()

    
    device_executor_map[aggregation_device] = server.default_device()

    # compile the concrete computation to use the BB
    # networking strategy; this will result in a se-
    # quence of TFE program steps to be executed in
    # order on the corresponding players
    networking = BulletinBoardStrategy()
    execution_plan = networking.compile(concrete_comp)

    # return result of running the plan
    return self._run_execution_plan(player_executor_map, execution_plan)

  def _compile_input_signature(self, xs, players):
    return [
        tfe.TensorSpec(
            base=PlaintextTensor,
            dtype=x.dtype,
            shape=x.shape,
            device=player.default_device(),
            secrecy={player})
        for x, player in zip(xs, players)
    ]

  def _run_concrete_computation(self, concrete_comp):


  def _run_execution_plan(self, player_executor_map, plan):
    for step in plan:
      executor = player_executor_map[step.player]
      # pre-routing: move inputs from server to executor
      local_inputs = [move(x, executor) for x in step.inputs]
      # execute graph locally
      # TODO assert that step is a TF graph
      local_outputs = self._run_graph(step.graph, local_inputs, executor)
      # post-routing: move outputs from executor to server
      outputs = [move(y, server) for y in local_outputs]

    # ...
```

-->

## Questions and Discussion Topics

- How are ground truth about identities defined and distributed?
