# Integrating TF Encrypted and TensorFlow Federated

| Status        | Draft |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl |
| **Sponsor**   | |
| **Updated**   | 2020-02-26 |

## Objective

This document describes an integration between TF Encrypted (TFE) and TensorFlow Federated (TFF), with the former providing secure computation functionality for the latter. It is being developed for both cross-silo and cross-device federated learning (FL) with the following additional design goals in mind:

- suitable for both easy experimentation and practical production deployment;
- support the use of a variety of secure computation techniques, including MPC and HE;
- support external actors without TFF placements, such as partially-trusted servers or enclaves.

The proposal is based on [TFF 0.12.0](https://github.com/tensorflow/federated/releases/tag/v0.12.0). Additional (and slightly outdated) background information is available in [Inside TensorFlow Federated](./inside-tensorflow-federated.md) and [Integration Strategies](./integration-strategies.md) which are based on [TFF 0.9.0](https://github.com/tensorflow/federated/releases/tag/v0.9.0).

## Motivation

Secure aggregation has been part of federated learning since early on (see e.g. [BIK+'17](https://eprint.iacr.org/2017/281)), yet remains an open area for experimentation as no single solution is a perfect fit for all environments. For example, secure aggregation protocols optimized for cross-device FL (e.g. large groups of volatile mobile devices) are from a cryptographic perspective vastly different than protocols optimized for cross-silo FL (e.g. small set of reliable servers running in a cluster).

The design and analysis of such protocols require expertise and tools that are mostly orthogonal to what is currently available in TFF, and suggests a modular approach where federated algorithms developed and executed with TFF are built on top of secure aggregation protocols developed and executed with TFE.

## Design Proposal

This design introduces server executors similar to the built-in `FederatedExecutor` but implementing secure versions of the supported intrinsic functions such as `tff.federated_secure_sum`, `tff.federated_sum`, and `tff.federated_mean`. As such, the specific secure aggregation protocol remains opaque from the perspective of the TFF glue language, and in the latter cases even that secure aggregation is used at all.

```python
# import integration module from TFE
from tf_encrypted.integrations import federated as tfe_integration

# define federated algorithm using TFF glue language
@tff.federated_computation
def my_computation(xs):
  return tff.federated_secure_sum(xs)

# create secure executor stack and set as default
executor_fn = tfe_integration.create_secure_executor(protocol='bik+17', ...)
tff.framework.set_default_executor(executor_fn)

# evaluate computations in the normal way
my_computation([1., 2., 3., 4., 5.]))
```

We assume a one-time installation of TFE on all involved parties, including clients. Beyond that, clients use the built-in executors, thus requiring no code changes between experiments and as part of deployment. Future work aim to improve upon this by also adding client support for enforcing security policies and trusted setup (e.g. PKI); until this is done we can only offer honest-but-curious security.

### Communication Channels

Indirect secure channels

Cryptographic protocols typically require different communication patterns than those otherwise used in TFF (e.g. `tff.federated_broadcast` and `tff.federated_collect`). In particular, it is often required that one party must send a authenticated and/or confidential message to a specific other party. Although this may change in the future, for this proposal we focus solely on implementing these communication patterns by sending all messages through the server; in cryptographic protocol theory this is known as the bulletin board model.

in the bulletin board it is not enough to have secure direct channels (eg gRPC over mTLS): we need something that can be embedded.

authenticated or secure (encrypted) channels between some of the participants are typically required, allowing one party to send an authenticated or confidential message to another.

This can be implemented by either allowing direct links to be established between executors or by routing all messages through a so-called bulletin board (public database) operated e.g. by the server. The former can be implemented securely using e.g. TLS and the latter without any changes to the client executors using [libsodium](https://github.com/jedisct1/libsodium) primitives exposed through TFE.

- How are ground truth about identities defined and distributed?

- What are alternatives for direct connections for high-performance use cases? out of scope!

### Roadmap

We propose the following implementation phases:

1. Implement and integrate specific executors and required sub-components such as secure channels;
2. Implement and integrate generic executor and required sub-components such as compilers;
3. To the extent possible, re-implement specific executors as instances of the generic executor.

Note that especially phase 2 depends on other TFE work around general language and compiler for encrypted computations.

### Phase 1: Specific Executors

For the first phase of the integration we propose several specific executors that each implement a fixed secure aggregation protocol. Besides being of practical use, this also allows us to develop the right abstracts and sub-components for the generic executor introduced in the next phase.

#### Trusted Executor

This server executor uses a distinct trusted third-party executor referenced through a `RemoteExecutor` and without formal placement. To compute an aggregation it collections values from the clients, sends them to the third-party executor, instructs it to aggregate them using TensorFlow, and finally pulls back the result. All of this is done over secure channels with the server as a bulletin board, yet the aggregation itself is performed on plaintext data by the trusted executor.

While useful on its own, this executor also functions as an important step towards a concrete template to build on, in addition to a workable solution for indirect secure channels and cryptographic setup.

This work depends on secure channel primitives being available in TFE.

#### Paillier Executor

This server executor is a natural evolution of the [trusted executor](#trusted-executor), following essentially the same pattern but where the aggregation done by the third-party is instead performed on data encrypted with the Paillier homomorphic encrypted scheme under a key pair owned by the server. This removes some trust in the third-party since it is now unable to see any of the plaintext values.

In addition to secure and authenticated channels, this work depends on Paillier primitives being available in TFE.

#### Keyed-PRG Executor

This server executor removes the need for a third-party aggregator by instead instructing the clients to generate and use correlated randomness generated by a keyed PRG. In particular, during setup all clients share keys between each other that during aggregation can be used to generate zero-sum masks for the values to be aggregated. Once masked by the client executors, the server executor can simply collect and sum. Besides the initial cost of setup, this executor allows secure aggregation without overhead.

This executor is particularly useful in a cross-silo setting, where the set of participating clients remain fixed for a longer period of time. It also poses relevant challenges wrt. longer-term setup material.

In addition to secure channels, this work depends only on a small extension of the secure randomness primitives available in TFE.

#### DPP'17 Executor

This server executor implements the [SDA secure sum protocol](https://eprint.iacr.org/2017/643) optimized for volatile clients.

It depends on primitives for both Paillier and Shamir (packed) secret sharing.

#### BIK+'17 Executor (Optional)

This server executor implements [Google's secure sum protocol](https://eprint.iacr.org/2017/281) optimized for volatile clients. Although a highly relevant protocol, it is also not clear at this point whether TFF will ship with an built-in implementation in the near future.

#### Enclave Executor (Optional)

This server executor is almost identical to the [trusted executor](#trusted-executor) except that the third-party executor is running inside an enclave. Assuming that enclave availability improves this may be an interesting alternative to the cryptographic solutions. An interesting aspect here is how attestation can be performed through indirect channels.

In addition to secure channels, this work must figure out how to run an executor inside an enclave, perhaps using TF Trusted and TF Lite.

### Phase 2: Generic Executor

The specific executors from the first phase are not intended for customization, and their implementation will likely require intimate knowledge of both cryptography and the TFF platform. As such, we do not imagine they represent a viable approach for designing and analyzing secure aggregation protocols. Instead, as a second phase we propose the implementation of a programmable executor parameterized by encrypted computations expressed in the high-level language of TFE.

```python
@tfe.computation
def secure_aggregation(aggregator, server, aggregation, xs):
  cs = [tfe.classify(x, {x.owner}) for x in xs]

  with aggregator:
    d = aggregation(cs)
    e = tfe.classify(d, {server.owner})

  with server:
    y = tfe.declassify(e)

  return y
```

```python
# define how we want to run the encrypted computations, in
# particular which cryptographic scheme we want for each device
aggregator = tfe.DeviceSpec(
    scheme=tfe.schemes.Pond(...)
)
server = tfe.DeviceSpec(
    scheme=tfe.schemes.Native(...)
)

# define encrypted computations lazily since the exact input
# signature is not known until runtime and may vary across runs

secure_mean_comp_fn = functools.partial(
    secure_aggregation.get_concrete_computation,
    aggregator=aggregator,
    server=server,
    aggregation_fn=lambda xs: tfe.add_n(xs) / len(xs),
)

secure_sum_comp_fn = functools.partial(
    secure_aggregation.get_concrete_computation,
    aggregator=aggregator,
    server=server,
    aggregation_fn=lambda xs: tfe.add_n(xs),
)
```

```python
from tensorflow_federated.python.core.impl.compiler import FEDERATED_MEAN
from tensorflow_federated.python.core.impl.compiler import FEDERATED_SUM

from tf_encrypted.integrations import federated as tfe_federated

executor_fn = tfe_federated.create_secure_executor(
    supported_aggregations={
        FEDERATED_MEAN: secure_mean_comp_fn,
        FEDERATED_SUM: secure_sum_comp_fn,
    }
)
tff.framework.set_default_executor(executor_fn)
```

The goal here is not necessarily to capture all possible secure aggregation protocols nor scenarios, and we may likely encounter protocols outside its scope; for this reason it may still be relevant to maintain the special executors from phase 1.

all runtime players each have a signature keypair registered in a PKI


As detailed later, to run secure aggregations the executor first derives concrete computation from the above using the specified protocol, and then uses a bulletin-board network strategy to compile these into a set of program steps and a plan for when and where they should be executed. Each of these steps is essentially a local TensorFlow graph that can be executed by the built-in `EagerExecutor`.

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
