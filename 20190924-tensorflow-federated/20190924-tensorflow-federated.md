# Integrating TF Encrypted and TensorFlow Federated

| Status        | Draft |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl |
| **Sponsor**   | |
| **Updated**   | 2019-09-24 |

## Objective

This document describes an integration between TF Encrypted (TFE) and TensorFlow Federated (TFF), with the former providing secure computation functionality for the latter. It is being developed for various environments, including both cross-silo and cross-device federated learning (FL), with the following design goals in mind:

- the approach should target both easy experimentation and practical production deployment;
- aggregation protocols may use a variety of secure computation techniques, including MPC and HE;
- computations may involve external actors without TFF placements, such as partially-trusted servers or enclaves.

The proposed design is based on [TFF 0.9.0](https://github.com/tensorflow/federated/releases/tag/v0.9.0) with additional background information available in [Inside TensorFlow Federated](./inside-tensorflow-federated.md) and [Integration Strategies](./integration-strategies.md).

## Motivation

Secure aggregation has been part of the privacy answer in federated learning since early on, yet remains an open area for experimentation as no single solution is a perfect fit for all environments. For example, secure aggregation protocols optimized for cross-device FL (eg a large group of volatile mobile devices) are vastly different from a cryptographic perspective than protocols optimized for cross-silo FL (eg a small set of reliable servers running in a cluster).

The design of such protocols require expertise and tools that are mostly orthogonal to what is  currently available in TFF, and this document proposes a modular solution to supporting experimentation and deployment of secure aggregation in TFF, suggesting the use of TFF for designing federated algorithms and TFE for designing secure aggregation protocols.

## Design Proposal

This document introduces custom server executors similar to the `FederatedExecutor` shipping with TFF. They implement encrypted versions of supported intrinsic functions, blocks the unsupported, and forwards unrelated commands to the next executor in the hierarchy. This means that the use of secure aggregation protocols remains opaque to the user designing federated algorithms using the TFF glue language.

Consider the following example:

```python
# define federated algorithm using TFF glue language
@tff.federated_computation
def my_computation(xs):
  return tff.federated_mean(xs)

# execute using default executor stack
my_computation([1., 2., 3., 4., 5.]))
```

The only change needed to instead 
we instead use one of the executors outlined in the following subsections:

```python
def secure_executor_stack(num_clients):

  def executor_fn(mapping):
    del mapping

    def _complete_stack(ex):
      return \
          LambdaExecutor(
          # CachingExecutor(
          # ConcurrentExecutor(
          ex
      )

    def _create_bottom_stack():
      return _complete_stack(EagerExecutor())

    executor_dict = {
        tff.CLIENTS: [_create_bottom_stack() for _ in range(num_clients)],
        tff.SERVER: _create_bottom_stack(),
        None: _create_bottom_stack(),
    }
    return _complete_stack(SecureFederatedExecutor(FederatedExecutor(executor_dict)))

  return executor_fn

executor_fn = tff.framework.create_local_executor(5)
tff.framework.set_default_executor(executor_fn)
```


A one-time installation of TFE on the clients is required in order for the needed cryptographic primitives to be available. Beyond that clients only use built-in executors, requiring no code changes between experiments and as part of deployment. We assume the existing of secure and authenticated channels between some of the participants; these may be implemented using TFE primitives if not offered by the runtime platform.

### Specific Encrypted Executors

We propose several executors for specific secure aggregation protocols implemented using the cryptographic primitives built into TFE and used as follows:

```python
from tf_encrypted.integrations import federated as tfe_federated

# create Paillier-based executor
executor = tfe_federated.create_paillier_executor(5)
```

We here focus on three specific executors based on respectively:

- Paillier encryption with distinct key holder;
- additive secret sharing among the clients with keyed PRF;
- secure enclave accessed through gRPC.

To the extent possible, the aim is to eventually define these as special cases of the generic executor described below.

### Generic Encrypted Executor

The above specific executors are not intended for customization, so to address the need for experimentation we also propose a programmable executor parameterized by encrypted computations expressed in the high-level language of TFE. While we believe that the special executors mentioned earlier can be naturally expressed as special cases of this executor, the goal is not to capture all possible scenarios, and we may encounter protocols outside its scope.

```python
from tensorflow_federated.python.core.impl.compiler import FEDERATED_MEAN
from tensorflow_federated.python.core.impl.compiler import FEDERATED_SUM

from tf_encrypted.integrations import federated as tfe_federated

# define symbolic player owning the output of the
# computation define below; this will later be lin-
# ked with the server executor
output_receiver = tfe.Player('output_receiver')

# define encrypted computations for the supported
# aggregation; these are defined lazily since their
# exact input signature is not known until runtime
# and may vary between invocations; details later
secure_mean_comp_fn = ...
secure_sum_comp_fn = ...

# define the concrete protocol we want to use, and
# fixes *how* we want the computation to take place;
# a new instance is created for each aggregation,
# which in turn creates a fresh keypair for each
paillier = tfe.protocols.Paillier(
    key_holder=tfe.Player('key_holder'),
    default_device=output_receiver.default_device)

# create custom executor supporting mean and sum
executor = tfe_federated.create_generic_executor(
    protocol=paillier,
    supported_aggregations={
        FEDERATED_MEAN: secure_mean_comp_fn,
        FEDERATED_SUM: secure_sum_comp_fn,
    },
    num_clients=5)
```

The exact language for expressing the encrypted computation is a work in progress but may be along the lines of the following:

```python
# define *what* we wish to compute; tfe.function is
# like tf.function but supporting TFE custom tensors:
# it represents a differentiable operation, indiffe-
# rent to whether or not it is applied to encrypted
# values, but expressed using the TFE API to support
# both; in the future we may support the TF API and
# automatically convert operations based on the graph
@tfe.function
def mean(cs):
  return tfe.add_n(cs) / len(cs)

# define a symbolic compute unit responsible for exe-
# cuting the above aggregation function; this will be
# linked with the server executor later
aggregation_device = tfe.Device('aggregation_device')

# define *where* we want the computation to be orche-
# strated and the security properties it must satis-
# fy; the resulting computation is abstract and made
# concrete below by combining it with the protocol;
# this also wraps up the computation as a single unit
# of work that the protocol can analyze and optimize
# as a whole
@tfe.computation(
    input_signature=[
        # we expect plaintext inputs, but cannot say
        # much else yet since the corresponding play-
        # ers will remain unknown until execute time;
        # this is only recorded for now and later re-
        # placed with a satisfying concrete signature
        List<tfe.PlaintextTensor>
    ],
    output_signature=[
        # we expect a plaintext result which must be
        # known only by the output receiver
        tfe.PlaintextTensor.with_secrecy({output_receiver})
    ]
)
def secure_aggregation(aggr_fn, xs):
  # encrypt inputs on their current devices; these
  # will be known when the concrete computation is
  # derived at runtime
  cs = [tfe.encrypt(x) for x in xs]

  # compute the aggregation; some protocols are more
  # picky w.r.t. to the device used, and may complain
  # when a concrete computation is derived; the pro-
  # tocol default device is used if left unspecified,
  # which in the concrete case here means the output
  # receiver
  with aggregation_device:
    d = aggr_fn(cs)

  # allow the output receiver to learn the result;
  # this is checkable at compile-time and runtime,
  # but can be configured to be a no-op
  d = tfe.allow(d, {output_receiver})

  # decrypt result for output receiver (on its de-
  # fault device); with the Paillier protocol this
  # will concretely happen via the key holder, but
  # masked by the output receiver
  with output_receiver.default_device():
    y = tfe.decrypt(d)

  # return (reference to) result
  return y

# we lazily combine the computation with a protocol
# to obtain concrete computations since the server
# executor is the only place where we know `xs`

secure_mean_fn = functools.partial(
    secure_aggregation.get_concrete_computation,
    aggr_fn=mean)

secure_sum_fn = functools.partial(
    secure_aggregation.get_concrete_computation,
    aggr_fn=tfe.add_n)

# where we derive concrete computations later we can
# check that the chosen security requirements are sa-
# tisfied; this would not be the case here if we had
# used a plaintext protocol instead
```

```python
xs = ...

# this is just a specification of the protocol we wish
# to use, and no operations are run as part of its in-
# stantiation; this allows the object to also be used
# when e.g. defining computations
protocol = tfe.protocols.Paillier(
    key_holder=tfe.Player('key_holder'),
    default_device=output_receiver.default_device)

# create a new session of the protocol; this is what
# happens behind the scenes if `protocol` was used di-
# rectly as a context handler; when a keypair is spe-
# cified then this is used, otherwise a new pair is
# created on the key holder; different sessions of the
# same protocol may or may not be compatible: for in-
# stance, in the case of Paillier it depends on whether
# the keypairs match, and in Pond on whether the same
# group of devices was used
protocol_session = protocol.new_session(
    # some protocols (such as enclaves and )
    allowed_computations=None
)

with protocol_session:

  # if no device is specified (between the session con-
  # text handler and here) then `encrypt` will try to
  # determine a device based on information in the type
  # of `x`; note that inputters do not know what their
  # values will be used for at this point, yet can still
  # specify secrecy properties
  cs = [tfe.encrypt(x) for x in xs]

  # compute the aggregation
  with aggregation_device:
    d = aggr_fn(cs)

  # allow the output receiver to learn the result
  d = tfe.allow(d, {output_receiver})

  # decrypt result for output receiver on its default
  # device; will check that the output_receiver is
  # allowed to see the value in plaintext
  with output_receiver.default_device():
    y = tfe.decrypt(d)
```

all runtime players each have a signature keypair registered in a PKI

```python
from tf_encrypted.protocols import paillier

protocol = tfe.protocols.Paillier(
    bit_precision=16,
    key_holder=tfe.Player('key_holder'),
    default_device=output_receiver.default_device)

@tfe.concrete_computation
def new_session():
  with protocol.key_holder.default_device():
    keypair = paillier.keypair()
    encryption_key = keypair.encryption_key()
    decryption_key = keypair.decryption_key()
  
  return PaillierSession(encryption_key, decryption_key)

@tfe.concrete_computation
def aggregation(session, xs):

  cs = []
  for x in xs:
    with x.player.default_device():
      x = paillier.encode(protocol.bit_precision, x)
      c = paillier.encrypt(session.encryption_key, x)
    cs.append(c)

  with aggregation_device:
    d = paillier.add_n(session.encryption_key, cs)
    d = paillier.div(session.encryption_key, d, len(cs))

  with output_receiver.default_device():
      y = paillier.decrypt(session.decryption_key, d)

  return y

xs = ...
paillier_session = new_session()
y = secure_mean(paillier_session, xs)
```

As detailed later, to run secure aggregations the executor first derives concrete computation from the above using the specified protocol, and then uses a bulletin-board network strategy to compile these into a set of program steps and a plan for when and where they should be executed. Each of these steps is essentially a local TensorFlow graph that can be executed by the built-in `EagerExecutor`.

### Network Strategy and Secure Channels

Secure aggregation protocols often require communication patterns outside of those otherwise used in TFF. In particular, channels between some of the participants are typically required, allowing one party to send a confidential and/or authenticated message to another.

This can be implemented by either allowing direct links to be established between executors or by routing all messages through a so-called bulletin board (public database) operated e.g. by the server. The former can be implemented securely using e.g. TLS and the latter without any changes to the client executors using [libsodium](https://github.com/jedisct1/libsodium) primitives exposed through TFE.

### External Services

Certain protocols and network strategies require the use of external services such as enclave instances and bulletin boards. Clients for these can be exposed through TF custom ops built into TFE as needed.

### Implementation Roadmap

We propose the following implementation phases:

1. Implement specific executors and required subcomponents such as secure channels;
2. Implement generic executor and required subcomponents such as compilers;
3. Re-implement specific executors as instances of the generic.

So of this work depends on work related to TFE architecture, especially around model and runtime system for encrypted computations.

## Detailed Design Proposal

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

```python
tfe.TensorSpec(
    base=tfe.PondTensor,
    dtype=dtype,
    shape=shape,
    protocol=Pond(dtype=dtype, server0, server1),
    device=pond123,
    location={server0, server1},
    secrecy={inputter},
)
```

```python
pond = tfe.protocols.Pond(...)

with pond.as_protocol():
  with pond:
    y = x * 2
)

with pond.as_protocol():
  with pond.as_device():
    y = x * 2
)
```

```python
# equivalent ways of defining tensor requirement;
# from this is follows that every tfe.Tensor type
# can also be used as a tfe.TensorSpec

# - TensorFlow style
tfe.TensorSpec(
    base=tfe.PlaintextTensor,
    dtype=dtype,
    shape=shape,
    location=location,
    secrecy=secrecy)

# - slightly more object-oriented
tfe.PlaintextTensor \
    .with_dtype(dtype) \
    .with_shape(shape) \
    .with_location(player) \
    .with_secrecy(player)
```

```python

# set up the concrete protocol that we want to use;
# focus is on *how* the computation takes place
protocol = tfe.protocols.Paillier(
    key_holder=key_holder,
    default_compute_player=server)
```



<!--
The above specific executors are easy to use but the general strategy makes for a more involved experimentation process. To that end we also propose a programmable executor 

the point is to have operations for eg sealed boxes/authenticated encrypted generated automatically when compiling for bulleting-board networking, allowing the protocol designer to simply assume secure channels. this reduces e.g. the Paillier aggregator to be expressed as follows:

```python
@tfe.computation(
    players={'client0', 'client1', 'server'},
    input_signature={
        'x0': tfe.PlainTensor.with_confidentiality({'client0'}),
        'x1': tfe.PlainTensor.with_confidentiality({'client1'}),
    },
    output_signature=[
        tfe.PlainTensor.with_confidentiality({'server'})
    ],
    local_variables={},
    local_protocols={},
)
def secure_mean(x0, x1):

  with tfe.Player('client0'):
    c0 = tfe.encrypt(x0)

  with tfe.Player('client1'):
    c1 = tfe.encrypt(x1)

  c = tfe.add_n(c0, c1) / 2
  c = tfe.allow(c, {tfe.Player('server')})

  with tfe.Player('server'):
    y = tfe.decrypt(c)
  
  return y
```

```python
paillier = tfe.protocols.Paillier(
    key_owner=server,
    default_computer=aggregator)

enclave = tfe.protocols.Enclave(
    host=server)

pond = tfe.protocols.Pond(
    computers=[aggregator0, aggregator1])

paillier_secure_mean = secure_mean.with_protocol(paillier)
enclave_secure_mean = secure_mean.with_protocol(enclave)
pond_secure_mean = secure_mean.with_protocol(pond)
```

general idea is that you can express secure aggregation mechanisms in the TFE glue language, and then through a two-step compilation process compile this down to a format that can be executed by the executor, such as a list of annotated TF computations.
-->
<!--
Aggregations are implemented by compiling the supplied TFE computations down to a sequence of local TF computations, together with a plan detailing when and where to execute each. Execution is carried out using `EagerExecutor`s on the clients, with tensors moved around by the server according to where they are needed next.


```python
import tensorflow_federated as tff

import tf_encrypted as tfe
import tf_encrypted.integrations.federated as tfe_federated

secure_mean = tfe_federated.FastMean()

# TODO
# specify which protocol to use somewhere without mentioning
# the concrete players to use, which may only be known elsewhere?

# TODO
# how do we allow more general functions to be computed?

def executor_fn(_):
  return tfe_federated.SecureFederatedExecutor(
      aggregation_protocols={'federated_mean': secure_mean},
      target_executor=tff.FederatedExecutor(...)
  )

tff.framework.set_default_executor(executor_fn)
```


```python
# use a function to delay execution
class SecureAggregation:

  def __init__(input_providers: List<tfe.Player>,
               aggregator: tfe.Player,
               output_receiver: tfe.Player):
    self.input_providers = input_providers
    self.aggregator = aggregator
    self.output_receiver = output_receiver

  # this gets turned into a TF graph without device
  # annotations, and in turn a tff.tf_computation
  # labelled for execution on the output receiver;
  #
  # as part of the compilation process, code is added
  # that converts the output tensors to raw TF tensors;
  #
  # the server executor receives
  # references to the locally stored output tensors
  # for further use in the execution
  def setup(self):
    with self.output_receiver:
      keypair = tfe.protocols.paillier.generate_keypair()
      public_key = keypair.public_key
      return keypair, public_key

  def decrypt(self, keypair, x_encrypted):
    with output_receiver:
      x = tfe.protocols.paillier.decrypt(keypair, x)
      return x


  @tfe.function
  def encrypt_input(x, input_provider):
    with input_provider:
      x_encrypted = 

  def run(xs)

  keypair, pk = setup()


  @tfe.function # ??
  def run(keypair, pk, xs):
    for inputter in inputters:
      with inputter

  return setup, run
```


```python
from tf_encrypted.integrations.federated import FederatedExecutionContext

class SecureFederatedExecutor:

  # ...

  def _run_tfe_function(self, fn, *args, **kwargs):
    execution_context = FederatedExecutionContext()
    execution_context.run(fn, *args, **kwargs)

  def _run_federated_mean(self):
    if self._mechanism is None:
      self._mechanism = SecureAggregation(
          input_providers=self._clients,
          aggregator=self._aggregator,
          ???
      )
      self._mechanism.setup()

````



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

-->

## Questions and Discussion Topics

- What are alternatives for implementing secure channels (authenticated encryption)?
  - Can/should we assume that it is offered by the execution platform?
  - How are ground truth about identities defined and distributed?

- What are alternatives for direct connections for high-performance use cases?
  - Is this out of scope for TFF?
  - Directly through TF Distributed Engine using a ClusterSpec?
  - Using a ClusterSpec but going through local proxies managed by the worker?
  - Side-stepping TF Distributed by using custom-ops with gRPC clients instead?
