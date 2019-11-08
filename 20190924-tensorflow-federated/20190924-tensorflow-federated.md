# Integrating TF Encrypted and TensorFlow Federated

| Status        | Draft |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl |
| **Sponsor**   | |
| **Updated**   | 2019-10-31 |

## Objective

This document describes an integration between TF Encrypted (TFE) and TensorFlow Federated (TFF), with the former providing runtime secure computation services for the latter. It is being developed with the following design goals in mind:

- aggregation protocols may involve external players without TFF placements;
- aggregation protocols may use a variety of secure computation techniques, including enclaves, MPC, and HE;
- the approach should target both easy experimentation as well as practical production deployment.

The proposed design is based on [TFF 0.9.0](https://github.com/tensorflow/federated/releases/tag/v0.9.0); please see [Inside TensorFlow Federated](./inside-tensorflow-federated.md) for more background information.

## Motivation

Secure aggregation has been part of the privacy answer in federated learning from early on, and remains an open area for experimentation as no single solution is a perfect fit for all environments. For example, secure aggregation protocols optimized for a large group of volatile Internet users is vastly different from protocols optimized for a small set of reliable servers running in a cluster. This proposal addresses this need in a modular approach, essentially suggesting the use of TFF for designing federated algorithms while using TFE for designing secure aggregation protocols.

## Design Proposal

The design introduces custom server executors similar to the `FederatedExecutor` shipping with TFF. They implement encrypted versions of (supported) intrinsic functions (blocking unsupported) and forwards other commands to the next executor in the hierarchy. From the perspective of the user designing federated algorithms using the TFF glue language, this means that the use of secure aggregation protocols remains opaque.

All protocols assume the existing of secure and authenticated channels between players,¨ which may either be offered directly by the runtime platform or implemented using TFE primitives. These may be indirect and for instance store all (encrypted) messages in a public database.

A one-time installation of TFE on the clients is required in order for the needed cryptographic primitives to be available. Beyond that clients only use built-in executors, requiring no code changes between experimentation and for deployment.

```python
# federated algorithm used in the running example
@tff.federated_computation
def my_computation(xs):
  return tff.federated_mean(xs)
```

```python
# create built-in executor
executor = tff.framework.create_local_executor(4)
```

```python
# execute using specific executor
tff.framework.set_default_executor(executor)

my_computation([1., 2., 3., 4.]))
```

### Specialized Encrypted Executors

We propose several easy-to-use executors for specific secure aggregation protocols: based on Paillier encryption with an external key holder, on additive secret sharing among the clients, and on an enclave hosted on the server. These can be implemented using the cryptographic primitives built into TFE and may be employed as follows:

```python
from tf_encrypted.integrations import federated as tfe_federated

# create Paillier-based executor
executor = tfe_federated.create_paillier_executor(4)
```

### General Encrypted Executor

The above specialized executors are not intended for customization, so to address the need for experimentation we also propose a programmable executor parameterized by an encrypted computation (or program) expressed in the high-level language of TFE.

```python
from tensorflow_federated.python.core.impl.compiler import FEDERATED_MEAN
from tensorflow_federated.python.core.impl.compiler import FEDERATED_SUM
from tf_encrypted.integrations import federated as tfe_federated

# create custom executor from encrypted computations
executor = tfe_federated.create_custom_executor(
    supported_aggregations={
        FEDERATED_MEAN.uri: secure_mean,
        FEDERATED_SUM.uri: secure_sum,
    },
    num_clients=4)
```

The exact language for expressing encrypted computation is a work in progress but may be along the lines of the following:

```python
# define symbolic players
server = tfe.Player('server')
key_holder = tfe.Player('key_holder')

# define the (abstract) encrypted computation;
# this is where we focus on *what* is computed
# and the desired security properties
@tfe.computation(
    input_signature={
        # we expect plaintext inputs, but cannot say
        # much else yet since the corresponding play-
        # ers will remain unknown until execute time
        'xs': List<tfe.PlaintextTensor>
    },
    output_signature=[
        # we expect a plaintext result which must be
        # known only by the server
        tfe.PlaintextTensor.with_secrecy({server})
    ]
)
def secure_aggregation(aggr_fn, xs):
  # encrypt inputs on their current devices; these
  # will be known when a concrete computation is
  # derived at runtime
  cs = [tfe.encrypt(x) for x in xs]

  # compute aggregation on default compute player;
  # due to the protocol picked later this will
  # concretely happen on the server
  d = aggr_fn(cs)

  # allow the server to learn the result; this is
  # checkable at compile-time and runtime but can
  # be configured to be a runtime no-op
  d = tfe.allow(d, {server})

  # decrypt result for server; due to the protocol
  # picked later this will concretely happen via
  # the key holder, but masked so that the server
  # is the only player that learns the plaintext
  with server:
    y = tfe.decrypt(d)

  # return (reference to) result
  return y

# tfe.function is like tf.function but supporting TFE
# custom tensors: it represents a differentiable ope-
# ration, indifferent to whether or not it is applied
# to encrypted values, but expressed using the TFE API
# to support both; in the future we may support using
# the TF API and automatically convert operations
@tfe.function
def mean(cs):
  return tfe.add_n(cs) / len(cs)
```

```python
# set up the concrete protocol that we want to use
paillier = tfe.protocols.Paillier(
    key_holder=key_holder,
    default_compute_player=server)

# combine computation with protocol; this allows us to
# derive concrete computations but is here done lazily
# since the executor is the only place where input `xs`
# is concretely known, including its cardinality; the
# focus is on *how* the computation happens, and it is
# finally clear that it is encrypted; when we derive
# concrete computations we can check that the chosen
# protocol satisfies the security requirements (which
# would not have been the case if we had used a plain-
# text protocol instead); `aggr_fn` is not require to
# be a tfe.function, but tfe.computation supports it

secure_mean = functools.partial(secure_aggregation,
    protocol=paillier, aggr_fn=mean)

secure_sum = functools.partial(secure_aggregation,
    protocol=paillier, aggr_fn=tfe.add_n)
```

```python
from tf_encrypted.networking import BulletinBoardStrategy

class CustomExecutor(Executor):

  # ...

  def _compute_federated_mean(self, xs):
    # assume correspondence between `xs` and clients;
    # create tfe.Players accordingly
    player_executor_map = {
        tfe.Player('client-%d' % i): client
        for i, client in enumerate(self._client_executors)
    }

    # derive concrete computation for `xs`; this is
    # essentially a graph similar to a TF graph but
    # with operations annotated with players, proto-
    # cols, secrecy requirements, etc.
    concrete_comp = self._get_concrete_computation(
        self._secure_mean, xs, player_executor_map.keys())

    # compile the concrete computation to use the BB
    # networking strategy; this will result in a se-
    # quence of ordinary TF graphs to be executed in
    # order on the corresponding players
    networking = BulletinBoardStrategy()
    execution_plan = networking.compile(concrete_comp)

    # add remaining executors
    player_executor_map['key_holder'] = self._keyholder_executor
    player_executor_map['server'] = self._local_executor

    # return result of running the plan
    return self._run_execution_plan(player_executor_map, execution_plan)

  def _get_concrete_computation(self, comp, inputs, players):
    # compile signature from inputs and players
    input_signature = [
        tfe.TensorSpec(
            base=PlaintextTensor,
            dtype=x.dtype,
            shape=x.shape,
            location=player,
            secrecy={player})
        for x, player in zip(inputs, players)
    ]
    # we can finally derive concrete computation
    return comp.get_concrete_computation(
        input_signature={'xs': input_signature})

  def _run_execution_plan(self, player_executor_map, plan):
    for step in plan:
      executor = player_executor_map[step.player]
      # pre-routing: move inputs from server to executor
      local_inputs = [move(x, executor) for x in step.inputs]
      # execute graph locally
      local_outputs = self._run_graph(step.graph, local_inputs, executor)
      # post-routing: move outputs from executor to server
      outputs = [move(y, server) for y in local_outputs]

    # ...
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


As detailed below, the executor will essentially compile such computations into a set of TensorFlow graphs and a plan for when and where they should be executed. Note that `tfe.allow` is a no-op at runtime yet enables automated reasoning about secrecy.

### Secure Channels

Secure aggregation protocols often require communication patterns outside of those otherwise used in TFF. In particular, secure channels between some of the parties are typically required, allowing one party to send a confidential and/or authenticated message to another. If a PKI is in place then this can in general be realized by routing all message through the server, requiring no changes to the client executors. This can either be achieved via the cryptographic primitives built into TFE or ideally via secure channel functionality built into TFF.

In some settings it is however interesting to instead allow direct communication between the parties using, for instance in high-performance cluster computations. This remain a work-in-progress but would likely happen via a custom `ConnectedEagerExecutor` that can establish secure connections through e.g. gRPC over TLS.

### External Services

### Implementation Roadmap

1. Implement specific executors;
2. Implement generic executor;
3. Re-implement (select) specific executors as instances of generic.

## Detailed Design Proposal

We here detail a few examples of custom executors for secure aggregation.

### Secure Aggregation using Paillier

This example consists of the following parties:

- a set of clients providing inputs to the aggregation;
- the server receiving the aggregated output and holding the decryption key;
- an aggregator responsible for combining ciphertexts using the homomorphic properties of the encryption scheme.

The aggregator does not have a formal TFF placement but is an `EagerExecutor` e.g. referenced through a `RemoteExecutor` by the custom executor running on the server (similar to the `None` executor used by `FederatedExecutor`). Secure channels from the clients to the aggregator are implemented by routing all messages through the server using the [libsodium](https://github.com/jedisct1/libsodium) operations for sealed boxes exposed through TFE; if desired the identity of clients can be ensured using authenticated encryption instead.

The protocol consists of the following steps split into two phases:

- Setup phase (for setup channels):

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

### Secure Aggregation using Enclaves

*(work in progress)*

<!--

This example consists of the following parties:

- a set of clients providing inputs to the aggregation;
- the server receiving the aggregated output and holding the decryption key;
- an external enclave offering aggregation services.

Everyone connects to the enclave using gRPC enclave client built into TFE. ?!?!?!?

Obstacles:

- allow gRPC connections from clients;
- launching the enclave and creating new sessions in it;
- manage associated state.
-->

### Secure Aggregation using Secret Sharing and Keyed PRGs

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

### Programmable Secure Aggregation

*(work in progress)*

<!--
The above specialized executors are easy to use but the general strategy makes for a more involved experimentation process. To that end we also propose a programmable executor 

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

- What are the options for implementing secure channels (authenticated encryption)?
  - Can/should we assume that it is offered by the execution platform?
  - How do parties verify the identity of each other?
  - How are ground truth about identities defined and distributed?

- What options do we have for direct connections?
  - Directly through TF Distributed using a ClusterSpec?
  - Using a ClusterSpec but going through local proxies managed by the worker?
  - Side-stepping TF Distributed by using custom-ops with gRPC clients instead?

- Is it interesting to build a custom executor for enclaves?

## Appendix: Integration Strategies

We here describe our motivation behind the particular integration strategy proposed above. There are at least two ways of adding support for secure aggregation to TFF:

1. by expressing secure aggregation as federated algorithms in the language of TFF;
2. via custom executors implementing corresponding intrinsic functions.

It is not clear that the former approach fits with the [overall aim](https://github.com/tensorflow/federated/blob/v0.9.0/docs/tutorials/custom_federated_algorithms_1.ipynb#design-overview) of TFF at a conceptual level, since aggregation algorithms are all currently implemented as intrinsic functions and secure computation protocols inherently need to express behavior at a more fine-grained, per-player level (although the code alludes to additional placement literals in the future, this still seems at a higher level). On the technical side, the currently implementation also lacks types to represent e.g. encrypted data and encrypted computations, which would hence have to be expressed in raw forms with the complexities involved. One advantage of the approach is that secure aggregation is a highly integrated part of federated algorithms, allowing developers to stay in the same framework when developing and experimenting with custom aggregation protocols.

The potential disadvantages of the second approach is that secure aggregation happens at a lower level and remain opaque to the federated algorithms expressed in TFF. This may require modifications of the execution platform as well as additional knowledge and toolchains when developing custom protocols. One advantage of this approach is the extra control over how secure computations are specified and executed. Another is the modularity that comes from keeping TFF operating at a higher level of abstraction, making it easy to test algorithms independently of the concrete aggregation method used. [Hints in the code](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/proto/v0/computation.proto#L625) furthermore seem to suggest this approach.

The second approach can be done:

- internally, where the secure computation is fully embedded in TFF and performed using its computation, communication, and orchestration mechanisms only;
- externally, where TFF may e.g. be used only for orchestrating input and output to an external service run by a set of players opaque to the federated algorithms.

Note that:

- The internal strategy is not the same as expressing secure aggregation as federated algorithms, since an executor on e.g. the server knows the exact players at runtime and may schedule computations independently on each.
- Both strategies may require additional functionality in e.g. the EagerExecutor, such as running cryptographic primitives and setting up distributed computations.
- The external strategy does not necessarily imply a distinct set of players, as the same set may e.g. run both the TFF worker and the external service.

The external strategy may turn out to be the best fit for employing certain robust or large-scale aggregation systems involving e.g. bulletin boards. One advantage of the internal strategy is that everything is managed within TFF and no other services have to be maintained. One disadvantage is that everything have to be compiled down to the TFF computation format, which in the external strategy might only be the case for logic dealing with input and output; this may make the internal strategy more suitable for computations with relatively simple communication patterns (e.g. based on HE) that can be completely implemented inside the executor, whereas the external strategy may be more suitable for complex computations (e.g. based on MPC).
