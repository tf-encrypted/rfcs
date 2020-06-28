# Secure Aggregation in TF Encrypted

| Status        | Proposed |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com) |
| **Sponsor**   | |
| **Updated**   | 2019-10-01 |

## Objective

This document describes the design and implementation of secure aggregation functionalities in TF Encrypted (TFE).

Requirements:

- Should be usable in plain TensorFlow (TF) in combination with e.g. `tfe.local_computation`, taking care of everything needed including encryption and networking; given a set of local tensors, performing secure aggregation might be as simple as instantiating a functionality and applying it as a function to these.

- Should be usable in TensorFlow Federated (TFF), integrating with how e.g. networking is handled in this framework.

- Should offer common implementations that can simply be instantiated and use as described above.

- Should be flexible enough to support custom protocols, from efficient protocols based in keyed secret sharing to robust protocols such as [SDA](https://eprint.iacr.org/2017/643).

## Motivation

Secure aggregation plays a key role in federated learning and private analytics. However, implementation of these is often non-trivial and may require custom made cryptographic protocols tailored to the particular setting.

## Design Proposal

```python
computers = [...]
output_receiver = ...
assert all(isinstance(computer, tfe.Player) for computer in computers + [output_receiver])


assert all(isinstance(computer, tfe.Player) for computer in computers)

xs = [...]
assert all(isinstance(x, tfe.LocalTensor) for x in xs)



secure_aggregation = tfe.functionalities.AdditiveSecureAverage(
    computers=computers,
    output_receiver=output_receiver
)

y = secure_aggregation(xs)
assert isinstance(y, tfe.LocalTensor)
assert y.placement == output_receiver
```


```python
class DataOwner(tfe.Player):
  @tfe.local_computation
  def provide_input(self):
    tf_tensor = ...
    return tf_tensor

data_owners = [
    DataOwner('data_owner_0', '/path/to/datafile/on/0'),
    DataOwner('data_owner_1', '/path/to/datafile/on/1'),
    DataOwner('data_owner_2', '/path/to/datafile/on/2'),
]

output_receiver = tfe.Player('output_receiver')

xs = [
    data_owner.provide_input()
    for data_owner in data_owners
]
assert all(isinstance(x, tfe.LocalTensor) for x in xs)

secure_aggregation = tfe.functionalities.AdditiveSecureAverage(
    computers=data_owners,
    output_receiver=output_receiver
)

y = secure_aggregation(xs)
assert isinstance(y, tfe.LocalTensor)
assert y.placement == output_receiver
```





TFE will offer several built-in functionalities for secure aggregation will be built into TFE, all mapping between local or raw tensors representing gradients. For example, secure aggregation using additive secret sharing could be expressed as follows:

```python
class AdditiveSecureAggregation:
  """
  Fast non-resilient secure average based on additive secret sharing.
  """

  def __init__(computers, output_receiver, aggregation_fn=None):
    self.protocol = tfe.protocols.Pond(computers)
    self.output_receiver = output_receiver
    if aggregation_fn is None:
      aggregation_fn = lambda xs: tfe.add_n(xs) / len(xs)
    self.aggregation_fn = aggregation_fn

  def initialize(self):
    # Some protocols will require e.g. a data-independent setup to
    # be run between the players before aggregation can take place

  @tfe.encrypted_computation
  def __call__(self, xs_plain):
    with self.protocol:
      xs = [
          # new version of tfe.define_private_input;
          # dtype could be left out here
          tfe.cast(x_plain, dtype=x.dtype, stype=tfe.private)
          for x_plain in xs_plain
      ]
      y = self.aggregation_fn(xs)
      y_plain = tfe.reveal(y, self.output_receiver)
      return y_plain
```

which may be used as follows:



And secure aggregation using Paillier encryption and a central aggregator as follows:

```python
class PaillierSecureAverage:
  """
  Centralized secure aggregation based on
  Paillier homomorphic encryption.
  """

  def __init__(input_providers, computer, output_receiver):
    self.input_providers = input_providers
    self.aggregator = aggregator
    self.output_receiver = output_receiver

  def initialize(self):
    # generate key pair to be used across iterations
    self.key_pair = tfe.protocols.Paillier.keypair(
        key_owner=self.output_receiver)

  @tfe.encrypted_computation
  def __call__(self, plaintext_grads):
    paillier = tfe.protocols.Paillier(
        evaluator=self.aggregator,
        key_owner=self.output_receiver,
        key_pair=self.key_pair)
    with paillier:
      grads = [
          tfe.define_private_input(grad, inputter)
          for grad, inputter in zip(
              plaintext_grads,
              self.input_providers)
      ]
      aggregated_grad = tfe.add_n(grads) / len(grads)
      return tfe.reveal(aggregated_grad, self.output_receiver)
```

Note that code used to express the specific aggregation type is generic between the two functionalities and could instead be given as an argument.

Instantiated functionalities are used as functions, matching in style with TF 2.0 while allowing for using optimized graphs under the hood via e.g. `tf.function`:

```python
# initialize once
aggregate.initialize()

# call repeatedly to compute aggregations
aggregated_grad = aggregate(grads)
```

## Detailed Design

This section outlines how TFE may accompanying these through existing primitives such as `tfe.local_computation`. It is roughly an update to the current FL example found in the TFE repo.

```python
class ModelOwner:
  # this will be called on both model and data owners;
  # the weights of the model build on the model owner
  # will be manipulated by the `FederatedLearning`
  # object constructed below
  def build_model(self):
    # use plain TensorFlow Keras
    model = tf.keras.Sequential()
    model.add(tf.keras.Dense())
    model.add(tf.keras.ReLU())
    model.compile(
        optimizer=tf.train.AdamOptimizer(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

model_owner = ModelOwner('model_owner', model_fn)

class DataOwner:

  def __init__(self, data_file):
    self.data_file = data_file

  def build_data_set(self):
    ...

data_owners = [
    DataOwner('data_owner_0', '/path/to/datafile/on/0'),
    DataOwner('data_owner_1', '/path/to/datafile/on/1'),
    DataOwner('data_owner_2', '/path/to/datafile/on/2'),
]

class FederatedLearning:

  TODO

# build the federated learning process
federated = TrainingProcess(
    model_owner=model_owner,
    data_owners=data_owners,
    aggregation=tfe.functionalities.AdditiveSecureAverage)

# initialize everything
federated.initialize()

# fit model for certain number of epochs
federated.fit(epochs=10)
```

Going a step further, some of plumbing could be wrapped up in reusable components under e.g. a `tfe.federated` namespace; candidates are generic versions of the `ModelOwner`, `DataOwner`, and `TrainingProcess` used above.

## Questions and Discussion Topics

- How do we support different communication models and device characteristics, and e.g. allow TFE to simply be used as an crypto oracle and leaving networking to other parts?

- More general question, and somewhat unrelated: Should `tfe.LocalTensor` be renamed to `tfe.PinnedTensor`, and `tfe.local_computation` to `tfe.pinned_computation`? Alternatively `tfe.PlaintextTensor` and `tfe.plaintext_computation`? Semantics are unchanged: all represent values and computations that happen on a specific device, with the former potentially being sensitive values.
