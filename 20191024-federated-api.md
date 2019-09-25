# Secure Aggregation with TF Encrypted

| Status        | Proposed |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com) |
| **Sponsor**   | |
| **Updated**   | 2019-09-25 |

## Objective

Secure aggregation has played a role in federated learning since the beginning. This document describes the secure aggregation functionalities available in TF Encrypted (TFE) and how they fit into federated learning (FL) processes expressed in plain TensorFlow (TF), in TensorFlow Federated (TFF), and via distribution strategies.

To this end the document presents a general design of secure aggregation functionalities as well as:

- a design for their use in TFF
- a design for their use in distribution strategies
- a discussion of how they can be used to build FL systems in plain TF

## Motivation

This design was developed under the following requirements:

- minimize cognitive burden on users:
  - match and integrate with TF and TFF as much as possible
  - maximize common ground across TFE and minimize the introduction of new concepts and terminology specific to this application
  - minimize ambiguity and ensure that every construct is well-motivated
- ensure flexibility for experimentation with:
  - machine learning models
  - aggregation functions
  - cryptographic techniques and guarantees
- be usable from TF 2.0, supporting eager mode as much as possible

The interest in building FL processes in plain TF was motivated by the yet unsettled landscape.

## Questions and Discussion Topics

- do we want generic classes and a `tfe.federated` namespace?
- how do we support different communication models and device characteristics, and e.g. allow TFE to simply be used as an crypto oracle and leaving networking to other parts

## Design Proposal

### Secure aggregation functionalities

Several functionalities for secure aggregation will be built into TFE, all mapping between local or raw tensors representing gradients. For example, secure aggregation using additive secret sharing via Pond could be expressed as follows:

```python
class AdditiveSecureAverage:
  """
  Non-resilient distributed secure aggregation
  based on additive secret sharing.
  """

  def __init__(input_providers, computers, output_receiver):
    self.input_providers = input_providers
    self.computers = computers
    self.output_receiver = output_receiver

  def initialize(self):
    # Some protocols will require e.g. a data-independent setup to
    # be run between the players before aggregation can take place

  def __call__(self, plaintext_grads):
    pond = tfe.protocols.Pond(self.computers)
    with pond:
      grads = [
          tfe.define_private_input(grad, inputter)
          for grad, inputter in zip(
              plaintext_grads,
              self.input_providers)
      ]
      aggregated_grad = tfe.add_n(grads) / len(grads)
      return tfe.reveal(aggregated_grad, self.output_receiver)
```

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

### From-scratch using plain TensorFlow

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

### Interfacing into TF Federated

**TODO**

### Interfacing with distribution strategies

It is not clear to what extent distribution strategies can use data already present on devices, as opposed to first being distributed from a central locations. 

From the [guide](https://www.tensorflow.org/guide/distributed_training):

> There are many all-reduce algorithms and implementations available, depending on the type of communication available between devices. By default, it uses NVIDIA NCCL as the all-reduce implementation. You can choose from a few other options we provide, or write your own.

> If you wish to override the cross device communication, you can do so using the cross_device_ops argument by supplying an instance of tf.distribute.CrossDeviceOps. Currently, tf.distribute.HierarchicalCopyAllReduce and tf.distribute.ReductionToOneDevice are two options other than tf.distribute.NcclAllReduce which is the default.

MirroredStrategy: "tf.distribute.MirroredStrategy supports synchronous distributed training on multiple GPUs on one machine."

MultiWorkerMirroredStrategy: "tf.distribute.experimental.MultiWorkerMirroredStrategy is very similar to MirroredStrategy. It implements synchronous distributed training across multiple workers, each with potentially multiple GPUs. Similar to MirroredStrategy, it creates copies of all variables in the model on each device across all workers."

subclassing [`CrossDeviceOps`](https://www.tensorflow.org/api_docs/python/tf/distribute/CrossDeviceOps)

## Appendix: Relevant Context

This section outlines background context relevant to understanding the decisions made here.

### TF Distribute Strategies

From [the docs](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/distribute/Strategy#class_strategy):

> distribute strategies are about *state & compute distribution policy on a list of devices*

From [the guide](https://www.tensorflow.org/guide/distribute_strategy#using_tfdistributestrategy_with_keras):

> The only things that need to change in a user's program are: (1) Create an instance of the appropriate `tf.distribute.Strategy` and (2) Move the creation and compiling of Keras model inside `strategy.scope`.

> `strategy.scope()` indicated which parts of the code to run distributed. Creating a model inside this scope allows us to create mirrored variables instead of regular variables. Compiling under the scope allows us to know that the user intends to train this model using this strategy. Once this is setup, you can fit your model like you would normally [i.e. outside scope].

```python
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  model.compile(loss='mse', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)
```

### TF Federated

From [the docs](https://www.tensorflow.org/federated/federated_learning#models): 

> Currently, TensorFlow does not fully support serializing and deserializing eager-mode TensorFlow. Thus, serialization in TFF currently follows the TF 1.0 pattern, where all code must be constructed inside a `tf.Graph` that TFF controls. This means currently TFF cannot consume an already-constructed model; instead, the model definition logic is packaged in a no-arg function that returns a `tff.learning.Model`. This function is then called by TFF to ensure all components of the model are serialized.

On TF distribution strategies vs Federated Core, from the [docs](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1#intended_uses):

> You may be aware of tf.contrib.distribute, and a natural question to ask at this point may be: in what ways does this framework differ? Both frameworks attempt at making TensorFlow computations distributed, after all.

> One way to think about it is that, whereas the stated goal of tf.contrib.distribute is to allow users to use existing models and training code with minimal changes to enable distributed training, and much focus is on how to take advantage of distributed infrastructure to make existing training code more efficient, the goal of TFF's Federated Core is to give researchers and practitioners explicit control over the specific patterns of distributed communication they will use in their systems. The focus in FC is on providing a flexible and extensible language for expressing distributed data flow algorithms, rather than a concrete set of implemented distributed training capabilities.

> One of the primary target audiences for TFF's FC API is researchers and practitioners who might want to experiment with new federated learning algorithms and evaluate the consequences of subtle design choices that affect the manner in which the flow of data in the distributed system is orchestrated, yet without getting bogged down by system implementation details. The level of abstraction that FC API is aiming for roughly corresponds to pseudocode one could use to describe the mechanics of a federated learning algorithm in a research publication - what data exists in the system and how it is transformed, but without dropping to the level of individual point-to-point network message exchanges.

> TFF as a whole is targeting scenarios in which data is distributed, and must remain such, e.g., for privacy reasons, and where collecting all data at a centralized location may not be a viable option. This has implication on the implementation of machine learning algorithms that require an increased degree of explicit control, as compared to scenarios in which all data can be accumulated in a centralized location at a data center.


> the stated goal of `tf.distribute` is to allow users *to use existing models and training code with minimal changes to enable distributed training*, and much focus is on how to take advantage of distributed infrastructure to make existing training code more efficient. The goal of TFF's Federated Core is to give researchers and practitioners explicit control over the specific patterns of distributed communication they will use in their systems. The focus in FC is on providing a flexible and extensible language for expressing distributed data flow algorithms, rather than a concrete set of implemented distributed training capabilities.
>
> One of the primary target audiences for TFF's FC API is researchers and practitioners who might want to experiment with new federated learning algorithms and evaluate the consequences of subtle design choices that affect the manner in which the flow of data in the distributed system is orchestrated, yet without getting bogged down by system implementation details. The level of abstraction that FC API is aiming for roughly corresponds to pseudocode one could use to describe the mechanics of a federated learning algorithm in a research publication - what data exists in the system and how it is transformed, but without dropping to the level of individual point-to-point network message exchanges.

From [the tutorial on text](https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation):

```python
def create_tff_model():
  ...
  keras_model_clone = compile(tf.keras.models.clone_model(keras_model))
  return tff.learning.from_compiled_keras_model(
      keras_model_clone, dummy_batch=dummy_batch)

# This command builds all the TensorFlow graphs and serializes them
fed_avg = tff.learning.build_federated_averaging_process(model_fn=create_tff_model)

# Perform federated training steps
state = fed_avg.initialize()
state, metrics = fed_avg.next(state, [example_dataset.take(1)])
print(metrics)
```

Note that `state` can used to update a local clone of the model for evaluation after each iteration:

```python
state = fed_avg.initialize()

state = tff.learning.state_with_new_model_weights(
    state,
    trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
    non_trainable_weights=[
        v.numpy() for v in keras_model.non_trainable_weights
    ])

def keras_evaluate(state, round_num):
  tff.learning.assign_weights_to_keras_model(keras_model, state.model)
  print('Evaluating before training round', round_num)
  keras_model.evaluate(example_dataset, steps=2)

for round_num in range(NUM_ROUNDS):
  keras_evaluate(state, round_num)
  state, metrics = fed_avg.next(state, train_datasets)
  print('Training metrics: ', metrics)

keras_evaluate(state, NUM_ROUNDS + 1)
```
