# Using TF Encrypted with Distribution Strategies

| Status        | Proposed |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com) |
| **Sponsor**   | |
| **Updated**   | 2019-09-25 |

## Objective

This document describes how TF Encrypted (TFE) may be used together with distribution strategies to for instance implement a simple federated learning framework.

## Motivation

## Design Proposal

`tfe.integrations.distribute`

From the [guide](https://www.tensorflow.org/guide/distributed_training):

> There are many all-reduce algorithms and implementations available, depending on the type of communication available between devices. By default, it uses NVIDIA NCCL as the all-reduce implementation. You can choose from a few other options we provide, or write your own.

> If you wish to override the cross device communication, you can do so using the cross_device_ops argument by supplying an instance of tf.distribute.CrossDeviceOps. Currently, tf.distribute.HierarchicalCopyAllReduce and tf.distribute.ReductionToOneDevice are two options other than tf.distribute.NcclAllReduce which is the default.

MirroredStrategy: "tf.distribute.MirroredStrategy supports synchronous distributed training on multiple GPUs on one machine."

MultiWorkerMirroredStrategy: "tf.distribute.experimental.MultiWorkerMirroredStrategy is very similar to MirroredStrategy. It implements synchronous distributed training across multiple workers, each with potentially multiple GPUs. Similar to MirroredStrategy, it creates copies of all variables in the model on each device across all workers."

subclassing [`CrossDeviceOps`](https://www.tensorflow.org/api_docs/python/tf/distribute/CrossDeviceOps)


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

## Detailed Design

## Questions and Discussion Topics
