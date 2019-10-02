# Secure Aggregation with TF Encrypted

| Status        | Proposed |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com) |
| **Sponsor**   | |
| **Updated**   | 2019-09-25 |

## Objective

This document describes the integration between TF Encrypted (TFE) and TensorFlow Federated (TFF), with the former providing encrypted computations for the latter. In particular, the use of TFE for secure aggregation is detailed.

## Motivation

## Design Proposal

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

## Questions and Discussion Topics

- Do we want generic classes and a `tfe.federated` namespace?
- How do we support different communication models and device characteristics, and e.g. allow TFE to simply be used as an crypto oracle and leaving networking to other parts
