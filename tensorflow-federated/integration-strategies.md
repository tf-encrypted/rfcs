# Appendix: Integration Strategies

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
