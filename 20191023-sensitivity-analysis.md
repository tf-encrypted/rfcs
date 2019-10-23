# Automated Sensitivity Analysis

| Status        | Draft |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com) |
| **Sponsor**   | |
| **Updated**   | 2019-10-23                                           |

## Objective

In this proposal we outline an approach for automating the security analysis of TF Encrypted (TFE) computations based on the sensitivity property of values.

Our approach takes the form of an effect type system and can be used at both compile and runtime time, the former allowing to catch errors even in computations where type erasure is applied before execution (such as when encrypted computation are compiled down to e.g. raw TensorFlow graphs).

We add tools and syntactic constructs to the `tfe.analysis` module.

## Motivation

Assuming perfect encryption, it can still be hard for users to keep track of who gets to see which values in a computation, i.e. whether the desired security policy is in fact enforced. This proposal introduces a formal framework to reason about security policies as well as tools to help automate the analysis process, thereby helping users' confidence.

## Design Proposal

To be expanded:

- Type system based on the built-in tensor types and their sensitivity property.
- An error is raised if a plaintext tensor is ever found on a player that is *not* in its sensitivity set; this can be checked at compile time and, optionally, at runtime.
- Subtyping allows for implicitly restricting sensitivity by removing players from the set: `T(S) <: T'(S')` if `S'` is subset of `S`.
- `tfe.analysis.broaden` must be used to broaden sensitivity by adding players to the set: `broaden_S(x) : T(S union S')` when `x: T(S')`; this makes it syntactically clear to the user where extra attention must be paid; no-op used by the type system, similar to type hints.
- `tfe.Tensor.with_sensitivity([])` is top, `tfe.Tensor.with_sensitivity(None)` is bottom; note that `None` here means the set of all players and hence `None != []`.
- When encrypting a tensor the sensitivity of the encrypted tensors is copied from the plaintext tensor; likewise when decrypting.
- When combining tensors the sensitivity must match (after a potential application of subtyping).

As an example, in the case of secure aggregation for federated learning we obtain:

1) Model weights with type `PlaintextTensor(None)`.
2) Local data with type `PlaintextTensor(do1)`, ..., `PlaintextTensor(doN)`.
3) Local gradients with type `PlaintextTensor(do1)`, ..., `PlaintextTensor(doN)` (after subtyping the weights).
4) Local encryptions with type `EncryptedTensor(do1)`, ..., `EncryptedTensor(doN)`.
5) Central encryptions with type `EncryptedTensor([])`, ..., `EncryptedTensor([])` (after subtyping).
6) Central encryption of aggregation with type `EncryptedTensor([])`.
7) Central encryption of aggregation with type `EncryptedTensor([mo])` after application of `broaden([mo])`; note that this is expressing the policy that it is okay for the model owner to learn aggregated gradients.
8) Plaintext aggregated gradient of type `PlaintextTensor([mo])` on the model owner.

## Detailed Design

## Questions and Discussion Topics
