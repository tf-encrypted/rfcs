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
- `tfe.Tensor.with_sensitivity({})` is top, `tfe.Tensor.with_sensitivity(None)` is bottom; note that `None` here means the set of all players and hence `None != {}`.
- When encrypting a tensor the sensitivity of the encrypted tensors is copied from the plaintext tensor; likewise when decrypting.
- When combining tensors the sensitivity must match (after a potential application of subtyping).

### Secure Aggregation Example

For secure aggregation for federated learning we obtain:

1) Model weights with type `PlaintextTensor(None)`.
2) Local data with type `PlaintextTensor(do1)`, ..., `PlaintextTensor(doN)`.
3) Local gradients with type `PlaintextTensor(do1)`, ..., `PlaintextTensor(doN)` (after subtyping the weights).
4) Local encryptions with type `EncryptedTensor(do1)`, ..., `EncryptedTensor(doN)`.
5) Central encryptions with type `EncryptedTensor({})`, ..., `EncryptedTensor({})` (after subtyping).
6) Central encryption of aggregation with type `EncryptedTensor({})`.
7) Central encryption of aggregation with type `EncryptedTensor({mo})` after application of `broaden({mo})`; note that this is expressing the policy that it is okay for the model owner to learn aggregated gradients.
8) Plaintext aggregated gradient of type `PlaintextTensor({mo})` on the model owner.

### Private Prediction Example

In this case the model owner is okay to share the weights with the compute servers in plaintext, but not to the prediction client (in which case the computation could just happen locally).

1) Model weights with type `PlaintextTensor({mo})`; broadened to `PlaintextTensor({mo, s0, s1})` to indicate security policy; note that sending these to the prediction client would hence raise an error.
2) Prediction input with type `PlaintextTensor({pc})`, encrypted to `EncryptedTensor({pc})` on the prediction client.
3) Central computation with inputs of type `PlaintextTensor({})` and `EncryptedTensor({})` after subtyping, and result of type `EncryptedTensor({})`.
4) `broaden({pc})` is applied to result, indicating that it's okay to release result back to prediction client (but no one else), and obtaining type `EncryptedTensor({pc})`.
5) Prediction client decrypts to obtain `PlaintextTensor({pc})`.

## Detailed Design

## Questions and Discussion Topics
