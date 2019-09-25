# Terminology in TF Encrypted

| Status        | Proposed |
:-------------- |:----------------------------------------------------
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com) |
| **Sponsor**   | |
| **Updated**   | 2019-10-25 |

## Objective

This document defines the general terminology used in TF Encrypted, offering guidelines for how to understand and name concepts.

## Design Proposal

- **Raw tensors** are the built-in TensorFlow tensors.

- **Local tensors** are a specific type of TFE tensors used to represent plaintext values held locally by a player. Technically they are just small wrappers over raw tensors, but including extra metadata about locality and ownership. This metadata is used by e.g. kernels and protocols, as well as for detecting privacy violations.

- **Operations** are abstract targets used when defining secure computations, for instance `tfe.matmul` and `tfe.keras.Dense`.

- **Kernels** are concrete implementations of operations.

- **Protocols** are sets of kernels, and are hence not specific about *what* is being computed but rather *how* it is. They are typically used as context handlers, under which an abstract specification of a function is made concrete via the implied mapping from operations to kernels. Note that this roughly matches how "protocol" is used in e.g. MPC.

- **Functionalities** are protocols implementing a specific task and exposed as functions, for instance mapping from and to local tensors. This roughly follows the idea of ideal functionality from the UC framework yet are more intended to be user-facing.
