
# Inside TensorFlow Federated

This text presents background material on TensorFlow Federated (TFF), in particular related to Federated Core (FC). It is based on [release 0.9.0](https://github.com/tensorflow/federated/releases/tag/v0.9.0).

Suggested reading, ordered:

- [Federated Core overview](https://github.com/tensorflow/federated/blob/v0.9.0/docs/federated_core.md)
- Custom Federated Algorithms, [Part 1](https://github.com/tensorflow/federated/blob/v0.9.0/docs/tutorials/custom_federated_algorithms_1.ipynb) and [Part 2](https://github.com/tensorflow/federated/blob/v0.9.0/docs/tutorials/custom_federated_algorithms_2.ipynb)
- [ProtoBuf files](https://github.com/tensorflow/federated/tree/v0.9.0/tensorflow_federated/proto/v0)
- [API source code](https://github.com/tensorflow/federated/tree/v0.9.0/tensorflow_federated/python/core/api)
- [Implementation source code](https://github.com/tensorflow/federated/tree/v0.9.0/tensorflow_federated/python/core/impl)

### Goals

- The [goal](https://github.com/tensorflow/federated/blob/v0.9.0/docs/federated_core.md#goals-intended-uses-and-scope) of Federated Core (FC) is to capture federated algorithms from a global system-view perspective that can be executed in a variety of environments, using building blocks such as `tff.federated_sum`, `tff.federated_reduce`, or `tff.federated_broadcast`, instead of the typical lower-level `send` and `receive`.

- These algorithms are expressed in a [strongly-typed](https://github.com/tensorflow/federated/blob/v0.9.0/docs/federated_core.md#type-system) functional glue [language](https://github.com/tensorflow/federated/blob/v0.9.0/docs/federated_core.md#language) focused on the global behavior of distributed systems, as opposed to the local behavior of individual participants; these [federated computations](https://github.com/tensorflow/federated/blob/v0.9.0/docs/tutorials/custom_federated_algorithms_1.ipynb#federated_computations) are typically specified using decorated Python functions.

- On [FC vs distribution strategies](https://github.com/tensorflow/federated/blob/v0.9.0/docs/tutorials/custom_federated_algorithms_1.ipynb#intended-uses): <em>The stated goal of tf.contrib.distribute is to allow users to use existing models and training code with minimal changes to enable distributed training, and much focus is on how to take advantage of distributed infrastructure to make existing training code more efficient. The goal of TFF's Federated Core is to give researchers and practitioners explicit control over the specific patterns of distributed communication they will use in their systems. The focus in FC is on providing a flexible and extensible language for expressing distributed data flow algorithms, rather than a concrete set of implemented distributed training capabilities.</em>

- Assumptions are that TFF can be used on devices where only a TF environment is available, e.g. without the ability to execute Python code.

### Data and Types

- [Federated data](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1#federated_data) is a collection of data items hosted across a group of devices in a distributed system and modelled as a single federated value.

- Types include [tensors similar to tf.Tensor and sequences similar tf.data.Dataset](https://www.tensorflow.org/federated/federated_core#type_system).

- [Types](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L423) in computations contain high-level [placement](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1#placements) [annotations](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L371) to identify abstract groups of protocol players such as `CLIENTS`, `AGGREGATORS`, and `SERVER`.

### Defining Computations

- [Computations](https://github.com/tensorflow/federated/blob/v0.9.0/docs/federated_core.md#building_blocks) are typically constructed using the [high-level Python API](https://github.com/tensorflow/federated/tree/v0.9.0/tensorflow_federated/python/core/api) and serialized as [protobuf](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L24) files.

- Ordinary [TensorFlow computations](https://github.com/tensorflow/federated/blob/v0.9.0/docs/tutorials/custom_federated_algorithms_1.ipynb#declaring_tensorflow_computations) are expressed using `tff.tf_computation` or `tff.tf2_computation` decorated functions; federated computations are expressed using `tff.federated_computation` decorated functions.

- The leaf nodes of computations are a mix of:

  - [Annotated TF GraphDefs](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L513) to be executed by TF.
  
  - [Intrinsic URIs](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L631) referencing [built-in federated operators](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/compiler/intrinsic_defs.py#L186-L469) such as `federated_mean`; the actual implementation of these is free and only given at runtime by the executor/compiler.
  
  - [Data references](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L818) to external sources.

- Higher-order computations are functional and expressible though [typed lambda expressions](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L652) and [blocks](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L710), allowing one to e.g. orchestrate client and server behavior.

The example below is given in [computation.proto](https://github.com/tensorflow/federated/blob/8cff0bf8703937393730ab6e635bc8dbe5f90cd6/tensorflow_federated/proto/v0/computation.proto#L45-L90), yet it seems slightly outdated. It shows that the following TFF code:

```python
@tff.computation
def fed_eval(model):

  @tf.function
  def local_eval(model):
    ...
    return {'loss': ..., 'accuracy': ...}

  client_model = tff.federated_broadcast(model)
  client_metrics = tff.federated_map(local_eval, client_model)
  return tff.federated_mean(client_metrics)
```

is serialized into the following AST:

```python
fed_eval = Computation(lambda=Lambda(
  parameter_name='model',
  result=Computation(block=Block(
    local=[
      Block.Local(name='local_eval', value=Computation(
        tensorflow=TensorFlow(...))),
      Block.Local(name='client_model', value=Computation(
        call=Call(
          function=Computation(
            intrinsic=Intrinsic(uri='federated_broadcast')),
          argument=Computation(
            reference=Reference(name='model'))))),
      Block.Local(name='client_metrics', value=Computation(
        call=Call(
          function=Computation(
            intrinsic=Intrinsic(uri='federated_map')),
          argument=Computation(
            tuple=Tuple(element=[
              Tuple.Element(
                value=Computation(
                  reference=Reference(
                    name='local_eval'))),
              Tuple.Element(
                value=Computation(
                  reference=Reference(
                    name='local_client_model')))
    ])))))],
    result=Computation(
      call=Call(
        function=Computation(
          intrinsic=Intrinsic(uri='federated_mean')),
      argument=Computation(
        reference=Reference(name='client_metrics')))
)))))
```

### Computation Contexts

- These are used when defining computations, as opposed to execution context which are used when running computations.

- [Computation wrappers](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_wrapper.py#L118-L356) are the decorators used to convert Python functions into computations; current instantiations are [tensorflow_wrapper](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_wrapper_instances.py#L48), [tf2_wrapper](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_wrapper_instances.py#L61), and [federated_computation_wrapper](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_wrapper_instances.py#L82), which are [exposed in the API](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/api/computations.py) as [tff.tf_computation](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/api/computations.py#L24), [tff.tf2_computation](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/api/computations.py#L159), and [tff.federated_computation](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/api/computations.py#L179), respectively.

- The wrapped function is either a [ConcreteFunction](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_wrapper.py#L115) or a [PolymorphicFunction](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_wrapper.py#L100), the former [in the form](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_wrapper_instances.py#L32-L83) of a [ComputationImpl](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/computation_impl.py#L30-L31); when called, the latter [instantiates and invokes itself as a concrete function](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/utils/function_utils.py#L685-L699).

- ttf.tf_computation [switches](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/tensorflow_serialization.py#L275-L277) the current context to a [TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/tf_computation_context.py#L31-L32), and tff.federated_computation [switches](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/federated_computation_utils.py#L66-L78) it to a [FederatedComputationContext](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/federated_computation_context.py#L31-L32); this changes the invocation behavior of previously defined computations.

### Execution and Execution Contexts

- The root context is used for [function invocation](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/utils/function_utils.py#L667); [the default](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/context_stack_impl.py#L31-L33) is a ReferenceExecutor but the intended behavior seems to be to [replace](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/context_stack_impl.py#L43-L50) this with an [ExecutionContext wrapping an executor constructor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/execution_context.py#L129-L130) via [set_default_executor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/set_default_executor.py#L23-L34):

  - [Simulation script](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/simulation/worker.py#L39-L41) where the executor is used directly via an [ExecutorService](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_service.py#L36-L37) without context
  - [Remote execution example](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/tools/client/test.py#L36-L40) which uses executors wrapping a RemoteExecutor
  - [Core test utilities](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/utils/test.py#L116) which uses [create_local_executor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_stacks.py#L152-L175)

- [ReferenceExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/reference_executor.py#L552) is a self-contained execution environment, [intended to be used](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/reference_executor.py#L555-L565) on its own in notebooks and in tests as a source of correctness; despite its name it is *not* an [Executor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_base.py) but instead a [Context](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/context_base.py#L27-L42) ([for now](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_base.py#L26)); it lacks all aspects of actually running in live environment.

- Several other executors are available for a more realistic execution, all implementing the abstract [Executor interface](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_base.py) and used in a [composable manner](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_stacks.py); note the pattern of first embedding items using e.g. [create_value](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_base.py#L33-L52) or [create_call](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_base.py#L56-L69) before subsequently calling [compute](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_value_base.py#L22-L42); *target executor* below refers to a wrapped executor:

  - [LambdaExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/lambda_executor.py#L173-L192) handles compositional constructs, including lambdas, blocks, references, calls, tuples, and selections, and orchestrates the execution of these constructs, while delegating all the non-compositional constructs (tensorflow, intrinsic, data, or placement) to a target executor.

  - [CachingExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/caching_executor.py#L169) delegates work to a target executor and caches the results.

  - [ConcurrentExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/concurrent_executor.py#L26-L30) runs everything on a target executor using a background thread.

  - [EagerExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/eager_executor.py#L295-L324) runs TF computations synchronously in eager mode, as well as external data identified by local filenames; it is able to place work on specific devices, can handle data sets in a pipelined fashion, does not place limits on the data set sizes, and avoids marshaling TensorFlow values in and out between calls.

  - [RemoteExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/remote_executor.py#L141-L142) is a local proxy for a [remote executor service](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_service.py#L36-L37) hosted on a separate machine and accessible through [gRPC](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/proto/v0/executor.proto); execution happens asynchronously at the protobuf level; currently only used by [simulation](https://github.com/tensorflow/federated/tree/v0.9.0/tensorflow_federated/python/simulation) through the [worker.py](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/simulation/worker.py#L57) script.

  - [FederatedExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/federated_executor.py#L120-L152) orchestrates federated computations by handling federated operations itself and delegating local operations to target executors associated with the concrete parties; implements a list of intrinsic functions such as [federated_reduce](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/federated_executor.py#L501-L541), [federated_sum](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/federated_executor.py#L583-L601), and [federated_mean](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/federated_executor.py#L603-L620).

  - [CompositeExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/composite_executor.py#L120) allows defining hierarchies of executors for running several executor stacks together; was some overlap with FederatedExecutor in that they both e.g. implement some of the same intrinsic functions; when used in combination, this means that aggregation is done is stages as per the hierarchy.

  - [TransformingExecutor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/transforming_executor.py#L25-L43) applies a given transformation function on each call to create_value; does not currently seem to be in use.

- [Simulation](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/simulation/worker.py) is currently done using [create_local_executor](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_stacks.py#L152), which creates and composes [executor stacks](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_stacks.py) on all placements; this is done by [composing](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_stacks.py) the executors above.

Calling the executor constructor produced by [create_local_executor(4, 2)](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/executor_stacks.py#L152-L175) results in the follow structure of executors:

```
Lambda(Caching(Concurrent(Composite)))
  Lambda(Caching(Concurrent(Eager)))  <-- parent
  Lambda(Caching(Concurrent(Federated)))  <-- child
    Lambda(Caching(Concurrent(Eager)))  <-- CLIENTS
    Lambda(Caching(Concurrent(Eager)))  <-- CLIENTS
    Lambda(Caching(Concurrent(Eager)))  <-- SERVER
    Lambda(Caching(Concurrent(Eager)))  <-- None
  Lambda(Caching(Concurrent(Federated)))  <-- child
    Lambda(Caching(Concurrent(Eager)))  <-- CLIENTS
    Lambda(Caching(Concurrent(Eager)))  <-- CLIENTS
    Lambda(Caching(Concurrent(Eager)))  <-- SERVER
    Lambda(Caching(Concurrent(Eager)))  <-- None
```

### Compiler

- Functionality from `core/impl/compiler` used all over the place, including currently supported [placement literals](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/compiler/placement_literals.py#L65-L75), [intrinsic](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/compiler/intrinsic_defs.py#L30-L472), and [building blocks](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/compiler/building_blocks.py).

- ReferenceExecutor is currently [the only user](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/context_stack_impl.py#L31-L33) of the [compiler pipeline](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/compiler_pipeline.py#L31-L40); it uses it to perform [a few transformations](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/compiler_pipeline.py#L51-L94), including replacing a subset of the intrinsic functions with [their definitions](https://github.com/tensorflow/federated/blob/v0.9.0/tensorflow_federated/python/core/impl/intrinsic_bodies.py).

### Extensibility

A custom executor can be implemented as follows (in this particular case delegating all work to the target executor):

```python
from tensorflow_federated.python.core.impl.executor_base import Executor

class MyExecutor(Executor):

  def __init__(self, target_executor):
    self._target_executor = target_executor

  async def create_value(self, value, type_spec=None):
    return await self._target_executor.create_value(value, type_spec=type_spec)

  async def create_call(self, comp, arg=None):
    return await self._target_executor.create_call(comp, arg)

  async def create_tuple(self, elements):
    return await self._target_executor.create_tuple(elements)

  async def create_selection(self, source, index=None, name=None):
    return await self._target_executor.create_selection(source, index=index, name=name)
```

A custom execution context can be implemented and used as follows (in this particular case delegating all work to the built-in execution context):

```python
from tensorflow_federated.python.core.impl.context_base import Context
from tensorflow_federated.python.core.impl.execution_context import ExecutionContext
from tensorflow_federated.python.core.impl.context_stack_impl import context_stack

class MyExecutionContext(Context):

  def __init__(self, executor_fn):
    self._target_context = ExecutionContext(executor_fn)

  def ingest(self, val, type_spec):
    return self._target_context.ingest(val, type_spec))

  def invoke(self, comp, arg):
    return self._target_context.invoke(comp, arg)

def set_default_executor(executor_fn):
  context = MyExecutionContext(executor_fn)
  context_stack.set_default_context(context)
```
