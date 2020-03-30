from enum import Enum

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow.python.framework.ops import EagerTensor as tf_EagerTensor

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.context_stack.context_base import Context
from tensorflow_federated.python.core.impl.context_stack.context_stack_impl import context_stack
from tensorflow_federated.python.core.impl.executors.execution_context import ExecutionContext
from tensorflow_federated.python.core.impl.executors.execution_context import ExecutionContextValue
from tensorflow_federated.python.core.impl.executors.executor_base import Executor
from tensorflow_federated.python.core.impl.executors.executor_factory import ExecutorFactoryImpl
from tensorflow_federated.python.core.impl.executors.executor_value_base import ExecutorValue

from tensorflow_federated.python.core.impl.executors.caching_executor import CachingExecutor
from tensorflow_federated.python.core.impl.executors.composing_executor import ComposingExecutor
from tensorflow_federated.python.core.impl.executors.thread_delegating_executor import ThreadDelegatingExecutor
from tensorflow_federated.python.core.impl.executors.eager_tf_executor import EagerTFExecutor
from tensorflow_federated.python.core.impl.executors.federating_executor import FederatingExecutor
from tensorflow_federated.python.core.impl.executors.reference_resolving_executor import ReferenceResolvingExecutor
from tensorflow_federated.python.core.impl.executors.remote_executor import RemoteExecutor
from tensorflow_federated.python.core.impl.executors.transforming_executor import TransformingExecutor

VISIT_RECURSION_STRATEGIES = dict()


def register_recursion_strategy(executor_type, recursion_strategy):
  VISIT_RECURSION_STRATEGIES[executor_type] = recursion_strategy


class ExecutorStackVisitor:

  def visit_before(self, executor):
    pass

  def visit_after(self, executor):
    pass


def visit_executor_stack(visitor, executor_stack):
  assert isinstance(visitor, ExecutorStackVisitor)
  visitor.visit_before(executor_stack)
  children = VISIT_RECURSION_STRATEGIES[type(executor_stack)](executor_stack)
  for child in children:
    visit_executor_stack(visitor, child)
  visitor.visit_after(executor_stack)


DEFAULT_STRATEGY = lambda x: "{} {}".format(x, type(x))
FORMATTING_STRATEGIES = dict()


def register_formatting_strategy(ty, formatting_strategy):
  FORMATTING_STRATEGIES[ty] = formatting_strategy


def format_object(x):
  strategy = FORMATTING_STRATEGIES.get(type(x), DEFAULT_STRATEGY)
  return strategy(x)


def monkey_patch_executor_value(executor_value):
  assert isinstance(executor_value, ExecutorValue)

  async def compute(self):
    return await executor_value._real_compute()

  executor_value._real_compute = executor_value.compute
  executor_value.compute = compute


class IndentStrategy(Enum):
  STACK_LEVEL = 0
  CALL_LEVEL = 1


_CURRENT_INDENT_STRATEGY = None
_CURRENT_CALL_LEVEL = 0


def set_indent_strategy(strategy):
  assert type(strategy) == IndentStrategy
  global _CURRENT_INDENT_STRATEGY
  _CURRENT_INDENT_STRATEGY = strategy


set_indent_strategy(IndentStrategy.CALL_LEVEL)


def monkey_patch_executor(executor):
  assert isinstance(executor, Executor)

  def prefix(executor):
    if _CURRENT_INDENT_STRATEGY == IndentStrategy.STACK_LEVEL:
      indentation = '  ' * executor.stack_level
    elif _CURRENT_INDENT_STRATEGY == IndentStrategy.CALL_LEVEL:
      indentation = '  ' * _CURRENT_CALL_LEVEL
    else:
      raise ValueError("Unknown indentation strategy '{}'".format(_CURRENT_INDENT_STRATEGY))

    return "{indentation}{type} {hash}".format(
        indentation=indentation,
        type=type(executor).__name__,
        hash=hash(executor),
    )

  async def create_value(value, type_spec=None):
    print(prefix(executor), "create_value call", format_object(value), ";", format_object(type_spec))
    global _CURRENT_CALL_LEVEL
    _CURRENT_CALL_LEVEL += 1
    res = await executor._real_create_value(value, type_spec=type_spec)
    _CURRENT_CALL_LEVEL -= 1
    print(prefix(executor), "create_value retr", format_object(res))
    return res

  executor._real_create_value = executor.create_value
  executor.create_value = create_value

  async def create_call(comp, arg=None):
    print(prefix(executor), "create_call call", format_object(comp), ";", format_object(arg))
    global _CURRENT_CALL_LEVEL
    _CURRENT_CALL_LEVEL += 1
    res = await executor._real_create_call(comp, arg)
    _CURRENT_CALL_LEVEL -= 1
    print(prefix(executor), "create_call retr", format_object(res))
    return res

  executor._real_create_call = executor.create_call
  executor.create_call = create_call

  async def create_tuple(elements):
    print(prefix(executor), "create_tuple call", format_object(elements))
    res = await executor._real_create_tuple(elements)
    print(prefix(executor), "create_tuple retr", format_object(res))
    return res

  executor._real_create_tuple = executor.create_tuple
  executor.create_tuple = create_tuple

  async def create_selection(source, index=None, name=None):
    print(prefix(executor), "create_selection call", format_object(index), format_object(name))
    res = await executor._real_create_selection(source, index=index, name=name)
    print(prefix(executor), "create_selection retr", format_object(res))
    return res

  executor._real_create_selection = executor.create_selection
  executor.create_selection = create_selection


def monkey_patch_executor_stack(executor_stack):

  class MonkeyPatchVisitor(ExecutorStackVisitor):
    level = 0
    def visit_before(self, executor):
      executor.stack_level = self.level
      monkey_patch_executor(executor)
      self.level += 1
    def visit_after(self, executor):
      self.level -= 1

  visit_executor_stack(MonkeyPatchVisitor(), executor_stack)


def monkey_patch_context(context):
  assert isinstance(context, Context)

  def prefix(context):
    return "{type} {hash}".format(
        type=type(context).__name__,
        hash=hash(context),
    )

  def ingest(val, type_spec):
    print(prefix(context), "ingest call", format_object(val), format_object(type_spec))
    res = context._real_ingest(val, type_spec)
    print(prefix(context), "ingest retr", format_object(res))
    return res

  context._real_ingest = context.ingest
  context.ingest = ingest

  def invoke(comp, arg):
    print(prefix(context), "invoke call", format_object(comp), format_object(arg))
    res = context._real_invoke(comp, arg)
    print(prefix(context), "invoke retr", format_object(res))
    return res

  context._real_invoke = context.invoke
  context.invoke = invoke


def dump_executor_stack(executor_stack):

  class DumpVisitor(ExecutorStackVisitor):
    level = 0
    def visit_before(self, executor):
      print("{indentation}{type} {hash}".format(
          indentation='  ' * self.level,
          type=type(executor).__name__,
          hash=hash(executor),
      ))
      self.level += 1
    def visit_after(self, executor):
      self.level -= 1

  visit_executor_stack(DumpVisitor(), executor_stack)


def dumping_executor_fn(executor_fn):
  def dump_executor_fn(mapping):
    executor_stack = executor_fn(mapping)
    print("Created executor stack")
    dump_executor_stack(executor_stack)
    return executor_stack
  return dump_executor_fn


def monkey_patching_executor_fn(executor_fn):
  def patched_executor_fn(mapping):
    executor_stack = executor_fn(mapping)
    monkey_patch_executor_stack(executor_stack)
    return executor_stack
  return patched_executor_fn


def set_default_executor(executor_fn):
  executor_fn = monkey_patching_executor_fn(executor_fn)
  executor_fn = dumping_executor_fn(executor_fn)
  executor_fn = ExecutorFactoryImpl(executor_fn)
  context = ExecutionContext(executor_fn)
  monkey_patch_context(context)
  context_stack.set_default_context(context)


#
# Registration of built-in types
#

register_recursion_strategy(CachingExecutor, lambda ex: [ex._target_executor])
register_recursion_strategy(ComposingExecutor, lambda ex: [ex._parent_executor] + ex._child_executors)
register_recursion_strategy(ThreadDelegatingExecutor, lambda ex: [ex._target_executor])
register_recursion_strategy(EagerTFExecutor, lambda _: [])
register_recursion_strategy(FederatingExecutor,
    lambda ex: ex._target_executors[None] \
             + ex._target_executors[tff.SERVER] \
             + ex._target_executors[tff.CLIENTS])
register_recursion_strategy(ReferenceResolvingExecutor, lambda ex: [ex._target_executor])
register_recursion_strategy(RemoteExecutor, lambda _: [])
register_recursion_strategy(TransformingExecutor, lambda ex: [ex._target_executor])


for ty in [
      tff.python.core.impl.executors.federating_executor.FederatingExecutor,
      tff.python.core.impl.executors.federating_executor.FederatingExecutorValue,
      tff.python.core.impl.executors.reference_resolving_executor.ReferenceResolvingExecutor,
      tff.python.core.impl.executors.reference_resolving_executor.ReferenceResolvingExecutorValue,
      tff.python.core.impl.executors.execution_context.ExecutionContextValue,
      tff.python.core.impl.executors.eager_tf_executor.EagerValue,
  ]:
  register_formatting_strategy(ty, lambda x: "<{} @{} : {}>".format(type(x).__name__, id(x), str(x.type_signature)))

for ty in [
      tff.python.core.api.computation_types.TensorType,
      tff.python.core.api.computation_types.FederatedType,
      tff.python.core.api.computation_types.FunctionType,
  ]:
  register_formatting_strategy(ty, lambda x: "<{}>".format(x))

register_formatting_strategy(float, lambda x: "<{} : float>".format(x))
register_formatting_strategy(type(None), lambda _: "-")
register_formatting_strategy(list, lambda x: [format_object(xi) for xi in x])
register_formatting_strategy(tuple, lambda x: tuple(format_object(xi) for xi in x))

register_formatting_strategy(tf_EagerTensor,
    lambda x: "<{} @{}>".format(type(x).__name__, id(x)))

register_formatting_strategy(tff.python.core.impl.compiler.intrinsic_defs.IntrinsicDef,
    lambda x: "<IntrinsicDef {} {} @{}>".format(x.uri, x.type_signature, id(x)))

register_formatting_strategy(tff.python.core.impl.computation_impl.ComputationImpl,
    lambda x: "<ComputationImpl {} @{} : {}>".format(x._computation_proto.WhichOneof('computation'), id(x), x.type_signature))

register_formatting_strategy(tff.proto.v0.computation_pb2.Computation,
    lambda x: "<ComputationPb {} @{} : {}>".format(x.WhichOneof('computation'), id(x), x.type.WhichOneof('type')))

register_formatting_strategy(tff.python.common_libs.anonymous_tuple.AnonymousTuple,
    lambda x: "<AnonymousTuple @{} : {}>".format(id(x), format_anon_tuple(x)))

def format_anon_tuple(tup):
  names = tup._name_array
  values = tup._element_array
  fmt_strs = []
  for n, v in zip(names, values):
    fmt_strs.append("{}: {}".format(n, format_object(v)))

  return "({})".format(", ".join(fmt_strs))
