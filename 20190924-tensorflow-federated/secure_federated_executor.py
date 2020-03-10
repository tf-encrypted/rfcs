import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.executors.executor_base import Executor
from tensorflow_federated.python.core.impl.executors.executor_value_base import ExecutorValue

import logging_helpers

class SecureFederatedExecutorValue(ExecutorValue):

  def __init__(self, value):
    self._value = value
    # self._type_signature = computation_types.to_type(type_spec)

  async def compute(self):
    print("compute")
    return self._value.compute()

  @property
  def type_signature(self):
    return self._value.type_signature


class SecureFederatedExecutor(Executor):

  def __init__(self, target_executor):
    self._target_executor = target_executor

  async def create_value(self, value, type_spec=None):
    if isinstance(value, intrinsic_defs.IntrinsicDef):
      print("*** {}".format(value))

    if isinstance(value, tff.proto.v0.computation_pb2.Computation):
      if (value.WhichOneof('computation') == 'intrinsic'):
        if (value.intrinsic.uri == 'federated_mean'):
          print("*** {}".format(value))

    res = await self._target_executor.create_value(value, type_spec=type_spec)
    return SecureFederatedExecutorValue(res)

  async def create_call(self, comp, arg=None):
    assert isinstance(comp, SecureFederatedExecutorValue)
    assert arg is None or isinstance(arg, SecureFederatedExecutorValue)
    _comp = comp._value
    _arg = arg._value if arg is not None else None
    res = await self._target_executor.create_call(_comp, _arg)
    return SecureFederatedExecutorValue(res)

  async def create_tuple(self, elements):
    print("TYPE TYPW TPYE", type(elements))
    print("ELEMENTS EKECSCDS", elements)
    _elements = elements._value if isinstance(elements, SecureFederatedExecutorValue) else elements
    res = await self._target_executor.create_tuple(_elements)
    return SecureFederatedExecutorValue(res)

  async def create_selection(self, source, index=None, name=None):
    return await self._target_executor.create_selection(source, index=index, name=name)

  def close(self):
    self._target_executor.close()


logging_helpers.register_recursion_strategy(SecureFederatedExecutor,
    lambda ex: [ex._target_executor])

logging_helpers.register_formatting_strategy(SecureFederatedExecutorValue,
    lambda x: "<{} @{} : {}>".format(type(x).__name__, id(x), str(x.type_signature)))
