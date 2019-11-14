import tensorflow_federated as tff

from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core.impl.eager_executor import EagerExecutor
from tensorflow_federated.python.core.impl.federated_executor import FederatedExecutor
from tensorflow_federated.python.core.impl.lambda_executor import LambdaExecutor

from secure_federated_executor import SecureFederatedExecutor


def builtin_executor_stack(num_clients):

  def executor_fn(mapping):
    del mapping

    def _complete_stack(ex):
      return \
          LambdaExecutor(
          # CachingExecutor(
          # ConcurrentExecutor(
          ex
      )

    def _create_bottom_stack():
      return _complete_stack(EagerExecutor())

    executor_dict = {
        tff.CLIENTS: [_create_bottom_stack() for _ in range(num_clients)],
        tff.SERVER: _create_bottom_stack(),
        None: _create_bottom_stack(),
    }
    return _complete_stack(FederatedExecutor(executor_dict))

  return executor_fn


def secure_executor_stack(num_clients):

  def executor_fn(mapping):
    del mapping

    def _complete_stack(ex):
      return \
          LambdaExecutor(
          # CachingExecutor(
          # ConcurrentExecutor(
          ex
      )

    def _create_bottom_stack():
      return _complete_stack(EagerExecutor())

    executor_dict = {
        tff.CLIENTS: [_create_bottom_stack() for _ in range(num_clients)],
        tff.SERVER: _create_bottom_stack(),
        None: _create_bottom_stack(),
    }
    return _complete_stack(SecureFederatedExecutor(FederatedExecutor(executor_dict)))

  return executor_fn


def framework_executor_stack(num_clients):
  return framework.create_local_executor(num_clients=num_clients)
 