import tensorflow_federated as tff

from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core.impl.executors.eager_tf_executor import EagerTFExecutor
from tensorflow_federated.python.core.impl.executors.federating_executor import FederatingExecutor
from tensorflow_federated.python.core.impl.executors.reference_resolving_executor import ReferenceResolvingExecutor

from secure_federated_executor import SecureFederatedExecutor


def builtin_executor_stack(num_clients):

  def executor_fn(mapping):
    del mapping

    def _complete_stack(ex):
      return \
          ReferenceResolvingExecutor(
          # CachingExecutor(
          # ConcurrentExecutor(
          ex
      )

    def _create_bottom_stack():
      return _complete_stack(EagerTFExecutor())

    executor_dict = {
        tff.CLIENTS: [_create_bottom_stack() for _ in range(num_clients)],
        tff.SERVER: _create_bottom_stack(),
        None: _create_bottom_stack(),
    }
    return _complete_stack(FederatingExecutor(executor_dict))

  return executor_fn


def secure_executor_stack(num_clients):

  def executor_fn(mapping):
    del mapping

    def _complete_stack(ex):
      return \
          ReferenceResolvingExecutor(
          # CachingExecutor(
          # ConcurrentExecutor(
          ex
      )

    def _create_bottom_stack():
      return _complete_stack(EagerTFExecutor())

    executor_dict = {
        tff.CLIENTS: [_create_bottom_stack() for _ in range(num_clients)],
        tff.SERVER: _create_bottom_stack(),
        None: _create_bottom_stack(),
    }
    return _complete_stack(SecureFederatedExecutor(FederatingExecutor(executor_dict)))

  return executor_fn


def framework_executor_stack(num_clients):
  return framework.local_executor_factory(num_clients=num_clients)
 