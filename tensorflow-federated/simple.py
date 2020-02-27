import tensorflow as tf
import tensorflow_federated as tff

import logging_helpers
import custom_executor_stacks


@tff.tf_computation(tf.float32)
def add_half(x):
  return tf.add(x, 0.5)

@tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
def add_half_on_clients(x):
  return tff.federated_map(add_half, x)

@tff.federated_computation(tff.FederatedType(tf.float32, tff.SERVER))
def send_to_clients(x):
  return tff.federated_broadcast(x)

@tff.federated_computation(tff.FederatedType(tf.float32, tff.SERVER))
def foo(x):
  x_on_clients = tff.federated_broadcast(x)
  y_on_clients = tff.federated_map(add_half, x_on_clients)
  y_mean = tff.federated_mean(y_on_clients)
  return y_mean

# executor_fn = custom_executor_stacks.secure_executor_stack(3)
# executor_fn = tff.framework.create_local_executor(3)
executor_fn = custom_executor_stacks.builtin_executor_stack(3)
logging_helpers.set_default_executor(executor_fn)

# print(foo(5.0))

res = add_half_on_clients([1.5, 2.5, 3.5])
print(res)
