import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.core.impl.federated_executor import FederatedExecutor

import logging_helpers
import custom_executor_stacks

#
# Dummy Paillier protocol
#

class EncryptionKey:

  def __init__(self, value):
    self.value = value

  def into_raw_tensor(self):
    return tf.strings.as_string(self.value)

  @classmethod
  def from_raw_tensor(cls, raw_value):
    return cls(tf.strings.to_number(raw_value))

class DecryptionKey:

  def __init__(self, p, q):
    self.p = p
    self.q = q

  def into_raw_tensor(self):
    return tf.strings.as_string(self.p) + "," + tf.strings.as_string(self.q)

  @classmethod
  def from_raw_tensor(cls, raw_value):
    p_and_q = tf.strings.split(raw_value, sep=',')
    p_raw, q_raw = p_and_q[0], p_and_q[1]
    return cls(tf.strings.to_number(p_raw), tf.strings.to_number(q_raw))

class Keypair:

  def __init__(self, p, q):
    self.p = p
    self.q = q

  def into_raw_tensor(self):
    return tf.strings.as_string(self.p) + "," + tf.strings.as_string(self.q)

  @classmethod
  def from_raw_tensor(cls, raw_value):
    p_and_q = tf.strings.split(raw_value, sep=',')
    p_raw, q_raw = p_and_q[0], p_and_q[1]
    return cls(tf.strings.to_number(p_raw), tf.strings.to_number(q_raw))

  def encryption_key(self):
    return EncryptionKey(self.p * self.q)

  def decryption_key(self):
    return DecryptionKey(self.p, self.q)

class PlaintextTensor:

  def __init__(self, value):
    self.value = value

  def into_raw_tensor(self):
    return tf.strings.as_string(self.value)

  @classmethod
  def from_raw_tensor(cls, raw_value):
    return cls(tf.strings.to_number(raw_value))

class EncryptedTensor:

  def __init__(self, value):
    self.value = value

  def into_raw_tensor(self):
    return tf.strings.as_string(self.value)

  @classmethod
  def from_raw_tensor(cls, raw_value):
    return cls(tf.strings.to_number(raw_value))

def generate_keypair():
  return Keypair(tf.constant(17), tf.constant(19))

def encrypt(ek, x):
  assert type(ek) is EncryptionKey, type(ek)
  assert type(x) is PlaintextTensor, type(x)
  c = EncryptedTensor(x.value)
  return c

def add(ek, c0, c1):
  assert type(ek) is EncryptionKey, type(ek)
  assert type(c0) is EncryptedTensor, type(c0)
  assert type(c1) is EncryptedTensor, type(c1)
  c = EncryptedTensor(tf.add(c0.value, c1.value))
  return c

def decrypt(dk, c):
  assert type(dk) is DecryptionKey
  assert type(c) is EncryptedTensor
  x = PlaintextTensor(c.value)
  return x

#
# Methods needed for secure aggregation mechanism based on Paillier
#

# @tff.tf_computation
def setup():
  keypair = generate_keypair()
  encryption_key = keypair.encryption_key()
  decryption_key = keypair.decryption_key()
  raw_encryption_key = encryption_key.into_raw_tensor()
  raw_decryption_key = decryption_key.into_raw_tensor()
  return raw_encryption_key, raw_decryption_key

# @tff.tf_computation(tf.int32)
def encode(x):
  return tf.strings.as_string(x)

# @tff.tf_computation(tf.string)
def decode(x):
  return tf.strings.to_number(x)

# @tff.tf_computation(tf.string, tf.string)
def encrypt_input(raw_encryption_key, raw_x):
  encryption_key = EncryptionKey.from_raw_tensor(raw_encryption_key)
  x = PlaintextTensor.from_raw_tensor(raw_x)
  c = encrypt(encryption_key, x)
  raw_c = c.into_raw_tensor()
  return raw_c

# @tff.tf_computation(tf.string)
def encrypt_zero(raw_encryption_key):
  return encrypt_input(raw_encryption_key, "0")

# @tff.tf_computation((tf.string, tf.string), (tf.string, tf.string))
def aggregate(raw_encryption_key0_raw_c0, raw_encryption_key1_raw_c1):
  raw_encryption_key0, raw_c0 = raw_encryption_key0_raw_c0
  raw_encryption_key1, raw_c1 = raw_encryption_key1_raw_c1
  encryption_key = EncryptionKey.from_raw_tensor(raw_encryption_key0)
  c0 = EncryptedTensor.from_raw_tensor(raw_c0)
  c1 = EncryptedTensor.from_raw_tensor(raw_c1)
  c = add(encryption_key, c0, c1) #/ len(cs)
  raw_c = c.into_raw_tensor()
  return raw_encryption_key0, raw_c

# @tff.tf_computation
def decrypt_output(raw_decryption_key, raw_c):
  decryption_key = DecryptionKey.from_raw_tensor(raw_decryption_key)
  c = EncryptedTensor.from_raw_tensor(raw_c)
  y = decrypt(decryption_key, c)
  raw_y = y.into_raw_tensor()
  return raw_y

#
# Specialized SecureFederatedExecutor for the above mechanism 
#

class PaillierFederatedExecutor(FederatedExecutor):
  async def _compute_intrinsic_federated_sum(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.FederatedType)
    zero, plus = tuple(await
                       asyncio.gather(*[
                           executor_utils.embed_tf_scalar_constant(
                               self, arg.type_signature.member, 0),
                           executor_utils.embed_tf_binary_operator(
                               self, arg.type_signature.member, tf.add)
                       ]))
    return await self._compute_intrinsic_federated_reduce(
        FederatedExecutorValue(
            anonymous_tuple.AnonymousTuple([
                (None, arg.internal_representation),
                (None, zero.internal_representation),
                (None, plus.internal_representation)
            ]),
            computation_types.NamedTupleType(
                [arg.type_signature, zero.type_signature,
                 plus.type_signature])))


# @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
# def secure_federated_sum(xs):
#   encryption_key, decryption_key = setup()
#   decryption_key_on_server = tff.federated_value(decryption_key, tff.SERVER)
#   encryption_key_on_server = tff.federated_value(encryption_key, tff.SERVER)

#   encryption_key_on_clients = tff.federated_broadcast(encryption_key_on_server)
#   xs_encoded = tff.federated_map(encode, xs)
#   cs = tff.federated_map(encrypt_input, (encryption_key_on_clients, xs_encoded))

#   # NOTE this should happen on the aggregator
#   pairs = tff.federated_zip((encryption_key_on_clients, cs))
#   zero = (encryption_key, encrypt_zero(encryption_key)) # == ("", "0")
#   _, c = tff.federated_reduce(pairs, zero, aggregate)
  
#   # foo = c._comp.compute()
#   # print(type(c._comp))
#   # y = decrypt_output(decryption_key, c)

# #   print(xs_encrypted.type_signature)

# #   # encrypted_inputs = tff.federated_map(encrypt_input, (encryption_key, xs))
# #   # return tff.federated_mean(xs)
#   return c


def paillier_executor_stack(num_clients):

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
    return _complete_stack(PaillierFederatedExecutor(executor_dict))

  return executor_fn


#
# Execution
#

executor_fn = custom_executor_stacks.builtin_executor_stack(2)
executor_fn = paillier_executor_stack(3)

# logging_helpers.set_default_executor(executor_fn)
tff.framework.set_default_executor(executor_fn)

print(secure_federated_sum([1, 2, 4]))





# @tff.tf_computation(tf.float32)
# def add_half(x):
#   return x + 0.5

# @tff.tf_computation((tf.float32, tf.float32), (tf.float32, tf.float32))
# def op(x, y):
#   x0, x1 = x
#   y0, y1 = y
#   return tf.add_n([x0, x1]), tf.add_n([y0, y1])

# @tff.federated_computation
# def foo():
#     xs = tff.federated_broadcast(tff.federated_value(1.0, tff.SERVER))
#     ys = tff.federated_map(add_half, tff.federated_broadcast(tff.federated_value(1.0, tff.SERVER)))
#     pairs = tff.federated_zip((xs, ys))
#     print(xs.type_signature, ys.type_signature, pairs.type_signature)
#     zero = (0.0, 0.0)
#     return tff.federated_reduce(pairs, zero, op)

# print(foo())
