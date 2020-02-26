# Python eDSL

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com)                 |
| **Sponsor**   |                                                      |
| **Updated**   | YYYY-MM-DD                                           |

## Objective

## Motivation

## Design Proposal

### Functional Paradigm

Both the Python eDSL and the target computation graphs follow a functional paradigm.

- immutable data is easier to (statically) analyse and values can safely be reused for optimization

### Tracing

- abstract computations (tfe.computation) is polymorphic, like tf.function

### Eager evaluation

### ...

- Functions are assigned to a single (potentially composite) device; makes 'with' context handler simpler since it makes sense for both execution and destination
- Python loops are unrolled during tracing; TFE functional constructs can be expressed by computation graph
- TF Keras models are automatically converted to TFE Keras models that fit with functional paradigm
- high-level functionality (such as ReLU/Dense/Keras) should be overloadable as well, https://github.com/tf-encrypted/tf-encrypted/issues/671; focusing on the Keras Functional API might be a natural solution to this

## Detailed Design

### Examples

#### Secure Aggregation

```python
# TFE functions are similar to TF functions in that they
# capture a (potentially differentiable) unit of work
@tfe.function
def mean(xs):

  # We may use ordinary Python functions, ideally even TF
  # operations and the related autograd functionality, yet
  # everything is eventually mapped to TFE operations
  return tfe.add_n(xs) / len(xs)
```

```python
# TFE computations captures how functions should be used
# in a multi-player scenario, supplying hints as to what
# must be encrypted and what can be used in plaintext
@tfe.computation
def secure_mean(xs):

  # specify classification of inputs; Python comprehension
  # is statically unrolled; the device for each operation
  # is inferred to be the same as the input
  cs = [tfe.classify(x, {x.owner}) for x in xs]

  # compute the mean on the designated device
  with compute_device:
    d = mean(cs)

  # allow result to be revealed below without causing a
  # violation; again the device is inferred to input
  e = tfe.classify(d, {result_device.owner})

  # reveal the result to designated device
  with result_device:
    y = tfe.declassify(e)

  return y
```

The above could also be expressed more directly as follows, which close mirrors how the computation is represented as a graph:

```python
@tfe.function
def mean(xs):
  return tfe.div(tfe.add_n(xs), len(xs))
```

```python
@tfe.computation
def secure_mean(xs):
  cs = [tfe.classify(x, {x.owner}, device=x.device)]
  d = tfe.call(mean, cs, device=compute_device)
  e = tfe.classify(d, {result_device.owner}, device=d.device)
  y = tfe.declassify(e, device=result_device)
  return y
```

#### Prediction

```python
class MyModel(tfe.Model):

  @tfe.function
  def forward(self, w, x):
    return tfe.matmul(w, x)

  def predict(self, x):
    return self.forward(self.w, x)
```

```python
@tfe.computation
def secure_prediction(m: tfe.Model, x: tfe.Tensor) -> tfe.Tensor:
  # use of device context is optional here and can be inferred
  with x.device:
    # indicate that we want the inputs to be protected; concrete
    # meaning of this is determined by schemes during compilation
    xe = tfe.classify(x, {x.owner})

  with compute_device:
    # run prediction
    ye = m.predict(xe)
    # indicate that the result may be seen by the owner of the
    # input; this is required to correctly decrypt below
    ze = tfe.classify(ye, {x.owner})
  
  # use of device context is required here since the inferred
  # device would be `compute_device`, which would violate the
  # result's classification
  with x.device:
    z = tfe.declassify(ze)

  # return reference to result
  return z
```

```python
function = MyModel.forward.get_concrete_function(...)

input_sig = [tfe.ModelSpec]
computation = secure_prediction.get_concrete_computation()
```

#### Training

```python
class MyModel(tfe.learning.Model):

  @tfe.function
  def forward(self, w, x):
    return tfe.matmul(w, x)

  def predict(self, w: tfe.Tensor, x: tfe.Tensor) -> tfe.Tensor:
    return self.forward(w, x)

  def fit(self, w: tfe.Tensor, xy: tfe.Sequence) -> tfe.Tensor:
    return tfe.foldl(self.fit_on_batch, w, xy)

  @tfe.function
  def fit_on_batch(self, w: tfe.Tensor, xy: tfe.Sequence) -> tfe.Tensor:
    raise NotImplementedError
```

```python
@tfe.function
def fit_batch(cur_model, xy):
  # generate new model from current model and xy
  return new_model

new_model = tfe.foldl(fit_batch, cur_model, training_data)
```

### Tracing

Tracing is used to turn `tfe.computation` (and `tfe.function`) decorated functions into computation graphs. However, in order to do so some information beyond input signature must be known about devices and their owners, including device type and owner identifier.

#### Secure aggregation

```python
native = tfe.schemes.Native()

# prepare input signature
num_inputs = 5
input_shape = (2, 3)
input_owners = [tfe.PlayerSpec('Client-{}' % i for i in range(num_inputs))]
input_devices = [native.create_device_spec(owner) for owner in input_owners]
input_sig = [tfe.TensorSpec(input_shape, device) for device in input_devices]

# prepare additional devices
result_owner = tfe.PlayerSpec('Outputter')
result_device = native.create_device_spec(result_owner)
```

With Paillier:

```python
paillier = tfe.schemes.Paillier()

compute_device = paillier.create_device_spec()
paillier_secure_mean = secure_mean.get_concrete_computation(input_sig)
```

With SDA:

```python
sda = tfe.schemes.SDA(num_clerks=2)

compute_device = sda.create_device_spec()
sda_secure_mean = secure_mean.get_concrete_computation(input_sig)
```

In principal we could also trace with an arbitrary device, however no compiler
chain nor executor may exist for the resulting graph:

```python
compute_device = tfe.DeviceSpec(device_type='Dummy')
dummy_secure_mean = secure_mean.get_concrete_computation(input_sig)
```


## Objective

- Front load what everyone involved in a computation is expected to do; this can for instance be used to assure a data owner that their dataset will only be used for encrypted model training (and e.g. not extracted).



What are we doing and why? What problem will this solve? What are the goals and
non-goals? This is your executive summary; keep it short, elaborate below.

## Motivation



Why this is a valuable problem to solve? What background information is needed
to show how this design addresses the problem?

Which users are affected by the problem? Why is it a problem? What data supports
this? What related work exists?

## Design Proposal

This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.

Factors to consider include:

* performance implications
* dependencies
* maintenance
* platforms and environments impacted (e.g. hardware, cloud, other software
  ecosystems)
* how will this change impact users, and how will that be managed?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
