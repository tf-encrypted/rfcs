# Secure Runtime

| Status        | WIP                                                  |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Morten Dahl (mortendahlcs@gmail.com)                 |
| **Sponsor**   |                                                      |
| **Updated**   | YYYY-MM-DD                                           |

## Objective

## Motivation

## Design Proposal



## Detailed Design

### Example: Enclaves

In this example we assume that some external coordinator has already created all services needed,
including launching a new enclave. As such, the runtime is parameterized by a configuration
that includes all connection information needed by the executors.

```python
@tfe.computation
def secure_sum(x0, x1, x2):
  with x0.device:
    c0 = tfe.classify(x0, {x0.owner})

  with x1.device:
    c1 = tfe.classify(x1, {x1.owner})

  with x2.device:
    c2 = tfe.classify(x2, {x2.owner})

  with compute_device:
    d = tfe.add_n(c0, c1, c2)
    e = tfe.classify(d, {result_device.owner})

  with result_device:
    y = tfe.declassify(e)

  return y

comp = secure_sum.get_concrete_computation(...)
```

Setup information provided by orchestrator (based on out-of-band agreement between players) and available on all workers:

```python
PLAYER_IDENTITIES = {
    "inputter-0": "<long-term public key fingerprint>",
    "inputter-1": "<long-term public key fingerprint>",
    "inputter-2": "<long-term public key fingerprint>",
    "compute_device": "<long-term public key fingerprint>",
    "result_device": "<long-term public key fingerprint>",
}
```

```python
PLAYER_CLAIMS = {
    "inputter-0": [],
    "inputter-1": [],
    "inputter-2": [],
    "compute_device": [tfe.schemes.enclave.AttestationClaim(...)],
    "result_device": [],
}
```

```python
ENDPOINTS = {
    "inputter-0": None,
    "inputter-1": None,
    "inputter-2": None,
    "compute_device": "<long-term public key fingerprint>",
    "result_device": "<long-term public key fingerprint>",
}
```

```python
compute_device_spec = tfe.DeviceSpec(
    identity='<short-term public key fingerprint>',
    claims=[
        
    ]
)
```

On `inputter-0`:

```python



compute_device = tfe.schemes.enclave.RemoteDevice(...)
result_device = tfe.schemes.Device(...)


do_worker = RemoteWorker('grpc://...')

mo_worker = RemoteWorker('grpc://...')

tfe.runtime.RemoteDevice
```

On `inputter-1`:

```python
ENDPOINTS = {
  'inputter-0': ...,
  'inputter-1:' ...,
  'compute_device': ...,
  'result_device': ...,
}

compute_worker = enclave_service.new_enclave_worker()
do_worker = RemoteWorker('grpc://...')

mo_worker = RemoteWorker('grpc://...')

tfe.runtime.RemoteDevice
```

On `inputter-2`:

```python
ENDPOINTS = {
  'inputter-0': ...,
  'inputter-1:' ...,
  'compute_device': ...,
  'result_device': ...,
}

compute_worker = enclave_service.new_enclave_worker()
do_worker = RemoteWorker('grpc://...')

mo_worker = RemoteWorker('grpc://...')

tfe.runtime.RemoteDevice
```


## Questions and Discussion Topics


## Dump

```python
# connect to the workers (which are also root executors)
compute_worker = enclave_service.new_enclave_worker()
do_worker = RemoteWorker('grpc://...')
mo_worker = RemoteWorker('grpc://...')

with mo_worker:
  # get references to local values (that was passed in when worker was launched
  # on host); these will have secrecy {model_owner} but not be restricted to
  # any particular computation; they will not be garbage collected as long as
  # the worker is around
  m = tfe.keras.load_named('model')
  w = tfe.io.load_named('weights')

# run the training computation; this will create new executors on each worker
# that will only be around for this computation, and with all local values
# restricted to this particular computation; return value will 
v = training_comp(m, w, executors={
    'do_executor': do_worker,
    'mo_executor': mo_worker,
    'compute_executor': compute_worker,
})

with mo_worker:
  tfe.keras.save_named('weights', v)
```