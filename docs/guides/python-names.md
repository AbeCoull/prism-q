# Errors, Typing and Names

## Errors

Every failure, from backend limits to unsupported operations and invalid arguments,
raises `prism_q.PrismError` with the Rust error's message:

```python
import prism_q

huge = prism_q.circuits.ghz(40)
try:
    simulate(huge).backend(prism_q.BackendKind.density_matrix()).expectation([[(0, "Z")]])
except prism_q.PrismError as exc:
    print(exc)   # backend `density_matrix` is incompatible: circuit has ... qubits
```

`PrismError.kind` names the Rust variant, such as `"parse"`, `"backend_unsupported"` or
`"step_limit"`, so a caller can branch without matching the message. A run over a memory
cap raises kind `"resource_limit"` with the numbers attached: `resource` names the unit
(`"qubits"`, `"amplitudes"`, `"entries"`, `"elements"` or `"bytes of device memory"`),
`required` and `limit` are integers in that unit, and `env_var` names the variable that
overrides the cap, or is `None`. `knob` names the [`Limits`](python-running.md#memory-caps)
field that overrides it, and `limit_source` says which layer set it: `"detected"`,
`"env"`, `"process_default"` or `"request"`. On every other kind those six attributes are
`None`.

## Typing

Type stubs (`prism_q/_prism_q.pyi`) and a `py.typed` marker give mypy and Pyright the
full surface.

## Names

Python names follow the Rust ones in Python conventions: a Rust constructor such as
`CircuitBuilder::new_with_classical` is the class call, a builder consumed by value is
a method chain on a mutable object, and a result field is a property. Two names differ:
`simulate()` returns a `Simulation` where Rust has the `Simulate` typestate builder (and
`simulate_program()` a `ProgramSimulation`), and the `circuits` generators drop the
`_circuit` suffix. `bindings/python/parity.toml` records, for every public Rust
function, method and result field, the Python name that reaches it or why it stays in
Rust only; a test keeps the record complete as the Rust API grows.
[Python API](../reference/python-api.md) lists the free functions.

Three classes have taken their Rust names: `Decoder` is now `UnionFindDecoder`,
`QecResult` is `QecSampleResult`, and `RecordRef` is `QecRecordRef`. The old names
still resolve, with a `DeprecationWarning`, until the next minor release, and pickles
written under the old names load.
