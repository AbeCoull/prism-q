# Dynamic Circuits

Goal: teleport a qubit using mid-circuit measurement and feed-forward, then reuse a
measured qubit with `reset` and branch on an outcome with `if`/`else`, all written in
OpenQASM 3.

## Teleportation

Qubit 0 holds `ry(1.2)|0>`. Qubits 1 and 2 share a Bell pair. Measuring qubits 0 and 1
mid-circuit and applying `x` and `z` to qubit 2 on those outcomes moves the state onto
qubit 2, so measuring it gives 1 with probability `sin(0.6)**2`.

```python
import math
from prism_q import parse_qasm, simulate

theta = 1.2
teleport = parse_qasm(f"""
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
ry({theta}) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1]) x q[2];
if (c[0]) z q[2];
c[2] = measure q[2];
""")

shots = simulate(teleport).seed(42).shots(100_000).shots
print(round(shots[:, 2].mean(), 4))           # 0.3208
print(round(math.sin(theta / 2) ** 2, 4))     # 0.3188
```

The sampled rate is 0.002 from the exact one, where one standard error at 100,000 shots
is 0.0015.
`shots` is a `(shots, bits)` bool array, so column 2 is classical bit `c[2]`.

A circuit with feed-forward has no single final state, so the per-shot terminals,
`shots()` and `sample_counts()`, are the ones to use. `run()` follows one sampled branch:
`classical_bits` holds that shot's outcomes and `probabilities` the state it ended in.
`state_vector()` declines any circuit that measures, resets or branches.

## Reset and if/else

`reset` returns a measured qubit to `|0>` so it can be used again. An `if` body can hold
any supported statement, and `else` takes the other branch:

```python
branch = parse_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[3] c;
h q[0];
c[0] = measure q[0];
reset q[0];
if (c[0]) {
  x q[1];
} else {
  h q[1];
}
c[1] = measure q[1];
c[2] = measure q[0];
""")

counts = simulate(branch).seed(42).sample_counts(1000).counts()
print(sorted(counts.items()))   # [('000', 270), ('010', 245), ('110', 485)]
```

Read the keys with `c[0]` on the left. Half the shots measured 1 first and took the `x`
branch, so they read `110`. The other half took `h`, which leaves `c[1]` a fair coin:
`000` or `010`. `c[2]` is always 0 because of the reset.

Conditions can also test a parity, `if (c[0] ^ c[1])`, and `switch` with `case` arms
lowers to the same guards. `while` is not supported, since a loop that exits on a
measurement has no fixed instruction list. The
[OpenQASM guide](../guides/openqasm.md#the-subset) has the full subset.

## In Rust

The same OpenQASM source parses with `openqasm::parse`. `CircuitBuilder` also has
`reset`, `conditional`, and `guarded` for building these circuits directly:

```rust
use prism_q::circuit::openqasm;
use prism_q::{bitstring, simulate};

let branch = openqasm::parse(
    r#"
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[3] c;
    h q[0];
    c[0] = measure q[0];
    reset q[0];
    if (c[0]) { x q[1]; } else { h q[1]; }
    c[1] = measure q[1];
    c[2] = measure q[0];
    "#,
)?;
let counts = simulate(&branch).seed(42).sample_counts(1000)?;
let mut keys: Vec<String> = counts
    .counts
    .keys()
    .map(|key| bitstring(key, counts.num_classical_bits))
    .collect();
keys.sort();
assert_eq!(keys, ["000", "010", "110"]);
# Ok::<(), prism_q::PrismError>(())
```

[`examples/dynamic_circuits.rs`](https://github.com/AbeCoull/prism-q/blob/main/examples/dynamic_circuits.rs)
runs both circuits.

Next: [Large and Structured Circuits](./large-circuits.md).
