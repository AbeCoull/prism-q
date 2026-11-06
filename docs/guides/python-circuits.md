# Circuits and Parameters

## Building circuits

Every `CircuitBuilder` gate method returns the builder; `build()` returns the `Circuit`.

```python
from prism_q import CircuitBuilder

circuit = (
    CircuitBuilder(3, 3)
    .h(0)
    .cx(0, 1)
    .rz(0.5, 2)
    .cphase(0.25, 1, 2)
    .measure_all()
    .build()
)
```

| Group | Methods |
|-------|---------|
| Single qubit | `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg` |
| Rotations | `rx(theta, q)`, `ry(theta, q)`, `rz(theta, q)`, `p(theta, q)`, `u(theta, phi, lam, q)` |
| Two qubit | `cx(control, target)`, `cy(control, target)`, `cz(q0, q1)`, `ch(control, target)`, `swap(q0, q1)`, `iswap(q0, q1)`, `cphase(theta, control, target)` |
| Controlled rotations | `crx(theta, control, target)`, `cry(theta, control, target)`, `crz(theta, control, target)` |
| Two-qubit rotations | `rxx(theta, q0, q1)`, `ryy(theta, q0, q1)`, `rzz(theta, q0, q1)` |
| Three qubit | `ccx(control0, control1, target)`, `cswap(control, q0, q1)` |
| Multi-qubit rotation | `pauli_rotation(theta, factors)` |
| Hamiltonian evolution | `trotter(hamiltonian, t, steps=1, order=2)` |
| Arbitrary unitary | `cu(matrix, control, target)`, `mcu(matrix, controls, target)`, `gate(gate, targets)` |
| Non-unitary | `measure(qubit, bit)`, `measure_all()`, `measure_in_basis(qubit, axis, bit)`, `measure_pauli_product(factors, bit)`, `reset(qubit)`, `barrier(qubits)` |
| Classical control | `conditional(condition, gate, targets)`, `guarded(condition, body, else_body=None)` |
| Parameters | `param(slot)`, `parameters()`, `parameter_links()`, `build_parametric()` |

`pauli_rotation(theta, factors)` appends `exp(-i * theta * P / 2)` for the Pauli
string `P` given as `(qubit, axis)` factors with `axis` one of `"X"`, `"Y"`, `"Z"`;
omit identity factors. A weight-1 string lowers to `rx`, `ry`, or `rz` and a two-qubit
`ZZ` string to `rzz`, so fusion and Clifford recognition still see them.
`Circuit.add_pauli_rotation` is the imperative spelling.

```python
builder = CircuitBuilder(4)
builder.pauli_rotation(0.4, [(0, "X"), (1, "Y"), (3, "Z")]).param(0)
```

`trotter(hamiltonian, t, steps=1, order=2)` approximates `exp(-i H t)` with `steps`
steps of length `t / steps`, each term `c P` appended as the `pauli_rotation`
`exp(-i c P dt)`. `order` is 1 for Lie-Trotter, 2 for the symmetric formula, or an even
`2k` for the Suzuki formula of that order, costing `5 ** (k - 1)` second-order substeps.
The Hamiltonian is a term list, as the expectation terminals take, or a
`PauliObservable`; identity terms drop out as a global phase. `Circuit.add_trotter` is
the imperative spelling.

```python
from prism_q import CircuitBuilder, PauliObservable, simulate

ising = PauliObservable([(1.0, [(0, "Z"), (1, "Z")]), (0.5, [(0, "X")]), (0.5, [(1, "X")])])
evolved = CircuitBuilder(2).h(0).h(1).trotter(ising, 1.0, steps=8, order=2).build()
energy = simulate(evolved).seed(42).expectation(ising).mean
assert abs(energy - 1.0) < 1e-2  # <H> is conserved up to the Trotter error
```

Standard gates lower exactly as the OpenQASM parser lowers them, so a builder circuit
matches the parsed one instruction for instruction: `crx`, `cy` and the other
controlled gates become one controlled-unitary, `rxx` and `ryy` become Pauli
rotations that `.param(slot)` can bind, `u` becomes one fused matrix, and `iswap`
and `cswap` expand to Clifford gates and a Toffoli.

`cu` and `mcu` take a 2x2 matrix as nested sequences of complex numbers. Out-of-range
or repeated qubits raise `PrismError` at build time.

Three other routes produce a `Circuit`:

```python
from prism_q import Circuit, circuits, parse_qasm

qasm_source = "OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];"
manual = Circuit(2, 2)                    # imperative, add_gate / add_measure / add_reset
ghz = circuits.ghz(10)                    # prebuilt corpus
parsed = parse_qasm(qasm_source)          # OpenQASM 3.0, with 2.0 accepted
```

The `circuits` submodule mirrors the Rust builders documented in
[Circuit Builders](../reference/builders.md): `qft`, `ghz`, `w_state`, `random`,
`hardware_efficient_ansatz`, `clifford_heavy`, `clifford_random_pairs`, `qaoa`,
`single_qubit_rotation`, `clifford_t`, `quantum_volume`, `cz_chain`,
`phase_estimation`, `independent_bell_pairs`, `independent_random_blocks`, and
`local_clifford_blocks`. Seeded builders default to seed 42.

A `Circuit` reports its own shape: `num_qubits`, `num_classical_bits`, `gate_count()`,
`t_count()` (T and T-dagger gates), `depth()` (layers with each operation as early as its
qubits allow, a measurement one layer and a barrier none), `is_clifford_only()`, and
`measurement_map()`, the `(qubit, classical_bit)` pair of each measurement in order.

```python
assert ghz.gate_count() == 10 and ghz.t_count() == 0 and ghz.is_clifford_only()
```

### Measuring in other bases

`measure_in_basis(qubit, axis, bit)` rotates `axis` onto Z and measures, recording
the `+1` eigenvalue as `0`. The qubit is left in the Z eigenstate, not the `axis` one.
`measure_pauli_product(factors, bit)` measures a product such as `X0 X1` without
measuring its factors. The first call appends one ancilla past the register to collect
the parity, so the circuit is one qubit wider; a second call resets that ancilla, which
takes the circuit off the compiled sampling route.

```python
from prism_q import CircuitBuilder, simulate

bell = CircuitBuilder(2, 2).h(0).cx(0, 1)
bell.measure_pauli_product([(0, "X"), (1, "X")], 0)
bell.measure_pauli_product([(0, "Z"), (1, "Z")], 1)
checked = bell.build()
print(checked.num_qubits)                              # 3
print(simulate(checked).seed(1).shots(100).counts())   # {'00': 100}
```

### Classical control

A `ClassicalCondition` tests measured bits at runtime. `ClassicalCondition.bit(b,
value=True)` reads one bit, `parity(bits, expected=True)` the XOR of several, and
`register_equals(offset, size, value)` and `register_not_equals` read `size` bits as an
integer with bit `offset` least significant. `~condition` is the negation.

`conditional(condition, gate, targets)` applies one gate when the condition holds:

```python
from prism_q import ClassicalCondition, Gate

teleport = CircuitBuilder(3, 3).ry(0.8, 0).h(1).cx(1, 2).cx(0, 1).h(0)
teleport.measure(0, 0).measure(1, 1)
teleport.conditional(ClassicalCondition.bit(1), Gate.x(), [2])
teleport.conditional(ClassicalCondition.bit(0), Gate.z(), [2])
teleport.ry(-0.8, 2).measure(2, 2)
received = simulate(teleport.build()).seed(5).shots(500)
assert not received.shots[:, 2].any()
```

`guarded(condition, body, else_body=None)` calls `body` with a fresh `CircuitBuilder` of
the same width and turns what it appends, measurements and resets included, into a
region that runs when the condition holds. `else_body` builds the region for the other
shots:

```python
coin = CircuitBuilder(2, 2).h(0).measure(0, 0)
coin.guarded(
    ClassicalCondition.bit(0),
    lambda then: then.x(1),
    lambda otherwise: otherwise.z(1),
)
coin.measure(1, 1)
print(simulate(coin.build()).seed(3).shots(1000).counts())   # {'00': 507, '11': 493}
```

The else region rereads the condition after the first region runs, so a `body` that
measures into a bit the condition reads cannot take an `else_body` (`PrismError`). A
body cannot widen the circuit, which rules out `measure_pauli_product` and
`measure_all`. Regions nest. A noise model for a circuit with regions carries slots for
the body gates, and their events fire only on shots that take the branch.
`NoiseModel.empty(circuit)` sizes the model to those slots too: index `i` below the
instruction count is top-level instruction `i`, and the body slots follow in region
order.

## Exporting and drawing circuits

`to_qasm()` writes OpenQASM 3.0 that `parse_qasm` reads back to the same instructions,
inline angles exact:

```python
from prism_q import CircuitBuilder, parse_qasm

circuit = (
    CircuitBuilder(3, 3)
    .h(0)
    .cx(0, 1)
    .pauli_rotation(0.3, [(0, "X"), (1, "Y"), (2, "Z")])
    .measure_all()
    .build()
)
text = circuit.to_qasm()
assert parse_qasm(text).to_qasm() == text
portable = circuit.to_qasm(expand_pauli_rotations=True)
```

A multi-letter Pauli rotation keeps its `rxyz(0.3)` spelling, which only PRISM-Q
parses. `expand_pauli_rotations=True` lowers it to basis changes around a CNOT ladder that
other toolchains read. Guarded regions export as `if` blocks. Save points and
dense unitaries on three or more qubits have no OpenQASM spelling and raise `PrismError`
with `kind == "export_unsupported"`.

`print(circuit)` draws the text wire diagram. `draw()` takes `fold_width`,
`show_idle_wires`, `show_barriers`, `max_qubits`, and `max_moments`; past 64 qubits or
500 moments it returns `summary()` instead (gate counts, connectivity, depth profile).
`heatmap()` draws gate density by qubit and moment.

```python
print(circuit)
print(circuit.draw(fold_width=80, show_idle_wires=False))
with open("circuit.svg", "w", encoding="utf-8") as f:
    f.write(circuit.to_svg(dark_mode=True, show_legend=True))
```

`to_svg()` writes a standalone SVG. Its keyword options cover the theme
(`dark_mode`, or `auto_theme` to embed both and follow the viewer's color scheme),
`animate`, `compact`, the `show_legend`, `show_stats_header`, and `show_topology`
layers, truncation (`max_qubits`, `max_moments`, and `ellipsis=(first, last)` to keep
only the leading and trailing moments), and geometry in SVG units (`wire_spacing`,
`moment_width`, `gate_height`, `gate_min_width`, `font_size`, `control_radius`, and
`padding` as `(left, top, right, bottom)`). `to_svg_heatmap()` renders the density
heatmap. In Jupyter a circuit displays as a static diagram in the page theme, cut to 64
wires and 200 moments.

## Parameter sweeps

`Parameters` names the angle slots a variational loop rebinds. `PreparedCircuit` holds
the circuit across bindings, so fusion and backend selection happen once rather than
per point. The [variational tutorial](../tutorials/variational.md) runs an optimization
loop on top of this.

```python
from prism_q import CircuitBuilder, PreparedCircuit

builder = CircuitBuilder(4)
for q in range(4):
    builder.ry(0.1, q).param(q)
for q in range(3):
    builder.cx(q, q + 1)

prepared = PreparedCircuit(builder.build(), builder.parameters())
for values in [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]:
    outcome = prepared.run(values, seed=42)
```

`run(values, seed=42)` returns the `RunOutcome` that `simulate(...).run()` would.
`expectation(values, observables)` reads `Simulation.expectation()` on the bound circuit
without dispatching or fusing again: a list of Pauli strings returns a `float64` ndarray,
one value per string, and a `PauliObservable` returns the `ObservableExpectation`.
`bind(values)` returns the bound `Circuit`, for `simulate(...)` with other options or any
other consumer.

`run_many` and `expectation_many` take a `(points, num_slots)` array and cross into Rust
once. `expectation_many` returns a `(points, observables)` array for a list of strings
and one `ObservableExpectation` per point for a `PauliObservable`. Up to 16 qubits the
rows split across cores, as in `run_batch`; each result matches the single call with the
same seed.

```python
from prism_q import PauliObservable

points = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
hamiltonian = [(1.0, [(0, "Z")]), (0.5, [(0, "Z"), (1, "Z")])]
energies = prepared.expectation_many(points, PauliObservable(hamiltonian), seed=42)
```

A term list is parsed, and its terms regrouped into qubit-wise commuting sets, on every
call. `PauliObservable` parses once and keeps the grouping after first use. Every
Hamiltonian argument accepts one, the gradients included, so build it outside the loop:

```python
h = PauliObservable(hamiltonian)
for values in points:
    mean = prepared.expectation(values, h, seed=42).mean
```

`expectation_gradient(values, h)` and `expectation_gradient_shift(values, h)` return the
`ExpectationGradient` that `Simulation.expectation_gradient()` and its shift twin give on
the bound circuit with the held `Parameters`. The `_many` forms return a `(points,)`
array of values and a `(points, num_slots)` array of gradients; rows split across cores up
to 14 qubits.

```python
values, gradients = prepared.expectation_gradient_many(points, h, seed=42)
```

Automatic dispatch reads the template, so build it at angles typical of the sweep. A
template with all rotations at zero reads as Clifford and picks a backend that then
rejects the bound circuit. An explicit backend as the third argument avoids this:

```python
from prism_q import BackendKind

circuit, params = builder.build(), builder.parameters()
prepared = PreparedCircuit(circuit, params, BackendKind.statevector())
```

`reuses_fusion_plan` reports whether the recorded fused structure is replayed. Results
agree either way; only speed differs.

`builder.parameters()` returns the slots recorded by `param(slot)`;
`Parameters.all_rotations(circuit)` gives every bindable gate its own slot in circuit
order; `Parameters(n)` plus `link(instruction, slot)` declares slots up front. Gates
sharing a slot all receive its angle.

`parse_qasm_parametric` returns a template and the named slots declared by OpenQASM
`input` statements, in declaration order:

```python
from prism_q import BackendKind, PreparedCircuit, parse_qasm_parametric

template, parameters = parse_qasm_parametric("""
OPENQASM 3.0;
input float[64] theta;
qubit[2] q;
ry(theta) q[0];
cx q[0], q[1];
ry(theta) q[1];
""")
assert parameters.slot_of("theta") == 0
qasm_sweep = PreparedCircuit(template, parameters, BackendKind.statevector())
outcomes = qasm_sweep.run_many([[0.2], [0.7]], seed=42)
```

Input angles start at zero, so bind the template before simulating and give
`PreparedCircuit` an explicit backend. Each input must be a whole angle argument of a
supported gate at top level; expressions such as `2 * theta` and inputs inside control
flow raise `PrismError`. `parse_qasm` still rejects unbound inputs.

| Method | Returns |
|--------|---------|
| `bind(template, values)` | `template` with the linked angles overwritten |
| `values(circuit)` | the angle each slot currently holds |
| `validate(circuit)` | nothing; raises if a link no longer points at a bindable gate |
| `with_names(names)` | a copy naming the slots, matching OpenQASM `input` declarations |
| `name_of(slot)`, `slot_of(name)` | the name and slot of a named set, else `None` |
| `unread_slots()` | declared slots no instruction reads, whose values are discarded |

A wrong-length value vector, a non-finite angle, and a link pointing at a gate
that carries no angle all raise `PrismError`.

## Dynamic programs

A `Circuit` is a finite instruction list and cannot hold a loop that repeats until a
measurement comes up. `parse_qasm_dynamic` reads OpenQASM with `while`, `break`,
`continue`, and runtime classical values, and returns a `DynamicProgram`: circuits
joined by branches on classical expressions, run once per shot. `simulate_program`
returns the same `ShotsResult` and `CountsResult` as a circuit run. The
[dynamic circuits tutorial](../tutorials/dynamic-circuits.md) ends with a subroutine
that returns a value computed from a measured bit.

```python
from prism_q import parse_qasm_dynamic, simulate_program

program = parse_qasm_dynamic("""
    OPENQASM 3.0;
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    while (c[0]) {
        reset q[0];
        h q[0];
        c[0] = measure q[0];
    }
""")
counts = simulate_program(program).seed(42).sample_counts(1000).counts()
assert counts == {"0": 1000}
```

`DynamicProgramBuilder` builds the same structure directly. Conditions, assigned
values, and rotation angles are OpenQASM expression strings over declared variables and
classical bits, `c[i]` for one bit and `c` for the
whole register read as an unsigned integer, or plain `bool`, `int`, and `float`
constants. `begin_while` and `begin_if` open a block that `end()` closes, and
`begin_else`, `break_loop`, and `continue_loop` work as their OpenQASM keywords
do. Variable types are `"bool"`, `"int[n]"`, `"uint[n]"`, `"float[n]"`, and
`"angle"`; integers wrap at their declared width, and `float[16]` and `float[32]` round
to half and single precision.

```python
from prism_q import DynamicProgramBuilder, Gate, simulate_program

b = DynamicProgramBuilder(2, 2)
b.declare("theta", "float", 0.0)
b.add_gate(Gate.h(), [0]).add_measure(0, 0)
b.begin_while("retry", "c[0] && theta < 1.0")
b.assign("theta", "theta + 0.25")
b.add_reset(0).add_gate(Gate.h(), [0]).add_measure(0, 0)
b.end()
b.add_rotation("ry", [1], "theta").add_measure(1, 1)
program = b.build()
shots = simulate_program(program).seed(7).shots(500)
```

A shot that runs more blocks than its bound, one million by default, raises
`PrismError` with `kind == "step_limit"` and the loop's name. `.max_steps(n)` changes
the bound:

```python
b = DynamicProgramBuilder(1, 0)
b.begin_while("forever", True).add_gate(Gate.x(), [0]).end()
simulate_program(b.build()).max_steps(100).shots(1)  # raises
```

`.noise(rules)` takes a `NoiseBuilder`, not a `NoiseModel`: a model is laid out against
one circuit and a program holds one per block. The rules compile against every block and
runtime rotation, and each shot runs one noisy trajectory. Readout error flips the
returned record, not the bits a branch reads. Leak flags and quasi-static draws last the
whole shot, so a qubit leaked in one block stays leaked in the next. The density matrix
and an over-rotation rule matching a runtime rotation raise `incompatible_backend`.

```python
from prism_q import GateFilter, NoiseBuilder, NoiseChannel

rules = NoiseBuilder().after_gates(
    GateFilter.all().named("h"), NoiseChannel.amplitude_damping(0.4)
)
noisy = simulate_program(program).noise(rules).seed(42).shots(1000)
```

A program without runtime control flow runs as its circuit, with the same seeded shots.
The statevector, stabilizer, sparse, MPS, product state, factored, tensor network and
density matrix backends run dynamic programs; the stabilizer rank and Pauli propagation
engines keep no state per shot and raise `incompatible_backend`.
