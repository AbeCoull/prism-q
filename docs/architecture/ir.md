# Parser and Circuit IR

## Parser

Handwritten parser targeting a practical OpenQASM 3.0 subset. It processes input line by line and converts `&str` directly to `Circuit` IR with no intermediate AST.

**Supported**: `qubit`/`bit` declarations, OpenQASM standard gates and aliases (x, y, z, h, s, sdg, t, tdg, sx, rx, ry, rz, p/phase, cx/CX/cnot, cy, cz, cp/cphase, crx, cry, crz, ch, swap, ccx/toffoli, cswap/fredkin, cu, u1, u2, u3/u/U), Qiskit and exporter gates (sxdg, cs, csdg, csx, ccz, r, rzz, rxx, ryy, xx_plus_yy, xx_minus_yy, ecr, iswap, dcx, c3x, c4x, mcx, rccx, rc3x/rcccx), hardware-native gates (gpi, gpi2, ms, syc, sqrt_iswap, sqrt_iswap_inv), gate modifiers (`ctrl @`, `inv @`, `pow(k) @`), user-defined `gate` blocks, classical `if` conditionals with a single statement or a braced body, multi-register broadcast, measure, barrier, expression evaluator with math functions. OpenQASM 2.0 backward compatibility (`qreg`/`creg`, `measure q -> c` syntax).

**Unsupported**: `for`/`while` loops, subroutines, classical expressions beyond `if`.

See the [OpenQASM Support guide](../guides/openqasm.md) for a user-facing walkthrough.

## Circuit IR

`Circuit` holds `num_qubits`, `num_classical_bits`, and `Vec<Instruction>`. Instructions are an enum:

| Variant | Fields | Description |
|---------|--------|-------------|
| `Gate` | `gate`, `targets` | Gate application |
| `Measure` | `qubit`, `classical_bit` | Destructive measurement |
| `Barrier` | `qubits` | Synchronization barrier |
| `Conditional` | `condition`, `gate`, `targets` | Classical-controlled gate |
| `Region` | `Box<GuardedRegion>` | Classical-controlled span of instructions |

Targets use `SmallVec<[usize; 4]>`, inline storage for up to 4 qubits, no heap allocation for typical gates.

### Guarded regions

`Region` carries a condition and a body that runs, in order, only when the
condition holds against the classical bits as they stand when control reaches
it. The body admits any instruction, measurement and reset included, and may
nest to `MAX_REGION_DEPTH`. The body is boxed, so `Instruction` stays at 96
bytes.

`GuardedRegion` caches the sorted union of the qubits its body touches, nested
bodies included. Passes read that set rather than re-walking the body: fusion
flushes exactly those qubits at the region boundary and is transparent
elsewhere, and no pass fuses across the boundary because a region is not an
`Instruction::Gate`.

`Conditional` is the single-gate lowering of the same construct. `if (c) x q[0];`
keeps that form, so the common guarded gate costs no allocation; anything else
becomes a `Region`. Build either through `circuit::guarded`, which picks the
form and returns `None` for an empty body.

A circuit holding a region runs once per shot: measurement-conditioned execution
has no single evolved distribution to sample, so the compiled samplers reject it
and the run falls back to replay. Routes requiring a unitary circuit (adjoint
gradients, exact expectation values, Pauli propagation, stabilizer-rank
probabilities) reject a region for the same reason they reject a bare
conditional. Noise models index one event slot per instruction, so they reject a
region rather than leave its body noiseless.

## Gate enum

`Gate` is a `Clone` enum kept at **16 bytes**. Simple variants carry parameters inline. Composite variants use `Box` to stay within the 16-byte budget for cache-friendly dispatch.

```admonish warning title="Keep the enum at 16 bytes"
Adding inline data larger than 16 bytes pollutes cache lines and has caused 40-130%
regressions. Always check `size_of::<Gate>()` after adding a variant, and `Box` large
payloads.
```

| Variant | Data | Size |
|---------|------|------|
| `Id`, `X`, `Y`, `Z`, `H`, `S`, `Sdg`, `T`, `Tdg`, `SX`, `SXdg` | None | 16B |
| `Rx(f64)`, `Ry(f64)`, `Rz(f64)`, `P(f64)`, `Rzz(f64)` | Inline f64 | 16B |
| `Cx`, `Cz`, `Swap` | None | 16B |
| `Cu(Box<[[Complex64; 2]; 2]>)` | Boxed 2×2 | 16B |
| `Mcu(Box<McuData>)` | Boxed matrix + control count | 16B |
| `Fused(Box<[[Complex64; 2]; 2]>)` | Boxed pre-fused 1q matrix | 16B |
| `Fused2q(Box<[[Complex64; 4]; 4]>)` | Boxed pre-fused 2q matrix | 16B |
| `MultiFused(Box<MultiFusedData>)` | Batched 1q gates for tiled pass | 16B |
| `Multi2q(Box<Multi2qData>)` | Batched 2q gates for tiled pass | 16B |
| `BatchPhase(Box<BatchPhaseData>)` | Batched cphase with shared control | 16B |
| `BatchRzz(Box<BatchRzzData>)` | Batched ZZ rotations | 16B |
| `DiagonalBatch(Box<DiagonalBatchData>)` | Mixed diagonal 1q/2q batch | 16B |
| `PauliRot(Box<PauliRotData>)` | Multi-qubit Pauli rotation, boxed angle plus letters | 16B |

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit. Applying `x q[0]` produces state index 1, not 2.
```
