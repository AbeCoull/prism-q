# Performance Decision Log

Dated record of decisions that changed how a performance-sensitive kernel or pipeline
is built. Each entry states the context, the options with their measured or estimated
cost, the decision, the measurement behind it, and the condition under which to
revisit. Newest entries first.

## 2026-08-22 Rank compute-bound work by whether the arithmetic is removable

- **Context**: two density-matrix channel kernels carried the same diagnosis,
  compute-bound at about 1.7x the one-sweep memory floor their one-qubit siblings set
  over the same `4^n` buffer. The diagnosis was correct both times and predicted the
  payoff only once.
- **Options considered**: remove the arithmetic algebraically, or reorganize how it
  issues. The two-qubit depolarizing kernel had removable arithmetic: the Pauli twirl
  collapses its 16x16 block superoperator to `alpha B + beta Tr(B) I4`, 256 complex
  multiply-accumulates per block down to 16 scales. The general two-qubit Kraus sweep
  did not: 16 multiply-accumulates per amplitude with no algebraic identity to delete
  them, leaving a wider sweep for instruction-level parallelism as the only lever.
- **Decision**: rank compute-bound optimization candidates by whether the arithmetic
  is removable (a closed form, operator structure such as diagonality) before
  considering vectorization. A removable-flops candidate is bounded by the gap to the
  memory floor. A reorganization candidate is bounded by the gap to the issue ceiling,
  which for interleaved `Complex64` arithmetic under AVX2 sits near 23% of FMA peak:
  AVX2 has no complex FMA, so each complex multiply pays a shuffle pair plus
  `vfmaddsub`, and lane width buys only address generation and load efficiency.
- **Measurement**: the closed-form depolarizing rewrite landed -49.0% at 10 qubits
  and -39.4% at 12 (`density_matrix/noisy_channels/depolarizing_2q/{10,12}`), moving
  the kernel from 1.72x to 1.04x its memory floor. Widening the general Kraus sweep
  landed -5.4% and -4.5% (`density_matrix/noisy_channels/kraus_2q_adjacent/{10,12}`);
  that kernel ran at 57.6 to 60.4 Gflop/s against a 256 Gflop/s AVX2 FMA peak. A
  first deliberate application of the rule, detecting diagonal superoperators in the
  same Kraus sweep, landed -53.0% to -55.6% at 10 qubits and -36.6% to -40.2% at 12
  on the `density_matrix/noisy_channels/kraus_2q` rows, again onto the floor. The
  statevector already encodes the same rule: all-diagonal `MultiFused` batches
  dispatch to a path costing one complex multiply per element instead of the dense
  kernel (see [Backends](./backends.md)).
- **Revisit trigger**: a state layout with split real and imaginary arrays, or
  hardware with a complex FMA instruction, changes the issue ceiling that bounds
  reorganization candidates.
