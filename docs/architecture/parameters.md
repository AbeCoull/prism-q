# Parameters, Binding and Gradients

A variational sweep holds one gate sequence and varies only the angles. Three pieces
share that view of a circuit: `Parameters` names the angles, `PreparedCircuit` reuses the
fusion work across bindings, and the gradient engines read the same links to know which
gates to differentiate.

## Parameter slots (`src/circuit/parameter.rs`)

`Parameters` is a side table over a circuit, not a field on the gates, so `Gate` stays at
16 bytes. Each `ParamLink` ties an instruction index to a slot in the value vector, and a
slot may drive several instructions: `bind` writes one angle to each, and the adjoint
gradient sums their contributions into one entry. `CircuitBuilder::param` records a
link on the gate it follows, and OpenQASM `input` declarations name the slots.

The arity is declared at construction, so a value vector of the wrong length is rejected
even when its trailing slots carry no link. The table also records the gate kind and
targets at each linked index, and `validate` compares them, because links are
instruction indices and an edited circuit would otherwise rebind the wrong gate without
complaint. `bind_into` overwrites a caller's circuit instead of allocating one per
point.

Bindable gates are `Rx`, `Ry`, `Rz`, `Rzz`, `P` and `PauliRot`, the `Gate` variants
carrying a rotation angle.

```admonish warning title="Adding a gate that carries an angle"
A new `Gate` variant with an angle owes three sites beyond its kernel:
`Gate::pauli_generator` for the gradients, `angle_of` and `angle_mut` in
`src/circuit/parameter.rs` for binding, and an angle recipe in `src/circuit/plan.rs` for
replay. Miss the third and `PreparedCircuit::bind_fused` keeps the template's angle with
no error, because replay patches only the sites it recorded.
```

## Plan capture and replay

Fusion decides the same block structure at every point of a sweep. `PreparedCircuit`
settles it once in a `FusionPlan` (`src/circuit/plan.rs`), along with the backend
choice, and rebinds against it. Capture gates the fusion floors on the buffer width of
the backend the prepared circuit holds, so a prepared density matrix fuses at `2n` as a
direct run does.

The plan is reusable. The matrices are not, since a changed angle changes every fused
matrix it feeds. `FusionPlan` records a recipe per payload derived from an angle: a list
of template instructions and how each one's matrix enters the product. Replay recomputes
the products and never caches them. Nested recipes splice by rewriting a placement flag
instead of materializing the inner product, which is sound because widening to a pair
and SWAP conjugation are both multiplicative.

The passes record under a `Tracer` that is inactive on the ordinary path, so fusion
outside the prepared form allocates what it always did.

Decisions that read a matrix rather than the gate sequence are guarded:

| Decision | Read by | Recorded as |
|----------|---------|-------------|
| A 1q run collapsing to the identity | `flush`, which elides it | `Guard::Fuses1q` |
| A 1q run matching a named gate | `Gate::recognize_matrix` | `Guard::Fuses1q` |
| Whether a 1q block is diagonal | `reorder_1q_gates`, commuting it past a control | `Guard::Fuses1q` |
| Whether a 2q block is diagonal | `PairRun::should_fuse` | `Guard::Diag4q` |
| Whether a pair is a controlled phase | `fuse_controlled_phases` | `Guard::Phase2q` |

Each guard is tested again per binding, and a binding that flips one falls back to
running the pipeline, so the stream matches an independently fused circuit either way.
The 1q diagonality guard lives on the run, not on a payload site: the block is often
absorbed into a `Fused2q` with no 2x2 of its own, and the reorder it steered has already
happened.

`DiagonalBatch` records one recipe per phase or parity entry. A diagonal pair is
decomposed after rebuilding its full matrix product, since its factors need not be
diagonal one by one. `BatchPhase` keeps source products through merging of repeated
pairs, batch splitting and changes of control qubit. The stage planner carries this
provenance through its nested diagonal passes, so a `PreparedCircuit` replays its
stages, and the device lowering runs after binding, so it replays under the profile of
the backend held.

A rewritten `CX Rz CX` records an `Rzz` matrix derived from the source angle, so
standalone parity batches and later pair fusion replay the same operation. A gate the
passes leave untouched still needs a site when it carries an angle: `PauliRot` holds its
angle in a boxed payload, not the enum slot, and replay patches it as it patches an `Rz`.

`PreparedCircuit` carries `run`, `expectation_values`, `observable_expectation`,
`expectation_gradient` and `expectation_gradient_shift`, each answering as the `Simulate`
terminal of that name does on the bound circuit, plus a `_many` form of each that takes
a list of bindings and splits it across cores where `run_batch` would. Gradient batches
split only below the statevector kernels' parallel width, 15 qubits, and run in order
above it. Where the held backend would answer differently, the call binds and hands the
circuit to `simulate`.

Capture pays most on narrow circuits, where
[fusion cost](./fusion.md#fusion-cost-against-apply-cost) is the larger share of a run.

## Adjoint gradient

`simulate(circuit).seed(seed).expectation_gradient(hamiltonian, params)` and
`sim::gradient::run_expectation_gradient` compute `⟨H⟩` and the exact gradient
`d⟨H⟩/dθ` for a weighted Pauli sum `H = Σ c_k P_k`, given as a `PauliObservable`, at a
cost independent of the parameter count (`src/sim/gradient.rs`).
`PreparedCircuit::expectation_gradient` runs the same pass on a held template, applying
its replayed fusion plan as the forward pass.

Two statevectors are propagated backwards. With `U = U_L…U_1` and `|φ⟩ = U|0⟩`:

1. The forward pass (fused) keeps `|φ⟩`. Build `|λ⟩ = H|φ⟩`; the value is
   `Re⟨φ|λ⟩`.
2. Sweep `i = L…1`. For a trainable gate with generator `G_i`, accumulate
   `Im⟨λ|G_i|φ⟩` (projector form for `P`), then step both states back through `U_i†`
   (`Gate::inverse()`).

The sweep is not gate by gate. A gate can be inverted once every gate above it that it
does not commute with has been, and a trainable gate's sandwich can be read once every
gate still above it commutes with its generator. A planner assigns each gate its
earliest such step, so one sandwich pass serves every gate ready at the same time (an Rz
layer, then an Ry layer of a hardware efficient ansatz, instead of one pass per qubit).
Gates inverted between two steps go through the fusion pipeline at the state width and
apply to both states as one stretch. Commutation is structural: exact between two Pauli
rotations, and per qubit otherwise, with each gate acting on a qubit as Z, X, Y, or
anything.

The sandwich `⟨λ|G|φ⟩` is the forward `pauli_expectation_from_masks` kernel
generalized to two vectors (`pauli_sandwich`, Rayon parallel from 16 qubits).

`Gate::pauli_generator` identifies the differentiable gates. A multi-qubit Pauli
rotation is `exp(-iθP/2)`, so its generator is the string itself:
`GeneratorKind::RotPauli` borrows the letters off the gate and the sandwich takes the
masks they imply. Trainable links on other gates, non-unitary instructions and
`QftBlock` are rejected.

The planner reads the unfused instruction stream so each gate keeps its link to its
generator, which fusion would erase along with the stored angle. Only the forward pass
and the inverted stretches are fused. Three prunings cut work without changing results:
the sweep stops at the earliest trainable gate in the cone, so a prefix with nothing
trainable costs no inverse applications; a gate no trainable gate below it waits on is
never inverted; and a trainable gate outside the Hamiltonian's inverse light cone has a
zero gradient, so its sandwich is skipped.

Memory is two statevectors, so the ceiling is about one qubit below a single run. Only
the statevector backend is supported.

## Parameter shift

`simulate(circuit).backend(kind).seed(seed).expectation_gradient_shift(hamiltonian,
params)` and `sim::gradient::run_expectation_gradient_shift` evaluate
`d⟨H⟩/dθ = (f(θ+π/2) - f(θ-π/2)) / 2` per trainable gate, exactly, from forward
`expectation_values` calls. The path inherits whatever the selected backend can
represent: sparse, MPS, factored, product state, density matrix and distributed, widths
past the statevector cap, and circuits containing `QftBlock`. A backend with no native
observable path reports `BackendUnsupported` naming itself.

The rule is exact because each differentiable gate is `exp(-iθG/2)` with `G` of
eigenvalues `±1`, so `⟨H⟩` is a trigonometric polynomial of degree 1 in that angle.
`P(θ)` has the projector `|1⟩⟨1|` as generator, eigenvalues `{0, 1}`, but
`P(θ) = e^{iθ/2} Rz(θ)` and the scalar cancels against its conjugate in `⟨ψ|H|ψ⟩`
wherever the gate sits, so the same shift applies.

The gate set is the adjoint's, so parameter shift reaches no gate the adjoint rejects.
It extends the backends and circuit shapes, not the gates. A `PauliRot` on a backend
without the native kernel is shifted in place, because the ladder expansion happens
below the forward evaluation.

Gates sharing a slot are shifted one at a time and their contributions summed. Shifting
them together gives a different quantity: two `Rx(θ)` on one qubit under `⟨Z⟩` give
`cos 2θ`, whose joint ±π/2 shift is zero rather than `-2 sin 2θ`.

Cost is `1 + 2 * links` evaluations against the adjoint's one, with no light cone
pruning: a trainable gate with a zero gradient is still evaluated twice. Evaluations run
in sequence, because each already drives a Rayon parallel backend run and several in
flight would multiply peak state memory by the parameter count, the resource this path
exists to stay under. With a noise model, only the density matrix kinds accept the pair
([noise across the terminals](./engine.md#noise-across-the-terminals)).
