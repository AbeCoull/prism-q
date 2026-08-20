//! Circuit intermediate representation.
//!
//! The IR is backend-agnostic. All frontends (OpenQASM, future programmatic builders)
//! target this IR. Backends consume it without knowledge of the source format.
//!
//! # Design notes
//! - `Instruction` uses `SmallVec<[usize; 4]>` for qubit targets. Most gates (1-2 qubits)
//!   store targets inline without heap allocation. Multi-controlled gates (≥5 qubits) spill
//!   to the heap transparently.
//! - The circuit is append-only during construction. Optimization passes (fusion, reordering,
//!   cancellation) operate on the instruction stream via [`fusion::fuse_circuit`].

pub mod builder;
mod draw;
pub use draw::TextOptions;
mod svg;
pub use svg::SvgOptions;
mod expr;
pub mod fusion;
mod fusion_phase;
mod fusion_rzz;
pub mod openqasm;
pub mod parameter;
pub(crate) mod plan;
pub mod prepared;
pub mod qasm_export;

pub use parameter::{ParamLink, Parameters};
pub use prepared::PreparedCircuit;

use crate::gates::{Gate, PauliRotData};
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};
pub use smallvec::{SmallVec, smallvec};
use std::borrow::Cow;

/// A quantum circuit in PRISM-Q's internal representation.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Total number of qubits.
    pub num_qubits: usize,
    /// Total number of classical bits.
    pub num_classical_bits: usize,
    /// Ordered sequence of instructions.
    pub instructions: Vec<Instruction>,
}

impl Circuit {
    /// Create an empty circuit with the given qubit and classical bit counts.
    pub fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self {
            num_qubits,
            num_classical_bits,
            instructions: Vec::new(),
        }
    }

    /// Append a gate operation.
    ///
    /// # Panics
    /// Panics if any target index is out of bounds or if the gate's arity
    /// does not match `targets.len()`. Bounds checks run in both debug and
    /// release builds: a bad index propagates into kernel pointer math and
    /// would corrupt or read uninitialised memory otherwise.
    #[inline]
    pub fn add_gate(&mut self, gate: Gate, targets: &[usize]) {
        assert_eq!(
            gate.num_qubits(),
            targets.len(),
            "gate `{}` expects {} qubits, got {}",
            gate.name(),
            gate.num_qubits(),
            targets.len()
        );
        for &t in targets {
            assert!(
                t < self.num_qubits,
                "qubit index {} out of bounds (circuit has {} qubits)",
                t,
                self.num_qubits
            );
        }
        self.instructions.push(Instruction::Gate {
            gate,
            targets: SmallVec::from_slice(targets),
        });
    }

    /// Append the Pauli rotation `exp(-i θ P / 2)` for the Pauli string `P`
    /// given as one factor per qubit.
    ///
    /// Recognized forms lower at insert: a weight-1 string appends `Rx`, `Ry`,
    /// or `Rz`, and a two-qubit `ZZ` string appends `Rzz`, so Clifford
    /// recognition, diagonal batching, and fusion keep firing on them. Any
    /// other string appends the native [`Gate::PauliRot`], factors sorted by
    /// qubit.
    ///
    /// # Panics
    /// Panics if `factors` is empty, names a qubit twice, or names a qubit out
    /// of bounds.
    pub fn add_pauli_rotation(&mut self, theta: f64, factors: &[PauliTerm]) {
        let (gate, targets) = pauli_rotation_gate(theta, factors);
        self.add_gate(gate, &targets);
    }

    /// Append a measurement operation.
    ///
    /// # Panics
    /// Panics if `qubit` or `classical_bit` is out of bounds.
    #[inline]
    pub fn add_measure(&mut self, qubit: usize, classical_bit: usize) {
        assert!(
            qubit < self.num_qubits,
            "qubit index {} out of bounds (circuit has {} qubits)",
            qubit,
            self.num_qubits
        );
        assert!(
            classical_bit < self.num_classical_bits,
            "classical bit index {} out of bounds (circuit has {} classical bits)",
            classical_bit,
            self.num_classical_bits
        );
        self.instructions.push(Instruction::Measure {
            qubit,
            classical_bit,
        });
    }

    /// Measure every qubit into the classical bit of the same index, growing
    /// `num_classical_bits` to `num_qubits` if needed.
    pub fn measure_all(&mut self) {
        let n = self.num_qubits;
        if self.num_classical_bits < n {
            self.num_classical_bits = n;
        }
        for q in 0..n {
            self.add_measure(q, q);
        }
    }

    /// Append a reset operation, returning the qubit to |0⟩.
    ///
    /// # Panics
    /// Panics if `qubit` is out of bounds.
    #[inline]
    pub fn add_reset(&mut self, qubit: usize) {
        assert!(
            qubit < self.num_qubits,
            "qubit index {} out of bounds (circuit has {} qubits)",
            qubit,
            self.num_qubits
        );
        self.instructions.push(Instruction::Reset { qubit });
    }

    /// Append a barrier (scheduling hint, no physical operation).
    ///
    /// # Panics
    /// Panics if any qubit index is out of bounds.
    #[inline]
    pub fn add_barrier(&mut self, qubits: &[usize]) {
        for &q in qubits {
            assert!(
                q < self.num_qubits,
                "qubit index {} out of bounds (circuit has {} qubits)",
                q,
                self.num_qubits
            );
        }
        self.instructions.push(Instruction::Barrier {
            qubits: SmallVec::from_slice(qubits),
        });
    }

    /// Count of gate instructions (excludes measurements and barriers).
    pub fn gate_count(&self) -> usize {
        let mut count = 0;
        for_each_gate(&self.instructions, &mut |_| count += 1);
        count
    }

    /// Count T and Tdg gates in the circuit.
    pub fn t_count(&self) -> usize {
        let mut count = 0;
        for_each_gate(&self.instructions, &mut |gate| {
            if matches!(gate, Gate::T | Gate::Tdg) {
                count += 1;
            }
        });
        count
    }

    /// Returns true if the circuit contains any T or Tdg gates.
    pub fn has_t_gates(&self) -> bool {
        any_gate(&self.instructions, &mut |gate| {
            matches!(gate, Gate::T | Gate::Tdg)
        })
    }

    /// True if every gate in the circuit is a Clifford gate.
    ///
    /// When true, the stabilizer backend can simulate this circuit exactly
    /// in O(n^2) time regardless of qubit count.
    pub fn is_clifford_only(&self) -> bool {
        !any_gate(&self.instructions, &mut |gate| !gate.is_clifford())
    }

    /// True if every gate is Clifford or T/Tdg.
    pub fn is_clifford_plus_t(&self) -> bool {
        !any_gate(&self.instructions, &mut |gate| {
            !(gate.is_clifford() || matches!(gate, Gate::T | Gate::Tdg))
        })
    }

    /// True if every gate preserves computational basis states (diagonal or
    /// permutation). When true, the sparse backend is optimal: the state
    /// always has exactly one non-zero amplitude, giving O(1) memory and O(n)
    /// per-gate cost regardless of qubit count.
    pub fn is_sparse_friendly(&self) -> bool {
        !any_gate(&self.instructions, &mut |gate| !gate.preserves_sparsity())
    }

    /// True if the circuit contains any multi-qubit (entangling) gates.
    ///
    /// When false, the product state backend can simulate in O(n) time.
    pub fn has_entangling_gates(&self) -> bool {
        any_gate(&self.instructions, &mut |gate| gate.num_qubits() >= 2)
    }

    /// Herfindahl-Hirschman index of the qubit interaction graph partition.
    ///
    /// Returns Σ(sᵢ/n)² where sᵢ is the size of each connected component.
    /// Ranges from 1/n (all singletons) to 1.0 (one component).
    /// Low values indicate many independent subsystems where factored
    /// backends amortize cost; 1.0 means fully connected (no benefit).
    pub fn connectivity_hhi(&self) -> f64 {
        let n = self.num_qubits;
        if n == 0 {
            return 1.0;
        }
        let components = self.independent_subsystems();
        let nf = n as f64;
        components
            .iter()
            .map(|c| {
                let s = c.len() as f64;
                (s * s) / (nf * nf)
            })
            .sum()
    }

    /// True if no gate or conditional appears after any measurement.
    pub fn has_terminal_measurements_only(&self) -> bool {
        let mut seen_measurement = false;
        for inst in &self.instructions {
            match inst {
                Instruction::Conditional { .. } | Instruction::Region(_) => return false,
                Instruction::Measure { .. } => {
                    seen_measurement = true;
                }
                Instruction::Gate { .. } | Instruction::Reset { .. } => {
                    if seen_measurement {
                        return false;
                    }
                }
                Instruction::Barrier { .. } => {}
            }
        }
        true
    }

    /// Drop guards whose condition no measurement can reach, and inline the
    /// ones that must be taken. Borrows unchanged when nothing folds.
    ///
    /// A condition reading only bits no preceding measurement writes evaluates
    /// against the initial classical state, which every backend zeroes.
    pub fn fold_static_guards(&self) -> Cow<'_, Circuit> {
        let guarded = self.instructions.iter().any(|inst| {
            matches!(
                inst,
                Instruction::Conditional { .. } | Instruction::Region(_)
            )
        });
        if !guarded {
            return Cow::Borrowed(self);
        }
        let mut state = FoldState {
            written: vec![false; self.num_classical_bits],
            zeros: vec![false; self.num_classical_bits],
            folded: false,
        };
        let mut instructions = Vec::with_capacity(self.instructions.len());
        fold_static_guards_into(&self.instructions, &mut state, &mut instructions);
        if !state.folded {
            return Cow::Borrowed(self);
        }
        Cow::Owned(Circuit {
            num_qubits: self.num_qubits,
            num_classical_bits: self.num_classical_bits,
            instructions,
        })
    }

    /// Extract the qubit-to-classical-bit mapping from all measurements.
    ///
    /// Measurements inside a guarded region are included; whether the region is
    /// taken is a runtime fact, so the map covers what the circuit can write.
    pub fn measurement_map(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for_each_measure(&self.instructions, &mut |qubit, classical_bit| {
            out.push((qubit, classical_bit))
        });
        out
    }

    pub fn without_measurements(&self) -> Circuit {
        let mut c = Circuit::new(self.num_qubits, self.num_classical_bits);
        c.instructions = strip_measurements(&self.instructions);
        c
    }

    pub fn has_resets(&self) -> bool {
        any_instruction(&self.instructions, &mut |i| {
            matches!(i, Instruction::Reset { .. })
        })
    }

    /// Partition qubits into independent (non-interacting) subsystems.
    ///
    /// Two qubits are in the same subsystem if any multi-qubit gate connects
    /// them, transitively. Classical dependencies (measure qubit → conditional
    /// target) also merge subsystems, since the conditional outcome depends on
    /// measurement results that must be available in the same simulation context.
    /// Returns a list of qubit groups, each sorted.
    /// A fully-entangled circuit returns a single group containing all qubits.
    pub fn independent_subsystems(&self) -> Vec<Vec<usize>> {
        let n = self.num_qubits;
        if n == 0 {
            return Vec::new();
        }
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0u8; n];

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            if rank[ra] < rank[rb] {
                parent[ra] = rb;
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra;
            } else {
                parent[rb] = ra;
                rank[ra] += 1;
            }
        }

        // Build cbit → measurement qubit map for classical dependency tracking
        let mut cbit_to_qubit: Vec<Option<usize>> = vec![None; self.num_classical_bits.max(1)];
        for_each_measure(&self.instructions, &mut |qubit, classical_bit| {
            cbit_to_qubit[classical_bit] = Some(qubit);
        });

        let mut condition_bits: Vec<usize> = Vec::new();
        for inst in &self.instructions {
            condition_bits.clear();
            let targets = match inst {
                Instruction::Gate { targets, .. } => targets.as_slice(),
                Instruction::Conditional {
                    condition, targets, ..
                } => {
                    collect_condition_bits(condition, &mut condition_bits);
                    targets.as_slice()
                }
                Instruction::Region(region) => {
                    collect_condition_bits(region.condition(), &mut condition_bits);
                    collect_body_condition_bits(region.body(), &mut condition_bits);
                    region.qubits()
                }
                _ => continue,
            };
            let Some(&first) = targets.first() else {
                continue;
            };
            for &bit in &condition_bits {
                if let Some(mq) = cbit_to_qubit.get(bit).copied().flatten() {
                    union(&mut parent, &mut rank, first, mq);
                }
            }
            for &t in &targets[1..] {
                union(&mut parent, &mut rank, first, t);
            }
        }

        let mut components: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for q in 0..n {
            let root = find(&mut parent, q);
            components.entry(root).or_default().push(q);
        }
        let mut result: Vec<Vec<usize>> = components.into_values().collect();
        result.sort_by_key(|group| group[0]);
        result
    }

    /// Extract a sub-circuit containing only the given qubits.
    ///
    /// Returns `(sub_circuit, qubit_map, classical_map)` where:
    /// - `sub_circuit` has remapped qubit/classical indices starting from 0
    /// - `qubit_map[local] = original` qubit index
    /// - `classical_map[local] = original` classical bit index
    pub fn extract_subcircuit(&self, qubit_set: &[usize]) -> (Circuit, Vec<usize>, Vec<usize>) {
        let mut old_to_new_qubit: Vec<Option<usize>> = vec![None; self.num_qubits];
        for (new_idx, &old_idx) in qubit_set.iter().enumerate() {
            old_to_new_qubit[old_idx] = Some(new_idx);
        }

        let mut classical_bits_used: Vec<usize> = Vec::new();
        let max_cb = self.num_classical_bits.max(1);
        let mut old_to_new_classical: Vec<Option<usize>> = vec![None; max_cb];

        for_each_measure(&self.instructions, &mut |qubit, classical_bit| {
            if old_to_new_qubit[qubit].is_some() && old_to_new_classical[classical_bit].is_none() {
                let new_idx = classical_bits_used.len();
                old_to_new_classical[classical_bit] = Some(new_idx);
                classical_bits_used.push(classical_bit);
            }
        });

        let mut sub = Circuit::new(qubit_set.len(), classical_bits_used.len());

        let qubit_of = |q: usize| old_to_new_qubit[q].expect("membership checked by the caller");
        let cbit_of = |c: usize| old_to_new_classical[c].unwrap_or(c);

        for inst in &self.instructions {
            match inst {
                Instruction::Gate { gate, targets } => {
                    if targets.iter().all(|&t| old_to_new_qubit[t].is_some()) {
                        let new_targets: SmallVec<[usize; 4]> = targets
                            .iter()
                            .map(|&t| old_to_new_qubit[t].unwrap())
                            .collect();
                        sub.instructions.push(Instruction::Gate {
                            gate: gate.clone(),
                            targets: new_targets,
                        });
                    }
                }
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    if let (Some(nq), Some(nc)) = (
                        old_to_new_qubit[*qubit],
                        old_to_new_classical[*classical_bit],
                    ) {
                        sub.instructions.push(Instruction::Measure {
                            qubit: nq,
                            classical_bit: nc,
                        });
                    }
                }
                Instruction::Reset { qubit } => {
                    if let Some(nq) = old_to_new_qubit[*qubit] {
                        sub.instructions.push(Instruction::Reset { qubit: nq });
                    }
                }
                Instruction::Barrier { qubits } => {
                    let new_qs: SmallVec<[usize; 4]> =
                        qubits.iter().filter_map(|&q| old_to_new_qubit[q]).collect();
                    if new_qs.len() >= 2 {
                        sub.instructions
                            .push(Instruction::Barrier { qubits: new_qs });
                    }
                }
                Instruction::Conditional { targets, .. } => {
                    if targets.iter().all(|&t| old_to_new_qubit[t].is_some()) {
                        sub.instructions
                            .push(remap_instruction(inst, &qubit_of, &cbit_of));
                    }
                }
                Instruction::Region(region) => {
                    if region
                        .qubits()
                        .iter()
                        .all(|&q| old_to_new_qubit[q].is_some())
                    {
                        sub.instructions
                            .push(remap_instruction(inst, &qubit_of, &cbit_of));
                    }
                }
            }
        }

        (sub, qubit_set.to_vec(), classical_bits_used)
    }

    /// Partition all instructions across independent subsystems in a single pass.
    ///
    /// Replaces K calls to `extract_subcircuit` (each scanning the full instruction
    /// stream) with two O(N) passes: one for classical bit discovery, one for
    /// instruction routing.
    pub fn partition_subcircuits(
        &self,
        components: &[Vec<usize>],
    ) -> Vec<(Circuit, Vec<usize>, Vec<usize>)> {
        let k = components.len();

        // Qubit → (component_index, new_qubit_index)
        let mut qubit_map: Vec<(usize, usize)> = vec![(0, 0); self.num_qubits];
        for (comp_idx, component) in components.iter().enumerate() {
            for (new_idx, &old_idx) in component.iter().enumerate() {
                qubit_map[old_idx] = (comp_idx, new_idx);
            }
        }

        // Pass 1: discover classical bits per component
        let mut classical_bits_per_comp: Vec<Vec<usize>> = vec![Vec::new(); k];
        let max_cb = self.num_classical_bits.max(1);
        let mut cbit_map: Vec<Option<(usize, usize)>> = vec![None; max_cb];

        for_each_measure(&self.instructions, &mut |qubit, classical_bit| {
            let (comp_idx, _) = qubit_map[qubit];
            if cbit_map[classical_bit].is_none() {
                let new_idx = classical_bits_per_comp[comp_idx].len();
                cbit_map[classical_bit] = Some((comp_idx, new_idx));
                classical_bits_per_comp[comp_idx].push(classical_bit);
            }
        });

        let mut subs: Vec<Circuit> = (0..k)
            .map(|i| Circuit::new(components[i].len(), classical_bits_per_comp[i].len()))
            .collect();

        let mut barrier_buf: Vec<SmallVec<[usize; 4]>> = (0..k).map(|_| SmallVec::new()).collect();

        let qubit_of = |q: usize| qubit_map[q].1;
        let cbit_of = |c: usize| cbit_map[c].map(|(_, nc)| nc).unwrap_or(c);

        // Pass 2: route each instruction to its component
        for inst in &self.instructions {
            match inst {
                Instruction::Gate { gate, targets } => {
                    let (comp_idx, _) = qubit_map[targets[0]];
                    let new_targets: SmallVec<[usize; 4]> =
                        targets.iter().map(|&t| qubit_map[t].1).collect();
                    subs[comp_idx].instructions.push(Instruction::Gate {
                        gate: gate.clone(),
                        targets: new_targets,
                    });
                }
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    let (comp_idx, nq) = qubit_map[*qubit];
                    if let Some((_, nc)) = cbit_map[*classical_bit] {
                        subs[comp_idx].instructions.push(Instruction::Measure {
                            qubit: nq,
                            classical_bit: nc,
                        });
                    }
                }
                Instruction::Reset { qubit } => {
                    let (comp_idx, nq) = qubit_map[*qubit];
                    subs[comp_idx]
                        .instructions
                        .push(Instruction::Reset { qubit: nq });
                }
                Instruction::Barrier { qubits } => {
                    for buf in barrier_buf.iter_mut() {
                        buf.clear();
                    }
                    for &q in qubits.iter() {
                        let (comp_idx, nq) = qubit_map[q];
                        barrier_buf[comp_idx].push(nq);
                    }
                    for (comp_idx, new_qs) in barrier_buf.iter().enumerate() {
                        if new_qs.len() >= 2 {
                            subs[comp_idx].instructions.push(Instruction::Barrier {
                                qubits: new_qs.clone(),
                            });
                        }
                    }
                }
                Instruction::Conditional { targets, .. } => {
                    let (comp_idx, _) = qubit_map[targets[0]];
                    subs[comp_idx]
                        .instructions
                        .push(remap_instruction(inst, &qubit_of, &cbit_of));
                }
                Instruction::Region(region) => {
                    let Some(&first) = region.qubits().first() else {
                        continue;
                    };
                    let (comp_idx, _) = qubit_map[first];
                    subs[comp_idx]
                        .instructions
                        .push(remap_instruction(inst, &qubit_of, &cbit_of));
                }
            }
        }

        subs.into_iter()
            .enumerate()
            .map(|(i, sub)| {
                (
                    sub,
                    components[i].clone(),
                    classical_bits_per_comp[i].clone(),
                )
            })
            .collect()
    }

    /// Split the circuit into a Clifford prefix and a non-Clifford tail.
    ///
    /// Returns `Some((prefix, tail))` if the circuit starts with at least one
    /// Clifford gate before the first non-Clifford gate. The prefix contains
    /// only Clifford gates, measurements, and barriers; the tail starts at
    /// the first non-Clifford gate and includes everything after it.
    ///
    /// Measurements terminate the prefix (they collapse state and must be
    /// committed before backend switch). Barriers are transparent.
    ///
    /// Returns `None` if the circuit has no Clifford prefix (first gate is
    /// non-Clifford) or is entirely Clifford.
    pub fn clifford_prefix_split(&self) -> Option<(Circuit, Circuit)> {
        let mut split_at = 0;

        for (i, inst) in self.instructions.iter().enumerate() {
            match inst {
                Instruction::Gate { gate, .. } => {
                    if !gate.is_clifford() {
                        split_at = i;
                        break;
                    }
                }
                Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
                | Instruction::Region(_) => {
                    split_at = i;
                    break;
                }
                Instruction::Barrier { .. } => {}
            }
            split_at = i + 1;
        }

        // No split if first gate is already non-Clifford or entire circuit is Clifford
        if split_at == 0 || split_at >= self.instructions.len() {
            return None;
        }

        let mut prefix = Circuit::new(self.num_qubits, self.num_classical_bits);
        prefix.instructions = self.instructions[..split_at].to_vec();

        let mut tail = Circuit::new(self.num_qubits, self.num_classical_bits);
        tail.instructions = self.instructions[split_at..].to_vec();

        Some((prefix, tail))
    }

    /// Same register shape as `self` with a replacement instruction list.
    ///
    /// The rebuild form every fusion pass uses to emit its rewritten circuit.
    pub(crate) fn with_instructions(&self, instructions: Vec<Instruction>) -> Circuit {
        Circuit {
            num_qubits: self.num_qubits,
            num_classical_bits: self.num_classical_bits,
            instructions,
        }
    }

    /// Circuit depth via greedy layer assignment.
    ///
    /// Each gate occupies the earliest layer where all its qubits are free.
    /// Measurements count as depth-1 operations. Barriers synchronize qubits
    /// to the same layer without adding depth.
    pub fn depth(&self) -> usize {
        let mut depth = 0usize;
        for_each_placement(&self.instructions, self.num_qubits, |inst, layer| {
            if !matches!(inst, Instruction::Barrier { .. }) {
                depth = depth.max(layer + 1);
            }
        });
        depth
    }
}

/// Walk instructions in greedy layer order, calling `visit(inst, layer)` with
/// the earliest layer where every touched qubit is free. Gates and
/// conditionals advance their qubits past the layer, measurements and resets
/// advance one step, and barriers synchronize their qubits to the layer
/// without occupying it. Shared by [`Circuit::depth`] and the drawing
/// moment assignment.
fn for_each_placement(
    instructions: &[Instruction],
    num_qubits: usize,
    mut visit: impl FnMut(&Instruction, usize),
) {
    let mut qubit_depth = vec![0usize; num_qubits];
    for inst in instructions {
        match inst {
            Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => {
                let d = targets.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                visit(inst, d);
                for &q in targets.iter() {
                    qubit_depth[q] = d + 1;
                }
            }
            Instruction::Region(region) => {
                let qubits = region.qubits();
                let d = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                visit(inst, d);
                for &q in qubits {
                    qubit_depth[q] = d + 1;
                }
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                let d = qubit_depth[*qubit];
                visit(inst, d);
                qubit_depth[*qubit] = d + 1;
            }
            Instruction::Barrier { qubits } => {
                let d = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                visit(inst, d);
                for &q in qubits.iter() {
                    qubit_depth[q] = d;
                }
            }
        }
    }
}

/// One step in the textbook QFT decomposition. Yielded by
/// [`qft_textbook_steps`] so backends and the sim layer share a single
/// definition of the H + controlled-phase + Swap pattern.
#[derive(Debug, Clone, Copy)]
pub enum QftTextbookStep {
    Hadamard(usize),
    CPhase {
        control: usize,
        target: usize,
        theta: f64,
    },
    Swap(usize, usize),
}

/// Yield the textbook QFT decomposition for `num` qubits starting at `start`.
/// Order: for each qubit `q` from `num - 1` down to `0`, `H q[start+q]` followed
/// by `cphase(TAU / 2^(q-k+1))` controlled by `q[start+k]` for `k in 0..q`, then
/// bit-reversal swaps. Matches the native FFT (`apply_qft_block`).
pub fn qft_textbook_steps(start: usize, num: usize) -> impl Iterator<Item = QftTextbookStep> {
    let outer = (0..num).rev().flat_map(move |q| {
        let head = std::iter::once(QftTextbookStep::Hadamard(start + q));
        let phases = (0..q).map(move |k| QftTextbookStep::CPhase {
            control: start + k,
            target: start + q,
            theta: std::f64::consts::TAU / (1u64 << (q - k + 1)) as f64,
        });
        head.chain(phases)
    });
    let swaps = (0..num / 2).map(move |i| QftTextbookStep::Swap(start + i, start + num - 1 - i));
    outer.chain(swaps)
}

/// Expand `Gate::QftBlock` instructions to textbook QFT gates.
///
/// Backends without native support call this before dispatch. Returns
/// `Cow::Borrowed` when there is nothing to expand.
pub fn expand_qft_blocks(circuit: &Circuit) -> std::borrow::Cow<'_, Circuit> {
    if !any_bare_gate(&circuit.instructions, &mut |gate| {
        matches!(gate, Gate::QftBlock { .. })
    }) {
        return std::borrow::Cow::Borrowed(circuit);
    }
    std::borrow::Cow::Owned(circuit.with_instructions(expanded_qft_blocks(&circuit.instructions)))
}

fn expanded_qft_blocks(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len() * 2);
    for inst in instructions {
        if let Instruction::Gate {
            gate: Gate::QftBlock { start, num },
            ..
        } = inst
        {
            for step in qft_textbook_steps(*start as usize, *num as usize) {
                match step {
                    QftTextbookStep::Hadamard(q) => out.push(Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q],
                    }),
                    QftTextbookStep::CPhase {
                        control,
                        target,
                        theta,
                    } => out.push(Instruction::Gate {
                        gate: Gate::cphase(theta),
                        targets: smallvec![control, target],
                    }),
                    QftTextbookStep::Swap(a, b) => out.push(Instruction::Gate {
                        gate: Gate::Swap,
                        targets: smallvec![a, b],
                    }),
                }
            }
        } else if let Instruction::Region(region) = inst {
            out.push(Instruction::Region(Box::new(GuardedRegion::new(
                region.condition().clone(),
                expanded_qft_blocks(region.body()),
            ))));
        } else {
            out.push(inst.clone());
        }
    }
    out
}

/// Emit the CNOT-ladder lowering of `exp(-i θ P / 2)` gate by gate.
///
/// Basis layer first (`H` for an X letter, `Sdg` then `H` for Y), a CX chain
/// accumulating the parity on the last target, one `Rz(θ)`, then the chain and
/// basis layer unwound. Shared by [`expand_pauli_rotations`] and the GPU
/// inline expansion so both routes lower identically.
pub(crate) fn pauli_rotation_lowering(
    theta: f64,
    targets: &[usize],
    axes: &[PauliAxis],
    mut emit: impl FnMut(Gate, &[usize]),
) {
    for (&q, axis) in targets.iter().zip(axes) {
        match axis {
            PauliAxis::X => emit(Gate::H, &[q]),
            PauliAxis::Y => {
                emit(Gate::Sdg, &[q]);
                emit(Gate::H, &[q]);
            }
            PauliAxis::Z => {}
        }
    }
    for pair in targets.windows(2) {
        emit(Gate::Cx, &[pair[0], pair[1]]);
    }
    emit(Gate::Rz(theta), &[targets[targets.len() - 1]]);
    for pair in targets.windows(2).rev() {
        emit(Gate::Cx, &[pair[0], pair[1]]);
    }
    for (&q, axis) in targets.iter().zip(axes) {
        match axis {
            PauliAxis::X => emit(Gate::H, &[q]),
            PauliAxis::Y => {
                emit(Gate::H, &[q]);
                emit(Gate::S, &[q]);
            }
            PauliAxis::Z => {}
        }
    }
}

/// Expand `Gate::PauliRot` instructions to the CNOT-ladder lowering.
///
/// Backends without the native kernel call this before dispatch, the same
/// probe-plus-expansion route as [`expand_qft_blocks`]. Returns
/// `Cow::Borrowed` when there is nothing to expand.
/// Lower a Pauli rotation to the gate and target order a circuit stores it as,
/// the recognizing step behind [`Circuit::add_pauli_rotation`].
///
/// Shared with the OpenQASM parser, which builds the instruction itself and so
/// cannot go through `add_pauli_rotation`. Targets come back sorted by qubit
/// with the letters aligned to them.
///
/// # Panics
/// Panics if `factors` is empty or names a qubit twice.
pub(crate) fn pauli_rotation_gate(
    theta: f64,
    factors: &[PauliTerm],
) -> (Gate, SmallVec<[usize; 4]>) {
    assert!(
        !factors.is_empty(),
        "Pauli rotation needs at least one factor"
    );
    let mut sorted: SmallVec<[PauliTerm; 4]> = SmallVec::from_slice(factors);
    sorted.sort_unstable_by_key(|term| term.qubit);
    for pair in sorted.windows(2) {
        assert_ne!(
            pair[0].qubit, pair[1].qubit,
            "Pauli rotation has duplicate factor on qubit {}",
            pair[0].qubit
        );
    }
    match sorted.as_slice() {
        [term] => {
            let gate = match term.axis {
                PauliAxis::X => Gate::Rx(theta),
                PauliAxis::Y => Gate::Ry(theta),
                PauliAxis::Z => Gate::Rz(theta),
            };
            (gate, smallvec![term.qubit])
        }
        [a, b] if a.axis == PauliAxis::Z && b.axis == PauliAxis::Z => {
            (Gate::Rzz(theta), smallvec![a.qubit, b.qubit])
        }
        _ => {
            let targets: SmallVec<[usize; 4]> = sorted.iter().map(|term| term.qubit).collect();
            let axes: Vec<PauliAxis> = sorted.iter().map(|term| term.axis).collect();
            (
                Gate::PauliRot(Box::new(PauliRotData { theta, axes })),
                targets,
            )
        }
    }
}

pub fn expand_pauli_rotations(circuit: &Circuit) -> std::borrow::Cow<'_, Circuit> {
    if !any_bare_gate(&circuit.instructions, &mut |gate| {
        matches!(gate, Gate::PauliRot(_))
    }) {
        return std::borrow::Cow::Borrowed(circuit);
    }
    std::borrow::Cow::Owned(
        circuit.with_instructions(expanded_pauli_rotations(&circuit.instructions)),
    )
}

fn expanded_pauli_rotations(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len() * 2);
    for inst in instructions {
        if let Instruction::Gate {
            gate: Gate::PauliRot(data),
            targets,
        } = inst
        {
            pauli_rotation_lowering(data.theta, targets, &data.axes, |gate, tgts| {
                out.push(Instruction::Gate {
                    gate,
                    targets: SmallVec::from_slice(tgts),
                });
            });
        } else if let Instruction::Region(region) = inst {
            out.push(Instruction::Region(Box::new(GuardedRegion::new(
                region.condition().clone(),
                expanded_pauli_rotations(region.body()),
            ))));
        } else {
            out.push(inst.clone());
        }
    }
    out
}

/// Condition for classically-controlled gate execution.
#[derive(Debug, Clone)]
pub enum ClassicalCondition {
    /// True when the classical bit at `bit` is 1.
    BitIsOne(usize),
    /// True when the classical bit at `bit` is 0.
    BitIsZero(usize),
    /// True when the classical register (bits `offset..offset+size`) equals `value`.
    RegisterEquals {
        offset: usize,
        size: usize,
        value: u64,
    },
    /// True when the classical register (bits `offset..offset+size`) does not equal `value`.
    RegisterNotEquals {
        offset: usize,
        size: usize,
        value: u64,
    },
    /// True when the XOR of the listed bits equals `expected`.
    ///
    /// The bits need not be contiguous, which is what a register comparison
    /// cannot express and what a detector-style predicate over measurement
    /// records needs. Boxed to keep the enum at the size the register variants
    /// already set. An empty bit list has parity zero and no OpenQASM form, so
    /// [`qasm_export`] rejects it.
    Parity { bits: Box<[usize]>, expected: bool },
}

impl ClassicalCondition {
    /// The condition that holds exactly when this one does not. Total: the
    /// language is closed under negation.
    pub fn negate(&self) -> ClassicalCondition {
        match self {
            ClassicalCondition::BitIsOne(bit) => ClassicalCondition::BitIsZero(*bit),
            ClassicalCondition::BitIsZero(bit) => ClassicalCondition::BitIsOne(*bit),
            ClassicalCondition::RegisterEquals {
                offset,
                size,
                value,
            } => ClassicalCondition::RegisterNotEquals {
                offset: *offset,
                size: *size,
                value: *value,
            },
            ClassicalCondition::RegisterNotEquals {
                offset,
                size,
                value,
            } => ClassicalCondition::RegisterEquals {
                offset: *offset,
                size: *size,
                value: *value,
            },
            ClassicalCondition::Parity { bits, expected } => ClassicalCondition::Parity {
                bits: bits.clone(),
                expected: !expected,
            },
        }
    }

    pub fn evaluate(&self, classical_bits: &[bool]) -> bool {
        match self {
            ClassicalCondition::BitIsOne(bit) => classical_bits[*bit],
            ClassicalCondition::BitIsZero(bit) => !classical_bits[*bit],
            ClassicalCondition::RegisterEquals {
                offset,
                size,
                value,
            }
            | ClassicalCondition::RegisterNotEquals {
                offset,
                size,
                value,
            } => {
                let mut reg_val = 0u64;
                for i in 0..*size {
                    if classical_bits[offset + i] {
                        reg_val |= 1u64 << i;
                    }
                }
                let eq = reg_val == *value;
                if matches!(self, ClassicalCondition::RegisterEquals { .. }) {
                    eq
                } else {
                    !eq
                }
            }
            ClassicalCondition::Parity { bits, expected } => {
                bits.iter()
                    .fold(false, |acc, &bit| acc ^ classical_bits[bit])
                    == *expected
            }
        }
    }
}

/// Deepest guarded-region nesting the parser accepts.
///
/// Every pass walks a region body recursively, so the depth is bounded at
/// construction to keep those walks finite.
pub const MAX_REGION_DEPTH: usize = 16;

/// A guarded region: `body` executes in order iff `condition` holds.
///
/// The condition is evaluated once, against the classical bits as they stand
/// when control reaches the region. Bits written by measurements inside a taken
/// body are visible to later instructions.
///
/// Fields are private because [`GuardedRegion::qubits`] caches the union over
/// the body; build a new region rather than editing one in place.
#[derive(Debug, Clone)]
pub struct GuardedRegion {
    condition: ClassicalCondition,
    body: Vec<Instruction>,
    qubits: SmallVec<[usize; 4]>,
}

impl GuardedRegion {
    /// Compute the body's qubit union once, so each pass reads it instead of
    /// re-walking the body.
    pub fn new(condition: ClassicalCondition, body: Vec<Instruction>) -> Self {
        let mut qubits: SmallVec<[usize; 4]> = SmallVec::new();
        collect_region_qubits(&body, &mut qubits);
        qubits.sort_unstable();
        qubits.dedup();
        Self {
            condition,
            body,
            qubits,
        }
    }

    pub fn condition(&self) -> &ClassicalCondition {
        &self.condition
    }

    pub fn body(&self) -> &[Instruction] {
        &self.body
    }

    /// Sorted union of every qubit the body touches, nested regions included.
    pub fn qubits(&self) -> &[usize] {
        &self.qubits
    }

    /// Nesting depth, counting this region as 1.
    pub fn depth(&self) -> usize {
        1 + self
            .body
            .iter()
            .filter_map(|inst| match inst {
                Instruction::Region(inner) => Some(inner.depth()),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

/// Visit every gate, including a guarded one, descending into region bodies.
fn for_each_gate(instructions: &[Instruction], visit: &mut impl FnMut(&Gate)) {
    for inst in instructions {
        match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => visit(gate),
            Instruction::Region(region) => for_each_gate(region.body(), visit),
            _ => {}
        }
    }
}

/// Short-circuiting scan over unguarded [`Instruction::Gate`] payloads only,
/// which is the set the lowering passes rewrite.
fn any_bare_gate(instructions: &[Instruction], pred: &mut impl FnMut(&Gate) -> bool) -> bool {
    instructions.iter().any(|inst| match inst {
        Instruction::Gate { gate, .. } => pred(gate),
        Instruction::Region(region) => any_bare_gate(region.body(), pred),
        _ => false,
    })
}

/// Short-circuiting [`for_each_gate`].
pub(crate) fn any_gate(instructions: &[Instruction], pred: &mut impl FnMut(&Gate) -> bool) -> bool {
    instructions.iter().any(|inst| match inst {
        Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => pred(gate),
        Instruction::Region(region) => any_gate(region.body(), pred),
        _ => false,
    })
}

/// True when `pred` holds for `inst` or for anything inside a region body.
fn any_instruction(
    instructions: &[Instruction],
    pred: &mut impl FnMut(&Instruction) -> bool,
) -> bool {
    instructions.iter().any(|inst| {
        pred(inst)
            || match inst {
                Instruction::Region(region) => any_instruction(region.body(), pred),
                _ => false,
            }
    })
}

fn strip_measurements(instructions: &[Instruction]) -> Vec<Instruction> {
    instructions
        .iter()
        .filter(|inst| !matches!(inst, Instruction::Measure { .. }))
        .filter_map(|inst| match inst {
            // A region whose body was all measurements drops out entirely,
            // rather than surviving as the empty region `guarded` never builds.
            Instruction::Region(region) => guarded(
                region.condition().clone(),
                strip_measurements(region.body()),
            ),
            other => Some(other.clone()),
        })
        .collect()
}

/// Visit every measurement in `instructions`, descending into region bodies.
fn for_each_measure(instructions: &[Instruction], visit: &mut impl FnMut(usize, usize)) {
    for inst in instructions {
        match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => visit(*qubit, *classical_bit),
            Instruction::Region(region) => for_each_measure(region.body(), visit),
            _ => {}
        }
    }
}

/// Which classical bits a preceding measurement may have written, plus the
/// all-zero vector a statically decidable condition evaluates against.
struct FoldState {
    written: Vec<bool>,
    zeros: Vec<bool>,
    folded: bool,
}

impl FoldState {
    /// `Some(value)` when no preceding measurement can have written a bit the
    /// condition reads, so its value is fixed before the run starts.
    fn static_value(&self, condition: &ClassicalCondition) -> Option<bool> {
        let mut read = Vec::new();
        collect_condition_bits(condition, &mut read);
        if read
            .iter()
            .any(|&bit| bit >= self.written.len() || self.written[bit])
        {
            return None;
        }
        Some(condition.evaluate(&self.zeros))
    }

    fn mark_written(&mut self, body: &[Instruction]) {
        let written = &mut self.written;
        for_each_measure(body, &mut |_, classical_bit| {
            if let Some(slot) = written.get_mut(classical_bit) {
                *slot = true;
            }
        });
    }
}

fn fold_static_guards_into(
    instructions: &[Instruction],
    state: &mut FoldState,
    out: &mut Vec<Instruction>,
) {
    for inst in instructions {
        match inst {
            Instruction::Measure { classical_bit, .. } => {
                if let Some(slot) = state.written.get_mut(*classical_bit) {
                    *slot = true;
                }
                out.push(inst.clone());
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => match state.static_value(condition) {
                Some(true) => {
                    state.folded = true;
                    out.push(Instruction::Gate {
                        gate: gate.clone(),
                        targets: targets.clone(),
                    });
                }
                Some(false) => state.folded = true,
                None => out.push(inst.clone()),
            },
            Instruction::Region(region) => match state.static_value(region.condition()) {
                Some(true) => {
                    state.folded = true;
                    fold_static_guards_into(region.body(), state, out);
                }
                Some(false) => state.folded = true,
                None => {
                    state.mark_written(region.body());
                    out.push(inst.clone());
                }
            },
            Instruction::Gate { .. } | Instruction::Reset { .. } | Instruction::Barrier { .. } => {
                out.push(inst.clone())
            }
        }
    }
}

/// True when a measurement anywhere in `body` writes a bit `condition` reads.
///
/// Lowering `else` and `switch` to a chain of guarded regions is only sound
/// while this is false. Each region in the chain re-evaluates its condition
/// after the earlier bodies have run, so a body that rewrites its own guard bits
/// could take two arms of what the source wrote as one choice.
pub(crate) fn body_writes_condition_bits(
    body: &[Instruction],
    condition: &ClassicalCondition,
) -> bool {
    let mut read = Vec::new();
    collect_condition_bits(condition, &mut read);
    let mut written = false;
    for_each_measure(body, &mut |_, classical_bit| {
        written |= read.contains(&classical_bit)
    });
    written
}

fn collect_condition_bits(condition: &ClassicalCondition, out: &mut Vec<usize>) {
    match condition {
        ClassicalCondition::BitIsOne(bit) | ClassicalCondition::BitIsZero(bit) => out.push(*bit),
        ClassicalCondition::RegisterEquals { offset, size, .. }
        | ClassicalCondition::RegisterNotEquals { offset, size, .. } => {
            out.extend(*offset..offset.saturating_add(*size))
        }
        ClassicalCondition::Parity { bits, .. } => out.extend_from_slice(bits),
    }
}

/// Every classical bit read by a condition anywhere in `body`, nested regions
/// included. A subsystem partition that ignored these would split a circuit
/// whose halves are coupled through the classical bits alone.
fn collect_body_condition_bits(body: &[Instruction], out: &mut Vec<usize>) {
    for inst in body {
        match inst {
            Instruction::Conditional { condition, .. } => collect_condition_bits(condition, out),
            Instruction::Region(region) => {
                collect_condition_bits(region.condition(), out);
                collect_body_condition_bits(region.body(), out);
            }
            _ => {}
        }
    }
}

fn remap_condition(
    condition: &ClassicalCondition,
    cbit: &impl Fn(usize) -> usize,
) -> ClassicalCondition {
    match condition {
        ClassicalCondition::BitIsOne(bit) => ClassicalCondition::BitIsOne(cbit(*bit)),
        ClassicalCondition::BitIsZero(bit) => ClassicalCondition::BitIsZero(cbit(*bit)),
        ClassicalCondition::RegisterEquals {
            offset,
            size,
            value,
        } => ClassicalCondition::RegisterEquals {
            offset: cbit(*offset),
            size: *size,
            value: *value,
        },
        ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        } => ClassicalCondition::RegisterNotEquals {
            offset: cbit(*offset),
            size: *size,
            value: *value,
        },
        ClassicalCondition::Parity { bits, expected } => ClassicalCondition::Parity {
            bits: bits.iter().map(|&bit| cbit(bit)).collect(),
            expected: *expected,
        },
    }
}

/// Rebuild `inst` under a qubit and classical-bit relabeling.
///
/// Callers must have established that every qubit `inst` touches has an image
/// under `qubit`, which for a region means its whole body.
fn remap_instruction(
    inst: &Instruction,
    qubit: &impl Fn(usize) -> usize,
    cbit: &impl Fn(usize) -> usize,
) -> Instruction {
    match inst {
        Instruction::Gate { gate, targets } => Instruction::Gate {
            gate: gate.clone(),
            targets: targets.iter().map(|&t| qubit(t)).collect(),
        },
        Instruction::Measure {
            qubit: q,
            classical_bit,
        } => Instruction::Measure {
            qubit: qubit(*q),
            classical_bit: cbit(*classical_bit),
        },
        Instruction::Reset { qubit: q } => Instruction::Reset { qubit: qubit(*q) },
        Instruction::Barrier { qubits } => Instruction::Barrier {
            qubits: qubits.iter().map(|&q| qubit(q)).collect(),
        },
        Instruction::Conditional {
            condition,
            gate,
            targets,
        } => Instruction::Conditional {
            condition: remap_condition(condition, cbit),
            gate: gate.clone(),
            targets: targets.iter().map(|&t| qubit(t)).collect(),
        },
        Instruction::Region(region) => Instruction::Region(Box::new(GuardedRegion::new(
            remap_condition(region.condition(), cbit),
            region
                .body()
                .iter()
                .map(|inner| remap_instruction(inner, qubit, cbit))
                .collect(),
        ))),
    }
}

fn collect_region_qubits(body: &[Instruction], out: &mut SmallVec<[usize; 4]>) {
    for inst in body {
        match inst {
            Instruction::Gate { targets, .. }
            | Instruction::Conditional { targets, .. }
            | Instruction::Barrier { qubits: targets } => out.extend_from_slice(targets),
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => out.push(*qubit),
            Instruction::Region(inner) => out.extend_from_slice(inner.qubits()),
        }
    }
}

/// Build the guarded form of `body`, or `None` when the body is empty.
///
/// A single-gate body lowers to [`Instruction::Conditional`] so the common
/// `if (c) x q[0];` keeps its allocation-free representation; anything else
/// becomes an [`Instruction::Region`].
pub fn guarded(condition: ClassicalCondition, mut body: Vec<Instruction>) -> Option<Instruction> {
    if body.is_empty() {
        return None;
    }
    if body.len() == 1
        && let Instruction::Gate { .. } = &body[0]
    {
        let Some(Instruction::Gate { gate, targets }) = body.pop() else {
            unreachable!("length and variant both checked above")
        };
        return Some(Instruction::Conditional {
            condition,
            gate,
            targets,
        });
    }
    Some(Instruction::Region(Box::new(GuardedRegion::new(
        condition, body,
    ))))
}

/// A single instruction in the circuit.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Apply a quantum gate to the specified qubits.
    Gate {
        gate: Gate,
        targets: SmallVec<[usize; 4]>,
    },
    /// Measure a qubit, storing the outcome in a classical bit.
    Measure { qubit: usize, classical_bit: usize },
    /// Reset a qubit to |0⟩. Destructive, non-unitary.
    Reset { qubit: usize },
    /// Barrier: scheduling hint, no physical operation.
    /// Backends should treat this as a no-op.
    Barrier { qubits: SmallVec<[usize; 4]> },
    /// Conditionally apply a gate based on classical measurement results.
    ///
    /// The single-gate lowering of [`Instruction::Region`]; see [`guarded`].
    Conditional {
        condition: ClassicalCondition,
        gate: Gate,
        targets: SmallVec<[usize; 4]>,
    },
    /// Conditionally execute a span of instructions, measure and reset included.
    ///
    /// Boxed so the variant costs a pointer: `Instruction` is 96 bytes and this
    /// keeps it there.
    Region(Box<GuardedRegion>),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every backend routes a conditional through ClassicalCondition::evaluate,
    // so the variant semantics are pinned once here rather than per backend.
    // Registers read bit `offset + i` as `1 << i`.
    #[test]
    fn classical_condition_evaluates_every_variant() {
        let bits = [true, false, true];

        assert!(ClassicalCondition::BitIsOne(0).evaluate(&bits));
        assert!(!ClassicalCondition::BitIsOne(1).evaluate(&bits));
        assert!(ClassicalCondition::BitIsZero(1).evaluate(&bits));
        assert!(!ClassicalCondition::BitIsZero(0).evaluate(&bits));

        let reg = |value| ClassicalCondition::RegisterEquals {
            offset: 0,
            size: 3,
            value,
        };
        assert!(reg(0b101).evaluate(&bits));
        assert!(!reg(0b100).evaluate(&bits));

        let reg_ne = |value| ClassicalCondition::RegisterNotEquals {
            offset: 0,
            size: 3,
            value,
        };
        assert!(reg_ne(0b100).evaluate(&bits));
        assert!(!reg_ne(0b101).evaluate(&bits));

        let parity = |list: &[usize], expected| ClassicalCondition::Parity {
            bits: list.to_vec().into(),
            expected,
        };
        assert!(parity(&[0, 1], true).evaluate(&bits), "one bit set is odd");
        assert!(!parity(&[0, 2], true).evaluate(&bits), "two set is even");
        assert!(parity(&[0, 2], false).evaluate(&bits));
        assert!(parity(&[0], true).evaluate(&bits), "one bit is a bit test");
        assert!(
            !parity(&[0, 0], true).evaluate(&bits),
            "a bit cancels itself"
        );
        assert!(!parity(&[], true).evaluate(&bits), "no bits is parity zero");
        assert!(parity(&[], false).evaluate(&bits));
    }

    #[test]
    fn classical_condition_register_honors_offset() {
        let bits = [true, true, false, true];
        let at = |offset| ClassicalCondition::RegisterEquals {
            offset,
            size: 2,
            value: 0b01,
        };
        assert!(at(1).evaluate(&bits), "bits 1..3 are 1,0 so the value is 1");
        assert!(
            !at(0).evaluate(&bits),
            "bits 0..2 are 1,1 so the value is 3"
        );
    }

    #[test]
    fn test_circuit_builder() {
        let mut c = Circuit::new(3, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);

        assert_eq!(c.num_qubits, 3);
        assert_eq!(c.num_classical_bits, 2);
        assert_eq!(c.gate_count(), 3);
        assert_eq!(c.instructions.len(), 5);
    }

    #[test]
    fn test_depth_linear() {
        // H(0), CX(0,1), CX(1,2): depth 3 (serial chain)
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        assert_eq!(c.depth(), 3);
    }

    #[test]
    fn test_depth_parallel() {
        // H(0), H(1), H(2): depth 1 (all parallel)
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::H, &[2]);
        assert_eq!(c.depth(), 1);
    }

    #[test]
    fn test_empty_depth() {
        let c = Circuit::new(4, 0);
        assert_eq!(c.depth(), 0);
    }

    #[test]
    fn test_clifford_prefix_split_basic() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Rx(0.5), &[1]);

        let (prefix, tail) = c.clifford_prefix_split().unwrap();
        assert_eq!(prefix.gate_count(), 2); // H, CX
        assert_eq!(tail.gate_count(), 2); // T, Rx
    }

    #[test]
    fn test_clifford_prefix_split_none_when_all_clifford() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::S, &[0]);
        assert!(c.clifford_prefix_split().is_none());
    }

    #[test]
    fn test_clifford_prefix_split_none_when_first_non_clifford() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);
        assert!(c.clifford_prefix_split().is_none());
    }

    #[test]
    fn test_clifford_prefix_split_stops_at_measure() {
        let mut c = Circuit::new(2, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_gate(Gate::H, &[1]);

        let (prefix, tail) = c.clifford_prefix_split().unwrap();
        assert_eq!(prefix.gate_count(), 2); // H, CX
        assert_eq!(tail.instructions.len(), 2); // measure, H
    }

    #[test]
    fn test_clifford_prefix_split_barrier_transparent() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_barrier(&[0, 1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[0]);

        let (prefix, tail) = c.clifford_prefix_split().unwrap();
        assert_eq!(prefix.instructions.len(), 3); // H, barrier, CX
        assert_eq!(tail.gate_count(), 1); // T
    }

    #[test]
    fn test_subsystems_fully_connected() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::Cx, &[2, 3]);
        let subs = c.independent_subsystems();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_subsystems_disjoint_pairs() {
        let mut c = Circuit::new(6, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::Cx, &[4, 5]);
        let subs = c.independent_subsystems();
        assert_eq!(subs.len(), 3);
        assert_eq!(subs[0], vec![0, 1]);
        assert_eq!(subs[1], vec![2, 3]);
        assert_eq!(subs[2], vec![4, 5]);
    }

    #[test]
    fn test_subsystems_no_entangling() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::Z, &[2]);
        let subs = c.independent_subsystems();
        assert_eq!(subs.len(), 3);
    }

    #[test]
    fn test_subsystems_empty() {
        let c = Circuit::new(0, 0);
        assert!(c.independent_subsystems().is_empty());
    }

    #[test]
    fn test_subsystems_classical_dependency_merges() {
        let mut c = Circuit::new(4, 2);
        // q0-q1 entangled, q2-q3 entangled, but q0 measured and conditional on q2
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_measure(0, 0);
        c.instructions.push(Instruction::Conditional {
            condition: ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: SmallVec::from_slice(&[2]),
        });
        let subs = c.independent_subsystems();
        // All four qubits should be in one group due to classical dependency
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_subsystems_no_classical_dependency() {
        let mut c = Circuit::new(4, 2);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_measure(0, 0);
        c.add_measure(2, 1);
        // No conditionals, subsystems remain independent
        let subs = c.independent_subsystems();
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_extract_subcircuit_basic() {
        let mut c = Circuit::new(4, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_measure(0, 0);
        c.add_measure(2, 1);

        let (sub, q_map, c_map) = c.extract_subcircuit(&[2, 3]);
        assert_eq!(sub.num_qubits, 2);
        assert_eq!(sub.num_classical_bits, 1);
        assert_eq!(sub.gate_count(), 2); // H(0), CX(0,1) remapped
        assert_eq!(sub.instructions.len(), 3); // 2 gates + 1 measure
        assert_eq!(q_map, vec![2, 3]);
        assert_eq!(c_map, vec![1]); // classical bit 1 maps to local 0
    }

    #[test]
    fn test_extract_subcircuit_remaps_indices() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::Cx, &[2, 3]);

        let (sub, _, _) = c.extract_subcircuit(&[2, 3]);
        if let Instruction::Gate { targets, .. } = &sub.instructions[0] {
            assert_eq!(targets.as_slice(), &[0, 1]);
        } else {
            panic!("expected gate instruction");
        }
    }

    #[test]
    #[should_panic(expected = "qubit index 5 out of bounds (circuit has 3 qubits)")]
    fn add_gate_panics_on_out_of_bounds_target_in_release() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[5]);
    }

    #[test]
    #[should_panic(expected = "qubit index 4 out of bounds (circuit has 3 qubits)")]
    fn add_gate_panics_on_second_target_out_of_bounds() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Cx, &[0, 4]);
    }

    #[test]
    #[should_panic(expected = "expects 2 qubits, got 1")]
    fn add_gate_panics_on_arity_mismatch() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Cx, &[0]);
    }

    #[test]
    #[should_panic(expected = "qubit index 2 out of bounds (circuit has 2 qubits)")]
    fn add_measure_panics_on_out_of_bounds_qubit() {
        let mut c = Circuit::new(2, 2);
        c.add_measure(2, 0);
    }

    #[test]
    #[should_panic(expected = "classical bit index 5 out of bounds (circuit has 2 classical bits)")]
    fn add_measure_panics_on_out_of_bounds_classical_bit() {
        let mut c = Circuit::new(2, 2);
        c.add_measure(0, 5);
    }

    #[test]
    #[should_panic(expected = "qubit index 9 out of bounds (circuit has 2 qubits)")]
    fn add_reset_panics_on_out_of_bounds() {
        let mut c = Circuit::new(2, 0);
        c.add_reset(9);
    }

    #[test]
    #[should_panic(expected = "qubit index 7 out of bounds (circuit has 4 qubits)")]
    fn add_barrier_panics_on_out_of_bounds() {
        let mut c = Circuit::new(4, 0);
        c.add_barrier(&[0, 7]);
    }
}
