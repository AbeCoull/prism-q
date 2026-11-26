//! Backend-agnostic circuit IR that every frontend lowers into.
//!
//! Targets are `SmallVec<[usize; 4]>`, inline up to four qubits. Optimization passes
//! rewrite the instruction stream through [`fusion::fuse_circuit`].

pub mod builder;
pub(crate) mod clifford_t;
mod draw;
pub use draw::TextOptions;
mod svg;
pub use svg::SvgOptions;
pub mod braket;
pub mod dynamic;
#[doc(hidden)]
pub mod fusion;
mod fusion_phase;
mod fusion_rzz;
pub(crate) mod fusion_stage;
pub(crate) mod gate_library;
pub mod openqasm;
pub mod parameter;
pub(crate) mod parity_ancilla;
mod pauli_lowering;
pub(crate) mod plan;
pub mod prepared;
pub(crate) mod qasm;
pub mod qasm_export;
mod qft_recognition;
mod region;
pub(crate) mod synthesis;
mod trotter;

pub use parameter::{ParamLink, Parameters};
pub use pauli_lowering::expand_pauli_rotations;
pub(crate) use pauli_lowering::{
    GateSink, append_axis_to_z_rotation, append_parity_rotations, append_z_to_axis_rotation,
    pauli_rotation_gate, pauli_rotation_lowering,
};
pub use prepared::PreparedCircuit;
pub use qft_recognition::{
    QftTextbookStep, expand_qft_blocks, qft_textbook_steps, recognize_qft_blocks,
};
use region::remap_gate;
pub use region::{ClassicalCondition, GuardedRegion, MAX_REGION_DEPTH, guarded};
use region::{
    FoldState, any_instruction, collect_body_condition_bits, collect_condition_bits,
    fold_static_guards_into, for_each_gate, for_each_measure, guarded_noise_slots, mark_coupled,
    strip_measurements, visit_noise_slots,
};
pub(crate) use region::{any_gate, body_writes_condition_bits, remap_instruction};
pub use trotter::TrotterOrder;

use crate::gates::Gate;
use crate::sim::unified_pauli::PauliTerm;
use num_complex::Complex64;
pub use smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// A quantum circuit in PRISM-Q's internal representation.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub num_qubits: usize,
    pub num_classical_bits: usize,
    pub instructions: Vec<Instruction>,
}

/// Component views of one circuit, each group sorted and the groups ordered by
/// their lowest qubit.
pub(crate) struct Subsystems {
    /// What [`Circuit::independent_subsystems`] returns.
    pub(crate) interaction: Vec<Vec<usize>>,
    /// `interaction` with every qubit measured into one classical bit joined, so
    /// no bit a measurement writes is written or read from two groups. `None`
    /// when no bit has two writers and the views coincide.
    pub(crate) closed: Option<Vec<Vec<usize>>>,
    pub(crate) clifford_only: bool,
}

impl Circuit {
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
    /// Panics if targets repeat, fall outside the circuit, or disagree with the gate's
    /// arity or embedded qubits.
    #[inline]
    pub fn add_gate(&mut self, gate: Gate, targets: &[usize]) {
        gate.validate_targets(targets, self.num_qubits)
            .unwrap_or_else(|err| panic!("{err}"));
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

    /// Append the `2^k x 2^k` unitary `mat` on the `k` qubits `targets`.
    ///
    /// Lowers the matrix the way [`Gate::unitary`] does, and also lowers a
    /// diagonal whose phases factor into one and two body terms to a
    /// [`Gate::DiagonalBatch`], which needs the targets the constructor lacks.
    /// A diagonal that does not factor stays dense (a `CCZ` reaches `Mcu`
    /// first; a general three-body phase does not).
    ///
    /// # Errors
    /// The errors of [`Gate::unitary`], with `k` read from `targets.len()`.
    ///
    /// # Panics
    /// Panics if any target index is out of bounds.
    pub fn add_unitary(
        &mut self,
        mat: Vec<Complex64>,
        targets: &[usize],
    ) -> crate::error::Result<()> {
        let gate = match Gate::unitary(mat, targets.len())? {
            Gate::Fused2q(m) => {
                let flat: Vec<Complex64> = m.iter().flat_map(|row| row.iter().copied()).collect();
                crate::gates::diagonal_batch(&flat, targets).unwrap_or(Gate::Fused2q(m))
            }
            Gate::Unitary(data) => {
                crate::gates::diagonal_batch(data.matrix(), targets).unwrap_or(Gate::Unitary(data))
            }
            other => other,
        };
        self.add_gate(gate, targets);
        Ok(())
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

    /// Append a save point recording `spec` under `label`.
    ///
    /// A save observes the whole register, so it is a fusion barrier across
    /// every qubit: no gate moves across it in either direction. Labels are not
    /// required to be unique; records come back in the order their points were
    /// reached. A backend that cannot produce `spec` fails the run rather than
    /// omitting the record.
    pub fn add_save(&mut self, spec: SaveSpec, label: impl Into<String>) {
        self.instructions.push(Instruction::Save {
            spec,
            qubits: (0..self.num_qubits).collect(),
            label: label.into(),
        });
    }

    pub fn save_count(&self) -> usize {
        self.save_points().count()
    }

    pub(crate) fn save_points(&self) -> impl Iterator<Item = (SaveSpec, &str)> {
        self.instructions.iter().filter_map(|inst| match inst {
            Instruction::Save { spec, label, .. } => Some((*spec, label.as_str())),
            _ => None,
        })
    }

    /// Count of gate instructions (excludes measurements and barriers).
    pub fn gate_count(&self) -> usize {
        let mut count = 0;
        for_each_gate(&self.instructions, &mut |_| count += 1);
        count
    }

    /// Count of T gates after Clifford+T lowering: `T` and `Tdg`, plus one per
    /// rotation at an odd multiple of pi/4 among the `Rz`, `P`, `Rzz`, `Rx`,
    /// `Ry`, `PauliRot`, `Fused`, and `Cu` forms that lower into the set. Gates
    /// outside the set count zero.
    pub fn t_count(&self) -> usize {
        let mut count = 0;
        for_each_gate(&self.instructions, &mut |gate| {
            count += clifford_t::clifford_t_count(gate).unwrap_or(0);
        });
        count
    }

    /// True if any gate lowers to a `T` (see [`Self::t_count`]).
    pub fn has_t_gates(&self) -> bool {
        any_gate(&self.instructions, &mut |gate| {
            clifford_t::clifford_t_count(gate).is_some_and(|t| t > 0)
        })
    }

    /// True if every gate, guarded ones included, is Clifford, so the stabilizer
    /// backend simulates it exactly in O(n^2) per gate at any width.
    pub fn is_clifford_only(&self) -> bool {
        !any_gate(&self.instructions, &mut |gate| !gate.is_clifford())
    }

    /// True if every gate is Clifford or lowers to Clifford+T: `T`, `Tdg`, and
    /// the rotation, `Fused`, and `Cu` forms whose angles sit on the pi/4 grid
    /// up to a global phase (see [`Self::t_count`]).
    pub fn is_clifford_plus_t(&self) -> bool {
        !any_gate(&self.instructions, &mut |gate| {
            clifford_t::clifford_t_count(gate).is_none()
        })
    }

    /// True if every gate is diagonal or a permutation, so a basis state keeps a
    /// single nonzero amplitude throughout: O(1) memory and O(n) per gate on the
    /// sparse backend.
    pub fn is_sparse_friendly(&self) -> bool {
        !any_gate(&self.instructions, &mut |gate| !gate.preserves_sparsity())
    }

    /// True if any gate touches two or more qubits. When false, the product state
    /// backend runs in O(n).
    pub fn has_entangling_gates(&self) -> bool {
        any_gate(&self.instructions, &mut |gate| gate.num_qubits() >= 2)
    }

    /// True if every shot can be sampled from one evolved pure state: no gate,
    /// reset, or conditional follows a measurement, and every reset acts on a
    /// qubit no multi-qubit gate has touched, so from `|0...0>` it is still a
    /// product factor and the reset is deterministic.
    ///
    /// A reset on an entangled qubit is a measurement in disguise, and each
    /// shot must draw its own outcome. A start state other than `|0...0>`
    /// voids the product argument, so callers holding one also check
    /// [`Self::has_resets`].
    pub fn has_terminal_measurements_only(&self) -> bool {
        self.terminal_scan(false)
    }

    /// [`Self::has_terminal_measurements_only`] with every reset before the
    /// first measurement accepted, which holds where reset is the channel on a
    /// mixed state rather than one sampled branch.
    pub(crate) fn has_terminal_measurements_under_reset_channel(&self) -> bool {
        self.terminal_scan(true)
    }

    fn terminal_scan(&self, reset_is_channel: bool) -> bool {
        let mut seen_measurement = false;
        // Built on the first reset only, so a reset-free circuit allocates nothing.
        let mut coupled: Option<Vec<u64>> = None;
        for (index, inst) in self.instructions.iter().enumerate() {
            match inst {
                Instruction::Conditional { .. } | Instruction::Region(_) => return false,
                Instruction::Measure { .. } => {
                    seen_measurement = true;
                }
                Instruction::Gate { targets, .. } => {
                    if seen_measurement {
                        return false;
                    }
                    if let Some(coupled) = coupled.as_mut() {
                        mark_coupled(coupled, targets);
                    }
                }
                Instruction::Reset { qubit } => {
                    if seen_measurement {
                        return false;
                    }
                    if reset_is_channel {
                        continue;
                    }
                    let coupled = coupled.get_or_insert_with(|| {
                        let mut words = vec![0u64; self.num_qubits.div_ceil(64)];
                        for prior in &self.instructions[..index] {
                            if let Instruction::Gate { targets, .. } = prior {
                                mark_coupled(&mut words, targets);
                            }
                        }
                        words
                    });
                    if (coupled[qubit / 64] >> (qubit % 64)) & 1 == 1 {
                        return false;
                    }
                }
                Instruction::Barrier { .. } | Instruction::Save { .. } => {}
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

    /// Which classical bits a condition reads, region bodies included, one flag per bit.
    pub(crate) fn condition_bits(&self) -> Vec<bool> {
        let mut read = vec![false; self.num_classical_bits];
        any_instruction(&self.instructions, &mut |inst| {
            let condition = match inst {
                Instruction::Conditional { condition, .. } => condition,
                Instruction::Region(region) => region.condition(),
                _ => return false,
            };
            condition.for_each_bit(|bit| {
                if let Some(flag) = read.get_mut(bit) {
                    *flag = true;
                }
            });
            false
        });
        read
    }

    /// Classical bit each measurement writes, in record order; the bit half of
    /// [`Circuit::measurement_map`], with the same region descent.
    pub(crate) fn classical_bit_order(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for_each_measure(&self.instructions, &mut |_, classical_bit| {
            out.push(classical_bit)
        });
        out
    }

    pub fn without_measurements(&self) -> Circuit {
        let mut c = Circuit::new(self.num_qubits, self.num_classical_bits);
        c.instructions = strip_measurements(&self.instructions);
        c
    }

    /// Length of [`NoiseModel::after_gate`](crate::sim::noise::NoiseModel::after_gate)
    /// for this circuit: one slot per top-level instruction, then the slots of
    /// guarded bodies.
    pub fn noise_slot_count(&self) -> usize {
        self.instructions.len() + guarded_noise_slots(&self.instructions)
    }

    /// Visit every instruction with its noise slot in the layout
    /// [`Circuit::noise_slot_count`] counts.
    pub(crate) fn for_each_noise_slot(&self, mut visit: impl FnMut(usize, &Instruction)) {
        visit_noise_slots(&self.instructions, 0, self.instructions.len(), &mut visit);
    }

    /// Noise slots the guards in `instructions[..from]` take, which puts the
    /// first guarded body at or after `from` at `instructions.len()` plus this.
    pub(crate) fn guarded_noise_slots_before(&self, from: usize) -> usize {
        guarded_noise_slots(&self.instructions[..from])
    }

    pub fn has_resets(&self) -> bool {
        any_instruction(&self.instructions, &mut |i| {
            matches!(i, Instruction::Reset { .. })
        })
    }

    /// Partition qubits into independent subsystems.
    ///
    /// Two qubits share a subsystem if a multi-qubit gate connects them,
    /// transitively, or if a condition on one's measurement guards the other.
    /// Each group is sorted, and groups are ordered by their lowest qubit.
    pub fn independent_subsystems(&self) -> Vec<Vec<usize>> {
        self.subsystems_and_clifford().interaction
    }

    /// Both component views and [`Circuit::is_clifford_only`] from one pass over
    /// the stream.
    pub(crate) fn subsystems_and_clifford(&self) -> Subsystems {
        let n = self.num_qubits;
        if n == 0 {
            return Subsystems {
                interaction: Vec::new(),
                closed: None,
                clifford_only: self.is_clifford_only(),
            };
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

        // Qubits arrive in ascending order, so each group opens at its lowest qubit
        // and the groups come out sorted by it.
        fn groups(parent: &mut [usize]) -> Vec<Vec<usize>> {
            let mut group_of = vec![usize::MAX; parent.len()];
            let mut result: Vec<Vec<usize>> = Vec::new();
            for q in 0..parent.len() {
                let root = find(parent, q);
                if group_of[root] == usize::MAX {
                    group_of[root] = result.len();
                    result.push(Vec::new());
                }
                result[group_of[root]].push(q);
            }
            result
        }

        // A condition joins its targets to the last qubit measured into each bit
        // it reads, anywhere in the circuit, so those unions wait for the loop to
        // end. `rewrites` pairs each later writer of a bit with the one before it.
        let mut clifford_only = true;
        let mut condition_bits: Vec<usize> = Vec::new();
        let mut guarded: Vec<(usize, usize)> = Vec::new();
        let mut writers: Vec<Option<usize>> = Vec::new();
        let mut rewrites: Vec<(usize, usize)> = Vec::new();
        let mut note_writer = |qubit: usize, classical_bit: usize| {
            if writers.is_empty() {
                writers.resize(self.num_classical_bits, None);
            }
            if let Some(slot) = writers.get_mut(classical_bit)
                && let Some(earlier) = slot.replace(qubit)
                && earlier != qubit
            {
                rewrites.push((earlier, qubit));
            }
        };
        for inst in &self.instructions {
            condition_bits.clear();
            let targets = match inst {
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    note_writer(*qubit, *classical_bit);
                    continue;
                }
                Instruction::Gate { gate, targets } => {
                    clifford_only &= gate.is_clifford();
                    targets.as_slice()
                }
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } => {
                    clifford_only &= gate.is_clifford();
                    collect_condition_bits(condition, &mut condition_bits);
                    targets.as_slice()
                }
                Instruction::Region(region) => {
                    clifford_only =
                        clifford_only && !any_gate(region.body(), &mut |gate| !gate.is_clifford());
                    collect_condition_bits(region.condition(), &mut condition_bits);
                    collect_body_condition_bits(region.body(), &mut condition_bits);
                    for_each_measure(region.body(), &mut note_writer);
                    region.qubits()
                }
                _ => continue,
            };
            let Some(&first) = targets.first() else {
                continue;
            };
            guarded.extend(condition_bits.iter().map(|&bit| (first, bit)));
            for &t in &targets[1..] {
                union(&mut parent, &mut rank, first, t);
            }
        }

        for (first, bit) in guarded {
            if let Some(writer) = writers.get(bit).copied().flatten() {
                union(&mut parent, &mut rank, first, writer);
            }
        }

        let interaction = groups(&mut parent);
        let closed = (!rewrites.is_empty()).then(|| {
            for (earlier, later) in rewrites {
                union(&mut parent, &mut rank, earlier, later);
            }
            groups(&mut parent)
        });
        Subsystems {
            interaction,
            closed,
            clifford_only,
        }
    }

    /// Split the circuit into one sub-circuit per component in two passes over the
    /// instruction stream.
    ///
    /// `components` must cover every qubit with no instruction spanning two of them,
    /// as [`Circuit::independent_subsystems`] returns. A caller that merges the
    /// blocks' classical bits passes [`Subsystems::closed`]. Each entry is
    /// `(sub_circuit, qubit_map, classical_map)` with `map[local] = original`.
    /// A block's classical register holds the bits its measurements write and its
    /// conditions read, in ascending order, so a register condition keeps its
    /// contiguous run.
    pub(crate) fn partition_subcircuits(
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

        // Pass 1: discover classical bits per component. A bit claimed by
        // several components gets a local index in each; the first claim stays
        // in the flat map and later claims go to `shared_cbits`. Local indices
        // are filled in once each component's bits are sorted.
        let mut classical_bits_per_comp: Vec<Vec<usize>> = vec![Vec::new(); k];
        let max_cb = self.num_classical_bits.max(1);
        let mut cbit_map: Vec<Option<(usize, usize)>> = vec![None; max_cb];
        let mut shared_cbits: HashMap<(usize, usize), usize> = HashMap::new();

        let mut claim = |comp_idx: usize, classical_bit: usize| {
            match cbit_map[classical_bit] {
                None => cbit_map[classical_bit] = Some((comp_idx, 0)),
                Some((owner, _)) if owner == comp_idx => return,
                Some(_) => match shared_cbits.entry((comp_idx, classical_bit)) {
                    Entry::Occupied(_) => return,
                    Entry::Vacant(slot) => {
                        slot.insert(0);
                    }
                },
            }
            classical_bits_per_comp[comp_idx].push(classical_bit);
        };
        let mut region_reads: Vec<usize> = Vec::new();
        for inst in &self.instructions {
            match inst {
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => claim(qubit_map[*qubit].0, *classical_bit),
                Instruction::Conditional {
                    condition, targets, ..
                } => {
                    let (comp_idx, _) = qubit_map[targets[0]];
                    condition.for_each_bit(|bit| claim(comp_idx, bit));
                }
                Instruction::Region(region) => {
                    let Some(&first) = region.qubits().first() else {
                        continue;
                    };
                    let (comp_idx, _) = qubit_map[first];
                    for_each_measure(region.body(), &mut |_, bit| claim(comp_idx, bit));
                    region_reads.clear();
                    collect_condition_bits(region.condition(), &mut region_reads);
                    collect_body_condition_bits(region.body(), &mut region_reads);
                    for &bit in &region_reads {
                        claim(comp_idx, bit);
                    }
                }
                Instruction::Gate { .. }
                | Instruction::Reset { .. }
                | Instruction::Barrier { .. }
                | Instruction::Save { .. } => {}
            }
        }
        for (comp_idx, bits) in classical_bits_per_comp.iter_mut().enumerate() {
            bits.sort_unstable();
            for (local, &classical_bit) in bits.iter().enumerate() {
                match &mut cbit_map[classical_bit] {
                    Some((owner, slot)) if *owner == comp_idx => *slot = local,
                    _ => {
                        shared_cbits.insert((comp_idx, classical_bit), local);
                    }
                }
            }
        }

        let mut subs: Vec<Circuit> = (0..k)
            .map(|i| Circuit::new(components[i].len(), classical_bits_per_comp[i].len()))
            .collect();

        let mut barrier_buf: Vec<SmallVec<[usize; 4]>> = (0..k).map(|_| SmallVec::new()).collect();

        let local_cbit = |comp_idx: usize, c: usize| match cbit_map[c] {
            Some((owner, nc)) if owner == comp_idx => Some(nc),
            Some(_) => shared_cbits.get(&(comp_idx, c)).copied(),
            None => None,
        };
        let qubit_of = |q: usize| qubit_map[q].1;

        // Pass 2: route each instruction to its component
        for inst in &self.instructions {
            match inst {
                Instruction::Gate { gate, targets } => {
                    let (comp_idx, _) = qubit_map[targets[0]];
                    let new_targets: SmallVec<[usize; 4]> =
                        targets.iter().map(|&t| qubit_map[t].1).collect();
                    subs[comp_idx].instructions.push(Instruction::Gate {
                        gate: remap_gate(gate, &qubit_of),
                        targets: new_targets,
                    });
                }
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    let (comp_idx, nq) = qubit_map[*qubit];
                    if let Some(nc) = local_cbit(comp_idx, *classical_bit) {
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
                Instruction::Save { .. } => {}
                Instruction::Conditional { targets, .. } => {
                    let (comp_idx, _) = qubit_map[targets[0]];
                    let cbit_of = |c: usize| {
                        local_cbit(comp_idx, c).expect("pass 1 claims every bit a condition reads")
                    };
                    subs[comp_idx]
                        .instructions
                        .push(remap_instruction(inst, &qubit_of, &cbit_of));
                }
                Instruction::Region(region) => {
                    let Some(&first) = region.qubits().first() else {
                        continue;
                    };
                    let (comp_idx, _) = qubit_map[first];
                    let cbit_of = |c: usize| {
                        local_cbit(comp_idx, c).expect("pass 1 claims every bit a condition reads")
                    };
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

    /// Split the circuit into a Clifford prefix and the rest.
    ///
    /// The prefix holds only Clifford gates and barriers. The tail starts at the
    /// first non-Clifford gate, measurement, reset, conditional, region, or save.
    /// Returns `None` when either part would be empty.
    pub(crate) fn clifford_prefix_split(&self) -> Option<(Circuit, Circuit)> {
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
                | Instruction::Region(_)
                | Instruction::Save { .. } => {
                    split_at = i;
                    break;
                }
                Instruction::Barrier { .. } => {}
            }
            split_at = i + 1;
        }

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
            Instruction::Barrier { qubits } | Instruction::Save { qubits, .. } => {
                let d = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                visit(inst, d);
                for &q in qubits.iter() {
                    qubit_depth[q] = d;
                }
            }
        }
    }
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
    /// Reset a qubit to |0⟩.
    Reset { qubit: usize },
    /// Scheduling hint with no physical effect.
    Barrier { qubits: SmallVec<[usize; 4]> },
    /// Single-gate form of [`Instruction::Region`]; see [`guarded`].
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
    /// Record the state as it stands at this point in the circuit.
    ///
    /// `qubits` lists every qubit the save observes, which is the whole
    /// register for the dense forms. Fusion reads it the same way it reads a
    /// barrier's, so no gate crosses a save point.
    Save {
        spec: SaveSpec,
        qubits: SmallVec<[usize; 4]>,
        label: String,
    },
}

impl Instruction {
    /// Qubits the instruction touches: a barrier's and a save's included, a
    /// region's body union.
    #[inline]
    pub(crate) fn qubits(&self) -> &[usize] {
        match self {
            Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => targets,
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                std::slice::from_ref(qubit)
            }
            Instruction::Barrier { qubits } | Instruction::Save { qubits, .. } => qubits,
            Instruction::Region(region) => region.qubits(),
        }
    }
}

/// What a [`Instruction::Save`] point records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SaveSpec {
    /// Amplitudes, `2^n` of them. Declined by backends that hold no dense
    /// statevector.
    StateVector,
    /// Probability of each basis state, `2^n` of them.
    Probabilities,
    /// The full density matrix, `2^n` by `2^n`, row major. Only the density
    /// matrix backend holds one.
    DensityMatrix,
}

/// One recorded save, in the order its point was reached.
#[derive(Debug, Clone, PartialEq)]
pub struct SaveRecord {
    /// The label given at [`Circuit::add_save`].
    pub label: String,
    pub value: SavedValue,
}

/// The state a save point recorded.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SavedValue {
    StateVector(Vec<Complex64>),
    Probabilities(Vec<f64>),
    /// Row major, `2^n` rows of `2^n`.
    DensityMatrix(Vec<Complex64>),
}

#[cfg(test)]
#[path = "circuit_tests.rs"]
mod tests;
