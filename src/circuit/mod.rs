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

use crate::gates::Gate;
pub use smallvec::{smallvec, SmallVec};

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
    /// # Panics (debug only)
    /// Debug-asserts that all target indices are within bounds and that the
    /// gate arity matches the target count.
    #[inline]
    pub fn add_gate(&mut self, gate: Gate, targets: &[usize]) {
        debug_assert_eq!(
            gate.num_qubits(),
            targets.len(),
            "gate `{}` expects {} qubits, got {}",
            gate.name(),
            gate.num_qubits(),
            targets.len()
        );
        debug_assert!(
            targets.iter().all(|&t| t < self.num_qubits),
            "qubit index out of bounds"
        );
        self.instructions.push(Instruction::Gate {
            gate,
            targets: SmallVec::from_slice(targets),
        });
    }

    /// Append a measurement operation.
    #[inline]
    pub fn add_measure(&mut self, qubit: usize, classical_bit: usize) {
        debug_assert!(qubit < self.num_qubits, "qubit index out of bounds");
        debug_assert!(
            classical_bit < self.num_classical_bits,
            "classical bit index out of bounds"
        );
        self.instructions.push(Instruction::Measure {
            qubit,
            classical_bit,
        });
    }

    /// Append a reset operation, returning the qubit to |0⟩.
    #[inline]
    pub fn add_reset(&mut self, qubit: usize) {
        debug_assert!(qubit < self.num_qubits, "qubit index out of bounds");
        self.instructions.push(Instruction::Reset { qubit });
    }

    /// Append a barrier (scheduling hint, no physical operation).
    #[inline]
    pub fn add_barrier(&mut self, qubits: &[usize]) {
        self.instructions.push(Instruction::Barrier {
            qubits: SmallVec::from_slice(qubits),
        });
    }

    /// Count of gate instructions (excludes measurements and barriers).
    pub fn gate_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Gate { .. } | Instruction::Conditional { .. }
                )
            })
            .count()
    }

    /// Count T and Tdg gates in the circuit.
    pub fn t_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|inst| match inst {
                Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                    matches!(gate, Gate::T | Gate::Tdg)
                }
                _ => false,
            })
            .count()
    }

    /// Returns true if the circuit contains any T or Tdg gates.
    pub fn has_t_gates(&self) -> bool {
        self.instructions.iter().any(|inst| match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                matches!(gate, Gate::T | Gate::Tdg)
            }
            _ => false,
        })
    }

    /// True if every gate in the circuit is a Clifford gate.
    ///
    /// When true, the stabilizer backend can simulate this circuit exactly
    /// in O(n^2) time regardless of qubit count.
    pub fn is_clifford_only(&self) -> bool {
        self.instructions.iter().all(|inst| match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                gate.is_clifford()
            }
            _ => true,
        })
    }

    /// True if every gate is Clifford or T/Tdg.
    pub fn is_clifford_plus_t(&self) -> bool {
        self.instructions.iter().all(|inst| match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                gate.is_clifford() || matches!(gate, Gate::T | Gate::Tdg)
            }
            _ => true,
        })
    }

    /// True if every gate preserves computational basis states (diagonal or
    /// permutation). When true, the sparse backend is optimal — the state
    /// always has exactly one non-zero amplitude, giving O(1) memory and O(n)
    /// per-gate cost regardless of qubit count.
    pub fn is_sparse_friendly(&self) -> bool {
        self.instructions.iter().all(|inst| match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                gate.preserves_sparsity()
            }
            _ => true,
        })
    }

    /// True if the circuit contains any multi-qubit (entangling) gates.
    ///
    /// When false, the product state backend can simulate in O(n) time.
    pub fn has_entangling_gates(&self) -> bool {
        self.instructions.iter().any(|inst| match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                gate.num_qubits() >= 2
            }
            _ => false,
        })
    }

    /// Herfindahl–Hirschman index of the qubit interaction graph partition.
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
                Instruction::Conditional { .. } => return false,
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

    /// Extract the qubit-to-classical-bit mapping from all measurements.
    pub fn measurement_map(&self) -> Vec<(usize, usize)> {
        self.instructions
            .iter()
            .filter_map(|inst| match inst {
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => Some((*qubit, *classical_bit)),
                _ => None,
            })
            .collect()
    }

    /// Return a copy of this circuit with all measurement instructions removed.
    pub fn without_measurements(&self) -> Circuit {
        let mut c = Circuit::new(self.num_qubits, self.num_classical_bits);
        c.instructions = self
            .instructions
            .iter()
            .filter(|inst| !matches!(inst, Instruction::Measure { .. }))
            .cloned()
            .collect();
        c
    }

    /// True if the circuit contains any reset instruction.
    pub fn has_resets(&self) -> bool {
        self.instructions
            .iter()
            .any(|i| matches!(i, Instruction::Reset { .. }))
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
        for inst in &self.instructions {
            if let Instruction::Measure {
                qubit,
                classical_bit,
            } = inst
            {
                cbit_to_qubit[*classical_bit] = Some(*qubit);
            }
        }

        for inst in &self.instructions {
            let targets = match inst {
                Instruction::Gate { targets, .. } => targets,
                Instruction::Conditional {
                    condition, targets, ..
                } => {
                    match condition {
                        ClassicalCondition::BitIsOne(bit) => {
                            if let Some(mq) = cbit_to_qubit[*bit] {
                                union(&mut parent, &mut rank, targets[0], mq);
                            }
                        }
                        ClassicalCondition::RegisterEquals { offset, size, .. } => {
                            for mq in cbit_to_qubit.iter().skip(*offset).take(*size).flatten() {
                                union(&mut parent, &mut rank, targets[0], *mq);
                            }
                        }
                    }
                    targets
                }
                _ => continue,
            };
            if targets.len() >= 2 {
                let first = targets[0];
                for &t in &targets[1..] {
                    union(&mut parent, &mut rank, first, t);
                }
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

        for inst in &self.instructions {
            if let Instruction::Measure {
                qubit,
                classical_bit,
            } = inst
            {
                if old_to_new_qubit[*qubit].is_some()
                    && old_to_new_classical[*classical_bit].is_none()
                {
                    let new_idx = classical_bits_used.len();
                    old_to_new_classical[*classical_bit] = Some(new_idx);
                    classical_bits_used.push(*classical_bit);
                }
            }
        }

        let mut sub = Circuit::new(qubit_set.len(), classical_bits_used.len());

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
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } => {
                    if targets.iter().all(|&t| old_to_new_qubit[t].is_some()) {
                        let new_targets: SmallVec<[usize; 4]> = targets
                            .iter()
                            .map(|&t| old_to_new_qubit[t].unwrap())
                            .collect();
                        let new_condition = match condition {
                            ClassicalCondition::BitIsOne(bit) => ClassicalCondition::BitIsOne(
                                old_to_new_classical[*bit].unwrap_or(*bit),
                            ),
                            ClassicalCondition::RegisterEquals {
                                offset,
                                size,
                                value,
                            } => ClassicalCondition::RegisterEquals {
                                offset: old_to_new_classical[*offset].unwrap_or(*offset),
                                size: *size,
                                value: *value,
                            },
                        };
                        sub.instructions.push(Instruction::Conditional {
                            condition: new_condition,
                            gate: gate.clone(),
                            targets: new_targets,
                        });
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

        for inst in &self.instructions {
            if let Instruction::Measure {
                qubit,
                classical_bit,
            } = inst
            {
                let (comp_idx, _) = qubit_map[*qubit];
                if cbit_map[*classical_bit].is_none() {
                    let new_idx = classical_bits_per_comp[comp_idx].len();
                    cbit_map[*classical_bit] = Some((comp_idx, new_idx));
                    classical_bits_per_comp[comp_idx].push(*classical_bit);
                }
            }
        }

        let mut subs: Vec<Circuit> = (0..k)
            .map(|i| Circuit::new(components[i].len(), classical_bits_per_comp[i].len()))
            .collect();

        let mut barrier_buf: Vec<SmallVec<[usize; 4]>> = (0..k).map(|_| SmallVec::new()).collect();

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
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } => {
                    let (comp_idx, _) = qubit_map[targets[0]];
                    let new_targets: SmallVec<[usize; 4]> =
                        targets.iter().map(|&t| qubit_map[t].1).collect();
                    let new_condition = match condition {
                        ClassicalCondition::BitIsOne(bit) => {
                            let (_, nc) = cbit_map[*bit].unwrap_or((comp_idx, *bit));
                            ClassicalCondition::BitIsOne(nc)
                        }
                        ClassicalCondition::RegisterEquals {
                            offset,
                            size,
                            value,
                        } => {
                            let new_offset = cbit_map[*offset].map(|(_, nc)| nc).unwrap_or(*offset);
                            ClassicalCondition::RegisterEquals {
                                offset: new_offset,
                                size: *size,
                                value: *value,
                            }
                        }
                    };
                    subs[comp_idx].instructions.push(Instruction::Conditional {
                        condition: new_condition,
                        gate: gate.clone(),
                        targets: new_targets,
                    });
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
                | Instruction::Conditional { .. } => {
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

    /// Circuit depth via greedy layer assignment.
    ///
    /// Each gate occupies the earliest layer where all its qubits are free.
    /// Measurements count as depth-1 operations. Barriers synchronize qubits
    /// to the same layer without adding depth.
    pub fn depth(&self) -> usize {
        if self.instructions.is_empty() {
            return 0;
        }
        let mut qubit_depth = vec![0usize; self.num_qubits];
        for inst in &self.instructions {
            match inst {
                Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => {
                    let max_d = targets.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                    for &q in targets {
                        qubit_depth[q] = max_d + 1;
                    }
                }
                Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                    qubit_depth[*qubit] += 1;
                }
                Instruction::Barrier { qubits } => {
                    let max_d = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                    for &q in qubits {
                        qubit_depth[q] = max_d;
                    }
                }
            }
        }
        qubit_depth.into_iter().max().unwrap_or(0)
    }
}

/// Condition for classically-controlled gate execution.
#[derive(Debug, Clone)]
pub enum ClassicalCondition {
    /// True when the classical bit at `bit` is 1.
    BitIsOne(usize),
    /// True when the classical register (bits `offset..offset+size`) equals `value`.
    RegisterEquals {
        offset: usize,
        size: usize,
        value: u64,
    },
}

impl ClassicalCondition {
    /// Evaluate this condition against a classical bit array.
    pub fn evaluate(&self, classical_bits: &[bool]) -> bool {
        match self {
            ClassicalCondition::BitIsOne(bit) => classical_bits[*bit],
            ClassicalCondition::RegisterEquals {
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
                reg_val == *value
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
    /// Reset a qubit to |0⟩. Destructive, non-unitary.
    Reset { qubit: usize },
    /// Barrier — scheduling hint, no physical operation.
    /// Backends should treat this as a no-op.
    Barrier { qubits: SmallVec<[usize; 4]> },
    /// Conditionally apply a gate based on classical measurement results.
    Conditional {
        condition: ClassicalCondition,
        gate: Gate,
        targets: SmallVec<[usize; 4]>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // H(0), CX(0,1), CX(1,2) — depth 3 (serial chain)
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        assert_eq!(c.depth(), 3);
    }

    #[test]
    fn test_depth_parallel() {
        // H(0), H(1), H(2) — depth 1 (all parallel)
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
        // No conditionals — subsystems remain independent
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
}
