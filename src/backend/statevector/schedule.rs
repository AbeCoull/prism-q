use num_complex::Complex64;

use crate::circuit::Instruction;
use crate::gates::{
    DiagEntry, Gate, is_diagonal_2x2, is_diagonal_4x4, multi_2q_high_budget_for, multi_2q_low_bits,
    pauli_rot_masks,
};

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(super) enum Action {
    One(usize, [[Complex64; 2]; 2]),
    Two(usize, usize, [[Complex64; 4]; 4]),
    Cx(usize, usize),
    Swap(usize, usize),
    Diagonal(usize, usize, [Complex64; 4]),
    Pauli(usize, usize, f64),
    Opaque(usize),
}

pub(super) fn one_gates(
    actions: &[Action],
    indices: &[usize],
) -> Vec<(usize, [[Complex64; 2]; 2])> {
    let mut gates: Vec<(usize, [[Complex64; 2]; 2])> = Vec::new();
    for &i in indices {
        let Action::One(q, matrix) = actions[i] else {
            unreachable!()
        };
        if let Some((_, prior)) = gates.iter_mut().find(|(target, _)| *target == q) {
            *prior = crate::gates::mat_mul_2x2(&matrix, prior);
        } else {
            gates.push((q, matrix));
        }
    }
    gates
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    touched: u64,
    moved: u64,
    kind: u8,
}

impl Action {
    fn shape(&self) -> Shape {
        let (touched, moved, kind) = match self {
            Self::One(q, _) => (1 << q, 1 << q, 0),
            Self::Two(a, b, _) => (1 << a | 1 << b, 1 << a | 1 << b, 1),
            Self::Cx(a, b) => (1 << a | 1 << b, 1 << b, 5),
            Self::Swap(a, b) => (1 << a | 1 << b, 1 << a | 1 << b, 6),
            Self::Diagonal(a, b, _) => (1 << a | 1 << b, 0, 2),
            Self::Pauli(x, z, _) => ((*x | *z) as u64, *x as u64, 3),
            Self::Opaque(i) => (*i as u64, 0, 4),
        };
        Shape {
            touched,
            moved,
            kind,
        }
    }

    fn conjugated(&self) -> Self {
        match self {
            Self::One(q, m) => Self::One(*q, m.map(|row| row.map(|z| z.conj()))),
            Self::Two(a, b, m) => Self::Two(*a, *b, m.map(|row| row.map(|z| z.conj()))),
            Self::Diagonal(a, b, d) => Self::Diagonal(*a, *b, d.map(|z| z.conj())),
            Self::Pauli(x, z, theta) => Self::Pauli(
                *x,
                *z,
                if (x & z).count_ones() % 2 == 0 {
                    -*theta
                } else {
                    *theta
                },
            ),
            Self::Cx(a, b) => Self::Cx(*a, *b),
            Self::Swap(a, b) => Self::Swap(*a, *b),
            Self::Opaque(_) => unreachable!(),
        }
    }

    fn shifted(&self, n: usize) -> Self {
        match self {
            Self::One(q, m) => Self::One(q + n, *m),
            Self::Two(a, b, m) => Self::Two(a + n, b + n, *m),
            Self::Diagonal(a, b, d) => Self::Diagonal(a + n, b + n, *d),
            Self::Pauli(x, z, theta) => Self::Pauli(x << n, z << n, *theta),
            Self::Cx(a, b) => Self::Cx(a + n, b + n),
            Self::Swap(a, b) => Self::Swap(a + n, b + n),
            Self::Opaque(_) => unreachable!(),
        }
    }
}

fn one(out: &mut Vec<Action>, q: usize, m: [[Complex64; 2]; 2]) {
    if is_diagonal_2x2(&m) {
        out.push(Action::Diagonal(q, q, [m[0][0], m[0][0], m[1][1], m[1][1]]));
    } else {
        out.push(Action::One(q, m));
    }
}

fn two(out: &mut Vec<Action>, a: usize, b: usize, m: [[Complex64; 4]; 4]) {
    if is_diagonal_4x4(&m) {
        out.push(Action::Diagonal(a, b, [m[0][0], m[1][1], m[2][2], m[3][3]]));
    } else {
        out.push(Action::Two(a, b, m));
    }
}

fn diagonal(out: &mut Vec<Action>, entry: &DiagEntry) {
    if let Some((q, m)) = entry.as_1q_matrix() {
        one(out, q, m);
    } else if let Some((a, b, m)) = entry.as_2q_matrix() {
        two(out, a, b, m);
    }
}

fn lower(out: &mut Vec<Action>, inst: &Instruction) -> bool {
    let Instruction::Gate { gate, targets } = inst else {
        return false;
    };
    match gate {
        Gate::MultiFused(data) => {
            for &(q, m) in &data.gates {
                one(out, q, m);
            }
        }
        Gate::Multi2q(data) => {
            for &(a, b, m) in &data.gates {
                two(out, a, b, m);
            }
        }
        Gate::MultiPauliRot(data) => {
            for &(x, z, theta) in &data.rotations {
                out.push(Action::Pauli(x, z, theta));
            }
        }
        Gate::PauliRot(data) => {
            let (x, z, _) = pauli_rot_masks(targets, data.axes());
            out.push(Action::Pauli(x, z, data.theta));
        }
        Gate::BatchRzz(data) => {
            for &(a, b, theta) in &data.edges {
                two(out, a, b, Gate::Rzz(theta).matrix_4x4());
            }
        }
        Gate::BatchPhase(data) => {
            for &(q, phase) in &data.phases {
                diagonal(
                    out,
                    &DiagEntry::Phase2q {
                        q0: targets[0],
                        q1: q,
                        phase,
                    },
                );
            }
        }
        Gate::DiagonalBatch(data) => {
            for entry in &data.entries {
                diagonal(out, entry);
            }
        }
        Gate::Id
        | Gate::X
        | Gate::Y
        | Gate::Z
        | Gate::H
        | Gate::S
        | Gate::Sdg
        | Gate::T
        | Gate::Tdg
        | Gate::SX
        | Gate::SXdg
        | Gate::Rx(_)
        | Gate::Ry(_)
        | Gate::Rz(_)
        | Gate::P(_)
        | Gate::Fused(_) => one(out, targets[0], gate.matrix_2x2()),
        Gate::Cx => out.push(Action::Cx(targets[0], targets[1])),
        Gate::Swap => out.push(Action::Swap(targets[0], targets[1])),
        Gate::Cz | Gate::Cu(_) | Gate::Rzz(_) | Gate::Fused2q(_) => {
            two(out, targets[0], targets[1], gate.matrix_4x4())
        }
        _ => return false,
    }
    true
}

pub(super) fn actions(instructions: &[Instruction], density_width: Option<usize>) -> Vec<Action> {
    let mut out = Vec::new();
    for (i, inst) in instructions.iter().enumerate() {
        let start = out.len();
        if !lower(&mut out, inst) {
            out.push(Action::Opaque(i));
        } else if let Some(n) = density_width {
            let end = out.len();
            for j in start..end {
                out.push(out[j].conjugated());
            }
            for action in &mut out[start..end] {
                *action = action.shifted(n);
            }
        }
    }
    out
}

#[derive(Default)]
pub(super) struct Schedule {
    width: usize,
    shapes: Vec<Shape>,
    pub(super) stages: Vec<Vec<usize>>,
}

impl Schedule {
    pub(super) fn update(&mut self, actions: &[Action], n: usize) {
        if self.width == n
            && self.shapes.len() == actions.len()
            && self
                .shapes
                .iter()
                .zip(actions)
                .all(|(shape, action)| *shape == action.shape())
        {
            return;
        }
        self.width = n;
        self.shapes = actions.iter().map(Action::shape).collect();
        self.stages.clear();
        let mut start = 0;
        for i in 0..=self.shapes.len() {
            if i == self.shapes.len() || self.shapes[i].kind == 4 {
                self.segment(start, i, n);
                if i < self.shapes.len() {
                    self.stages.push(vec![i]);
                }
                start = i + 1;
            }
        }
    }

    fn scan(&self, remaining: &[usize], high: u64, mask: u64, out: &mut Vec<usize>) {
        out.clear();
        let (mut blocked, mut moved) = (0, 0);
        let mut skipped = 0;
        for &i in remaining {
            let shape = self.shapes[i];
            if shape.moved & mask & !high == 0
                && shape.touched & moved == 0
                && shape.moved & blocked == 0
            {
                out.push(i);
            } else {
                blocked |= shape.touched;
                moved |= shape.moved;
                skipped += 1;
                if skipped > 4 * self.width {
                    break;
                }
            }
        }
    }

    fn segment(&mut self, start: usize, end: usize, n: usize) {
        let mask = (if n == 64 { u64::MAX } else { (1u64 << n) - 1 })
            & !((1u64 << multi_2q_low_bits().min(n)) - 1);
        let budget = multi_2q_high_budget_for(n) as u32;
        let mut remaining: Vec<_> = (start..end).collect();
        let mut taken = Vec::new();
        while !remaining.is_empty() {
            let mut high = 0;
            loop {
                self.scan(&remaining, high, mask, &mut taken);
                let base = taken.len();
                let mut best = None;
                let mut seen = Vec::new();
                for &i in remaining.iter().take(4 * n) {
                    let add = self.shapes[i].moved & mask & !high;
                    if add == 0 || (high | add).count_ones() > budget || seen.contains(&add) {
                        continue;
                    }
                    seen.push(add);
                    self.scan(&remaining, high | add, mask, &mut taken);
                    let gain = taken.len().saturating_sub(base);
                    if gain > 0
                        && best.is_none_or(|(bits, score): (u64, usize)| {
                            gain * bits.count_ones() as usize > score * add.count_ones() as usize
                        })
                    {
                        best = Some((add, gain));
                    }
                }
                match best {
                    Some((add, _)) => high |= add,
                    None => break,
                }
            }
            self.scan(&remaining, high, mask, &mut taken);
            if taken.is_empty() {
                taken.push(remaining[0]);
            }
            remaining.retain(|i| taken.binary_search(i).is_err());
            self.stages.push(taken.clone());
        }
    }
}

pub(crate) fn enabled(n: usize) -> bool {
    crate::circuit::fusion_stage::stages_at(n, n)
}

impl super::StatevectorBackend {
    pub(super) fn apply_schedule(
        &mut self,
        instructions: &[Instruction],
    ) -> crate::error::Result<()> {
        self.apply_schedule_for(instructions, None)
    }

    pub(crate) fn apply_doubled_schedule(
        &mut self,
        instructions: &[Instruction],
        n: usize,
    ) -> crate::error::Result<()> {
        self.apply_schedule_for(instructions, Some(n))
    }

    fn apply_schedule_for(
        &mut self,
        instructions: &[Instruction],
        density_width: Option<usize>,
    ) -> crate::error::Result<()> {
        use crate::backend::Backend;
        let actions = actions(instructions, density_width);
        let mut schedule = std::mem::take(&mut self.schedule);
        schedule.update(&actions, self.num_qubits);
        let result = (|| {
            for stage in &schedule.stages {
                if let Action::Opaque(i) = actions[stage[0]] {
                    self.apply(&instructions[i])?;
                } else {
                    self.apply_mixed_stage(&actions, stage);
                }
            }
            Ok(())
        })();
        self.schedule = schedule;
        result
    }
}

pub(crate) fn supported(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Gate {
            gate: Gate::Id
                | Gate::X
                | Gate::Y
                | Gate::Z
                | Gate::H
                | Gate::S
                | Gate::Sdg
                | Gate::T
                | Gate::Tdg
                | Gate::SX
                | Gate::SXdg
                | Gate::Rx(_)
                | Gate::Ry(_)
                | Gate::Rz(_)
                | Gate::P(_)
                | Gate::Fused(_)
                | Gate::Cx
                | Gate::Cz
                | Gate::Swap
                | Gate::Cu(_)
                | Gate::Rzz(_)
                | Gate::Fused2q(_)
                | Gate::MultiFused(_)
                | Gate::Multi2q(_)
                | Gate::MultiPauliRot(_)
                | Gate::PauliRot(_)
                | Gate::BatchRzz(_)
                | Gate::BatchPhase(_)
                | Gate::DiagonalBatch(_),
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        Backend, density_matrix::DensityMatrixBackend, statevector::StatevectorBackend,
    };
    use crate::circuit::{Circuit, smallvec};
    use crate::gates::{BatchPhaseData, DiagonalBatchData, PauliRotData};
    use crate::sim::unified_pauli::PauliAxis;

    fn fixture(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 1);
        for q in 0..n {
            c.add_gate(Gate::Ry(0.17 + q as f64 * 0.13), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q + 1, q]);
            c.add_gate(Gate::Rzz(0.31), &[q, n - 1]);
            c.add_gate(Gate::Rz(-0.43), &[n - 1]);
            c.add_gate(Gate::Rx(0.12), &[q]);
        }
        c.add_gate(
            Gate::PauliRot(Box::new(PauliRotData {
                theta: 0.27,
                axes: vec![PauliAxis::X, PauliAxis::Y, PauliAxis::Z],
            })),
            &[0, 2, n - 1],
        );
        c.add_gate(
            Gate::BatchPhase(Box::new(BatchPhaseData::new([(
                0,
                Complex64::from_polar(1.0, 0.37),
            )]))),
            &[n - 1, 0],
        );
        c.add_gate(
            Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(vec![DiagEntry::Phase1q {
                qubit: n - 1,
                d0: Complex64::from_polar(1.0, 0.11),
                d1: Complex64::from_polar(1.0, -0.19),
            }]))),
            &[n - 1],
        );
        c
    }

    fn compare(c: &Circuit) -> Vec<Complex64> {
        let mut a = StatevectorBackend::new(42);
        let mut b = StatevectorBackend::new(42);
        a.init(c.num_qubits, 1).unwrap();
        b.init(c.num_qubits, 1).unwrap();
        for inst in &c.instructions {
            a.apply(inst).unwrap();
        }
        b.apply_schedule(&c.instructions).unwrap();
        for (x, y) in a.state.iter().zip(&b.state) {
            assert!((*x - *y).norm() < 1e-11, "{x} != {y}");
        }
        b.state
    }

    #[test]
    fn mixed_schedule_matches_native_gates() {
        for n in [6, 16, 18, 20] {
            compare(&fixture(n));
        }
        assert_eq!(size_of::<Gate>(), 16);
    }

    #[test]
    fn mixed_schedule_rebuilds_on_structure_and_refreshes_payloads() {
        let mut c = fixture(8);
        let mut backend = StatevectorBackend::new(42);
        for gate in [Gate::Ry(0.2), Gate::Rz(0.3), Gate::Ry(-0.4)] {
            c.instructions[0] = Instruction::Gate {
                gate,
                targets: smallvec![7],
            };
            backend.init(8, 1).unwrap();
            backend.apply_schedule(&c.instructions).unwrap();
            let expected = compare(&c);
            assert_eq!(backend.state, expected);
        }
    }

    #[test]
    fn mixed_schedule_refreshes_target_order_and_pauli_letters() {
        let mut c = fixture(8);
        let mut backend = StatevectorBackend::new(42);
        let mut replacements = vec![
            Instruction::Gate {
                gate: Gate::Cx,
                targets: smallvec![7, 0],
            },
            Instruction::Gate {
                gate: Gate::Cx,
                targets: smallvec![0, 7],
            },
        ];
        for axis in [PauliAxis::X, PauliAxis::Y, PauliAxis::Z] {
            replacements.push(Instruction::Gate {
                gate: Gate::PauliRot(Box::new(PauliRotData {
                    theta: 0.47,
                    axes: vec![axis, PauliAxis::Y, PauliAxis::Z],
                })),
                targets: smallvec![0, 2, 7],
            });
        }
        for inst in replacements {
            c.instructions[0] = inst;
            backend.init(8, 1).unwrap();
            backend.apply_schedule(&c.instructions).unwrap();
            assert_eq!(backend.state, compare(&c));
        }
    }

    #[test]
    fn doubled_schedule_matches_density_matrix_native_gates() {
        let c = fixture(7);
        let mut a = DensityMatrixBackend::new(42);
        a.init(7, 1).unwrap();
        for inst in &c.instructions {
            a.apply(inst).unwrap();
        }
        let mut b = StatevectorBackend::new(42);
        b.init(14, 0).unwrap();
        b.apply_doubled_schedule(&c.instructions, 7).unwrap();
        for (x, y) in a.density_matrix().unwrap().iter().zip(&b.state) {
            assert!((*x - *y).norm() < 1e-11, "{x} != {y}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn mixed_schedule_is_equal_across_thread_counts() {
        let c = fixture(18);
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| compare(&c));
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| compare(&c));
        assert_eq!(one, four);
    }

    #[test]
    fn mixed_schedule_keeps_opaque_boundaries() {
        let mut c = fixture(8);
        c.instructions.insert(
            9,
            Instruction::Barrier {
                qubits: smallvec![0, 7],
            },
        );
        c.instructions.insert(
            20,
            Instruction::Measure {
                qubit: 7,
                classical_bit: 0,
            },
        );
        c.instructions.insert(24, Instruction::Reset { qubit: 2 });
        c.instructions.insert(
            21,
            Instruction::Conditional {
                condition: crate::circuit::ClassicalCondition::BitIsOne(0),
                gate: Gate::Ry(0.53),
                targets: smallvec![0],
            },
        );
        compare(&c);
    }

    #[test]
    fn mixed_schedule_folds_repeated_homogeneous_targets_in_order() {
        let mut circuit = Circuit::new(20, 1);
        for (gate, q) in [
            (Gate::Rx(0.31), 19),
            (Gate::Ry(0.27), 0),
            (Gate::Ry(0.59), 19),
            (Gate::Rx(-0.43), 0),
        ] {
            circuit.add_gate(gate, &[q]);
        }
        compare(&circuit);
    }

    #[test]
    fn mixed_schedule_long_stream_keeps_every_operation() {
        let actions: Vec<_> = (0..4096)
            .map(|i| Action::Cx((i + 1) % 24, i % 24))
            .collect();
        let mut schedule = Schedule::default();
        schedule.update(&actions, 24);
        let mut emitted: Vec<_> = schedule.stages.iter().flatten().copied().collect();
        emitted.sort_unstable();
        assert_eq!(emitted, (0..actions.len()).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod census_tests {
    use crate::backend::{
        Backend,
        statevector::{StatevectorBackend, sweeps},
    };
    use crate::circuit::fusion::fuse_circuit_for_width;

    #[test]
    fn doubled_schedule_reduces_native_passes() {
        use crate::backend::density_matrix::DensityMatrixBackend;
        for n in [10, 12] {
            for (name, c) in [
                (
                    "random_d10",
                    crate::circuits::random_circuit(n, 10, 0xDEAD_BEEF),
                ),
                ("qaoa_l6", crate::circuits::qaoa_circuit(n, 6, 0xDEAD_BEEF)),
            ] {
                let fused = fuse_circuit_for_width(&c, true, 2 * n);
                let mut backend = DensityMatrixBackend::new(42);
                backend.init(n, 0).unwrap();
                sweeps::take_ticks();
                for inst in &fused.instructions {
                    backend.apply(inst).unwrap();
                }
                let native = sweeps::take_ticks();
                let expected = backend.density_matrix().unwrap();
                backend.init(n, 0).unwrap();
                backend.apply_instructions(&fused.instructions).unwrap();
                let scheduled = sweeps::take_ticks();
                eprintln!("density_matrix/{name}/{n}: {native} -> {scheduled} kernel passes");
                assert!(scheduled <= native);
                for (a, b) in expected.iter().zip(backend.density_matrix().unwrap()) {
                    assert!((*a - b).norm() < 1e-11);
                }
            }
        }
    }

    #[test]
    fn mixed_schedule_counts_match_kernel_entries() {
        for (n, density) in [(20, false), (10, true)] {
            let c = crate::circuits::diagonal_mixed_circuit(n, 3, 0xDEAD_BEEF);
            let width = if density { 2 * n } else { n };
            let fused = fuse_circuit_for_width(&c, true, width);
            let mut backend = StatevectorBackend::new(42);
            backend.init(width, 0).unwrap();
            sweeps::take_ticks();
            if density {
                backend
                    .apply_doubled_schedule(&fused.instructions, n)
                    .unwrap();
            } else {
                backend.apply_schedule(&fused.instructions).unwrap();
            }
            let actual = sweeps::take_ticks();
            #[cfg(feature = "parallel")]
            let threads = rayon::current_num_threads();
            #[cfg(not(feature = "parallel"))]
            let threads = 1;
            let expected = sweeps::scheduled_sweep_count(&c, threads, density);
            assert_eq!(actual, expected, "density={density}");
        }
    }
}
