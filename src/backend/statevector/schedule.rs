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

fn single_batch_kind(actions: &[Action]) -> bool {
    let mut batched = actions
        .iter()
        .filter(|action| !matches!(action, Action::Opaque(_)));
    let Some(first) = batched.next() else {
        return true;
    };
    matches!(first, Action::One(..) | Action::Two(..) | Action::Pauli(..))
        && batched.all(|action| std::mem::discriminant(action) == std::mem::discriminant(first))
}

pub(crate) fn enabled(n: usize) -> bool {
    crate::circuit::fusion_stage::stages_at(n, n)
}

impl super::StatevectorBackend {
    #[cfg(test)]
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

    /// Schedule `instructions`, or apply them natively when they lower to one batch kind:
    /// the staged stream already runs that kind on the kernel each stage would use.
    pub(super) fn apply_mixed_schedule(
        &mut self,
        instructions: &[Instruction],
    ) -> crate::error::Result<()> {
        use crate::backend::Backend;
        let actions = actions(instructions, None);
        if single_batch_kind(&actions) {
            for instruction in instructions {
                self.apply(instruction)?;
            }
            return Ok(());
        }
        self.run_schedule(instructions, &actions)
    }

    fn apply_schedule_for(
        &mut self,
        instructions: &[Instruction],
        density_width: Option<usize>,
    ) -> crate::error::Result<()> {
        self.run_schedule(instructions, &actions(instructions, density_width))
    }

    fn run_schedule(
        &mut self,
        instructions: &[Instruction],
        actions: &[Action],
    ) -> crate::error::Result<()> {
        use crate::backend::Backend;
        let mut schedule = std::mem::take(&mut self.schedule);
        schedule.update(actions, self.num_qubits);
        let result = (|| {
            for stage in &schedule.stages {
                if let Action::Opaque(i) = actions[stage[0]] {
                    self.apply(&instructions[i])?;
                } else {
                    self.apply_mixed_stage(actions, stage);
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
#[path = "schedule_tests.rs"]
mod tests;

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
