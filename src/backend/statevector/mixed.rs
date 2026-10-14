use super::*;
use crate::backend::statevector::schedule::Action;

#[allow(clippy::large_enum_variant)]
enum Prepared {
    One(usize, simd::PreparedGate1q),
    Two(usize, usize, simd::PreparedGate2q),
    Diagonal {
        a: usize,
        b: usize,
        outer_a: usize,
        outer_b: usize,
        phases: [Complex64; 4],
    },
    DiagonalBatch(Box<DiagBatchTables>),
    Pauli(TiledPauliRot),
    Cx {
        control: usize,
        outer: usize,
        target: usize,
    },
    Swap(usize, usize),
}

#[inline(always)]
fn apply_ops(tile: &mut [Complex64], base: usize, bits: usize, rest: &[usize], ops: &[Prepared]) {
    let virtual_base = rest
        .iter()
        .enumerate()
        .fold(0, |index, (j, &q)| index | ((base >> q & 1) << (bits + j)));
    for op in ops {
        match op {
            Prepared::One(q, gate) => gate.apply_tiled(tile, *q),
            Prepared::Two(a, b, gate) => gate.apply_tiled(tile, bits, *a, *b),
            Prepared::Pauli(rotation) => rotation.apply(tile, base),
            Prepared::DiagonalBatch(tables) => {
                #[cfg(target_arch = "x86_64")]
                if simd::has_bmi2() && simd::has_fma() {
                    // SAFETY: BMI2 and FMA detected; the tables cover the mapped tile coordinates.
                    unsafe {
                        diagonal_batch_tile_bmi2(
                            tile,
                            virtual_base,
                            &tables.tables,
                            &tables.group_pext_masks,
                            tables.num_groups,
                        );
                    }
                    continue;
                }
                let shifts = diag_batch_group_shifts(&tables.unique_qubits, &tables.group_sizes);
                diagonal_batch_tile_scalar(
                    tile,
                    virtual_base,
                    &tables.tables,
                    &shifts,
                    &tables.group_sizes,
                    tables.num_groups,
                );
            }
            Prepared::Cx {
                control,
                outer,
                target,
            } => {
                for i in 0..tile.len() {
                    if i & target == 0 && (i & control != 0 || base & outer != 0) {
                        tile.swap(i, i | target);
                    }
                }
            }
            Prepared::Swap(a, b) => {
                for i in 0..tile.len() {
                    if i & a == 0 && i & b != 0 {
                        tile.swap(i, i ^ a ^ b);
                    }
                }
            }
            Prepared::Diagonal {
                a,
                b,
                outer_a,
                outer_b,
                phases,
            } => {
                let fixed = usize::from(base & outer_a != 0) * 2 + usize::from(base & outer_b != 0);
                for (i, value) in tile.iter_mut().enumerate() {
                    let index = fixed | (usize::from(i & a != 0) * 2) | usize::from(i & b != 0);
                    *value *= phases[index];
                }
            }
        }
    }
}

// SAFETY: callers partition the state into disjoint subcubes matching `plan`.
#[inline(never)]
unsafe fn apply_mixed_subcube(
    state: *mut Complex64,
    plan: &SubcubePlan,
    rest: &[usize],
    outer: usize,
    ops: &[Prepared],
) {
    let base = rest
        .iter()
        .enumerate()
        .fold(0, |base, (j, &q)| base | ((outer >> j & 1) << q));
    let offset = |c: usize| {
        plan.high
            .iter()
            .enumerate()
            .fold(base, |off, (j, &q)| off | ((c >> j & 1) << q))
    };
    let run = 1usize << plan.low;
    with_subcube_tile(plan.tile_bits, |tile| {
        for c in 0..1usize << plan.high.len() {
            // SAFETY: same contract as the enclosing unsafe fn.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.add(offset(c)),
                    tile.as_mut_ptr().add(c * run),
                    run,
                );
            }
        }
        if tile.iter().all(|z| z.re == 0.0 && z.im == 0.0) {
            return;
        }
        apply_ops(tile, base, plan.tile_bits, rest, ops);
        for c in 0..1usize << plan.high.len() {
            // SAFETY: same contract as the enclosing unsafe fn.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tile.as_ptr().add(c * run),
                    state.add(offset(c)),
                    run,
                );
            }
        }
    });
}

fn push_diagonal(entries: &mut Vec<DiagEntry>, a: usize, b: usize, d: [Complex64; 4]) {
    if a == b {
        entries.push(DiagEntry::Phase1q {
            qubit: a,
            d0: d[0],
            d1: d[3],
        });
    } else {
        entries.push(DiagEntry::Phase1q {
            qubit: a,
            d0: d[0],
            d1: d[2],
        });
        entries.push(DiagEntry::Phase1q {
            qubit: b,
            d0: Complex64::new(1.0, 0.0),
            d1: d[1] / d[0],
        });
        entries.push(DiagEntry::Phase2q {
            q0: a,
            q1: b,
            phase: d[3] * d[0] / (d[1] * d[2]),
        });
    }
}

impl StatevectorBackend {
    pub(in crate::backend::statevector) fn apply_mixed_stage(
        &mut self,
        actions: &[Action],
        indices: &[usize],
    ) {
        if indices
            .iter()
            .all(|&i| matches!(actions[i], Action::Two(..)))
        {
            let gates: Vec<_> = indices
                .iter()
                .map(|&i| match actions[i] {
                    Action::Two(a, b, m) => (a, b, m),
                    _ => unreachable!(),
                })
                .collect();
            self.apply_multi_2q(&gates);
            return;
        }
        if indices
            .iter()
            .all(|&i| matches!(actions[i], Action::One(..)))
        {
            let gates = crate::backend::statevector::schedule::one_gates(actions, indices);
            self.apply_multi_1q(&gates);
            return;
        }
        if indices
            .iter()
            .all(|&i| matches!(actions[i], Action::Pauli(..)))
        {
            let gates: Vec<_> = indices
                .iter()
                .map(|&i| match actions[i] {
                    Action::Pauli(x, z, t) => (x, z, t),
                    _ => unreachable!(),
                })
                .collect();
            self.apply_multi_pauli_rot(&gates);
            return;
        }
        if indices
            .iter()
            .all(|&i| matches!(actions[i], Action::Diagonal(..)))
        {
            let mut entries = Vec::new();
            for &i in indices {
                let Action::Diagonal(a, b, d) = actions[i] else {
                    unreachable!();
                };
                push_diagonal(&mut entries, a, b, d);
            }
            self.apply_diagonal_batch(&entries);
            return;
        }
        let moved = indices.iter().fold(0usize, |mask, &i| {
            mask | match actions[i] {
                Action::One(q, _) => 1 << q,
                Action::Two(a, b, _) => 1 << a | 1 << b,
                Action::Pauli(x, _, _) => x,
                Action::Cx(_, b) => 1 << b,
                Action::Swap(a, b) => 1 << a | 1 << b,
                _ => 0,
            }
        });
        let qubits: Vec<_> = (0..self.num_qubits)
            .filter(|q| moved >> q & 1 != 0)
            .collect();
        let bits = multi_2q_tile_bits_for(self.num_qubits).min(self.num_qubits);
        let plan =
            qubit_subcube_plan(qubits.iter().copied(), self.num_qubits).unwrap_or_else(|| {
                assert_eq!(moved >> bits, 0, "mixed stage exceeds the tile budget");
                SubcubePlan {
                    tile_bits: bits,
                    low: bits,
                    high: SmallVec::new(),
                    map: (0..self.num_qubits)
                        .map(|q| if q < bits { q } else { usize::MAX })
                        .collect(),
                }
            });
        let bit = |q: usize| (plan.map[q] != usize::MAX).then_some(plan.map[q]);
        let rest = plan.rest(self.num_qubits);
        let virtual_bit = |q: usize| {
            bit(q).unwrap_or_else(|| bits + rest.iter().position(|&outside| outside == q).unwrap())
        };
        let mut prepared = Vec::new();
        let mut cursor = 0;
        while cursor < indices.len() {
            if matches!(actions[indices[cursor]], Action::Diagonal(..)) {
                let start = cursor;
                let mut touched = 0usize;
                let mut entries = Vec::new();
                while cursor < indices.len() {
                    let Action::Diagonal(a, b, d) = actions[indices[cursor]] else {
                        break;
                    };
                    let joined = touched | 1 << a | 1 << b;
                    if joined.count_ones() as usize > DIAG_BATCH_MAX_QUBITS_PER_GROUP {
                        break;
                    }
                    touched = joined;
                    push_diagonal(&mut entries, virtual_bit(a), virtual_bit(b), d);
                    cursor += 1;
                }
                if cursor - start > 1 {
                    let mut tables = Box::new(DiagBatchTables::empty());
                    assert!(fill_diagonal_batch_tables(&entries, &mut tables));
                    prepared.push(Prepared::DiagonalBatch(tables));
                    continue;
                }
                cursor = start;
            }
            prepared.push(match &actions[indices[cursor]] {
                Action::One(q, m) => Prepared::One(plan.map[*q], simd::PreparedGate1q::new(m)),
                Action::Two(a, b, m) => {
                    Prepared::Two(plan.map[*a], plan.map[*b], simd::PreparedGate2q::new(m))
                }
                Action::Pauli(x, z, t) => Prepared::Pauli(TiledPauliRot::new(*x, *z, *t, bit)),
                Action::Diagonal(a, b, phases) => Prepared::Diagonal {
                    a: bit(*a).map_or(0, |q| 1 << q),
                    b: bit(*b).map_or(0, |q| 1 << q),
                    outer_a: if bit(*a).is_none() { 1 << a } else { 0 },
                    outer_b: if bit(*b).is_none() { 1 << b } else { 0 },
                    phases: *phases,
                },
                Action::Cx(a, b) => Prepared::Cx {
                    control: bit(*a).map_or(0, |q| 1 << q),
                    outer: if bit(*a).is_none() { 1 << a } else { 0 },
                    target: 1 << plan.map[*b],
                },
                Action::Swap(a, b) => Prepared::Swap(1 << plan.map[*a], 1 << plan.map[*b]),
                Action::Opaque(_) => unreachable!(),
            });
            cursor += 1;
        }
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        if plan.high.is_empty() {
            let len = 1usize << bits;
            #[cfg(feature = "parallel")]
            if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
                self.state
                    .par_chunks_mut(len)
                    .enumerate()
                    .for_each(|(i, tile)| apply_ops(tile, i * len, bits, &rest, &prepared));
                return;
            }
            for (i, tile) in self.state.chunks_mut(len).enumerate() {
                apply_ops(tile, i * len, bits, &rest, &prepared);
            }
            return;
        }
        let count = 1usize << (self.num_qubits - plan.tile_bits);
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            let ptr = SendPtr(self.state.as_mut_ptr());
            (0..count).into_par_iter().for_each(|outer| {
                // SAFETY: each outer index owns a disjoint subcube of the initialized state.
                unsafe {
                    apply_mixed_subcube(ptr.as_complex_ptr(), &plan, &rest, outer, &prepared);
                }
            });
            return;
        }
        for outer in 0..count {
            // SAFETY: each outer index owns a disjoint subcube of the initialized state.
            unsafe {
                apply_mixed_subcube(self.state.as_mut_ptr(), &plan, &rest, outer, &prepared);
            }
        }
    }
}
