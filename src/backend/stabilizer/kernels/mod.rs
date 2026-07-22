mod batch;
pub(crate) mod rowops;
mod simd;

pub(crate) use batch::MIN_WORDS_FOR_BATCH;
pub(crate) use simd::{rowmul_words, xor_words};

use smallvec::SmallVec;

use crate::backend::Backend;
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use rand::RngExt;

use super::StabilizerBackend;

#[cfg(feature = "parallel")]
pub(super) use crate::backend::MIN_QUBITS_FOR_PAR_GATES;

#[cfg(feature = "parallel")]
use crate::backend::MIN_ANTI_ROWS_FOR_PAR;

#[cfg(feature = "parallel")]
struct SendU64Ptr(*mut u64);
#[cfg(feature = "parallel")]
impl SendU64Ptr {
    #[inline(always)]
    fn ptr(&self) -> *mut u64 {
        self.0
    }
}
#[cfg(feature = "parallel")]
// SAFETY: Used only when each parallel task accesses non-overlapping row regions.
unsafe impl Send for SendU64Ptr {}
#[cfg(feature = "parallel")]
// SAFETY: The pointer itself is read-only; mutation goes through derived slices at disjoint offsets.
unsafe impl Sync for SendU64Ptr {}

#[cfg(feature = "parallel")]
struct SendBoolPtr(*mut bool);
#[cfg(feature = "parallel")]
impl SendBoolPtr {
    #[inline(always)]
    fn ptr(&self) -> *mut bool {
        self.0
    }
}
#[cfg(feature = "parallel")]
// SAFETY: Used only when each parallel task accesses a distinct phase element.
unsafe impl Send for SendBoolPtr {}
#[cfg(feature = "parallel")]
// SAFETY: The pointer itself is read-only; mutation goes through distinct indices.
unsafe impl Sync for SendBoolPtr {}

impl StabilizerBackend {
    #[inline(always)]
    pub(super) fn stride(&self) -> usize {
        2 * self.num_words
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn par_rows(&self) -> bool {
        self.n >= MIN_QUBITS_FOR_PAR_GATES
    }

    #[cfg(not(feature = "parallel"))]
    #[inline(always)]
    fn par_rows(&self) -> bool {
        false
    }

    #[inline(always)]
    fn apply_h(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::h_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    #[inline(always)]
    fn apply_s(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::s_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    #[inline(always)]
    fn apply_sdg(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::sdg_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    #[inline(always)]
    fn apply_x(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::x_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    #[inline(always)]
    fn apply_y(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::y_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    #[inline(always)]
    fn apply_z(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::z_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    #[inline(always)]
    fn apply_cx(&mut self, control: usize, target: usize) {
        let par = self.par_rows();
        rowops::cx_all(
            &mut self.xz,
            &mut self.phase,
            self.num_words,
            par,
            control,
            target,
        );
    }

    #[inline(always)]
    fn apply_cz(&mut self, a: usize, b: usize) {
        let par = self.par_rows();
        rowops::cz_all(&mut self.xz, &mut self.phase, self.num_words, par, a, b);
    }

    #[inline(always)]
    fn apply_swap(&mut self, a: usize, b: usize) {
        let par = self.par_rows();
        rowops::swap_all(&mut self.xz, &mut self.phase, self.num_words, par, a, b);
    }

    pub(super) fn sgi_enabled(&self) -> bool {
        let n = self.n;
        if n < 256 {
            return false;
        }
        let total_rows = 2 * n;
        let avg = self.total_weight / total_rows;
        avg < n / 8 && self.sgi_max_active < total_rows / 16
    }

    fn sgi_apply_1q(&mut self, gate: &Gate, q: usize) {
        let qb = rowops::QubitBit::of(q);
        let stride = self.stride();
        let nw = self.num_words;

        for &g in &self.qubit_active[q] {
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];
            let phase = &mut self.phase[g as usize];

            match gate {
                Gate::H => rowops::h_row(row, phase, nw, qb),
                Gate::S => rowops::s_row(row, phase, nw, qb),
                Gate::Sdg => rowops::sdg_row(row, phase, nw, qb),
                Gate::X => rowops::x_row(row, phase, nw, qb),
                Gate::Y => rowops::y_row(row, phase, nw, qb),
                Gate::Z => rowops::z_row(row, phase, nw, qb),
                Gate::SX => rowops::sx_row(row, phase, nw, qb),
                Gate::SXdg => rowops::sxdg_row(row, phase, nw, qb),
                _ => {}
            }
        }
    }

    fn sgi_merge_active(&mut self, q_a: usize, q_b: usize) {
        self.sgi_merge_buf.clear();
        let list_a = &self.qubit_active[q_a];
        let list_b = &self.qubit_active[q_b];
        let (mut ia, mut ib) = (0, 0);
        while ia < list_a.len() && ib < list_b.len() {
            if list_a[ia] < list_b[ib] {
                self.sgi_merge_buf.push(list_a[ia]);
                ia += 1;
            } else if list_a[ia] > list_b[ib] {
                self.sgi_merge_buf.push(list_b[ib]);
                ib += 1;
            } else {
                self.sgi_merge_buf.push(list_a[ia]);
                ia += 1;
                ib += 1;
            }
        }
        if ia < list_a.len() {
            self.sgi_merge_buf.extend_from_slice(&list_a[ia..]);
        }
        if ib < list_b.len() {
            self.sgi_merge_buf.extend_from_slice(&list_b[ib..]);
        }
    }

    fn sgi_apply_cx(&mut self, ctrl: usize, tgt: usize) {
        let c = rowops::QubitBit::of(ctrl);
        let t = rowops::QubitBit::of(tgt);
        let (c_word, c_bit) = (c.word, c.bit);
        let (t_word, t_bit) = (t.word, t.bit);
        let stride = self.stride();
        let nw = self.num_words;

        self.sgi_merge_active(ctrl, tgt);
        self.sgi_new_a.clear();
        self.sgi_new_b.clear();

        for i in 0..self.sgi_merge_buf.len() {
            let g = self.sgi_merge_buf[i];
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];
            let phase = &mut self.phase[g as usize];

            let xa = (row[c_word] >> c_bit) & 1;
            let za = (row[nw + c_word] >> c_bit) & 1;
            let xb = (row[t_word] >> t_bit) & 1;
            let zb = (row[nw + t_word] >> t_bit) & 1;

            rowops::cx_row(row, phase, nw, c, t);

            let new_xa = (row[c_word] >> c_bit) & 1;
            let new_za = (row[nw + c_word] >> c_bit) & 1;
            let new_xb = (row[t_word] >> t_bit) & 1;
            let new_zb = (row[nw + t_word] >> t_bit) & 1;

            let old_a = xa | za;
            let new_a = new_xa | new_za;
            let old_b = xb | zb;
            let new_b = new_xb | new_zb;

            if new_a != 0 {
                self.sgi_new_a.push(g);
            }
            if old_a != 0 && new_a == 0 {
                self.total_weight -= 1;
            } else if old_a == 0 && new_a != 0 {
                self.total_weight += 1;
            }

            if new_b != 0 {
                self.sgi_new_b.push(g);
            }
            if old_b != 0 && new_b == 0 {
                self.total_weight -= 1;
            } else if old_b == 0 && new_b != 0 {
                self.total_weight += 1;
            }
        }

        std::mem::swap(&mut self.qubit_active[ctrl], &mut self.sgi_new_a);
        std::mem::swap(&mut self.qubit_active[tgt], &mut self.sgi_new_b);
        let ma = self.qubit_active[ctrl].len();
        let mb = self.qubit_active[tgt].len();
        if ma > self.sgi_max_active {
            self.sgi_max_active = ma;
        }
        if mb > self.sgi_max_active {
            self.sgi_max_active = mb;
        }
    }

    fn sgi_apply_cz(&mut self, a: usize, b: usize) {
        let qa = rowops::QubitBit::of(a);
        let qb = rowops::QubitBit::of(b);
        let (a_word, a_bit) = (qa.word, qa.bit);
        let (b_word, b_bit) = (qb.word, qb.bit);
        let stride = self.stride();
        let nw = self.num_words;

        self.sgi_merge_active(a, b);
        self.sgi_new_a.clear();
        self.sgi_new_b.clear();

        for i in 0..self.sgi_merge_buf.len() {
            let g = self.sgi_merge_buf[i];
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];
            let phase = &mut self.phase[g as usize];

            let xa = (row[a_word] >> a_bit) & 1;
            let xb = (row[b_word] >> b_bit) & 1;
            let za = (row[nw + a_word] >> a_bit) & 1;
            let zb = (row[nw + b_word] >> b_bit) & 1;

            rowops::cz_row(row, phase, nw, qa, qb);

            let new_xa = (row[a_word] >> a_bit) & 1;
            let new_za = (row[nw + a_word] >> a_bit) & 1;
            let new_xb = (row[b_word] >> b_bit) & 1;
            let new_zb = (row[nw + b_word] >> b_bit) & 1;

            let old_a_active = xa | za;
            let new_a_active = new_xa | new_za;
            let old_b_active = xb | zb;
            let new_b_active = new_xb | new_zb;

            if new_a_active != 0 {
                self.sgi_new_a.push(g);
            }
            if old_a_active != 0 && new_a_active == 0 {
                self.total_weight -= 1;
            } else if old_a_active == 0 && new_a_active != 0 {
                self.total_weight += 1;
            }

            if new_b_active != 0 {
                self.sgi_new_b.push(g);
            }
            if old_b_active != 0 && new_b_active == 0 {
                self.total_weight -= 1;
            } else if old_b_active == 0 && new_b_active != 0 {
                self.total_weight += 1;
            }
        }

        std::mem::swap(&mut self.qubit_active[a], &mut self.sgi_new_a);
        std::mem::swap(&mut self.qubit_active[b], &mut self.sgi_new_b);
        let ma = self.qubit_active[a].len();
        let mb = self.qubit_active[b].len();
        if ma > self.sgi_max_active {
            self.sgi_max_active = ma;
        }
        if mb > self.sgi_max_active {
            self.sgi_max_active = mb;
        }
    }

    fn sgi_apply_swap(&mut self, a: usize, b: usize) {
        let qa = rowops::QubitBit::of(a);
        let qb = rowops::QubitBit::of(b);
        let (a_word, a_bit) = (qa.word, qa.bit);
        let (b_word, b_bit) = (qb.word, qb.bit);
        let stride = self.stride();
        let nw = self.num_words;

        self.sgi_merge_active(a, b);
        self.sgi_new_a.clear();
        self.sgi_new_b.clear();

        for i in 0..self.sgi_merge_buf.len() {
            let g = self.sgi_merge_buf[i];
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];

            rowops::swap_row(row, nw, qa, qb);

            let new_xa = (row[a_word] >> a_bit) & 1;
            let new_za = (row[nw + a_word] >> a_bit) & 1;
            let new_xb = (row[b_word] >> b_bit) & 1;
            let new_zb = (row[nw + b_word] >> b_bit) & 1;

            if (new_xa | new_za) != 0 {
                self.sgi_new_a.push(g);
            }
            if (new_xb | new_zb) != 0 {
                self.sgi_new_b.push(g);
            }
        }

        std::mem::swap(&mut self.qubit_active[a], &mut self.sgi_new_a);
        std::mem::swap(&mut self.qubit_active[b], &mut self.sgi_new_b);
        let ma = self.qubit_active[a].len();
        let mb = self.qubit_active[b].len();
        if ma > self.sgi_max_active {
            self.sgi_max_active = ma;
        }
        if mb > self.sgi_max_active {
            self.sgi_max_active = mb;
        }
    }

    fn sgi_dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Id => {}
            Gate::X | Gate::Y | Gate::Z | Gate::H | Gate::S | Gate::Sdg | Gate::SX | Gate::SXdg => {
                self.sgi_apply_1q(gate, targets[0]);
            }
            Gate::Cx => self.sgi_apply_cx(targets[0], targets[1]),
            Gate::Cz => self.sgi_apply_cz(targets[0], targets[1]),
            Gate::Swap => self.sgi_apply_swap(targets[0], targets[1]),
            Gate::T
            | Gate::Tdg
            | Gate::Rx(_)
            | Gate::Ry(_)
            | Gate::Rz(_)
            | Gate::P(_)
            | Gate::Rzz(_)
            | Gate::Cu(_)
            | Gate::Mcu(_)
            | Gate::Fused(_)
            | Gate::BatchPhase(_)
            | Gate::BatchRzz(_)
            | Gate::DiagonalBatch(_)
            | Gate::MultiFused(_)
            | Gate::Fused2q(_)
            | Gate::Multi2q(_)
            | Gate::QftBlock { .. } => {
                return Err(PrismError::BackendUnsupported {
                    backend: self.name().to_string(),
                    operation: format!(
                        "non-Clifford gate `{}` (stabilizer backend supports Clifford gates only)",
                        gate.name()
                    ),
                });
            }
        }
        Ok(())
    }

    fn sgi_measure(&mut self, qubit: usize, classical_bit: usize) {
        let n = self.n;
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);
        let stride = self.stride();

        let mut p_row: Option<usize> = None;
        for &g in &self.qubit_active[qubit] {
            let g = g as usize;
            if g >= n && g < 2 * n && self.xz[g * stride + word] & bit_mask != 0 {
                p_row = Some(g);
                break;
            }
        }

        if let Some(p_row) = p_row {
            self.sgi_measure_random(p_row, qubit, classical_bit);
        } else {
            let scratch = 2 * n;
            self.zero_row(scratch);

            let destab_active: SmallVec<[usize; 32]> = self.qubit_active[qubit]
                .iter()
                .filter_map(|&g| {
                    let g = g as usize;
                    if g < n && self.xz[g * stride + word] & bit_mask != 0 {
                        Some(g)
                    } else {
                        None
                    }
                })
                .collect();

            for g in destab_active {
                self.rowmul(scratch, g + n);
            }

            let outcome = self.phase[scratch];
            self.classical_bits[classical_bit] = outcome;
        }
    }

    fn sgi_measure_random(&mut self, p_row: usize, qubit: usize, classical_bit: usize) {
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);

        let p_base = p_row * stride;
        let p_data: SmallVec<[u64; 32]> = SmallVec::from_slice(&self.xz[p_base..p_base + stride]);
        let p_phase = self.phase[p_row];

        for i in 0..2 * n {
            if i == p_row {
                continue;
            }
            if self.xz[i * stride + word] & bit_mask != 0 {
                let initial_sum =
                    if p_phase { 2u64 } else { 0 } + if self.phase[i] { 2u64 } else { 0 };
                let row = &mut self.xz[i * stride..(i + 1) * stride];
                let (rx, rz) = row.split_at_mut(nw);
                let sum = rowmul_words(
                    rx,
                    &mut rz[..nw],
                    &p_data[..nw],
                    &p_data[nw..2 * nw],
                    initial_sum,
                );
                self.phase[i] = (sum & 3) >= 2;
            }
        }

        let d_row = p_row - n;
        let d_base = d_row * stride;
        self.xz.copy_within(p_base..p_base + stride, d_base);
        self.phase[d_row] = self.phase[p_row];

        self.zero_row(p_row);
        let outcome: bool = self.rng.random();
        self.phase[p_row] = outcome;
        self.xz[p_row * stride + nw + word] |= bit_mask;

        self.classical_bits[classical_bit] = outcome;

        self.rebuild_qubit_active();
    }

    fn rebuild_qubit_active(&mut self) {
        let n = self.n;
        let stride = self.stride();
        let nw = self.num_words;

        for list in &mut self.qubit_active {
            list.clear();
        }
        self.total_weight = 0;

        for g in 0..2 * n {
            let row = &self.xz[g * stride..(g + 1) * stride];
            for w in 0..nw {
                let active = row[w] | row[nw + w];
                let mut bits = active;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    let q = w * 64 + b;
                    if q < n {
                        self.qubit_active[q].push(g as u32);
                        self.total_weight += 1;
                    }
                    bits &= bits - 1;
                }
            }
        }
        self.sgi_max_active = self.qubit_active.iter().map(|a| a.len()).max().unwrap_or(0);
    }

    /// Random-outcome measurement: rowmul anti-commuting rows against the pivot,
    /// then collapse to a Z-eigenstate. Parallelizable (disjoint destinations).
    pub(super) fn measure_random(
        &mut self,
        p_row: usize,
        word: usize,
        bit_mask: u64,
        classical_bit: usize,
    ) {
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();

        let p_base = p_row * stride;
        let p_data: SmallVec<[u64; 32]> = SmallVec::from_slice(&self.xz[p_base..p_base + stride]);
        let p_phase = self.phase[p_row];

        let anti_rows: SmallVec<[usize; 16]> = (0..2 * n)
            .filter(|&r| r != p_row && self.xz[r * stride + word] & bit_mask != 0)
            .collect();

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES && anti_rows.len() >= MIN_ANTI_ROWS_FOR_PAR {
            use rayon::prelude::*;

            let xz_ptr = SendU64Ptr(self.xz.as_mut_ptr());
            let phase_ptr = SendBoolPtr(self.phase.as_mut_ptr());

            // SAFETY: Each row index in anti_rows is unique (collected from a filter
            // with no duplicates) and none equals p_row. Each row occupies
            // [row*stride .. (row+1)*stride] in xz, non-overlapping regions.
            // Phase elements are at distinct indices. p_data is a separate copy.
            anti_rows.par_iter().for_each(|&r| {
                let xz_base = xz_ptr.ptr();
                let ph_base = phase_ptr.ptr();
                // SAFETY: r is unique within anti_rows and identifies one
                // in-bounds row region of length stride.
                let row =
                    unsafe { std::slice::from_raw_parts_mut(xz_base.add(r * stride), stride) };
                // SAFETY: r is unique within anti_rows, so this mutable
                // reference targets one distinct phase element.
                let phase = unsafe { &mut *ph_base.add(r) };

                let initial_sum = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
                let (rx, rz) = row.split_at_mut(nw);
                let sum = rowmul_words(
                    rx,
                    &mut rz[..nw],
                    &p_data[..nw],
                    &p_data[nw..2 * nw],
                    initial_sum,
                );
                *phase = (sum & 3) >= 2;
            });
        } else {
            for &r in &anti_rows {
                let base = r * stride;
                let row = &mut self.xz[base..base + stride];
                let phase = &mut self.phase[r];
                let initial_sum = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
                let (rx, rz) = row.split_at_mut(nw);
                let sum = rowmul_words(
                    rx,
                    &mut rz[..nw],
                    &p_data[..nw],
                    &p_data[nw..2 * nw],
                    initial_sum,
                );
                *phase = (sum & 3) >= 2;
            }
        }

        #[cfg(not(feature = "parallel"))]
        for &r in &anti_rows {
            let base = r * stride;
            let row = &mut self.xz[base..base + stride];
            let phase = &mut self.phase[r];
            let initial_sum = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
            let (rx, rz) = row.split_at_mut(nw);
            let sum = rowmul_words(
                rx,
                &mut rz[..nw],
                &p_data[..nw],
                &p_data[nw..2 * nw],
                initial_sum,
            );
            *phase = (sum & 3) >= 2;
        }

        let dest_row = p_row - n;
        self.copy_row(dest_row, p_row);

        self.zero_row(p_row);
        self.xz[p_row * stride + nw + word] |= bit_mask;

        let outcome: bool = self.rng.random();
        self.phase[p_row] = outcome;
        self.classical_bits[classical_bit] = outcome;
    }

    #[inline(always)]
    fn apply_sx(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::sx_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    #[inline(always)]
    fn apply_sxdg(&mut self, a: usize) {
        let par = self.par_rows();
        rowops::sxdg_all(&mut self.xz, &mut self.phase, self.num_words, par, a);
    }

    pub(super) fn apply_instructions_sgi(&mut self, instructions: &[Instruction]) -> Result<()> {
        for (idx, instruction) in instructions.iter().enumerate() {
            if !self.sgi_enabled() {
                return self.apply_instructions_word_batch(&instructions[idx..]);
            }

            match instruction {
                Instruction::Gate { gate, targets } => {
                    self.sgi_dispatch_gate(gate, targets)?;
                }
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    self.sgi_measure(*qubit, *classical_bit);
                }
                Instruction::Reset { qubit } => {
                    self.apply_reset(*qubit)?;
                }
                Instruction::Barrier { .. } => {}
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } if condition.evaluate(&self.classical_bits) => {
                    self.sgi_dispatch_gate(gate, targets)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Id => {}
            Gate::X => self.apply_x(targets[0]),
            Gate::Y => self.apply_y(targets[0]),
            Gate::Z => self.apply_z(targets[0]),
            Gate::H => self.apply_h(targets[0]),
            Gate::S => self.apply_s(targets[0]),
            Gate::Sdg => self.apply_sdg(targets[0]),
            Gate::SX => self.apply_sx(targets[0]),
            Gate::SXdg => self.apply_sxdg(targets[0]),
            Gate::Cx => self.apply_cx(targets[0], targets[1]),
            Gate::Cz => self.apply_cz(targets[0], targets[1]),
            Gate::Swap => self.apply_swap(targets[0], targets[1]),
            Gate::T
            | Gate::Tdg
            | Gate::Rx(_)
            | Gate::Ry(_)
            | Gate::Rz(_)
            | Gate::P(_)
            | Gate::Rzz(_)
            | Gate::Cu(_)
            | Gate::Mcu(_)
            | Gate::Fused(_)
            | Gate::BatchPhase(_)
            | Gate::BatchRzz(_)
            | Gate::DiagonalBatch(_)
            | Gate::MultiFused(_)
            | Gate::Fused2q(_)
            | Gate::Multi2q(_)
            | Gate::QftBlock { .. } => {
                return Err(PrismError::BackendUnsupported {
                    backend: self.name().to_string(),
                    operation: format!(
                        "non-Clifford gate `{}` (stabilizer backend supports Clifford gates only)",
                        gate.name()
                    ),
                });
            }
        }
        Ok(())
    }

    pub(super) fn apply_gates_only_sgi(&mut self, instructions: &[Instruction]) -> Result<()> {
        for (idx, instruction) in instructions.iter().enumerate() {
            if !self.sgi_enabled() {
                return self.apply_gates_only_word_batch(&instructions[idx..]);
            }

            match instruction {
                Instruction::Gate { gate, targets } => {
                    self.sgi_dispatch_gate(gate, targets)?;
                }
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } if condition.evaluate(&self.classical_bits) => {
                    self.sgi_dispatch_gate(gate, targets)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}
