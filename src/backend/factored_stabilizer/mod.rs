//! Factored stabilizer simulation backend.
//!
//! Dynamically-partitioned stabilizer simulator: maintains independent
//! sub-tableaux per disentangled qubit group. Merges on-demand when
//! entangling gates bridge groups, splits when measurement reveals
//! product structure. O((a+b)²/64) polynomial merge cost.

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::stabilizer::kernels::rowmul_words;
use crate::backend::Backend;
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;

#[cfg(feature = "parallel")]
use crate::backend::{MIN_ANTI_ROWS_FOR_PAR, MIN_QUBITS_FOR_PAR_GATES};

#[cfg(test)]
mod tests;

struct SubTableau {
    n: usize,
    num_words: usize,
    xz: Vec<u64>,
    phase: Vec<bool>,
    qubits: SmallVec<[usize; 8]>,
}

impl SubTableau {
    fn new_single(global_qubit: usize) -> Self {
        let n = 1;
        let num_words = 1;
        let stride = 2;
        let total_rows = 3;
        let mut xz = vec![0u64; total_rows * stride];
        let phase = vec![false; total_rows];
        xz[0] = 1;
        xz[stride + num_words] = 1;
        SubTableau {
            n,
            num_words,
            xz,
            phase,
            qubits: SmallVec::from_elem(global_qubit, 1),
        }
    }

    #[inline(always)]
    fn stride(&self) -> usize {
        2 * self.num_words
    }

    fn local_qubit(&self, global: usize) -> usize {
        self.qubits.iter().position(|&q| q == global).unwrap()
    }

    fn apply_h(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let xw = row[word];
            let zw = row[nw + word];
            let x = xw & bit_mask;
            let z = zw & bit_mask;
            *phase ^= (x != 0) && (z != 0);
            row[word] = (xw & !bit_mask) | z;
            row[nw + word] = (zw & !bit_mask) | x;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_s(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            *phase ^= (x != 0) && (z != 0);
            row[nw + word] ^= x;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_sdg(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            row[nw + word] ^= x;
            let z_new = row[nw + word] & bit_mask;
            *phase ^= (x != 0) && (z_new != 0);
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_x(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            *phase ^= (row[nw + word] & bit_mask) != 0;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_y(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            *phase ^= ((row[word] ^ row[nw + word]) & bit_mask) != 0;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_z(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();

        let row_op = |row: &mut [u64], phase: &mut bool| {
            *phase ^= (row[word] & bit_mask) != 0;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_sx(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            *phase ^= (z != 0) && (x == 0);
            row[word] ^= z;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_sxdg(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            *phase ^= (x != 0) && (z != 0);
            row[word] ^= z;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_cx(&mut self, ctrl: usize, tgt: usize) {
        let cw = ctrl / 64;
        let cb = 1u64 << (ctrl % 64);
        let tw = tgt / 64;
        let tb = 1u64 << (tgt % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let xc = (row[cw] & cb) != 0;
            let zc = (row[nw + cw] & cb) != 0;
            let xt = (row[tw] & tb) != 0;
            let zt = (row[nw + tw] & tb) != 0;
            *phase ^= xc && zt && (xt == zc);
            if xc {
                row[tw] ^= tb;
            }
            if zt {
                row[nw + cw] ^= cb;
            }
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_cz(&mut self, a: usize, b: usize) {
        let aw = a / 64;
        let ab = 1u64 << (a % 64);
        let bw = b / 64;
        let bb = 1u64 << (b % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let xa = (row[aw] & ab) != 0;
            let za = (row[nw + aw] & ab) != 0;
            let xb = (row[bw] & bb) != 0;
            let zb = (row[nw + bw] & bb) != 0;
            *phase ^= xa && xb && (za != zb);
            if xa {
                row[nw + bw] ^= bb;
            }
            if xb {
                row[nw + aw] ^= ab;
            }
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn apply_swap(&mut self, a: usize, b: usize) {
        let aw = a / 64;
        let ab = 1u64 << (a % 64);
        let bw = b / 64;
        let bb = 1u64 << (b % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], _phase: &mut bool| {
            let xa = (row[aw] & ab) != 0;
            let za = (row[nw + aw] & ab) != 0;
            let xb = (row[bw] & bb) != 0;
            let zb = (row[nw + bw] & bb) != 0;
            row[aw] = (row[aw] & !ab) | if xb { ab } else { 0 };
            row[bw] = (row[bw] & !bb) | if xa { bb } else { 0 };
            row[nw + aw] = (row[nw + aw] & !ab) | if zb { ab } else { 0 };
            row[nw + bw] = (row[nw + bw] & !bb) | if za { bb } else { 0 };
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    fn dispatch_gate(&mut self, gate: &Gate, local_targets: &[usize]) -> Result<()> {
        match gate {
            Gate::H => self.apply_h(local_targets[0]),
            Gate::S => self.apply_s(local_targets[0]),
            Gate::Sdg => self.apply_sdg(local_targets[0]),
            Gate::X => self.apply_x(local_targets[0]),
            Gate::Y => self.apply_y(local_targets[0]),
            Gate::Z => self.apply_z(local_targets[0]),
            Gate::SX => self.apply_sx(local_targets[0]),
            Gate::SXdg => self.apply_sxdg(local_targets[0]),
            Gate::Id => {}
            Gate::Cx => self.apply_cx(local_targets[0], local_targets[1]),
            Gate::Cz => self.apply_cz(local_targets[0], local_targets[1]),
            Gate::Swap => self.apply_swap(local_targets[0], local_targets[1]),
            _ => {
                return Err(PrismError::BackendUnsupported {
                    backend: "factored-stabilizer".to_string(),
                    operation: format!("gate {:?}", gate),
                });
            }
        }
        Ok(())
    }

    fn rowmul(&mut self, h: usize, i: usize) {
        debug_assert!(h != i);
        let stride = self.stride();
        let nw = self.num_words;
        let base_h = h * stride;
        let base_i = i * stride;
        let initial = if self.phase[i] { 2u64 } else { 0 } + if self.phase[h] { 2u64 } else { 0 };
        // SAFETY: h != i so row regions are non-overlapping.
        let (dst_x, dst_z, src_x, src_z) = unsafe {
            let ptr = self.xz.as_mut_ptr();
            (
                std::slice::from_raw_parts_mut(ptr.add(base_h), nw),
                std::slice::from_raw_parts_mut(ptr.add(base_h + nw), nw),
                std::slice::from_raw_parts(ptr.add(base_i) as *const u64, nw),
                std::slice::from_raw_parts(ptr.add(base_i + nw) as *const u64, nw),
            )
        };
        let sum = rowmul_words(dst_x, dst_z, src_x, src_z, initial);
        self.phase[h] = (sum & 3) >= 2;
    }

    fn copy_row(&mut self, dst: usize, src: usize) {
        let stride = self.stride();
        let src_start = src * stride;
        let dst_start = dst * stride;
        self.xz
            .copy_within(src_start..src_start + stride, dst_start);
        self.phase[dst] = self.phase[src];
    }

    fn zero_row(&mut self, r: usize) {
        let stride = self.stride();
        let start = r * stride;
        self.xz[start..start + stride].fill(0);
        self.phase[r] = false;
    }

    fn measure(&mut self, local_q: usize, rng: &mut ChaCha8Rng) -> bool {
        let n = self.n;
        let word = local_q / 64;
        let bit_mask = 1u64 << (local_q % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let mut p: Option<usize> = None;
        for i in n..2 * n {
            if self.xz[i * stride + word] & bit_mask != 0 {
                p = Some(i);
                break;
            }
        }

        if let Some(p_row) = p {
            let p_base = p_row * stride;
            let p_data: SmallVec<[u64; 32]> =
                SmallVec::from_slice(&self.xz[p_base..p_base + stride]);
            let p_phase = self.phase[p_row];

            let anti_rows: SmallVec<[usize; 16]> = (0..2 * n)
                .filter(|&r| r != p_row && self.xz[r * stride + word] & bit_mask != 0)
                .collect();

            #[cfg(feature = "parallel")]
            if n >= MIN_QUBITS_FOR_PAR_GATES && anti_rows.len() >= MIN_ANTI_ROWS_FOR_PAR {
                use rayon::prelude::*;

                struct SendU64Ptr(*mut u64);
                impl SendU64Ptr {
                    #[inline(always)]
                    fn ptr(&self) -> *mut u64 {
                        self.0
                    }
                }
                // SAFETY: Each parallel task accesses non-overlapping row regions.
                unsafe impl Send for SendU64Ptr {}
                // SAFETY: The pointer itself is read-only; mutation goes through derived slices.
                unsafe impl Sync for SendU64Ptr {}

                struct SendBoolPtr(*mut bool);
                impl SendBoolPtr {
                    #[inline(always)]
                    fn ptr(&self) -> *mut bool {
                        self.0
                    }
                }
                // SAFETY: Each parallel task accesses a distinct phase element.
                unsafe impl Send for SendBoolPtr {}
                // SAFETY: The pointer itself is read-only; mutation goes through distinct indices.
                unsafe impl Sync for SendBoolPtr {}

                let xz_ptr = SendU64Ptr(self.xz.as_mut_ptr());
                let phase_ptr = SendBoolPtr(self.phase.as_mut_ptr());

                // SAFETY: Each row index in anti_rows is unique and none equals p_row.
                // Row regions [r*stride .. (r+1)*stride] are non-overlapping.
                // p_data is a separate copy.
                anti_rows.par_iter().for_each(|&r| {
                    let row = unsafe {
                        std::slice::from_raw_parts_mut(xz_ptr.ptr().add(r * stride), stride)
                    };
                    let phase = unsafe { &mut *phase_ptr.ptr().add(r) };
                    let initial = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
                    let (rx, rz) = row.split_at_mut(nw);
                    let sum = rowmul_words(
                        rx,
                        &mut rz[..nw],
                        &p_data[..nw],
                        &p_data[nw..2 * nw],
                        initial,
                    );
                    *phase = (sum & 3) >= 2;
                });
            }

            #[cfg(feature = "parallel")]
            if n < MIN_QUBITS_FOR_PAR_GATES || anti_rows.len() < MIN_ANTI_ROWS_FOR_PAR {
                for &r in &anti_rows {
                    let base = r * stride;
                    let row = &mut self.xz[base..base + stride];
                    let phase = &mut self.phase[r];
                    let initial = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
                    let (rx, rz) = row.split_at_mut(nw);
                    let sum = rowmul_words(
                        rx,
                        &mut rz[..nw],
                        &p_data[..nw],
                        &p_data[nw..2 * nw],
                        initial,
                    );
                    *phase = (sum & 3) >= 2;
                }
            }

            #[cfg(not(feature = "parallel"))]
            for &r in &anti_rows {
                let base = r * stride;
                let row = &mut self.xz[base..base + stride];
                let phase = &mut self.phase[r];
                let initial = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
                let (rx, rz) = row.split_at_mut(nw);
                let sum = rowmul_words(
                    rx,
                    &mut rz[..nw],
                    &p_data[..nw],
                    &p_data[nw..2 * nw],
                    initial,
                );
                *phase = (sum & 3) >= 2;
            }

            self.copy_row(p_row - n, p_row);
            self.zero_row(p_row);
            let outcome: bool = rng.gen();
            self.xz[p_row * stride + nw + word] |= bit_mask;
            self.phase[p_row] = outcome;
            outcome
        } else {
            let scratch = 2 * n;
            self.zero_row(scratch);
            for i in 0..n {
                if self.xz[i * stride + word] & bit_mask != 0 {
                    self.rowmul(scratch, i + n);
                }
            }
            self.phase[scratch]
        }
    }

    fn reset_qubit(&mut self, local_q: usize, rng: &mut ChaCha8Rng) -> Result<()> {
        let outcome = self.measure(local_q, rng);
        if outcome {
            self.apply_x(local_q);
        }
        Ok(())
    }

    fn compute_probabilities(&self) -> Result<Vec<f64>> {
        let n = self.n;
        if n > crate::backend::MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: "factored-stabilizer".to_string(),
                operation: format!(
                    "probabilities for {} qubits (max {})",
                    n,
                    crate::backend::MAX_PROB_QUBITS
                ),
            });
        }

        let mut work_xz = self.xz.clone();
        let mut work_phase = self.phase.clone();
        let nw = self.num_words;
        let stride = self.stride();

        let (k, col_map) = gauss_eliminate_x(&mut work_xz, &mut work_phase, n, nw, stride);

        let seed = solve_diagonal_seed(&work_xz, &work_phase, n, nw, stride, &col_map, k);

        let dim = 1usize << n;
        let mut probs = vec![0.0f64; dim];
        let num_coset = 1usize << k;
        let amp_sq = 1.0 / num_coset as f64;

        let mut gen_xparts = Vec::with_capacity(k);
        for g in 0..k {
            let row = n + g;
            let mut xval = 0usize;
            for q in 0..n {
                let w = q / 64;
                let b = q % 64;
                if work_xz[row * stride + w] & (1u64 << b) != 0 {
                    xval |= 1 << q;
                }
            }
            gen_xparts.push(xval);
        }

        let mut state = seed;
        probs[state] = amp_sq;
        for gray in 1..num_coset {
            let bit = gray.trailing_zeros() as usize;
            state ^= gen_xparts[bit];
            probs[state] = amp_sq;
        }

        Ok(probs)
    }

    fn compute_statevector(&self) -> Result<Vec<Complex64>> {
        let n = self.n;
        if n > crate::backend::MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: "factored-stabilizer".to_string(),
                operation: format!("statevector for {} qubits", n),
            });
        }

        let mut work_xz = self.xz.clone();
        let mut work_phase = self.phase.clone();
        let nw = self.num_words;
        let stride = self.stride();

        let (k, col_map) = gauss_eliminate_x(&mut work_xz, &mut work_phase, n, nw, stride);
        let seed = solve_diagonal_seed(&work_xz, &work_phase, n, nw, stride, &col_map, k);

        let dim = 1usize << n;
        let zero = Complex64::new(0.0, 0.0);
        let mut sv = vec![zero; dim];
        sv[seed] = Complex64::new(1.0, 0.0);

        let powers_of_i = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];

        let mut visited_gen = vec![0u32; dim];
        let mut current_gen = 0u32;
        for i in 0..n {
            let row = (i + n) * stride;
            let mut x_bits = 0usize;
            let mut z_bits = 0usize;
            for w in 0..nw {
                let shift = w * 64;
                if shift < usize::BITS as usize {
                    x_bits |= (self.xz[row + w] as usize) << shift;
                    z_bits |= (self.xz[row + nw + w] as usize) << shift;
                }
            }
            let r = self.phase[i + n];
            let m = (x_bits & z_bits).count_ones() as usize;
            let i_factor = powers_of_i[m & 3];
            let base_sign = if r { -1.0 } else { 1.0 };

            if x_bits == 0 {
                for (y, s) in sv.iter_mut().enumerate() {
                    let dot_parity = (z_bits & y).count_ones() & 1;
                    let phase_val = if dot_parity == 0 {
                        base_sign
                    } else {
                        -base_sign
                    };
                    if phase_val < 0.0 {
                        *s = zero;
                    }
                }
            } else {
                current_gen += 1;
                for y in 0..dim {
                    if visited_gen[y] == current_gen {
                        continue;
                    }
                    let partner = y ^ x_bits;
                    visited_gen[partner] = current_gen;

                    let a = sv[y];
                    let b = sv[partner];

                    let dot_y = (z_bits & y).count_ones() & 1;
                    let real_y = if dot_y == 0 { base_sign } else { -base_sign };
                    let gy_phase = i_factor * real_y;

                    let dot_p = (z_bits & partner).count_ones() & 1;
                    let real_p = if dot_p == 0 { base_sign } else { -base_sign };
                    let gp_phase = i_factor * real_p;

                    sv[y] = (a + b * gp_phase) * 0.5;
                    sv[partner] = (b + a * gy_phase) * 0.5;
                }
            }
        }

        let norm_sq: f64 = sv.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq > 1e-30 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for amp in &mut sv {
                *amp *= inv_norm;
            }
        }

        Ok(sv)
    }
}

/// Factored stabilizer simulation: dynamic sub-tableau partitioning.
pub struct FactoredStabilizerBackend {
    num_qubits: usize,
    qubit_to_sub: Vec<usize>,
    subs: Vec<Option<SubTableau>>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
}

impl FactoredStabilizerBackend {
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            qubit_to_sub: Vec::new(),
            subs: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn ensure_same_sub(&mut self, targets: &[usize]) -> usize {
        let first = self.qubit_to_sub[targets[0]];
        let mut need_merge: SmallVec<[usize; 4]> = SmallVec::new();
        for &q in &targets[1..] {
            let s = self.qubit_to_sub[q];
            if s != first && !need_merge.contains(&s) {
                need_merge.push(s);
            }
        }
        for other in need_merge {
            self.merge_subs(first, other);
        }
        first
    }

    fn merge_subs(&mut self, dst_idx: usize, src_idx: usize) {
        let src = self.subs[src_idx].take().unwrap();
        let dst = self.subs[dst_idx].as_ref().unwrap();

        let a = dst.n;
        let b = src.n;
        let total_n = a + b;
        let new_nw = total_n.div_ceil(64);
        let new_stride = 2 * new_nw;
        let total_rows = 2 * total_n + 1;

        let mut merged_qubits: SmallVec<[usize; 8]> = SmallVec::with_capacity(total_n);
        let mut dst_positions: SmallVec<[usize; 8]> = SmallVec::new();
        let mut src_positions: SmallVec<[usize; 8]> = SmallVec::new();

        let (mut di, mut si) = (0, 0);
        while di < a || si < b {
            if di < a && (si >= b || dst.qubits[di] < src.qubits[si]) {
                dst_positions.push(merged_qubits.len());
                merged_qubits.push(dst.qubits[di]);
                di += 1;
            } else {
                src_positions.push(merged_qubits.len());
                merged_qubits.push(src.qubits[si]);
                si += 1;
            }
        }

        let mut new_xz = vec![0u64; total_rows * new_stride];
        let mut new_phase = vec![false; total_rows];

        let dst_ref = self.subs[dst_idx].as_ref().unwrap();
        #[allow(clippy::needless_range_loop)]
        for r in 0..a {
            remap_row(
                &dst_ref.xz,
                dst_ref.stride(),
                r,
                &dst_positions,
                dst_ref.num_words,
                &mut new_xz,
                new_stride,
                r,
                new_nw,
            );
            new_phase[r] = dst_ref.phase[r];
        }
        for r in 0..b {
            remap_row(
                &src.xz,
                src.stride(),
                r,
                &src_positions,
                src.num_words,
                &mut new_xz,
                new_stride,
                a + r,
                new_nw,
            );
            new_phase[a + r] = src.phase[r];
        }
        for r in 0..a {
            remap_row(
                &dst_ref.xz,
                dst_ref.stride(),
                a + r,
                &dst_positions,
                dst_ref.num_words,
                &mut new_xz,
                new_stride,
                total_n + r,
                new_nw,
            );
            new_phase[total_n + r] = dst_ref.phase[a + r];
        }
        for r in 0..b {
            remap_row(
                &src.xz,
                src.stride(),
                b + r,
                &src_positions,
                src.num_words,
                &mut new_xz,
                new_stride,
                total_n + a + r,
                new_nw,
            );
            new_phase[total_n + a + r] = src.phase[b + r];
        }

        let d = self.subs[dst_idx].as_mut().unwrap();
        d.n = total_n;
        d.num_words = new_nw;
        d.xz = new_xz;
        d.phase = new_phase;
        d.qubits = merged_qubits;

        for &q in &src.qubits {
            self.qubit_to_sub[q] = dst_idx;
        }
    }

    fn try_split(&mut self, sub_idx: usize) -> bool {
        let sub = self.subs[sub_idx].as_ref().unwrap();
        let k = sub.n;
        if k <= 1 {
            return false;
        }

        let mut parent: Vec<usize> = (0..k).collect();
        let mut uf_rank = vec![0u8; k];
        let stride = sub.stride();
        let nw = sub.num_words;

        for r in k..2 * k {
            let base = r * stride;
            let mut first: Option<usize> = None;
            for q in 0..k {
                let w = q / 64;
                let bit = 1u64 << (q % 64);
                if (sub.xz[base + w] & bit) != 0 || (sub.xz[base + nw + w] & bit) != 0 {
                    if let Some(f) = first {
                        uf_union(&mut parent, &mut uf_rank, f, q);
                    } else {
                        first = Some(q);
                    }
                }
            }
        }

        let mut num_components = 0usize;
        let mut component_root = [usize::MAX; 64];
        let mut qubit_component = vec![0usize; k];
        #[allow(clippy::needless_range_loop)]
        for q in 0..k {
            let r = uf_find(&mut parent, q);
            let mut found = false;
            for c in 0..num_components {
                if component_root[c] == r {
                    qubit_component[q] = c;
                    found = true;
                    break;
                }
            }
            if !found {
                if num_components >= 64 {
                    return false;
                }
                component_root[num_components] = r;
                qubit_component[q] = num_components;
                num_components += 1;
            }
        }

        if num_components <= 1 {
            return false;
        }

        let sub = self.subs[sub_idx].take().unwrap();
        let mut comp_qubits: Vec<SmallVec<[usize; 8]>> = vec![SmallVec::new(); num_components];
        #[allow(clippy::needless_range_loop)]
        for q in 0..k {
            comp_qubits[qubit_component[q]].push(q);
        }

        let mut new_sub_indices = Vec::with_capacity(num_components);

        #[allow(clippy::needless_range_loop)]
        for c in 0..num_components {
            let local_qs = &comp_qubits[c];
            let cn = local_qs.len();
            let cnw = cn.div_ceil(64);
            let cstride = 2 * cnw;
            let ctotal = 2 * cn + 1;
            let mut cxz = vec![0u64; ctotal * cstride];
            let mut cphase = vec![false; ctotal];
            let mut cqubits: SmallVec<[usize; 8]> = SmallVec::with_capacity(cn);
            for &lq in local_qs {
                cqubits.push(sub.qubits[lq]);
            }

            let mut destab_idx = 0usize;
            let mut stab_idx = 0usize;

            for r in 0..k {
                let base = r * sub.stride();
                let mut has_support = false;
                for &lq in local_qs {
                    let w = lq / 64;
                    let bit = 1u64 << (lq % 64);
                    if (sub.xz[base + w] & bit) != 0
                        || (sub.xz[base + sub.num_words + w] & bit) != 0
                    {
                        has_support = true;
                        break;
                    }
                }
                if !has_support {
                    continue;
                }
                if destab_idx >= cn {
                    continue;
                }
                let dst_row = destab_idx;
                for (new_local, &old_local) in local_qs.iter().enumerate() {
                    let ow = old_local / 64;
                    let ob = old_local % 64;
                    let nww = new_local / 64;
                    let nb = new_local % 64;
                    if sub.xz[base + ow] & (1u64 << ob) != 0 {
                        cxz[dst_row * cstride + nww] |= 1u64 << nb;
                    }
                    if sub.xz[base + sub.num_words + ow] & (1u64 << ob) != 0 {
                        cxz[dst_row * cstride + cnw + nww] |= 1u64 << nb;
                    }
                }
                cphase[dst_row] = sub.phase[r];
                destab_idx += 1;
            }

            for r in k..2 * k {
                let base = r * sub.stride();
                let mut has_support = false;
                for &lq in local_qs {
                    let w = lq / 64;
                    let bit = 1u64 << (lq % 64);
                    if (sub.xz[base + w] & bit) != 0
                        || (sub.xz[base + sub.num_words + w] & bit) != 0
                    {
                        has_support = true;
                        break;
                    }
                }
                if !has_support {
                    continue;
                }
                if stab_idx >= cn {
                    continue;
                }
                let dst_row = cn + stab_idx;
                for (new_local, &old_local) in local_qs.iter().enumerate() {
                    let ow = old_local / 64;
                    let ob = old_local % 64;
                    let nww = new_local / 64;
                    let nb = new_local % 64;
                    if sub.xz[base + ow] & (1u64 << ob) != 0 {
                        cxz[dst_row * cstride + nww] |= 1u64 << nb;
                    }
                    if sub.xz[base + sub.num_words + ow] & (1u64 << ob) != 0 {
                        cxz[dst_row * cstride + cnw + nww] |= 1u64 << nb;
                    }
                }
                cphase[dst_row] = sub.phase[r];
                stab_idx += 1;
            }

            while destab_idx < cn {
                let q = destab_idx;
                let w = q / 64;
                let b = q % 64;
                cxz[destab_idx * cstride + w] |= 1u64 << b;
                destab_idx += 1;
            }
            while stab_idx < cn {
                let q = stab_idx;
                let w = q / 64;
                let b = q % 64;
                cxz[(cn + stab_idx) * cstride + cnw + w] |= 1u64 << b;
                stab_idx += 1;
            }

            let new_sub = SubTableau {
                n: cn,
                num_words: cnw,
                xz: cxz,
                phase: cphase,
                qubits: cqubits,
            };

            let slot = self.find_free_slot();
            new_sub_indices.push(slot);
            let global_qs: SmallVec<[usize; 8]> = new_sub.qubits.clone();
            self.subs[slot] = Some(new_sub);
            for &gq in &global_qs {
                self.qubit_to_sub[gq] = slot;
            }
        }

        true
    }

    fn find_free_slot(&mut self) -> usize {
        for (i, s) in self.subs.iter().enumerate() {
            if s.is_none() {
                return i;
            }
        }
        self.subs.push(None);
        self.subs.len() - 1
    }
}

impl Backend for FactoredStabilizerBackend {
    fn name(&self) -> &'static str {
        "factored-stabilizer"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.qubit_to_sub.clear();
        self.subs.clear();
        for q in 0..num_qubits {
            self.qubit_to_sub.push(q);
            self.subs.push(Some(SubTableau::new_single(q)));
        }
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => {
                let ss = self.ensure_same_sub(targets);
                let sub = self.subs[ss].as_mut().unwrap();
                let mut local = SmallVec::<[usize; 4]>::new();
                for &t in targets.iter() {
                    local.push(sub.local_qubit(t));
                }
                sub.dispatch_gate(gate, &local)?;
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                let ss = self.qubit_to_sub[*qubit];
                let sub = self.subs[ss].as_mut().unwrap();
                let local_q = sub.local_qubit(*qubit);
                let outcome = sub.measure(local_q, &mut self.rng);
                self.classical_bits[*classical_bit] = outcome;
                self.try_split(ss);
            }
            Instruction::Reset { qubit } => {
                let ss = self.qubit_to_sub[*qubit];
                let sub = self.subs[ss].as_mut().unwrap();
                let local_q = sub.local_qubit(*qubit);
                sub.reset_qubit(local_q, &mut self.rng)?;
                self.try_split(ss);
            }
            Instruction::Barrier { .. } => {}
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    let ss = self.ensure_same_sub(targets);
                    let sub = self.subs[ss].as_mut().unwrap();
                    let mut local = SmallVec::<[usize; 4]>::new();
                    for &t in targets.iter() {
                        local.push(sub.local_qubit(t));
                    }
                    sub.dispatch_gate(gate, &local)?;
                }
            }
        }
        Ok(())
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let active: Vec<&SubTableau> = self.subs.iter().filter_map(|s| s.as_ref()).collect();

        if active.len() == 1 && active[0].n == self.num_qubits {
            return active[0].compute_probabilities();
        }

        let blocks: Vec<(Vec<f64>, Vec<usize>)> = active
            .iter()
            .map(|sub| {
                let probs = sub
                    .compute_probabilities()
                    .unwrap_or_else(|_| vec![0.0; 1 << sub.n]);
                let qubits: Vec<usize> = sub.qubits.to_vec();
                (probs, qubits)
            })
            .collect();

        Ok(crate::sim::merge_probabilities(&blocks, self.num_qubits))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_fused_gates(&self) -> bool {
        false
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        let ss = self.qubit_to_sub[qubit];
        let sub = self.subs[ss].as_mut().unwrap();
        let local_q = sub.local_qubit(qubit);
        sub.reset_qubit(local_q, &mut self.rng)?;
        self.try_split(ss);
        Ok(())
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let active: Vec<&SubTableau> = self.subs.iter().filter_map(|s| s.as_ref()).collect();

        if active.len() == 1 && active[0].n == self.num_qubits {
            return active[0].compute_statevector();
        }

        Err(PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "statevector export for multiple sub-tableaux".to_string(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn remap_row(
    src_xz: &[u64],
    src_stride: usize,
    src_row: usize,
    positions: &[usize],
    src_nw: usize,
    dst_xz: &mut [u64],
    dst_stride: usize,
    dst_row: usize,
    dst_nw: usize,
) {
    let src_base = src_row * src_stride;
    let dst_base = dst_row * dst_stride;
    let src_n = positions.len();
    #[allow(clippy::needless_range_loop)]
    for local in 0..src_n {
        let sw = local / 64;
        let sb = local % 64;
        let merged = positions[local];
        let dw = merged / 64;
        let db = merged % 64;
        if src_xz[src_base + sw] & (1u64 << sb) != 0 {
            dst_xz[dst_base + dw] |= 1u64 << db;
        }
        if src_xz[src_base + src_nw + sw] & (1u64 << sb) != 0 {
            dst_xz[dst_base + dst_nw + dw] |= 1u64 << db;
        }
    }
}

fn uf_find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = uf_find(parent, parent[x]);
    }
    parent[x]
}

fn uf_union(parent: &mut [usize], rank: &mut [u8], x: usize, y: usize) {
    let rx = uf_find(parent, x);
    let ry = uf_find(parent, y);
    if rx == ry {
        return;
    }
    if rank[rx] < rank[ry] {
        parent[rx] = ry;
    } else if rank[rx] > rank[ry] {
        parent[ry] = rx;
    } else {
        parent[ry] = rx;
        rank[rx] += 1;
    }
}

fn gauss_eliminate_x(
    xz: &mut [u64],
    phase: &mut [bool],
    n: usize,
    nw: usize,
    stride: usize,
) -> (usize, Vec<usize>) {
    let mut col_map: Vec<usize> = Vec::new();
    let mut k = 0usize;

    for col in 0..n {
        let word = col / 64;
        let bit = 1u64 << (col % 64);

        let mut pivot = None;
        for row in k..n {
            if xz[(n + row) * stride + word] & bit != 0 {
                pivot = Some(row);
                break;
            }
        }

        let row = match pivot {
            Some(r) => r,
            None => continue,
        };

        if row != k {
            let a_off = (n + k) * stride;
            let b_off = (n + row) * stride;
            for w in 0..stride {
                xz.swap(a_off + w, b_off + w);
            }
            phase.swap(n + k, n + row);
        }

        for other in 0..n {
            if other == k {
                continue;
            }
            if xz[(n + other) * stride + word] & bit != 0 {
                let src: Vec<u64> = xz[(n + k) * stride..(n + k + 1) * stride].to_vec();
                let sp = phase[n + k];
                let dst = &mut xz[(n + other) * stride..(n + other + 1) * stride];
                let (dx, dz) = dst.split_at_mut(nw);
                let initial = if sp { 2u64 } else { 0 } + if phase[n + other] { 2u64 } else { 0 };
                let sum = rowmul_words(dx, &mut dz[..nw], &src[..nw], &src[nw..2 * nw], initial);
                phase[n + other] = (sum & 3) >= 2;
            }
        }

        col_map.push(col);
        k += 1;
    }

    (k, col_map)
}

fn solve_diagonal_seed(
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    stride: usize,
    _col_map: &[usize],
    k: usize,
) -> usize {
    let d = n - k;
    if d == 0 {
        return 0;
    }

    let mut z_rows = Vec::with_capacity(d * nw);
    let mut phases = Vec::with_capacity(d);
    for g in k..n {
        let row = n + g;
        let base = row * stride + nw;
        z_rows.extend_from_slice(&xz[base..base + nw]);
        phases.push(phase[row]);
    }

    let mut pivot_col = vec![usize::MAX; d];
    let mut available_cols: Vec<usize> = (0..n).collect();

    for row in 0..d {
        let row_off = row * nw;
        let mut found = None;
        for (ci, &col) in available_cols.iter().enumerate() {
            if (z_rows[row_off + col / 64] >> (col % 64)) & 1 == 1 {
                found = Some(ci);
                break;
            }
        }
        if let Some(ci) = found {
            let col = available_cols.swap_remove(ci);
            pivot_col[row] = col;
            let w = col / 64;
            let b = col % 64;
            let pivot_z: Vec<u64> = z_rows[row_off..row_off + nw].to_vec();
            let pivot_phase = phases[row];
            #[allow(clippy::needless_range_loop)]
            for other in 0..d {
                if other == row {
                    continue;
                }
                let other_off = other * nw;
                if (z_rows[other_off + w] >> b) & 1 == 1 {
                    for ww in 0..nw {
                        z_rows[other_off + ww] ^= pivot_z[ww];
                    }
                    phases[other] ^= pivot_phase;
                }
            }
        }
    }

    let mut seed = 0usize;
    for row in 0..d {
        if pivot_col[row] != usize::MAX && phases[row] {
            seed |= 1 << pivot_col[row];
        }
    }
    seed
}
