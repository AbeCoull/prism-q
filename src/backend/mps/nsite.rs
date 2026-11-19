//! Gates on more than two adjacent MPS sites: contraction of the block into one
//! tensor, the gate applied across it, and the sweep of SVDs that splits it back.

use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::MIN_BOND_FOR_PAR;
use super::svd::{
    discards_resolvable_weight, fill_isometry_site_data, fill_scaled_vt_data, svd,
    truncated_svd_rank, widest_block_cut,
};
use super::{MpsBackend, SiteTensor, ZERO};
use crate::error::Result;
use crate::gates::Gate;

impl MpsBackend {
    /// Contract N adjacent site tensors into Θ with shape (χ_left, 2^N, χ_right).
    ///
    /// Returns (theta, bl, br) where theta is indexed as Θ[α, s, β] at
    /// `α * (dim * br) + s * br + β`, with dim = 2^n.
    fn contract_n_sites(&self, start: usize, n: usize) -> (Vec<Complex64>, usize, usize) {
        debug_assert!(n >= 2);
        let bl = self.sites[start].bond_left;
        let br = self.sites[start + n - 1].bond_right;

        let t0 = &self.sites[start];
        let mut theta = t0.data.clone();
        let mut cur_dim = 2usize;
        let mut cur_right = t0.bond_right;
        let mut new_theta = Vec::new();

        for k in 1..n {
            let t = &self.sites[start + k];
            let bond_mid = cur_right;
            let next_right = t.bond_right;
            let new_dim = cur_dim * 2;

            let nsite_chunk = new_dim * next_right;
            let needed = bl * nsite_chunk;
            new_theta.clear();
            new_theta.resize(needed, ZERO);
            let t_data = &t.data;
            let theta_ref = &theta;

            #[cfg(feature = "parallel")]
            if bl >= MIN_BOND_FOR_PAR {
                new_theta
                    .par_chunks_mut(nsite_chunk)
                    .enumerate()
                    .for_each(|(alpha, chunk)| {
                        for s in 0..cur_dim {
                            for j in 0..2 {
                                for beta in 0..next_right {
                                    let mut val = ZERO;
                                    for gamma in 0..bond_mid {
                                        val += theta_ref
                                            [alpha * (cur_dim * bond_mid) + s * bond_mid + gamma]
                                            * t_data
                                                [gamma * (2 * next_right) + j * next_right + beta];
                                    }
                                    let new_s = s * 2 + j;
                                    chunk[new_s * next_right + beta] = val;
                                }
                            }
                        }
                    });
            } else {
                for alpha in 0..bl {
                    for s in 0..cur_dim {
                        for j in 0..2 {
                            for beta in 0..next_right {
                                let mut val = ZERO;
                                for gamma in 0..bond_mid {
                                    val += theta_ref
                                        [alpha * (cur_dim * bond_mid) + s * bond_mid + gamma]
                                        * t_data[gamma * (2 * next_right) + j * next_right + beta];
                                }
                                let new_s = s * 2 + j;
                                new_theta[alpha * nsite_chunk + new_s * next_right + beta] = val;
                            }
                        }
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            for alpha in 0..bl {
                for s in 0..cur_dim {
                    for j in 0..2 {
                        for beta in 0..next_right {
                            let mut val = ZERO;
                            for gamma in 0..bond_mid {
                                val += theta_ref
                                    [alpha * (cur_dim * bond_mid) + s * bond_mid + gamma]
                                    * t_data[gamma * (2 * next_right) + j * next_right + beta];
                            }
                            let new_s = s * 2 + j;
                            new_theta[alpha * nsite_chunk + new_s * next_right + beta] = val;
                        }
                    }
                }
            }
            std::mem::swap(&mut theta, &mut new_theta);
            cur_dim = new_dim;
            cur_right = next_right;
        }

        (theta, bl, br)
    }

    /// Apply a 2^N × 2^N gate matrix (row-major) to a contracted tensor in-place.
    fn apply_gate_to_theta(
        theta: &[Complex64],
        gate: &[Complex64],
        dim: usize,
        bl: usize,
        br: usize,
    ) -> Vec<Complex64> {
        let gate_chunk = dim * br;
        let mut out = vec![ZERO; bl * gate_chunk];

        #[cfg(feature = "parallel")]
        if bl >= MIN_BOND_FOR_PAR {
            out.par_chunks_mut(gate_chunk)
                .enumerate()
                .for_each(|(alpha, chunk)| {
                    for sp in 0..dim {
                        for beta in 0..br {
                            let mut val = ZERO;
                            for s in 0..dim {
                                val +=
                                    gate[sp * dim + s] * theta[alpha * gate_chunk + s * br + beta];
                            }
                            chunk[sp * br + beta] = val;
                        }
                    }
                });
            return out;
        }

        for alpha in 0..bl {
            for sp in 0..dim {
                for beta in 0..br {
                    let mut val = ZERO;
                    for s in 0..dim {
                        val += gate[sp * dim + s] * theta[alpha * gate_chunk + s * br + beta];
                    }
                    out[alpha * gate_chunk + sp * br + beta] = val;
                }
            }
        }
        out
    }

    /// Decompose a contracted tensor back into N site tensors via N-1 SVDs.
    ///
    /// Left-to-right sweep: at each step, reshape the remainder to
    /// (χ × 2) rows × (remaining_dim × χ_right) cols, SVD, truncate,
    /// extract left site from U, and continue with S·V† as the new remainder.
    fn decompose_n_sites(
        &mut self,
        theta: &[Complex64],
        start: usize,
        n: usize,
        bl: usize,
        br: usize,
        judge: bool,
    ) -> bool {
        let mut remaining = theta.to_vec();
        let mut cur_bl = bl;
        let mut remaining_dim = 1usize << n;
        let mut mat = Vec::new();
        let mut new_remaining = Vec::new();

        for k in 0..n - 1 {
            remaining_dim /= 2;
            let rows = cur_bl * 2;
            let cols = remaining_dim * br;

            // Reshape to column-major matrix for SVD: M[r, c] at mat[c * rows + r]
            // r = α * 2 + i, c = s_rest * br + β
            mat.clear();
            mat.resize(rows * cols, ZERO);
            #[cfg(feature = "parallel")]
            if cur_bl >= MIN_BOND_FOR_PAR * 2 {
                let rem = &remaining;
                let rd = remaining_dim;
                mat.par_iter_mut().enumerate().for_each(|(out_idx, elem)| {
                    let c = out_idx / rows;
                    let r = out_idx % rows;
                    let alpha = r / 2;
                    let i = r & 1;
                    let s_rest = c / br;
                    let beta = c % br;
                    let combined_s = i * rd + s_rest;
                    *elem = rem[alpha * (2 * rd * br) + combined_s * br + beta];
                });
            } else {
                for alpha in 0..cur_bl {
                    for i in 0..2 {
                        for s_rest in 0..remaining_dim {
                            for beta in 0..br {
                                let r = alpha * 2 + i;
                                let c = s_rest * br + beta;
                                let combined_s = i * remaining_dim + s_rest;
                                mat[c * rows + r] = remaining
                                    [alpha * (2 * remaining_dim * br) + combined_s * br + beta];
                            }
                        }
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            for alpha in 0..cur_bl {
                for i in 0..2 {
                    for s_rest in 0..remaining_dim {
                        for beta in 0..br {
                            let r = alpha * 2 + i;
                            let c = s_rest * br + beta;
                            let combined_s = i * remaining_dim + s_rest;
                            mat[c * rows + r] = remaining
                                [alpha * (2 * remaining_dim * br) + combined_s * br + beta];
                        }
                    }
                }
            }

            let svd_result = svd(&mat, rows, cols);
            let chi_new = truncated_svd_rank(&svd_result.s, self.svd_epsilon, self.max_bond_dim);
            if judge && discards_resolvable_weight(&svd_result.s, chi_new) {
                return false;
            }
            self.record_truncation(&svd_result.s, chi_new);
            self.bond_high_water = self.bond_high_water.max(chi_new);

            // Extract left site from U: shape (cur_bl, 2, chi_new)
            let mut left_data = std::mem::take(&mut self.sites[start + k].data);
            fill_isometry_site_data(
                &mut left_data,
                &svd_result.u,
                svd_result.u_rows,
                cur_bl,
                chi_new,
            );

            self.sites[start + k] = SiteTensor {
                bond_left: cur_bl,
                bond_right: chi_new,
                data: left_data,
            };

            if k < n - 2 {
                // Build remainder from S·V†: shape (chi_new, remaining_dim, br)
                fill_scaled_vt_data(&mut new_remaining, &svd_result, chi_new, remaining_dim, br);
                std::mem::swap(&mut remaining, &mut new_remaining);
                cur_bl = chi_new;
            } else {
                // Last site: S·V† reshaped to (chi_new, 2, br)
                let mut right_data = std::mem::take(&mut self.sites[start + n - 1].data);
                fill_scaled_vt_data(&mut right_data, &svd_result, chi_new, 2, br);
                self.sites[start + n - 1] = SiteTensor {
                    bond_left: chi_new,
                    bond_right: br,
                    data: right_data,
                };
            }
        }
        true
    }

    /// Reject a dense `2^n x 2^n` gate matrix over the workspace budget before
    /// it is built.
    ///
    /// The N-site path holds two of them at peak, the assembled matrix and the
    /// role-reordered copy, and both are allocated before
    /// [`Self::apply_adjacent_n_qubit`] reaches its own theta check.
    pub(super) fn check_n_qubit_gate_matrix(&self, n: usize) -> Result<()> {
        // Rejected outright rather than by comparison: `2^(2n + 1)` is not
        // representable here, and the cap is itself `u128::MAX` when memory
        // detection is unavailable, so a comparison would pass and leave the
        // caller to shift `1usize` by more than its width.
        if n >= 63 {
            return Err(crate::backend::workspace_allocation_error(
                "mps",
                "multi-qubit gate matrix",
                u128::MAX,
                &self.limits,
            ));
        }
        let elements = 2u128 << (2 * n);
        if elements > self.workspace_cap {
            return Err(crate::backend::workspace_allocation_error(
                "mps",
                "multi-qubit gate matrix",
                elements,
                &self.limits,
            ));
        }
        Ok(())
    }

    pub(super) fn apply_adjacent_n_qubit(
        &mut self,
        gate: &[Complex64],
        dim: usize,
        start_site: usize,
    ) -> Result<()> {
        let n = dim.trailing_zeros() as usize; // dim = 2^n
        // Contraction, gate application, and decomposition each hold a
        // (bl, 2^n, br) tensor, with the decomposition remainder and SVD input
        // making about four live at peak.
        let bl = self.sites[start_site].bond_left;
        let br = self.sites[start_site + n - 1].bond_right;
        let workspace = 4u128 * dim as u128 * bl as u128 * br as u128;
        if workspace > self.workspace_cap {
            return Err(crate::backend::workspace_allocation_error(
                "mps",
                "multi-qubit gate application",
                workspace,
                &self.limits,
            ));
        }
        if !self.cut_block(gate, dim, start_site, n) {
            self.establish_center(start_site);
            let gauged = self.cut_block(gate, dim, start_site, n);
            debug_assert!(gauged, "a block cut in gauge judged its spectrum");
        }
        Ok(())
    }

    /// One attempt at [`Self::apply_adjacent_n_qubit`]. Returns `false`,
    /// having put the block's sites back, when the chain has no center and a
    /// cut of the decomposition drops a value the factorization resolves.
    ///
    /// The decomposition sweeps left to right and leaves the weight on the
    /// block's last site, so a center anywhere in the block ends there. One
    /// outside it would leave the sites between non-orthogonal, and the
    /// block's own cuts would be the ones paying for it. The two-site kernel
    /// judges its cut before writing anything, where the spectrum that shows
    /// it here is only reached once the sites before that cut are written, so
    /// the retry restores them from a copy. Only a threshold can drop a value
    /// on the ungauged arm, since a cut that could reach the cap gauges
    /// instead, so at a threshold of zero there is nothing to judge and no
    /// copy is taken.
    pub(super) fn cut_block(
        &mut self,
        gate: &[Complex64],
        dim: usize,
        start_site: usize,
        n: usize,
    ) -> bool {
        let block = start_site..start_site + n;
        let bl = self.sites[start_site].bond_left;
        let br = self.sites[block.end - 1].bond_right;
        let maintained = self.center.is_some() || self.cut_can_lose(widest_block_cut(bl, br, n));
        if maintained {
            match self.center {
                Some(at) => self.move_center(at.clamp(start_site, block.end - 1)),
                None => self.establish_center(start_site),
            }
        }
        let judging = !maintained && self.svd_epsilon > 0.0;

        let (theta, bl, br) = self.contract_n_sites(start_site, n);
        let theta_prime = Self::apply_gate_to_theta(&theta, gate, dim, bl, br);
        if !judging {
            self.decompose_n_sites(&theta_prime, start_site, n, bl, br, false);
            if maintained {
                self.center = Some(block.end - 1);
            }
            return true;
        }

        let saved = self.sites[block.clone()].to_vec();
        let booked = self.truncation_discarded;
        let high_water = self.bond_high_water;
        if self.decompose_n_sites(&theta_prime, start_site, n, bl, br, true) {
            return true;
        }
        self.sites[block].clone_from_slice(&saved);
        self.truncation_discarded = booked;
        self.bond_high_water = high_water;
        false
    }

    /// Apply an N-qubit gate to arbitrary (possibly non-adjacent) qubits.
    ///
    /// SWAP-routes qubits into a contiguous block, applies the gate, then
    /// reverses the SWAPs.
    pub(super) fn apply_n_qubit_gate(
        &mut self,
        gate: &[Complex64],
        dim: usize,
        qubits: &[usize],
    ) -> Result<()> {
        let n = qubits.len();
        debug_assert_eq!(dim, 1 << n);

        let mut indexed: Vec<(usize, usize)> = qubits
            .iter()
            .enumerate()
            .map(|(role, &pos)| (pos, role))
            .collect();
        indexed.sort_unstable_by_key(|&(pos, _)| pos);

        let sorted_positions: Vec<usize> = indexed.iter().map(|&(pos, _)| pos).collect();
        let start = sorted_positions[0];

        let contiguous = sorted_positions.last().unwrap() - start + 1 == n;

        if contiguous {
            let qubit_order: Vec<usize> = indexed.iter().map(|&(_, role)| role).collect();
            let reordered_gate = Self::reorder_n_gate(gate, dim, &qubit_order, n);
            self.apply_adjacent_n_qubit(&reordered_gate, dim, start)?;
        } else {
            let swap_mat = Gate::Swap.matrix_4x4();
            let mut current_positions = sorted_positions;
            let mut swap_log: Vec<usize> = Vec::new();

            for i in 1..n {
                let target_pos = start + i;
                while current_positions[i] > target_pos {
                    let s = current_positions[i] - 1;
                    self.apply_adjacent_two_qubit(&swap_mat, s, true)?;
                    swap_log.push(s);
                    // Update any tracked qubit that was displaced
                    for cp in &mut current_positions[(i + 1)..n] {
                        if *cp == s {
                            *cp = s + 1;
                        }
                    }
                    current_positions[i] -= 1;
                }
            }

            let qubit_order: Vec<usize> = indexed.iter().map(|&(_, role)| role).collect();
            let reordered_gate = Self::reorder_n_gate(gate, dim, &qubit_order, n);
            self.apply_adjacent_n_qubit(&reordered_gate, dim, start)?;

            for &s in swap_log.iter().rev() {
                self.apply_adjacent_two_qubit(&swap_mat, s, true)?;
            }
        }
        Ok(())
    }

    /// Reorder an N-qubit gate matrix to match the physical qubit ordering.
    ///
    /// `qubit_order[block_pos]` = the MCU role index for that block position.
    /// The original gate assumes role order 0,1,...,N-1 (controls then target).
    fn reorder_n_gate(
        gate: &[Complex64],
        dim: usize,
        qubit_order: &[usize],
        n: usize,
    ) -> Vec<Complex64> {
        let identity_order = qubit_order.iter().enumerate().all(|(i, &r)| r == i);
        if identity_order {
            return gate.to_vec();
        }

        let mut out = vec![ZERO; dim * dim];
        for s_in in 0..dim {
            for s_out in 0..dim {
                let mut logical_in = 0usize;
                let mut logical_out = 0usize;
                for (pos, &role) in qubit_order.iter().enumerate() {
                    let bit_pos = n - 1 - pos; // bit position for block pos
                    let role_bit = n - 1 - role; // bit position in role order
                    if s_in & (1 << bit_pos) != 0 {
                        logical_in |= 1 << role_bit;
                    }
                    if s_out & (1 << bit_pos) != 0 {
                        logical_out |= 1 << role_bit;
                    }
                }
                out[s_out * dim + s_in] = gate[logical_out * dim + logical_in];
            }
        }
        out
    }
}
