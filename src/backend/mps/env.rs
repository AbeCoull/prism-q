//! Environment contractions over the MPS chain: overlaps, Pauli expectations, and
//! the left and right environments the reductions read.

use num_complex::Complex64;

use super::{MpsBackend, ONE, ZERO};
use crate::backend::gemm::{gemm_nh, gemm_nn, gemm_tn};
use crate::error::Result;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Largest `bond_left * bond_right` at which a single-site environment step keeps
/// the four-index sum and its zero skip rather than two dense passes.
const DENSE_ENV_MAX_BONDS: usize = 4;

/// Pauli axis for [`MpsBackend::pauli_expectation`]. Identity factors
/// are conveyed by omission from the factor list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsPauliAxis {
    X,
    Y,
    Z,
}

pub(super) fn mps_pauli_factors(observable: &[PauliTerm]) -> Vec<(usize, MpsPauliAxis)> {
    observable
        .iter()
        .map(|term| {
            let axis = match term.axis {
                PauliAxis::X => MpsPauliAxis::X,
                PauliAxis::Y => MpsPauliAxis::Y,
                PauliAxis::Z => MpsPauliAxis::Z,
            };
            (term.qubit, axis)
        })
        .collect()
}

/// Replace each `[j, α']` pair of rows in `tmp`, laid out `[β, j, α']` with
/// rows of `br`, by `Σ_j tmp[β, j, α'] · M[j, k]` for the Pauli matrix `M`.
pub(super) fn fold_pauli_into_rows(tmp: &mut [Complex64], br: usize, axis: Option<MpsPauliAxis>) {
    let Some(axis) = axis else {
        return;
    };
    for pair in tmp.chunks_exact_mut(2 * br) {
        let (zero, one) = pair.split_at_mut(br);
        match axis {
            MpsPauliAxis::X => zero.swap_with_slice(one),
            MpsPauliAxis::Y => {
                for (z, o) in zero.iter_mut().zip(one) {
                    let (z0, o0) = (*z, *o);
                    *z = Complex64::new(-o0.im, o0.re);
                    *o = Complex64::new(z0.im, -z0.re);
                }
            }
            MpsPauliAxis::Z => {
                for o in one {
                    *o = -*o;
                }
            }
        }
    }
}

pub(super) fn identity_env(dim: usize) -> Vec<Complex64> {
    let mut env = vec![ZERO; dim * dim];
    for i in 0..dim {
        env[i * dim + i] = ONE;
    }
    env
}

impl MpsBackend {
    /// `<self|other>` for two chains with the same site layout; an error when
    /// the qubit counts or layouts differ.
    ///
    /// Contracts left-to-right through the environment tensor in two passes
    /// per site, `O(n χ³)` against the naive four-index sum.
    pub fn inner_product(&self, other: &Self) -> Result<Complex64> {
        self.inner_product_with_scratch(other, &mut Vec::new(), &mut Vec::new())
    }

    /// [`Self::inner_product`] with caller-owned scratch. `tmp` and
    /// `next_env` are cleared and resized per site, so a caller evaluating
    /// many overlaps pays one small environment allocation per call instead
    /// of two per site.
    pub(crate) fn inner_product_with_scratch(
        &self,
        other: &Self,
        tmp: &mut Vec<Complex64>,
        next_env: &mut Vec<Complex64>,
    ) -> Result<Complex64> {
        if self.num_qubits != other.num_qubits {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "MPS inner product requires equal qubit counts; got {} vs {}",
                    self.num_qubits, other.num_qubits,
                ),
            });
        }
        if self.logical_to_site != other.logical_to_site {
            return Err(crate::error::PrismError::InvalidParameter {
                message: "MPS inner product requires identical logical_to_site permutations"
                    .to_string(),
            });
        }
        if self.num_qubits == 0 {
            return Ok(ONE);
        }

        let mut env: Vec<Complex64> = vec![ONE; 1];
        let mut env_cols: usize = 1;

        for i in 0..self.num_qubits {
            let a = &self.sites[i];
            let b = &other.sites[i];
            let bl_a = a.bond_left;
            let br_a = a.bond_right;
            let bl_b = b.bond_left;
            let br_b = b.bond_right;

            // Step 1: tmp[β, j, α'] = Σ_α env[α, β] · conj(A[α, j, α'])
            tmp.clear();
            tmp.resize(bl_b * 2 * br_a, ZERO);
            for alpha in 0..bl_a {
                for j in 0..2 {
                    for alpha_p in 0..br_a {
                        let a_val = a.data[a.idx(alpha, j, alpha_p)].conj();
                        if a_val == ZERO {
                            continue;
                        }
                        for beta in 0..bl_b {
                            let e_ab = env[alpha * env_cols + beta];
                            tmp[beta * (2 * br_a) + j * br_a + alpha_p] += a_val * e_ab;
                        }
                    }
                }
            }

            // Step 2: next_env[α', β'] = Σ_{β, j} tmp[β, j, α'] · B[β, j, β']
            next_env.clear();
            next_env.resize(br_a * br_b, ZERO);
            for beta in 0..bl_b {
                for j in 0..2 {
                    for beta_p in 0..br_b {
                        let b_val = b.data[b.idx(beta, j, beta_p)];
                        if b_val == ZERO {
                            continue;
                        }
                        for alpha_p in 0..br_a {
                            let t = tmp[beta * (2 * br_a) + j * br_a + alpha_p];
                            next_env[alpha_p * br_b + beta_p] += t * b_val;
                        }
                    }
                }
            }

            std::mem::swap(&mut env, next_env);
            env_cols = br_b;
        }

        Ok(env[0])
    }

    /// `⟨ψ|P|ψ⟩` for the Pauli string with one factor per listed
    /// `(qubit, axis)`, identity elsewhere. Walks the contraction once
    /// (`O(N·χ³)`) and applies each site's Pauli matrix inline, with no MPS
    /// clone and no gate apply pass.
    /// Duplicate factors on the same qubit are rejected.
    pub fn pauli_expectation(&self, pauli_factors: &[(usize, MpsPauliAxis)]) -> Result<Complex64> {
        if self.num_qubits == 0 {
            return Ok(ONE);
        }

        let site_axis = self.pauli_site_axes(pauli_factors)?;

        let mut env: Vec<Complex64> = vec![ONE; 1];
        let mut next = Vec::new();
        let mut tmp = Vec::new();
        for (site, &axis) in site_axis.iter().enumerate() {
            self.contract_pauli_env(site, &env, axis, &mut tmp, &mut next);
            std::mem::swap(&mut env, &mut next);
        }

        Ok(env[0])
    }

    /// Site index of each requested Pauli factor, `None` where the string acts
    /// as identity. Rejects out-of-range qubits and duplicate factors.
    pub(super) fn pauli_site_axes(
        &self,
        pauli_factors: &[(usize, MpsPauliAxis)],
    ) -> Result<Vec<Option<MpsPauliAxis>>> {
        let mut site_axis: Vec<Option<MpsPauliAxis>> = vec![None; self.num_qubits];
        for &(qubit, axis) in pauli_factors {
            if qubit >= self.num_qubits {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!(
                        "MPS Pauli expectation: qubit {qubit} out of range (n={})",
                        self.num_qubits
                    ),
                });
            }
            let site = self.logical_to_site[qubit];
            if site_axis[site].is_some() {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!("MPS Pauli expectation: duplicate factor on qubit {qubit}"),
                });
            }
            site_axis[site] = Some(axis);
        }
        Ok(site_axis)
    }

    /// Absorb one site into the `⟨ψ|P|ψ⟩` environment, applying `axis` as the
    /// site's Pauli matrix (identity when `None`), as two matrix products.
    ///
    /// `env` is the `bond_left` square indexed `[bra bond, ket bond]`; `out`
    /// is resized to the `bond_right` square indexed the same way. `tmp` is
    /// scratch held across a sweep.
    pub(super) fn contract_pauli_env(
        &self,
        site: usize,
        env: &[Complex64],
        axis: Option<MpsPauliAxis>,
        tmp: &mut Vec<Complex64>,
        out: &mut Vec<Complex64>,
    ) {
        let a = &self.sites[site];
        let bl = a.bond_left;
        let br = a.bond_right;
        debug_assert_eq!(env.len(), bl * bl);

        // tmp[β, j, α'] = Σ_α env[α, β] · conj(A[α, j, α']), then the Pauli
        // matrix folded over j.
        tmp.resize(bl * 2 * br, ZERO);
        gemm_tn::<true>(tmp, env, &a.data, bl, bl, 2 * br);
        fold_pauli_into_rows(tmp, br, axis);

        // out[α', β'] = Σ_{β, k} tmp[β, k, α'] · A[β, k, β']
        out.resize(br * br, ZERO);
        gemm_tn::<false>(out, tmp, &a.data, 2 * bl, br, br);
    }

    /// Identity environments left of every site, plus `⟨ψ|ψ⟩` as the last
    /// entry, in the `[bra bond, ket bond]` convention of
    /// [`MpsBackend::contract_pauli_env`].
    ///
    /// Entry `s` is the contraction of sites `0..s`, so it has `bond_left(s)`
    /// columns and entry `num_qubits` is the `1x1` norm.
    pub(super) fn pauli_env_prefixes(&self) -> Vec<Vec<Complex64>> {
        let mut prefixes = Vec::with_capacity(self.num_qubits + 1);
        prefixes.push(vec![ONE; 1]);
        let mut tmp = Vec::new();
        for site in 0..self.num_qubits {
            let mut next = Vec::new();
            self.contract_pauli_env(site, prefixes.last().unwrap(), None, &mut tmp, &mut next);
            prefixes.push(next);
        }
        prefixes
    }

    /// Left environment of `site`, contracted from `from`; every site before
    /// `from` must be a left isometry, so the sweep starts there at identity.
    pub(super) fn compute_left_env(&self, from: usize, site: usize) -> Vec<Complex64> {
        let mut env_dim = self.sites[from].bond_left;
        let mut env = identity_env(env_dim);
        let mut tmp = Vec::new();

        for s in from..site {
            let t = &self.sites[s];
            let bl = t.bond_left;
            let br = t.bond_right;
            debug_assert_eq!(env_dim, bl);

            if bl * br > DENSE_ENV_MAX_BONDS {
                env = self.contract_left_env(s, &env, &mut tmp);
                env_dim = br;
                continue;
            }

            let mut new_env = vec![ZERO; br * br];
            for gamma in 0..br {
                for gamma_p in 0..br {
                    let mut val = ZERO;
                    for alpha in 0..bl {
                        for alpha_p in 0..bl {
                            let env_val = env[alpha * bl + alpha_p];
                            if env_val == ZERO {
                                continue;
                            }
                            for i in 0..2 {
                                val += env_val
                                    * t.data[t.idx(alpha, i, gamma)]
                                    * t.data[t.idx(alpha_p, i, gamma_p)].conj();
                            }
                        }
                    }
                    new_env[gamma * br + gamma_p] = val;
                }
            }
            env = new_env;
            env_dim = br;
        }
        env
    }

    /// Absorb one site into a left environment, the mirror of
    /// [`Self::contract_right_env`].
    fn contract_left_env(
        &self,
        site: usize,
        env: &[Complex64],
        tmp: &mut Vec<Complex64>,
    ) -> Vec<Complex64> {
        let t = &self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;

        // tmp[α', i, γ] = Σ_α env[α, α'] · A[α, i, γ]
        tmp.clear();
        tmp.resize(bl * 2 * br, ZERO);
        for alpha in 0..bl {
            for alpha_p in 0..bl {
                let e = env[alpha * bl + alpha_p];
                if e == ZERO {
                    continue;
                }
                for i in 0..2 {
                    let src = t.idx(alpha, i, 0);
                    let dst = (alpha_p * 2 + i) * br;
                    for gamma in 0..br {
                        tmp[dst + gamma] += e * t.data[src + gamma];
                    }
                }
            }
        }

        // new_env[γ, γ'] = Σ_{α', i} tmp[α', i, γ] · conj(A[α', i, γ'])
        let mut new_env = vec![ZERO; br * br];
        for alpha_p in 0..bl {
            for i in 0..2 {
                let row = (alpha_p * 2 + i) * br;
                let src = t.idx(alpha_p, i, 0);
                for gamma in 0..br {
                    let x = tmp[row + gamma];
                    if x == ZERO {
                        continue;
                    }
                    for gamma_p in 0..br {
                        new_env[gamma * br + gamma_p] += x * t.data[src + gamma_p].conj();
                    }
                }
            }
        }
        new_env
    }

    /// Absorb one site into a right environment as two matrix products:
    /// `O(χ³)` instead of the four-index sum's `O(χ⁴)`. `out` is resized to the
    /// `bond_left` square, and `tmp` carries the intermediate across a sweep so
    /// the inner buffer is allocated once rather than per site.
    ///
    /// [`Self::compute_right_env`] calls this only past
    /// [`DENSE_ENV_MAX_BONDS`]. Below it the chains a projection leaves behind
    /// carry zeros, and the four-index form's `env_val == ZERO` skip beat dense
    /// `O(χ³)` arithmetic by 14% on `mps/hotspots/measure_reset_32q_r3` (496 µs
    /// against 574 µs) while that row still went through the environments.
    pub(super) fn contract_right_env(
        &self,
        site: usize,
        env: &[Complex64],
        tmp: &mut Vec<Complex64>,
        out: &mut Vec<Complex64>,
    ) {
        let t = &self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;
        debug_assert_eq!(env.len(), br * br);

        // tmp[α, i, β'] = Σ_β A[α, i, β] · env[β, β']
        tmp.resize(bl * 2 * br, ZERO);
        gemm_nn(tmp, &t.data, env, 2 * bl, br, br);

        // out[α, α'] = Σ_{i, β'} tmp[α, i, β'] · conj(A[α', i, β'])
        out.resize(bl * bl, ZERO);
        gemm_nh(out, tmp, &t.data, bl, 2 * br, bl);
    }

    /// Right environment of `site`, contracted from `to`; every site after `to`
    /// must be a right isometry, so the sweep starts there at identity.
    pub(super) fn compute_right_env(&self, site: usize, to: usize) -> Vec<Complex64> {
        let mut env_dim = self.sites[to].bond_right;
        let mut env = identity_env(env_dim);
        let mut tmp = Vec::new();
        let mut next = Vec::new();

        for s in (site + 1..=to).rev() {
            let t = &self.sites[s];
            let bl = t.bond_left;
            let br = t.bond_right;
            debug_assert_eq!(env_dim, br);

            if bl * br > DENSE_ENV_MAX_BONDS {
                self.contract_right_env(s, &env, &mut tmp, &mut next);
                std::mem::swap(&mut env, &mut next);
                env_dim = bl;
                continue;
            }

            let mut new_env = vec![ZERO; bl * bl];
            for alpha in 0..bl {
                for alpha_p in 0..bl {
                    let mut val = ZERO;
                    for beta in 0..br {
                        for beta_p in 0..br {
                            let env_val = env[beta * br + beta_p];
                            if env_val == ZERO {
                                continue;
                            }
                            for i in 0..2 {
                                val += t.data[t.idx(alpha, i, beta)]
                                    * t.data[t.idx(alpha_p, i, beta_p)].conj()
                                    * env_val;
                            }
                        }
                    }
                    new_env[alpha * bl + alpha_p] = val;
                }
            }
            env = new_env;
            env_dim = bl;
        }
        env
    }

    /// Right environment of every site, in one backward sweep.
    ///
    /// `envs[s]` is what [`Self::compute_right_env`] returns for `s`, but the
    /// sweep shares each partial contraction with the site to its left, so all
    /// `n` cost one pass instead of `n`.
    pub(super) fn right_environments(&self) -> Vec<Vec<Complex64>> {
        let n = self.num_qubits;
        let mut envs: Vec<Vec<Complex64>> = Vec::with_capacity(n);
        let mut tmp = Vec::new();
        envs.push(vec![ONE]);
        for site in (1..n).rev() {
            let mut next = Vec::new();
            self.contract_right_env(site, envs.last().unwrap(), &mut tmp, &mut next);
            envs.push(next);
        }
        envs.reverse();
        envs
    }
}
