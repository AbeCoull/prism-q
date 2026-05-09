/// A single block in a factored probability distribution.
///
/// Each block represents the marginal probabilities for one independent
/// subsystem. The `mask` indicates which global qubit positions belong
/// to this block, and `probs` holds the 2^k marginal distribution.
#[derive(Debug, Clone)]
pub struct FactoredBlock {
    /// Marginal probability vector for this block (length 2^k).
    pub probs: Vec<f64>,
    /// Bitmask of global qubit positions belonging to this block.
    pub mask: u64,
}

/// Probability distribution over computational basis states.
///
/// For monolithic simulations this wraps a dense `Vec<f64>` of length 2^n.
/// For decomposed simulations with independent subsystems, this stores
/// per-block marginal distributions that are multiplied on demand,
/// avoiding the O(2^N) Kronecker product unless explicitly requested.
#[derive(Debug, Clone)]
pub enum Probabilities {
    /// Full probability vector of length 2^n.
    Dense(Vec<f64>),
    /// Lazy Kronecker product of independent block distributions.
    Factored {
        /// Per-block marginal probability vectors and bitmasks.
        blocks: Vec<FactoredBlock>,
        /// Total qubit count across all blocks.
        total_qubits: usize,
    },
}

impl Probabilities {
    /// Number of basis states (2^n).
    pub fn len(&self) -> usize {
        match self {
            Probabilities::Dense(v) => v.len(),
            Probabilities::Factored { total_qubits, .. } => 1 << total_qubits,
        }
    }

    /// Always false, a probability distribution has at least one state.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Probability of a single computational basis state. O(1) for dense,
    /// O(K) for factored where K is the number of independent blocks.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    pub fn get(&self, index: usize) -> f64 {
        match self {
            Probabilities::Dense(v) => v[index],
            Probabilities::Factored { blocks, .. } => {
                let mut p = 1.0;
                for block in blocks {
                    let local = extract_block_bits(index, block.mask);
                    p *= block.probs[local];
                }
                p
            }
        }
    }

    /// Iterate over all basis-state probabilities in order.
    ///
    /// For `Dense` this is a direct slice iteration. For `Factored` each
    /// probability is computed on the fly in O(K) per element.
    pub fn iter(&self) -> ProbabilitiesIter<'_> {
        match self {
            Probabilities::Dense(v) => ProbabilitiesIter {
                inner: ProbabilitiesIterInner::Dense(v.iter().copied()),
            },
            Probabilities::Factored {
                blocks,
                total_qubits,
            } => ProbabilitiesIter {
                inner: ProbabilitiesIterInner::Factored {
                    blocks,
                    next: 0,
                    len: 1usize << total_qubits,
                },
            },
        }
    }

    /// Materialize the full probability vector. O(1) clone for dense,
    /// O(K x 2^N) for factored. Prefer [`Probabilities::get`] for spot-checking.
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            Probabilities::Dense(v) => v.clone(),
            Probabilities::Factored {
                blocks,
                total_qubits,
            } => {
                let n = 1usize << total_qubits;
                let mut result = vec![0.0f64; n];
                #[cfg(feature = "parallel")]
                {
                    const MIN_PAR_STATES: usize = 1 << 14;
                    if n >= MIN_PAR_STATES {
                        use rayon::prelude::*;
                        crate::backend::init_thread_pool();
                        result.par_iter_mut().enumerate().for_each(|(i, slot)| {
                            let mut p = 1.0;
                            for block in blocks {
                                let local = extract_block_bits(i, block.mask);
                                p *= block.probs[local];
                            }
                            *slot = p;
                        });
                        return result;
                    }
                }
                for (i, slot) in result.iter_mut().enumerate() {
                    let mut p = 1.0;
                    for block in blocks {
                        let local = extract_block_bits(i, block.mask);
                        p *= block.probs[local];
                    }
                    *slot = p;
                }
                result
            }
        }
    }
}

impl std::ops::Index<usize> for Probabilities {
    type Output = f64;

    /// Index into a dense probability vector.
    ///
    /// Only works for `Dense`. Panics on `Factored` because `Index` must
    /// return `&f64` and factored values are computed, not stored.
    /// Use [`Probabilities::get`] or [`Probabilities::iter`] instead.
    fn index(&self, index: usize) -> &f64 {
        match self {
            Probabilities::Dense(v) => &v[index],
            Probabilities::Factored { .. } => {
                panic!("cannot index Factored probabilities; use .get(i) or .to_vec()")
            }
        }
    }
}

/// Concrete iterator for [`Probabilities::iter`].
pub struct ProbabilitiesIter<'a> {
    inner: ProbabilitiesIterInner<'a>,
}

enum ProbabilitiesIterInner<'a> {
    Dense(std::iter::Copied<std::slice::Iter<'a, f64>>),
    Factored {
        blocks: &'a [FactoredBlock],
        next: usize,
        len: usize,
    },
}

impl Iterator for ProbabilitiesIter<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            ProbabilitiesIterInner::Dense(iter) => iter.next(),
            ProbabilitiesIterInner::Factored { blocks, next, len } => {
                if *next >= *len {
                    return None;
                }
                let index = *next;
                *next += 1;
                let mut p = 1.0;
                for block in *blocks {
                    let local = extract_block_bits(index, block.mask);
                    p *= block.probs[local];
                }
                Some(p)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            ProbabilitiesIterInner::Dense(iter) => iter.size_hint(),
            ProbabilitiesIterInner::Factored { next, len, .. } => {
                let remaining = len.saturating_sub(*next);
                (remaining, Some(remaining))
            }
        }
    }
}

impl ExactSizeIterator for ProbabilitiesIter<'_> {}

/// Extract the bits of `global_index` at positions set in `mask`,
/// packing them into contiguous low bits.
#[inline]
fn extract_block_bits(global_index: usize, mask: u64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            // SAFETY: BMI2 availability is checked immediately before this call.
            return unsafe { core::arch::x86_64::_pext_u64(global_index as u64, mask) as usize };
        }
    }
    let mut result = 0usize;
    let mut bit = 0;
    let mut m = mask;
    while m != 0 {
        let pos = m.trailing_zeros() as usize;
        if global_index & (1 << pos) != 0 {
            result |= 1 << bit;
        }
        bit += 1;
        m &= m.wrapping_sub(1);
    }
    result
}
