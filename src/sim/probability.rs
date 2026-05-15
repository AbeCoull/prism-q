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

#[cfg(test)]
mod tests {
    use super::*;

    fn factored_2x3() -> Probabilities {
        Probabilities::Factored {
            blocks: vec![
                FactoredBlock {
                    probs: vec![0.25, 0.75],
                    mask: 0b001,
                },
                FactoredBlock {
                    probs: vec![0.1, 0.2, 0.3, 0.4],
                    mask: 0b110,
                },
            ],
            total_qubits: 3,
        }
    }

    #[test]
    fn dense_basic_accessors() {
        let p = Probabilities::Dense(vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(p.len(), 4);
        assert!(!p.is_empty());
        assert_eq!(p.get(2), 0.3);
        assert_eq!(p[3], 0.4);
        assert_eq!(p.to_vec(), vec![0.1, 0.2, 0.3, 0.4]);
        let collected: Vec<f64> = p.iter().collect();
        assert_eq!(collected, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn factored_get_matches_to_vec() {
        let p = factored_2x3();
        assert_eq!(p.len(), 8);
        let dense = p.to_vec();
        for (i, d) in dense.iter().enumerate() {
            assert!((p.get(i) - d).abs() < 1e-12);
        }
        let sum: f64 = dense.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn factored_iter_matches_to_vec() {
        let p = factored_2x3();
        let dense = p.to_vec();
        let iter_vec: Vec<f64> = p.iter().collect();
        assert_eq!(iter_vec.len(), dense.len());
        for (a, b) in iter_vec.iter().zip(dense.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn factored_iter_size_hint_exact() {
        let p = factored_2x3();
        let mut it = p.iter();
        assert_eq!(it.size_hint(), (8, Some(8)));
        assert_eq!(it.len(), 8);
        it.next();
        assert_eq!(it.size_hint(), (7, Some(7)));
        for _ in 0..7 {
            it.next();
        }
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn dense_iter_size_hint_exact() {
        let p = Probabilities::Dense(vec![0.5, 0.5]);
        let it = p.iter();
        assert_eq!(it.size_hint(), (2, Some(2)));
    }

    #[test]
    #[should_panic(expected = "cannot index Factored")]
    fn factored_index_panics() {
        let p = factored_2x3();
        let _ = p[0];
    }

    #[test]
    fn extract_block_bits_scalar_via_get() {
        let blocks = vec![FactoredBlock {
            probs: vec![0.0, 0.0, 0.0, 1.0],
            mask: 0b1010,
        }];
        let p = Probabilities::Factored {
            blocks,
            total_qubits: 4,
        };
        assert!((p.get(0b1010) - 1.0).abs() < 1e-12);
        assert!(p.get(0b0010).abs() < 1e-12);
        assert!(p.get(0b1000).abs() < 1e-12);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn factored_to_vec_parallel_path() {
        let mut blocks = Vec::new();
        for i in 0..5 {
            blocks.push(FactoredBlock {
                probs: vec![0.125; 8],
                mask: 0b111u64 << (3 * i),
            });
        }
        let p = Probabilities::Factored {
            blocks,
            total_qubits: 15,
        };
        let v = p.to_vec();
        assert_eq!(v.len(), 1 << 15);
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        for (i, val) in v.iter().enumerate() {
            assert!((p.get(i) - val).abs() < 1e-12);
        }
    }
}
