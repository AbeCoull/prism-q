use crate::circuit::Circuit;
use crate::error::Result;

use super::{
    execute_circuit, select_backend, validate_explicit_backend, BackendKind, FactoredBlock,
    Probabilities, SimOptions, SimulationResult,
};

pub(super) const MIN_DECOMPOSITION_QUBITS: usize = 8;

pub(super) fn should_decompose(components: &[Vec<usize>], total_qubits: usize) -> bool {
    let max_block = components.iter().map(|c| c.len()).max().unwrap_or(0);
    max_block + 3 <= total_qubits
}

#[cfg(feature = "parallel")]
const MAX_BLOCK_QUBITS_FOR_PAR: usize = 14;

fn run_blocks_maybe_par(
    kind: &BackendKind,
    partitions: &[(Circuit, Vec<usize>, Vec<usize>)],
    _components: &[Vec<usize>],
    seed: u64,
    opts: &SimOptions,
    k: usize,
) -> Vec<Result<SimulationResult>> {
    #[cfg(feature = "parallel")]
    {
        let all_small = _components
            .iter()
            .all(|c| c.len() < MAX_BLOCK_QUBITS_FOR_PAR);
        if all_small && k >= 2 {
            use rayon::prelude::*;
            crate::backend::init_thread_pool();
            return (0..k)
                .into_par_iter()
                .map(|i| run_subcircuit(kind, &partitions[i].0, seed.wrapping_add(i as u64), opts))
                .collect();
        }
    }
    (0..k)
        .map(|i| run_subcircuit(kind, &partitions[i].0, seed.wrapping_add(i as u64), opts))
        .collect()
}

fn run_subcircuit(
    kind: &BackendKind,
    sub_circuit: &Circuit,
    block_seed: u64,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    let block_kind = if matches!(kind, BackendKind::Auto) {
        BackendKind::Auto
    } else {
        kind.clone()
    };
    super::run_with_internal(block_kind, sub_circuit, block_seed, *opts)
}

pub(super) fn run_decomposed(
    kind: &BackendKind,
    components: &[Vec<usize>],
    circuit: &Circuit,
    seed: u64,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    let partitions = circuit.partition_subcircuits(components);
    let k = partitions.len();

    let block_opts = if circuit.num_qubits > 64 {
        SimOptions::classical_only()
    } else {
        *opts
    };
    let results: Vec<Result<SimulationResult>> =
        run_blocks_maybe_par(kind, &partitions, components, seed, &block_opts, k);

    merge_decomposed_results(
        results,
        components,
        &partitions,
        circuit.num_classical_bits,
        circuit.num_qubits,
        opts,
    )
}

fn merge_decomposed_results(
    results: Vec<Result<SimulationResult>>,
    components: &[Vec<usize>],
    partitions: &[(Circuit, Vec<usize>, Vec<usize>)],
    num_classical_bits: usize,
    num_qubits: usize,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    let mut factored_blocks: Vec<FactoredBlock> = Vec::new();
    let mut merged_classical = vec![false; num_classical_bits];

    for (i, result) in results.into_iter().enumerate() {
        let result = result?;
        let (_, ref qubit_map, ref classical_map) = partitions[i];

        for (local_idx, &global_idx) in classical_map.iter().enumerate() {
            merged_classical[global_idx] = result.classical_bits[local_idx];
        }

        if opts.probabilities && num_qubits <= 64 {
            if let Some(probs) = result.probabilities {
                let dense = probs.to_vec();
                let mut mask = 0u64;
                for &global_qubit in qubit_map {
                    mask |= 1u64 << global_qubit;
                }
                factored_blocks.push(FactoredBlock { probs: dense, mask });
            }
        }
    }

    let probabilities = if opts.probabilities && factored_blocks.len() == components.len() {
        Some(Probabilities::Factored {
            blocks: factored_blocks,
            total_qubits: num_qubits,
        })
    } else {
        None
    };

    Ok(SimulationResult {
        classical_bits: merged_classical,
        probabilities,
    })
}

pub(super) fn run_decomposed_prefused(
    kind: &BackendKind,
    components: &[Vec<usize>],
    partitions: &[(Circuit, Vec<usize>, Vec<usize>)],
    fused_blocks: &[std::borrow::Cow<'_, Circuit>],
    seed: u64,
    opts: &SimOptions,
    original_circuit: &Circuit,
) -> Result<SimulationResult> {
    let k = partitions.len();
    let block_opts = if original_circuit.num_qubits > 64 {
        SimOptions::classical_only()
    } else {
        *opts
    };
    let results: Vec<Result<SimulationResult>> = (0..k)
        .map(|i| {
            let block_seed = seed.wrapping_add(i as u64);
            let sub = &partitions[i].0;
            let block_kind = if matches!(kind, BackendKind::Auto) {
                BackendKind::Auto
            } else {
                kind.clone()
            };
            if !matches!(block_kind, BackendKind::Auto) {
                validate_explicit_backend(&block_kind, sub)?;
            }
            let mut backend = select_backend(&block_kind, sub, block_seed, false);
            execute_circuit(&mut *backend, &fused_blocks[i], &block_opts)
        })
        .collect();
    merge_decomposed_results(
        results,
        components,
        partitions,
        original_circuit.num_classical_bits,
        original_circuit.num_qubits,
        opts,
    )
}

/// Combine per-block probability vectors via Kronecker product.
///
/// Two-pass O(2^N) algorithm:
/// 1. In-place Kronecker product — single allocation, reverse-iteration expansion
/// 2. Bit permutation to map natural (block-sequential) bit positions
///    to global qubit positions (parallelized at ≥2^14 states)
pub(crate) fn merge_probabilities(
    blocks: &[(Vec<f64>, Vec<usize>)],
    total_qubits: usize,
) -> Vec<f64> {
    let n_states = 1usize << total_qubits;

    let mut result = vec![0.0f64; n_states];
    result[0] = 1.0;
    let mut cur_len = 1usize;

    for (probs, _) in blocks {
        let block_len = probs.len();
        let new_len = cur_len * block_len;
        for i in (0..cur_len).rev() {
            let r = result[i];
            for j in 0..block_len {
                result[i * block_len + j] = r * probs[j];
            }
        }
        cur_len = new_len;
    }
    debug_assert_eq!(cur_len, n_states);

    let mut natural_to_global = Vec::with_capacity(total_qubits);
    for (_probs, qubits) in blocks.iter().rev() {
        natural_to_global.extend_from_slice(qubits);
    }

    if natural_to_global.iter().enumerate().all(|(i, &g)| i == g) {
        return result;
    }

    let mut perm = vec![0usize; n_states];
    for (nat_bit, &global_bit) in natural_to_global.iter().enumerate() {
        let half = 1usize << nat_bit;
        for i in 0..half {
            perm[i | half] = perm[i] | (1 << global_bit);
        }
    }

    let mut permuted = vec![0.0f64; n_states];

    #[cfg(feature = "parallel")]
    {
        const MIN_PAR_PERM: usize = 1 << 14;
        if n_states >= MIN_PAR_PERM {
            use rayon::prelude::*;
            result
                .par_iter()
                .zip(perm.par_iter())
                .for_each(|(&prob, &global_idx)| {
                    // SAFETY: perm is a bijection so each global_idx is written exactly once.
                    // No two threads write to the same index.
                    unsafe {
                        let ptr = permuted.as_ptr() as *mut f64;
                        *ptr.add(global_idx) = prob;
                    }
                });
            return permuted;
        }
    }

    for (nat_idx, &prob) in result.iter().enumerate() {
        permuted[perm[nat_idx]] = prob;
    }
    permuted
}
