//! Reusable benchmark and test circuit builders.
//!
//! Shared across benchmarks, profiling tools, and the cross-simulator
//! comparison runner. All randomized builders use `ChaCha8Rng` for
//! deterministic output given the same seed.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::Circuit;
use crate::gates::Gate;

/// Build an n-qubit textbook QFT circuit (Hadamard + controlled-phase + SWAP).
pub fn qft_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for i in 0..n {
        c.add_gate(Gate::H, &[i]);
        for j in (i + 1)..n {
            let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
            c.add_gate(Gate::cphase(theta), &[i, j]);
        }
    }
    for i in 0..n / 2 {
        c.add_gate(Gate::Swap, &[i, n - 1 - i]);
    }
    c
}

/// Build a random circuit with `n` qubits, `depth` layers of 1q + brick-layer CX.
pub fn random_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let singles = [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::T];
    for layer in 0..depth {
        for q in 0..n {
            c.add_gate(singles[rng.gen_range(0..singles.len())].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n - 1).step_by(2) {
            if rng.gen_bool(0.5) {
                c.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    c
}

/// Build a hardware-efficient ansatz: `layers` of random Ry/Rz + linear CX.
pub fn hardware_efficient_ansatz(n: usize, layers: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for _ in 0..layers {
        for q in 0..n {
            c.add_gate(Gate::Ry(rng.gen::<f64>() * std::f64::consts::TAU), &[q]);
            c.add_gate(Gate::Rz(rng.gen::<f64>() * std::f64::consts::TAU), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c
}

/// Build a Clifford-dominated circuit: random {H, S, X, Y, Z} + brick-layer CX.
pub fn clifford_heavy_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let cliffords = [Gate::H, Gate::S, Gate::X, Gate::Y, Gate::Z];
    for layer in 0..depth {
        for q in 0..n {
            c.add_gate(cliffords[rng.gen_range(0..cliffords.len())].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c
}

pub fn clifford_random_pairs(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let cliffords = [Gate::H, Gate::S, Gate::X, Gate::Y, Gate::Z];
    for _ in 0..depth {
        for q in 0..n {
            c.add_gate(cliffords[rng.gen_range(0..cliffords.len())].clone(), &[q]);
        }
        let num_pairs = n / 2;
        let mut available: Vec<usize> = (0..n).collect();
        for _ in 0..num_pairs {
            if available.len() < 2 {
                break;
            }
            let i = rng.gen_range(0..available.len());
            let q0 = available.swap_remove(i);
            let j = rng.gen_range(0..available.len());
            let q1 = available.swap_remove(j);
            c.add_gate(Gate::Cx, &[q0, q1]);
        }
    }
    c
}

/// N independent Bell pairs: qubits (0,1), (2,3), ..., (2N-2, 2N-1).
///
/// Decomposes into `n_pairs` blocks of 2 qubits each. Useful for
/// benchmarking subsystem decomposition overhead vs monolithic simulation.
pub fn independent_bell_pairs(n_pairs: usize) -> Circuit {
    let n = n_pairs * 2;
    let mut c = Circuit::new(n, 0);
    for i in 0..n_pairs {
        c.add_gate(Gate::H, &[2 * i]);
        c.add_gate(Gate::Cx, &[2 * i, 2 * i + 1]);
    }
    c
}

/// K independent random sub-circuits of `block_size` qubits each.
///
/// Total qubits = `num_blocks * block_size`. Each block has its own
/// brick-layer CX entanglement but no inter-block connections.
pub fn independent_random_blocks(
    num_blocks: usize,
    block_size: usize,
    depth: usize,
    seed: u64,
) -> Circuit {
    let n = num_blocks * block_size;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let singles = [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::T];

    for block in 0..num_blocks {
        let base = block * block_size;
        for layer in 0..depth {
            for q in 0..block_size {
                c.add_gate(
                    singles[rng.gen_range(0..singles.len())].clone(),
                    &[base + q],
                );
            }
            if block_size >= 2 {
                let offset = layer % 2;
                for q in (offset..block_size - 1).step_by(2) {
                    if rng.gen_bool(0.5) {
                        c.add_gate(Gate::Cx, &[base + q, base + q + 1]);
                    }
                }
            }
        }
    }
    c
}

/// Build a GHZ state preparation circuit: H on qubit 0, then CX chain.
///
/// # Panics
/// Panics if `n == 0` (underflow in CX chain loop).
pub fn ghz_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

/// Build a QAOA-style circuit: `layers` of nearest-neighbor ZZ interactions + Rx mixer.
///
/// ZZ(theta) is decomposed as CX - Rz(theta) - CX. The mixer applies Rx(beta)
/// to every qubit. Angles are drawn randomly from the given seed.
pub fn qaoa_circuit(n: usize, layers: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for _ in 0..layers {
        for q in 0..n - 1 {
            let gamma: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Rzz(gamma), &[q, q + 1]);
        }
        for q in 0..n {
            let beta: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Rx(beta), &[q]);
        }
    }
    c
}

/// Build a circuit with only single-qubit rotation gates (no entanglement).
///
/// `depth` layers of random Rx/Ry/Rz on every qubit. Useful for benchmarking
/// product-state and single-qubit gate throughput.
pub fn single_qubit_rotation_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            let choice: usize = rng.gen_range(0..3);
            let angle: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            match choice {
                0 => c.add_gate(Gate::Rx(angle), &[q]),
                1 => c.add_gate(Gate::Ry(angle), &[q]),
                _ => c.add_gate(Gate::Rz(angle), &[q]),
            }
        }
    }
    c
}

/// Build a Clifford+T circuit with controlled T-count.
///
/// Clifford depth-10 base with `t_fraction` of single-qubit gates replaced by T/Tdg.
/// For benchmarking stabilizer rank and quasi-probability dispatch.
pub fn clifford_t_circuit(n: usize, depth: usize, t_fraction: f64, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let cliffords = [Gate::H, Gate::S, Gate::X, Gate::Y, Gate::Z];
    for layer in 0..depth {
        for q in 0..n {
            if rng.gen::<f64>() < t_fraction {
                if rng.gen_bool(0.5) {
                    c.add_gate(Gate::T, &[q]);
                } else {
                    c.add_gate(Gate::Tdg, &[q]);
                }
            } else {
                c.add_gate(cliffords[rng.gen_range(0..cliffords.len())].clone(), &[q]);
            }
        }
        let offset = layer % 2;
        for q in (offset..n.saturating_sub(1)).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c
}

/// Build a W-state preparation circuit.
///
/// Produces the n-qubit W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n.
/// Uses a cascade of controlled rotations and CX gates.
pub fn w_state_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::X, &[0]);
    for i in 0..n - 1 {
        let remaining = (n - i) as f64;
        let theta = 2.0 * (1.0 / remaining).sqrt().acos();
        c.add_gate(Gate::Ry(theta), &[i + 1]);
        c.add_gate(Gate::Cx, &[i + 1, i]);
        c.add_gate(Gate::Ry(-theta), &[i + 1]);
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    c
}

/// Build a quantum volume-style circuit.
///
/// `depth` layers, each applying a random permutation of qubit pairs followed
/// by random SU(4) gates (decomposed into CX + single-qubit rotations).
pub fn quantum_volume_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let n_pairs = n / 2;
    for _ in 0..depth {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        for p in 0..n_pairs {
            let q0 = perm[2 * p];
            let q1 = perm[2 * p + 1];
            let a0: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            let a1: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            let a2: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Ry(a0), &[q0]);
            c.add_gate(Gate::Rz(a1), &[q0]);
            c.add_gate(Gate::Ry(a2), &[q1]);
            c.add_gate(Gate::Cx, &[q0, q1]);
            let b0: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            let b1: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            let b2: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Ry(b0), &[q0]);
            c.add_gate(Gate::Rz(b1), &[q1]);
            c.add_gate(Gate::Cx, &[q0, q1]);
            c.add_gate(Gate::Ry(b2), &[q0]);
        }
    }
    c
}

/// K independent Clifford-only blocks of `block_size` qubits each.
///
/// Total qubits = `num_blocks * block_size`. Each block has brick-layer CX
/// entanglement with random Clifford 1q gates, but no inter-block connections.
/// For benchmarking factored stabilizer vs monolithic stabilizer.
pub fn local_clifford_blocks(
    num_blocks: usize,
    block_size: usize,
    depth: usize,
    seed: u64,
) -> Circuit {
    let n = num_blocks * block_size;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let cliffords = [Gate::H, Gate::S, Gate::X, Gate::Y, Gate::Z];

    for block in 0..num_blocks {
        let base = block * block_size;
        for layer in 0..depth {
            for q in 0..block_size {
                c.add_gate(
                    cliffords[rng.gen_range(0..cliffords.len())].clone(),
                    &[base + q],
                );
            }
            if block_size >= 2 {
                let offset = layer % 2;
                for q in (offset..block_size - 1).step_by(2) {
                    c.add_gate(Gate::Cx, &[base + q, base + q + 1]);
                }
            }
        }
    }
    c
}

/// Build a linearly-connected circuit with only CZ + single-qubit gates.
///
/// `depth` layers of random {H, S, T, X} + linear CZ chain. CZ-heavy circuits
/// exercise different fusion/reordering paths than CX-heavy ones.
pub fn cz_chain_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let singles = [Gate::H, Gate::S, Gate::T, Gate::X];
    for layer in 0..depth {
        for q in 0..n {
            c.add_gate(singles[rng.gen_range(0..singles.len())].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n.saturating_sub(1)).step_by(2) {
            c.add_gate(Gate::Cz, &[q, q + 1]);
        }
    }
    c
}

/// Append an inverse QFT on `n` qubits starting at index `start`.
pub(crate) fn apply_inverse_qft(c: &mut Circuit, start: usize, n: usize) {
    for i in 0..n / 2 {
        c.add_gate(Gate::Swap, &[start + i, start + n - 1 - i]);
    }
    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
            c.add_gate(Gate::cphase(-theta), &[start + i, start + j]);
        }
        c.add_gate(Gate::H, &[start + i]);
    }
}

/// Build a quantum phase estimation circuit.
///
/// `n` is the total qubit count (n-1 counting qubits + 1 target qubit).
///
/// # Panics
/// Panics if `n < 2` (need at least 1 counting qubit + 1 target qubit).
pub fn phase_estimation_circuit(n: usize) -> Circuit {
    let n_counting = n - 1;
    let target = n_counting;
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::X, &[target]);
    for i in 0..n_counting {
        c.add_gate(Gate::H, &[i]);
    }
    for k in 0..n_counting {
        let theta = std::f64::consts::FRAC_PI_4 * (1u64 << k) as f64;
        c.add_gate(Gate::cphase(theta), &[k, target]);
    }
    apply_inverse_qft(&mut c, 0, n_counting);
    c
}
