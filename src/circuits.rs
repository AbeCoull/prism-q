//! Reusable benchmark and test circuit builders.
//!
//! Shared across benchmarks, profiling tools, and the cross-simulator
//! comparison runner. All randomized builders use `ChaCha8Rng` for
//! deterministic output given the same seed.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::Circuit;
use crate::gates::Gate;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Build an n-qubit QFT circuit.
///
/// Emits one `Gate::QftBlock` over `0..n`. CPU statevector runs the native
/// FFT path; other backends expand the block before execution.
pub fn qft_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    debug_assert!(n <= u8::MAX as usize, "QFT block size must fit in u8");
    let targets: Vec<usize> = (0..n).collect();
    c.add_gate(
        Gate::QftBlock {
            start: 0,
            num: n as u8,
        },
        &targets,
    );
    c
}

/// Build a random circuit with `n` qubits, `depth` layers of 1q + brick-layer CX.
///
/// The interaction graph is connected at `depth >= 2`: the first two layers lay
/// down both halves of the brick pattern unconditionally, and later layers keep
/// each bond at random. Without that guarantee a dropped bond can split the
/// register into independent blocks, which routes the circuit to the decomposed
/// path instead of the backend a caller selected.
pub fn random_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let singles = [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::T];
    for layer in 0..depth {
        for q in 0..n {
            c.add_gate(singles[rng.random_range(0..singles.len())].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n - 1).step_by(2) {
            let keep = rng.random_bool(0.5);
            if layer < 2 || keep {
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
            c.add_gate(Gate::Ry(rng.random::<f64>() * std::f64::consts::TAU), &[q]);
            c.add_gate(Gate::Rz(rng.random::<f64>() * std::f64::consts::TAU), &[q]);
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
            c.add_gate(
                cliffords[rng.random_range(0..cliffords.len())].clone(),
                &[q],
            );
        }
        let offset = layer % 2;
        for q in (offset..n - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c
}

/// Build a Clifford circuit with random long-range CX pairings each layer,
/// unlike the brick-layer nearest-neighbor pattern in [`clifford_heavy_circuit`].
pub fn clifford_random_pairs(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let cliffords = [Gate::H, Gate::S, Gate::X, Gate::Y, Gate::Z];
    for _ in 0..depth {
        for q in 0..n {
            c.add_gate(
                cliffords[rng.random_range(0..cliffords.len())].clone(),
                &[q],
            );
        }
        let num_pairs = n / 2;
        let mut available: Vec<usize> = (0..n).collect();
        for _ in 0..num_pairs {
            if available.len() < 2 {
                break;
            }
            let i = rng.random_range(0..available.len());
            let q0 = available.swap_remove(i);
            let j = rng.random_range(0..available.len());
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

/// One connected block spanning `n - 2` qubits plus one independent pair.
///
/// Sized so the decomposition heuristic declines: it needs the largest
/// component to be at least three qubits smaller than the register, which
/// `n - 2` fails. The circuit is therefore partially independent but runs whole
/// on one backend, which is the shape that reaches the factored backend's
/// multi-block probability terminal.
pub fn partially_independent_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    assert!(
        n >= 4,
        "partially_independent_circuit needs at least 4 qubits"
    );
    let big = n - 2;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let singles = [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::T];

    for layer in 0..depth {
        for q in 0..big {
            c.add_gate(singles[rng.random_range(0..singles.len())].clone(), &[q]);
        }
        for q in (layer % 2..big - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    // The chain runs unconditionally so the large block is one component at
    // every seed; group count must not be a seed property.
    for q in 0..big - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }

    for layer in 0..depth {
        c.add_gate(singles[rng.random_range(0..singles.len())].clone(), &[big]);
        c.add_gate(
            singles[rng.random_range(0..singles.len())].clone(),
            &[big + 1],
        );
        if layer == 0 {
            c.add_gate(Gate::Cx, &[big, big + 1]);
        }
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
                    singles[rng.random_range(0..singles.len())].clone(),
                    &[base + q],
                );
            }
            if block_size >= 2 {
                let offset = layer % 2;
                for q in (offset..block_size - 1).step_by(2) {
                    if rng.random_bool(0.5) {
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
/// Each ZZ interaction is a native `Rzz(theta)`. The mixer applies Rx(beta)
/// to every qubit. Angles are drawn randomly from the given seed.
pub fn qaoa_circuit(n: usize, layers: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for _ in 0..layers {
        for q in 0..n - 1 {
            let gamma: f64 = rng.random::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Rzz(gamma), &[q, q + 1]);
        }
        for q in 0..n {
            let beta: f64 = rng.random::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Rx(beta), &[q]);
        }
    }
    c
}

/// Build a mixed diagonal-phase circuit: one H layer, then `layers` of random
/// 1q diagonals (Rz, T, P) plus long-range CZ/CPhase/Rzz pairs at a per-layer
/// stride. The long-range strides keep the pairs out of the same-pair 2q fusion
/// blocks, so the runs collapse into `DiagonalBatch` instructions at 16 qubits
/// and above.
///
/// Pairs sweep the register modulo `n`, which is what keeps the interaction
/// graph connected at `layers >= 2` for any `n`. Anchoring them to the low half
/// instead leaves every qubit above `n / 2 + layers` untouched by any 2q gate,
/// splitting the register once that tail is wide enough for the decomposed
/// route to claim the circuit.
///
/// # Panics
/// Panics if `n < 2` (stride selection divides by `n / 2`).
pub fn diagonal_mixed_circuit(n: usize, layers: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for layer in 0..layers {
        for q in 0..n {
            let theta = rng.random::<f64>() * std::f64::consts::TAU;
            match rng.random_range(0..3) {
                0 => c.add_gate(Gate::Rz(theta), &[q]),
                1 => c.add_gate(Gate::T, &[q]),
                _ => c.add_gate(Gate::P(theta), &[q]),
            }
        }
        let stride = 1 + layer % (n / 2);
        for q in 0..n / 2 {
            let q0 = 2 * q;
            let q1 = (q0 + stride) % n;
            let theta = rng.random::<f64>() * std::f64::consts::TAU;
            match rng.random_range(0..3) {
                0 => c.add_gate(Gate::Cz, &[q0, q1]),
                1 => c.add_gate(Gate::cphase(theta), &[q0, q1]),
                _ => c.add_gate(Gate::Rzz(theta), &[q0, q1]),
            }
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
            let choice: usize = rng.random_range(0..3);
            let angle: f64 = rng.random::<f64>() * std::f64::consts::TAU;
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
            if rng.random::<f64>() < t_fraction {
                if rng.random_bool(0.5) {
                    c.add_gate(Gate::T, &[q]);
                } else {
                    c.add_gate(Gate::Tdg, &[q]);
                }
            } else {
                c.add_gate(
                    cliffords[rng.random_range(0..cliffords.len())].clone(),
                    &[q],
                );
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
            let j = rng.random_range(0..=i);
            perm.swap(i, j);
        }
        for p in 0..n_pairs {
            let q0 = perm[2 * p];
            let q1 = perm[2 * p + 1];
            let a0: f64 = rng.random::<f64>() * std::f64::consts::TAU;
            let a1: f64 = rng.random::<f64>() * std::f64::consts::TAU;
            let a2: f64 = rng.random::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Ry(a0), &[q0]);
            c.add_gate(Gate::Rz(a1), &[q0]);
            c.add_gate(Gate::Ry(a2), &[q1]);
            c.add_gate(Gate::Cx, &[q0, q1]);
            let b0: f64 = rng.random::<f64>() * std::f64::consts::TAU;
            let b1: f64 = rng.random::<f64>() * std::f64::consts::TAU;
            let b2: f64 = rng.random::<f64>() * std::f64::consts::TAU;
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
                    cliffords[rng.random_range(0..cliffords.len())].clone(),
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
            c.add_gate(singles[rng.random_range(0..singles.len())].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n.saturating_sub(1)).step_by(2) {
            c.add_gate(Gate::Cz, &[q, q + 1]);
        }
    }
    c
}

/// Build a non-Clifford brick-wall circuit: `depth` layers of random Ry/Rz on
/// every qubit, each layer closed by CZ on alternating adjacent pairs.
///
/// No layer resets entanglement, so the Schmidt rank across a cut doubles each
/// time a CZ crosses it (once per two layers) until it saturates at
/// `2^min(c, n - c)` for the cut after qubit `c - 1`. The `mps/brickwork_d24`
/// bench rows rely on that growth to reach their bond caps;
/// `tests/bench_fixture_routing.rs` pins it through the instruction stream.
pub fn brickwork_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for layer in 0..depth {
        for q in 0..n {
            c.add_gate(Gate::Ry(rng.random::<f64>() * std::f64::consts::TAU), &[q]);
            c.add_gate(Gate::Rz(rng.random::<f64>() * std::f64::consts::TAU), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n.saturating_sub(1)).step_by(2) {
            c.add_gate(Gate::Cz, &[q, q + 1]);
        }
    }
    c
}

/// Build a brick-wall circuit whose entangling layer is a fresh random
/// perfect matching: `depth` layers of random Ry/Rz on every qubit, then CZ
/// on each pair of a seeded shuffle of the register.
///
/// Pair distances average about `n / 3` and change every layer, so gates
/// stay mostly non-adjacent under any site layout and entanglement grows
/// across every cut; a fixed pairing would instead let SWAP routing park
/// each pair adjacent after one layer and hold the chain at bond 2. At odd
/// `n` one qubit of the shuffle sits out each layer.
/// `tests/bench_fixture_routing.rs` pins both properties.
pub fn matched_brickwork_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            c.add_gate(Gate::Ry(rng.random::<f64>() * std::f64::consts::TAU), &[q]);
            c.add_gate(Gate::Rz(rng.random::<f64>() * std::f64::consts::TAU), &[q]);
        }
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        for pair in order.chunks_exact(2) {
            c.add_gate(Gate::Cz, &[pair[0], pair[1]]);
        }
    }
    c
}

/// Build a sparse-regime workload: H on the low `k` qubits, then `depth`
/// layers of seeded diagonal phases (Rz, P, T, Rzz) and basis permutations
/// (X, brick-layer Cx, Swap) over the whole register.
///
/// The prefix puts the amplitude map at exactly `2^k` entries and no later
/// gate can grow or drain it: diagonals scale amplitudes in place and
/// permutations remap keys one to one, so the entry count is pinned for the
/// whole walk while the keys spread over all `n` bits. The brick Cx layers
/// keep the interaction graph connected at `depth >= 2`, which keeps the
/// decomposed route from claiming the register.
/// `tests/bench_fixture_routing.rs` pins both properties.
///
/// # Panics
/// Panics if `k > n` or `n < 2`.
pub fn sparse_walk_circuit(n: usize, k: usize, depth: usize, seed: u64) -> Circuit {
    assert!(
        k <= n && n >= 2,
        "sparse_walk_circuit needs k <= n and n >= 2"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for q in 0..k {
        c.add_gate(Gate::H, &[q]);
    }
    for layer in 0..depth {
        for q in 0..n {
            let theta = rng.random::<f64>() * std::f64::consts::TAU;
            match rng.random_range(0..4) {
                0 => c.add_gate(Gate::Rz(theta), &[q]),
                1 => c.add_gate(Gate::P(theta), &[q]),
                2 => c.add_gate(Gate::T, &[q]),
                _ => c.add_gate(Gate::X, &[q]),
            }
        }
        let offset = layer % 2;
        for q in (offset..n - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in (offset..n - 1).step_by(5) {
            let theta = rng.random::<f64>() * std::f64::consts::TAU;
            c.add_gate(Gate::Rzz(theta), &[q, q + 1]);
        }
        c.add_gate(Gate::Swap, &[layer % (n - 1), layer % (n - 1) + 1]);
    }
    c
}

/// Append an inverse QFT on `n` qubits starting at index `start`.
///
/// Exact inverse of the forward decomposition in `qft_textbook_steps`: reverse
/// the gate order and negate every controlled-phase angle.
pub(crate) fn apply_inverse_qft(c: &mut Circuit, start: usize, n: usize) {
    for i in 0..n / 2 {
        c.add_gate(Gate::Swap, &[start + i, start + n - 1 - i]);
    }
    for q in 0..n {
        for k in (0..q).rev() {
            let theta = std::f64::consts::TAU / (1u64 << (q - k + 1)) as f64;
            c.add_gate(Gate::cphase(-theta), &[start + k, start + q]);
        }
        c.add_gate(Gate::H, &[start + q]);
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

/// Jordan-Wigner Pauli strings of a seeded random number-conserving two-body
/// fermionic operator on `n` modes, truncated to the `max_terms` largest
/// coefficients.
///
/// One-body terms cover every `p <= q` pair; two-body terms draw `3 * n^2`
/// quadruples `a_dag_p a_dag_q a_r a_s + h.c.` with `p < q`, `r < s`. Ladder
/// operators expand through explicit Pauli-string products, so the output is
/// the exact transform of the sampled operator and Hermitian by construction.
/// Sorted by descending coefficient magnitude; deterministic for a given seed.
pub fn jordan_wigner_hamiltonian(
    n: usize,
    max_terms: usize,
    seed: u64,
) -> Vec<(f64, Vec<PauliTerm>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut accumulated: std::collections::HashMap<Vec<PauliTerm>, Complex64> =
        std::collections::HashMap::new();

    let mut absorb = |strings: Vec<PauliString>, weight: f64| {
        for string in strings {
            let mut factors: Vec<PauliTerm> = string
                .axes
                .iter()
                .enumerate()
                .filter_map(|(qubit, axis)| axis.map(|axis| PauliTerm::new(qubit, axis)))
                .collect();
            factors.sort_unstable_by_key(|term| term.qubit);
            *accumulated
                .entry(factors)
                .or_insert(Complex64::new(0.0, 0.0)) += string.phase * weight;
        }
    };

    for p in 0..n {
        for q in p..n {
            let t = rng.random::<f64>() * 2.0 - 1.0;
            let hop = ladder_product(n, &[(p, true), (q, false)]);
            if p == q {
                absorb(hop, t);
            } else {
                absorb(hop, t);
                absorb(ladder_product(n, &[(q, true), (p, false)]), t);
            }
        }
    }

    for _ in 0..3 * n * n {
        let mut pair = || loop {
            let a = rng.random_range(0..n);
            let b = rng.random_range(0..n);
            if a != b {
                return (a.min(b), a.max(b));
            }
        };
        let (p, q) = pair();
        let (r, s) = pair();
        let v = rng.random::<f64>() * 2.0 - 1.0;
        absorb(
            ladder_product(n, &[(p, true), (q, true), (r, false), (s, false)]),
            v,
        );
        absorb(
            ladder_product(n, &[(s, true), (r, true), (q, false), (p, false)]),
            v,
        );
    }

    let mut terms: Vec<(f64, Vec<PauliTerm>)> = accumulated
        .into_iter()
        .filter(|(_, coefficient)| coefficient.re.abs() > 1e-12)
        .map(|(factors, coefficient)| {
            debug_assert!(coefficient.im.abs() < 1e-9);
            (coefficient.re, factors)
        })
        .collect();
    terms.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()).then_with(|| a.1.cmp(&b.1)));
    terms.truncate(max_terms);
    terms
}

/// One Pauli string of a fermionic-operator expansion: a complex phase and an
/// optional axis per mode.
struct PauliString {
    phase: Complex64,
    axes: Vec<Option<PauliAxis>>,
}

/// `a * b` for single-qubit Paulis: the phase and the resulting axis, `None`
/// for identity.
fn pauli_axis_mul(a: PauliAxis, b: PauliAxis) -> (Complex64, Option<PauliAxis>) {
    use PauliAxis::{X, Y, Z};
    let i = Complex64::new(0.0, 1.0);
    match (a, b) {
        (X, X) | (Y, Y) | (Z, Z) => (Complex64::new(1.0, 0.0), None),
        (X, Y) => (i, Some(Z)),
        (Y, X) => (-i, Some(Z)),
        (Y, Z) => (i, Some(X)),
        (Z, Y) => (-i, Some(X)),
        (Z, X) => (i, Some(Y)),
        (X, Z) => (-i, Some(Y)),
    }
}

fn pauli_string_mul(a: &PauliString, b: &PauliString) -> PauliString {
    let mut phase = a.phase * b.phase;
    let axes = a
        .axes
        .iter()
        .zip(&b.axes)
        .map(|(&left, &right)| match (left, right) {
            (None, axis) | (axis, None) => axis,
            (Some(left), Some(right)) => {
                let (factor, axis) = pauli_axis_mul(left, right);
                phase *= factor;
                axis
            }
        })
        .collect();
    PauliString { phase, axes }
}

/// Product of Jordan-Wigner ladder operators, `(mode, is_creation)` in
/// operator order, expanded as a sum of Pauli strings.
///
/// `a_p = Z_0..Z_{p-1} (X_p + i Y_p) / 2`, with `-i` for the creation
/// operator.
fn ladder_product(n: usize, ops: &[(usize, bool)]) -> Vec<PauliString> {
    let mut product = vec![PauliString {
        phase: Complex64::new(1.0, 0.0),
        axes: vec![None; n],
    }];
    for &(mode, is_creation) in ops {
        let mut x_axes = vec![None; n];
        for chain in x_axes.iter_mut().take(mode) {
            *chain = Some(PauliAxis::Z);
        }
        let mut y_axes = x_axes.clone();
        x_axes[mode] = Some(PauliAxis::X);
        y_axes[mode] = Some(PauliAxis::Y);
        let y_phase = Complex64::new(0.0, if is_creation { -0.5 } else { 0.5 });
        let factor = [
            PauliString {
                phase: Complex64::new(0.5, 0.0),
                axes: x_axes,
            },
            PauliString {
                phase: y_phase,
                axes: y_axes,
            },
        ];
        product = product
            .iter()
            .flat_map(|left| factor.iter().map(|right| pauli_string_mul(left, right)))
            .collect();
    }
    product
}
