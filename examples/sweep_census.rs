//! Static census of statevector passes against the tile-schedule lower bound.
//!
//! Counts the full-state passes each fused bench circuit makes with the batching passes
//! and with the stage planner that replaces them past the last-level cache, the gates
//! applied inside cache tiles, the fewest passes a tile schedule of the same stream
//! needs, and the planner's own time. Nothing is simulated, so it runs at widths no
//! state vector here fits.
//!
//! ```text
//! cargo run --release --features bench-internal --example sweep_census
//! ```

use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::{
    SweepKind, scheduled_sweep_count, stage_planning_time, sweep_lower_bound, sweep_profile_for,
    sweep_profile_unstaged,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 0xDEAD_BEEF;

/// QAOA p = 3 on a ring plus a random perfect matching, as the
/// `statevector/qaoa_ring_matching` bench row builds it.
fn qaoa_ring_matching(n: usize) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    let mut edges: Vec<(usize, usize)> = (0..n).map(|q| (q, (q + 1) % n)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.random_range(0..=i));
    }
    edges.extend(
        order
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0], pair[1])),
    );
    for _ in 0..3 {
        let gamma = rng.random::<f64>() * std::f64::consts::PI;
        let beta = rng.random::<f64>() * std::f64::consts::PI;
        for &(a, b) in &edges {
            circuit.add_gate(Gate::Rzz(gamma), &[a, b]);
        }
        for q in 0..n {
            circuit.add_gate(Gate::Rx(beta), &[q]);
        }
    }
    circuit
}

/// One Trotter step of the seeded Jordan-Wigner operator, as the native
/// `statevector/trotter` bench row builds it.
fn trotter(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for (coefficient, factors) in &circuits::jordan_wigner_hamiltonian(n, 200, SEED) {
        if !factors.is_empty() {
            circuit.add_pauli_rotation(2.0 * coefficient * 0.05, factors);
        }
    }
    circuit
}

fn build(name: &str, n: usize) -> Circuit {
    match name {
        "hea_l5" => circuits::hardware_efficient_ansatz(n, 5, SEED),
        "random_d10" => circuits::random_circuit(n, 10, SEED),
        "qaoa_l3" => circuits::qaoa_circuit(n, 3, SEED),
        "qaoa_ring_matching" => qaoa_ring_matching(n),
        "qv" => circuits::quantum_volume_circuit(n, n, SEED),
        "qft_textbook" => circuits::qft_circuit(n),
        "diag_mixed_l6" => circuits::diagonal_mixed_circuit(n, 6, SEED),
        "trotter" => trotter(n),
        _ => unreachable!(),
    }
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let families = [
        "hea_l5",
        "random_d10",
        "qaoa_l3",
        "qaoa_ring_matching",
        "qv",
        "qft_textbook",
        "diag_mixed_l6",
        "trotter",
    ];
    println!("threads {threads}");
    println!(
        "{:<20} {:>3} {:>8} {:>7} {:>6} {:>7} {:>8} {:>9}  passes/tile_apps by kind, staged",
        "circuit", "n", "unstaged", "staged", "bound", "removed", "plan_us", "tile_apps"
    );
    for name in families {
        for n in (16..=30).step_by(2) {
            let circuit = build(name, n);
            let unstaged = sweep_profile_unstaged(&circuit, threads);
            let profile = sweep_profile_for(&circuit, threads);
            let scheduled = scheduled_sweep_count(&circuit, threads, false);
            let bound = sweep_lower_bound(&circuit, threads);
            let plan = stage_planning_time(&circuit, 20);
            let kinds: Vec<String> = SweepKind::ALL
                .into_iter()
                .filter(|&kind| profile.count(kind).instructions > 0)
                .map(|kind| {
                    let count = profile.count(kind);
                    format!("{}={}/{}", kind.name(), count.passes, count.tile_apps)
                })
                .collect();
            println!(
                "{:<20} {:>3} {:>8} {:>7} {:>6} {:>7} {:>8.1} {:>4}/{:<4}  mixed_forced={scheduled} {}",
                name,
                n,
                unstaged.passes(),
                profile.passes(),
                bound,
                unstaged.passes() as i64 - profile.passes() as i64,
                plan.as_secs_f64() * 1e6,
                unstaged.tile_apps(),
                profile.tile_apps(),
                kinds.join(" ")
            );
        }
    }
    for n in [10, 12] {
        for (name, circuit) in [
            ("random_d10", circuits::random_circuit(n, 10, SEED)),
            ("qaoa_l6", circuits::qaoa_circuit(n, 6, SEED)),
        ] {
            println!(
                "density_matrix/{name}/{n} scheduled={}",
                scheduled_sweep_count(&circuit, threads, true)
            );
        }
    }
    for n in 12..=20 {
        for (name, circuit) in [
            ("qaoa_l3", circuits::qaoa_circuit(n, 3, SEED)),
            (
                "controlled_rotation_l5",
                circuits::controlled_rotation_circuit(n, 5, SEED),
            ),
        ] {
            let profile = sweep_profile_for(&circuit, threads);
            println!(
                "boundary/{name}/{n} passes={} tile_apps={}",
                profile.passes(),
                profile.tile_apps()
            );
        }
    }
}
