//! Seeded shots from the split Clifford register are part of the output contract: each
//! tier's draws are pinned, and a run's shots depend on the seed and the shot index alone,
//! never on the shot count or the thread count.

use prism_q::circuit::Circuit;
use prism_q::{BackendKind, Gate, sim};

use crate::bench_fixtures;

const SEED: u64 = 0xDEAD_BEEF;

fn fnv1a(digest: &mut u64, byte: u8) {
    *digest ^= u64::from(byte);
    *digest = digest.wrapping_mul(0x0000_0100_0000_01B3);
}

/// Digest of the shots, shot-major bit order, as `qec_seeded_records` digests records.
fn digest(shots: &[Vec<bool>]) -> u64 {
    let mut digest = 0xCBF2_9CE4_8422_2325u64;
    for shot in shots {
        let mut byte = 0u8;
        for (i, &b) in shot.iter().enumerate() {
            byte = (byte << 1) | u8::from(b);
            if i % 8 == 7 {
                fnv1a(&mut digest, byte);
                byte = 0;
            }
        }
        fnv1a(&mut digest, byte);
    }
    digest
}

fn scrambled(n: usize, t: usize) -> Circuit {
    let mut circuit = bench_fixtures::scrambled_clifford_t_circuit(n, t, SEED);
    circuit.measure_all();
    circuit
}

/// 72 qubits under `H`, one rotated by a T gate: 71 uniform rows, two coin words.
fn wide_coins() -> Circuit {
    let n = 72;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::T, &[5]);
    circuit.add_gate(Gate::H, &[5]);
    circuit.add_gate(Gate::Cx, &[5, 6]);
    circuit.measure_all();
    circuit
}

/// One circuit per tier at the default limits, with the digest of 1000 shots at seed 42:
/// one pack; packs alone at 32 qubits; one table of `2^18`; a cut taken ahead of a whole
/// part of 23 qubits; a cut of a part past 24; two coin words. The parts and cuts are
/// pinned in `bench_fixture_routing`.
fn fixtures() -> Vec<(&'static str, Circuit, u64)> {
    vec![
        (
            "clifford_t_shots/16q_t10",
            bench_fixtures::random_clifford_t_layers(16, 10, 10, SEED),
            0x93f7_c6f3_d109_d5fa,
        ),
        (
            "layers_32q_t24",
            bench_fixtures::random_clifford_t_layers(32, 10, 24, SEED),
            0x4e41_cded_e46f_f651,
        ),
        (
            "scrambled_22q_t20",
            scrambled(22, 20),
            0xaf34_0f95_4050_d6ec,
        ),
        (
            "scrambled_32q_t24",
            scrambled(32, 24),
            0x9210_3298_dd7d_d16b,
        ),
        (
            "scrambled_32q_t28",
            scrambled(32, 28),
            0x2a9a_0ba0_b194_ca0d,
        ),
        ("wide_coins_72q", wide_coins(), 0xc079_223c_a7b2_b0e7),
    ]
}

fn shots(circuit: &Circuit, num_shots: usize) -> Vec<Vec<bool>> {
    sim::simulate(circuit)
        .backend(BackendKind::StabilizerRank)
        .seed(42)
        .shots(num_shots)
        .unwrap()
        .shots
}

#[test]
#[cfg_attr(
    prism_lowered_thresholds,
    ignore = "the lowered split limits tier these circuits differently"
)]
fn register_tiers_draw_pinned_shots() {
    let (got, want): (Vec<_>, Vec<_>) = fixtures()
        .into_iter()
        .map(|(label, circuit, expected)| {
            let got = format!("{label}: {:#018x}", digest(&shots(&circuit, 1000)));
            (got, format!("{label}: {expected:#018x}"))
        })
        .unzip();
    assert_eq!(got, want);
}

#[cfg(feature = "parallel")]
#[test]
#[cfg_attr(
    prism_lowered_thresholds,
    ignore = "the lowered split limits decline some of these circuits"
)]
fn register_shots_ignore_the_shot_and_thread_counts() {
    use prism_q::ThreadPool;

    for (label, circuit, _) in fixtures() {
        let one = ThreadPool::with_threads(1).unwrap();
        let four = ThreadPool::with_threads(4).unwrap();
        let serial = one.install(|| shots(&circuit, 1000));
        assert_eq!(four.install(|| shots(&circuit, 1000)), serial, "{label}");
        assert_eq!(shots(&circuit, 300), serial[..300], "{label}");
        let counts = |pool: &ThreadPool| {
            pool.install(|| {
                sim::simulate(&circuit)
                    .backend(BackendKind::StabilizerRank)
                    .seed(42)
                    .sample_counts(1000)
                    .unwrap()
                    .counts
            })
        };
        let want = sim::simulate(&circuit)
            .backend(BackendKind::StabilizerRank)
            .seed(42)
            .shots(1000)
            .unwrap()
            .counts();
        assert_eq!(counts(&one), counts(&four), "{label}");
        assert_eq!(counts(&one), want, "{label}");
    }
}
