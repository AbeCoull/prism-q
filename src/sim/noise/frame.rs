//! Pauli frame sampler for shallow noisy Clifford circuits.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::backend::Backend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::ShotsResult;

use super::NoiseModel;
use super::compiled::{
    PauliSplit, apply_readout_packed, for_each_event_shot, pair_branch, record_readout,
    unpack_and_remap,
};

const FRAME_BATCH_SIZE: usize = 256;

/// Gate count per qubit below which the Pauli frame sampler beats the compiled
/// sensitivity map, whose build cost amortizes only over deeper circuits. Wide
/// circuits carry a higher cutoff: the frame walks one word per qubit per gate,
/// so its per-gate cost grows with the register while the map's does not.
const FRAME_DEPTH_RATIO: f64 = 3.0;
#[cfg(not(prism_lowered_thresholds))]
const WIDE_FRAME_QUBITS: usize = 200;
#[cfg(prism_lowered_thresholds)]
const WIDE_FRAME_QUBITS: usize = 8;
const WIDE_FRAME_DEPTH_RATIO: f64 = 5.0;

/// True when [`run_shots_noisy`](super::run_shots_noisy) takes the Pauli frame
/// sampler rather than the compiled one. Both answer from the same distribution
/// and stamp the same resolved backend, so a result carries nothing that
/// separates them.
pub(in crate::sim) fn use_frame_sampler(circuit: &Circuit) -> bool {
    let n = circuit.num_qubits;
    let gate_count = circuit
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate { .. } | Instruction::Conditional { .. }
            )
        })
        .count();

    let depth_ratio = gate_count as f64 / n.max(1) as f64;
    depth_ratio < FRAME_DEPTH_RATIO
        || (n >= WIDE_FRAME_QUBITS && depth_ratio < WIDE_FRAME_DEPTH_RATIO)
}

struct ReferenceInfo {
    outcomes: Vec<bool>,
    is_random: Vec<bool>,
    random_x_support: Vec<Vec<usize>>,
}

fn reference_simulation(circuit: &Circuit, seed: u64) -> Result<ReferenceInfo> {
    let mut stab = StabilizerBackend::new(seed);
    stab.init(circuit.num_qubits, circuit.num_classical_bits)?;

    let meas_info: Vec<(usize, usize, usize)> = circuit
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => Some((i, *qubit, *classical_bit)),
            _ => None,
        })
        .collect();

    let num_meas = meas_info.len();
    if num_meas == 0 {
        return Ok(ReferenceInfo {
            outcomes: Vec::new(),
            is_random: Vec::new(),
            random_x_support: Vec::new(),
        });
    }

    let first_meas_idx = meas_info[0].0;
    let all_at_end = meas_info
        .iter()
        .enumerate()
        .all(|(i, &(inst_idx, _, _))| inst_idx == first_meas_idx + i);
    let mut seen_qubits = vec![false; circuit.num_qubits];
    let mut seen_bits = vec![false; circuit.num_classical_bits];
    let distinct = meas_info.iter().all(|&(_, qubit, classical_bit)| {
        !std::mem::replace(&mut seen_qubits[qubit], true)
            && !std::mem::replace(&mut seen_bits[classical_bit], true)
    });

    if all_at_end && distinct {
        stab.apply_gates_only(&circuit.instructions[..first_meas_idx])?;

        let measurements: Vec<(usize, usize)> = meas_info
            .iter()
            .map(|&(_, qubit, classical_bit)| (qubit, classical_bit))
            .collect();
        let (is_random, random_x_support, outcomes) =
            stab.batch_measure_ref_info(&measurements, true);

        return Ok(ReferenceInfo {
            outcomes,
            is_random,
            random_x_support,
        });
    }

    let mut outcomes = Vec::with_capacity(num_meas);
    let mut is_random = Vec::with_capacity(num_meas);
    let mut random_x_support: Vec<Vec<usize>> = Vec::with_capacity(num_meas);

    let mut seg_start = 0usize;
    for &(meas_inst_idx, qubit, classical_bit) in &meas_info {
        if seg_start < meas_inst_idx {
            stab.apply_gates_only(&circuit.instructions[seg_start..meas_inst_idx])?;
        }

        let (meas_random, support) = stab.apply_measure_with_info(qubit, classical_bit);
        outcomes.push(stab.classical_results()[classical_bit]);
        is_random.push(meas_random);
        random_x_support.push(support);
        seg_start = meas_inst_idx + 1;
    }

    Ok(ReferenceInfo {
        outcomes,
        is_random,
        random_x_support,
    })
}

#[inline(always)]
fn apply_gate_to_frame(
    gate: &Gate,
    targets: &[usize],
    x_frame: &mut [Vec<u64>],
    z_frame: &mut [Vec<u64>],
    bw: usize,
) {
    match gate {
        Gate::H => {
            let q = targets[0];
            std::mem::swap(&mut x_frame[q], &mut z_frame[q]);
        }
        Gate::S | Gate::Sdg => {
            let q = targets[0];
            for w in 0..bw {
                z_frame[q][w] ^= x_frame[q][w];
            }
        }
        Gate::SX | Gate::SXdg => {
            let q = targets[0];
            for w in 0..bw {
                x_frame[q][w] ^= z_frame[q][w];
            }
        }
        Gate::X | Gate::Y | Gate::Z | Gate::Id => {}
        Gate::Cx => {
            let ctrl = targets[0];
            let tgt = targets[1];
            for w in 0..bw {
                x_frame[tgt][w] ^= x_frame[ctrl][w];
                z_frame[ctrl][w] ^= z_frame[tgt][w];
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            for w in 0..bw {
                z_frame[q0][w] ^= x_frame[q1][w];
                z_frame[q1][w] ^= x_frame[q0][w];
            }
        }
        Gate::Swap => {
            let q0 = targets[0];
            let q1 = targets[1];
            x_frame.swap(q0, q1);
            z_frame.swap(q0, q1);
        }
        _ => {
            debug_assert!(
                false,
                "apply_gate_to_frame: unhandled Clifford gate {:?}",
                gate
            );
        }
    }
}

#[inline(always)]
fn flip_frame_letter(
    x_frame: &mut [Vec<u64>],
    z_frame: &mut [Vec<u64>],
    q: usize,
    pos: usize,
    letter: usize,
) {
    let w = pos / 64;
    let bit = 1u64 << (pos % 64);
    match letter {
        1 => x_frame[q][w] ^= bit,
        2 => {
            x_frame[q][w] ^= bit;
            z_frame[q][w] ^= bit;
        }
        3 => z_frame[q][w] ^= bit,
        _ => {}
    }
}

pub(super) fn run_shots_noisy_frame(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let n = circuit.num_qubits;
    let num_classical = circuit.num_classical_bits;

    let ref_info = reference_simulation(circuit, seed)?;

    let classical_bit_order = circuit.classical_bit_order();
    let num_measurements = classical_bit_order.len();
    let m_words = num_measurements.div_ceil(64);

    let mut all_packed = vec![0u64; num_shots * m_words];
    let readout = record_readout(circuit, noise);
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xFAAB_E001));

    let max_bw = FRAME_BATCH_SIZE.div_ceil(64);
    let mut x_frame: Vec<Vec<u64>> = vec![vec![0u64; max_bw]; n];
    let mut z_frame: Vec<Vec<u64>> = vec![vec![0u64; max_bw]; n];

    for batch_start in (0..num_shots).step_by(FRAME_BATCH_SIZE) {
        let batch_end = (batch_start + FRAME_BATCH_SIZE).min(num_shots);
        let batch_n = batch_end - batch_start;
        let bw = batch_n.div_ceil(64);

        for row in x_frame.iter_mut().chain(z_frame.iter_mut()) {
            row[..bw].fill(0);
        }

        let mut meas_idx = 0usize;

        for (idx, inst) in circuit.instructions.iter().enumerate() {
            match inst {
                Instruction::Gate { gate, targets }
                | Instruction::Conditional { gate, targets, .. } => {
                    apply_gate_to_frame(gate, targets.as_slice(), &mut x_frame, &mut z_frame, bw);
                }
                Instruction::Measure {
                    qubit,
                    classical_bit: _,
                } => {
                    if ref_info.is_random[meas_idx] {
                        let support = &ref_info.random_x_support[meas_idx];
                        #[allow(clippy::needless_range_loop)]
                        for w in 0..bw {
                            let random_word: u64 = rand::RngExt::random(&mut rng);
                            let mask = if w == bw - 1 && !batch_n.is_multiple_of(64) {
                                random_word & ((1u64 << (batch_n % 64)) - 1)
                            } else {
                                random_word
                            };
                            if mask != 0 {
                                for &q in support {
                                    x_frame[q][w] ^= mask;
                                }
                            }
                        }
                    }

                    let ref_bit = ref_info.outcomes[meas_idx];
                    let mi_word = meas_idx / 64;
                    let mi_bit = meas_idx % 64;
                    #[allow(clippy::needless_range_loop)]
                    for w in 0..bw {
                        let frame_word = x_frame[*qubit][w];
                        let effective = if ref_bit { !frame_word } else { frame_word };
                        let num_bits = if w == bw - 1 && !batch_n.is_multiple_of(64) {
                            batch_n % 64
                        } else {
                            64
                        };
                        let mask = if num_bits == 64 {
                            effective
                        } else {
                            effective & ((1u64 << num_bits) - 1)
                        };
                        let mut bits = mask;
                        while bits != 0 {
                            let s = bits.trailing_zeros() as usize;
                            let gs = batch_start + w * 64 + s;
                            all_packed[gs * m_words + mi_word] |= 1u64 << mi_bit;
                            bits &= bits - 1;
                        }
                    }

                    meas_idx += 1;
                }
                _ => {}
            }

            for event in &noise.after_gate[idx] {
                let frames = &mut (x_frame.as_mut_slice(), z_frame.as_mut_slice());
                if let Some((q0, q1, p)) = event.pauli_pair() {
                    if p == 0.0 {
                        continue;
                    }
                    for_each_event_shot(
                        &mut rng,
                        p,
                        0,
                        batch_n,
                        frames,
                        |(x, z), shot, r, dense| {
                            if dense && r >= p {
                                return;
                            }
                            let (l0, l1) = pair_branch(if dense { r / p } else { r });
                            flip_frame_letter(x, z, q0, shot, l0);
                            flip_frame_letter(x, z, q1, shot, l1);
                        },
                    );
                    continue;
                }

                let (px, py, pz) = event.pauli_probs();
                let q = event.qubit();
                let Some(split) = PauliSplit::new(px, py, pz) else {
                    continue;
                };
                for_each_event_shot(
                    &mut rng,
                    split.p_event(),
                    0,
                    batch_n,
                    frames,
                    |(x, z), shot, r, dense| {
                        flip_frame_letter(x, z, q, shot, split.letter(r, dense));
                    },
                );
            }
        }
    }

    apply_readout_packed(&mut all_packed, num_shots, m_words, &readout, &mut rng);

    let shots = unpack_and_remap(
        &all_packed,
        m_words,
        num_shots,
        &classical_bit_order,
        num_classical,
    );

    Ok(
        ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(
            crate::sim::RunMetadata::exact(crate::sim::ResolvedBackend::CompiledStabilizer)
                .with_engine(crate::sim::Engine::FrameSampler)
                .with_reason(crate::sim::RouteRule::CompiledCliffordSampler),
        ),
    )
}
