use std::fmt::Write as _;

use super::*;
use crate::backend::Backend;
use crate::backend::statevector::{StatevectorBackend, sweeps};

fn matrix_json<const N: usize>(matrix: &[[Complex64; N]; N]) -> String {
    let rows: Vec<_> = matrix
        .iter()
        .map(|row| {
            let entries: Vec<_> = row
                .iter()
                .map(|z| format!("[{:.17e},{:.17e}]", z.re, z.im))
                .collect();
            format!("[{}]", entries.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}

fn payload_json(targets: &[usize], matrix: String) -> String {
    format!("{{\"targets\":{targets:?},\"matrix\":{matrix}}}")
}

fn variant(gate: &Gate) -> &'static str {
    match gate {
        Gate::Id => "Id",
        Gate::X => "X",
        Gate::Y => "Y",
        Gate::Z => "Z",
        Gate::H => "H",
        Gate::S => "S",
        Gate::Sdg => "Sdg",
        Gate::T => "T",
        Gate::Tdg => "Tdg",
        Gate::SX => "SX",
        Gate::SXdg => "SXdg",
        Gate::Rx(_) => "Rx",
        Gate::Ry(_) => "Ry",
        Gate::Rz(_) => "Rz",
        Gate::P(_) => "P",
        Gate::Rzz(_) => "Rzz",
        Gate::Cx => "Cx",
        Gate::Cz => "Cz",
        Gate::Swap => "Swap",
        Gate::Cu(_) => "Cu",
        Gate::Fused(_) => "Fused",
        Gate::Fused2q(_) => "Fused2q",
        Gate::MultiFused(_) => "MultiFused",
        Gate::Multi2q(_) => "Multi2q",
        Gate::BatchRzz(_) => "BatchRzz",
        Gate::BatchPhase(_) => "BatchPhase",
        Gate::DiagonalBatch(_) => "DiagonalBatch",
        other => panic!("unexpected atlas gate: {other:?}"),
    }
}

fn instruction_json(index: usize, instruction: &Instruction, kernel_sweeps: usize) -> String {
    let Instruction::Gate { gate, targets } = instruction else {
        panic!("atlas circuits contain only gates");
    };
    let mut support = targets.to_vec();
    let entries: Vec<String> = match gate {
        Gate::MultiFused(data) => data
            .gates()
            .iter()
            .map(|(q, matrix)| payload_json(&[*q], matrix_json(matrix)))
            .collect(),
        Gate::Multi2q(data) => data
            .gates()
            .iter()
            .map(|(q0, q1, matrix)| payload_json(&[*q0, *q1], matrix_json(matrix)))
            .collect(),
        Gate::BatchRzz(data) => data
            .edges()
            .iter()
            .map(|&(q0, q1, theta)| {
                payload_json(&[q0, q1], matrix_json(&Gate::Rzz(theta).matrix_4x4()))
            })
            .collect(),
        Gate::BatchPhase(data) => data
            .phases()
            .iter()
            .map(|&(q, phase)| {
                support.push(q);
                let z = Complex64::new(0.0, 0.0);
                let one = Complex64::new(1.0, 0.0);
                let matrix = [
                    [one, z, z, z],
                    [z, one, z, z],
                    [z, z, one, z],
                    [z, z, z, phase],
                ];
                payload_json(&[targets[0], q], matrix_json(&matrix))
            })
            .collect(),
        Gate::DiagonalBatch(data) => data
            .entries()
            .iter()
            .map(|entry| {
                if let Some((q, matrix)) = entry.as_1q_matrix() {
                    support.push(q);
                    payload_json(&[q], matrix_json(&matrix))
                } else {
                    let (q0, q1, matrix) = entry.as_2q_matrix().unwrap();
                    support.extend([q0, q1]);
                    payload_json(&[q0, q1], matrix_json(&matrix))
                }
            })
            .collect(),
        _ if gate.num_qubits() == 1 => {
            vec![payload_json(targets, matrix_json(&gate.matrix_2x2()))]
        }
        _ if gate.num_qubits() == 2 => {
            vec![payload_json(targets, matrix_json(&gate.matrix_4x4()))]
        }
        other => panic!("unexpected atlas gate: {other:?}"),
    };
    support.sort_unstable();
    support.dedup();
    format!(
        "{{\"index\":{index},\"gate\":{:?},\"variant\":{:?},\"targets\":{targets:?},\"support\":{support:?},\"kernelSweeps\":{kernel_sweeps},\"payload\":[{}]}}",
        gate.name(),
        variant(gate),
        entries.join(",")
    )
}

fn initial_state(width: usize) -> Vec<Complex64> {
    let mut state: Vec<_> = (0..1usize << width)
        .map(|j| {
            Complex64::new(
                ((j * 17 + 42) % 101) as f64 - 50.0,
                ((j * 31 + 42) % 97) as f64 - 48.0,
            )
        })
        .collect();
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    state.iter_mut().for_each(|z| *z /= norm);
    state
}

fn execute(circuit: &Circuit, initial: &[Complex64]) -> (Vec<Complex64>, Vec<usize>) {
    let mut backend = StatevectorBackend::new(42);
    backend.init_from_amplitudes(initial.to_vec(), 0).unwrap();
    sweeps::take_ticks();
    let mut ticks = Vec::with_capacity(circuit.instructions.len());
    for instruction in &circuit.instructions {
        backend.apply(instruction).unwrap();
        ticks.push(sweeps::take_ticks());
    }
    (backend.state_vector().to_vec(), ticks)
}

struct Snapshot {
    name: &'static str,
    path: &'static str,
    symbol: &'static str,
    enabled: bool,
    floor: Option<usize>,
    circuit: Circuit,
}

fn pipeline(circuit: &Circuit, profile: FusionProfile) -> Vec<Snapshot> {
    let n = circuit.num_qubits;
    let mut stream = Cow::Borrowed(circuit);
    let mut tracer = Tracer::off();
    let mut snapshots = vec![Snapshot {
        name: "original",
        path: "src/circuit/fusion_atlas_tests.rs",
        symbol: "family",
        enabled: true,
        floor: None,
        circuit: circuit.clone(),
    }];
    macro_rules! pass {
        ($name:literal, $path:literal, $symbol:literal, $enabled:expr, $floor:expr, $body:expr) => {{
            let enabled = $enabled;
            if enabled {
                stream = $body;
            }
            snapshots.push(Snapshot {
                name: $name,
                path: $path,
                symbol: $symbol,
                enabled,
                floor: $floor,
                circuit: stream.clone().into_owned(),
            });
        }};
    }
    pass!(
        "regions",
        "src/circuit/fusion.rs",
        "fuse_region_bodies",
        true,
        None,
        fuse_region_bodies(circuit, n, profile)
    );
    pass!(
        "cancel",
        "src/circuit/fusion.rs",
        "cancel_self_inverse_pairs",
        true,
        None,
        apply_pass(stream, &mut tracer, cancel_self_inverse_pairs)
    );
    pass!(
        "recognize-rzz",
        "src/circuit/fusion_rzz.rs",
        "fuse_rzz",
        true,
        None,
        apply_pass(stream, &mut tracer, fuse_rzz)
    );
    if profile == FusionProfile::Host {
        pass!(
            "batch-rzz",
            "src/circuit/fusion_rzz.rs",
            "fuse_batch_rzz",
            n >= MIN_QUBITS_FOR_DIAG_BATCH,
            Some(MIN_QUBITS_FOR_DIAG_BATCH),
            apply_pass(stream, &mut tracer, fuse_batch_rzz)
        );
    }
    let active = n >= MIN_QUBITS_FOR_FUSION;
    pass!(
        "fuse-1q",
        "src/circuit/fusion.rs",
        "fuse_single_qubit_gates",
        active,
        Some(MIN_QUBITS_FOR_FUSION),
        apply_pass(stream, &mut tracer, fuse_single_qubit_gates)
    );
    pass!(
        "reorder-1q",
        "src/circuit/fusion.rs",
        "reorder_1q_gates",
        active,
        Some(MIN_QUBITS_FOR_FUSION),
        reorder_1q_gates(stream, &mut tracer)
    );
    pass!(
        "cancel-after-reorder",
        "src/circuit/fusion.rs",
        "cancel_self_inverse_pairs",
        active,
        Some(MIN_QUBITS_FOR_FUSION),
        apply_pass(stream, &mut tracer, cancel_self_inverse_pairs)
    );
    pass!(
        "refuse-1q",
        "src/circuit/fusion.rs",
        "fuse_single_qubit_gates",
        active,
        Some(MIN_QUBITS_FOR_FUSION),
        apply_pass(stream, &mut tracer, fuse_single_qubit_gates)
    );
    let pair_active = active && n >= MIN_QUBITS_FOR_2Q_FUSION;
    if profile == FusionProfile::Device {
        pass!(
            "absorb-2q-same-pair-only",
            "src/circuit/fusion.rs",
            "fuse_2q_gates_where",
            pair_active,
            Some(MIN_QUBITS_FOR_2Q_FUSION),
            {
                let runs = same_pair_run_members(&stream);
                fuse_2q_gates_where(stream, &mut tracer, false, |i| runs[i])
            }
        );
    } else {
        pass!(
            "absorb-2q",
            "src/circuit/fusion.rs",
            "fuse_2q_gates",
            pair_active,
            Some(MIN_QUBITS_FOR_2Q_FUSION),
            fuse_2q_gates(stream, &mut tracer, n)
        );
    }
    pass!(
        "same-pair-2q",
        "src/circuit/fusion.rs",
        "fuse_same_pair_2q_blocks",
        pair_active,
        Some(MIN_QUBITS_FOR_2Q_FUSION),
        fuse_same_pair_2q_blocks(stream, &mut tracer)
    );
    if profile == FusionProfile::Host {
        if active && stages_at(n, circuit.num_qubits) {
            pass!(
                "stage-planner",
                "src/circuit/fusion_stage.rs",
                "plan_stages",
                true,
                None,
                plan_stages(stream, &mut tracer)
            );
        } else {
            pass!(
                "multi-1q",
                "src/circuit/fusion.rs",
                "fuse_multi_1q_gates",
                active && n >= MIN_QUBITS_FOR_MULTI_FUSION,
                Some(MIN_QUBITS_FOR_MULTI_FUSION),
                fuse_multi_1q_gates(stream, &mut tracer)
            );
            pass!(
                "reorder-2q-tiles",
                "src/circuit/fusion.rs",
                "reorder_fused2q_into_tiles",
                active && n >= MIN_QUBITS_FOR_MULTI_2Q_FUSION && reorder_2q_enabled(),
                Some(MIN_QUBITS_FOR_MULTI_2Q_FUSION),
                reorder_fused2q_into_tiles(stream, &mut tracer)
            );
            pass!(
                "multi-2q",
                "src/circuit/fusion.rs",
                "fuse_multi_2q_gates",
                active && n >= MIN_QUBITS_FOR_MULTI_2Q_FUSION,
                Some(MIN_QUBITS_FOR_MULTI_2Q_FUSION),
                fuse_multi_2q_gates(stream, &mut tracer)
            );
            pass!(
                "pauli-rotation-batches",
                "src/circuit/fusion.rs",
                "fuse_pauli_rot_batches",
                active && n >= MIN_QUBITS_FOR_PAULI_ROT_BATCH,
                Some(MIN_QUBITS_FOR_PAULI_ROT_BATCH),
                fuse_pauli_rot_batches(stream, &mut tracer)
            );
            pass!(
                "controlled-phases",
                "src/circuit/fusion_phase.rs",
                "fuse_controlled_phases",
                active && n >= MIN_QUBITS_FOR_DIAG_BATCH,
                Some(MIN_QUBITS_FOR_DIAG_BATCH),
                fuse_controlled_phases(stream, &mut tracer)
            );
            pass!(
                "diagonal-batch",
                "src/circuit/fusion.rs",
                "fuse_diagonal_batch",
                active && n >= MIN_QUBITS_FOR_DIAG_BATCH,
                Some(MIN_QUBITS_FOR_DIAG_BATCH),
                fuse_diagonal_batch(stream, &mut tracer)
            );
            pass!(
                "rebatch-1q",
                "src/circuit/fusion_phase.rs",
                "batch_post_phase_1q",
                active && n >= MIN_QUBITS_FOR_DIAG_BATCH,
                Some(MIN_QUBITS_FOR_DIAG_BATCH),
                batch_post_phase_1q(stream, &mut tracer)
            );
        }
    }
    let public = fuse_circuit_for_profile(circuit, true, n, profile);
    assert_eq!(stream.instructions.len(), public.instructions.len());
    for (actual, expected) in stream.instructions.iter().zip(&public.instructions) {
        match (actual, expected) {
            (
                Instruction::Gate {
                    gate: a,
                    targets: at,
                },
                Instruction::Gate {
                    gate: b,
                    targets: bt,
                },
            ) => {
                assert_eq!(a, b);
                assert_eq!(at, bt);
            }
            _ => panic!("atlas circuits contain only gates"),
        }
    }
    snapshots
}

fn family(name: &str, width: usize) -> Circuit {
    let mut circuit = Circuit::new(width, 0);
    match name {
        "legal-disjoint" => {
            circuit.add_gate(Gate::Cx, &[4, 5]);
            circuit.add_gate(Gate::Ry(0.11), &[6]);
            circuit.add_gate(Gate::Cx, &[4, 5]);
            circuit.add_gate(Gate::X, &[0]);
            circuit.add_gate(Gate::Ry(0.31), &[1]);
            circuit.add_gate(Gate::X, &[0]);
            circuit.add_gate(Gate::H, &[2]);
            circuit.add_gate(Gate::Rx(0.23), &[3]);
            circuit.add_gate(Gate::H, &[2]);
            for q in 0..width {
                circuit.add_gate(Gate::Ry(0.17 + q as f64 * 0.01), &[q]);
            }
            for q in (0..width).rev() {
                circuit.add_gate(Gate::Rx(-0.29), &[q]);
            }
        }
        "blocked-dependency" => {
            circuit.add_gate(Gate::X, &[0]);
            circuit.add_gate(Gate::Cx, &[0, 1]);
            circuit.add_gate(Gate::X, &[0]);
            circuit.add_gate(Gate::H, &[1]);
            circuit.add_gate(Gate::Rz(0.37), &[1]);
            circuit.add_gate(Gate::H, &[1]);
            for q in 2..width {
                circuit.add_gate(Gate::Ry(0.19), &[q]);
                circuit.add_gate(Gate::Cx, &[q - 1, q]);
                circuit.add_gate(Gate::Rx(0.27), &[q - 1]);
            }
        }
        "diagonal-pair-runs" => {
            for q in 0..width {
                circuit.add_gate(Gate::H, &[q]);
                circuit.add_gate(Gate::Ry(0.13), &[q]);
            }
            for q in (0..width - 1).step_by(2) {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
                circuit.add_gate(Gate::Rz(0.41), &[q + 1]);
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
            for q in 0..width {
                circuit.add_gate(Gate::Rz(0.21), &[q]);
            }
            for q in (0..width - 1).step_by(2) {
                circuit.add_gate(Gate::Cz, &[q, q + 1]);
            }
            for q in (0..width - 1).step_by(2) {
                circuit.add_gate(Gate::Rx(0.23), &[q]);
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
                circuit.add_gate(Gate::Ry(-0.17), &[q + 1]);
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
                circuit.add_gate(Gate::Rz(0.09), &[q]);
            }
        }
        _ => panic!("unknown atlas family"),
    }
    circuit
}

fn append_snapshot(
    output: &mut String,
    snapshot: &Snapshot,
    index: usize,
    initial: &[Complex64],
    reference: &[Complex64],
    threads: usize,
) {
    let (state, instruction_ticks) = execute(&snapshot.circuit, initial);
    let ticks = instruction_ticks.iter().sum::<usize>();
    let max_error = state
        .iter()
        .zip(reference)
        .map(|(actual, expected)| (*actual - expected).norm())
        .fold(0.0, f64::max);
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>();
    assert!(max_error < 1e-12, "{}: {max_error}", snapshot.name);
    assert!((norm - 1.0).abs() < 1e-12);
    let profile = sweeps::sweep_profile(
        &snapshot.circuit.instructions,
        snapshot.circuit.num_qubits,
        threads,
    );
    assert_eq!(ticks, profile.passes(), "{} sweep count", snapshot.name);
    let instructions: Vec<_> = snapshot
        .circuit
        .instructions
        .iter()
        .enumerate()
        .map(|(i, inst)| instruction_json(i, inst, instruction_ticks[i]))
        .collect();
    let sample_indices = [0, 1, 2, 3, state.len() / 2, state.len() - 1];
    let samples: Vec<_> = sample_indices
        .iter()
        .map(|&i| {
            format!(
                "{{\"index\":{i},\"re\":{:.17e},\"im\":{:.17e}}}",
                state[i].re, state[i].im
            )
        })
        .collect();
    let floor = snapshot.floor.map_or("null".to_owned(), |n| n.to_string());
    write!(output,
        "{{\"id\":{index},\"name\":{:?},\"source\":{{\"path\":{:?},\"symbol\":{:?}}},\"enabled\":{},\"floor\":{floor},\"instructions\":[{}],\"metrics\":{{\"instructionCount\":{},\"measuredKernelSweeps\":{ticks},\"modeledFullStatePasses\":{},\"tileApplications\":{}}},\"state\":{{\"normSquared\":{norm:.17e},\"maxAbsoluteError\":{max_error:.17e},\"samples\":[{}]}}}}",
        snapshot.name, snapshot.path, snapshot.symbol, snapshot.enabled,
        instructions.join(","), instructions.len(),
        profile.passes(), profile.tile_apps(), samples.join(",")
    ).unwrap();
}

#[test]
#[ignore = "writes native compiler evidence only with PRISM_ATLAS_FUSION_OUTPUT set"]
fn export_atlas_fusion_traces() {
    let path = std::env::var_os("PRISM_ATLAS_FUSION_OUTPUT")
        .expect("set PRISM_ATLAS_FUSION_OUTPUT to the fixture destination");
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(revision.status.success());
    let revision = String::from_utf8(revision.stdout).unwrap();
    let status = std::process::Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--",
            "src",
            "Cargo.toml",
            "Cargo.lock",
        ])
        .output()
        .unwrap();
    assert!(status.status.success());
    let dirty_tree = !status.stdout.is_empty();
    let llc_bytes = crate::backend::cache::last_level_cache()
        .map_or("null".to_owned(), |cache| cache.bytes.to_string());
    let stage_floor = (0..=64).find(|&n| stages_at(n, n)).unwrap();
    let mut output = format!(
        "{{\"schemaVersion\":1,\"generator\":{{\"source\":\"src/circuit/fusion_atlas_tests.rs\",\"sourceRevision\":{:?},\"dirtyTree\":{dirty_tree},\"seed\":42,\"threads\":{threads},\"lastLevelCacheBytes\":{llc_bytes},\"stagePlannerFloor\":{stage_floor},\"execution\":\"native CPU statevector of every intermediate IR\",\"initialState\":\"normalized dense complex formula with seed offset 42\",\"counterSemantics\":\"Instrumented CPU full-state kernel entry ticks, checked against the static raw-stream sweep model; not measured bytes, time, cache misses, or GPU launches. Zero fill excluded.\",\"provenanceUnavailable\":\"Tracer recipe internals are private; stage indices do not claim fused source origins.\"}},\"records\":[",
        revision.trim()
    );
    let mut first = true;
    for name in ["legal-disjoint", "blocked-dependency", "diagonal-pair-runs"] {
        for width in [8, 10, 12, 14, 16, 20] {
            if width == 20 && name != "diagonal-pair-runs" {
                continue;
            }
            let circuit = family(name, width);
            let initial = initial_state(width);
            let (reference, _) = execute(&circuit, &initial);
            for profile in [FusionProfile::Host, FusionProfile::Device] {
                if width == 20 && profile == FusionProfile::Device {
                    continue;
                }
                if !first {
                    output.push(',');
                }
                first = false;
                write!(output,
                    "{{\"id\":\"{name}-{width}-{profile:?}\",\"family\":\"{name}\",\"width\":{width},\"profile\":\"{profile:?}\",\"finalMatchesPublicPipeline\":true,\"stages\":["
                ).unwrap();
                for (i, snapshot) in pipeline(&circuit, profile).iter().enumerate() {
                    if i != 0 {
                        output.push(',');
                    }
                    append_snapshot(&mut output, snapshot, i, &initial, &reference, threads);
                }
                output.push_str("]}");
            }
        }
    }
    output.push_str("]}\n");
    std::fs::write(path, output).unwrap();
}
