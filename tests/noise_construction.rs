//! The rule-based noise builder and the channels it can reach.
//!
//! Channel semantics are anchored against hand-computed values in
//! `tests/golden_small_circuits.rs`; what is pinned here is which events a rule
//! set emits, where they land, and that the two exact routes agree.

mod common;

use num_complex::Complex64;
use prism_q::circuit::{Circuit, ClassicalCondition, Instruction, guarded};
use prism_q::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel};
use prism_q::{
    BackendKind, CircuitBuilder, Gate, GateFilter, NoiseBuilder, PauliTerm, PrismError,
    density_matrix_expectation_values, simulate,
};
use smallvec::smallvec;

const SEED: u64 = 42;

fn mixed_circuit() -> Circuit {
    let mut c = Circuit::new(3, 3);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_barrier(&[0, 1, 2]);
    c.add_reset(2);
    c.add_gate(Gate::Rx(0.7), &[2]);
    for q in 0..3 {
        c.add_measure(q, q);
    }
    c
}

fn assert_same_events(built: &NoiseModel, manual: &NoiseModel) {
    assert_eq!(built.after_gate.len(), manual.after_gate.len());
    for (idx, (a, b)) in built
        .after_gate
        .iter()
        .zip(manual.after_gate.iter())
        .enumerate()
    {
        assert_eq!(a, b, "instruction {idx}");
    }
    assert_eq!(built.readout, manual.readout);
}

// Acceptance: a rule-based model reproduces the vector the manual constructor
// builds, event for event and slot for slot.
#[test]
fn uniform_rule_matches_the_manual_constructor() {
    let circuit = mixed_circuit();
    let p = 0.012;
    let branch = p / 3.0;

    let built = NoiseBuilder::new()
        .after_gates(
            GateFilter::all(),
            NoiseChannel::Pauli {
                px: branch,
                py: branch,
                pz: branch,
            },
        )
        .build(&circuit)
        .unwrap();

    assert_same_events(&built, &NoiseModel::uniform_depolarizing(&circuit, p));
}

#[test]
fn amplitude_damping_rule_matches_the_manual_constructor() {
    let circuit = mixed_circuit();
    let built = NoiseBuilder::new()
        .after_gates(
            GateFilter::all(),
            NoiseChannel::AmplitudeDamping { gamma: 0.03 },
        )
        .build(&circuit)
        .unwrap();

    assert_same_events(&built, &NoiseModel::with_amplitude_damping(&circuit, 0.03));
}

// Per-gate-type and per-qubit rates, the two selectors the constructors have no
// way to express.
#[test]
fn rules_select_by_gate_name_and_by_qubit() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[2]);

    let one_qubit = NoiseChannel::Depolarizing { p: 0.001 };
    let two_qubit = NoiseChannel::Depolarizing { p: 0.02 };
    let noise = NoiseBuilder::new()
        .after_gates(GateFilter::all().arity(1), one_qubit.clone())
        .after_gates(GateFilter::all().named("cx"), two_qubit.clone())
        .build(&circuit)
        .unwrap();

    assert_eq!(noise.after_gate[0], vec![event(&one_qubit, &[0])]);
    assert_eq!(
        noise.after_gate[1],
        vec![event(&two_qubit, &[0]), event(&two_qubit, &[1])]
    );
    assert_eq!(noise.after_gate[2], vec![event(&one_qubit, &[2])]);

    // The same rule narrowed to one qubit fires only there.
    let narrowed = NoiseBuilder::new()
        .after_gates(
            GateFilter::all().named("cx").on_qubits([1]),
            two_qubit.clone(),
        )
        .build(&circuit)
        .unwrap();
    assert!(narrowed.after_gate[0].is_empty());
    assert_eq!(narrowed.after_gate[1], vec![event(&two_qubit, &[1])]);
}

fn event(channel: &NoiseChannel, qubits: &[usize]) -> NoiseEvent {
    NoiseEvent {
        channel: channel.clone(),
        qubits: qubits.iter().copied().collect(),
    }
}

// Acceptance: idle decoherence lands on the qubits a layer does not touch.
// H on 0 and H on 1 share a layer, so only qubit 2 is idle and the single event
// rides the layer's last instruction. A per-instruction rule would instead emit
// two events per slot, and one that closed a layer at every instruction would
// put an event on slot 0.
#[test]
fn idle_noise_fires_on_untouched_qubits_per_layer() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);

    let idle = NoiseChannel::PhaseDamping { gamma: 0.02 };
    let noise = NoiseBuilder::new()
        .on_idle_qubits(idle.clone())
        .build(&circuit)
        .unwrap();

    assert!(noise.after_gate[0].is_empty());
    assert_eq!(noise.after_gate[1], vec![event(&idle, &[2])]);
}

// A circuit where every qubit is busy in every layer emits nothing. Packing
// instructions in index order rather than greedily would split this into three
// layers and charge two idle events against a register that never idles.
#[test]
fn idle_noise_charges_nothing_when_no_qubit_idles() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_gate(Gate::X, &[1]);

    let noise = NoiseBuilder::new()
        .on_idle_qubits(NoiseChannel::PhaseDamping { gamma: 0.02 })
        .build(&circuit)
        .unwrap();

    assert!(
        noise.after_gate.iter().all(|events| events.is_empty()),
        "{:?}",
        noise.after_gate
    );
}

// A barrier synchronizes its qubits without occupying a layer, so the gate
// after it starts a new one and the qubit the barrier crossed idles there.
#[test]
fn idle_noise_accounts_for_a_barrier() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_barrier(&[0, 1]);
    circuit.add_gate(Gate::X, &[0]);

    let idle = NoiseChannel::PhaseDamping { gamma: 0.02 };
    let noise = NoiseBuilder::new()
        .on_idle_qubits(idle.clone())
        .build(&circuit)
        .unwrap();

    assert_eq!(noise.after_gate[0], vec![event(&idle, &[1])]);
    assert!(noise.after_gate[1].is_empty());
    assert_eq!(noise.after_gate[2], vec![event(&idle, &[1])]);
}

// A qubit left out of a circuit entirely still decoheres: amplitude damping at
// gamma = 1 on the idle qubit of an X-prepared register drives it back to |0>,
// so the joint distribution collapses onto the one basis state.
#[test]
fn idle_noise_reaches_a_qubit_no_instruction_names() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_gate(Gate::Id, &[0]);

    let noise = NoiseBuilder::new()
        .on_idle_qubits(NoiseChannel::AmplitudeDamping { gamma: 1.0 })
        .build(&circuit)
        .unwrap();

    // Layers: {X0, X1} then {Id0}, so qubit 1 is idle in the second and decays.
    let z = density_matrix_expectation_values(
        &circuit,
        &[vec![PauliTerm::z(0)], vec![PauliTerm::z(1)]],
        Some(&noise),
        SEED,
    )
    .unwrap();
    assert!((z[0] + 1.0).abs() < 1e-12, "qubit 0 stays in |1>: {}", z[0]);
    assert!(
        (z[1] - 1.0).abs() < 1e-12,
        "qubit 1 decays to |0>: {}",
        z[1]
    );
}

// Reset error: a deterministic X after the reset leaves the qubit in |1>.
#[test]
fn reset_noise_fires_after_a_reset() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_reset(0);

    let noise = NoiseBuilder::new()
        .after_resets(NoiseChannel::Pauli {
            px: 1.0,
            py: 0.0,
            pz: 0.0,
        })
        .build(&circuit)
        .unwrap();

    assert!(noise.after_gate[0].is_empty());
    let z =
        density_matrix_expectation_values(&circuit, &[vec![PauliTerm::z(0)]], Some(&noise), SEED)
            .unwrap();
    assert!(
        (z[0] + 1.0).abs() < 1e-12,
        "expected |1>, got <Z> = {}",
        z[0]
    );
}

// Mid-circuit measurement error rides the preceding instruction's slot, so the
// bit a conditional reads is the faulty one. Every measurement gets its own
// deterministic flip: bit 0 reads 1 off a qubit prepared in |0>, and bit 1
// reads 0 only if the conditional fired on that faulty bit and the second flip
// then undid its X. A conditional that never fired would report bit 1 as 1.
#[test]
fn measurement_noise_reaches_the_bit_a_conditional_reads() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Id, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(
        guarded(
            ClassicalCondition::BitIsOne(0),
            vec![Instruction::Gate {
                gate: Gate::X,
                targets: smallvec![1],
            }],
        )
        .unwrap(),
    );
    circuit.add_measure(1, 1);

    let noise = NoiseBuilder::new()
        .before_measurements(NoiseChannel::Pauli {
            px: 1.0,
            py: 0.0,
            pz: 0.0,
        })
        .build(&circuit)
        .unwrap();
    assert_eq!(noise.after_gate[0].len(), 1);
    assert_eq!(noise.after_gate[0][0].qubits.as_slice(), &[0]);

    let shots = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(64)
        .unwrap();
    for shot in &shots.shots {
        assert_eq!(
            shot,
            &vec![true, false],
            "flip must feed the guarded branch"
        );
    }
}

#[test]
fn measurement_noise_needs_a_slot_before_the_measurement() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_measure(0, 0);

    let err = NoiseBuilder::new()
        .before_measurements(NoiseChannel::Depolarizing { p: 0.01 })
        .build(&circuit)
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::InvalidParameter { message }
            if message.contains("instruction 0 is a measurement")),
        "{err:?}"
    );
}

// Crosstalk names spectators through a coupling map: a Cx on (0, 1) reaches
// qubit 2 through the 1-2 edge and nothing else, and the gate's own targets are
// never their own spectators.
#[test]
fn crosstalk_fires_on_coupled_spectators_only() {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);

    let channel = NoiseChannel::Depolarizing { p: 0.005 };
    let noise = NoiseBuilder::new()
        .crosstalk(
            GateFilter::all().arity(2),
            [(0, 1), (1, 2), (2, 3)],
            channel.clone(),
        )
        .build(&circuit)
        .unwrap();

    assert_eq!(noise.after_gate[0], vec![event(&channel, &[2])]);
}

// Two targets sharing one spectator give it a single one-qubit event, not one
// per target.
#[test]
fn crosstalk_applies_a_shared_spectator_once() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);

    let channel = NoiseChannel::Depolarizing { p: 0.005 };
    let noise = NoiseBuilder::new()
        .crosstalk(
            GateFilter::all().arity(2),
            [(0, 2), (1, 2)],
            channel.clone(),
        )
        .build(&circuit)
        .unwrap();

    assert_eq!(noise.after_gate[0], vec![event(&channel, &[2])]);
}

// A crosstalk rule honours the filter's qubit set: narrowing it to target 1
// drops the spectators reached through target 0.
#[test]
fn crosstalk_honours_the_filter_qubit_set() {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);

    let channel = NoiseChannel::Depolarizing { p: 0.005 };
    let noise = NoiseBuilder::new()
        .crosstalk(
            GateFilter::all().arity(2).on_qubits([1]),
            [(0, 2), (1, 3)],
            channel.clone(),
        )
        .build(&circuit)
        .unwrap();

    assert_eq!(noise.after_gate[0], vec![event(&channel, &[3])]);
}

// `on_targets` is directed, so a rule keyed on the edge (0, 1) leaves cx(1, 0)
// alone. A coupling-map calibration table needs that distinction; `on_qubits`
// is a set and cannot make it.
#[test]
fn on_targets_distinguishes_the_two_directions_of_an_edge() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Cx, &[1, 0]);

    let channel = NoiseChannel::Depolarizing { p: 0.02 };
    let noise = NoiseBuilder::new()
        .after_gates_joint(
            GateFilter::all().on_targets([0, 1]),
            NoiseChannel::TwoQubitDepolarizing { p: 0.02 },
        )
        .after_gates(GateFilter::all().on_targets([1, 0]), channel.clone())
        .build(&circuit)
        .unwrap();

    assert_eq!(noise.after_gate[0].len(), 1);
    assert_eq!(noise.after_gate[0][0].qubits.as_slice(), &[0, 1]);
    assert_eq!(
        noise.after_gate[1],
        vec![event(&channel, &[1]), event(&channel, &[0])]
    );
}

// A two-qubit crosstalk channel lands on the (target, spectator) pair with the
// target first, which is the packing the channel's Kraus operators use.
#[test]
fn two_qubit_crosstalk_pairs_the_target_with_the_spectator() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);

    let noise = NoiseBuilder::new()
        .crosstalk(
            GateFilter::all().arity(2),
            [(1, 2)],
            NoiseChannel::TwoQubitDepolarizing { p: 0.004 },
        )
        .build(&circuit)
        .unwrap();

    assert_eq!(noise.after_gate[0].len(), 1);
    assert_eq!(noise.after_gate[0][0].qubits.as_slice(), &[1, 2]);
}

// Coherent over-rotation: an rx(theta) under a relative excess of `eps` behaves
// as rx(theta * (1 + eps)), so P(1) on |0> is sin^2(theta(1+eps)/2).
#[test]
fn over_rotation_scales_the_gate_angle() {
    let theta = std::f64::consts::FRAC_PI_2;
    let relative = 0.1;
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Rx(theta), &[0]);

    let noise = NoiseBuilder::new()
        .over_rotation(GateFilter::all().named("rx"), relative)
        .build(&circuit)
        .unwrap();

    let z =
        density_matrix_expectation_values(&circuit, &[vec![PauliTerm::z(0)]], Some(&noise), SEED)
            .unwrap();
    let expected = (theta * (1.0 + relative)).cos();
    assert!(
        (z[0] - expected).abs() < 1e-12,
        "expected <Z> = {expected}, got {}",
        z[0]
    );
}

#[test]
fn over_rotation_skips_gates_with_no_single_angle() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);

    let noise = NoiseBuilder::new()
        .over_rotation(GateFilter::all(), 0.1)
        .build(&circuit)
        .unwrap();
    assert!(noise.after_gate[0].is_empty());
}

// Acceptance: per-bit readout without reaching into `NoiseModel::readout`.
#[test]
fn per_bit_readout_is_reachable_from_both_construction_paths() {
    let circuit = CircuitBuilder::new_with_classical(2, 2).h(0).build();

    let built = NoiseBuilder::new()
        .uniform_readout_error(0.01, 0.02)
        .readout_error(1, 0.3, 0.4)
        .build(&circuit)
        .unwrap();
    assert_eq!(built.readout[0].as_ref().unwrap().p01, 0.01);
    assert_eq!(built.readout[1].as_ref().unwrap().p01, 0.3);

    let mut manual = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    manual.set_bit_readout_error(1, 0.3, 0.4);
    assert!(manual.readout[0].is_none());
    assert_eq!(manual.readout[1].as_ref().unwrap().p10, 0.4);
}

#[test]
fn readout_error_outside_the_register_rejected() {
    let circuit = CircuitBuilder::new_with_classical(1, 1).h(0).build();
    let err = NoiseBuilder::new()
        .readout_error(3, 0.1, 0.1)
        .build(&circuit)
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::InvalidParameter { message }
            if message.contains("classical bit 3")),
        "{err:?}"
    );
}

/// `sqrt(1-p) I` and `sqrt(p) (X (x) X)`, a correlated bit flip: the two qubits
/// flip together or not at all, so `01` and `10` never appear.
fn correlated_bit_flip(p: f64) -> NoiseChannel {
    let zero = Complex64::new(0.0, 0.0);
    let mut keep = [[zero; 4]; 4];
    let mut flip = [[zero; 4]; 4];
    for t in 0..4 {
        keep[t][t] = Complex64::new((1.0 - p).sqrt(), 0.0);
        flip[t][3 - t] = Complex64::new(p.sqrt(), 0.0);
    }
    NoiseChannel::Kraus2q {
        kraus: vec![keep, flip],
    }
}

// Acceptance: a two-qubit Kraus channel on the density matrix. The exact
// mixture is (1-p)|00><00| + p|11><11|, hand-computed from the Kraus pair.
#[test]
fn two_qubit_kraus_runs_on_the_density_matrix() {
    let p = 0.3;
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::Id, &[0]);

    let noise = NoiseBuilder::new()
        .after_gates_joint(GateFilter::all().arity(1), correlated_bit_flip(p))
        .build(&circuit);
    // A joint rule needs a gate whose arity matches the channel; a one-qubit
    // gate is not one, so the rule stays silent.
    assert!(noise.unwrap().after_gate[0].is_empty());

    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = NoiseBuilder::new()
        .after_gates_joint(GateFilter::all().arity(2), correlated_bit_flip(p))
        .build(&circuit)
        .unwrap();

    let z = density_matrix_expectation_values(
        &circuit,
        &[
            vec![PauliTerm::z(0)],
            vec![PauliTerm::z(1)],
            vec![PauliTerm::z(0), PauliTerm::z(1)],
        ],
        Some(&noise),
        SEED,
    )
    .unwrap();
    let expected = [1.0 - 2.0 * p, 1.0 - 2.0 * p, 1.0];
    for (i, (a, e)) in z.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-12, "term {i}: expected {e}, got {a}");
    }
}

// Acceptance: the same channel on trajectories. A trajectory result is a
// sampled estimate, so it is compared to the exact rate with a stated tolerance
// at a fixed seed, never for equality. Three standard errors of a binomial at
// 4096 shots and p = 0.3 is 3*sqrt(0.3*0.7/4096) = 0.0215.
#[test]
fn two_qubit_kraus_runs_on_trajectories() {
    let p = 0.3;
    let shots = 4096;
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);

    let noise = NoiseBuilder::new()
        .after_gates_joint(GateFilter::all().arity(2), correlated_bit_flip(p))
        .build(&circuit)
        .unwrap();

    let result = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(shots)
        .unwrap();

    let mut both = 0usize;
    for shot in &result.shots {
        // Structural, not sampled: no operator of this channel produces a split
        // outcome from |00>.
        assert_eq!(
            shot[0], shot[1],
            "a correlated bit flip never produces a split outcome"
        );
        both += usize::from(shot[0]);
    }
    let measured = both as f64 / shots as f64;
    assert!(
        (measured - p).abs() < 0.0215,
        "trajectory estimate {measured} against the exact {p}"
    );
}

/// `sqrt(1-p) I` and `sqrt(p) (Z (x) I)`: dephasing of the high-index qubit
/// alone, which is the first qubit the event names. Asymmetric under swapping
/// the pair, unlike `X (x) X` or `Z (x) Z`.
fn dephase_high_qubit(p: f64) -> NoiseChannel {
    let zero = Complex64::new(0.0, 0.0);
    let diag = |scale: f64, signs: [f64; 4]| {
        let mut m = [[zero; 4]; 4];
        for (t, sign) in signs.iter().enumerate() {
            m[t][t] = Complex64::new(scale * sign, 0.0);
        }
        m
    };
    NoiseChannel::Kraus2q {
        kraus: vec![
            diag((1.0 - p).sqrt(), [1.0, 1.0, 1.0, 1.0]),
            diag(p.sqrt(), [1.0, 1.0, -1.0, -1.0]),
        ],
    }
}

// The trajectory route composes three index conventions: the pair passed to
// the two-qubit reduction, the row and column order the branch weight reads,
// and the target order the selected operator is applied on. A channel symmetric
// under the qubit swap tells none of them apart. Z on one qubit does: it
// dephases whichever qubit the event names first, which reads as a loss of <X>
// there and nowhere else, and swapping the event's qubits moves it.
#[test]
fn two_qubit_kraus_trajectory_dephases_the_first_named_qubit() {
    let p = 0.5;
    let shots = 2048;

    for (q0, q1, x0, x1) in [(0usize, 1usize, 0.0, 1.0), (1, 0, 1.0, 0.0)] {
        let mut circuit = Circuit::new(2, 2);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[1]);
        let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        noise.after_gate[1] = vec![NoiseEvent {
            channel: dephase_high_qubit(p),
            qubits: smallvec![q0, q1],
        }];
        // Read <X> by rotating into the Z basis before measuring.
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        noise
            .after_gate
            .resize_with(circuit.instructions.len(), Vec::new);

        let result = simulate(&circuit)
            .backend(BackendKind::Statevector)
            .noise(&noise)
            .seed(SEED)
            .shots(shots)
            .unwrap();

        let mean = |bit: usize| {
            let ones = result.shots.iter().filter(|s| s[bit]).count();
            1.0 - 2.0 * (ones as f64 / shots as f64)
        };
        // Dephasing at p = 0.5 takes <X> to 0 on the named qubit and leaves the
        // other at 1. Three standard errors at 2048 shots is 0.033.
        assert!(
            (mean(0) - x0).abs() < 0.04,
            "({q0},{q1}) <X0>: expected {x0}, got {}",
            mean(0)
        );
        assert!(
            (mean(1) - x1).abs() < 0.04,
            "({q0},{q1}) <X1>: expected {x1}, got {}",
            mean(1)
        );
    }
}

// A branch weight that depends on the state, which every channel above hides
// behind an effect proportional to the identity. The jump operator
// sqrt(gamma) |00><11| has its effect supported on |11> alone, so the branch
// rate tracks the |11> population instead of a constant, and the reduction the
// weight is drawn from has to be the real one. Compared against the exact
// mixture, not against another sampler.
#[test]
fn two_qubit_kraus_trajectory_branch_weight_reads_the_state() {
    let gamma = 0.6f64;
    let shots = 8192;
    let zero = Complex64::new(0.0, 0.0);
    let mut keep = [[zero; 4]; 4];
    for (t, row) in keep.iter_mut().enumerate() {
        row[t] = Complex64::new(if t == 3 { (1.0 - gamma).sqrt() } else { 1.0 }, 0.0);
    }
    let mut jump = [[zero; 4]; 4];
    jump[0][3] = Complex64::new(gamma.sqrt(), 0.0);
    let channel = NoiseChannel::Kraus2q {
        kraus: vec![keep, jump],
    };

    // Ry puts qubit 0 at P(1) = 0.75 and the Cx copies it, so the |11> weight
    // the jump reads is 0.75 rather than the 0.5 a Bell pair would give.
    let theta = 2.0 * 0.25f64.sqrt().acos();
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Ry(theta), &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[1] = vec![NoiseEvent {
        channel,
        qubits: smallvec![0, 1],
    }];
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    noise
        .after_gate
        .resize_with(circuit.instructions.len(), Vec::new);

    let exact = density_matrix_expectation_values(
        &circuit,
        &[vec![PauliTerm::z(0)], vec![PauliTerm::z(1)]],
        Some(&noise),
        SEED,
    )
    .unwrap();

    let result = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    let mean = |bit: usize| {
        let ones = result.shots.iter().filter(|s| s[bit]).count();
        1.0 - 2.0 * (ones as f64 / shots as f64)
    };

    // Three standard errors at 8192 shots is 0.033. A weight drawn from a
    // constant instead of the state would put <Z> at 1 - 2*0.75*(1-gamma/2),
    // which is 0.175 away from the exact value here.
    for (bit, expected) in exact.iter().enumerate() {
        assert!(
            (mean(bit) - expected).abs() < 0.035,
            "bit {bit}: trajectory {} against the exact {expected}",
            mean(bit)
        );
    }
}

// A two-qubit Kraus channel needs a backend that can reduce a pair, and only
// the host statevector can. The rejection lands at dispatch, before a shot
// allocates state, rather than part way through the first trajectory.
#[test]
fn two_qubit_kraus_rejected_on_a_backend_without_the_reduction() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);

    let noise = NoiseBuilder::new()
        .after_gates_joint(GateFilter::all().arity(2), correlated_bit_flip(0.1))
        .build(&circuit)
        .unwrap();

    let err = simulate(&circuit)
        .backend(BackendKind::Mps { max_bond_dim: 16 })
        .noise(&noise)
        .seed(SEED)
        .shots(4)
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { reason, .. }
            if reason.contains("two-qubit reduced density matrix")),
        "{err:?}"
    );
}

#[test]
fn two_qubit_kraus_rejects_a_non_trace_preserving_set() {
    let zero = Complex64::new(0.0, 0.0);
    let mut scaled = [[zero; 4]; 4];
    for (t, row) in scaled.iter_mut().enumerate() {
        row[t] = Complex64::new(0.5, 0.0);
    }
    let channel = NoiseChannel::Kraus2q {
        kraus: vec![scaled],
    };
    assert!(channel.validate().is_err());
    assert!(!channel.is_exactly_samplable());
}

// A guarded region leaves its body without event slots, so a model carrying
// quantum events is still rejected. A readout-only model has nothing to lose
// there and is accepted.
#[test]
fn guarded_regions_reject_quantum_events_but_not_readout_error() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(
        guarded(
            ClassicalCondition::BitIsOne(0),
            vec![
                Instruction::Gate {
                    gate: Gate::X,
                    targets: smallvec![1],
                },
                Instruction::Reset { qubit: 1 },
            ],
        )
        .unwrap(),
    );

    let with_events = NoiseBuilder::new()
        .after_gates(GateFilter::all(), NoiseChannel::Depolarizing { p: 0.01 })
        .build(&circuit)
        .unwrap_err();
    assert!(
        matches!(&with_events, PrismError::IncompatibleBackend { reason, .. }
            if reason.contains("guarded region")),
        "{with_events:?}"
    );

    let readout_only = NoiseBuilder::new()
        .uniform_readout_error(0.05, 0.05)
        .build(&circuit)
        .unwrap();
    assert!(readout_only.after_gate.iter().all(|e| e.is_empty()));
    assert!(readout_only.has_noise());
}

// The remaining single-angle rotations the rule accepts. Ry moves <X> off a
// |0>-prepared qubit; Rz and P move <Y> off a |+>-prepared one.
#[test]
fn over_rotation_covers_every_rotation_it_accepts() {
    let theta = std::f64::consts::FRAC_PI_2;
    let relative = 0.1;

    let mut ry = Circuit::new(1, 0);
    ry.add_gate(Gate::Ry(theta), &[0]);
    let noise = NoiseBuilder::new()
        .over_rotation(GateFilter::all().named("ry"), relative)
        .build(&ry)
        .unwrap();
    let z = density_matrix_expectation_values(&ry, &[vec![PauliTerm::z(0)]], Some(&noise), SEED)
        .unwrap();
    assert!(
        (z[0] - (theta * (1.0 + relative)).cos()).abs() < 1e-12,
        "ry: got {}",
        z[0]
    );

    for (name, gate) in [("rz", Gate::Rz(theta)), ("p", Gate::P(theta))] {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(gate, &[0]);
        let noise = NoiseBuilder::new()
            .over_rotation(GateFilter::all().named(name), relative)
            .build(&c)
            .unwrap();
        let y = density_matrix_expectation_values(&c, &[vec![PauliTerm::y(0)]], Some(&noise), SEED)
            .unwrap();
        assert!(
            (y[0] - (theta * (1.0 + relative)).sin()).abs() < 1e-12,
            "{name}: got {}",
            y[0]
        );
    }
}

// Both dense Kraus arities reject an empty operator list and a non-finite
// entry, and both name the variant in the message.
#[test]
fn dense_kraus_sets_reject_empty_and_non_finite() {
    let zero = Complex64::new(0.0, 0.0);

    let empty_1q = NoiseChannel::Custom { kraus: Vec::new() };
    let empty_2q = NoiseChannel::Kraus2q { kraus: Vec::new() };
    for (label, channel) in [("Custom", empty_1q), ("Kraus2q", empty_2q)] {
        let err = channel.validate().unwrap_err();
        assert!(
            matches!(&err, PrismError::InvalidParameter { message }
                if message.contains(label) && message.contains("at least one operator")),
            "{err:?}"
        );
    }

    let nan = Complex64::new(f64::NAN, 0.0);
    let bad_1q = NoiseChannel::Custom {
        kraus: vec![[[nan, zero], [zero, zero]]],
    };
    let mut bad_2q_op = [[zero; 4]; 4];
    bad_2q_op[2][1] = nan;
    let bad_2q = NoiseChannel::Kraus2q {
        kraus: vec![bad_2q_op],
    };
    for (label, channel) in [("Custom", bad_1q), ("Kraus2q", bad_2q)] {
        let err = channel.validate().unwrap_err();
        assert!(
            matches!(&err, PrismError::InvalidParameter { message }
                if message.contains(label) && message.contains("must be finite")),
            "{err:?}"
        );
    }
}

// Per-bit readout reaching a shot, not just the struct: bit 1 inverts on every
// shot and bit 0 never does.
#[test]
fn per_bit_readout_reaches_the_reported_bits() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Id, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);

    let noise = NoiseBuilder::new()
        .readout_error(1, 1.0, 1.0)
        .build(&circuit)
        .unwrap();

    let result = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(32)
        .unwrap();
    for shot in &result.shots {
        assert_eq!(shot, &vec![false, true], "only bit 1 carries readout error");
    }
}

#[test]
#[should_panic(expected = "outside the 2-bit register")]
fn set_bit_readout_error_rejects_a_bit_outside_the_register() {
    let circuit = CircuitBuilder::new_with_classical(2, 2).h(0).build();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.set_bit_readout_error(5, 0.1, 0.1);
}
