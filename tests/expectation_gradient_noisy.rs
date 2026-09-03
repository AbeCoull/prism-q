//! Parameter-shift gradients under a noise model: the density-matrix route
//! against central finite differences of the noisy expectation values, and the
//! rejections every other backend keeps.

mod common;

use common::SEED;
use prism_q::circuits;
use prism_q::{
    BackendKind, Circuit, Gate, Instruction, NoiseModel, Parameters, PauliTerm, simulate,
};

type Hamiltonian = Vec<(f64, Vec<PauliTerm>)>;

fn fixture() -> (Circuit, NoiseModel, Parameters, Hamiltonian) {
    let circuit = circuits::hardware_efficient_ansatz(6, 2, SEED);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
    let params = Parameters::all_rotations(&circuit);
    let hamiltonian = vec![
        (0.7, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (-0.3, vec![PauliTerm::x(2)]),
        (0.5, vec![PauliTerm::y(3), PauliTerm::z(4)]),
        (0.2, vec![PauliTerm::z(5)]),
    ];
    (circuit, noise, params, hamiltonian)
}

fn noisy_expval(circuit: &Circuit, noise: &NoiseModel, hamiltonian: &Hamiltonian) -> f64 {
    let observables: Vec<Vec<PauliTerm>> = hamiltonian.iter().map(|(_, p)| p.clone()).collect();
    let per_term = simulate(circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(noise)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    hamiltonian
        .iter()
        .zip(per_term)
        .map(|((c, _), v)| c * v)
        .sum()
}

fn shift_slot(circuit: &Circuit, params: &Parameters, slot: usize, delta: f64) -> Circuit {
    let mut out = circuit.clone();
    for link in params.links().iter().filter(|l| l.slot == slot) {
        if let Instruction::Gate { gate, .. } = &mut out.instructions[link.instruction] {
            *gate = match gate {
                Gate::Ry(t) => Gate::Ry(*t + delta),
                Gate::Rz(t) => Gate::Rz(*t + delta),
                other => panic!("fixture carries only Ry and Rz rotations, found {other:?}"),
            };
        }
    }
    out
}

#[test]
fn noisy_shift_matches_finite_differences_of_noisy_expectation_values() {
    let (circuit, noise, params, hamiltonian) = fixture();
    let g = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .expectation_gradient_shift(&hamiltonian, &params)
        .unwrap();

    assert!((g.value - noisy_expval(&circuit, &noise, &hamiltonian)).abs() < 1e-12);
    assert_eq!(g.gradient.len(), params.num_slots());
    let eps = 1e-4;
    let mut largest = 0.0_f64;
    for slot in 0..params.num_slots() {
        let plus = noisy_expval(
            &shift_slot(&circuit, &params, slot, eps),
            &noise,
            &hamiltonian,
        );
        let minus = noisy_expval(
            &shift_slot(&circuit, &params, slot, -eps),
            &noise,
            &hamiltonian,
        );
        let fd = (plus - minus) / (2.0 * eps);
        assert!(
            (g.gradient[slot] - fd).abs() < 1e-6,
            "slot {slot}: shift {} vs finite difference {fd}",
            g.gradient[slot]
        );
        largest = largest.max(g.gradient[slot].abs());
    }
    assert!(
        largest > 1e-3,
        "the fixture must have a non-trivial gradient"
    );
}

#[test]
fn noisy_shift_differs_from_the_noiseless_gradient() {
    let (circuit, noise, params, hamiltonian) = fixture();
    let noisy = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .expectation_gradient_shift(&hamiltonian, &params)
        .unwrap();
    let clean = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .expectation_gradient_shift(&hamiltonian, &params)
        .unwrap();
    let moved = noisy
        .gradient
        .iter()
        .zip(&clean.gradient)
        .any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(moved, "depolarizing at 2% per gate must move the gradient");
}

#[test]
fn noisy_shift_rejects_backends_without_the_mixture() {
    let (circuit, noise, params, hamiltonian) = fixture();
    for kind in [
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::Sparse,
    ] {
        let err = simulate(&circuit)
            .backend(kind.clone())
            .noise(&noise)
            .seed(SEED)
            .expectation_gradient_shift(&hamiltonian, &params)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("density-matrix") && err.contains("noise model"),
            "{kind:?}: {err}"
        );
    }
}

#[test]
fn adjoint_still_rejects_noise_and_names_the_shift_route() {
    let (circuit, noise, params, hamiltonian) = fixture();
    for kind in [BackendKind::Auto, BackendKind::DensityMatrix] {
        let err = simulate(&circuit)
            .backend(kind.clone())
            .noise(&noise)
            .seed(SEED)
            .expectation_gradient(&hamiltonian, &params)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("expectation_gradient_shift"),
            "{kind:?}: {err}"
        );
    }
}

#[test]
fn noisy_shift_rejects_mid_circuit_measurement() {
    let (mut circuit, _, _, hamiltonian) = fixture();
    circuit.num_classical_bits = 1;
    circuit.instructions.insert(
        3,
        Instruction::Measure {
            qubit: 0,
            classical_bit: 0,
        },
    );
    let params = Parameters::all_rotations(&circuit);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
    let err = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .expectation_gradient_shift(&hamiltonian, &params)
        .unwrap_err()
        .to_string();
    assert!(err.contains("measurement"), "{err}");
}
