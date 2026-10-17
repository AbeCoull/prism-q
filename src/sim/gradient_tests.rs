use super::*;
use crate::sim::unified_pauli::PauliTerm;

fn z_obs(qubit: usize) -> Vec<(f64, Vec<PauliTerm>)> {
    vec![(1.0, vec![PauliTerm::z(qubit)])]
}

#[test]
fn single_rx_gradient_matches_analytic() {
    // Rx(θ)|0>, <Z> = cos θ, d/dθ = -sin θ.
    let theta = 0.7;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(theta), &[0]);
    let mut params = Parameters::new(1);
    params.link(0, 0);

    let g = run_expectation_gradient(&c, &z_obs(0), &params, 42).unwrap();
    assert!((g.value - theta.cos()).abs() < 1e-12);
    assert!((g.gradient[0] - (-theta.sin())).abs() < 1e-9);
}

#[test]
fn ry_generator_carries_num_y() {
    // Ry(θ)|0>, <Z> = cos θ, d/dθ = -sin θ. Generator Y has num_y = 1.
    let theta = 1.3;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Ry(theta), &[0]);
    let mut params = Parameters::new(1);
    params.link(0, 0);

    let g = run_expectation_gradient(&c, &z_obs(0), &params, 42).unwrap();
    assert!((g.gradient[0] - (-theta.sin())).abs() < 1e-9);
}

#[test]
fn phase_projector_gradient() {
    // H then P(θ): |ψ> = (|0> + e^{iθ}|1>)/√2, <X> = cos θ, d/dθ = -sin θ.
    let theta = 0.9;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::P(theta), &[0]);
    let mut params = Parameters::new(1);
    params.link(1, 0);

    let obs = vec![(1.0, vec![PauliTerm::x(0)])];
    let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
    assert!((g.value - theta.cos()).abs() < 1e-12);
    assert!((g.gradient[0] - (-theta.sin())).abs() < 1e-9);
}

#[test]
fn shared_parameter_accumulates() {
    // Two Rx gates on separate qubits sharing one slot; each contributes
    // -sin θ to <Z0 + Z1>, so the shared gradient is -2 sin θ.
    let theta = 0.4;
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Rx(theta), &[0]);
    c.add_gate(Gate::Rx(theta), &[1]);
    let mut params = Parameters::new(1);
    params.link(0, 0);
    params.link(1, 0);

    let obs = vec![(1.0, vec![PauliTerm::z(0)]), (1.0, vec![PauliTerm::z(1)])];
    let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
    assert_eq!(g.gradient.len(), 1);
    assert!((g.gradient[0] - (-2.0 * theta.sin())).abs() < 1e-9);
}

fn hea_with_shared_slots(n: usize) -> (Circuit, Parameters) {
    let circuit = crate::circuits::hardware_efficient_ansatz(n, 2, 7);
    let mut params = Parameters::new(3);
    let rotations = circuit.instructions.iter().enumerate().filter(|(_, inst)| {
        matches!(
            inst,
            Instruction::Gate {
                gate: Gate::Ry(_) | Gate::Rz(_),
                ..
            }
        )
    });
    for (k, (i, _)) in rotations.enumerate() {
        params.link(i, k % 3);
    }
    (circuit, params)
}

fn slot_per_link(params: &Parameters) -> Parameters {
    let links = params.links();
    Parameters::from_links(
        links
            .iter()
            .enumerate()
            .map(|(slot, link)| ParamLink {
                instruction: link.instruction,
                slot,
            })
            .collect(),
        links.len(),
    )
}

fn weighted_energy(ham: &[(f64, Vec<PauliTerm>)], per_term: Vec<f64>) -> f64 {
    ham.iter()
        .zip(per_term)
        .map(|((coeff, _), v)| coeff * v)
        .sum()
}

// Ten qubits splits the links across workers under `parallel` and replays a
// fusion plan; the reference sums the same prepared terms in link order on
// one thread, so the bits must agree.
#[test]
fn shift_gradient_matches_a_sequential_sum_bit_for_bit() {
    let (circuit, params) = hea_with_shared_slots(10);
    let ham = vec![
        (0.5, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (1.5, vec![PauliTerm::x(3)]),
    ];
    let observables: Vec<Vec<PauliTerm>> = ham.iter().map(|(_, t)| t.clone()).collect();
    let links = params.links();
    let sites = slot_per_link(&params);
    let base = sites.values(&circuit).unwrap();
    let mut prepared = PreparedCircuit::new(circuit.clone(), sites).unwrap();
    assert!(prepared.reuses_fusion_plan());
    let mut energy = |values: &[f64]| {
        weighted_energy(
            &ham,
            prepared
                .expectation_values(values, &observables, 42)
                .unwrap(),
        )
    };
    let shift = std::f64::consts::FRAC_PI_2;
    let mut expected = vec![0.0; 3];
    for (k, link) in links.iter().enumerate() {
        let mut values = base.clone();
        values[k] = base[k] + shift;
        let plus = energy(&values);
        values[k] = base[k] - shift;
        let minus = energy(&values);
        expected[link.slot] += 0.5 * (plus - minus);
    }

    let g = run_expectation_gradient_shift(&circuit, &ham, &params, 42).unwrap();
    assert_eq!(g.gradient, expected);
}

// Angles of pi/2 on half the rotations shift to 0 and pi, which collapse a
// fused block to the identity or a named gate and force the replay onto the
// full pass pipeline.
#[test]
fn shift_gradient_matches_fresh_fusion_on_degenerate_shifts() {
    for n in [6, 10] {
        let (mut circuit, params) = hea_with_shared_slots(n);
        for link in params.links().iter().step_by(2) {
            *angle_mut(&mut circuit.instructions[link.instruction]) = std::f64::consts::FRAC_PI_2;
        }
        let ham: Vec<(f64, Vec<PauliTerm>)> = (0..n - 1)
            .map(|q| (1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]))
            .chain([(0.7, vec![PauliTerm::x(2)])])
            .collect();
        let observables: Vec<Vec<PauliTerm>> = ham.iter().map(|(_, t)| t.clone()).collect();
        let energy = |c: &Circuit| {
            weighted_energy(
                &ham,
                super::super::run_expectation_values_with(
                    BackendKind::Auto,
                    c,
                    &observables,
                    42,
                    &ResolvedLimits::process_default(),
                )
                .unwrap(),
            )
        };
        let shift = std::f64::consts::FRAC_PI_2;
        let mut expected = vec![0.0; params.num_slots()];
        for link in params.links() {
            let mut shifted = circuit.clone();
            let base = *angle_mut(&mut shifted.instructions[link.instruction]);
            *angle_mut(&mut shifted.instructions[link.instruction]) = base + shift;
            let plus = energy(&shifted);
            *angle_mut(&mut shifted.instructions[link.instruction]) = base - shift;
            let minus = energy(&shifted);
            expected[link.slot] += 0.5 * (plus - minus);
        }

        if n == 10 {
            let sites = slot_per_link(&params);
            let base = sites.values(&circuit).unwrap();
            let mut prepared = PreparedCircuit::new(circuit.clone(), sites).unwrap();
            assert!(prepared.reuses_fusion_plan());
            let replayed = prepared.bind_fused(&base).unwrap().instructions.len();
            let fell_back = (0..base.len()).any(|k| {
                let mut values = base.clone();
                values[k] -= shift;
                prepared.bind_fused(&values).unwrap().instructions.len() != replayed
            });
            assert!(fell_back, "no shifted binding left the captured plan");
        }

        let g = run_expectation_gradient_shift(&circuit, &ham, &params, 42).unwrap();
        assert!((g.value - energy(&circuit)).abs() < 1e-10, "{n}q value");
        for (slot, (got, want)) in g.gradient.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "{n}q slot {slot}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn empty_params_returns_value_only() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(0.5), &[0]);
    let g = run_expectation_gradient(&c, &z_obs(0), &Parameters::new(0), 42).unwrap();
    assert!(g.gradient.is_empty());
    assert!((g.value - 0.5f64.cos()).abs() < 1e-12);
}

#[test]
fn nondifferentiable_trainable_gate_is_rejected() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let mut params = Parameters::new(1);
    params.link(0, 0);
    assert!(run_expectation_gradient(&c, &z_obs(0), &params, 42).is_err());
}

#[test]
fn nonunitary_circuit_is_rejected() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::Rx(0.3), &[0]);
    c.add_measure(0, 0);
    assert!(run_expectation_gradient(&c, &z_obs(0), &Parameters::new(0), 42).is_err());
}

#[test]
fn single_term_hamiltonian_matches_analytic() {
    // Ry(theta)|0>, <X> = sin theta, d/dtheta = cos theta.
    let theta = 0.6;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Ry(theta), &[0]);
    let params = Parameters::all_rotations(&c);

    let obs = vec![(1.0, vec![PauliTerm::x(0)])];
    let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
    assert!((g.value - theta.sin()).abs() < 1e-12);
    assert!((g.gradient[0] - theta.cos()).abs() < 1e-12);
}

#[test]
fn diagonal_hamiltonian_matches_analytic() {
    // Two independent Ry rotations under Z0, Z1 and Z0Z1: every term is
    // diagonal, so nothing moves an amplitude.
    let (a, b) = (0.4, 1.1);
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Ry(a), &[0]);
    c.add_gate(Gate::Ry(b), &[1]);
    let params = Parameters::all_rotations(&c);

    let obs = vec![
        (1.0, vec![PauliTerm::z(0)]),
        (0.5, vec![PauliTerm::z(1)]),
        (0.25, vec![PauliTerm::z(0), PauliTerm::z(1)]),
    ];
    let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
    let value = a.cos() + 0.5 * b.cos() + 0.25 * a.cos() * b.cos();
    assert!((g.value - value).abs() < 1e-12);
    assert!((g.gradient[0] - (-a.sin() - 0.25 * a.sin() * b.cos())).abs() < 1e-12);
    assert!((g.gradient[1] - (-0.5 * b.sin() - 0.25 * a.cos() * b.sin())).abs() < 1e-12);
}

#[test]
fn grouped_x_mask_terms_match_parameter_shift() {
    // X0, Y0, X0Z1 and Y0Z2 all carry the X mask of qubit 0, four terms
    // reading the same amplitude.
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Ry(0.4), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rx(0.9), &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Rz(0.3), &[2]);
    let params = Parameters::all_rotations(&c);

    let obs = vec![
        (1.0, vec![PauliTerm::x(0)]),
        (-0.5, vec![PauliTerm::y(0)]),
        (0.75, vec![PauliTerm::x(0), PauliTerm::z(1)]),
        (0.25, vec![PauliTerm::y(0), PauliTerm::z(2)]),
    ];
    let adjoint = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
    let shift = run_expectation_gradient_shift(&c, &obs, &params, 42).unwrap();
    assert!((adjoint.value - shift.value).abs() < 1e-12);
    for (slot, (&got, &want)) in adjoint.gradient.iter().zip(&shift.gradient).enumerate() {
        assert!((got - want).abs() < 1e-12, "slot {slot}: {got} vs {want}");
    }
}

#[test]
fn parallel_threshold_width_matches_parameter_shift() {
    // 2^15 amplitudes, the width at which the gather fans out to Rayon.
    let mut c = Circuit::new(15, 0);
    c.add_gate(Gate::Ry(0.3), &[0]);
    c.add_gate(Gate::Cx, &[0, 7]);
    c.add_gate(Gate::Rx(0.8), &[7]);
    c.add_gate(Gate::Cx, &[7, 14]);
    c.add_gate(Gate::Rz(0.5), &[14]);
    let params = Parameters::all_rotations(&c);

    let obs = vec![
        (1.0, vec![PauliTerm::z(0)]),
        (0.5, vec![PauliTerm::x(7)]),
        (-0.25, vec![PauliTerm::y(7), PauliTerm::z(14)]),
        (0.75, vec![PauliTerm::x(7), PauliTerm::z(0)]),
    ];
    let adjoint = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
    let shift = run_expectation_gradient_shift(&c, &obs, &params, 42).unwrap();
    assert!((adjoint.value - shift.value).abs() < 1e-12);
    for (slot, (&got, &want)) in adjoint.gradient.iter().zip(&shift.gradient).enumerate() {
        assert!((got - want).abs() < 1e-12, "slot {slot}: {got} vs {want}");
    }
}

#[test]
fn the_gather_matches_a_term_by_term_reference() {
    // Sum the terms one at a time, the shape the gather replaces, and
    // compare on one register. Split across two chunks so the gather's base
    // arithmetic is covered as well.
    let dim = 1usize << 6;
    let phi: Vec<Complex64> = (0..dim)
        .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
        .collect();
    let masked = vec![
        (1.0, 0usize, 0b101usize, 0u32),
        (-0.5, 0b010, 0b100, 1),
        (0.75, 0b010, 0b001, 0),
        (0.25, 0b110, 0b011, 2),
    ];

    let mut want = vec![Complex64::new(0.0, 0.0); dim];
    for &(coeff, xmask, zmask, num_y) in &masked {
        let factor = Complex64::new(coeff, 0.0) * i_pow(num_y);
        for (j, &amp) in phi.iter().enumerate() {
            let sign = if (j & zmask).count_ones() & 1 == 1 {
                -1.0
            } else {
                1.0
            };
            want[j ^ xmask] += factor * sign * amp;
        }
    }
    let want_value: f64 = phi.iter().zip(&want).map(|(p, l)| (p.conj() * l).re).sum();

    let groups = group_terms_by_xmask(&masked);
    let mut got = vec![Complex64::new(0.0, 0.0); dim];
    let (lo, hi) = got.split_at_mut(dim / 2);
    let got_value =
        gather_lambda_chunk(&groups, &phi, 0, lo) + gather_lambda_chunk(&groups, &phi, dim / 2, hi);

    assert!((want_value - got_value).abs() < 1e-12);
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!((w - g).norm() < 1e-12, "slot {i}: {w} vs {g}");
    }
}

#[test]
fn pauli_rot_generators_commute_on_an_even_anticommuting_count() {
    use crate::sim::unified_pauli::PauliAxis;
    let gate = |axes: &[PauliAxis], targets: &[usize]| {
        Footprint::pauli(
            generator_masks(GeneratorKind::RotPauli(axes), targets),
            targets,
        )
    };
    let x0y1 = gate(&[PauliAxis::X, PauliAxis::Y], &[0, 1]);
    let y0x1 = gate(&[PauliAxis::Y, PauliAxis::X], &[0, 1]);
    let wide = gate(
        &[PauliAxis::Y, PauliAxis::X, PauliAxis::X, PauliAxis::X],
        &[0, 1, 2, 3],
    );
    let x1y2 = gate(&[PauliAxis::X, PauliAxis::Y], &[1, 2]);

    assert!(x0y1.commutes_with(&y0x1));
    assert!(x0y1.commutes_with(&x0y1));
    assert!(!wide.commutes_with(&x1y2));
    assert!(wide.commutes_with(&y0x1));
}

#[test]
fn footprints_commute_through_control_and_target_actions() {
    let cx = Footprint::of(&Gate::Cx, &[0, 1]);
    let commutes = |gate: Gate, targets: &[usize]| {
        let f = Footprint::of(&gate, targets);
        assert_eq!(f.commutes_with(&cx), cx.commutes_with(&f));
        f.commutes_with(&cx)
    };
    assert!(commutes(Gate::Rz(0.3), &[0]));
    assert!(commutes(Gate::Rx(0.3), &[1]));
    assert!(commutes(Gate::Cz, &[0, 2]));
    assert!(commutes(Gate::H, &[2]));
    assert!(!commutes(Gate::Rz(0.3), &[1]));
    assert!(!commutes(Gate::Ry(0.3), &[0]));
    assert!(!commutes(Gate::Rzz(0.3), &[0, 1]));
    assert!(!commutes(Gate::H, &[0]));
    assert!(!commutes(Gate::Cx, &[1, 0]));
}

fn plan_for(circuit: &Circuit, params: &Parameters) -> SweepPlan {
    let mut linked: Vec<usize> = params.links().iter().map(|l| l.instruction).collect();
    linked.sort_unstable();
    let start = linked[0];
    let sweep: Vec<usize> = (start..circuit.instructions.len()).collect();
    let trainable: Vec<bool> = sweep
        .iter()
        .map(|i| linked.binary_search(i).is_ok())
        .collect();
    let footprints: Vec<Footprint> = sweep
        .iter()
        .map(|&i| {
            let Instruction::Gate { gate, targets } = &circuit.instructions[i] else {
                unreachable!()
            };
            Footprint::of(gate, targets)
        })
        .collect();
    plan_sweep(&footprints, &trainable, circuit.num_qubits)
}

// Interleaved Ry and Rz on each qubit evaluate as one Rz step and one Ry
// step per layer, where a run that must commute internally splits at every
// qubit.
#[test]
fn a_hardware_efficient_layer_takes_two_steps() {
    let (n, layers) = (8, 3);
    let circuit = crate::circuits::hardware_efficient_ansatz(n, layers, 7);
    let plan = plan_for(&circuit, &Parameters::all_rotations(&circuit));
    assert_eq!(plan.evaluate.len(), 2 * n * layers);
    assert!(
        plan.last_step <= 2 * layers,
        "{} steps for {layers} layers",
        plan.last_step + 1
    );
    let stretches = plan
        .invert
        .iter()
        .map(|&(t, _)| t)
        .collect::<std::collections::BTreeSet<_>>();
    assert!(stretches.len() <= 2 * layers);
}

#[test]
fn a_qaoa_layer_takes_two_steps() {
    let circuit = crate::circuits::qaoa_circuit(8, 3, 7);
    let plan = plan_for(&circuit, &Parameters::all_rotations(&circuit));
    assert_eq!(plan.last_step, 5);
}

#[test]
fn the_plan_never_inverts_the_earliest_trainable_gate_or_an_idle_one() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Ry(0.4), &[0]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.7), &[1]);
    let mut params = Parameters::new(2);
    params.link(0, 0);
    params.link(3, 1);

    let plan = plan_for(&c, &params);
    let inverted: Vec<usize> = plan.invert.iter().map(|&(_, s)| s).collect();
    assert_eq!(inverted, [3, 2]);
    assert_eq!(plan.last_step, 1);
}

fn assert_adjoint_matches_shift(
    circuit: &Circuit,
    ham: &[(f64, Vec<PauliTerm>)],
    params: &Parameters,
) {
    let adjoint = run_expectation_gradient(circuit, ham, params, 42).unwrap();
    let shift = run_expectation_gradient_shift(circuit, ham, params, 42).unwrap();
    let n = circuit.num_qubits;
    assert!((adjoint.value - shift.value).abs() < 1e-10, "{n}q value");
    for (slot, (got, want)) in adjoint.gradient.iter().zip(&shift.gradient).enumerate() {
        assert!(
            (got - want).abs() < 1e-10,
            "{n}q slot {slot}: {got} vs {want}"
        );
    }
}

fn mixed_hamiltonian(n: usize) -> Vec<(f64, Vec<PauliTerm>)> {
    (0..n - 1)
        .map(|q| {
            (
                0.5 + 0.1 * q as f64,
                vec![PauliTerm::z(q), PauliTerm::z(q + 1)],
            )
        })
        .chain([
            (0.7, vec![PauliTerm::x(1)]),
            (-0.4, vec![PauliTerm::y(0), PauliTerm::x(n - 1)]),
        ])
        .collect()
}

// Widths on both sides of the fusion floors, so the inverted stretches run
// both unfused and through the 1q, 2q, and batch passes.
#[test]
fn frontier_sweep_matches_parameter_shift_on_hea_and_qaoa() {
    for n in [5, 12, 16] {
        let ham = mixed_hamiltonian(n);
        let (hea, shared) = hea_with_shared_slots(n);
        assert_adjoint_matches_shift(&hea, &ham, &shared);
        assert_adjoint_matches_shift(&hea, &ham, &Parameters::all_rotations(&hea));

        let qaoa = crate::circuits::qaoa_circuit(n, 2, 11);
        let mut params = Parameters::new(4);
        for (k, link) in Parameters::all_rotations(&qaoa).links().iter().enumerate() {
            let layer = k / (2 * n - 1);
            let mixer = k % (2 * n - 1) >= n - 1;
            params.link(link.instruction, 2 * layer + usize::from(mixer));
        }
        assert_adjoint_matches_shift(&qaoa, &ham, &params);
    }
}

// Fixed gates of every commutation kind sit between trainable ones, some
// trainable gates share a slot, one rotation stays fixed, and a fixed
// prefix precedes the earliest trainable gate.
#[test]
fn frontier_sweep_matches_parameter_shift_with_fixed_gates_between() {
    use crate::sim::unified_pauli::PauliAxis;
    for n in [4, 12, 16] {
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        for layer in 0..2 {
            for q in 0..n {
                let angle = 0.3 + 0.17 * (q + layer * n) as f64;
                c.add_gate(Gate::Ry(angle), &[q]);
                c.add_gate(if q % 3 == 0 { Gate::T } else { Gate::SX }, &[q]);
                c.add_gate(Gate::Rz(angle * 0.5), &[q]);
            }
            for q in (layer..n - 1).step_by(2) {
                c.add_gate(Gate::Cx, &[q, q + 1]);
                c.add_gate(Gate::Rzz(0.2 + 0.05 * q as f64), &[q, q + 1]);
            }
            c.add_gate(Gate::Cz, &[0, n - 1]);
            c.add_gate(Gate::Swap, &[1, 2]);
            c.add_gate(Gate::P(0.6), &[n - 1]);
            c.add_pauli_rotation(
                0.45,
                &[
                    PauliTerm::new(0, PauliAxis::X),
                    PauliTerm::new(2, PauliAxis::Y),
                    PauliTerm::new(3, PauliAxis::Z),
                ],
            );
            c.add_gate(Gate::Rx(0.9), &[n / 2]);
        }

        let mut params = Parameters::new(5);
        let rotations = c
                .instructions
                .iter()
                .enumerate()
                .filter(|(_, inst)| {
                    matches!(inst, Instruction::Gate { gate, .. } if gate.pauli_generator().is_some())
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
        for (k, &i) in rotations.iter().enumerate() {
            if k % 7 != 3 {
                params.link(i, k % 5);
            }
        }
        assert_adjoint_matches_shift(&c, &mixed_hamiltonian(n), &params);
    }
}

#[test]
fn a_full_light_cone_borrows_the_circuit_it_was_cut_from() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::Rx(0.3), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(0.7), &[1]);

    let all: Vec<usize> = (0..c.instructions.len()).collect();
    assert!(matches!(kept_subcircuit(&c, &all), Cow::Borrowed(_)));

    let pruned = kept_subcircuit(&c, &[0, 2]);
    assert!(matches!(pruned, Cow::Owned(_)));
    assert_eq!(pruned.num_qubits, c.num_qubits);
    assert_eq!(pruned.num_classical_bits, c.num_classical_bits);
    let gates: Vec<&Gate> = pruned
        .instructions
        .iter()
        .map(|inst| {
            let Instruction::Gate { gate, .. } = inst else {
                unreachable!()
            };
            gate
        })
        .collect();
    assert!(matches!(gates.as_slice(), [Gate::Rx(_), Gate::Ry(_)]));
}
