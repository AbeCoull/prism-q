use super::*;

/// Width of the register `resolver` stands for.
const REGISTER: usize = 2;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// `q[k]` resolves to k, a bare `q` to the whole two-qubit register.
fn resolver() -> impl Fn(&str) -> Result<Vec<usize>> {
    |token: &str| match token {
        "q" => Ok(vec![0, 1]),
        _ => token
            .trim_start_matches("q[")
            .trim_start_matches('$')
            .trim_end_matches(']')
            .parse::<usize>()
            .map(|index| vec![index])
            .map_err(|_| PrismError::UndefinedRegister {
                name: token.to_string(),
                line: 1,
            }),
    }
}

fn result_of(body: &str) -> ResultSpec {
    parse_result_pragma(body, 1, REGISTER, &resolver()).unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

fn noise_of(body: &str) -> NoiseSpec {
    parse_noise_pragma(body, 1, &resolver()).unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

#[test]
fn complex_literal_forms() {
    let cases = [
        ("0", c(0.0, 0.0)),
        ("-1", c(-1.0, 0.0)),
        ("1.5", c(1.5, 0.0)),
        ("1im", c(0.0, 1.0)),
        ("-1im", c(0.0, -1.0)),
        ("2.5im", c(0.0, 2.5)),
        ("0.7 + 0.7im", c(0.7, 0.7)),
        ("1 - 2im", c(1.0, -2.0)),
        ("-1 - 2im", c(-1.0, -2.0)),
        // A sign inside an exponent is not a term separator.
        ("1e-3", c(0.001, 0.0)),
        ("1e-3 + 2e-3im", c(0.001, 0.002)),
    ];
    for (text, expected) in cases {
        let parsed = parse_complex(text, 1).unwrap_or_else(|e| panic!("`{text}`: {e}"));
        assert!(
            (parsed - expected).norm() < 1e-12,
            "`{text}` parsed as {parsed}, expected {expected}"
        );
    }
}

#[test]
fn complex_literal_rejections() {
    for text in ["", "abc", "1 + 2", "1im + 2im"] {
        assert!(parse_complex(text, 1).is_err(), "`{text}` should not parse");
    }
}

#[test]
fn matrix_literal_parses_pauli_y() {
    let parsed = parse_matrix("[[0, -1im], [1im, 0]]", 1).unwrap();
    assert_eq!(
        parsed,
        vec![
            vec![c(0.0, 0.0), c(0.0, -1.0)],
            vec![c(0.0, 1.0), c(0.0, 0.0)]
        ]
    );
}

#[test]
fn matrix_literal_rejects_non_square_and_odd_sides() {
    for text in [
        "[[1, 0]]",
        "[[1, 0], [0, 1], [0, 0]]",
        "[[1, 0, 0], [0, 1, 0]]",
    ] {
        assert!(parse_matrix(text, 1).is_err(), "`{text}` should not parse");
    }
}

#[test]
fn result_pragmas_without_observables() {
    assert_eq!(result_of("state_vector"), ResultSpec::StateVector);
    assert_eq!(
        result_of("density_matrix all"),
        ResultSpec::DensityMatrix(Targets::All)
    );
    // An omitted target list means the same as `all`.
    assert_eq!(
        result_of("density_matrix"),
        ResultSpec::DensityMatrix(Targets::All)
    );
    assert_eq!(
        result_of("probability q[0], q[1]"),
        ResultSpec::Probability(Targets::These(vec![0, 1]))
    );
    assert_eq!(
        result_of("amplitude \"01\", \"10\""),
        ResultSpec::Amplitude(vec!["01".to_string(), "10".to_string()])
    );
}

#[test]
fn observable_result_pragmas() {
    assert_eq!(
        result_of("expectation x(q[0])"),
        ResultSpec::Expectation(Observable {
            factors: vec![ObservableFactor::Pauli {
                axis: PauliAxis::X,
                targets: Targets::These(vec![0])
            }]
        })
    );
    assert_eq!(
        result_of("sample h(q[1])"),
        ResultSpec::Sample(Observable {
            factors: vec![ObservableFactor::Hadamard {
                targets: Targets::These(vec![1])
            }]
        })
    );
    assert_eq!(
        result_of("expectation i all"),
        ResultSpec::Expectation(Observable {
            factors: vec![ObservableFactor::Identity {
                targets: Targets::All
            }]
        })
    );
}

#[test]
fn tensor_product_observable_keeps_factor_order() {
    let ResultSpec::Expectation(observable) = result_of("expectation x(q[0]) @ z(q[1])") else {
        panic!("expected an expectation");
    };
    assert_eq!(
        observable.factors,
        vec![
            ObservableFactor::Pauli {
                axis: PauliAxis::X,
                targets: Targets::These(vec![0])
            },
            ObservableFactor::Pauli {
                axis: PauliAxis::Z,
                targets: Targets::These(vec![1])
            },
        ]
    );
}

#[test]
fn hermitian_observable_carries_its_matrix() {
    let ResultSpec::Variance(observable) =
        result_of("variance hermitian([[0, -1im], [1im, 0]]) $0")
    else {
        panic!("expected a variance");
    };
    assert_eq!(
        observable.factors,
        vec![ObservableFactor::Hermitian {
            matrix: vec![
                vec![c(0.0, 0.0), c(0.0, -1.0)],
                vec![c(0.0, 1.0), c(0.0, 0.0)]
            ],
            targets: Targets::These(vec![0]),
        }]
    );
}

// A 2x2 matrix names one qubit; naming two would silently measure the wrong
// operator, so the arity is checked against the matrix side.
#[test]
fn hermitian_observable_arity_must_match_its_matrix() {
    let err = parse_result_pragma(
        "variance hermitian([[0, -1im], [1im, 0]]) q[0], q[1]",
        1,
        REGISTER,
        &resolver(),
    )
    .expect_err("a 2x2 matrix on two qubits");
    let text = format!("{err}");
    assert!(
        matches!(err, PrismError::Parse { line: 1, .. })
            && text.contains("needs a 4x4 matrix, got 2x2"),
        "got {text}"
    );
}

#[test]
fn shots_mode_split_matches_braket() {
    for body in ["state_vector", "density_matrix all", "amplitude \"0\""] {
        assert!(result_of(body).requires_exact(), "{body}");
    }
    for body in ["probability all", "expectation z(q[0])", "sample z(q[0])"] {
        assert!(!result_of(body).requires_exact(), "{body}");
    }
}

#[test]
fn unsupported_result_types_are_named() {
    for body in ["adjoint_gradient expectation(z(q[0])) all", "nonsense q[0]"] {
        assert!(matches!(
            parse_result_pragma(body, 1, REGISTER, &resolver()),
            Err(PrismError::UnsupportedConstruct { .. })
        ));
    }
}

#[test]
fn single_qubit_noise_pragmas() {
    assert!(matches!(
        noise_of("bit_flip(0.1) q[0]").channel,
        NoiseChannel::Pauli { px, py, pz } if px == 0.1 && py == 0.0 && pz == 0.0
    ));
    assert!(matches!(
        noise_of("phase_flip(0.1) q[0]").channel,
        NoiseChannel::Pauli { px, pz, .. } if px == 0.0 && pz == 0.1
    ));
    assert!(matches!(
        noise_of("pauli_channel(0.1, 0.2, 0.3) q[0]").channel,
        NoiseChannel::Pauli { px, py, pz } if px == 0.1 && py == 0.2 && pz == 0.3
    ));
    assert!(matches!(
        noise_of("depolarizing(0.3) q[0]").channel,
        NoiseChannel::Depolarizing { p } if p == 0.3
    ));
    assert!(matches!(
        noise_of("amplitude_damping(0.2) q[0]").channel,
        NoiseChannel::AmplitudeDamping { gamma } if gamma == 0.2
    ));
    assert!(matches!(
        noise_of("phase_damping(0.2) q[0]").channel,
        NoiseChannel::PhaseDamping { gamma } if gamma == 0.2
    ));
    assert_eq!(noise_of("bit_flip(0.1) q[1]").qubits, vec![1]);
}

#[test]
fn two_qubit_noise_pragmas() {
    let spec = noise_of("two_qubit_depolarizing(0.2) q[0], q[1]");
    assert_eq!(spec.qubits, vec![0, 1]);
    assert!(matches!(
        spec.channel,
        NoiseChannel::TwoQubitDepolarizing { p } if p == 0.2
    ));
    let spec = noise_of("two_qubit_dephasing(0.2) q[0], q[1]");
    assert_eq!(spec.qubits, vec![0, 1]);
    assert!(matches!(spec.channel, NoiseChannel::Kraus2q { .. }));
}

// The two channels with no named variant go in as explicit operator sets, so
// the CPTP check is what says they were built correctly.
#[test]
fn lowered_channels_are_cptp() {
    for body in [
        "generalized_amplitude_damping(0.3, 0.7) q[0]",
        "two_qubit_dephasing(0.4) q[0], q[1]",
    ] {
        noise_of(body).channel.validate().expect(body);
    }
}

#[test]
fn kraus_pragma_takes_one_and_two_qubit_operator_sets() {
    // sqrt(0.9) and sqrt(0.1): a bit-flip channel at p = 0.1 written out.
    let spec = noise_of(
        "kraus([[0.9486832980505138, 0], [0, 0.9486832980505138]],          [[0, 0.31622776601683794], [0.31622776601683794, 0]]) q[0]",
    );
    assert!(matches!(spec.channel, NoiseChannel::Custom { ref kraus } if kraus.len() == 2));

    let identity_4 = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]";
    let spec = noise_of(&format!("kraus({identity_4}) q[0], q[1]"));
    assert!(matches!(spec.channel, NoiseChannel::Kraus2q { ref kraus } if kraus.len() == 1));
}

#[test]
fn noise_probability_ranges_are_enforced() {
    // Braket's own bounds: bit flip stops at a half, depolarizing at 3/4.
    for body in [
        "bit_flip(0.6) q[0]",
        "phase_flip(0.6) q[0]",
        "depolarizing(0.8) q[0]",
        "two_qubit_depolarizing(0.99) q[0], q[1]",
        "two_qubit_dephasing(0.8) q[0], q[1]",
    ] {
        assert!(
            parse_noise_pragma(body, 1, &resolver()).is_err(),
            "`{body}` should be out of range"
        );
    }
}

#[test]
fn noise_arity_is_enforced() {
    for body in [
        "bit_flip(0.1) q[0], q[1]",
        "two_qubit_depolarizing(0.1) q[0]",
        "bit_flip(0.1, 0.2) q[0]",
    ] {
        assert!(
            parse_noise_pragma(body, 1, &resolver()).is_err(),
            "`{body}` should not parse"
        );
    }
}

#[test]
fn a_non_cptp_kraus_set_is_rejected() {
    assert!(
        parse_noise_pragma(
            "kraus([[1, 0], [0, 1]], [[1, 0], [0, 1]]) q[0]",
            1,
            &resolver()
        )
        .is_err()
    );
}

// The rotation and the eigenvalues have to stay paired by column: outcome `j`
// reads eigenvalue `j`. Checked by rebuilding the observable from the pair.
#[test]
fn a_hermitian_measurement_pairs_its_eigenvalues_with_its_rotation() {
    let matrices: Vec<Vec<Vec<Complex64>>> = vec![
        vec![
            vec![c(0.0, 0.0), c(1.0, 0.0)],
            vec![c(1.0, 0.0), c(0.0, 0.0)],
        ],
        vec![
            vec![c(1.0, 0.0), c(2.0, -1.0)],
            vec![c(2.0, 1.0), c(-3.0, 0.0)],
        ],
        vec![
            vec![c(1.0, 0.0), c(0.0, 0.5), c(0.0, 0.0), c(0.2, 0.0)],
            vec![c(0.0, -0.5), c(-2.0, 0.0), c(0.3, 0.1), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(0.3, -0.1), c(0.0, 0.0), c(0.7, 0.0)],
            vec![c(0.2, 0.0), c(0.0, 0.0), c(0.7, 0.0), c(3.0, 0.0)],
        ],
    ];
    for matrix in &matrices {
        let dim = matrix.len();
        let width = dim.trailing_zeros() as usize;
        let targets: Vec<usize> = (0..width).collect();
        let (values, rotation) =
            hermitian_measurement(matrix, &targets).expect("inside the reduction span");
        assert_eq!(values.len(), dim);
        assert!(!rotation.is_empty());

        // `U*` applied to eigenvector `j` gives `e_j`, so rebuilding
        // `sum_j values[j] |v_j><v_j|` has to return the matrix. The
        // eigenvectors are the rows of `U*` conjugated.
        let dagger = realized_rotation(&rotation, width, &targets);
        for row in 0..dim {
            for column in 0..dim {
                let rebuilt: Complex64 = (0..dim)
                    .map(|j| dagger[j * dim + row].conj() * values[j] * dagger[j * dim + column])
                    .sum();
                assert!(
                    (rebuilt - matrix[row][column]).norm() < 1e-9,
                    "({row}, {column}): {rebuilt} against {}",
                    matrix[row][column]
                );
            }
        }
    }
}

/// The unitary a rotation implements, row major with `targets[0]` the high bit.
fn realized_rotation(instrs: &[Instruction], width: usize, targets: &[usize]) -> Vec<Complex64> {
    let dim = 1usize << width;
    let place = |state: usize| -> usize {
        targets.iter().enumerate().fold(0usize, |index, (at, &q)| {
            index | (state >> (width - 1 - at) & 1) << q
        })
    };
    let circuit = crate::circuit::Circuit {
        num_qubits: width,
        num_classical_bits: 0,
        instructions: instrs.to_vec(),
    };
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for column in 0..dim {
        let mut start = vec![Complex64::new(0.0, 0.0); dim];
        start[place(column)] = Complex64::new(1.0, 0.0);
        let state = crate::simulate(&circuit)
            .seed(42)
            .initial_state(&start)
            .state_vector()
            .expect("a unitary rotation");
        for row in 0..dim {
            out[row * dim + column] = state[place(row)];
        }
    }
    out
}

// An observable wider than the reduction covers has no rotation to carry it,
// and says so rather than dropping the request.
#[test]
fn a_hermitian_observable_past_the_reduction_span_declines() {
    let dim = 32usize;
    let matrix: Vec<Vec<Complex64>> = (0..dim)
        .map(|row| {
            (0..dim)
                .map(|column| c(f64::from(u8::from(row == column)) * row as f64, 0.0))
                .collect()
        })
        .collect();
    let targets: Vec<usize> = (0..5).collect();
    let err = hermitian_measurement(&matrix, &targets).expect_err("five qubits is too wide");
    assert!(
        format!("{err}").contains("the reduction covers"),
        "got {err}"
    );
}
