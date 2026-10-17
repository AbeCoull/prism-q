use super::*;
use crate::circuit::builder::CircuitBuilder;

#[test]
fn gate_display_labels() {
    assert_eq!(Gate::H.to_string(), "H");
    assert_eq!(Gate::Cx.to_string(), "CX");
    assert_eq!(Gate::Rx(std::f64::consts::FRAC_PI_2).to_string(), "Rx(π/2)");
    assert_eq!(Gate::Rz(0.5).to_string(), "Rz(0.5000)");
}

#[test]
fn bell_pair_diagram() {
    let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains("q[0]"));
    assert!(text.contains("q[1]"));
    assert!(text.contains("H"));
    assert!(text.contains(CTRL.to_string().as_str()));
    assert!(text.contains("X"));
}

#[test]
fn ghz_3q_diagram() {
    let circuit = CircuitBuilder::new(3).h(0).cx(0, 1).cx(1, 2).build();
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains("q[0]"));
    assert!(text.contains("q[1]"));
    assert!(text.contains("q[2]"));
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines.len() >= 3);
}

fn char_col(s: &str, ch: char) -> Option<usize> {
    s.chars().position(|c| c == ch)
}

#[test]
fn connector_alignment_even_width() {
    let mut builder = CircuitBuilder::new(12);
    for i in (0..12).step_by(2) {
        builder.h(i).cx(i, i + 1);
    }
    let circuit = builder.build();
    let text = circuit.draw(&TextOptions::default());
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if let Some(pos) = char_col(line, VERT) {
            if i > 0
                && let Some(ctrl_pos) = char_col(lines[i - 1], CTRL)
            {
                assert_eq!(
                    ctrl_pos,
                    pos,
                    "connector at line {} misaligned with control at line {}\n{}",
                    i,
                    i - 1,
                    text,
                );
            }
            if i + 1 < lines.len()
                && let Some(x_pos) = char_col(lines[i + 1], 'X')
            {
                assert_eq!(
                    x_pos,
                    pos,
                    "connector at line {} misaligned with target at line {}\n{}",
                    i,
                    i + 1,
                    text,
                );
            }
        }
    }
}

#[test]
fn empty_circuit() {
    let circuit = Circuit::new(2, 0);
    let text = circuit.draw(&TextOptions::default());
    assert_eq!(text, "(empty circuit)");
}

#[test]
fn single_gate() {
    let circuit = CircuitBuilder::new(1).h(0).build();
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains("H"));
    assert!(text.contains("q[0]"));
}

#[test]
fn measurement_shown() {
    let circuit = CircuitBuilder::new(1).h(0).measure_all().build();
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains("M"));
}

#[test]
fn idle_wire_elision() {
    let circuit = CircuitBuilder::new(5).h(0).cx(0, 4).build();
    let opts = TextOptions {
        show_idle_wires: false,
        ..Default::default()
    };
    let text = circuit.draw(&opts);
    assert!(text.contains("q[0]"));
    assert!(text.contains("q[4]"));
    assert!(!text.contains("q[1]"));
    assert!(!text.contains("q[2]"));
    assert!(!text.contains("q[3]"));
}

#[test]
fn summary_mode() {
    let circuit = crate::circuits::random_circuit(100, 10, 42);
    let summary = circuit.summary();
    assert!(summary.contains("Circuit: 100 qubits"));
    assert!(summary.contains("Gate counts:"));
    assert!(summary.contains("Qubit activity"));
}

#[test]
fn auto_summary_for_large() {
    let circuit = crate::circuits::random_circuit(100, 10, 42);
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains("Circuit: 100 qubits"));
}

#[test]
fn display_impl() {
    let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
    let text = format!("{}", circuit);
    assert!(text.contains("q[0]"));
    assert!(text.contains("q[1]"));
}

#[test]
fn fold_wraps_long_circuits() {
    let circuit = crate::circuits::random_circuit(3, 50, 42);
    let opts = TextOptions {
        fold_width: 60,
        ..Default::default()
    };
    let text = circuit.draw(&opts);
    assert!(text.contains("\u{254C}"));
}

#[test]
fn max_moments_truncates_and_says_so() {
    let circuit = CircuitBuilder::new(1).h(0).x(0).h(0).x(0).build();
    let opts = TextOptions {
        max_moments: Some(2),
        ..Default::default()
    };
    let text = circuit.draw(&opts);
    assert!(text.contains("truncated at moment 2 of 4"), "{text}");
    assert!(!circuit.draw(&TextOptions::default()).contains("truncated"));
}

#[test]
fn swap_gate_display() {
    let circuit = CircuitBuilder::new(2).swap(0, 1).build();
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains(SWAP_X.to_string().as_str()));
}

#[test]
fn parametric_gate_display() {
    let circuit = CircuitBuilder::new(1)
        .rx(std::f64::consts::FRAC_PI_4, 0)
        .build();
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains("Rx(π/4)"));
}

#[test]
fn qft_4q_diagram() {
    // `qft_circuit` emits one `Gate::QftBlock`; `expand_qft_blocks` unrolls it.
    let circuit = crate::circuits::qft_circuit(4);
    let text = circuit.draw(&TextOptions::default());
    assert!(text.contains("q[0]"));
    assert!(text.contains("q[3]"));

    let unrolled = crate::circuit::expand_qft_blocks(&circuit);
    let unrolled_text = unrolled.draw(&TextOptions::default());
    assert!(unrolled_text.contains("H"));
}

#[test]
fn summary_has_connectivity() {
    let circuit = crate::circuits::random_circuit(100, 10, 42);
    let summary = circuit.summary();
    assert!(summary.contains("Connectivity:"));
    assert!(summary.contains("max degree:"));
}

#[test]
fn summary_no_2q_connectivity() {
    let circuit = CircuitBuilder::new(4).h(0).h(1).h(2).h(3).build();
    let summary = circuit.summary();
    assert!(summary.contains("Connectivity: none"));
}

#[test]
fn summary_has_depth_profile() {
    let circuit = crate::circuits::random_circuit(100, 10, 42);
    let summary = circuit.summary();
    assert!(summary.contains("Depth profile"));
}

#[test]
fn summary_has_heatmap() {
    let circuit = crate::circuits::random_circuit(100, 10, 42);
    let summary = circuit.summary();
    assert!(summary.contains("Gate density heatmap"));
    assert!(summary.contains("Legend:"));
}

#[test]
fn heatmap_standalone() {
    let circuit = crate::circuits::ghz_circuit(10);
    let hm = circuit.heatmap(&TextOptions::default());
    assert!(hm.contains("Gate density heatmap"));
    assert!(hm.contains("q[0]"));
}

#[test]
fn connectivity_nearest_neighbor() {
    let circuit = crate::circuits::ghz_circuit(8);
    let summary = circuit.summary();
    assert!(summary.contains("nearest-neighbor"));
}

#[test]
fn multibyte_labels_pad_by_char_count() {
    let rotation = CircuitBuilder::new(2)
        .ry(std::f64::consts::FRAC_PI_2, 0)
        .h(1)
        .build();
    assert_eq!(
        rotation.draw(&TextOptions::default()),
        ["q[0]: ─Ry(π/2)─", "q[1]: ────H────"].join("\n"),
    );

    let mut builder = CircuitBuilder::new(2);
    builder.h(0).h(1).reset(0).x(1);
    assert_eq!(
        builder.build().draw(&TextOptions::default()),
        ["q[0]: ─H──|0⟩─", "q[1]: ─H───X──"].join("\n"),
    );
}

#[test]
fn swap_connector_aligns_with_markers() {
    let circuit = CircuitBuilder::new(3).swap(0, 2).h(0).h(2).build();
    assert_eq!(
        circuit.draw(&TextOptions::default()),
        [
            "q[0]: ──×────H─",
            "        │",
            "q[1]: ──│──────",
            "        │",
            "q[2]: ──×────H─",
        ]
        .join("\n"),
    );

    let odd_width = CircuitBuilder::new(3)
        .swap(0, 1)
        .rx(std::f64::consts::FRAC_PI_4, 2)
        .build();
    assert_eq!(
        odd_width.draw(&TextOptions::default()),
        [
            "q[0]: ────×────",
            "          │",
            "q[1]: ────×────",
            "q[2]: ─Rx(π/4)─",
        ]
        .join("\n"),
    );
}

#[test]
fn barrier_gets_its_own_column() {
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).h(1).barrier(&[0, 1]).x(0).x(1);
    let circuit = builder.build();
    assert_eq!(
        circuit.draw(&TextOptions::default()),
        ["q[0]: ─H─┊┊┊─X─", "q[1]: ─H─┊┊┊─X─"].join("\n"),
    );

    let hidden = TextOptions {
        show_barriers: false,
        ..Default::default()
    };
    assert_eq!(
        circuit.draw(&hidden),
        ["q[0]: ─H──X─", "q[1]: ─H──X─"].join("\n"),
    );
}

#[test]
fn partial_barrier_precedes_later_gates() {
    let mut builder = CircuitBuilder::new(3);
    builder.h(0).h(2).barrier(&[0, 1]).h(1);
    assert_eq!(
        builder.build().draw(&TextOptions::default()),
        ["q[0]: ─H─┊┊┊───", "q[1]: ───┊┊┊─H─", "q[2]: ─H───────"].join("\n"),
    );
}

#[test]
fn measurement_column_fits_its_classical_bit() {
    let circuit = CircuitBuilder::new_with_classical(1, 101)
        .h(0)
        .measure(0, 100)
        .x(0)
        .build();
    assert_eq!(circuit.draw(&TextOptions::default()), "q[0]: ─H──M100──X─");
}

#[test]
fn conditional_column_fits_its_condition() {
    let circuit = CircuitBuilder::new_with_classical(2, 1)
        .h(0)
        .measure(0, 0)
        .x(1)
        .x(1)
        .conditional(ClassicalCondition::BitIsOne(0), Gate::X, &[1])
        .build();
    assert_eq!(
        circuit.draw(&TextOptions::default()),
        ["q[0]: ─H──M0─────────", "q[1]: ─X──X───c[0]?X─"].join("\n"),
    );
}
