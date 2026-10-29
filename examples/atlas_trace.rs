use std::fmt::Write as _;
use std::io::Write as _;
use std::process::Command;

use num_complex::Complex64;
use prism_q::StatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::{Instruction, SmallVec};
use prism_q::gates::Gate;

fn append_state(output: &mut String, state: &[Complex64]) {
    output.push('[');
    for (i, amplitude) in state.iter().enumerate() {
        if i != 0 {
            output.push(',');
        }
        write!(output, "{:.17e},{:.17e}", amplitude.re, amplitude.im).unwrap();
    }
    output.push(']');
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let revision = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let revision = String::from_utf8(revision.stdout)?;
    let revision = revision.trim();
    if revision.len() != 40 || !revision.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("export requires a Git source revision".into());
    }
    let theta = std::f64::consts::PI / 3.0;
    let gates = [
        ("H", Gate::H),
        ("X", Gate::X),
        ("Y", Gate::Y),
        ("Z", Gate::Z),
        ("Rx", Gate::Rx(theta)),
        ("Ry", Gate::Ry(theta)),
        ("Rz", Gate::Rz(theta)),
    ];
    let mut output = format!(
        "{{\"schemaVersion\":1,\"provenance\":{{\"sourceRevision\":\"{revision}\",\"crateVersion\":\"{}\",\"backend\":\"StatevectorBackend\",\"execution\":\"native\",\"precision\":\"complex128\",\"seed\":42,\"features\":[\"parallel\"],\"exporter\":\"examples/atlas_trace.rs\",\"instruction\":\"Gate::Fused\",\"timing\":null}},\"traces\":[",
        env!("CARGO_PKG_VERSION")
    );
    let mut first = true;
    for n in 2..=6 {
        let mut state: Vec<_> = (0..1usize << n)
            .map(|j| Complex64::new((j + 1) as f64, ((j * 7) % 11) as f64 - 5.0))
            .collect();
        let scale = state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        state.iter_mut().for_each(|a| *a /= scale);
        for target in 0..n {
            for (name, gate) in &gates {
                if !first {
                    output.push(',');
                }
                first = false;
                write!(
                    output,
                    "{{\"id\":\"{n}-{target}-{name}\",\"n\":{n},\"target\":{target},\"gate\":\"{name}\",\"theta\":{theta:.17e},\"before\":"
                )?;
                append_state(&mut output, &state);
                output.push_str(",\"steps\":[");
                let mut backend = StatevectorBackend::new(42);
                backend.init_from_amplitudes(state.clone(), 0)?;
                let instruction = Instruction::Gate {
                    gate: Gate::Fused(Box::new(gate.matrix_2x2())),
                    targets: SmallVec::from_slice(&[target]),
                };
                for step in 0..4 {
                    if step != 0 {
                        output.push(',');
                    }
                    backend.apply(&instruction)?;
                    append_state(&mut output, backend.state_vector());
                }
                output.push_str("]}");
            }
        }
    }
    output.push_str("],\"wideTraces\":[");
    let n = 16;
    let mut state: Vec<_> = (0..1usize << n)
        .map(|j| Complex64::new((j + 1) as f64, ((j * 7) % 11) as f64 - 5.0))
        .collect();
    let scale = state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    state.iter_mut().for_each(|a| *a /= scale);
    let mut first = true;
    for (name, gate) in [("H", Gate::H), ("Z", Gate::Z), ("Rz", Gate::Rz(theta))] {
        for target in [0, 2, 13, 15] {
            let matrix = gate.matrix_2x2();
            let mut reference = state.clone();
            let half = 1usize << target;
            for i0 in (0..reference.len()).filter(|i| i & half == 0) {
                let (a, b) = (reference[i0], reference[i0 | half]);
                reference[i0] = matrix[0][0] * a + matrix[0][1] * b;
                reference[i0 | half] = matrix[1][0] * a + matrix[1][1] * b;
            }
            let mut backend = StatevectorBackend::new(42);
            backend.init_from_amplitudes(state.clone(), 0)?;
            backend.apply(&Instruction::Gate {
                gate: Gate::Fused(Box::new(matrix)),
                targets: SmallVec::from_slice(&[target]),
            })?;
            let after = backend.state_vector();
            let max_error = after
                .iter()
                .zip(&reference)
                .map(|(x, y)| (x - y).norm())
                .fold(0.0, f64::max);
            let norm = after.iter().map(Complex64::norm_sqr).sum::<f64>();
            if !first {
                output.push(',');
            }
            first = false;
            write!(
                output,
                "{{\"id\":\"{n}-{target}-{name}\",\"n\":{n},\"target\":{target},\"gate\":\"{name}\",\"theta\":{theta:.17e},\"normSquared\":{norm:.17e},\"maxAbsoluteError\":{max_error:.17e},\"samples\":["
            )?;
            let last = after.len() - 1;
            let indices = [
                0,
                1,
                2,
                3,
                half - 1,
                half,
                half + 1,
                last / 2,
                last - half,
                last,
            ];
            for (k, &i) in indices.iter().enumerate() {
                if k != 0 {
                    output.push(',');
                }
                write!(
                    output,
                    "{{\"index\":{i},\"re\":{:.17e},\"im\":{:.17e}}}",
                    after[i].re, after[i].im
                )?;
            }
            output.push_str("]}");
        }
    }
    output.push_str("]}\n");
    std::io::stdout().lock().write_all(output.as_bytes())?;
    Ok(())
}
