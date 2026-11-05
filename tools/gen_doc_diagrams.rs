//! Generates the circuit SVGs embedded in the mdBook docs.
//!
//! The output files in `docs/diagrams/` are committed to the repository so the docs CI
//! job stays a pure `mdbook build` with no Rust toolchain. Regenerate after changing the
//! SVG renderer or the circuits below, then commit the result:
//!
//! ```bash
//! cargo run --example gen_doc_diagrams
//! ```
//!
//! `auto_theme` emits self-contained SVGs that adapt to the reader's color scheme.

use prism_q::circuit::expand_qft_blocks;
use prism_q::circuits::{
    ghz_circuit, hardware_efficient_ansatz, phase_estimation_circuit, qaoa_circuit, qft_circuit,
    w_state_circuit,
};
use prism_q::{CircuitBuilder, Instruction, SvgOptions};
use std::fs;
use std::path::Path;

const SEED: u64 = 0xDEAD_BEEF;

fn doc_opts() -> SvgOptions {
    SvgOptions {
        auto_theme: true,
        animate: false,
        ..SvgOptions::default()
    }
}

fn main() {
    let dir = Path::new("docs/diagrams");
    fs::create_dir_all(dir).expect("create docs/diagrams");
    let opts = doc_opts();
    let bell = CircuitBuilder::new(2).h(0).cx(0, 1).build();
    let measured = CircuitBuilder::new_with_classical(2, 2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .build();
    let mut ansatz = hardware_efficient_ansatz(4, 2, SEED);
    let layer_boundary = ansatz.instructions.len() / 2;
    ansatz.instructions.insert(
        layer_boundary,
        Instruction::Barrier {
            qubits: (0..ansatz.num_qubits).collect(),
        },
    );

    let diagrams = [
        ("bell_2", bell.to_svg(&opts)),
        ("bell_2_measured", measured.to_svg(&opts)),
        ("ghz_5", ghz_circuit(5).to_svg(&opts)),
        ("qft_4", expand_qft_blocks(&qft_circuit(4)).to_svg(&opts)),
        ("w_state_4", w_state_circuit(4).to_svg(&opts)),
        ("qaoa_4", qaoa_circuit(4, 1, SEED).to_svg(&opts)),
        (
            "hea_4",
            ansatz.to_svg(&SvgOptions {
                show_barriers: false,
                ..opts
            }),
        ),
        (
            "qpe_4",
            expand_qft_blocks(&phase_estimation_circuit(4)).to_svg(&opts),
        ),
        (
            "hea_12_heatmap",
            hardware_efficient_ansatz(12, 4, SEED).to_svg_heatmap(&opts),
        ),
    ];

    for (name, svg) in diagrams {
        let path = dir.join(format!("{name}.svg"));
        fs::write(&path, svg).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        println!("wrote {}", path.display());
    }
}
