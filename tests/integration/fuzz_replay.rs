//! Stable-toolchain replay of the fuzz targets in `fuzz/`: the checked-in seeds plus a
//! seeded sweep through the same decoders, so the checks run in the ordinary suite.

#[path = "../../fuzz/src/lib.rs"]
mod fuzz_checks;

use std::path::Path;

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use fuzz_checks::{
    SimulateCase, check_fusion, check_parse, check_parse_bytes, check_parse_tokens, check_simulate,
    check_simulate_case, simulate_case_from_bytes,
};
use prism_q::{Gate, Instruction};

fn seeds(target: &str) -> Vec<Vec<u8>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/seeds")
        .join(target);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut inputs: Vec<Vec<u8>> = entries
        .map(|entry| std::fs::read(entry.unwrap().path()).unwrap())
        .collect();
    inputs.sort();
    inputs
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

#[test]
fn qasm_seeds_and_their_truncations_parse_without_panic() {
    let corpus = seeds("qasm_parse");
    assert!(
        corpus.len() > 100,
        "seed corpus holds {} files",
        corpus.len()
    );
    for seed in &corpus {
        check_parse_bytes(seed);
        for cut in (0..seed.len()).step_by(5) {
            check_parse_bytes(&seed[..cut]);
        }
    }
}

#[test]
fn spliced_qasm_seeds_parse_without_panic() {
    let corpus = seeds("qasm_parse");
    assert!(!corpus.is_empty(), "qasm_parse seed corpus is empty");
    let mut rng = SplitMix(42);
    for _ in 0..3000 {
        let text = String::from_utf8_lossy(&corpus[rng.below(corpus.len())]).into_owned();
        let donor = String::from_utf8_lossy(&corpus[rng.below(corpus.len())]).into_owned();
        let at = floor_char_boundary(&text, rng.below(text.len() + 1));
        let from = floor_char_boundary(&donor, rng.below(donor.len() + 1));
        let to = floor_char_boundary(&donor, from + rng.below(donor.len() - from + 1));
        let mut spliced = String::with_capacity(text.len() + to - from);
        spliced.push_str(&text[..at]);
        spliced.push_str(&donor[from..to]);
        spliced.push_str(&text[at..]);
        check_parse(&spliced);
    }
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[test]
fn seeded_token_programs_parse_without_panic() {
    let corpus = seeds("qasm_tokens");
    assert!(!corpus.is_empty(), "qasm_tokens seed corpus is empty");
    for seed in corpus {
        check_parse_tokens(&seed);
    }
    let mut rng = SplitMix(42);
    for _ in 0..3000 {
        let len = rng.below(96);
        check_parse_tokens(&rng.bytes(len));
    }
}

#[test]
fn seeded_fusion_circuits_match_unfused() {
    let corpus = seeds("fusion");
    assert!(!corpus.is_empty(), "fusion seed corpus is empty");
    for seed in corpus {
        check_fusion(&seed);
    }
    let mut rng = SplitMix(42);
    for _ in 0..150 {
        let len = 64 + rng.below(448);
        check_fusion(&rng.bytes(len));
    }
}

// Open defects the simulate loop runs into. A case inside one of these shapes may fail;
// the seeds in `fuzz/seeds/simulate_known_defects` pin one failure each. When a fix lands,
// move its seed to `fuzz/seeds/simulate` and narrow or delete the shape here.
fn inside_known_defect(case: &SimulateCase) -> bool {
    let wide_gate = |gate: &Gate| matches!(gate, Gate::QftBlock { .. } | Gate::PauliRot(_));
    let has = |pred: &dyn Fn(&Instruction) -> bool| case.base.instructions.iter().any(pred);

    // A run of eight or more qubits that splits into independent blocks drops its
    // save points.
    let splits = case.padded.num_qubits >= 8 && case.padded.num_qubits > case.base.num_qubits;
    let block_sensitive = has(&|instruction| matches!(instruction, Instruction::Save { .. }));
    // `Auto` above the statevector cap probes sparse support with a gate that has no
    // 2x2 matrix.
    let wide_over_cap = case.statevector_cap.is_some()
        && has(
            &|instruction| matches!(instruction, Instruction::Gate { gate, .. } if wide_gate(gate)),
        );
    (splits && block_sensitive) || wide_over_cap
}

#[test]
fn seeded_simulate_cases_match_the_reference() {
    let corpus = seeds("simulate");
    assert!(!corpus.is_empty(), "simulate seed corpus is empty");
    for seed in corpus {
        check_simulate(&seed);
    }
    let mut rng = SplitMix(42);
    let mut tolerated = 0;
    for _ in 0..2000 {
        let len = 8 + rng.below(120);
        let case = simulate_case_from_bytes(&rng.bytes(len));
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| check_simulate_case(&case))) {
            if !inside_known_defect(&case) {
                resume_unwind(panic);
            }
            tolerated += 1;
        }
    }
    assert!(
        tolerated > 0,
        "no case failed inside a known defect shape; delete `inside_known_defect`"
    );
}

// Default build only: the lowered floors send some of these circuits down other routes,
// where the same input passes.
#[cfg(not(prism_lowered_thresholds))]
#[test]
fn known_defect_seeds_still_fail() {
    let corpus = seeds("simulate_known_defects");
    assert_eq!(corpus.len(), 2, "one seed per open defect");
    for seed in corpus {
        let case = simulate_case_from_bytes(&seed);
        assert!(
            inside_known_defect(&case),
            "seed lies outside every known defect shape\n{case:#?}"
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| check_simulate_case(&case))).is_err(),
            "a known defect no longer reproduces; move its seed to fuzz/seeds/simulate\n{case:#?}"
        );
    }
}
