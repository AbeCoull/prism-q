//! Stable-toolchain replay of the fuzz targets in `fuzz/`: the checked-in seeds plus a
//! seeded sweep through the same decoders, so the checks run in the ordinary suite.

#[path = "../../fuzz/src/lib.rs"]
mod fuzz_checks;

use std::path::Path;

use fuzz_checks::{
    FuzzQuery, check_fusion, check_parse, check_parse_bytes, check_parse_tokens, check_simulate,
    check_simulate_case, simulate_case_from_bytes,
};
use prism_q::simulate;

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

#[test]
fn seeded_simulate_cases_match_the_reference() {
    let corpus = seeds("simulate");
    assert!(corpus.len() >= 27, "expected promoted simulate seeds");
    for seed in corpus {
        check_simulate(&seed);
    }
    let mut rng = SplitMix(42);
    for _ in 0..1800 {
        let len = 12 + rng.below(120);
        let case = simulate_case_from_bytes(&rng.bytes(len));
        check_simulate_case(&case);
    }
}

#[test]
fn bounded_tn_shot_seed_reports_nontrivial_fidelity_bound() {
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/simulate/bounded_tn_shot_bound"),
    )
    .expect("bounded_tn_shot_bound seed");
    let case = simulate_case_from_bytes(&bytes);
    let shots = match case.query {
        FuzzQuery::Shots(s) => s,
        other => panic!("expected Shots query, got {other:?}"),
    };
    let result = simulate(&case.base)
        .backend(case.kind)
        .limits(case.limits)
        .seed(42)
        .shots(shots)
        .expect("bounded TN shots");
    let bound = result
        .metadata
        .fidelity_lower_bound()
        .expect("bounded TN reports fidelity lower bound");
    assert!(
        bound < 1.0,
        "expected nontrivial truncation bound < 1.0, got {bound}"
    );
}

#[cfg(feature = "distributed")]
#[test]
fn distributed_serial_simulate_cases_match_the_reference() {
    use prism_q::{BackendKind, DistributedContext};

    let mut rng = SplitMix(0x5EED_D157);
    for _ in 0..80 {
        let len = 16 + rng.below(100);
        let mut bytes = rng.bytes(len);
        // Force Stage 2 header without batch check and select index 15 (StatevectorDistributed).
        bytes[0] = (bytes[0] & !0x20) | 0x40;
        if bytes.len() > 1 {
            bytes[1] = 15;
        }
        let mut case = simulate_case_from_bytes(&bytes);
        case.kind = BackendKind::StatevectorDistributed {
            context: DistributedContext::serial(),
        };
        check_simulate_case(&case);
    }
}
