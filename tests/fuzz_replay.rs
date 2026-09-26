//! Stable-toolchain replay of the fuzz targets in `fuzz/`: the checked-in seeds plus a
//! seeded sweep through the same decoders, so the checks run in the ordinary suite.

#[path = "../fuzz/src/lib.rs"]
mod fuzz_checks;

use std::path::Path;

use fuzz_checks::{check_fusion, check_parse, check_parse_bytes, check_parse_tokens};

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
    for seed in seeds("qasm_tokens") {
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
    for seed in seeds("fusion") {
        check_fusion(&seed);
    }
    let mut rng = SplitMix(0xDEAD_BEEF);
    for _ in 0..150 {
        let len = 64 + rng.below(448);
        check_fusion(&rng.bytes(len));
    }
}
