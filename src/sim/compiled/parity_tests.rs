use super::*;
use crate::circuits;
use crate::sim::compiled::compile_measurements;

#[test]
fn block_detection_independent_pairs() {
    let mut c = circuits::independent_bell_pairs(4);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    let sparse = sampler.sparse.as_ref().unwrap();
    let blocks = sparse.find_blocks(sampler.rank);
    assert!(blocks.is_some());
    let blocks = blocks.unwrap();
    assert_eq!(blocks.len(), 4);
    for b in &blocks {
        assert_eq!(b.len(), 2);
    }
    let firsts: Vec<usize> = blocks.iter().map(|b| b[0]).collect();
    assert!(firsts.is_sorted(), "blocks out of row order: {firsts:?}");
}

#[test]
fn block_detection_single_block() {
    let mut c = circuits::clifford_random_pairs(8, 20, 42);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert!(sampler.parity_blocks.is_none());
}

#[test]
fn early_exit_blocks_match_the_column_lists() {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut split = 0;
    for case in 0..300 {
        let num_meas = rng.random_range(2..200usize);
        let rank = rng.random_range(1..150usize);
        let groups = rng.random_range(1..6usize);
        let density = [0.02, 0.1, 0.5][case % 3];
        let cover_every_row = case % 2 == 0;
        let words = num_meas.div_ceil(64);
        let mut flip_rows = vec![vec![0u64; words]; rank];
        for m in 0..num_meas {
            let group = m % groups;
            let cols: Vec<usize> = (0..rank).filter(|c| c % groups == group).collect();
            for &c in &cols {
                if rng.random_bool(density) {
                    flip_rows[c][m / 64] |= 1 << (m % 64);
                }
            }
            if cover_every_row && !cols.is_empty() {
                let c = cols[rng.random_range(0..cols.len())];
                flip_rows[c][m / 64] |= 1 << (m % 64);
            }
        }
        let sparse = SparseParity::from_flip_rows(&flip_rows, num_meas);
        let expected = sparse.find_blocks_by_column_lists(rank);
        assert_eq!(sparse.find_blocks(rank), expected, "case {case}");

        let ref_bits = vec![0u64; words];
        let expected_useful =
            expected.filter(|blocks| blocks.len() >= 2 && blocks.iter().all(|b| b.len() >= 2));
        let actual_useful = build_parity_blocks_if_useful(&sparse, rank, &ref_bits).map(|p| {
            p.blocks
                .into_iter()
                .map(|b| b.meas_indices)
                .collect::<Vec<_>>()
        });
        split += usize::from(actual_useful.is_some());
        assert_eq!(actual_useful, expected_useful, "case {case}");
    }
    assert!(split > 0);
}
