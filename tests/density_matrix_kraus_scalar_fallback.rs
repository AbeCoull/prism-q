//! With `PRISM_NO_AVX2_KRAUS` set, the dense two-qubit Kraus sweep takes the
//! blocked scalar path on hosts whose detected AVX2 would otherwise route it
//! to the wide kernel. Own file: the switch is read once per process.

mod common;

use common::{DM_EPS, SEED, all_pauli_masks, depolarizing_2q_kraus};
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::{circuits, sim};

#[test]
fn dense_scalar_sweep_matches_closed_form_depolarizing() {
    // SAFETY: no other thread touches the environment at this point.
    unsafe { std::env::set_var("PRISM_NO_AVX2_KRAUS", "1") };
    let p = 0.3;
    let kraus = depolarizing_2q_kraus(p);
    // (3,0,1) and (5,2,3) run the serial arm at W = 1 and W = 4; the 7-qubit
    // pairs run the rayon arm (2n >= 14) at both widths.
    for (n, q0, q1) in [(3usize, 0usize, 1usize), (5, 2, 3), (7, 0, 6), (7, 2, 5)] {
        let circuit = circuits::random_circuit(n, 4, SEED);
        let masks = all_pauli_masks(n);

        let mut closed = DensityMatrixBackend::new(SEED);
        sim::run_on(&mut closed, &circuit).unwrap();
        closed.apply_2q_depolarizing(q0, q1, p);

        let mut swept = DensityMatrixBackend::new(SEED);
        sim::run_on(&mut swept, &circuit).unwrap();
        swept.apply_2q_kraus(q0, q1, &kraus);

        let want = closed.expectations_pauli(&masks);
        let got = swept.expectations_pauli(&masks);
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() < DM_EPS,
                "n={n} pair=({q0},{q1}) pauli {:?}: sweep {a}, closed {b}",
                masks[k]
            );
        }
    }
}
