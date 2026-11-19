use super::tiles::{QftPass, qft_pass_plan};
use super::{Route2q, route_2q};
use crate::gates::Gate;
use num_complex::Complex64;

#[test]
fn route_2q_reads_exact_structure_only() {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    let phase = Complex64::from_polar(1.0, 0.4);
    assert_eq!(
        route_2q(&Gate::Cx.matrix_4x4()),
        Route2q::Cx { control_high: true }
    );
    assert_eq!(
        route_2q(&[[o, z, z, z], [z, z, z, o], [z, z, o, z], [z, o, z, z]]),
        Route2q::Cx {
            control_high: false
        }
    );
    assert_eq!(route_2q(&Gate::Swap.matrix_4x4()), Route2q::Swap);
    assert_eq!(route_2q(&Gate::Cz.matrix_4x4()), Route2q::Cz);
    assert_eq!(
        route_2q(&[[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, phase]]),
        Route2q::ControlledPhase(phase)
    );
    assert_eq!(
        route_2q(&[[z, o, z, z], [phase, z, z, z], [z, z, o, z], [z, z, z, o]]),
        Route2q::Monomial {
            src: [1, 0, 2, 3],
            scale: [o, phase, o, o]
        }
    );
    let mut near = Gate::Cx.matrix_4x4();
    near[0][1] = Complex64::new(1e-300, 0.0);
    assert_eq!(route_2q(&near), Route2q::Dense);
    let mut singular = Gate::Swap.matrix_4x4();
    singular[3][3] = z;
    assert_eq!(route_2q(&singular), Route2q::Dense);
    let mut repeated = Gate::Swap.matrix_4x4();
    repeated[0] = [z, o, z, z];
    assert_eq!(route_2q(&repeated), Route2q::Dense);
}

#[test]
fn qft_pass_plan_splits_high_stages_evenly_above_the_tile() {
    let pass = |lo, bits| QftPass { lo, bits };
    assert_eq!(qft_pass_plan(0, 10), [pass(0, 10)]);
    assert_eq!(qft_pass_plan(0, 20), [pass(11, 9), pass(0, 11)]);
    assert_eq!(
        qft_pass_plan(0, 28),
        [pass(19, 9), pass(11, 8), pass(0, 11)]
    );
    assert_eq!(qft_pass_plan(1, 12), [pass(10, 2), pass(0, 10)]);
    assert_eq!(qft_pass_plan(5, 17), [pass(9, 8), pass(0, 9)]);
}
