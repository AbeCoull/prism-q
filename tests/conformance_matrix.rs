//! Cross-backend differential conformance over a seeded circuit corpus.
//!
//! The generator, capability matrix, and comparison engine live in
//! `tests/common/conformance.rs`; that module's docs carry the oracle policy,
//! the comparison regimes, and the skip contract. This file is the entry point
//! and the place where the fired skip rules are pinned.

mod common;

use std::collections::BTreeMap;

use common::conformance::{
    self, Anchor, AnchorPolicy, CONFORMANCE_SEED, Family, MIN_CONSENSUS_PARTICIPANTS, Regime,
    SkipReason, assert_amplitudes, assert_deterministic_case, assert_exact_mixture_case,
    assert_expectation_values, assert_native_sampling, assert_sampled_mixture_case, cases_in,
    generated_cases, participants,
};

/// Skip rules the corpus is expected to fire. A rule that starts or stops
/// firing changes what the harness silently stops checking, so the set is
/// pinned rather than left to drift.
const EXPECTED_FIRING: &[SkipReason] = &[
    SkipReason::NonClifford,
    SkipReason::EntanglingGates,
    SkipReason::StabilizerRankNoTGates,
    SkipReason::StabilizerRankNonUnitary,
    SkipReason::MpsTruncation,
    SkipReason::DensityMatrixMixesBranches,
    SkipReason::NotDecomposable,
    SkipReason::NoAmplitudeExport,
    SkipReason::FactoredBlocksHaveNoJointState,
    SkipReason::BranchingTrajectory,
    SkipReason::OutsideQueryMatrix,
    SkipReason::QueryNeedsUnitaryCircuit,
];

/// Rules the corpus does not reach, each with why. Encoded so the matrix still
/// carries the rule and a future corpus that does reach it fails loudly rather
/// than skipping quietly.
const EXPECTED_DORMANT: &[(SkipReason, &str)] = &[
    (
        SkipReason::StabilizerRankExpansionCost,
        "the Clifford+T family caps its T budget below the expansion bound",
    ),
    (
        SkipReason::StabilizerRankStatistical,
        "the sparsified route needs a stabilizer-rank budget that only tens of qubits admit",
    ),
    (
        SkipReason::TensorNetworkOutputLimit,
        "the widest case is 9 qubits, far below the dense-output bound",
    ),
    (
        SkipReason::DensityMatrixQubitCap,
        "the widest case is 9 qubits, below the 4^n guard",
    ),
    (
        SkipReason::DecompositionPerBlockSeed,
        "the decomposable family is unitary, so no block branches",
    ),
];

fn check_deterministic(family: Family) {
    let participants = participants();
    let cases = cases_in(family);
    assert!(!cases.is_empty(), "{} generated no cases", family.name());
    for case in &cases {
        assert_deterministic_case(case, &participants);
    }
}

#[test]
fn conformance_clifford_unitary_agrees_across_backends() {
    check_deterministic(Family::CliffordUnitary);
}

#[test]
fn conformance_clifford_t_unitary_agrees_across_backends() {
    check_deterministic(Family::CliffordTUnitary);
}

#[test]
fn conformance_rotation_unitary_agrees_across_backends() {
    check_deterministic(Family::RotationUnitary);
}

#[test]
fn conformance_separable_agrees_across_backends() {
    check_deterministic(Family::Separable);
}

#[test]
fn conformance_forced_trajectory_agrees_across_backends() {
    check_deterministic(Family::ForcedTrajectory);
}

#[test]
fn conformance_decomposable_agrees_across_backends() {
    check_deterministic(Family::Decomposable);
}

#[test]
fn conformance_wide_entangled_agrees_across_backends() {
    check_deterministic(Family::WideEntangled);
}

#[test]
fn conformance_entangled_reset_branches_recombine_into_the_channel() {
    let participants = participants();
    let mut widest = 0;
    for case in cases_in(Family::EntangledReset) {
        assert!(
            case.profile.has_reset && case.profile.entangling,
            "{}the entangled-reset family must reset a qubit inside an entangled block",
            case.context()
        );
        widest = widest.max(assert_exact_mixture_case(&case, &participants));
    }
    assert!(
        widest >= 2,
        "no entangled-reset case produced distinguishable branches, so the family never \
         exercised the trajectory split that separates the channel from a projection"
    );
}

#[test]
fn conformance_mid_circuit_measurement_branches_recombine_into_the_channel() {
    let participants = participants();
    let mut widest = 0;
    for case in cases_in(Family::MidCircuitMeasure) {
        widest = widest.max(assert_exact_mixture_case(&case, &participants));
    }
    assert!(
        widest >= 2,
        "no mid-circuit measurement case produced distinguishable branches"
    );
}

#[test]
fn conformance_classical_conditioning_branches_agree_across_backends() {
    let participants = participants();
    for case in cases_in(Family::ConditionalBranch) {
        assert!(
            case.profile.has_conditional && case.profile.has_measure,
            "{}the conditional family must condition on a measured bit",
            case.context()
        );
        assert_sampled_mixture_case(&case, &participants);
    }
}

/// Shot sampling on the backends that draw from their own representation, over
/// the whole unitary corpus. Each participant is checked against its own
/// probability vector, so a disagreement names the sampler rather than the
/// evolution.
#[test]
fn conformance_native_sampling_matches_each_backend_distribution() {
    let participants = participants();
    for case in generated_cases() {
        assert_native_sampling(&case, &participants);
    }
}

/// The query matrices are differential, so a unitary case that reaches fewer
/// than two participants compares nothing while still reporting pass.
#[test]
fn conformance_query_matrix_compares_every_unitary_case() {
    let participants = participants();
    let mut unitary_cases = 0;
    for case in generated_cases() {
        if !case.profile.unitary() {
            continue;
        }
        unitary_cases += 1;
        let compared: Vec<&str> = participants
            .iter()
            .filter(|participant| participant.query_eligibility(&case.profile).is_compared())
            .map(|participant| participant.name)
            .collect();
        assert!(
            compared.len() >= 2,
            "{}only {:?} reached the query matrix; shots and observables need at least two \
             participants to compare",
            case.context(),
            compared
        );
    }
    assert!(
        unitary_cases > 0,
        "the corpus produced no unitary case, so the query matrices ran empty"
    );
}

#[test]
fn conformance_expectation_values_agree_where_the_matrix_exposes_them() {
    let participants = participants();
    for case in generated_cases() {
        assert_expectation_values(&case, &participants);
    }
}

#[test]
fn conformance_amplitudes_agree_where_the_matrix_exposes_them() {
    let participants = participants();
    for case in generated_cases() {
        assert_amplitudes(&case, &participants);
    }
}

#[test]
fn conformance_generator_is_reproducible_from_the_seed() {
    let first = generated_cases();
    let second = generated_cases();
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(
            conformance::describe(&a.circuit),
            conformance::describe(&b.circuit),
            "case {} is not reproducible from seed {CONFORMANCE_SEED}",
            a.name()
        );
        assert_eq!(a.profile, b.profile, "profile drift on case {}", a.name());
    }
}

#[test]
fn conformance_corpus_covers_every_declared_shape() {
    let cases = generated_cases();
    for family in Family::ALL {
        assert!(
            cases.iter().any(|case| case.family == family),
            "family {} produced no cases",
            family.name()
        );
    }
    assert!(
        cases.iter().any(|case| case.profile.clifford_only),
        "no Clifford-only case"
    );
    assert!(
        cases.iter().any(|case| !case.profile.clifford_only),
        "no non-Clifford case"
    );
    assert!(
        cases.iter().any(|case| case.profile.has_measure),
        "no mid-circuit measurement case"
    );
    assert!(
        cases.iter().any(|case| case.profile.has_reset),
        "no reset case"
    );
    assert!(
        cases.iter().any(|case| case.profile.has_conditional),
        "no classical conditioning case"
    );
    assert!(
        cases.iter().any(|case| case.profile.decomposes),
        "no case reaches the subsystem-decomposition route"
    );
    assert!(
        cases
            .iter()
            .any(|case| case.family.regime() == Regime::ExactMixture
                && case.profile.has_reset
                && case.profile.entangling),
        "no generated case resets a qubit inside an entangled block"
    );
}

#[test]
fn conformance_skip_rules_are_explicit_and_pinned() {
    let ledger = conformance::skip_ledger();
    let fired: Vec<SkipReason> = ledger.keys().copied().collect();
    let expected: Vec<SkipReason> = {
        let mut sorted = EXPECTED_FIRING.to_vec();
        sorted.sort();
        sorted
    };
    assert_eq!(
        fired,
        expected,
        "the set of fired skip rules moved.\nledger: {ledger:?}\nmatrix:\n{}",
        conformance::matrix_report()
    );

    let dormant: BTreeMap<SkipReason, &str> = EXPECTED_DORMANT.iter().copied().collect();
    for reason in SkipReason::ALL {
        let in_expected = EXPECTED_FIRING.contains(&reason);
        let in_dormant = dormant.contains_key(&reason);
        assert!(
            in_expected != in_dormant,
            "skip rule `{}` must be listed as firing or as dormant with a reason, not both or \
             neither",
            reason.name()
        );
        if in_dormant {
            assert!(
                !ledger.contains_key(&reason),
                "skip rule `{}` was recorded dormant ({}) but fired {} time(s)",
                reason.name(),
                dormant[&reason],
                ledger[&reason]
            );
        }
    }
}

#[test]
fn conformance_every_case_has_an_anchor_or_a_quorum() {
    let participants = participants();
    for case in generated_cases() {
        let compared: Vec<&str> = participants
            .iter()
            .filter(|participant| participant.eligibility(&case.profile).is_compared())
            .map(|participant| participant.name)
            .collect();
        assert!(
            compared.len() >= 2,
            "{}only {:?} were compared; a differential check needs at least two",
            case.context(),
            compared
        );
        match case.family.anchor_policy() {
            AnchorPolicy::Required(anchor) => {
                let from_participant = participants.iter().any(|participant| {
                    participant.anchor == Some(anchor)
                        && participant.eligibility(&case.profile).is_compared()
                });
                let from_oracle =
                    anchor == Anchor::Channel && case.family.regime() == Regime::ExactMixture;
                assert!(
                    from_participant || from_oracle,
                    "{}declares the {anchor:?} anchor but nothing provides it",
                    case.context()
                );
            }
            AnchorPolicy::ConsensusOnly => assert!(
                compared.len() >= MIN_CONSENSUS_PARTICIPANTS,
                "{}has no independent anchor and only {} compared participants (minimum {})",
                case.context(),
                compared.len(),
                MIN_CONSENSUS_PARTICIPANTS
            ),
        }
    }
}
