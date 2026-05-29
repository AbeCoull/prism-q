//! Product-Z observable rerouting for QEC analytical paths.
//!
//! This module implements the production-safe core of Method F for explicit
//! stabilizer input. It does not infer code stabilizers. Callers must supply
//! stabilizers that fix the evaluated state or conditioned subspace.

use std::collections::HashSet;

use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::unified_pauli::{inverse_light_cone, PauliTerm};

/// Upper bound on the number of supplied stabilizers the reroute search will
/// enumerate. The search is a brute-force scan over the `2^k` subset sums of
/// the stabilizer span, each costing a full light-cone walk, so the candidate
/// count must stay tractable. `2^20 ≈ 1M` walks is the ceiling.
pub const MAX_REROUTE_STABILIZERS: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConeTelemetry {
    pub gates_total: usize,
    pub gates_in_cone: usize,
    pub t_total: usize,
    pub t_in_cone: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservableRerouteResult {
    pub original_support: Vec<usize>,
    pub rerouted_support: Vec<usize>,
    pub original: ConeTelemetry,
    pub rerouted: ConeTelemetry,
}

#[inline]
fn is_t_gate(gate: &Gate) -> bool {
    matches!(gate, Gate::T | Gate::Tdg)
}

fn z_terms(support: &[usize]) -> Vec<PauliTerm> {
    let mut out: Vec<_> = support.iter().copied().map(PauliTerm::z).collect();
    out.sort_by_key(|t| t.qubit);
    out
}

pub fn cone_telemetry(circuit: &Circuit, observable: &[PauliTerm]) -> ConeTelemetry {
    let keep = inverse_light_cone(circuit, observable);
    let mut out = ConeTelemetry {
        gates_total: 0,
        gates_in_cone: 0,
        t_total: 0,
        t_in_cone: 0,
    };
    for (idx, inst) in circuit.instructions.iter().enumerate() {
        if let Instruction::Gate { gate, .. } = inst {
            out.gates_total += 1;
            if is_t_gate(gate) {
                out.t_total += 1;
            }
            if keep[idx] {
                out.gates_in_cone += 1;
                if is_t_gate(gate) {
                    out.t_in_cone += 1;
                }
            }
        }
    }
    out
}

pub fn xor_z_support(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut support: HashSet<usize> = a.iter().copied().collect();
    for &q in b {
        if !support.insert(q) {
            support.remove(&q);
        }
    }
    let mut out: Vec<_> = support.into_iter().collect();
    out.sort_unstable();
    out
}

pub fn min_cone_z_representative(
    circuit: &Circuit,
    observable: &[usize],
    stabilizers: &[Vec<usize>],
) -> Result<ObservableRerouteResult> {
    if stabilizers.len() > MAX_REROUTE_STABILIZERS {
        return Err(PrismError::BackendUnsupported {
            backend: "QEC observable reroute".to_string(),
            operation: format!(
                "reroute search over {} stabilizers (max {}); the 2^k subset scan is intractable beyond the cap",
                stabilizers.len(),
                MAX_REROUTE_STABILIZERS
            ),
        });
    }

    let mut original_support = observable.to_vec();
    original_support.sort_unstable();
    let original = cone_telemetry(circuit, &z_terms(&original_support));
    let mut best_support = original_support.clone();
    let mut best = original;

    let candidate_count = 1usize << stabilizers.len();
    for mask in 0..candidate_count {
        let mut candidate = original_support.clone();
        for (idx, stabilizer) in stabilizers.iter().enumerate() {
            if (mask >> idx) & 1 == 1 {
                candidate = xor_z_support(&candidate, stabilizer);
            }
        }
        let telemetry = cone_telemetry(circuit, &z_terms(&candidate));
        let score = (
            telemetry.t_in_cone,
            telemetry.gates_in_cone,
            candidate.len(),
            candidate.clone(),
        );
        let best_score = (
            best.t_in_cone,
            best.gates_in_cone,
            best_support.len(),
            best_support.clone(),
        );
        if score < best_score {
            best = telemetry;
            best_support = candidate;
        }
    }

    Ok(ObservableRerouteResult {
        original_support,
        rerouted_support: best_support,
        original,
        rerouted: best,
    })
}
