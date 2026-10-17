use super::*;
use crate::circuit::{Instruction, SmallVec};
use crate::gates::{BatchPhaseData, Gate};
use num_complex::Complex64;

fn path_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for i in 0..(n - 1) {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    c
}

fn grid_circuit(d: usize) -> Circuit {
    let mut c = Circuit::new(d * d, 0);
    for r in 0..d {
        for cc in 0..d {
            let v = r * d + cc;
            if cc + 1 < d {
                c.add_gate(Gate::Cx, &[v, v + 1]);
            }
            if r + 1 < d {
                c.add_gate(Gate::Cx, &[v, v + d]);
            }
        }
    }
    c
}

#[test]
fn interaction_graph_path_has_n_minus_one_edges() {
    let g = InteractionGraph::from_circuit(&path_circuit(6));
    assert_eq!(g.num_edges(), 5);
    for v in 1..5 {
        assert_eq!(g.adj[v].len(), 2);
    }
    assert_eq!(g.adj[0].len(), 1);
    assert_eq!(g.adj[5].len(), 1);
}

#[test]
fn interaction_graph_batch_phase_uses_internal_phase_targets() {
    let mut c = Circuit::new(3, 0);
    c.instructions.push(Instruction::Gate {
        gate: Gate::BatchPhase(Box::new(BatchPhaseData {
            phases: vec![
                (1, Complex64::new(0.0, 1.0)),
                (2, Complex64::new(-1.0, 0.0)),
            ]
            .into(),
        })),
        targets: SmallVec::from_slice(&[0]),
    });

    let g = InteractionGraph::from_circuit(&c);
    assert_eq!(g.num_edges(), 2);
    assert!(g.adj[0].contains(&1));
    assert!(g.adj[0].contains(&2));
    assert!(!g.adj[1].contains(&2));
}

#[test]
fn min_fill_path_has_clique_two() {
    let g = InteractionGraph::from_circuit(&path_circuit(10));
    let w = min_fill_treewidth_proxy(&g, &HashSet::new());
    assert_eq!(w, 2, "path treewidth proxy should be 2 (clique size)");
}

#[test]
fn min_fill_3x3_grid_has_clique_four() {
    let g = InteractionGraph::from_circuit(&grid_circuit(3));
    let w = min_fill_treewidth_proxy(&g, &HashSet::new());
    assert!(
        (3..=4).contains(&w),
        "3x3 grid treewidth proxy expected 3 or 4 (got {w})"
    );
}

#[test]
fn components_after_cutting_path_midpoint() {
    let g = InteractionGraph::from_circuit(&path_circuit(7));
    let mut cut = HashSet::new();
    cut.insert(3);
    let comps = g.components_excluding(&cut);
    assert_eq!(comps.len(), 2);
    assert_eq!(comps[0], vec![0, 1, 2]);
    assert_eq!(comps[1], vec![4, 5, 6]);
}

#[test]
fn cut_score_uncut_path_polynomial() {
    let g = InteractionGraph::from_circuit(&path_circuit(20));
    let score = cut_score(&g, &HashSet::new(), 1.0);
    assert!(score.is_finite());
    assert!(score < 1e4);
}

#[test]
fn cut_score_grid_decreases_with_separator() {
    let g = InteractionGraph::from_circuit(&grid_circuit(4));
    let uncut = cut_score(&g, &HashSet::new(), 2.0);

    let mut cut: HashSet<usize> = HashSet::new();
    for r in 0..4 {
        cut.insert(r * 4 + 2);
    }
    let separated = cut_score(&g, &cut, 2.0);

    assert!(
        separated < uncut,
        "4-qubit separator on 4x4 grid should beat uncut (got separated={separated}, uncut={uncut})"
    );
}
