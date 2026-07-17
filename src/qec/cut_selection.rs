//! Treewidth-aware cut-cost analysis for QEC observable simulation.
//!
//! Scores candidate qubit cuts by the joint cost of branch count `B` and
//! post-cut contraction width `w_post`, rather than by branch count alone.
//!
//! Status: this is analysis tooling, not a live dispatch path. The QEC auto
//! T-strategy (`run_qec_program_auto`) does not yet consult these scores; it
//! follows a fixed SPD -> CAMPS -> tensor-network ladder.
//! These functions are exercised by the cut-cost benchmarks and are intended
//! to inform a future width-aware dispatcher. Wiring them into the dispatcher
//! is deferred until the cost model is validated against benchmark results.
//!
//! # Definitions
//!
//! Let `G(V, E)` be the *interaction graph* of a circuit: `V` is the qubit
//! set, and `{u, v} in E` if some two-qubit (or larger) gate has support
//! `{u, v} subseteq T`. A *cut* `S subseteq V` is any qubit subset; removing
//! `S` from `G` yields a graph `G - S` whose connected components are the
//! independent sub-problems produced by cutting at those qubits.
//!
//! The *post-cut contraction width* `w_post(S)` is the maximum over
//! components `C` of `G - S` of an upper bound on the treewidth of `C`,
//! computed by a min-fill elimination heuristic. `w_post` is an upper
//! bound on the actual treewidth (treewidth itself is NP-hard); using an
//! upper bound is sound for the dispatch decision (it can only reject a
//! backend that would have succeeded, never accept one that will fail).
//!
//! # Cut score
//!
//! For a cut `S` of size `k = |S|` admitting `B(S) = 2^k` branches (each
//! branch fixes the `k` cut qubits to a computational basis state), the
//! joint cost model is
//!
//! ```text
//! score(S) = B(S) * exp(c * w_post(S))
//! ```
//!
//! where `c` is the per-treewidth-bit cost constant. The right cut is
//! `argmin_S score(S)`.

use std::collections::{BTreeSet, HashSet};

use crate::circuit::{Circuit, Instruction};
use crate::gates::Gate;

/// Undirected interaction graph of a circuit. Adjacency is stored as a
/// `Vec<BTreeSet<usize>>` indexed by qubit id; `BTreeSet` keeps the
/// elimination order deterministic.
#[derive(Debug, Clone)]
pub struct InteractionGraph {
    pub num_qubits: usize,
    pub adj: Vec<BTreeSet<usize>>,
}

impl InteractionGraph {
    /// Extract the interaction graph of a circuit. For each instruction with
    /// `k >= 2` targets, add the complete clique on those targets.
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let n = circuit.num_qubits;
        let mut adj = vec![BTreeSet::<usize>::new(); n];
        for inst in &circuit.instructions {
            if let Instruction::Gate { gate, targets } = inst {
                match gate {
                    Gate::BatchPhase(data) => {
                        if let Some(&control) = targets.first() {
                            for &(target, _) in &data.phases {
                                add_interaction_edge(&mut adj, control, target);
                            }
                        }
                        continue;
                    }
                    Gate::BatchRzz(data) => {
                        for &(q0, q1, _) in &data.edges {
                            add_interaction_edge(&mut adj, q0, q1);
                        }
                        continue;
                    }
                    Gate::DiagonalBatch(data) => {
                        for entry in &data.entries {
                            if let Some((q0, q1, _)) = entry.as_2q_matrix() {
                                add_interaction_edge(&mut adj, q0, q1);
                            }
                        }
                        continue;
                    }
                    Gate::Multi2q(data) => {
                        for &(q0, q1, _) in &data.gates {
                            add_interaction_edge(&mut adj, q0, q1);
                        }
                        continue;
                    }
                    _ => {}
                }
                if targets.len() < 2 {
                    continue;
                }
                for i in 0..targets.len() {
                    for j in (i + 1)..targets.len() {
                        let (a, b) = (targets[i], targets[j]);
                        add_interaction_edge(&mut adj, a, b);
                    }
                }
            }
        }
        Self { num_qubits: n, adj }
    }

    pub fn num_edges(&self) -> usize {
        self.adj.iter().map(|s| s.len()).sum::<usize>() / 2
    }

    /// Connected components of `G - excluded`. Returns one `Vec<usize>` per
    /// component, in increasing-min-vertex order.
    pub fn components_excluding(&self, excluded: &HashSet<usize>) -> Vec<Vec<usize>> {
        let mut visited: HashSet<usize> = (0..self.num_qubits)
            .filter(|v| excluded.contains(v))
            .collect();
        let mut comps = Vec::new();
        for start in 0..self.num_qubits {
            if visited.contains(&start) {
                continue;
            }
            let mut stack = vec![start];
            let mut comp = Vec::new();
            while let Some(v) = stack.pop() {
                if !visited.insert(v) {
                    continue;
                }
                comp.push(v);
                for &u in &self.adj[v] {
                    if !visited.contains(&u) {
                        stack.push(u);
                    }
                }
            }
            comp.sort_unstable();
            comps.push(comp);
        }
        comps
    }
}

fn add_interaction_edge(adj: &mut [BTreeSet<usize>], a: usize, b: usize) {
    if a != b {
        adj[a].insert(b);
        adj[b].insert(a);
    }
}

/// Upper bound on the treewidth of `G - excluded` via the min-fill
/// elimination heuristic.
///
/// Returns the maximum, over all eliminated vertices, of `|N(v) cap remaining|`
/// at elimination time. This is the standard `tw <= max_clique_during_elim - 1`
/// bound expressed as the clique size; the clique size itself is reported
/// (i.e., `tw_proxy = max_clique`) because the cost model is `exp(c * w)`
/// and the additive constant absorbs into `c`.
///
/// **Soundness.** The produced elimination order has a concrete maximum
/// induced clique size, and that value is an upper bound on the optimal
/// treewidth plus one. The heuristic may overestimate, which is acceptable
/// for dispatch because it can reject a backend but cannot understate the
/// width of the chosen elimination order.
pub fn min_fill_treewidth_proxy(graph: &InteractionGraph, excluded: &HashSet<usize>) -> usize {
    let mut remaining: HashSet<usize> = (0..graph.num_qubits)
        .filter(|v| !excluded.contains(v))
        .collect();
    if remaining.is_empty() {
        return 0;
    }

    let mut adj = graph.adj.clone();
    let mut max_clique = 0usize;

    while !remaining.is_empty() {
        let mut best: Option<(usize, usize, usize)> = None;
        for &v in &remaining {
            let nbrs_in: Vec<usize> = adj[v]
                .iter()
                .copied()
                .filter(|u| remaining.contains(u))
                .collect();
            let deg = nbrs_in.len();
            let mut fill = 0usize;
            for i in 0..nbrs_in.len() {
                for j in (i + 1)..nbrs_in.len() {
                    if !adj[nbrs_in[i]].contains(&nbrs_in[j]) {
                        fill += 1;
                    }
                }
            }
            let key = (fill, deg, v);
            if best.is_none_or(|b| key < b) {
                best = Some(key);
            }
        }
        let (_, deg, v) = best.unwrap();
        let clique = deg + 1;
        if clique > max_clique {
            max_clique = clique;
        }
        let nbrs_in: Vec<usize> = adj[v]
            .iter()
            .copied()
            .filter(|u| remaining.contains(u))
            .collect();
        for i in 0..nbrs_in.len() {
            for j in (i + 1)..nbrs_in.len() {
                let (a, b) = (nbrs_in[i], nbrs_in[j]);
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        remaining.remove(&v);
    }
    max_clique
}

/// Score combining branch count and post-cut contraction width.
///
/// `score(S) = B(S) * exp(c * w_post(S))`
/// where `B(S) = 2^|S|` and `w_post(S)` is the per-component max of the
/// min-fill treewidth proxy on `G - S`.
///
/// Returns `f64::INFINITY` if either factor overflows.
pub fn cut_score(graph: &InteractionGraph, cut: &HashSet<usize>, c: f64) -> f64 {
    let k = cut.len() as f64;
    let comps = graph.components_excluding(cut);
    let mut w_post = 0usize;
    for comp in &comps {
        let comp_excluded: HashSet<usize> = (0..graph.num_qubits)
            .filter(|v| !comp.contains(v))
            .collect();
        let w = min_fill_treewidth_proxy(graph, &comp_excluded);
        if w > w_post {
            w_post = w;
        }
    }
    let log_score = k * std::f64::consts::LN_2 + c * (w_post as f64);
    if log_score > 700.0 {
        f64::INFINITY
    } else {
        log_score.exp()
    }
}

/// Find the best single-qubit cut by exhaustive search.
///
/// Used as a baseline and for small interaction graphs (n <= ~30). Returns
/// the cut qubit and its score, or `None` if no cut improves on the
/// uncut score `exp(c * w(G))`.
pub fn best_single_cut(graph: &InteractionGraph, c: f64) -> Option<(usize, f64)> {
    let uncut = cut_score(graph, &HashSet::new(), c);
    let mut best: Option<(usize, f64)> = None;
    for v in 0..graph.num_qubits {
        let mut s = HashSet::new();
        s.insert(v);
        let score = cut_score(graph, &s, c);
        if score < uncut && best.is_none_or(|(_, sb)| score < sb) {
            best = Some((v, score));
        }
    }
    best
}

/// Nested-dissection cut sequence on a path or grid-like graph.
///
/// Recursively cuts at the vertex (or pair of vertices) whose removal
/// produces the most-balanced split. Returns the sequence of cut sets, in
/// the order they should be applied. Each successive cut operates on one
/// component of the previous step.
///
/// For a `1 x n` path this returns roughly `log_2(n)` single-vertex cuts at
/// midpoints. For a `d x d` grid this returns `O(d)` linear separators.
pub fn nested_dissection_cuts(graph: &InteractionGraph, max_component: usize) -> Vec<Vec<usize>> {
    let mut cuts = Vec::new();
    let mut work: Vec<Vec<usize>> = vec![(0..graph.num_qubits).collect()];

    while let Some(component) = work.pop() {
        if component.len() <= max_component {
            continue;
        }
        let mut best: Option<(usize, isize)> = None;
        for &v in &component {
            let mut excl: HashSet<usize> = (0..graph.num_qubits)
                .filter(|u| !component.contains(u))
                .collect();
            excl.insert(v);
            let sub_comps = graph.components_excluding(&excl);
            if sub_comps.len() < 2 {
                continue;
            }
            let max_part = sub_comps.iter().map(|c| c.len()).max().unwrap();
            let balance = -(max_part as isize);
            if best.is_none_or(|(_, b)| balance > b) {
                best = Some((v, balance));
            }
        }
        let Some((v, _)) = best else { continue };
        cuts.push(vec![v]);
        let mut excl: HashSet<usize> = (0..graph.num_qubits)
            .filter(|u| !component.contains(u))
            .collect();
        excl.insert(v);
        let new_comps = graph.components_excluding(&excl);
        for sub in new_comps {
            work.push(sub);
        }
    }
    cuts
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn nested_dissection_path_recurses() {
        let g = InteractionGraph::from_circuit(&path_circuit(8));
        let cuts = nested_dissection_cuts(&g, 2);
        assert!(!cuts.is_empty());
        assert!(cuts.iter().all(|c| c.len() == 1));
    }

    #[test]
    fn best_single_cut_no_help_on_minimal_width_path() {
        let g = InteractionGraph::from_circuit(&path_circuit(7));
        let result = best_single_cut(&g, 1.0);
        assert!(
            result.is_none(),
            "paths already have width 2; branch cost 2 cannot improve"
        );
    }

    #[test]
    fn best_single_cut_helps_on_complete_graph() {
        let mut c = Circuit::new(5, 0);
        for i in 0..5 {
            for j in (i + 1)..5 {
                c.add_gate(Gate::Cx, &[i, j]);
            }
        }
        let g = InteractionGraph::from_circuit(&c);
        let result = best_single_cut(&g, 1.0);
        assert!(
            result.is_some(),
            "K_5 has width 5; cutting one vertex gives width 4, saves exp(1) versus branch cost 2"
        );
    }
}
