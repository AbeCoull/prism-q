//! Treewidth-aware cut-cost analysis for QEC observable simulation.
//!
//! Scores candidate qubit cuts by the joint cost of branch count `B` and post-cut
//! contraction width `w_post`, rather than by branch count alone.
//!
//! # Definitions
//!
//! Let `G(V, E)` be the *interaction graph* of a circuit: `V` is the qubit set, and
//! `{u, v} in E` if some two-qubit (or larger) gate has support `{u, v} subseteq T`. A
//! *cut* `S subseteq V` is any qubit subset; the connected components of `G - S` are the
//! independent sub-problems produced by cutting at those qubits.
//!
//! The *post-cut contraction width* `w_post(S)` is the maximum over components `C` of
//! `G - S` of an upper bound on the treewidth of `C`, computed by a min-fill elimination
//! heuristic (treewidth itself is NP-hard). An upper bound is sound for the dispatch
//! decision: it can only reject a backend that would have succeeded, never accept one
//! that will fail.
//!
//! # Cut score
//!
//! A cut of size `k = |S|` admits `B(S) = 2^k` branches, each fixing the cut qubits to a
//! computational basis state. The joint cost model is
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

/// Undirected qubit interaction graph; `BTreeSet` adjacency keeps the elimination order
/// deterministic.
#[derive(Debug, Clone)]
pub struct InteractionGraph {
    pub num_qubits: usize,
    pub adj: Vec<BTreeSet<usize>>,
}

impl InteractionGraph {
    /// Add a clique over each multi-qubit gate's targets, and one edge per pair inside a
    /// batched gate.
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
    pub(crate) fn components_excluding(&self, excluded: &HashSet<usize>) -> Vec<Vec<usize>> {
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

/// Upper bound on the treewidth of `G - excluded` via min-fill elimination.
///
/// Returns the largest clique `|N(v) cap remaining| + 1` formed during elimination, the
/// standard `tw <= max_clique - 1` bound reported as the clique size, since the constant
/// folds into `c` in `exp(c * w)`. The heuristic may overestimate, which is acceptable
/// for dispatch: it can reject a backend but never understates the width of the
/// elimination order it built.
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

/// Cut cost `2^|S| * exp(c * w_post(S))`, `w_post` the largest min-fill proxy over the
/// components of `G - S`.
///
/// Returns `f64::INFINITY` once the log score passes 700.
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

#[cfg(test)]
#[path = "cut_selection_tests.rs"]
mod tests;
