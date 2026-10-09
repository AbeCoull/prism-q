use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::*;

const PROBABILITIES: [f64; 5] = [0.01, 0.05, 0.1, 0.2, 0.3];

// Half the draws come from a short list so equal-weight ties are common.
fn random_probability(rng: &mut ChaCha8Rng) -> f64 {
    if rng.random_range(0..2) == 0 {
        PROBABILITIES[rng.random_range(0..PROBABILITIES.len())]
    } else {
        rng.random_range(0.002..0.45)
    }
}

fn random_graph(rng: &mut ChaCha8Rng, vertices: usize, max_edges: usize) -> GraphEdges {
    let mut candidates: Vec<(u32, u32)> = Vec::new();
    for u in 0..vertices as u32 {
        candidates.push((u, BOUNDARY));
        for v in u + 1..vertices as u32 {
            candidates.push((u, v));
        }
    }
    let mut graph = GraphEdges {
        num_detectors: vertices,
        num_observables: 2,
        edge_u: Vec::new(),
        edge_v: Vec::new(),
        edge_p: Vec::new(),
        edge_obs: Vec::new(),
    };
    while graph.edge_u.len() < max_edges && !candidates.is_empty() {
        let (u, v) = candidates.swap_remove(rng.random_range(0..candidates.len()));
        graph.edge_u.push(u);
        graph.edge_v.push(v);
        graph.edge_p.push(random_probability(rng));
        graph.edge_obs.push(rng.random_range(0..4u64));
    }
    graph
}

/// Minimum correction weight per syndrome over every edge subset, and the
/// observable masks reached at that weight as a bitset over the four values.
fn brute_force_corrections(graph: &GraphEdges, weights: &[i64]) -> (Vec<i64>, Vec<u8>) {
    let syndromes = 1usize << graph.num_detectors;
    let mut best = vec![INF; syndromes];
    let mut masks = vec![0u8; syndromes];
    for subset in 0..1u32 << graph.edge_u.len() {
        let mut syndrome = 0usize;
        let mut weight = 0i64;
        let mut obs = 0u64;
        let mut bits = subset;
        while bits != 0 {
            let edge = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            syndrome ^= 1 << graph.edge_u[edge];
            if graph.edge_v[edge] != BOUNDARY {
                syndrome ^= 1 << graph.edge_v[edge];
            }
            weight += weights[edge];
            obs ^= graph.edge_obs[edge];
        }
        if weight < best[syndrome] {
            best[syndrome] = weight;
            masks[syndrome] = 1 << obs;
        } else if weight == best[syndrome] {
            masks[syndrome] |= 1 << obs;
        }
    }
    (best, masks)
}

#[test]
fn matching_equals_brute_force_minimum_on_random_graphs() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..400 {
        let vertices = rng.random_range(2..=8);
        let max_edges = rng.random_range(vertices..=14);
        let graph = random_graph(&mut rng, vertices, max_edges);
        let weights = quantized_weights(&graph.edge_p);
        let (best, masks) = brute_force_corrections(&graph, &weights);
        let decoder = MatchingDecoder::from_graph(graph);
        let mut scratch = decoder.scratch();
        for syndrome in 0..best.len() {
            let mut out = [0u64; 1];
            let matched = decoder.match_shot(&[syndrome as u64], &mut out, &mut scratch);
            if best[syndrome] == INF {
                assert!(matched.is_err(), "syndrome {syndrome:b} has no correction");
                continue;
            }
            let weight = matched.unwrap_or_else(|_| panic!("syndrome {syndrome:b} stuck"));
            assert_eq!(weight, best[syndrome], "syndrome {syndrome:b}");
            assert!(
                masks[syndrome] >> out[0] & 1 == 1,
                "syndrome {syndrome:b}: observable mask {} is not reached by a minimum correction",
                out[0]
            );
        }
    }
}

/// Exact minimum-weight perfect matching over the metric closure by subset
/// recursion: the lowest defect pairs with another or with the boundary.
fn brute_force_matching(dist: &[Vec<i64>], boundary: &[i64]) -> i64 {
    let k = boundary.len();
    let mut memo = vec![INF; 1 << k];
    memo[0] = 0;
    for set in 1..1usize << k {
        let i = set.trailing_zeros() as usize;
        let rest = set & !(1 << i);
        let mut best = INF;
        if boundary[i] < INF && memo[rest] < INF {
            best = boundary[i] + memo[rest];
        }
        let mut others = rest;
        while others != 0 {
            let j = others.trailing_zeros() as usize;
            others &= others - 1;
            let remaining = rest & !(1 << j);
            if dist[i][j] < INF && memo[remaining] < INF {
                best = best.min(dist[i][j] + memo[remaining]);
            }
        }
        memo[set] = best;
    }
    memo[(1 << k) - 1]
}

/// A connected random graph of mean degree three, a quarter of the vertices
/// carrying a boundary edge.
fn random_sparse_graph(rng: &mut ChaCha8Rng, vertices: usize) -> GraphEdges {
    let mut edges = std::collections::BTreeSet::new();
    for v in 1..vertices as u32 {
        edges.insert((rng.random_range(0..v), v));
    }
    while edges.len() < vertices * 3 / 2 {
        let u = rng.random_range(0..vertices as u32);
        let v = rng.random_range(0..vertices as u32);
        if u != v {
            edges.insert((u.min(v), u.max(v)));
        }
    }
    for v in 0..vertices as u32 {
        if rng.random_range(0..4) == 0 {
            edges.insert((v, BOUNDARY));
        }
    }
    let mut graph = GraphEdges {
        num_detectors: vertices,
        num_observables: 1,
        edge_u: Vec::new(),
        edge_v: Vec::new(),
        edge_p: Vec::new(),
        edge_obs: Vec::new(),
    };
    for (u, v) in edges {
        graph.edge_u.push(u);
        graph.edge_v.push(v);
        graph.edge_p.push(random_probability(rng));
        graph.edge_obs.push(rng.random_range(0..2u64));
    }
    graph
}

#[test]
fn matching_equals_metric_closure_minimum_on_large_syndromes() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..60 {
        let vertices = rng.random_range(20..=48);
        let graph = random_sparse_graph(&mut rng, vertices);
        let weights = quantized_weights(&graph.edge_p);
        let mut dist = vec![vec![INF; vertices]; vertices];
        let mut boundary = vec![INF; vertices];
        for (v, row) in dist.iter_mut().enumerate() {
            row[v] = 0;
        }
        for (edge, &weight) in weights.iter().enumerate() {
            let u = graph.edge_u[edge] as usize;
            if graph.edge_v[edge] == BOUNDARY {
                boundary[u] = boundary[u].min(weight);
            } else {
                let v = graph.edge_v[edge] as usize;
                dist[u][v] = dist[u][v].min(weight);
                dist[v][u] = dist[u][v];
            }
        }
        for m in 0..vertices {
            for u in 0..vertices {
                for v in 0..vertices {
                    if dist[u][m] < INF && dist[m][v] < INF {
                        dist[u][v] = dist[u][v].min(dist[u][m] + dist[m][v]);
                    }
                }
            }
        }
        let direct = boundary.clone();
        for u in 0..vertices {
            for v in 0..vertices {
                if dist[u][v] < INF && direct[v] < INF {
                    boundary[u] = boundary[u].min(dist[u][v] + direct[v]);
                }
            }
        }

        let decoder = MatchingDecoder::from_graph(graph);
        let mut scratch = decoder.scratch();
        for _ in 0..40 {
            let k = rng.random_range(1..=12usize);
            let mut defects: Vec<usize> = (0..vertices).collect();
            for at in 0..k {
                let pick = rng.random_range(at..vertices);
                defects.swap(at, pick);
            }
            defects.truncate(k);
            defects.sort_unstable();
            let mut row = [0u64; 1];
            for &d in &defects {
                row[0] |= 1 << d;
            }
            let local_dist: Vec<Vec<i64>> = defects
                .iter()
                .map(|&a| defects.iter().map(|&b| dist[a][b]).collect())
                .collect();
            let local_boundary: Vec<i64> = defects.iter().map(|&d| boundary[d]).collect();
            let expected = brute_force_matching(&local_dist, &local_boundary);
            let mut out = [0u64; 1];
            let matched = decoder.match_shot(&row, &mut out, &mut scratch);
            if expected >= INF {
                assert!(matched.is_err(), "defects {defects:?} have no matching");
            } else {
                let weight = matched.unwrap_or_else(|_| panic!("defects {defects:?} stuck"));
                assert_eq!(weight, expected, "defects {defects:?}");
            }
        }
    }
}

#[test]
fn quantized_weights_are_even_and_scale_to_the_largest() {
    let weights = quantized_weights(&[0.5, 0.25, 0.01, 0.75]);
    assert_eq!(weights[0], 0);
    assert_eq!(weights[3], 0, "p above one half clamps at zero");
    assert_eq!(weights[2], 2 * (1 << 23));
    assert!(weights.iter().all(|w| w % 2 == 0));
}
