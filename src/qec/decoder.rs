//! Union-find decoder over graphlike detector error models.
//!
//! Weighted cluster growth in the `ln((1-p)/p)` metric followed by peeling on
//! the grown erasure, per Delfosse and Nickerson (arXiv:1709.06218).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use super::DetectorErrorModel;
use super::dem::symptom_label;
use crate::error::{PrismError, Result};
use crate::sim::compiled::{PackedShots, ShotLayout};

const BOUNDARY: u32 = u32::MAX;
const EDGE_NONE: u32 = u32::MAX;
const VERTEX_NONE: u32 = u32::MAX;
const GROWTH_EPS: f64 = 1e-9;
#[cfg(feature = "parallel")]
const PARALLEL_SHOT_THRESHOLD: usize = 1024;
#[cfg(feature = "parallel")]
const SHOT_CHUNK: usize = 256;

/// Union-find decoder compiled from a graphlike detector error model.
///
/// Detectors are vertices. A two-detector mechanism becomes an internal edge
/// and a one-detector mechanism a boundary edge, each weighted `ln((1-p)/p)`
/// clamped at zero. Mechanisms flipping no detector cannot enter the graph;
/// their probability mass bounds the logical error rate any decoder over the
/// model can reach. Mechanisms sharing one detector set collapse to the most
/// probable of them. Decoding uses no randomness: growth, fusion, and peeling
/// break every tie by ascending edge index in mechanism order, so equal
/// inputs give equal outputs on any thread count.
#[derive(Debug, Clone)]
pub struct UnionFindDecoder {
    num_detectors: usize,
    num_observables: usize,
    obs_words: usize,
    edge_u: Vec<u32>,
    edge_v: Vec<u32>,
    edge_weight: Vec<f64>,
    edge_obs: Vec<u64>,
    adj_offsets: Vec<u32>,
    adj_edge: Vec<u32>,
}

impl UnionFindDecoder {
    /// Compile a decoder from a graphlike detector error model.
    ///
    /// # Errors
    ///
    /// A mechanism flipping more than two detectors is rejected with a
    /// pointer to [`DetectorErrorModel::decompose_graphlike`]. Mechanism
    /// probabilities must lie in `[0, 1)`; a zero-probability mechanism is
    /// skipped rather than rejected.
    pub fn from_model(model: &DetectorErrorModel) -> Result<Self> {
        if model.num_detectors() >= BOUNDARY as usize {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "{} detectors exceed the decoder's index range",
                    model.num_detectors()
                ),
            });
        }
        let num_detectors = model.num_detectors();
        let num_observables = model.num_observables();
        let obs_words = num_observables.div_ceil(64);

        let mut edge_u: Vec<u32> = Vec::new();
        let mut edge_v: Vec<u32> = Vec::new();
        let mut edge_p: Vec<f64> = Vec::new();
        let mut edge_obs_rows: Vec<&[usize]> = Vec::new();
        let mut slots: HashMap<(u32, u32), u32> = HashMap::new();
        for mechanism in model.mechanisms() {
            let p = mechanism.probability();
            if !(0.0..1.0).contains(&p) {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "mechanism `{}` has probability {p}, outside [0, 1)",
                        symptom_label(mechanism)
                    ),
                });
            }
            if p == 0.0 {
                continue;
            }
            let endpoints = match *mechanism.detectors() {
                [] => continue,
                [d] => (d as u32, BOUNDARY),
                [d0, d1] => (d0 as u32, d1 as u32),
                _ => {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "mechanism `{}` flips {} detectors; union-find decoding needs a \
                             graphlike model, apply `decompose_graphlike` first",
                            symptom_label(mechanism),
                            mechanism.detectors().len()
                        ),
                    });
                }
            };
            match slots.entry(endpoints) {
                Entry::Occupied(slot) => {
                    let at = *slot.get() as usize;
                    if p > edge_p[at] {
                        edge_p[at] = p;
                        edge_obs_rows[at] = mechanism.observables();
                    }
                }
                Entry::Vacant(slot) => {
                    slot.insert(edge_u.len() as u32);
                    edge_u.push(endpoints.0);
                    edge_v.push(endpoints.1);
                    edge_p.push(p);
                    edge_obs_rows.push(mechanism.observables());
                }
            }
        }

        let edge_weight: Vec<f64> = edge_p
            .iter()
            .map(|&p| ((1.0 - p) / p).ln().max(0.0))
            .collect();
        let mut edge_obs = vec![0u64; edge_u.len() * obs_words];
        for (edge, row) in edge_obs_rows.iter().enumerate() {
            for &observable in *row {
                edge_obs[edge * obs_words + observable / 64] |= 1u64 << (observable % 64);
            }
        }

        let mut adj_offsets = vec![0u32; num_detectors + 1];
        for edge in 0..edge_u.len() {
            adj_offsets[edge_u[edge] as usize + 1] += 1;
            if edge_v[edge] != BOUNDARY {
                adj_offsets[edge_v[edge] as usize + 1] += 1;
            }
        }
        for v in 0..num_detectors {
            adj_offsets[v + 1] += adj_offsets[v];
        }
        let mut cursor = adj_offsets.clone();
        let mut adj_edge = vec![0u32; *adj_offsets.last().unwrap() as usize];
        for edge in 0..edge_u.len() {
            let u = edge_u[edge] as usize;
            adj_edge[cursor[u] as usize] = edge as u32;
            cursor[u] += 1;
            if edge_v[edge] != BOUNDARY {
                let v = edge_v[edge] as usize;
                adj_edge[cursor[v] as usize] = edge as u32;
                cursor[v] += 1;
            }
        }

        Ok(Self {
            num_detectors,
            num_observables,
            obs_words,
            edge_u,
            edge_v,
            edge_weight,
            edge_obs,
            adj_offsets,
            adj_edge,
        })
    }

    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Decode packed detector samples into predicted observable flips.
    ///
    /// Accepts either layout with one bit per detector per shot, detector `d`
    /// at bit index `d`. Returns shot-major records with one bit per
    /// observable per shot, observable `o` at bit index `o`.
    ///
    /// # Errors
    ///
    /// The input measurement count must equal the model's detector count, and
    /// every shot must be explainable: a detector component with odd defect
    /// parity and no boundary edge rejects the batch, naming the first such
    /// shot.
    pub fn decode_packed(&self, detectors: &PackedShots) -> Result<PackedShots> {
        if detectors.num_measurements() != self.num_detectors {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "detector shots carry {} measurements, the model has {} detectors",
                    detectors.num_measurements(),
                    self.num_detectors
                ),
            });
        }
        let num_shots = detectors.num_shots();
        let m_words = self.num_detectors.div_ceil(64);
        let transposed;
        let rows: &[u64] = match detectors.layout() {
            ShotLayout::ShotMajor => detectors.raw_data(),
            ShotLayout::MeasMajor => {
                transposed = detectors.clone().into_shot_major_data();
                &transposed
            }
        };
        let out_words = self.obs_words;
        let mut out = vec![0u64; num_shots * out_words];

        #[cfg(feature = "parallel")]
        if num_shots >= PARALLEL_SHOT_THRESHOLD && out_words > 0 {
            use rayon::prelude::*;
            let failure = out
                .par_chunks_mut(SHOT_CHUNK * out_words)
                .enumerate()
                .map_init(
                    || DecodeScratch::new(self),
                    |scratch, (chunk, chunk_out)| {
                        for (offset, shot_out) in chunk_out.chunks_mut(out_words).enumerate() {
                            let shot = chunk * SHOT_CHUNK + offset;
                            let row = &rows[shot * m_words..(shot + 1) * m_words];
                            if let Err(stuck) = self.decode_shot(row, shot_out, scratch) {
                                return Some((shot, stuck));
                            }
                        }
                        None
                    },
                )
                .reduce(
                    || None,
                    |a, b| match (a, b) {
                        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                        (a, b) => a.or(b),
                    },
                );
            if let Some((shot, stuck)) = failure {
                return Err(stuck.into_error(shot));
            }
            return Ok(PackedShots::from_shot_major(
                out,
                num_shots,
                self.num_observables,
            ));
        }

        let mut scratch = DecodeScratch::new(self);
        for shot in 0..num_shots {
            let row = &rows[shot * m_words..(shot + 1) * m_words];
            let shot_out = &mut out[shot * out_words..(shot + 1) * out_words];
            self.decode_shot(row, shot_out, &mut scratch)
                .map_err(|stuck| stuck.into_error(shot))?;
        }
        Ok(PackedShots::from_shot_major(
            out,
            num_shots,
            self.num_observables,
        ))
    }

    fn decode_shot(
        &self,
        row: &[u64],
        out_row: &mut [u64],
        s: &mut DecodeScratch,
    ) -> std::result::Result<(), Stuck> {
        s.stamp += 1;
        s.shot_stamp = s.stamp;

        s.defects.clear();
        for (word_index, &bits) in row.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                s.defects
                    .push((word_index * 64) as u32 + bits.trailing_zeros());
                bits &= bits - 1;
            }
        }
        if s.defects.is_empty() {
            return Ok(());
        }

        s.active.clear();
        let mut i = 0;
        while i < s.defects.len() {
            let defect = s.defects[i];
            i += 1;
            s.activate(defect);
            s.parity[defect as usize] = true;
            s.defect_stamp[defect as usize] = s.shot_stamp;
            s.active.push(defect);
        }

        self.grow_clusters(s)?;

        let mut i = 0;
        while i < s.defects.len() {
            let root = s.find(s.defects[i]);
            i += 1;
            if s.peeled_stamp[root as usize] == s.shot_stamp {
                continue;
            }
            s.peeled_stamp[root as usize] = s.shot_stamp;
            self.peel_cluster(root, out_row, s);
        }
        Ok(())
    }

    // Each round grows every active cluster's non-saturated incident edges by
    // one shared increment: the minimum slack over those edges, divided by how
    // many active clusters touch the edge, so at least one edge saturates per
    // round and the loop is bounded by the edge count.
    fn grow_clusters(&self, s: &mut DecodeScratch) -> std::result::Result<(), Stuck> {
        loop {
            s.stamp += 1;
            let round = s.stamp;

            let mut live = 0usize;
            let mut i = 0;
            while i < s.active.len() {
                let root = s.find(s.active[i]);
                i += 1;
                if s.seen_stamp[root as usize] == round {
                    continue;
                }
                s.seen_stamp[root as usize] = round;
                if s.parity[root as usize] && !s.boundary[root as usize] {
                    s.active[live] = root;
                    live += 1;
                }
            }
            s.active.truncate(live);
            if s.active.is_empty() {
                return Ok(());
            }

            s.touched.clear();
            for &root in &s.active {
                let mut grew = false;
                let mut lowest = root;
                let mut v = root;
                while v != VERTEX_NONE {
                    lowest = lowest.min(v);
                    let begin = self.adj_offsets[v as usize] as usize;
                    let end = self.adj_offsets[v as usize + 1] as usize;
                    for &edge in &self.adj_edge[begin..end] {
                        let e = edge as usize;
                        if s.edge_stamp[e] == s.shot_stamp && s.edge_saturated[e] {
                            continue;
                        }
                        grew = true;
                        if s.touch_stamp[e] == round {
                            s.touch_count[e] += 1;
                        } else {
                            s.touch_stamp[e] = round;
                            s.touch_count[e] = 1;
                            s.touched.push(edge);
                        }
                    }
                    v = s.list_next[v as usize];
                }
                if !grew {
                    return Err(Stuck { detector: lowest });
                }
            }

            let mut delta = f64::INFINITY;
            for &edge in &s.touched {
                let e = edge as usize;
                let growth = if s.edge_stamp[e] == s.shot_stamp {
                    s.edge_growth[e]
                } else {
                    0.0
                };
                let step = (self.edge_weight[e] - growth) / f64::from(s.touch_count[e]);
                if step < delta {
                    delta = step;
                }
            }

            s.fused.clear();
            for &edge in &s.touched {
                let e = edge as usize;
                if s.edge_stamp[e] != s.shot_stamp {
                    s.edge_stamp[e] = s.shot_stamp;
                    s.edge_growth[e] = 0.0;
                    s.edge_saturated[e] = false;
                }
                s.edge_growth[e] += f64::from(s.touch_count[e]) * delta;
                if s.edge_growth[e] + GROWTH_EPS >= self.edge_weight[e] {
                    s.fused.push(e as u32);
                }
            }
            s.fused.sort_unstable();
            let mut i = 0;
            while i < s.fused.len() {
                let edge = s.fused[i];
                i += 1;
                s.edge_saturated[edge as usize] = true;
                let u = self.edge_u[edge as usize];
                let v = self.edge_v[edge as usize];
                s.activate(u);
                if v == BOUNDARY {
                    let root = s.find(u);
                    s.boundary[root as usize] = true;
                    s.boundary_edge[root as usize] = s.boundary_edge[root as usize].min(edge);
                } else {
                    s.activate(v);
                    let ru = s.find(u);
                    let rv = s.find(v);
                    if ru != rv {
                        s.union(ru, rv);
                    }
                }
            }
        }
    }

    // Spanning-forest peel: leaves flush their defect through the tree edge
    // toward the root, which is the interior endpoint of the designated
    // boundary edge when the cluster touches the boundary.
    fn peel_cluster(&self, root: u32, out_row: &mut [u64], s: &mut DecodeScratch) {
        let start = if s.boundary[root as usize] {
            self.edge_u[s.boundary_edge[root as usize] as usize]
        } else {
            let mut lowest = root;
            let mut v = root;
            while v != VERTEX_NONE {
                lowest = lowest.min(v);
                v = s.list_next[v as usize];
            }
            lowest
        };

        s.order.clear();
        s.stack.clear();
        s.dfs_stamp[start as usize] = s.shot_stamp;
        s.stack.push(start);
        while let Some(v) = s.stack.pop() {
            let begin = self.adj_offsets[v as usize] as usize;
            let end = self.adj_offsets[v as usize + 1] as usize;
            for &edge in &self.adj_edge[begin..end] {
                let e = edge as usize;
                if s.edge_stamp[e] != s.shot_stamp || !s.edge_saturated[e] {
                    continue;
                }
                if self.edge_v[e] == BOUNDARY {
                    continue;
                }
                let other = if self.edge_u[e] == v {
                    self.edge_v[e]
                } else {
                    self.edge_u[e]
                };
                if s.dfs_stamp[other as usize] == s.shot_stamp {
                    continue;
                }
                s.dfs_stamp[other as usize] = s.shot_stamp;
                s.order.push((other, edge, v));
                s.stack.push(other);
            }
        }

        for &(vertex, edge, parent) in s.order.iter().rev() {
            if s.defect_stamp[vertex as usize] != s.shot_stamp {
                continue;
            }
            s.defect_stamp[vertex as usize] = 0;
            if s.defect_stamp[parent as usize] == s.shot_stamp {
                s.defect_stamp[parent as usize] = 0;
            } else {
                s.defect_stamp[parent as usize] = s.shot_stamp;
            }
            self.xor_edge_observables(edge, out_row);
        }

        if s.defect_stamp[start as usize] == s.shot_stamp {
            s.defect_stamp[start as usize] = 0;
            debug_assert!(s.boundary[root as usize]);
            self.xor_edge_observables(s.boundary_edge[root as usize], out_row);
        }
    }

    #[inline]
    fn xor_edge_observables(&self, edge: u32, out_row: &mut [u64]) {
        let base = edge as usize * self.obs_words;
        for (word, mask) in out_row
            .iter_mut()
            .zip(&self.edge_obs[base..base + self.obs_words])
        {
            *word ^= mask;
        }
    }
}

struct Stuck {
    detector: u32,
}

impl Stuck {
    fn into_error(self, shot: usize) -> PrismError {
        PrismError::InvalidParameter {
            message: format!(
                "shot {shot}: the detector component containing D{} has odd syndrome parity \
                 but no boundary edge, so the syndrome is impossible under the model",
                self.detector
            ),
        }
    }
}

/// Reusable per-shot decode state. Vertex and edge slots are validated by
/// stamp comparison against the current shot or round, so nothing is cleared
/// between shots and untouched slots cost nothing.
struct DecodeScratch {
    stamp: u64,
    shot_stamp: u64,
    parent: Vec<u32>,
    size: Vec<u32>,
    parity: Vec<bool>,
    boundary: Vec<bool>,
    boundary_edge: Vec<u32>,
    list_tail: Vec<u32>,
    list_next: Vec<u32>,
    vertex_stamp: Vec<u64>,
    seen_stamp: Vec<u64>,
    defect_stamp: Vec<u64>,
    dfs_stamp: Vec<u64>,
    peeled_stamp: Vec<u64>,
    edge_stamp: Vec<u64>,
    edge_growth: Vec<f64>,
    edge_saturated: Vec<bool>,
    touch_stamp: Vec<u64>,
    touch_count: Vec<u8>,
    defects: Vec<u32>,
    active: Vec<u32>,
    touched: Vec<u32>,
    fused: Vec<u32>,
    stack: Vec<u32>,
    order: Vec<(u32, u32, u32)>,
}

impl DecodeScratch {
    fn new(decoder: &UnionFindDecoder) -> Self {
        let vertices = decoder.num_detectors;
        let edges = decoder.edge_u.len();
        Self {
            stamp: 0,
            shot_stamp: 0,
            parent: vec![0; vertices],
            size: vec![0; vertices],
            parity: vec![false; vertices],
            boundary: vec![false; vertices],
            boundary_edge: vec![0; vertices],
            list_tail: vec![0; vertices],
            list_next: vec![0; vertices],
            vertex_stamp: vec![0; vertices],
            seen_stamp: vec![0; vertices],
            defect_stamp: vec![0; vertices],
            dfs_stamp: vec![0; vertices],
            peeled_stamp: vec![0; vertices],
            edge_stamp: vec![0; edges],
            edge_growth: vec![0.0; edges],
            edge_saturated: vec![false; edges],
            touch_stamp: vec![0; edges],
            touch_count: vec![0; edges],
            defects: Vec::new(),
            active: Vec::new(),
            touched: Vec::new(),
            fused: Vec::new(),
            stack: Vec::new(),
            order: Vec::new(),
        }
    }

    fn activate(&mut self, v: u32) {
        let at = v as usize;
        if self.vertex_stamp[at] == self.shot_stamp {
            return;
        }
        self.vertex_stamp[at] = self.shot_stamp;
        self.parent[at] = v;
        self.size[at] = 1;
        self.parity[at] = false;
        self.boundary[at] = false;
        self.boundary_edge[at] = EDGE_NONE;
        self.list_tail[at] = v;
        self.list_next[at] = VERTEX_NONE;
    }

    fn find(&mut self, mut v: u32) -> u32 {
        while self.parent[v as usize] != v {
            let grand = self.parent[self.parent[v as usize] as usize];
            self.parent[v as usize] = grand;
            v = grand;
        }
        v
    }

    fn union(&mut self, a: u32, b: u32) {
        let (big, small) = if self.size[a as usize] > self.size[b as usize]
            || (self.size[a as usize] == self.size[b as usize] && a < b)
        {
            (a, b)
        } else {
            (b, a)
        };
        let (big_at, small_at) = (big as usize, small as usize);
        self.parent[small_at] = big;
        self.size[big_at] += self.size[small_at];
        self.parity[big_at] ^= self.parity[small_at];
        self.boundary[big_at] |= self.boundary[small_at];
        self.boundary_edge[big_at] = self.boundary_edge[big_at].min(self.boundary_edge[small_at]);
        self.list_next[self.list_tail[big_at] as usize] = small;
        self.list_tail[big_at] = self.list_tail[small_at];
    }
}
