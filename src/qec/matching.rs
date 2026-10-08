//! Minimum-weight perfect matching decoder over graphlike detector error models.
//!
//! Exact primal-dual blossom matching on each shot's defects, with defect-to-defect
//! distances discovered on demand by one Dijkstra ball per defect.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::DetectorErrorModel;
use super::decoder::{
    BOUNDARY, GraphEdges, ShotDecoder, Stuck, decode_batch, graphlike_edges, logical_error_rate,
};
use crate::error::{PrismError, Result};
use crate::sim::compiled::PackedShots;

const NONE: u32 = u32::MAX;
const DEAD: u32 = u32::MAX - 1;
const INF: i64 = i64::MAX / 4;
/// Quantization levels for the largest edge weight; every quantized weight is
/// then doubled so all dual events fall on integer times.
const WEIGHT_LEVELS: f64 = (1u64 << 23) as f64;

const EVENT_FRONTIER: u8 = 0;
const EVENT_PAIR: u8 = 1;
const EVENT_BOUNDARY: u8 = 2;
const EVENT_EXPAND: u8 = 3;

/// Exact minimum-weight perfect matching decoder compiled from a graphlike
/// detector error model.
///
/// The decoding graph is the union-find decoder's: detectors are vertices, a
/// two-detector mechanism an internal edge and a one-detector mechanism a
/// boundary edge, each weighted `ln((1-p)/p)` clamped at zero, with mechanisms
/// sharing one detector set collapsed to the most probable. Weights are
/// quantized to `2^-23` of the largest edge weight, and the matching is exact
/// for the quantized weights: each shot's correction has the minimum total
/// weight of any edge set whose boundary is the shot's defects, the open
/// boundary absorbing any number of chain ends. Decoding uses no randomness,
/// so equal inputs give equal outputs on any thread count. The algorithm is
/// described in the decoding section of `docs/architecture/qec-programs.md`.
#[derive(Debug, Clone)]
pub struct MatchingDecoder {
    num_detectors: usize,
    num_observables: usize,
    obs_words: usize,
    adj_offsets: Vec<u32>,
    adj_target: Vec<u32>,
    adj_weight: Vec<i64>,
    adj_edge: Vec<u32>,
    edge_obs: Vec<u64>,
    boundary_dist: Vec<i64>,
    boundary_obs: Vec<u64>,
}

impl MatchingDecoder {
    /// Compile a decoder from a graphlike detector error model.
    ///
    /// # Errors
    ///
    /// As [`super::UnionFindDecoder::from_model`]: a mechanism flipping more
    /// than two detectors is rejected with a pointer to
    /// [`DetectorErrorModel::decompose_graphlike`], and probabilities must
    /// lie in `[0, 1)`.
    pub fn from_model(model: &DetectorErrorModel) -> Result<Self> {
        Ok(Self::from_graph(graphlike_edges(model, "matching")?))
    }

    fn from_graph(graph: GraphEdges) -> Self {
        let GraphEdges {
            num_detectors,
            num_observables,
            edge_u,
            edge_v,
            edge_p,
            edge_obs,
        } = graph;
        let obs_words = num_observables.div_ceil(64);
        let weights = quantized_weights(&edge_p);

        let mut adj_offsets = vec![0u32; num_detectors + 1];
        let mut boundary_edge = vec![NONE; num_detectors];
        for edge in 0..edge_u.len() {
            if edge_v[edge] == BOUNDARY {
                boundary_edge[edge_u[edge] as usize] = edge as u32;
            } else {
                adj_offsets[edge_u[edge] as usize + 1] += 1;
                adj_offsets[edge_v[edge] as usize + 1] += 1;
            }
        }
        for v in 0..num_detectors {
            adj_offsets[v + 1] += adj_offsets[v];
        }
        let slots = *adj_offsets.last().unwrap() as usize;
        let mut cursor = adj_offsets.clone();
        let mut adj_target = vec![0u32; slots];
        let mut adj_weight = vec![0i64; slots];
        let mut adj_edge = vec![0u32; slots];
        for edge in 0..edge_u.len() {
            if edge_v[edge] == BOUNDARY {
                continue;
            }
            for (from, to) in [(edge_u[edge], edge_v[edge]), (edge_v[edge], edge_u[edge])] {
                let at = cursor[from as usize] as usize;
                adj_target[at] = to;
                adj_weight[at] = weights[edge];
                adj_edge[at] = edge as u32;
                cursor[from as usize] += 1;
            }
        }

        // Multi-source Dijkstra from every boundary edge: the exact distance
        // from each detector to the boundary, and the observable mask of one
        // shortest such path.
        let mut boundary_dist = vec![INF; num_detectors];
        let mut boundary_obs = vec![0u64; num_detectors * obs_words];
        let mut settled = vec![false; num_detectors];
        let mut heap = Frontier::new();
        for (v, &edge) in boundary_edge.iter().enumerate() {
            if edge != NONE {
                heap.push(Reverse((weights[edge as usize], v as u32, NONE, edge)));
            }
        }
        while let Some(Reverse((dist, v, from, edge))) = heap.pop() {
            let at = v as usize;
            if settled[at] {
                continue;
            }
            settled[at] = true;
            boundary_dist[at] = dist;
            let edge_row = &edge_obs[edge as usize * obs_words..(edge as usize + 1) * obs_words];
            for word in 0..obs_words {
                let base = if from == NONE {
                    0
                } else {
                    boundary_obs[from as usize * obs_words + word]
                };
                boundary_obs[at * obs_words + word] = base ^ edge_row[word];
            }
            for slot in adj_offsets[at] as usize..adj_offsets[at + 1] as usize {
                let next = adj_target[slot];
                if !settled[next as usize] {
                    heap.push(Reverse((dist + adj_weight[slot], next, v, adj_edge[slot])));
                }
            }
        }

        Self {
            num_detectors,
            num_observables,
            obs_words,
            adj_offsets,
            adj_target,
            adj_weight,
            adj_edge,
            edge_obs,
            boundary_dist,
            boundary_obs,
        }
    }

    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Decode packed detector samples into predicted observable flips.
    ///
    /// Layouts and bit order as [`super::UnionFindDecoder::decode_packed`].
    ///
    /// # Errors
    ///
    /// The input measurement count must equal the model's detector count, and
    /// a shot with an odd number of defects in a detector component without a
    /// boundary edge rejects the batch, naming the first such shot.
    pub fn decode_packed(&self, detectors: &PackedShots) -> Result<PackedShots> {
        decode_batch(self, detectors)
    }

    /// Decode `detectors` and return the fraction of shots whose predicted
    /// flips differ from `observables` in any observable; `0.0` for no shots.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packed`], and `observables` must hold one bit per
    /// observable for the same shot count.
    pub fn logical_error_rate(
        &self,
        detectors: &PackedShots,
        observables: &PackedShots,
    ) -> Result<f64> {
        logical_error_rate(&self.decode_packed(detectors)?, observables)
    }

    /// Match one shot, XOR its predicted flips into `out_row`, and return the
    /// correction's quantized weight.
    fn match_shot(
        &self,
        row: &[u64],
        out_row: &mut [u64],
        s: &mut MatchScratch,
    ) -> std::result::Result<i64, Stuck> {
        s.begin_shot();
        for (word_index, &bits) in row.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                s.defects
                    .push((word_index * 64) as u32 + bits.trailing_zeros());
                bits &= bits - 1;
            }
        }
        let k = s.defects.len();
        if k == 0 {
            return Ok(0);
        }
        s.prepare(k);
        for defect in 0..k as u32 {
            let vertex = s.defects[defect as usize];
            self.settle(s, defect, vertex, 0, NONE, NONE);
        }
        for defect in 0..k as u32 {
            self.schedule_growth(s, defect);
        }

        while s.roots > 0 {
            let Some(Reverse(event)) = s.events.pop() else {
                return Err(s.stuck());
            };
            debug_assert!(event.time >= s.now);
            s.now = event.time;
            match event.kind {
                EVENT_FRONTIER => self.on_frontier(s, event.id),
                EVENT_PAIR => self.on_pair(s, event.id),
                EVENT_BOUNDARY => self.on_boundary(s, event.id),
                _ => self.on_expand(s, event.id),
            }
        }
        Ok(self.extract(s, out_row))
    }

    /// Settle `vertex` into `defect`'s ball, offer every pair the new entry
    /// meets (at the vertex or across one edge), and push unsettled neighbors.
    fn settle(
        &self,
        s: &mut MatchScratch,
        defect: u32,
        vertex: u32,
        dist: i64,
        prev: u32,
        via: u32,
    ) {
        let at = vertex as usize;
        if s.vertex_stamp[at] != s.stamp {
            s.vertex_stamp[at] = s.stamp;
            s.vertex_head[at] = NONE;
        }
        let entry = s.entries.len() as u32;
        let mut other = s.vertex_head[at];
        while other != NONE {
            let met = s.entries[other as usize];
            s.offer(defect, entry, met.defect, other, dist + met.dist, NONE);
            other = met.next;
        }
        s.entries.push(BallEntry {
            defect,
            dist,
            prev,
            via,
            next: s.vertex_head[at],
        });
        s.vertex_head[at] = entry;

        for slot in self.adj_offsets[at] as usize..self.adj_offsets[at + 1] as usize {
            let next = self.adj_target[slot];
            let reach = dist + self.adj_weight[slot];
            let mut reached = false;
            if s.vertex_stamp[next as usize] == s.stamp {
                let mut other = s.vertex_head[next as usize];
                while other != NONE {
                    let met = s.entries[other as usize];
                    if met.defect == defect {
                        reached = true;
                    } else {
                        s.offer(
                            defect,
                            entry,
                            met.defect,
                            other,
                            reach + met.dist,
                            self.adj_edge[slot],
                        );
                    }
                    other = met.next;
                }
            }
            if !reached {
                s.heaps[defect as usize].push(Reverse((reach, next, entry, self.adj_edge[slot])));
            }
        }
    }

    fn schedule_growth(&self, s: &mut MatchScratch, defect: u32) {
        let radius = s.radius(defect);
        if let Some(&Reverse((dist, ..))) = s.heaps[defect as usize].peek() {
            s.push_event((dist - radius).max(0), EVENT_FRONTIER, defect);
        }
        let boundary = self.boundary_dist[s.defects[defect as usize] as usize];
        if boundary < INF {
            debug_assert!(boundary >= radius);
            s.push_event(boundary - radius, EVENT_BOUNDARY, defect);
        }
    }

    fn reschedule(&self, s: &mut MatchScratch, node: u32) {
        let label = s.nodes[node as usize].label;
        let tail = s.nodes[node as usize].tail;
        let mut defect = s.nodes[node as usize].head;
        loop {
            for at in 0..s.incident[defect as usize].len() {
                let pair = s.incident[defect as usize][at];
                s.schedule_pair(pair);
            }
            if label == Label::S {
                self.schedule_growth(s, defect);
            }
            if defect == tail {
                break;
            }
            defect = s.defect_next[defect as usize];
        }
        if label == Label::T && node as usize >= s.defects.len() {
            let delay = s.dual(node);
            s.push_event(delay, EVENT_EXPAND, node);
        }
    }

    fn on_frontier(&self, s: &mut MatchScratch, defect: u32) {
        if s.nodes[s.top[defect as usize] as usize].label != Label::S {
            return;
        }
        let radius = s.radius(defect);
        while let Some(&Reverse((dist, vertex, prev, via))) = s.heaps[defect as usize].peek() {
            if dist > radius {
                break;
            }
            s.heaps[defect as usize].pop();
            if s.vertex_stamp[vertex as usize] == s.stamp {
                let mut other = s.vertex_head[vertex as usize];
                let mut reached = false;
                while other != NONE {
                    let met = s.entries[other as usize];
                    if met.defect == defect {
                        reached = true;
                        break;
                    }
                    other = met.next;
                }
                if reached {
                    continue;
                }
            }
            self.settle(s, defect, vertex, dist, prev, via);
        }
        if let Some(&Reverse((dist, ..))) = s.heaps[defect as usize].peek() {
            s.push_event(dist - radius, EVENT_FRONTIER, defect);
        }
    }

    fn on_pair(&self, s: &mut MatchScratch, pair: u32) {
        let Pair { a, b, cand, .. } = s.pairs[pair as usize];
        let (x, y) = (s.top[a as usize], s.top[b as usize]);
        if x == y || cand - s.radius(a) - s.radius(b) != 0 {
            return;
        }
        match (s.nodes[x as usize].label, s.nodes[y as usize].label) {
            (Label::S, Label::S) => {
                if s.nodes[x as usize].tree == s.nodes[y as usize].tree {
                    self.form_blossom(s, x, y, pair);
                } else {
                    self.augment(s, x, y, pair);
                }
            }
            (Label::S, Label::Frozen) => self.hit_frozen(s, x, y, pair),
            (Label::Frozen, Label::S) => self.hit_frozen(s, y, x, pair),
            _ => {}
        }
    }

    fn on_boundary(&self, s: &mut MatchScratch, defect: u32) {
        let x = s.top[defect as usize];
        if s.nodes[x as usize].label != Label::S
            || s.radius(defect) != self.boundary_dist[s.defects[defect as usize] as usize]
        {
            return;
        }
        let tree = s.nodes[x as usize].tree;
        s.augment_path(x);
        s.nodes[x as usize].mate = Mate::Boundary(defect);
        self.dissolve(s, tree);
    }

    fn on_expand(&self, s: &mut MatchScratch, node: u32) {
        let n = &s.nodes[node as usize];
        if n.blossom_parent != NONE || n.label != Label::T || s.dual(node) != 0 {
            return;
        }
        self.expand(s, node);
    }

    fn augment(&self, s: &mut MatchScratch, x: u32, y: u32, pair: u32) {
        let (tx, ty) = (s.nodes[x as usize].tree, s.nodes[y as usize].tree);
        s.augment_path(x);
        s.augment_path(y);
        s.nodes[x as usize].mate = Mate::Pair(pair);
        s.nodes[y as usize].mate = Mate::Pair(pair);
        self.dissolve(s, tx);
        self.dissolve(s, ty);
    }

    /// Tree node `x` reaches the out-of-tree node `y` through `pair`: grow
    /// the tree through `y` and its mate, or augment when `y` is matched to
    /// the boundary.
    fn hit_frozen(&self, s: &mut MatchScratch, x: u32, y: u32, pair: u32) {
        let tree = s.nodes[x as usize].tree;
        match s.nodes[y as usize].mate {
            Mate::Pair(mate_pair) => {
                let z = s.other_top(mate_pair, y);
                s.set_slope(y, -1);
                s.set_slope(z, 1);
                let ny = &mut s.nodes[y as usize];
                ny.label = Label::T;
                ny.tree = tree;
                ny.parent = pair;
                let nz = &mut s.nodes[z as usize];
                nz.label = Label::S;
                nz.tree = tree;
                nz.parent = mate_pair;
                s.trees[tree as usize].push(y);
                s.trees[tree as usize].push(z);
                self.reschedule(s, y);
                self.reschedule(s, z);
            }
            Mate::Boundary(_) => {
                s.augment_path(x);
                s.nodes[x as usize].mate = Mate::Pair(pair);
                s.nodes[y as usize].mate = Mate::Pair(pair);
                self.dissolve(s, tree);
            }
            Mate::None => unreachable!("out-of-tree nodes are matched"),
        }
    }

    fn dissolve(&self, s: &mut MatchScratch, tree: u32) {
        s.roots -= 1;
        let mut members = std::mem::take(&mut s.trees[tree as usize]);
        members.retain(|&node| {
            let n = &s.nodes[node as usize];
            n.blossom_parent == NONE && n.tree == tree
        });
        for &node in &members {
            s.set_slope(node, 0);
            let n = &mut s.nodes[node as usize];
            n.label = Label::Frozen;
            n.tree = NONE;
            n.parent = NONE;
        }
        for &node in &members {
            self.reschedule(s, node);
        }
        members.clear();
        s.trees[tree as usize] = members;
    }

    fn form_blossom(&self, s: &mut MatchScratch, x: u32, y: u32, pair: u32) {
        if s.mark.len() < s.nodes.len() {
            s.mark.resize(s.nodes.len(), 0);
        }
        s.mark_stamp += 1;
        let stamp = s.mark_stamp;
        let mut n = x;
        loop {
            s.mark[n as usize] = stamp;
            if s.nodes[n as usize].parent == NONE {
                break;
            }
            n = s.parent_node(n);
        }
        s.path_y.clear();
        let mut n = y;
        while s.mark[n as usize] != stamp {
            s.path_y.push(n);
            n = s.parent_node(n);
        }
        let lca = n;
        s.path_x.clear();
        let mut n = x;
        while n != lca {
            s.path_x.push(n);
            n = s.parent_node(n);
        }

        let blossom = s.nodes.len() as u32;
        let cycle_start = s.cycles.len() as u32;
        let first_link = match s.path_x.last() {
            Some(&top_x) => s.nodes[top_x as usize].parent,
            None => pair,
        };
        s.cycles.push(CycleLink {
            child: lca,
            pair: first_link,
        });
        for at in (0..s.path_x.len()).rev() {
            let child = s.path_x[at];
            let link = if at > 0 {
                s.nodes[s.path_x[at - 1] as usize].parent
            } else {
                pair
            };
            s.cycles.push(CycleLink { child, pair: link });
        }
        for at in 0..s.path_y.len() {
            let child = s.path_y[at];
            let link = s.nodes[child as usize].parent;
            s.cycles.push(CycleLink { child, pair: link });
        }
        let cycle_len = s.cycles.len() as u32 - cycle_start;
        debug_assert!(cycle_len % 2 == 1);

        let base = &s.nodes[lca as usize];
        let node = Node {
            y: 0,
            y_time: s.now,
            slope: 1,
            label: Label::S,
            tree: base.tree,
            parent: base.parent,
            mate: base.mate,
            blossom_parent: NONE,
            cycle_start,
            cycle_len,
            head: base.head,
            tail: s.nodes[s.cycles[(cycle_start + cycle_len - 1) as usize].child as usize].tail,
        };
        s.nodes.push(node);
        for at in cycle_start..cycle_start + cycle_len {
            let child = s.cycles[at as usize].child;
            s.set_slope(child, 0);
            let c = &mut s.nodes[child as usize];
            c.blossom_parent = blossom;
            c.label = Label::Frozen;
            c.tree = NONE;
            let (head, tail, dual) = (c.head, c.tail, c.y);
            if at + 1 < cycle_start + cycle_len {
                let next = s.cycles[at as usize + 1].child;
                s.defect_next[tail as usize] = s.nodes[next as usize].head;
            }
            let mut defect = head;
            loop {
                s.inner[defect as usize] += dual;
                s.top[defect as usize] = blossom;
                if defect == tail {
                    break;
                }
                defect = s.defect_next[defect as usize];
            }
        }
        let tree = s.nodes[blossom as usize].tree;
        s.trees[tree as usize].push(blossom);
        self.reschedule(s, blossom);
    }

    /// Expand a T blossom whose dual reached zero: the even-length cycle arc
    /// from the child holding the tree edge to the child holding the match
    /// stays in the tree as alternating T and S nodes, and the rest of the
    /// cycle leaves the tree as matched pairs.
    fn expand(&self, s: &mut MatchScratch, blossom: u32) {
        let b = s.nodes[blossom as usize];
        let Mate::Pair(out_pair) = b.mate else {
            unreachable!("T blossoms are matched to their tree child");
        };
        let in_defect = s.endpoint_in(b.parent, blossom);
        let out_defect = s.endpoint_in(out_pair, blossom);
        let start = b.cycle_start as usize;
        let len = b.cycle_len as usize;
        for at in start..start + len {
            let child = s.cycles[at].child;
            let c = &mut s.nodes[child as usize];
            c.blossom_parent = NONE;
            let (head, tail, dual) = (c.head, c.tail, c.y);
            let mut defect = head;
            loop {
                s.inner[defect as usize] -= dual;
                s.top[defect as usize] = child;
                if defect == tail {
                    break;
                }
                defect = s.defect_next[defect as usize];
            }
        }
        s.nodes[blossom as usize].blossom_parent = DEAD;

        let child_in = s.top[in_defect as usize];
        let child_out = s.top[out_defect as usize];
        let position = |s: &MatchScratch, child: u32| {
            (0..len)
                .find(|&at| s.cycles[start + at].child == child)
                .unwrap()
        };
        let p_in = position(s, child_in);
        let p_out = position(s, child_out);
        let forward = (p_out + len - p_in) % len;
        let (step, path_len) = if forward.is_multiple_of(2) {
            (1, forward)
        } else {
            (len - 1, len - forward)
        };
        // Link between cycle positions `at` and `at + step`.
        let link = |s: &MatchScratch, at: usize| {
            if step == 1 {
                s.cycles[start + at].pair
            } else {
                s.cycles[start + (at + len - 1) % len].pair
            }
        };

        let tree = b.tree;
        let mut at = p_in;
        let mut prev_link = b.parent;
        for k in 0..=path_len {
            let child = s.cycles[start + at].child;
            let next_link = if k == path_len { out_pair } else { link(s, at) };
            let (label, slope, mate) = if k % 2 == 0 {
                (Label::T, -1, next_link)
            } else {
                (Label::S, 1, prev_link)
            };
            s.set_slope(child, slope);
            let c = &mut s.nodes[child as usize];
            c.label = label;
            c.tree = tree;
            c.parent = prev_link;
            c.mate = Mate::Pair(mate);
            s.trees[tree as usize].push(child);
            prev_link = next_link;
            at = (at + step) % len;
        }
        for _ in 0..(len - path_len - 1) / 2 {
            let pair_link = link(s, at);
            let u = s.cycles[start + at].child;
            at = (at + step) % len;
            let v = s.cycles[start + at].child;
            at = (at + step) % len;
            for node in [u, v] {
                s.set_slope(node, 0);
                let c = &mut s.nodes[node as usize];
                c.label = Label::Frozen;
                c.tree = NONE;
                c.parent = NONE;
                c.mate = Mate::Pair(pair_link);
            }
        }
        for at in start..start + len {
            let child = s.cycles[at].child;
            self.reschedule(s, child);
        }
    }

    /// Resolve every blossom to its internal matching, XOR the observable
    /// mask of each matched path into `out_row`, and return the total weight.
    fn extract(&self, s: &mut MatchScratch, out_row: &mut [u64]) -> i64 {
        let k = s.defects.len() as u32;
        let mut weight = 0i64;
        s.stack.clear();
        for node in 0..s.nodes.len() as u32 {
            let Node {
                blossom_parent,
                mate,
                ..
            } = s.nodes[node as usize];
            if blossom_parent != NONE {
                continue;
            }
            match mate {
                Mate::Pair(pair) => {
                    let inside = s.endpoint_in(pair, node);
                    if inside == s.pairs[pair as usize].a {
                        weight += self.pair_flips(s, pair, out_row);
                    }
                    s.stack.push((node, inside));
                }
                Mate::Boundary(defect) => {
                    weight += self.boundary_flips(s, defect, out_row);
                    s.stack.push((node, defect));
                }
                Mate::None => unreachable!("every defect is matched when no tree is left"),
            }
        }
        while let Some((node, outside_defect)) = s.stack.pop() {
            if node < k {
                continue;
            }
            let start = s.nodes[node as usize].cycle_start as usize;
            let len = s.nodes[node as usize].cycle_len as usize;
            let entry_child = s.child_holding(node, outside_defect);
            let p = (0..len)
                .find(|&at| s.cycles[start + at].child == entry_child)
                .unwrap();
            s.stack.push((entry_child, outside_defect));
            for j in (1..len).step_by(2) {
                let pair = s.cycles[start + (p + j) % len].pair;
                let u = s.cycles[start + (p + j) % len].child;
                let v = s.cycles[start + (p + j + 1) % len].child;
                weight += self.pair_flips(s, pair, out_row);
                let Pair { a, b, .. } = s.pairs[pair as usize];
                let (in_u, in_v) = if s.child_holding(node, a) == u {
                    (a, b)
                } else {
                    (b, a)
                };
                s.stack.push((u, in_u));
                s.stack.push((v, in_v));
            }
        }
        weight
    }

    fn pair_flips(&self, s: &MatchScratch, pair: u32, out_row: &mut [u64]) -> i64 {
        let p = s.pairs[pair as usize];
        if p.via != NONE {
            self.xor_edge(p.via, out_row);
        }
        for start in [p.meet_a, p.meet_b] {
            let mut entry = start;
            while entry != NONE {
                let e = s.entries[entry as usize];
                if e.via != NONE {
                    self.xor_edge(e.via, out_row);
                }
                entry = e.prev;
            }
        }
        p.cand
    }

    fn boundary_flips(&self, s: &MatchScratch, defect: u32, out_row: &mut [u64]) -> i64 {
        let vertex = s.defects[defect as usize] as usize;
        let row = &self.boundary_obs[vertex * self.obs_words..(vertex + 1) * self.obs_words];
        for (word, mask) in out_row.iter_mut().zip(row) {
            *word ^= mask;
        }
        self.boundary_dist[vertex]
    }

    #[inline]
    fn xor_edge(&self, edge: u32, out_row: &mut [u64]) {
        let base = edge as usize * self.obs_words;
        for (word, mask) in out_row
            .iter_mut()
            .zip(&self.edge_obs[base..base + self.obs_words])
        {
            *word ^= mask;
        }
    }
}

impl ShotDecoder for MatchingDecoder {
    type Scratch = MatchScratch;
    type Failure = Stuck;

    fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    fn num_observables(&self) -> usize {
        self.num_observables
    }

    fn scratch(&self) -> MatchScratch {
        MatchScratch::new(self.num_detectors)
    }

    fn decode_shot(
        &self,
        row: &[u64],
        out_row: &mut [u64],
        scratch: &mut MatchScratch,
    ) -> std::result::Result<(), Stuck> {
        self.match_shot(row, out_row, scratch).map(|_| ())
    }

    fn failure_error(failure: Stuck, shot: usize) -> PrismError {
        failure.into_error(shot)
    }
}

/// Map each mechanism probability to an even integer weight proportional to
/// `ln((1-p)/p)` clamped at zero.
fn quantized_weights(edge_p: &[f64]) -> Vec<i64> {
    let raw: Vec<f64> = edge_p
        .iter()
        .map(|&p| ((1.0 - p) / p).ln().max(0.0))
        .collect();
    let largest = raw.iter().copied().fold(0.0f64, f64::max);
    let scale = if largest > 0.0 {
        WEIGHT_LEVELS / largest
    } else {
        0.0
    };
    raw.iter()
        .map(|&w| 2 * (w * scale).round() as i64)
        .collect()
}

/// Dijkstra frontier: `(distance, vertex, predecessor, edge)` per pending
/// vertex, nearest first.
type Frontier = BinaryHeap<Reverse<(i64, u32, u32, u32)>>;

/// One defect's settled vertex. `prev` and `via` trace the shortest path back
/// to the defect; `next` chains the entries settled at the same vertex.
#[derive(Debug, Clone, Copy)]
struct BallEntry {
    defect: u32,
    dist: i64,
    prev: u32,
    via: u32,
    next: u32,
}

/// Best known path between two defects: `cand` is the length through the
/// meeting entries (joined by graph edge `via`, or at one vertex when `via`
/// is `NONE`), exact once both balls cover it.
#[derive(Debug, Clone, Copy)]
struct Pair {
    a: u32,
    b: u32,
    cand: i64,
    meet_a: u32,
    meet_b: u32,
    via: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    S,
    T,
    Frozen,
}

#[derive(Debug, Clone, Copy)]
enum Mate {
    None,
    Pair(u32),
    Boundary(u32),
}

/// Matching node: a defect (ids below the shot's defect count) or a blossom.
/// The dual is `y + slope * (now - y_time)`; only top-level nodes move. Tree
/// and match edges are pair ids, resolved to nodes through `top`, so blossom
/// formation and expansion rewire nothing outside the blossom.
#[derive(Debug, Clone, Copy)]
struct Node {
    y: i64,
    y_time: i64,
    slope: i64,
    label: Label,
    tree: u32,
    parent: u32,
    mate: Mate,
    blossom_parent: u32,
    cycle_start: u32,
    cycle_len: u32,
    head: u32,
    tail: u32,
}

/// Blossom cycle child and the pair linking it to the next child.
#[derive(Debug, Clone, Copy)]
struct CycleLink {
    child: u32,
    pair: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Event {
    time: i64,
    kind: u8,
    id: u32,
}

/// Reusable per-shot matching state. Vertex slots are validated by stamp, so
/// nothing sized by the graph is cleared between shots; the per-defect pools
/// keep their capacity.
pub(super) struct MatchScratch {
    stamp: u64,
    vertex_stamp: Vec<u64>,
    vertex_head: Vec<u32>,
    entries: Vec<BallEntry>,
    heaps: Vec<Frontier>,
    defects: Vec<u32>,
    pairs: Vec<Pair>,
    incident: Vec<Vec<u32>>,
    nodes: Vec<Node>,
    cycles: Vec<CycleLink>,
    defect_next: Vec<u32>,
    inner: Vec<i64>,
    top: Vec<u32>,
    trees: Vec<Vec<u32>>,
    events: BinaryHeap<Reverse<Event>>,
    mark: Vec<u64>,
    mark_stamp: u64,
    path_x: Vec<u32>,
    path_y: Vec<u32>,
    stack: Vec<(u32, u32)>,
    now: i64,
    roots: usize,
}

impl MatchScratch {
    fn new(num_detectors: usize) -> Self {
        Self {
            stamp: 0,
            vertex_stamp: vec![0; num_detectors],
            vertex_head: vec![NONE; num_detectors],
            entries: Vec::new(),
            heaps: Vec::new(),
            defects: Vec::new(),
            pairs: Vec::new(),
            incident: Vec::new(),
            nodes: Vec::new(),
            cycles: Vec::new(),
            defect_next: Vec::new(),
            inner: Vec::new(),
            top: Vec::new(),
            trees: Vec::new(),
            events: BinaryHeap::new(),
            mark: Vec::new(),
            mark_stamp: 0,
            path_x: Vec::new(),
            path_y: Vec::new(),
            stack: Vec::new(),
            now: 0,
            roots: 0,
        }
    }

    fn begin_shot(&mut self) {
        self.stamp += 1;
        self.defects.clear();
    }

    fn prepare(&mut self, k: usize) {
        self.entries.clear();
        self.pairs.clear();
        self.nodes.clear();
        self.cycles.clear();
        self.events.clear();
        self.defect_next.clear();
        self.inner.clear();
        self.top.clear();
        if self.heaps.len() < k {
            self.heaps.resize_with(k, BinaryHeap::new);
            self.incident.resize_with(k, Vec::new);
            self.trees.resize_with(k, Vec::new);
        }
        for defect in 0..k as u32 {
            self.heaps[defect as usize].clear();
            self.incident[defect as usize].clear();
            self.trees[defect as usize].clear();
            self.trees[defect as usize].push(defect);
            self.nodes.push(Node {
                y: 0,
                y_time: 0,
                slope: 1,
                label: Label::S,
                tree: defect,
                parent: NONE,
                mate: Mate::None,
                blossom_parent: NONE,
                cycle_start: 0,
                cycle_len: 0,
                head: defect,
                tail: defect,
            });
            self.defect_next.push(NONE);
            self.inner.push(0);
            self.top.push(defect);
        }
        self.now = 0;
        self.roots = k;
    }

    #[inline]
    fn dual(&self, node: u32) -> i64 {
        let n = &self.nodes[node as usize];
        n.y + n.slope * (self.now - n.y_time)
    }

    /// Total dual of every node containing `defect`: its ball radius.
    #[inline]
    fn radius(&self, defect: u32) -> i64 {
        self.inner[defect as usize] + self.dual(self.top[defect as usize])
    }

    fn set_slope(&mut self, node: u32, slope: i64) {
        let y = self.dual(node);
        let now = self.now;
        let n = &mut self.nodes[node as usize];
        n.y = y;
        n.y_time = now;
        n.slope = slope;
    }

    fn push_event(&mut self, delay: i64, kind: u8, id: u32) {
        self.events.push(Reverse(Event {
            time: self.now + delay,
            kind,
            id,
        }));
    }

    fn offer(
        &mut self,
        defect: u32,
        entry: u32,
        other: u32,
        other_entry: u32,
        cand: i64,
        via: u32,
    ) {
        let found = self.incident[defect as usize]
            .iter()
            .copied()
            .find(|&pair| {
                let p = &self.pairs[pair as usize];
                p.a == other || p.b == other
            });
        let pair = match found {
            Some(pair) => {
                let p = &mut self.pairs[pair as usize];
                if cand >= p.cand {
                    return;
                }
                p.cand = cand;
                p.via = via;
                if p.a == defect {
                    p.meet_a = entry;
                    p.meet_b = other_entry;
                } else {
                    p.meet_a = other_entry;
                    p.meet_b = entry;
                }
                pair
            }
            None => {
                let pair = self.pairs.len() as u32;
                self.pairs.push(Pair {
                    a: defect,
                    b: other,
                    cand,
                    meet_a: entry,
                    meet_b: other_entry,
                    via,
                });
                self.incident[defect as usize].push(pair);
                self.incident[other as usize].push(pair);
                pair
            }
        };
        self.schedule_pair(pair);
    }

    fn schedule_pair(&mut self, pair: u32) {
        let Pair { a, b, cand, .. } = self.pairs[pair as usize];
        let (x, y) = (self.top[a as usize], self.top[b as usize]);
        if x == y {
            return;
        }
        let rate = self.nodes[x as usize].slope + self.nodes[y as usize].slope;
        if rate <= 0 {
            return;
        }
        let slack = cand - self.radius(a) - self.radius(b);
        debug_assert!(slack >= 0, "dual infeasible pair: slack {slack}");
        let delay = if rate == 2 {
            debug_assert!(slack % 2 == 0, "odd slack {slack} between growing nodes");
            slack / 2
        } else {
            slack
        };
        self.push_event(delay, EVENT_PAIR, pair);
    }

    /// The endpoint of `pair` inside top-level `node`.
    fn endpoint_in(&self, pair: u32, node: u32) -> u32 {
        let p = &self.pairs[pair as usize];
        if self.top[p.a as usize] == node {
            p.a
        } else {
            p.b
        }
    }

    /// The child of `blossom` that contains `defect`.
    fn child_holding(&self, blossom: u32, defect: u32) -> u32 {
        let mut n = defect;
        while self.nodes[n as usize].blossom_parent != blossom {
            n = self.nodes[n as usize].blossom_parent;
        }
        n
    }

    fn other_top(&self, pair: u32, node: u32) -> u32 {
        let p = &self.pairs[pair as usize];
        let ta = self.top[p.a as usize];
        if ta == node {
            self.top[p.b as usize]
        } else {
            ta
        }
    }

    fn parent_node(&self, node: u32) -> u32 {
        self.other_top(self.nodes[node as usize].parent, node)
    }

    /// Flip matched and unmatched tree edges from S node `node` up to its root.
    fn augment_path(&mut self, node: u32) {
        let mut n = node;
        while self.nodes[n as usize].parent != NONE {
            let t = self.parent_node(n);
            let edge = self.nodes[t as usize].parent;
            let s = self.parent_node(t);
            self.nodes[t as usize].mate = Mate::Pair(edge);
            self.nodes[s as usize].mate = Mate::Pair(edge);
            n = s;
        }
    }

    fn stuck(&self) -> Stuck {
        let detector = (0..self.defects.len())
            .filter(|&d| self.nodes[self.top[d] as usize].label == Label::S)
            .map(|d| self.defects[d])
            .min()
            .unwrap_or(0);
        Stuck { detector }
    }
}

#[cfg(test)]
#[path = "matching_tests.rs"]
mod tests;
