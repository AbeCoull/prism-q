#[derive(Debug, Clone)]
pub struct SparseParity {
    pub col_indices: Vec<u32>,
    pub row_offsets: Vec<u32>,
    pub num_rows: usize,
}

impl SparseParity {
    pub fn from_flip_rows(flip_rows: &[Vec<u64>], num_measurements: usize) -> Self {
        let num_rows = num_measurements;
        let rank = flip_rows.len();
        let mut row_offsets = Vec::with_capacity(num_rows + 1);
        let mut col_indices = Vec::new();

        for m in 0..num_rows {
            row_offsets.push(col_indices.len() as u32);
            let w = m / 64;
            let bit = m % 64;
            for (j, row) in flip_rows.iter().enumerate().take(rank) {
                if (row[w] >> bit) & 1 != 0 {
                    col_indices.push(j as u32);
                }
            }
        }
        row_offsets.push(col_indices.len() as u32);

        Self {
            col_indices,
            row_offsets,
            num_rows,
        }
    }

    #[inline(always)]
    pub fn row_weight(&self, row: usize) -> usize {
        (self.row_offsets[row + 1] - self.row_offsets[row]) as usize
    }

    pub fn row_cols(&self, row: usize) -> &[u32] {
        let start = self.row_offsets[row] as usize;
        let end = self.row_offsets[row + 1] as usize;
        &self.col_indices[start..end]
    }

    pub fn build_xor_dag(&self) -> XorDag {
        let n = self.num_rows;
        let mut entries: Vec<XorDagEntry> = Vec::with_capacity(n);

        for m in 0..n {
            let cols = self.row_cols(m);
            let weight = cols.len();

            let mut best_parent = None;
            let mut best_residual_weight = weight;

            for p in 0..m {
                let parent_cols = self.row_cols(p);
                let sym_diff_size = symmetric_difference_size(cols, parent_cols);
                if sym_diff_size < best_residual_weight {
                    best_residual_weight = sym_diff_size;
                    best_parent = Some(p);
                }
            }

            if let Some(p) = best_parent {
                if best_residual_weight < weight {
                    let parent_cols = self.row_cols(p);
                    let residual = symmetric_difference(cols, parent_cols);
                    entries.push(XorDagEntry {
                        parent: Some(p),
                        residual_cols: residual,
                    });
                } else {
                    entries.push(XorDagEntry {
                        parent: None,
                        residual_cols: cols.to_vec(),
                    });
                }
            } else {
                entries.push(XorDagEntry {
                    parent: None,
                    residual_cols: cols.to_vec(),
                });
            }
        }

        let original_weight: usize = (0..n).map(|m| self.row_weight(m)).sum();
        let dag_weight: usize = entries.iter().map(|e| e.residual_cols.len()).sum();

        XorDag {
            entries,
            original_weight,
            dag_weight,
        }
    }

    pub fn stats(&self) -> ParityStats {
        if self.num_rows == 0 {
            return ParityStats {
                min_weight: 0,
                max_weight: 0,
                mean_weight: 0.0,
                total_weight: 0,
                num_deterministic: 0,
            };
        }
        let mut min_w = usize::MAX;
        let mut max_w = 0usize;
        let mut total = 0usize;
        let mut num_det = 0usize;
        for r in 0..self.num_rows {
            let w = self.row_weight(r);
            min_w = min_w.min(w);
            max_w = max_w.max(w);
            total += w;
            if w == 0 {
                num_det += 1;
            }
        }
        ParityStats {
            min_weight: min_w,
            max_weight: max_w,
            mean_weight: total as f64 / self.num_rows as f64,
            total_weight: total,
            num_deterministic: num_det,
        }
    }

    pub fn find_blocks(&self, rank: usize) -> Option<Vec<Vec<usize>>> {
        if self.num_rows <= 1 || rank == 0 {
            return None;
        }

        let mut col_to_rows: Vec<Vec<usize>> = vec![Vec::new(); rank];
        for m in 0..self.num_rows {
            for &c in self.row_cols(m) {
                col_to_rows[c as usize].push(m);
            }
        }

        let mut parent: Vec<usize> = (0..self.num_rows).collect();
        let mut size: Vec<usize> = vec![1; self.num_rows];

        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut root = x;
            while parent[root] != root {
                root = parent[root];
            }
            let mut cur = x;
            while parent[cur] != root {
                let next = parent[cur];
                parent[cur] = root;
                cur = next;
            }
            root
        }

        fn union(parent: &mut [usize], size: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            if size[ra] < size[rb] {
                parent[ra] = rb;
                size[rb] += size[ra];
            } else {
                parent[rb] = ra;
                size[ra] += size[rb];
            }
        }

        for rows in &col_to_rows {
            for i in 1..rows.len() {
                union(&mut parent, &mut size, rows[0], rows[i]);
            }
        }

        let mut block_map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for m in 0..self.num_rows {
            let root = find(&mut parent, m);
            block_map.entry(root).or_default().push(m);
        }

        let blocks: Vec<Vec<usize>> = block_map.into_values().collect();
        if blocks.len() <= 1 {
            return None;
        }

        Some(blocks)
    }

    pub fn compile_detection_events(&self, pairs: &[(usize, usize)]) -> SparseParity {
        let num_events = pairs.len();
        let mut row_offsets = Vec::with_capacity(num_events + 1);
        let mut col_indices = Vec::new();

        for &(m_a, m_b) in pairs {
            row_offsets.push(col_indices.len() as u32);
            let cols_a = self.row_cols(m_a);
            let cols_b = self.row_cols(m_b);
            let sym_diff = symmetric_difference(cols_a, cols_b);
            col_indices.extend_from_slice(&sym_diff);
        }
        row_offsets.push(col_indices.len() as u32);

        SparseParity {
            col_indices,
            row_offsets,
            num_rows: num_events,
        }
    }
}

#[derive(Debug, Clone)]
pub struct XorDagEntry {
    pub parent: Option<usize>,
    pub residual_cols: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct XorDag {
    pub entries: Vec<XorDagEntry>,
    pub original_weight: usize,
    pub dag_weight: usize,
}

fn symmetric_difference_size(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                count += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    count + (a.len() - i) + (b.len() - j)
}

fn symmetric_difference(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

#[derive(Debug, Clone)]
pub struct ParityStats {
    pub min_weight: usize,
    pub max_weight: usize,
    pub mean_weight: f64,
    pub total_weight: usize,
    pub num_deterministic: usize,
}

pub struct ParityBlock {
    pub meas_indices: Vec<usize>,
    pub sparse: SparseParity,
    pub block_rank: usize,
    pub ref_bits_packed: Vec<u64>,
}

pub struct ParityBlocks {
    pub blocks: Vec<ParityBlock>,
}

impl ParityBlocks {
    pub(super) fn build(
        global_sparse: &SparseParity,
        block_meas: Vec<Vec<usize>>,
        rank: usize,
        ref_bits: &[u64],
    ) -> Self {
        let mut blocks = Vec::with_capacity(block_meas.len());
        for meas_indices in block_meas {
            let mut col_set: Vec<u32> = Vec::new();
            for &m in &meas_indices {
                for &c in global_sparse.row_cols(m) {
                    col_set.push(c);
                }
            }
            col_set.sort_unstable();
            col_set.dedup();
            let block_rank = col_set.len();

            let mut col_remap = vec![0u32; rank];
            for (new_idx, &old_idx) in col_set.iter().enumerate() {
                col_remap[old_idx as usize] = new_idx as u32;
            }

            let num_rows = meas_indices.len();
            let mut row_offsets = Vec::with_capacity(num_rows + 1);
            let mut col_indices = Vec::new();
            for &m in &meas_indices {
                row_offsets.push(col_indices.len() as u32);
                for &c in global_sparse.row_cols(m) {
                    col_indices.push(col_remap[c as usize]);
                }
            }
            row_offsets.push(col_indices.len() as u32);

            let sparse = SparseParity {
                col_indices,
                row_offsets,
                num_rows,
            };

            let ref_words = num_rows.div_ceil(64);
            let mut block_ref = vec![0u64; ref_words];
            for (local_m, &global_m) in meas_indices.iter().enumerate() {
                let ref_bit = (ref_bits[global_m / 64] >> (global_m % 64)) & 1;
                if ref_bit != 0 {
                    block_ref[local_m / 64] |= 1u64 << (local_m % 64);
                }
            }

            blocks.push(ParityBlock {
                meas_indices,
                sparse,
                block_rank,
                ref_bits_packed: block_ref,
            });
        }
        ParityBlocks { blocks }
    }
}

pub(super) fn row_weight(row: &[u64]) -> u32 {
    row.iter().map(|w| w.count_ones()).sum()
}

const MAX_MEASUREMENTS_FOR_DAG: usize = 2000;
const MIN_DAG_REDUCTION_PCT: usize = 20;
const MIN_MEAN_WEIGHT_FOR_DAG: usize = 3;

pub(super) fn build_xor_dag_if_useful(sparse: &SparseParity) -> Option<XorDag> {
    if sparse.num_rows <= 1 || sparse.num_rows > MAX_MEASUREMENTS_FOR_DAG {
        return None;
    }
    let stats = sparse.stats();
    if stats.mean_weight < MIN_MEAN_WEIGHT_FOR_DAG as f64 {
        return None;
    }
    let dag = sparse.build_xor_dag();
    if dag.original_weight == 0 {
        return None;
    }
    let saved = dag.original_weight - dag.dag_weight;
    let reduction_pct = 100 * saved / dag.original_weight;
    if reduction_pct >= MIN_DAG_REDUCTION_PCT {
        Some(dag)
    } else {
        None
    }
}

const MIN_BLOCKS_FOR_PARALLEL: usize = 2;
const MIN_BLOCK_MEASUREMENTS: usize = 2;

pub(super) fn build_parity_blocks_if_useful(
    sparse: &SparseParity,
    rank: usize,
    ref_bits: &[u64],
) -> Option<ParityBlocks> {
    let block_meas = sparse.find_blocks(rank)?;

    if block_meas.len() < MIN_BLOCKS_FOR_PARALLEL {
        return None;
    }
    if block_meas.iter().any(|b| b.len() < MIN_BLOCK_MEASUREMENTS) {
        return None;
    }

    Some(ParityBlocks::build(sparse, block_meas, rank, ref_bits))
}

const MAX_RANK_FOR_WEIGHT_MIN: usize = 500;
const MAX_WEIGHT_MIN_ROUNDS: usize = 5;

pub(super) fn minimize_flip_row_weight(flip_rows: &mut [Vec<u64>]) -> (usize, usize) {
    let rank = flip_rows.len();
    let total: usize = flip_rows.iter().map(|r| row_weight(r) as usize).sum();
    if rank <= 1 || rank > MAX_RANK_FOR_WEIGHT_MIN {
        return (total, total);
    }

    let before = total;
    let mut weights: Vec<u32> = flip_rows.iter().map(|r| row_weight(r)).collect();

    for _round in 0..MAX_WEIGHT_MIN_ROUNDS {
        let mut improved = false;
        for i in 0..rank {
            let wi = weights[i];
            if wi == 0 {
                continue;
            }
            let mut best_j = usize::MAX;
            let mut best_w = wi;
            for j in 0..rank {
                if j == i {
                    continue;
                }
                let xor_w: u32 = flip_rows[i]
                    .iter()
                    .zip(flip_rows[j].iter())
                    .map(|(&a, &b)| (a ^ b).count_ones())
                    .sum();
                if xor_w < best_w {
                    best_w = xor_w;
                    best_j = j;
                }
            }
            if best_j != usize::MAX {
                for w in 0..flip_rows[i].len() {
                    flip_rows[i][w] ^= flip_rows[best_j][w];
                }
                weights[i] = best_w;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    let after: usize = weights.iter().map(|&w| w as usize).sum();
    (before, after)
}
