//! Sparse parity structures for compiled sampling: the GF(2) system mapping
//! random bits to measurement flips, plus derived evaluation orders.

/// CSR matrix over GF(2) mapping random-bit columns to measurement rows.
#[derive(Debug, Clone)]
pub struct SparseParity {
    pub col_indices: Vec<u32>,
    pub row_offsets: Vec<u32>,
    pub num_rows: usize,
    pub non_det_rows: Vec<u32>,
}

impl SparseParity {
    pub fn from_flip_rows(flip_rows: &[Vec<u64>], num_measurements: usize) -> Self {
        let num_rows = num_measurements;
        let m_words = num_rows.div_ceil(64);
        let tail_mask = match num_rows % 64 {
            0 => u64::MAX,
            tail => (1u64 << tail) - 1,
        };
        let mut row_offsets = vec![0u32; num_rows + 1];
        for row in flip_rows {
            for (w, &word) in row[..m_words].iter().enumerate() {
                let mut bits = if w + 1 == m_words {
                    word & tail_mask
                } else {
                    word
                };
                while bits != 0 {
                    row_offsets[w * 64 + bits.trailing_zeros() as usize + 1] += 1;
                    bits &= bits - 1;
                }
            }
        }
        for m in 0..num_rows {
            row_offsets[m + 1] += row_offsets[m];
        }
        let mut col_indices = vec![0u32; row_offsets[num_rows] as usize];
        let mut cursor = row_offsets[..num_rows].to_vec();
        for (j, row) in flip_rows.iter().enumerate() {
            for (w, &word) in row[..m_words].iter().enumerate() {
                let mut bits = if w + 1 == m_words {
                    word & tail_mask
                } else {
                    word
                };
                while bits != 0 {
                    let m = w * 64 + bits.trailing_zeros() as usize;
                    col_indices[cursor[m] as usize] = j as u32;
                    cursor[m] += 1;
                    bits &= bits - 1;
                }
            }
        }

        let non_det_rows: Vec<u32> = (0..num_rows as u32)
            .filter(|&m| row_offsets[m as usize + 1] != row_offsets[m as usize])
            .collect();

        Self {
            col_indices,
            row_offsets,
            num_rows,
            non_det_rows,
        }
    }

    #[cfg(test)]
    pub(super) fn from_flip_rows_by_probe(flip_rows: &[Vec<u64>], num_measurements: usize) -> Self {
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

        let non_det_rows: Vec<u32> = (0..num_rows as u32)
            .filter(|&m| row_offsets[m as usize + 1] != row_offsets[m as usize])
            .collect();

        Self {
            col_indices,
            row_offsets,
            num_rows,
            non_det_rows,
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
        let mut parent: Vec<usize> = (0..self.num_rows).collect();
        let mut size: Vec<usize> = vec![1; self.num_rows];
        let mut components = self.num_rows;
        let mut first_row = vec![usize::MAX; rank];
        for m in 0..self.num_rows {
            for &c in self.row_cols(m) {
                let first = &mut first_row[c as usize];
                if *first == usize::MAX {
                    *first = m;
                } else if union_rows(&mut parent, &mut size, *first, m) {
                    components -= 1;
                    if components == 1 {
                        return None;
                    }
                }
            }
        }

        blocks_by_root(&mut parent)
    }

    #[cfg(test)]
    pub(super) fn find_blocks_by_column_lists(&self, rank: usize) -> Option<Vec<Vec<usize>>> {
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
        for rows in &col_to_rows {
            for i in 1..rows.len() {
                union_rows(&mut parent, &mut size, rows[0], rows[i]);
            }
        }

        blocks_by_root(&mut parent)
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

        let non_det_rows: Vec<u32> = (0..num_events as u32)
            .filter(|&m| row_offsets[m as usize + 1] != row_offsets[m as usize])
            .collect();

        SparseParity {
            col_indices,
            row_offsets,
            num_rows: num_events,
            non_det_rows,
        }
    }
}

fn find_root(parent: &mut [usize], x: usize) -> usize {
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

/// Join the sets holding `a` and `b`; false when they already share one.
fn union_rows(parent: &mut [usize], size: &mut [usize], a: usize, b: usize) -> bool {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    if ra == rb {
        return false;
    }
    if size[ra] < size[rb] {
        parent[ra] = rb;
        size[rb] += size[ra];
    } else {
        parent[rb] = ra;
        size[ra] += size[rb];
    }
    true
}

/// Rows grouped by set, blocks ordered by their first row; `None` for a single set.
fn blocks_by_root(parent: &mut [usize]) -> Option<Vec<Vec<usize>>> {
    let num_rows = parent.len();
    let mut block_of_root = vec![usize::MAX; num_rows];
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for m in 0..num_rows {
        let root = find_root(parent, m);
        if block_of_root[root] == usize::MAX {
            block_of_root[root] = blocks.len();
            blocks.push(Vec::new());
        }
        blocks[block_of_root[root]].push(m);
    }
    (blocks.len() > 1).then_some(blocks)
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

/// Row-weight statistics of a parity matrix; a row's weight is the number of
/// random bits whose XOR determines that measurement.
#[derive(Debug, Clone)]
pub struct ParityStats {
    pub min_weight: usize,
    pub max_weight: usize,
    pub mean_weight: f64,
    pub total_weight: usize,
    /// Rows with zero weight (fixed outcomes).
    pub num_deterministic: usize,
}

#[derive(Debug, Clone)]
pub struct ParityBlock {
    pub meas_indices: Vec<usize>,
    pub sparse: SparseParity,
    pub block_rank: usize,
    pub ref_bits_packed: Vec<u64>,
}

/// Measurement blocks sharing no random bits, sampled independently in
/// parallel.
#[derive(Debug, Clone)]
pub struct ParityBlocks {
    pub blocks: Vec<ParityBlock>,
    pub direct_scatter: bool,
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

            let non_det_rows: Vec<u32> = (0..num_rows as u32)
                .filter(|&m| row_offsets[m as usize + 1] != row_offsets[m as usize])
                .collect();

            let sparse = SparseParity {
                col_indices,
                row_offsets,
                num_rows,
                non_det_rows,
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
        ParityBlocks {
            blocks,
            direct_scatter: false,
        }
    }

    pub(super) fn from_blocks_if_useful(blocks: Vec<ParityBlock>) -> Option<Self> {
        if blocks.len() < MIN_BLOCKS_FOR_PARALLEL {
            return None;
        }
        if blocks
            .iter()
            .any(|block| block.meas_indices.len() < MIN_BLOCK_MEASUREMENTS)
        {
            return None;
        }
        Some(Self {
            blocks,
            direct_scatter: false,
        })
    }
}

pub(super) fn row_weight(row: &[u64]) -> u32 {
    row.iter().map(|w| w.count_ones()).sum()
}

const MIN_BLOCKS_FOR_PARALLEL: usize = 2;
const MIN_BLOCK_MEASUREMENTS: usize = 2;

pub(super) fn build_parity_blocks_if_useful(
    sparse: &SparseParity,
    rank: usize,
    ref_bits: &[u64],
) -> Option<ParityBlocks> {
    // A fixed outcome is a block of one row, which MIN_BLOCK_MEASUREMENTS rejects.
    if sparse.non_det_rows.len() < sparse.num_rows {
        return None;
    }
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

/// Greedy basis change: XOR one flip row into another when that lowers its
/// weight. Elementary row operations preserve the sampled distribution.
/// Returns total weight (before, after).
pub(super) fn minimize_flip_row_weight(flip_rows: &mut [Vec<u64>]) -> (usize, usize) {
    let rank = flip_rows.len();
    let total: usize = flip_rows.iter().map(|r| row_weight(r) as usize).sum();
    if rank <= 1 || rank > MAX_RANK_FOR_WEIGHT_MIN {
        return (total, total);
    }
    let words = flip_rows[0].len();
    let mut flat: Vec<u64> = Vec::with_capacity(rank * words);
    for row in flip_rows.iter() {
        flat.extend_from_slice(row);
    }
    let after = match words {
        1 => minimize_flat::<1>(&mut flat, rank),
        2 => minimize_flat::<2>(&mut flat, rank),
        3 => minimize_flat::<3>(&mut flat, rank),
        4 => minimize_flat::<4>(&mut flat, rank),
        _ => minimize_flat_dyn(&mut flat, rank, words),
    };
    for (row, src) in flip_rows.iter_mut().zip(flat.chunks_exact(words)) {
        row.copy_from_slice(src);
    }
    (total, after)
}

/// [`minimize_flip_row_weight`] over `rank` contiguous rows of `W` words, so the
/// inner scan reads one flat slice instead of chasing a pointer per row.
fn minimize_flat<const W: usize>(flat: &mut [u64], rank: usize) -> usize {
    let rows = flat.as_chunks_mut::<W>().0;
    let mut weights: Vec<u32> = rows.iter().map(|r| row_weight(r)).collect();
    for _round in 0..MAX_WEIGHT_MIN_ROUNDS {
        let mut improved = false;
        for i in 0..rank {
            let wi = weights[i];
            if wi == 0 {
                continue;
            }
            let ri = rows[i];
            let mut best_j = usize::MAX;
            let mut best_w = wi;
            for (j, rj) in rows.iter().enumerate() {
                let mut xor_w = 0u32;
                for w in 0..W {
                    xor_w += (ri[w] ^ rj[w]).count_ones();
                }
                if xor_w < best_w && j != i {
                    best_w = xor_w;
                    best_j = j;
                }
            }
            if best_j != usize::MAX {
                let rj = rows[best_j];
                for w in 0..W {
                    rows[i][w] ^= rj[w];
                }
                weights[i] = best_w;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    weights.iter().map(|&w| w as usize).sum()
}

fn minimize_flat_dyn(flat: &mut [u64], rank: usize, words: usize) -> usize {
    let mut weights: Vec<u32> = flat.chunks_exact(words).map(row_weight).collect();
    for _round in 0..MAX_WEIGHT_MIN_ROUNDS {
        let mut improved = false;
        for i in 0..rank {
            let wi = weights[i];
            if wi == 0 {
                continue;
            }
            let ri = &flat[i * words..(i + 1) * words];
            let mut best_j = usize::MAX;
            let mut best_w = wi;
            for (j, rj) in flat.chunks_exact(words).enumerate() {
                let xor_w: u32 = ri.iter().zip(rj).map(|(&a, &b)| (a ^ b).count_ones()).sum();
                if xor_w < best_w && j != i {
                    best_w = xor_w;
                    best_j = j;
                }
            }
            if best_j != usize::MAX {
                for w in 0..words {
                    flat[i * words + w] ^= flat[best_j * words + w];
                }
                weights[i] = best_w;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    weights.iter().map(|&w| w as usize).sum()
}
