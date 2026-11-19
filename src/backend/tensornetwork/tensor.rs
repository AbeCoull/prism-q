//! Dense tensors, the buffer pool and the transpose and contraction kernels.

use std::borrow::Cow;

use num_complex::Complex64;
use smallvec::SmallVec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::backend::MIN_PAR_ELEMS;

use super::LegId;

/// `m*k*n` at or above which a contraction goes to faer instead of the scalar
/// loop below.
///
/// A 64-cubed complex product. Routing everything to faer instead costs 23% on
/// `tn/scalar_depth_20q/4`, where the operands are too small to cover its
/// packing.
#[cfg(feature = "parallel")]
const MIN_FAER_GEMM_WORK: usize = 1 << 18;

/// Dense multidimensional tensor with named legs for contraction.
///
/// Legs with matching `LegId` across two tensors are contracted (summed over)
/// when those tensors are pairwise contracted.
#[derive(Clone, Debug)]
pub(super) struct Tensor {
    pub(super) data: Vec<Complex64>,
    pub(super) shape: SmallVec<[usize; 6]>,
    pub(super) legs: SmallVec<[LegId; 6]>,
}

impl Tensor {
    pub(super) fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub(super) fn rank(&self) -> usize {
        self.legs.len()
    }
}

/// Elements from which a freed buffer is worth holding for reuse, 512 KiB. Smaller
/// ones the allocator recycles without faulting in new pages.
const POOL_MIN_ELEMENTS: usize = 1 << 15;

/// Freed buffers a [`BufferPool`] holds at once: the two operands and two
/// transposes one contraction step releases.
const POOL_BUFFERS: usize = 4;

/// Large tensor buffers one plan replay frees and reuses.
///
/// A fresh buffer past the allocator's mapping threshold faults in every page on
/// first write, and each contraction step of a replay allocates and frees a few
/// of these. Recycling them keeps the replay on pages it has already touched.
/// The pool holds only what earlier steps released, so resident memory stays at
/// the largest single step instead of dropping between steps.
#[derive(Default)]
pub(super) struct BufferPool {
    free: Vec<Vec<Complex64>>,
}

impl BufferPool {
    /// A buffer of `len` elements whose contents the caller overwrites.
    fn take(&mut self, len: usize) -> Vec<Complex64> {
        self.reuse(len)
            .unwrap_or_else(|| vec![Complex64::new(0.0, 0.0); len])
    }

    fn take_zeroed(&mut self, len: usize) -> Vec<Complex64> {
        let zero = Complex64::new(0.0, 0.0);
        match self.reuse(len) {
            Some(mut buf) => {
                buf.fill(zero);
                buf
            }
            None => vec![zero; len],
        }
    }

    /// The smallest held buffer of at least `len` elements, cut to `len`.
    fn reuse(&mut self, len: usize) -> Option<Vec<Complex64>> {
        if len < POOL_MIN_ELEMENTS {
            return None;
        }
        let (index, _) = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, buf)| buf.len() >= len)
            .min_by_key(|(_, buf)| buf.len())?;
        let mut buf = self.free.swap_remove(index);
        buf.truncate(len);
        Some(buf)
    }

    /// Hold `buf` for reuse, displacing the smallest held buffer when full.
    pub(super) fn give(&mut self, buf: Vec<Complex64>) {
        if buf.len() < POOL_MIN_ELEMENTS {
            return;
        }
        if self.free.len() == POOL_BUFFERS {
            let (index, smallest) = self
                .free
                .iter()
                .enumerate()
                .min_by_key(|(_, held)| held.len())
                .expect("a full pool holds buffers");
            if smallest.len() >= buf.len() {
                return;
            }
            self.free.swap_remove(index);
        }
        self.free.push(buf);
    }
}

/// Extent and source stride of each output axis, in output order.
type Axes = SmallVec<[(usize, usize); 6]>;

/// Source elements a tiled transpose reads per contiguous run, two cache lines.
const TRANSPOSE_TILE: usize = 8;

/// Elements from which a strided transpose tiles its reads: twice the tile budget,
/// 512 KiB on the x86 cores measured, past a private L2.
#[inline(always)]
fn transpose_tile_min() -> usize {
    2 * crate::backend::cache::tile_budget_bytes() / size_of::<Complex64>()
}

/// Output axes of `perm` over a tensor of `shape`, with neighbours that stay
/// adjacent in the source merged into one axis and unit extents dropped.
///
/// Merging keeps every walk below on the fewest axes: a permutation that moves
/// a few legs of a rank-20 tensor becomes a handful of long runs.
fn fused_axes(shape: &[usize], perm: &[usize]) -> Axes {
    let mut strides: SmallVec<[usize; 6]> = SmallVec::from_elem(0, shape.len());
    let mut stride = 1;
    for (axis, &extent) in shape.iter().enumerate().rev() {
        strides[axis] = stride;
        stride *= extent;
    }
    let mut axes = Axes::new();
    for &old in perm {
        let (extent, stride) = (shape[old], strides[old]);
        if extent == 1 {
            continue;
        }
        match axes.last_mut() {
            Some((outer_extent, outer_stride)) if *outer_stride == stride * extent => {
                *outer_extent *= extent;
                *outer_stride = stride;
            }
            _ => axes.push((extent, stride)),
        }
    }
    if axes.is_empty() {
        axes.push((1, 1));
    }
    axes
}

/// Source offset of a row-major walk over `axes`.
struct Odometer<'a> {
    axes: &'a [(usize, usize)],
    counter: SmallVec<[usize; 6]>,
    offset: usize,
}

impl<'a> Odometer<'a> {
    /// Start the walk at row-major index `start`; only a nonzero start divides.
    fn new(axes: &'a [(usize, usize)], start: usize) -> Self {
        let mut counter: SmallVec<[usize; 6]> = SmallVec::from_elem(0, axes.len());
        let mut offset = 0;
        let mut rem = start;
        if start != 0 {
            for (slot, &(extent, stride)) in counter.iter_mut().zip(axes).rev() {
                *slot = rem % extent;
                rem /= extent;
                offset += *slot * stride;
            }
        }
        Self {
            axes,
            counter,
            offset,
        }
    }

    #[inline(always)]
    fn advance(&mut self) {
        for (slot, &(extent, stride)) in self.counter.iter_mut().zip(self.axes).rev() {
            *slot += 1;
            self.offset += stride;
            if *slot < extent {
                return;
            }
            *slot = 0;
            self.offset -= stride * extent;
        }
    }
}

/// Write `out`, the output elements from index `start`, as whole runs of the
/// innermost axis, which the source holds contiguously.
fn transpose_runs(out: &mut [Complex64], src: &[Complex64], axes: &[(usize, usize)], start: usize) {
    let (run, _) = axes[axes.len() - 1];
    let mut outer = Odometer::new(&axes[..axes.len() - 1], start / run);
    for dst in out.chunks_exact_mut(run) {
        dst.copy_from_slice(&src[outer.offset..outer.offset + run]);
        outer.advance();
    }
}

/// Write `out`, [`TRANSPOSE_TILE`] consecutive indices of the source-contiguous
/// axis `b` starting at output index `start`.
///
/// Output index is `prefix | b | mid | inner`. Walking `b` innermost reads whole
/// cache lines of the source and writes the tile's output rows in step, where the
/// plain output-order walk reads one element per line on a strided inner axis.
fn transpose_tile(
    out: &mut [Complex64],
    src: &[Complex64],
    axes: &[(usize, usize)],
    b: usize,
    start: usize,
) {
    let (inner_extent, inner_stride) = axes[axes.len() - 1];
    let mid = &axes[b + 1..axes.len() - 1];
    let row: usize = axes[b + 1..].iter().map(|&(extent, _)| extent).product();
    let (b_extent, _) = axes[b];
    let first = start / row;
    let base = Odometer::new(&axes[..b], first / b_extent).offset + first % b_extent;

    let mut walk = Odometer::new(mid, 0);
    for mid_out in (0..row).step_by(inner_extent) {
        let mut from = base + walk.offset;
        for a in 0..inner_extent {
            let lane = &src[from..from + TRANSPOSE_TILE];
            for (t, &value) in lane.iter().enumerate() {
                out[t * row + mid_out + a] = value;
            }
            from += inner_stride;
        }
        walk.advance();
    }
}

/// Write `out`, the output elements from index `start`, one at a time.
fn transpose_elements(
    out: &mut [Complex64],
    src: &[Complex64],
    axes: &[(usize, usize)],
    start: usize,
) {
    let mut walk = Odometer::new(axes, start);
    for slot in out {
        *slot = src[walk.offset];
        walk.advance();
    }
}

/// Fill `out` with `src` read along `axes`.
///
/// Runs of at least [`TRANSPOSE_TILE`] copy whole. A strided inner axis on a
/// tensor past [`transpose_tile_min`] goes through [`transpose_tile`]; below it
/// the source sits in cache and the element walk has less setup per element.
fn transpose_into(out: &mut [Complex64], src: &[Complex64], axes: &[(usize, usize)]) {
    let (inner_extent, inner_stride) = axes[axes.len() - 1];
    if inner_stride == 1 && inner_extent >= TRANSPOSE_TILE {
        let chunk = inner_extent * (MIN_TRANSPOSE_TASK / inner_extent).max(1);
        return for_each_chunk(out, chunk, |start, dst| {
            transpose_runs(dst, src, axes, start)
        });
    }
    if inner_stride != 1 && out.len() >= transpose_tile_min() {
        let b = axes
            .iter()
            .position(|&(_, stride)| stride == 1)
            .expect("the source's last axis has unit stride");
        let (b_extent, _) = axes[b];
        if b_extent % TRANSPOSE_TILE == 0 {
            let row: usize = axes[b + 1..].iter().map(|&(extent, _)| extent).product();
            let part = TRANSPOSE_TILE * row;
            let per_task = (MIN_TRANSPOSE_TASK / part).max(1);
            let chunk = part * gcd(b_extent / TRANSPOSE_TILE, per_task);
            return for_each_chunk(out, chunk, |start, dst| {
                for (index, piece) in dst.chunks_exact_mut(part).enumerate() {
                    transpose_tile(piece, src, axes, b, start + index * part);
                }
            });
        }
    }
    for_each_chunk(out, MIN_TRANSPOSE_TASK, |start, dst| {
        transpose_elements(dst, src, axes, start)
    })
}

/// Output elements below which a transpose task is not worth a Rayon split.
#[cfg(feature = "parallel")]
const MIN_TRANSPOSE_TASK: usize = MIN_PAR_ELEMS;
#[cfg(not(feature = "parallel"))]
const MIN_TRANSPOSE_TASK: usize = usize::MAX;

/// Run `body` over `out` in `chunk`-sized pieces with each piece's start index,
/// in parallel when there is more than one.
fn for_each_chunk(
    out: &mut [Complex64],
    chunk: usize,
    body: impl Fn(usize, &mut [Complex64]) + Sync,
) {
    #[cfg(feature = "parallel")]
    if out.len() > chunk {
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(index, dst)| body(index * chunk, dst));
        return;
    }
    let _ = chunk;
    body(0, out);
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Transpose a tensor by permuting its axes.
///
/// `perm[new_axis] = old_axis`. The output tensor has shape
/// `[input.shape[perm[0]], input.shape[perm[1]], ...]`.
pub(super) fn transpose(t: &Tensor, perm: &[usize]) -> Tensor {
    transpose_with(t, perm, &mut BufferPool::default())
}

/// [`transpose`] into a buffer from `pool`.
fn transpose_with(t: &Tensor, perm: &[usize], pool: &mut BufferPool) -> Tensor {
    debug_assert_eq!(perm.len(), t.rank());
    let shape: SmallVec<[usize; 6]> = perm.iter().map(|&p| t.shape[p]).collect();
    let legs: SmallVec<[LegId; 6]> = perm.iter().map(|&p| t.legs[p]).collect();
    let mut data = pool.take(t.num_elements());
    transpose_into(&mut data, &t.data, &fused_axes(&t.shape, perm));
    Tensor { data, shape, legs }
}

/// Multiply the row-major `m` by `k` and `k` by `n` operands into `c`.
///
/// A row-major array is its own transpose read column-major, so `C = A*B` is
/// issued as `C^T = B^T * A^T` over the same buffers with no repacking.
#[cfg(feature = "parallel")]
fn faer_gemm(a: &[Complex64], b: &[Complex64], c: &mut [Complex64], m: usize, k: usize, n: usize) {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef, Par};

    matmul(
        MatMut::from_column_major_slice_mut(c, n, m),
        Accum::Replace,
        MatRef::from_column_major_slice(b, n, k),
        MatRef::from_column_major_slice(a, k, m),
        Complex64::new(1.0, 0.0),
        Par::rayon(0),
    );
}

/// Contract two tensors over shared legs (matching LegId).
///
/// Standard tensordot: find shared legs, reshape both to 2D matrices,
/// multiply, reshape result.
pub(super) fn contract(a: &Tensor, b: &Tensor) -> Tensor {
    contract_with(a, b, &mut BufferPool::default())
}

/// [`contract`] with its transposes and result drawn from `pool`, and the
/// transposes handed back to it.
pub(super) fn contract_with(a: &Tensor, b: &Tensor, pool: &mut BufferPool) -> Tensor {
    let mut a_shared: SmallVec<[usize; 4]> = SmallVec::new();
    let mut b_shared: SmallVec<[usize; 4]> = SmallVec::new();
    for (ai, &a_leg) in a.legs.iter().enumerate() {
        for (bi, &b_leg) in b.legs.iter().enumerate() {
            if a_leg == b_leg {
                a_shared.push(ai);
                b_shared.push(bi);
            }
        }
    }

    let a_free: SmallVec<[usize; 6]> = (0..a.rank()).filter(|i| !a_shared.contains(i)).collect();
    let b_free: SmallVec<[usize; 6]> = (0..b.rank()).filter(|i| !b_shared.contains(i)).collect();

    let mut a_perm: SmallVec<[usize; 6]> = SmallVec::new();
    a_perm.extend_from_slice(&a_free);
    a_perm.extend_from_slice(&a_shared);

    let mut b_perm: SmallVec<[usize; 6]> = SmallVec::new();
    b_perm.extend_from_slice(&b_shared);
    b_perm.extend_from_slice(&b_free);

    let a_t = if a_perm.iter().enumerate().all(|(i, &p)| i == p) {
        Cow::Borrowed(a)
    } else {
        Cow::Owned(transpose_with(a, &a_perm, pool))
    };

    let b_t = if b_perm.iter().enumerate().all(|(i, &p)| i == p) {
        Cow::Borrowed(b)
    } else {
        Cow::Owned(transpose_with(b, &b_perm, pool))
    };

    let m: usize = a_free.iter().map(|&i| a.shape[i]).product::<usize>().max(1);
    let k: usize = a_shared
        .iter()
        .map(|&i| a.shape[i])
        .product::<usize>()
        .max(1);
    let n: usize = b_free.iter().map(|&i| b.shape[i]).product::<usize>().max(1);

    let zero = Complex64::new(0.0, 0.0);
    #[cfg(feature = "parallel")]
    let faer = m * k * n >= MIN_FAER_GEMM_WORK;
    #[cfg(not(feature = "parallel"))]
    let faer = false;
    let mut c_data = if faer {
        pool.take(m * n)
    } else {
        pool.take_zeroed(m * n)
    };

    #[cfg(feature = "parallel")]
    if faer {
        faer_gemm(&a_t.data, &b_t.data, &mut c_data, m, k, n);
    } else if m * n >= MIN_PAR_ELEMS {
        let a_data = &a_t.data;
        let b_data = &b_t.data;
        c_data.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            for j in 0..k {
                let a_val = a_data[i * k + j];
                if a_val == zero {
                    continue;
                }
                let b_row = &b_data[j * n..(j + 1) * n];
                for (c_elem, &b_val) in c_row.iter_mut().zip(b_row) {
                    *c_elem += a_val * b_val;
                }
            }
        });
    } else {
        for i in 0..m {
            for j in 0..k {
                let a_val = a_t.data[i * k + j];
                if a_val == zero {
                    continue;
                }
                let b_row = &b_t.data[j * n..(j + 1) * n];
                let c_row = &mut c_data[i * n..(i + 1) * n];
                for (c_elem, &b_val) in c_row.iter_mut().zip(b_row) {
                    *c_elem += a_val * b_val;
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    for i in 0..m {
        for j in 0..k {
            let a_val = a_t.data[i * k + j];
            if a_val == zero {
                continue;
            }
            let b_row = &b_t.data[j * n..(j + 1) * n];
            let c_row = &mut c_data[i * n..(i + 1) * n];
            for (c_elem, &b_val) in c_row.iter_mut().zip(b_row) {
                *c_elem += a_val * b_val;
            }
        }
    }

    if let Cow::Owned(t) = a_t {
        pool.give(t.data);
    }
    if let Cow::Owned(t) = b_t {
        pool.give(t.data);
    }

    let mut result_shape: SmallVec<[usize; 6]> = SmallVec::new();
    let mut result_legs: SmallVec<[LegId; 6]> = SmallVec::new();
    for &i in &a_free {
        result_shape.push(a.shape[i]);
        result_legs.push(a.legs[i]);
    }
    for &i in &b_free {
        result_shape.push(b.shape[i]);
        result_legs.push(b.legs[i]);
    }

    if result_shape.is_empty() {
        result_shape.push(1);
    }

    Tensor {
        data: c_data,
        shape: result_shape,
        legs: result_legs,
    }
}
