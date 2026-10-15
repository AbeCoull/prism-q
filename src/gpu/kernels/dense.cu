
// Template constants substituted at device construction from the Rust constants.
#define TILE_Q          {{TILE_Q}}
#define TILE_SIZE       {{TILE_SIZE}}
#define BP_TABLE_SIZE   {{BP_TABLE_SIZE}}
#define BP_GROUP_SIZE   {{BP_GROUP_SIZE}}
#define BR_TABLE_SIZE   {{BR_TABLE_SIZE}}
#define BR_GROUP_SIZE   {{BR_GROUP_SIZE}}
#define DB_TABLE_SIZE   {{DB_TABLE_SIZE}}
#define DB_MAX_QUBITS   {{DB_MAX_QUBITS}}
#define BP_MAX_GROUPS   {{BP_MAX_GROUPS}}
#define PARAM_QUBITS    {{PARAM_QUBITS}}
#define BR_MAX_GROUPS   {{BR_MAX_GROUPS}}
#define DB_MAX_GROUPS   {{DB_MAX_GROUPS}}
#define QFT_TILE_Q      {{QFT_TILE_Q}}
#define MIX_TILE_Q      {{MIX_TILE_Q}}
#define MIX_BLOCK_LOG   {{MIX_BLOCK_LOG}}
#define MIX_MAX_OPS     {{MIX_MAX_OPS}}
#define MIX_MAX_COEFS   {{MIX_MAX_COEFS}}

// The fused 2q matrix travels in kernel parameter space: a pageable host to device
// copy synchronizes the stream before it starts, so uploading the matrix per gate
// serialized the host against every queued kernel. Entry m = a + ib is stored as
// (a, b - a, a + b), row-major. Layout mirrors `FusedMatArg`.
struct FusedMat { double v[48]; };
struct MonoScale { double v[8]; };
struct QubitList { int q[PARAM_QUBITS]; };
struct DiagList { double d[4 * PARAM_QUBITS]; int t[PARAM_QUBITS]; };
struct TileGates { double v[8 * TILE_Q]; unsigned long long qubits; unsigned long long targets; };
struct MixedOps { double c[MIX_MAX_COEFS]; unsigned int op[MIX_MAX_OPS]; unsigned char q[16]; int num_ops; };

// ============================================================================
// Shared device helpers
// ============================================================================

__device__ __forceinline__ void apply2x2(double2 *state,
    unsigned long long i0, unsigned long long i1,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    double2 a = state[i0];
    double2 b = state[i1];
    state[i0].x = m00r*a.x - m00i*a.y + m01r*b.x - m01i*b.y;
    state[i0].y = m00r*a.y + m00i*a.x + m01r*b.y + m01i*b.x;
    state[i1].x = m10r*a.x - m10i*a.y + m11r*b.x - m11i*b.y;
    state[i1].y = m10r*a.y + m10i*a.x + m11r*b.y + m11i*b.x;
}

__device__ __forceinline__ void apply_phase(double2 *state, unsigned long long i, double pr, double pi)
{
    double2 a = state[i];
    state[i].x = pr*a.x - pi*a.y;
    state[i].y = pr*a.y + pi*a.x;
}

__device__ __forceinline__ double2 cmul(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ============================================================================
// Initialisation
// ============================================================================

extern "C" __global__ void set_initial_state(double2 *state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        state[0] = make_double2(1.0, 0.0);
    }
}

// ============================================================================
// Single-qubit gate: generic 2x2
// ============================================================================
//
// Launch: 2^(n-1) threads. Each thread handles one (lo, hi) amplitude pair.

extern "C" __global__ void apply_gate_1q(
    double2 *state, unsigned long long pair_count, int target,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;

    unsigned long long mask = (1ULL << target) - 1;
    unsigned long long i0 = ((k & ~mask) << 1) | (k & mask);
    unsigned long long i1 = i0 | (1ULL << target);

    apply2x2(state, i0, i1, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i);
}

// Diagonal 2x2 specialisation.
// state[i0] *= d0, state[i1] *= d1, no cross terms.

extern "C" __global__ void apply_diagonal_1q(
    double2 *state, unsigned long long pair_count, int target,
    double d0r, double d0i, double d1r, double d1i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;

    unsigned long long mask = (1ULL << target) - 1;
    unsigned long long i0 = ((k & ~mask) << 1) | (k & mask);
    unsigned long long i1 = i0 | (1ULL << target);

    apply_phase(state, i0, d0r, d0i);
    apply_phase(state, i1, d1r, d1i);
}

// ============================================================================
// Two-qubit gates (CX / CZ / SWAP)
// ============================================================================
//
// All take `pair_count = 2^(n-2)` threads. Each thread computes a compressed index
// and expands via chained insert_zero_bit (q0, q1 in either order).

__device__ __forceinline__ unsigned long long expand_2q(unsigned long long k, int q0, int q1) {
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long lo_mask = (1ULL << lo_q) - 1;
    unsigned long long lo = k & lo_mask;
    unsigned long long mid_hi = k >> lo_q;
    mid_hi = (mid_hi << 1);                                 // insert 0 at lo_q
    unsigned long long base = (mid_hi << lo_q) | lo;        // reassemble with gap at lo_q
    // second insertion at hi_q (already accounts for +1 shift from first)
    unsigned long long hi_mask = (1ULL << hi_q) - 1;
    unsigned long long low = base & hi_mask;
    unsigned long long high = base >> hi_q;
    return (high << (hi_q + 1)) | low;
}

extern "C" __global__ void apply_cx(
    double2 *state, unsigned long long pair_count, int control, int target)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    unsigned long long idx = expand_2q(k, control, target);
    unsigned long long i0 = idx | (1ULL << control);
    unsigned long long i1 = i0 | (1ULL << target);
    double2 tmp = state[i0];
    state[i0] = state[i1];
    state[i1] = tmp;
}

extern "C" __global__ void apply_cz(
    double2 *state, unsigned long long pair_count, int q0, int q1)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    unsigned long long idx = expand_2q(k, q0, q1);
    unsigned long long i = idx | (1ULL << q0) | (1ULL << q1);
    state[i].x = -state[i].x;
    state[i].y = -state[i].y;
}

extern "C" __global__ void apply_swap(
    double2 *state, unsigned long long pair_count, int q0, int q1)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    unsigned long long idx = expand_2q(k, q0, q1);
    unsigned long long i01 = idx | (1ULL << q0);
    unsigned long long i10 = idx | (1ULL << q1);
    double2 tmp = state[i01];
    state[i01] = state[i10];
    state[i10] = tmp;
}

// Parity-dependent phase (Rzz and DiagEntry::Parity2q).
// state[i] *= same when ((i>>q0) ^ (i>>q1)) & 1 == 0, else state[i] *= diff.
// Launch over 2^n threads.

extern "C" __global__ void apply_parity_phase(
    double2 *state, unsigned long long dim, int q0, int q1,
    double same_r, double same_i, double diff_r, double diff_i)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    unsigned long long parity = ((i >> q0) ^ (i >> q1)) & 1ULL;
    double pr = parity ? diff_r : same_r;
    double pi = parity ? diff_i : same_i;
    apply_phase(state, i, pr, pi);
}

// ============================================================================
// Controlled-unitary gates
// ============================================================================

extern "C" __global__ void apply_cu(
    double2 *state, unsigned long long pair_count, int control, int target,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    unsigned long long idx = expand_2q(k, control, target);
    unsigned long long i0 = idx | (1ULL << control);
    unsigned long long i1 = i0 | (1ULL << target);

    apply2x2(state, i0, i1, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i);
}

// Controlled-phase optimisation: state[both_set] *= phase.
// Launch over pair_count = 2^(n-2) threads. Only acts on ctrl=1, tgt=1.

extern "C" __global__ void apply_cu_phase(
    double2 *state, unsigned long long pair_count, int control, int target,
    double pr, double pi)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    unsigned long long idx = expand_2q(k, control, target);
    unsigned long long i = idx | (1ULL << control) | (1ULL << target);
    apply_phase(state, i, pr, pi);
}

// Insert a zero bit at each of the `count` ascending positions in `sorted`.
__device__ __forceinline__ unsigned long long insert_zero_bits(
    unsigned long long idx, const int *sorted, int count)
{
    for (int i = 0; i < count; ++i) {
        int bit = sorted[i];
        unsigned long long mask_lo = (1ULL << bit) - 1;
        unsigned long long lo = idx & mask_lo;
        unsigned long long hi = idx >> bit;
        idx = (hi << (bit + 1)) | lo;
    }
    return idx;
}

// Multi-controlled unitary. `sorted` contains all controls + target, sorted ascending.
// `num_sorted = num_controls + 1`. `ctrl_mask` has bits set at control positions only;
// `tgt_mask = 1 << target`.
// Launch over 2^(n - num_sorted) threads.

extern "C" __global__ void apply_mcu(
    double2 *state, unsigned long long iter_count,
    QubitList sorted, int num_sorted,
    unsigned long long ctrl_mask, unsigned long long tgt_mask,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= iter_count) return;
    unsigned long long idx = insert_zero_bits(k, sorted.q, num_sorted);
    unsigned long long i0 = idx | ctrl_mask;
    unsigned long long i1 = i0 | tgt_mask;

    apply2x2(state, i0, i1, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i);
}

extern "C" __global__ void apply_mcu_phase(
    double2 *state, unsigned long long iter_count,
    QubitList sorted, int num_sorted,
    unsigned long long all_mask,
    double pr, double pi)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= iter_count) return;
    unsigned long long idx = insert_zero_bits(k, sorted.q, num_sorted);
    unsigned long long i = idx | all_mask;
    apply_phase(state, i, pr, pi);
}

// ============================================================================
// Fused 2q: generic 4x4 matrix.
// Launch over pair_count = 2^(n-2) threads; each processes one 4-element group.
// ============================================================================
//
// Each complex product m * x is three multiply-adds in the Gauss form,
// re = a (x + y) - y (a + b) and im = a (x + y) + x (b - a), accumulated per row in
// three independent chains against x + y, x and y. The row costs 12 FMAs and two adds
// where `acc += a*x - b*y` compiles to a multiply, an FMA and an add per component,
// so the group issues 60 FP64 operations instead of 96. The multiply is FP64-issue
// bound on reduced-rate parts (2.33 ms at 24 qubits on a GTX 1080 Ti, against 1.53 ms
// for a pure 4-load 4-store sweep) and lands on the sweep floor in this form.

extern "C" __global__ void apply_fused_2q(
    double2 *state, unsigned long long pair_count, int q0, int q1,
    FusedMat mat)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    unsigned long long idx = expand_2q(k, q0, q1);

    // Basis b = (q0_bit << 1) | q1_bit, as the CPU PreparedGate2q::apply_full.
    unsigned long long indices[4] = {
        idx, idx | (1ULL << q1), idx | (1ULL << q0), idx | (1ULL << q0) | (1ULL << q1)};
    double2 in[4];
    double sum[4];
    #pragma unroll
    for (int col = 0; col < 4; ++col) {
        in[col] = state[indices[col]];
        sum[col] = in[col].x + in[col].y;
    }

    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        double k1 = 0.0, k2 = 0.0, k3 = 0.0;
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            const double *e = &mat.v[3 * (row * 4 + col)];
            k1 = fma(e[0], sum[col], k1);
            k2 = fma(e[1], in[col].x, k2);
            k3 = fma(e[2], in[col].y, k3);
        }
        state[indices[row]] = make_double2(k1 - k3, k1 + k2);
    }
}

// Monomial 4x4 in the apply_fused_2q basis: row r takes the amplitude of basis state
// `(src >> 2r) & 3` times scale r. Every source is read before any row is written.

extern "C" __global__ void apply_monomial_2q(
    double2 *state, unsigned long long pair_count, int q0, int q1,
    MonoScale scale, unsigned int src)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    unsigned long long idx = expand_2q(k, q0, q1);
    unsigned long long b0 = 1ULL << q0;
    unsigned long long b1 = 1ULL << q1;

    double2 in[4];
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        unsigned int c = (src >> (2 * row)) & 3u;
        in[row] = state[idx | ((c & 2u) ? b0 : 0ULL) | ((c & 1u) ? b1 : 0ULL)];
    }
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        unsigned long long i = idx | ((row & 2) ? b0 : 0ULL) | ((row & 1) ? b1 : 0ULL);
        state[i] = cmul(make_double2(scale.v[2 * row], scale.v[2 * row + 1]), in[row]);
    }
}

// ============================================================================
// QftBlock: DIF FFT passes over the field of `num` qubits starting at `start`
// ============================================================================
//
// A field index is x = (high | r << lo | L) for the stages lo .. lo + bits of one pass,
// where L < 2^lo. A block gathers 2^bits rows of 2^row_log consecutive amplitudes each,
// row r at stride 2^(start + lo), and runs the pass's stages in shared memory on the
// plain per-stride twiddles. The L-dependent part of every twiddle collects into one
// phase exp(2 pi i L rev(r) / 2^(lo + bits)) on output row r, read as the product of the
// three staged tables at `phases` (low table scaled by 2^(-bits/2)). With lo = 0, L is
// zero and only the scale applies.

__device__ __forceinline__ double2 qft_phase(
    const double2 * __restrict__ low, unsigned long long x, int low_bits, int mid_bits)
{
    const double2 *mid = low + (1 << low_bits);
    const double2 *high = mid + (1 << mid_bits);
    double2 p = cmul(__ldg(&low[x & ((1ULL << low_bits) - 1ULL)]),
                     __ldg(&mid[(x >> low_bits) & ((1ULL << mid_bits) - 1ULL)]));
    return cmul(p, __ldg(&high[x >> (low_bits + mid_bits)]));
}

extern "C" __global__ void qft_dif_pass(
    double2 *state, int abs_lo, int bits, int row_log, int start,
    const double2 * __restrict__ twiddles,
    const double2 * __restrict__ phases, int low_bits, int mid_bits, double scale)
{
    __shared__ double2 qft_tile[1 << QFT_TILE_Q];
    const int size = 1 << (bits + row_log);
    const int row_mask = (1 << row_log) - 1;
    const int col_blocks_log = abs_lo - row_log;
    unsigned long long g = blockIdx.x;
    unsigned long long col0 = (g & ((1ULL << col_blocks_log) - 1ULL)) << row_log;
    unsigned long long base = ((g >> col_blocks_log) << (abs_lo + bits)) + col0;

    for (int t = threadIdx.x; t < size; t += blockDim.x) {
        qft_tile[t] = state[base + ((unsigned long long)(t >> row_log) << abs_lo) + (t & row_mask)];
    }
    __syncthreads();

    for (int u = bits - 1; u >= 0; --u) {
        const int sh = u + row_log;
        const int low_mask = (1 << sh) - 1;
        const int k_mask = (1 << u) - 1;
        for (int p = threadIdx.x; p < (size >> 1); p += blockDim.x) {
            int t0 = ((p & ~low_mask) << 1) | (p & low_mask);
            int t1 = t0 | (1 << sh);
            double2 w = __ldg(&twiddles[(1 << u) + ((t0 >> row_log) & k_mask)]);
            double2 a = qft_tile[t0];
            double2 b = qft_tile[t1];
            qft_tile[t0] = make_double2(a.x + b.x, a.y + b.y);
            qft_tile[t1] = cmul(make_double2(a.x - b.x, a.y - b.y), w);
        }
        __syncthreads();
    }

    for (int t = threadIdx.x; t < size; t += blockDim.x) {
        int r = t >> row_log;
        double2 v = qft_tile[t];
        if (phases) {
            unsigned long long l = (col0 + (t & row_mask)) >> start;
            unsigned long long rev = __brev((unsigned int)r) >> (32 - bits);
            v = cmul(v, qft_phase(phases, l * rev, low_bits, mid_bits));
        } else {
            v.x *= scale;
            v.y *= scale;
        }
        state[base + ((unsigned long long)r << abs_lo) + (t & row_mask)] = v;
    }
}

// Reverse the `num` field bits at `start` of every index: one thread per index, the
// lower of each pair swaps.
extern "C" __global__ void qft_bit_reverse(
    double2 *state, unsigned long long dim, int start, int num)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    unsigned long long x = (i >> start) & ((1ULL << num) - 1ULL);
    unsigned long long xr = __brevll(x) >> (64 - num);
    if (x >= xr) return;
    unsigned long long j = i ^ ((x ^ xr) << start);
    double2 tmp = state[i];
    state[i] = state[j];
    state[j] = tmp;
}

// Field at bit 0, num >= 10: x = (a << (num - 5)) | (m << 5) | c reverses to
// (rev5(c), rev(m), rev5(a)). A 32 x 8 block loads the 32 x 32 tiles of m and rev(m)
// with coalesced rows and writes each into the other transposed, so stores stay
// coalesced too. The block of the larger middle exits.
extern "C" __global__ void qft_bit_reverse_tiled(double2 *state, int num)
{
    __shared__ double2 tile_a[32][33];
    __shared__ double2 tile_b[32][33];
    const int mid_bits = num - 10;
    unsigned long long m = (unsigned long long)blockIdx.x & ((1ULL << mid_bits) - 1ULL);
    unsigned long long mr = mid_bits ? (__brevll(m) >> (64 - mid_bits)) : 0ULL;
    if (m > mr) return;
    unsigned long long hb = ((unsigned long long)blockIdx.x >> mid_bits) << num;
    const int c = threadIdx.x;
    for (int a = threadIdx.y; a < 32; a += blockDim.y) {
        unsigned long long row = hb | ((unsigned long long)a << (num - 5));
        tile_a[a][c] = state[row | (m << 5) | c];
        tile_b[a][c] = state[row | (mr << 5) | c];
    }
    __syncthreads();
    const int rc = __brev((unsigned int)c) >> 27;
    for (int a = threadIdx.y; a < 32; a += blockDim.y) {
        const int ra = __brev((unsigned int)a) >> 27;
        unsigned long long row = hb | ((unsigned long long)a << (num - 5));
        state[row | (mr << 5) | c] = tile_a[rc][ra];
        state[row | (m << 5) | c] = tile_b[rc][ra];
    }
}

// ============================================================================
// Measurement
// ============================================================================
//
// measure_prob_one: per-block reduction of sum(|amp|^2) over elements where qubit bit is 1.
// Each block reduces 2*BLOCK_SIZE elements (using shared memory) into out_partials, which
// measure_prob_one_finalize then sums on the device.

extern "C" __global__ void measure_prob_one(
    const double2 *state, unsigned long long dim, int qubit, double *out_partials)
{
    extern __shared__ double sdata[];
    unsigned long long tid = threadIdx.x;
    unsigned long long i = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    double s = 0.0;
    if (i < dim && ((i >> qubit) & 1ULL)) {
        double2 a = state[i];
        s += a.x*a.x + a.y*a.y;
    }
    unsigned long long i2 = i + blockDim.x;
    if (i2 < dim && ((i2 >> qubit) & 1ULL)) {
        double2 a = state[i2];
        s += a.x*a.x + a.y*a.y;
    }
    sdata[tid] = s;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_partials[blockIdx.x] = sdata[0];
}

// Single-block reduction over `partials` (length `count`) into a single f64
// at `result[0]`. Caller launches with a single block of `blockDim.x` threads;
// each thread loops with stride `blockDim.x` over `partials` to accumulate.
// Replaces a num_blocks-element D2H plus a host sum with one 8-byte D2H.
extern "C" __global__ void measure_prob_one_finalize(
    const double *partials, unsigned int count, double *result)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    double s = 0.0;
    for (unsigned int i = tid; i < count; i += blockDim.x) {
        s += partials[i];
    }
    sdata[tid] = s;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

// rdm_qubit: per-block reduction of the four 1q reduced-density-matrix sums over
// amplitude pairs (i0, i1 = i0 | 1<<qubit): sum |a0|^2, sum |a1|^2, and sum
// a1 * conj(a0) (re, im). Pair index p maps to i0 by inserting a zero bit at
// `qubit`. Each block reduces 2*BLOCK_SIZE pairs into out_partials[4*blockIdx..].
// Shared memory holds four blockDim.x columns.

extern "C" __global__ void rdm_qubit(
    const double2 *state, unsigned long long pairs, int qubit, double *out_partials)
{
    extern __shared__ double sdata[];
    double *s0 = sdata;
    double *s1 = sdata + blockDim.x;
    double *sr = sdata + 2 * blockDim.x;
    double *si = sdata + 3 * blockDim.x;
    unsigned long long tid = threadIdx.x;
    unsigned long long low_mask = (1ULL << qubit) - 1ULL;
    double p0 = 0.0, p1 = 0.0, rr = 0.0, ri = 0.0;
    unsigned long long p = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    for (int rep = 0; rep < 2; ++rep) {
        if (p < pairs) {
            unsigned long long i0 = ((p & ~low_mask) << 1) | (p & low_mask);
            double2 a0 = state[i0];
            double2 a1 = state[i0 | (1ULL << qubit)];
            p0 += a0.x*a0.x + a0.y*a0.y;
            p1 += a1.x*a1.x + a1.y*a1.y;
            rr += a1.x*a0.x + a1.y*a0.y;
            ri += a1.y*a0.x - a1.x*a0.y;
        }
        p += blockDim.x;
    }
    s0[tid] = p0; s1[tid] = p1; sr[tid] = rr; si[tid] = ri;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_partials[4 * blockIdx.x + 0] = s0[0];
        out_partials[4 * blockIdx.x + 1] = s1[0];
        out_partials[4 * blockIdx.x + 2] = sr[0];
        out_partials[4 * blockIdx.x + 3] = si[0];
    }
}

// Single-block reduction of the four-column partials (length 4*count) into
// result[0..4]. Same shape as measure_prob_one_finalize; replaces a
// 4*num_blocks-element D2H plus a host sum with one 32-byte D2H.
extern "C" __global__ void rdm_qubit_finalize(
    const double *partials, unsigned int count, double *result)
{
    extern __shared__ double sdata[];
    double *s0 = sdata;
    double *s1 = sdata + blockDim.x;
    double *sr = sdata + 2 * blockDim.x;
    double *si = sdata + 3 * blockDim.x;
    unsigned int tid = threadIdx.x;
    double p0 = 0.0, p1 = 0.0, rr = 0.0, ri = 0.0;
    for (unsigned int i = tid; i < count; i += blockDim.x) {
        p0 += partials[4 * i + 0];
        p1 += partials[4 * i + 1];
        rr += partials[4 * i + 2];
        ri += partials[4 * i + 3];
    }
    s0[tid] = p0; s1[tid] = p1; sr[tid] = rr; si[tid] = ri;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = s0[0];
        result[1] = s1[0];
        result[2] = sr[0];
        result[3] = si[0];
    }
}

// sv_pauli_expect: per-block complex partials of
// sum_j conj(state[j ^ x]) * state[j] * (-1)^{popcount(j & z)} for every mask of
// group g = blockIdx.y, PAULI_AMPS_PER_THREAD amplitudes per thread. A group holds up to
// PAULI_GROUP_MASKS masks sharing one x, so each amplitude pair is read and
// multiplied once for the whole group and only the sign differs per mask. The
// product order matches the host reduction, so an x == 0 mask lands exactly on
// |amp|^2 with a zero imaginary part.
//
// groups = [x_0 .. x_{G-1}, start_0 .. start_G] and members = [z_0 .. z_{M-1},
// slot_0 .. slot_{M-1}], a group's members contiguous. Partials go to the
// member's slot, laid out [slot][block][re, im], and are reduced by
// dm_pauli_expect_finalize.
#define PAULI_GROUP_MASKS {{PAULI_GROUP_MASKS}}
#define PAULI_AMPS_PER_THREAD {{PAULI_AMPS_PER_THREAD}}
extern "C" __global__ void sv_pauli_expect(
    const double2 *state, unsigned long long dim,
    const unsigned long long *groups, unsigned int num_groups,
    const unsigned long long *members, unsigned int num_masks, double *out_partials)
{
    __shared__ double warp_sums[PAULI_GROUP_MASKS][2][32];
    unsigned int g = blockIdx.y;
    unsigned long long x = groups[g];
    unsigned int start = (unsigned int)groups[num_groups + g];
    unsigned int count = (unsigned int)groups[num_groups + g + 1] - start;
    unsigned long long z[PAULI_GROUP_MASKS];
    double re[PAULI_GROUP_MASKS], im[PAULI_GROUP_MASKS];
#pragma unroll
    for (int m = 0; m < PAULI_GROUP_MASKS; ++m) {
        z[m] = m < count ? members[start + m] : 0ULL;
        re[m] = 0.0;
        im[m] = 0.0;
    }
    unsigned long long j =
        (unsigned long long)blockIdx.x * (blockDim.x * PAULI_AMPS_PER_THREAD) + threadIdx.x;
    for (int rep = 0; rep < PAULI_AMPS_PER_THREAD; ++rep) {
        if (j < dim) {
            double2 a = state[j];
            double2 p = state[j ^ x];
            double pr = p.x * a.x + p.y * a.y;
            double pi = p.x * a.y - p.y * a.x;
#pragma unroll
            for (int m = 0; m < PAULI_GROUP_MASKS; ++m) {
                if (m < count) {
                    double sign = (__popcll(j & z[m]) & 1) ? -1.0 : 1.0;
                    re[m] += sign * pr;
                    im[m] += sign * pi;
                }
            }
        }
        j += blockDim.x;
    }
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp = threadIdx.x >> 5;
#pragma unroll
    for (int m = 0; m < PAULI_GROUP_MASKS; ++m) {
        if (m < count) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                re[m] += __shfl_down_sync(0xffffffffu, re[m], offset);
                im[m] += __shfl_down_sync(0xffffffffu, im[m], offset);
            }
            if (lane == 0) {
                warp_sums[m][0][warp] = re[m];
                warp_sums[m][1][warp] = im[m];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int num_warps = blockDim.x >> 5;
        for (unsigned int m = 0; m < count; ++m) {
            double r = lane < num_warps ? warp_sums[m][0][lane] : 0.0;
            double i = lane < num_warps ? warp_sums[m][1][lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                r += __shfl_down_sync(0xffffffffu, r, offset);
                i += __shfl_down_sync(0xffffffffu, i, offset);
            }
            if (lane == 0) {
                unsigned long long slot =
                    (members[num_masks + start + m] * gridDim.x + blockIdx.x) * 2ULL;
                out_partials[slot] = r;
                out_partials[slot + 1] = i;
            }
        }
    }
}

// sv_group_moments: per-block partials (sum p, sum p h, sum p h^2), p = |state[j]|^2,
// of one commuting group's operator h(j) = sum_i c_i (-1)^popcount(j & z_i). Each
// warp takes 64-index blocks. Within a block the high bits of j are fixed, so each
// term folds its sign into a histogram bin keyed by its low six mask bits, and a
// Walsh-Hadamard transform over the 64 bins gives h at every index of the block,
// as the host weighted_group_moments does. Each lane holds bins lane and lane + 32;
// the top level combines them in registers and the other five go through shuffles.
// Partials land at out_partials[3 * blockIdx.x + {0, 1, 2}].
extern "C" __global__ void sv_group_moments(
    const double2 *state, unsigned long long dim,
    const unsigned long long *zmasks, const double *coeffs, unsigned int num_terms,
    double *out_partials)
{
    __shared__ double bins[32][64];
    __shared__ double warp_sums[3][32];
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int num_warps = blockDim.x >> 5;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    unsigned long long num_blocks = (dim + 63ULL) >> 6;
    for (unsigned long long b = (unsigned long long)blockIdx.x * num_warps + warp;
         b < num_blocks; b += (unsigned long long)gridDim.x * num_warps) {
        unsigned long long base = b << 6;
        bins[warp][lane] = 0.0;
        bins[warp][lane + 32] = 0.0;
        __syncwarp();
        for (unsigned int t = lane; t < num_terms; t += 32) {
            unsigned long long z = zmasks[t];
            double c = (__popcll(base & z) & 1) ? -coeffs[t] : coeffs[t];
            atomicAdd(&bins[warp][z & 63ULL], c);
        }
        __syncwarp();
        double lo = bins[warp][lane];
        double hi = bins[warp][lane + 32];
        __syncwarp();
        double sum = lo + hi;
        hi = lo - hi;
        lo = sum;
        for (int half = 16; half > 0; half >>= 1) {
            double plo = __shfl_xor_sync(0xffffffffu, lo, half);
            double phi = __shfl_xor_sync(0xffffffffu, hi, half);
            if (lane & half) {
                lo = plo - lo;
                hi = phi - hi;
            } else {
                lo += plo;
                hi += phi;
            }
        }
        unsigned long long j = base + lane;
        if (j < dim) {
            double2 a = state[j];
            double p = a.x * a.x + a.y * a.y;
            s0 += p;
            s1 += p * lo;
            s2 += p * lo * lo;
        }
        j += 32;
        if (j < dim) {
            double2 a = state[j];
            double p = a.x * a.x + a.y * a.y;
            s0 += p;
            s1 += p * hi;
            s2 += p * hi * hi;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down_sync(0xffffffffu, s0, offset);
        s1 += __shfl_down_sync(0xffffffffu, s1, offset);
        s2 += __shfl_down_sync(0xffffffffu, s2, offset);
    }
    if (lane == 0) {
        warp_sums[0][warp] = s0;
        warp_sums[1][warp] = s1;
        warp_sums[2][warp] = s2;
    }
    __syncthreads();
    if (warp == 0) {
        s0 = lane < num_warps ? warp_sums[0][lane] : 0.0;
        s1 = lane < num_warps ? warp_sums[1][lane] : 0.0;
        s2 = lane < num_warps ? warp_sums[2][lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            s0 += __shfl_down_sync(0xffffffffu, s0, offset);
            s1 += __shfl_down_sync(0xffffffffu, s1, offset);
            s2 += __shfl_down_sync(0xffffffffu, s2, offset);
        }
        if (lane == 0) {
            out_partials[3ULL * blockIdx.x] = s0;
            out_partials[3ULL * blockIdx.x + 1] = s1;
            out_partials[3ULL * blockIdx.x + 2] = s2;
        }
    }
}

// measure_collapse: zero amplitudes where qubit bit != outcome.
// Launch over 2^n threads, block size BLOCK_SIZE.

extern "C" __global__ void measure_collapse(
    double2 *state, unsigned long long dim, int qubit, int outcome)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int bit = (int)((i >> qubit) & 1ULL);
    if (bit != outcome) {
        state[i].x = 0.0;
        state[i].y = 0.0;
    }
}

// compute_probabilities: out[i] = (amp.x² + amp.y²) * norm_sq for every basis state.
// Launched over 2^n threads. Saves half the PCIe traffic compared to dtoh'ing raw amplitudes
// and squaring on the host.

extern "C" __global__ void compute_probabilities(
    const double2 *state, unsigned long long dim, double norm_sq, double *out)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double2 a = state[i];
    out[i] = (a.x * a.x + a.y * a.y) * norm_sq;
}

// apply_multi_fused_tiled: batched non-diagonal MultiFused via shared-memory tiles.
//
// Each block loads the TILE_SIZE amplitudes spanned by the tile qubits into shared memory,
// then applies every sub-gate with no further global memory reads. Pairs (i0, i1) for a
// given gate stay within the tile because every target is a tile qubit; the block id
// supplies the bits of the other qubits.
//
// Gate data rides in parameter space: six bits of `gates.qubits` per tile qubit, four
// bits of `gates.targets` per gate (a position in the tile qubit list), and
// gates.v[8g .. 8g+8] the row-major matrix as re/im pairs.
//
// TILE_Q = 10, TILE_SIZE = 1024, block_size = 512 threads, each thread handles one pair.
// Shared memory usage: 1024 × 16 bytes = 16 KB. Pascal-friendly.
// (TILE_Q / TILE_SIZE are defined in the template header at the top of the file.)

// Scatter the TILE_Q bits of `j` onto the qubit positions `q`.
__device__ __forceinline__ unsigned long long deposit_tile_bits(int j, const int *q)
{
    unsigned long long idx = 0;
    #pragma unroll
    for (int k = 0; k < TILE_Q; ++k) {
        idx |= (unsigned long long)((j >> k) & 1) << q[k];
    }
    return idx;
}

// The tile is the 2^TILE_Q amplitudes spanned by the tile qubits (sorted ascending),
// with every other bit fixed per block. Tile-local index j maps to the global index
// block_base | deposit(j); the five lowest qubits are always in the tile, so a warp's 32
// consecutive j hit 32 consecutive amplitudes and the loads stay coalesced.
extern "C" __global__ void apply_multi_fused_tiled(
    double2 *state, unsigned long long dim,
    TileGates gates,
    int num_gates)
{
    __shared__ double2 tile[TILE_SIZE];

    unsigned long long tiles = dim >> TILE_Q;
    if (blockIdx.x >= tiles) return;

    int q[TILE_Q];
    #pragma unroll
    for (int k = 0; k < TILE_Q; ++k) q[k] = (int)((gates.qubits >> (6 * k)) & 63ULL);
    unsigned long long block_base = blockIdx.x;
    #pragma unroll
    for (int k = 0; k < TILE_Q; ++k) {
        unsigned long long lo = block_base & ((1ULL << q[k]) - 1ULL);
        block_base = ((block_base >> q[k]) << (q[k] + 1)) | lo;
    }

    int tid = (int)threadIdx.x;
    // Load the tile cooperatively, two amplitudes per thread (block_dim = TILE_SIZE/2).
    int a_off = tid;
    int b_off = tid + (TILE_SIZE / 2);
    unsigned long long a_idx = block_base | deposit_tile_bits(a_off, q);
    unsigned long long b_idx = block_base | deposit_tile_bits(b_off, q);
    tile[a_off] = state[a_idx];
    tile[b_off] = state[b_idx];
    __syncthreads();

    // Each thread owns one pair per gate. k is the compressed pair index (0 .. TILE_SIZE/2).
    int k = tid;

    for (int g = 0; g < num_gates; ++g) {
        int t = (int)((gates.targets >> (4 * g)) & 15ULL);
        int mask = (1 << t) - 1;
        int i0 = ((k & ~mask) << 1) | (k & mask);
        int i1 = i0 | (1 << t);

        const double *m = gates.v + 8 * g;
        double2 m00 = make_double2(m[0], m[1]);
        double2 m01 = make_double2(m[2], m[3]);
        double2 m10 = make_double2(m[4], m[5]);
        double2 m11 = make_double2(m[6], m[7]);

        // No sync needed between read and write here: each thread owns an
        // exclusive (i0, i1) pair within a single gate iteration (the compressed-
        // index mapping partitions [0, TILE_SIZE)). The trailing sync below
        // serialises writes across gate boundaries, which is the only hazard.

        apply2x2(tile, i0, i1, m00.x, m00.y, m01.x, m01.y, m10.x, m10.y, m11.x, m11.y);
        __syncthreads();
    }

    // Store tile back to global memory.
    state[a_idx] = tile[a_off];
    state[b_idx] = tile[b_off];
}

// apply_diagonal_batch: applies a mixed batch of diagonal entries (1q / 2q / parity-2q)
// via precomputed per-group LUTs (built on the host by build_diagonal_batch_tables).
// Replaces per-entry dispatch. Launches 2^n threads; each thread folds all group phases.
//
// Table layout: group g owns `1 << group_lens[g]` entries starting at
// group_tables[group_offsets[g]], indexed by the bit pattern `bits`. `meta` packs
// group_shifts[DB_MAX_GROUPS * DB_MAX_QUBITS] (qubit index of the j-th bit in group g),
// then group_lens[DB_MAX_GROUPS], then group_offsets[DB_MAX_GROUPS].

extern "C" __global__ void apply_diagonal_batch(
    double2 *state, unsigned long long dim,
    const double2 * __restrict__ group_tables,
    const int * __restrict__ meta,
    int num_groups)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const int *group_shifts = meta;
    const int *group_lens = meta + DB_MAX_GROUPS * DB_MAX_QUBITS;
    const int *group_offsets = group_lens + DB_MAX_GROUPS;

    double2 c = make_double2(1.0, 0.0);
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *shifts = group_shifts + g * DB_MAX_QUBITS;
        int bits = 0;
        for (int j = 0; j < len; ++j) {
            bits |= (int)(((i >> shifts[j]) & 1ULL) << j);
        }
        c = cmul(c, group_tables[group_offsets[g] + bits]);
    }

    apply_phase(state, i, c.x, c.y);
}

// apply_batch_rzz: applies a batch of Rzz gates via precomputed parity-phase LUTs (built
// on the host by build_batch_rzz_tables). Replaces per-edge apply_parity_phase launches.
// Launches 2^n threads across the full state; each thread computes parity bits per edge
// per group, indexes the 256-entry LUT, chains multiplies.
//
// Table layout: group g owns `1 << group_lens[g]` entries starting at
// group_tables[group_offsets[g]], indexed by the parity pattern `bits`. `meta` packs
// group_q0s[BR_MAX_GROUPS * BR_GROUP_SIZE] and group_q1s[BR_MAX_GROUPS * BR_GROUP_SIZE]
// (qubit pair of the k-th edge in group g), then group_lens[BR_MAX_GROUPS], then
// group_offsets[BR_MAX_GROUPS].

extern "C" __global__ void apply_batch_rzz(
    double2 *state, unsigned long long dim,
    const double2 * __restrict__ group_tables,
    const int * __restrict__ meta,
    int num_groups)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const int *group_q0s = meta;
    const int *group_q1s = meta + BR_MAX_GROUPS * BR_GROUP_SIZE;
    const int *group_lens = group_q1s + BR_MAX_GROUPS * BR_GROUP_SIZE;
    const int *group_offsets = group_lens + BR_MAX_GROUPS;

    double2 c = make_double2(1.0, 0.0);
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *q0s = group_q0s + g * BR_GROUP_SIZE;
        const int *q1s = group_q1s + g * BR_GROUP_SIZE;
        int bits = 0;
        for (int k = 0; k < len; ++k) {
            bits |= (int)((((i >> q0s[k]) ^ (i >> q1s[k])) & 1ULL) << k);
        }
        c = cmul(c, group_tables[group_offsets[g] + bits]);
    }

    apply_phase(state, i, c.x, c.y);
}

// apply_batch_phase: applies a batch of controlled-phase gates sharing a control qubit via
// precomputed LUTs (built on the host by build_batch_phase_tables). Replaces per-phase
// launches of apply_cu_phase. One DRAM read/write per amplitude in the ctrl=1 subspace.
//
// Table layout: group g owns `1 << group_lens[g]` entries starting at
// group_tables[group_offsets[g]], indexed by the bit pattern `bits`. `meta` packs
// group_shifts[BP_MAX_GROUPS * BP_GROUP_SIZE] (qubit index of the j-th bit in group g),
// then group_lens[BP_MAX_GROUPS], then group_offsets[BP_MAX_GROUPS].

extern "C" __global__ void apply_batch_phase(
    double2 *state, unsigned long long half_count,
    int control,
    const double2 * __restrict__ group_tables,
    const int * __restrict__ meta,
    int num_groups)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= half_count) return;
    const int *group_shifts = meta;
    const int *group_lens = meta + BP_MAX_GROUPS * BP_GROUP_SIZE;
    const int *group_offsets = group_lens + BP_MAX_GROUPS;
    unsigned long long ctrl_mask = 1ULL << control;
    unsigned long long mask = ctrl_mask - 1ULL;
    unsigned long long idx = ((k & ~mask) << 1) | (k & mask) | ctrl_mask;

    double2 c = make_double2(1.0, 0.0);
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *shifts = group_shifts + g * BP_GROUP_SIZE;
        int bits = 0;
        for (int j = 0; j < len; ++j) {
            bits |= (int)(((idx >> shifts[j]) & 1ULL) << j);
        }
        c = cmul(c, group_tables[group_offsets[g] + bits]);
    }

    apply_phase(state, idx, c.x, c.y);
}

// apply_multi_fused_diagonal: batch of diagonal 1q gates in a single pass. Replaces the
// per sub-gate decomposition on the host for `Gate::MultiFused { all_diagonal: true }`.
//
// Each thread handles one amplitude and folds all `num_gates` diagonal multiplications
// based on the bit pattern of its own index. One DRAM read + one DRAM write per thread;
// compute = num_gates complex multiplies (typically 1-10).
//
// Gate data rides in parameter space: gates.t[g] the target, gates.d[4g .. 4g+4] the
// (d0, d1) re/im pairs. Both entries are loaded uniformly and selected per thread.

extern "C" __global__ void apply_multi_fused_diagonal(
    double2 *state, unsigned long long dim,
    DiagList gates,
    int num_gates)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double2 a = state[i];
    for (int g = 0; g < num_gates; ++g) {
        int t = gates.t[g];
        int bit = (int)((i >> t) & 1ULL);
        const double *e = gates.d + 4 * g;
        double2 p = bit ? make_double2(e[2], e[3]) : make_double2(e[0], e[1]);
        double nx = p.x * a.x - p.y * a.y;
        double ny = p.x * a.y + p.y * a.x;
        a.x = nx;
        a.y = ny;
    }
    state[i] = a;
}

// ============================================================================
// Mixed shared-memory tile
// ============================================================================
//
// One block gathers the 2^MIX_TILE_Q amplitudes spanned by the tile qubits ops.q
// (ascending, every other bit fixed per block) and applies the op list to them. Op word:
// kind in bits 0..3, tile positions a and b in bits 3..7 and 7..11, a flag in bit 11,
// and the offset of its doubles in ops.c from bit 12. A 1q matrix carries its form in
// place of b: real, or real diagonal with imaginary off-diagonal as Rx has, each a half
// of the general multiply. Thread t owns tile entries
// t + m * MIX_BLOCK for the gather, the scatter and the diagonal ops, so consecutive
// diagonal ops need no barrier between them.

#define MIX_TILE        (1 << MIX_TILE_Q)
#define MIX_BLOCK       (1 << MIX_BLOCK_LOG)
#define MIX_PER_THREAD  (MIX_TILE / MIX_BLOCK)
#define MIX_MAT1  0
#define MIX_DIAG1 1
#define MIX_ANTI1 2
#define MIX_CX    3
#define MIX_SWAP  4
#define MIX_DIAG2 5
#define MIX_FORM_REAL 1
#define MIX_FORM_RX   2

__device__ __forceinline__ int insert_zero_bit(int k, int bit)
{
    return ((k >> bit) << (bit + 1)) | (k & ((1 << bit) - 1));
}

__device__ __forceinline__ void swap_entries(double2 *tile, int i, int j)
{
    double2 t = tile[i];
    tile[i] = tile[j];
    tile[j] = t;
}

extern "C" __global__ void apply_mixed_tile(double2 *state, MixedOps ops)
{
    __shared__ double2 tile[MIX_TILE];
    const int tid = (int)threadIdx.x;

    unsigned long long own = blockIdx.x;
    #pragma unroll
    for (int k = 0; k < MIX_TILE_Q; ++k) {
        int q = ops.q[k];
        own = ((own >> q) << (q + 1)) | (own & ((1ULL << q) - 1ULL));
    }
    #pragma unroll
    for (int k = 0; k < MIX_BLOCK_LOG; ++k) {
        own |= (unsigned long long)((tid >> k) & 1) << ops.q[k];
    }
    unsigned long long high[MIX_PER_THREAD];
    #pragma unroll
    for (int m = 0; m < MIX_PER_THREAD; ++m) {
        unsigned long long h = own;
        #pragma unroll
        for (int k = MIX_BLOCK_LOG; k < MIX_TILE_Q; ++k) {
            h |= (unsigned long long)((m >> (k - MIX_BLOCK_LOG)) & 1) << ops.q[k];
        }
        high[m] = h;
        tile[tid + m * MIX_BLOCK] = state[h];
    }

    bool elementwise = true;
    const int num_ops = ops.num_ops;
    for (int i = 0; i < num_ops; ++i) {
        const unsigned int w = ops.op[i];
        const int kind = w & 7;
        const int a = (w >> 3) & 15;
        const int b = (w >> 7) & 15;
        const bool flag = (w >> 11) & 1;
        const double *c = ops.c + (w >> 12);
        const bool diagonal = kind == MIX_DIAG1 || kind == MIX_DIAG2;
        if (!(diagonal && elementwise)) __syncthreads();
        elementwise = diagonal;

        if (diagonal && i + 1 < num_ops
            && ((ops.op[i + 1] & 7) == MIX_DIAG1 || (ops.op[i + 1] & 7) == MIX_DIAG2)) {
            int end = i + 2;
            while (end < num_ops && ((ops.op[end] & 7) == MIX_DIAG1 || (ops.op[end] & 7) == MIX_DIAG2)) ++end;
            for (int m = 0; m < MIX_PER_THREAD; ++m) {
                const int e = tid + m * MIX_BLOCK;
                double2 value = tile[e];
                for (int j = i; j < end; ++j) {
                    const unsigned int dw = ops.op[j];
                    const int dk = dw & 7;
                    const int da = (dw >> 3) & 15;
                    const int db = (dw >> 7) & 15;
                    const bool df = (dw >> 11) & 1;
                    const double *dc = ops.c + (dw >> 12);
                    if (dk == MIX_DIAG1) {
                        const int s = (e >> da) & 1;
                        if (s || !df) value = cmul(value, make_double2(dc[2 * s], dc[2 * s + 1]));
                    } else {
                        const int s = (((e >> da) & 1) << 1) | ((e >> db) & 1);
                        if (s == 3 || !df) value = cmul(value, make_double2(dc[2 * s], dc[2 * s + 1]));
                    }
                }
                tile[e] = value;
            }
            i = end - 1;
            continue;
        }

        if (kind == MIX_MAT1) {
            const double m00r = c[0], m00i = c[1], m01r = c[2], m01i = c[3];
            const double m10r = c[4], m10i = c[5], m11r = c[6], m11i = c[7];
            #pragma unroll
            for (int m = 0; m < MIX_PER_THREAD / 2; ++m) {
                int i0 = insert_zero_bit(tid + m * MIX_BLOCK, a);
                int i1 = i0 | (1 << a);
                double2 x = tile[i0];
                double2 y = tile[i1];
                if (b == MIX_FORM_REAL) {
                    tile[i0] = make_double2(m00r * x.x + m01r * y.x, m00r * x.y + m01r * y.y);
                    tile[i1] = make_double2(m10r * x.x + m11r * y.x, m10r * x.y + m11r * y.y);
                } else if (b == MIX_FORM_RX) {
                    tile[i0] = make_double2(m00r * x.x - m01i * y.y, m00r * x.y + m01i * y.x);
                    tile[i1] = make_double2(m11r * y.x - m10i * x.y, m11r * y.y + m10i * x.x);
                } else {
                    apply2x2(tile, i0, i1, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i);
                }
            }
        } else if (kind == MIX_DIAG1) {
            const double2 d0 = make_double2(c[0], c[1]);
            const double2 d1 = make_double2(c[2], c[3]);
            #pragma unroll
            for (int m = 0; m < MIX_PER_THREAD; ++m) {
                int e = tid + m * MIX_BLOCK;
                if ((e >> a) & 1) {
                    tile[e] = cmul(tile[e], d1);
                } else if (!flag) {
                    tile[e] = cmul(tile[e], d0);
                }
            }
        } else if (kind == MIX_ANTI1) {
            const double2 d0 = make_double2(c[0], c[1]);
            const double2 d1 = make_double2(c[2], c[3]);
            #pragma unroll
            for (int m = 0; m < MIX_PER_THREAD / 2; ++m) {
                int i0 = insert_zero_bit(tid + m * MIX_BLOCK, a);
                int i1 = i0 | (1 << a);
                if (flag) {
                    swap_entries(tile, i0, i1);
                } else {
                    double2 x = tile[i0];
                    tile[i0] = cmul(d0, tile[i1]);
                    tile[i1] = cmul(d1, x);
                }
            }
        } else if (kind == MIX_CX || kind == MIX_SWAP) {
            const int lo = a < b ? a : b;
            const int hi = a < b ? b : a;
            #pragma unroll
            for (int m = 0; m < MIX_PER_THREAD / 4; ++m) {
                int x = insert_zero_bit(insert_zero_bit(tid + m * MIX_BLOCK, lo), hi);
                if (kind == MIX_CX) {
                    swap_entries(tile, x | (1 << a), x | (1 << a) | (1 << b));
                } else {
                    swap_entries(tile, x | (1 << a), x | (1 << b));
                }
            }
        } else {
            const double2 d3 = make_double2(c[6], c[7]);
            if (flag) {
                #pragma unroll
                for (int m = 0; m < MIX_PER_THREAD; ++m) {
                    int e = tid + m * MIX_BLOCK;
                    if ((e >> a) & (e >> b) & 1) tile[e] = cmul(tile[e], d3);
                }
            } else {
                const double2 d0 = make_double2(c[0], c[1]);
                const double2 d1 = make_double2(c[2], c[3]);
                const double2 d2 = make_double2(c[4], c[5]);
                #pragma unroll
                for (int m = 0; m < MIX_PER_THREAD; ++m) {
                    int e = tid + m * MIX_BLOCK;
                    int s = (((e >> a) & 1) << 1) | ((e >> b) & 1);
                    double2 d = s == 0 ? d0 : s == 1 ? d1 : s == 2 ? d2 : d3;
                    tile[e] = cmul(tile[e], d);
                }
            }
        }
    }
    if (!elementwise) __syncthreads();

    #pragma unroll
    for (int m = 0; m < MIX_PER_THREAD; ++m) {
        state[high[m]] = tile[tid + m * MIX_BLOCK];
    }
}
