
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

// The fused 2q matrix travels in kernel parameter space: a pageable host to device
// copy synchronizes the stream before it starts, so uploading 32 doubles per gate
// serialized the host against every queued kernel. Layout mirrors `FusedMatArg`.
struct FusedMat { double v[32]; };
struct QubitList { int q[PARAM_QUBITS]; };
struct DiagList { double d[4 * PARAM_QUBITS]; int t[PARAM_QUBITS]; };
struct TileGates { double v[8 * TILE_Q]; unsigned long long qubits; unsigned long long targets; };

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
// and expands via chained insert_zero_bit (q0, q1 sorted).

__device__ __forceinline__ unsigned long long expand_2q(unsigned long long k, int lo_q, int hi_q) {
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
    int lo_q = control < target ? control : target;
    int hi_q = control < target ? target : control;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
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
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
    unsigned long long i = idx | (1ULL << q0) | (1ULL << q1);
    state[i].x = -state[i].x;
    state[i].y = -state[i].y;
}

extern "C" __global__ void apply_swap(
    double2 *state, unsigned long long pair_count, int q0, int q1)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
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
    int lo_q = control < target ? control : target;
    int hi_q = control < target ? target : control;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
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
    int lo_q = control < target ? control : target;
    int hi_q = control < target ? target : control;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
    unsigned long long i = idx | (1ULL << control) | (1ULL << target);
    apply_phase(state, i, pr, pi);
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
    unsigned long long idx = k;
    for (int i = 0; i < num_sorted; ++i) {
        int bit = sorted.q[i];
        unsigned long long mask_lo = (1ULL << bit) - 1;
        unsigned long long lo = idx & mask_lo;
        unsigned long long hi = idx >> bit;
        idx = (hi << (bit + 1)) | lo;
    }
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
    unsigned long long idx = k;
    for (int i = 0; i < num_sorted; ++i) {
        int bit = sorted.q[i];
        unsigned long long mask_lo = (1ULL << bit) - 1;
        unsigned long long lo = idx & mask_lo;
        unsigned long long hi = idx >> bit;
        idx = (hi << (bit + 1)) | lo;
    }
    unsigned long long i = idx | all_mask;
    apply_phase(state, i, pr, pi);
}

// ============================================================================
// Fused 2q: generic 4x4 matrix. mat is row-major 16 Complex64 = 32 f64.
// Launch over pair_count = 2^(n-2) threads; each processes one 4-element group.
// ============================================================================

extern "C" __global__ void apply_fused_2q(
    double2 *state, unsigned long long pair_count, int q0, int q1,
    FusedMat mat)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);

    // Basis ordering matches the CPU PreparedGate2q::apply_full in src/backend/simd.rs:
    //   basis index b = (q0_bit << 1) | q1_bit   i.e. q1 is LSB of the 4-element basis.
    //   b=0 → (q0=0, q1=0); b=1 → (q0=0, q1=1); b=2 → (q0=1, q1=0); b=3 → (q0=1, q1=1).
    unsigned long long i00 = idx;
    unsigned long long i01 = idx | (1ULL << q1);
    unsigned long long i10 = idx | (1ULL << q0);
    unsigned long long i11 = idx | (1ULL << q0) | (1ULL << q1);

    double2 in[4] = {state[i00], state[i01], state[i10], state[i11]};
    unsigned long long indices[4] = {i00, i01, i10, i11};

    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        double rr = 0.0, ri = 0.0;
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            // mat row-major: 32 f64s, row r col c → mat.v[2*(r*4+c)]=re, mat.v[2*(r*4+c)+1]=im
            double mr = mat.v[2*(row*4 + col)];
            double mi = mat.v[2*(row*4 + col) + 1];
            rr += mr*in[col].x - mi*in[col].y;
            ri += mr*in[col].y + mi*in[col].x;
        }
        state[indices[row]].x = rr;
        state[indices[row]].y = ri;
    }
}

// ============================================================================
// Measurement
// ============================================================================
//
// measure_prob_one: per-block reduction of sum(|amp|^2) over elements where qubit bit is 1.
// Each block reduces 2*BLOCK_SIZE elements (using shared memory). Host sums the
// out_partials array afterward.

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
// sum_j conj(state[j ^ x]) * state[j] * (-1)^{popcount(j & z)} for mask
// k = blockIdx.y, two amplitudes per thread; the same product order as the
// host reduction, so an x == 0 mask lands exactly on |amp|^2 with a zero
// imaginary part. Partials are laid out [k][block][re, im] and reduced by
// dm_pauli_expect_finalize.
extern "C" __global__ void sv_pauli_expect(
    const double2 *state, unsigned long long dim,
    const unsigned long long *xmasks, const unsigned long long *zmasks, double *out_partials)
{
    extern __shared__ double sdata[];
    double *sr = sdata;
    double *si = sdata + blockDim.x;
    unsigned long long tid = threadIdx.x;
    unsigned int k = blockIdx.y;
    unsigned long long x = xmasks[k];
    unsigned long long z = zmasks[k];
    double re = 0.0, im = 0.0;
    unsigned long long j = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    for (int rep = 0; rep < 2; ++rep) {
        if (j < dim) {
            double2 a = state[j];
            double2 p = state[j ^ x];
            double sign = (__popcll(j & z) & 1) ? -1.0 : 1.0;
            re += sign * (p.x * a.x + p.y * a.y);
            im += sign * (p.x * a.y - p.y * a.x);
        }
        j += blockDim.x;
    }
    sr[tid] = re;
    si[tid] = im;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        unsigned long long slot = ((unsigned long long)k * gridDim.x + blockIdx.x) * 2ULL;
        out_partials[slot] = sr[0];
        out_partials[slot + 1] = si[0];
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
// Each block loads a TILE_SIZE slice of amplitudes into shared memory, then applies every
// sub-gate whose target bit is inside the tile (target < TILE_Q) with no further global
// memory reads. Pairs (i0, i1) for a given gate stay within the tile because the target
// bit is a low bit of the global index; the high bits (> TILE_Q) are the block id.
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

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *shifts = group_shifts + g * DB_MAX_QUBITS;
        int bits = 0;
        for (int j = 0; j < len; ++j) {
            bits |= (int)(((i >> shifts[j]) & 1ULL) << j);
        }
        double2 ph = group_tables[group_offsets[g] + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    apply_phase(state, i, cr, ci);
}

// apply_batch_rzz: applies a batch of Rzz gates via precomputed parity-phase LUTs (built
// built on the host by build_batch_rzz_tables). Replaces per-edge apply_parity_phase launches.
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

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *q0s = group_q0s + g * BR_GROUP_SIZE;
        const int *q1s = group_q1s + g * BR_GROUP_SIZE;
        int bits = 0;
        for (int k = 0; k < len; ++k) {
            bits |= (int)((((i >> q0s[k]) ^ (i >> q1s[k])) & 1ULL) << k);
        }
        double2 ph = group_tables[group_offsets[g] + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    apply_phase(state, i, cr, ci);
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

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *shifts = group_shifts + g * BP_GROUP_SIZE;
        int bits = 0;
        for (int j = 0; j < len; ++j) {
            bits |= (int)(((idx >> shifts[j]) & 1ULL) << j);
        }
        double2 ph = group_tables[group_offsets[g] + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    apply_phase(state, idx, cr, ci);
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
