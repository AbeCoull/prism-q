
// ============================================================================
// Density-matrix sweeps. `n` is the circuit width; the buffer holds 4^n
// amplitudes indexed (ket << n) | bra.
// ============================================================================

__device__ __forceinline__ unsigned long long dm_insert_zero(unsigned long long m, int pos)
{
    unsigned long long low = (1ULL << pos) - 1ULL;
    return ((m & ~low) << 1) | (m & low);
}

__device__ __forceinline__ double2 dm_mul(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Base of the (q0, q1) block for compacted index m: a zero bit inserted at
// each of the four block positions in ascending order.
__device__ __forceinline__ unsigned long long dm_block_base(
    unsigned long long m, int q0, int q1, int n)
{
    int p[4] = {q0, q1, q0 + n, q1 + n};
    for (int i = 1; i < 4; ++i) {
        int v = p[i];
        int j = i - 1;
        while (j >= 0 && p[j] > v) { p[j + 1] = p[j]; --j; }
        p[j + 1] = v;
    }
    unsigned long long base = m;
    for (int i = 0; i < 4; ++i) base = dm_insert_zero(base, p[i]);
    return base;
}

// Offset of block slot 4 * tr + tc from the block base.
__device__ __forceinline__ unsigned long long dm_block_offset(int slot, int q0, int q1, int n)
{
    int tr = slot >> 2;
    int tc = slot & 3;
    unsigned long long off = 0;
    if (tr & 2) off |= 1ULL << (q0 + n);
    if (tr & 1) off |= 1ULL << (q1 + n);
    if (tc & 2) off |= 1ULL << q0;
    if (tc & 1) off |= 1ULL << q1;
    return off;
}

// out[r] = Re rho[r][r]. Launch over d = 2^n threads.
extern "C" __global__ void dm_diagonal(const double2 *state, unsigned long long d, double *out)
{
    unsigned long long r = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= d) return;
    out[r] = state[r * d + r].x;
}

// Per-block partials of sum |a|^2 over the whole buffer; finalize with
// measure_prob_one_finalize.
extern "C" __global__ void dm_norm_sqr(const double2 *state, unsigned long long len, double *out_partials)
{
    extern __shared__ double sdata[];
    unsigned long long tid = threadIdx.x;
    unsigned long long i = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    double s = 0.0;
    if (i < len) { double2 a = state[i]; s += a.x * a.x + a.y * a.y; }
    unsigned long long i2 = i + blockDim.x;
    if (i2 < len) { double2 a = state[i2]; s += a.x * a.x + a.y * a.y; }
    sdata[tid] = s;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_partials[blockIdx.x] = sdata[0];
}

// Keep the entries whose row and column both agree with `outcome` on the
// measured qubit, scaled by `scale`; zero the rest.
extern "C" __global__ void dm_project(
    double2 *state, unsigned long long len,
    unsigned long long rmask, unsigned long long cmask, int outcome, double scale)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    int row = ((i & rmask) != 0ULL) ? 1 : 0;
    int col = ((i & cmask) != 0ULL) ? 1 : 0;
    double2 a = state[i];
    if (row == outcome && col == outcome) {
        state[i] = make_double2(a.x * scale, a.y * scale);
    } else {
        state[i] = make_double2(0.0, 0.0);
    }
}

// rho -> |0><0| (x) tr_q rho. One thread per (row-clear, col-clear) entry owns
// its three siblings, so no two threads touch the same amplitude.
extern "C" __global__ void dm_reset(
    double2 *state, unsigned long long groups, int qubit, int n)
{
    unsigned long long m = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= groups) return;
    unsigned long long base = dm_insert_zero(dm_insert_zero(m, qubit), qubit + n);
    unsigned long long rmask = 1ULL << (qubit + n);
    unsigned long long cmask = 1ULL << qubit;
    double2 a = state[base];
    double2 b = state[base | rmask | cmask];
    state[base] = make_double2(a.x + b.x, a.y + b.y);
    state[base | rmask] = make_double2(0.0, 0.0);
    state[base | cmask] = make_double2(0.0, 0.0);
    state[base | rmask | cmask] = make_double2(0.0, 0.0);
}

extern "C" __global__ void dm_conjugate(double2 *state, unsigned long long len)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    state[i].y = -state[i].y;
}

// rho[r][c] *= f(r) * conj(f(c)) for the 2^n-entry phase table `table`.
extern "C" __global__ void dm_diagonal_sandwich(
    double2 *state, unsigned long long len, int n, const double2 *table)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned long long d = 1ULL << n;
    double2 k = table[i >> n];
    double2 b = table[i & (d - 1ULL)];
    b.y = -b.y;
    state[i] = dm_mul(state[i], dm_mul(k, b));
}

// Diagonal 16-entry block superoperator: one complex multiply per amplitude,
// slot 4 * tr + tc read from the ket bits (tr) and bra bits (tc) of (q0, q1).
extern "C" __global__ void dm_kraus_2q_diagonal(
    double2 *state, unsigned long long len, int q0, int q1, int n, const double2 *diag)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    int slot = (int)(((i >> (q0 + n)) & 1ULL) << 3)
             | (int)(((i >> (q1 + n)) & 1ULL) << 2)
             | (int)(((i >> q0) & 1ULL) << 1)
             | (int)((i >> q1) & 1ULL);
    state[i] = dm_mul(state[i], diag[slot]);
}

// Dense 16x16 block superoperator `s` (row-major, 256 complex entries), one
// block of 16 amplitudes per thread.
extern "C" __global__ void dm_kraus_2q_dense(
    double2 * __restrict__ state, unsigned long long groups, int q0, int q1, int n,
    const double2 * __restrict__ s)
{
    unsigned long long m = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= groups) return;
    unsigned long long base = dm_block_base(m, q0, q1, n);
    unsigned long long idx[16];
    double2 v[16];
    for (int k = 0; k < 16; ++k) {
        idx[k] = base | dm_block_offset(k, q0, q1, n);
        v[k] = state[idx[k]];
    }
    for (int row = 0; row < 16; ++row) {
        double re = 0.0, im = 0.0;
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            double2 c = s[row * 16 + col];
            re += c.x * v[col].x - c.y * v[col].y;
            im += c.x * v[col].y + c.y * v[col].x;
        }
        state[idx[row]] = make_double2(re, im);
    }
}

// Symmetric two-qubit depolarizing on each block: B -> alpha B + beta Tr(B) I.
extern "C" __global__ void dm_depolarizing_2q(
    double2 *state, unsigned long long groups, int q0, int q1, int n, double alpha, double beta)
{
    unsigned long long m = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= groups) return;
    unsigned long long base = dm_block_base(m, q0, q1, n);
    double tr_re = 0.0, tr_im = 0.0;
    for (int t = 0; t < 4; ++t) {
        double2 a = state[base | dm_block_offset(5 * t, q0, q1, n)];
        tr_re += a.x;
        tr_im += a.y;
    }
    double shift_re = tr_re * beta;
    double shift_im = tr_im * beta;
    for (int k = 0; k < 16; ++k) {
        unsigned long long i = base | dm_block_offset(k, q0, q1, n);
        double2 a = state[i];
        double re = a.x * alpha;
        double im = a.y * alpha;
        if ((k % 5) == 0) { re += shift_re; im += shift_im; }
        state[i] = make_double2(re, im);
    }
}

// rho = |psi><psi| from the 2^n amplitudes `amps`. Launch over 4^n threads.
extern "C" __global__ void dm_outer_product(
    double2 *state, unsigned long long len, int n, const double2 *amps)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned long long d = 1ULL << n;
    double2 k = amps[i >> n];
    double2 b = amps[i & (d - 1ULL)];
    b.y = -b.y;
    state[i] = dm_mul(k, b);
}

// Per-block complex partials of sum_j (-1)^{popcount(j & z)} rho[j][j ^ x] for
// mask k = blockIdx.y; j runs over the 2^n rows, two per thread. Partials are
// laid out [k][block][re, im].
extern "C" __global__ void dm_pauli_expect(
    const double2 *state, unsigned long long d,
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
        if (j < d) {
            double2 e = state[j * d + (j ^ x)];
            double sign = (__popcll(j & z) & 1) ? -1.0 : 1.0;
            re += sign * e.x;
            im += sign * e.y;
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

// One block per mask: sums that mask's `count` partial pairs into result[2k..].
extern "C" __global__ void dm_pauli_expect_finalize(
    const double *partials, unsigned int count, double *result)
{
    extern __shared__ double sdata[];
    double *sr = sdata;
    double *si = sdata + blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int k = blockIdx.x;
    double re = 0.0, im = 0.0;
    for (unsigned int i = tid; i < count; i += blockDim.x) {
        unsigned long long slot = ((unsigned long long)k * count + i) * 2ULL;
        re += partials[slot];
        im += partials[slot + 1];
    }
    sr[tid] = re;
    si[tid] = im;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[2 * k] = sr[0];
        result[2 * k + 1] = si[0];
    }
}
