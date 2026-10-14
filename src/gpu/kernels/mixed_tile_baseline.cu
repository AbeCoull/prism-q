
extern "C" __global__ void apply_mixed_tile_baseline(double2 *state, MixedOps ops)
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
    for (int i = 0; i < ops.num_ops; ++i) {
        const unsigned int w = ops.op[i];
        const int kind = w & 7;
        const int a = (w >> 3) & 15;
        const int b = (w >> 7) & 15;
        const bool flag = (w >> 11) & 1;
        const double *c = ops.c + (w >> 12);
        const bool diagonal = kind == MIX_DIAG1 || kind == MIX_DIAG2;
        if (!(diagonal && elementwise)) __syncthreads();
        elementwise = diagonal;

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
