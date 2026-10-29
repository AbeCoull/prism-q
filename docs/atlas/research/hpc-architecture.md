# Dense execution and hardware evidence

Source snapshot: `7adde35d0bacaff71711148223ac8df06a008d01`. References give
source paths, symbols and inclusive line ranges. Everything here is read from the
source or derived as a model; no benchmark was run for this audit.

## State, indices, and pair ownership

`StatevectorBackend` holds one contiguous `Vec<Complex64>`, with basis index
`i = sum(bit(q) * 2^q)`. Qubit 0 is the least significant bit. Each amplitude
occupies two adjacent `f64` values, `[re, im]`, so the amplitude payload is
`16 * 2^n` bytes. For example, 25 qubits require 512 MiB of amplitudes and
30 qubits require 16 GiB. These totals exclude output probabilities, scratch,
and allocator overhead. Evidence: `src/backend/statevector/mod.rs:8-11,248-250`;
the casts and two-double pointer increments in `src/backend/simd.rs:348-371`.

For target `t`, define `half = 2^t`, `mask = half - 1`, and enumerate
`k in [0, 2^(n-1))`. The kernel inserts a zero bit at target position:

```text
i0 = ((k & ~mask) << 1) | (k & mask)
i1 = i0 | half
a = state[i0]
b = state[i1]
state[i0] = m00*a + m01*b
state[i1] = m10*a + m11*b
```

This is a bijection from `(k, target bit)` to state indices. Pairs are disjoint,
and both inputs are loaded before either output is stored. The two members of a
pair are `16 * 2^t` bytes apart, but each half of a block is a contiguous stream,
so a high target does not mean random access. Evidence: `src/backend/simd.rs:343-374`.

## Register lanes

Lanes are listed in ascending lane number, which for a load is also ascending
memory address. For two neighboring pair indices within a block:

| Register                        | Lane 0   | Lane 1   | Lane 2   | Lane 3   |
| ------------------------------- | -------- | -------- | -------- | -------- |
| AVX2 `a`                        | `re(a0)` | `im(a0)` | `re(a1)` | `im(a1)` |
| AVX2 `b`                        | `re(b0)` | `im(b0)` | `re(b1)` | `im(b1)` |
| `permute(a, 0101)`              | `im(a0)` | `re(a0)` | `im(a1)` | `re(a1)` |
| coefficient real broadcast      | `cr`     | `cr`     | `cr`     | `cr`     |
| coefficient imaginary broadcast | `ci`     | `ci`     | `ci`     | `ci`     |

An AVX2 register holds two complex amplitudes from the same half, interleaved
as real and imaginary parts. The two input registers together supply two pairs.
For a coefficient `c = cr + i*ci`, the intrinsic sequence is:

```text
z_swap = _mm256_permute_pd(z, 0b0101)
t = _mm256_mul_pd(c_ii, z_swap)
result = _mm256_fmaddsub_pd(c_rr, z, t)
```

Even lanes subtract and odd lanes add, giving
`[cr*x0-ci*y0, cr*y0+ci*x0, cr*x1-ci*y1, cr*y1+ci*x1]`.
Four complex multiplies and two vector adds implement the two output rows.
The stores preserve the original array-of-structs layout. Evidence:
`src/backend/simd.rs:72-85,258-307,376-413`.

The 128-bit FMA version packs one amplitude as `[re, im]`, shuffles with
`_mm_shuffle_pd(z,z,0b01)`, multiplies `ci * swapped`, and calls
`_mm_fmaddsub_pd(cr,z,t)`. SSE2 uses two multiplies, a shuffle, an XOR sign
change in the real lane, and an add. NEON packs `[re,im]` as well, but
prepares the imaginary coefficient as `[-ci,+ci]`, swaps lanes with
`vextq_f64(z,z,1)`, multiplies `cr*z`, and uses
`vfmaq_f64(product,imaginary_coeff,swapped)`. Evidence:
`src/backend/simd.rs:101-153,187-194,227-256`.

### Exact routing controls

For the ordinary dense, non-diagonal one-qubit matrix path, the kernel width
depends on both the `parallel` build feature and the qubit count.

| Context on an AVX2+FMA x86_64 CPU        | Arithmetic                                                       |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Sequential, `n <= 13`, `t >= 2`          | AVX2, two pairs per iteration                                    |
| Sequential, `t < 2` or `n >= 14`         | 128-bit FMA, one pair per iteration                              |
| `parallel` enabled, `n >= 15`, `t <= 12` | Tasks over 8192-amplitude tiles; AVX2 if `t >= 2`, FMA otherwise |
| `parallel` enabled, `n >= 15`, `t >= 13` | Tasks apply AVX2 to paired lower/upper slices                    |
| `PreparedGate1q::apply_tiled`, `t >= 2`  | AVX2 without the full-state length limit                         |
| `PreparedGate1q::apply_pair_ptr`         | 128-bit FMA even when AVX2 is available                          |

For the ordinary path, valid `0 <= t < n`, and non-miri builds, the branch
conditions combine to `use_avx2 = cpu_avx2 && cpu_fma && t >= 2 &&
(n <= 13 || (parallel_feature && n >= 15))`. At 14 qubits the ordinary path
uses 128-bit FMA. The predicate does not cover diagonal gates, controlled pointer
kernels, MultiFused tile coordinates or GPU execution. A host without FMA
uses SSE2 on x86_64; aarch64 uses NEON; other architectures use scalar code.
Evidence: `src/backend/simd.rs:574-643,645-733,743-769`;
`src/backend/statevector/kernels.rs:3075-3114,3626-3639`;
`src/backend/mod.rs:88-97`.

### Diagonal gates use a different route

The dispatch arm for one-qubit gates checks `Gate::is_diagonal_1q()` before
choosing the general matrix kernel. Z and Rz are diagonal variants. A
`Gate::Fused` matrix is also checked for diagonality, so wrapping Z in Fused
does not force general matrix arithmetic. Evidence:
`src/backend/statevector/mod.rs:851-872`; `src/gates/mod.rs:1561-1575`.

For diagonal gates on an AVX2+FMA x86_64 CPU, the exact width predicate is
`use_avx2 = parallel_feature && n >= 15 && t >= n-2`. All other diagonal
routes use 128-bit FMA when FMA is available. Without FMA the diagonal
sequential path is scalar, where the general kernel would fall back to SSE2.
On aarch64 the diagonal arithmetic remains 128-bit NEON. The high-target
parallel branch runs when fewer than four target blocks exist, splits each
pair of halves into 4096-amplitude chunks, and calls the AVX2-capable complex
scale helper. The other branch calls the 128-bit diagonal loop over tiles.
Evidence: `src/backend/statevector/kernels.rs:601-625,3642-3673`;
`src/backend/simd.rs:876-945,1355-1431`.

Each diagonal output depends only on its own input: `a'=d0*a`, `b'=d1*b`.
There are no cross-input products or row additions. Z leaves the lower
member untouched because `d0=1`, scaling only the upper member by -1. Rz
usually scales both. The lower member is skipped when both components of
`d0-1` are within the phase-one tolerance, so diagonal traffic differs from the
general matrix model. Evidence:
`src/backend/simd.rs:789-829`; `src/backend/mod.rs:274-278`.

## Threading and cache traffic

The ordinary one-qubit parallel path starts at 15 qubits. Let
`B = 2^(t+1)` and `T = max(8192,B)`. If `B <= 8192` and at least four
tiles exist, Rayon partitions the state into 8192-amplitude tiles. Otherwise,
with at least four `T` tiles, each block is split into lower and upper slices.
With fewer tiles, each block is traversed separately while paired slices are
partitioned into `min(8192,2^t)`-amplitude subtiles. Task count need not
match worker count, and the code fixes no CPU affinity, NUMA placement,
cache-line size, bandwidth or core count. Evidence: `src/backend/statevector/kernels.rs:3075-3114`.

For a general dense one-qubit matrix, conventional arithmetic counting gives
four complex multiplies (six FLOPs each) and two complex additions (two
FLOPs each), or 28 FLOPs per pair and `14 * 2^n` FLOPs per gate. Counting
two amplitude reads and two amplitude writes gives 64 logical bytes per
pair, or `32 * 2^n` bytes per gate and `28/64 = 0.4375` FLOP per byte.
The payload model leaves out coefficients, write allocation, cache reuse, page
effects and addressing instructions, and counts an FMA as two FLOPs. With
sustained bandwidth `W`, `32 * 2^n / W` is a lower bound on time if the whole
logical stream crosses that memory level.

MultiFused changes the stream count: apply all eligible gates inside each tile
before moving on. The tile budget is a power of two from 256 KiB through
1 MiB by default, derived from L2 per physical core and LLC share, with
`PRISM_TILE_KB` allowing 128 KiB through 2 MiB. Parallel tiles can shrink to
expose enough tiles for hardware concurrency. The tile width uses available
hardware parallelism, not Rayon pool width. High targets can also be gathered
into contiguous subcube scratch, processed locally, and scattered back.
The gather path skips an exactly zero subcube. With these routes a gate does
not always cost a full DRAM pass. Evidence:
`src/backend/cache.rs:22-29,53-89,96-134`;
`src/backend/statevector/kernels.rs:36-60,456-539,3536-3609`.

## CUDA execution

The device state is `2 * 2^n` interleaved `f64` values, interpreted by CUDA as
`double2*`. The amplitude payload stays 16 bytes per amplitude. The resource
gate budgets 25 bytes per amplitude: the 16-byte payload, probability
scratch and a margin. The 25 is a budgeting figure, not the state layout. On successful device initialization the host amplitude
vector is cleared. Evidence: `src/gpu/mod.rs:1-5,108-112,299-332`;
`src/backend/statevector/mod.rs:955-966`.

The ordinary CUDA 1q kernel launches blocks of 256 threads and assigns one
compressed pair index to each thread, using the same inserted-zero-bit map
as the CPU. Extra threads return at the bounds check. Kernel coefficients
travel as scalar launch arguments. The kernel contains no per-gate shared
memory tile. Coalescing depends on target: for `t=0`, each stream selects
every other `double2`; for `t>=5`, a warp's corresponding inputs are 32
consecutive amplitudes while it remains inside a half block. Address
patterns alone do not give transaction efficiency or occupancy.
Evidence: `src/gpu/kernels/dense.rs:21,206-247`;
`src/gpu/kernels/dense.cu:36-47,74-90`.

The non-diagonal MultiFused CUDA route keeps data on chip between gates:
one block loads 1024 amplitudes into a 16 KiB shared-memory tile, applies
several gates, then stores it. Each of 512 threads owns one pair during
each gate, and a block barrier separates gates. The ten tile qubits always
include the lowest five, so consecutive warp lanes load consecutive
amplitudes even when the other selected qubits are high. Each pass admits
five higher targets. Low-target gates enter only the first pass. Fewer
than three gates in a pass or `n<=10` uses ordinary per-gate launches.
Evidence: `src/gpu/kernels/dense.cu:803-889`;
`src/gpu/kernels/dense.rs:1750-1764,1911-1963`.

## Distributed memory and MPI

For `P=2^p` ranks, each rank stores `L=2^(n-p)` amplitudes. Initially,
`global_index = rank*L + local_index`: low bits index the local state and
high bits encode rank. A direct gate on physical global qubit `q` exchanges
with `rank XOR 2^(q-(n-p))`. A rank with that bit zero computes the first
matrix row; a rank with that bit one computes the second. A direct general
1q gate sends `16L` bytes and receives `16L` bytes per rank, split into
messages bounded by the exchange chunk. Global diagonal action is a scale
of the local slice and requires no amplitude exchange. These costs exclude
transport headers and configuration collectives. Evidence:
`src/backend/distributed_statevector/mod.rs:1-29,1035-1072,1103-1137`.

Relabeling is on by default, so most gates on a global qubit do not take the
direct exchange path. A circuit-to-physical map turns SWAP
into metadata updates. Before non-diagonal action on a global position,
the backend can exchange only the half slice where the local bit differs
from the rank bit, move the qubit into a local position, then reuse local
SIMD kernels for later gates. One relabel sends `8L` bytes and receives
`8L` bytes per rank. Batched instruction execution plans windows that fit
the local positions and uses future required use to select victims;
single-gate execution uses least recently used eviction. The mapping from
circuit qubit to rank bit therefore changes during a run. Evidence:
`src/backend/distributed_statevector/mod.rs:60-91,746-791`;
`src/distributed/mod.rs:61-79`.

`MpiComm` requests at least `MPI_THREAD_FUNNELED`. Rayon workers perform
memory work while MPI calls stay on the constructing thread below
`MPI_THREAD_MULTIPLE`. Transfers pass complex buffers as contiguous doubles
and split past the MPI count limit. The default exchange chunk is `2^29`
complex amplitudes, a maximum message size of 8 GiB; it is a ceiling, not a
buffer allocated on every exchange. Nothing in this source overlaps
communication with compute, uses GPU-aware MPI or combines the CUDA and
distributed backends, and no multi-node speedup was measured. Evidence: `src/distributed/comm.rs:248-285,323-386,454-462`;
`src/distributed/mod.rs:39-58`.

## Candidate scenes

1. Follow the same two amplitude pairs from basis indices through AoS memory,
   load, lane permutation, multiply, alternating FMA, row addition, and store.
   The route selector names the context: full state, parallel slice, or tile.
2. Apply a gate sequence gate by gate, then tile by tile. Show logical memory
   crossings as a model with explicit state and tile sizes, labeled as modeled
   traffic.
3. Transfer the tile mechanism to CUDA: global memory, anchored gather,
   16 KiB shared tile, repeated pair updates and barriers, final store.
4. Turn a high state-index bit into a rank bit, animate an XOR partner
   exchange, then enable relabeling to trade a half-slice exchange for later
   local work. Show circuit and physical qubit labels separately.

The source fixes ownership, memory order, dispatch predicates and
deterministic work counts. CPU frequency, cache residency, memory bandwidth,
occupancy, host-device crossover and network scaling need measurements on a
named machine.
