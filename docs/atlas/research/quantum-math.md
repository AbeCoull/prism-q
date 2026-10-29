# Mathematical basis and source audit

Audited revision: `7adde35d0bacaff71711148223ac8df06a008d01`. Source ranges refer to that revision. The formulas below follow from the stated conventions; none of them is a measurement.

## Statevector and binary indexing

The CPU backend stores a contiguous `Vec<Complex64>` with `2^n` amplitudes. Basis index `x = sum_q b_q 2^q` denotes `|b_(n-1)...b_0>`. Qubit 0 is the least significant bit, so `X q[0]` takes index 0 to index 1. The normalized mathematical state is

```text
|psi> = sum_(x=0)^(2^n-1) a_x |x>,   sum_x |a_x|^2 = 1.
```

Source: `src/backend/statevector/mod.rs:8-18,248-254`; `src/gates/mod.rs:1017-1029`.

For a gate `U` on target `q`, enumerate `k = 0,...,2^(n-1)-1`. Insert a zero at bit `q`:

```text
half = 1 << q
mask = half - 1
i = ((k & ~mask) << 1) | (k & mask)
j = i | half = i XOR half
a'_i = U00 a_i + U01 a_j
a'_j = U10 a_i + U11 a_j
```

Both original amplitudes must be loaded before either store, or the second result reads a modified input. The scalar pair kernel saves `v0` and `v1` before updating them. Source: `src/backend/simd.rs:516-523,680-692`.

For `n=3`, the complete pair lists are:

| Target | Pairs                      | Separation in amplitudes | Separation in bytes |
| ------ | -------------------------- | ------------------------ | ------------------- |
| 0      | (0,1), (2,3), (4,5), (6,7) | 1                        | 16                  |
| 1      | (0,2), (1,3), (4,6), (5,7) | 2                        | 32                  |
| 2      | (0,4), (1,5), (2,6), (3,7) | 4                        | 64                  |

For complex128 amplitudes the pair separation is `16 * 2^q` bytes. Within a `2^(q+1)` block the lower and upper streams each advance one amplitude at a time: `(base+t, base+half+t)` for `t=0,...,half-1`. A high target gives two long sequential streams far apart in memory; individual loads do not stride by `2^q`. At target 0 the two streams interleave adjacent amplitudes. Cache behavior also depends on line alignment, kernel routing, state size and cache capacity.

The visualization uses relative byte offsets. A Rust `Vec<Complex64>` guarantees no particular base address or 64-byte alignment, so the cache-line overlay assumes a base offset of zero.

## Gates and checkable examples

The implemented matrices include:

```text
H = (1/sqrt(2)) [[1, 1], [1, -1]]
X = [[0, 1], [1, 0]]
Y = [[0, -i], [i, 0]]
Rx(theta) = [[cos(theta/2), -i sin(theta/2)],
             [-i sin(theta/2), cos(theta/2)]]
Ry(theta) = [[cos(theta/2), -sin(theta/2)],
             [sin(theta/2), cos(theta/2)]]
Rz(theta) = diag(exp(-i theta/2), exp(i theta/2))
P(theta) = diag(1, exp(i theta))
```

`Rz(theta)` and `P(theta)` differ by a global phase when applied alone, so their raw amplitude arrays can differ. Under control the phase matters: a phase on one control branch is relative to the other branch. Source: `src/gates/mod.rs:1025-1073`.

For a one-qubit pure state `[a,b]`, the Bloch vector is `(2 Re(conj(a)b), 2 Im(conj(a)b), |a|^2-|b|^2)`. These are Pauli expectations, not amplitude coordinates. For an entangled qubit use its reduced density matrix; the resulting Bloch vector can have length less than 1.

Independently checkable examples:

1. `n=3`, `X q[0]` on zero state: only amplitude 1 is 1, basis label `001`.
2. `n=3`, `H q[2]` on zero state: amplitudes 0 and 4 equal `1/sqrt(2)`, each probability `1/2`.
3. `n=2`, `H q[0]; CX q[0],q[1]`: amplitudes 0 and 3 equal `1/sqrt(2)`, producing `(|00>+|11>)/sqrt(2)`.
4. `n=2`, `X q[0]; CX q[0],q[1]`: basis index 1 becomes index 3. Reversing control and target instead leaves index 1 unchanged.
5. `n=1`, `H; H` restores `[1,0]` to floating point tolerance.
6. `n=1`, `Ry(pi/2)` on zero state gives `[1/sqrt(2),1/sqrt(2)]`; `Rx(pi/2)` gives `[1/sqrt(2),-i/sqrt(2)]`. Equal computational probabilities do not mean equal states.

The specialized CPU CX kernel swaps amplitudes only in the control-one subspace. It does not materialize a 4 by 4 matrix. Source: `src/backend/statevector/kernels.rs:3677-3713`.

## Measurement and normalization

For a Z measurement on target `q`, the Born probability of outcome one is

```text
p1 = sum_(x: bit_q(x)=1) |a_x|^2.
P(m=1) = p1;   P(m=0) = 1-p1.
a'_x = a_x / sqrt(P(m)) if bit_q(x)=m, otherwise 0.
```

The CPU implementation samples `rng.random::<f64>() < p1`, stores a classical bit, and zeros the rejected half of every block. It defers the rescaling through `pending_norm` rather than multiplying every retained amplitude immediately. Actual logical amplitudes are `pending_norm * state[x]`, and logical probabilities use `pending_norm^2`. `state_vector()` exposes this working buffer; `export_statevector()` returns a normalized copy. A visualization that uses the raw slice after measurement must show the separate pending scale or normalize the exported state. Source: `src/backend/statevector/kernels.rs:4944-4985`; `src/backend/statevector/mod.rs:575-583,601-615,1087-1105`.

Bell measurement example: measuring q0 gives 0 or 1 with probability `1/2`, leaving `|00>` or `|11>`. Measuring q1 afterwards is deterministic and agrees. The two qubits are not independent fair coins.

## Stabilizer tableau

Each Pauli row stores `(x,z,r)`, with tensor factors encoded as `I=(0,0)`, `X=(1,0)`, `Z=(0,1)`, `Y=(1,1)` and sign `(-1)^r`. A full expression is `(-1)^r i^(sum_q x_q z_q) X^x Z^z`, where the exponent uses the integer count of Y factors. Two rows commute when their binary symplectic product vanishes:

```text
x dot z' + z dot x' = 0 mod 2.
```

The tableau has `2n+1` rows: n destabilizers, n stabilizers, and one scratch row. X and Z bits are packed into u64 words; a separate Boolean records each row's negative sign. The quantum state is the simultaneous +1 eigenstate of the signed stabilizer rows. Source: `src/backend/stabilizer/mod.rs:8-21`; `src/backend/stabilizer/kernels/rowops.rs:431-439`.

All updates below use input bits on the right:

```text
H(q): r' = r XOR (x_q AND z_q);  x'_q=z_q; z'_q=x_q
S(q): r' = r XOR (x_q AND z_q);  x'_q=x_q; z'_q=z_q XOR x_q
CX(c,t): r' = r XOR [x_c AND z_t AND (x_t XOR z_c XOR 1)]
         x'_t = x_t XOR x_c; z'_c = z_c XOR z_t
```

Source: `src/backend/stabilizer/kernels/rowops.rs:33-109`. The phase bit matters: H sends Y to -Y, and S sends Y to -X.

For two qubits initially in `|00>`, use stabilizers `Z0, Z1`. H0 produces `X0, Z1`. CX(0,1) produces `X0 X1, Z0 Z1`, the Bell stabilizers. With columns ordered q0 then q1 the stabilizer half is `x=[1,1], z=[0,0], r=0` and `x=[0,0], z=[1,1], r=0`. Basis ket labels still display q1 then q0. Keep tableau column order explicit.

Measuring Zq is random exactly when a stabilizer generator has X support at q and therefore anticommutes with Zq. That branch samples a fair bit, clears other anticommuting rows by row multiplication, and replaces a generator with the signed measured Z. When no such stabilizer exists, a product of generators determines the sign. Source: `src/backend/stabilizer/mod.rs:876-905`; `src/backend/stabilizer/kernels/mod.rs:674-763`.

Storage is O(n^2) bits, an elementary Clifford update is O(n) row work, and measurement is O(n^2) in the worst case. Bit packing and sparse generator indexing change the constants and the rows visited. Dense amplitude export is still exponential, and a tableau can only represent stabilizer states.

The stabilizer backend rejects non-Clifford gates, but a non-Clifford circuit does not always need a dense statevector. PRISM-Q also has stabilizer-rank, Pauli-propagation and QEC T strategies; routing depends on the operations and the workload. Source: `src/backend/stabilizer/mod.rs:18-35`; `src/sim/stabilizer_rank.rs`; `src/qec/t_sampler.rs`; `src/qec/camps_prefix.rs`.

## Matrix product states

The MPS factors an amplitude as a contraction of site tensors:

```text
a_(b0,...,b(n-1)) = sum_(alpha1,...,alpha(n-1))
  A[0]_(1,b0,alpha1) A[1]_(alpha1,b1,alpha2) ... A[n-1]_(alpha(n-1),b(n-1),1).
```

A site has shape `(chi_left,2,chi_right)` and flat offset `alpha*(2*chi_right)+b*chi_right+beta`. PRISM-Q can move logical qubits between sites, so a renderer must distinguish physical chain site from logical qubit. Source: `src/backend/mps/mod.rs:9-21,147-165,734-736`.

Across an orthonormal bipartition, Schmidt decomposition is `|psi> = sum_i lambda_i |u_i>|v_i>`, with `lambda_i >= 0` and `sum lambda_i^2=1`. Schmidt rank is the minimum exact bond dimension at that cut. Let `p_i=lambda_i^2`. The reduced density matrix has eigenvalues p, so

```text
S = -sum_i p_i log2(p_i) <= log2(rank) <= log2(chi).
```

The first inequality follows because entropy at fixed rank is maximized by a uniform distribution. Qubit count alone cannot determine chi: an arbitrarily long GHZ chain has rank 2 across each interior cut, while a generic balanced bipartition can require rank `2^floor(n/2)`.

For an adjacent two-qubit gate, contract two sites into `theta[alpha,i,j,beta]`, apply the 4 by 4 gate to physical indices `(i,j)`, and reshape into matrix `M[(alpha*2+i),(j*chi_right+beta)]`. Factor `M=U diag(s) Vdagger`, truncate, and reshape to site shapes `(chi_left,2,chi_new)` and `(chi_new,2,chi_right)`. Internal SVD matrix storage is column-major even though each site tensor uses the flat layout above. Source: `src/backend/mps/mod.rs:1120-1149,1300-1361`.

The retained rank is the number of singular values strictly larger than `epsilon*s_max`, at least 1 and at most `max_bond_dim`. Default epsilon is `1e-12`; automatic dispatch sets a cap of 256. The cutoff is relative to the largest singular value and puts no limit on cumulative discarded squared weight. Source: `src/backend/mps/svd.rs:233-245`; `src/backend/mps/mod.rs:18-21,411-427`.

For normalized Schmidt values, a single orthonormal cut discards weight `delta=sum_(i>=chi) lambda_i^2`. The unnormalized projection differs from the input by 2-norm `sqrt(delta)`. After normalizing the retained projection, squared overlap fidelity is `1-delta` and squared Euclidean distance is `2-2*sqrt(1-delta)`. The three quantities are different.

Numerical example: `lambda=[sqrt(0.9),sqrt(0.1)]` has entropy approximately `0.4689955936` bits. Retaining chi=1 discards weight 0.1, leaves norm squared 0.9, and has normalized fidelity 0.9. The Bell state has lambda `[1/sqrt(2),1/sqrt(2)]`, entropy 1 bit, and needs chi=2 for exact representation. A rank-1 cut discards weight 0.5, regardless of which tied component is retained.

PRISM-Q accumulates each cut's discarded fraction `sum(discarded s^2)/sum(all s^2)`. The sum can exceed 1 and is not the final state's infidelity. Truncation leaves the chain unnormalized; probability, expectation, sampling and export reads rescale. For chains with a wider bond the backend reports approximate metadata with lower bound `max(0,1-cumulative_discarded)`. That figure is bookkeeping from local cuts, not a proven bound for arbitrary sequences of truncations. Source: `src/backend/mps/mod.rs:440-485,2273-2318,2938-2950`.

## Noise, QEC, and stochastic interpretation

A one-qubit Pauli channel is `E(rho)=(1-px-py-pz)rho+px XrhoX+py YrhoY+pz ZrhoZ`. The PRISM-Q depolarizing parameter p is total nonidentity probability, so px=py=pz=p/3. Two-qubit depolarizing noise samples each of the 15 nonidentity Paulis with p/15. Conventions that mix directly with the maximally mixed state use a different p. Source: `src/sim/noise.rs:36-65`.

For Kraus operators satisfying `sum K_k^dagger K_k=I`, density-matrix evolution is `rho'=sum K_k rho K_k^dagger`. A trajectory samples branch k with `p_k=Tr(K_k^dagger K_k rho)`, then applies `K_k/sqrt(p_k)`. PRISM-Q's custom one-qubit trajectory obtains the reduced density matrix, including off-diagonal coherence, to evaluate that probability. Populations alone are insufficient for arbitrary Kraus channels. Source: `src/sim/noise.rs:66-79,283-329`; `src/sim/trajectory.rs:563-617`.

Amplitude damping example: on `|1>` with gamma=0.2, the jump to `|0>` occurs with probability 0.2 and no-jump `|1>` with probability 0.8. The density matrix is `diag(0.2,0.8)`. One trajectory is pure and does not equal this density matrix; its ensemble reproduces the channel. A finite sample adds uncertainty, even when each conditional evolution is exact.

A Pauli frame propagates algebraically through Clifford gates. Through CX(c,t), `Xc -> Xc Xt`, `Zt -> Zc Zt`, `Zc -> Zc`, and `Xt -> Xt`. A Pauli error flips a measured stabilizer exactly when the two anticommute. The same binary symplectic parity tests this. For Bell stabilizers XX and ZZ, a Z0 error flips XX and preserves ZZ; an X0 error preserves XX and flips ZZ. Source basis: `src/backend/stabilizer/kernels/rowops.rs:95-109,431-439`; `src/sim/noise.rs:2649`.

QEC detector rows are XOR parities over measurement records. Default detector reference XORs a deterministic noiseless baseline whose random measurement outcomes are fixed to zero. Raw parity is separately available. A detector that is random without noise does not flag an error; programs with non-Clifford gates report raw parity under either reference setting. Source: `src/qec/mod.rs:296-305,559-560,600-615`.

The decoders are weighted union-find for graphlike models, blossom minimum-weight matching (exact for quantized graph weights), and BP+OSD, which also handles hypergraph mechanisms. Union-find and matching reject mechanisms touching more than two detectors unless the model is decomposed. Matching quantizes relative to the largest edge weight at resolution `2^-23`; exact matching on those weights is not optimal decoding under every physical noise model. BP+OSD searches exhaustively only when its configured search space covers all free columns, and is not maximum-likelihood in general. Source: `src/qec/decoder.rs:20-64`; `src/qec/matching.rs:28-63`; `src/qec/bposd.rs:1-46`.

For N independent Bernoulli shots estimating event probability p, the sample fraction has variance `p(1-p)/N`; the plug-in standard error is `sqrt(p_hat*(1-p_hat)/N)`. At p=0.01 and N=10000 the true standard deviation is approximately 0.0009949874. Zero observed failures does not mean zero probability. Quasiprobability estimators and postselected records need their own variance and accepted-shot accounting; the Bernoulli interval does not apply to them.

## Validation contracts for the atlas

- Pair enumeration visits each basis index exactly once and the target bit differs across each pair.
- X, H, Rx, Ry, and Rz preserve squared norm to numerical tolerance for normalized inputs.
- Applying U then Udagger restores the input, including complex phases.
- Matrix results for controlled gates preserve control and target ordering.
- Measurement probabilities sum to one, and conditional states have the sampled bit fixed.
- Bell and GHZ examples retain their correlated measurement outcomes.
- H, S, and CX tableau updates match the corresponding Pauli conjugations, including negative phases.
- MPS tensor contractions match dense vectors before truncation; one-cut fidelity and discarded weight match the explicit Schmidt example.
- Anything computed in the browser is labeled as a teaching model, separate from native PRISM-Q execution.

Existing regression evidence can be inspected in `src/backend/statevector/tests.rs:1603`, `tests/integration/stabilizer_correctness.rs`, `tests/integration/mps_correctness.rs`, `tests/integration/noise_validation.rs`, and `tests/integration/qec_detector_reference.rs`. Reading them shows coverage; it does not show they pass in a given environment.
