//! Signed-Clifford prefix tracker for the CAMPS path.
//!
//! Maintains a Clifford unitary `C` implicitly via its inverse tableau:
//! `inv_x[q] = C† · X_q · C` and `inv_z[q] = C† · Z_q · C`, each a
//! signed Pauli string with a phase in `{1, i, -1, -i}` (encoded as
//! `phase4 ∈ {0, 1, 2, 3}` so that the operator value is `i^phase4`).
//! For Hermitian images of Hermitian Paulis the phase is always `0`
//! or `2` once a row is settled; intermediate `phase4` of `1` or `3`
//! can occur during a `rowmul` and clears by the end of the gate.
//!
//! The OFD disentangler reads `C† · Z_q · C` directly from the
//! inverse-tableau row. Final-observable evaluation
//! `⟨ψ| O |ψ⟩ = ⟨ψ'| C† O C |ψ'⟩` factors the observable into single-
//! qubit `X`/`Z` components and reads from the same row.
//!
//! Per-gate updates use the rule
//! `new inv_x[q] = C† · (U · X_q · U†) · C` (and analogous for Z),
//! expanded by linearity over the existing tableau rows. This is the
//! correct cumulative composition for sequential state-gate
//! application `C ← U·C`.

use crate::gates::Gate;

/// Packed signed Pauli row for the inverse Clifford tableau. `(x, z)` bit
/// pairs encode the letter directly ((0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=Y)
/// and `phase4` is the `i^{phase4}` global factor; this matches the
/// convention in [`crate::sim::stabilizer_rank`]'s `SignedPauli`. The two
/// are deliberately distinct: this one uses packed `Vec<u64>` rows and
/// `rowmul` for full-tableau composition, while that one uses dense
/// `Vec<bool>` storage and forward conjugation for a single string. Keep
/// the letter and phase conventions in sync across both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SignedPauli {
    pub x: Vec<u64>,
    pub z: Vec<u64>,
    pub phase4: u8,
}

impl SignedPauli {
    fn zero(num_words: usize) -> Self {
        Self {
            x: vec![0u64; num_words],
            z: vec![0u64; num_words],
            phase4: 0,
        }
    }

    #[inline(always)]
    pub fn get_x(&self, q: usize) -> bool {
        (self.x[q >> 6] >> (q & 63)) & 1 == 1
    }

    #[inline(always)]
    pub fn get_z(&self, q: usize) -> bool {
        (self.z[q >> 6] >> (q & 63)) & 1 == 1
    }

    #[inline(always)]
    fn set_x(&mut self, q: usize, b: bool) {
        let m = 1u64 << (q & 63);
        if b {
            self.x[q >> 6] |= m;
        } else {
            self.x[q >> 6] &= !m;
        }
    }

    #[inline(always)]
    fn set_z(&mut self, q: usize, b: bool) {
        let m = 1u64 << (q & 63);
        if b {
            self.z[q >> 6] |= m;
        } else {
            self.z[q >> 6] &= !m;
        }
    }

    pub fn pauli_at(&self, q: usize) -> PauliKind {
        match (self.get_x(q), self.get_z(q)) {
            (false, false) => PauliKind::I,
            (true, false) => PauliKind::X,
            (true, true) => PauliKind::Y,
            (false, true) => PauliKind::Z,
        }
    }

    /// Collect the non-identity letters as MPS Pauli-axis factors, ready
    /// for [`crate::backend::mps::MpsBackend::pauli_expectation`].
    pub(crate) fn mps_factors(&self, n: usize) -> Vec<(usize, crate::backend::mps::MpsPauliAxis)> {
        (0..n)
            .filter_map(|q| self.pauli_at(q).to_mps_axis().map(|axis| (q, axis)))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PauliKind {
    I,
    X,
    Y,
    Z,
}

impl PauliKind {
    /// Map a Pauli letter to its MPS axis, or `None` for identity.
    pub(crate) fn to_mps_axis(self) -> Option<crate::backend::mps::MpsPauliAxis> {
        use crate::backend::mps::MpsPauliAxis;
        match self {
            PauliKind::I => None,
            PauliKind::X => Some(MpsPauliAxis::X),
            PauliKind::Y => Some(MpsPauliAxis::Y),
            PauliKind::Z => Some(MpsPauliAxis::Z),
        }
    }
}

/// Per-qubit phase contribution when multiplying Pauli letter `a` by
/// Pauli letter `b` (in that order). Returns the `phase4` increment
/// such that `(letter a) · (letter b) = i^{increment} · (letter a XOR b
/// in (x,z) bits)`.
#[inline(always)]
fn letter_product_phase(ax: bool, az: bool, bx: bool, bz: bool) -> u8 {
    match ((ax, az), (bx, bz)) {
        ((false, false), _) | (_, (false, false)) => 0,
        ((true, false), (true, false)) => 0,
        ((false, true), (false, true)) => 0,
        ((true, true), (true, true)) => 0,
        ((true, false), (true, true)) => 1,
        ((true, false), (false, true)) => 3,
        ((true, true), (true, false)) => 3,
        ((true, true), (false, true)) => 1,
        ((false, true), (true, false)) => 1,
        ((false, true), (true, true)) => 3,
    }
}

/// `dst ← (i^extra_phase4) · dst · src`, with phase tracking across
/// each qubit position.
fn rowmul_into(dst: &mut SignedPauli, src: &SignedPauli, n: usize, extra_phase4: u8) {
    let mut total: u32 = u32::from(dst.phase4) + u32::from(src.phase4) + u32::from(extra_phase4);
    for q in 0..n {
        let ax = dst.get_x(q);
        let az = dst.get_z(q);
        let bx = src.get_x(q);
        let bz = src.get_z(q);
        total += u32::from(letter_product_phase(ax, az, bx, bz));
    }
    for w in 0..dst.x.len() {
        dst.x[w] ^= src.x[w];
        dst.z[w] ^= src.z[w];
    }
    dst.phase4 = (total & 3) as u8;
}

/// `rows[dst] ← (i^extra_phase4) · rows[dst] · rows[src]` for two distinct
/// indices into the same row vector. Splits the borrow so neither row is
/// cloned. `dst == src` is a logic error (a row times itself).
fn rowmul_within(rows: &mut [SignedPauli], dst: usize, src: usize, n: usize, extra_phase4: u8) {
    debug_assert_ne!(dst, src, "rowmul_within requires distinct rows");
    let hi = dst.max(src);
    let (left, right) = rows.split_at_mut(hi);
    if dst < src {
        rowmul_into(&mut left[dst], &right[0], n, extra_phase4);
    } else {
        rowmul_into(&mut right[0], &left[src], n, extra_phase4);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SignedCliffordPrefix {
    num_qubits: usize,
    pub(crate) inv_x: Vec<SignedPauli>,
    pub(crate) inv_z: Vec<SignedPauli>,
}

impl SignedCliffordPrefix {
    pub fn identity(num_qubits: usize) -> Self {
        let num_words = num_qubits.div_ceil(64).max(1);
        let mut inv_x = Vec::with_capacity(num_qubits);
        let mut inv_z = Vec::with_capacity(num_qubits);
        for q in 0..num_qubits {
            let mut x = SignedPauli::zero(num_words);
            x.set_x(q, true);
            inv_x.push(x);
            let mut z = SignedPauli::zero(num_words);
            z.set_z(q, true);
            inv_z.push(z);
        }
        Self {
            num_qubits,
            inv_x,
            inv_z,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn conjugate_z(&self, q: usize) -> SignedPauli {
        self.inv_z[q].clone()
    }

    pub fn conjugate_x(&self, q: usize) -> SignedPauli {
        self.inv_x[q].clone()
    }

    pub fn apply_state_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<(), &'static str> {
        match gate {
            Gate::Id => Ok(()),
            Gate::H => {
                self.apply_h(targets[0]);
                Ok(())
            }
            Gate::S => {
                self.apply_s(targets[0]);
                Ok(())
            }
            Gate::Sdg => {
                self.apply_sdg(targets[0]);
                Ok(())
            }
            Gate::SX => {
                self.apply_sx(targets[0]);
                Ok(())
            }
            Gate::SXdg => {
                self.apply_sxdg(targets[0]);
                Ok(())
            }
            Gate::X => {
                self.apply_x(targets[0]);
                Ok(())
            }
            Gate::Y => {
                self.apply_y(targets[0]);
                Ok(())
            }
            Gate::Z => {
                self.apply_z(targets[0]);
                Ok(())
            }
            Gate::Cx => {
                self.apply_cx(targets[0], targets[1]);
                Ok(())
            }
            Gate::Cz => {
                self.apply_cz(targets[0], targets[1]);
                Ok(())
            }
            Gate::Swap => {
                self.apply_swap(targets[0], targets[1]);
                Ok(())
            }
            _ => Err("non-Clifford gate cannot be absorbed into the SignedCliffordPrefix"),
        }
    }

    fn apply_h(&mut self, p: usize) {
        std::mem::swap(&mut self.inv_x[p], &mut self.inv_z[p]);
    }

    fn apply_s(&mut self, p: usize) {
        // State-gate S: new inv_x[p] = C† (S† X S) C = C† (-Y) C = -i X Z propagated.
        let n = self.num_qubits;
        rowmul_into(&mut self.inv_x[p], &self.inv_z[p], n, 3);
    }

    fn apply_sdg(&mut self, p: usize) {
        // State-gate Sdg: new inv_x[p] = C† (S X S†) C = C† Y C = i X Z propagated.
        let n = self.num_qubits;
        rowmul_into(&mut self.inv_x[p], &self.inv_z[p], n, 1);
    }

    fn apply_sx(&mut self, p: usize) {
        // State-gate SX: new inv_z[p] = C† (SX† Z SX) C = C† Y C.
        // Rowmul order here is inv_z[p] · inv_x[p] = Z·X = iY, so landing on
        // +Y needs an extra `i^3 = -i` factor: (-i)·iY = Y.
        let n = self.num_qubits;
        rowmul_into(&mut self.inv_z[p], &self.inv_x[p], n, 3);
    }

    fn apply_sxdg(&mut self, p: usize) {
        // State-gate SXdg: new inv_z[p] = C† (-Y) C. With Z·X = iY order and
        // an extra factor `i`: i · iY = -Y.
        let n = self.num_qubits;
        rowmul_into(&mut self.inv_z[p], &self.inv_x[p], n, 1);
    }

    fn apply_x(&mut self, p: usize) {
        // X X X = X, X Z X = -Z → flip inv_z[p] sign (phase4 += 2)
        self.inv_z[p].phase4 = (self.inv_z[p].phase4 + 2) & 3;
    }

    fn apply_y(&mut self, p: usize) {
        self.inv_x[p].phase4 = (self.inv_x[p].phase4 + 2) & 3;
        self.inv_z[p].phase4 = (self.inv_z[p].phase4 + 2) & 3;
    }

    fn apply_z(&mut self, p: usize) {
        self.inv_x[p].phase4 = (self.inv_x[p].phase4 + 2) & 3;
    }

    fn apply_cx(&mut self, ctrl: usize, tgt: usize) {
        // CX X_c CX = X_c X_t → new inv_x[c] = inv_x[c] · inv_x[t]
        // CX X_t CX = X_t → unchanged
        // CX Z_c CX = Z_c → unchanged
        // CX Z_t CX = Z_c Z_t → new inv_z[t] = inv_z[c] · inv_z[t]
        let n = self.num_qubits;
        rowmul_within(&mut self.inv_x, ctrl, tgt, n, 0);
        rowmul_within(&mut self.inv_z, tgt, ctrl, n, 0);
    }

    fn apply_cz(&mut self, a: usize, b: usize) {
        // CZ X_a CZ = X_a Z_b → new inv_x[a] = inv_x[a] · inv_z[b]
        // CZ X_b CZ = Z_a X_b → new inv_x[b] = inv_z[a] · inv_x[b]
        // Z_a, Z_b unchanged
        let n = self.num_qubits;
        rowmul_into(&mut self.inv_x[a], &self.inv_z[b], n, 0);
        rowmul_into(&mut self.inv_x[b], &self.inv_z[a], n, 0);
    }

    fn apply_swap(&mut self, a: usize, b: usize) {
        self.inv_x.swap(a, b);
        self.inv_z.swap(a, b);
    }

    /// Right-composition state-gate fold: `C ← C · U`. Used to absorb
    /// the disentangler inverse `D†` into the Clifford prefix after a
    /// CAMPS T-gate. Implementing this via `apply_state_gate` would
    /// compose on the wrong side (`U · C`), which only coincides with
    /// `C · U` when `C` and `U` commute or `D` is trivial.
    ///
    /// Each inverse-tableau row `R = C† P C` transforms as
    /// `R → U† R U` (Heisenberg conjugation by `U` of the existing
    /// Pauli row), so the update is local to the columns touched by
    /// `U` and tracks any phase introduced by the conjugation.
    pub(crate) fn fold_right_state_gate(
        &mut self,
        gate: &Gate,
        targets: &[usize],
    ) -> Result<(), &'static str> {
        match gate {
            Gate::Id => Ok(()),
            Gate::H => {
                self.fold_right_h(targets[0]);
                Ok(())
            }
            Gate::S => {
                self.fold_right_s(targets[0]);
                Ok(())
            }
            Gate::Sdg => {
                self.fold_right_sdg(targets[0]);
                Ok(())
            }
            Gate::X => {
                self.fold_right_x(targets[0]);
                Ok(())
            }
            Gate::Y => {
                self.fold_right_y(targets[0]);
                Ok(())
            }
            Gate::Z => {
                self.fold_right_z(targets[0]);
                Ok(())
            }
            Gate::Cx => {
                self.fold_right_cx(targets[0], targets[1]);
                Ok(())
            }
            Gate::Cz => {
                self.fold_right_cz(targets[0], targets[1]);
                Ok(())
            }
            _ => Err("gate not supported in fold_right_state_gate"),
        }
    }

    fn fold_right_h(&mut self, p: usize) {
        for row in self.inv_x.iter_mut().chain(self.inv_z.iter_mut()) {
            let xp = row.get_x(p);
            let zp = row.get_z(p);
            row.set_x(p, zp);
            row.set_z(p, xp);
            if xp && zp {
                row.phase4 = (row.phase4 + 2) & 3;
            }
        }
    }

    fn fold_right_s(&mut self, p: usize) {
        // Sdg · P · S at position p: X → -Y, Y → X, Z/I unchanged.
        for row in self.inv_x.iter_mut().chain(self.inv_z.iter_mut()) {
            if row.get_x(p) {
                let had_z = row.get_z(p);
                row.set_z(p, !had_z);
                if !had_z {
                    row.phase4 = (row.phase4 + 2) & 3;
                }
            }
        }
    }

    fn fold_right_sdg(&mut self, p: usize) {
        // S · P · Sdg at position p: X → Y, Y → -X, Z/I unchanged.
        for row in self.inv_x.iter_mut().chain(self.inv_z.iter_mut()) {
            if row.get_x(p) {
                let had_z = row.get_z(p);
                row.set_z(p, !had_z);
                if had_z {
                    row.phase4 = (row.phase4 + 2) & 3;
                }
            }
        }
    }

    fn fold_right_x(&mut self, p: usize) {
        // X · P · X at position p: Y → -Y, Z → -Z, X/I unchanged.
        for row in self.inv_x.iter_mut().chain(self.inv_z.iter_mut()) {
            if row.get_z(p) {
                row.phase4 = (row.phase4 + 2) & 3;
            }
        }
    }

    fn fold_right_y(&mut self, p: usize) {
        // Y · P · Y at position p: X → -X, Z → -Z, Y/I unchanged.
        for row in self.inv_x.iter_mut().chain(self.inv_z.iter_mut()) {
            let xp = row.get_x(p);
            let zp = row.get_z(p);
            if xp ^ zp {
                row.phase4 = (row.phase4 + 2) & 3;
            }
        }
    }

    fn fold_right_z(&mut self, p: usize) {
        // Z · P · Z at position p: X → -X, Y → -Y, Z/I unchanged.
        for row in self.inv_x.iter_mut().chain(self.inv_z.iter_mut()) {
            if row.get_x(p) {
                row.phase4 = (row.phase4 + 2) & 3;
            }
        }
    }

    fn fold_right_cx(&mut self, c: usize, t: usize) {
        // CX · P · CX: X_c → X_c X_t, Z_t → Z_c Z_t. Phase increment per
        // Aaronson-Gottesman: x_c · z_t · (x_t XOR z_c XOR 1).
        for row in self.inv_x.iter_mut().chain(self.inv_z.iter_mut()) {
            let xc = row.get_x(c);
            let zc = row.get_z(c);
            let xt = row.get_x(t);
            let zt = row.get_z(t);
            if xc && zt && (xt ^ zc ^ true) {
                row.phase4 = (row.phase4 + 2) & 3;
            }
            if xc {
                row.set_x(t, !xt);
            }
            if zt {
                row.set_z(c, !zc);
            }
        }
    }

    fn fold_right_cz(&mut self, a: usize, b: usize) {
        // CZ = H_b · CX(a,b) · H_b, fold-right composes left-to-right.
        self.fold_right_h(b);
        self.fold_right_cx(a, b);
        self.fold_right_h(b);
    }
}

/// Optimization-Free Disentangler (Algorithm 1 of Liu & Clark
/// arXiv:2412.17209). Given an MPS state `|ψ'⟩` and a Pauli string `P`
/// expressed as a [`SignedPauli`], constructs a Clifford disentangler
/// `D` such that applying `D` to `|ψ'⟩` leaves at least one qubit
/// disentangled in the `|0⟩` state and rotates `P` to act trivially
/// on that qubit.
///
/// The returned cascade is a sequence of gates with their target
/// qubits, intended to be applied to the MPS via the existing
/// [`crate::backend::Backend`] dispatch. All gates share a single
/// control qubit `n` chosen as the first index where MPS site `n` is
/// in `|0⟩` and `P[n] ∈ {X, Y}`. For each other qubit `m` with non-
/// identity Pauli factor:
/// - `P[m] = X` → `CX(n, m)`
/// - `P[m] = Y` → `Sdg(m), CX(n, m), S(m)` (CY decomposition)
/// - `P[m] = Z` → `CZ(n, m)`
///
/// Returns an empty cascade when no such `n` exists. The disentangler
/// inverse `D†` is what gets folded into the Clifford prefix.
pub(crate) type OfdGate = (Gate, Vec<usize>);

fn build_xy_anchor_cascade(p: &SignedPauli, n: usize, num_qubits: usize) -> Vec<OfdGate> {
    let mut cascade: Vec<OfdGate> = Vec::new();
    for m in 0..num_qubits {
        if m == n {
            continue;
        }
        match p.pauli_at(m) {
            PauliKind::I => continue,
            PauliKind::X => cascade.push((Gate::Cx, vec![n, m])),
            PauliKind::Y => {
                cascade.push((Gate::Sdg, vec![m]));
                cascade.push((Gate::Cx, vec![n, m]));
                cascade.push((Gate::S, vec![m]));
            }
            PauliKind::Z => cascade.push((Gate::Cz, vec![n, m])),
        }
    }
    cascade
}

fn support_qubits(p: &SignedPauli, num_qubits: usize) -> Vec<usize> {
    (0..num_qubits)
        .filter(|&q| !matches!(p.pauli_at(q), PauliKind::I))
        .collect()
}

/// Sum of MPS-site distances over every two-qubit gate in a cascade.
/// Same routing-cost proxy as [`anchor_routing_cost`] but evaluated on
/// the assembled cascade so OFD and OFDS variants (which can pick
/// different anchors and different gate sequences) can be compared
/// directly. Single-qubit cascade gates contribute 0 since they do not
/// route across MPS sites.
pub(crate) fn cascade_routing_cost(
    mps: &crate::backend::mps::MpsBackend,
    cascade: &[OfdGate],
) -> usize {
    cascade
        .iter()
        .filter(|(_, targets)| targets.len() == 2)
        .map(|(_, targets)| {
            mps.site_for_qubit(targets[0])
                .abs_diff(mps.site_for_qubit(targets[1]))
        })
        .sum()
}

/// Which disentangler tier produced a chosen cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisentanglerKind {
    /// OFD (Algorithm 1), bond-dimension preserving when applied to
    /// the chosen `|0⟩` anchor.
    Ofd,
    /// OFDS (Algorithm 2), works without the `|0⟩` precondition; may
    /// grow bond dimension on the MPS.
    Ofds,
}

/// Cost-compare OFD vs OFDS and return the cheaper cascade, biased to
/// OFD on ties since OFD is bond-dimension safe by construction. Caller
/// must handle empty / single-qubit support before calling this. Only
/// multi-qubit support paths reach disentangler dispatch.
///
/// Returns `Ok(None)` when neither OFD nor OFDS can produce a cascade
/// (an invariant violation given the empty / single-qubit short-circuit
/// happened upstream).
pub(crate) fn choose_disentangler(
    mps: &crate::backend::mps::MpsBackend,
    p: &SignedPauli,
    num_qubits: usize,
    tol: f64,
) -> crate::error::Result<Option<(usize, Vec<OfdGate>, DisentanglerKind)>> {
    let ofd = build_ofd_disentangler(mps, p, num_qubits, tol)?;
    let ofds = build_ofds_disentangler(mps, p, num_qubits);
    Ok(match (ofd, ofds) {
        (Some((n, c_ofd)), Some((m, c_ofds))) => {
            if cascade_routing_cost(mps, &c_ofds) < cascade_routing_cost(mps, &c_ofd) {
                Some((m, c_ofds, DisentanglerKind::Ofds))
            } else {
                Some((n, c_ofd, DisentanglerKind::Ofd))
            }
        }
        (Some((n, c)), None) => Some((n, c, DisentanglerKind::Ofd)),
        (None, Some((m, c))) => Some((m, c, DisentanglerKind::Ofds)),
        (None, None) => None,
    })
}

/// Sum of MPS-site distances from `anchor` to every other qubit in
/// `support`. Proxy for the SWAP-routing cost of the resulting CX/CZ
/// cascade and therefore for bond-dimension growth on the MPS.
fn anchor_routing_cost(
    mps: &crate::backend::mps::MpsBackend,
    anchor: usize,
    support: &[usize],
) -> usize {
    let anchor_site = mps.site_for_qubit(anchor);
    support
        .iter()
        .filter(|&&q| q != anchor)
        .map(|&q| mps.site_for_qubit(q).abs_diff(anchor_site))
        .sum()
}

pub(crate) fn build_ofd_disentangler(
    mps: &crate::backend::mps::MpsBackend,
    p: &SignedPauli,
    num_qubits: usize,
    tol: f64,
) -> crate::error::Result<Option<(usize, Vec<OfdGate>)>> {
    let support = support_qubits(p, num_qubits);
    let mut best: Option<(usize, usize)> = None;
    for &n in &support {
        if !matches!(p.pauli_at(n), PauliKind::X | PauliKind::Y) {
            continue;
        }
        if !mps.is_qubit_in_zero_state(n, tol)? {
            continue;
        }
        let cost = anchor_routing_cost(mps, n, &support);
        if best.is_none_or(|(_, c)| cost < c) {
            best = Some((n, cost));
        }
    }
    Ok(best.map(|(n, _)| (n, build_xy_anchor_cascade(p, n, num_qubits))))
}

/// Optimization-Free Disentangler with State support (Algorithm 2 of
/// Liu & Clark arXiv:2412.17209). Same cascade structure as
/// [`build_ofd_disentangler`] but with no `|0⟩` precondition on the
/// anchor qubit. The `|0⟩` requirement in OFD is a bond-dimension-
/// preservation optimization; OFDS produces a correct disentangler for
/// any MPS state at the cost of possibly growing bond dimension when
/// the cascade is applied.
///
/// Anchor selection:
/// - First qubit with `P[n] ∈ {X, Y}` if any. Cascade matches OFD.
/// - Otherwise (`P` has only `Z` letters and `>= 2` of them), anchors
///   at the last `Z` qubit and reduces support via a CX ladder
///   `CX(q_i, q_{i+1})` over consecutive `Z` positions. Each rung
///   maps `Z_{q_i} Z_{q_{i+1}} → Z_{q_{i+1}}` (Heisenberg picture),
///   leaving a single `Z` on the anchor after `k - 1` CXs.
///
/// Returns `None` only when `P` has fewer than two non-identity
/// letters and none of them are `X`/`Y`. The caller already handles
/// the empty- and single-support cases directly.
pub(crate) fn build_ofds_disentangler(
    mps: &crate::backend::mps::MpsBackend,
    p: &SignedPauli,
    num_qubits: usize,
) -> Option<(usize, Vec<OfdGate>)> {
    let support = support_qubits(p, num_qubits);
    let xy_candidates: Vec<usize> = support
        .iter()
        .copied()
        .filter(|&n| matches!(p.pauli_at(n), PauliKind::X | PauliKind::Y))
        .collect();
    if let Some(&anchor) = xy_candidates
        .iter()
        .min_by_key(|&&n| anchor_routing_cost(mps, n, &support))
    {
        return Some((anchor, build_xy_anchor_cascade(p, anchor, num_qubits)));
    }
    let z_support: Vec<usize> = support
        .iter()
        .copied()
        .filter(|&q| matches!(p.pauli_at(q), PauliKind::Z))
        .collect();
    if z_support.len() < 2 {
        return None;
    }
    let anchor = *z_support
        .iter()
        .min_by_key(|&&q| anchor_routing_cost(mps, q, &z_support))
        .unwrap();
    let mut z_sites: Vec<(usize, usize)> = z_support
        .iter()
        .map(|&q| (mps.site_for_qubit(q), q))
        .collect();
    z_sites.sort_by_key(|&(s, _)| s);
    let ordered: Vec<usize> = z_sites.into_iter().map(|(_, q)| q).collect();
    let anchor_pos = ordered.iter().position(|&q| q == anchor).unwrap();
    let mut cascade: Vec<OfdGate> = Vec::with_capacity(ordered.len() - 1);
    for i in 0..anchor_pos {
        cascade.push((Gate::Cx, vec![ordered[i], ordered[i + 1]]));
    }
    for i in (anchor_pos + 1..ordered.len()).rev() {
        cascade.push((Gate::Cx, vec![ordered[i], ordered[i - 1]]));
    }
    Some((anchor, cascade))
}

/// Evaluate `⟨ψ|Π_k P_k|ψ⟩` for a general Pauli product over X/Y/Z terms
/// on a CAMPS state `|ψ⟩ = C|ϕ⟩` where the Clifford prefix `C` is tracked
/// by `prefix` and the MPS holds `|ϕ⟩`.
///
/// Rewrites the observable as `⟨ϕ| C† (Π P_k) C |ϕ⟩` by composing the
/// conjugated rows via signed-Pauli multiplication: `C† Z_q C` and
/// `C† X_q C` come straight from the inverse tableau, and a Y term is
/// `C† Y_q C = i · (C† X_q C)(C† Z_q C)`, with the `i` supplied as
/// `extra_phase4 = 1` on that product. The result is evaluated on the MPS
/// via [`crate::backend::mps::MpsBackend::pauli_expectation`].
///
/// The composed string is canonicalized for the MPS evaluator: each
/// qubit's `(x, z)` bit pattern is mapped to letter `I`/`X`/`Y`/`Z`,
/// with the residual `(-i)` factor from rewriting `X·Z = -i·Y`
/// absorbed into the overall coefficient alongside the stored `i^phase4`.
/// For a Hermitian observable (which `C† O C` is whenever `O` is
/// Hermitian and `C` unitary) the coefficient lands at `±1`.
pub(crate) fn evaluate_pauli_observable_camps(
    prefix: &SignedCliffordPrefix,
    mps: &crate::backend::mps::MpsBackend,
    terms: &[crate::sim::unified_pauli::PauliTerm],
) -> crate::error::Result<f64> {
    use crate::sim::unified_pauli::PauliAxis;
    let n = prefix.num_qubits();
    let num_words = n.div_ceil(64).max(1);
    let mut combined = SignedPauli::zero(num_words);
    for term in terms {
        let row = match term.axis {
            PauliAxis::Z => prefix.conjugate_z(term.qubit),
            PauliAxis::X => prefix.conjugate_x(term.qubit),
            PauliAxis::Y => {
                let mut row = prefix.conjugate_x(term.qubit);
                rowmul_into(&mut row, &prefix.inv_z[term.qubit], n, 1);
                row
            }
        };
        rowmul_into(&mut combined, &row, n, 0);
    }
    let factors = combined.mps_factors(n);
    let p = u32::from(combined.phase4);
    let coef_re = match p {
        0 => 1.0,
        2 => -1.0,
        _ => {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "CAMPS observable: expected Hermitian (real ±1) twisted coefficient, \
                     got i^{p}"
                ),
            });
        }
    };
    let val = mps.pauli_expectation(&factors)?;
    Ok(coef_re * val.re)
}

/// Maximum relative state weight a CAMPS T-gate may discard to SVD truncation
/// before its result is rejected as inexact. Epsilon-threshold truncation
/// contributes `~svd_epsilon²` (negligible); a value above this means the MPS
/// bond-dim cap discarded real weight and the observable would be wrong.
const CAMPS_TRUNCATION_TOL: f64 = 1e-12;

/// Reject a CAMPS T-gate application that silently truncated state weight.
///
/// The MPS truncates inside `apply` (clamping bond dim to its cap) with no
/// signal, so peeking at the post-application bond dimension misses both
/// already-applied truncations and transient peaks that relaxed below the cap.
/// Reading the cumulative discarded weight catches every truncation since the
/// tracker was reset, regardless of the final bond dimension. The corrupted
/// state is discarded by erroring, letting the auto dispatcher fall back.
fn check_camps_truncation(
    mps: &crate::backend::mps::MpsBackend,
    target: usize,
) -> crate::error::Result<()> {
    let discarded = mps.truncation_discarded();
    if discarded > CAMPS_TRUNCATION_TOL {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!(
                "CAMPS T-gate on qubit {target}: disentangler cascade exceeded the MPS bond-dim \
                 cap and SVD truncation discarded {discarded:.3e} of the state weight. Raise \
                 `max_bond_dim` or use a less-entangling disentangler."
            ),
        });
    }
    Ok(())
}

/// Apply a T or Tdg gate to a CAMPS state `|ψ⟩ = C |ϕ⟩`.
///
/// The twisted Pauli `C† Z_target C` is reduced to single-qubit support with
/// a chosen disentangler, then absorbed as a one-qubit MPS rotation. Identity
/// support is a global phase, single-qubit support uses the direct rotation
/// when OFD cannot apply, and multi-qubit support chooses the cheaper OFD or
/// OFDS cascade. The truncation tracker is reset around each cascade so any
/// discarded weight becomes a hard error.
pub(crate) fn apply_t_via_camps(
    prefix: &mut SignedCliffordPrefix,
    mps: &mut crate::backend::mps::MpsBackend,
    target: usize,
    is_dagger: bool,
    tol: f64,
) -> crate::error::Result<()> {
    let z_bar = prefix.conjugate_z(target);
    let n_qubits = prefix.num_qubits();

    let support: Vec<usize> = (0..n_qubits)
        .filter(|&q| !matches!(z_bar.pauli_at(q), PauliKind::I))
        .collect();
    if support.is_empty() {
        return Ok(());
    }

    if support.len() == 1 {
        mps.reset_truncation_tracking();
        match build_ofd_disentangler(mps, &z_bar, n_qubits, tol)? {
            Some((n, cascade)) => {
                apply_cascade_and_rotate(prefix, mps, &cascade, n, target, is_dagger)?;
            }
            _ => {
                apply_single_qubit_rotation_to_mps(mps, &z_bar, support[0], is_dagger)?;
            }
        }
        return check_camps_truncation(mps, target);
    }

    match choose_disentangler(mps, &z_bar, n_qubits, tol)? {
        Some((n, cascade, _kind)) => {
            mps.reset_truncation_tracking();
            apply_cascade_and_rotate(prefix, mps, &cascade, n, target, is_dagger)?;
            check_camps_truncation(mps, target)
        }
        None => {
            let letters: String = (0..n_qubits)
                .map(|q| match z_bar.pauli_at(q) {
                    PauliKind::I => '.',
                    PauliKind::X => 'X',
                    PauliKind::Y => 'Y',
                    PauliKind::Z => 'Z',
                })
                .collect();
            Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "CAMPS T-gate on qubit {target}: invariant violation in disentangler dispatch. \
                     Twisted Pauli has support size {sz} (>=2 expected) at qubits {support:?} with letters \
                     `{letters}`, phase4={phase}. Both OFD and OFDS declined a multi-qubit support. \
                     Add an explicit fallback for this support pattern in `apply_t_via_camps`.",
                    sz = support.len(),
                    phase = z_bar.phase4,
                ),
            })
        }
    }
}

fn apply_cascade_and_rotate(
    prefix: &mut SignedCliffordPrefix,
    mps: &mut crate::backend::mps::MpsBackend,
    cascade: &[OfdGate],
    anchor_n: usize,
    target: usize,
    is_dagger: bool,
) -> crate::error::Result<()> {
    use crate::backend::Backend;
    use crate::circuit::{Instruction, SmallVec};

    for (gate, targets) in cascade {
        mps.apply(&Instruction::Gate {
            gate: gate.clone(),
            targets: SmallVec::from_slice(targets),
        })?;
    }

    for (gate, targets) in cascade.iter() {
        let inv = match gate {
            Gate::S => Gate::Sdg,
            Gate::Sdg => Gate::S,
            Gate::Cx | Gate::Cz => gate.clone(),
            other => {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!(
                        "CAMPS T-gate: cascade emitted unexpected gate {other:?} (no inverse rule)"
                    ),
                });
            }
        };
        prefix.fold_right_state_gate(&inv, targets).map_err(|e| {
            crate::error::PrismError::InvalidParameter {
                message: format!("CAMPS T-gate: prefix update failed: {e}"),
            }
        })?;
    }

    let new_z_bar = prefix.conjugate_z(target);
    let n_qubits = prefix.num_qubits();
    let stray: Vec<usize> = (0..n_qubits)
        .filter(|&q| q != anchor_n && !matches!(new_z_bar.pauli_at(q), PauliKind::I))
        .collect();
    if !stray.is_empty() {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!(
                "CAMPS T-gate on qubit {target}: disentangler did not concentrate the twisted \
                 Pauli onto anchor qubit {anchor_n}; residual support remains on qubits {stray:?}. \
                 The single-qubit rotation would silently drop those factors and corrupt the state."
            ),
        });
    }
    apply_single_qubit_rotation_to_mps(mps, &new_z_bar, anchor_n, is_dagger)
}

fn apply_single_qubit_rotation_to_mps(
    mps: &mut crate::backend::mps::MpsBackend,
    pauli: &SignedPauli,
    q: usize,
    is_dagger: bool,
) -> crate::error::Result<()> {
    use crate::backend::Backend;
    use crate::circuit::{Instruction, SmallVec};
    use num_complex::Complex64;

    let phase = match pauli.phase4 & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        3 => Complex64::new(0.0, -1.0),
        _ => unreachable!(),
    };
    let bx = pauli.get_x(q);
    let bz = pauli.get_z(q);
    let zc = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let op_at_q: [[Complex64; 2]; 2] = match (bx, bz) {
        (true, false) => [[zc, phase], [phase, zc]],
        (true, true) => [[zc, -i * phase], [i * phase, zc]],
        (false, true) => [[phase, zc], [zc, -phase]],
        _ => {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!("CAMPS rotation: Pauli at qubit {q} is identity; expected X/Y/Z"),
            });
        }
    };

    let alpha = (std::f64::consts::FRAC_PI_8).cos();
    let sin_pi8 = (std::f64::consts::FRAC_PI_8).sin();
    let beta = if is_dagger {
        Complex64::new(0.0, sin_pi8)
    } else {
        Complex64::new(0.0, -sin_pi8)
    };

    let alpha_c = Complex64::new(alpha, 0.0);
    let mat: [[Complex64; 2]; 2] = [
        [alpha_c + beta * op_at_q[0][0], beta * op_at_q[0][1]],
        [beta * op_at_q[1][0], alpha_c + beta * op_at_q[1][1]],
    ];

    mps.apply(&Instruction::Gate {
        gate: Gate::Fused(Box::new(mat)),
        targets: SmallVec::from_slice(&[q]),
    })?;

    Ok(())
}

#[cfg(test)]
#[path = "camps_prefix_tests.rs"]
mod tests;
