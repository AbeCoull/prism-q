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

    #[cfg(test)]
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
        if best.map_or(true, |(_, c)| cost < c) {
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

/// Evaluate `⟨ψ|Π_q Z_q|ψ⟩` for a CAMPS state `|ψ⟩ = C|ϕ⟩` where the
/// Clifford prefix `C` is tracked by `prefix` and the MPS holds `|ϕ⟩`.
///
/// Rewrites the observable as `⟨ϕ| C† (Π Z_q) C |ϕ⟩` by composing
/// `Π_q (C† Z_q C)` via signed-Pauli multiplication, then evaluates
/// the resulting Pauli string on the MPS via [`crate::backend::mps::MpsBackend::pauli_expectation`].
///
/// The composed string is canonicalized for the MPS evaluator: each
/// qubit's `(x, z)` bit pattern is mapped to letter `I`/`X`/`Y`/`Z`,
/// with the residual `(-i)` factor from rewriting `X·Z = -i·Y`
/// absorbed into the overall coefficient alongside the stored `i^phase4`.
/// For a Hermitian observable (which `C† O C` is whenever `O` is
/// Hermitian and `C` unitary) the coefficient lands at `±1`.
pub(crate) fn evaluate_z_observable_camps(
    prefix: &SignedCliffordPrefix,
    mps: &crate::backend::mps::MpsBackend,
    z_qubits: &[usize],
) -> crate::error::Result<f64> {
    let n = prefix.num_qubits();
    let num_words = n.div_ceil(64).max(1);
    let mut combined = SignedPauli::zero(num_words);
    for &q in z_qubits {
        let row = prefix.conjugate_z(q);
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
        if let Some((n, cascade)) = build_ofd_disentangler(mps, &z_bar, n_qubits, tol)? {
            apply_cascade_and_rotate(prefix, mps, &cascade, n, target, is_dagger)?;
        } else {
            apply_single_qubit_rotation_to_mps(mps, &z_bar, support[0], is_dagger)?;
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
mod tests {
    use super::*;
    use crate::backend::statevector::StatevectorBackend;
    use crate::backend::Backend;
    use crate::circuit::{Instruction, SmallVec};
    use num_complex::Complex64;

    fn pauli_action_on_basis(p: &SignedPauli, i: usize, n: usize) -> (Complex64, usize) {
        let mut j = i;
        let mut phase4: u32 = u32::from(p.phase4);
        for q in 0..n {
            let b = (i >> q) & 1;
            match p.pauli_at(q) {
                PauliKind::I => {}
                PauliKind::X => {
                    j ^= 1 << q;
                }
                PauliKind::Y => {
                    j ^= 1 << q;
                    phase4 = phase4.wrapping_add(if b == 0 { 1 } else { 3 });
                }
                PauliKind::Z => {
                    if b == 1 {
                        phase4 = phase4.wrapping_add(2);
                    }
                }
            }
        }
        let phase = match phase4 & 3 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };
        (phase, j)
    }

    fn pauli_string_expectation(state: &[Complex64], n: usize, p: &SignedPauli) -> Complex64 {
        let dim = 1usize << n;
        assert_eq!(state.len(), dim);
        let mut acc = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            let (phase, j) = pauli_action_on_basis(p, i, n);
            acc += state[j].conj() * (phase * state[i]);
        }
        acc
    }

    fn run_gates(n: usize, gates: &[(Gate, Vec<usize>)]) -> Vec<Complex64> {
        let mut backend = StatevectorBackend::new(0);
        backend.init(n, 0).unwrap();
        let instrs: Vec<Instruction> = gates
            .iter()
            .map(|(g, t)| Instruction::Gate {
                gate: g.clone(),
                targets: SmallVec::from_slice(t),
            })
            .collect();
        backend.apply_instructions(&instrs).unwrap();
        backend.export_statevector().unwrap()
    }

    fn randomizing_prefix(n: usize, seed: u64) -> Vec<(Gate, Vec<usize>)> {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut g = vec![(Gate::H, vec![0])];
        for q in 0..n {
            g.push((Gate::Ry(rng.random_range(0.3..2.7)), vec![q]));
            g.push((Gate::Rz(rng.random_range(0.3..2.7)), vec![q]));
            if q + 1 < n {
                g.push((Gate::Cx, vec![q, q + 1]));
            }
            g.push((Gate::Rx(rng.random_range(0.3..2.7)), vec![q]));
        }
        g
    }

    fn assert_consistent(n: usize, gates: &[(Gate, Vec<usize>)], seed: u64) {
        let pre_gates = randomizing_prefix(n, seed);
        let pre = run_gates(n, &pre_gates);
        let mut combined = pre_gates;
        combined.extend_from_slice(gates);
        let post = run_gates(n, &combined);

        let mut prefix = SignedCliffordPrefix::identity(n);
        for (g, t) in gates {
            prefix.apply_state_gate(g, t).unwrap();
        }
        for q in 0..n {
            let twisted_z = prefix.conjugate_z(q);
            let twisted_x = prefix.conjugate_x(q);
            let num_words = n.div_ceil(64).max(1);
            let mut z_only = SignedPauli::zero(num_words);
            z_only.set_z(q, true);
            let mut x_only = SignedPauli::zero(num_words);
            x_only.set_x(q, true);
            let lhs_z = pauli_string_expectation(&post, n, &z_only);
            let rhs_z = pauli_string_expectation(&pre, n, &twisted_z);
            assert!(
                (lhs_z - rhs_z).norm() < 1e-9,
                "Z_{q} mismatch: post={lhs_z} expected_from_prefix={rhs_z} (gates={gates:?})"
            );
            let lhs_x = pauli_string_expectation(&post, n, &x_only);
            let rhs_x = pauli_string_expectation(&pre, n, &twisted_x);
            assert!(
                (lhs_x - rhs_x).norm() < 1e-9,
                "X_{q} mismatch: post={lhs_x} expected_from_prefix={rhs_x} (gates={gates:?})"
            );
        }
    }

    #[test]
    fn identity_is_identity() {
        let p = SignedCliffordPrefix::identity(3);
        for q in 0..3 {
            assert_eq!(p.conjugate_z(q).pauli_at(q), PauliKind::Z);
            assert_eq!(p.conjugate_z(q).phase4, 0);
            assert_eq!(p.conjugate_x(q).pauli_at(q), PauliKind::X);
            assert_eq!(p.conjugate_x(q).phase4, 0);
        }
    }

    #[test]
    fn h_swaps_x_z() {
        assert_consistent(2, &[(Gate::H, vec![0])], 42);
        assert_consistent(2, &[(Gate::H, vec![1])], 42);
    }

    #[test]
    fn s_and_sdg_signs() {
        assert_consistent(2, &[(Gate::S, vec![0])], 42);
        assert_consistent(2, &[(Gate::Sdg, vec![0])], 42);
        assert_consistent(2, &[(Gate::S, vec![0]), (Gate::S, vec![0])], 42);
    }

    #[test]
    fn sx_and_sxdg() {
        assert_consistent(2, &[(Gate::SX, vec![0])], 42);
        assert_consistent(2, &[(Gate::SXdg, vec![0])], 42);
    }

    #[test]
    fn pauli_gates_only_flip_signs() {
        assert_consistent(2, &[(Gate::X, vec![0])], 42);
        assert_consistent(2, &[(Gate::Y, vec![0])], 42);
        assert_consistent(2, &[(Gate::Z, vec![0])], 42);
    }

    #[test]
    fn cx_and_cz() {
        assert_consistent(3, &[(Gate::Cx, vec![0, 1])], 42);
        assert_consistent(3, &[(Gate::Cz, vec![1, 2])], 42);
    }

    #[test]
    fn swap_permutes_columns() {
        assert_consistent(3, &[(Gate::Swap, vec![0, 2])], 42);
    }

    #[test]
    fn compound_h_cx() {
        assert_consistent(3, &[(Gate::H, vec![0]), (Gate::Cx, vec![0, 1])], 7);
    }

    #[test]
    fn compound_h_cx_s() {
        assert_consistent(
            3,
            &[
                (Gate::H, vec![0]),
                (Gate::Cx, vec![0, 1]),
                (Gate::S, vec![1]),
            ],
            7,
        );
    }

    #[test]
    fn compound_h_cx_s_cz() {
        assert_consistent(
            3,
            &[
                (Gate::H, vec![0]),
                (Gate::Cx, vec![0, 1]),
                (Gate::S, vec![1]),
                (Gate::Cz, vec![1, 2]),
            ],
            7,
        );
    }

    #[test]
    fn random_clifford_sequence_4q() {
        let seq = vec![
            (Gate::H, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::S, vec![1]),
            (Gate::Cz, vec![1, 2]),
            (Gate::H, vec![3]),
            (Gate::Cx, vec![2, 3]),
            (Gate::Sdg, vec![2]),
            (Gate::SX, vec![3]),
            (Gate::Z, vec![0]),
            (Gate::Swap, vec![0, 3]),
            (Gate::Y, vec![2]),
            (Gate::SXdg, vec![1]),
        ];
        assert_consistent(4, &seq, 17);
        assert_consistent(4, &seq, 31);
    }

    fn make_pauli(n: usize, factors: &[(usize, PauliKind)]) -> SignedPauli {
        let nw = n.div_ceil(64).max(1);
        let mut p = SignedPauli::zero(nw);
        for &(q, kind) in factors {
            match kind {
                PauliKind::I => {}
                PauliKind::X => p.set_x(q, true),
                PauliKind::Y => {
                    p.set_x(q, true);
                    p.set_z(q, true);
                }
                PauliKind::Z => p.set_z(q, true),
            }
        }
        p
    }

    fn empty_mps(n: usize) -> crate::backend::mps::MpsBackend {
        let mut b = crate::backend::mps::MpsBackend::new(0, 64);
        b.init(n, 0).unwrap();
        b
    }

    #[test]
    fn ofd_no_xy_returns_none() {
        let mps = empty_mps(3);
        let p = make_pauli(3, &[(0, PauliKind::Z), (1, PauliKind::Z)]);
        assert!(build_ofd_disentangler(&mps, &p, 3, 1e-10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn ofd_x_only_no_other_support_returns_empty_cascade() {
        let mps = empty_mps(3);
        let p = make_pauli(3, &[(0, PauliKind::X)]);
        let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
        assert_eq!(n, 0);
        assert!(d.is_empty());
    }

    #[test]
    fn ofd_x0_z1_builds_cz() {
        let mps = empty_mps(3);
        let p = make_pauli(3, &[(0, PauliKind::X), (1, PauliKind::Z)]);
        let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
        assert_eq!(n, 0);
        assert_eq!(d.len(), 1);
        assert!(matches!(d[0].0, Gate::Cz));
        assert_eq!(d[0].1, vec![0, 1]);
    }

    #[test]
    fn ofd_x0_x1_builds_cx() {
        let mps = empty_mps(3);
        let p = make_pauli(3, &[(0, PauliKind::X), (1, PauliKind::X)]);
        let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
        assert_eq!(n, 0);
        assert_eq!(d.len(), 1);
        assert!(matches!(d[0].0, Gate::Cx));
        assert_eq!(d[0].1, vec![0, 1]);
    }

    #[test]
    fn ofd_x0_y1_builds_cy_decomposition() {
        let mps = empty_mps(3);
        let p = make_pauli(3, &[(0, PauliKind::X), (1, PauliKind::Y)]);
        let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
        assert_eq!(n, 0);
        assert_eq!(d.len(), 3);
        assert!(matches!(d[0].0, Gate::Sdg));
        assert_eq!(d[0].1, vec![1]);
        assert!(matches!(d[1].0, Gate::Cx));
        assert_eq!(d[1].1, vec![0, 1]);
        assert!(matches!(d[2].0, Gate::S));
        assert_eq!(d[2].1, vec![1]);
    }

    #[test]
    fn ofd_multi_target_x_z() {
        let mps = empty_mps(4);
        let p = make_pauli(
            4,
            &[
                (0, PauliKind::X),
                (1, PauliKind::X),
                (2, PauliKind::Z),
                (3, PauliKind::I),
            ],
        );
        let (n, d) = build_ofd_disentangler(&mps, &p, 4, 1e-10).unwrap().unwrap();
        assert_eq!(n, 1, "anchor should sit at the routing-cost minimum");
        assert_eq!(d.len(), 2);
        assert!(matches!(d[0].0, Gate::Cx) && d[0].1 == vec![1, 0]);
        assert!(matches!(d[1].0, Gate::Cz) && d[1].1 == vec![1, 2]);
    }

    #[test]
    fn ofd_skips_qubit_not_in_zero_state() {
        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(3, 0).unwrap();
        mps.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: SmallVec::from_slice(&[0]),
        })
        .unwrap();
        let p = make_pauli(
            3,
            &[(0, PauliKind::X), (1, PauliKind::Y), (2, PauliKind::Z)],
        );
        let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
        assert_eq!(n, 1);
        assert_eq!(d.len(), 2);
        assert!(matches!(d[0].0, Gate::Cx) && d[0].1 == vec![1, 0]);
        assert!(matches!(d[1].0, Gate::Cz) && d[1].1 == vec![1, 2]);
    }

    #[test]
    fn ofd_post_apply_target_qubit_is_disentangled_zero() {
        let mut mps = empty_mps(3);
        mps.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[1]),
        })
        .unwrap();
        mps.apply(&Instruction::Gate {
            gate: Gate::Cx,
            targets: SmallVec::from_slice(&[1, 2]),
        })
        .unwrap();
        let p = make_pauli(
            3,
            &[(0, PauliKind::X), (1, PauliKind::X), (2, PauliKind::X)],
        );
        let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
        assert_eq!(n, 0);
        for (gate, targets) in &d {
            mps.apply(&Instruction::Gate {
                gate: gate.clone(),
                targets: SmallVec::from_slice(targets),
            })
            .unwrap();
        }
        assert!(mps.is_qubit_in_zero_state(0, 1e-10).unwrap());
    }

    // apply_t_via_camps end-to-end tests

    fn direct_state(n: usize, gates: &[(Gate, Vec<usize>)]) -> Vec<Complex64> {
        let mut sb = StatevectorBackend::new(0);
        sb.init(n, 0).unwrap();
        for (gate, targets) in gates {
            sb.apply(&Instruction::Gate {
                gate: gate.clone(),
                targets: SmallVec::from_slice(targets),
            })
            .unwrap();
        }
        sb.export_statevector().unwrap()
    }

    fn run_camps_then_t(
        n: usize,
        prep: &[(Gate, Vec<usize>)],
        t_target: usize,
        t_dagger: bool,
    ) -> (
        SignedCliffordPrefix,
        Vec<(Gate, Vec<usize>)>,
        crate::backend::mps::MpsBackend,
    ) {
        let mut prefix = SignedCliffordPrefix::identity(n);
        let mut prefix_gates: Vec<(Gate, Vec<usize>)> = Vec::new();
        for (gate, targets) in prep {
            prefix.apply_state_gate(gate, targets).unwrap();
            prefix_gates.push((gate.clone(), targets.clone()));
        }
        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        apply_t_via_camps(&mut prefix, &mut mps, t_target, t_dagger, 1e-10).unwrap();
        // After call, prefix has been updated to C·D† via fold_right_state_gate.
        // The tracker doesn't store gates, so callers that need to replay
        // the prefix onto a materialized state compare via Pauli expectations
        // instead of gate-list reconstruction.
        (prefix, prefix_gates, mps)
    }

    #[test]
    fn t_via_camps_identity_prefix_falls_back_to_direct_t() {
        // C = I, so Z̄ = Z_target. OFD finds no disentangler, so the
        // pure-Z fallback applies T directly on the target qubit.
        let n = 2;
        let mut prefix = SignedCliffordPrefix::identity(n);
        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();

        let direct = direct_state(n, &[(Gate::T, vec![0])]);
        for q in 0..n {
            let zbar = prefix.conjugate_z(q);
            let factors = zbar.mps_factors(n);
            let zc = mps.pauli_expectation(&factors).unwrap();
            let phase_sign = match zbar.phase4 & 3 {
                0 => 1.0,
                2 => -1.0,
                _ => panic!("non-real Z̄ phase {}", zbar.phase4),
            };
            let z_camps = phase_sign * zc.re;
            let z_direct: f64 = direct
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * amp.norm_sqr()
                })
                .sum();
            assert!(
                (z_camps - z_direct).abs() < 1e-9,
                "Z_{q}: camps={z_camps} direct={z_direct}"
            );
        }
    }

    #[test]
    fn t_via_camps_with_h_prefix_matches_direct() {
        // Prep: H on qubit 0. Then T on qubit 0.
        // Direct: H, T on statevector.
        // CAMPS: prefix absorbs H, then apply_t_via_camps(0, T).
        let n = 2;
        let prep = vec![(Gate::H, vec![0])];
        let (prefix, _, mps) = run_camps_then_t(n, &prep, 0, false);

        // Prefix unitary is now H·... folded right by D†, giving C·D†.
        // Reconstructing C from the inverse tableau is impractical, so
        // this test compares via Pauli expectations instead of replay.
        let direct = direct_state(n, &[(Gate::H, vec![0]), (Gate::T, vec![0])]);

        // CAMPS oracle: ⟨ψ|Z_0|ψ⟩ = ⟨ϕ|C†Z_0C|ϕ⟩ = ⟨ϕ|Z̄'|ϕ⟩
        // where Z̄' = (new_C)†·Z_0·(new_C). Use prefix.conjugate_z(0)
        // and evaluate via mps.pauli_expectation.
        let zbar = prefix.conjugate_z(0);
        let factors = zbar.mps_factors(n);
        let z0_camps = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!(
                "expected Hermitian Z̄ with phase4∈{{0,2}}, got {}",
                zbar.phase4
            ),
        };
        let z0_camps_real = phase_sign * z0_camps.re;

        let z0_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();

        assert!(
            (z0_camps_real - z0_direct).abs() < 1e-9,
            "Z_0: camps={z0_camps_real} direct={z0_direct}"
        );
    }

    #[test]
    fn t_via_camps_single_y_twisted_pauli_matches_direct() {
        let n = 1;
        let mut prefix = SignedCliffordPrefix::identity(n);
        prefix.apply_state_gate(&Gate::SX, &[0]).unwrap();

        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
        prefix.apply_state_gate(&Gate::H, &[0]).unwrap();

        let z_camps = evaluate_z_observable_camps(&prefix, &mps, &[0]).unwrap();
        let direct = direct_state(
            n,
            &[(Gate::SX, vec![0]), (Gate::T, vec![0]), (Gate::H, vec![0])],
        );
        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();

        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "single-Y twisted Pauli: camps={z_camps} direct={z_direct}"
        );
    }

    #[test]
    fn t_via_camps_h_cx_prefix_multi_qubit_pauli() {
        // Prep: H_0, CX(0,1). Then T on qubit 0.
        // Z̄_0 = (H_0 CX_01)† Z_0 (H_0 CX_01) = H_0 Z_0 H_0 ⊗ ... etc.
        // Actually: (CX)†(H†) Z_0 (H)(CX). H Z_0 H = X_0. CX X_0 CX = X_0 X_1.
        // So Z̄ = X_0 X_1, letter at 0 is X, qubit 0 in |0⟩ (yes, fresh init), OFD succeeds.
        let n = 2;
        let prep = vec![(Gate::H, vec![0]), (Gate::Cx, vec![0, 1])];
        let (prefix, _, mps) = run_camps_then_t(n, &prep, 0, false);

        let direct = direct_state(
            n,
            &[
                (Gate::H, vec![0]),
                (Gate::Cx, vec![0, 1]),
                (Gate::T, vec![0]),
            ],
        );

        for q in 0..n {
            let zbar = prefix.conjugate_z(q);
            let factors = zbar.mps_factors(n);
            let zc = mps.pauli_expectation(&factors).unwrap();
            let phase_sign = match zbar.phase4 & 3 {
                0 => 1.0,
                2 => -1.0,
                _ => panic!("non-real Z̄ phase {}", zbar.phase4),
            };
            let z_camps = phase_sign * zc.re;

            let z_direct: f64 = direct
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * amp.norm_sqr()
                })
                .sum();

            assert!(
                (z_camps - z_direct).abs() < 1e-9,
                "Z_{q}: camps={z_camps} direct={z_direct}"
            );
        }
    }

    // deeper multi-qubit OFD success-path coverage

    #[test]
    fn t_via_camps_ofd_3q_mixed_cascade_matches_direct() {
        // Build a prefix that produces a 3-qubit twisted Pauli with
        // mixed X/Y/Z letters anchored on a |0⟩ qubit. The fresh-init
        // MPS has every qubit in |0⟩, so OFD's anchor selection picks
        // the first X/Y letter and emits a non-trivial cascade spanning
        // all remaining qubits (CX + CZ + Sdg/CX/S triplet for Y).
        //
        // Prefix construction: H_1, CX(1,0), CX(1,2), S_2.
        //   prefix C = S_2 · CX(1,2) · CX(1,0) · H_1
        // Twisted Z_0:
        //   C† Z_0 C = H_1 · CX(1,0)† · CX(1,2)† · S_2† · Z_0 · S_2 · CX(1,2) · CX(1,0) · H_1
        // S_2 commutes with Z_0 (different qubit), so reduces to
        //   H_1 · CX(1,0) · CX(1,2) · Z_0 · CX(1,2) · CX(1,0) · H_1
        // CX(1,2) commutes with Z_0 (different qubits), so
        //   H_1 · CX(1,0) · Z_0 · CX(1,0) · H_1 = H_1 · Z_0 Z_1 · H_1 = Z_0 X_1
        // That setup only gives 2-qubit support, so use a richer prefix.
        //
        // Use: H_0, CX(0,1), CX(0,2), S_1. Then for T_target=0:
        //   C = S_1 · CX(0,2) · CX(0,1) · H_0
        //   C† Z_0 C = H_0 · CX(0,1) · CX(0,2) · S_1† · Z_0 · S_1 · CX(0,2) · CX(0,1) · H_0
        //   S_1 commutes with Z_0
        //   CX(0,2)† Z_0 CX(0,2) = Z_0 (control Z unchanged)
        //   Still 1-qubit Z, then H_0 gives X_0, a single-qubit Pauli.
        //
        // Need to use CX on the *target*-qubit position. T on qubit 2:
        //   C† Z_2 C, with C above:
        //   S_1† CX(0,2)† Z_2 CX(0,2) S_1 = S_1† (Z_0 Z_2) S_1 = Z_0 Z_2 (S commutes with Z)
        //   CX(0,1)† (Z_0 Z_2) CX(0,1) = Z_0 Z_2 (Z_0 unchanged by CX(0,1) control)
        //   H_0 (Z_0 Z_2) H_0 = X_0 Z_2, still 2-qubit.
        //
        // To get 3-qubit, need to fan through both Hadamard and CX
        // structure. Use: H_0, H_2, CX(0,1), CX(2,1), S_1; T on 1.
        //   C = S_1 · CX(2,1) · CX(0,1) · H_2 · H_0
        //   C† Z_1 C = (left-to-right inverse) H_0 H_2 CX(0,1) CX(2,1) S_1† Z_1 S_1 CX(2,1) CX(0,1) H_2 H_0
        //   S_1† Z_1 S_1 = Z_1
        //   CX(2,1)† Z_1 CX(2,1) = Z_1 Z_2
        //   CX(0,1)† (Z_1 Z_2) CX(0,1) = Z_0 Z_1 Z_2 (CX flips Z on target → Z_c Z_t)
        //   H_2 (Z_0 Z_1 Z_2) H_2 = Z_0 Z_1 X_2
        //   H_0 (Z_0 Z_1 X_2) H_0 = X_0 Z_1 X_2, 3-qubit mixed letters.
        // OFD on this: first X/Y letter is q=0 (X). mps[0] is a valid |0> anchor.
        // Cascade emits CZ(0,1) and CX(0,2). Non-trivial 3-qubit OFD.
        let n = 3;
        let mut prefix = SignedCliffordPrefix::identity(n);
        for (g, t) in [
            (Gate::H, vec![0]),
            (Gate::H, vec![2]),
            (Gate::Cx, vec![0, 1]),
            (Gate::Cx, vec![2, 1]),
            (Gate::S, vec![1]),
        ] {
            prefix.apply_state_gate(&g, &t).unwrap();
        }

        let zbar_pre = prefix.conjugate_z(1);
        assert_eq!(zbar_pre.pauli_at(0), PauliKind::X);
        assert_eq!(zbar_pre.pauli_at(1), PauliKind::Z);
        assert_eq!(zbar_pre.pauli_at(2), PauliKind::X);

        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        // Verify OFD succeeds (fresh MPS, all qubits in |0⟩, X letter at qubit 0).
        let (anchor, cascade) = build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
            .unwrap()
            .unwrap();
        assert_eq!(anchor, 0);
        assert_eq!(cascade.len(), 2);
        assert!(matches!(cascade[0].0, Gate::Cz) && cascade[0].1 == vec![0, 1]);
        assert!(matches!(cascade[1].0, Gate::Cx) && cascade[1].1 == vec![0, 2]);

        apply_t_via_camps(&mut prefix, &mut mps, 1, false, 1e-10).unwrap();

        let direct = direct_state(
            n,
            &[
                (Gate::H, vec![0]),
                (Gate::H, vec![2]),
                (Gate::Cx, vec![0, 1]),
                (Gate::Cx, vec![2, 1]),
                (Gate::S, vec![1]),
                (Gate::T, vec![1]),
            ],
        );

        for q in 0..n {
            let zbar = prefix.conjugate_z(q);
            let factors = zbar.mps_factors(n);
            let zc = mps.pauli_expectation(&factors).unwrap();
            let phase_sign = match zbar.phase4 & 3 {
                0 => 1.0,
                2 => -1.0,
                _ => panic!("non-real Z̄ phase {}", zbar.phase4),
            };
            let z_camps = phase_sign * zc.re;
            let z_direct: f64 = direct
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * amp.norm_sqr()
                })
                .sum();
            assert!(
                (z_camps - z_direct).abs() < 1e-9,
                "OFD-3q Z_{q}: camps={z_camps} direct={z_direct}"
            );
        }
    }

    #[test]
    fn t_via_camps_ofd_with_y_letter_triplet_cascade() {
        // Force a Y letter to land in the twisted Pauli so OFD emits
        // the (Sdg, CX, S) Y-decomposition triplet.
        //
        // Prefix: S_1, H_0, CX(0,1). For T on qubit 0:
        //   C = CX(0,1) · H_0 · S_1
        //   C† Z_0 C = S_1† H_0 CX(0,1)† Z_0 CX(0,1) H_0 S_1
        //   CX(0,1)† Z_0 CX(0,1) = Z_0
        //   H_0 Z_0 H_0 = X_0
        //   S_1† X_0 S_1 = X_0, still single-qubit. Won't trigger Y.
        //
        // Different angle: prefix S_1, CX(0,1), H_1. T on qubit 1.
        //   C = H_1 · CX(0,1) · S_1
        //   C† Z_1 C = S_1† CX(0,1)† H_1 Z_1 H_1 CX(0,1) S_1
        //   H_1 Z_1 H_1 = X_1
        //   CX(0,1)† X_1 CX(0,1) = X_1 (target X unchanged by CX with target = X_1's qubit? no, X_target → X_target stays, control unchanged for X_t)
        //   Actually CX(c,t) X_t CX(c,t) = X_t. So X_1 stays.
        //   S_1† X_1 S_1 = -Y_1. So Z̄ = -Y_1, a single-qubit Y that would use single-qubit fallback, not OFD.
        //
        // For multi-qubit with Y letter, need a more layered prefix.
        // Use: H_0, CX(0,1), S_1, CX(0,2), H_2. T on 2.
        //   C = H_2 · CX(0,2) · S_1 · CX(0,1) · H_0
        //   C† Z_2 C = H_0 CX(0,1) S_1† CX(0,2)† H_2 Z_2 H_2 CX(0,2) S_1 CX(0,1) H_0
        //   H_2 Z_2 H_2 = X_2
        //   CX(0,2)† X_2 CX(0,2) = X_2 (target X unchanged by CX, but CX(c,t) X_t = X_t. So X_2 stays)
        //   S_1† X_2 S_1 = X_2 (different qubit)
        //   CX(0,1)† X_2 CX(0,1) = X_2 (different qubits)
        //   H_0 X_2 H_0 = X_2, single-qubit Z̄ = X_2. Trivial.
        //
        // The issue: CX(0,2)·X_2 doesn't grow because target X is invariant.
        // To get growth on the target side, the inner gate at the target
        // must be Z (or Y), not X. Re-pick: H_2 → not at end.
        //
        // Try: CX(0,1), H_1, CX(1,2), S_2, H_2. T on 2.
        //   C = H_2 · S_2 · CX(1,2) · H_1 · CX(0,1)
        //   C† Z_2 C = CX(0,1) H_1 CX(1,2)† S_2† H_2 Z_2 H_2 S_2 CX(1,2) H_1 CX(0,1)
        //   H_2 Z_2 H_2 = X_2
        //   Sdg_2 X_2 S_2 = -Y_2 by direct multiplication.
        //   So -Y_2 after S†.
        //   CX(1,2)† (-Y_2) CX(1,2) = -CX(1,2)† Y_2 CX(1,2). Y_2 = i X_2 Z_2. CX(1,2) maps X_2 → X_2, Z_2 → Z_1 Z_2. So Y_2 → i X_2 Z_1 Z_2 = Z_1 · i X_2 Z_2 = Z_1 Y_2.
        //   So get -Z_1 Y_2.
        //   H_1 (-Z_1 Y_2) H_1 = -X_1 Y_2.
        //   CX(0,1)† (-X_1 Y_2) CX(0,1) = -CX(0,1)† X_1 CX(0,1) · Y_2 = -X_1 · Y_2 (X_t unchanged) = -X_1 Y_2.
        //   So Zbar = -X_1 Y_2, phase4=2. Letters at (0,1,2): I, X, Y.
        // OFD: first X/Y is q=1 (X). mps[1] is a valid |0> anchor. Cascade emits triplet for Y at q=2.
        let n = 3;
        let mut prefix = SignedCliffordPrefix::identity(n);
        for (g, t) in [
            (Gate::Cx, vec![0, 1]),
            (Gate::H, vec![1]),
            (Gate::Cx, vec![1, 2]),
            (Gate::S, vec![2]),
            (Gate::H, vec![2]),
        ] {
            prefix.apply_state_gate(&g, &t).unwrap();
        }

        let zbar_pre = prefix.conjugate_z(2);
        assert_eq!(zbar_pre.pauli_at(0), PauliKind::I);
        assert_eq!(zbar_pre.pauli_at(1), PauliKind::X);
        assert_eq!(zbar_pre.pauli_at(2), PauliKind::Y);
        assert_eq!(zbar_pre.phase4, 2);

        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        let (anchor, cascade) = build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
            .unwrap()
            .unwrap();
        assert_eq!(anchor, 1);
        assert_eq!(
            cascade.len(),
            3,
            "Y partner should emit (Sdg, CX, S) triplet"
        );
        assert!(matches!(cascade[0].0, Gate::Sdg) && cascade[0].1 == vec![2]);
        assert!(matches!(cascade[1].0, Gate::Cx) && cascade[1].1 == vec![1, 2]);
        assert!(matches!(cascade[2].0, Gate::S) && cascade[2].1 == vec![2]);

        apply_t_via_camps(&mut prefix, &mut mps, 2, false, 1e-10).unwrap();

        let direct = direct_state(
            n,
            &[
                (Gate::Cx, vec![0, 1]),
                (Gate::H, vec![1]),
                (Gate::Cx, vec![1, 2]),
                (Gate::S, vec![2]),
                (Gate::H, vec![2]),
                (Gate::T, vec![2]),
            ],
        );

        for q in 0..n {
            let zbar = prefix.conjugate_z(q);
            let factors = zbar.mps_factors(n);
            let zc = mps.pauli_expectation(&factors).unwrap();
            let phase_sign = match zbar.phase4 & 3 {
                0 => 1.0,
                2 => -1.0,
                _ => panic!("non-real Z̄ phase {}", zbar.phase4),
            };
            let z_camps = phase_sign * zc.re;
            let z_direct: f64 = direct
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * amp.norm_sqr()
                })
                .sum();
            assert!(
                (z_camps - z_direct).abs() < 1e-9,
                "OFD-Y-triplet Z_{q}: camps={z_camps} direct={z_direct}"
            );
        }
    }

    // OFDS tests

    #[test]
    fn ofds_x_anchor_no_zero_state_requirement() {
        let mut mps = empty_mps(2);
        mps.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: SmallVec::from_slice(&[0]),
        })
        .unwrap();
        mps.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: SmallVec::from_slice(&[1]),
        })
        .unwrap();
        let p = make_pauli(2, &[(0, PauliKind::X), (1, PauliKind::Z)]);
        assert!(build_ofd_disentangler(&mps, &p, 2, 1e-10)
            .unwrap()
            .is_none());
        let (n, d) = build_ofds_disentangler(&mps, &p, 2).unwrap();
        assert_eq!(n, 0);
        assert_eq!(d.len(), 1);
        assert!(matches!(d[0].0, Gate::Cz) && d[0].1 == vec![0, 1]);
    }

    fn fresh_mps(n: usize) -> crate::backend::mps::MpsBackend {
        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        mps
    }

    #[test]
    fn ofds_all_z_anchor_minimizes_routing() {
        let p = make_pauli(
            4,
            &[
                (0, PauliKind::Z),
                (1, PauliKind::I),
                (2, PauliKind::Z),
                (3, PauliKind::Z),
            ],
        );
        let mps = fresh_mps(4);
        let (n, d) = build_ofds_disentangler(&mps, &p, 4).unwrap();
        assert_eq!(n, 2, "anchor should sit at the routing-cost minimum");
        assert_eq!(d.len(), 2);
        assert!(matches!(d[0].0, Gate::Cx) && d[0].1 == vec![0, 2]);
        assert!(matches!(d[1].0, Gate::Cx) && d[1].1 == vec![3, 2]);
    }

    #[test]
    fn ofds_single_z_returns_none() {
        let p = make_pauli(3, &[(1, PauliKind::Z)]);
        let mps = fresh_mps(3);
        assert!(build_ofds_disentangler(&mps, &p, 3).is_none());
    }

    #[test]
    fn ofds_empty_returns_none() {
        let p = SignedPauli::zero(1);
        let mps = fresh_mps(3);
        assert!(build_ofds_disentangler(&mps, &p, 3).is_none());
    }

    #[test]
    fn t_via_camps_ofds_xy_path_matches_direct() {
        let n = 2;
        let mut prefix = SignedCliffordPrefix::identity(n);
        prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
        prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();

        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        mps.apply(&Instruction::Gate {
            gate: Gate::Ry(0.7),
            targets: SmallVec::from_slice(&[0]),
        })
        .unwrap();
        mps.apply(&Instruction::Gate {
            gate: Gate::Ry(0.5),
            targets: SmallVec::from_slice(&[1]),
        })
        .unwrap();

        let zbar_pre = prefix.conjugate_z(0);
        assert!(
            build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
                .unwrap()
                .is_none(),
            "test setup error: OFD should have failed so OFDS path is exercised"
        );

        apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();

        let direct = direct_state(
            n,
            &[
                (Gate::Ry(0.7), vec![0]),
                (Gate::Ry(0.5), vec![1]),
                (Gate::H, vec![0]),
                (Gate::Cx, vec![0, 1]),
                (Gate::T, vec![0]),
            ],
        );

        for q in 0..n {
            let zbar = prefix.conjugate_z(q);
            let factors = zbar.mps_factors(n);
            let zc = mps.pauli_expectation(&factors).unwrap();
            let phase_sign = match zbar.phase4 & 3 {
                0 => 1.0,
                2 => -1.0,
                _ => panic!("non-real Z̄ phase {}", zbar.phase4),
            };
            let z_camps = phase_sign * zc.re;
            let z_direct: f64 = direct
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * amp.norm_sqr()
                })
                .sum();
            assert!(
                (z_camps - z_direct).abs() < 1e-9,
                "OFDS Z_{q}: camps={z_camps} direct={z_direct}"
            );
        }
    }

    #[test]
    fn t_via_camps_ofds_all_z_path_matches_direct() {
        let n = 3;
        let mut prefix = SignedCliffordPrefix::identity(n);
        prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
        prefix.apply_state_gate(&Gate::Cx, &[1, 2]).unwrap();

        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();

        let zbar_pre = prefix.conjugate_z(2);
        for q in 0..n {
            assert!(
                matches!(zbar_pre.pauli_at(q), PauliKind::Z),
                "test setup: expected all-Z twisted Pauli, got {:?} at q={q}",
                zbar_pre.pauli_at(q)
            );
        }
        assert!(
            build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
                .unwrap()
                .is_none(),
            "test setup: OFD should reject all-Z support"
        );

        apply_t_via_camps(&mut prefix, &mut mps, 2, false, 1e-10).unwrap();

        let direct = direct_state(
            n,
            &[
                (Gate::Cx, vec![0, 1]),
                (Gate::Cx, vec![1, 2]),
                (Gate::T, vec![2]),
            ],
        );

        for q in 0..n {
            let zbar = prefix.conjugate_z(q);
            let factors = zbar.mps_factors(n);
            let zc = mps.pauli_expectation(&factors).unwrap();
            let phase_sign = match zbar.phase4 & 3 {
                0 => 1.0,
                2 => -1.0,
                _ => panic!("non-real Z̄ phase {}", zbar.phase4),
            };
            let z_camps = phase_sign * zc.re;
            let z_direct: f64 = direct
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * amp.norm_sqr()
                })
                .sum();
            assert!(
                (z_camps - z_direct).abs() < 1e-9,
                "OFDS all-Z Z_{q}: camps={z_camps} direct={z_direct}"
            );
        }
    }

    #[test]
    fn tdag_via_camps_h_prefix() {
        let n = 2;
        let prep = vec![(Gate::H, vec![0])];
        let (prefix, _, mps) = run_camps_then_t(n, &prep, 0, true);

        let direct = direct_state(n, &[(Gate::H, vec![0]), (Gate::Tdg, vec![0])]);

        for q in 0..n {
            let zbar = prefix.conjugate_z(q);
            let factors = zbar.mps_factors(n);
            let zc = mps.pauli_expectation(&factors).unwrap();
            let phase_sign = match zbar.phase4 & 3 {
                0 => 1.0,
                2 => -1.0,
                _ => panic!("non-real Z̄ phase {}", zbar.phase4),
            };
            let z_camps = phase_sign * zc.re;
            let z_direct: f64 = direct
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * amp.norm_sqr()
                })
                .sum();
            assert!(
                (z_camps - z_direct).abs() < 1e-9,
                "Tdg Z_{q}: camps={z_camps} direct={z_direct}"
            );
        }
    }

    fn two_t_zz_probe(post_cliffords: &[(Gate, Vec<usize>)]) -> (f64, f64) {
        let n = 2;
        let mut prefix = SignedCliffordPrefix::identity(n);
        prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
        prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
        prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
        apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
        for (g, t) in post_cliffords {
            prefix.apply_state_gate(g, t).unwrap();
        }
        let zz_camps = evaluate_z_observable_camps(&prefix, &mps, &[0, 1]).unwrap();
        let mut gates: Vec<(Gate, Vec<usize>)> = vec![
            (Gate::H, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::T, vec![0]),
            (Gate::H, vec![0]),
            (Gate::T, vec![0]),
        ];
        gates.extend_from_slice(post_cliffords);
        let direct = direct_state(n, &gates);
        let zz_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let s0 = if i & 1 == 0 { 1.0 } else { -1.0 };
                let s1 = if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 };
                s0 * s1 * amp.norm_sqr()
            })
            .sum();
        (zz_camps, zz_direct)
    }

    #[test]
    fn t_via_camps_two_t_bisect_post_cliffords() {
        type BisectCase<'a> = (&'a str, &'a [(Gate, &'a [usize])]);
        let cases: &[BisectCase] = &[
            ("no_post", &[]),
            ("h0", &[(Gate::H, &[0])]),
            ("h1", &[(Gate::H, &[1])]),
            ("h0_h1", &[(Gate::H, &[0]), (Gate::H, &[1])]),
            ("h1_h0", &[(Gate::H, &[1]), (Gate::H, &[0])]),
            ("x0_x1", &[(Gate::X, &[0]), (Gate::X, &[1])]),
            ("z0_z1", &[(Gate::Z, &[0]), (Gate::Z, &[1])]),
            ("cx_01", &[(Gate::Cx, &[0, 1])]),
            ("s0_s1", &[(Gate::S, &[0]), (Gate::S, &[1])]),
        ];
        let mut failures = Vec::new();
        for (label, ops) in cases {
            let owned: Vec<(Gate, Vec<usize>)> =
                ops.iter().map(|(g, t)| (g.clone(), t.to_vec())).collect();
            let (c, d) = two_t_zz_probe(&owned);
            if (c - d).abs() >= 1e-9 {
                failures.push(format!("{label}: camps={c} direct={d}"));
            }
        }
        assert!(failures.is_empty(), "fails:\n  {}", failures.join("\n  "));
    }

    #[test]
    fn t_via_camps_two_t_multi_qubit_zz_observable() {
        // Reproduces the prior multi-qubit two-T failure shape on a small fixture
        // and checks the joint ⟨Z_0 Z_1⟩ against statevector.
        // Sequence: H_0; CX(0,1); T_0; H_0; T_0
        let n = 2;
        let mut prefix = SignedCliffordPrefix::identity(n);
        prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
        prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();

        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();

        apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
        prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
        apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
        prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
        prefix.apply_state_gate(&Gate::H, &[1]).unwrap();

        let zz_camps = evaluate_z_observable_camps(&prefix, &mps, &[0, 1]).unwrap();

        let direct = direct_state(
            n,
            &[
                (Gate::H, vec![0]),
                (Gate::Cx, vec![0, 1]),
                (Gate::T, vec![0]),
                (Gate::H, vec![0]),
                (Gate::T, vec![0]),
                (Gate::H, vec![0]),
                (Gate::H, vec![1]),
            ],
        );
        let zz_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let s0 = if i & 1 == 0 { 1.0 } else { -1.0 };
                let s1 = if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 };
                s0 * s1 * amp.norm_sqr()
            })
            .sum();

        assert!(
            (zz_camps - zz_direct).abs() < 1e-9,
            "two-T ⟨Z_0 Z_1⟩: camps={zz_camps} direct={zz_direct}"
        );
    }

    #[test]
    fn t_via_camps_bond_dim_stays_bounded_on_ofd_and_ofds_paths() {
        let max_bond = 64;

        let n_ofd = 3;
        let mut prefix_ofd = SignedCliffordPrefix::identity(n_ofd);
        prefix_ofd.apply_state_gate(&Gate::H, &[0]).unwrap();
        prefix_ofd.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
        prefix_ofd.apply_state_gate(&Gate::Cx, &[1, 2]).unwrap();
        let mut mps_ofd = crate::backend::mps::MpsBackend::new(0, max_bond);
        mps_ofd.init(n_ofd, 0).unwrap();

        let zbar_ofd = prefix_ofd.conjugate_z(0);
        assert!(
            build_ofd_disentangler(&mps_ofd, &zbar_ofd, n_ofd, 1e-10)
                .unwrap()
                .is_some(),
            "test setup: OFD should succeed for this prefix"
        );

        apply_t_via_camps(&mut prefix_ofd, &mut mps_ofd, 0, false, 1e-10).unwrap();
        let ofd_peak = mps_ofd.current_max_bond_dim();
        assert!(
            ofd_peak <= max_bond,
            "OFD path exceeded max_bond_dim cap: peak={ofd_peak} cap={max_bond}"
        );

        let n_ofds = 3;
        let mut prefix_ofds = SignedCliffordPrefix::identity(n_ofds);
        prefix_ofds.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
        prefix_ofds.apply_state_gate(&Gate::Cx, &[1, 2]).unwrap();
        let mut mps_ofds = crate::backend::mps::MpsBackend::new(0, max_bond);
        mps_ofds.init(n_ofds, 0).unwrap();

        let zbar_ofds = prefix_ofds.conjugate_z(2);
        assert!(
            build_ofd_disentangler(&mps_ofds, &zbar_ofds, n_ofds, 1e-10)
                .unwrap()
                .is_none(),
            "test setup: OFD should reject all-Z support so OFDS is exercised"
        );

        apply_t_via_camps(&mut prefix_ofds, &mut mps_ofds, 2, false, 1e-10).unwrap();
        let ofds_peak = mps_ofds.current_max_bond_dim();
        assert!(
            ofds_peak <= max_bond,
            "OFDS path exceeded max_bond_dim cap: peak={ofds_peak} cap={max_bond}"
        );
    }

    #[test]
    fn t_via_camps_ofds_saturation_guard_errors_when_cap_too_small() {
        let n = 4;
        let mut prefix = SignedCliffordPrefix::identity(n);
        for q in 0..n - 1 {
            prefix.apply_state_gate(&Gate::Cx, &[q, q + 1]).unwrap();
        }

        // Mixed product state: even qubits in |+>, odd qubits in |0>. The
        // all-Z OFDS CX ladder then has |+>-control / |0>-target rungs, which
        // create Bell pairs the bond-dim-1 cap must truncate.
        let mut mps = crate::backend::mps::MpsBackend::new(0, 1);
        mps.init(n, 0).unwrap();
        for q in (0..n).step_by(2) {
            mps.apply(&crate::circuit::Instruction::Gate {
                gate: Gate::H,
                targets: crate::circuit::SmallVec::from_slice(&[q]),
            })
            .unwrap();
        }

        let zbar = prefix.conjugate_z(n - 1);
        assert!(
            build_ofd_disentangler(&mps, &zbar, n, 1e-10)
                .unwrap()
                .is_none(),
            "test setup: OFD must decline so OFDS path runs"
        );

        let result = apply_t_via_camps(&mut prefix, &mut mps, n - 1, false, 1e-10);
        let err = result.expect_err("OFDS truncation at bond cap=1 should trip the guard");
        let msg = format!("{err}");
        assert!(
            msg.contains("SVD truncation discarded"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn choose_disentangler_prefers_ofds_when_routing_cost_strictly_lower() {
        let n = 5;
        let mut mps = fresh_mps(n);
        mps.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[2]),
        })
        .unwrap();
        let p = make_pauli(
            n,
            &[(0, PauliKind::X), (2, PauliKind::X), (4, PauliKind::X)],
        );

        let (ofd_anchor, ofd_cascade) = build_ofd_disentangler(&mps, &p, n, 1e-10)
            .unwrap()
            .expect("OFD has |0⟩ X anchors at q0 and q4");
        assert!(
            ofd_anchor == 0 || ofd_anchor == 4,
            "OFD must reject q2 because it is in |+⟩, picked {ofd_anchor}"
        );
        let ofd_cost = cascade_routing_cost(&mps, &ofd_cascade);
        assert_eq!(ofd_cost, 6, "OFD anchor at q0/q4 routes |0-2|+|0-4|=6");

        let (ofds_anchor, ofds_cascade) =
            build_ofds_disentangler(&mps, &p, n).expect("OFDS has full X anchor freedom");
        assert_eq!(
            ofds_anchor, 2,
            "OFDS picks q2 (middle of support, no |0⟩ requirement)"
        );
        let ofds_cost = cascade_routing_cost(&mps, &ofds_cascade);
        assert_eq!(ofds_cost, 4, "OFDS anchor at q2 routes |2-0|+|2-4|=4");

        let (chosen_anchor, chosen_cascade, chosen_kind) = choose_disentangler(&mps, &p, n, 1e-10)
            .unwrap()
            .expect("cost-compare picks the cheaper cascade");
        assert_eq!(chosen_kind, DisentanglerKind::Ofds);
        assert_eq!(chosen_anchor, 2);
        assert_eq!(cascade_routing_cost(&mps, &chosen_cascade), 4);
    }

    #[test]
    fn choose_disentangler_breaks_ties_in_favor_of_ofd_for_bond_dim_safety() {
        let n = 3;
        let mps = fresh_mps(n);
        let p = make_pauli(
            n,
            &[(0, PauliKind::X), (1, PauliKind::X), (2, PauliKind::X)],
        );

        let (_, ofd_cascade) = build_ofd_disentangler(&mps, &p, n, 1e-10)
            .unwrap()
            .expect("OFD finds anchor on |0...0⟩");
        let (_, ofds_cascade) =
            build_ofds_disentangler(&mps, &p, n).expect("OFDS also finds anchor");
        assert_eq!(
            cascade_routing_cost(&mps, &ofd_cascade),
            cascade_routing_cost(&mps, &ofds_cascade),
            "this fixture must produce a routing-cost tie"
        );

        let (_, _, chosen_kind) = choose_disentangler(&mps, &p, n, 1e-10)
            .unwrap()
            .expect("dispatch returns Some on a viable support");
        assert_eq!(
            chosen_kind,
            DisentanglerKind::Ofd,
            "ties must go to OFD because OFD preserves bond dimension"
        );
    }

    #[test]
    fn apply_t_via_camps_with_cost_compare_dispatch_matches_statevector() {
        let n = 5;
        let mut prefix = SignedCliffordPrefix::identity(n);
        let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
        mps.init(n, 0).unwrap();
        mps.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[2]),
        })
        .unwrap();

        let mut sv = StatevectorBackend::new(0);
        sv.init(n, 0).unwrap();
        sv.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[2]),
        })
        .unwrap();

        apply_t_via_camps(&mut prefix, &mut mps, 2, false, 1e-10).unwrap();
        sv.apply(&Instruction::Gate {
            gate: Gate::T,
            targets: SmallVec::from_slice(&[2]),
        })
        .unwrap();

        let probs = sv.probabilities().unwrap();
        for q in 0..n {
            let camps = evaluate_z_observable_camps(&prefix, &mps, &[q]).unwrap();
            let mask = 1usize << q;
            let direct: f64 = probs
                .iter()
                .enumerate()
                .map(|(i, &p)| if i & mask == 0 { p } else { -p })
                .sum();
            assert!(
                (camps - direct).abs() < 1e-9,
                "Z_{q}: camps={camps:.12}, statevector={direct:.12}"
            );
        }
    }

    #[test]
    fn ofd_anchor_heuristic_picks_routing_minimum() {
        let n = 5;
        let mps = fresh_mps(n);
        let p = make_pauli(
            n,
            &[
                (0, PauliKind::X),
                (2, PauliKind::Z),
                (3, PauliKind::X),
                (4, PauliKind::Z),
            ],
        );
        let (anchor, _) = build_ofd_disentangler(&mps, &p, n, 1e-10)
            .unwrap()
            .expect("OFD should succeed on fresh |0...0⟩");
        let support = support_qubits(&p, n);
        let xy_candidates: Vec<usize> = support
            .iter()
            .copied()
            .filter(|&q| matches!(p.pauli_at(q), PauliKind::X | PauliKind::Y))
            .collect();
        let chosen_cost = anchor_routing_cost(&mps, anchor, &support);
        for &alt in &xy_candidates {
            let alt_cost = anchor_routing_cost(&mps, alt, &support);
            assert!(
                chosen_cost <= alt_cost,
                "heuristic picked anchor {anchor} (cost {chosen_cost}) but {alt} costs {alt_cost}"
            );
        }
    }
}
