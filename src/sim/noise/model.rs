//! Noise channels and events, readout error, and the [`NoiseModel`] that places them,
//! with their validation.

use num_complex::Complex64;
use smallvec::smallvec;

use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::Result;
use crate::sim::unified_pauli::PauliAxis;

/// A noise channel on one qubit, or on two for `TwoQubitDepolarizing` and
/// `Kraus2q`.
///
/// Fires after each instruction whose `NoiseModel::after_gate` slot holds a
/// `NoiseEvent` carrying it. On a Clifford circuit, channels that sample as a
/// Pauli frame draw take the compiled stabilizer sampler; see
/// [`NoiseChannel::is_pauli`] and
/// [`NoiseChannel::TwoQubitDepolarizing`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NoiseChannel {
    /// Independent Pauli X/Y/Z error with given per-branch probabilities.
    Pauli { px: f64, py: f64, pz: f64 },
    /// Symmetric depolarizing: `I` with probability `1-p`, each of `X/Y/Z`
    /// with probability `p/3`.
    Depolarizing { p: f64 },
    /// T1 relaxation: `|1⟩ → |0⟩` with amplitude `sqrt(gamma)`.
    AmplitudeDamping { gamma: f64 },
    /// Pure dephasing: phase randomisation with amplitude `sqrt(gamma)` on the
    /// `|1⟩` state. Populations are preserved.
    PhaseDamping { gamma: f64 },
    /// Combined T1 + T2 relaxation over a gate of duration `gate_time`,
    /// requiring `t2 <= 2·t1`. Populations decay as `exp(-gate_time/t1)` and
    /// coherences as `exp(-gate_time/t2)`; both the exact and the trajectory
    /// route realize that as generalized amplitude damping composed with pure
    /// dephasing.
    ///
    /// `excited_population` is the steady state the qubit relaxes toward: 0 for
    /// a cold qubit that settles in the ground state, 0.5 for one whose steady
    /// state is maximally mixed. A device's value is
    /// `1 / (1 + exp(h f / k T))` at frequency `f` and temperature `T`.
    ThermalRelaxation {
        t1: f64,
        t2: f64,
        gate_time: f64,
        excited_population: f64,
    },
    /// Symmetric two-qubit depolarizing: each of the 15 non-identity
    /// two-qubit Paulis occurs with probability `p/15`.
    TwoQubitDepolarizing { p: f64 },
    /// General single-qubit channel described by a set of Kraus operators.
    ///
    /// Dense Kraus operators are supported by trajectory backends that expose
    /// `Backend::reduced_density_matrix_1q`. Branch probabilities use the full
    /// one-qubit reduced density matrix, so channels may depend on coherence
    /// as well as populations. Call [`NoiseModel::validate`] in setup code to
    /// catch empty or non-finite custom channels before sampling.
    Custom { kraus: Vec<[[Complex64; 2]; 2]> },
    /// General two-qubit channel described by a set of 4x4 Kraus operators.
    ///
    /// Each operator is indexed `K[t][t']` with `t = 2 * bit(qubits[0]) +
    /// bit(qubits[1])`, the packing [`crate::gates::Gate::matrix_4x4`] uses, so
    /// the first target is the high bit. Runs exactly on the density matrix and
    /// on trajectory backends reporting [`Backend::supports_two_qubit_kraus`].
    ///
    /// Correlated `ZZ` dephasing of strength `p`, as the diagonal pair
    /// `sqrt(1-p) I` and `sqrt(p) (Z (x) Z)`:
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use prism_q::NoiseChannel;
    ///
    /// let p = 0.02_f64;
    /// let diag = |scale: f64, signs: [f64; 4]| {
    ///     let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
    ///     for (t, sign) in signs.iter().enumerate() {
    ///         m[t][t] = Complex64::new(scale * sign, 0.0);
    ///     }
    ///     m
    /// };
    /// let zz = NoiseChannel::Kraus2q {
    ///     kraus: vec![
    ///         diag((1.0 - p).sqrt(), [1.0, 1.0, 1.0, 1.0]),
    ///         diag(p.sqrt(), [1.0, -1.0, -1.0, 1.0]),
    ///     ],
    /// };
    /// zz.validate()?;
    /// # Ok::<(), prism_q::PrismError>(())
    /// ```
    ///
    /// [`Backend::supports_two_qubit_kraus`]: crate::backend::Backend::supports_two_qubit_kraus
    Kraus2q { kraus: Vec<[[Complex64; 4]; 4]> },
    /// With probability `p`, an unleaked qubit leaves the computational subspace.
    ///
    /// A leaked qubit is a classical flag in the trajectory: gates on it do not
    /// act, a gate pairing it with unleaked qubits applies a uniformly random
    /// Pauli (`I`, `X`, `Y`, `Z` at a quarter each) to each of them instead,
    /// measuring it reports 1, and a reset or a [`NoiseChannel::Seepage`] clears
    /// the flag. Every other channel skips an event naming a leaked qubit.
    /// [`ShotsResult::leaked`](crate::ShotsResult::leaked) records which qubits
    /// leaked in each shot. Runs on the per-shot trajectory engines only.
    Leakage { p: f64 },
    /// With probability `p`, a leaked qubit returns to the computational
    /// subspace in a uniformly random basis state, `|0>` or `|1>` at one half
    /// each and uncorrelated with the rest of the register.
    Seepage { p: f64 },
    /// With probability `p`, leakage spreads across a pair: when exactly one of
    /// the two qubits is leaked as the event fires, the other leaks too.
    ///
    /// The flags are read when the event fires, so declare it ahead of a
    /// [`NoiseChannel::Leakage`] event in the same slot to spread only the
    /// leakage the gate itself saw.
    LeakageTransport { p: f64 },
    /// Coherent rotation `exp(-i angle P / 2)` about `axis`, with an angle
    /// drawn once per shot as `sum_k weight_k * z[source_k]` over the pairs in
    /// `weights`.
    ///
    /// `z` holds independent standard normal draws, one per source index, made
    /// at the start of each shot from its noise stream and shared by every event
    /// of the shot, so two events naming one source rotate by correlated angles.
    /// That makes each shot a different unitary, which only the per-shot
    /// trajectory engines can sample; [`NoiseBuilder`](super::NoiseBuilder) lowers
    /// detuning and amplitude drift onto it.
    QuasiStatic {
        axis: PauliAxis,
        weights: Vec<(usize, f64)>,
    },
}

impl NoiseChannel {
    /// Number of qubits an event carrying this channel must name.
    pub fn num_qubits(&self) -> usize {
        match self {
            NoiseChannel::TwoQubitDepolarizing { .. }
            | NoiseChannel::Kraus2q { .. }
            | NoiseChannel::LeakageTransport { .. } => 2,
            _ => 1,
        }
    }

    /// Per-Pauli `(px, py, pz)` probabilities for a Pauli or Depolarizing
    /// channel, `None` for every other channel.
    pub fn as_pauli(&self) -> Option<(f64, f64, f64)> {
        match self {
            NoiseChannel::Pauli { px, py, pz } => Some((*px, *py, *pz)),
            NoiseChannel::Depolarizing { p } => {
                let pp = p / 3.0;
                Some((pp, pp, pp))
            }
            _ => None,
        }
    }

    /// Total error probability of a two-qubit Pauli channel, `None` for every
    /// other channel. The 15 non-identity products share it equally.
    pub fn pauli_pair_rate(&self) -> Option<f64> {
        match self {
            NoiseChannel::TwoQubitDepolarizing { p } => Some(*p),
            _ => None,
        }
    }

    /// True when no branch of the channel can fire, so it moves no state and
    /// decides no route, the reading [`ReadoutError::is_inert`] gives a
    /// zero-rate readout entry. Only the Pauli family is considered: a damping
    /// or Kraus channel stays outside the Pauli frame whatever its parameters.
    pub(crate) fn is_inert(&self) -> bool {
        match self {
            NoiseChannel::Pauli { px, py, pz } => *px == 0.0 && *py == 0.0 && *pz == 0.0,
            NoiseChannel::Depolarizing { p } | NoiseChannel::TwoQubitDepolarizing { p } => {
                *p == 0.0
            }
            _ => false,
        }
    }

    /// True for the Pauli and Depolarizing channels.
    pub fn is_pauli(&self) -> bool {
        matches!(
            self,
            NoiseChannel::Pauli { .. } | NoiseChannel::Depolarizing { .. }
        )
    }

    /// True for the channels a Pauli frame can carry, which is
    /// [`NoiseChannel::is_pauli`] plus
    /// [`NoiseChannel::TwoQubitDepolarizing`]. The pair channel needs a joint
    /// draw over its 15 branches rather than one per qubit, so the samplers
    /// hold it apart from the single-qubit rows.
    pub(crate) fn is_pauli_frame(&self) -> bool {
        self.is_pauli() || self.pauli_pair_rate().is_some()
    }

    /// True when `validate` accepts the channel.
    pub fn is_exactly_samplable(&self) -> bool {
        self.validate().is_ok()
    }

    /// Validate that this channel is usable by the trajectory engine.
    pub fn validate(&self) -> Result<()> {
        match self {
            NoiseChannel::Pauli { px, py, pz } => {
                validate_probability("px", *px)?;
                validate_probability("py", *py)?;
                validate_probability("pz", *pz)?;
                if *px + *py + *pz > 1.0 + 1e-12 {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: "Pauli channel probabilities must sum to <= 1".into(),
                    });
                }
            }
            NoiseChannel::Depolarizing { p } | NoiseChannel::TwoQubitDepolarizing { p } => {
                validate_probability("p", *p)?;
            }
            NoiseChannel::AmplitudeDamping { gamma } | NoiseChannel::PhaseDamping { gamma } => {
                validate_probability("gamma", *gamma)?;
            }
            NoiseChannel::ThermalRelaxation {
                t1,
                t2,
                gate_time,
                excited_population,
            } => {
                if !t1.is_finite() || *t1 <= 0.0 {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: "thermal relaxation t1 must be finite and positive".into(),
                    });
                }
                if !t2.is_finite() || *t2 <= 0.0 {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: "thermal relaxation t2 must be finite and positive".into(),
                    });
                }
                if !gate_time.is_finite() || *gate_time < 0.0 {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: "thermal relaxation gate_time must be finite and non-negative"
                            .into(),
                    });
                }
                if *t2 > 2.0 * *t1 * (1.0 + 1e-9) {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: "thermal relaxation requires t2 <= 2*t1".into(),
                    });
                }
                validate_probability("excited_population", *excited_population)?;
            }
            NoiseChannel::Custom { kraus } => validate_kraus_set("Custom", kraus)?,
            NoiseChannel::Kraus2q { kraus } => validate_kraus_set("Kraus2q", kraus)?,
            NoiseChannel::Leakage { p }
            | NoiseChannel::Seepage { p }
            | NoiseChannel::LeakageTransport { p } => validate_probability("p", *p)?,
            NoiseChannel::QuasiStatic { weights, .. } => {
                if weights.iter().any(|(_, weight)| !weight.is_finite()) {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: "quasi-static weights must be finite".into(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Deviation from `sum_k Kdagger_k K_k = I` tolerated in a declared Custom
/// channel. Loose enough for a set assembled from square roots, tight enough
/// that a channel scaled by a constant is rejected.
const CPTP_TOLERANCE: f64 = 1e-9;

/// Reject an empty operator set, a non-finite entry, and any deviation from
/// `sum_k Kdagger_k K_k = I` past [`CPTP_TOLERANCE`]. Shared by the one-qubit
/// and two-qubit dense channels, which differ only in `D`.
fn validate_kraus_set<const D: usize>(name: &str, kraus: &[[[Complex64; D]; D]]) -> Result<()> {
    if kraus.is_empty() {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!("{name} Kraus channel must contain at least one operator"),
        });
    }
    for (op_idx, op) in kraus.iter().enumerate() {
        for (row_idx, row) in op.iter().enumerate() {
            for (col_idx, val) in row.iter().enumerate() {
                if !val.re.is_finite() || !val.im.is_finite() {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: format!(
                            "{name} Kraus operator {op_idx} entry ({row_idx},{col_idx}) must be finite"
                        ),
                    });
                }
            }
        }
    }

    let mut sum = [[Complex64::new(0.0, 0.0); D]; D];
    for op in kraus {
        for (row, sum_row) in sum.iter_mut().enumerate() {
            for (col, entry) in sum_row.iter_mut().enumerate() {
                for op_row in op.iter() {
                    *entry += op_row[row].conj() * op_row[col];
                }
            }
        }
    }
    for (row, sum_row) in sum.iter().enumerate() {
        for (col, entry) in sum_row.iter().enumerate() {
            let expected = if row == col { 1.0 } else { 0.0 };
            if (entry.re - expected).abs() > CPTP_TOLERANCE || entry.im.abs() > CPTP_TOLERANCE {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!(
                        "{name} Kraus operators must be trace preserving: \
                         sum of Kdagger K at ({row},{col}) is {entry}, expected {expected}"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_probability(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!("{name} must be finite and in [0, 1]"),
        });
    }
    Ok(())
}

/// A noise channel bound to its target qubits; fires after the instruction
/// its [`NoiseModel::after_gate`] slot is attached to.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseEvent {
    pub channel: NoiseChannel,
    /// As many entries as [`NoiseChannel::num_qubits`] requires, in the order
    /// the channel indexes them.
    pub qubits: SmallVec<[usize; 2]>,
}

impl NoiseEvent {
    pub fn pauli(qubit: usize, px: f64, py: f64, pz: f64) -> Self {
        Self {
            channel: NoiseChannel::Pauli { px, py, pz },
            qubits: smallvec![qubit],
        }
    }

    /// First target qubit (the only one for single-qubit channels).
    pub fn qubit(&self) -> usize {
        self.qubits[0]
    }

    /// Return the per-Pauli probabilities for a Pauli/Depolarizing channel.
    ///
    /// # Panics
    ///
    /// Panics if the channel is not Pauli or Depolarizing. Callers on the
    /// compiled stabilizer sampler path must guard with
    /// [`NoiseModel::is_pauli_only`] before invoking this method, and read a
    /// two-qubit channel through [`NoiseEvent::pauli_pair`] instead.
    pub fn pauli_probs(&self) -> (f64, f64, f64) {
        self.channel
            .as_pauli()
            .expect("pauli_probs called on non-Pauli channel; caller must check is_pauli_only")
    }

    /// Both targets and the total rate of a two-qubit Pauli channel, `None`
    /// when the channel names one qubit. An event answers this or
    /// [`NoiseEvent::pauli_probs`], never both.
    pub fn pauli_pair(&self) -> Option<(usize, usize, f64)> {
        self.channel
            .pauli_pair_rate()
            .map(|p| (self.qubits[0], self.qubits[1], p))
    }
}

/// Classical bit-flip error applied to a measurement outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutError {
    /// Probability a measured 0 reads out as 1.
    pub p01: f64,
    /// Probability a measured 1 reads out as 0.
    pub p10: f64,
}

impl ReadoutError {
    /// True when neither rate can flip a bit, so the entry changes no draw and
    /// a sweep that starts at zero is not a readout model yet.
    pub(crate) fn is_inert(&self) -> bool {
        self.p01 == 0.0 && self.p10 == 0.0
    }
}

/// Per-instruction noise attached to a specific circuit.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, NoiseModel, run_shots_noisy};
///
/// let mut circuit = Circuit::new(2, 2);
/// circuit.add_gate(Gate::H, &[0]);
/// circuit.add_gate(Gate::Cx, &[0, 1]);
/// circuit.add_measure(0, 0);
/// circuit.add_measure(1, 1);
///
/// let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
/// let result = run_shots_noisy(&circuit, &noise, 100, 42)?;
/// assert_eq!(result.shots.len(), 100);
/// # Ok::<(), prism_q::PrismError>(())
/// ```
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Events fired after each instruction. Slot `i < n` follows top-level
    /// instruction `i` and fires every shot, a guarded one's included. Guarded
    /// bodies follow from `n`: each region in order places its body's slots,
    /// then its nested guards' slots, and a conditional takes one slot for its
    /// gate. A body slot fires only when its instruction runs.
    ///
    /// A circuit without guards therefore has `n` slots. A model of exactly `n`
    /// slots is also accepted on a circuit whose guards are all conditional
    /// gates, which then run without gate noise, and on one with a region when
    /// it carries no quantum event.
    pub after_gate: Vec<Vec<NoiseEvent>>,
    /// Per-classical-bit readout error; `None` means ideal readout. Either empty,
    /// for no readout error at all, or one entry per classical bit.
    ///
    /// A measurement flips the bit it writes, so a classical condition reads the
    /// noisy bit, each measurement into a bit draws its own flip, and the quantum
    /// state is left alone. A bit no measurement writes is never flipped, and a
    /// classical write stores its value unflipped.
    pub readout: Vec<Option<ReadoutError>>,
}

impl NoiseModel {
    /// Depolarizing noise with total probability `p` (not per branch) on
    /// every target of every gate, guarded gates included, ideal readout.
    pub fn uniform_depolarizing(circuit: &Circuit, p: f64) -> Self {
        let pp = p / 3.0;
        Self::per_gate_target(circuit, |q| NoiseEvent::pauli(q, pp, pp, pp))
    }

    /// Amplitude damping with rate `gamma` on every target of every gate,
    /// guarded gates included, ideal readout.
    pub fn with_amplitude_damping(circuit: &Circuit, gamma: f64) -> Self {
        Self::per_gate_target(circuit, |q| NoiseEvent {
            channel: NoiseChannel::AmplitudeDamping { gamma },
            qubits: smallvec![q],
        })
    }

    /// No noise events and ideal readout, sized for `circuit`.
    pub(crate) fn noiseless_for(circuit: &Circuit) -> Self {
        Self {
            after_gate: vec![Vec::new(); circuit.noise_slot_count()],
            readout: vec![None; circuit.num_classical_bits],
        }
    }

    fn per_gate_target(circuit: &Circuit, event: impl Fn(usize) -> NoiseEvent) -> Self {
        let mut after_gate = vec![Vec::new(); circuit.noise_slot_count()];
        circuit.for_each_noise_slot(|slot, instr| {
            if let Instruction::Gate { targets, .. } = instr {
                after_gate[slot] = targets.iter().map(|&q| event(q)).collect();
            }
        });
        Self {
            after_gate,
            readout: vec![None; circuit.num_classical_bits],
        }
    }

    /// Set the same readout error on every classical bit, flipped as each
    /// measurement writes it (see [`NoiseModel::readout`]).
    pub fn with_readout_error(&mut self, p01: f64, p10: f64) -> &mut Self {
        self.readout.fill(Some(ReadoutError { p01, p10 }));
        self
    }

    /// Set the readout error of one classical bit, leaving the rest alone.
    ///
    /// Indexed by classical bit rather than by qubit: a qubit measured into
    /// several bits carries an independent rate per bit. Each measurement into
    /// `bit` draws its own flip (see [`NoiseModel::readout`]).
    ///
    /// # Panics
    ///
    /// Panics if `bit` is outside the classical register the model was built
    /// for, matching the index-bounds policy of the circuit builders.
    pub fn set_bit_readout_error(&mut self, bit: usize, p01: f64, p10: f64) -> &mut Self {
        assert!(
            bit < self.readout.len(),
            "classical bit {bit} is outside the {}-bit register",
            self.readout.len()
        );
        self.readout[bit] = Some(ReadoutError { p01, p10 });
        self
    }

    /// True when every channel is a single-qubit Pauli or Depolarizing channel
    /// and no readout entry can flip a bit. A live two-qubit channel answers
    /// `false` here and still runs on the Clifford samplers; an inert one
    /// blocks nothing.
    pub fn is_pauli_only(&self) -> bool {
        self.has_chain_complex_channels()
            && self.readout.iter().flatten().all(ReadoutError::is_inert)
    }

    /// True when every channel fits one column of the error chain complex: a
    /// single-qubit Pauli draw, or inert. Readout is not a channel and is
    /// judged separately, since only some of the routes over the complex need
    /// it folded into a syndrome class.
    pub(crate) fn has_chain_complex_channels(&self) -> bool {
        self.after_gate
            .iter()
            .flat_map(|events| events.iter())
            .all(|e| e.channel.is_pauli() || e.channel.is_inert())
    }

    /// True when every channel samples as a Pauli frame draw, whatever the
    /// readout entries hold. Readout acts on the measurement record, not the
    /// state, so the Clifford samplers apply it themselves and only the channels
    /// decide whether a model can take that route.
    pub(crate) fn has_only_pauli_channels(&self) -> bool {
        self.after_gate
            .iter()
            .flat_map(|events| events.iter())
            .all(|e| e.channel.is_pauli_frame())
    }

    /// True when any event carries a [`NoiseChannel::Kraus2q`], the one channel
    /// needing a two-qubit reduction from the backend running the trajectory.
    pub(crate) fn has_two_qubit_kraus(&self) -> bool {
        self.after_gate
            .iter()
            .flat_map(|events| events.iter())
            .any(|event| matches!(event.channel, NoiseChannel::Kraus2q { .. }))
    }

    /// True when any event carries a leakage channel, which gives each
    /// trajectory a leak flag per qubit and each shot a leaked-qubit record.
    pub(crate) fn has_leakage(&self) -> bool {
        self.after_gate.iter().flatten().any(|event| {
            matches!(
                event.channel,
                NoiseChannel::Leakage { .. }
                    | NoiseChannel::Seepage { .. }
                    | NoiseChannel::LeakageTransport { .. }
            )
        })
    }

    /// Standard normal draws each shot makes for its
    /// [`NoiseChannel::QuasiStatic`] events: one past the highest source index.
    pub(crate) fn quasi_static_sources(&self) -> usize {
        self.after_gate
            .iter()
            .flatten()
            .filter_map(|event| match &event.channel {
                NoiseChannel::QuasiStatic { weights, .. } => {
                    weights.iter().map(|&(source, _)| source + 1).max()
                }
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// Return an error naming `engine` if any channel needs per-shot memory:
    /// a leak flag or a quasi-static draw. A mixture averages each event on its
    /// own, which neither survives.
    pub(crate) fn ensure_memoryless(&self, engine: &str) -> Result<()> {
        if self.has_leakage() || self.quasi_static_sources() > 0 {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: engine.into(),
                reason: "leakage and quasi-static channels carry state from one event to the \
                         next within a shot, which a mixture evolved event by event does not \
                         hold; sample them through per-shot trajectories"
                    .into(),
            });
        }
        Ok(())
    }

    pub fn has_noise(&self) -> bool {
        self.after_gate.iter().any(|events| !events.is_empty())
            || self.readout.iter().any(|r| r.is_some())
    }

    /// Return an error naming `engine` when a guarded body slot of `circuit`
    /// carries an event. Such an event fires only on the shots whose condition
    /// holds, which a route without one state per shot cannot decide.
    pub(crate) fn ensure_no_guarded_events(&self, circuit: &Circuit, engine: &str) -> Result<()> {
        let guarded = self.after_gate.get(circuit.instructions.len()..);
        if guarded.is_some_and(|slots| slots.iter().any(|events| !events.is_empty())) {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: engine.into(),
                reason: "noise inside a guarded body fires only on the shots its condition \
                         holds, which this route cannot decide; sample through \
                         `run_shots_noisy` or a trajectory backend"
                    .into(),
            });
        }
        Ok(())
    }

    /// Return an error if any channel falls outside the error chain complex,
    /// which holds one column per single-qubit Pauli error.
    pub(crate) fn ensure_chain_complex_channels(&self) -> Result<()> {
        if !self.has_chain_complex_channels() {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "error chain complex".into(),
                reason: "non-Pauli noise channels (amplitude damping, phase damping, \
                         thermal relaxation, custom Kraus) carry no Pauli frame, and a \
                         two-qubit channel spans two error columns where the complex \
                         holds one; sample through `run_shots_noisy`"
                    .into(),
            });
        }
        Ok(())
    }

    /// Return an error if any channel is non-Pauli, the precondition of the
    /// Clifford samplers. Readout error is allowed: they apply it to the
    /// measurement record after the reference bits are folded in.
    pub(crate) fn ensure_only_pauli_channels(&self) -> Result<()> {
        if !self.has_only_pauli_channels() {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "compiled stabilizer sampler".into(),
                reason: "non-Pauli noise channels (amplitude damping, phase damping, \
                         thermal relaxation, custom Kraus) require the trajectory \
                         engine; use a non-stabilizer backend or a Pauli-only noise \
                         model"
                    .into(),
            });
        }
        Ok(())
    }

    /// Validate as [`NoiseModel::validate`] does, plus the checks that need the
    /// circuit: the slot count [`NoiseModel::after_gate`] lays out, a readout
    /// table that is empty or holds one entry per classical bit, and every
    /// target qubit inside the register.
    ///
    /// Every noisy entry point calls this before allocating state. Qubit bounds
    /// cannot be checked without the circuit, and an out-of-range target reaches
    /// kernels that index amplitudes without one.
    pub fn validate_for(&self, circuit: &Circuit) -> Result<()> {
        let slots = circuit.noise_slot_count();
        let top_level = circuit.instructions.len();
        let legacy = self.after_gate.len() == top_level
            && (self.after_gate.iter().all(Vec::is_empty)
                || !circuit
                    .instructions
                    .iter()
                    .any(|inst| matches!(inst, Instruction::Region(_))));
        if self.after_gate.len() != slots && !legacy {
            let carried = self.after_gate.len();
            let message = if slots == top_level {
                format!("noise model carries {carried} event slots for {top_level} instructions")
            } else {
                format!(
                    "noise model carries {carried} event slots for {top_level} instructions \
                     and {} guarded body slots",
                    slots - top_level
                )
            };
            return Err(crate::error::PrismError::InvalidParameter { message });
        }
        if !self.readout.is_empty() && self.readout.len() != circuit.num_classical_bits {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "noise model carries {} readout entries for {} classical bits",
                    self.readout.len(),
                    circuit.num_classical_bits
                ),
            });
        }
        self.validate()?;

        for (instr_idx, events) in self.after_gate.iter().enumerate() {
            for (event_idx, event) in events.iter().enumerate() {
                for &qubit in &event.qubits {
                    if qubit >= circuit.num_qubits {
                        return Err(crate::error::PrismError::InvalidParameter {
                            message: format!(
                                "noise event {event_idx} after instruction {instr_idx}: qubit {qubit} is outside the {}-qubit register",
                                circuit.num_qubits
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate all channels and readout errors in this noise model.
    pub fn validate(&self) -> Result<()> {
        for (instr_idx, events) in self.after_gate.iter().enumerate() {
            for (event_idx, event) in events.iter().enumerate() {
                event.channel.validate().map_err(|err| {
                    crate::error::PrismError::InvalidParameter {
                        message: format!(
                            "noise event {event_idx} after instruction {instr_idx}: {err}"
                        ),
                    }
                })?;

                let expected_qubits = event.channel.num_qubits();
                if event.qubits.len() != expected_qubits {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: format!(
                            "noise event {event_idx} after instruction {instr_idx}: expected {expected_qubits} qubit(s), got {}",
                            event.qubits.len()
                        ),
                    });
                }
                if expected_qubits == 2 && event.qubits[0] == event.qubits[1] {
                    return Err(crate::error::PrismError::InvalidParameter {
                        message: format!(
                            "noise event {event_idx} after instruction {instr_idx}: two-qubit channel needs distinct targets, got qubit {} twice",
                            event.qubits[0]
                        ),
                    });
                }
            }
        }

        for (bit_idx, readout) in self.readout.iter().enumerate() {
            if let Some(readout) = readout {
                validate_probability("readout p01", readout.p01).map_err(|err| {
                    crate::error::PrismError::InvalidParameter {
                        message: format!("readout error {bit_idx}: {err}"),
                    }
                })?;
                validate_probability("readout p10", readout.p10).map_err(|err| {
                    crate::error::PrismError::InvalidParameter {
                        message: format!("readout error {bit_idx}: {err}"),
                    }
                })?;
            }
        }

        Ok(())
    }
}
