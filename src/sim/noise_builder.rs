//! Rule-based construction of a [`NoiseModel`] against a circuit.
//!
//! Rules are declared once and compiled against an instruction stream by
//! [`NoiseBuilder::build`], which emits the same per-instruction event vector
//! the manual `after_gate` path builds. Nothing here runs per shot.

use smallvec::smallvec;

use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::calibration::GateTimes;
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, ReadoutError};
use crate::sim::unified_pauli::PauliAxis;

/// Which gates a rule fires after.
///
/// An unset field imposes no restriction, so [`GateFilter::all`] matches every
/// gate on every qubit.
#[derive(Debug, Clone, Default)]
pub struct GateFilter {
    arity: Option<usize>,
    name: Option<String>,
    qubits: Option<Vec<usize>>,
    targets: Option<Vec<usize>>,
}

impl GateFilter {
    pub fn all() -> Self {
        Self::default()
    }

    /// Restrict to gates with exactly `arity` targets.
    pub fn arity(mut self, arity: usize) -> Self {
        self.arity = Some(arity);
        self
    }

    /// Restrict to gates whose [`Gate::name`] equals `name`, such as `"cx"`.
    ///
    /// A name no gate in the circuit carries is not an error: the rule emits
    /// nothing. Fusion renames gates (`"fused"`, `"fused_2q"`, `"multi_fused"`),
    /// so build the model against the circuit the caller holds rather than a
    /// fused stream.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Restrict to the listed qubits, as an unordered set. Every rule kind
    /// honours it: one acting per target emits events only on the targets in
    /// the set, and one acting on a gate's whole target list, or on the
    /// spectators of a target, fires only for targets the set admits.
    pub fn on_qubits(mut self, qubits: impl IntoIterator<Item = usize>) -> Self {
        let mut listed: Vec<usize> = qubits.into_iter().collect();
        listed.sort_unstable();
        listed.dedup();
        self.qubits = Some(listed);
        self
    }

    /// Restrict to gates whose target list equals `targets` in order.
    ///
    /// Directed, unlike [`GateFilter::on_qubits`]: `on_targets([0, 1])` matches
    /// `cx(0, 1)` and not `cx(1, 0)`, which is what a calibration table keyed by
    /// coupling-map edge needs.
    pub fn on_targets(mut self, targets: impl IntoIterator<Item = usize>) -> Self {
        self.targets = Some(targets.into_iter().collect());
        self
    }

    fn matches(&self, gate: &Gate, targets: &[usize]) -> bool {
        if self.arity.is_some_and(|arity| arity != targets.len()) {
            return false;
        }
        if self.name.as_deref().is_some_and(|name| name != gate.name()) {
            return false;
        }
        if self
            .targets
            .as_deref()
            .is_some_and(|listed| listed != targets)
        {
            return false;
        }
        true
    }

    fn allows(&self, qubit: usize) -> bool {
        self.qubits
            .as_ref()
            .is_none_or(|listed| listed.binary_search(&qubit).is_ok())
    }
}

enum Rule {
    PerTarget {
        filter: GateFilter,
        channel: NoiseChannel,
    },
    Joint {
        filter: GateFilter,
        channel: NoiseChannel,
    },
    Crosstalk {
        filter: GateFilter,
        coupling: Vec<(usize, usize)>,
        channel: NoiseChannel,
    },
    OverRotation {
        filter: GateFilter,
        relative: f64,
    },
    Idle {
        channel: NoiseChannel,
    },
    AfterReset {
        channel: NoiseChannel,
    },
    BeforeMeasure {
        channel: NoiseChannel,
    },
    ScheduledIdle {
        coherence: Vec<(f64, f64)>,
    },
    Detuning {
        drift: DriftDistribution,
    },
    OverRotationDrift {
        filter: GateFilter,
        sigma: f64,
    },
    OverRotationDriftPerQubit {
        filter: GateFilter,
        drift: DriftDistribution,
    },
}

/// A zero-mean Gaussian over one quasi-static offset per qubit, drawn once per
/// shot: a detuning in radians per second, or a fractional over-rotation.
///
/// Held as a covariance matrix over qubits `0..n`. A model built from it draws
/// `n` independent standard normals per shot and correlates them through the
/// Cholesky factor of the matrix, so the offsets have exactly this covariance.
/// The matrix is checked when the model is built: symmetric, finite, and
/// positive semidefinite, singular allowed.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftDistribution {
    num_qubits: usize,
    covariance: Vec<f64>,
}

impl DriftDistribution {
    /// Independent offsets with standard deviation `sigmas[q]` on qubit `q`.
    pub fn independent(sigmas: impl IntoIterator<Item = f64>) -> Self {
        let sigmas: Vec<f64> = sigmas.into_iter().collect();
        let n = sigmas.len();
        let mut covariance = vec![0.0; n * n];
        for (q, sigma) in sigmas.iter().enumerate() {
            covariance[q * n + q] = sigma * sigma;
        }
        Self {
            num_qubits: n,
            covariance,
        }
    }

    /// Independent detunings whose free-induction decay `exp(-(sigma t)^2 / 2)`
    /// falls to `1/e` at `t2_star[q]` seconds, which is
    /// `sigma = sqrt(2) / t2_star`.
    pub fn from_t2_star(t2_star: impl IntoIterator<Item = f64>) -> Self {
        Self::independent(t2_star.into_iter().map(|t| std::f64::consts::SQRT_2 / t))
    }

    /// Covariance given in full, `matrix[a][b]` between qubits `a` and `b`. A
    /// ragged matrix is reported when the model is built.
    pub fn from_covariance(matrix: Vec<Vec<f64>>) -> Self {
        let n = matrix.len();
        let mut covariance = vec![f64::NAN; n * n];
        for (a, row) in matrix.iter().enumerate().filter(|(_, row)| row.len() == n) {
            covariance[a * n..(a + 1) * n].copy_from_slice(row);
        }
        Self {
            num_qubits: n,
            covariance,
        }
    }

    /// Correlate the offsets across each undirected `coupling` edge with
    /// coefficient `rho`, leaving every other pair as it was.
    ///
    /// # Panics
    ///
    /// Panics if an edge names a qubit outside the distribution.
    pub fn with_neighbour_correlation(
        mut self,
        coupling: impl IntoIterator<Item = (usize, usize)>,
        rho: f64,
    ) -> Self {
        let n = self.num_qubits;
        for (a, b) in coupling {
            assert!(
                a < n && b < n,
                "coupling edge ({a}, {b}) is outside the {n}-qubit drift distribution"
            );
            if a == b {
                continue;
            }
            let value = rho * (self.covariance[a * n + a] * self.covariance[b * n + b]).sqrt();
            self.covariance[a * n + b] = value;
            self.covariance[b * n + a] = value;
        }
        self
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The covariance matrix, row major, `num_qubits` squared entries.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Lower-triangular `L` with `L L^T` equal to the covariance, row major.
    ///
    /// A pivot within rounding of zero is a direction with no variance, so its
    /// column stays zero; the matrix is rejected when that column would need to
    /// carry weight, or when a pivot is negative.
    fn cholesky(&self) -> Result<Vec<f64>> {
        let n = self.num_qubits;
        let cov = &self.covariance;
        let invalid = |message: String| PrismError::InvalidParameter { message };
        for a in 0..n {
            for b in 0..n {
                let value = cov[a * n + b];
                if !value.is_finite() {
                    return Err(invalid(format!(
                        "drift covariance entry ({a}, {b}) is {value}; give a finite square matrix"
                    )));
                }
                if (value - cov[b * n + a]).abs() > 1e-12 * value.abs().max(1.0) {
                    return Err(invalid(format!(
                        "drift covariance is not symmetric at ({a}, {b})"
                    )));
                }
            }
        }
        let scale = (0..n).map(|q| cov[q * n + q]).fold(0.0f64, f64::max);
        let tolerance = 1e-12 * scale;
        let mut l = vec![0.0f64; n * n];
        for j in 0..n {
            let pivot = cov[j * n + j] - (0..j).map(|k| l[j * n + k].powi(2)).sum::<f64>();
            if pivot < -tolerance {
                return Err(invalid(format!(
                    "drift covariance is not positive semidefinite (pivot {pivot} at qubit {j})"
                )));
            }
            let diagonal = if pivot > tolerance { pivot.sqrt() } else { 0.0 };
            l[j * n + j] = diagonal;
            for i in j + 1..n {
                let residual =
                    cov[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
                if diagonal > 0.0 {
                    l[i * n + j] = residual / diagonal;
                } else if residual.abs() > 1e-6 * scale {
                    return Err(invalid(format!(
                        "drift covariance is not positive semidefinite: qubit {j} has no \
                         variance left but still covaries with qubit {i}"
                    )));
                }
            }
        }
        Ok(l)
    }

    /// Row `qubit` of the Cholesky `factor` as weights over the sources from
    /// `base`, each scaled by `scale`.
    fn weights(
        factor: &[f64],
        n: usize,
        qubit: usize,
        base: usize,
        scale: f64,
    ) -> Vec<(usize, f64)> {
        (0..=qubit)
            .filter(|&k| factor[qubit * n + k] != 0.0)
            .map(|k| (base + k, factor[qubit * n + k] * scale))
            .collect()
    }
}

/// Declarative noise-model construction.
///
/// Each method appends a rule; [`NoiseBuilder::build`] walks a circuit once and
/// evaluates the rules in registration order at every instruction, so the event
/// order inside a slot is the order the rules were declared. Rules match
/// [`Instruction::Gate`] only: a [`Instruction::Conditional`] carries noise that
/// would fire whether or not its gate ran, so it is left alone.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, GateFilter, NoiseBuilder, NoiseChannel};
///
/// let mut circuit = Circuit::new(3, 3);
/// circuit.add_gate(Gate::H, &[0]);
/// circuit.add_gate(Gate::Cx, &[0, 1]);
///
/// let noise = NoiseBuilder::new()
///     .after_gates(GateFilter::all().arity(1), NoiseChannel::Depolarizing { p: 1e-4 })
///     .after_gates(GateFilter::all().named("cx"), NoiseChannel::Depolarizing { p: 2e-3 })
///     .on_idle_qubits(NoiseChannel::PhaseDamping { gamma: 1e-5 })
///     .readout_error(2, 0.02, 0.03)
///     .build(&circuit)?;
///
/// assert_eq!(noise.after_gate.len(), circuit.instructions.len());
/// # Ok::<(), prism_q::PrismError>(())
/// ```
#[derive(Default)]
pub struct NoiseBuilder {
    rules: Vec<Rule>,
    readout: Vec<(usize, ReadoutError)>,
    uniform_readout: Option<ReadoutError>,
    schedule: Option<GateTimes>,
}

impl NoiseBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// One single-qubit channel on each matching target of each matching gate.
    pub fn after_gates(mut self, filter: GateFilter, channel: NoiseChannel) -> Self {
        self.rules.push(Rule::PerTarget { filter, channel });
        self
    }

    /// One channel on a matching gate's whole target list, for the correlated
    /// two-qubit channels ([`NoiseChannel::TwoQubitDepolarizing`],
    /// [`NoiseChannel::Kraus2q`]). The rule fires only on gates whose target
    /// count equals the channel's own arity.
    pub fn after_gates_joint(mut self, filter: GateFilter, channel: NoiseChannel) -> Self {
        self.rules.push(Rule::Joint { filter, channel });
        self
    }

    /// One channel per spectator qubit coupled to a target of a matching gate.
    ///
    /// `coupling` is an undirected edge list. A single-qubit channel lands once
    /// on each distinct spectator; a two-qubit channel lands on each
    /// `(target, spectator)` pair with the target as its first qubit. Qubits
    /// among the gate's own targets are not spectators of it, and a target the
    /// filter's qubit set excludes contributes no spectators.
    pub fn crosstalk(
        mut self,
        filter: GateFilter,
        coupling: impl IntoIterator<Item = (usize, usize)>,
        channel: NoiseChannel,
    ) -> Self {
        self.rules.push(Rule::Crosstalk {
            filter,
            coupling: coupling.into_iter().collect(),
            channel,
        });
        self
    }

    /// A coherent rotation of `relative * theta` about a matching rotation
    /// gate's own axis, appended after it: the gate turns `theta` into
    /// `theta * (1 + relative)`.
    ///
    /// Fires on `rx`, `ry`, `rz`, and `p` only. The excess is emitted as a
    /// one-qubit Kraus set, so the multi-qubit rotations (`rzz`, `pauli_rot`)
    /// do not fit even though they carry a single angle. Other gates matching
    /// `filter` are skipped.
    pub fn over_rotation(mut self, filter: GateFilter, relative: f64) -> Self {
        self.rules.push(Rule::OverRotation { filter, relative });
        self
    }

    /// One single-qubit channel on every qubit no instruction of a layer
    /// touches, fired at the end of the layer.
    ///
    /// Layers come from the greedy assignment [`Circuit::depth`] reports, so
    /// the layer count a caller can print is the one the idle budget is charged
    /// against. A layer's events ride its highest-indexed instruction. Greedy
    /// assignment can place a later instruction in an earlier layer, and on
    /// such a circuit a layer's idle events fire after the events of the layer
    /// that follows it.
    ///
    /// The event count grows as layers times idle qubits, which on a wide
    /// shallow circuit is most of the register on every layer. Every engine
    /// walks that stream per shot.
    pub fn on_idle_qubits(mut self, channel: NoiseChannel) -> Self {
        self.rules.push(Rule::Idle { channel });
        self
    }

    /// One single-qubit channel on the reset qubit, after each reset.
    pub fn after_resets(mut self, channel: NoiseChannel) -> Self {
        self.rules.push(Rule::AfterReset { channel });
        self
    }

    /// One single-qubit channel on the measured qubit, immediately before each
    /// measurement.
    ///
    /// Distinct from [`NoiseModel::readout`], which flips reported bits once at
    /// the end of a shot: this damages the state the measurement then projects,
    /// so a mid-circuit outcome feeding a classical conditional is the faulty
    /// one. A purely classical fault that leaves the state intact is not
    /// expressible this way.
    ///
    /// The channel is carried by the slot of the preceding instruction, so a
    /// measurement at instruction 0 has nowhere to put it and
    /// [`NoiseBuilder::build`] rejects the circuit; prepend a barrier to make
    /// room. Sharing that slot with the preceding gate's own error means
    /// declaration order decides which fires first, so declare this rule after
    /// the gate rules.
    pub fn before_measurements(mut self, channel: NoiseChannel) -> Self {
        self.rules.push(Rule::BeforeMeasure { channel });
        self
    }

    /// Time the circuit with `times`, for the rules that act over elapsed time:
    /// [`scheduled_idle`](NoiseBuilder::scheduled_idle) and
    /// [`quasi_static_detuning`](NoiseBuilder::quasi_static_detuning).
    ///
    /// Layers are the greedy assignment
    /// [`on_idle_qubits`](NoiseBuilder::on_idle_qubits) uses, run as soon as
    /// possible: each layer lasts as long as its longest instruction, and a
    /// qubit's events for a layer land just before its next instruction.
    pub fn schedule(mut self, times: GateTimes) -> Self {
        self.schedule = Some(times);
        self
    }

    /// Idle relaxation over the real elapsed time of each scheduled layer, in
    /// place of a fixed per-layer idle channel.
    ///
    /// `coherence[q]` is the `(t1, t2)` of qubit `q` in seconds, and must cover
    /// the register. At the end of each layer every qubit gets a
    /// [`NoiseChannel::ThermalRelaxation`] relaxing to the ground state over the
    /// part of the layer it spends idle: the whole layer when nothing touches
    /// it, the remainder after a shorter instruction, nothing after the longest
    /// one. Decay during a gate is left to the gate rules, as
    /// [`DeviceCalibration::to_noise_model`](crate::DeviceCalibration::to_noise_model)
    /// models it.
    pub fn scheduled_idle(mut self, coherence: impl IntoIterator<Item = (f64, f64)>) -> Self {
        self.rules.push(Rule::ScheduledIdle {
            coherence: coherence.into_iter().collect(),
        });
        self
    }

    /// Quasi-static detuning: each shot draws one frequency offset per qubit,
    /// in radians per second, from `drift`, and at the end of each scheduled
    /// layer every unleaked qubit turns about `Z` by its offset times the
    /// duration of the layer.
    ///
    /// Averaged over shots, a qubit left for time `t` dephases as
    /// `exp(-(sigma t)^2 / 2)`, the Gaussian free-induction decay a `T2*`
    /// measurement reports. `drift` must cover the register.
    pub fn quasi_static_detuning(mut self, drift: DriftDistribution) -> Self {
        self.rules.push(Rule::Detuning { drift });
        self
    }

    /// Amplitude drift shared by a gate family: each shot draws one fractional
    /// error `epsilon` with standard deviation `sigma`, and every matching `rx`,
    /// `ry`, `rz` or `p` gate of the shot turns `theta` into
    /// `theta * (1 + epsilon)`.
    pub fn over_rotation_drift(mut self, filter: GateFilter, sigma: f64) -> Self {
        self.rules.push(Rule::OverRotationDrift { filter, sigma });
        self
    }

    /// [`over_rotation_drift`](NoiseBuilder::over_rotation_drift) with one
    /// fractional error per target qubit, drawn jointly from `drift`, which must
    /// cover the register.
    pub fn over_rotation_drift_per_qubit(
        mut self,
        filter: GateFilter,
        drift: DriftDistribution,
    ) -> Self {
        self.rules
            .push(Rule::OverRotationDriftPerQubit { filter, drift });
        self
    }

    /// Readout error on one classical bit. Overrides any uniform rate whatever
    /// order the two are declared in.
    pub fn readout_error(mut self, bit: usize, p01: f64, p10: f64) -> Self {
        self.readout.push((bit, ReadoutError { p01, p10 }));
        self
    }

    /// Readout error on every classical bit of the register the model is built
    /// for, including bits no measurement writes.
    pub fn uniform_readout_error(mut self, p01: f64, p10: f64) -> Self {
        self.uniform_readout = Some(ReadoutError { p01, p10 });
        self
    }

    /// Compile the rules against `circuit`.
    ///
    /// # Errors
    ///
    /// Reports a per-bit readout rate outside the classical register, a
    /// measurement at instruction 0 under a
    /// [`before_measurements`](NoiseBuilder::before_measurements) rule, a timed
    /// rule without a [`schedule`](NoiseBuilder::schedule), fixed and scheduled
    /// idle rules together, coherence times or a drift distribution that do not
    /// cover the register, a drift covariance that is not positive
    /// semidefinite, and everything [`NoiseModel::validate_for`] rejects.
    pub fn build(&self, circuit: &Circuit) -> Result<NoiseModel> {
        let mut after_gate: Vec<Vec<NoiseEvent>> = vec![Vec::new(); circuit.instructions.len()];
        let context = BuildContext::new(self, circuit)?;

        for (idx, instr) in circuit.instructions.iter().enumerate() {
            for (rule_index, rule) in self.rules.iter().enumerate() {
                emit(rule, rule_index, idx, instr, &context, &mut after_gate);
            }
        }

        let mut readout = vec![self.uniform_readout.clone(); circuit.num_classical_bits];
        for (bit, error) in &self.readout {
            if *bit >= readout.len() {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "readout error on classical bit {bit} is outside the {}-bit register",
                        readout.len()
                    ),
                });
            }
            readout[*bit] = Some(error.clone());
        }

        let model = NoiseModel {
            after_gate,
            readout,
        };
        model.validate_for(circuit)?;
        Ok(model)
    }
}

/// Everything [`emit`] reads besides the rule and the instruction, computed
/// once per build.
struct BuildContext {
    idle: Option<Vec<Vec<usize>>>,
    pre_measure: Option<Vec<Option<usize>>>,
    layers: Option<ScheduledLayers>,
    /// `(first source, Cholesky factor)` of each drift rule by rule index; the
    /// factor is empty for a rule drawing one shared source. Sources are
    /// numbered in rule order, so no two rules share a draw.
    drift: Vec<Option<(usize, Vec<f64>)>>,
    num_qubits: usize,
}

impl BuildContext {
    fn new(builder: &NoiseBuilder, circuit: &Circuit) -> Result<Self> {
        let rules = &builder.rules;
        let has = |test: fn(&Rule) -> bool| rules.iter().any(test);
        let idle =
            has(|rule| matches!(rule, Rule::Idle { .. })).then(|| idle_qubits_by_layer(circuit));
        let pre_measure = has(|rule| matches!(rule, Rule::BeforeMeasure { .. }))
            .then(|| pre_measure_qubits(circuit))
            .transpose()?;
        let invalid = |message: String| PrismError::InvalidParameter { message };
        if idle.is_some() && has(|rule| matches!(rule, Rule::ScheduledIdle { .. })) {
            return Err(invalid(
                "a scheduled idle rule replaces the fixed per-layer idle channel; declare one \
                 of the two"
                    .into(),
            ));
        }
        let timed = has(|rule| matches!(rule, Rule::ScheduledIdle { .. } | Rule::Detuning { .. }));
        let layers = match (&builder.schedule, timed) {
            (_, false) => None,
            (Some(times), true) => {
                times.validate()?;
                Some(scheduled_layers(circuit, times))
            }
            (None, true) => {
                return Err(invalid(
                    "scheduled idling and detuning need gate times; call NoiseBuilder::schedule"
                        .into(),
                ));
            }
        };

        let num_qubits = circuit.num_qubits;
        let mut next = 0usize;
        let mut drift = Vec::with_capacity(rules.len());
        for rule in rules {
            let entry = match rule {
                Rule::ScheduledIdle { coherence } if coherence.len() < num_qubits => {
                    return Err(invalid(format!(
                        "scheduled idle coherence covers {} qubits and the circuit has \
                         {num_qubits}",
                        coherence.len()
                    )));
                }
                Rule::Detuning { drift } | Rule::OverRotationDriftPerQubit { drift, .. } => {
                    if drift.num_qubits() < num_qubits {
                        return Err(invalid(format!(
                            "drift distribution covers {} qubits and the circuit has \
                             {num_qubits}",
                            drift.num_qubits()
                        )));
                    }
                    let factor = drift.cholesky()?;
                    let base = next;
                    next += drift.num_qubits();
                    Some((base, factor))
                }
                Rule::OverRotationDrift { sigma, .. } => {
                    if !sigma.is_finite() || *sigma < 0.0 {
                        return Err(invalid(format!(
                            "over-rotation drift sigma = {sigma} must be finite and non-negative"
                        )));
                    }
                    let base = next;
                    next += 1;
                    Some((base, Vec::new()))
                }
                _ => None,
            };
            drift.push(entry);
        }
        Ok(Self {
            idle,
            pre_measure,
            layers,
            drift,
            num_qubits,
        })
    }
}

fn emit(
    rule: &Rule,
    rule_index: usize,
    idx: usize,
    instr: &Instruction,
    context: &BuildContext,
    after_gate: &mut [Vec<NoiseEvent>],
) {
    let idle = &context.idle;
    let pre_measure = &context.pre_measure;
    let slot = &mut after_gate[idx];
    match rule {
        Rule::PerTarget { filter, channel } => {
            let Instruction::Gate { gate, targets } = instr else {
                return;
            };
            if !filter.matches(gate, targets) {
                return;
            }
            for &qubit in targets.iter().filter(|&&q| filter.allows(q)) {
                slot.push(NoiseEvent {
                    channel: channel.clone(),
                    qubits: smallvec![qubit],
                });
            }
        }
        Rule::Joint { filter, channel } => {
            let Instruction::Gate { gate, targets } = instr else {
                return;
            };
            if targets.len() != channel.num_qubits()
                || !filter.matches(gate, targets)
                || !targets.iter().all(|&q| filter.allows(q))
            {
                return;
            }
            slot.push(NoiseEvent {
                channel: channel.clone(),
                qubits: targets.iter().copied().collect(),
            });
        }
        Rule::Crosstalk {
            filter,
            coupling,
            channel,
        } => {
            let Instruction::Gate { gate, targets } = instr else {
                return;
            };
            if !filter.matches(gate, targets) {
                return;
            }
            emit_crosstalk(filter, coupling, channel, targets, slot);
        }
        Rule::OverRotation { filter, relative } => {
            let Instruction::Gate { gate, targets } = instr else {
                return;
            };
            if !filter.matches(gate, targets) {
                return;
            }
            let Some(channel) = over_rotation_channel(gate, *relative) else {
                return;
            };
            for &qubit in targets.iter().filter(|&&q| filter.allows(q)) {
                slot.push(NoiseEvent {
                    channel: channel.clone(),
                    qubits: smallvec![qubit],
                });
            }
        }
        Rule::Idle { channel } => {
            let Some(idle) = idle else { return };
            for &qubit in &idle[idx] {
                slot.push(NoiseEvent {
                    channel: channel.clone(),
                    qubits: smallvec![qubit],
                });
            }
        }
        Rule::AfterReset { channel } => {
            if let Instruction::Reset { qubit } = instr {
                slot.push(NoiseEvent {
                    channel: channel.clone(),
                    qubits: smallvec![*qubit],
                });
            }
        }
        Rule::BeforeMeasure { channel } => {
            let Some(pre_measure) = pre_measure else {
                return;
            };
            if let Some(qubit) = pre_measure[idx] {
                slot.push(NoiseEvent {
                    channel: channel.clone(),
                    qubits: smallvec![qubit],
                });
            }
        }
        Rule::ScheduledIdle { coherence } => {
            let Some(layer) = context.layers.as_ref().and_then(|l| l.ending_at(idx)) else {
                return;
            };
            for (qubit, &busy) in layer.busy.iter().enumerate() {
                let idle_time = layer.duration - busy;
                if idle_time <= 0.0 {
                    continue;
                }
                let (t1, t2) = coherence[qubit];
                after_gate[layer.slots[qubit]].push(NoiseEvent {
                    channel: NoiseChannel::ThermalRelaxation {
                        t1,
                        t2,
                        gate_time: idle_time,
                        excited_population: 0.0,
                    },
                    qubits: smallvec![qubit],
                });
            }
        }
        Rule::Detuning { drift } => {
            let Some(layer) = context.layers.as_ref().and_then(|l| l.ending_at(idx)) else {
                return;
            };
            let Some((base, factor)) = &context.drift[rule_index] else {
                return;
            };
            if layer.duration <= 0.0 {
                return;
            }
            for qubit in 0..context.num_qubits {
                let weights = DriftDistribution::weights(
                    factor,
                    drift.num_qubits(),
                    qubit,
                    *base,
                    layer.duration,
                );
                if !weights.is_empty() {
                    after_gate[layer.slots[qubit]].push(NoiseEvent {
                        channel: NoiseChannel::QuasiStatic {
                            axis: PauliAxis::Z,
                            weights,
                        },
                        qubits: smallvec![qubit],
                    });
                }
            }
        }
        Rule::OverRotationDrift { filter, sigma } => {
            let Instruction::Gate { gate, targets } = instr else {
                return;
            };
            let Some((axis, theta)) = rotation_axis(gate) else {
                return;
            };
            let Some((base, _)) = &context.drift[rule_index] else {
                return;
            };
            if !filter.matches(gate, targets) || *sigma == 0.0 || theta == 0.0 {
                return;
            }
            for &qubit in targets.iter().filter(|&&q| filter.allows(q)) {
                slot.push(NoiseEvent {
                    channel: NoiseChannel::QuasiStatic {
                        axis,
                        weights: vec![(*base, sigma * theta)],
                    },
                    qubits: smallvec![qubit],
                });
            }
        }
        Rule::OverRotationDriftPerQubit { filter, drift } => {
            let Instruction::Gate { gate, targets } = instr else {
                return;
            };
            let Some((axis, theta)) = rotation_axis(gate) else {
                return;
            };
            let Some((base, factor)) = &context.drift[rule_index] else {
                return;
            };
            if !filter.matches(gate, targets) || theta == 0.0 {
                return;
            }
            for &qubit in targets.iter().filter(|&&q| filter.allows(q)) {
                let weights =
                    DriftDistribution::weights(factor, drift.num_qubits(), qubit, *base, theta);
                if !weights.is_empty() {
                    slot.push(NoiseEvent {
                        channel: NoiseChannel::QuasiStatic { axis, weights },
                        qubits: smallvec![qubit],
                    });
                }
            }
        }
    }
}

/// Axis and angle of a gate carrying one single-qubit rotation angle, the
/// gates the over-rotation rules fire on; `p` is `rz` up to a global phase.
fn rotation_axis(gate: &Gate) -> Option<(PauliAxis, f64)> {
    match *gate {
        Gate::Rx(theta) => Some((PauliAxis::X, theta)),
        Gate::Ry(theta) => Some((PauliAxis::Y, theta)),
        Gate::Rz(theta) | Gate::P(theta) => Some((PauliAxis::Z, theta)),
        _ => None,
    }
}

fn emit_crosstalk(
    filter: &GateFilter,
    coupling: &[(usize, usize)],
    channel: &NoiseChannel,
    targets: &[usize],
    slot: &mut Vec<NoiseEvent>,
) {
    let mut seen: Vec<usize> = Vec::new();
    for &target in targets.iter().filter(|&&q| filter.allows(q)) {
        let mut spectators: Vec<usize> = coupling
            .iter()
            .filter_map(|&(a, b)| match (a == target, b == target) {
                (true, false) => Some(b),
                (false, true) => Some(a),
                _ => None,
            })
            .filter(|spectator| !targets.contains(spectator))
            .collect();
        spectators.sort_unstable();
        spectators.dedup();

        for spectator in spectators {
            let qubits: SmallVec<[usize; 2]> = if channel.num_qubits() == 2 {
                smallvec![target, spectator]
            } else {
                if seen.contains(&spectator) {
                    continue;
                }
                seen.push(spectator);
                smallvec![spectator]
            };
            slot.push(NoiseEvent {
                channel: channel.clone(),
                qubits,
            });
        }
    }
}

/// The unitary an over-rotation of `relative` appends after `gate`, as a
/// one-operator Kraus set. `None` for a gate carrying no single rotation angle.
fn over_rotation_channel(gate: &Gate, relative: f64) -> Option<NoiseChannel> {
    let excess = match gate {
        Gate::Rx(theta) => Gate::Rx(relative * theta),
        Gate::Ry(theta) => Gate::Ry(relative * theta),
        Gate::Rz(theta) => Gate::Rz(relative * theta),
        Gate::P(theta) => Gate::P(relative * theta),
        _ => return None,
    };
    Some(NoiseChannel::Custom {
        kraus: vec![excess.matrix_2x2()],
    })
}

/// Qubits an instruction occupies. Empty for a barrier, which schedules
/// rather than acts; [`idle_qubits_by_layer`] handles that case itself.
fn instruction_qubits(instr: &Instruction) -> SmallVec<[usize; 4]> {
    match instr {
        Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => {
            targets.clone()
        }
        Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => smallvec![*qubit],
        Instruction::Barrier { qubits } | Instruction::Save { qubits, .. } => qubits.clone(),
        Instruction::Region(region) => region.qubits().iter().copied().collect(),
    }
}

/// For the highest-indexed instruction of each layer, the qubits that layer
/// left idle; empty everywhere else.
///
/// Layer assignment is the greedy rule `Circuit::depth` reports: an instruction
/// takes the earliest layer where every qubit it touches is free, and a barrier
/// synchronizes its qubits to that layer without occupying it.
fn idle_qubits_by_layer(circuit: &Circuit) -> Vec<Vec<usize>> {
    let num_qubits = circuit.num_qubits;
    let mut qubit_depth = vec![0usize; num_qubits];
    let mut layers: Vec<(Vec<bool>, usize)> = Vec::new();

    for (idx, instr) in circuit.instructions.iter().enumerate() {
        let qubits = instruction_qubits(instr);
        if qubits.is_empty() {
            continue;
        }
        let layer = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
        if matches!(instr, Instruction::Barrier { .. }) {
            for &qubit in &qubits {
                qubit_depth[qubit] = layer;
            }
            continue;
        }
        while layers.len() <= layer {
            layers.push((vec![false; num_qubits], 0));
        }
        let (touched, last) = &mut layers[layer];
        for &qubit in &qubits {
            touched[qubit] = true;
            qubit_depth[qubit] = layer + 1;
        }
        *last = (*last).max(idx);
    }

    let mut idle = vec![Vec::new(); circuit.instructions.len()];
    for (touched, last) in layers {
        idle[last] = touched
            .iter()
            .enumerate()
            .filter_map(|(qubit, &used)| (!used).then_some(qubit))
            .collect();
    }
    idle
}

/// One layer of an as-soon-as-possible schedule.
struct ScheduledLayer {
    /// Highest-indexed instruction in the layer, which emits its events.
    last: usize,
    /// Longest instruction duration in the layer, in seconds.
    duration: f64,
    /// Time each qubit spends inside an instruction of the layer.
    busy: Vec<f64>,
    /// Slot carrying each qubit's events for the layer: the one just before the
    /// qubit's first instruction in a later layer, or the last slot when it has
    /// none. Instruction order alone can put a later layer's gate on the qubit,
    /// or its measurement, ahead of the layer's highest index.
    slots: Vec<usize>,
}

struct ScheduledLayers {
    layers: Vec<ScheduledLayer>,
    /// The layer each instruction ends, if any.
    ending: Vec<Option<usize>>,
}

impl ScheduledLayers {
    fn ending_at(&self, idx: usize) -> Option<&ScheduledLayer> {
        self.ending[idx].map(|layer| &self.layers[layer])
    }
}

/// The greedy layers of [`idle_qubits_by_layer`], timed with `times`.
fn scheduled_layers(circuit: &Circuit, times: &GateTimes) -> ScheduledLayers {
    let num_qubits = circuit.num_qubits;
    let mut qubit_depth = vec![0usize; num_qubits];
    let mut layers: Vec<ScheduledLayer> = Vec::new();
    let mut uses: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_qubits];
    let final_slot = circuit.instructions.len().saturating_sub(1);

    for (idx, instr) in circuit.instructions.iter().enumerate() {
        let qubits = instruction_qubits(instr);
        if qubits.is_empty() {
            continue;
        }
        let layer = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
        if matches!(instr, Instruction::Barrier { .. }) {
            for &qubit in &qubits {
                qubit_depth[qubit] = layer;
            }
            continue;
        }
        while layers.len() <= layer {
            layers.push(ScheduledLayer {
                last: 0,
                duration: 0.0,
                busy: vec![0.0; num_qubits],
                slots: vec![final_slot; num_qubits],
            });
        }
        let duration = times.duration(instr);
        let entry = &mut layers[layer];
        for &qubit in &qubits {
            entry.busy[qubit] = entry.busy[qubit].max(duration);
            qubit_depth[qubit] = layer + 1;
            uses[qubit].push((layer, idx));
        }
        entry.duration = entry.duration.max(duration);
        entry.last = entry.last.max(idx);
    }

    for (qubit, uses) in uses.iter().enumerate() {
        let mut next = 0;
        for (index, layer) in layers.iter_mut().enumerate() {
            while next < uses.len() && uses[next].0 <= index {
                next += 1;
            }
            let slot = uses.get(next).map_or(final_slot, |&(_, idx)| idx - 1);
            layer.slots[qubit] = slot;
        }
    }

    let mut ending = vec![None; circuit.instructions.len()];
    for (index, layer) in layers.iter().enumerate() {
        ending[layer.last] = Some(index);
    }
    ScheduledLayers { layers, ending }
}

/// The qubit measured by instruction `idx + 1`, indexed by `idx`, so a
/// pre-measurement channel rides the preceding instruction's slot.
fn pre_measure_qubits(circuit: &Circuit) -> Result<Vec<Option<usize>>> {
    if let Some(Instruction::Measure { .. }) = circuit.instructions.first() {
        return Err(PrismError::InvalidParameter {
            message: "pre-measurement noise needs a preceding instruction to attach to, and \
                      instruction 0 is a measurement; prepend a barrier"
                .into(),
        });
    }
    let mut out = vec![None; circuit.instructions.len()];
    for (idx, instr) in circuit.instructions.iter().enumerate().skip(1) {
        if let Instruction::Measure { qubit, .. } = instr {
            out[idx - 1] = Some(*qubit);
        }
    }
    Ok(out)
}
