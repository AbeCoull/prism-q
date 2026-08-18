//! Rule-based construction of a [`NoiseModel`] against a circuit.
//!
//! Rules are declared once and compiled against an instruction stream by
//! [`NoiseBuilder::build`], which emits the same per-instruction event vector
//! the manual `after_gate` path builds. Nothing here runs per shot.

use smallvec::smallvec;

use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, ReadoutError};

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
    /// [`before_measurements`](NoiseBuilder::before_measurements) rule, and
    /// everything [`NoiseModel::validate_for`] rejects.
    pub fn build(&self, circuit: &Circuit) -> Result<NoiseModel> {
        let mut after_gate: Vec<Vec<NoiseEvent>> = vec![Vec::new(); circuit.instructions.len()];

        let idle = self
            .rules
            .iter()
            .any(|rule| matches!(rule, Rule::Idle { .. }))
            .then(|| idle_qubits_by_layer(circuit));
        let pre_measure = self
            .rules
            .iter()
            .any(|rule| matches!(rule, Rule::BeforeMeasure { .. }))
            .then(|| pre_measure_qubits(circuit))
            .transpose()?;

        for (idx, instr) in circuit.instructions.iter().enumerate() {
            for rule in &self.rules {
                emit(rule, idx, instr, &idle, &pre_measure, &mut after_gate);
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

fn emit(
    rule: &Rule,
    idx: usize,
    instr: &Instruction,
    idle: &Option<Vec<Vec<usize>>>,
    pre_measure: &Option<Vec<Option<usize>>>,
    after_gate: &mut [Vec<NoiseEvent>],
) {
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
        Instruction::Barrier { qubits } => qubits.clone(),
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
