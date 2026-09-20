//! Evaluating the result requests a Braket program declared, exactly or from a
//! shot record.
//!
//! Values come back in Braket's conventions, where qubit 0 is the most
//! significant bit of a basis index and `q[0]` is the least significant one
//! here, so an exported vector is reversed on the way out.

use std::collections::{BTreeMap, HashMap};

use num_complex::Complex64;

use crate::circuit::Instruction;
use crate::circuit::braket::{MeasuredFactor, ObservableFactor, ResultSpec, Targets};
use crate::error::{PrismError, Result};
use crate::sim::observable::PauliObservable;
use crate::sim::unified_pauli::PauliTerm;
use crate::sim::{Seeded, Simulate};

/// One computed `#pragma braket result`, in Braket's conventions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ResultValue {
    /// Amplitudes with qubit 0 in the most significant bit.
    StateVector(Vec<Complex64>),
    /// Row major over the requested targets, `targets[0]` the most significant
    /// bit of both indices.
    DensityMatrix(Vec<Vec<Complex64>>),
    /// One amplitude per requested basis state, in the order requested.
    Amplitude(Vec<(String, Complex64)>),
    /// Joint probabilities over the requested targets, `targets[0]` the most
    /// significant bit.
    Probability(Vec<f64>),
    /// One value, or one per qubit in qubit order when the observable named no
    /// targets.
    Expectation(Vec<f64>),
    /// `<O^2> - <O>^2`, shaped like [`ResultValue::Expectation`].
    Variance(Vec<f64>),
    /// One eigenvalue per shot, in one series per value
    /// [`ResultValue::Expectation`] would report.
    Sample(Vec<Vec<f64>>),
}

/// The Pauli sums one result request reports, each beside what a variance
/// needs: the constant term held out of the square, and the square of the rest.
type Requested = Vec<(PauliObservable, Option<(f64, PauliObservable)>)>;

impl<'c> Simulate<'c, Seeded> {
    /// Evaluate the result requests a Braket program declared.
    ///
    /// Every `expectation` and `variance` request is served by a single
    /// traversal: their observables are lowered to Pauli sums, the distinct
    /// strings across all of them are evaluated together through
    /// [`Simulate::expectation_values`], and each requested value is then a
    /// weighted sum over that one evaluation. `state_vector` and `amplitude`
    /// share one export, and a `probability` or `density_matrix` request beside
    /// them is read off that same export. Without one it runs on its own,
    /// keeping the routing its own width earns: a subset marginal of a wide
    /// Clifford circuit stays on the tableau rather than forcing a dense
    /// export.
    ///
    /// `sample` is declined here: it reports per-shot eigenvalues, which
    /// [`Simulate::braket_results_sampled`] answers. So is a circuit carrying a
    /// measurement, reset or conditional, which has no one exact output state
    /// to read: Braket rejects the same programs at zero shots.
    ///
    /// # Errors
    /// Returns `BackendUnsupported` for a `sample` request,
    /// `IncompatibleBackend` for a circuit that is not unitary, and whatever
    /// the underlying terminal returns for a route that cannot serve a request.
    pub fn braket_results(self, specs: &[ResultSpec]) -> Result<Vec<ResultValue>> {
        let num_qubits = self.circuit.num_qubits;
        crate::sim::require_unitary_circuit(
            &self.kind,
            self.circuit,
            "an exact result request reads",
        )?;
        let requested = specs
            .iter()
            .map(|spec| observables_of(spec, num_qubits))
            .collect::<Result<Vec<_>>>()?;

        let mut index: BTreeMap<Vec<PauliTerm>, usize> = BTreeMap::new();
        for sums in requested.iter().flatten() {
            for (sum, squared) in sums {
                let squared = squared.as_ref().map(|(_, square)| square);
                for source in [Some(sum), squared].into_iter().flatten() {
                    for (_, string) in source.terms() {
                        if !string.is_empty() {
                            let next = index.len();
                            index.entry(string.clone()).or_insert(next);
                        }
                    }
                }
            }
        }
        let values = if index.is_empty() {
            Vec::new()
        } else {
            let mut strings = vec![Vec::new(); index.len()];
            for (string, &slot) in &index {
                strings[slot] = string.clone();
            }
            self.fork().expectation_values(&strings)?
        };

        let mut state = None;
        if specs
            .iter()
            .any(|spec| matches!(spec, ResultSpec::StateVector | ResultSpec::Amplitude(_)))
        {
            self.state_once(&mut state)?;
        }
        let mut computed = Vec::with_capacity(specs.len());
        for (spec, sums) in specs.iter().zip(&requested) {
            computed.push(match (spec, sums) {
                (ResultSpec::StateVector, _) => {
                    let amplitudes = self.state_once(&mut state)?;
                    ResultValue::StateVector(reverse_state(amplitudes, num_qubits))
                }
                (ResultSpec::Amplitude(states), _) => {
                    let amplitudes = self.state_once(&mut state)?;
                    ResultValue::Amplitude(
                        states
                            .iter()
                            .map(|label| {
                                let at = basis_index(label, num_qubits)?;
                                Ok((label.clone(), amplitudes[at]))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )
                }
                (ResultSpec::Probability(targets), _) => {
                    let targets = resolve(targets, num_qubits);
                    let joint = match state.as_deref() {
                        Some(amplitudes) => {
                            crate::backend::schmidt::validate_qubit_set(&targets, num_qubits)?;
                            marginal_of(amplitudes, &targets, num_qubits)
                        }
                        None => self.fork().probabilities_of(&targets)?,
                    };
                    ResultValue::Probability(reverse_state(&joint, targets.len()))
                }
                (ResultSpec::DensityMatrix(targets), _) => {
                    let targets = resolve(targets, num_qubits);
                    let data = match state.as_deref() {
                        Some(amplitudes) => {
                            crate::backend::schmidt::validate_qubit_set(&targets, num_qubits)?;
                            reduced_of(amplitudes, &targets, num_qubits)
                        }
                        None => self.fork().reduced_density_matrix(&targets)?.data,
                    };
                    ResultValue::DensityMatrix(reverse_matrix(&data, targets.len()))
                }
                (ResultSpec::Expectation(_), Some(sums)) => ResultValue::Expectation(
                    sums.iter()
                        .map(|(sum, _)| weighted(sum, &index, &values))
                        .collect(),
                ),
                (ResultSpec::Variance(_), Some(sums)) => ResultValue::Variance(
                    sums.iter()
                        .map(|(sum, squared)| {
                            let (offset, square) =
                                squared.as_ref().expect("a variance carries its square");
                            let centered = weighted(sum, &index, &values) - offset;
                            weighted(square, &index, &values) - centered * centered
                        })
                        .collect(),
                ),
                (spec, _) => {
                    return Err(PrismError::BackendUnsupported {
                        backend: format!("{:?}", self.kind),
                        operation: format!(
                            "`{}`, which reports per-shot eigenvalues and so needs measurement \
                             in the observable's own basis rather than an exact value",
                            spec.name()
                        ),
                    });
                }
            });
        }
        Ok(computed)
    }

    /// Evaluate the result requests a Braket program declared, from a shot
    /// record rather than from the exact state.
    ///
    /// Each observable is diagonalized and the rotations carrying them onto the
    /// computational basis are appended to the circuit once, so a single
    /// sampling pass answers every `sample`, `expectation` and `variance`
    /// request together. Two observables reading one qubit in different bases
    /// cannot share a record and are rejected rather than answered from
    /// whichever basis was applied first. A `probability` request reads the
    /// computational basis and so takes its own unrotated pass whenever any
    /// rotation was applied.
    ///
    /// # Errors
    /// Returns `BackendUnsupported` for `state_vector`, `density_matrix` and
    /// `amplitude`, which report the state itself and which Braket admits only
    /// at zero shots, and `InvalidParameter` for zero shots or for observables
    /// that cannot share one measurement.
    pub fn braket_results_sampled(
        self,
        specs: &[ResultSpec],
        shots: usize,
    ) -> Result<Vec<ResultValue>> {
        let num_qubits = self.circuit.num_qubits;
        if shots == 0 {
            return Err(PrismError::InvalidParameter {
                message: "a shot-based evaluation needs at least one shot".into(),
            });
        }
        if let Some(spec) = specs.iter().find(|spec| spec.requires_exact()) {
            return Err(PrismError::BackendUnsupported {
                backend: format!("{:?}", self.kind),
                operation: format!(
                    "`{}` above zero shots, since it reports the state itself rather than a \
                     measurement of it",
                    spec.name()
                ),
            });
        }

        let measured = specs
            .iter()
            .map(|spec| match spec {
                ResultSpec::Expectation(observable)
                | ResultSpec::Variance(observable)
                | ResultSpec::Sample(observable) => observable.diagonalize(num_qubits).map(Some),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;

        let rotations = merge_rotations(&measured)?;
        let record = self.sample_record(&rotations, shots)?;
        let unrotated = if rotations.is_empty()
            || !specs
                .iter()
                .any(|spec| matches!(spec, ResultSpec::Probability(_)))
        {
            None
        } else {
            Some(self.sample_record(&Rotations::new(), shots)?)
        };

        let series = |groups: &[Vec<MeasuredFactor>]| -> Vec<Vec<f64>> {
            groups
                .iter()
                .map(|group| record.iter().map(|bits| shot_value(group, bits)).collect())
                .collect()
        };
        let mut computed = Vec::with_capacity(specs.len());
        for (spec, groups) in specs.iter().zip(&measured) {
            computed.push(match (spec, groups) {
                (ResultSpec::Sample(_), Some(groups)) => ResultValue::Sample(series(groups)),
                (ResultSpec::Expectation(_), Some(groups)) => ResultValue::Expectation(
                    series(groups).iter().map(|values| mean(values)).collect(),
                ),
                (ResultSpec::Variance(_), Some(groups)) => ResultValue::Variance(
                    series(groups)
                        .iter()
                        .map(|values| variance(values))
                        .collect(),
                ),
                (ResultSpec::Probability(targets), _) => {
                    let targets = resolve(targets, num_qubits);
                    crate::backend::schmidt::validate_qubit_set(&targets, num_qubits)?;
                    if targets.len() > crate::backend::schmidt::export_cap() {
                        return Err(crate::backend::schmidt::export_cap_exceeded(
                            &format!("{:?}", self.kind),
                            format!("a probability over {} qubits", targets.len()),
                        ));
                    }
                    let source = unrotated.as_ref().unwrap_or(&record);
                    ResultValue::Probability(histogram(&targets, source))
                }
                (spec, _) => unreachable!("`{}` was screened above", spec.name()),
            });
        }
        Ok(computed)
    }

    /// One shot record of every qubit, taken after `rotations`, with only the
    /// bits this call appended.
    ///
    /// An attached noise model is indexed per instruction, so it is extended
    /// with empty slots over the appended rotation and readout: the basis
    /// change is a reading device rather than part of the program, and the
    /// program's own readout error stays on the bits it was declared for.
    fn sample_record(&self, rotations: &Rotations, shots: usize) -> Result<Vec<Vec<bool>>> {
        let num_qubits = self.circuit.num_qubits;
        let mut circuit = self.circuit.clone();
        for instrs in rotations.values() {
            circuit.instructions.extend(instrs.iter().cloned());
        }
        let base = circuit.num_classical_bits;
        circuit.num_classical_bits = base + num_qubits;
        for qubit in 0..num_qubits {
            circuit.add_measure(qubit, base + qubit);
        }
        let extended = self.noise_model.map(|model| {
            let mut model = model.clone();
            model
                .after_gate
                .resize(circuit.instructions.len(), Vec::new());
            model.readout.resize(circuit.num_classical_bits, None);
            model
        });
        let sampled = Simulate::<Seeded> {
            circuit: &circuit,
            kind: self.kind.clone(),
            seed: self.seed,
            noise_model: extended.as_ref(),
            initial_state: self.initial_state,
            require_exact: self.require_exact,
        }
        .shots(shots)?;
        // The appended readout sits at the end of the record, so trimming in
        // place leaves the bits this call wrote without copying every shot.
        Ok(sampled
            .shots
            .into_iter()
            .map(|mut bits| {
                bits.truncate(base + num_qubits);
                bits.drain(..base);
                bits
            })
            .collect())
    }

    /// Export the output state, reusing an earlier export within the same call.
    fn state_once<'s>(&self, cache: &'s mut Option<Vec<Complex64>>) -> Result<&'s [Complex64]> {
        if cache.is_none() {
            *cache = Some(self.fork().state_vector()?);
        }
        Ok(cache.as_deref().expect("just filled"))
    }

    /// A copy of the request, so one builder can drive several terminals.
    fn fork(&self) -> Simulate<'c, Seeded> {
        Simulate {
            circuit: self.circuit,
            kind: self.kind.clone(),
            seed: self.seed,
            noise_model: self.noise_model,
            initial_state: self.initial_state,
            require_exact: self.require_exact,
        }
    }
}

/// Pauli sums a request needs, `None` for a request that reads the state
/// rather than an observable.
fn observables_of(spec: &ResultSpec, num_qubits: usize) -> Result<Option<Requested>> {
    let (observable, squared) = match spec {
        ResultSpec::Expectation(observable) => (observable, false),
        ResultSpec::Variance(observable) => (observable, true),
        _ => return Ok(None),
    };
    Ok(Some(
        observable
            .lower(num_qubits)?
            .into_iter()
            .map(|(_, sum)| {
                let square = squared.then(|| {
                    let (offset, traceless) = sum.split_identity();
                    (offset, traceless.square())
                });
                (sum, square)
            })
            .collect(),
    ))
}

fn weighted(sum: &PauliObservable, index: &BTreeMap<Vec<PauliTerm>, usize>, values: &[f64]) -> f64 {
    sum.terms()
        .iter()
        .map(|(coefficient, string)| {
            if string.is_empty() {
                *coefficient
            } else {
                coefficient * values[index[string]]
            }
        })
        .sum()
}

/// The rotation each measured qubit set takes before it is read, keyed by the
/// qubits so the order a circuit receives them in is fixed.
type Rotations = BTreeMap<Vec<usize>, Vec<Instruction>>;

/// One rotation per qubit set, rejecting observables that cannot share a
/// measurement: two reading a qubit in different bases, or reading overlapping
/// but unequal qubit sets.
fn merge_rotations(measured: &[Option<Vec<Vec<MeasuredFactor>>>]) -> Result<Rotations> {
    let conflict = |qubit: usize| PrismError::InvalidParameter {
        message: format!(
            "two observables read qubit {qubit} in different bases, which one measurement cannot \
             serve; request them separately"
        ),
    };
    let mut rotations = Rotations::new();
    let mut bases: BTreeMap<Vec<usize>, ObservableFactor> = BTreeMap::new();
    let mut claimed: HashMap<usize, Vec<usize>> = HashMap::new();
    for factor in measured.iter().flatten().flatten().flatten() {
        let Some(gates) = &factor.rotation else {
            continue;
        };
        for &qubit in &factor.targets {
            match claimed.get(&qubit) {
                Some(owner) if *owner != factor.targets => return Err(conflict(qubit)),
                Some(_) => {}
                None => {
                    claimed.insert(qubit, factor.targets.clone());
                }
            }
        }
        // Two factors on the same qubits share a measurement only when they
        // read the same basis, which the factor itself decides rather than the
        // instructions it happened to lower to.
        match bases.get(&factor.targets) {
            Some(existing) if *existing != factor.basis => {
                return Err(conflict(factor.targets[0]));
            }
            Some(_) => {}
            None => {
                bases.insert(factor.targets.clone(), factor.basis.clone());
                rotations.insert(factor.targets.clone(), gates.clone());
            }
        }
    }
    Ok(rotations)
}

/// Eigenvalue one shot gives a tensor product: the product over its factors of
/// the eigenvalue each reads.
fn shot_value(group: &[MeasuredFactor], bits: &[bool]) -> f64 {
    group
        .iter()
        .map(|factor| factor.eigenvalues[outcome(&factor.targets, bits)])
        .product()
}

/// Basis index measured bits give a target list, `targets[0]` the most
/// significant, which is how Braket and a gate matrix both pack one.
fn outcome(targets: &[usize], bits: &[bool]) -> usize {
    targets.iter().fold(0usize, |index, &qubit| {
        index << 1 | usize::from(bits[qubit])
    })
}

fn histogram(targets: &[usize], record: &[Vec<bool>]) -> Vec<f64> {
    let mut counts = vec![0.0f64; 1usize << targets.len()];
    for bits in record {
        counts[outcome(targets, bits)] += 1.0;
    }
    let shots = record.len() as f64;
    for count in &mut counts {
        *count /= shots;
    }
    counts
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population variance, the estimator Braket reports for a shot series.
fn variance(values: &[f64]) -> f64 {
    let mean = mean(values);
    values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64
}

fn resolve(targets: &Targets, num_qubits: usize) -> Vec<usize> {
    match targets {
        Targets::All => (0..num_qubits).collect(),
        Targets::These(qubits) => qubits.clone(),
    }
}

/// Index of the basis state a Braket bitstring names: the leftmost character is
/// qubit 0, which is the lowest bit of a PRISM-Q basis index.
fn basis_index(label: &str, num_qubits: usize) -> Result<usize> {
    if label.len() != num_qubits {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "basis state `{label}` names {} qubit(s) of {num_qubits}",
                label.len()
            ),
        });
    }
    label
        .chars()
        .enumerate()
        .try_fold(0usize, |index, (qubit, bit)| match bit {
            '0' => Ok(index),
            '1' => Ok(index | 1 << qubit),
            other => Err(PrismError::InvalidParameter {
                message: format!("basis state `{label}` has `{other}` where a bit belongs"),
            }),
        })
}

/// Basis index `targets` reads out of a full index, `targets[0]` the lowest
/// bit, beside the index the untargeted qubits carry in their own order.
fn split_index(index: usize, targets: &[usize], num_qubits: usize) -> (usize, usize) {
    let mut read = 0usize;
    for (bit, &qubit) in targets.iter().enumerate() {
        read |= (index >> qubit & 1) << bit;
    }
    let mut rest = 0usize;
    let mut bit = 0usize;
    for qubit in 0..num_qubits {
        if targets.contains(&qubit) {
            continue;
        }
        rest |= (index >> qubit & 1) << bit;
        bit += 1;
    }
    (read, rest)
}

/// Joint distribution over `targets` read off an exported state, shaped like
/// [`Simulate::probabilities_of`].
fn marginal_of(amplitudes: &[Complex64], targets: &[usize], num_qubits: usize) -> Vec<f64> {
    let mut joint = vec![0.0f64; 1usize << targets.len()];
    for (index, amplitude) in amplitudes.iter().enumerate() {
        joint[split_index(index, targets, num_qubits).0] += amplitude.norm_sqr();
    }
    joint
}

/// Reduced density matrix of `targets` read off an exported state, shaped like
/// [`ReducedDensityMatrix::data`](crate::ReducedDensityMatrix::data).
fn reduced_of(amplitudes: &[Complex64], targets: &[usize], num_qubits: usize) -> Vec<Complex64> {
    let side = 1usize << targets.len();
    let mut blocks = vec![Complex64::new(0.0, 0.0); amplitudes.len()];
    for (index, amplitude) in amplitudes.iter().enumerate() {
        let (read, rest) = split_index(index, targets, num_qubits);
        blocks[rest * side + read] = *amplitude;
    }
    let mut reduced = vec![Complex64::new(0.0, 0.0); side * side];
    for block in blocks.chunks(side) {
        for row in 0..side {
            if block[row] == Complex64::new(0.0, 0.0) {
                continue;
            }
            for column in 0..side {
                reduced[row * side + column] += block[row] * block[column].conj();
            }
        }
    }
    reduced
}

/// Width of one table-driven reversal step. A table of this many bits costs
/// 4 KiB and covers a 33-qubit index in three lookups.
const CHUNK_BITS: usize = 11;

/// Reversal of every `CHUNK_BITS`-bit value, so reversing a whole index costs
/// one lookup per chunk rather than one pass per bit.
fn chunk_table() -> Vec<u16> {
    (0..1u32 << CHUNK_BITS)
        .map(|value| {
            (0..CHUNK_BITS).fold(0u16, |acc, bit| {
                acc | ((value as u16 >> bit) & 1) << (CHUNK_BITS - 1 - bit)
            })
        })
        .collect()
}

fn reverse_bits_with(table: &[u16], index: usize, width: usize) -> usize {
    let mut reversed = 0usize;
    let mut remaining = width;
    let mut rest = index;
    while remaining > 0 {
        let take = remaining.min(CHUNK_BITS);
        let chunk = rest & ((1usize << take) - 1);
        reversed |= ((table[chunk] >> (CHUNK_BITS - take)) as usize) << (remaining - take);
        rest >>= take;
        remaining -= take;
    }
    reversed
}

fn reverse_state<T: Copy>(values: &[T], width: usize) -> Vec<T> {
    let table = chunk_table();
    (0..values.len())
        .map(|index| values[reverse_bits_with(&table, index, width)])
        .collect()
}

fn reverse_matrix(data: &[Complex64], width: usize) -> Vec<Vec<Complex64>> {
    let side = 1usize << width;
    let table = chunk_table();
    let reversed: Vec<usize> = (0..side)
        .map(|index| reverse_bits_with(&table, index, width))
        .collect();
    reversed
        .iter()
        .map(|&source| {
            reversed
                .iter()
                .map(|&column| data[source * side + column])
                .collect()
        })
        .collect()
}
