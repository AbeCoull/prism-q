//! Detector error model derivation and text export for native QEC programs.
//!
//! Expands each Pauli-noise annotation into its fault branches, propagates
//! every branch to the detectors and observables it flips, and merges branches
//! with identical symptoms into weighted error mechanisms.

use std::collections::HashMap;

use super::noise::lower_qec_program_to_deferred_circuit;
use super::parity_walk::compile_fault_sites;
use super::runner::QecParityProjection;
use super::{QecOp, QecProgram};
use crate::error::{PrismError, Result};

/// One error mechanism: an independent fault process that flips a fixed set of
/// detectors and observables with the given probability.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorMechanism {
    probability: f64,
    detectors: Vec<usize>,
    observables: Vec<usize>,
}

impl ErrorMechanism {
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Detector indices this mechanism flips, ascending.
    pub fn detectors(&self) -> &[usize] {
        &self.detectors
    }

    /// Observable indices this mechanism flips, ascending.
    pub fn observables(&self) -> &[usize] {
        &self.observables
    }
}

/// Detector error model: independent error mechanisms over a program's
/// detectors and observables.
///
/// Derived by [`QecProgram::detector_error_model`]. Mechanisms are ordered by
/// the program position of the noise annotation that first produced each
/// symptom.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorErrorModel {
    mechanisms: Vec<ErrorMechanism>,
    detector_coords: Vec<Vec<f64>>,
    num_detectors: usize,
    num_observables: usize,
}

impl DetectorErrorModel {
    pub fn mechanisms(&self) -> &[ErrorMechanism] {
        &self.mechanisms
    }

    pub fn num_mechanisms(&self) -> usize {
        self.mechanisms.len()
    }

    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Coordinates per detector, verbatim from the program; empty when the op carried none.
    pub fn detector_coords(&self) -> &[Vec<f64>] {
        &self.detector_coords
    }

    /// Decompose hypergraph mechanisms into graphlike components.
    ///
    /// Returns a model in which every mechanism flips at most two detectors.
    /// A mechanism with more is replaced by existing graphlike mechanisms
    /// whose non-empty detector sets partition its detectors and whose
    /// observable XOR matches; its probability composes into every component
    /// as `p = p1(1-p2) + p2(1-p1)`. Cross-component correlations are lost;
    /// single-detector marginals are unchanged. Deterministic: retained
    /// mechanisms keep their order, the first cover in mechanism order wins.
    ///
    /// # Errors
    ///
    /// A hypergraph mechanism with no cover; the error names its symptom.
    pub fn decompose_graphlike(&self) -> Result<DetectorErrorModel> {
        let mut graphlike: Vec<ErrorMechanism> = Vec::new();
        for mechanism in &self.mechanisms {
            if mechanism.detectors.len() <= 2 {
                graphlike.push(mechanism.clone());
            }
        }

        for mechanism in &self.mechanisms {
            if mechanism.detectors.len() <= 2 {
                continue;
            }
            let Some(components) = partition_cover(mechanism, &graphlike) else {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "graphlike decomposition failed: mechanism `{}` has no \
                         cover by graphlike mechanisms",
                        symptom_label(mechanism)
                    ),
                });
            };
            let p = mechanism.probability;
            for at in components {
                let prior = graphlike[at].probability;
                graphlike[at].probability = prior * (1.0 - p) + p * (1.0 - prior);
            }
        }

        Ok(DetectorErrorModel {
            mechanisms: graphlike,
            detector_coords: self.detector_coords.clone(),
            num_detectors: self.num_detectors,
            num_observables: self.num_observables,
        })
    }

    /// Render the model in the common detector error model text format.
    ///
    /// One `error(p) D.. L..` line per mechanism in mechanism order, then one
    /// `detector` line per detector (with its coordinates when present), then
    /// one `logical_observable` line per observable slot. The grammar is
    /// documented in `docs/architecture/qec-programs.md`.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for mechanism in &self.mechanisms {
            out.push_str(&format!("error({})", mechanism.probability));
            for detector in &mechanism.detectors {
                out.push_str(&format!(" D{detector}"));
            }
            for observable in &mechanism.observables {
                out.push_str(&format!(" L{observable}"));
            }
            out.push('\n');
        }
        for (detector, coords) in self.detector_coords.iter().enumerate() {
            if coords.is_empty() {
                out.push_str(&format!("detector D{detector}\n"));
            } else {
                let coords = coords
                    .iter()
                    .map(f64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push_str(&format!("detector({coords}) D{detector}\n"));
            }
        }
        for observable in 0..self.num_observables {
            out.push_str(&format!("logical_observable L{observable}\n"));
        }
        out
    }
}

impl QecProgram {
    /// Derive the detector error model implied by the program's Pauli-noise
    /// annotations, detectors, and observables.
    ///
    /// Each annotation expands into Pauli fault branches per fault site (the nonzero
    /// ones of three per target for a one-qubit channel, of fifteen per target pair
    /// for a two-qubit channel), each propagated to the detectors and observables
    /// it flips. Exclusive branches at one site with the same symptom sum;
    /// independent sites (distinct targets of one annotation included) with the same
    /// symptom compose as `p = p1(1-p2) + p2(1-p1)`.
    /// Faults that flip nothing are omitted. Mechanisms are independent in the
    /// model, so its joint statistics agree with the sampler to second order in
    /// the branch probabilities.
    ///
    /// # Errors
    ///
    /// Requires the compiled Clifford path: non-Clifford gates and reuse of a
    /// measured qubit without reset are rejected.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_q::QecProgram;
    ///
    /// let program = QecProgram::from_text(
    ///     "X_ERROR(0.05) 0 1 2
    ///      CX 0 3 1 3 1 4 2 4
    ///      M 3 4
    ///      DETECTOR rec[-2]
    ///      DETECTOR rec[-1]",
    /// )?;
    /// let model = program.detector_error_model()?;
    /// assert_eq!(model.num_detectors(), 2);
    /// assert_eq!(model.num_mechanisms(), 3);
    /// # Ok::<(), prism_q::PrismError>(())
    /// ```
    pub fn detector_error_model(&self) -> Result<DetectorErrorModel> {
        derive_detector_error_model(self)
    }
}

/// Flipped (detector indices, observable indices), both ascending.
type Symptom = (Vec<usize>, Vec<usize>);

fn derive_detector_error_model(program: &QecProgram) -> Result<DetectorErrorModel> {
    let detector_rows = program.detector_rows()?;
    let observable_rows = program.observable_rows()?;
    let num_detectors = detector_rows.len();
    let num_observables = observable_rows.len();
    let projection =
        QecParityProjection::new(program.num_measurements(), &detector_rows, &observable_rows);
    let deferred = lower_qec_program_to_deferred_circuit(program)?;
    let sites = compile_fault_sites(&deferred, &projection)?;

    let mut index: HashMap<Symptom, usize> = HashMap::new();
    let mut mechanisms: Vec<ErrorMechanism> = Vec::new();
    let mut local: Vec<(Symptom, f64)> = Vec::new();
    for site in sites.sites() {
        local.clear();
        for (probability, outputs) in site {
            if outputs.is_empty() {
                continue;
            }
            let split = outputs.partition_point(|&output| (output as usize) < num_detectors);
            let symptom: Symptom = (
                outputs[..split]
                    .iter()
                    .map(|&output| output as usize)
                    .collect(),
                outputs[split..]
                    .iter()
                    .map(|&output| output as usize - num_detectors)
                    .collect(),
            );
            match local.iter_mut().find(|(existing, _)| *existing == symptom) {
                Some((_, total)) => *total += probability,
                None => local.push((symptom, probability)),
            }
        }
        for (symptom, probability) in local.drain(..) {
            match index.get(&symptom) {
                Some(&at) => {
                    let prior = mechanisms[at].probability;
                    mechanisms[at].probability =
                        prior * (1.0 - probability) + probability * (1.0 - prior);
                }
                None => {
                    index.insert(symptom.clone(), mechanisms.len());
                    let (detectors, observables) = symptom;
                    mechanisms.push(ErrorMechanism {
                        probability,
                        detectors,
                        observables,
                    });
                }
            }
        }
    }

    Ok(DetectorErrorModel {
        mechanisms,
        detector_coords: detector_coordinates(program),
        num_detectors,
        num_observables,
    })
}

/// Depth-first over candidates in mechanism order; the first cover found is
/// the result.
fn partition_cover(mechanism: &ErrorMechanism, graphlike: &[ErrorMechanism]) -> Option<Vec<usize>> {
    fn search(
        remaining: &[usize],
        observables: &[usize],
        start: usize,
        graphlike: &[ErrorMechanism],
        chosen: &mut Vec<usize>,
    ) -> bool {
        if remaining.is_empty() {
            return observables.is_empty();
        }
        for at in start..graphlike.len() {
            let candidate = &graphlike[at];
            if candidate.detectors.is_empty() || !is_subset(&candidate.detectors, remaining) {
                continue;
            }
            let next_remaining = symmetric_difference(remaining, &candidate.detectors);
            let next_observables = symmetric_difference(observables, &candidate.observables);
            chosen.push(at);
            if search(
                &next_remaining,
                &next_observables,
                at + 1,
                graphlike,
                chosen,
            ) {
                return true;
            }
            chosen.pop();
        }
        false
    }

    let mut chosen = Vec::new();
    search(
        &mechanism.detectors,
        &mechanism.observables,
        0,
        graphlike,
        &mut chosen,
    )
    .then_some(chosen)
}

/// True when every element of ascending `a` appears in ascending `b`.
fn is_subset(a: &[usize], b: &[usize]) -> bool {
    let mut j = 0;
    'outer: for &x in a {
        while j < b.len() {
            match b[j].cmp(&x) {
                std::cmp::Ordering::Less => j += 1,
                std::cmp::Ordering::Equal => {
                    j += 1;
                    continue 'outer;
                }
                std::cmp::Ordering::Greater => return false,
            }
        }
        return false;
    }
    true
}

/// Symmetric difference of two ascending index lists, ascending.
fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

pub(super) fn symptom_label(mechanism: &ErrorMechanism) -> String {
    let mut label = String::new();
    for detector in &mechanism.detectors {
        if !label.is_empty() {
            label.push(' ');
        }
        label.push_str(&format!("D{detector}"));
    }
    for observable in &mechanism.observables {
        if !label.is_empty() {
            label.push(' ');
        }
        label.push_str(&format!("L{observable}"));
    }
    label
}

fn detector_coordinates(program: &QecProgram) -> Vec<Vec<f64>> {
    program
        .ops()
        .iter()
        .filter_map(|op| match op {
            QecOp::Detector { coords, .. } => Some(coords.clone()),
            _ => None,
        })
        .collect()
}
