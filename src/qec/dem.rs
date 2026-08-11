//! Detector error model derivation and text export for native QEC programs.
//!
//! Expands each Pauli-noise annotation into its fault branches, propagates
//! every branch to the detectors and observables it flips, and merges branches
//! with identical symptoms into weighted error mechanisms.

use std::collections::HashMap;

use super::noise::{
    QecDeferredNoiseEvent, append_qec_pauli_noise_effect, lower_qec_program_to_deferred_circuit,
    walk_qec_noise_sensitivity,
};
use super::{QecNoise, QecOp, QecProgram};
use crate::error::Result;
use crate::sim::compiled::xor_words;

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
/// symptom. Detector coordinates are carried verbatim from the program's
/// detector ops, one entry per detector, empty when the op carried none.
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

    /// Coordinates per detector, in detector order. Empty for detectors whose
    /// op carried no coordinates.
    pub fn detector_coords(&self) -> &[Vec<f64>] {
        &self.detector_coords
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
    /// Every annotation expands into its Pauli fault branches per fault site
    /// (one branch per target for `X_ERROR` / `Z_ERROR`, three per target for
    /// `DEPOLARIZE1`, fifteen per target pair for `DEPOLARIZE2`). Each branch
    /// is propagated through the circuit to the set of detectors and
    /// observables it flips. Mutually exclusive branches at one fault site
    /// with the same symptom sum; independent fault sites (distinct targets
    /// of one annotation included) with the same symptom compose as
    /// `p = p1(1-p2) + p2(1-p1)`. Faults that flip no detector and no
    /// observable are omitted. Mechanisms are independent in the model, so
    /// its joint statistics agree with the sampler to second order in the
    /// branch probabilities.
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

/// One independent random draw in the sampler: the mutually exclusive Pauli
/// branches of a single noise annotation on a single target (or target pair),
/// each with its packed measurement-record flip mask.
struct FaultUnit {
    position: usize,
    branches: Vec<(f64, Vec<u64>)>,
}

/// Flipped (detector indices, observable indices), both ascending.
type Symptom = (Vec<usize>, Vec<usize>);

fn derive_detector_error_model(program: &QecProgram) -> Result<DetectorErrorModel> {
    let detector_rows = program.detector_rows()?;
    let observable_rows = program.observable_rows()?;
    let num_detectors = detector_rows.len();
    let num_observables = observable_rows.len();
    let m_words = program.num_measurements().div_ceil(64);
    let detector_masks = pack_record_rows(&detector_rows, m_words);
    let observable_masks = pack_record_rows(&observable_rows, m_words);

    let deferred = lower_qec_program_to_deferred_circuit(program)?;
    let mut units: Vec<FaultUnit> = Vec::new();
    walk_qec_noise_sensitivity(&deferred, |event, x_packed, z_packed| {
        collect_fault_units(event, x_packed, z_packed, &mut units);
    })?;
    units.sort_by_key(|unit| unit.position);

    let mut index: HashMap<Symptom, usize> = HashMap::new();
    let mut mechanisms: Vec<ErrorMechanism> = Vec::new();
    for unit in units {
        for (symptom, probability) in unit_symptoms(&unit, &detector_masks, &observable_masks) {
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

fn collect_fault_units(
    event: &QecDeferredNoiseEvent,
    x_packed: &[Vec<u64>],
    z_packed: &[Vec<u64>],
    units: &mut Vec<FaultUnit>,
) {
    match event.channel {
        QecNoise::XError(p) => {
            for &target in &event.targets {
                units.push(FaultUnit {
                    position: event.position,
                    branches: vec![(p, z_packed[target].clone())],
                });
            }
        }
        QecNoise::ZError(p) => {
            for &target in &event.targets {
                units.push(FaultUnit {
                    position: event.position,
                    branches: vec![(p, x_packed[target].clone())],
                });
            }
        }
        QecNoise::Depolarize1(p) => {
            let branch_p = p / 3.0;
            for &target in &event.targets {
                let mut y_mask = x_packed[target].clone();
                xor_words(&mut y_mask, &z_packed[target]);
                units.push(FaultUnit {
                    position: event.position,
                    branches: vec![
                        (branch_p, z_packed[target].clone()),
                        (branch_p, y_mask),
                        (branch_p, x_packed[target].clone()),
                    ],
                });
            }
        }
        QecNoise::Depolarize2(p) => {
            let branch_p = p / 15.0;
            for pair in event.targets.chunks_exact(2) {
                let m_words = z_packed[pair[0]].len();
                let mut branches = Vec::with_capacity(15);
                for sample in 1..=15 {
                    let mut mask = vec![0u64; m_words];
                    append_qec_pauli_noise_effect(
                        &mut mask,
                        sample / 4,
                        &x_packed[pair[0]],
                        &z_packed[pair[0]],
                    );
                    append_qec_pauli_noise_effect(
                        &mut mask,
                        sample % 4,
                        &x_packed[pair[1]],
                        &z_packed[pair[1]],
                    );
                    branches.push((branch_p, mask));
                }
                units.push(FaultUnit {
                    position: event.position,
                    branches,
                });
            }
        }
    }
}

/// Project a unit's branches onto (detectors, observables) symptoms, summing
/// exclusive branches that share a symptom and dropping branches that flip
/// nothing.
fn unit_symptoms(
    unit: &FaultUnit,
    detector_masks: &[Vec<u64>],
    observable_masks: &[Vec<u64>],
) -> Vec<(Symptom, f64)> {
    let mut local: Vec<(Symptom, f64)> = Vec::new();
    for (probability, mask) in &unit.branches {
        let detectors = flipped_rows(mask, detector_masks);
        let observables = flipped_rows(mask, observable_masks);
        if detectors.is_empty() && observables.is_empty() {
            continue;
        }
        let symptom = (detectors, observables);
        match local.iter_mut().find(|(existing, _)| *existing == symptom) {
            Some((_, total)) => *total += probability,
            None => local.push((symptom, *probability)),
        }
    }
    local
}

fn flipped_rows(mask: &[u64], rows: &[Vec<u64>]) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter(|(_, row)| odd_overlap(mask, row))
        .map(|(row_index, _)| row_index)
        .collect()
}

fn odd_overlap(a: &[u64], b: &[u64]) -> bool {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x & y).count_ones())
        .sum::<u32>()
        % 2
        == 1
}

/// Pack record-index rows into bit masks. XOR rather than OR: a record listed
/// twice in a row cancels in the parity, and the mask must agree.
fn pack_record_rows(rows: &[Vec<usize>], m_words: usize) -> Vec<Vec<u64>> {
    rows.iter()
        .map(|row| {
            let mut mask = vec![0u64; m_words];
            for &record in row {
                mask[record / 64] ^= 1u64 << (record % 64);
            }
            mask
        })
        .collect()
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
