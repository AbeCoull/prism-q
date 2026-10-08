//! Detector error model derivation and text export for native QEC programs.
//!
//! Expands each Pauli-noise annotation into its fault branches, propagates
//! every branch to the detectors and observables it flips, and merges branches
//! with identical symptoms into weighted error mechanisms.

use std::collections::HashMap;
use std::fmt::Write;

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
    components: Vec<Symptom>,
}

impl ErrorMechanism {
    pub(super) fn new(
        probability: f64,
        detectors: Vec<usize>,
        observables: Vec<usize>,
        components: Vec<Symptom>,
    ) -> Self {
        Self {
            probability,
            detectors,
            observables,
            components,
        }
    }

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

    /// Suggested split into components, each `(detectors, observables)` ascending, whose
    /// symmetric difference is this mechanism's symptom; empty when none is carried. Read
    /// from the `^` separators of imported text.
    pub fn suggested_decomposition(&self) -> &[(Vec<usize>, Vec<usize>)] {
        &self.components
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
    pub(super) fn from_parts(
        mechanisms: Vec<ErrorMechanism>,
        detector_coords: Vec<Vec<f64>>,
        num_detectors: usize,
        num_observables: usize,
    ) -> Self {
        Self {
            mechanisms,
            detector_coords,
            num_detectors,
            num_observables,
        }
    }

    /// Parse the common detector error model text format.
    ///
    /// Reads `error(p)` with `D` and `L` targets and `^` separators, `detector` with
    /// optional coordinates, `logical_observable`, `shift_detectors`, and `repeat N {
    /// ... }` blocks, which expand as they are read. Detector targets add the
    /// accumulated `shift_detectors` offset and coordinates its accumulated coordinate
    /// shift. Each `error` line becomes one mechanism, kept in order and not merged with
    /// others of the same symptom; `^` separators become its
    /// [`ErrorMechanism::suggested_decomposition`]. Instruction tags (`error[tag](p)`)
    /// are ignored and `#` starts a comment. The detector and observable counts are one
    /// past the highest index any instruction names.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_q::DetectorErrorModel;
    ///
    /// let model = DetectorErrorModel::from_text(
    ///     "error(0.1) D0 D1 ^ D2 L0
    ///      repeat 2 {
    ///          error(0.01) D0
    ///          shift_detectors 1
    ///      }
    ///      detector(1, 0) D0",
    /// )?;
    /// assert_eq!(model.num_mechanisms(), 3);
    /// assert_eq!(model.mechanisms()[0].detectors(), [0, 1, 2]);
    /// assert_eq!(model.mechanisms()[2].detectors(), [1]);
    /// assert_eq!(model.detector_coords()[2], [1.0, 0.0]);
    /// # Ok::<(), prism_q::PrismError>(())
    /// ```
    pub fn from_text(text: &str) -> Result<Self> {
        super::dem_text::parse_detector_error_model(text)
    }

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
    /// as `p = p1(1-p2) + p2(1-p1)`. A mechanism carrying a
    /// [`suggested_decomposition`](ErrorMechanism::suggested_decomposition) whose
    /// components each flip one or two detectors splits along it instead, a component
    /// with no matching mechanism being appended at the end. Cross-component
    /// correlations are lost; single-detector marginals are unchanged. Deterministic:
    /// retained mechanisms keep their order, the first cover in mechanism order wins.
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
        let index = CoverIndex::new(&graphlike, self.num_detectors);
        let mut by_symptom: Option<HashMap<Symptom, usize>> = None;

        for mechanism in &self.mechanisms {
            if mechanism.detectors.len() <= 2 {
                continue;
            }
            let p = mechanism.probability;
            if !mechanism.components.is_empty()
                && mechanism
                    .components
                    .iter()
                    .all(|(detectors, _)| matches!(detectors.len(), 1 | 2))
            {
                let by_symptom = by_symptom.get_or_insert_with(|| {
                    graphlike
                        .iter()
                        .enumerate()
                        .map(|(at, m)| ((m.detectors.clone(), m.observables.clone()), at))
                        .collect()
                });
                for component in &mechanism.components {
                    let at = *by_symptom.entry(component.clone()).or_insert_with(|| {
                        graphlike.push(ErrorMechanism::new(
                            0.0,
                            component.0.clone(),
                            component.1.clone(),
                            Vec::new(),
                        ));
                        graphlike.len() - 1
                    });
                    let prior = graphlike[at].probability;
                    graphlike[at].probability = prior * (1.0 - p) + p * (1.0 - prior);
                }
                continue;
            }
            let Some(components) = partition_cover(mechanism, &graphlike, &index) else {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "graphlike decomposition failed: mechanism `{}` has no \
                         cover by graphlike mechanisms",
                        symptom_label(mechanism)
                    ),
                });
            };
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
    /// one `logical_observable` line per observable slot. A mechanism on more than
    /// two detectors is written as `^`-separated components: its suggested
    /// decomposition when it carries one, else the cover
    /// [`decompose_graphlike`](Self::decompose_graphlike) would use, else flat. The
    /// output is flat otherwise: no `repeat` blocks and no `shift_detectors`. The
    /// grammar is documented in `docs/architecture/qec-programs.md`.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let mut cover: Option<(Vec<ErrorMechanism>, CoverIndex)> = None;
        for mechanism in &self.mechanisms {
            write!(out, "error({})", mechanism.probability).expect("String write");
            if !mechanism.components.is_empty() {
                for (k, (detectors, observables)) in mechanism.components.iter().enumerate() {
                    if k > 0 {
                        out.push_str(" ^");
                    }
                    write_symptom(&mut out, detectors, observables);
                }
            } else if mechanism.detectors.len() > 2 {
                let (graphlike, index) = cover.get_or_insert_with(|| {
                    let graphlike: Vec<ErrorMechanism> = self
                        .mechanisms
                        .iter()
                        .filter(|m| m.detectors.len() <= 2)
                        .cloned()
                        .collect();
                    let index = CoverIndex::new(&graphlike, self.num_detectors);
                    (graphlike, index)
                });
                match partition_cover(mechanism, graphlike, index) {
                    Some(components) => {
                        for (k, at) in components.into_iter().enumerate() {
                            if k > 0 {
                                out.push_str(" ^");
                            }
                            let component = &graphlike[at];
                            write_symptom(&mut out, &component.detectors, &component.observables);
                        }
                    }
                    None => write_symptom(&mut out, &mechanism.detectors, &mechanism.observables),
                }
            } else {
                write_symptom(&mut out, &mechanism.detectors, &mechanism.observables);
            }
            out.push('\n');
        }
        for (detector, coords) in self.detector_coords.iter().enumerate() {
            out.push_str("detector");
            if !coords.is_empty() {
                out.push('(');
                for (k, c) in coords.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    write!(out, "{c}").expect("String write");
                }
                out.push(')');
            }
            writeln!(out, " D{detector}").expect("String write");
        }
        for observable in 0..self.num_observables {
            writeln!(out, "logical_observable L{observable}").expect("String write");
        }
        out
    }
}

fn write_symptom(out: &mut String, detectors: &[usize], observables: &[usize]) {
    for detector in detectors {
        write!(out, " D{detector}").expect("String write");
    }
    for observable in observables {
        write!(out, " L{observable}").expect("String write");
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
pub(super) type Symptom = (Vec<usize>, Vec<usize>);

fn derive_detector_error_model(program: &QecProgram) -> Result<DetectorErrorModel> {
    if program.has_leakage() {
        return Err(super::leakage::leakage_rejection("detector error model"));
    }
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
                    mechanisms.push(ErrorMechanism::new(
                        probability,
                        detectors,
                        observables,
                        Vec::new(),
                    ));
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

/// Graphlike mechanisms by the detectors they flip, for [`partition_cover`].
struct CoverIndex {
    by_detector: Vec<Vec<u32>>,
}

impl CoverIndex {
    fn new(graphlike: &[ErrorMechanism], num_detectors: usize) -> Self {
        let mut by_detector = vec![Vec::new(); num_detectors];
        for (at, mechanism) in graphlike.iter().enumerate() {
            for &detector in &mechanism.detectors {
                by_detector[detector].push(at as u32);
            }
        }
        Self { by_detector }
    }
}

/// Depth-first over candidates in mechanism order; the first cover found is
/// the result. Only graphlike mechanisms whose detectors all belong to `mechanism` can
/// sit in a cover, so the search walks those alone, which finds the same first cover.
fn partition_cover(
    mechanism: &ErrorMechanism,
    graphlike: &[ErrorMechanism],
    index: &CoverIndex,
) -> Option<Vec<usize>> {
    fn search(
        remaining: &[usize],
        observables: &[usize],
        start: usize,
        candidates: &[usize],
        graphlike: &[ErrorMechanism],
        chosen: &mut Vec<usize>,
    ) -> bool {
        if remaining.is_empty() {
            return observables.is_empty();
        }
        for (next, &at) in candidates.iter().enumerate().skip(start) {
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
                next + 1,
                candidates,
                graphlike,
                chosen,
            ) {
                return true;
            }
            chosen.pop();
        }
        false
    }

    let mut candidates: Vec<usize> = mechanism
        .detectors
        .iter()
        .filter_map(|&detector| index.by_detector.get(detector))
        .flatten()
        .map(|&at| at as usize)
        .filter(|&at| is_subset(&graphlike[at].detectors, &mechanism.detectors))
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    let mut chosen = Vec::new();
    search(
        &mechanism.detectors,
        &mechanism.observables,
        0,
        &candidates,
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
