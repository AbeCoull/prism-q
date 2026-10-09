//! Memory-experiment generators: repetition, rotated surface, and triangular 6.6.6 color
//! code programs with detectors, coordinates, one logical observable, and circuit-level
//! noise.

use std::collections::HashMap;

use super::{QecBasis, QecNoise, QecOp, QecOptions, QecProgram, QecRecordRef};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

/// Circuit-level noise for the memory-experiment generators. A zero rate emits no
/// annotation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QecCircuitNoise {
    /// `DEPOLARIZE1` after every `H`, `DEPOLARIZE2` after every `CX`.
    pub after_clifford_depolarization: f64,
    /// Flip before every measurement, `X_ERROR` before `M` and `Z_ERROR` before `MX`.
    pub before_measure_flip_probability: f64,
    /// Flip after every reset, `X_ERROR` after `R` and `Z_ERROR` after `RX`.
    pub after_reset_flip_probability: f64,
    /// `DEPOLARIZE1` on every data qubit at the start of each round.
    pub before_round_data_depolarization: f64,
}

impl QecCircuitNoise {
    /// Every term at rate `p`.
    pub fn uniform(p: f64) -> Self {
        Self {
            after_clifford_depolarization: p,
            before_measure_flip_probability: p,
            after_reset_flip_probability: p,
            before_round_data_depolarization: p,
        }
    }

    fn validate(&self) -> Result<()> {
        for (name, p) in [
            (
                "after_clifford_depolarization",
                self.after_clifford_depolarization,
            ),
            (
                "before_measure_flip_probability",
                self.before_measure_flip_probability,
            ),
            (
                "after_reset_flip_probability",
                self.after_reset_flip_probability,
            ),
            (
                "before_round_data_depolarization",
                self.before_round_data_depolarization,
            ),
        ] {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(PrismError::InvalidParameter {
                    message: format!("{name} must be finite and in [0, 1], got {p}"),
                });
            }
        }
        Ok(())
    }
}

impl QecProgram {
    /// Repetition-code Z memory: `distance` data qubits on the even indices, `distance -
    /// 1` ZZ ancillas between them on the odd ones, `rounds` rounds of extraction, then
    /// a Z readout of the data. Protects against X errors only.
    ///
    /// Detectors carry `(x, t)` with `x` the ancilla's qubit index and `t` the round; the
    /// final detectors use `t = rounds`. Observable 0 is the readout of data qubit 0.
    ///
    /// # Errors
    ///
    /// `distance < 2`, `rounds == 0`, or a noise rate outside `[0, 1]`.
    pub fn repetition_memory(
        distance: usize,
        rounds: usize,
        noise: &QecCircuitNoise,
    ) -> Result<Self> {
        validate_memory(distance, rounds, 2, noise)?;
        let data: Vec<usize> = (0..distance).map(|i| 2 * i).collect();
        let ancillas: Vec<usize> = (0..distance - 1).map(|i| 2 * i + 1).collect();
        let first_layer: Vec<usize> = ancillas.iter().flat_map(|&a| [a - 1, a]).collect();
        let second_layer: Vec<usize> = ancillas.iter().flat_map(|&a| [a + 1, a]).collect();
        let all: Vec<usize> = (0..2 * distance - 1).collect();

        let mut b = MemoryBuilder::new(all.len(), *noise, rounds * (6 * distance + 8));
        b.reset(QecBasis::Z, &all);
        b.tick();
        let mut previous = None;
        for round in 0..rounds {
            b.data_noise(&data);
            b.cx(&first_layer);
            b.cx(&second_layer);
            let first = b.measure_reset(&ancillas);
            for (i, &ancilla) in ancillas.iter().enumerate() {
                b.compare_detector(
                    first + i,
                    previous.map(|p: usize| p + i),
                    &[ancilla as f64, round as f64],
                );
            }
            previous = Some(first);
        }
        let readout = b.measure(QecBasis::Z, &data);
        let last = previous.expect("rounds >= 1");
        for (i, &ancilla) in ancillas.iter().enumerate() {
            b.detector(
                &[readout + i, readout + i + 1, last + i],
                &[ancilla as f64, rounds as f64],
            );
        }
        b.observable(&[readout]);
        b.finish()
    }

    /// Rotated surface-code memory in the `basis` (X or Z) logical basis.
    ///
    /// Data qubit `r * distance + c` sits at `(2c + 1, 2r + 1)`; one ancilla per
    /// stabilizer follows the data, at the plaquette centre. Each round runs four `CX`
    /// layers, X stabilizers in the order NW, SW, NE, SE and Z stabilizers in NW, NE,
    /// SW, SE, so hook errors run perpendicular to the logical they could extend.
    /// Detectors carry `(x, y, t)`; the first round has detectors only on stabilizers of
    /// the memory basis, as does the final data readout at `t = rounds`. Observable 0 is
    /// the left data column for Z memory and the top row for X memory.
    ///
    /// # Errors
    ///
    /// `distance < 2`, `rounds == 0`, a `Y` basis, or a noise rate outside `[0, 1]`.
    pub fn surface_memory(
        distance: usize,
        rounds: usize,
        basis: QecBasis,
        noise: &QecCircuitNoise,
    ) -> Result<Self> {
        validate_memory(distance, rounds, 2, noise)?;
        validate_memory_basis(basis)?;
        let d = distance;
        let num_data = d * d;
        let data: Vec<usize> = (0..num_data).collect();
        let mut checks: Vec<Check> = Vec::new();
        for i in 0..=d {
            for j in 0..=d {
                let corner = |r: usize, c: usize| {
                    ((1..=d).contains(&r) && (1..=d).contains(&c)).then(|| (r - 1) * d + (c - 1))
                };
                let [nw, ne, sw, se] = [
                    corner(i, j),
                    corner(i, j + 1),
                    corner(i + 1, j),
                    corner(i + 1, j + 1),
                ];
                let weight = [nw, ne, sw, se].iter().flatten().count();
                let z_type = (i + j) % 2 == 1;
                let boundary_kept = if z_type {
                    i == 0 || i == d
                } else {
                    j == 0 || j == d
                };
                if weight == 4 || (weight == 2 && boundary_kept) {
                    let basis = if z_type { QecBasis::Z } else { QecBasis::X };
                    let order = if z_type {
                        vec![nw, ne, sw, se]
                    } else {
                        vec![nw, sw, ne, se]
                    };
                    checks.push(Check {
                        basis,
                        ancilla: num_data + checks.len(),
                        order,
                        coords: vec![2.0 * j as f64, 2.0 * i as f64],
                    });
                }
            }
        }
        let observable: Vec<usize> = match basis {
            QecBasis::Z => (0..d).map(|r| r * d).collect(),
            _ => (0..d).collect(),
        };
        let num_qubits = num_data + checks.len();
        let mut b = MemoryBuilder::new(num_qubits, *noise, rounds * 9 * num_qubits);
        b.reset(basis, &data);
        let ancillas: Vec<usize> = checks.iter().map(|check| check.ancilla).collect();
        b.reset(QecBasis::Z, &ancillas);
        b.tick();
        let mut previous = None;
        for round in 0..rounds {
            b.data_noise(&data);
            let first = b.extract(&checks, 4);
            b.round_detectors(&checks, first, previous, basis, round);
            previous = Some(first);
        }
        b.final_detectors(&checks, &data, previous, basis, rounds, &observable);
        b.finish()
    }

    /// Triangular 6.6.6 color-code memory in the `basis` (X or Z) logical basis, for odd
    /// `distance >= 3` on `(3 distance^2 + 1) / 4` data qubits.
    ///
    /// Data qubits are the honeycomb vertices inside a triangle with one boundary of
    /// each color; every face (hexagon, or a weight-4 half hexagon on the boundary) has
    /// one ancilla, after the data. Each round measures every X stabilizer and then
    /// every Z stabilizer on that ancilla, six `CX` layers apiece, visiting a face's
    /// vertices in angular order. No flag qubits are used, so an ancilla fault midway
    /// through a hexagon can spread to three data qubits.
    ///
    /// Detectors carry `(x, y, t, c)`: the face centre, with neighbouring centres at
    /// `(±2, 0)` and `(±1, ±1)`, the round, and the face color `0..3` for X stabilizers
    /// or `3..6` for Z stabilizers. Observable 0 is the readout of every data qubit.
    ///
    /// # Errors
    ///
    /// An even `distance` or one below 3, `rounds == 0`, a `Y` basis, or a noise rate
    /// outside `[0, 1]`.
    pub fn color_memory(
        distance: usize,
        rounds: usize,
        basis: QecBasis,
        noise: &QecCircuitNoise,
    ) -> Result<Self> {
        validate_memory(distance, rounds, 3, noise)?;
        validate_memory_basis(basis)?;
        if distance.is_multiple_of(2) {
            return Err(PrismError::InvalidParameter {
                message: format!("color code distance must be odd, got {distance}"),
            });
        }
        let (num_data, faces) = color_code_layout(distance);
        let data: Vec<usize> = (0..num_data).collect();
        let ancilla_of = |f: usize| num_data + f;
        let mut checks = Vec::with_capacity(2 * faces.len());
        for check_basis in [QecBasis::X, QecBasis::Z] {
            for (f, face) in faces.iter().enumerate() {
                let tag = face.color as f64 + if check_basis == QecBasis::Z { 3.0 } else { 0.0 };
                checks.push(Check {
                    basis: check_basis,
                    ancilla: ancilla_of(f),
                    order: face.vertices.to_vec(),
                    coords: vec![face.centre.0, face.centre.1, tag],
                });
            }
        }
        let (x_checks, z_checks) = checks.split_at(faces.len());
        let num_qubits = num_data + faces.len();
        let mut b = MemoryBuilder::new(num_qubits, *noise, rounds * 18 * num_qubits);
        b.reset(basis, &data);
        let ancillas: Vec<usize> = (0..faces.len()).map(ancilla_of).collect();
        b.reset(QecBasis::Z, &ancillas);
        b.tick();
        let mut previous: Option<(usize, usize)> = None;
        for round in 0..rounds {
            b.data_noise(&data);
            let first_x = b.extract(x_checks, 6);
            let first_z = b.extract(z_checks, 6);
            b.round_detectors(x_checks, first_x, previous.map(|p| p.0), basis, round);
            b.round_detectors(z_checks, first_z, previous.map(|p| p.1), basis, round);
            previous = Some((first_x, first_z));
        }
        let (memory_checks, last) = match (basis, previous.expect("rounds >= 1")) {
            (QecBasis::Z, (_, z)) => (z_checks, z),
            (_, (x, _)) => (x_checks, x),
        };
        b.final_detectors(memory_checks, &data, Some(last), basis, rounds, &data);
        b.finish()
    }
}

/// One stabilizer measurement: the ancilla, the data qubit it couples to at each `CX`
/// layer (`None` idles), and detector coordinates without the round.
struct Check {
    basis: QecBasis,
    ancilla: usize,
    order: Vec<Option<usize>>,
    coords: Vec<f64>,
}

struct ColorFace {
    color: u8,
    centre: (f64, f64),
    vertices: [Option<usize>; 6],
}

/// Data qubit count and faces of the distance-`d` triangular 6.6.6 code.
///
/// Face centres are triangular-lattice points `(a, b)`, colored `(a - b) mod 3`; a data
/// qubit is a lattice triangle, kept when three times its centroid `(sa, sb)` satisfies
/// `sa >= sb`, `sa + 2 sb >= 1`, and `2 sa + sb <= (9d + 1) / 2`. Up triangles sit at
/// steps 0, 2, 4 of their three faces and down triangles at 1, 3, 5, so no data qubit
/// meets two ancillas in one `CX` layer.
fn color_code_layout(d: usize) -> (usize, Vec<ColorFace>) {
    let bound = (9 * d as i64 + 1) / 2;
    let span = 3 * d as i64;
    let mut vertices = Vec::new();
    for a in -span..=span {
        for b in -span..=span {
            let up = (
                (3 * a + 1, 3 * b + 1),
                [((a, b), 0), ((a + 1, b), 2), ((a, b + 1), 4)],
            );
            let down = (
                (3 * a + 2, 3 * b + 2),
                [((a + 1, b), 1), ((a + 1, b + 1), 3), ((a, b + 1), 5)],
            );
            for (sum, corners) in [up, down] {
                let (sa, sb) = sum;
                if sa >= sb && sa + 2 * sb >= 1 && 2 * sa + sb <= bound {
                    vertices.push((sum, corners));
                }
            }
        }
    }
    vertices.sort_by_key(|&((sa, sb), _)| (sb, sa));

    let mut faces: HashMap<(i64, i64), [Option<usize>; 6]> = HashMap::new();
    for (qubit, (_, corners)) in vertices.iter().enumerate() {
        for &(centre, step) in corners {
            faces.entry(centre).or_insert([None; 6])[step] = Some(qubit);
        }
    }
    let mut faces: Vec<_> = faces
        .into_iter()
        .filter(|(_, steps)| steps.iter().flatten().count() >= 4)
        .collect();
    faces.sort_by_key(|&((a, b), _)| (b, a));
    let faces = faces
        .into_iter()
        .map(|((a, b), vertices)| ColorFace {
            color: (a - b).rem_euclid(3) as u8,
            centre: ((2 * a + b) as f64, b as f64),
            vertices,
        })
        .collect();
    (vertices.len(), faces)
}

fn validate_memory(
    distance: usize,
    rounds: usize,
    min_distance: usize,
    noise: &QecCircuitNoise,
) -> Result<()> {
    if distance < min_distance {
        return Err(PrismError::InvalidParameter {
            message: format!("memory distance must be at least {min_distance}, got {distance}"),
        });
    }
    if rounds == 0 {
        return Err(PrismError::InvalidParameter {
            message: "memory experiments need at least one round".to_string(),
        });
    }
    noise.validate()
}

fn validate_memory_basis(basis: QecBasis) -> Result<()> {
    if basis == QecBasis::Y {
        return Err(PrismError::InvalidParameter {
            message: "memory basis must be X or Z".to_string(),
        });
    }
    Ok(())
}

/// Op list under construction, with the record count and the noise to weave in. The
/// program is validated once, by [`QecProgram::from_ops`], when it is finished.
struct MemoryBuilder {
    num_qubits: usize,
    ops: Vec<QecOp>,
    records: usize,
    noise: QecCircuitNoise,
    pairs: Vec<usize>,
}

impl MemoryBuilder {
    fn new(num_qubits: usize, noise: QecCircuitNoise, capacity: usize) -> Self {
        Self {
            num_qubits,
            ops: Vec::with_capacity(capacity),
            records: 0,
            noise,
            pairs: Vec::new(),
        }
    }

    fn finish(self) -> Result<QecProgram> {
        QecProgram::from_ops(self.num_qubits, QecOptions::default(), self.ops)
    }

    fn tick(&mut self) {
        self.ops.push(QecOp::Tick);
    }

    fn noise(&mut self, channel: QecNoise, targets: &[usize]) {
        if channel.probability() > 0.0 && !targets.is_empty() {
            self.ops.push(QecOp::Noise {
                channel,
                targets: targets.to_vec(),
            });
        }
    }

    fn data_noise(&mut self, data: &[usize]) {
        let p = self.noise.before_round_data_depolarization;
        self.noise(QecNoise::Depolarize1(p), data);
    }

    fn reset(&mut self, basis: QecBasis, qubits: &[usize]) {
        self.ops
            .extend(qubits.iter().map(|&qubit| QecOp::Reset { basis, qubit }));
        let p = self.noise.after_reset_flip_probability;
        self.noise(basis_flip(basis, p), qubits);
    }

    /// Measure `qubits` in `basis` and return the first record.
    fn measure(&mut self, basis: QecBasis, qubits: &[usize]) -> usize {
        let p = self.noise.before_measure_flip_probability;
        self.noise(basis_flip(basis, p), qubits);
        self.ops
            .extend(qubits.iter().map(|&qubit| QecOp::Measure { basis, qubit }));
        let first = self.records;
        self.records += qubits.len();
        first
    }

    /// Z-measure and reset `qubits`, then close the layer; returns the first record.
    fn measure_reset(&mut self, qubits: &[usize]) -> usize {
        let first = self.measure(QecBasis::Z, qubits);
        self.reset(QecBasis::Z, qubits);
        self.tick();
        first
    }

    fn h(&mut self, qubits: &[usize]) {
        if qubits.is_empty() {
            return;
        }
        self.ops.extend(qubits.iter().map(|&q| QecOp::Gate {
            gate: Gate::H,
            targets: vec![q],
        }));
        let p = self.noise.after_clifford_depolarization;
        self.noise(QecNoise::Depolarize1(p), qubits);
        self.tick();
    }

    /// `CX` on each `(control, target)` pair of the flat list, then close the layer.
    fn cx(&mut self, pairs: &[usize]) {
        if pairs.is_empty() {
            return;
        }
        self.ops
            .extend(pairs.chunks_exact(2).map(|pair| QecOp::Gate {
                gate: Gate::Cx,
                targets: pair.to_vec(),
            }));
        let p = self.noise.after_clifford_depolarization;
        self.noise(QecNoise::Depolarize2(p), pairs);
        self.tick();
    }

    /// Measure every check through its ancilla over `layers` `CX` layers; returns the
    /// first record, check `k` landing on record `first + k`.
    fn extract(&mut self, checks: &[Check], layers: usize) -> usize {
        let x_ancillas: Vec<usize> = checks
            .iter()
            .filter(|check| check.basis == QecBasis::X)
            .map(|check| check.ancilla)
            .collect();
        self.h(&x_ancillas);
        for layer in 0..layers {
            let mut pairs = std::mem::take(&mut self.pairs);
            pairs.clear();
            for check in checks {
                if let Some(Some(qubit)) = check.order.get(layer) {
                    let pair = if check.basis == QecBasis::X {
                        [check.ancilla, *qubit]
                    } else {
                        [*qubit, check.ancilla]
                    };
                    pairs.extend_from_slice(&pair);
                }
            }
            self.cx(&pairs);
            self.pairs = pairs;
        }
        self.h(&x_ancillas);
        let ancillas: Vec<usize> = checks.iter().map(|check| check.ancilla).collect();
        self.measure_reset(&ancillas)
    }

    /// One detector per check: its record against the previous round's, or alone in
    /// the first round for checks of the memory basis.
    fn round_detectors(
        &mut self,
        checks: &[Check],
        first: usize,
        previous: Option<usize>,
        basis: QecBasis,
        round: usize,
    ) {
        for (k, check) in checks.iter().enumerate() {
            if previous.is_none() && check.basis != basis {
                continue;
            }
            let coords = with_round(&check.coords, round);
            self.compare_detector(first + k, previous.map(|p| p + k), &coords);
        }
    }

    /// Read out the data in `basis`, close each memory-basis check against its last
    /// record, and include `observable`'s readouts in observable 0.
    fn final_detectors(
        &mut self,
        checks: &[Check],
        data: &[usize],
        previous: Option<usize>,
        basis: QecBasis,
        rounds: usize,
        observable: &[usize],
    ) {
        let readout = self.measure(basis, data);
        let last = previous.expect("rounds >= 1");
        let mut records = Vec::new();
        for (k, check) in checks.iter().enumerate() {
            if check.basis != basis {
                continue;
            }
            records.clear();
            records.extend(check.order.iter().flatten().map(|&q| readout + q));
            records.push(last + k);
            let coords = with_round(&check.coords, rounds);
            self.detector(&records, &coords);
        }
        let observable: Vec<usize> = observable.iter().map(|&q| readout + q).collect();
        self.observable(&observable);
    }

    fn compare_detector(&mut self, record: usize, previous: Option<usize>, coords: &[f64]) {
        match previous {
            Some(previous) => self.detector(&[record, previous], coords),
            None => self.detector(&[record], coords),
        }
    }

    fn detector(&mut self, records: &[usize], coords: &[f64]) {
        self.ops.push(QecOp::Detector {
            records: records.iter().map(|&r| QecRecordRef::Absolute(r)).collect(),
            coords: coords.to_vec(),
        });
    }

    fn observable(&mut self, records: &[usize]) {
        self.ops.push(QecOp::ObservableInclude {
            observable: 0,
            records: records.iter().map(|&r| QecRecordRef::Absolute(r)).collect(),
        });
    }
}

/// `coords` with the round spliced in after the two spatial coordinates.
fn with_round(coords: &[f64], round: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(coords.len() + 1);
    out.extend_from_slice(&coords[..2]);
    out.push(round as f64);
    out.extend_from_slice(&coords[2..]);
    out
}

/// The flip that a basis reset or measurement is sensitive to.
fn basis_flip(basis: QecBasis, p: f64) -> QecNoise {
    match basis {
        QecBasis::Z => QecNoise::XError(p),
        _ => QecNoise::ZError(p),
    }
}
