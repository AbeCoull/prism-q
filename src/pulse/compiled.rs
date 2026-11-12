//! A compiled propagator, its computational block and what the gate level takes from it.

use num_complex::Complex64;

use super::integrate::multiply;
use super::{Device, PulseError};
use crate::gates::spectral::hermitian_eigen;
use crate::{Gate, NoiseChannel};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Smallest singular value of the computational block that still has a nearest unitary
/// worth reporting.
const MIN_SINGULAR_VALUE: f64 = 1e-6;

/// The result of [`compile`](super::compile).
///
/// Every matrix is row major. [`CompiledGate::propagator`] is in the device basis of
/// [`Device`]; [`CompiledGate::subspace`] and [`CompiledGate::unitary`] use the
/// [`Gate::unitary`] packing, site 0 the most significant bit, so they apply with targets
/// `[q0, q1, ...]` naming the qubits that stand for sites 0, 1 and so on.
///
/// # Panics
///
/// [`CompiledGate::average_fidelity`] and [`CompiledGate::error_after`] panic when the
/// ideal gate's width differs from the number of sites, or on a three-qubit gate other
/// than [`Gate::Unitary`] and [`Gate::Mcu`]; [`CompiledGate::error_after`] also panics on
/// a three-site device, which no [`NoiseChannel`] variant can carry.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledGate {
    sites: usize,
    propagator: Vec<Complex64>,
    subspace: Vec<Complex64>,
    unitary: Vec<Complex64>,
    leakage: Vec<[f64; 2]>,
}

impl CompiledGate {
    pub(super) fn new(device: &Device, propagator: Vec<Complex64>) -> Result<Self, PulseError> {
        let levels: Vec<usize> = device.sites().iter().map(|site| site.levels).collect();
        let sites = levels.len();
        let dim = propagator.len().isqrt();
        let strides = device.strides();
        let width = 1usize << sites;
        let device_index = |packed: usize| -> usize {
            (0..sites)
                .filter(|&site| (packed >> (sites - 1 - site)) & 1 == 1)
                .map(|site| strides[site])
                .sum()
        };

        let mut block = vec![ZERO; width * width];
        for column in 0..width {
            for row in 0..width {
                block[column * width + row] =
                    propagator[device_index(column) * dim + device_index(row)];
            }
        }
        let unitary = nearest_unitary(&block, width)?;

        let leakage = (0..sites)
            .map(|site| {
                [0, 1].map(|level| {
                    let column = level * strides[site];
                    (0..dim)
                        .filter(|&row| (row / strides[site]) % levels[site] >= 2)
                        .map(|row| propagator[column * dim + row].norm_sqr())
                        .sum()
                })
            })
            .collect();

        Ok(CompiledGate {
            sites,
            propagator: to_row_major(&propagator, dim),
            subspace: to_row_major(&block, width),
            unitary: to_row_major(&unitary, width),
            leakage,
        })
    }

    /// The full propagator over the device basis.
    pub fn propagator(&self) -> &[Complex64] {
        &self.propagator
    }

    /// The propagator restricted to states with every site in `|0>` or `|1>`; not unitary
    /// when anything leaks.
    pub fn subspace(&self) -> &[Complex64] {
        &self.subspace
    }

    /// The unitary nearest [`CompiledGate::subspace`] by polar decomposition, ready for
    /// [`Gate::unitary`] or [`Circuit::add_unitary`](crate::Circuit::add_unitary).
    pub fn unitary(&self) -> Vec<Complex64> {
        self.unitary.clone()
    }

    /// Per site, the probability of leaving `{|0>, |1>}` from `|0>` and from `|1>`, with
    /// every other site starting in `|0>`.
    pub fn leakage(&self) -> &[[f64; 2]] {
        &self.leakage
    }

    /// Average gate fidelity of [`CompiledGate::subspace`] against `ideal`, counting
    /// leakage out of the subspace as error.
    pub fn average_fidelity(&self, ideal: &Gate) -> f64 {
        let target = ideal_matrix(ideal, self.sites);
        let width = 1usize << self.sites;
        let overlap: Complex64 = (0..width * width)
            .map(|index| target[index].conj() * self.subspace[index])
            .sum();
        let retained: f64 = self.subspace.iter().map(Complex64::norm_sqr).sum();
        let d = width as f64;
        (retained + overlap.norm_sqr()) / (d * (d + 1.0))
    }

    /// The coherent residual `E` with `E * ideal` equal to [`CompiledGate::unitary`], as a
    /// one-operator Kraus set to attach after `ideal`.
    pub fn error_after(&self, ideal: &Gate) -> NoiseChannel {
        assert!(
            self.sites <= 2,
            "error_after needs one or two sites; no channel carries a three-qubit Kraus set"
        );
        let target = ideal_matrix(ideal, self.sites);
        let width = 1usize << self.sites;
        let residual = |row: usize, column: usize| -> Complex64 {
            (0..width)
                .map(|k| self.unitary[row * width + k] * target[column * width + k].conj())
                .sum()
        };
        if self.sites == 1 {
            NoiseChannel::Custom {
                kraus: vec![[
                    [residual(0, 0), residual(0, 1)],
                    [residual(1, 0), residual(1, 1)],
                ]],
            }
        } else {
            NoiseChannel::Kraus2q {
                kraus: vec![std::array::from_fn(|row| {
                    std::array::from_fn(|column| residual(row, column))
                })],
            }
        }
    }
}

/// `ideal` as a row-major matrix in the [`Gate::unitary`] packing.
fn ideal_matrix(ideal: &Gate, sites: usize) -> Vec<Complex64> {
    assert_eq!(
        ideal.num_qubits(),
        sites,
        "the ideal gate acts on {} qubits, the device has {sites} sites",
        ideal.num_qubits()
    );
    match sites {
        1 => ideal.matrix_2x2().concat(),
        2 => ideal.matrix_4x4().concat(),
        _ => match ideal {
            Gate::Unitary(data) => data.matrix().to_vec(),
            Gate::Mcu(data) => {
                let width = 1usize << sites;
                let mut out = vec![ZERO; width * width];
                for index in 0..width - 2 {
                    out[index * width + index] = Complex64::new(1.0, 0.0);
                }
                for (row, entries) in data.mat().iter().enumerate() {
                    for (column, &entry) in entries.iter().enumerate() {
                        out[(width - 2 + row) * width + width - 2 + column] = entry;
                    }
                }
                out
            }
            other => panic!(
                "a three-site ideal gate must be Unitary or Mcu, got `{}`",
                other.name()
            ),
        },
    }
}

/// `block (block^dagger block)^(-1/2)`, column major.
fn nearest_unitary(block: &[Complex64], width: usize) -> Result<Vec<Complex64>, PulseError> {
    let gram = multiply(&adjoint(block, width), block, width);
    let (values, vectors) = hermitian_eigen(&gram, width);
    let smallest = values[0].max(0.0).sqrt();
    if smallest < MIN_SINGULAR_VALUE {
        return Err(PulseError::Leaked(smallest));
    }
    let mut inverse_root = vec![ZERO; width * width];
    for column in 0..width {
        for row in 0..width {
            inverse_root[column * width + row] = (0..width)
                .map(|k| {
                    vectors[k * width + row] * vectors[k * width + column].conj() / values[k].sqrt()
                })
                .sum();
        }
    }
    Ok(multiply(block, &inverse_root, width))
}

fn to_row_major(matrix: &[Complex64], dim: usize) -> Vec<Complex64> {
    (0..dim * dim)
        .map(|index| matrix[(index % dim) * dim + index / dim])
        .collect()
}

fn adjoint(matrix: &[Complex64], dim: usize) -> Vec<Complex64> {
    (0..dim * dim)
        .map(|index| matrix[(index % dim) * dim + index / dim].conj())
        .collect()
}
