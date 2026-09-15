//! Amazon Braket's `#pragma braket` extensions: result requests, noise
//! channels, and inline unitaries.
//!
//! A result request reaches the caller beside the circuit rather than inside
//! it, [`Instruction`] describing what runs rather than
//! what to report.

use num_complex::Complex64;

use crate::PauliObservable;
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::noise::NoiseChannel;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Qubits a pragma names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Targets {
    /// Every declared qubit, which `all` and an omitted target list both mean.
    All,
    These(Vec<usize>),
}

/// One tensor factor of a Braket observable.
///
/// Braket admits `x`, `y`, `z`, `h` and `i` alongside an explicit Hermitian
/// matrix; only the first three are Pauli axes, so the others are their own
/// variants rather than a widened [`PauliAxis`].
#[derive(Debug, Clone, PartialEq)]
pub enum ObservableFactor {
    Pauli {
        axis: PauliAxis,
        targets: Targets,
    },
    Hadamard {
        targets: Targets,
    },
    Identity {
        targets: Targets,
    },
    Hermitian {
        matrix: Vec<Vec<Complex64>>,
        targets: Targets,
    },
}

/// A Braket observable: the tensor product its `@`-separated factors spell.
#[derive(Debug, Clone, PartialEq)]
pub struct Observable {
    pub factors: Vec<ObservableFactor>,
}

/// One observable factor reduced to a computational-basis measurement.
#[derive(Debug, Clone)]
pub struct MeasuredFactor {
    pub targets: Vec<usize>,
    /// The factor with its target list dropped, which is what decides whether
    /// two requests can share one measurement: the same basis on the same
    /// qubits may, a different one may not.
    pub basis: ObservableFactor,
    /// Instructions carrying the factor's eigenbasis onto the computational
    /// basis, run after the circuit. `None` is the identity, which reads no
    /// basis and so shares its qubits with any other observable; an empty list
    /// is the computational basis itself.
    pub rotation: Option<Vec<Instruction>>,
    /// Eigenvalue per outcome of `targets`, `targets[0]` the high bit.
    pub eigenvalues: Vec<f64>,
}

impl ObservableFactor {
    pub fn targets(&self) -> &Targets {
        match self {
            ObservableFactor::Pauli { targets, .. }
            | ObservableFactor::Hadamard { targets }
            | ObservableFactor::Identity { targets }
            | ObservableFactor::Hermitian { targets, .. } => targets,
        }
    }

    /// Qubits the factor acts on, which a Hermitian matrix fixes by its side.
    pub fn width(&self) -> usize {
        match self {
            ObservableFactor::Hermitian { matrix, .. } => matrix.len().trailing_zeros() as usize,
            _ => 1,
        }
    }

    fn check_width(&self, targets: &[usize]) -> Result<()> {
        if targets.len() != self.width() {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "an observable on {} qubit(s) was given {} target(s)",
                    self.width(),
                    targets.len()
                ),
            });
        }
        Ok(())
    }

    /// Rotation onto the computational basis and the eigenvalue each outcome
    /// then carries.
    ///
    /// The Pauli and Hadamard cases are the named single-qubit rotations; an
    /// explicit matrix is diagonalized numerically, which needs a gate wide
    /// enough to carry the result and so stops at two qubits.
    fn diagonalize(&self, targets: &[usize]) -> Result<MeasuredFactor> {
        self.check_width(targets)?;
        let single = |gate: Gate| {
            vec![Instruction::Gate {
                gate,
                targets: crate::circuit::SmallVec::from_slice(&targets[..1]),
            }]
        };
        let (rotation, eigenvalues) = match self {
            ObservableFactor::Pauli { axis, .. } => {
                let rotation = match axis {
                    PauliAxis::X => single(Gate::H),
                    PauliAxis::Y => {
                        let mut both = single(Gate::Sdg);
                        both.extend(single(Gate::H));
                        both
                    }
                    PauliAxis::Z => Vec::new(),
                };
                (Some(rotation), vec![1.0, -1.0])
            }
            ObservableFactor::Hadamard { .. } => (
                Some(single(Gate::Ry(-std::f64::consts::FRAC_PI_4))),
                vec![1.0, -1.0],
            ),
            ObservableFactor::Identity { .. } => (None, vec![1.0, 1.0]),
            ObservableFactor::Hermitian { matrix, .. } => {
                let (values, rotation) = hermitian_measurement(matrix, targets)?;
                (Some(rotation), values)
            }
        };
        Ok(MeasuredFactor {
            targets: targets.to_vec(),
            basis: self.with_targets(Targets::All),
            rotation,
            eigenvalues,
        })
    }

    /// A copy reading the same thing on `targets`, which normalizes the target
    /// list out of an equality comparison.
    fn with_targets(&self, targets: Targets) -> ObservableFactor {
        match self {
            ObservableFactor::Pauli { axis, .. } => ObservableFactor::Pauli {
                axis: *axis,
                targets,
            },
            ObservableFactor::Hadamard { .. } => ObservableFactor::Hadamard { targets },
            ObservableFactor::Identity { .. } => ObservableFactor::Identity { targets },
            ObservableFactor::Hermitian { matrix, .. } => ObservableFactor::Hermitian {
                matrix: matrix.clone(),
                targets,
            },
        }
    }

    fn lower(&self, targets: &[usize]) -> Result<PauliObservable> {
        self.check_width(targets)?;
        match self {
            ObservableFactor::Pauli { axis, .. } => {
                PauliObservable::from_terms([(1.0, vec![PauliTerm::new(targets[0], *axis)])])
            }
            ObservableFactor::Hadamard { .. } => PauliObservable::from_terms([
                (
                    std::f64::consts::FRAC_1_SQRT_2,
                    vec![PauliTerm::x(targets[0])],
                ),
                (
                    std::f64::consts::FRAC_1_SQRT_2,
                    vec![PauliTerm::z(targets[0])],
                ),
            ]),
            ObservableFactor::Identity { .. } => PauliObservable::from_terms([(1.0, Vec::new())]),
            ObservableFactor::Hermitian { matrix, .. } => {
                PauliObservable::from_terms(pauli_decomposition(matrix, targets))
            }
        }
    }
}

impl Observable {
    /// Lower to one Pauli sum per reported value, each beside the qubits it
    /// reads. A single-qubit observable with no target list reports one value
    /// per qubit; every other form reports one.
    ///
    /// `h` expands as `(X + Z)/sqrt(2)` and an explicit Hermitian matrix by its
    /// Pauli decomposition, so every observable reaches the same weighted-sum
    /// evaluation the native terminals take.
    pub fn lower(&self, num_qubits: usize) -> Result<Vec<(Vec<usize>, PauliObservable)>> {
        self.grouped(num_qubits)?
            .into_iter()
            .map(|group| {
                let mut targets = Vec::new();
                let mut product = PauliObservable::from_terms([(1.0, Vec::new())])?;
                for (factor, these) in group {
                    product = tensor(&product, &factor.lower(&these)?)?;
                    targets.extend_from_slice(&these);
                }
                Ok((targets, product))
            })
            .collect()
    }

    /// Reduce to measurements, one group of factors per reported value, shaped
    /// like [`Observable::lower`].
    pub fn diagonalize(&self, num_qubits: usize) -> Result<Vec<Vec<MeasuredFactor>>> {
        self.grouped(num_qubits)?
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|(factor, these)| factor.diagonalize(&these))
                    .collect()
            })
            .collect()
    }

    /// Factors beside the qubits each reads, one group per reported value.
    ///
    /// Braket applies a single-qubit observable with no target list to every
    /// qubit in parallel and reports one value for each, so `expectation z all`
    /// on a three-qubit register is three numbers and not one. Every other form
    /// is a single tensor product and a single value.
    #[allow(clippy::type_complexity)]
    fn grouped(&self, num_qubits: usize) -> Result<Vec<Vec<(&ObservableFactor, Vec<usize>)>>> {
        if self
            .factors
            .iter()
            .any(|factor| matches!(factor.targets(), Targets::All))
        {
            let [factor] = self.factors.as_slice() else {
                return Err(PrismError::InvalidParameter {
                    message: "a tensor-product observable names its targets on every factor".into(),
                });
            };
            if factor.width() != 1 {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "an observable applied to every qubit acts on one qubit, this one acts on {}",
                        factor.width()
                    ),
                });
            }
            return Ok((0..num_qubits)
                .map(|qubit| vec![(factor, vec![qubit])])
                .collect());
        }

        let mut group = Vec::with_capacity(self.factors.len());
        let mut named = Vec::new();
        for factor in &self.factors {
            let Targets::These(these) = factor.targets() else {
                unreachable!("the `all` case returned above");
            };
            for &qubit in these {
                if qubit >= num_qubits {
                    return Err(PrismError::InvalidQubit {
                        index: qubit,
                        register_size: num_qubits,
                    });
                }
                if named.contains(&qubit) {
                    return Err(PrismError::InvalidParameter {
                        message: format!("a tensor-product observable names qubit {qubit} twice"),
                    });
                }
                named.push(qubit);
            }
            group.push((factor, these.clone()));
        }
        Ok(vec![group])
    }
}

/// Product of two Pauli sums over disjoint qubits, which the caller has
/// checked.
fn tensor(left: &PauliObservable, right: &PauliObservable) -> Result<PauliObservable> {
    let mut product = PauliObservable::new();
    for (left_coefficient, left_string) in left.terms() {
        for (right_coefficient, right_string) in right.terms() {
            let mut string = left_string.clone();
            string.extend_from_slice(right_string);
            product
                .add_term(left_coefficient * right_coefficient, string)
                .map_err(|_| PrismError::InvalidParameter {
                    message: "a tensor-product observable names one qubit twice".into(),
                })?;
        }
    }
    Ok(product)
}

/// Decompose a Hermitian matrix over `targets` into `c_P = Tr(P M) / 2^k`.
///
/// `targets[0]` is the high bit of the matrix index, the packing
/// [`Gate::matrix_4x4`](crate::gates::Gate::matrix_4x4) uses. Coefficients are
/// real because the matrix is Hermitian, which the pragma checks when it reads
/// the literal.
///
/// A Pauli string holds one non-zero per column, so the trace walks the matrix
/// by rows and picks the single column that string pairs each with.
fn pauli_decomposition(matrix: &[Vec<Complex64>], targets: &[usize]) -> Vec<(f64, Vec<PauliTerm>)> {
    let width = targets.len();
    let dim = matrix.len();
    let scale = matrix
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f64, |peak, entry| peak.max(entry.norm()));
    let tolerance = f64::EPSILON * scale * dim as f64;
    let mut terms = Vec::new();
    for assignment in 0..4usize.pow(width as u32) {
        let axes: Vec<usize> = (0..width).map(|l| assignment >> (2 * l) & 3).collect();
        let mut trace = Complex64::new(0.0, 0.0);
        for (row, entries) in matrix.iter().enumerate() {
            let mut column = row;
            let mut value = Complex64::new(1.0, 0.0);
            for (l, &axis) in axes.iter().enumerate() {
                let position = width - 1 - l;
                let bit = row >> position & 1;
                match axis {
                    1 => column ^= 1 << position,
                    2 => {
                        column ^= 1 << position;
                        value *= Complex64::new(0.0, if bit == 0 { 1.0 } else { -1.0 });
                    }
                    3 if bit == 1 => value = -value,
                    _ => {}
                }
            }
            trace += value * entries[column];
        }
        let coefficient = trace.re / dim as f64;
        if coefficient.abs() <= tolerance {
            continue;
        }
        let string = axes
            .iter()
            .enumerate()
            .filter_map(|(l, &axis)| {
                let axis = match axis {
                    1 => PauliAxis::X,
                    2 => PauliAxis::Y,
                    3 => PauliAxis::Z,
                    _ => return None,
                };
                Some(PauliTerm::new(targets[l], axis))
            })
            .collect();
        terms.push((coefficient, string));
    }
    terms
}

/// Eigenvalues of a Hermitian matrix and the instructions carrying its
/// eigenbasis onto the computational basis.
///
/// Eigenvalue `j` is the one an outcome of `j` then reads, so the two halves
/// are paired by column and must not be sorted apart.
fn hermitian_measurement(
    matrix: &[Vec<Complex64>],
    targets: &[usize],
) -> Result<(Vec<f64>, Vec<Instruction>)> {
    let dim = matrix.len();
    let width = targets.len();
    if width > crate::circuit::synthesis::MAX_COMPOSED_QUBITS {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "sampling a {width}-qubit hermitian observable needs a rotation past the {} \
                 qubits the reduction covers",
                crate::circuit::synthesis::MAX_COMPOSED_QUBITS
            ),
        });
    }
    let mut columns = vec![Complex64::new(0.0, 0.0); dim * dim];
    for (row, entries) in matrix.iter().enumerate() {
        for (column, entry) in entries.iter().enumerate() {
            columns[column * dim + row] = *entry;
        }
    }
    let (values, vectors) = crate::gates::spectral::hermitian_eigen(&columns, dim);
    // The rotation is `U*`, whose row `a` is the conjugate of eigenvector `a`.
    let mut dagger = vec![Complex64::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        for column in 0..dim {
            dagger[row * dim + column] = vectors[row * dim + column].conj();
        }
    }
    Ok((
        values,
        crate::circuit::synthesis::dense_unitary(&dagger, targets),
    ))
}

/// Largest entrywise deviation from `M = M-dagger` a matrix literal may carry.
const HERMITIAN_TOLERANCE: f64 = 1e-9;

fn assert_hermitian(matrix: &[Vec<Complex64>], line: usize) -> Result<()> {
    for (row, entries) in matrix.iter().enumerate() {
        for (column, entry) in entries.iter().enumerate() {
            if (entry - matrix[column][row].conj()).norm() > HERMITIAN_TOLERANCE {
                return Err(PrismError::Parse {
                    line,
                    message: format!(
                        "observable matrix is not hermitian: entry ({row}, {column}) is {entry} \
                         against {} at ({column}, {row})",
                        matrix[column][row]
                    ),
                });
            }
        }
    }
    Ok(())
}

/// One `#pragma braket result` request, in declaration order.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultSpec {
    StateVector,
    DensityMatrix(Targets),
    /// Basis states named as big-endian bitstrings, the order Braket writes
    /// them: the leftmost character is qubit 0.
    Amplitude(Vec<String>),
    Probability(Targets),
    Expectation(Observable),
    Variance(Observable),
    Sample(Observable),
}

impl ResultSpec {
    /// The name Braket spells this request with.
    pub fn name(&self) -> &'static str {
        match self {
            ResultSpec::StateVector => "state_vector",
            ResultSpec::DensityMatrix(_) => "density_matrix",
            ResultSpec::Amplitude(_) => "amplitude",
            ResultSpec::Probability(_) => "probability",
            ResultSpec::Expectation(_) => "expectation",
            ResultSpec::Variance(_) => "variance",
            ResultSpec::Sample(_) => "sample",
        }
    }

    /// Whether the request is only meaningful at `shots = 0`.
    ///
    /// Braket rejects `state_vector`, `density_matrix` and `amplitude` above
    /// zero shots, and rejects `sample` at zero.
    pub fn requires_exact(&self) -> bool {
        matches!(
            self,
            ResultSpec::StateVector | ResultSpec::DensityMatrix(_) | ResultSpec::Amplitude(_)
        )
    }
}

/// Split on `delimiter` at bracket depth zero, so a matrix literal or a call
/// argument list keeps its own separators.
fn split_top_level(text: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, ch) in text.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = depth.saturating_sub(1),
            _ if ch == delimiter && depth == 0 => {
                parts.push(&text[start..i]);
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

/// Body of the outermost bracket pair, and whatever trails it.
fn split_bracketed(text: &str, open: char, close: char, line: usize) -> Result<(&str, &str)> {
    let text = text.trim();
    let start = text.find(open).ok_or_else(|| PrismError::Parse {
        line,
        message: format!("expected `{open}` in `{text}`"),
    })?;
    let mut depth = 0usize;
    for (i, ch) in text[start..].char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                let end = start + i;
                return Ok((
                    &text[start + open.len_utf8()..end],
                    &text[end + close.len_utf8()..],
                ));
            }
        }
    }
    Err(PrismError::Parse {
        line,
        message: format!("unclosed `{open}` in `{text}`"),
    })
}

/// Parse one complex literal in Braket's matrix notation.
///
/// Accepts a real (`0`, `-1.5`), an imaginary (`1im`, `-1im`), or a sum of the
/// two (`0.7 + 0.7im`). This is the whole grammar Braket admits inside a
/// matrix; a general expression is not one of the forms.
pub(crate) fn parse_complex(text: &str, line: usize) -> Result<Complex64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(PrismError::Parse {
            line,
            message: "empty matrix entry".to_string(),
        });
    }
    // A sign that opens the literal belongs to the first term, and one that
    // follows an exponent marker belongs to the exponent.
    let bytes = trimmed.as_bytes();
    let mut split = None;
    for (i, ch) in trimmed.char_indices().skip(1) {
        if (ch == '+' || ch == '-') && !matches!(bytes[i - 1], b'e' | b'E') {
            split = Some(i);
            break;
        }
    }
    match split {
        Some(i) => {
            let real = parse_real(&trimmed[..i], line)?;
            let imag = parse_imaginary(&trimmed[i..], line)?;
            Ok(Complex64::new(real, imag))
        }
        None if trimmed.ends_with("im") => Ok(Complex64::new(0.0, parse_imaginary(trimmed, line)?)),
        None => Ok(Complex64::new(parse_real(trimmed, line)?, 0.0)),
    }
}

fn parse_real(text: &str, line: usize) -> Result<f64> {
    text.trim().parse::<f64>().map_err(|_| PrismError::Parse {
        line,
        message: format!("`{}` is not a real literal", text.trim()),
    })
}

fn parse_imaginary(text: &str, line: usize) -> Result<f64> {
    let trimmed = text.trim();
    let digits = trimmed
        .strip_suffix("im")
        .ok_or_else(|| PrismError::Parse {
            line,
            message: format!("`{trimmed}` is not an imaginary literal"),
        })?
        .trim();
    // `f64` parsing rejects a space between the sign and the digits, which
    // `0.7 + 0.7im` puts there, so the sign is taken off first.
    let (sign, rest) = match digits.strip_prefix(['+', '-']) {
        Some(rest) => (
            if digits.starts_with('-') { -1.0 } else { 1.0 },
            rest.trim(),
        ),
        None => (1.0, digits),
    };
    match rest {
        "" => Ok(sign),
        other => Ok(sign * parse_real(other, line)?),
    }
}

/// Parse a `[[a, b], [c, d]]` matrix literal.
pub(crate) fn parse_matrix(text: &str, line: usize) -> Result<Vec<Vec<Complex64>>> {
    let (body, trailing) = split_bracketed(text, '[', ']', line)?;
    if !trailing.trim().is_empty() {
        return Err(PrismError::Parse {
            line,
            message: format!("trailing `{}` after a matrix literal", trailing.trim()),
        });
    }
    let mut rows = Vec::new();
    for row in split_top_level(body, ',') {
        let row = row.trim();
        if row.is_empty() {
            continue;
        }
        let (entries, trailing) = split_bracketed(row, '[', ']', line)?;
        if !trailing.trim().is_empty() {
            return Err(PrismError::Parse {
                line,
                message: format!("trailing `{}` after a matrix row", trailing.trim()),
            });
        }
        rows.push(
            split_top_level(entries, ',')
                .into_iter()
                .map(|entry| parse_complex(entry, line))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    let side = rows.len();
    if side == 0 || !side.is_power_of_two() || rows.iter().any(|row| row.len() != side) {
        return Err(PrismError::Parse {
            line,
            message: format!("expected a square matrix with a power-of-two side, got {side} rows"),
        });
    }
    Ok(rows)
}

/// Body of a leading parenthesised group, and whatever trails it.
pub(crate) fn split_paren_body(text: &str, line: usize) -> Result<(&str, &str)> {
    split_bracketed(text, '(', ')', line)
}

/// The instructions an inline `unitary` pragma names.
///
/// One and two qubit matrices keep their own gate variants; a wider one is
/// reduced to multi-controlled gates by
/// [`synthesis::dense_unitary`](crate::circuit::synthesis::dense_unitary).
pub(crate) fn unitary_instructions(
    matrix: &[Vec<Complex64>],
    targets: &[usize],
    line: usize,
) -> Result<Vec<crate::circuit::Instruction>> {
    let side = matrix.len();
    let num_targets = targets.len();
    if num_targets == 0 || side != 1usize << num_targets {
        return Err(PrismError::Parse {
            line,
            message: format!(
                "a unitary on {num_targets} qubit(s) needs a {}x{} matrix, got {side}x{side}",
                1usize << num_targets,
                1usize << num_targets
            ),
        });
    }
    if num_targets > crate::circuit::synthesis::MAX_COMPOSED_QUBITS {
        return Err(PrismError::UnsupportedConstruct {
            construct: format!(
                "`unitary` on {num_targets} qubits, past the {} the reduction covers",
                crate::circuit::synthesis::MAX_COMPOSED_QUBITS
            ),
            line,
        });
    }
    check_unitary(matrix, line)?;
    let flat: Vec<Complex64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
    Ok(crate::circuit::synthesis::dense_unitary(&flat, targets))
}

fn check_unitary(matrix: &[Vec<Complex64>], line: usize) -> Result<()> {
    let side = matrix.len();
    for row in 0..side {
        for col in 0..side {
            let entry: Complex64 = (0..side)
                .map(|k| matrix[k][row].conj() * matrix[k][col])
                .sum();
            let want = if row == col { 1.0 } else { 0.0 };
            if (entry - Complex64::new(want, 0.0)).norm() > 1e-9 {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "matrix at line {line} is not unitary: column dot product at ({row}, {col}) is {entry}"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Resolve a qubit token (`q[0]`, a whole register, or `$0`) to global indices.
pub(crate) type QubitResolver<'r> = dyn Fn(&str) -> Result<Vec<usize>> + 'r;

fn parse_targets(text: &str, line: usize, resolve: &QubitResolver<'_>) -> Result<Targets> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "all" {
        return Ok(Targets::All);
    }
    let mut indices = Vec::new();
    for token in split_top_level(trimmed, ',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        indices.extend(resolve(token)?);
    }
    if indices.is_empty() {
        return Err(PrismError::Parse {
            line,
            message: format!("`{trimmed}` names no qubit"),
        });
    }
    Ok(Targets::These(indices))
}

fn parse_observable_factor(
    text: &str,
    line: usize,
    resolve: &QubitResolver<'_>,
) -> Result<ObservableFactor> {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("hermitian") {
        let (body, tail) = split_bracketed(rest, '(', ')', line)?;
        let matrix = parse_matrix(body, line)?;
        let targets = parse_targets(tail, line, resolve)?;
        if let Targets::These(indices) = &targets {
            let expected = matrix.len();
            if 1usize << indices.len() != expected {
                return Err(PrismError::Parse {
                    line,
                    message: format!(
                        "a hermitian observable on {} qubit(s) needs a {}x{} matrix, got {expected}x{expected}",
                        indices.len(),
                        1usize << indices.len(),
                        1usize << indices.len()
                    ),
                });
            }
        }
        assert_hermitian(&matrix, line)?;
        return Ok(ObservableFactor::Hermitian { matrix, targets });
    }

    let (name, targets) = match trimmed.find('(') {
        Some(_) => {
            let open = trimmed.find('(').unwrap();
            let (body, tail) = split_bracketed(&trimmed[open..], '(', ')', line)?;
            if !tail.trim().is_empty() {
                return Err(PrismError::Parse {
                    line,
                    message: format!("trailing `{}` after an observable", tail.trim()),
                });
            }
            (trimmed[..open].trim(), parse_targets(body, line, resolve)?)
        }
        None => match trimmed.split_once(char::is_whitespace) {
            Some((name, rest)) => (name.trim(), parse_targets(rest, line, resolve)?),
            None => (trimmed, Targets::All),
        },
    };

    match name {
        "x" | "y" | "z" => Ok(ObservableFactor::Pauli {
            axis: PauliAxis::from_letter(name.chars().next().unwrap()).unwrap(),
            targets,
        }),
        "h" => Ok(ObservableFactor::Hadamard { targets }),
        "i" => Ok(ObservableFactor::Identity { targets }),
        other => Err(PrismError::UnsupportedConstruct {
            construct: format!("observable `{other}`"),
            line,
        }),
    }
}

fn parse_observable(text: &str, line: usize, resolve: &QubitResolver<'_>) -> Result<Observable> {
    let factors = split_top_level(text, '@')
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_observable_factor(part, line, resolve))
        .collect::<Result<Vec<_>>>()?;
    if factors.is_empty() {
        return Err(PrismError::Parse {
            line,
            message: "an observable result needs an observable".to_string(),
        });
    }
    Ok(Observable { factors })
}

/// Parse the body of a `#pragma braket result` line.
pub(crate) fn parse_result_pragma(
    body: &str,
    line: usize,
    num_qubits: usize,
    resolve: &QubitResolver<'_>,
) -> Result<ResultSpec> {
    let spec = read_result_pragma(body, line, resolve)?;
    // Lower once here so an observable that cannot be evaluated is reported
    // against the line that wrote it rather than against the run.
    match &spec {
        ResultSpec::Expectation(observable)
        | ResultSpec::Variance(observable)
        | ResultSpec::Sample(observable) => {
            observable
                .lower(num_qubits)
                .map_err(|e| PrismError::Parse {
                    line,
                    message: e.to_string(),
                })?;
        }
        _ => {}
    }
    Ok(spec)
}

fn read_result_pragma(body: &str, line: usize, resolve: &QubitResolver<'_>) -> Result<ResultSpec> {
    let trimmed = body.trim();
    let (name, rest) = match trimmed.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => (trimmed, ""),
    };
    match name {
        "state_vector" => {
            if rest.is_empty() {
                Ok(ResultSpec::StateVector)
            } else {
                Err(PrismError::Parse {
                    line,
                    message: format!("`state_vector` takes no target, got `{rest}`"),
                })
            }
        }
        "density_matrix" => Ok(ResultSpec::DensityMatrix(parse_targets(
            rest, line, resolve,
        )?)),
        "probability" => Ok(ResultSpec::Probability(parse_targets(rest, line, resolve)?)),
        "amplitude" => {
            let states = split_top_level(rest, ',')
                .into_iter()
                .map(|state| {
                    let state = state.trim().trim_matches('"');
                    if state.is_empty() || state.chars().any(|c| c != '0' && c != '1') {
                        return Err(PrismError::Parse {
                            line,
                            message: format!("`{state}` is not a basis-state bitstring"),
                        });
                    }
                    Ok(state.to_string())
                })
                .collect::<Result<Vec<_>>>()?;
            if states.is_empty() {
                return Err(PrismError::Parse {
                    line,
                    message: "`amplitude` needs at least one basis state".to_string(),
                });
            }
            Ok(ResultSpec::Amplitude(states))
        }
        "expectation" => Ok(ResultSpec::Expectation(parse_observable(
            rest, line, resolve,
        )?)),
        "variance" => Ok(ResultSpec::Variance(parse_observable(rest, line, resolve)?)),
        "sample" => Ok(ResultSpec::Sample(parse_observable(rest, line, resolve)?)),
        "adjoint_gradient" => Err(PrismError::UnsupportedConstruct {
            construct: "`adjoint_gradient`, which Braket serves on SV1 only".to_string(),
            line,
        }),
        other => Err(PrismError::UnsupportedConstruct {
            construct: format!("result type `{other}`"),
            line,
        }),
    }
}

/// A noise channel bound to the qubits a pragma named.
#[derive(Debug, Clone)]
pub(crate) struct NoiseSpec {
    pub channel: NoiseChannel,
    pub qubits: Vec<usize>,
}

fn check_range(name: &str, value: f64, high: f64, line: usize) -> Result<f64> {
    if !(0.0..=high).contains(&value) {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "`{name}` at line {line} takes a probability in [0, {high}], got {value}"
            ),
        });
    }
    Ok(value)
}

fn kraus_1q(entries: &[Vec<Vec<Complex64>>], line: usize) -> Result<Vec<[[Complex64; 2]; 2]>> {
    entries
        .iter()
        .map(|m| {
            if m.len() != 2 {
                return Err(PrismError::Parse {
                    line,
                    message: "expected a 2x2 Kraus operator".to_string(),
                });
            }
            Ok([[m[0][0], m[0][1]], [m[1][0], m[1][1]]])
        })
        .collect()
}

fn kraus_2q(entries: &[Vec<Vec<Complex64>>], line: usize) -> Result<Vec<[[Complex64; 4]; 4]>> {
    entries
        .iter()
        .map(|m| {
            if m.len() != 4 {
                return Err(PrismError::Parse {
                    line,
                    message: "expected a 4x4 Kraus operator".to_string(),
                });
            }
            let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
            for (r, row) in m.iter().enumerate() {
                out[r].copy_from_slice(&row[..4]);
            }
            Ok(out)
        })
        .collect()
}

/// Kraus operators of the generalized amplitude damping channel.
///
/// [`NoiseChannel::AmplitudeDamping`] is the zero-temperature case, so the
/// finite-temperature form has no named variant and goes in as an explicit
/// operator set.
fn generalized_amplitude_damping(gamma: f64, probability: f64) -> Vec<[[Complex64; 2]; 2]> {
    let c = |re: f64| Complex64::new(re, 0.0);
    let (p, q) = (probability.sqrt(), (1.0 - probability).sqrt());
    let (g, g1) = (gamma.sqrt(), (1.0 - gamma).sqrt());
    vec![
        [[c(p), c(0.0)], [c(0.0), c(p * g1)]],
        [[c(0.0), c(p * g)], [c(0.0), c(0.0)]],
        [[c(q * g1), c(0.0)], [c(0.0), c(q)]],
        [[c(0.0), c(0.0)], [c(q * g), c(0.0)]],
    ]
}

/// Kraus operators of the correlated two-qubit dephasing channel.
fn two_qubit_dephasing(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let diag = |a: f64, b: f64, c: f64, d: f64| {
        let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
        for (i, value) in [a, b, c, d].into_iter().enumerate() {
            m[i][i] = Complex64::new(value, 0.0);
        }
        m
    };
    let keep = (1.0 - p).sqrt();
    let each = (p / 3.0).sqrt();
    vec![
        diag(keep, keep, keep, keep),
        diag(each, -each, each, -each),
        diag(each, each, -each, -each),
        diag(each, -each, -each, each),
    ]
}

/// Parse the body of a `#pragma braket noise` line.
pub(crate) fn parse_noise_pragma(
    body: &str,
    line: usize,
    resolve: &QubitResolver<'_>,
) -> Result<NoiseSpec> {
    let trimmed = body.trim();
    let open = trimmed.find('(').ok_or_else(|| PrismError::Parse {
        line,
        message: format!("`{trimmed}` names no noise channel arguments"),
    })?;
    let name = trimmed[..open].trim();
    let (args_text, tail) = split_bracketed(&trimmed[open..], '(', ')', line)?;
    let targets = parse_targets(tail, line, resolve)?;
    let qubits = match targets {
        Targets::These(indices) => indices,
        Targets::All => {
            return Err(PrismError::Parse {
                line,
                message: format!("`{name}` needs an explicit qubit target"),
            });
        }
    };

    if name == "kraus" {
        let entries = split_top_level(args_text, ',')
            .into_iter()
            .filter(|part| !part.trim().is_empty())
            .map(|part| parse_matrix(part, line))
            .collect::<Result<Vec<_>>>()?;
        let channel = match qubits.len() {
            1 => NoiseChannel::Custom {
                kraus: kraus_1q(&entries, line)?,
            },
            2 => NoiseChannel::Kraus2q {
                kraus: kraus_2q(&entries, line)?,
            },
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`kraus` on {other} qubits, which Braket caps at two"),
                    line,
                });
            }
        };
        channel.validate()?;
        return Ok(NoiseSpec { channel, qubits });
    }

    let args = split_top_level(args_text, ',')
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_real(part, line))
        .collect::<Result<Vec<f64>>>()?;
    let expect = |count: usize| -> Result<()> {
        if args.len() == count {
            Ok(())
        } else {
            Err(PrismError::InvalidParameter {
                message: format!(
                    "`{name}` at line {line} takes {count} probability argument(s), got {}",
                    args.len()
                ),
            })
        }
    };
    let expect_qubits = |count: usize| -> Result<()> {
        if qubits.len() == count {
            Ok(())
        } else {
            Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: count,
                got: qubits.len(),
            })
        }
    };

    let channel = match name {
        "bit_flip" => {
            expect(1)?;
            expect_qubits(1)?;
            NoiseChannel::Pauli {
                px: check_range(name, args[0], 0.5, line)?,
                py: 0.0,
                pz: 0.0,
            }
        }
        "phase_flip" => {
            expect(1)?;
            expect_qubits(1)?;
            NoiseChannel::Pauli {
                px: 0.0,
                py: 0.0,
                pz: check_range(name, args[0], 0.5, line)?,
            }
        }
        "pauli_channel" => {
            expect(3)?;
            expect_qubits(1)?;
            NoiseChannel::Pauli {
                px: check_range(name, args[0], 1.0, line)?,
                py: check_range(name, args[1], 1.0, line)?,
                pz: check_range(name, args[2], 1.0, line)?,
            }
        }
        "depolarizing" => {
            expect(1)?;
            expect_qubits(1)?;
            NoiseChannel::Depolarizing {
                p: check_range(name, args[0], 0.75, line)?,
            }
        }
        "amplitude_damping" => {
            expect(1)?;
            expect_qubits(1)?;
            NoiseChannel::AmplitudeDamping {
                gamma: check_range(name, args[0], 1.0, line)?,
            }
        }
        "phase_damping" => {
            expect(1)?;
            expect_qubits(1)?;
            NoiseChannel::PhaseDamping {
                gamma: check_range(name, args[0], 1.0, line)?,
            }
        }
        "generalized_amplitude_damping" => {
            expect(2)?;
            expect_qubits(1)?;
            NoiseChannel::Custom {
                kraus: generalized_amplitude_damping(
                    check_range(name, args[0], 1.0, line)?,
                    check_range(name, args[1], 1.0, line)?,
                ),
            }
        }
        "two_qubit_depolarizing" => {
            expect(1)?;
            expect_qubits(2)?;
            NoiseChannel::TwoQubitDepolarizing {
                p: check_range(name, args[0], 0.9375, line)?,
            }
        }
        "two_qubit_dephasing" => {
            expect(1)?;
            expect_qubits(2)?;
            NoiseChannel::Kraus2q {
                kraus: two_qubit_dephasing(check_range(name, args[0], 0.75, line)?),
            }
        }
        other => {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("noise channel `{other}`"),
                line,
            });
        }
    };
    channel.validate()?;
    Ok(NoiseSpec { channel, qubits })
}

#[cfg(test)]
#[path = "braket_tests.rs"]
mod tests;
