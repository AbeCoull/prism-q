use super::{QecBasis, QecNoise, QecOp, QecOptions, QecPauli, QecProgram, QecRecordRef};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

const MAX_QEC_EXPANDED_LINES: usize = 1_000_000;

/// Parse a native measurement-record QEC program from text.
///
/// Recognized instructions: `H`, `S`, `S_DAG`, `T`, `T_DAG`, `CX`, `CZ`,
/// `R`/`RX`/`RY`, `M`/`MX`/`MY`, `MR`/`MRX`/`MRY`, `MPP`, `DETECTOR`,
/// `OBSERVABLE_INCLUDE`, `POSTSELECT`, `EXP_VAL`, `X_ERROR`, `Z_ERROR`,
/// `DEPOLARIZE1`, `DEPOLARIZE2`, `TICK`, `QUBIT_COORDS`, `SHIFT_COORDS`, and
/// flattened `REPEAT` blocks. `rec[-k]` measurement record references are
/// resolved against records emitted up to that point. Comments use `#`.
///
/// `M(p)` and `MR(p)` lower the optional measurement-error probability into a
/// pre-measurement Pauli flip annotation; `MPP` does not currently accept a
/// measurement-error argument.
pub fn parse_qec_program(input: &str) -> Result<QecProgram> {
    let lines = expand_repeats(input)?;
    let mut parser = QecTextParser::default();
    for (line_num, line) in lines {
        parser.parse_line(line_num, &line)?;
    }
    let num_qubits = parser.max_qubit.map_or(0, |qubit| qubit + 1);
    QecProgram::from_ops(num_qubits, QecOptions::default(), parser.ops)
}

#[derive(Default)]
struct QecTextParser {
    ops: Vec<QecOp>,
    measurement_count: usize,
    max_qubit: Option<usize>,
}

impl QecTextParser {
    fn parse_line(&mut self, line_num: usize, line: &str) -> Result<()> {
        let (name, args, targets) = split_instruction(line, line_num)?;
        match name.as_str() {
            "I" | "X" | "Y" | "Z" | "H" | "S" | "S_DAG" | "SDAG" | "T" | "T_DAG" | "TDG" => {
                self.parse_single_qubit_gate(&name, &targets, line_num)
            }
            "CX" | "CNOT" | "CZ" => self.parse_two_qubit_gate(&name, &targets, line_num),
            "R" | "RZ" | "RX" | "RY" => {
                self.parse_reset(measurement_basis(&name), &targets, line_num)
            }
            "M" | "MZ" | "MX" | "MY" => self.parse_measurement(
                measurement_basis(&name),
                args.as_deref(),
                &targets,
                line_num,
            ),
            "MR" | "MRZ" | "MRX" | "MRY" => self.parse_measure_reset(
                measurement_basis(&name),
                args.as_deref(),
                &targets,
                line_num,
            ),
            "MPP" => self.parse_mpp(args.as_deref(), &targets, line_num),
            "DETECTOR" => self.parse_detector(args.as_deref(), &targets, line_num),
            "OBSERVABLE_INCLUDE" => {
                self.parse_observable_include(args.as_deref(), &targets, line_num)
            }
            "POSTSELECT" => self.parse_postselect(args.as_deref(), &targets, line_num),
            "EXP_VAL" => self.parse_exp_val(args.as_deref(), &targets, line_num),
            "X_ERROR" | "Z_ERROR" | "DEPOLARIZE1" | "DEPOLARIZE2" => {
                self.parse_noise(&name, args.as_deref(), &targets, line_num)
            }
            "QUBIT_COORDS" => self.parse_qubit_coords(&targets, line_num),
            "SHIFT_COORDS" => Ok(()),
            "TICK" => {
                self.ops.push(QecOp::Tick);
                Ok(())
            }
            _ => Err(qec_parse_error(
                line_num,
                format!("unsupported QEC instruction `{name}`"),
            )),
        }
    }

    fn parse_single_qubit_gate(
        &mut self,
        name: &str,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let gate = match name {
            "I" => Gate::Id,
            "X" => Gate::X,
            "Y" => Gate::Y,
            "Z" => Gate::Z,
            "H" => Gate::H,
            "S" => Gate::S,
            "S_DAG" | "SDAG" => Gate::Sdg,
            "T" => Gate::T,
            "T_DAG" | "TDG" => Gate::Tdg,
            _ => unreachable!(),
        };
        let qubits = parse_qubit_targets(targets, line_num)?;
        require_non_empty_targets(&qubits, name, line_num)?;
        for qubit in qubits {
            self.note_qubit(qubit);
            self.ops.push(QecOp::Gate {
                gate: gate.clone(),
                targets: vec![qubit],
            });
        }
        Ok(())
    }

    fn parse_two_qubit_gate(
        &mut self,
        name: &str,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let qubits = parse_qubit_targets(targets, line_num)?;
        require_non_empty_targets(&qubits, name, line_num)?;
        if qubits.len() % 2 != 0 {
            return Err(qec_parse_error(
                line_num,
                format!("`{name}` requires an even number of qubit targets"),
            ));
        }
        let gate = if name == "CZ" { Gate::Cz } else { Gate::Cx };
        for pair in qubits.chunks_exact(2) {
            self.note_qubit(pair[0]);
            self.note_qubit(pair[1]);
            self.ops.push(QecOp::Gate {
                gate: gate.clone(),
                targets: pair.to_vec(),
            });
        }
        Ok(())
    }

    fn parse_reset(&mut self, basis: QecBasis, targets: &[String], line_num: usize) -> Result<()> {
        let qubits = parse_qubit_targets(targets, line_num)?;
        require_non_empty_targets(&qubits, "reset", line_num)?;
        for qubit in qubits {
            self.note_qubit(qubit);
            self.ops.push(QecOp::Reset { basis, qubit });
        }
        Ok(())
    }

    fn parse_measurement(
        &mut self,
        basis: QecBasis,
        args: Option<&str>,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let measurement_error = parse_optional_probability_arg(args, "measurement", line_num)?;
        let qubits = parse_qubit_targets(targets, line_num)?;
        require_non_empty_targets(&qubits, "measurement", line_num)?;
        for qubit in qubits {
            self.note_qubit(qubit);
            if let Some(p) = measurement_error.filter(|&p| p > 0.0) {
                self.ops.push(QecOp::Noise {
                    channel: measurement_error_channel(basis, p),
                    targets: vec![qubit],
                });
            }
            self.ops.push(QecOp::Measure { basis, qubit });
            self.measurement_count += 1;
        }
        Ok(())
    }

    fn parse_measure_reset(
        &mut self,
        basis: QecBasis,
        args: Option<&str>,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let measurement_error = parse_optional_probability_arg(args, "measure-reset", line_num)?;
        let qubits = parse_qubit_targets(targets, line_num)?;
        require_non_empty_targets(&qubits, "measure-reset", line_num)?;
        for qubit in qubits {
            self.note_qubit(qubit);
            if let Some(p) = measurement_error.filter(|&p| p > 0.0) {
                self.ops.push(QecOp::Noise {
                    channel: measurement_error_channel(basis, p),
                    targets: vec![qubit],
                });
            }
            self.ops.push(QecOp::Measure { basis, qubit });
            self.measurement_count += 1;
            self.ops.push(QecOp::Reset { basis, qubit });
        }
        Ok(())
    }

    fn parse_mpp(&mut self, args: Option<&str>, targets: &[String], line_num: usize) -> Result<()> {
        if !parse_f64_args(args, line_num)?.is_empty() {
            return Err(qec_parse_error(
                line_num,
                "`MPP` does not support measurement-error arguments yet",
            ));
        }
        if targets.is_empty() {
            return Err(qec_parse_error(
                line_num,
                "`MPP` requires at least one Pauli product",
            ));
        }
        for target in targets {
            let terms = parse_pauli_product(target, line_num)?;
            for term in &terms {
                self.note_qubit(term.qubit);
            }
            self.ops.push(QecOp::MeasurePauliProduct { terms });
            self.measurement_count += 1;
        }
        Ok(())
    }

    fn parse_detector(
        &mut self,
        args: Option<&str>,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let coords = parse_f64_args(args, line_num)?;
        let records = self.parse_record_targets(targets, line_num)?;
        self.ops.push(QecOp::Detector { records, coords });
        Ok(())
    }

    fn parse_observable_include(
        &mut self,
        args: Option<&str>,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let observable = parse_single_usize_arg(args, "OBSERVABLE_INCLUDE", line_num)?;
        let records = self.parse_record_targets(targets, line_num)?;
        self.ops.push(QecOp::ObservableInclude {
            observable,
            records,
        });
        Ok(())
    }

    fn parse_postselect(
        &mut self,
        args: Option<&str>,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let expected = parse_optional_bool_arg(args, "POSTSELECT", line_num)?;
        let records = self.parse_record_targets(targets, line_num)?;
        self.ops.push(QecOp::Postselect { records, expected });
        Ok(())
    }

    fn parse_exp_val(
        &mut self,
        args: Option<&str>,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let coefficient = parse_optional_coefficient(args, line_num)?;
        if targets.is_empty() {
            return Err(qec_parse_error(
                line_num,
                "`EXP_VAL` requires at least one Pauli target",
            ));
        }
        let mut terms = Vec::new();
        for target in targets {
            let mut product = parse_pauli_product(target, line_num)?;
            terms.append(&mut product);
        }
        for term in &terms {
            self.note_qubit(term.qubit);
        }
        self.ops
            .push(QecOp::ExpectationValue { terms, coefficient });
        Ok(())
    }

    fn parse_noise(
        &mut self,
        name: &str,
        args: Option<&str>,
        targets: &[String],
        line_num: usize,
    ) -> Result<()> {
        let p = parse_single_f64_arg(args, name, line_num)?;
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(qec_parse_error(
                line_num,
                format!("`{name}` probability must be finite and in [0, 1]"),
            ));
        }
        let qubits = parse_qubit_targets(targets, line_num)?;
        require_non_empty_targets(&qubits, name, line_num)?;
        for &qubit in &qubits {
            self.note_qubit(qubit);
        }
        let channel = match name {
            "X_ERROR" => QecNoise::XError(p),
            "Z_ERROR" => QecNoise::ZError(p),
            "DEPOLARIZE1" => QecNoise::Depolarize1(p),
            "DEPOLARIZE2" => QecNoise::Depolarize2(p),
            _ => unreachable!(),
        };
        if p > 0.0 {
            self.ops.push(QecOp::Noise {
                channel,
                targets: qubits,
            });
        }
        Ok(())
    }

    fn parse_qubit_coords(&mut self, targets: &[String], line_num: usize) -> Result<()> {
        let qubits = parse_qubit_targets(targets, line_num)?;
        for qubit in qubits {
            self.note_qubit(qubit);
        }
        Ok(())
    }

    fn parse_record_targets(
        &self,
        targets: &[String],
        line_num: usize,
    ) -> Result<Vec<QecRecordRef>> {
        targets
            .iter()
            .map(|target| parse_record_ref(target, self.measurement_count, line_num))
            .collect()
    }

    fn note_qubit(&mut self, qubit: usize) {
        self.max_qubit = Some(self.max_qubit.map_or(qubit, |max| max.max(qubit)));
    }
}

fn measurement_basis(name: &str) -> QecBasis {
    if name.ends_with('X') {
        QecBasis::X
    } else if name.ends_with('Y') {
        QecBasis::Y
    } else {
        QecBasis::Z
    }
}

fn expand_repeats(input: &str) -> Result<Vec<(usize, String)>> {
    let mut lines = Vec::new();
    for (idx, raw) in input.lines().enumerate() {
        let line = strip_qec_comment(raw).trim();
        if !line.is_empty() {
            lines.push((idx + 1, line.to_string()));
        }
    }

    let mut index = 0;
    let expanded = expand_repeat_block(&lines, &mut index, false)?;
    if index != lines.len() {
        let (line_num, _) = &lines[index];
        return Err(qec_parse_error(
            *line_num,
            "unexpected trailing repeat block",
        ));
    }
    Ok(expanded)
}

fn expand_repeat_block(
    lines: &[(usize, String)],
    index: &mut usize,
    in_repeat: bool,
) -> Result<Vec<(usize, String)>> {
    let mut expanded = Vec::new();
    while *index < lines.len() {
        let (line_num, line) = &lines[*index];
        if line == "}" {
            if !in_repeat {
                return Err(qec_parse_error(*line_num, "unmatched `}`"));
            }
            *index += 1;
            return Ok(expanded);
        }

        if line.to_ascii_uppercase().starts_with("REPEAT ") {
            let count = parse_repeat_header(line, *line_num)?;
            *index += 1;
            let body = expand_repeat_block(lines, index, true)?;
            let repeated_len = body.len().checked_mul(count).ok_or_else(|| {
                qec_parse_error(
                    *line_num,
                    format!("`REPEAT` expansion exceeds {MAX_QEC_EXPANDED_LINES} instructions"),
                )
            })?;
            if expanded.len().saturating_add(repeated_len) > MAX_QEC_EXPANDED_LINES {
                return Err(qec_parse_error(
                    *line_num,
                    format!("`REPEAT` expansion exceeds {MAX_QEC_EXPANDED_LINES} instructions"),
                ));
            }
            for _ in 0..count {
                expanded.extend(body.iter().cloned());
            }
            continue;
        }

        if line == "{" || line.contains('{') {
            return Err(qec_parse_error(*line_num, "unexpected `{`"));
        }

        if expanded.len() == MAX_QEC_EXPANDED_LINES {
            return Err(qec_parse_error(
                *line_num,
                format!("QEC text exceeds {MAX_QEC_EXPANDED_LINES} instructions"),
            ));
        }
        expanded.push((*line_num, line.clone()));
        *index += 1;
    }

    if in_repeat {
        return Err(PrismError::Parse {
            line: lines.last().map_or(1, |(line_num, _)| *line_num),
            message: "unterminated `REPEAT` block".to_string(),
        });
    }
    Ok(expanded)
}

fn parse_repeat_header(line: &str, line_num: usize) -> Result<usize> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 3 || !parts[0].eq_ignore_ascii_case("REPEAT") || parts[2] != "{" {
        return Err(qec_parse_error(
            line_num,
            "`REPEAT` must have form `REPEAT count {`",
        ));
    }
    parts[1]
        .parse::<usize>()
        .map_err(|_| qec_parse_error(line_num, "`REPEAT` count must be an unsigned integer"))
}

fn strip_qec_comment(line: &str) -> &str {
    line.split_once('#').map_or(line, |(before, _)| before)
}

fn split_instruction(line: &str, line_num: usize) -> Result<(String, Option<String>, Vec<String>)> {
    let line = line.trim();
    let (head, rest) = if let Some(open) = line.find('(') {
        let first_ws = line.find(char::is_whitespace);
        if first_ws.map_or(true, |ws| open < ws) {
            let close = line[open..]
                .find(')')
                .map(|offset| open + offset)
                .ok_or_else(|| qec_parse_error(line_num, "unterminated instruction arguments"))?;
            (&line[..=close], line[close + 1..].trim())
        } else {
            split_head_word(line)
        }
    } else {
        split_head_word(line)
    };

    let (name, args) = if let Some(open) = head.find('(') {
        if !head.ends_with(')') {
            return Err(qec_parse_error(line_num, "malformed instruction arguments"));
        }
        (
            head[..open].to_ascii_uppercase(),
            Some(head[open + 1..head.len() - 1].trim().to_string()),
        )
    } else {
        (head.to_ascii_uppercase(), None)
    };
    let targets = rest.split_whitespace().map(ToString::to_string).collect();
    Ok((name, args, targets))
}

fn split_head_word(line: &str) -> (&str, &str) {
    match line.find(char::is_whitespace) {
        Some(idx) => (&line[..idx], line[idx..].trim()),
        None => (line, ""),
    }
}

fn parse_f64_args(args: Option<&str>, line_num: usize) -> Result<Vec<f64>> {
    let Some(args) = args else {
        return Ok(Vec::new());
    };
    if args.trim().is_empty() {
        return Ok(Vec::new());
    }
    args.split(',')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .map_err(|_| qec_parse_error(line_num, "expected numeric argument"))
        })
        .collect()
}

fn parse_single_f64_arg(args: Option<&str>, name: &str, line_num: usize) -> Result<f64> {
    let values = parse_f64_args(args, line_num)?;
    if values.len() != 1 {
        return Err(qec_parse_error(
            line_num,
            format!("`{name}` requires one numeric argument"),
        ));
    }
    Ok(values[0])
}

fn parse_optional_coefficient(args: Option<&str>, line_num: usize) -> Result<f64> {
    let values = parse_f64_args(args, line_num)?;
    match values.as_slice() {
        [] => Ok(1.0),
        [coefficient] => Ok(*coefficient),
        _ => Err(qec_parse_error(
            line_num,
            "`EXP_VAL` accepts zero or one numeric argument",
        )),
    }
}

fn parse_optional_probability_arg(
    args: Option<&str>,
    name: &str,
    line_num: usize,
) -> Result<Option<f64>> {
    let values = parse_f64_args(args, line_num)?;
    match values.as_slice() {
        [] => Ok(None),
        [p] if p.is_finite() && (0.0..=1.0).contains(p) => Ok(Some(*p)),
        [_] => Err(qec_parse_error(
            line_num,
            format!("`{name}` probability must be finite and in [0, 1]"),
        )),
        _ => Err(qec_parse_error(
            line_num,
            format!("`{name}` accepts zero or one numeric argument"),
        )),
    }
}

fn measurement_error_channel(basis: QecBasis, p: f64) -> QecNoise {
    match basis {
        QecBasis::X | QecBasis::Y => QecNoise::ZError(p),
        QecBasis::Z => QecNoise::XError(p),
    }
}

fn parse_optional_bool_arg(args: Option<&str>, name: &str, line_num: usize) -> Result<bool> {
    let values = parse_f64_args(args, line_num)?;
    match values.as_slice() {
        [] => Ok(false),
        [v] if *v == 0.0 => Ok(false),
        [v] if *v == 1.0 => Ok(true),
        [_] => Err(qec_parse_error(
            line_num,
            format!("`{name}` expected value must be 0 or 1"),
        )),
        _ => Err(qec_parse_error(
            line_num,
            format!("`{name}` accepts zero or one expected-value argument"),
        )),
    }
}

fn parse_single_usize_arg(args: Option<&str>, name: &str, line_num: usize) -> Result<usize> {
    let Some(args) = args else {
        return Err(qec_parse_error(
            line_num,
            format!("`{name}` requires one integer argument"),
        ));
    };
    if args.split(',').count() != 1 {
        return Err(qec_parse_error(
            line_num,
            format!("`{name}` requires one integer argument"),
        ));
    }
    args.trim()
        .parse::<usize>()
        .map_err(|_| qec_parse_error(line_num, format!("`{name}` argument must be an integer")))
}

fn parse_qubit_targets(targets: &[String], line_num: usize) -> Result<Vec<usize>> {
    targets
        .iter()
        .map(|target| parse_qubit_target(target, line_num))
        .collect()
}

fn parse_qubit_target(target: &str, line_num: usize) -> Result<usize> {
    reject_inverted_target(target, line_num)?;
    target
        .parse::<usize>()
        .map_err(|_| qec_parse_error(line_num, format!("expected qubit target, got `{target}`")))
}

fn parse_pauli_product(target: &str, line_num: usize) -> Result<Vec<QecPauli>> {
    reject_inverted_target(target, line_num)?;
    if target.is_empty() {
        return Err(qec_parse_error(line_num, "empty Pauli product"));
    }
    target
        .split('*')
        .map(|term| parse_pauli_term(term, line_num))
        .collect()
}

fn parse_pauli_term(term: &str, line_num: usize) -> Result<QecPauli> {
    reject_inverted_target(term, line_num)?;
    let mut chars = term.chars();
    let basis = match chars.next() {
        Some('X') | Some('x') => QecBasis::X,
        Some('Y') | Some('y') => QecBasis::Y,
        Some('Z') | Some('z') => QecBasis::Z,
        _ => {
            return Err(qec_parse_error(
                line_num,
                format!("expected Pauli target, got `{term}`"),
            ))
        }
    };
    let qubit_text = chars.as_str();
    if qubit_text.is_empty() {
        return Err(qec_parse_error(
            line_num,
            format!("Pauli target `{term}` is missing a qubit index"),
        ));
    }
    let qubit = qubit_text.parse::<usize>().map_err(|_| {
        qec_parse_error(
            line_num,
            format!("Pauli target `{term}` has an invalid qubit index"),
        )
    })?;
    Ok(QecPauli::new(basis, qubit))
}

fn parse_record_ref(
    target: &str,
    measurement_count: usize,
    line_num: usize,
) -> Result<QecRecordRef> {
    reject_inverted_target(target, line_num)?;
    if !target.starts_with("rec[") || !target.ends_with(']') {
        return Err(qec_parse_error(
            line_num,
            format!("expected measurement record target, got `{target}`"),
        ));
    }
    let offset = target[4..target.len() - 1].parse::<isize>().map_err(|_| {
        qec_parse_error(
            line_num,
            format!("measurement record target `{target}` has invalid offset"),
        )
    })?;
    if offset >= 0 {
        return Err(qec_parse_error(
            line_num,
            "measurement record offsets must be negative",
        ));
    }
    let distance = offset.unsigned_abs();
    if distance == 0 || distance > measurement_count {
        return Err(qec_parse_error(
            line_num,
            format!("measurement record `{target}` out of bounds for {measurement_count} records"),
        ));
    }
    Ok(QecRecordRef::Absolute(measurement_count - distance))
}

fn reject_inverted_target(target: &str, line_num: usize) -> Result<()> {
    if target.starts_with('!') {
        return Err(qec_parse_error(
            line_num,
            format!("inverted target `{target}` is not supported in native QEC text"),
        ));
    }
    Ok(())
}

fn require_non_empty_targets(targets: &[usize], name: &str, line_num: usize) -> Result<()> {
    if targets.is_empty() {
        return Err(qec_parse_error(
            line_num,
            format!("`{name}` requires at least one target"),
        ));
    }
    Ok(())
}

fn qec_parse_error(line: usize, message: impl Into<String>) -> PrismError {
    PrismError::Parse {
        line,
        message: message.into(),
    }
}
