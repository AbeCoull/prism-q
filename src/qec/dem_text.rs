//! Reader for the common detector error model text format.

use super::dem::{DetectorErrorModel, ErrorMechanism, Symptom};
use crate::error::{PrismError, Result};

/// Instructions the expanded text may hold, charged before each top-level instruction or
/// `repeat` block runs, so a pathological repeat count fails at once instead of
/// exhausting memory. A distance-13 surface memory of 1000 rounds writes about 3.6M.
const MAX_DEM_EXPANDED_INSTRUCTIONS: u64 = 1 << 25;

enum Instruction {
    Error {
        probability: f64,
        targets: Vec<Target>,
    },
    Detector {
        coords: Vec<f64>,
        targets: Vec<u64>,
    },
    Observable {
        targets: Vec<u64>,
    },
    Shift {
        coords: Vec<f64>,
        detectors: u64,
    },
    Repeat {
        count: u64,
        body: Vec<(usize, Instruction)>,
    },
}

#[derive(Clone, Copy)]
enum Target {
    Detector(u64),
    Observable(u64),
    Separator,
}

pub(super) fn parse_detector_error_model(text: &str) -> Result<DetectorErrorModel> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(index, raw)| (index + 1, raw.split_once('#').map_or(raw, |(code, _)| code)))
        .map(|(line, code)| (line, code.trim()))
        .filter(|(_, code)| !code.is_empty());
    let mut state = State::default();
    let mut scratch = Vec::new();
    while let Some((line, code)) = lines.next() {
        if code == "}" {
            return Err(dem_error(line, "unmatched `}`"));
        }
        let (name, args, targets) = split_line(code, line)?;
        if name.eq_ignore_ascii_case("error") {
            let probability = parse_error(args, targets, line, &mut scratch)?;
            state.charge(1, line)?;
            state.error(probability, &scratch, line)?;
            continue;
        }
        let block = [(
            line,
            parse_instruction(name, args, targets, line, &mut lines)?,
        )];
        state.charge(expanded_len(&block), line)?;
        state.run(&block)?;
    }
    Ok(state.finish())
}

fn expanded_len(block: &[(usize, Instruction)]) -> u64 {
    block.iter().fold(0u64, |total, (_, instruction)| {
        let own = match instruction {
            Instruction::Repeat { count, body } => count.saturating_mul(expanded_len(body)),
            _ => 1,
        };
        total.saturating_add(own)
    })
}

/// Read a `repeat` body, opened on line `opened_at`, through its closing `}`.
fn parse_block<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
    opened_at: usize,
) -> Result<Vec<(usize, Instruction)>> {
    let mut block = Vec::new();
    while let Some((line, code)) = lines.next() {
        if code == "}" {
            return Ok(block);
        }
        let (name, args, targets) = split_line(code, line)?;
        let instruction = parse_instruction(name, args, targets, line, lines)?;
        block.push((line, instruction));
    }
    Err(dem_error(opened_at, "unterminated `repeat` block"))
}

fn parse_instruction<'a>(
    name: &str,
    args: &str,
    targets: &str,
    line: usize,
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<Instruction> {
    let is = |expected: &str| name.eq_ignore_ascii_case(expected);
    Ok(if is("error") {
        let mut parsed = Vec::new();
        let probability = parse_error(args, targets, line, &mut parsed)?;
        Instruction::Error {
            probability,
            targets: parsed,
        }
    } else if is("detector") {
        Instruction::Detector {
            coords: parse_args(args, line)?,
            targets: parse_indexed(targets, 'D', line)?,
        }
    } else if is("logical_observable") {
        if !args.is_empty() {
            return Err(dem_error(line, "`logical_observable` takes no arguments"));
        }
        Instruction::Observable {
            targets: parse_indexed(targets, 'L', line)?,
        }
    } else if is("shift_detectors") {
        let mut shift = targets.split_whitespace();
        let detectors = match shift.next() {
            Some(count) => parse_u64(count, line)?,
            None => 0,
        };
        if shift.next().is_some() {
            return Err(dem_error(line, "`shift_detectors` takes one target"));
        }
        Instruction::Shift {
            coords: parse_args(args, line)?,
            detectors,
        }
    } else if is("repeat") {
        let mut parts = targets.split_whitespace();
        let (Some(count), Some("{"), None) = (parts.next(), parts.next(), parts.next()) else {
            return Err(dem_error(line, "`repeat` must have the form `repeat N {`"));
        };
        Instruction::Repeat {
            count: parse_u64(count, line)?,
            body: parse_block(lines, line)?,
        }
    } else {
        return Err(dem_error(
            line,
            format!("unsupported detector error model instruction `{name}`"),
        ));
    })
}

/// Split `name[tag](args) targets` into the name, the argument text, and the target
/// text. The tag is dropped.
fn split_line(code: &str, line: usize) -> Result<(&str, &str, &str)> {
    let name_end = code
        .find(|c: char| c == '(' || c == '[' || c.is_whitespace())
        .unwrap_or(code.len());
    let name = &code[..name_end];
    let mut rest = &code[name_end..];
    if let Some(tagged) = rest.strip_prefix('[') {
        let close = tagged
            .find(']')
            .ok_or_else(|| dem_error(line, "unterminated instruction tag"))?;
        rest = &tagged[close + 1..];
    }
    let mut args = "";
    if let Some(opened) = rest.strip_prefix('(') {
        let close = opened
            .find(')')
            .ok_or_else(|| dem_error(line, "unterminated instruction arguments"))?;
        args = &opened[..close];
        rest = &opened[close + 1..];
    }
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return Err(dem_error(line, format!("malformed instruction `{code}`")));
    }
    Ok((name, args, rest.trim()))
}

/// Parse an `error` line into its probability and `parsed`, the targets in order with
/// `^` as [`Target::Separator`].
fn parse_error(args: &str, targets: &str, line: usize, parsed: &mut Vec<Target>) -> Result<f64> {
    if args.contains(',') || args.trim().is_empty() {
        return Err(dem_error(line, "`error` takes one probability argument"));
    }
    let probability = args
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|p| (0.0..=1.0).contains(p))
        .ok_or_else(|| dem_error(line, "`error` probability must be a number in [0, 1]"))?;
    parsed.clear();
    let mut component_len = 0usize;
    for token in targets.split_whitespace() {
        if token == "^" {
            if component_len == 0 {
                return Err(dem_error(line, "`^` must separate non-empty components"));
            }
            parsed.push(Target::Separator);
            component_len = 0;
            continue;
        }
        let target = match token.as_bytes().first() {
            Some(b'D') => Target::Detector(parse_u64(&token[1..], line)?),
            Some(b'L') => Target::Observable(parse_u64(&token[1..], line)?),
            _ => {
                return Err(dem_error(
                    line,
                    format!("expected a `D`, `L`, or `^` target, got `{token}`"),
                ));
            }
        };
        parsed.push(target);
        component_len += 1;
    }
    if component_len == 0 && !parsed.is_empty() {
        return Err(dem_error(line, "`^` must separate non-empty components"));
    }
    Ok(probability)
}

fn parse_args(args: &str, line: usize) -> Result<Vec<f64>> {
    if args.trim().is_empty() {
        return Ok(Vec::new());
    }
    args.split(',')
        .map(|arg| {
            let value = arg
                .trim()
                .parse::<f64>()
                .map_err(|_| dem_error(line, format!("expected a number, got `{}`", arg.trim())))?;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(dem_error(line, "arguments must be finite"))
            }
        })
        .collect()
}

fn parse_indexed(targets: &str, prefix: char, line: usize) -> Result<Vec<u64>> {
    targets
        .split_whitespace()
        .map(|token| match token.strip_prefix(prefix) {
            Some(index) => parse_u64(index, line),
            None => Err(dem_error(
                line,
                format!("expected a `{prefix}` target, got `{token}`"),
            )),
        })
        .collect()
}

fn parse_u64(text: &str, line: usize) -> Result<u64> {
    text.parse::<u64>()
        .map_err(|_| dem_error(line, format!("expected an unsigned integer, got `{text}`")))
}

fn dem_error(line: usize, message: impl Into<String>) -> PrismError {
    PrismError::Parse {
        line,
        message: message.into(),
    }
}

#[derive(Default)]
struct State {
    mechanisms: Vec<ErrorMechanism>,
    coords: Vec<Vec<f64>>,
    detector_offset: u64,
    coord_shift: Vec<f64>,
    num_detectors: u64,
    num_observables: u64,
    executed: u64,
}

impl State {
    /// Count `instructions` more toward the expansion limit before they run.
    fn charge(&mut self, instructions: u64, line: usize) -> Result<()> {
        self.executed = self.executed.saturating_add(instructions);
        if self.executed > MAX_DEM_EXPANDED_INSTRUCTIONS {
            return Err(dem_error(
                line,
                format!("`repeat` expansion exceeds {MAX_DEM_EXPANDED_INSTRUCTIONS} instructions"),
            ));
        }
        Ok(())
    }

    fn run(&mut self, block: &[(usize, Instruction)]) -> Result<()> {
        for (line, instruction) in block {
            match instruction {
                Instruction::Error {
                    probability,
                    targets,
                } => self.error(*probability, targets, *line)?,
                Instruction::Detector { coords, targets } => {
                    for &target in targets {
                        let detector = self.detector_index(target, *line)?;
                        let shifted = coords
                            .iter()
                            .enumerate()
                            .map(|(axis, c)| c + self.coord_shift.get(axis).copied().unwrap_or(0.0))
                            .collect();
                        if self.coords.len() <= detector {
                            self.coords.resize_with(detector + 1, Vec::new);
                        }
                        self.coords[detector] = shifted;
                    }
                }
                Instruction::Observable { targets } => {
                    for &target in targets {
                        self.num_observables = self.num_observables.max(target + 1);
                    }
                }
                Instruction::Shift { coords, detectors } => {
                    if self.coord_shift.len() < coords.len() {
                        self.coord_shift.resize(coords.len(), 0.0);
                    }
                    for (shift, c) in self.coord_shift.iter_mut().zip(coords) {
                        *shift += c;
                    }
                    self.detector_offset = self
                        .detector_offset
                        .checked_add(*detectors)
                        .ok_or_else(|| dem_error(*line, "detector shift overflows"))?;
                }
                Instruction::Repeat { count, body } => {
                    for _ in 0..*count {
                        self.run(body)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn detector_index(&mut self, target: u64, line: usize) -> Result<usize> {
        let index = target
            .checked_add(self.detector_offset)
            .filter(|&index| index < u32::MAX as u64)
            .ok_or_else(|| dem_error(line, "detector index out of range"))?;
        self.num_detectors = self.num_detectors.max(index + 1);
        Ok(index as usize)
    }

    fn error(&mut self, probability: f64, targets: &[Target], line: usize) -> Result<()> {
        let mut components: Vec<Symptom> = Vec::new();
        if targets.iter().any(|t| matches!(t, Target::Separator)) {
            for component in targets.split(|t| matches!(t, Target::Separator)) {
                components.push(self.symptom(component, line)?);
            }
        }
        let (detectors, observables) = if components.is_empty() {
            self.symptom(targets, line)?
        } else {
            let mut detectors: Vec<usize> = components.iter().flat_map(|c| c.0.clone()).collect();
            let mut observables: Vec<usize> = components.iter().flat_map(|c| c.1.clone()).collect();
            cancel_pairs(&mut detectors);
            cancel_pairs(&mut observables);
            (detectors, observables)
        };
        self.mechanisms.push(ErrorMechanism::new(
            probability,
            detectors,
            observables,
            components,
        ));
        Ok(())
    }

    fn symptom(&mut self, targets: &[Target], line: usize) -> Result<Symptom> {
        let mut detectors = Vec::with_capacity(targets.len());
        let mut observables = Vec::new();
        for &target in targets {
            match target {
                Target::Detector(index) => detectors.push(self.detector_index(index, line)?),
                Target::Observable(index) => {
                    self.num_observables = self.num_observables.max(index + 1);
                    observables.push(index as usize);
                }
                Target::Separator => unreachable!("components are split on separators"),
            }
        }
        cancel_pairs(&mut detectors);
        cancel_pairs(&mut observables);
        Ok((detectors, observables))
    }

    fn finish(mut self) -> DetectorErrorModel {
        let num_detectors = self.num_detectors as usize;
        self.coords.resize_with(num_detectors, Vec::new);
        DetectorErrorModel::from_parts(
            self.mechanisms,
            self.coords,
            num_detectors,
            self.num_observables as usize,
        )
    }
}

/// Sort `indices` and drop every index that appears an even number of times.
fn cancel_pairs(indices: &mut Vec<usize>) {
    indices.sort_unstable();
    let mut kept = 0;
    for at in 0..indices.len() {
        if kept > 0 && indices[kept - 1] == indices[at] {
            kept -= 1;
        } else {
            indices[kept] = indices[at];
            kept += 1;
        }
    }
    indices.truncate(kept);
}
