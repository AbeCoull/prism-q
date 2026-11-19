//! Timing under a model where every gate takes no time: the length
//! `durationof` folds to, and the stretches a fixed-length `box` resolves.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use super::eval::replace_nodes;
use super::*;
use crate::circuit::qasm::ast::{Block, Operand, StmtKind};
use crate::circuit::qasm::expr::{Duration, Expr, Stretched, Timed};
use crate::circuit::qasm::parser as syntax;
use crate::circuit::qasm::stream::Stream;

/// Why a block has no length under the zero-duration model.
enum Unscheduled {
    /// A statement other than a gate, `measure`, `reset`, `barrier`, `delay`,
    /// `nop` or `box`.
    Statement,
    /// Qubit timelines whose order depends on the length of `dt`.
    Mixed,
    /// A stretch with a negative coefficient, which backtracks.
    Backtrack,
}

/// Each qubit's time since the block opened, and the equations its stretches
/// meet at every point where qubits synchronise.
#[derive(Default)]
struct Schedule {
    at: BTreeMap<usize, Stretched>,
    /// The stretch terms of one qubit's timeline, and the length they make up.
    equations: Vec<(Stretched, Duration)>,
}

impl Schedule {
    fn touch(&mut self, qubit: usize) {
        self.at
            .entry(qubit)
            .or_insert_with(|| Stretched::fixed(Duration::default()));
    }

    /// Bring `qubits` to one time, `forced` or else the latest fixed part among
    /// them, each stretch on the way growing to fill its qubit's idle time, and
    /// return that time. A lone qubit with nothing forced keeps its timeline.
    fn align(
        &mut self,
        qubits: &[usize],
        forced: Option<Duration>,
        line: usize,
    ) -> Result<std::result::Result<Stretched, Unscheduled>> {
        let zero = Stretched::fixed(Duration::default());
        if forced.is_none() && qubits.len() < 2 {
            return Ok(Ok(match qubits.first() {
                Some(&qubit) => {
                    self.touch(qubit);
                    self.at[&qubit]
                }
                None => zero,
            }));
        }
        let forms: Vec<Stretched> = qubits
            .iter()
            .map(|q| self.at.get(q).copied().unwrap_or(zero))
            .collect();
        if forms
            .iter()
            .any(|form| form.terms().iter().any(|&(_, coeff)| coeff < 0.0))
        {
            return Ok(Err(Unscheduled::Backtrack));
        }
        let target = match forced {
            Some(length) => {
                for form in &forms {
                    match form.base.order(length) {
                        Some(Ordering::Greater) => {
                            return Err(PrismError::Parse {
                                line,
                                message: "`box` holds delays longer than its duration".into(),
                            });
                        }
                        Some(_) => {}
                        None => return Ok(Err(Unscheduled::Mixed)),
                    }
                }
                length
            }
            None => {
                let mut latest = Duration::default();
                for form in &forms {
                    match form.base.order(latest) {
                        Some(Ordering::Greater) => latest = form.base,
                        Some(_) => {}
                        None => return Ok(Err(Unscheduled::Mixed)),
                    }
                }
                latest
            }
        };
        for form in forms.into_iter().filter(|form| !form.terms().is_empty()) {
            let mut terms = form;
            terms.base = Duration::default();
            self.equations.push((terms, target.minus(form.base)));
        }
        let target = Stretched::fixed(target);
        for &qubit in qubits {
            self.at.insert(qubit, target);
        }
        Ok(Ok(target))
    }

    /// Close the block: every qubit it touched ends at one time, its length.
    fn finish(
        &mut self,
        forced: Option<Duration>,
        line: usize,
    ) -> Result<std::result::Result<(Vec<usize>, Stretched), Unscheduled>> {
        let used: Vec<usize> = self.at.keys().copied().collect();
        Ok(self
            .align(&used, forced, line)?
            .map(|length| (used, length)))
    }
}

/// The stretch growths the equations fix uniquely and non-negatively. An
/// inconsistent system, or one forcing a negative growth, resolves nothing.
fn solve(equations: &[(Stretched, Duration)]) -> HashMap<u32, Duration> {
    let mut ids: Vec<u32> = equations
        .iter()
        .flat_map(|(terms, _)| terms.terms().iter().map(|&(id, _)| id))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let cols = ids.len();
    let mut rows: Vec<(Vec<f64>, [f64; 2])> = equations
        .iter()
        .map(|(terms, rhs)| {
            let mut row = vec![0.0; cols];
            for &(id, coeff) in terms.terms() {
                row[ids.binary_search(&id).expect("collected above")] += coeff;
            }
            (row, [rhs.ns, rhs.dt])
        })
        .collect();
    let scale = rows
        .iter()
        .flat_map(|(_, rhs)| rhs.iter().map(|v| v.abs()))
        .fold(1.0, f64::max);
    let eps = 1e-9;
    let mut pivots = Vec::new();
    let mut top = 0;
    for col in 0..cols {
        let Some(best) = (top..rows.len())
            .max_by(|&a, &b| rows[a].0[col].abs().total_cmp(&rows[b].0[col].abs()))
        else {
            break;
        };
        if rows[best].0[col].abs() < eps {
            continue;
        }
        rows.swap(top, best);
        let pivot = rows[top].0[col];
        for v in &mut rows[top].0 {
            *v /= pivot;
        }
        for v in &mut rows[top].1 {
            *v /= pivot;
        }
        let pivot_row = rows[top].clone();
        let (head, rest) = rows.split_at_mut(top);
        for row in head.iter_mut().chain(rest[1..].iter_mut()) {
            let factor = row.0[col];
            if factor != 0.0 {
                for (v, p) in row.0.iter_mut().zip(&pivot_row.0) {
                    *v -= factor * p;
                }
                for (v, p) in row.1.iter_mut().zip(&pivot_row.1) {
                    *v -= factor * p;
                }
            }
        }
        pivots.push(col);
        top += 1;
    }
    if rows[top..]
        .iter()
        .any(|(_, rhs)| rhs.iter().any(|v| v.abs() > eps * scale))
    {
        return HashMap::new();
    }
    let mut resolved = HashMap::new();
    for (row, &col) in rows.iter().zip(&pivots) {
        let free = row
            .0
            .iter()
            .enumerate()
            .any(|(c, v)| c != col && !pivots.contains(&c) && v.abs() >= eps);
        if free {
            continue;
        }
        let [ns, dt] = row.1;
        if ns < -eps * scale || dt < -eps * scale {
            return HashMap::new();
        }
        resolved.insert(
            ids[col],
            Duration {
                ns: ns.max(0.0),
                dt: dt.max(0.0),
            },
        );
    }
    resolved
}

impl<'a> Parser<'a> {
    /// `expr` with each `durationof` replaced by the length of its block.
    pub(super) fn with_durationof<'e, 'x>(
        &self,
        expr: &'e Expr<'x>,
        line: usize,
    ) -> Result<Cow<'e, Expr<'x>>> {
        if !expr.has_durationof() {
            return Ok(Cow::Borrowed(expr));
        }
        replace_nodes(expr, &|node| match node {
            Expr::DurationOf(tokens) => {
                let block = syntax::braced_block(&mut Stream::new(tokens))?;
                Ok(Some(Expr::Duration(self.durationof(&block, line)?)))
            }
            _ => Ok(None),
        })
        .map(Cow::Owned)
    }

    /// The length of `block` when every gate takes no time: the longest qubit
    /// timeline its delays and fixed-length boxes build. The block does not run.
    fn durationof(&self, block: &Block<'_>, line: usize) -> Result<Duration> {
        let mut schedule = Schedule::default();
        let outcome = match self.schedule(block, &mut schedule)? {
            Ok(()) => schedule.finish(None, line)?.map(|(_, length)| length.base),
            Err(why) => Err(why),
        };
        outcome.map_err(|why| PrismError::UnsupportedConstruct {
            construct: match why {
                Unscheduled::Statement => {
                    "`durationof` over a block holding anything but gate calls, `measure`, \
                     `reset`, `barrier`, `delay`, `nop` and `box`"
                }
                Unscheduled::Mixed => {
                    "`durationof` over delays mixing `dt` with SI units, whose order depends \
                     on the backend's sample period"
                }
                Unscheduled::Backtrack => "`durationof` over a stretch that backtracks",
            }
            .into(),
            line,
        })
    }

    /// After a `box[length]` body has run, check its delays fit and fix the
    /// stretches its alignment determines, substituting them into every duration
    /// in scope. A body the schedule cannot model is neither checked nor solved.
    pub(super) fn resolve_stretches(
        &mut self,
        body: &Block<'a>,
        length: Duration,
        line: usize,
    ) -> Result<()> {
        let mut schedule = Schedule::default();
        if !matches!(self.schedule(body, &mut schedule), Ok(Ok(())))
            || schedule.finish(Some(length), line)?.is_err()
            || schedule.equations.is_empty()
        {
            return Ok(());
        }
        let resolved = solve(&schedule.equations);
        if resolved.is_empty() {
            return Ok(());
        }
        for value in self.durations.values_mut() {
            if let Timed::Stretch(form) = *value {
                *value = form.substitute(&resolved);
            }
        }
        if self
            .classical
            .values()
            .any(|decl| matches!(decl.value, Some(ConstValue::Duration(Timed::Stretch(_)))))
        {
            for decl in Rc::make_mut(&mut self.classical).values_mut() {
                if let Some(ConstValue::Duration(Timed::Stretch(form))) = decl.value {
                    decl.value = Some(ConstValue::Duration(form.substitute(&resolved)));
                }
            }
        }
        Ok(())
    }

    fn schedule(
        &self,
        block: &Block<'_>,
        schedule: &mut Schedule,
    ) -> Result<std::result::Result<(), Unscheduled>> {
        for stmt in block {
            let line = stmt.line;
            match &stmt.kind {
                StmtKind::Empty | StmtKind::Pragma(_) => {}
                StmtKind::Call { name, operands, .. } => {
                    if self.def_defs.contains_key(*name) {
                        return Ok(Err(Unscheduled::Statement));
                    }
                    let lists = operands
                        .iter()
                        .map(|operand| self.qubits_of(operand))
                        .collect::<Result<Vec<_>>>()?;
                    let width = lists.iter().map(SmallVec::len).max().unwrap_or(0);
                    for at in 0..width {
                        let qubits: SmallVec<[usize; 4]> = lists
                            .iter()
                            .map(|list| if list.len() == 1 { list[0] } else { list[at] })
                            .collect();
                        if let Err(why) = schedule.align(&qubits, None, line)? {
                            return Ok(Err(why));
                        }
                    }
                }
                StmtKind::Measure(measure) => {
                    for qubit in self.qubits_of(&measure.source)? {
                        schedule.touch(qubit);
                    }
                }
                StmtKind::Reset { targets } | StmtKind::Nop { targets } => {
                    for target in targets {
                        for qubit in self.qubits_of(target)? {
                            schedule.touch(qubit);
                        }
                    }
                }
                StmtKind::Barrier { targets } => {
                    let qubits = self.timing_qubits(targets)?;
                    if let Err(why) = schedule.align(&qubits, None, line)? {
                        return Ok(Err(why));
                    }
                }
                StmtKind::Delay { duration, targets } => {
                    self.span_of(duration, "delay", line)?;
                    let length = self
                        .duration_of(duration, line)?
                        .stretched()
                        .expect("duration_of rejects numbers");
                    let qubits = self.timing_qubits(targets)?;
                    if let Err(why) = schedule.align(&qubits, None, line)? {
                        return Ok(Err(why));
                    }
                    for qubit in qubits {
                        let at = schedule.at[&qubit];
                        schedule.at.insert(qubit, at.plus(length, 1.0, line)?);
                    }
                }
                StmtKind::Box { duration, body } => {
                    let forced = match duration {
                        Some(duration) => match self.span_of(duration, "box", line)? {
                            Some(length) => Some(length),
                            None => return Ok(Err(Unscheduled::Statement)),
                        },
                        None => None,
                    };
                    let mut inner = Schedule::default();
                    if let Err(why) = self.schedule(body, &mut inner)? {
                        return Ok(Err(why));
                    }
                    let (used, length) = match inner.finish(forced, line)? {
                        Ok(done) => done,
                        Err(why) => return Ok(Err(why)),
                    };
                    schedule.equations.extend(inner.equations);
                    let start = match schedule.align(&used, None, line)? {
                        Ok(start) => start,
                        Err(why) => return Ok(Err(why)),
                    };
                    let end = start.plus(length, 1.0, line)?;
                    for qubit in used {
                        schedule.at.insert(qubit, end);
                    }
                }
                _ => return Ok(Err(Unscheduled::Statement)),
            }
        }
        Ok(Ok(()))
    }

    /// The qubits a `barrier` or `delay` spans: those named, or every qubit.
    fn timing_qubits(&self, targets: &[Operand]) -> Result<Vec<usize>> {
        if targets.is_empty() {
            return Ok((0..self.total_qubits).collect());
        }
        let mut qubits = Vec::new();
        for target in targets {
            qubits.extend(self.qubits_of(target)?);
        }
        Ok(qubits)
    }
}
