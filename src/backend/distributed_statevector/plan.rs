//! Lookahead relabel planning: windows of the instruction stream whose
//! non-diagonal targets fit the local positions, relabeled in one batch with
//! eviction by furthest next use.

use super::{DistributedStatevectorBackend, required_local_qubits};
use crate::backend::Backend;
use crate::circuit::{Instruction, SmallVec};
use crate::error::Result;
use crate::gates::Gate;

/// Next use beyond what the scan can see: past the horizon, past the end of
/// the slice, or behind a construct the scan cannot alias through (a region
/// body, a conditional SWAP).
const NEVER: usize = usize::MAX;

/// Instructions the next-use scan reads past a window end before it treats
/// every unresolved candidate as equally far away. Bounds the planning cost of
/// a window to `HORIZON` instructions.
const HORIZON: usize = 1 << 12;

/// Gate and targets of an instruction the planner sees through, with SWAPs
/// separated because they permute the map without needing locality.
enum Step<'a> {
    Gate(&'a Gate, &'a [usize]),
    Swap(usize, usize),
    Skip,
    Opaque,
}

fn step(instruction: &Instruction) -> Step<'_> {
    match instruction {
        Instruction::Gate { gate, targets } => match gate {
            Gate::Swap => Step::Swap(targets[0], targets[1]),
            _ => Step::Gate(gate, targets),
        },
        Instruction::Conditional { gate, targets, .. } => match gate {
            Gate::Swap => Step::Opaque,
            _ => Step::Gate(gate, targets),
        },
        Instruction::Measure { .. } | Instruction::Reset { .. } | Instruction::Barrier { .. } => {
            Step::Skip
        }
        Instruction::Region(_) => Step::Opaque,
    }
}

impl DistributedStatevectorBackend {
    /// Apply `instructions` window by window: each window's required qubits
    /// are relabeled into local positions before its first gate, so the gates
    /// inside dispatch locally with no further relabel.
    pub(super) fn apply_planned(&mut self, instructions: &[Instruction]) -> Result<()> {
        let mut start = 0;
        while start < instructions.len() {
            let end = self.plan_window(instructions, start);
            for instruction in &instructions[start..end] {
                self.apply(instruction)?;
            }
            start = end;
        }
        Ok(())
    }

    /// Delimit the window at `start`, relabel its required qubits into local
    /// positions, and return the exclusive window end (always past `start`).
    ///
    /// The window extends while the union of required qubits fits the local
    /// positions and splits at the first instruction that overflows. A single
    /// instruction that overflows on its own forms a window of one and is not
    /// relabeled for: however the map is arranged, the same number of its
    /// qubits stay global, so a relabel would move amplitudes without sparing
    /// an exchange. The per-gate path applies it, relabeling only while an
    /// unreferenced victim exists. A region or conditional SWAP also ends the
    /// window: a region body plans itself when taken. SWAPs inside the window are followed through an
    /// alias map, so a requirement names the qubit that holds the position at
    /// window start. Victims are the local qubits outside the window whose
    /// next required use is furthest away; ties go to the lowest position.
    fn plan_window(&mut self, instructions: &[Instruction], start: usize) -> usize {
        let local = self.local_qubits();
        let mut alias: Vec<usize> = (0..self.num_qubits).collect();
        let mut required: SmallVec<[usize; 16]> = SmallVec::new();
        let mut fresh: SmallVec<[usize; 8]> = SmallVec::new();
        let mut end = start;
        while end < instructions.len() {
            let (gate, targets) = match step(&instructions[end]) {
                Step::Gate(gate, targets) => (gate, targets),
                Step::Swap(a, b) => {
                    alias.swap(a, b);
                    end += 1;
                    continue;
                }
                Step::Skip => {
                    end += 1;
                    continue;
                }
                Step::Opaque => {
                    if end == start {
                        end += 1;
                    }
                    break;
                }
            };
            fresh.clear();
            for &q in &required_local_qubits(gate, targets) {
                let a = alias[q];
                if !required.contains(&a) && !fresh.contains(&a) {
                    fresh.push(a);
                }
            }
            if required.len() + fresh.len() > local {
                if end == start {
                    return start + 1;
                }
                break;
            }
            required.extend(fresh.iter().copied());
            end += 1;
        }

        let incoming: SmallVec<[usize; 8]> = required
            .iter()
            .copied()
            .filter(|&q| self.qubit_map[q] >= local)
            .collect();
        if incoming.is_empty() {
            return end;
        }
        let mut candidates: SmallVec<[(usize, usize); 16]> = (0..local)
            .filter(|&pos| !required.contains(&self.phys_map[pos]))
            .map(|pos| (pos, NEVER))
            .collect();
        self.resolve_next_uses(
            instructions,
            end,
            &mut alias,
            &mut candidates,
            incoming.len(),
        );
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (&q, &(victim, _)) in incoming.iter().zip(candidates.iter()) {
            let global_pos = self.qubit_map[q];
            self.relabel_swap(victim, global_pos);
        }
        end
    }

    /// Fill each candidate's next required use, scanning from `from` with the
    /// window's alias map. Stops once at most `victims` candidates are
    /// unresolved: those are then the furthest, whatever the rest of the
    /// stream holds.
    fn resolve_next_uses(
        &self,
        instructions: &[Instruction],
        from: usize,
        alias: &mut [usize],
        candidates: &mut [(usize, usize)],
        victims: usize,
    ) {
        let mut unresolved = candidates.len();
        let horizon = from.saturating_add(HORIZON).min(instructions.len());
        for (t, instruction) in instructions[from..horizon].iter().enumerate() {
            if unresolved <= victims {
                break;
            }
            let (gate, targets) = match step(instruction) {
                Step::Gate(gate, targets) => (gate, targets),
                Step::Swap(a, b) => {
                    alias.swap(a, b);
                    continue;
                }
                Step::Skip => continue,
                Step::Opaque => break,
            };
            for &q in &required_local_qubits(gate, targets) {
                let a = alias[q];
                if let Some(c) = candidates
                    .iter_mut()
                    .find(|c| c.1 == NEVER && self.phys_map[c.0] == a)
                {
                    c.1 = from + t;
                    unresolved -= 1;
                }
            }
        }
    }
}
