//! Noiseless mid-circuit shots on the host statevector by outcome branching:
//! each measurement splits the shots binomially and evolves every outcome once.

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use super::compiled::rng::{Xoshiro256PlusPlus, binomial_sample};
use super::{ShotsResult, apply_recording_saves, backend_metadata, mix_seed, read_save};
use crate::backend::statevector::StatevectorBackend;
use crate::backend::{Backend, max_statevector_qubits};
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;

/// Stream of the run seed whose generator draws the root's outcome splits.
const SPLIT_STREAM: u64 = u64::MAX;
/// Stream of the run seed whose generator shuffles the finished rows.
const SHUFFLE_STREAM: u64 = u64::MAX - 1;

/// Width from which the two branches of a split may run on separate workers.
/// Below it a branch evolves in about the time fork-join takes to dispatch it.
#[cfg(feature = "parallel")]
pub(super) const MIN_QUBITS_FOR_PARALLEL_SPLITS: usize = 10;

/// Splits that may hand their two branches to separate workers, counted from
/// the root. Past it the walk stays on the worker it reached.
#[cfg(feature = "parallel")]
const MAX_PARALLEL_SPLIT_DEPTH: usize = 12;

/// Outcome record and the number of shots that ended on it.
type Leaf = (Vec<bool>, usize);

/// How [`branch_shots`] may spend memory and threads.
pub(super) struct BranchLimits {
    /// States of the circuit's width the memory budget holds at once.
    pub(super) max_live_states: usize,
    /// Whether the two branches of a split may run on separate workers.
    #[cfg(feature = "parallel")]
    pub(super) parallel: bool,
}

impl BranchLimits {
    pub(super) fn for_width(num_qubits: usize, parallel: bool) -> Self {
        let headroom = max_statevector_qubits().saturating_sub(num_qubits);
        let max_live_states = u32::try_from(headroom)
            .ok()
            .and_then(|shift| 1usize.checked_shl(shift))
            .unwrap_or(usize::MAX);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        Self {
            max_live_states,
            #[cfg(feature = "parallel")]
            parallel,
        }
    }
}

/// Run `num_shots` shots of `fused` from `|0...0>` on the host statevector.
///
/// At a measurement, or at a reset that a later measurement can observe, the
/// shots still on a branch split by an exact Binomial(shots, P(1)) draw and
/// each non-empty outcome evolves once, so a circuit costs one evolution per
/// distinct outcome history rather than one per shot. Draws come from a
/// generator seeded by the run seed and the branch's outcome path, never by
/// the worker, so the shots do not depend on thread count.
///
/// The walk recurses into the branch with fewer shots and continues on the
/// other, cloning a state only at a split, so a path holds at most
/// `log2(num_shots) + 1` states plus one per parallel split above it. A split
/// that would pass
/// [`BranchLimits::max_live_states`] runs its shots one at a time from the
/// branch state instead. Instructions after the last one that writes a
/// classical bit or reads a save are never applied: nothing they do reaches a
/// result.
///
/// Branching groups shots by outcome, so the rows are shuffled with a seeded
/// Fisher-Yates pass when `shuffle` is set; a caller that only counts them
/// skips it.
pub(super) fn branch_shots(
    fused: &Circuit,
    num_shots: usize,
    seed: u64,
    limits: &BranchLimits,
    shuffle: bool,
) -> Result<ShotsResult> {
    let mut root = StatevectorBackend::new(seed);
    let metadata = backend_metadata(&root);
    let bits = fused.num_classical_bits;
    if num_shots == 0 {
        return Ok(ShotsResult::from_shots(Vec::new(), bits).with_metadata(metadata));
    }
    root.init(fused.num_qubits, bits)?;

    let live_end = fused
        .instructions
        .iter()
        .rposition(needs_state)
        .map_or(0, |last| last + 1);
    let mut cursor = Cursor::default();
    cursor.enter(&fused.instructions[..live_end]);

    let root = Branch {
        sv: root,
        shots: num_shots,
        rng: stream_rng(seed, SPLIT_STREAM),
    };
    let mut leaves = Vec::new();
    Walk { limits }.run(root, cursor, 1, 0, &mut leaves)?;

    // Traversal order depends on which branches ran in parallel, so the rows
    // are laid out in record order; equal records are interchangeable.
    leaves.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut shots = Vec::with_capacity(num_shots);
    for (record, count) in leaves {
        shots.extend(std::iter::repeat_n(record, count));
    }
    if shuffle {
        let mut rng = stream_rng(seed, SHUFFLE_STREAM);
        for i in (1..shots.len()).rev() {
            shots.swap(i, rng.random_range(0..=i));
        }
    }
    Ok(ShotsResult::from_shots(shots, bits).with_metadata(metadata))
}

fn stream_rng(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream);
    rng
}

/// Whether `inst` needs the quantum state to produce what it does: a
/// measurement, a save, or a region holding either.
fn needs_state(inst: &Instruction) -> bool {
    match inst {
        Instruction::Measure { .. } | Instruction::Save { .. } => true,
        Instruction::Region(region) => region.body().iter().any(needs_state),
        _ => false,
    }
}

fn draw_ones(rng: &mut ChaCha8Rng, shots: usize, prob_one: f64) -> usize {
    let mut fast = Xoshiro256PlusPlus::from_chacha(rng);
    binomial_sample(&mut fast, shots, prob_one)
}

/// Position in the instruction stream: the top-level list, then the body of
/// each region entered and not yet finished.
#[derive(Clone, Default)]
struct Cursor<'a> {
    frames: SmallVec<[(&'a [Instruction], usize); 2]>,
}

impl<'a> Cursor<'a> {
    fn enter(&mut self, body: &'a [Instruction]) {
        self.frames.push((body, 0));
    }

    fn next(&mut self) -> Option<&'a Instruction> {
        while let Some((body, idx)) = self.frames.last_mut() {
            let body: &'a [Instruction] = body;
            if let Some(inst) = body.get(*idx) {
                *idx += 1;
                return Some(inst);
            }
            self.frames.pop();
        }
        None
    }

    /// Step back over the instruction [`Self::next`] just returned.
    fn rewind(&mut self) {
        if let Some((_, idx)) = self.frames.last_mut() {
            *idx -= 1;
        }
    }

    /// Whether an instruction still ahead needs the state. The top-level list
    /// ends on such an instruction, so anything left in it does.
    fn state_ahead(&self) -> bool {
        self.frames.iter().enumerate().any(|(depth, &(body, idx))| {
            if depth == 0 {
                idx < body.len()
            } else {
                body[idx..].iter().any(needs_state)
            }
        })
    }
}

/// One branch of the walk: its state, the shots that followed it, and the
/// generator its outcome splits draw from.
struct Branch {
    sv: StatevectorBackend,
    shots: usize,
    rng: ChaCha8Rng,
}

struct Walk<'l> {
    limits: &'l BranchLimits,
}

impl Walk<'_> {
    /// Carry `branch` from `cursor` to the end, pushing one leaf per outcome
    /// history. `live` counts the states held on this path, the branch's own
    /// included, and `depth` the splits above it.
    fn run(
        &self,
        mut branch: Branch,
        mut cursor: Cursor<'_>,
        live: usize,
        depth: usize,
        leaves: &mut Vec<Leaf>,
    ) -> Result<()> {
        #[cfg(not(feature = "parallel"))]
        let _ = depth;
        while let Some(inst) = cursor.next() {
            let sv = &mut branch.sv;
            let (qubit, classical_bit) = match inst {
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => (*qubit, Some(*classical_bit)),
                Instruction::Reset { qubit } => {
                    if !cursor.state_ahead() {
                        continue;
                    }
                    (*qubit, None)
                }
                Instruction::Region(region) => {
                    if region.condition().evaluate(&sv.classical_bits) {
                        cursor.enter(region.body());
                    }
                    continue;
                }
                Instruction::Save { spec, label, .. } => {
                    read_save(sv, *spec, label)?;
                    continue;
                }
                _ => {
                    sv.apply(inst)?;
                    continue;
                }
            };

            let prob_one = sv.qubit_probability(qubit)?.clamp(0.0, 1.0);
            let shots = branch.shots;
            let ones = draw_ones(&mut branch.rng, shots, prob_one);
            let state_ahead = cursor.state_ahead();
            if ones == 0 || ones == shots {
                let outcome = ones > 0;
                if let Some(bit) = classical_bit {
                    sv.classical_bits[bit] = outcome;
                }
                if state_ahead {
                    project(sv, qubit, classical_bit, outcome, prob_one);
                }
                continue;
            }

            if let (false, Some(bit)) = (state_ahead, classical_bit) {
                let mut record = std::mem::take(&mut sv.classical_bits);
                record[bit] = false;
                leaves.push((record.clone(), shots - ones));
                record[bit] = true;
                leaves.push((record, ones));
                return Ok(());
            }

            if live.saturating_add(2) > self.limits.max_live_states {
                cursor.rewind();
                return per_shot(&mut branch, &cursor, leaves);
            }

            let mut sv_one = StatevectorBackend::new(0);
            sv_one.copy_state_from(sv);
            if let Some(bit) = classical_bit {
                sv.classical_bits[bit] = false;
                sv_one.classical_bits[bit] = true;
            }
            project(sv, qubit, classical_bit, false, prob_one);
            project(&mut sv_one, qubit, classical_bit, true, prob_one);

            let seed_zero = branch.rng.next_u64();
            let seed_one = branch.rng.next_u64();
            let zero = Branch {
                sv: branch.sv,
                shots: shots - ones,
                rng: ChaCha8Rng::seed_from_u64(seed_zero),
            };
            let one = Branch {
                sv: sv_one,
                shots: ones,
                rng: ChaCha8Rng::seed_from_u64(seed_one),
            };
            let (small, large) = if one.shots <= zero.shots {
                (one, zero)
            } else {
                (zero, one)
            };

            #[cfg(feature = "parallel")]
            if self.limits.parallel && depth < MAX_PARALLEL_SPLIT_DEPTH {
                let small_cursor = cursor.clone();
                let mut small_leaves = Vec::new();
                let (small_run, large_run) = rayon::join(
                    || self.run(small, small_cursor, live + 1, depth + 1, &mut small_leaves),
                    || self.run(large, cursor, live + 1, depth + 1, leaves),
                );
                small_run?;
                large_run?;
                leaves.append(&mut small_leaves);
                return Ok(());
            }

            self.run(small, cursor.clone(), live + 1, depth + 1, leaves)?;
            branch = large;
        }
        leaves.push((branch.sv.classical_bits, branch.shots));
        Ok(())
    }
}

fn project(
    sv: &mut StatevectorBackend,
    qubit: usize,
    classical_bit: Option<usize>,
    outcome: bool,
    prob_one: f64,
) {
    match classical_bit {
        Some(_) => sv.collapse_qubit(qubit, outcome, prob_one),
        None => sv.reset_qubit_from(qubit, outcome, prob_one),
    }
}

/// Run each of the branch's shots on its own from its state and `cursor`, one
/// scratch state reused across them, for a split the memory budget cannot
/// hold.
fn per_shot(branch: &mut Branch, cursor: &Cursor<'_>, leaves: &mut Vec<Leaf>) -> Result<()> {
    let base = branch.rng.next_u64();
    let mut scratch = StatevectorBackend::new(base);
    for i in 0..branch.shots {
        scratch.copy_state_from(&branch.sv);
        scratch.rng = ChaCha8Rng::seed_from_u64(mix_seed(base, i));
        for &(body, idx) in cursor.frames.iter().rev() {
            apply_recording_saves(&mut scratch, &body[idx..])?;
        }
        leaves.push((scratch.classical_bits.clone(), 1));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackendKind;
    use crate::circuit::{ClassicalCondition, SaveSpec, guarded};
    use crate::gates::Gate;
    use crate::sim::run_shots_with;

    fn frequency(shots: &[Vec<bool>], bit: usize) -> f64 {
        shots.iter().filter(|s| s[bit]).count() as f64 / shots.len() as f64
    }

    fn assert_near(observed: f64, p: f64, shots: usize, label: &str) {
        let sigma = (p * (1.0 - p) / shots as f64).sqrt().max(1e-9);
        assert!(
            (observed - p).abs() < 5.0 * sigma,
            "{label}: observed {observed:.5}, expected {p:.5}"
        );
    }

    fn rotation_layers(c: &mut Circuit, n: usize, offset: f64) {
        for q in 0..n {
            c.add_gate(Gate::Ry(offset + 0.37 * q as f64), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..n {
            c.add_gate(Gate::Rx(offset + 0.21 * q as f64), &[q]);
        }
    }

    fn mid_then_last(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 2);
        rotation_layers(&mut c, n, 0.4);
        c.add_measure(2, 0);
        rotation_layers(&mut c, n, 1.1);
        c.add_measure(n - 1, 1);
        c
    }

    // Exact law of (mid, last) for `mid_then_last`, cell `mid + 2 * last`: the
    // prefix's Born probability for the mid measurement, then the suffix run
    // once per outcome on the collapsed state.
    fn exact_joint(n: usize) -> [f64; 4] {
        let mut prefix = Circuit::new(n, 0);
        rotation_layers(&mut prefix, n, 0.4);
        let mut suffix = Circuit::new(n, 0);
        rotation_layers(&mut suffix, n, 1.1);

        let mut sv = StatevectorBackend::new(0);
        sv.init(n, 0).unwrap();
        sv.apply_instructions(&prefix.instructions).unwrap();
        let p_mid = sv.qubit_probability(2).unwrap();
        let mut joint = [0.0; 4];
        for mid in [false, true] {
            let mut branch = StatevectorBackend::new(0);
            branch.copy_state_from(&sv);
            branch.collapse_qubit(2, mid, p_mid);
            branch.apply_instructions(&suffix.instructions).unwrap();
            let p_last = branch.qubit_probability(n - 1).unwrap();
            let p_branch = if mid { p_mid } else { 1.0 - p_mid };
            joint[usize::from(mid)] = p_branch * (1.0 - p_last);
            joint[2 + usize::from(mid)] = p_branch * p_last;
        }
        joint
    }

    // Pearson statistic over the four cells; 16.27 is the 0.001 critical value
    // at three degrees of freedom.
    fn assert_joint_law(shots: &[Vec<bool>], expected: [f64; 4]) {
        let mut counts = [0usize; 4];
        for s in shots {
            counts[usize::from(s[0]) + 2 * usize::from(s[1])] += 1;
        }
        let total = shots.len() as f64;
        let chi: f64 = counts
            .iter()
            .zip(expected)
            .map(|(&c, p)| (c as f64 - p * total).powi(2) / (p * total))
            .sum();
        assert!(
            chi < 16.27,
            "chi-square {chi:.2}: {counts:?} against {expected:?}"
        );
    }

    #[test]
    fn mid_circuit_outcomes_follow_the_born_rule() {
        let n = 8;
        let shots = run_shots_with(BackendKind::Statevector, &mid_then_last(n), 20_000, 42)
            .unwrap()
            .shots;
        assert_joint_law(&shots, exact_joint(n));
    }

    #[test]
    fn a_forced_per_shot_fallback_keeps_the_born_rule() {
        let n = 6;
        let circuit = mid_then_last(n);
        for max_live_states in [1, 2, 3] {
            let limits = BranchLimits {
                max_live_states,
                #[cfg(feature = "parallel")]
                parallel: false,
            };
            let result = branch_shots(&circuit, 8_000, 42, &limits, true).unwrap();
            assert_eq!(result.shots.len(), 8_000);
            assert_joint_law(&result.shots, exact_joint(n));
        }
    }

    #[test]
    fn rows_are_shuffled_and_count_like_the_unshuffled_walk() {
        let circuit = mid_then_last(6);
        let limits = BranchLimits::for_width(6, false);
        let shuffled = branch_shots(&circuit, 2_000, 42, &limits, true).unwrap();
        let ordered = branch_shots(&circuit, 2_000, 42, &limits, false).unwrap();
        assert_eq!(shuffled.counts(), ordered.counts());
        assert_ne!(shuffled.shots, ordered.shots);
        let mut sorted = ordered.shots.clone();
        sorted.sort();
        assert_eq!(ordered.shots, sorted, "unshuffled rows run in record order");
    }

    #[test]
    fn teleportation_delivers_the_state_on_every_branch() {
        let theta = 0.8;
        let mut c = Circuit::new(3, 3);
        c.add_gate(Gate::Ry(theta), &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        c.instructions.push(Instruction::Conditional {
            condition: ClassicalCondition::BitIsOne(1),
            gate: Gate::X,
            targets: smallvec::smallvec![2],
        });
        c.instructions.push(Instruction::Conditional {
            condition: ClassicalCondition::BitIsOne(0),
            gate: Gate::Z,
            targets: smallvec::smallvec![2],
        });
        c.add_gate(Gate::Ry(-theta), &[2]);
        c.add_measure(2, 2);

        let shots = run_shots_with(BackendKind::Statevector, &c, 8_000, 42)
            .unwrap()
            .shots;
        assert!(shots.iter().all(|s| !s[2]), "teleported qubit read one");
        assert_near(frequency(&shots, 0), 0.5, shots.len(), "bit 0");
        assert_near(frequency(&shots, 1), 0.5, shots.len(), "bit 1");
    }

    // Repeat until success: a second attempt, reset and measured inside a
    // region, runs only on the shots whose first attempt read zero.
    #[test]
    fn repeat_until_success_regions_branch_inside_the_body() {
        let p = (0.6f64 / 2.0).sin().powi(2);
        let mut c = Circuit::new(2, 3);
        c.add_gate(Gate::Ry(0.6), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        let mut retry = Circuit::new(2, 3);
        retry.add_reset(0);
        retry.add_reset(1);
        retry.add_gate(Gate::Ry(0.6), &[0]);
        retry.add_gate(Gate::Cx, &[0, 1]);
        retry.add_measure(0, 1);
        c.instructions
            .push(guarded(ClassicalCondition::BitIsZero(0), retry.instructions).unwrap());
        c.add_measure(1, 2);

        let shots = run_shots_with(BackendKind::Statevector, &c, 20_000, 42)
            .unwrap()
            .shots;
        assert_near(frequency(&shots, 0), p, shots.len(), "first attempt");
        let retried: Vec<Vec<bool>> = shots.iter().filter(|s| !s[0]).cloned().collect();
        assert_near(frequency(&retried, 1), p, retried.len(), "second attempt");
        assert!(shots.iter().filter(|s| s[0]).all(|s| !s[1]));
        // Qubit 1 copies whichever attempt ran last.
        assert!(shots.iter().all(|s| s[2] == (s[0] || s[1])));
    }

    #[test]
    fn a_reset_on_an_entangled_qubit_splits_its_partner() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_reset(0);
        c.add_gate(Gate::Cx, &[1, 0]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        let limits = BranchLimits::for_width(2, false);
        let shots = branch_shots(&c, 4_000, 42, &limits, true).unwrap().shots;
        assert!(shots.iter().all(|s| s[0] == s[1]));
        assert_near(frequency(&shots, 1), 0.5, shots.len(), "partner");
    }

    // A million shots over fifty measurements of one qubit: the first fixes the
    // rest, so the walk keeps two branches however many shots ride on them.
    #[test]
    fn many_shots_over_many_measurements_stay_two_branches() {
        let mut c = Circuit::new(3, 50);
        c.add_gate(Gate::Ry(1.1), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        for bit in 0..50 {
            c.add_gate(Gate::Rz(0.3), &[2]);
            c.add_measure(0, bit);
        }
        let result = run_shots_with(BackendKind::Statevector, &c, 1_000_000, 42).unwrap();
        let counts = result.counts();
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.values().sum::<u64>(), 1_000_000);
        let p = (1.1f64 / 2.0).sin().powi(2);
        assert_near(
            frequency(&result.shots, 49),
            p,
            1_000_000,
            "last measurement",
        );
    }

    #[test]
    fn the_tail_after_the_last_measurement_is_skipped_but_a_save_is_not() {
        let mut c = Circuit::new(2, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        c.add_gate(Gate::H, &[1]);
        let limits = BranchLimits::for_width(2, false);
        let result = branch_shots(&c, 1_000, 42, &limits, true).unwrap();
        assert_near(frequency(&result.shots, 0), 0.5, 1_000, "bit 0");

        assert!(!needs_state(&c.instructions[2]));
        c.add_save(SaveSpec::Probabilities, "probs");
        assert!(needs_state(&c.instructions[3]));
        let result = branch_shots(&c, 1_000, 42, &limits, true).unwrap();
        assert_near(frequency(&result.shots, 0), 0.5, 1_000, "bit 0 with a save");
    }
}
