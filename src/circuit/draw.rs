use std::fmt;

use crate::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec};
use crate::gates::Gate;

const SUMMARY_QUBIT_THRESHOLD: usize = 64;
const SUMMARY_MOMENT_THRESHOLD: usize = 500;
const DEFAULT_FOLD_WIDTH: usize = 120;

pub struct TextOptions {
    pub fold_width: usize,
    pub show_idle_wires: bool,
    pub show_barriers: bool,
    pub max_qubits: Option<usize>,
    pub max_moments: Option<usize>,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            fold_width: DEFAULT_FOLD_WIDTH,
            show_idle_wires: true,
            show_barriers: true,
            max_qubits: None,
            max_moments: None,
        }
    }
}

pub(super) struct PlacedOp {
    pub(super) label: String,
    pub(super) qubits: SmallVec<[usize; 4]>,
    pub(super) kind: OpKind,
}

pub(super) enum OpKind {
    Single,
    Controlled { controls: Vec<usize>, target: usize },
    TwoQubit,
    Swap,
    Barrier,
    Measure { cbit: usize },
    Conditional { cbit_label: String },
    MultiFused,
}

fn classify_op(gate: &Gate, targets: &[usize]) -> (String, OpKind) {
    let label = gate.to_string();
    let kind = match gate {
        Gate::Cx => OpKind::Controlled {
            controls: vec![targets[0]],
            target: targets[1],
        },
        Gate::Cu(_) => OpKind::Controlled {
            controls: vec![targets[0]],
            target: targets[1],
        },
        Gate::Mcu(data) => {
            let nc = data.num_controls as usize;
            OpKind::Controlled {
                controls: targets[..nc].to_vec(),
                target: targets[nc],
            }
        }
        Gate::Cz => OpKind::Controlled {
            controls: vec![targets[0]],
            target: targets[1],
        },
        Gate::Swap => OpKind::Swap,
        Gate::Rzz(_)
        | Gate::Fused2q(_)
        | Gate::BatchRzz(_)
        | Gate::DiagonalBatch(_)
        | Gate::Multi2q(_) => OpKind::TwoQubit,
        Gate::BatchPhase(_) => OpKind::Controlled {
            controls: vec![targets[0]],
            target: *targets.last().unwrap(),
        },
        Gate::MultiFused(_) => OpKind::MultiFused,
        _ => OpKind::Single,
    };
    (label, kind)
}

pub(super) fn assign_moments(circuit: &Circuit) -> Vec<Vec<PlacedOp>> {
    let n = circuit.num_qubits;
    let mut qubit_depth = vec![0usize; n];
    let mut moments: Vec<Vec<PlacedOp>> = Vec::new();

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                let max_d = targets.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                let (label, kind) = classify_op(gate, targets);
                let op = PlacedOp {
                    label,
                    qubits: SmallVec::from_slice(targets),
                    kind,
                };
                if max_d >= moments.len() {
                    moments.resize_with(max_d + 1, Vec::new);
                }
                moments[max_d].push(op);
                for &q in targets.iter() {
                    qubit_depth[q] = max_d + 1;
                }
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                let d = qubit_depth[*qubit];
                let op = PlacedOp {
                    label: "M".into(),
                    qubits: smallvec::smallvec![*qubit],
                    kind: OpKind::Measure {
                        cbit: *classical_bit,
                    },
                };
                if d >= moments.len() {
                    moments.resize_with(d + 1, Vec::new);
                }
                moments[d].push(op);
                qubit_depth[*qubit] = d + 1;
            }
            Instruction::Barrier { qubits } => {
                let max_d = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                let op = PlacedOp {
                    label: String::new(),
                    qubits: SmallVec::from_slice(qubits),
                    kind: OpKind::Barrier,
                };
                if max_d >= moments.len() {
                    moments.resize_with(max_d + 1, Vec::new);
                }
                moments[max_d].push(op);
                for &q in qubits.iter() {
                    qubit_depth[q] = max_d;
                }
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                let max_d = targets.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
                let cbit_label = match condition {
                    ClassicalCondition::BitIsOne(b) => format!("c[{}]", b),
                    ClassicalCondition::RegisterEquals {
                        offset,
                        size,
                        value,
                    } => {
                        format!("c[{}..{}]=={}", offset, offset + size, value)
                    }
                };
                let op = PlacedOp {
                    label: gate.to_string(),
                    qubits: SmallVec::from_slice(targets),
                    kind: OpKind::Conditional { cbit_label },
                };
                if max_d >= moments.len() {
                    moments.resize_with(max_d + 1, Vec::new);
                }
                moments[max_d].push(op);
                for &q in targets.iter() {
                    qubit_depth[q] = max_d + 1;
                }
            }
        }
    }
    moments
}

const WIRE: char = '\u{2500}'; // ─
const VERT: char = '\u{2502}'; // │
const CTRL: char = '@';
const SWAP_X: char = '\u{00D7}'; // ×

fn qubit_label_width(n: usize) -> usize {
    if n <= 1 {
        return 4;
    }
    let digits = ((n - 1) as f64).log10().floor() as usize + 1;
    3 + digits // "q[" + digits + "]"
}

struct GridCell {
    content: String,
    is_vert_connector: bool,
}

impl GridCell {
    fn wire(width: usize) -> Self {
        Self {
            content: WIRE.to_string().repeat(width),
            is_vert_connector: false,
        }
    }

    fn gate(label: &str, width: usize) -> Self {
        let pad_total = width.saturating_sub(label.len());
        let pad_left = pad_total / 2;
        let pad_right = pad_total - pad_left;
        let content = format!(
            "{}{}{}",
            WIRE.to_string().repeat(pad_left),
            label,
            WIRE.to_string().repeat(pad_right),
        );
        Self {
            content,
            is_vert_connector: false,
        }
    }

    fn control(width: usize) -> Self {
        Self::gate(&CTRL.to_string(), width)
    }

    fn swap_marker(width: usize) -> Self {
        Self::gate(&SWAP_X.to_string(), width)
    }

    fn vert_connector(width: usize) -> Self {
        let pad_left = width.saturating_sub(1) / 2;
        let pad_right = width - pad_left - 1;
        let content = format!("{}{}{}", " ".repeat(pad_left), VERT, " ".repeat(pad_right),);
        Self {
            content,
            is_vert_connector: true,
        }
    }

    fn barrier(width: usize) -> Self {
        Self {
            content: "\u{250A}".to_string().repeat(width), // ┊
            is_vert_connector: false,
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn render_moments(moments: &[Vec<PlacedOp>], num_qubits: usize, opts: &TextOptions) -> Vec<String> {
    if moments.is_empty() || num_qubits == 0 {
        return Vec::new();
    }

    let max_moments = opts.max_moments.unwrap_or(moments.len()).min(moments.len());
    let moments = &moments[..max_moments];

    let label_w = qubit_label_width(num_qubits);

    let mut col_widths: Vec<usize> = Vec::with_capacity(moments.len());
    for moment in moments {
        let mut max_label = 1usize;
        for op in moment {
            if matches!(op.kind, OpKind::Barrier) {
                continue;
            }
            max_label = max_label.max(op.label.len());
        }
        col_widths.push(max_label + 2);
    }

    let mut active_qubits: Vec<bool> = vec![false; num_qubits];
    if !opts.show_idle_wires {
        for moment in moments {
            for op in moment {
                for &q in &op.qubits {
                    if q < num_qubits {
                        active_qubits[q] = true;
                    }
                }
            }
        }
    } else {
        active_qubits.fill(true);
    }

    let visible_qubits: Vec<usize> = (0..num_qubits).filter(|&q| active_qubits[q]).collect();
    if visible_qubits.is_empty() {
        return vec!["(empty circuit)".to_string()];
    }

    let max_vis_qubits = opts.max_qubits.unwrap_or(visible_qubits.len());
    let show_qubits = &visible_qubits[..max_vis_qubits.min(visible_qubits.len())];
    let elided = visible_qubits.len().saturating_sub(show_qubits.len());

    let qubit_to_row: Vec<Option<usize>> = {
        let mut map = vec![None; num_qubits];
        for (row, &q) in show_qubits.iter().enumerate() {
            map[q] = Some(row);
        }
        map
    };

    let num_rows = show_qubits.len() * 2 - 1;

    let mut grid: Vec<Vec<GridCell>> = Vec::with_capacity(num_rows);
    for r in 0..num_rows {
        let mut row = Vec::with_capacity(moments.len());
        for &w in &col_widths {
            if r % 2 == 0 {
                row.push(GridCell::wire(w));
            } else {
                row.push(GridCell {
                    content: " ".repeat(w),
                    is_vert_connector: false,
                });
            }
        }
        grid.push(row);
    }

    for (m_idx, moment) in moments.iter().enumerate() {
        let w = col_widths[m_idx];
        for op in moment {
            match &op.kind {
                OpKind::Single | OpKind::MultiFused => {
                    for &q in &op.qubits {
                        if let Some(row) = qubit_to_row.get(q).copied().flatten() {
                            grid[row * 2][m_idx] = GridCell::gate(&op.label, w);
                        }
                    }
                }
                OpKind::Controlled { controls, target } => {
                    for &c in controls {
                        if let Some(row) = qubit_to_row.get(c).copied().flatten() {
                            grid[row * 2][m_idx] = GridCell::control(w);
                        }
                    }
                    if let Some(row) = qubit_to_row.get(*target).copied().flatten() {
                        let tgt_label = match op.label.as_str() {
                            "CX" => "X",
                            "CZ" => "Z",
                            "CU" => "U",
                            other => {
                                if other.starts_with("MCU") {
                                    "U"
                                } else {
                                    &op.label
                                }
                            }
                        };
                        grid[row * 2][m_idx] = GridCell::gate(tgt_label, w);
                    }
                    let all_rows: Vec<usize> = controls
                        .iter()
                        .chain(std::iter::once(target))
                        .filter_map(|&q| qubit_to_row.get(q).copied().flatten())
                        .collect();
                    if all_rows.len() >= 2 {
                        let min_r = *all_rows.iter().min().unwrap();
                        let max_r = *all_rows.iter().max().unwrap();
                        for r in (min_r * 2 + 1)..=(max_r * 2 - 1) {
                            if r % 2 == 1 {
                                grid[r][m_idx] = GridCell::vert_connector(w);
                            } else {
                                let row_qubit_idx = r / 2;
                                if !all_rows.contains(&row_qubit_idx) {
                                    grid[r][m_idx] = GridCell::gate(&VERT.to_string(), w);
                                }
                            }
                        }
                    }
                }
                OpKind::TwoQubit => {
                    let rows: Vec<usize> = op
                        .qubits
                        .iter()
                        .filter_map(|&q| qubit_to_row.get(q).copied().flatten())
                        .collect();
                    if rows.len() >= 2 {
                        let min_r = *rows.iter().min().unwrap();
                        let max_r = *rows.iter().max().unwrap();
                        grid[min_r * 2][m_idx] = GridCell::gate(&op.label, w);
                        grid[max_r * 2][m_idx] = GridCell::gate(&op.label, w);
                        for r in (min_r * 2 + 1)..=(max_r * 2 - 1) {
                            if r % 2 == 1 {
                                grid[r][m_idx] = GridCell::vert_connector(w);
                            } else {
                                let row_q = r / 2;
                                if !rows.contains(&row_q) {
                                    grid[r][m_idx] = GridCell::gate(&VERT.to_string(), w);
                                }
                            }
                        }
                    } else if let Some(&row) = rows.first() {
                        grid[row * 2][m_idx] = GridCell::gate(&op.label, w);
                    }
                }
                OpKind::Swap => {
                    let rows: Vec<usize> = op
                        .qubits
                        .iter()
                        .filter_map(|&q| qubit_to_row.get(q).copied().flatten())
                        .collect();
                    if rows.len() == 2 {
                        let (min_r, max_r) = (rows[0].min(rows[1]), rows[0].max(rows[1]));
                        grid[min_r * 2][m_idx] = GridCell::swap_marker(w);
                        grid[max_r * 2][m_idx] = GridCell::swap_marker(w);
                        for r in (min_r * 2 + 1)..=(max_r * 2 - 1) {
                            if r % 2 == 1 {
                                grid[r][m_idx] = GridCell::vert_connector(w);
                            } else {
                                let row_q = r / 2;
                                if row_q != min_r && row_q != max_r {
                                    grid[r][m_idx] = GridCell::gate(&VERT.to_string(), w);
                                }
                            }
                        }
                    }
                }
                OpKind::Barrier => {
                    if opts.show_barriers {
                        for &q in &op.qubits {
                            if let Some(row) = qubit_to_row.get(q).copied().flatten() {
                                grid[row * 2][m_idx] = GridCell::barrier(w);
                            }
                        }
                    }
                }
                OpKind::Measure { cbit } => {
                    if let Some(row) = op
                        .qubits
                        .first()
                        .and_then(|&q| qubit_to_row.get(q).copied().flatten())
                    {
                        let label = format!("M{}", cbit);
                        grid[row * 2][m_idx] = GridCell::gate(&label, w);
                    }
                }
                OpKind::Conditional { cbit_label } => {
                    for &q in &op.qubits {
                        if let Some(row) = qubit_to_row.get(q).copied().flatten() {
                            let label = format!("{}?{}", cbit_label, op.label);
                            grid[row * 2][m_idx] = GridCell::gate(&label, w);
                        }
                    }
                }
            }
        }
    }

    let usable_width = opts.fold_width.saturating_sub(label_w + 2);

    let mut sections: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    let mut cur_width = 0;
    for (i, &w) in col_widths.iter().enumerate() {
        if cur_width + w > usable_width && i > start {
            sections.push((start, i));
            start = i;
            cur_width = 0;
        }
        cur_width += w;
    }
    if start < col_widths.len() {
        sections.push((start, col_widths.len()));
    }

    let mut lines: Vec<String> = Vec::new();

    for (sec_idx, &(col_start, col_end)) in sections.iter().enumerate() {
        if sec_idx > 0 {
            let sec_width: usize =
                col_widths[col_start..col_end].iter().sum::<usize>() + label_w + 2;
            lines.push("\u{254C}".repeat(sec_width));
        }
        for (vis_row, &q) in show_qubits.iter().enumerate() {
            let label = format!("q[{}]", q);
            let padded = format!("{:>width$}: ", label, width = label_w);

            let wire_row = vis_row * 2;
            let mut line = padded;
            for col in col_start..col_end {
                line.push_str(&grid[wire_row][col].content);
            }
            let trimmed = line.trim_end();
            lines.push(trimmed.to_string());

            if vis_row < show_qubits.len() - 1 {
                let conn_row = vis_row * 2 + 1;
                let padding = " ".repeat(label_w + 2);
                let mut conn_line = padding;
                let mut has_content = false;
                for col in col_start..col_end {
                    let cell = &grid[conn_row][col];
                    if cell.is_vert_connector {
                        has_content = true;
                    }
                    conn_line.push_str(&cell.content);
                }
                if has_content {
                    let trimmed = conn_line.trim_end();
                    lines.push(trimmed.to_string());
                }
            }
        }
    }

    if elided > 0 {
        lines.push(format!("  ... and {} more qubits", elided));
    }

    if max_moments < moments.len() {
        lines.push(format!(
            "  ... truncated at moment {} of {}",
            max_moments,
            moments.len()
        ));
    }

    lines
}

fn collect_2q_edges(circuit: &Circuit) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for inst in &circuit.instructions {
        let targets: &[usize] = match inst {
            Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => targets,
            _ => continue,
        };
        if targets.len() == 2 {
            let (a, b) = (targets[0].min(targets[1]), targets[0].max(targets[1]));
            edges.push((a, b));
        }
    }
    edges
}

fn render_connectivity(lines: &mut Vec<String>, circuit: &Circuit) {
    let n = circuit.num_qubits;
    let edges = collect_2q_edges(circuit);
    if edges.is_empty() {
        lines.push("Connectivity: none (no 2q gates)".to_string());
        return;
    }

    let mut degree = vec![0usize; n];
    let mut unique_neighbors: Vec<std::collections::HashSet<usize>> =
        vec![std::collections::HashSet::new(); n];
    for &(a, b) in &edges {
        degree[a] += 1;
        degree[b] += 1;
        unique_neighbors[a].insert(b);
        unique_neighbors[b].insert(a);
    }

    let total_2q = edges.len();
    let unique_pairs: std::collections::HashSet<(usize, usize)> = edges.iter().copied().collect();
    let num_unique_pairs = unique_pairs.len();

    let max_degree_q = degree
        .iter()
        .enumerate()
        .max_by_key(|(_, &d)| d)
        .map(|(q, _)| q)
        .unwrap_or(0);
    let max_degree = degree[max_degree_q];

    let max_neighbors = unique_neighbors.iter().map(|s| s.len()).max().unwrap_or(0);

    let all_nn = unique_pairs.iter().all(|&(a, b)| b.saturating_sub(a) == 1);
    let max_possible = n * (n - 1) / 2;
    let topology = if all_nn && num_unique_pairs <= n {
        "nearest-neighbor"
    } else if num_unique_pairs == max_possible {
        "all-to-all"
    } else if max_neighbors <= 4 && n > 8 {
        "sparse"
    } else {
        "mixed"
    };

    lines.push(format!(
        "Connectivity: {} 2q gates across {} unique pairs ({})",
        total_2q, num_unique_pairs, topology,
    ));
    lines.push(format!(
        "  max degree: {} (q[{}], {} unique neighbors)",
        max_degree,
        max_degree_q,
        unique_neighbors[max_degree_q].len(),
    ));

    let bar_max = 30usize;
    let mut deg_dist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &d in &degree {
        if d > 0 {
            *deg_dist.entry(d).or_default() += 1;
        }
    }
    if deg_dist.len() > 1 {
        let mut sorted_degrees: Vec<(usize, usize)> = deg_dist.into_iter().collect();
        sorted_degrees.sort_by_key(|&(d, _)| d);
        let max_freq = sorted_degrees.iter().map(|(_, c)| *c).max().unwrap_or(1);
        lines.push("  degree distribution:".to_string());
        for (d, count) in &sorted_degrees {
            let bar_len = (*count * bar_max) / max_freq;
            lines.push(format!(
                "    {:>3}: {} {}",
                d,
                "\u{2588}".repeat(bar_len),
                count,
            ));
        }
    }
}

fn render_depth_profile(lines: &mut Vec<String>, moments: &[Vec<PlacedOp>]) {
    if moments.is_empty() {
        return;
    }
    let depth = moments.len();
    let num_buckets = 20.min(depth);
    let bucket_size = depth.div_ceil(num_buckets).max(1);
    let actual_buckets = depth.div_ceil(bucket_size);

    let mut bucket_counts = vec![0usize; actual_buckets];
    let mut bucket_2q = vec![0usize; actual_buckets];
    for (m_idx, moment) in moments.iter().enumerate() {
        let b = m_idx / bucket_size;
        for op in moment {
            bucket_counts[b] += 1;
            if op.qubits.len() >= 2 {
                bucket_2q[b] += 1;
            }
        }
    }

    let max_gates = *bucket_counts.iter().max().unwrap_or(&1).max(&1);
    let bar_max = 40usize;

    lines.push("Depth profile (gates per time slice):".to_string());
    let depth_w = ((depth - 1) as f64).log10().floor() as usize + 1;
    for (b, &count) in bucket_counts.iter().enumerate() {
        let start = b * bucket_size;
        let end = ((b + 1) * bucket_size).min(depth) - 1;
        let bar_1q = ((count - bucket_2q[b]) * bar_max) / max_gates;
        let bar_2q = (bucket_2q[b] * bar_max) / max_gates;
        lines.push(format!(
            "  {:>w$}-{:<w$}: {}{} {}",
            start,
            end,
            "\u{2588}".repeat(bar_1q),
            "\u{2593}".repeat(bar_2q),
            count,
            w = depth_w,
        ));
    }
    if bucket_2q.iter().any(|&c| c > 0) {
        lines.push("  (\u{2588} = 1q  \u{2593} = 2q)".to_string());
    }
}

const HEATMAP_CHARS: &[char] = &[' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];

fn render_heatmap(circuit: &Circuit, opts: &TextOptions) -> Vec<String> {
    let moments = assign_moments(circuit);
    if moments.is_empty() || circuit.num_qubits == 0 {
        return vec!["(empty circuit)".to_string()];
    }

    let n = circuit.num_qubits;
    let depth = moments.len();

    let max_cols = (opts.fold_width.saturating_sub(14)).max(20);
    let max_rows = 40.min(n);

    let col_bucket = depth.div_ceil(max_cols).max(1);
    let row_bucket = n.div_ceil(max_rows).max(1);
    let actual_cols = depth.div_ceil(col_bucket);
    let actual_rows = n.div_ceil(row_bucket);

    let mut grid = vec![vec![0usize; actual_cols]; actual_rows];
    for (m_idx, moment) in moments.iter().enumerate() {
        let col = m_idx / col_bucket;
        for op in moment {
            for &q in &op.qubits {
                if q < n {
                    let row = q / row_bucket;
                    grid[row][col] += 1;
                }
            }
        }
    }

    let max_density = grid
        .iter()
        .flat_map(|r| r.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);

    let mut lines = Vec::new();
    lines.push(format!(
        "Gate density heatmap ({} qubits x {} moments):",
        n, depth,
    ));

    let label_w = if row_bucket == 1 {
        qubit_label_width(n)
    } else {
        let w = ((n - 1) as f64).log10().floor() as usize + 1;
        w * 2 + 4
    };

    for (r, row) in grid.iter().enumerate() {
        let label = if row_bucket == 1 {
            format!("q[{}]", r)
        } else {
            let start = r * row_bucket;
            let end = ((r + 1) * row_bucket).min(n) - 1;
            format!("{}-{}", start, end)
        };

        let mut line = format!("{:>width$}: ", label, width = label_w);
        for &count in row {
            let level = if count == 0 {
                0
            } else {
                (count * 4 / max_density).clamp(1, 4)
            };
            line.push(HEATMAP_CHARS[level]);
        }
        lines.push(line);
    }

    let moment_axis = format!("{:>width$}  ", "", width = label_w);
    let end_label = format!("{}", depth - 1);
    let axis_line = format!(
        "{}0{}{}",
        moment_axis,
        " ".repeat(actual_cols.saturating_sub(1 + end_label.len())),
        end_label,
    );
    lines.push(axis_line);

    if max_density <= 4 {
        let mut legend = String::from("  Legend: ' '=0");
        for level in 1..=max_density {
            let ch = HEATMAP_CHARS[((level * 4) / max_density).clamp(1, 4)];
            legend.push_str(&format!("  {ch}={level}"));
        }
        lines.push(legend);
    } else {
        lines.push(format!(
            "  Legend: ' '=0  \u{2591}=1-{}  \u{2592}={}-{}  \u{2593}={}-{}  \u{2588}={}-{}",
            max_density / 4,
            max_density / 4 + 1,
            max_density / 2,
            max_density / 2 + 1,
            3 * max_density / 4,
            3 * max_density / 4 + 1,
            max_density,
        ));
    }

    lines
}

fn render_summary(circuit: &Circuit) -> Vec<String> {
    let mut lines = Vec::new();
    let moments = assign_moments(circuit);
    let depth = moments.len();
    lines.push(format!(
        "Circuit: {} qubits, {} gates, depth {}",
        circuit.num_qubits,
        circuit.gate_count(),
        depth,
    ));
    lines.push(String::new());

    let mut gate_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut measure_count = 0usize;
    let mut barrier_count = 0usize;
    let mut conditional_count = 0usize;

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, .. } => {
                *gate_counts.entry(gate.name()).or_default() += 1;
            }
            Instruction::Measure { .. } => measure_count += 1,
            Instruction::Barrier { .. } => barrier_count += 1,
            Instruction::Conditional { gate, .. } => {
                conditional_count += 1;
                *gate_counts.entry(gate.name()).or_default() += 1;
            }
        }
    }

    let mut sorted: Vec<(&&str, &usize)> = gate_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    lines.push("Gate counts:".to_string());
    let mut gate_line = String::from("  ");
    for (i, (name, count)) in sorted.iter().enumerate() {
        if i > 0 {
            gate_line.push_str("  ");
        }
        let entry = format!("{}: {}", name, count);
        if gate_line.len() + entry.len() > 100 {
            lines.push(gate_line);
            gate_line = format!("  {}", entry);
        } else {
            gate_line.push_str(&entry);
        }
    }
    if !gate_line.trim().is_empty() {
        lines.push(gate_line);
    }

    if measure_count > 0 {
        lines.push(format!("  measurements: {}", measure_count));
    }
    if barrier_count > 0 {
        lines.push(format!("  barriers: {}", barrier_count));
    }
    if conditional_count > 0 {
        lines.push(format!("  conditionals: {}", conditional_count));
    }

    lines.push(String::new());

    render_connectivity(&mut lines, circuit);
    lines.push(String::new());

    render_depth_profile(&mut lines, &moments);
    lines.push(String::new());

    let n = circuit.num_qubits;
    let mut qubit_gate_count = vec![0usize; n];
    for inst in &circuit.instructions {
        let targets: &[usize] = match inst {
            Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => targets,
            Instruction::Measure { qubit, .. } => std::slice::from_ref(qubit),
            Instruction::Barrier { .. } => continue,
        };
        for &q in targets {
            if q < n {
                qubit_gate_count[q] += 1;
            }
        }
    }

    let max_count = *qubit_gate_count.iter().max().unwrap_or(&1).max(&1);
    let bar_max = 40usize;

    if n <= 32 {
        lines.push("Qubit activity:".to_string());
        for (q, &c) in qubit_gate_count.iter().enumerate() {
            let bar_len = (c * bar_max) / max_count;
            lines.push(format!(
                "  q[{:>width$}]: {} {}",
                q,
                "\u{2588}".repeat(bar_len),
                c,
                width = ((n - 1) as f64).log10().floor() as usize + 1,
            ));
        }
    } else {
        lines.push("Qubit activity (bucketed):".to_string());
        let bucket_size = n.div_ceil(10).max(1);
        let mut bucket_start = 0;
        while bucket_start < n {
            let bucket_end = (bucket_start + bucket_size).min(n);
            let min_c = qubit_gate_count[bucket_start..bucket_end]
                .iter()
                .copied()
                .min()
                .unwrap_or(0);
            let max_c = qubit_gate_count[bucket_start..bucket_end]
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            let avg_c = qubit_gate_count[bucket_start..bucket_end]
                .iter()
                .sum::<usize>()
                / (bucket_end - bucket_start);
            let bar_len = (avg_c * bar_max) / max_count;
            lines.push(format!(
                "  q[{:>4}..{:<4}]: {} {}-{}",
                bucket_start,
                bucket_end - 1,
                "\u{2588}".repeat(bar_len),
                min_c,
                max_c,
            ));
            bucket_start = bucket_end;
        }
    }

    lines.push(String::new());

    let heatmap = render_heatmap(circuit, &TextOptions::default());
    lines.extend(heatmap);
    lines.push(String::new());

    let classification = if circuit.is_clifford_only() {
        "Clifford-only"
    } else if circuit.is_clifford_plus_t() {
        "Clifford+T"
    } else {
        "General"
    };
    lines.push(format!("Classification: {}", classification));

    if circuit.has_terminal_measurements_only() {
        lines.push(format!("Measurements: terminal-only ({})", measure_count));
    } else if measure_count > 0 {
        lines.push(format!("Measurements: mid-circuit ({})", measure_count));
    } else {
        lines.push("Measurements: none".to_string());
    }

    let components = circuit.independent_subsystems();
    if components.len() > 1 {
        let sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();
        let max_block = sizes.iter().max().unwrap_or(&0);
        lines.push(format!(
            "Components: {} (max block: {}q)",
            components.len(),
            max_block,
        ));
    }

    lines
}

impl Circuit {
    pub fn draw(&self, opts: &TextOptions) -> String {
        let moments = assign_moments(self);
        let use_summary =
            self.num_qubits > SUMMARY_QUBIT_THRESHOLD || moments.len() > SUMMARY_MOMENT_THRESHOLD;

        if use_summary {
            return render_summary(self).join("\n");
        }

        let lines = render_moments(&moments, self.num_qubits, opts);
        if lines.is_empty() {
            return "(empty circuit)".to_string();
        }
        lines.join("\n")
    }

    pub fn summary(&self) -> String {
        render_summary(self).join("\n")
    }

    pub fn heatmap(&self, opts: &TextOptions) -> String {
        render_heatmap(self, opts).join("\n")
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opts = TextOptions::default();
        write!(f, "{}", self.draw(&opts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::builder::CircuitBuilder;

    #[test]
    fn gate_display_labels() {
        assert_eq!(Gate::H.to_string(), "H");
        assert_eq!(Gate::Cx.to_string(), "CX");
        assert_eq!(Gate::Rx(std::f64::consts::FRAC_PI_2).to_string(), "Rx(π/2)");
        assert_eq!(Gate::Rz(0.5).to_string(), "Rz(0.5000)");
    }

    #[test]
    fn bell_pair_diagram() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains("q[0]"));
        assert!(text.contains("q[1]"));
        assert!(text.contains("H"));
        assert!(text.contains(CTRL.to_string().as_str()));
        assert!(text.contains("X"));
    }

    #[test]
    fn ghz_3q_diagram() {
        let circuit = CircuitBuilder::new(3).h(0).cx(0, 1).cx(1, 2).build();
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains("q[0]"));
        assert!(text.contains("q[1]"));
        assert!(text.contains("q[2]"));
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.len() >= 3);
    }

    fn char_col(s: &str, ch: char) -> Option<usize> {
        s.chars().position(|c| c == ch)
    }

    #[test]
    fn connector_alignment_even_width() {
        let mut builder = CircuitBuilder::new(12);
        for i in (0..12).step_by(2) {
            builder.h(i).cx(i, i + 1);
        }
        let circuit = builder.build();
        let text = circuit.draw(&TextOptions::default());
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if let Some(pos) = char_col(line, VERT) {
                if i > 0 {
                    if let Some(ctrl_pos) = char_col(lines[i - 1], CTRL) {
                        assert_eq!(
                            ctrl_pos,
                            pos,
                            "connector at line {} misaligned with control at line {}\n{}",
                            i,
                            i - 1,
                            text,
                        );
                    }
                }
                if i + 1 < lines.len() {
                    if let Some(x_pos) = char_col(lines[i + 1], 'X') {
                        assert_eq!(
                            x_pos,
                            pos,
                            "connector at line {} misaligned with target at line {}\n{}",
                            i,
                            i + 1,
                            text,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn empty_circuit() {
        let circuit = Circuit::new(2, 0);
        let text = circuit.draw(&TextOptions::default());
        assert_eq!(text, "(empty circuit)");
    }

    #[test]
    fn single_gate() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains("H"));
        assert!(text.contains("q[0]"));
    }

    #[test]
    fn measurement_shown() {
        let circuit = CircuitBuilder::new(1).h(0).measure_all().build();
        // Circuit already has a measurement from measure_all()
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains("M"));
    }

    #[test]
    fn idle_wire_elision() {
        let circuit = CircuitBuilder::new(5).h(0).cx(0, 4).build();
        let opts = TextOptions {
            show_idle_wires: false,
            ..Default::default()
        };
        let text = circuit.draw(&opts);
        assert!(text.contains("q[0]"));
        assert!(text.contains("q[4]"));
        assert!(!text.contains("q[1]"));
        assert!(!text.contains("q[2]"));
        assert!(!text.contains("q[3]"));
    }

    #[test]
    fn summary_mode() {
        let circuit = crate::circuits::random_circuit(100, 10, 42);
        let summary = circuit.summary();
        assert!(summary.contains("Circuit: 100 qubits"));
        assert!(summary.contains("Gate counts:"));
        assert!(summary.contains("Qubit activity"));
    }

    #[test]
    fn auto_summary_for_large() {
        let circuit = crate::circuits::random_circuit(100, 10, 42);
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains("Circuit: 100 qubits"));
    }

    #[test]
    fn display_impl() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let text = format!("{}", circuit);
        assert!(text.contains("q[0]"));
        assert!(text.contains("q[1]"));
    }

    #[test]
    fn fold_wraps_long_circuits() {
        let circuit = crate::circuits::random_circuit(3, 50, 42);
        let opts = TextOptions {
            fold_width: 60,
            ..Default::default()
        };
        let text = circuit.draw(&opts);
        assert!(text.contains("\u{254C}"));
    }

    #[test]
    fn swap_gate_display() {
        let circuit = CircuitBuilder::new(2).swap(0, 1).build();
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains(SWAP_X.to_string().as_str()));
    }

    #[test]
    fn parametric_gate_display() {
        let circuit = CircuitBuilder::new(1)
            .rx(std::f64::consts::FRAC_PI_4, 0)
            .build();
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains("Rx(π/4)"));
    }

    #[test]
    fn qft_4q_diagram() {
        let circuit = crate::circuits::qft_circuit(4);
        let text = circuit.draw(&TextOptions::default());
        assert!(text.contains("q[0]"));
        assert!(text.contains("q[3]"));
        assert!(text.contains("H"));
    }

    #[test]
    fn summary_has_connectivity() {
        let circuit = crate::circuits::random_circuit(100, 10, 42);
        let summary = circuit.summary();
        assert!(summary.contains("Connectivity:"));
        assert!(summary.contains("max degree:"));
    }

    #[test]
    fn summary_no_2q_connectivity() {
        let circuit = CircuitBuilder::new(4).h(0).h(1).h(2).h(3).build();
        let summary = circuit.summary();
        assert!(summary.contains("Connectivity: none"));
    }

    #[test]
    fn summary_has_depth_profile() {
        let circuit = crate::circuits::random_circuit(100, 10, 42);
        let summary = circuit.summary();
        assert!(summary.contains("Depth profile"));
    }

    #[test]
    fn summary_has_heatmap() {
        let circuit = crate::circuits::random_circuit(100, 10, 42);
        let summary = circuit.summary();
        assert!(summary.contains("Gate density heatmap"));
        assert!(summary.contains("Legend:"));
    }

    #[test]
    fn heatmap_standalone() {
        let circuit = crate::circuits::ghz_circuit(10);
        let hm = circuit.heatmap(&TextOptions::default());
        assert!(hm.contains("Gate density heatmap"));
        assert!(hm.contains("q[0]"));
    }

    #[test]
    fn connectivity_nearest_neighbor() {
        let circuit = crate::circuits::ghz_circuit(8);
        let summary = circuit.summary();
        assert!(summary.contains("nearest-neighbor"));
    }
}
