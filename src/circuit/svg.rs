//! SVG rendering for circuit diagrams and gate-density heatmaps:
//! [`Circuit::to_svg`] and [`Circuit::to_svg_heatmap`], configured through
//! [`SvgOptions`].

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use super::Circuit;
use super::draw::{OpKind, PlacedOp, assign_moments, barrier_columns};
use crate::gates::Gate;

const GATE_CORNER_RADIUS: f64 = 1.0;
const LABEL_GAP: f64 = 8.0;
const CHAR_WIDTH_FACTOR: f64 = 0.62;
const LABEL_PADDING: f64 = 16.0;

/// Layout and theme options for SVG rendering of a [`Circuit`].
///
/// All lengths are in SVG user units. `Default` gives the light theme with
/// animations on and every annotation layer (legend, header, topology) off.
pub struct SvgOptions {
    pub wire_spacing: f64,
    /// Minimum width of a moment column; wide gate labels stretch it.
    pub moment_width: f64,
    pub gate_height: f64,
    pub gate_min_width: f64,
    /// Margin on the left edge, containing the qubit labels.
    pub padding_left: f64,
    pub padding_top: f64,
    pub padding_right: f64,
    pub padding_bottom: f64,
    /// Base font size in pixels.
    pub font_size: f64,
    pub control_radius: f64,
    pub max_moments: Option<usize>,
    pub max_qubits: Option<usize>,
    pub show_idle_wires: bool,
    pub show_barriers: bool,
    pub dark_mode: bool,
    /// Embed both themes, switched by `prefers-color-scheme`; overrides `dark_mode`.
    pub auto_theme: bool,
    /// Include CSS animations; a `prefers-reduced-motion` query disables them.
    pub animate: bool,
    /// Append a gate-category color legend below the circuit.
    pub show_legend: bool,
    /// Add a qubit, gate, and depth summary line above the circuit.
    pub show_stats_header: bool,
    /// Shrink spacing, padding, and font sizes for dense embedding.
    pub compact: bool,
    /// `Some((first, last))` draws only the first and last moments with an
    /// ellipsis gap between them; ignored when the circuit already fits.
    pub ellipsis_mode: Option<(usize, usize)>,
    /// Append a circular qubit-connectivity graph beside the circuit.
    pub show_topology: bool,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            wire_spacing: 64.0,
            moment_width: 72.0,
            gate_height: 46.0,
            gate_min_width: 46.0,
            padding_left: 64.0,
            padding_top: 32.0,
            padding_right: 24.0,
            padding_bottom: 32.0,
            font_size: 20.0,
            control_radius: 4.2,
            max_moments: None,
            max_qubits: None,
            show_idle_wires: true,
            show_barriers: true,
            dark_mode: false,
            auto_theme: false,
            animate: true,
            show_legend: false,
            show_stats_header: false,
            compact: false,
            ellipsis_mode: None,
            show_topology: false,
        }
    }
}

struct GateStyle {
    fill: &'static str,
    stroke: &'static str,
    text: &'static str,
}

struct Theme {
    bg: &'static str,
    wire: &'static str,
    text: &'static str,
    control_dot: &'static str,
    control_border: &'static str,
    swap_stroke: &'static str,
    barrier: &'static str,
    standard: GateStyle,
    parametric: GateStyle,
    phase: GateStyle,
    controlled: GateStyle,
    measure: GateStyle,
    multi: GateStyle,
}

const LIGHT: Theme = Theme {
    bg: "#ffffff",
    wire: "#626262",
    text: "#252525",
    control_dot: "#252525",
    control_border: "#ffffff",
    swap_stroke: "#454545",
    barrier: "#b0b0b0",
    standard: GateStyle {
        fill: "#ffffff",
        stroke: "#363636",
        text: "#252525",
    },
    parametric: GateStyle {
        fill: "#ffffff",
        stroke: "#4c6775",
        text: "#252525",
    },
    phase: GateStyle {
        fill: "#ffffff",
        stroke: "#5d6e64",
        text: "#252525",
    },
    controlled: GateStyle {
        fill: "#ffffff",
        stroke: "#303030",
        text: "#252525",
    },
    measure: GateStyle {
        fill: "#f2f2f2",
        stroke: "#7b6650",
        text: "#54432f",
    },
    multi: GateStyle {
        fill: "#ffffff",
        stroke: "#707070",
        text: "#454545",
    },
};

const DARK: Theme = Theme {
    bg: "#181818",
    wire: "#858585",
    text: "#dddddd",
    control_dot: "#dddddd",
    control_border: "#181818",
    swap_stroke: "#cccccc",
    barrier: "#777777",
    standard: GateStyle {
        fill: "#181818",
        stroke: "#cccccc",
        text: "#dddddd",
    },
    parametric: GateStyle {
        fill: "#181818",
        stroke: "#8fa7b5",
        text: "#dddddd",
    },
    phase: GateStyle {
        fill: "#181818",
        stroke: "#9aafa2",
        text: "#dddddd",
    },
    controlled: GateStyle {
        fill: "#181818",
        stroke: "#dddddd",
        text: "#dddddd",
    },
    measure: GateStyle {
        fill: "#242424",
        stroke: "#ab9986",
        text: "#c5b5a2",
    },
    multi: GateStyle {
        fill: "#181818",
        stroke: "#a2a2a2",
        text: "#cccccc",
    },
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum GateCategory {
    Standard,
    Parametric,
    Phase,
    Controlled,
    Measure,
    Multi,
}

impl GateCategory {
    fn css_class(&self) -> &'static str {
        match self {
            Self::Standard => "gate-std",
            Self::Parametric => "gate-par",
            Self::Phase => "gate-pha",
            Self::Controlled => "gate-ctrl",
            Self::Measure => "gate-meas",
            Self::Multi => "gate-mfus",
        }
    }

    fn display_name(&self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Parametric => "Parametric",
            Self::Phase => "Phase",
            Self::Controlled => "Controlled",
            Self::Measure => "Measure",
            Self::Multi => "Fused",
        }
    }

    fn css_prefix(&self) -> &'static str {
        match self {
            Self::Standard => "std",
            Self::Parametric => "par",
            Self::Phase => "pha",
            Self::Controlled => "ctrl",
            Self::Measure => "meas",
            Self::Multi => "mfus",
        }
    }
}

impl Theme {
    fn emit_css_vars(&self, svg: &mut String) {
        let _ = write!(
            svg,
            "--bg:{};--wire:{};--text:{};--ctrl-dot:{};--ctrl-bdr:{};--swap:{};--barrier:{};",
            self.bg,
            self.wire,
            self.text,
            self.control_dot,
            self.control_border,
            self.swap_stroke,
            self.barrier,
        );
        for (prefix, sty) in [
            ("std", &self.standard),
            ("par", &self.parametric),
            ("pha", &self.phase),
            ("ctrl", &self.controlled),
            ("meas", &self.measure),
            ("mfus", &self.multi),
        ] {
            let _ = write!(
                svg,
                "--{prefix}-fill:{};--{prefix}-str:{};--{prefix}-txt:{};",
                sty.fill, sty.stroke, sty.text,
            );
        }
    }
}

fn classify_gate(gate: Option<&Gate>) -> GateCategory {
    match gate {
        Some(Gate::S | Gate::T | Gate::Sdg | Gate::Tdg) => GateCategory::Phase,
        Some(Gate::Rx(_) | Gate::Ry(_) | Gate::Rz(_) | Gate::P(_) | Gate::Rzz(_)) => {
            GateCategory::Parametric
        }
        _ => GateCategory::Standard,
    }
}

fn heatmap_stops(dark_mode: bool) -> [(f64, f64, f64); 5] {
    if dark_mode {
        [
            (24.0, 24.0, 24.0),
            (33.0, 68.0, 105.0),
            (50.0, 108.0, 158.0),
            (92.0, 164.0, 206.0),
            (185.0, 230.0, 243.0),
        ]
    } else {
        [
            (241.0, 241.0, 241.0),
            (199.0, 217.0, 240.0),
            (143.0, 181.0, 222.0),
            (77.0, 135.0, 194.0),
            (36.0, 90.0, 146.0),
        ]
    }
}

fn heatmap_color(t: f64, dark_mode: bool) -> String {
    let t = t.clamp(0.0, 1.0);
    let stops = heatmap_stops(dark_mode);
    let idx = t * 4.0;
    let i = (idx as usize).min(3);
    let f = idx - i as f64;
    let r = (stops[i].0 + f * (stops[i + 1].0 - stops[i].0)).clamp(0.0, 255.0) as u8;
    let g = (stops[i].1 + f * (stops[i + 1].1 - stops[i].1)).clamp(0.0, 255.0) as u8;
    let b = (stops[i].2 + f * (stops[i + 1].2 - stops[i].2)).clamp(0.0, 255.0) as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn escape_xml(s: &str) -> std::borrow::Cow<'_, str> {
    if !s.contains(&['<', '>', '&', '"', '\''][..]) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    std::borrow::Cow::Owned(out)
}

fn label_width(label: &str, font_size: f64) -> f64 {
    label.chars().count() as f64 * font_size * CHAR_WIDTH_FACTOR + LABEL_PADDING
}

fn parameter_parts(label: &str) -> Option<(&str, &str)> {
    let (name, value) = label.split_once('(')?;
    if !matches!(name, "Rx" | "Ry" | "Rz" | "Rzz" | "P") {
        return None;
    }
    Some((name, value.strip_suffix(')')?))
}

fn display_parameter(value: &str) -> Cow<'_, str> {
    let Ok(angle) = value.parse::<f64>() else {
        return Cow::Borrowed(value);
    };
    if angle == 0.0 {
        return Cow::Borrowed("0");
    }
    if angle.abs() < 0.001 {
        return Cow::Owned(format!("{angle:.2e}"));
    }
    let rounded = format!("{angle:.3}");
    Cow::Owned(
        rounded
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned(),
    )
}

fn gate_label(op: &PlacedOp) -> Cow<'_, str> {
    if let Some((name, value)) = parameter_parts(&op.label)
        && !value.contains('π')
        && let Some(
            Gate::Rx(angle) | Gate::Ry(angle) | Gate::Rz(angle) | Gate::P(angle) | Gate::Rzz(angle),
        ) = op.gate.as_ref()
    {
        return Cow::Owned(format!("{name}({angle})"));
    }
    Cow::Borrowed(&op.label)
}

fn conditional_label(op: &PlacedOp, cbit_label: &str) -> String {
    let label = gate_label(op);
    if let Some((name, value)) = parameter_parts(&label) {
        format!("{cbit_label}?{name}({})", display_parameter(value))
    } else {
        format!("{cbit_label}?{label}")
    }
}

fn gate_width(label: &str, opts: &SvgOptions) -> f64 {
    let width = if let Some((name, value)) = parameter_parts(label) {
        label_width(name, opts.font_size).max(label_width(
            &display_parameter(value),
            (opts.font_size * 0.6).max(8.0),
        ))
    } else {
        label_width(label, opts.font_size)
    };
    width.max(opts.gate_min_width)
}

fn controlled_target_label(op: &PlacedOp) -> Cow<'_, str> {
    if let Some(phase) = op.gate.as_ref().and_then(Gate::controlled_phase) {
        let angle = phase.arg();
        let label = Gate::P(angle).to_string();
        return Cow::Owned(if label.contains('π') {
            label
        } else {
            format!("P({angle})")
        });
    }
    Cow::Borrowed(match op.label.as_str() {
        "CZ" => "Z",
        "CU" => "U",
        other if other.starts_with("MCU") => "U",
        other => other,
    })
}

fn emit_theme_vars(svg: &mut String, theme: &Theme, auto_theme: bool) {
    let _ = write!(svg, "<style>.pq-circuit{{");
    if auto_theme {
        LIGHT.emit_css_vars(svg);
    } else {
        theme.emit_css_vars(svg);
    }
    let _ = write!(svg, "}}");
    if auto_theme {
        let _ = write!(svg, "@media(prefers-color-scheme:dark){{.pq-circuit{{");
        DARK.emit_css_vars(svg);
        let _ = write!(svg, "}}}}");
    }
}

fn empty_svg(theme: &Theme) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"60\" viewBox=\"0 0 200 60\">\
         <rect width=\"100%\" height=\"100%\" fill=\"{}\"/>\
         <text x=\"100\" y=\"35\" text-anchor=\"middle\" \
         font-family=\"monospace\" font-size=\"14\" fill=\"{}\">empty circuit</text>\
         </svg>",
        theme.bg, theme.text,
    )
}

fn tooltip(op: &PlacedOp) -> String {
    let qs: Vec<String> = op.qubits.iter().map(|q| format!("q[{q}]")).collect();
    let mut tip = match &op.kind {
        OpKind::Controlled { controls, target } => {
            let ctrls: Vec<String> = controls.iter().map(|c| format!("q[{c}]")).collect();
            format!(
                "{}: ctrl {} → target q[{}]",
                controlled_target_label(op),
                ctrls.join(", "),
                target
            )
        }
        OpKind::Swap => format!("SWAP {} ↔ {}", qs[0], qs[1]),
        OpKind::Measure { cbit } => format!("Measure {} → c[{}]", qs[0], cbit),
        OpKind::Barrier => format!("Barrier on {}", qs.join(", ")),
        _ => format!("{} on {}", gate_label(op), qs.join(", ")),
    };
    let angle = match op.gate.as_ref() {
        Some(
            Gate::Rx(angle) | Gate::Ry(angle) | Gate::Rz(angle) | Gate::P(angle) | Gate::Rzz(angle),
        ) => Some(*angle),
        Some(gate) => gate.controlled_phase().map(|phase| phase.arg()),
        None => None,
    };
    if let Some(angle) = angle {
        let _ = write!(tip, "; angle {angle} rad");
    }
    tip
}

const CLASSICAL_WIRE_SPACING: f64 = 30.0;

fn render_svg(
    moments: &[Vec<PlacedOp>],
    num_qubits: usize,
    num_classical_bits: usize,
    opts: &SvgOptions,
) -> String {
    let compact_opts;
    let opts = if opts.compact {
        compact_opts = SvgOptions {
            wire_spacing: 32.0,
            moment_width: 44.0,
            gate_height: 26.0,
            gate_min_width: 28.0,
            padding_left: 48.0,
            padding_top: 18.0,
            padding_right: 14.0,
            padding_bottom: 18.0,
            font_size: 11.0,
            control_radius: 3.0,
            ..*opts
        };
        &compact_opts
    } else {
        opts
    };

    let theme = if opts.dark_mode { &DARK } else { &LIGHT };

    if moments.is_empty() || num_qubits == 0 {
        return empty_svg(theme);
    }

    let content_columns: Vec<usize> = moments
        .iter()
        .enumerate()
        .filter_map(|(i, moment)| {
            moment
                .iter()
                .any(|op| !matches!(op.kind, OpKind::Barrier))
                .then_some(i)
        })
        .collect();
    let max_moments = opts
        .max_moments
        .unwrap_or(content_columns.len())
        .min(content_columns.len());
    let moment_end = if max_moments < content_columns.len() {
        max_moments
            .checked_sub(1)
            .map_or(0, |i| content_columns[i] + 1)
    } else {
        moments.len()
    };
    let moments = &moments[..moment_end];
    let content_columns = &content_columns[..max_moments];

    let mut active_qubits = vec![false; num_qubits];
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
        return empty_svg(theme);
    }

    let max_vis = opts
        .max_qubits
        .unwrap_or(visible_qubits.len())
        .min(visible_qubits.len());
    let show_qubits = &visible_qubits[..max_vis];

    let qubit_to_row: Vec<Option<usize>> = {
        let mut map = vec![None; num_qubits];
        for (row, &q) in show_qubits.iter().enumerate() {
            map[q] = Some(row);
        }
        map
    };

    let num_rows = show_qubits.len();

    let ellipsis = opts.ellipsis_mode.and_then(|(first, last)| {
        if first.saturating_add(last) < content_columns.len() {
            let first_end = first.checked_sub(1).map_or(0, |i| content_columns[i] + 1);
            let last_start = if last == 0 {
                moments.len()
            } else {
                content_columns[content_columns.len() - last]
            };
            Some((first_end, last_start))
        } else {
            None
        }
    });

    let mut col_widths: Vec<f64> = Vec::with_capacity(moments.len());
    for moment in moments {
        if moment.iter().all(|op| matches!(op.kind, OpKind::Barrier)) {
            col_widths.push(opts.moment_width * 0.4);
            continue;
        }
        let mut max_w = opts.gate_min_width;
        for op in moment {
            let w = match &op.kind {
                OpKind::Conditional { cbit_label } => {
                    gate_width(&conditional_label(op, cbit_label), opts)
                }
                OpKind::Controlled { .. } => gate_width(&controlled_target_label(op), opts),
                _ => gate_width(&gate_label(op), opts),
            };
            if w > max_w {
                max_w = w;
            }
        }
        let measurements = moment
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Measure { .. }))
            .count();
        let shared_lane = usize::from(
            moment
                .iter()
                .any(|op| !matches!(op.kind, OpKind::Measure { .. })),
        );
        col_widths
            .push((max_w + LABEL_GAP).max(opts.moment_width) * (measurements + shared_lane) as f64);
    }

    let ellipsis_gap = 40.0;
    let mut col_x: Vec<f64> = Vec::with_capacity(moments.len());
    let mut cx_pos = opts.padding_left;
    let mut ellipsis_x = 0.0_f64;
    for (i, &w) in col_widths.iter().enumerate() {
        if let Some((first_end, last_start)) = ellipsis {
            if i == first_end {
                ellipsis_x = cx_pos;
                cx_pos += ellipsis_gap;
            }
            if i >= first_end && i < last_start {
                col_x.push(f64::NAN);
                continue;
            }
        }
        col_x.push(cx_pos);
        cx_pos += w;
    }

    let stats = if opts.show_stats_header {
        let gate_count = moments
            .iter()
            .flatten()
            .filter(|op| !matches!(op.kind, OpKind::Barrier | OpKind::Measure { .. }))
            .count();
        let depth = moments
            .iter()
            .filter(|m| m.iter().any(|op| !matches!(op.kind, OpKind::Barrier)))
            .count();
        format!(
            "{} qubit{} \u{00b7} {} gate{} \u{00b7} depth {}",
            num_qubits,
            if num_qubits == 1 { "" } else { "s" },
            gate_count,
            if gate_count == 1 { "" } else { "s" },
            depth,
        )
    } else {
        String::new()
    };
    let mut circuit_width = cx_pos + opts.padding_right;
    if opts.show_stats_header {
        circuit_width = circuit_width
            .max(opts.padding_left + label_width(&stats, opts.font_size) + opts.padding_right);
    }
    if opts.show_legend {
        let present: HashSet<_> = moments
            .iter()
            .flatten()
            .filter_map(|op| match op.kind {
                OpKind::Barrier | OpKind::Swap => None,
                OpKind::MultiFused => Some(GateCategory::Multi),
                OpKind::Controlled { .. } => Some(GateCategory::Controlled),
                OpKind::Measure { .. } | OpKind::Reset => Some(GateCategory::Measure),
                _ => Some(classify_gate(op.gate.as_ref())),
            })
            .collect();
        let font = opts.font_size * 0.92;
        let legend_width: f64 = present.iter().map(|cat| legend_item_width(cat, font)).sum();
        circuit_width = circuit_width.max(opts.padding_left + legend_width + opts.padding_right);
    }
    let header_extra = if opts.show_stats_header {
        opts.font_size + opts.gate_height / 2.0 + LABEL_GAP
    } else {
        0.0
    };
    let legend_extra = if opts.show_legend {
        let swatch_h = opts.font_size * 0.92 * 0.9;
        4.0 + swatch_h + 8.0 + opts.padding_bottom * 0.5
    } else {
        0.0
    };
    let pad_top = opts.padding_top + header_extra;
    let classical_gap = opts.gate_height / 2.0 + 24.0;
    let classical_extra = if num_classical_bits > 0 {
        classical_gap + (num_classical_bits as f64 - 1.0) * CLASSICAL_WIRE_SPACING + 10.0
    } else {
        0.0
    };
    let circuit_height = pad_top
        + (num_rows as f64 - 1.0) * opts.wire_spacing
        + classical_extra
        + opts.padding_bottom;

    let topo_edges = if opts.show_topology {
        extract_topology(moments, num_qubits)
    } else {
        Vec::new()
    };
    let has_topo = opts.show_topology && !topo_edges.is_empty();
    let topo_diameter = if has_topo {
        (circuit_height - pad_top - opts.padding_bottom).clamp(80.0, 200.0)
    } else {
        0.0
    };
    let topo_extra = if has_topo { topo_diameter + 40.0 } else { 0.0 };

    let total_width = circuit_width + topo_extra;
    let total_height = circuit_height + legend_extra;

    let wire_y = |row: usize| -> f64 { pad_top + row as f64 * opts.wire_spacing };
    let last_quantum_y = wire_y(num_rows - 1);
    let classical_y = |cbit: usize| -> f64 {
        last_quantum_y + classical_gap + cbit as f64 * CLASSICAL_WIRE_SPACING
    };

    let mut svg = String::with_capacity(moments.len() * num_rows * 150 + 4096);

    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_width:.1}\" height=\"{total_height:.1}\" \
         viewBox=\"0 0 {:.1} {:.1}\" \
         class=\"pq-circuit\" role=\"img\" aria-label=\"Quantum circuit: {} qubits, {} moments\">",
        total_width, total_height, num_qubits, max_moments,
    );

    let _ = write!(svg, "<defs>");
    let _ = write!(
        svg,
        "<marker id=\"arrow-meas\" viewBox=\"0 0 6 6\" refX=\"3\" refY=\"3\" \
         markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\">\
         <path d=\"M 0 0 L 6 3 L 0 6 z\" fill=\"var(--meas-str)\"/></marker>",
    );
    let _ = writeln!(svg, "</defs>");

    emit_theme_vars(&mut svg, theme, opts.auto_theme);
    let anim_gate = if opts.animate {
        "animation:gate-in 0.3s ease both;"
    } else {
        ""
    };
    let _ = write!(
        svg,
        ".pq-circuit{{font-size:{:.0}px;}}\
         .pq-circuit text{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;\
         text-rendering:optimizeLegibility;\
         font-variant-numeric:tabular-nums;}}\
         .pq-circuit .gate text{{font-family:Cambria,'Times New Roman',serif;}}\
         .pq-circuit .gate .gate-name{{font-style:italic;}}\
         .pq-circuit .gate .gate-param{{font-family:Arial,Helvetica,sans-serif;fill:var(--text);opacity:0.85;}}\
         .pq-circuit .gate,.pq-circuit .ctrl{{{anim_gate}}}",
        opts.font_size,
    );
    for prefix in ["std", "par", "pha", "ctrl", "meas", "mfus"] {
        let _ = write!(
            svg,
            ".pq-circuit .gate-{prefix} rect,.pq-circuit .gate-{prefix} circle\
             {{fill:var(--{prefix}-fill);stroke:var(--{prefix}-str);}}\
             .pq-circuit .gate-{prefix} text{{fill:var(--{prefix}-txt);}}",
        );
    }
    let _ = write!(
        svg,
        ".pq-circuit .gate-ctrl line{{stroke:var(--ctrl-str);}}\
         .pq-circuit .gate-meas path,.pq-circuit .gate-meas line{{stroke:var(--meas-txt);}}"
    );
    let _ = write!(
        svg,
        ".pq-circuit .qlabel{{font-family:Cambria,'Times New Roman',serif;}}\
         .pq-circuit .stats{{font-family:Arial,Helvetica,sans-serif;}}\
         .pq-circuit .qlabel{{fill:var(--text);}}\
         .pq-circuit .qlabel:not([font-size]){{font-size:{:.1}px;}}",
        opts.font_size * 0.92,
    );
    if opts.animate {
        let _ = write!(
            svg,
            "@keyframes gate-in{{from{{opacity:0;transform:translateY(4px)}}\
             to{{opacity:1;transform:translateY(0)}}}}"
        );
        let max_m = moments.len().min(50);
        for i in 0..=max_m {
            let _ = write!(svg, ".pq-circuit .m{i}{{animation-delay:{}ms}}", i * 30);
        }
        let _ = write!(
            svg,
            "@keyframes wire-flow{{to{{stroke-dashoffset:-12}}}}\
             .pq-circuit .wire-anim{{stroke-dasharray:1 11;animation:wire-flow 2s linear infinite;}}"
        );
        let _ = write!(
            svg,
            "@media(prefers-reduced-motion:reduce){{\
             .pq-circuit .gate,.pq-circuit .ctrl{{animation:none!important;}}\
             .pq-circuit .wire-anim{{animation:none!important;stroke-dasharray:none;}}}}"
        );
    }
    let _ = writeln!(svg, "</style>");

    svg.push_str("<g id=\"layer-bg\" role=\"presentation\">\n");
    let _ = writeln!(
        svg,
        "<rect width=\"100%\" height=\"100%\" fill=\"var(--bg)\"/>",
    );
    svg.push_str("</g>\n");

    if opts.show_stats_header {
        let _ = writeln!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" class=\"stats\" font-size=\"{:.1}px\" \
             fill=\"var(--text)\" dominant-baseline=\"central\">{stats}</text>",
            opts.padding_left,
            opts.padding_top + opts.font_size / 2.0,
            opts.font_size,
        );
    }

    svg.push_str("<g id=\"layer-wires\">\n");
    for (row, &q) in show_qubits.iter().enumerate() {
        let y = wire_y(row);
        emit_register_label(&mut svg, "q", q, opts.padding_left - LABEL_GAP, y, opts);
    }
    let wire_end = circuit_width - opts.padding_right;
    let wire_extra = if opts.animate {
        "class=\"wire-anim\""
    } else {
        "shape-rendering=\"crispEdges\""
    };
    for row in 0..num_rows {
        let y = wire_y(row);
        let _ = writeln!(
            svg,
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"var(--wire)\" stroke-width=\"1\" stroke-linecap=\"round\" \
             {wire_extra}/>",
            opts.padding_left, y, wire_end, y,
        );
    }
    for c in 0..num_classical_bits {
        let cy = classical_y(c);
        emit_register_label(&mut svg, "c", c, opts.padding_left - LABEL_GAP, cy, opts);
        let _ = writeln!(
            svg,
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"var(--wire)\" stroke-width=\"0.5\" stroke-linecap=\"round\"/>",
            opts.padding_left,
            cy - 1.0,
            wire_end,
            cy - 1.0,
        );
        let _ = writeln!(
            svg,
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"var(--wire)\" stroke-width=\"0.5\" stroke-linecap=\"round\"/>",
            opts.padding_left,
            cy + 1.0,
            wire_end,
            cy + 1.0,
        );
    }
    svg.push_str("</g>\n");

    svg.push_str("<g id=\"layer-gates\">\n");
    let row_of = |q: usize| -> Option<usize> { qubit_to_row.get(q).copied().flatten() };
    let mut present_cats: HashSet<GateCategory> = HashSet::new();

    if ellipsis.is_some() {
        let ecx = ellipsis_x + ellipsis_gap / 2.0;
        for row in 0..num_rows {
            let y = wire_y(row);
            let _ = writeln!(
                svg,
                "<text x=\"{ecx:.1}\" y=\"{y:.1}\" text-anchor=\"middle\" \
                 dominant-baseline=\"central\" fill=\"var(--wire)\" \
                 font-size=\"{:.0}px\">\u{22ef}</text>",
                opts.font_size + 4.0,
            );
        }
    }

    for (m_idx, moment) in moments.iter().enumerate() {
        let mx = col_x[m_idx];
        if mx.is_nan() {
            continue;
        }
        let mw = col_widths[m_idx];
        let measurements = moment
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Measure { .. }))
            .count();
        let shared_lane = usize::from(
            moment
                .iter()
                .any(|op| !matches!(op.kind, OpKind::Measure { .. })),
        );
        let lane_width = mw / (measurements + shared_lane) as f64;
        let center_x = mx + lane_width / 2.0;
        let mut measurement_lane = shared_lane;
        let mc = if opts.animate {
            format!(" m{}", m_idx.min(50))
        } else {
            String::new()
        };

        for op in moment {
            let tip_raw = tooltip(op);
            let tip = escape_xml(&tip_raw);

            match &op.kind {
                OpKind::Single => {
                    let cat = classify_gate(op.gate.as_ref());
                    present_cats.insert(cat);
                    let cls = cat.css_class();
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_gate_box(
                                &mut svg,
                                center_x,
                                cy,
                                &gate_label(op),
                                opts,
                                cls,
                                &tip,
                                &mc,
                            );
                        }
                    }
                }

                OpKind::MultiFused => {
                    present_cats.insert(GateCategory::Multi);
                    let cls = GateCategory::Multi.css_class();
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_gate_box(
                                &mut svg,
                                center_x,
                                cy,
                                &gate_label(op),
                                opts,
                                cls,
                                &tip,
                                &mc,
                            );
                        }
                    }
                }

                OpKind::Controlled { controls, target } => {
                    present_cats.insert(GateCategory::Controlled);
                    let all_rows: Vec<usize> = controls
                        .iter()
                        .chain(std::iter::once(target))
                        .filter_map(|&q| row_of(q))
                        .collect();

                    if all_rows.len() >= 2 {
                        let min_row = *all_rows.iter().min().unwrap();
                        let max_row = *all_rows.iter().max().unwrap();
                        let min_y = wire_y(min_row);
                        let max_y = wire_y(max_row);
                        let _ = writeln!(
                            svg,
                            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                             stroke=\"var(--ctrl-str)\" stroke-width=\"1\" \
                             shape-rendering=\"crispEdges\"/>",
                            center_x, min_y, center_x, max_y,
                        );
                    }

                    for &c in controls {
                        if let Some(row) = row_of(c) {
                            let cy = wire_y(row);
                            let _ = writeln!(
                                svg,
                                "<g class=\"ctrl{mc}\"><title>{tip}</title>\
                                 <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" \
                                 fill=\"var(--ctrl-dot)\" stroke=\"var(--ctrl-bdr)\" \
                                 stroke-width=\"1\"/></g>",
                                center_x, cy, opts.control_radius,
                            );
                        }
                    }

                    if let Some(row) = row_of(*target) {
                        let cy = wire_y(row);
                        let is_cx = op.label == "CX";
                        if is_cx {
                            emit_cnot_target(&mut svg, center_x, cy, opts, &tip, &mc);
                        } else {
                            let tgt_label = controlled_target_label(op);
                            let cls = GateCategory::Controlled.css_class();
                            emit_gate_box(&mut svg, center_x, cy, &tgt_label, opts, cls, &tip, &mc);
                        }
                    }
                }

                OpKind::TwoQubit => {
                    let rows: Vec<usize> = op.qubits.iter().filter_map(|&q| row_of(q)).collect();
                    let cat = classify_gate(op.gate.as_ref());
                    present_cats.insert(cat);

                    if rows.len() >= 2 {
                        let min_y = wire_y(*rows.iter().min().unwrap());
                        let max_y = wire_y(*rows.iter().max().unwrap());
                        let var_str = match cat {
                            GateCategory::Standard => "var(--std-str)",
                            GateCategory::Parametric => "var(--par-str)",
                            GateCategory::Phase => "var(--pha-str)",
                            GateCategory::Controlled => "var(--ctrl-str)",
                            GateCategory::Measure => "var(--meas-str)",
                            GateCategory::Multi => "var(--mfus-str)",
                        };
                        let _ = writeln!(
                            svg,
                            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                             stroke=\"{}\" stroke-width=\"1\" \
                             shape-rendering=\"crispEdges\"/>",
                            center_x, min_y, center_x, max_y, var_str,
                        );
                    }

                    let cls = cat.css_class();
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_gate_box(
                                &mut svg,
                                center_x,
                                cy,
                                &gate_label(op),
                                opts,
                                cls,
                                &tip,
                                &mc,
                            );
                        }
                    }
                }

                OpKind::Swap => {
                    let rows: Vec<usize> = op.qubits.iter().filter_map(|&q| row_of(q)).collect();

                    if rows.len() == 2 {
                        let min_row = rows[0].min(rows[1]);
                        let max_row = rows[0].max(rows[1]);
                        let min_y = wire_y(min_row);
                        let max_y = wire_y(max_row);
                        let _ = writeln!(
                            svg,
                            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                             stroke=\"var(--swap)\" stroke-width=\"1\" \
                             shape-rendering=\"crispEdges\"/>",
                            center_x, min_y, center_x, max_y,
                        );
                    }

                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_swap_cross(&mut svg, center_x, cy, opts, &tip, &mc);
                        }
                    }
                }

                OpKind::Barrier => {
                    if opts.show_barriers {
                        let rows: Vec<usize> =
                            op.qubits.iter().filter_map(|&q| row_of(q)).collect();
                        if rows.len() >= 2 {
                            let min_y =
                                wire_y(*rows.iter().min().unwrap()) - opts.gate_height / 2.0;
                            let max_y =
                                wire_y(*rows.iter().max().unwrap()) + opts.gate_height / 2.0;
                            let bar_w = 4.0;
                            let _ = writeln!(
                                svg,
                                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{bar_w}\" \
                                 height=\"{:.1}\" rx=\"1\" \
                                 fill=\"var(--barrier)\" fill-opacity=\"0.55\" \
                                 stroke=\"var(--barrier)\" stroke-width=\"0.5\"/>",
                                center_x - bar_w / 2.0,
                                min_y,
                                max_y - min_y,
                            );
                        }
                    }
                }

                OpKind::Reset => {
                    present_cats.insert(GateCategory::Measure);
                    let cls = GateCategory::Measure.css_class();
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_gate_box(&mut svg, center_x, cy, &op.label, opts, cls, &tip, &mc);
                        }
                    }
                }

                OpKind::Measure { cbit } => {
                    present_cats.insert(GateCategory::Measure);
                    let meter_x = mx + lane_width * (measurement_lane as f64 + 0.5);
                    measurement_lane += 1;
                    if let Some(row) = op.qubits.first().and_then(|&q| row_of(q)) {
                        let cy = wire_y(row);
                        emit_measure_box(&mut svg, meter_x, cy, opts, &tip, &mc);
                        if *cbit < num_classical_bits {
                            let cbit_y = classical_y(*cbit);
                            let arrow_top = cy + opts.gate_height / 2.0;
                            let _ = writeln!(
                                svg,
                                "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                                 stroke=\"var(--meas-str)\" stroke-width=\"0.75\" \
                                 stroke-dasharray=\"3,3\" marker-end=\"url(#arrow-meas)\"/>",
                                meter_x,
                                arrow_top,
                                meter_x,
                                cbit_y - 1.0,
                            );
                        }
                    }
                }

                OpKind::Conditional { cbit_label } => {
                    let cat = classify_gate(op.gate.as_ref());
                    present_cats.insert(cat);
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            let lbl = conditional_label(op, cbit_label);
                            let cls = cat.css_class();
                            emit_gate_box_inner(
                                &mut svg, center_x, cy, &lbl, opts, cls, &tip, &mc, true,
                            );
                            let cbit_idx = cbit_label
                                .strip_prefix("c[")
                                .and_then(|s| s.strip_suffix(']'))
                                .and_then(|s| s.parse::<usize>().ok());
                            if let Some(ci) = cbit_idx
                                && ci < num_classical_bits
                            {
                                let cbit_y = classical_y(ci);
                                let gate_bottom = cy + opts.gate_height / 2.0;
                                let _ = writeln!(
                                    svg,
                                    "<line x1=\"{:.1}\" y1=\"{:.1}\" \
                                         x2=\"{:.1}\" y2=\"{:.1}\" \
                                         stroke=\"var(--wire)\" stroke-width=\"0.75\" \
                                         stroke-dasharray=\"3,3\"/>",
                                    center_x, gate_bottom, center_x, cbit_y,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    svg.push_str("</g>\n");

    if opts.show_legend && !present_cats.is_empty() {
        emit_legend(&mut svg, opts, &present_cats, circuit_height, total_width);
    }

    if has_topo {
        emit_topology(
            &mut svg,
            &topo_edges,
            num_qubits,
            circuit_width + 20.0,
            pad_top,
            topo_diameter,
        );
    }

    svg.push_str("</svg>\n");
    svg
}

struct TopoEdge {
    a: usize,
    b: usize,
    weight: usize,
}

fn extract_topology(moments: &[Vec<PlacedOp>], num_qubits: usize) -> Vec<TopoEdge> {
    let mut counts = HashMap::new();
    for moment in moments {
        for op in moment {
            if op.qubits.len() >= 2 {
                match &op.kind {
                    OpKind::Controlled { controls, target } => {
                        for &c in controls {
                            if c < num_qubits && *target < num_qubits {
                                let (a, b) = if c < *target {
                                    (c, *target)
                                } else {
                                    (*target, c)
                                };
                                *counts.entry((a, b)).or_insert(0usize) += 1;
                            }
                        }
                    }
                    OpKind::Barrier => {}
                    _ => {
                        for i in 0..op.qubits.len() {
                            for j in (i + 1)..op.qubits.len() {
                                let (a, b) = if op.qubits[i] < op.qubits[j] {
                                    (op.qubits[i], op.qubits[j])
                                } else {
                                    (op.qubits[j], op.qubits[i])
                                };
                                if a < num_qubits && b < num_qubits {
                                    *counts.entry((a, b)).or_insert(0usize) += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut edges: Vec<TopoEdge> = counts
        .into_iter()
        .map(|((a, b), w)| TopoEdge { a, b, weight: w })
        .collect();
    edges.sort_by_key(|e| (e.a, e.b));
    edges
}

fn emit_topology(
    svg: &mut String,
    edges: &[TopoEdge],
    num_qubits: usize,
    origin_x: f64,
    origin_y: f64,
    diameter: f64,
) {
    let radius = diameter / 2.0;
    let cx = origin_x + radius;
    let cy = origin_y + radius;
    let node_r = if num_qubits <= 8 {
        radius * 0.14
    } else if num_qubits <= 16 {
        radius * 0.10
    } else {
        radius * 0.06
    };

    let qubit_pos: Vec<(f64, f64)> = (0..num_qubits)
        .map(|i| {
            let angle = -std::f64::consts::FRAC_PI_2
                + 2.0 * std::f64::consts::PI * i as f64 / num_qubits as f64;
            (
                cx + radius * 0.8 * angle.cos(),
                cy + radius * 0.8 * angle.sin(),
            )
        })
        .collect();

    svg.push_str("<g id=\"layer-topology\">\n");

    let title_y = origin_y - 2.0;
    let title_w = 56.0;
    let title_h = 14.0;
    let _ = writeln!(
        svg,
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{title_w}\" height=\"{title_h}\" \
         rx=\"4\" fill=\"var(--bg)\" fill-opacity=\"0.85\" \
         stroke=\"var(--wire)\" stroke-width=\"0.5\" stroke-opacity=\"0.3\"/>",
        cx - title_w / 2.0,
        title_y - title_h * 0.65,
    );
    let _ = writeln!(
        svg,
        "<text x=\"{cx:.1}\" y=\"{title_y:.1}\" text-anchor=\"middle\" \
         class=\"qlabel\" font-size=\"9px\" fill=\"var(--text)\" opacity=\"0.6\">topology</text>",
    );

    let max_w = edges.iter().map(|e| e.weight).max().unwrap_or(1).max(1);
    for edge in edges {
        let (ax, ay) = qubit_pos[edge.a];
        let (bx, by) = qubit_pos[edge.b];
        let t = edge.weight as f64 / max_w as f64;
        let sw = 0.5 + t * 2.5;
        let opacity = 0.25 + t * 0.55;
        let _ = writeln!(
            svg,
            "<line x1=\"{ax:.1}\" y1=\"{ay:.1}\" x2=\"{bx:.1}\" y2=\"{by:.1}\" \
             stroke=\"var(--std-str)\" stroke-width=\"{sw:.1}\" \
             stroke-opacity=\"{opacity:.2}\" stroke-linecap=\"round\"/>",
        );
    }

    for (i, &(px, py)) in qubit_pos.iter().enumerate() {
        let has_edge = edges.iter().any(|e| e.a == i || e.b == i);
        let fill = if has_edge {
            "var(--std-str)"
        } else {
            "var(--wire)"
        };
        let _ = writeln!(
            svg,
            "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"{node_r:.1}\" \
             fill=\"{fill}\" stroke=\"var(--bg)\" stroke-width=\"1\"/>",
        );
        if num_qubits <= 16 {
            let _ = writeln!(
                svg,
                "<text x=\"{px:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
                 font-size=\"7px\" fill=\"var(--text)\" \
                 dominant-baseline=\"central\">{i}</text>",
                py + 0.5,
            );
        }
    }

    svg.push_str("</g>\n");
}

const LEGEND_ORDER: [GateCategory; 6] = [
    GateCategory::Standard,
    GateCategory::Parametric,
    GateCategory::Phase,
    GateCategory::Controlled,
    GateCategory::Measure,
    GateCategory::Multi,
];

fn legend_item_width(cat: &GateCategory, font_size: f64) -> f64 {
    let swatch = font_size * 0.9;
    let gap = font_size * 0.33;
    let text_w = cat.display_name().len() as f64 * font_size * CHAR_WIDTH_FACTOR;
    let spacing = font_size * 1.15;
    swatch + gap + text_w + spacing
}

#[allow(clippy::too_many_arguments)]
fn emit_legend(
    svg: &mut String,
    opts: &SvgOptions,
    present: &HashSet<GateCategory>,
    circuit_height: f64,
    total_width: f64,
) {
    let avail_w = total_width - opts.padding_left - opts.padding_right;
    let base_font = opts.font_size * 0.92;
    let trailing_base = base_font * 1.15;
    let full_w: f64 = LEGEND_ORDER
        .iter()
        .filter(|c| present.contains(c))
        .map(|c| legend_item_width(c, base_font))
        .sum::<f64>()
        - trailing_base;

    let scale = if full_w > avail_w {
        (avail_w / full_w).max(0.6)
    } else {
        1.0
    };
    let font = base_font * scale;
    let swatch_size = (font * 0.9).max(6.0);
    let swatch_r = 2.0_f64.min(swatch_size * 0.2);
    let gap = font * 0.33;
    let trailing = font * 1.15;

    let total_legend_w: f64 = LEGEND_ORDER
        .iter()
        .filter(|c| present.contains(c))
        .map(|c| legend_item_width(c, font))
        .sum::<f64>()
        - trailing;

    let legend_y = circuit_height + 4.0;
    let start_x = ((total_width - total_legend_w) / 2.0).max(opts.padding_left);

    svg.push_str("<g id=\"layer-legend\">\n");
    let mut x = start_x;
    for cat in &LEGEND_ORDER {
        if !present.contains(cat) {
            continue;
        }
        let prefix = cat.css_prefix();
        let _ = writeln!(
            svg,
            "<rect x=\"{x:.1}\" y=\"{:.1}\" width=\"{swatch_size:.1}\" \
             height=\"{swatch_size:.1}\" rx=\"{swatch_r:.1}\" \
             fill=\"var(--{prefix}-fill)\" stroke=\"var(--{prefix}-str)\" \
             stroke-width=\"0.75\"/>",
            legend_y,
        );
        let _ = writeln!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" class=\"qlabel\" font-size=\"{font:.1}px\" \
             dominant-baseline=\"central\">{}</text>",
            x + swatch_size + gap,
            legend_y + swatch_size / 2.0,
            cat.display_name(),
        );
        x += legend_item_width(cat, font);
    }
    svg.push_str("</g>\n");
}

fn emit_register_label(
    svg: &mut String,
    register: &str,
    index: usize,
    x: f64,
    y: f64,
    opts: &SvgOptions,
) {
    let _ = writeln!(
        svg,
        "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"end\" \
         dominant-baseline=\"central\" class=\"qlabel\" aria-label=\"{register}[{index}]\">\
         <tspan font-style=\"italic\">{register}</tspan>\
         <tspan baseline-shift=\"sub\" font-size=\"{:.1}px\">{index}</tspan></text>",
        opts.font_size * 0.64,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_gate_box_inner(
    svg: &mut String,
    cx: f64,
    cy: f64,
    label: &str,
    opts: &SvgOptions,
    cat_class: &str,
    tip: &str,
    mc: &str,
    dashed: bool,
) {
    let w = gate_width(label, opts);
    let h = opts.gate_height;
    let dash_attr = if dashed {
        " stroke-dasharray=\"4,2\""
    } else {
        ""
    };
    let _ = writeln!(
        svg,
        "<g class=\"gate {cat_class}{mc}\" aria-label=\"{}\"><title>{tip}</title>\
         <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
         rx=\"{GATE_CORNER_RADIUS}\" stroke-width=\"1.25\"{dash_attr}/>",
        escape_xml(label),
        cx - w / 2.0,
        cy - h / 2.0,
        w,
        h,
    );
    if let Some((name, value)) = parameter_parts(label) {
        let _ = write!(
            svg,
            "<text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
             class=\"gate-name\" font-size=\"{:.1}px\" \
             dominant-baseline=\"central\">{}",
            cy - h * 0.22,
            opts.font_size * 0.9,
            &name[..1],
        );
        if name.len() > 1 {
            let _ = write!(
                svg,
                "<tspan baseline-shift=\"sub\" font-style=\"normal\" \
                 font-size=\"{:.1}px\">{}</tspan>",
                opts.font_size * 0.6,
                &name[1..],
            );
        }
        let _ = writeln!(
            svg,
            "</text><text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
             class=\"gate-param\" font-size=\"{:.1}px\" \
             dominant-baseline=\"central\">{}</text></g>",
            cy + h * 0.27,
            (opts.font_size * 0.6).max(8.0),
            escape_xml(&display_parameter(value)),
        );
    } else {
        let text_w = label.chars().count() as f64 * opts.font_size * CHAR_WIDTH_FACTOR;
        let inner = w - LABEL_PADDING * 0.5;
        let font_attr = if text_w > inner * 0.9 {
            let shrunk = (opts.font_size * inner * 0.9 / text_w).max(8.0);
            format!(" font-size=\"{shrunk:.1}px\"")
        } else {
            String::new()
        };
        let class = if label.chars().count() == 1 {
            "gate-name"
        } else {
            "gate-label"
        };
        let _ = writeln!(
            svg,
            "<text x=\"{cx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
             class=\"{class}\" dominant-baseline=\"central\"{font_attr}>{}</text></g>",
            escape_xml(label),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_gate_box(
    svg: &mut String,
    cx: f64,
    cy: f64,
    label: &str,
    opts: &SvgOptions,
    cat_class: &str,
    tip: &str,
    mc: &str,
) {
    emit_gate_box_inner(svg, cx, cy, label, opts, cat_class, tip, mc, false);
}

fn emit_cnot_target(svg: &mut String, cx: f64, cy: f64, opts: &SvgOptions, tip: &str, mc: &str) {
    let r = opts.gate_height * 0.3;
    let _ = writeln!(
        svg,
        "<g class=\"gate gate-ctrl{mc}\"><title>{tip}</title>\
         <circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r:.1}\" stroke-width=\"1.25\"/>\
         <line x1=\"{:.1}\" y1=\"{cy:.1}\" x2=\"{:.1}\" y2=\"{cy:.1}\" \
         stroke-width=\"1.25\"/>\
         <line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" \
         stroke-width=\"1.25\"/></g>",
        cx - r,
        cx + r,
        cy - r,
        cy + r,
    );
}

fn emit_swap_cross(svg: &mut String, cx: f64, cy: f64, opts: &SvgOptions, tip: &str, mc: &str) {
    let s = opts.gate_height * 0.16;
    let _ = writeln!(
        svg,
        "<g class=\"gate{mc}\"><title>{tip}</title>\
         <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke=\"var(--swap)\" stroke-width=\"1.25\" stroke-linecap=\"round\"/>\
         <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke=\"var(--swap)\" stroke-width=\"1.25\" stroke-linecap=\"round\"/></g>",
        cx - s,
        cy - s,
        cx + s,
        cy + s,
        cx - s,
        cy + s,
        cx + s,
        cy - s,
    );
}

fn emit_measure_box(svg: &mut String, cx: f64, cy: f64, opts: &SvgOptions, tip: &str, mc: &str) {
    let w = opts.gate_min_width.max(36.0);
    let h = opts.gate_height;
    let arc_y = cy + 2.0;
    let arc_r = w * 0.3;
    let _ = writeln!(
        svg,
        "<g class=\"gate gate-meas{mc}\"><title>{tip}</title>\
         <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
         rx=\"{GATE_CORNER_RADIUS}\" stroke-width=\"1\"/>\
         <path d=\"M {:.1} {:.1} A {:.1} {:.1} 0 0 1 {:.1} {:.1}\" \
         fill=\"none\" stroke-width=\"1\"/>\
         <line x1=\"{cx:.1}\" y1=\"{arc_y:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke-width=\"1\"/></g>",
        cx - w / 2.0,
        cy - h / 2.0,
        w,
        h,
        cx - arc_r,
        arc_y,
        arc_r,
        arc_r,
        cx + arc_r,
        arc_y,
        cx + arc_r * 0.6,
        cy - h * 0.3,
    );
}

fn render_svg_heatmap(moments: &[Vec<PlacedOp>], num_qubits: usize, opts: &SvgOptions) -> String {
    let theme = if opts.dark_mode { &DARK } else { &LIGHT };

    if moments.is_empty() || num_qubits == 0 {
        return empty_svg(theme);
    }

    let depth = moments.len();
    let n = num_qubits;

    let max_cols = 200usize;
    let max_rows = 200usize;
    let col_bucket = depth.div_ceil(max_cols).max(1);
    let row_bucket = n.div_ceil(max_rows).max(1);
    let actual_cols = depth.div_ceil(col_bucket);
    let actual_rows = n.div_ceil(row_bucket);
    let aggregated = row_bucket > 1 || col_bucket > 1;

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
    let binary = max_density == 1;
    let row_totals: Vec<usize> = grid.iter().map(|r| r.iter().sum()).collect();
    let col_totals: Vec<usize> = (0..actual_cols)
        .map(|c| grid.iter().map(|r| r[c]).sum())
        .collect();
    let max_row_total = *row_totals.iter().max().unwrap_or(&1).max(&1);
    let max_col_total = *col_totals.iter().max().unwrap_or(&1).max(&1);

    let cell_w = (640.0 / actual_cols as f64).clamp(4.0, 16.0);
    let cell_h = (320.0 / actual_rows as f64).clamp(4.0, 16.0);
    let row_digits = (n - 1).to_string().len();
    let row_label_chars = if row_bucket > 1 {
        row_digits * 2 + 1
    } else {
        row_digits
    };
    let label_pad = (row_label_chars as f64 * 6.0 + 34.0).max(70.0);
    let top_pad = 64.0;
    let spark_h = 24.0;
    let spark_w = 32.0;
    let mut notes = vec![String::from(
        "One count per wire touched; CX contributes 2.",
    )];
    if aggregated {
        let row_unit = if row_bucket == 1 { "qubit" } else { "qubits" };
        let col_unit = if col_bucket == 1 { "moment" } else { "moments" };
        notes.push(format!(
            "Cells: up to {row_bucket} {row_unit} × {col_bucket} {col_unit}."
        ));
        notes.push(String::from(
            "Counts are sums, not normalized for cell size.",
        ));
        let last_rows = n - (actual_rows - 1) * row_bucket;
        let last_cols = depth - (actual_cols - 1) * col_bucket;
        if last_rows < row_bucket || last_cols < col_bucket {
            let mut partial = Vec::new();
            if last_rows < row_bucket {
                let unit = if last_rows == 1 { "qubit" } else { "qubits" };
                let range = if last_rows == 1 {
                    (n - 1).to_string()
                } else {
                    format!("{}–{}", n - last_rows, n - 1)
                };
                partial.push(format!("final row q[{range}] ({last_rows} {unit})"));
            }
            if last_cols < col_bucket {
                let unit = if last_cols == 1 { "moment" } else { "moments" };
                let range = if last_cols == 1 {
                    (depth - 1).to_string()
                } else {
                    format!("{}–{}", depth - last_cols, depth - 1)
                };
                partial.push(format!("final column m[{range}] ({last_cols} {unit})"));
            }
            notes.push(format!("Partial {}.", partial.join("; ")));
        }
    }
    let bottom_pad = 76.0 + spark_h + (notes.len() - 1) as f64 * 16.0;
    let count_width = max_row_total.to_string().len() as f64 * 6.0 + 10.0;
    let right_pad = 98.0 + spark_w + count_width;

    let grid_w = actual_cols as f64 * cell_w;
    let grid_h = actual_rows as f64 * cell_h;
    let summary = if binary {
        format!("{n} qubits × {depth} moments · active/idle")
    } else {
        format!("{n} qubits × {depth} moments · max cell count {max_density}")
    };
    let mut total_w =
        (label_pad + grid_w + right_pad).max(label_pad + label_width(&summary, 11.0) + 16.0);
    for note in &notes {
        total_w = total_w.max(label_pad + label_width(note, 10.0) + 16.0);
    }
    let total_h = top_pad + grid_h + bottom_pad;

    let use_tooltips = (actual_rows * actual_cols) < 5000;

    let mut svg = String::with_capacity(actual_rows * actual_cols * 90 + 4096);

    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_w:.0}\" height=\"{total_h:.0}\" viewBox=\"0 0 {:.0} {:.0}\" \
         class=\"pq-circuit\" role=\"img\" aria-label=\"Gate activity counts: {n} qubits, {depth} moments\">",
        total_w, total_h,
    );
    emit_theme_vars(&mut svg, theme, opts.auto_theme);
    let _ = writeln!(
        svg,
        ".pq-circuit text{{font-family:Arial,Helvetica,sans-serif;\
         text-rendering:optimizeLegibility;}}\
         .pq-circuit .hm rect{{stroke:var(--bg);stroke-width:0.5;transition:filter 0.1s ease;}}\
         .pq-circuit .hm rect:hover{{stroke:var(--text);stroke-width:1;filter:brightness(1.15);cursor:crosshair;}}",
    );
    if opts.auto_theme {
        let _ = write!(
            svg,
            ".pq-circuit .hm rect,.pq-circuit .hm-key .hm-active{{fill:var(--hm-light);}}\
             .pq-circuit .hm-scale stop{{stop-color:var(--hm-light);}}\
             @media(prefers-color-scheme:dark){{\
             .pq-circuit .hm rect,.pq-circuit .hm-key .hm-active{{fill:var(--hm-dark);}}\
             .pq-circuit .hm-scale stop{{stop-color:var(--hm-dark);}}}}"
        );
    }
    svg.push_str("</style>\n");
    let _ = writeln!(
        svg,
        "<rect width=\"100%\" height=\"100%\" fill=\"var(--bg)\"/>",
    );

    let _ = writeln!(
        svg,
        "<text x=\"{:.0}\" y=\"20\" font-size=\"14\" font-weight=\"600\" \
         fill=\"var(--text)\">Gate activity counts</text>",
        label_pad,
    );
    let _ = writeln!(
        svg,
        "<text x=\"{:.0}\" y=\"36\" font-size=\"11\" fill=\"var(--text)\">\
         {summary}</text>",
        label_pad,
    );

    let cw = cell_w - 1.0;
    let ch = cell_h - 1.0;
    let crisp = cell_w < 6.0;
    let (cell_rx, cell_sr) = if crisp {
        ("0", " shape-rendering=\"crispEdges\"")
    } else {
        ("1", "")
    };
    svg.push_str("<g class=\"hm\">\n");
    for (r, row) in grid.iter().enumerate() {
        for (c, &count) in row.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let ratio = count as f64 / max_density as f64;
            let fill = heatmap_color(ratio, opts.dark_mode && !opts.auto_theme);
            let theme_style = if opts.auto_theme {
                format!(
                    " style=\"--hm-light:{};--hm-dark:{}\"",
                    heatmap_color(ratio, false),
                    heatmap_color(ratio, true),
                )
            } else {
                String::new()
            };
            let x = label_pad + c as f64 * cell_w;
            let y = top_pad + r as f64 * cell_h;
            if use_tooltips {
                let q_lo = r * row_bucket;
                let q_hi = ((r + 1) * row_bucket).min(n) - 1;
                let m_lo = c * col_bucket;
                let m_hi = ((c + 1) * col_bucket).min(depth) - 1;
                let _ = writeln!(
                    svg,
                    "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cw}\" height=\"{ch}\" \
                     rx=\"{cell_rx}\"{cell_sr} fill=\"{fill}\"{theme_style}>\
                     <title>q[{q_lo}..{q_hi}] m[{m_lo}..{m_hi}]: activity {count}</title></rect>",
                );
            } else {
                let _ = writeln!(
                    svg,
                    "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cw}\" height=\"{ch}\" \
                     rx=\"{cell_rx}\"{cell_sr} fill=\"{fill}\"{theme_style}/>",
                );
            }
        }
    }
    svg.push_str("</g>\n");

    let _ = writeln!(
        svg,
        "<rect x=\"{:.0}\" y=\"{:.0}\" width=\"{:.0}\" height=\"{:.0}\" \
         fill=\"none\" stroke=\"var(--wire)\" stroke-width=\"0.5\" \
         shape-rendering=\"crispEdges\"/>",
        label_pad, top_pad, grid_w, grid_h,
    );

    let label_font = 10.0;
    let tick_spacing_rows = (18.0 / cell_h).ceil() as usize;
    for r in (0..actual_rows).step_by(tick_spacing_rows) {
        let q_start = r * row_bucket;
        let q_end = ((r + 1) * row_bucket).min(n) - 1;
        let label = if q_start == q_end {
            q_start.to_string()
        } else {
            format!("{q_start}–{q_end}")
        };
        let y = top_pad + r as f64 * cell_h + cell_h / 2.0;
        let _ = writeln!(
            svg,
            "<text x=\"{:.0}\" y=\"{y:.1}\" text-anchor=\"end\" \
             font-size=\"{label_font:.0}\" fill=\"var(--text)\" \
             dominant-baseline=\"central\">{label}</text>",
            label_pad - 4.0,
        );
    }

    let col_digits = (depth - 1).to_string().len();
    let col_label_chars = if col_bucket > 1 {
        col_digits * 2 + 1
    } else {
        col_digits
    };
    let tick_spacing_cols =
        ((col_label_chars as f64 * 5.4 + 8.0).max(36.0) / cell_w).ceil() as usize;
    let axis_y = top_pad + grid_h + spark_h + 32.0;
    for c in (0..actual_cols).step_by(tick_spacing_cols) {
        let m = c * col_bucket;
        let m_end = ((c + 1) * col_bucket).min(depth) - 1;
        let label = if m == m_end {
            m.to_string()
        } else {
            format!("{m}–{m_end}")
        };
        let x = label_pad + c as f64 * cell_w;
        let _ = writeln!(
            svg,
            "<text x=\"{x:.1}\" y=\"{axis_y:.0}\" text-anchor=\"middle\" \
             font-size=\"9\" fill=\"var(--text)\">{label}</text>",
        );
    }

    let _ = writeln!(
        svg,
        "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" \
         font-size=\"10\" fill=\"var(--text)\">moment</text>",
        label_pad + grid_w / 2.0,
        axis_y + 14.0,
    );
    let qx = 18.0;
    let qy = top_pad + grid_h / 2.0;
    let _ = writeln!(
        svg,
        "<text x=\"{qx:.0}\" y=\"{qy:.0}\" text-anchor=\"middle\" \
         font-size=\"10\" fill=\"var(--text)\" transform=\"rotate(-90 {qx:.0} {qy:.0})\">\
         qubit</text>",
    );

    let spark_x = label_pad + grid_w + 4.0;
    let count_spacing_rows = (12.0 / cell_h).ceil() as usize;
    let _ = writeln!(
        svg,
        "<text x=\"{spark_x:.1}\" y=\"{:.1}\" font-size=\"10\" fill=\"var(--text)\">count</text>",
        top_pad - 8.0,
    );
    for (r, &total) in row_totals.iter().enumerate() {
        let bar_w = (total as f64 / max_row_total as f64) * spark_w;
        let y = top_pad + r as f64 * cell_h;
        let q_lo = r * row_bucket;
        let q_hi = ((r + 1) * row_bucket).min(n) - 1;
        let _ = writeln!(
            svg,
            "<rect x=\"{spark_x:.1}\" y=\"{y:.1}\" width=\"{bar_w:.1}\" height=\"{ch:.0}\" \
             rx=\"0.5\" fill=\"var(--wire)\" opacity=\"0.4\">\
             <title>q[{q_lo}..{q_hi}]: activity {total}</title></rect>",
        );
        if r % count_spacing_rows == 0 {
            let _ = writeln!(
                svg,
                "<text x=\"{:.1}\" y=\"{:.1}\" class=\"hm-row-total\" font-size=\"10\" \
                 dominant-baseline=\"central\" fill=\"var(--text)\" \
                 aria-label=\"q[{q_lo}..{q_hi}]: activity {total}\">{total}</text>",
                spark_x + spark_w + 6.0,
                y + cell_h / 2.0,
            );
        }
    }

    let spark_y = top_pad + grid_h + 4.0;
    let _ = writeln!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-size=\"10\" \
         fill=\"var(--text)\">count</text>",
        label_pad - 8.0,
        spark_y + spark_h + 12.0,
    );
    let count_spacing_cols =
        ((max_col_total.to_string().len() as f64 * 6.0 + 3.0) / cell_w).ceil() as usize;
    for (c, &total) in col_totals.iter().enumerate() {
        let bar_h = (total as f64 / max_col_total as f64) * spark_h;
        let x = label_pad + c as f64 * cell_w;
        let y_bar = spark_y + spark_h - bar_h;
        let m_lo = c * col_bucket;
        let m_hi = ((c + 1) * col_bucket).min(depth) - 1;
        let _ = writeln!(
            svg,
            "<rect x=\"{x:.1}\" y=\"{y_bar:.1}\" width=\"{cw:.0}\" height=\"{bar_h:.1}\" \
             rx=\"0.5\" fill=\"var(--wire)\" opacity=\"0.4\">\
             <title>m[{m_lo}..{m_hi}]: activity {total}</title></rect>",
        );
        if c % count_spacing_cols == 0 {
            let _ = writeln!(
                svg,
                "<text x=\"{:.1}\" y=\"{:.1}\" class=\"hm-col-total\" text-anchor=\"middle\" \
                 font-size=\"10\" fill=\"var(--text)\" \
                 aria-label=\"m[{m_lo}..{m_hi}]: activity {total}\">{total}</text>",
                x + cell_w / 2.0,
                spark_y + spark_h + 12.0,
            );
        }
    }

    let legend_x = spark_x + spark_w + count_width + 12.0;
    let legend_y = top_pad;
    if binary {
        let fill = heatmap_color(1.0, opts.dark_mode && !opts.auto_theme);
        let theme_style = if opts.auto_theme {
            format!(
                " style=\"--hm-light:{};--hm-dark:{}\"",
                heatmap_color(1.0, false),
                heatmap_color(1.0, true),
            )
        } else {
            String::new()
        };
        let _ = writeln!(
            svg,
            "<g class=\"hm-key\" aria-label=\"Cell activity\">\
             <rect x=\"{legend_x:.1}\" y=\"{legend_y:.1}\" width=\"10\" height=\"10\" \
             class=\"hm-active\" fill=\"{fill}\"{theme_style}/>\
             <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"var(--text)\">Active</text>\
             <rect x=\"{legend_x:.1}\" y=\"{:.1}\" width=\"10\" height=\"10\" \
             fill=\"var(--bg)\" stroke=\"var(--wire)\" stroke-width=\"0.5\"/>\
             <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"var(--text)\">Idle</text></g>",
            legend_x + 16.0,
            legend_y + 9.0,
            legend_y + 22.0,
            legend_x + 16.0,
            legend_y + 31.0,
        );
    } else {
        let legend_h = grid_h.clamp(64.0, 120.0);
        let legend_w = 14.0;
        let legend_id = "hm-legend";
        let _ = write!(
            svg,
            "<defs><linearGradient id=\"{legend_id}\" class=\"hm-scale\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">"
        );
        for i in 0..5 {
            let offset = i as f64 / 4.0 * 100.0;
            let ratio = 1.0 - i as f64 / 4.0;
            let fill = heatmap_color(ratio, opts.dark_mode && !opts.auto_theme);
            let theme_style = if opts.auto_theme {
                format!(
                    " style=\"--hm-light:{};--hm-dark:{}\"",
                    heatmap_color(ratio, false),
                    heatmap_color(ratio, true),
                )
            } else {
                String::new()
            };
            let _ = write!(
                svg,
                "<stop offset=\"{offset:.0}%\" stop-color=\"{fill}\"{theme_style}/>"
            );
        }
        let _ = writeln!(svg, "</linearGradient></defs>");
        let _ = writeln!(
            svg,
            "<rect x=\"{legend_x:.0}\" y=\"{legend_y:.0}\" width=\"{legend_w}\" \
         height=\"{legend_h:.0}\" rx=\"2\" fill=\"url(#{legend_id})\" \
         stroke=\"var(--wire)\" stroke-width=\"0.5\"/>",
        );
        let _ = writeln!(
            svg,
            "<text x=\"{:.0}\" y=\"{:.1}\" font-size=\"9\" fill=\"var(--text)\">{max_density}</text>",
            legend_x + legend_w + 4.0,
            legend_y + 6.0,
        );
        let mid_val = if max_density % 2 == 0 {
            (max_density / 2).to_string()
        } else {
            format!("{:.1}", max_density as f64 / 2.0)
        };
        let _ = writeln!(
            svg,
            "<text x=\"{:.0}\" y=\"{:.1}\" font-size=\"9\" fill=\"var(--text)\">{mid_val}</text>",
            legend_x + legend_w + 4.0,
            legend_y + legend_h / 2.0 + 3.0,
        );
        let _ = writeln!(
            svg,
            "<text x=\"{:.0}\" y=\"{:.1}\" font-size=\"9\" fill=\"var(--text)\">0</text>",
            legend_x + legend_w + 4.0,
            legend_y + legend_h,
        );
    }

    for (i, note) in notes.iter().enumerate() {
        let _ = writeln!(
            svg,
            "<text x=\"{label_pad:.1}\" y=\"{:.1}\" class=\"hm-note\" font-size=\"10\" \
             fill=\"var(--text)\">{}</text>",
            axis_y + 36.0 + i as f64 * 16.0,
            escape_xml(note),
        );
    }

    svg.push_str("</svg>\n");
    svg
}

impl Circuit {
    /// Render the circuit as a self-contained SVG wire diagram.
    pub fn to_svg(&self, opts: &SvgOptions) -> String {
        let moments = barrier_columns(assign_moments(self), true);
        render_svg(&moments, self.num_qubits, self.num_classical_bits, opts)
    }

    /// Render the gate-density heatmap (qubits by moments) as self-contained
    /// SVG with marginal activity bars and a color legend.
    pub fn to_svg_heatmap(&self, opts: &SvgOptions) -> String {
        let moments = assign_moments(self);
        render_svg_heatmap(&moments, self.num_qubits, opts)
    }
}

#[cfg(test)]
#[path = "svg_tests.rs"]
mod tests;
