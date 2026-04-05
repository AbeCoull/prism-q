use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use super::draw::{assign_moments, OpKind, PlacedOp};
use super::Circuit;

const GATE_CORNER_RADIUS: f64 = 1.0;
const JUNCTION_DOT_RADIUS: f64 = 1.5;
const SWAP_CROSS_SIZE: f64 = 7.0;
const LABEL_GAP: f64 = 8.0;
const CHAR_WIDTH_FACTOR: f64 = 0.62;
const LABEL_PADDING: f64 = 16.0;

pub struct SvgOptions {
    pub wire_spacing: f64,
    pub moment_width: f64,
    pub gate_height: f64,
    pub gate_min_width: f64,
    pub padding_left: f64,
    pub padding_top: f64,
    pub padding_right: f64,
    pub padding_bottom: f64,
    pub font_size: f64,
    pub control_radius: f64,
    pub max_moments: Option<usize>,
    pub max_qubits: Option<usize>,
    pub show_idle_wires: bool,
    pub show_barriers: bool,
    pub dark_mode: bool,
    pub auto_theme: bool,
    pub animate: bool,
    pub show_legend: bool,
    pub show_stats_header: bool,
    pub compact: bool,
    pub ellipsis_mode: Option<(usize, usize)>,
    pub show_topology: bool,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            wire_spacing: 40.0,
            moment_width: 60.0,
            gate_height: 28.0,
            gate_min_width: 36.0,
            padding_left: 60.0,
            padding_top: 20.0,
            padding_right: 20.0,
            padding_bottom: 20.0,
            font_size: 13.0,
            control_radius: 3.5,
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
    gradient_top: &'static str,
}

struct Theme {
    bg: &'static str,
    wire: &'static str,
    text: &'static str,
    stripe: &'static str,
    control_dot: &'static str,
    control_border: &'static str,
    swap_stroke: &'static str,
    barrier: &'static str,
    glow_color: &'static str,
    shadow_opacity: f64,
    standard: GateStyle,
    parametric: GateStyle,
    phase: GateStyle,
    controlled: GateStyle,
    measure: GateStyle,
    multi: GateStyle,
}

const LIGHT: Theme = Theme {
    bg: "#ffffff",
    wire: "#c4c8cc",
    text: "#2b2f33",
    stripe: "#f6f7f8",
    control_dot: "#2b2f33",
    control_border: "#ffffff",
    swap_stroke: "#6b7075",
    barrier: "#dcdfe2",
    glow_color: "#5a5e63",
    shadow_opacity: 0.05,
    standard: GateStyle {
        fill: "#c8daf0",
        stroke: "#3872a8",
        text: "#1e4d7a",
        gradient_top: "#dce8f6",
    },
    parametric: GateStyle {
        fill: "#d8c8ee",
        stroke: "#6840a0",
        text: "#42267a",
        gradient_top: "#e6d8f4",
    },
    phase: GateStyle {
        fill: "#c0e4d0",
        stroke: "#2d7a52",
        text: "#1a5438",
        gradient_top: "#d4eedf",
    },
    controlled: GateStyle {
        fill: "#ecd8b8",
        stroke: "#98682e",
        text: "#6a4418",
        gradient_top: "#f2e4cc",
    },
    measure: GateStyle {
        fill: "#eac8ca",
        stroke: "#a03840",
        text: "#701820",
        gradient_top: "#f2d8da",
    },
    multi: GateStyle {
        fill: "#d0c4e8",
        stroke: "#5c40a0",
        text: "#3a2070",
        gradient_top: "#ddd4f0",
    },
};

const DARK: Theme = Theme {
    bg: "#000000",
    wire: "#2a2d31",
    text: "#b0b4ba",
    stripe: "#07080a",
    control_dot: "#d0d3d8",
    control_border: "#000000",
    swap_stroke: "#6b7075",
    barrier: "#2a2d31",
    glow_color: "#7a7e84",
    shadow_opacity: 0.3,
    standard: GateStyle {
        fill: "#122844",
        stroke: "#5a9ed6",
        text: "#90c4ec",
        gradient_top: "#1a3858",
    },
    parametric: GateStyle {
        fill: "#20123a",
        stroke: "#9868d0",
        text: "#c4a0ec",
        gradient_top: "#2e1e50",
    },
    phase: GateStyle {
        fill: "#0e2818",
        stroke: "#40b878",
        text: "#80e0b0",
        gradient_top: "#183c28",
    },
    controlled: GateStyle {
        fill: "#281c08",
        stroke: "#d09848",
        text: "#ecc888",
        gradient_top: "#382c14",
    },
    measure: GateStyle {
        fill: "#281014",
        stroke: "#c85060",
        text: "#ec9098",
        gradient_top: "#381c22",
    },
    multi: GateStyle {
        fill: "#1a1030",
        stroke: "#8860c8",
        text: "#b898e8",
        gradient_top: "#261c44",
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
            "--bg:{};--wire:{};--text:{};--stripe:{};--ctrl-dot:{};--ctrl-bdr:{};--swap:{};--barrier:{};",
            self.bg, self.wire, self.text, self.stripe,
            self.control_dot, self.control_border, self.swap_stroke, self.barrier,
        );
        let _ = write!(
            svg,
            "--shadow-filter:drop-shadow(0 0.5px 0.5px rgba(0,0,0,{:.2}));\
             --hover-filter:drop-shadow(0 0 3px {}50);",
            self.shadow_opacity, self.glow_color,
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
                "--{prefix}-top:{};--{prefix}-fill:{};--{prefix}-str:{};--{prefix}-txt:{};",
                sty.gradient_top, sty.fill, sty.stroke, sty.text,
            );
        }
    }
}

fn classify_gate(label: &str) -> GateCategory {
    match label {
        "S" | "T" | "Sdg" | "Tdg" => GateCategory::Phase,
        "H" | "X" | "Y" | "Z" | "Id" | "SX" | "SXdg" => GateCategory::Standard,
        _ if label.starts_with("Rx(")
            || label.starts_with("Ry(")
            || label.starts_with("Rz(")
            || label.starts_with("P(")
            || label.starts_with("Rzz(") =>
        {
            GateCategory::Parametric
        }
        _ => GateCategory::Standard,
    }
}

fn heatmap_stops(dark_mode: bool) -> [(f64, f64, f64); 5] {
    if dark_mode {
        [
            (8.0, 8.0, 18.0),
            (20.0, 50.0, 130.0),
            (50.0, 160.0, 140.0),
            (230.0, 200.0, 50.0),
            (252.0, 250.0, 230.0),
        ]
    } else {
        [
            (240.0, 244.0, 250.0),
            (100.0, 170.0, 220.0),
            (60.0, 150.0, 100.0),
            (220.0, 170.0, 50.0),
            (160.0, 30.0, 40.0),
        ]
    }
}

fn heatmap_color(t: f64, dark_mode: bool) -> String {
    let t = t.clamp(0.0, 1.0);
    let stops = heatmap_stops(dark_mode);
    let idx = (t * 4.0).min(3.999);
    let i = idx as usize;
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

fn empty_svg(theme: &Theme) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 200 60\">\
         <rect width=\"100%\" height=\"100%\" fill=\"{}\"/>\
         <text x=\"100\" y=\"35\" text-anchor=\"middle\" \
         font-family=\"monospace\" font-size=\"14\" fill=\"{}\">empty circuit</text>\
         </svg>",
        theme.bg, theme.text,
    )
}

fn tooltip(op: &PlacedOp) -> String {
    let qs: Vec<String> = op.qubits.iter().map(|q| format!("q[{q}]")).collect();
    match &op.kind {
        OpKind::Controlled { controls, target } => {
            let ctrls: Vec<String> = controls.iter().map(|c| format!("q[{c}]")).collect();
            format!(
                "{}: ctrl {} → target q[{}]",
                op.label,
                ctrls.join(", "),
                target
            )
        }
        OpKind::Swap => format!("SWAP {} ↔ {}", qs[0], qs[1]),
        OpKind::Measure { cbit } => format!("Measure {} → c[{}]", qs[0], cbit),
        OpKind::Barrier => format!("Barrier on {}", qs.join(", ")),
        _ => format!("{} on {}", op.label, qs.join(", ")),
    }
}

const CLASSICAL_WIRE_SPACING: f64 = 30.0;
const CLASSICAL_WIRE_GAP: f64 = 20.0;

fn render_svg(
    moments: &[Vec<PlacedOp>],
    num_qubits: usize,
    num_classical_bits: usize,
    opts: &SvgOptions,
) -> String {
    let compact_opts;
    let opts = if opts.compact {
        compact_opts = SvgOptions {
            wire_spacing: 28.0,
            moment_width: 44.0,
            gate_height: 22.0,
            gate_min_width: 28.0,
            padding_left: 48.0,
            padding_top: 14.0,
            padding_right: 14.0,
            padding_bottom: 14.0,
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

    let max_moments = opts.max_moments.unwrap_or(moments.len()).min(moments.len());
    let moments = &moments[..max_moments];

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
        if first + last < moments.len() {
            Some((first, moments.len() - last))
        } else {
            None
        }
    });

    let mut col_widths: Vec<f64> = Vec::with_capacity(moments.len());
    for moment in moments {
        let mut max_w = opts.gate_min_width;
        for op in moment {
            let w = label_width(&op.label, opts.font_size);
            if w > max_w {
                max_w = w;
            }
        }
        col_widths.push(max_w.max(opts.moment_width));
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

    let circuit_width = cx_pos + opts.padding_right;
    let header_extra = if opts.show_stats_header { 20.0 } else { 0.0 };
    let legend_extra = if opts.show_legend {
        let swatch_h = opts.font_size * 0.92 * 0.9;
        4.0 + swatch_h + 8.0 + opts.padding_bottom * 0.5
    } else {
        0.0
    };
    let pad_top = opts.padding_top + header_extra;
    let classical_extra = if num_classical_bits > 0 {
        CLASSICAL_WIRE_GAP + (num_classical_bits as f64 - 1.0) * CLASSICAL_WIRE_SPACING + 10.0
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
        last_quantum_y + CLASSICAL_WIRE_GAP + cbit as f64 * CLASSICAL_WIRE_SPACING
    };

    let mut svg = String::with_capacity(moments.len() * num_rows * 150 + 4096);

    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {:.1} {:.1}\" \
         role=\"img\" aria-label=\"Quantum circuit: {} qubits, {} moments\">",
        total_width,
        total_height,
        num_qubits,
        moments.len(),
    );

    let _ = write!(svg, "<defs>");
    for (prefix, id) in [
        ("std", "g-std"),
        ("par", "g-par"),
        ("pha", "g-pha"),
        ("ctrl", "g-ctrl"),
        ("meas", "g-meas"),
        ("mfus", "g-mfus"),
    ] {
        let _ = write!(
            svg,
            "<linearGradient id=\"{id}\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\
             <stop offset=\"0%\" stop-color=\"var(--{prefix}-top)\"/>\
             <stop offset=\"100%\" stop-color=\"var(--{prefix}-fill)\"/>\
             </linearGradient>",
        );
    }
    let _ = write!(
        svg,
        "<marker id=\"arrow-meas\" viewBox=\"0 0 6 6\" refX=\"3\" refY=\"3\" \
         markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\">\
         <path d=\"M 0 0 L 6 3 L 0 6 z\" fill=\"var(--meas-str)\"/></marker>",
    );
    let _ = writeln!(svg, "</defs>");

    let _ = write!(svg, "<style>:root{{");
    if opts.auto_theme {
        LIGHT.emit_css_vars(&mut svg);
    } else {
        theme.emit_css_vars(&mut svg);
    }
    let _ = write!(svg, "}}");
    if opts.auto_theme {
        let _ = write!(svg, "@media(prefers-color-scheme:dark){{:root{{");
        DARK.emit_css_vars(&mut svg);
        let _ = write!(svg, "}}}}");
    }
    let anim_gate = if opts.animate {
        "animation:gate-in 0.3s ease both;"
    } else {
        ""
    };
    let _ = write!(
        svg,
        "text{{font-family:Inter,-apple-system,system-ui,sans-serif;\
         font-size:{:.0}px;text-rendering:optimizeLegibility;\
         font-variant-numeric:tabular-nums;}}\
         .gate text{{font-family:'JetBrains Mono','Fira Code',Consolas,monospace;\
         font-weight:600;letter-spacing:-0.01em;}}\
         .gate{{filter:var(--shadow-filter);cursor:pointer;\
         transition:filter 0.15s ease,opacity 0.1s ease;{anim_gate}}}\
         .gate:hover{{filter:var(--hover-filter);opacity:0.92;}}\
         .ctrl{{cursor:pointer;transition:r 0.1s ease;{anim_gate}}}\
         .ctrl:hover circle{{r:{:.1};}}",
        opts.font_size,
        opts.control_radius + 1.5,
    );
    for (prefix, id) in [
        ("std", "g-std"),
        ("par", "g-par"),
        ("pha", "g-pha"),
        ("ctrl", "g-ctrl"),
        ("meas", "g-meas"),
        ("mfus", "g-mfus"),
    ] {
        let _ = write!(
            svg,
            ".gate-{prefix} rect,.gate-{prefix} circle\
             {{fill:url(#{id});stroke:var(--{prefix}-str);}}\
             .gate-{prefix} text{{fill:var(--{prefix}-txt);}}",
        );
    }
    let _ = write!(
        svg,
        ".gate-ctrl line{{stroke:var(--ctrl-str);}}\
         .gate-meas path,.gate-meas line{{stroke:var(--meas-txt);}}"
    );
    let _ = write!(
        svg,
        ".qlabel{{font-family:'JetBrains Mono','Fira Code',Consolas,monospace;\
         font-weight:500;font-size:{:.1}px;letter-spacing:0.02em;fill:var(--text);}}",
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
            let _ = write!(svg, ".m{i}{{animation-delay:{}ms}}", i * 30);
        }
        let _ = write!(
            svg,
            "@keyframes wire-flow{{to{{stroke-dashoffset:-12}}}}\
             .wire-anim{{stroke-dasharray:1 11;animation:wire-flow 2s linear infinite;}}"
        );
        let _ = write!(
            svg,
            "@media(prefers-reduced-motion:reduce){{\
             .gate,.ctrl{{animation:none!important;}}\
             .wire-anim{{animation:none!important;stroke-dasharray:none;}}}}"
        );
    }
    let _ = writeln!(svg, "</style>");

    if opts.show_stats_header {
        let gate_count: usize = moments
            .iter()
            .flat_map(|m| m.iter())
            .filter(|op| !matches!(op.kind, OpKind::Barrier | OpKind::Measure { .. }))
            .count();
        let _ = writeln!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" class=\"qlabel\" \
             dominant-baseline=\"central\">{} qubit{} \u{00b7} {} gate{} \u{00b7} depth {}</text>",
            opts.padding_left,
            opts.padding_top + header_extra / 2.0,
            num_qubits,
            if num_qubits == 1 { "" } else { "s" },
            gate_count,
            if gate_count == 1 { "" } else { "s" },
            moments.len(),
        );
    }

    svg.push_str("<g id=\"layer-bg\" role=\"presentation\">\n");
    let _ = writeln!(
        svg,
        "<rect width=\"100%\" height=\"100%\" fill=\"var(--bg)\"/>",
    );
    for row in (1..num_rows).step_by(2) {
        let y = wire_y(row) - opts.wire_spacing / 2.0;
        let _ = writeln!(
            svg,
            "<rect x=\"0\" y=\"{y:.1}\" width=\"100%\" height=\"{:.1}\" \
             fill=\"var(--stripe)\" shape-rendering=\"crispEdges\"/>",
            opts.wire_spacing,
        );
    }
    svg.push_str("</g>\n");

    svg.push_str("<g id=\"layer-wires\">\n");
    for (row, &q) in show_qubits.iter().enumerate() {
        let y = wire_y(row);
        let _ = writeln!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" \
             dominant-baseline=\"central\" class=\"qlabel\">q[{}]</text>",
            opts.padding_left - LABEL_GAP,
            y,
            q,
        );
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
        let _ = writeln!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" \
             dominant-baseline=\"central\" class=\"qlabel\">c[{}]</text>",
            opts.padding_left - LABEL_GAP,
            cy,
            c,
        );
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
        let center_x = mx + mw / 2.0;
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
                    let cat = classify_gate(&op.label);
                    present_cats.insert(cat);
                    let cls = cat.css_class();
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_gate_box(&mut svg, center_x, cy, &op.label, opts, cls, &tip, &mc);
                        }
                    }
                }

                OpKind::MultiFused => {
                    present_cats.insert(GateCategory::Multi);
                    let cls = GateCategory::Multi.css_class();
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_gate_box(&mut svg, center_x, cy, &op.label, opts, cls, &tip, &mc);
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
                        for row in (min_row + 1)..max_row {
                            if !all_rows.contains(&row) {
                                let jy = wire_y(row);
                                let _ = writeln!(
                                    svg,
                                    "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{JUNCTION_DOT_RADIUS}\" \
                                     fill=\"var(--wire)\"/>",
                                    center_x, jy,
                                );
                            }
                        }
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
                            let tgt_label = match op.label.as_str() {
                                "CZ" => "Z",
                                "CU" => "U",
                                other if other.starts_with("MCU") => "U",
                                other => other,
                            };
                            let cls = GateCategory::Controlled.css_class();
                            emit_gate_box(&mut svg, center_x, cy, tgt_label, opts, cls, &tip, &mc);
                        }
                    }
                }

                OpKind::TwoQubit => {
                    let rows: Vec<usize> = op.qubits.iter().filter_map(|&q| row_of(q)).collect();
                    let cat = classify_gate(&op.label);
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
                            emit_gate_box(&mut svg, center_x, cy, &op.label, opts, cls, &tip, &mc);
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
                        for row in (min_row + 1)..max_row {
                            let jy = wire_y(row);
                            let _ = writeln!(
                                svg,
                                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{JUNCTION_DOT_RADIUS}\" \
                                 fill=\"var(--wire)\"/>",
                                center_x, jy,
                            );
                        }
                    }

                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            emit_swap_cross(&mut svg, center_x, cy, &tip, &mc);
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

                OpKind::Measure { cbit } => {
                    present_cats.insert(GateCategory::Measure);
                    if let Some(row) = op.qubits.first().and_then(|&q| row_of(q)) {
                        let cy = wire_y(row);
                        emit_measure_box(&mut svg, center_x, cy, opts, &tip, &mc);
                        if *cbit < num_classical_bits {
                            let cbit_y = classical_y(*cbit);
                            let arrow_top = cy + opts.gate_height / 2.0;
                            let _ = writeln!(
                                svg,
                                "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                                 stroke=\"var(--meas-str)\" stroke-width=\"0.75\" \
                                 stroke-dasharray=\"3,3\" marker-end=\"url(#arrow-meas)\"/>",
                                center_x,
                                arrow_top,
                                center_x,
                                cbit_y - 1.0,
                            );
                        }
                    }
                }

                OpKind::Conditional { cbit_label } => {
                    let cat = classify_gate(&op.label);
                    present_cats.insert(cat);
                    for &q in &op.qubits {
                        if let Some(row) = row_of(q) {
                            let cy = wire_y(row);
                            let lbl = format!("{}?{}", cbit_label, op.label);
                            let cls = cat.css_class();
                            emit_gate_box_inner(
                                &mut svg, center_x, cy, &lbl, opts, cls, &tip, &mc, true,
                            );
                            let cbit_idx = cbit_label
                                .strip_prefix("c[")
                                .and_then(|s| s.strip_suffix(']'))
                                .and_then(|s| s.parse::<usize>().ok());
                            if let Some(ci) = cbit_idx {
                                if ci < num_classical_bits {
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
    let text_w = cat.display_name().len() as f64 * font_size * 0.52;
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
    let pill_pad_x = 8.0;
    let pill_pad_y = 4.0;
    let pill_h = swatch_size + pill_pad_y * 2.0;
    let pill_w = (total_legend_w + pill_pad_x * 2.0).min(total_width - 4.0);
    let _ = writeln!(
        svg,
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{pill_w:.1}\" height=\"{pill_h:.1}\" \
         rx=\"{:.1}\" fill=\"var(--bg)\" fill-opacity=\"0.92\" \
         stroke=\"var(--wire)\" stroke-width=\"0.5\" stroke-opacity=\"0.4\"/>",
        start_x - pill_pad_x,
        legend_y - pill_pad_y,
        pill_h / 2.0,
    );
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
             fill=\"url(#g-{prefix})\" stroke=\"var(--{prefix}-str)\" \
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
    let w = label_width(label, opts.font_size).max(opts.gate_min_width);
    let h = opts.gate_height;
    let text_w = label.chars().count() as f64 * opts.font_size * CHAR_WIDTH_FACTOR;
    let inner = w - LABEL_PADDING * 0.5;
    let font_attr = if text_w > inner * 0.9 {
        let shrunk = (opts.font_size * inner * 0.9 / text_w).max(8.0);
        format!(" font-size=\"{shrunk:.1}px\"")
    } else {
        String::new()
    };
    let dash_attr = if dashed {
        " stroke-dasharray=\"4,2\""
    } else {
        ""
    };
    let _ = writeln!(
        svg,
        "<g class=\"gate {cat_class}{mc}\"><title>{tip}</title>\
         <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
         rx=\"{GATE_CORNER_RADIUS}\" stroke-width=\"1\"{dash_attr}/>\
         <text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
         dominant-baseline=\"central\"{font_attr}>{}</text></g>",
        cx - w / 2.0,
        cy - h / 2.0,
        w,
        h,
        cx,
        cy,
        escape_xml(label),
    );
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
    let r = opts.gate_height / 2.0 - 2.0;
    let _ = writeln!(
        svg,
        "<g class=\"gate gate-ctrl{mc}\"><title>{tip}</title>\
         <circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r:.1}\" stroke-width=\"1\"/>\
         <line x1=\"{:.1}\" y1=\"{cy:.1}\" x2=\"{:.1}\" y2=\"{cy:.1}\" \
         stroke-width=\"1\" shape-rendering=\"crispEdges\"/>\
         <line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" \
         stroke-width=\"1\" shape-rendering=\"crispEdges\"/></g>",
        cx - r,
        cx + r,
        cy - r,
        cy + r,
    );
}

fn emit_swap_cross(svg: &mut String, cx: f64, cy: f64, tip: &str, mc: &str) {
    let s = SWAP_CROSS_SIZE;
    let _ = writeln!(
        svg,
        "<g class=\"gate{mc}\"><title>{tip}</title>\
         <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke=\"var(--swap)\" stroke-width=\"1.5\" stroke-linecap=\"round\"/>\
         <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke=\"var(--swap)\" stroke-width=\"1.5\" stroke-linecap=\"round\"/></g>",
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

    let cell_w = 4.0_f64;
    let cell_h = 4.0_f64;
    let label_pad = 70.0;
    let top_pad = 50.0;
    let spark_h = 24.0;
    let spark_w = 32.0;
    let bottom_pad = 50.0 + spark_h;
    let right_pad = 90.0 + spark_w;

    let grid_w = actual_cols as f64 * cell_w;
    let grid_h = actual_rows as f64 * cell_h;
    let total_w = label_pad + grid_w + right_pad;
    let total_h = top_pad + grid_h + bottom_pad;

    let row_totals: Vec<usize> = grid.iter().map(|r| r.iter().sum()).collect();
    let col_totals: Vec<usize> = (0..actual_cols)
        .map(|c| grid.iter().map(|r| r[c]).sum())
        .collect();
    let max_row_total = *row_totals.iter().max().unwrap_or(&1).max(&1);
    let max_col_total = *col_totals.iter().max().unwrap_or(&1).max(&1);

    let use_tooltips = (actual_rows * actual_cols) < 5000;

    let mut svg = String::with_capacity(actual_rows * actual_cols * 90 + 4096);

    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {:.0} {:.0}\" \
         role=\"img\" aria-label=\"Gate density heatmap: {n} qubits, {depth} moments\">",
        total_w, total_h,
    );
    let _ = write!(svg, "<style>:root{{");
    if opts.auto_theme {
        LIGHT.emit_css_vars(&mut svg);
    } else {
        theme.emit_css_vars(&mut svg);
    }
    let _ = write!(svg, "}}");
    if opts.auto_theme {
        let _ = write!(svg, "@media(prefers-color-scheme:dark){{:root{{");
        DARK.emit_css_vars(&mut svg);
        let _ = write!(svg, "}}}}");
    }
    let _ = writeln!(
        svg,
        "text{{font-family:Inter,-apple-system,system-ui,sans-serif;\
         text-rendering:optimizeLegibility;}}\
         .hm rect{{stroke:var(--bg);stroke-width:0.5;transition:filter 0.1s ease;}}\
         .hm rect:hover{{stroke:var(--text);stroke-width:1;filter:brightness(1.15);cursor:crosshair;}}\
         </style>",
    );
    let _ = writeln!(
        svg,
        "<rect width=\"100%\" height=\"100%\" fill=\"var(--bg)\"/>",
    );

    let _ = writeln!(
        svg,
        "<text x=\"{:.0}\" y=\"20\" font-size=\"14\" font-weight=\"600\" \
         fill=\"var(--text)\">Gate density heatmap</text>",
        label_pad,
    );
    let _ = writeln!(
        svg,
        "<text x=\"{:.0}\" y=\"36\" font-size=\"11\" font-weight=\"300\" fill=\"var(--wire)\">\
         {} qubits × {} moments · max density {}</text>",
        label_pad, n, depth, max_density,
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
            let fill = heatmap_color(ratio, opts.dark_mode);
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
                     rx=\"{cell_rx}\"{cell_sr} fill=\"{fill}\">\
                     <title>q[{q_lo}..{q_hi}] m[{m_lo}..{m_hi}]: {count}</title></rect>",
                );
            } else {
                let _ = writeln!(
                    svg,
                    "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{cw}\" height=\"{ch}\" \
                     rx=\"{cell_rx}\"{cell_sr} fill=\"{fill}\"/>",
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

    let label_font = 10.0_f64.min(cell_h * row_bucket as f64);
    let tick_spacing_rows = (actual_rows / 5).max(1);
    for r in (0..actual_rows).step_by(tick_spacing_rows) {
        let q_start = r * row_bucket;
        let y = top_pad + r as f64 * cell_h + cell_h / 2.0;
        let _ = writeln!(
            svg,
            "<text x=\"{:.0}\" y=\"{y:.1}\" text-anchor=\"end\" \
             font-size=\"{label_font:.0}\" fill=\"var(--text)\" \
             dominant-baseline=\"central\">{q_start}</text>",
            label_pad - 4.0,
        );
    }

    let tick_spacing_cols = (actual_cols / 5).max(1);
    let axis_y = top_pad + grid_h + 12.0;
    for c in (0..actual_cols).step_by(tick_spacing_cols) {
        let m = c * col_bucket;
        let x = label_pad + c as f64 * cell_w;
        let _ = writeln!(
            svg,
            "<text x=\"{x:.1}\" y=\"{axis_y:.0}\" text-anchor=\"middle\" \
             font-size=\"9\" fill=\"var(--text)\">{m}</text>",
        );
    }

    let _ = writeln!(
        svg,
        "<text x=\"{:.0}\" y=\"{:.0}\" text-anchor=\"middle\" \
         font-size=\"10\" fill=\"var(--text)\">moment</text>",
        label_pad + grid_w / 2.0,
        axis_y + 14.0,
    );
    let qx = label_pad - label_pad * 0.6;
    let qy = top_pad + grid_h / 2.0;
    let _ = writeln!(
        svg,
        "<text x=\"{qx:.0}\" y=\"{qy:.0}\" text-anchor=\"middle\" \
         font-size=\"10\" fill=\"var(--text)\" transform=\"rotate(-90 {qx:.0} {qy:.0})\">\
         qubit</text>",
    );

    let spark_x = label_pad + grid_w + 4.0;
    for (r, &total) in row_totals.iter().enumerate() {
        if total == 0 {
            continue;
        }
        let bar_w = (total as f64 / max_row_total as f64) * spark_w;
        let y = top_pad + r as f64 * cell_h;
        let _ = writeln!(
            svg,
            "<rect x=\"{spark_x:.1}\" y=\"{y:.1}\" width=\"{bar_w:.1}\" height=\"{ch:.0}\" \
             rx=\"0.5\" fill=\"var(--wire)\" opacity=\"0.4\"/>",
        );
    }

    let spark_y = top_pad + grid_h + 4.0;
    for (c, &total) in col_totals.iter().enumerate() {
        if total == 0 {
            continue;
        }
        let bar_h = (total as f64 / max_col_total as f64) * spark_h;
        let x = label_pad + c as f64 * cell_w;
        let y_bar = spark_y + spark_h - bar_h;
        let _ = writeln!(
            svg,
            "<rect x=\"{x:.1}\" y=\"{y_bar:.1}\" width=\"{cw:.0}\" height=\"{bar_h:.1}\" \
             rx=\"0.5\" fill=\"var(--wire)\" opacity=\"0.4\"/>",
        );
    }

    let legend_x = spark_x + spark_w + 12.0;
    let legend_y = top_pad;
    let legend_h = grid_h.min(120.0);
    let legend_w = 14.0;
    let legend_id = "hm-legend";
    let _ = write!(
        svg,
        "<defs><linearGradient id=\"{legend_id}\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">"
    );
    let stops = heatmap_stops(opts.dark_mode);
    for (i, &(r, g, b)) in stops.iter().rev().enumerate() {
        let offset = i as f64 / 4.0 * 100.0;
        let _ = write!(
            svg,
            "<stop offset=\"{offset:.0}%\" stop-color=\"rgb({:.0},{:.0},{:.0})\"/>",
            r, g, b,
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
    let mid_val = max_density / 2;
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

    svg.push_str("</svg>\n");
    svg
}

impl Circuit {
    pub fn to_svg(&self, opts: &SvgOptions) -> String {
        let moments = assign_moments(self);
        render_svg(&moments, self.num_qubits, self.num_classical_bits, opts)
    }

    pub fn to_svg_heatmap(&self, opts: &SvgOptions) -> String {
        let moments = assign_moments(self);
        render_svg_heatmap(&moments, self.num_qubits, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::builder::CircuitBuilder;

    #[test]
    fn empty_circuit_svg() {
        let circuit = Circuit::new(0, 0);
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.starts_with("<svg xmlns="));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("empty circuit"));
    }

    #[test]
    fn bell_pair_svg() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.starts_with("<svg xmlns="));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("viewBox"));
        assert!(svg.contains(">H</text>"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("q[0]"));
        assert!(svg.contains("q[1]"));
    }

    #[test]
    fn cnot_target_oplus() {
        let circuit = CircuitBuilder::new(2).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            svg.matches("<circle").count() >= 2,
            "CX needs control dot + target circle"
        );
        assert!(
            !svg.contains(">X</text>"),
            "CX target uses oplus, not X box"
        );
    }

    #[test]
    fn gate_type_coloring() {
        let pi = std::f64::consts::PI;
        let circuit = CircuitBuilder::new(3).h(0).s(1).rx(pi / 4.0, 2).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains(LIGHT.standard.fill), "Standard gate fill");
        assert!(svg.contains(LIGHT.phase.fill), "Phase gate fill");
        assert!(svg.contains(LIGHT.parametric.fill), "Parametric gate fill");
    }

    #[test]
    fn tooltips_present() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("<title>"));
        assert!(svg.contains("H on q[0]"));
        assert!(svg.contains("ctrl q[0]"));
    }

    #[test]
    fn css_hover_present() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains(".gate:hover"));
        assert!(svg.contains("drop-shadow"));
        assert!(svg.contains("var(--shadow-filter)"));
    }

    #[test]
    fn swap_svg() {
        let circuit = CircuitBuilder::new(2).swap(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("stroke-width=\"1.5\""));
        assert!(svg.contains("stroke-linecap=\"round\""));
    }

    #[test]
    fn measurement_svg() {
        let circuit = CircuitBuilder::new(1).h(0).measure_all().build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("<path d=\"M"));
        assert!(svg.contains(LIGHT.measure.fill));
    }

    #[test]
    fn parametric_gate_svg() {
        let circuit = CircuitBuilder::new(1)
            .rx(std::f64::consts::FRAC_PI_4, 0)
            .build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("Rx(π/4)"));
        assert!(svg.contains(LIGHT.parametric.fill));
    }

    #[test]
    fn dark_mode_svg() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let opts = SvgOptions {
            dark_mode: true,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains(DARK.bg));
    }

    #[test]
    fn idle_wire_elision_svg() {
        let circuit = CircuitBuilder::new(5).h(0).cx(0, 4).build();
        let opts = SvgOptions {
            show_idle_wires: false,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("q[0]"));
        assert!(svg.contains("q[4]"));
        assert!(!svg.contains("q[1]"));
        assert!(!svg.contains("q[2]"));
        assert!(!svg.contains("q[3]"));
    }

    #[test]
    fn max_moments_svg() {
        let circuit = crate::circuits::random_circuit(3, 20, 42);
        let opts = SvgOptions {
            max_moments: Some(2),
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("<svg"));
        let gate_count = svg.matches("<g class=\"gate\"").count();
        assert!(gate_count < 20);
    }

    #[test]
    fn barrier_svg() {
        let circuit = CircuitBuilder::new(2)
            .h(0)
            .h(1)
            .barrier(&[0, 1])
            .cx(0, 1)
            .build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            svg.contains("fill-opacity=\"0.55\""),
            "barrier uses semi-transparent fill"
        );
        assert!(
            svg.contains("fill=\"var(--barrier)\""),
            "barrier uses theme color"
        );
    }

    #[test]
    fn valid_svg_structure() {
        let circuit = crate::circuits::qft_circuit(4);
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.trim().ends_with("</svg>"));
        assert!(svg.contains("viewBox=\"0 0"));
        assert!(svg.contains("<defs>"));
    }

    #[test]
    fn heatmap_light_colors() {
        let circuit = crate::circuits::ghz_circuit(5);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(svg.starts_with("<svg xmlns="));
        assert!(svg.contains("Gate density heatmap"));
        assert!(svg.contains("5 qubits"));
        assert!(svg.trim().ends_with("</svg>"));
        assert!(
            svg.contains("--bg:#ffffff"),
            "light heatmap uses white background"
        );
        let color_count = svg.matches("fill=\"#").count();
        assert!(color_count > 3, "light heatmap has distinct cell colors");
    }

    #[test]
    fn heatmap_large_circuit() {
        let circuit = crate::circuits::random_circuit(200, 50, 42);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(svg.contains("200 qubits"));
        let rect_count = svg.matches("<rect").count();
        assert!(rect_count > 10);
        assert!(rect_count < 50_000);
    }

    #[test]
    fn heatmap_tooltips_small() {
        let circuit = crate::circuits::ghz_circuit(5);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(
            svg.contains("<title>"),
            "small heatmap should have tooltips"
        );
    }

    #[test]
    fn heatmap_dark_mode() {
        let circuit = crate::circuits::random_circuit(30, 20, 42);
        let opts = SvgOptions {
            dark_mode: true,
            ..Default::default()
        };
        let svg = circuit.to_svg_heatmap(&opts);
        assert!(svg.contains(DARK.bg));
    }

    #[test]
    fn heatmap_empty() {
        let circuit = Circuit::new(0, 0);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(svg.contains("empty circuit"));
    }

    #[test]
    fn heatmap_hover_css() {
        let circuit = crate::circuits::ghz_circuit(5);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(svg.contains(".hm rect:hover"));
        assert!(svg.contains("class=\"hm\""));
        assert!(svg.contains("brightness(1.15)"), "hover brightness filter");
    }

    #[test]
    fn heatmap_gradient_legend() {
        let circuit = crate::circuits::random_circuit(30, 20, 42);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(
            svg.contains("<linearGradient id=\"hm-legend\""),
            "continuous gradient legend"
        );
        assert!(svg.contains("url(#hm-legend)"), "legend rect uses gradient");
    }

    #[test]
    fn heatmap_sparklines() {
        let circuit = crate::circuits::random_circuit(30, 20, 42);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(svg.contains("opacity=\"0.4\""), "sparkline bars present");
    }

    #[test]
    fn heatmap_crisp_small_cells() {
        let circuit = crate::circuits::ghz_circuit(5);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(
            svg.contains("shape-rendering=\"crispEdges\""),
            "small heatmap cells use crisp rendering"
        );
        assert!(
            svg.contains("rx=\"0\""),
            "small cells have no rounded corners"
        );
    }

    #[test]
    fn chessboard_stripes() {
        let circuit = CircuitBuilder::new(4).h(0).h(1).h(2).h(3).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains(LIGHT.stripe), "odd rows should have stripe bg");
    }

    #[test]
    fn sharp_gate_corners() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("rx=\"1\""), "gate corners should be sharp");
        assert!(!svg.contains("rx=\"4\""), "no rounded gate corners");
    }

    #[test]
    fn thin_strokes() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            svg.contains("stroke-width=\"1\""),
            "gates use thin 1px strokes"
        );
    }

    #[test]
    fn heatmap_dark_colors() {
        let circuit = crate::circuits::random_circuit(30, 20, 42);
        let opts = SvgOptions {
            dark_mode: true,
            ..Default::default()
        };
        let svg = circuit.to_svg_heatmap(&opts);
        assert!(
            svg.contains("--bg:#000000"),
            "dark heatmap uses true black background"
        );
        let color_count = svg.matches("fill=\"#").count();
        assert!(color_count > 5, "dark heatmap has distinct cell colors");
    }

    #[test]
    fn gradient_defs_present() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("<linearGradient id=\"g-std\""));
        assert!(svg.contains("<linearGradient id=\"g-ctrl\""));
        assert!(svg.contains("url(#g-std)"), "CSS references gradient");
        assert!(svg.contains("url(#g-ctrl)"), "CSS references gradient");
        assert!(svg.contains("var(--std-top)"), "gradient uses CSS vars");
    }

    #[test]
    fn css_transitions_present() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            svg.contains("transition:"),
            "gates should have CSS transitions"
        );
        assert!(svg.contains("tabular-nums"), "text should use tabular-nums");
    }

    #[test]
    fn css_custom_properties() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains(":root{"), "should have CSS custom properties");
        assert!(svg.contains("--bg:"), "should define --bg variable");
        assert!(svg.contains("var(--bg)"), "background uses CSS variable");
        assert!(svg.contains("var(--wire)"), "wires use CSS variable");
        assert!(svg.contains("var(--text)"), "labels use CSS variable");
        assert!(svg.contains("gate gate-std"), "gate uses CSS class");
    }

    #[test]
    fn auto_theme_svg() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let opts = SvgOptions {
            auto_theme: true,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(
            svg.contains("prefers-color-scheme:dark"),
            "auto-theme media query"
        );
        assert!(svg.contains(LIGHT.bg), "light theme variables present");
        assert!(svg.contains(DARK.bg), "dark theme variables present");
    }

    #[test]
    fn wire_junction_dots() {
        let circuit = CircuitBuilder::new(4).cx(0, 3).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        let junction_dots = svg.matches("r=\"1.5\"").count();
        assert_eq!(
            junction_dots, 2,
            "CX spanning q[0]→q[3] should have 2 junction dots on q[1] and q[2]"
        );
    }

    #[test]
    fn entrance_animation_present() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            svg.contains("@keyframes gate-in"),
            "gate entrance keyframes"
        );
        assert!(
            svg.contains("animation:gate-in"),
            "gates have entrance animation"
        );
        assert!(
            svg.contains(".m0{animation-delay:0ms}"),
            "moment 0 delay class"
        );
        assert!(
            svg.contains(".m1{animation-delay:30ms}"),
            "moment 1 delay class"
        );
        assert!(svg.contains("gate gate-std m0"), "gate has moment class");
    }

    #[test]
    fn wire_shimmer_present() {
        let circuit = CircuitBuilder::new(2).h(0).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            svg.contains("@keyframes wire-flow"),
            "wire shimmer keyframes"
        );
        assert!(
            svg.contains("class=\"wire-anim\""),
            "wires have shimmer class"
        );
        assert!(
            svg.contains("stroke-dasharray:1 11"),
            "shimmer dash pattern"
        );
    }

    #[test]
    fn animate_false_no_animation() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let opts = SvgOptions {
            animate: false,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(
            !svg.contains("@keyframes"),
            "no keyframes when animate=false"
        );
        assert!(
            !svg.contains("wire-anim"),
            "no wire shimmer when animate=false"
        );
        assert!(!svg.contains(" m0"), "no moment classes when animate=false");
        assert!(
            svg.contains("shape-rendering=\"crispEdges\""),
            "wires use crispEdges when static"
        );
    }

    #[test]
    fn aria_attributes() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("role=\"img\""), "root svg has role=img");
        assert!(
            svg.contains("aria-label=\"Quantum circuit: 2 qubits"),
            "aria-label present"
        );
    }

    #[test]
    fn semantic_layers() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("id=\"layer-bg\""), "background layer");
        assert!(svg.contains("id=\"layer-wires\""), "wires layer");
        assert!(svg.contains("id=\"layer-gates\""), "gates layer");
    }

    #[test]
    fn heatmap_aria() {
        let circuit = crate::circuits::ghz_circuit(5);
        let svg = circuit.to_svg_heatmap(&SvgOptions::default());
        assert!(svg.contains("role=\"img\""), "heatmap has role=img");
        assert!(
            svg.contains("aria-label=\"Gate density heatmap:"),
            "heatmap aria-label"
        );
    }

    #[test]
    fn qlabel_class() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains("class=\"qlabel\""), "labels use qlabel class");
        assert!(svg.contains(".qlabel{"), "qlabel CSS rule emitted");
    }

    #[test]
    fn reduced_motion_media_query() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            svg.contains("prefers-reduced-motion:reduce"),
            "reduced motion media query present when animate=true"
        );
        let static_opts = SvgOptions {
            animate: false,
            ..Default::default()
        };
        let svg_static = circuit.to_svg(&static_opts);
        assert!(
            !svg_static.contains("prefers-reduced-motion"),
            "no reduced motion query when animate=false"
        );
    }

    #[test]
    fn stats_header() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let opts = SvgOptions {
            show_stats_header: true,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("2 qubits"), "stats shows qubit count");
        assert!(svg.contains("depth 2"), "stats shows depth");
    }

    #[test]
    fn legend_shows_present_categories() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let opts = SvgOptions {
            show_legend: true,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("id=\"layer-legend\""), "legend layer present");
        assert!(svg.contains(">Standard</text>"), "Standard category shown");
        assert!(
            svg.contains(">Controlled</text>"),
            "Controlled category shown"
        );
        assert!(
            !svg.contains(">Parametric</text>"),
            "Parametric not shown (not present)"
        );
    }

    #[test]
    fn legend_hidden_by_default() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            !svg.contains("id=\"layer-legend\""),
            "legend not present by default"
        );
    }

    #[test]
    fn classical_wires() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).measure_all().build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(svg.contains(">c[0]</text>"), "classical wire 0 label");
        assert!(svg.contains(">c[1]</text>"), "classical wire 1 label");
        assert!(
            svg.contains("stroke-dasharray=\"3,3\""),
            "measurement arrow"
        );
        assert!(
            svg.contains("marker-end=\"url(#arrow-meas)\""),
            "SVG marker arrowhead"
        );
    }

    #[test]
    fn no_classical_wires_when_zero() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(
            !svg.contains(">c["),
            "no classical wires without measurement"
        );
    }

    #[test]
    fn compact_mode_smaller() {
        let circuit = crate::circuits::ghz_circuit(5);
        let normal = circuit.to_svg(&SvgOptions::default());
        let compact = circuit.to_svg(&SvgOptions {
            compact: true,
            ..Default::default()
        });
        assert!(
            compact.len() < normal.len(),
            "compact SVG should be smaller"
        );
        assert!(compact.contains("font-size:10.1px"), "compact font size");
    }

    #[test]
    fn ellipsis_mode() {
        let circuit = crate::circuits::random_circuit(3, 15, 42);
        let opts = SvgOptions {
            ellipsis_mode: Some((3, 3)),
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("\u{22ef}"), "ellipsis character present");
    }

    #[test]
    fn ellipsis_noop_small() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let opts = SvgOptions {
            ellipsis_mode: Some((5, 5)),
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(!svg.contains("\u{22ef}"), "no ellipsis when circuit fits");
    }

    #[test]
    fn topology_graph() {
        let circuit = CircuitBuilder::new(3).cx(0, 1).cx(1, 2).build();
        let opts = SvgOptions {
            show_topology: true,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(
            svg.contains("id=\"layer-topology\""),
            "topology layer present"
        );
        assert!(svg.contains(">topology</text>"), "topology label present");
        let node_count = svg.matches("layer-topology").count();
        assert!(node_count >= 1, "topology has nodes");
    }

    #[test]
    fn topology_hidden_by_default() {
        let circuit = CircuitBuilder::new(2).cx(0, 1).build();
        let svg = circuit.to_svg(&SvgOptions::default());
        assert!(!svg.contains("layer-topology"), "no topology by default");
    }

    #[test]
    fn topology_no_graph_without_2q() {
        let circuit = CircuitBuilder::new(2).h(0).h(1).build();
        let opts = SvgOptions {
            show_topology: true,
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(
            !svg.contains("layer-topology"),
            "no topology when no 2q gates"
        );
    }
}
