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
