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
fn swap_svg() {
    let circuit = CircuitBuilder::new(2).swap(0, 1).build();
    let svg = circuit.to_svg(&SvgOptions::default());
    assert!(svg.contains("stroke-width=\"1.25\""));
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
    assert!(svg.contains("aria-label=\"Rx(π/4)\""));
    assert!(svg.contains("Rx(π/4) on q[0]"));
    assert!(svg.contains("baseline-shift=\"sub\""));
    assert!(svg.contains(">x</tspan>"));
    assert!(svg.contains(">π/4</text>"));
    assert!(!svg.contains(">Rx(π/4)</text>"));
    assert!(svg.contains(LIGHT.parametric.fill));
    let label = &svg[svg.find("aria-label=\"Rx(π/4)\"").unwrap()..];
    let label = label.split("</g>").next().unwrap();
    let name_y = x_before(label, "class=\"gate-name\"", " y=\"");
    let angle_y = x_before(label, "class=\"gate-param\"", " y=\"");
    let top: f64 = label
        .split(" y=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(angle_y > name_y);
    assert!(name_y > top);
    assert!(angle_y < top + SvgOptions::default().gate_height);
}

#[test]
fn controlled_phase_shows_its_angle_without_relabeling_other_unitaries() {
    let circuit = CircuitBuilder::new(2)
        .cphase(std::f64::consts::FRAC_PI_4, 0, 1)
        .cu(Gate::H.matrix_2x2(), 0, 1)
        .cu(Gate::Rz(0.5).matrix_2x2(), 0, 1)
        .build();
    let svg = circuit.to_svg(&SvgOptions::default());
    assert!(svg.contains("aria-label=\"P(π/4)\""));
    assert!(svg.contains(">P</text>"));
    assert!(svg.contains(">π/4</text>"));
    assert_eq!(svg.matches(">U</text>").count(), 2);
}

#[test]
fn rounded_parameters_keep_full_values_and_small_nonzero_angles() {
    let angle = 1.23456789;
    let circuit = CircuitBuilder::new(1)
        .ry(angle, 0)
        .rz(-0.000012345, 0)
        .rx(-0.0, 0)
        .build();
    let svg = circuit.to_svg(&SvgOptions::default());
    assert!(svg.contains(">1.235</text>"));
    assert!(svg.contains("angle 1.23456789 rad"));
    assert!(svg.contains("aria-label=\"Ry(1.23456789)\""));
    assert!(svg.contains(">-1.23e-5</text>"));
    assert!(svg.contains("angle -0.000012345 rad"));
    assert!(svg.contains(">0</text>"));
    assert!(!svg.contains(">-0</text>"));
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
    assert!(svg.contains("Gate activity counts"));
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
    let mut builder = CircuitBuilder::new(201);
    for q in 0..201 {
        builder.h(q);
    }
    let circuit = builder.build();
    let svg = circuit.to_svg_heatmap(&SvgOptions::default());
    assert!(
        svg.contains("<linearGradient id=\"hm-legend\""),
        "continuous gradient legend"
    );
    assert!(svg.contains("url(#hm-legend)"), "legend rect uses gradient");
    assert!(svg.contains("max cell count 2"));
    assert!(!svg.contains(">Active</text>"));
    assert!(svg.contains("q[0..1]: activity 2\">2</text>"));
    assert!(svg.contains("m[0..0]: activity 201\">201</text>"));
    assert!(svg.contains("Cells: up to 2 qubits × 1 moment."));
    assert!(svg.contains("Counts are sums, not normalized for cell size."));
    assert!(svg.contains("Partial final row q[200] (1 qubit)."));
    assert!(svg.contains(">0–1</text>"));
}

#[test]
fn heatmap_binary_activity_has_a_discrete_key() {
    let circuit = CircuitBuilder::new(3).h(0).cx(0, 1).build();
    let svg = circuit.to_svg_heatmap(&SvgOptions::default());
    assert!(svg.contains(">Active</text>"));
    assert!(svg.contains(">Idle</text>"));
    assert!(!svg.contains("<linearGradient"));
    assert!(!svg.contains(">0.5</text>"));
    assert!(svg.contains("One count per wire touched; CX contributes 2."));
    assert_eq!(svg.matches("class=\"hm-note\"").count(), 1);
}

#[test]
fn heatmap_aggregation_explains_both_dimensions_without_cell_tooltips() {
    let mut builder = CircuitBuilder::new(201);
    for _ in 0..201 {
        for q in 0..201 {
            builder.h(q);
        }
    }
    let svg = builder.build().to_svg_heatmap(&SvgOptions::default());
    assert!(svg.contains("Cells: up to 2 qubits × 2 moments."));
    assert!(svg.contains("max cell count 4"));
    assert!(svg.contains("Counts are sums, not normalized for cell size."));
    assert!(svg.contains("Partial final row q[200] (1 qubit); final column m[200] (1 moment)."));
    assert_eq!(svg.matches(">0–1</text>").count(), 2);
    let cells = svg
        .split("<g class=\"hm\">")
        .nth(1)
        .unwrap()
        .split("</g>")
        .next()
        .unwrap();
    assert!(!cells.contains("<title>"));
}

#[test]
fn heatmap_tiny_chart_reserves_space_for_the_unit_note() {
    let circuit = CircuitBuilder::new(1).h(0).build();
    let svg = circuit.to_svg_heatmap(&SvgOptions::default());
    let note = "One count per wire touched; CX contributes 2.";
    let x = x_before(&svg, note, "<text x=\"");
    assert!(x + label_width(note, 10.0) < viewbox_width(&svg));
    assert_eq!(svg.matches("class=\"hm-note\"").count(), 1);
}

#[test]
fn heatmap_marginal_labels_count_activity_and_idle_qubits() {
    let circuit = CircuitBuilder::new(3).h(0).cx(0, 1).build();
    let svg = circuit.to_svg_heatmap(&SvgOptions::default());
    for expected in [
        "q[0..0]: activity 2\">2</text>",
        "q[1..1]: activity 1\">1</text>",
        "q[2..2]: activity 0\">0</text>",
        "m[0..0]: activity 1\">1</text>",
        "m[1..1]: activity 2\">2</text>",
    ] {
        assert!(svg.contains(expected), "missing total {expected}");
    }
    assert_eq!(svg.matches("class=\"hm-row-total\"").count(), 3);
    assert_eq!(svg.matches("class=\"hm-col-total\"").count(), 2);
}

#[test]
fn heatmap_sparklines() {
    let circuit = crate::circuits::random_circuit(30, 20, 42);
    let svg = circuit.to_svg_heatmap(&SvgOptions::default());
    assert!(svg.contains("opacity=\"0.4\""), "sparkline bars present");
}

#[test]
fn heatmap_crisp_small_cells() {
    let circuit = crate::circuits::ghz_circuit(200);
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
        svg.contains(&format!("--bg:{}", DARK.bg)),
        "dark heatmap uses the dark canvas"
    );
    let color_count = svg.matches("fill=\"#").count();
    assert!(color_count > 5, "dark heatmap has distinct cell colors");
}

#[test]
fn gate_fills_follow_theme() {
    let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
    let svg = circuit.to_svg(&SvgOptions::default());
    assert!(svg.contains("fill:var(--std-fill)"));
    assert!(svg.contains("fill:var(--ctrl-fill)"));
}

#[test]
fn css_custom_properties() {
    let circuit = CircuitBuilder::new(1).h(0).build();
    let svg = circuit.to_svg(&SvgOptions::default());
    assert!(
        svg.contains(".pq-circuit{"),
        "should have CSS custom properties"
    );
    assert!(
        !svg.contains(":root"),
        "styles must stay scoped to the figure"
    );
    assert!(svg.contains("--bg:"), "should define --bg variable");
    assert!(svg.contains("var(--bg)"), "background uses CSS variable");
    assert!(svg.contains("var(--wire)"), "wires use CSS variable");
    assert!(svg.contains("var(--text)"), "labels use CSS variable");
    assert!(svg.contains("gate gate-std"), "gate uses CSS class");
}

fn style_selectors(svg: &str) -> Vec<String> {
    let mut selectors = Vec::new();
    let mut rest = svg;
    while let Some(start) = rest.find("<style>") {
        let body_start = start + "<style>".len();
        let end = body_start + rest[body_start..].find("</style>").unwrap();
        let mut prelude = String::new();
        let mut depth = 0usize;
        let mut skip_to = None;
        for c in rest[body_start..end].chars() {
            match c {
                '{' => {
                    depth += 1;
                    let p = prelude.trim().to_string();
                    prelude.clear();
                    if skip_to.is_some() || p.starts_with("@media") {
                        continue;
                    }
                    if p.starts_with("@keyframes") {
                        skip_to = Some(depth - 1);
                        continue;
                    }
                    selectors.extend(p.split(',').map(|s| s.trim().to_string()));
                    skip_to = Some(depth - 1);
                }
                '}' => {
                    depth -= 1;
                    prelude.clear();
                    if skip_to == Some(depth) {
                        skip_to = None;
                    }
                }
                _ if skip_to.is_none() => prelude.push(c),
                _ => {}
            }
        }
        rest = &rest[end..];
    }
    selectors
}

#[test]
fn every_style_rule_is_scoped_to_the_figure() {
    let circuit = CircuitBuilder::new(3)
        .h(0)
        .cx(0, 1)
        .rz(0.3, 2)
        .swap(1, 2)
        .build();
    for dark_mode in [false, true] {
        for auto_theme in [false, true] {
            for animate in [false, true] {
                let opts = SvgOptions {
                    dark_mode,
                    auto_theme,
                    animate,
                    ..SvgOptions::default()
                };
                for svg in [circuit.to_svg(&opts), circuit.to_svg_heatmap(&opts)] {
                    let selectors = style_selectors(&svg);
                    for s in &selectors {
                        assert!(
                            s == ".pq-circuit" || s.starts_with(".pq-circuit "),
                            "unscoped selector {s:?} (dark {dark_mode}, auto {auto_theme}, animate {animate})"
                        );
                    }
                }
            }
        }
    }
    assert!(!style_selectors(&circuit.to_svg(&SvgOptions::default())).is_empty());
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
fn wire_crossings_do_not_add_controls() {
    let circuit = CircuitBuilder::new(4).cx(0, 3).build();
    let svg = circuit.to_svg(&SvgOptions::default());
    assert_eq!(
        svg.matches("<circle").count(),
        2,
        "a distant CX has one control and one target"
    );
    let swap = CircuitBuilder::new(4).swap(0, 3).build();
    assert!(!swap.to_svg(&SvgOptions::default()).contains("<circle"));
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
    assert!(svg.contains("aria-label=\"q[0]\""));
    assert!(svg.contains("aria-label=\"q[1]\""));
    assert!(svg.contains("baseline-shift=\"sub\""));
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
        svg.contains("aria-label=\"Gate activity counts:"),
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
    let background = svg.find("id=\"layer-bg\"").unwrap();
    let background_end = background + svg[background..].find("</g>").unwrap();
    assert!(svg.find("depth 2</text>").unwrap() > background_end);
}

fn viewbox_width(svg: &str) -> f64 {
    svg.split("viewBox=\"0 0 ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn short_circuit_annotations_fit_without_shrinking() {
    let circuit = CircuitBuilder::new(3).h(0).rx(0.25, 1).t(2).build();
    let opts = SvgOptions {
        show_stats_header: true,
        show_legend: true,
        ..Default::default()
    };
    let svg = circuit.to_svg(&opts);
    let width = viewbox_width(&svg);
    let legend = &svg[svg.find("id=\"layer-legend\"").unwrap()..];
    let last_label = legend.rfind("<text x=\"").unwrap() + "<text x=\"".len();
    let label_x: f64 = legend[last_label..]
        .split('"')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(label_x + "Phase".len() as f64 * opts.font_size * 0.62 < width);
    let legend_font = opts.font_size * 0.92;
    assert!(legend.contains(&format!("font-size=\"{legend_font:.1}px\"")));
    assert!(svg.contains(".qlabel:not([font-size])"));
}

#[test]
fn conditional_labels_leave_space_for_the_following_gate() {
    let circuit = CircuitBuilder::new_with_classical(1, 8)
        .conditional(
            crate::circuit::ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 8,
                value: 255,
            },
            Gate::X,
            &[0],
        )
        .h(0)
        .build();
    let svg = circuit.to_svg(&SvgOptions::default());
    let conditional = ">c[0..8]==255?X</text>";
    let left = x_before(&svg, conditional, "<rect x=\"");
    let width = x_before(&svg, conditional, " width=\"");
    let next_left = x_before(&svg, "central\">H</text>", "<rect x=\"");
    assert!(left + width < next_left);
}

#[test]
fn heatmap_labels_stay_readable_and_below_activity_bars() {
    let circuit = crate::circuits::hardware_efficient_ansatz(12, 4, 42);
    let svg = circuit.to_svg_heatmap(&SvgOptions::default());
    assert!(!svg.contains("font-size=\"4\""));
    assert!(svg.contains("font-size=\"10\""));
    let axis_y = x_before(&svg, ">moment</text>", " y=\"");
    let mut max_bar_bottom = 0.0_f64;
    for bar in svg.split("<rect ").skip(1) {
        let bar = bar.split("/>").next().unwrap();
        if !bar.contains("opacity=\"0.4\"") {
            continue;
        }
        let y: f64 = bar
            .split("y=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let h: f64 = bar
            .split("height=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        max_bar_bottom = max_bar_bottom.max(y + h);
    }
    assert!(axis_y - 14.0 > max_bar_bottom);
    let count_y = x_before(&svg, "class=\"hm-col-total\"", " y=\"");
    assert!(count_y > max_bar_bottom);
    assert!(axis_y - 14.0 > count_y);
}

#[test]
fn heatmap_auto_theme_switches_cells_and_scale_together() {
    let mut builder = CircuitBuilder::new(201);
    for q in 0..201 {
        builder.h(q);
    }
    let circuit = builder.build();
    let opts = SvgOptions {
        auto_theme: true,
        ..Default::default()
    };
    let svg = circuit.to_svg_heatmap(&opts);
    let max_light = heatmap_color(1.0, false);
    let max_dark = heatmap_color(1.0, true);
    let colors = format!("--hm-light:{max_light};--hm-dark:{max_dark}");
    let cells = &svg[svg.find("<g class=\"hm\">").unwrap()..];
    assert!(cells.contains(&colors));
    let scale = &svg[svg.find("<linearGradient id=\"hm-legend\"").unwrap()..];
    assert!(scale.contains(&colors));
    assert!(
        svg.contains(".pq-circuit .hm rect,.pq-circuit .hm-key .hm-active{fill:var(--hm-dark);}")
    );
    assert!(svg.contains(".pq-circuit .hm-scale stop{stop-color:var(--hm-dark);}"));
    let dark_override = circuit.to_svg_heatmap(&SvgOptions {
        dark_mode: true,
        ..opts
    });
    assert_eq!(svg, dark_override);
}

#[test]
fn heatmap_binary_key_uses_the_cell_theme() {
    let circuit = crate::circuits::ghz_circuit(5);
    let svg = circuit.to_svg_heatmap(&SvgOptions {
        auto_theme: true,
        ..Default::default()
    });
    let colors = format!(
        "--hm-light:{};--hm-dark:{}",
        heatmap_color(1.0, false),
        heatmap_color(1.0, true),
    );
    let key = &svg[svg.find("<g class=\"hm-key\"").unwrap()..];
    assert!(key.contains(&colors));
    assert!(key.contains("fill=\"var(--bg)\""));
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
    assert!(
        svg.contains("aria-label=\"c[0]\""),
        "classical wire 0 label"
    );
    assert!(
        svg.contains("aria-label=\"c[1]\""),
        "classical wire 1 label"
    );
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
fn measurement_arrow_clears_the_quantum_gate() {
    let circuit = CircuitBuilder::new_with_classical(1, 1)
        .measure(0, 0)
        .build();
    let svg = circuit.to_svg(&SvgOptions::default());
    let marker = "marker-end=\"url(#arrow-meas)\"";
    let top = x_before(&svg, marker, " y1=\"");
    let bottom = x_before(&svg, marker, " y2=\"");
    assert!(bottom > top);
}

#[test]
fn simultaneous_measurements_have_separate_paths_past_other_gates() {
    let circuit = CircuitBuilder::new_with_classical(3, 2)
        .h(2)
        .measure(0, 0)
        .measure(1, 1)
        .build();
    for compact in [false, true] {
        let opts = SvgOptions {
            compact,
            show_stats_header: true,
            max_moments: Some(1),
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("depth 1"));
        let arrows: Vec<_> = svg
            .split("<line ")
            .filter(|line| line.split("/>").next().unwrap().contains("marker-end="))
            .map(|line| {
                line.split("x1=\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
            })
            .collect();
        assert_eq!(arrows.len(), 2);
        let h = x_before(&svg, "central\">H</text>", "<text x=\"");
        let meter_width = if compact { 36.0 } else { opts.gate_min_width };
        assert!(arrows[0] - h > meter_width);
        assert!(arrows[1] - arrows[0] > meter_width);
    }
}

#[test]
fn no_classical_wires_when_zero() {
    let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
    let svg = circuit.to_svg(&SvgOptions::default());
    assert!(
        !svg.contains("aria-label=\"c["),
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
    assert!(viewbox_width(&compact) < viewbox_width(&normal));
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

fn x_before(svg: &str, marker: &str, element: &str) -> f64 {
    let end = svg
        .find(marker)
        .unwrap_or_else(|| panic!("{marker} missing"));
    let start = svg[..end].rfind(element).unwrap() + element.len();
    svg[start..].split('"').next().unwrap().parse().unwrap()
}

#[test]
fn barrier_sits_between_the_gates_it_separates() {
    let circuit = CircuitBuilder::new(2)
        .h(0)
        .h(1)
        .barrier(&[0, 1])
        .x(0)
        .x(1)
        .build();
    let opts = SvgOptions {
        show_stats_header: true,
        ..Default::default()
    };
    let svg = circuit.to_svg(&opts);
    let barrier_left = x_before(&svg, "fill=\"var(--barrier)\"", "<rect x=\"");
    let h = x_before(&svg, "central\">H</text>", "<text x=\"");
    let x = x_before(&svg, "central\">X</text>", "<text x=\"");
    let barrier = barrier_left + 2.0;
    assert!(
        h < barrier && barrier < x,
        "H {h}, barrier {barrier}, X {x}"
    );
    let gate_half_width = SvgOptions::default().gate_min_width / 2.0;
    assert!(
        x - barrier > gate_half_width,
        "barrier {barrier} under X {x}"
    );
    assert!(svg.contains("depth 2"), "a barrier adds no depth");
}

#[test]
fn hidden_barriers_preserve_alignment_with_a_narrow_gap() {
    let circuit = CircuitBuilder::new(3)
        .cx(0, 1)
        .cx(1, 2)
        .barrier(&[0, 1, 2])
        .h(0)
        .h(1)
        .h(2)
        .build();
    let opts = SvgOptions {
        show_barriers: false,
        show_stats_header: true,
        ..Default::default()
    };
    let svg = circuit.to_svg(&opts);
    assert!(!svg.contains("fill=\"var(--barrier)\""));
    assert!(svg.contains("depth 3"));
    let positions: Vec<_> = svg
        .split("<text x=\"")
        .skip(1)
        .filter(|text| text.split("</text>").next().unwrap().ends_with(">H"))
        .map(|text| text.split('"').next().unwrap().parse::<f64>().unwrap())
        .collect();
    assert_eq!(positions.len(), 3);
    assert!(positions.iter().all(|x| *x == positions[0]));
    let last_target = svg.rfind("<circle cx=\"").unwrap() + "<circle cx=\"".len();
    let target: f64 = svg[last_target..]
        .split('"')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let separation = positions[0] - target;
    assert!(separation > opts.moment_width);
    assert!(separation < opts.moment_width * 1.5);
}

#[test]
fn barrier_columns_do_not_consume_moment_limits() {
    let prefix = CircuitBuilder::new(2).h(0).barrier(&[0, 1]).x(0).build();
    let circuit = CircuitBuilder::new(2)
        .h(0)
        .barrier(&[0, 1])
        .x(0)
        .barrier(&[0, 1])
        .z(0)
        .build();
    for show_barriers in [false, true] {
        let opts = SvgOptions {
            show_barriers,
            show_stats_header: true,
            ..Default::default()
        };
        let limited = circuit.to_svg(&SvgOptions {
            max_moments: Some(2),
            ..opts
        });
        assert_eq!(limited, prefix.to_svg(&opts));
        assert!(limited.contains("2 gates"));
        assert!(limited.contains("2 moments\""));
        let complete = circuit.to_svg(&opts);
        for budget in [3, 4] {
            assert_eq!(
                circuit.to_svg(&SvgOptions {
                    max_moments: Some(budget),
                    ..opts
                }),
                complete,
            );
        }
    }
}

#[test]
fn barrier_columns_stay_inside_retained_ellipsis_sections() {
    let circuit = CircuitBuilder::new(2)
        .h(0)
        .barrier(&[0, 1])
        .x(0)
        .barrier(&[0, 1])
        .y(0)
        .barrier(&[0, 1])
        .z(0)
        .barrier(&[0, 1])
        .s(0)
        .barrier(&[0, 1])
        .t(0)
        .build();
    for show_barriers in [false, true] {
        let opts = SvgOptions {
            show_barriers,
            show_stats_header: true,
            max_moments: Some(5),
            ellipsis_mode: Some((2, 2)),
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("\u{22ef}"));
        for label in ["H", "X", "Z", "S"] {
            assert!(svg.contains(&format!("aria-label=\"{label}\"")));
        }
        for label in ["Y", "T"] {
            assert!(!svg.contains(&format!("aria-label=\"{label}\"")));
        }
        assert!(svg.contains("depth 5"));
        assert!(svg.contains("5 gates"));
        assert!(svg.contains("5 moments\""));
        assert_eq!(
            svg.matches("fill=\"var(--barrier)\"").count(),
            if show_barriers { 2 } else { 0 },
        );
        let center =
            |label: &str| x_before(&svg, &format!("central\">{label}</text>"), "<text x=\"");
        let separated = opts.moment_width * 1.4;
        assert!((center("X") - center("H") - separated).abs() < 0.1);
        assert!((center("S") - center("Z") - separated).abs() < 0.1);
        assert!((center("Z") - center("X") - opts.moment_width - 40.0).abs() < 0.1);

        let complete = circuit.to_svg(&SvgOptions {
            ellipsis_mode: None,
            ..opts
        });
        assert_eq!(
            circuit.to_svg(&SvgOptions {
                ellipsis_mode: Some((3, 2)),
                ..opts
            }),
            complete,
        );
    }
}

#[test]
fn one_sided_ellipsis_leaves_no_barrier_beside_the_gap() {
    let circuit = CircuitBuilder::new(2)
        .h(0)
        .barrier(&[0, 1])
        .x(0)
        .barrier(&[0, 1])
        .z(0)
        .build();
    for retained in [(0, 1), (1, 0), (0, 0)] {
        let opts = SvgOptions {
            ellipsis_mode: Some(retained),
            ..Default::default()
        };
        let svg = circuit.to_svg(&opts);
        assert!(svg.contains("\u{22ef}"));
        assert!(!svg.contains("fill=\"var(--barrier)\""));
        let columns = retained.0 + retained.1;
        let expected_width =
            opts.padding_left + columns as f64 * opts.moment_width + 40.0 + opts.padding_right;
        assert!((viewbox_width(&svg) - expected_width).abs() < 0.1);
    }
}

#[test]
fn legend_svg_is_identical_across_renders() {
    let circuit = CircuitBuilder::new(2)
        .h(0)
        .rx(0.3, 1)
        .t(0)
        .cx(0, 1)
        .measure_all()
        .build();
    let opts = SvgOptions {
        show_legend: true,
        ..SvgOptions::default()
    };
    let first = circuit.to_svg(&opts);
    for _ in 0..32 {
        assert_eq!(circuit.to_svg(&opts), first);
    }
}
