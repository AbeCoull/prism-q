//! Every routing and fusion threshold names the bench rows on either side of it, or says
//! that none exist, and every group it names is a Criterion group under `benches/`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const BRACKETED: &str = "/// Bracketed by ";
const UNBRACKETED: &str = "/// No bench row brackets this yet.";

const THRESHOLDS: &[(&str, &str)] = &[
    ("src/sim/dispatch.rs", "MPS_GATE_CUT_OVERHEAD"),
    ("src/sim/dispatch.rs", "MAX_AUTO_T_COUNT_SHOTS"),
    ("src/sim/dispatch.rs", "MAX_STABILIZER_RANK_QUBITS"),
    ("src/sim/dispatch.rs", "MIN_QUBITS_FOR_SPD_AUTO"),
    ("src/sim/dispatch.rs", "AUTO_SPD_AMPLITUDES_PER_TERM"),
    ("src/sim/dispatch.rs", "MIN_FACTORED_STABILIZER_QUBITS"),
    ("src/sim/dispatch.rs", "MIN_BLOCK_FOR_FACTORED_STAB"),
    ("src/sim/dispatch.rs", "AUTO_SPARSE_MAX_SUPPORT"),
    ("src/sim/route.rs", "TENSOR_ROUTE_MIN_QUBITS"),
    ("src/sim/route.rs", "TENSOR_ROUTE_MPS_COST_PER_OBSERVABLE"),
    ("src/sim/route.rs", "TENSOR_ELEMENT_MPS_COST"),
    ("src/circuit/fusion.rs", "MIN_QUBITS_FOR_FUSION"),
    ("src/circuit/fusion.rs", "MIN_QUBITS_FOR_MULTI_FUSION"),
    ("src/circuit/fusion.rs", "MIN_QUBITS_FOR_DIAG_BATCH"),
    ("src/circuit/fusion.rs", "MIN_QUBITS_FOR_POST_PHASE_BATCH"),
    ("src/circuit/fusion.rs", "MIN_QUBITS_FOR_2Q_FUSION"),
    ("src/circuit/fusion.rs", "MIN_QUBITS_FOR_PAULI_ROT_BATCH"),
    ("src/sim/mod.rs", "RUN_SPLIT_QUBITS"),
    ("src/sim/mod.rs", "MIN_AUTO_CLIFFORD_REGISTER_QUBITS"),
    ("src/sim/mod.rs", "MAX_PAIR_MASKS_PER_GROUP"),
    ("src/backend/mod.rs", "PARALLEL_THRESHOLD_QUBITS"),
];

fn manifest_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn rust_files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files_under(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn bench_groups() -> BTreeSet<String> {
    let mut files = Vec::new();
    rust_files_under(&manifest_path("benches"), &mut files);
    let mut groups = BTreeSet::new();
    for file in files {
        let source = read(&file);
        for chunk in source.split("benchmark_group(\"").skip(1) {
            let end = chunk.find('"').expect("unterminated group name");
            groups.insert(chunk[..end].to_string());
        }
    }
    groups
}

/// The doc lines directly above the first `const NAME:` in `source`, past any attributes.
fn doc_block<'a>(source: &'a str, file: &str, name: &str) -> Vec<&'a str> {
    let lines: Vec<&str> = source.lines().map(str::trim).collect();
    let decl = format!("const {name}:");
    let at = lines
        .iter()
        .position(|line| {
            let line = ["pub(crate) ", "pub(super) ", "pub "]
                .iter()
                .find_map(|vis| line.strip_prefix(vis))
                .unwrap_or(line);
            line.starts_with(&decl)
        })
        .unwrap_or_else(|| panic!("{file}: no definition of {name}"));
    let mut i = at;
    while i > 0 && lines[i - 1].starts_with("#[") {
        i -= 1;
        assert!(
            !lines[i].starts_with("#[cfg") || lines[i].contains("not("),
            "{file}: the first definition of {name} is not the native one: {}",
            lines[i]
        );
    }
    let end = i;
    while i > 0 && lines[i - 1].starts_with("///") {
        i -= 1;
    }
    lines[i..end].to_vec()
}

/// Group names a `Bracketed by` line cites: the backticked spans outside parentheses.
fn cited_groups(line: &str) -> Vec<String> {
    let mut outside = String::new();
    let mut depth = 0usize;
    for c in line.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.checked_sub(1).expect("unbalanced parentheses"),
            _ if depth == 0 => outside.push(c),
            _ => {}
        }
    }
    assert_eq!(depth, 0, "unbalanced parentheses: {line}");
    outside
        .split('`')
        .skip(1)
        .step_by(2)
        .map(String::from)
        .collect()
}

#[test]
fn threshold_names_are_unique() {
    let names: BTreeSet<&str> = THRESHOLDS.iter().map(|&(_, name)| name).collect();
    assert_eq!(names.len(), THRESHOLDS.len());
}

#[test]
fn threshold_docs_cite_bench_groups_that_exist() {
    let groups = bench_groups();
    assert!(!groups.is_empty());
    for &(file, name) in THRESHOLDS {
        let source = read(&manifest_path(file));
        let docs = doc_block(&source, file, name);
        let cites: Vec<&str> = docs
            .iter()
            .copied()
            .filter(|line| line.starts_with(BRACKETED) || *line == UNBRACKETED)
            .collect();
        assert_eq!(
            cites.len(),
            1,
            "{file}: {name} needs exactly one `{}` or `{UNBRACKETED}` doc line, found {cites:?}",
            BRACKETED.trim_end()
        );
        let line = cites[0];
        if line == UNBRACKETED {
            continue;
        }
        assert!(
            line.ends_with('.'),
            "{file}: {name}: unfinished line {line:?}"
        );
        let cited = cited_groups(line);
        assert!(!cited.is_empty(), "{file}: {name} cites no group: {line:?}");
        for group in cited {
            assert!(
                groups.contains(&group),
                "{file}: {name} cites `{group}`, which no `benchmark_group(\"...\")` under \
                 benches/ creates"
            );
        }
    }
}
