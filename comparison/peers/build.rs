//! Reads the resolved Spinoza and qip versions out of `Cargo.lock` so the
//! provenance the binary reports cannot drift from what it was built against.

use std::fs;
use std::path::Path;

fn locked(lock: &str, name: &str) -> String {
    let header = format!("name = \"{name}\"");
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() != header {
            continue;
        }
        let mut version = String::new();
        let mut rev = String::new();
        for field in lines.by_ref().take_while(|l| !l.is_empty()) {
            if let Some(v) = field.strip_prefix("version = ") {
                version = v.trim_matches('"').to_string();
            } else if let Some(src) = field.strip_prefix("source = ") {
                if let Some(hash) = src.rsplit('#').next().filter(|_| src.contains("git+")) {
                    rev = hash.trim_matches('"').chars().take(7).collect();
                }
            }
        }
        return if rev.is_empty() { version } else { format!("{version} (git {rev})") };
    }
    panic!("{name} is not in Cargo.lock");
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let lock_path = Path::new(&manifest_dir).join("Cargo.lock");
    let lock = fs::read_to_string(&lock_path).expect("read Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!("cargo:rustc-env=SPINOZA_VERSION={}", locked(&lock, "spinoza"));
    println!("cargo:rustc-env=QIP_VERSION={}", locked(&lock, "qip"));
}
