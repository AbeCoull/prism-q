# PRISM-Q Makefile
# For Unix/WSL environments. Windows users: see scripts/*.ps1

.PHONY: all build test bench bench-smoke bench-full baseline compare fmt clippy check clean

all: fmt clippy test

build:
	cargo build --release

check:
	cargo check --all-targets

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test

# Quick benchmark smoke test (subset for PRs)
bench-smoke:
	cargo bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10 "single_qubit_gates/h_gate/4"

# Full benchmark suite
bench-full:
	cargo bench

# Save baseline for regression comparison
baseline:
	./scripts/bench_baseline.sh

# Compare current performance against saved baseline
compare:
	./scripts/bench_compare.sh

clean:
	cargo clean
