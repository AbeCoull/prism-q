//! Peak-memory profiler for the chunked (streaming) compiled-sampling path.
//!
//! `sample_chunked` folds one `PackedShots` chunk at a time into a bounded
//! accumulator, so peak heap allocation should track the per-chunk budget
//! (`default_chunk_size`) rather than the total shot count. A tracking global
//! allocator records the allocated-bytes high-water mark of each run; the run
//! passes when peak stays flat as total shots grow.
//!
//! Histogram is profiled on GHZ (two distinct outcomes) because its state grows
//! with the number of distinct outcomes and is only chunk-bounded on a
//! low-entropy circuit. Marginals and expectation are bounded on any circuit and
//! use a high-rank Clifford circuit.
//!
//! ```text
//! cargo run --release --features parallel --example profile_chunked_memory
//! cargo run --release --features parallel --example profile_chunked_memory -- 1000000000
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::sim::compiled::default_chunk_size;
use prism_q::{
    CompiledSampler, HistogramAccumulator, MarginalsAccumulator, PauliExpectationAccumulator,
    compile_measurements,
};

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct TrackingAlloc;

// SAFETY: every method forwards to the System allocator with the caller's
// original layout and only adds atomic byte counting around it, so all
// GlobalAlloc safety obligations are discharged by System.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller's layout unchanged to System.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(now, Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Relaxed);
        // SAFETY: `ptr`/`layout` are the pair returned by our `alloc`.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarding the caller's pointer, layout, and new size to System.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                let now =
                    CURRENT.fetch_add(new_size - layout.size(), Relaxed) + new_size - layout.size();
                PEAK.fetch_max(now, Relaxed);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: TrackingAlloc = TrackingAlloc;

/// Pin the peak to the current live bytes and return that baseline, so a run's
/// peak growth is measured above the memory already resident before it starts.
fn reset_peak() -> usize {
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    base
}

fn peak() -> usize {
    PEAK.load(Relaxed)
}

const SEED: u64 = 0xDEAD_BEEF;
const QUBITS: usize = 100;
const DEFAULT_SHOTS: [usize; 2] = [20_000_000, 100_000_000];
/// Peak may vary run-to-run from allocator jitter and sub-chunk rounding on the
/// smallest shot count; flag growth only when it clears this factor.
const TOLERANCE: f64 = 1.5;

fn measured(mut c: Circuit) -> Circuit {
    let n = c.num_qubits;
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    c
}

fn mib(bytes: usize) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn profile_family(
    name: &str,
    num_meas: usize,
    mut sampler: CompiledSampler,
    shot_counts: &[usize],
    mut run: impl FnMut(&mut CompiledSampler, usize) -> f64,
) -> bool {
    let m_words = num_meas.div_ceil(64);
    let chunk = default_chunk_size(num_meas);
    println!(
        "\n{name}  (chunk_size={chunk} shots, chunk_buf~{})",
        mib(chunk * m_words * 8)
    );

    let mut deltas = Vec::new();
    for &shots in shot_counts {
        let base = reset_peak();
        let checksum = run(&mut sampler, shots);
        let delta = peak().saturating_sub(base);
        deltas.push(delta);
        println!(
            "  shots={shots:>13}  peak_alloc={:>11}  checksum={checksum:.4}",
            mib(delta)
        );
    }

    let first = *deltas.first().unwrap_or(&0);
    let last = *deltas.last().unwrap_or(&0);
    let bounded = last as f64 <= first as f64 * TOLERANCE;
    println!(
        "  -> {}",
        if bounded {
            "bounded"
        } else {
            "GROWS WITH SHOTS"
        }
    );
    bounded
}

fn main() {
    let shot_counts: Vec<usize> = match std::env::args()
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
    {
        Some(extra) => DEFAULT_SHOTS.iter().copied().chain([extra]).collect(),
        None => DEFAULT_SHOTS.to_vec(),
    };

    let clifford = measured(circuits::clifford_heavy_circuit(QUBITS, 10, SEED));
    let ghz = measured(circuits::ghz_circuit(QUBITS));

    let mut ok = true;

    ok &= profile_family(
        "histogram / ghz_100q",
        QUBITS,
        compile_measurements(&ghz, SEED).unwrap(),
        &shot_counts,
        |s, shots| {
            let mut acc = HistogramAccumulator::new();
            s.sample_chunked(shots, &mut acc);
            acc.into_counts().len() as f64
        },
    );

    ok &= profile_family(
        "marginals / clifford_d10_100q",
        QUBITS,
        compile_measurements(&clifford, SEED).unwrap(),
        &shot_counts,
        |s, shots| {
            let mut acc = MarginalsAccumulator::new(QUBITS);
            s.sample_chunked(shots, &mut acc);
            acc.marginals().iter().sum()
        },
    );

    ok &= profile_family(
        "expectation / clifford_d10_100q",
        QUBITS,
        compile_measurements(&clifford, SEED).unwrap(),
        &shot_counts,
        |s, shots| {
            let mut acc =
                PauliExpectationAccumulator::new(vec![vec![0, 1], vec![10, 20], vec![50, 99]]);
            s.sample_chunked(shots, &mut acc);
            acc.expectations().iter().sum()
        },
    );

    if ok {
        println!("\nPASS: peak allocation stays bounded by chunk_size across shot counts");
    } else {
        eprintln!("\nFAIL: peak allocation grew with total shots");
        std::process::exit(1);
    }
}
