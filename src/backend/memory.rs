//! Memory caps for dense state allocation and dense outputs, derived from
//! detected physical memory with a per-cap environment variable override.

use std::mem::size_of;

use num_complex::Complex64;

use crate::error::{PrismError, Result};

const MEMORY_BUDGET_DIVISOR: u64 = 2;

pub(crate) fn max_statevector_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        configured_or_detected_dense_qubits(
            "PRISM_MAX_SV_QUBITS",
            size_of::<Complex64>(),
            "statevector qubit cap",
        )
    })
}

/// Environment advice for the density-matrix cap. Both variables bind: the
/// backend allocates its `4^n` state as a `2n`-qubit statevector, so
/// `PRISM_MAX_DM_QUBITS` raises the cap only as far as half the statevector
/// cap, and going past that needs `PRISM_MAX_SV_QUBITS` raised with it.
pub(crate) const DM_QUBIT_CAP_ENV: &str = "PRISM_MAX_DM_QUBITS and PRISM_MAX_SV_QUBITS";

/// Qubit cap for the exact density-matrix backend. Its state is `4^n`
/// `Complex64` entries, the element count of a `2n`-qubit statevector, so the
/// cap is `floor(cap_sv / 2)`: 14 on a 16 GiB host, 15 on 32 GiB.
/// `PRISM_MAX_DM_QUBITS` moves it within that bound; see [`DM_QUBIT_CAP_ENV`].
/// This is the only density-matrix cap, so dispatch-time validation and the
/// backend's own `init` guard cannot disagree about where the ceiling is.
pub(crate) fn max_density_matrix_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let budget = max_statevector_qubits() / 2;
        env_qubit_override("PRISM_MAX_DM_QUBITS").map_or(budget, |n| n.min(budget))
    })
}

/// Read a qubit-cap override from `env_var`, returning `None` when unset or
/// unparseable.
fn env_qubit_override(env_var: &str) -> Option<usize> {
    std::env::var(env_var).ok().and_then(|val| val.parse().ok())
}

pub(crate) fn max_dense_probability_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        configured_or_detected_dense_qubits(
            "PRISM_MAX_PROB_QUBITS",
            size_of::<f64>(),
            "dense probability cap",
        )
    })
}

pub(crate) fn max_dense_statevector_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        configured_or_detected_dense_qubits(
            "PRISM_MAX_EXPORT_QUBITS",
            size_of::<Complex64>(),
            "dense statevector export cap",
        )
    })
}

pub(crate) fn max_tensor_probability_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        configured_or_detected_dense_qubits(
            "PRISM_MAX_PROB_QUBITS",
            size_of::<Complex64>() + size_of::<f64>(),
            "tensor-network dense probability cap",
        )
    })
}

/// Measured-bit cap for the dense terminal-sampling path, which materializes
/// one outcome distribution plus one CDF (two f64 per outcome). Above the cap
/// the samplers stream with sorted thresholds instead.
pub(crate) fn max_dense_outcome_bits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        configured_or_detected_dense_qubits(
            "PRISM_MAX_DENSE_OUTCOME_BITS",
            2 * size_of::<f64>(),
            "dense outcome sampling cap",
        )
    })
}

/// Entry cap for the sparse backend's amplitude map.
///
/// Derived from the memory budget at 64 bytes per entry, the worst-case peak
/// across the double-buffered maps (24 payload bytes, table slot overhead, and
/// load-factor headroom on both buffers). `PRISM_MAX_SPARSE_QUBITS` overrides
/// the cap as a power of two: the map may hold at most `2^q` entries.
pub(crate) fn max_sparse_entries() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let q =
            configured_or_detected_dense_qubits("PRISM_MAX_SPARSE_QUBITS", 64, "sparse entry cap");
        if q >= usize::BITS as usize - 1 {
            usize::MAX
        } else {
            1usize << q
        }
    })
}

/// Merged-block qubit cap for the factored backend, checked at merge time.
///
/// Defaults to the detected-memory dense budget, independent of the
/// statevector override: lowering `PRISM_MAX_SV_QUBITS` to steer routing must
/// not shrink the block a factored run may legitimately hold.
pub(crate) fn max_factored_merge_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        configured_or_detected_dense_qubits(
            "PRISM_MAX_FACTORED_MERGE_QUBITS",
            size_of::<Complex64>(),
            "factored merge cap",
        )
    })
}

/// Words a stabilizer tableau of `n` qubits occupies: `2n + 1` rows of
/// `2 * ceil(n / 64)` words each.
fn stabilizer_tableau_words(n: u128) -> u128 {
    (2 * n + 1) * 2 * n.div_ceil(64)
}

/// Merged-cluster qubit cap for the factored stabilizer backend, checked at
/// merge time.
///
/// Separate from [`max_factored_merge_qubits`] because the resources differ in
/// kind: a factored merge allocates `2^n` amplitudes, while a stabilizer merge
/// allocates a tableau of `O(n^2 / 64)` words. Reusing the dense cap here would
/// reject the first merge of a 128-qubit Clifford circuit, which is the shape
/// dispatch selects this backend for. `PRISM_MAX_STABILIZER_CLUSTER_QUBITS`
/// overrides it.
pub(crate) fn max_stabilizer_cluster_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Some(n) = env_qubit_override("PRISM_MAX_STABILIZER_CLUSTER_QUBITS") {
            return n;
        }
        match detect_physical_memory_bytes() {
            Some(bytes) => {
                // The old pair is still live when the joint tableau allocates,
                // and is strictly smaller than it, so peak is under twice the
                // merged size.
                let budget_words = u128::from(bytes / MEMORY_BUDGET_DIVISOR) / 8 / 2;
                let mut n: u128 = 1;
                while stabilizer_tableau_words(2 * n) <= budget_words
                    && 2 * n < u128::from(u32::MAX)
                {
                    n *= 2;
                }
                let mut step = n / 2;
                while step > 0 {
                    if stabilizer_tableau_words(n + step) <= budget_words {
                        n += step;
                    }
                    step /= 2;
                }
                usize::try_from(n).unwrap_or(usize::MAX)
            }
            None => {
                eprintln!(
                    "warning: could not detect system memory; stabilizer cluster cap is \
                     disabled. Large merges may abort on allocation. Set \
                     PRISM_MAX_STABILIZER_CLUSTER_QUBITS to suppress."
                );
                usize::MAX
            }
        }
    })
}

/// Error for a stabilizer cluster merge of `total_n` qubits over the cluster
/// budget, raised before the joint tableau allocates.
#[cold]
pub(crate) fn stabilizer_cluster_error(total_n: usize, cap: usize) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: "factored-stabilizer".to_string(),
        reason: format!(
            "merging entangled clusters needs a {total_n}-qubit joint tableau, exceeding \
             the cap of {cap} on this machine \
             (set PRISM_MAX_STABILIZER_CLUSTER_QUBITS to override)"
        ),
    }
}

/// Workspace cap for MPS gate application, as `2^q` amplitudes of live
/// contraction buffers. Independent of the statevector override for the same
/// reason as the factored merge cap: MPS exists to run above that cap.
fn max_mps_workspace_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        configured_or_detected_dense_qubits(
            "PRISM_MAX_MPS_WORKSPACE_QUBITS",
            size_of::<Complex64>(),
            "MPS workspace cap",
        )
    })
}

/// MPS workspace cap in amplitudes: `2^q` from the budget above, or
/// `u128::MAX` when detection is disabled. Read once per backend at
/// construction so the per-gate check is a field compare, not an atomic load.
pub(crate) fn mps_workspace_cap_elements() -> u128 {
    let cap = max_mps_workspace_qubits();
    if cap >= u128::BITS as usize {
        u128::MAX
    } else {
        1u128 << cap
    }
}

/// Error for a dense transient workspace of `elements` amplitudes over the
/// MPS workspace budget: the growth-path analogue of the
/// [`check_state_allocation`] rejection, for scratch whose natural unit is
/// amplitudes rather than circuit qubits.
#[cold]
pub(crate) fn workspace_allocation_error(backend: &str, what: &str, elements: u128) -> PrismError {
    let cap = max_mps_workspace_qubits();
    PrismError::IncompatibleBackend {
        backend: backend.to_string(),
        reason: format!(
            "{what} needs {elements} amplitudes of workspace, exceeding the cap of \
             2^{cap} on this machine (set PRISM_MAX_MPS_WORKSPACE_QUBITS to override)"
        ),
    }
}

fn configured_or_detected_dense_qubits(
    env_var: &str,
    bytes_per_basis_state: usize,
    warning_label: &str,
) -> usize {
    if let Some(n) = env_qubit_override(env_var) {
        return n;
    }
    match detect_physical_memory_bytes().and_then(|bytes| {
        max_dense_qubits_for_budget(bytes / MEMORY_BUDGET_DIVISOR, bytes_per_basis_state)
    }) {
        Some(n) => n,
        None => {
            eprintln!(
                "warning: could not detect system memory; {warning_label} is disabled. \
                 Large outputs may abort on allocation. Set {env_var} to suppress."
            );
            usize::MAX
        }
    }
}

fn max_dense_qubits_for_budget(budget_bytes: u64, bytes_per_basis_state: usize) -> Option<usize> {
    let bytes_per_basis_state = u64::try_from(bytes_per_basis_state).ok()?;
    if bytes_per_basis_state == 0 {
        return None;
    }
    let max_elements = budget_bytes / bytes_per_basis_state;
    if max_elements == 0 {
        return None;
    }
    let max_qubits = (u64::BITS - 1 - max_elements.leading_zeros()) as usize;
    Some(max_qubits.min(usize::BITS as usize - 1))
}

/// Reject a dense state that would exceed `cap` before the backend reserves it.
///
/// Backends call this at the top of `init`, the one point every execution path
/// passes through before allocating, so a caller that drives a backend directly
/// through [`run_on`](crate::run_on) cannot bypass the dispatch-level caps and
/// reach an allocation the machine cannot satisfy. Exceeding a cap is an error,
/// never a silent fallback to another backend.
pub(crate) fn check_state_allocation(
    backend: &str,
    num_qubits: usize,
    cap: usize,
    env_var: &str,
) -> Result<()> {
    if num_qubits >= usize::BITS as usize {
        return Err(PrismError::IncompatibleBackend {
            backend: backend.to_string(),
            reason: format!("circuit has {num_qubits} qubits, exceeding addressable memory"),
        });
    }
    if num_qubits > cap {
        return Err(PrismError::IncompatibleBackend {
            backend: backend.to_string(),
            reason: format!(
                "circuit has {num_qubits} qubits, exceeding the cap of {cap} on this machine \
                 (set {env_var} to override)"
            ),
        });
    }
    Ok(())
}

pub(crate) fn dense_probability_len(backend: &str, num_qubits: usize) -> Result<usize> {
    dense_output_len(
        backend,
        "probabilities",
        num_qubits,
        size_of::<f64>(),
        max_dense_probability_qubits(),
    )
}

pub(crate) fn dense_statevector_len(
    backend: &str,
    operation: &str,
    num_qubits: usize,
) -> Result<usize> {
    dense_output_len(
        backend,
        operation,
        num_qubits,
        size_of::<Complex64>(),
        max_dense_statevector_qubits(),
    )
}

pub(crate) fn tensor_probability_len(backend: &str, num_qubits: usize) -> Result<usize> {
    dense_output_len(
        backend,
        "probabilities",
        num_qubits,
        size_of::<Complex64>() + size_of::<f64>(),
        max_tensor_probability_qubits(),
    )
}

fn dense_output_len(
    backend: &str,
    operation: &str,
    num_qubits: usize,
    bytes_per_basis_state: usize,
    max_qubits: usize,
) -> Result<usize> {
    if num_qubits >= usize::BITS as usize {
        return Err(PrismError::BackendUnsupported {
            backend: backend.to_string(),
            operation: format!("{operation} for {num_qubits} qubits (exceeds addressable memory)"),
        });
    }
    if num_qubits > max_qubits {
        return Err(PrismError::BackendUnsupported {
            backend: backend.to_string(),
            operation: format!(
                "{operation} for {num_qubits} qubits (max {max_qubits} on this machine, {} bytes required)",
                required_dense_bytes(num_qubits, bytes_per_basis_state)
            ),
        });
    }
    Ok(1usize << num_qubits)
}

pub(crate) fn reserve_dense_output<T>(
    out: &mut Vec<T>,
    len: usize,
    backend: &str,
    operation: &str,
) -> Result<()> {
    out.try_reserve_exact(len)
        .map_err(|_| PrismError::BackendUnsupported {
            backend: backend.to_string(),
            operation: format!(
                "{operation} for {} elements ({} bytes required)",
                len,
                len.saturating_mul(size_of::<T>())
            ),
        })
}

fn required_dense_bytes(num_qubits: usize, bytes_per_basis_state: usize) -> usize {
    (1usize << num_qubits).saturating_mul(bytes_per_basis_state)
}

#[cfg(windows)]
fn detect_physical_memory_bytes() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    // SAFETY: signature matches the documented kernel32 GlobalMemoryStatusEx
    // ABI: one pointer argument to a struct whose dw_length field is set
    // before the call, returning BOOL as i32.
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }

    // SAFETY: MemoryStatusEx is a repr(C) data struct and the all-zero pattern is valid.
    let mut status: MemoryStatusEx = unsafe { std::mem::zeroed() };
    status.dw_length = size_of::<MemoryStatusEx>() as u32;
    // SAFETY: status points to a valid MemoryStatusEx with dw_length set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return None;
    }

    Some(status.ull_total_phys)
}

#[cfg(target_os = "macos")]
fn detect_physical_memory_bytes() -> Option<u64> {
    // SAFETY: signature matches the documented libSystem sysctlbyname ABI:
    // a C-string name, an output buffer with its length passed by pointer,
    // and an unused input buffer, returning 0 on success.
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }

    let mut memsize: u64 = 0;
    let mut len = size_of::<u64>();
    // SAFETY: oldp points to an 8-byte buffer and oldlenp holds its size;
    // hw.memsize is a u64 sysctl.
    let ret = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut memsize as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 || len != size_of::<u64>() || memsize == 0 {
        return None;
    }
    Some(memsize)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn detect_physical_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

#[cfg(not(any(windows, unix)))]
fn detect_physical_memory_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_budget_counts_fit_elements() {
        assert_eq!(max_dense_qubits_for_budget(8, 8), Some(0));
        assert_eq!(max_dense_qubits_for_budget(16, 8), Some(1));
        assert_eq!(max_dense_qubits_for_budget(31, 8), Some(1));
        assert_eq!(max_dense_qubits_for_budget(32, 8), Some(2));
    }

    #[test]
    fn dense_output_rejects_unaddressable_shift() {
        let err = dense_output_len("test", "probabilities", usize::BITS as usize, 8, usize::MAX)
            .unwrap_err();
        match err {
            PrismError::BackendUnsupported { operation, .. } => {
                assert!(operation.contains("exceeds addressable memory"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
