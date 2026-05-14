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

fn configured_or_detected_dense_qubits(
    env_var: &str,
    bytes_per_basis_state: usize,
    warning_label: &str,
) -> usize {
    if let Ok(val) = std::env::var(env_var) {
        if let Ok(n) = val.parse::<usize>() {
            return n;
        }
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

    extern "system" {
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

#[cfg(unix)]
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
