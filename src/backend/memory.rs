//! Memory detection and the `PRISM_MAX_*` environment layer that resource limits
//! resolve over, plus the width checks backends apply against a resolved cap.

use std::mem::size_of;
use std::sync::OnceLock;

use crate::error::{PrismError, ResourceKind, Result};
use crate::limits::{Limit, LimitKnob, LimitSource, Limits, ResolvedLimits, resource_limit_error};

const MEMORY_BUDGET_DIVISOR: u64 = 2;

/// Default memory budget: half of detected physical memory, `None` when detection
/// fails.
pub(crate) fn detected_budget_bytes() -> Option<u64> {
    static CACHED: OnceLock<Option<u64>> = OnceLock::new();
    *CACHED.get_or_init(|| detect_physical_memory_bytes().map(|b| b / MEMORY_BUDGET_DIVISOR))
}

/// The `PRISM_MAX_*` variables, read once per process. A value that is not a count
/// warns and is ignored, so detection applies rather than a typo silently lifting a
/// cap.
pub(crate) struct EnvLimits {
    pub(crate) limits: Limits,
    /// `PRISM_MAX_DENSE_OUTCOME_BITS`, the one environment cap with no [`Limits`]
    /// field.
    pub(crate) dense_outcome_bits: Option<usize>,
}

pub(crate) fn env_limits() -> &'static EnvLimits {
    static CACHED: OnceLock<EnvLimits> = OnceLock::new();
    CACHED.get_or_init(|| {
        let qubits = |var| crate::env_knobs::usize_override(var, 0);
        let mut limits = Limits::new();
        limits.max_statevector_qubits = qubits("PRISM_MAX_SV_QUBITS");
        limits.max_density_matrix_qubits = qubits("PRISM_MAX_DM_QUBITS");
        limits.max_probability_qubits = qubits("PRISM_MAX_PROB_QUBITS");
        limits.max_export_qubits = qubits("PRISM_MAX_EXPORT_QUBITS");
        limits.max_sparse_qubits = qubits("PRISM_MAX_SPARSE_QUBITS");
        limits.max_factored_merge_qubits = qubits("PRISM_MAX_FACTORED_MERGE_QUBITS");
        limits.max_stabilizer_cluster_qubits = qubits("PRISM_MAX_STABILIZER_CLUSTER_QUBITS");
        limits.max_mps_workspace_qubits = qubits("PRISM_MAX_MPS_WORKSPACE_QUBITS");
        limits.max_tn_peak_qubits = qubits("PRISM_MAX_TN_PEAK_QUBITS");
        limits.max_tn_slices = crate::env_knobs::usize_override("PRISM_MAX_TN_SLICES", 1);
        EnvLimits {
            limits,
            dense_outcome_bits: qubits("PRISM_MAX_DENSE_OUTCOME_BITS"),
        }
    })
}

/// Error for a stabilizer cluster merge of `total_n` qubits over the cluster
/// budget, raised before the joint tableau allocates.
#[cold]
pub(crate) fn stabilizer_cluster_error(total_n: usize, cap: Limit) -> PrismError {
    resource_limit_error(
        "factored-stabilizer",
        "the joint tableau of merged clusters",
        ResourceKind::Qubits,
        total_n as u128,
        cap.value as u128,
        LimitKnob::StabilizerClusterQubits,
        cap.source,
    )
}

/// Error for a dense transient workspace of `elements` amplitudes over the
/// MPS workspace budget: the growth-path analogue of the
/// [`check_state_allocation`] rejection, for scratch whose natural unit is
/// amplitudes rather than circuit qubits.
#[cold]
pub(crate) fn workspace_allocation_error(
    backend: &str,
    what: &str,
    elements: u128,
    limits: &ResolvedLimits,
) -> PrismError {
    resource_limit_error(
        backend,
        what,
        ResourceKind::Amplitudes,
        elements,
        limits.mps_workspace_elements(),
        LimitKnob::MpsWorkspaceQubits,
        limits.mps_workspace_qubits.source,
    )
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
    cap: Limit,
    knob: LimitKnob,
) -> Result<()> {
    if num_qubits >= usize::BITS as usize {
        return Err(PrismError::ResourceLimit(Box::new(
            crate::error::ResourceLimit {
                backend: backend.to_string(),
                operation: "the circuit".to_string(),
                resource: ResourceKind::Qubits,
                required: num_qubits as u128,
                limit: usize::BITS as u128 - 1,
                env_var: None,
                knob: None,
                limit_source: LimitSource::Detected,
            },
        )));
    }
    if num_qubits > cap.value {
        return Err(resource_limit_error(
            backend,
            "the circuit",
            ResourceKind::Qubits,
            num_qubits as u128,
            cap.value as u128,
            knob,
            cap.source,
        ));
    }
    Ok(())
}

/// Width check for the dense probability terminal of a backend that exists to
/// run past the dense cap (sparse, MPS, stabilizer, factored, product).
///
/// Raises `BackendUnsupported`, which the dispatch layer reads from a
/// probability query as "no dense terminal" and reports as `None`: a wide
/// register on these backends is the normal case, and `run` still answers
/// with the classical bits.
pub(crate) fn dense_probability_len(
    backend: &str,
    num_qubits: usize,
    limits: &ResolvedLimits,
) -> Result<usize> {
    dense_output_len(
        backend,
        "probabilities",
        num_qubits,
        size_of::<f64>(),
        limits.probability_qubits.value,
    )
}

/// Width check for the statevector's dense probability terminal.
///
/// The register already fits the state, so a width past the probability cap is a
/// cap to name rather than a terminal that does not exist: raises `ResourceLimit`,
/// which the dispatch layer's `None` arm does not absorb.
pub(crate) fn statevector_probability_len(
    backend: &str,
    num_qubits: usize,
    limits: &ResolvedLimits,
) -> Result<usize> {
    capped_output_len(
        backend,
        "probabilities",
        num_qubits,
        limits.probability_qubits,
        LimitKnob::ProbabilityQubits,
    )
}

/// Width check for a statevector export. Every backend exports through it, and
/// nothing absorbs the error, so the cap surfaces naming itself.
pub(crate) fn dense_statevector_len(
    backend: &str,
    operation: &str,
    num_qubits: usize,
    limits: &ResolvedLimits,
) -> Result<usize> {
    capped_output_len(
        backend,
        operation,
        num_qubits,
        limits.export_qubits,
        LimitKnob::ExportQubits,
    )
}

/// Width check for the tensor-network dense probability terminal, raising
/// `ResourceLimit` for the same reason as [`statevector_probability_len`].
pub(crate) fn tensor_probability_len(
    backend: &str,
    num_qubits: usize,
    limits: &ResolvedLimits,
) -> Result<usize> {
    capped_output_len(
        backend,
        "probabilities",
        num_qubits,
        limits.tensor_probability_qubits,
        LimitKnob::ProbabilityQubits,
    )
}

fn capped_output_len(
    backend: &str,
    operation: &str,
    num_qubits: usize,
    cap: Limit,
    knob: LimitKnob,
) -> Result<usize> {
    let max = cap.value.min(usize::BITS as usize - 1);
    if num_qubits > max {
        return Err(resource_limit_error(
            backend,
            operation,
            ResourceKind::Qubits,
            num_qubits as u128,
            max as u128,
            knob,
            cap.source,
        ));
    }
    Ok(1usize << num_qubits)
}

/// Rejection for a contraction whose peak of `peak` elements is still over
/// `cap` once slicing has been tried, raised before the replay allocates it.
#[cold]
pub(crate) fn tensor_peak_error(
    backend: &str,
    operation: &str,
    peak: usize,
    cap: usize,
    source: LimitSource,
) -> PrismError {
    resource_limit_error(
        backend,
        format!("the peak intermediate of {operation}"),
        ResourceKind::Elements,
        peak as u128,
        cap as u128,
        LimitKnob::TnPeakQubits,
        source,
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

    #[test]
    fn capped_output_past_the_cap_names_it() {
        let cap = Limit {
            value: 4,
            source: LimitSource::Env,
        };
        let err = capped_output_len(
            "test",
            "probabilities",
            5,
            cap,
            LimitKnob::ProbabilityQubits,
        )
        .unwrap_err();
        match err {
            PrismError::ResourceLimit(data) => {
                let crate::error::ResourceLimit {
                    required,
                    limit,
                    env_var,
                    knob,
                    limit_source,
                    ..
                } = crate::error::ResourceLimit::clone(&data);
                assert_eq!((required, limit), (5, 4));
                assert_eq!(env_var, Some("PRISM_MAX_PROB_QUBITS"));
                assert_eq!(knob, Some(LimitKnob::ProbabilityQubits));
                assert_eq!(limit_source, LimitSource::Env);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
