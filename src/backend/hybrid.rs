//! Performance cores on hybrid CPUs, and the narrower Rayon pool that routes which lose
//! to efficiency cores run on there.

use std::sync::OnceLock;

/// Widest statevector run that takes the narrow pool. On an 8P+16E host eight workers
/// beat 24 by 14.5% at 16 qubits, read even at 17 and lost 15.9% at 20.
pub(crate) const MAX_NARROW_STATEVECTOR_QUBITS: usize = 16;

/// The highest efficiency class on a hybrid CPU: its logical CPU count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PerformanceCores {
    pub(crate) count: usize,
}

/// Read the performance cores once per process. `None` on a CPU with one core class.
pub(crate) fn performance_cores() -> Option<PerformanceCores> {
    static CACHED: OnceLock<Option<PerformanceCores>> = OnceLock::new();
    *CACHED.get_or_init(detect)
}

/// Whether a run of `route` over `width` qubits should take the narrow pool: MPS runs,
/// whose faer kernels split evenly and wait at a barrier for the slowest worker, and
/// statevector runs from the parallel threshold to [`MAX_NARROW_STATEVECTOR_QUBITS`].
pub(crate) fn route_wants_narrow_pool(route: crate::sim::ResolvedBackend, width: usize) -> bool {
    use crate::sim::ResolvedBackend;
    match route {
        ResolvedBackend::Mps => true,
        ResolvedBackend::Statevector => {
            (super::PARALLEL_THRESHOLD_QUBITS..=MAX_NARROW_STATEVECTOR_QUBITS).contains(&width)
        }
        _ => false,
    }
}

/// Run `f` on the performance core pool when `wanted`, or on the current pool. The
/// current pool stays when the caller already runs inside a Rayon pool, when
/// `RAYON_NUM_THREADS` is set, and on a CPU with one core class.
pub(crate) fn on_narrow_pool<R: Send>(wanted: bool, f: impl FnOnce() -> R + Send) -> R {
    match narrow_pool().filter(|_| wanted) {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

fn narrow_pool() -> Option<&'static rayon::ThreadPool> {
    if rayon::current_thread_index().is_some() || std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return None;
    }
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let cores = performance_cores()?;
        rayon::ThreadPoolBuilder::new()
            .num_threads(cores.count)
            .thread_name(|index| format!("prism-perf-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// Count the logical CPUs of the highest class in `(efficiency_class, group, mask)` core
/// records: `None` unless two classes exist.
#[cfg_attr(not(all(windows, not(miri))), allow(dead_code))]
fn classify(cores: impl IntoIterator<Item = (u8, u16, usize)> + Clone) -> Option<PerformanceCores> {
    let (lowest, highest) =
        cores
            .clone()
            .into_iter()
            .fold(None, |range: Option<(u8, u8)>, (class, _, _)| {
                Some(range.map_or((class, class), |(lo, hi)| (lo.min(class), hi.max(class))))
            })?;
    if lowest == highest {
        return None;
    }
    let count = cores
        .into_iter()
        .filter(|&(class, _, _)| class == highest)
        .map(|(_, _, core_mask)| core_mask.count_ones() as usize)
        .sum();
    Some(PerformanceCores { count })
}

#[cfg(all(windows, not(miri)))]
fn detect() -> Option<PerformanceCores> {
    // SAFETY: signature matches the documented kernel32 GetLogicalProcessorInformationEx
    // ABI: a relationship type, a byte buffer and an in/out byte length, returning BOOL.
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
    }
    const RELATION_PROCESSOR_CORE: u32 = 0;

    let mut len = 0u32;
    // SAFETY: a null buffer with a zero length asks for the required length only.
    unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len)
    };
    if len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    // SAFETY: buf holds len bytes.
    if unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len)
    } == 0
    {
        return None;
    }
    buf.truncate(len as usize);
    classify(processor_core_records(&buf))
}

/// `(efficiency_class, group, mask)` per core from a `RelationProcessorCore` buffer: an
/// 8-byte record header, then `PROCESSOR_RELATIONSHIP` with the class at byte 9 and the
/// first `GROUP_AFFINITY` (mask, group) at byte 32. A core spans one group.
#[cfg_attr(not(all(windows, not(miri))), allow(dead_code))]
fn processor_core_records(buf: &[u8]) -> impl Iterator<Item = (u8, u16, usize)> + Clone + '_ {
    let word = |bytes: &[u8], at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let mut offset = 0usize;
    std::iter::from_fn(move || {
        let rest = buf.get(offset..)?;
        if rest.len() < 48 {
            return None;
        }
        let size = word(rest, 4) as usize;
        if size < 48 || size > rest.len() {
            return None;
        }
        offset += size;
        let class = rest[9];
        let mask = u64::from_le_bytes(rest[32..40].try_into().unwrap()) as usize;
        let group = u16::from_le_bytes(rest[40..42].try_into().unwrap());
        Some((class, group, mask))
    })
}

#[cfg(all(unix, not(target_os = "macos"), not(miri)))]
fn detect() -> Option<PerformanceCores> {
    let list = |path: &str| std::fs::read_to_string(path).ok();
    list("/sys/devices/cpu_atom/cpus").and_then(|atoms| super::cache::count_cpu_list(&atoms))?;
    let count = list("/sys/devices/cpu_core/cpus")
        .and_then(|cores| super::cache::count_cpu_list(&cores))?;
    Some(PerformanceCores { count })
}

#[cfg(any(miri, target_os = "macos", not(any(windows, unix))))]
fn detect() -> Option<PerformanceCores> {
    None
}

#[cfg(test)]
#[path = "hybrid_tests.rs"]
mod tests;
