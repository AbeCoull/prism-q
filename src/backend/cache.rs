//! Last-level cache size and sharing, read from the OS once per process.

use std::sync::OnceLock;

/// The largest data or unified cache and the number of logical CPUs that share one
/// instance of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SharedCache {
    pub(crate) bytes: usize,
    pub(crate) sharing: usize,
}

/// Read the last-level cache once per process. `None` when the OS does not report one,
/// and callers keep their untuned behavior.
pub(crate) fn last_level_cache() -> Option<SharedCache> {
    static CACHED: OnceLock<Option<SharedCache>> = OnceLock::new();
    *CACHED.get_or_init(detect)
}

/// Whether `workers` tiles of `tile_bytes` each fit one instance of `cache` together,
/// within half its capacity: an inclusive cache also holds every private cache below
/// it, and tiles that fill it exactly still evict each other.
pub(crate) fn tiles_fit(cache: SharedCache, tile_bytes: usize, workers: usize) -> bool {
    let concurrent = workers.min(cache.sharing).max(1);
    tile_bytes.saturating_mul(concurrent) <= cache.bytes / 2
}

#[cfg(all(windows, not(miri)))]
fn detect() -> Option<SharedCache> {
    // The union after `relationship` holds a `ULONGLONG[2]`, so it is 8-byte aligned;
    // its `CACHE_DESCRIPTOR` arm is level, associativity, line size, size, type.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct LogicalProcessorInformation {
        processor_mask: usize,
        relationship: u32,
        union: [u64; 2],
    }

    impl LogicalProcessorInformation {
        fn cache(&self) -> (u8, u32, u32) {
            let lo = self.union[0].to_le_bytes();
            let hi = self.union[1].to_le_bytes();
            let size = u32::from_le_bytes(lo[4..8].try_into().unwrap());
            let cache_type = u32::from_le_bytes(hi[..4].try_into().unwrap());
            (lo[0], size, cache_type)
        }
    }

    // SAFETY: signature matches the documented kernel32 GetLogicalProcessorInformation
    // ABI: a buffer pointer and an in/out byte length, returning BOOL as i32.
    unsafe extern "system" {
        fn GetLogicalProcessorInformation(
            buffer: *mut LogicalProcessorInformation,
            returned_length: *mut u32,
        ) -> i32;
    }

    const RELATION_CACHE: u32 = 2;
    const CACHE_UNIFIED: u32 = 0;
    const CACHE_DATA: u32 = 2;

    let entry = std::mem::size_of::<LogicalProcessorInformation>();
    let mut len = 0u32;
    // SAFETY: a null buffer with a zero length asks for the required length only.
    unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len) };
    if len == 0 {
        return None;
    }
    let count = len as usize / entry;
    // SAFETY: LogicalProcessorInformation is a repr(C) data struct and the all-zero
    // pattern is valid.
    let mut infos = vec![unsafe { std::mem::zeroed::<LogicalProcessorInformation>() }; count];
    let mut len = (count * entry) as u32;
    // SAFETY: infos holds `count` entries and len is their size in bytes.
    if unsafe { GetLogicalProcessorInformation(infos.as_mut_ptr(), &mut len) } == 0 {
        return None;
    }
    infos.truncate(len as usize / entry);

    infos
        .iter()
        .filter(|info| info.relationship == RELATION_CACHE)
        .filter_map(|info| {
            let (level, size, cache_type) = info.cache();
            let kept = (cache_type == CACHE_UNIFIED || cache_type == CACHE_DATA) && size > 0;
            kept.then_some((level, size, info.processor_mask.count_ones() as usize))
        })
        .max_by_key(|&(level, size, _)| (level, size))
        .map(|(_, size, sharing)| SharedCache {
            bytes: size as usize,
            sharing: sharing.max(1),
        })
}

#[cfg(all(target_os = "macos", not(miri)))]
fn detect() -> Option<SharedCache> {
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

    let read = |name: &std::ffi::CStr| -> Option<u64> {
        let mut value = [0u8; 8];
        let mut len = value.len();
        // SAFETY: oldp points to an 8-byte buffer and oldlenp holds its size; the
        // integer sysctls read here are 4 or 8 bytes, little-endian on every macOS
        // target.
        let ret = unsafe {
            sysctlbyname(
                name.as_ptr(),
                value.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        match (ret, len) {
            (0, 4) => Some(u32::from_le_bytes(value[..4].try_into().unwrap()) as u64),
            (0, 8) => Some(u64::from_le_bytes(value)),
            _ => None,
        }
        .filter(|&v| v > 0)
    };

    if let Some(bytes) = read(c"hw.l3cachesize") {
        let sharing = read(c"hw.logicalcpu").unwrap_or(1);
        return Some(SharedCache {
            bytes: bytes as usize,
            sharing: sharing as usize,
        });
    }
    let bytes = read(c"hw.perflevel0.l2cachesize").or_else(|| read(c"hw.l2cachesize"))?;
    let sharing = read(c"hw.perflevel0.cpusperl2").unwrap_or(1);
    Some(SharedCache {
        bytes: bytes as usize,
        sharing: sharing as usize,
    })
}

#[cfg(all(unix, not(target_os = "macos"), not(miri)))]
fn detect() -> Option<SharedCache> {
    let mut best: Option<(u32, SharedCache)> = None;
    for index in 0.. {
        let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
        let Ok(level) = std::fs::read_to_string(format!("{dir}/level")) else {
            break;
        };
        let kind = std::fs::read_to_string(format!("{dir}/type")).unwrap_or_default();
        if !matches!(kind.trim(), "Unified" | "Data") {
            continue;
        }
        let (Ok(level), Some(bytes)) = (
            level.trim().parse::<u32>(),
            std::fs::read_to_string(format!("{dir}/size"))
                .ok()
                .and_then(|size| parse_sysfs_size(&size)),
        ) else {
            continue;
        };
        let sharing = std::fs::read_to_string(format!("{dir}/shared_cpu_list"))
            .ok()
            .and_then(|list| count_cpu_list(&list))
            .unwrap_or(1);
        let cache = SharedCache { bytes, sharing };
        if best.is_none_or(|(l, b)| (level, bytes) > (l, b.bytes)) {
            best = Some((level, cache));
        }
    }
    best.map(|(_, cache)| cache)
}

#[cfg(any(miri, not(any(windows, unix))))]
fn detect() -> Option<SharedCache> {
    None
}

/// Parse a sysfs cache size such as `32K`, `8192K` or `1M`.
#[cfg_attr(not(all(unix, not(target_os = "macos"), not(miri))), allow(dead_code))]
fn parse_sysfs_size(text: &str) -> Option<usize> {
    let text = text.trim();
    let (digits, scale) = match text.as_bytes().last()? {
        b'K' => (&text[..text.len() - 1], 1 << 10),
        b'M' => (&text[..text.len() - 1], 1 << 20),
        b'G' => (&text[..text.len() - 1], 1 << 30),
        _ => (text, 1),
    };
    digits
        .parse::<usize>()
        .ok()?
        .checked_mul(scale)
        .filter(|&b| b > 0)
}

/// Count the CPUs in a sysfs list such as `0-7` or `0-3,8-11`.
#[cfg_attr(not(all(unix, not(target_os = "macos"), not(miri))), allow(dead_code))]
fn count_cpu_list(text: &str) -> Option<usize> {
    let mut count = 0usize;
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        count += match part.split_once('-') {
            Some((lo, hi)) => hi.parse::<usize>().ok()?.checked_sub(lo.parse().ok()?)? + 1,
            None => {
                part.parse::<usize>().ok()?;
                1
            }
        };
    }
    (count > 0).then_some(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysfs_sizes_parse_with_and_without_suffix() {
        assert_eq!(parse_sysfs_size("32K\n"), Some(32 << 10));
        assert_eq!(parse_sysfs_size("8192K"), Some(8 << 20));
        assert_eq!(parse_sysfs_size("1M"), Some(1 << 20));
        assert_eq!(parse_sysfs_size("4096"), Some(4096));
        assert_eq!(parse_sysfs_size("0K"), None);
        assert_eq!(parse_sysfs_size("K"), None);
    }

    #[test]
    fn cpu_lists_count_ranges_and_singletons() {
        assert_eq!(count_cpu_list("0-7\n"), Some(8));
        assert_eq!(count_cpu_list("0-3,8-11"), Some(8));
        assert_eq!(count_cpu_list("0,4"), Some(2));
        assert_eq!(count_cpu_list("5"), Some(1));
        assert_eq!(count_cpu_list(""), None);
        assert_eq!(count_cpu_list("7-3"), None);
    }

    #[test]
    fn tiles_fit_counts_only_the_workers_sharing_one_cache() {
        let skylake = SharedCache {
            bytes: 8 << 20,
            sharing: 8,
        };
        assert!(!tiles_fit(skylake, 2 << 20, 8));
        assert!(!tiles_fit(skylake, 2 << 20, 4));
        assert!(tiles_fit(skylake, 2 << 20, 2));
        assert!(tiles_fit(skylake, 2 << 20, 1));

        let stacked = SharedCache {
            bytes: 96 << 20,
            sharing: 16,
        };
        assert!(tiles_fit(stacked, 2 << 20, 32));
        assert!(!tiles_fit(stacked, 4 << 20, 32));
        assert!(tiles_fit(stacked, 2 << 20, 0));
    }

    #[cfg(all(
        not(miri),
        any(
            windows,
            target_os = "macos",
            all(target_os = "linux", target_arch = "x86_64")
        )
    ))]
    #[test]
    fn last_level_cache_is_reported_on_supported_hosts() {
        let cache = last_level_cache().expect("the OS reports a last-level cache");
        assert!(
            (64 << 10..=1 << 30).contains(&cache.bytes),
            "implausible cache size {cache:?}"
        );
        assert!(cache.sharing >= 1);
    }
}
