//! Cache topology read from the OS once per process, and the tile budget derived
//! from it.

use std::sync::OnceLock;

/// One cache level: its size and the number of logical CPUs that share one instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SharedCache {
    pub(crate) bytes: usize,
    pub(crate) sharing: usize,
}

/// The caches CPU 0 sees: its L2, the largest data or unified level it shares, and the
/// logical CPUs on its core. On Apple silicon the cluster L2 is both `l2` and `llc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CacheTopology {
    pub(crate) l2: Option<SharedCache>,
    pub(crate) llc: SharedCache,
    pub(crate) smt: usize,
}

impl CacheTopology {
    /// L2 bytes per physical core: the private L2 on x86, a core's share of the
    /// cluster L2 on Apple silicon.
    pub(crate) fn l2_per_core(&self) -> Option<usize> {
        let l2 = self.l2?;
        let cores = (l2.sharing / self.smt.max(1)).max(1);
        Some(l2.bytes / cores)
    }
}

/// Read the cache topology once per process. `None` when the OS does not report a
/// cache, and callers keep their untuned behavior.
pub(crate) fn topology() -> Option<CacheTopology> {
    static CACHED: OnceLock<Option<CacheTopology>> = OnceLock::new();
    *CACHED.get_or_init(detect)
}

/// The last-level cache from [`topology`].
#[cfg(feature = "parallel")]
pub(crate) fn last_level_cache() -> Option<SharedCache> {
    topology().map(|topology| topology.llc)
}

/// Whether `workers` tiles of `tile_bytes` each fit one instance of `cache` together,
/// within half its capacity: an inclusive cache also holds every private cache below
/// it, and tiles that fill it exactly still evict each other.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn tiles_fit(cache: SharedCache, tile_bytes: usize, workers: usize) -> bool {
    let concurrent = workers.min(cache.sharing).max(1);
    tile_bytes.saturating_mul(concurrent) <= cache.bytes / 2
}

/// The tile budget on an unreported cache, and the floor of the derived one: 256 KB,
/// the private L2 of the x86 cores every tile was measured on.
pub(crate) const MIN_TILE_BYTES: usize = 256 << 10;
const MAX_TILE_BYTES: usize = 1 << 20;
const MIN_TILE_OVERRIDE_KB: usize = 128;
const MAX_TILE_OVERRIDE_BYTES: usize = 2 << 20;

/// Bytes a cache-resident tile may take, a power of two from 256 KB to 1 MB.
///
/// The smaller of the L2 per core and a quarter of the last-level cache per logical
/// CPU sharing it, so every concurrent tile and the source lines it gathers fit half
/// that cache: on an i7-6700K eight 1 MB tiles filled the 8 MB L3 and ran 2x slower
/// than 256 KB ones, while 512 KB read flat. [`MIN_TILE_BYTES`] when the OS reports
/// no cache. `PRISM_TILE_KB` overrides it, rounded down to a power of two between
/// 128 KB and 2 MB, for sweeps past the derived range.
pub(crate) fn tile_budget_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Some(kb) = crate::env_knobs::usize_override("PRISM_TILE_KB", MIN_TILE_OVERRIDE_KB) {
            return floor_pow2((kb << 10).min(MAX_TILE_OVERRIDE_BYTES));
        }
        topology().map_or(MIN_TILE_BYTES, tile_budget_for)
    })
}

/// The derived tile budget for `topology`; see [`tile_budget_bytes`].
pub(crate) fn tile_budget_for(topology: CacheTopology) -> usize {
    let llc_share = topology.llc.bytes / (4 * topology.llc.sharing.max(1));
    let budget = topology
        .l2_per_core()
        .unwrap_or(usize::MAX)
        .min(llc_share.max(MIN_TILE_BYTES));
    floor_pow2(budget.clamp(MIN_TILE_BYTES, MAX_TILE_BYTES))
}

fn floor_pow2(n: usize) -> usize {
    1 << n.ilog2()
}

/// Assemble the topology from the caches CPU 0 sits in, in `(level, bytes, sharing)`
/// form, and its logical CPUs per core.
fn assemble(
    caches: impl IntoIterator<Item = (u32, usize, usize)>,
    smt: usize,
) -> Option<CacheTopology> {
    let mut l2 = None;
    let mut llc: Option<(u32, SharedCache)> = None;
    for (level, bytes, sharing) in caches {
        let cache = SharedCache { bytes, sharing };
        if level == 2 && l2.is_none_or(|known: SharedCache| bytes > known.bytes) {
            l2 = Some(cache);
        }
        if llc.is_none_or(|(l, known)| (level, bytes) > (l, known.bytes)) {
            llc = Some((level, cache));
        }
    }
    Some(CacheTopology {
        l2,
        llc: llc?.1,
        smt: smt.max(1),
    })
}

#[cfg(all(windows, not(miri)))]
fn detect() -> Option<CacheTopology> {
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

    const RELATION_PROCESSOR_CORE: u32 = 0;
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

    let on_cpu0 = |info: &&LogicalProcessorInformation| info.processor_mask & 1 == 1;
    let smt = infos
        .iter()
        .filter(on_cpu0)
        .find(|info| info.relationship == RELATION_PROCESSOR_CORE)
        .map_or(1, |info| info.processor_mask.count_ones() as usize);
    let caches = infos
        .iter()
        .filter(on_cpu0)
        .filter(|info| info.relationship == RELATION_CACHE)
        .filter_map(|info| {
            let (level, size, cache_type) = info.cache();
            let kept = (cache_type == CACHE_UNIFIED || cache_type == CACHE_DATA) && size > 0;
            kept.then_some((
                u32::from(level),
                size as usize,
                info.processor_mask.count_ones() as usize,
            ))
        });
    assemble(caches, smt)
}

#[cfg(all(target_os = "macos", not(miri)))]
fn detect() -> Option<CacheTopology> {
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

    let read = |name: &std::ffi::CStr| -> Option<usize> {
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
            (0, 4) => Some(u32::from_le_bytes(value[..4].try_into().unwrap()) as usize),
            (0, 8) => Some(u64::from_le_bytes(value) as usize),
            _ => None,
        }
        .filter(|&v| v > 0)
    };

    let logical = read(c"hw.logicalcpu").unwrap_or(1);
    let smt = (logical / read(c"hw.physicalcpu").unwrap_or(logical)).max(1);
    let l2_bytes = read(c"hw.perflevel0.l2cachesize").or_else(|| read(c"hw.l2cachesize"));
    let l2 = l2_bytes.map(|bytes| SharedCache {
        bytes,
        sharing: read(c"hw.perflevel0.cpusperl2").unwrap_or(1) * smt,
    });
    let l3 = read(c"hw.l3cachesize").map(|bytes| SharedCache {
        bytes,
        sharing: logical,
    });
    Some(CacheTopology {
        l2,
        llc: l3.or(l2)?,
        smt,
    })
}

#[cfg(all(unix, not(target_os = "macos"), not(miri)))]
fn detect() -> Option<CacheTopology> {
    let cpu0 = "/sys/devices/system/cpu/cpu0";
    let mut caches = Vec::new();
    for index in 0.. {
        let dir = format!("{cpu0}/cache/index{index}");
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
        caches.push((level, bytes, sharing));
    }
    let smt = std::fs::read_to_string(format!("{cpu0}/topology/thread_siblings_list"))
        .ok()
        .and_then(|list| count_cpu_list(&list))
        .unwrap_or(1);
    assemble(caches, smt)
}

#[cfg(any(miri, not(any(windows, unix))))]
fn detect() -> Option<CacheTopology> {
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

    const KB: usize = 1 << 10;
    const MB: usize = 1 << 20;

    fn cache(bytes: usize, sharing: usize) -> SharedCache {
        SharedCache { bytes, sharing }
    }

    fn skylake_4c() -> CacheTopology {
        CacheTopology {
            l2: Some(cache(256 * KB, 2)),
            llc: cache(8 * MB, 8),
            smt: 2,
        }
    }

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
        let skylake = cache(8 * MB, 8);
        assert!(!tiles_fit(skylake, 2 * MB, 8));
        assert!(!tiles_fit(skylake, 2 * MB, 4));
        assert!(tiles_fit(skylake, 2 * MB, 2));
        assert!(tiles_fit(skylake, 2 * MB, 1));

        let stacked = cache(96 * MB, 16);
        assert!(tiles_fit(stacked, 2 * MB, 32));
        assert!(!tiles_fit(stacked, 4 * MB, 32));
        assert!(tiles_fit(stacked, 2 * MB, 0));
    }

    #[test]
    fn assemble_takes_the_l2_and_the_largest_level_cpu0_sits_in() {
        let topology = assemble([(1, 32 * KB, 2), (2, 256 * KB, 2), (3, 8 * MB, 8)], 2).unwrap();
        assert_eq!(topology, skylake_4c());
        assert_eq!(
            assemble([(2, 16 * MB, 4)], 1).unwrap().llc,
            cache(16 * MB, 4)
        );
        assert!(assemble([], 1).is_none());
    }

    #[test]
    fn l2_per_core_discounts_smt_siblings_and_cluster_sharing() {
        assert_eq!(skylake_4c().l2_per_core(), Some(256 * KB));
        let apple = CacheTopology {
            l2: Some(cache(16 * MB, 4)),
            llc: cache(16 * MB, 4),
            smt: 1,
        };
        assert_eq!(apple.l2_per_core(), Some(4 * MB));
        let no_l2 = CacheTopology {
            l2: None,
            llc: cache(8 * MB, 8),
            smt: 1,
        };
        assert_eq!(no_l2.l2_per_core(), None);
    }

    // Every value the kernels were tuned with comes back on the host they were tuned
    // on, and the rule shrinks or grows elsewhere only within the derived range.
    #[test]
    fn tile_budget_reproduces_the_measured_host_and_bounds_the_rest() {
        assert_eq!(tile_budget_for(skylake_4c()), 256 * KB);

        let zen4_ccd = CacheTopology {
            l2: Some(cache(MB, 2)),
            llc: cache(32 * MB, 16),
            smt: 2,
        };
        assert_eq!(tile_budget_for(zen4_ccd), 512 * KB);

        let raptor_lake_p = CacheTopology {
            l2: Some(cache(2 * MB, 2)),
            llc: cache(36 * MB, 32),
            smt: 2,
        };
        assert_eq!(tile_budget_for(raptor_lake_p), 256 * KB);

        let apple_m2 = CacheTopology {
            l2: Some(cache(16 * MB, 4)),
            llc: cache(16 * MB, 4),
            smt: 1,
        };
        assert_eq!(tile_budget_for(apple_m2), MB);

        let graviton3 = CacheTopology {
            l2: Some(cache(MB, 1)),
            llc: cache(32 * MB, 64),
            smt: 1,
        };
        assert_eq!(tile_budget_for(graviton3), 256 * KB);

        let huge_everything = CacheTopology {
            l2: Some(cache(8 * MB, 1)),
            llc: cache(256 * MB, 1),
            smt: 1,
        };
        assert_eq!(
            tile_budget_for(huge_everything),
            MB,
            "capped at the derived maximum"
        );

        let tiny_l2 = CacheTopology {
            l2: Some(cache(128 * KB, 1)),
            llc: cache(4 * MB, 4),
            smt: 1,
        };
        assert_eq!(
            tile_budget_for(tiny_l2),
            256 * KB,
            "never below the measured floor"
        );

        let no_l2 = CacheTopology {
            l2: None,
            llc: cache(32 * MB, 16),
            smt: 1,
        };
        assert_eq!(tile_budget_for(no_l2), 512 * KB);
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
    fn topology_is_reported_on_supported_hosts() {
        let topology = topology().expect("the OS reports a cache topology");
        let plausible = 64 * KB..=1 << 30;
        assert!(
            plausible.contains(&topology.llc.bytes),
            "implausible cache size {topology:?}"
        );
        assert!(topology.llc.sharing >= 1);
        assert!(
            (1..=4).contains(&topology.smt),
            "implausible SMT width {topology:?}"
        );
        if let Some(l2) = topology.l2 {
            assert!(plausible.contains(&l2.bytes), "implausible L2 {topology:?}");
            assert!(
                l2.sharing >= topology.smt,
                "an L2 holds at least one core {topology:?}"
            );
        }
        let budget = tile_budget_bytes();
        assert!(budget.is_power_of_two());
        assert!((128 * KB..=2 * MB).contains(&budget));
    }
}
