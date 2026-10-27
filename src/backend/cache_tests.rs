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
fn pool_tile_cuts_the_budget_until_the_pass_has_its_tile_count() {
    let budget = 1 << 14;
    let floor = 1 << 12;
    assert_eq!(tile_len_for_pool(1 << 20, budget, floor, 16), budget);
    assert_eq!(tile_len_for_pool(1 << 18, budget, floor, 16), budget);
    assert_eq!(tile_len_for_pool(1 << 17, budget, floor, 16), 1 << 13);
    assert_eq!(tile_len_for_pool(1 << 16, budget, floor, 16), 1 << 12);
    assert_eq!(tile_len_for_pool(1 << 15, budget, floor, 16), floor);
    assert_eq!(tile_len_for_pool(1 << 10, budget, floor, 16), floor);
    assert_eq!(tile_len_for_pool(1 << 15, budget, floor, 1), budget);
    assert!(min_parallel_tiles().is_power_of_two());
}

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

#[test]
fn tile_override_rounds_into_its_range_for_any_parsed_value() {
    assert_eq!(tile_override_bytes(128), 128 * KB);
    assert_eq!(tile_override_bytes(700), 512 * KB);
    assert_eq!(tile_override_bytes(4096), 2 * MB);
    assert_eq!(tile_override_bytes(1 << 54), 2 * MB);
    assert_eq!(tile_override_bytes((1 << 54) + 1), 2 * MB);
    assert_eq!(tile_override_bytes(usize::MAX), 2 * MB);
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
