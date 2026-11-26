use super::*;

#[test]
fn stub_reports_as_stub() {
    let dev = GpuDevice::stub_for_tests();
    assert!(dev.is_stub());
}

#[test]
fn stub_stream_returns_unsupported() {
    let dev = GpuDevice::stub_for_tests();
    assert!(matches!(
        dev.stream().unwrap_err(),
        PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn arch_maps_exact_capabilities() {
    assert_eq!(arch_for_capability((6, 1)), Some("compute_61"));
    assert_eq!(arch_for_capability((8, 9)), Some("compute_89"));
    assert_eq!(arch_for_capability((9, 0)), Some("compute_90"));
}

#[test]
fn arch_clamps_down_to_the_newest_entry_at_or_below() {
    // Blackwell (10.0 / 12.0) and any gap inside the table take the newest
    // entry below them, never the oldest.
    assert_eq!(arch_for_capability((10, 0)), Some("compute_90"));
    assert_eq!(arch_for_capability((12, 0)), Some("compute_90"));
    assert_eq!(arch_for_capability((8, 8)), Some("compute_87"));
    assert_eq!(arch_for_capability((7, 1)), Some("compute_70"));
}

#[test]
fn arch_rejects_capabilities_below_the_floor() {
    assert_eq!(arch_for_capability((5, 2)), None);
    assert_eq!(arch_for_capability((3, 5)), None);
}

// Skips without a usable GPU, matching the golden suites.
#[test]
fn second_device_reuses_the_process_cached_image() {
    if !GpuDevice::is_available() {
        eprintln!("SKIP: no usable GPU");
        return;
    }
    let context = CudaContext::new(0).unwrap();
    let capability = detect_capability(&context).unwrap();
    let _first = GpuDevice::new(0).unwrap();
    let before = Arc::clone(&IMAGE_BY_CAPABILITY.lock().unwrap()[&capability]);
    let _second = GpuDevice::new(0).unwrap();
    let after = Arc::clone(&IMAGE_BY_CAPABILITY.lock().unwrap()[&capability]);
    assert!(Arc::ptr_eq(&before, &after));
}

// Skips without a usable GPU or without NVRTC.
#[test]
fn device_architecture_known_to_nvrtc_compiles_to_sass() {
    if !GpuDevice::is_available() || !nvrtc_present() {
        eprintln!("SKIP: no usable GPU or NVRTC");
        return;
    }
    let context = CudaContext::new(0).unwrap();
    let capability = detect_capability(&context).unwrap();
    if !nvrtc_sass_archs()
        .unwrap()
        .contains(&(capability.0 * 10 + capability.1))
    {
        eprintln!("SKIP: NVRTC cannot target this device");
        return;
    }
    let image = compile_kernels(&kernel_source(), capability).unwrap();
    assert!(matches!(image, KernelImage::Cubin(_)));
    image.load(&context).unwrap();
}

// A cached cubin matches whatever NVRTC version its name records, so a cubin
// labelled with a foreign version loads and nothing is recompiled or written.
#[test]
fn cached_cubin_from_another_nvrtc_version_loads_without_a_recompile() {
    if !GpuDevice::is_available() || !nvrtc_present() {
        eprintln!("SKIP: no usable GPU or NVRTC");
        return;
    }
    let context = CudaContext::new(0).unwrap();
    let capability = detect_capability(&context).unwrap();
    let image = compile_kernels(&kernel_source(), capability).unwrap();
    if !matches!(image, KernelImage::Cubin(_)) {
        eprintln!("SKIP: NVRTC cannot target this device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("prism-q-cubin-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = cache_stem(capability, &kernel_source());
    let path = dir.join(cache_file_name(&stem, (12, 0), "cubin"));
    std::fs::write(&path, image.bytes()).unwrap();

    let (loaded, _module) = load_or_compile(&context, capability, &dir).unwrap();
    assert_eq!(loaded.bytes(), image.bytes());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

// A corrupt cache file is overwritten by a recompile; the valid file is then
// loaded without a rewrite (the miss path always writes, so an unchanged mtime
// is the hit).
#[test]
fn corrupt_disk_cache_falls_back_to_a_recompile() {
    if !GpuDevice::is_available() || !nvrtc_present() {
        eprintln!("SKIP: no usable GPU or NVRTC");
        return;
    }
    let context = CudaContext::new(0).unwrap();
    let capability = detect_capability(&context).unwrap();
    let dir = std::env::temp_dir().join(format!("prism-q-ptx-test-{}", std::process::id()));
    let (image, _module) = load_or_compile(&context, capability, &dir).unwrap();
    let path = dir.join(cache_file_name(
        &cache_stem(capability, &kernel_source()),
        nvrtc_version().unwrap(),
        image.extension(),
    ));
    std::fs::write(&path, "not a kernel image").unwrap();

    let (image, _module) = load_or_compile(&context, capability, &dir).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), image.bytes());
    let written = std::fs::metadata(&path).unwrap().modified().unwrap();

    let (_image, _module) = load_or_compile(&context, capability, &dir).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().modified().unwrap(),
        written
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "parallel")]
#[test]
fn staging_cell_transitions_empty_idle_checked_out_and_restores_on_drop() {
    let cell = StagingCell::<Vec<u8>>::new();
    let mut inits = 0usize;
    {
        let mut guard = cell
            .try_checkout_with(|| {
                inits += 1;
                Ok(vec![1, 2, 3])
            })
            .unwrap()
            .expect("initial checkout from Empty should succeed");
        assert_eq!(*guard, vec![1, 2, 3]);
        assert!(
            matches!(*cell.0.lock().unwrap(), StagingSlotState::CheckedOut),
            "cell must remain CheckedOut while guard is alive"
        );
        guard.push(4);
    }
    assert!(matches!(*cell.0.lock().unwrap(), StagingSlotState::Idle(_)));

    let second = cell
        .try_checkout_with(|| {
            inits += 1;
            Ok(vec![99])
        })
        .unwrap()
        .expect("checkout from Idle should reuse existing value");
    assert_eq!(*second, vec![1, 2, 3, 4]);
    assert_eq!(inits, 1);
}

#[cfg(feature = "parallel")]
#[test]
fn staging_cell_failed_init_restores_empty_and_retries() {
    let cell = StagingCell::<u64>::new();
    let err = cell
        .try_checkout_with(|| {
            Err(PrismError::BackendUnsupported {
                backend: "gpu".to_string(),
                operation: "alloc staging".to_string(),
            })
        })
        .unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    assert!(matches!(*cell.0.lock().unwrap(), StagingSlotState::Empty));

    let guard = cell
        .try_checkout_with(|| Ok(42))
        .unwrap()
        .expect("retry after failed init must succeed");
    assert_eq!(*guard, 42);
}

#[cfg(feature = "parallel")]
#[test]
fn staging_cell_panicking_init_restores_empty() {
    let cell = StagingCell::<u64>::new();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = cell.try_checkout_with(|| -> Result<u64> {
            panic!("simulated staging allocation panic");
        });
    }));
    assert!(panicked.is_err());
    assert!(matches!(*cell.0.lock().unwrap(), StagingSlotState::Empty));

    let guard = cell
        .try_checkout_with(|| Ok(7))
        .unwrap()
        .expect("checkout after panicking init must recover");
    assert_eq!(*guard, 7);
}

#[cfg(feature = "parallel")]
#[test]
fn staging_cell_panicking_checkout_restores_idle() {
    let cell = StagingCell::<Vec<u8>>::new();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = cell
            .try_checkout_with(|| Ok(vec![10]))
            .unwrap()
            .expect("initial checkout");
        guard.push(20);
        panic!("simulated readback panic");
    }));
    assert!(panicked.is_err());
    assert!(matches!(*cell.0.lock().unwrap(), StagingSlotState::Idle(_)));

    let guard = cell
        .try_checkout_with(|| unreachable!("Idle value must be reused after panic"))
        .unwrap()
        .expect("checkout after panicking holder must recover Idle value");
    assert_eq!(*guard, vec![10, 20]);
}

#[cfg(feature = "parallel")]
#[test]
fn staging_cell_contention_returns_none_and_rayon_nested_checkout_does_not_deadlock() {
    let cell = StagingCell::<u64>::new();

    // Contention during initial allocation (no mutex held across init or rayon::join).
    let guard = cell
        .try_checkout_with(|| {
            let (nested, _) = rayon::join(
                || cell.try_checkout_with(|| Ok(999)),
                || (0..64).sum::<u64>(),
            );
            assert!(
                nested.unwrap().is_none(),
                "nested checkout during init must return Ok(None) without deadlocking"
            );
            Ok(55)
        })
        .unwrap()
        .expect("outer init checkout should succeed");
    assert_eq!(*guard, 55);

    // Contention while CheckedOut inside nested Rayon tasks.
    let (nested_a, nested_b) = rayon::join(
        || cell.try_checkout_with(|| Ok(1)),
        || cell.try_checkout_with(|| Ok(2)),
    );
    assert!(nested_a.unwrap().is_none());
    assert!(nested_b.unwrap().is_none());
    drop(guard);

    let reused = cell
        .try_checkout_with(|| unreachable!("Idle value should be returned"))
        .unwrap()
        .expect("after outer drop, cell must be Idle");
    assert_eq!(*reused, 55);
}
