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
