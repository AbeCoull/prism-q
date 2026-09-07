//! CUDA device wrapper. Isolates cudarc so alternative backends (wgpu, ROCm) can be substituted
//! by replacing this file.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::error::{PrismError, Result};

use super::kernels::{KERNEL_NAMES, kernel_source};

/// Handle to a CUDA-capable device.
///
/// Owns the CUDA context, default stream, and compiled PTX module. The `Stub` variant exists
/// so unit tests can exercise the `with_gpu()` builder path without CUDA available.
#[derive(Debug)]
pub struct GpuDevice {
    inner: DeviceInner,
}

#[derive(Debug)]
enum DeviceInner {
    Real {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        #[allow(dead_code)]
        module: Arc<CudaModule>,
        functions: HashMap<&'static str, CudaFunction>,
    },
    #[allow(dead_code)]
    Stub,
}

impl GpuDevice {
    /// Open the device with the given ordinal and load the kernel module.
    ///
    /// PTX is compiled targeting the newest supported arch at or below the device's
    /// compute capability so the running NVIDIA driver can load it regardless of the
    /// toolkit NVRTC version. A capability below 6.0 is rejected rather than targeted.
    /// The compiled PTX is shared by every device opened in the process and cached on
    /// disk in `prism-q-ptx` under the user cache dir (`XDG_CACHE_HOME`, `LOCALAPPDATA`,
    /// or `HOME/.cache`), so NVRTC runs once per source change per user and host.
    pub fn new(device_id: usize) -> Result<Self> {
        let context = CudaContext::new(device_id).map_err(|e| Self::driver_err("init", e))?;
        let stream = context.default_stream();
        let arch = detect_arch(&context)?;
        let module = load_kernel_module(&context, arch)?;
        // Pre-resolve every kernel once, to amortise driver lookups away from the gate
        // dispatch hot path.
        let mut functions = HashMap::with_capacity(KERNEL_NAMES.len());
        for &name in KERNEL_NAMES {
            let func = module
                .load_function(name)
                .map_err(|e| Self::driver_err(&format!("load_function `{name}`"), e))?;
            functions.insert(name, func);
        }
        Ok(Self {
            inner: DeviceInner::Real {
                context,
                stream,
                module,
                functions,
            },
        })
    }

    /// Query whether any CUDA-capable GPU is available on this system.
    ///
    /// Safe to call without a device; returns `false` if detection fails for any reason.
    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// Total VRAM on the selected device in bytes.
    pub fn vram_bytes(&self) -> Result<usize> {
        match &self.inner {
            DeviceInner::Real { context, .. } => context
                .total_mem()
                .map_err(|e| Self::driver_err("vram_bytes", e)),
            DeviceInner::Stub => Err(Self::stub_unsupported("vram_bytes")),
        }
    }

    /// Free VRAM currently available on the selected device in bytes.
    ///
    /// Reflects allocations by all processes sharing the device, including the
    /// current process's own outstanding `GpuBuffer`s. Useful for deciding
    /// whether a pending statevector allocation is likely to fit.
    pub fn vram_available(&self) -> Result<usize> {
        match &self.inner {
            DeviceInner::Real { context, .. } => context
                .mem_get_info()
                .map(|(free, _total)| free)
                .map_err(|e| Self::driver_err("vram_available", e)),
            DeviceInner::Stub => Err(Self::stub_unsupported("vram_available")),
        }
    }

    /// Maximum qubits representable as a Complex64 statevector in the currently
    /// free VRAM.
    ///
    /// Computed as `floor(log2(vram_available / 16))`. Each amplitude is two f64s =
    /// 16 bytes. Free memory moves with other processes sharing the device, so the
    /// value is advisory and can differ between calls.
    pub fn max_qubits_for_statevector(&self) -> Result<usize> {
        let bytes = self.vram_available()?;
        let elements = bytes / 16;
        if elements == 0 {
            return Ok(0);
        }
        Ok(63 - elements.leading_zeros() as usize)
    }

    #[cfg(test)]
    pub(crate) fn stub_for_tests() -> Self {
        Self {
            inner: DeviceInner::Stub,
        }
    }

    pub(crate) fn stream(&self) -> Result<&Arc<CudaStream>> {
        match &self.inner {
            DeviceInner::Real { stream, .. } => Ok(stream),
            DeviceInner::Stub => Err(Self::stub_unsupported("stream access")),
        }
    }

    pub(crate) fn function(&self, name: &str) -> Result<CudaFunction> {
        match &self.inner {
            DeviceInner::Real { functions, .. } => {
                functions
                    .get(name)
                    .cloned()
                    .ok_or_else(|| PrismError::BackendUnsupported {
                        backend: "gpu".to_string(),
                        operation: format!("unknown kernel `{name}` (not in KERNEL_NAMES)"),
                    })
            }
            DeviceInner::Stub => Err(Self::stub_unsupported(&format!("function `{name}`"))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_stub(&self) -> bool {
        matches!(self.inner, DeviceInner::Stub)
    }

    fn driver_err(op: &str, err: impl std::fmt::Display) -> PrismError {
        PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: format!("{op}: {err}"),
        }
    }

    fn stub_unsupported(op: &str) -> PrismError {
        PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: format!("{op} (stub device)"),
        }
    }
}

/// Compiled PTX per target arch, shared by every device opened in this process.
static PTX_BY_ARCH: LazyLock<Mutex<HashMap<&'static str, Arc<Ptx>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn load_kernel_module(context: &Arc<CudaContext>, arch: &'static str) -> Result<Arc<CudaModule>> {
    let mut cache = PTX_BY_ARCH.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(ptx) = cache.get(arch) {
        return context
            .load_module(Ptx::clone(ptx))
            .map_err(|e| GpuDevice::driver_err("load_module", e));
    }
    let (ptx, module) = load_or_compile(context, arch, &ptx_cache_dir())?;
    cache.insert(arch, ptx);
    Ok(module)
}

/// Load the module from the PTX cached in `cache_dir`, or compile through NVRTC when
/// the file is missing or the driver rejects its contents. The file name binds arch,
/// NVRTC options, crate version, and a hash of the source text, so a changed kernel
/// never resolves to stale PTX. A fresh compile is written back atomically; a write
/// failure is ignored, the module is already loaded.
fn load_or_compile(
    context: &Arc<CudaContext>,
    arch: &'static str,
    cache_dir: &Path,
) -> Result<(Arc<Ptx>, Arc<CudaModule>)> {
    let opts = CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let source = kernel_source();
    let path = cache_dir.join(ptx_cache_file_name(arch, &opts, &source));
    if let Ok(text) = std::fs::read_to_string(&path) {
        let ptx = Ptx::from_src(text);
        if let Ok(module) = context.load_module(ptx.clone()) {
            return Ok((Arc::new(ptx), module));
        }
    }
    let ptx = compile_ptx_with_opts(&source, opts).map_err(|e| PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("PTX compilation (arch={arch}): {e}"),
    })?;
    let module = context
        .load_module(ptx.clone())
        .map_err(|e| GpuDevice::driver_err("load_module", e))?;
    write_ptx_cache(&path, &ptx.to_src());
    Ok((Arc::new(ptx), module))
}

/// Per-user cache location: `XDG_CACHE_HOME`, `LOCALAPPDATA`, or `HOME/.cache`, falling
/// back to the OS temp dir. A shared temp dir is avoided where a user dir exists so
/// that no other local user can seed the file another process will load.
fn ptx_cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("prism-q-ptx")
}

fn ptx_cache_file_name(arch: &str, opts: &CompileOptions, source: &str) -> String {
    let mut hasher = DefaultHasher::new();
    opts.hash(&mut hasher);
    source.hash(&mut hasher);
    format!(
        "{}-{arch}-{:016x}.ptx",
        env!("CARGO_PKG_VERSION"),
        hasher.finish()
    )
}

fn write_ptx_cache(path: &Path, ptx_src: &str) {
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, ptx_src).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Virtual architectures NVRTC is asked to target, ascending by capability.
///
/// The ceiling tracks the newest arch the pinned `cudarc` NVRTC binding
/// (`cuda-12040` in `Cargo.toml`) accepts. Naming a newer one fails compilation
/// outright on a 12.x toolkit, so raise the pin before extending this table.
const KNOWN_ARCHS: &[((i32, i32), &str)] = &[
    ((6, 0), "compute_60"),
    ((6, 1), "compute_61"),
    ((6, 2), "compute_62"),
    ((7, 0), "compute_70"),
    ((7, 2), "compute_72"),
    ((7, 5), "compute_75"),
    ((8, 0), "compute_80"),
    ((8, 6), "compute_86"),
    ((8, 7), "compute_87"),
    ((8, 9), "compute_89"),
    ((9, 0), "compute_90"),
];

/// Newest entry of [`KNOWN_ARCHS`] at or below `capability`, or `None` below
/// the oldest entry (6.0, the floor for double-precision atomics).
///
/// Capabilities past the table clamp down rather than fail: the driver JITs PTX
/// forward, so an under-targeted module still runs, and the clamped string stays
/// clear of the pre-Turing range NVRTC 13.x refuses.
fn arch_for_capability(capability: (i32, i32)) -> Option<&'static str> {
    KNOWN_ARCHS
        .iter()
        .rev()
        .find(|&&(known, _)| known <= capability)
        .map(|&(_, arch)| arch)
}

fn detect_arch(context: &Arc<CudaContext>) -> Result<&'static str> {
    let capability = context
        .compute_capability()
        .map_err(|e| GpuDevice::driver_err("compute_capability", e))?;
    arch_for_capability(capability).ok_or_else(|| PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!(
            "compute capability {}.{} is below the 6.0 floor the kernels target",
            capability.0, capability.1
        ),
    })
}

#[cfg(test)]
mod tests {
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
    fn second_device_reuses_the_process_cached_ptx() {
        let Ok(context) = CudaContext::new(0) else {
            eprintln!("SKIP: no usable GPU");
            return;
        };
        let arch = detect_arch(&context).unwrap();
        let _first = GpuDevice::new(0).unwrap();
        let before = Arc::clone(&PTX_BY_ARCH.lock().unwrap()[arch]);
        let _second = GpuDevice::new(0).unwrap();
        let after = Arc::clone(&PTX_BY_ARCH.lock().unwrap()[arch]);
        assert!(Arc::ptr_eq(&before, &after));
    }

    // A corrupt cache file is overwritten by a recompile; the valid file is then
    // loaded without a rewrite (the miss path always writes, so an unchanged mtime
    // is the hit).
    #[test]
    fn corrupt_disk_cache_falls_back_to_a_recompile() {
        let Ok(context) = CudaContext::new(0) else {
            eprintln!("SKIP: no usable GPU");
            return;
        };
        let arch = detect_arch(&context).unwrap();
        let dir = std::env::temp_dir().join(format!("prism-q-ptx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let path = dir.join(ptx_cache_file_name(arch, &opts, &kernel_source()));
        std::fs::write(&path, "not ptx").unwrap();

        let (ptx, _module) = load_or_compile(&context, arch, &dir).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), ptx.to_src());
        let written = std::fs::metadata(&path).unwrap().modified().unwrap();

        let (_ptx, _module) = load_or_compile(&context, arch, &dir).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            written
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
