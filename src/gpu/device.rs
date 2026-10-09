//! CUDA device handle, kernel module loading, and the on-disk kernel image cache.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::{
    CompileOptions, Ptx, compile_ptx_with_opts, result as nvrtc, sys as nvrtc_sys,
};

use crate::error::{PrismError, Result};

use super::driver_err;
use super::kernels::{KERNEL_NAMES, kernel_source};

/// Handle to a CUDA-capable device.
///
/// Owns the CUDA context, default stream, and compiled kernel module. A stub variant lets unit
/// tests exercise the `with_gpu()` builder path without CUDA.
#[derive(Debug)]
pub struct GpuDevice {
    inner: DeviceInner,
}

#[derive(Debug)]
enum DeviceInner {
    Real {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        _module: Arc<CudaModule>,
        functions: HashMap<&'static str, CudaFunction>,
    },
    #[cfg_attr(not(test), allow(dead_code))]
    Stub,
}

impl GpuDevice {
    /// Open the device with the given ordinal and load the kernel module.
    ///
    /// NVRTC compiles the kernels to SASS for the device's exact architecture, which any
    /// CUDA 12 driver loads whatever the NVRTC minor version. When the NVRTC predates the
    /// device, it emits PTX for the newest known arch below it instead, which the driver
    /// JITs. A capability below 6.0 is rejected rather than targeted. The image is shared
    /// by every device opened in the process and cached on disk in `prism-q-ptx` under the
    /// user cache dir (`XDG_CACHE_HOME`, `LOCALAPPDATA`, or `HOME/.cache`), so NVRTC runs
    /// once per source change, NVRTC version, user, and host.
    pub fn new(device_id: usize) -> Result<Self> {
        let version = driver_version()?;
        if version < MIN_DRIVER_VERSION {
            return Err(gpu_unusable(format!(
                "the NVIDIA driver supports CUDA {}, and CUDA 12.0 or newer is required",
                cuda_version(version)
            )));
        }
        let context = CudaContext::new(device_id).map_err(|e| driver_err("init", e))?;
        let stream = context.default_stream();
        let capability = detect_capability(&context)?;
        let module = load_kernel_module(&context, capability)?;
        // Pre-resolve every kernel once, to amortise driver lookups away from the gate
        // dispatch hot path.
        let mut functions = HashMap::with_capacity(KERNEL_NAMES.len());
        for &name in KERNEL_NAMES {
            let func = module
                .load_function(name)
                .map_err(|e| driver_err(&format!("load_function `{name}`"), e))?;
            functions.insert(name, func);
        }
        Ok(Self {
            inner: DeviceInner::Real {
                context,
                stream,
                _module: module,
                functions,
            },
        })
    }

    /// Whether device 0 opens; any detection failure reads as `false`.
    pub fn is_available() -> bool {
        driver_version().is_ok_and(|v| v >= MIN_DRIVER_VERSION) && CudaContext::new(0).is_ok()
    }

    /// Product name of the selected device, as the driver reports it.
    pub fn name(&self) -> Result<String> {
        match &self.inner {
            DeviceInner::Real { context, .. } => context.name().map_err(|e| driver_err("name", e)),
            DeviceInner::Stub => Err(Self::stub_unsupported("name")),
        }
    }

    /// Total VRAM on the selected device in bytes.
    pub fn vram_bytes(&self) -> Result<usize> {
        match &self.inner {
            DeviceInner::Real { context, .. } => {
                context.total_mem().map_err(|e| driver_err("vram_bytes", e))
            }
            DeviceInner::Stub => Err(Self::stub_unsupported("vram_bytes")),
        }
    }

    /// Free VRAM currently available on the selected device in bytes.
    ///
    /// Reflects allocations by all processes sharing the device, including the
    /// current process's own outstanding `GpuBuffer`s.
    pub fn vram_available(&self) -> Result<usize> {
        match &self.inner {
            DeviceInner::Real { context, .. } => context
                .mem_get_info()
                .map(|(free, _total)| free)
                .map_err(|e| driver_err("vram_available", e)),
            DeviceInner::Stub => Err(Self::stub_unsupported("vram_available")),
        }
    }

    /// Maximum qubits representable as a Complex64 statevector in the currently
    /// free VRAM.
    ///
    /// Computed as `floor(log2(vram_available / 16))`. Free memory moves with other
    /// processes sharing the device, so the value is advisory and can differ between calls.
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

    #[cfg(test)]
    pub(crate) fn is_stub(&self) -> bool {
        matches!(self.inner, DeviceInner::Stub)
    }

    fn stub_unsupported(op: &str) -> PrismError {
        PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: format!("{op} (stub device)"),
        }
    }
}

/// Compiled kernels for one device architecture.
#[derive(Debug)]
enum KernelImage {
    /// SASS for the exact device architecture.
    Cubin(Vec<u8>),
    /// PTX for the driver to JIT, used when NVRTC cannot target the device.
    Ptx(String),
}

impl KernelImage {
    fn load(
        &self,
        context: &Arc<CudaContext>,
    ) -> std::result::Result<Arc<CudaModule>, DriverError> {
        context.load_module(match self {
            Self::Cubin(bytes) => Ptx::from_binary(bytes.clone()),
            Self::Ptx(src) => Ptx::from_src(src.clone()),
        })
    }

    fn extension(&self) -> &'static str {
        match self {
            Self::Cubin(_) => "cubin",
            Self::Ptx(_) => "ptx",
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Cubin(bytes) => bytes,
            Self::Ptx(src) => src.as_bytes(),
        }
    }

    fn read(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        match path.extension()?.to_str()? {
            "cubin" => Some(Self::Cubin(bytes)),
            "ptx" => String::from_utf8(bytes).ok().map(Self::Ptx),
            _ => None,
        }
    }
}

/// Compiled kernels per device compute capability, shared by every device opened in
/// this process.
static IMAGE_BY_CAPABILITY: LazyLock<Mutex<ImageCache>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

type ImageCache = HashMap<(i32, i32), Arc<KernelImage>>;

fn load_kernel_module(
    context: &Arc<CudaContext>,
    capability: (i32, i32),
) -> Result<Arc<CudaModule>> {
    let mut cache = IMAGE_BY_CAPABILITY
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(image) = cache.get(&capability) {
        return image
            .load(context)
            .map_err(|e| driver_err("load_module", e));
    }
    let (image, module) = load_or_compile(context, capability, &ptx_cache_dir())?;
    cache.insert(capability, image);
    Ok(module)
}

/// Load the module from an image cached in `cache_dir`, or compile through NVRTC when
/// none loads. File names bind crate version, device architecture, a hash of the source
/// text, and the NVRTC version that compiled them. A cached cubin matches whichever
/// CUDA 12 NVRTC made it, so it is tried first without opening NVRTC. Cached PTX is
/// tried next, from this NVRTC only when one loads, since a driver older than the NVRTC
/// refuses it. A fresh compile is written back atomically; a write failure is ignored,
/// the module is already loaded.
fn load_or_compile(
    context: &Arc<CudaContext>,
    capability: (i32, i32),
    cache_dir: &Path,
) -> Result<(Arc<KernelImage>, Arc<CudaModule>)> {
    let source = kernel_source();
    let stem = cache_stem(capability, &source);
    let cached = |extension: &str| -> Vec<PathBuf> {
        let prefix = format!("{stem}-nvrtc");
        let suffix = format!(".{extension}");
        std::fs::read_dir(cache_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(&suffix))
            })
            .collect()
    };
    let try_load = |paths: &[PathBuf]| {
        paths
            .iter()
            .filter_map(|path| KernelImage::read(path))
            .find_map(|image| Some((image.load(context).ok()?, image)))
            .map(|(module, image)| (Arc::new(image), module))
    };
    if let Some(hit) = try_load(&cached("cubin")) {
        return Ok(hit);
    }
    if !nvrtc_present() {
        return try_load(&cached("ptx")).ok_or_else(|| {
            gpu_unusable(
                "NVRTC, the CUDA runtime compiler, did not load (looked for libnvrtc.so.12 or \
                 nvrtc64_120_0.dll); install a CUDA 12 toolkit or put its NVRTC library on the \
                 loader path"
                    .to_string(),
            )
        });
    }
    let version =
        nvrtc_version().ok_or_else(|| gpu_unusable("NVRTC did not report its version".into()))?;
    if let Some(hit) = try_load(&[cache_dir.join(cache_file_name(&stem, version, "ptx"))]) {
        return Ok(hit);
    }
    let image = compile_kernels(&source, capability)?;
    let module = image.load(context).map_err(fresh_module_err)?;
    write_image_cache(
        &cache_dir.join(cache_file_name(&stem, version, image.extension())),
        image.bytes(),
    );
    Ok((Arc::new(image), module))
}

/// SASS for `sm_XY` when this NVRTC can emit it, otherwise PTX for the newest
/// [`KNOWN_ARCHS`] entry at or below the device.
fn compile_kernels(source: &str, capability: (i32, i32)) -> Result<KernelImage> {
    if nvrtc_sass_archs()?.contains(&(capability.0 * 10 + capability.1)) {
        let arch = format!("sm_{}{}", capability.0, capability.1);
        return compile_cubin(source, &arch).map(KernelImage::Cubin);
    }
    let arch = arch_for_capability(capability).ok_or_else(|| below_floor(capability))?;
    let opts = CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(source, opts)
        .map_err(|e| driver_err(&format!("PTX compilation (arch={arch})"), e))?;
    Ok(KernelImage::Ptx(ptx.to_src()))
}

/// NVRTC program handle, destroyed on drop.
struct NvrtcProgram(nvrtc_sys::nvrtcProgram);

impl Drop for NvrtcProgram {
    fn drop(&mut self) {
        // SAFETY: the handle came from create_program and is destroyed only here.
        let _ = unsafe { nvrtc::destroy_program(self.0) };
    }
}

/// Compile `source` to a cubin for `arch`. cudarc's safe wrapper returns PTX only, so
/// this drives the NVRTC calls directly.
fn compile_cubin(source: &str, arch: &str) -> Result<Vec<u8>> {
    let src = CString::new(source).expect("kernel source contains no NUL byte");
    let program = NvrtcProgram(
        nvrtc::create_program(&src, None).map_err(|e| driver_err("NVRTC create_program", e))?,
    );
    let options = [format!("--gpu-architecture={arch}")];
    // SAFETY: `program` is live, and `src` outlives it.
    if let Err(e) = unsafe { nvrtc::compile_program(program.0, &options) } {
        // SAFETY: `program` is live.
        let log = unsafe { nvrtc::get_program_log(program.0) }
            .map(|log| {
                // SAFETY: NVRTC NUL-terminates the log it writes.
                unsafe { CStr::from_ptr(log.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        return Err(driver_err(
            &format!("SASS compilation (arch={arch})"),
            format!("{e}: {}", log.trim()),
        ));
    }
    let mut size = 0;
    // SAFETY: `program` compiled; nvrtcGetCUBINSize only writes its out-parameter.
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program.0, &mut size) }
        .result()
        .map_err(|e| driver_err("NVRTC cubin size", e))?;
    let mut cubin = vec![0u8; size];
    // SAFETY: `cubin` holds the `size` bytes nvrtcGetCUBINSize reported.
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program.0, cubin.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| driver_err("NVRTC cubin", e))?;
    Ok(cubin)
}

/// Real architectures this NVRTC emits SASS for, as `10 * major + minor`.
fn nvrtc_sass_archs() -> Result<Vec<i32>> {
    let mut count = 0;
    // SAFETY: NVRTC loads (checked by the caller); the call only writes its out-parameter.
    unsafe { nvrtc_sys::nvrtcGetNumSupportedArchs(&mut count) }
        .result()
        .map_err(|e| driver_err("NVRTC supported archs", e))?;
    let mut archs = vec![0; usize::try_from(count).unwrap_or(0)];
    // SAFETY: `archs` holds the `count` entries nvrtcGetNumSupportedArchs reported.
    unsafe { nvrtc_sys::nvrtcGetSupportedArchs(archs.as_mut_ptr()) }
        .result()
        .map_err(|e| driver_err("NVRTC supported archs", e))?;
    Ok(archs)
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

fn cache_stem(capability: (i32, i32), source: &str) -> String {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    format!(
        "{}-sm_{}{}-{:016x}",
        env!("CARGO_PKG_VERSION"),
        capability.0,
        capability.1,
        hasher.finish()
    )
}

fn cache_file_name(stem: &str, nvrtc_version: (i32, i32), extension: &str) -> String {
    format!(
        "{stem}-nvrtc{}.{}.{extension}",
        nvrtc_version.0, nvrtc_version.1
    )
}

fn write_image_cache(path: &Path, bytes: &[u8]) {
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Oldest driver, as `cuDriverGetVersion` encodes it, that the pinned `cudarc` bindings
/// and NVRTC 12 output run on.
const MIN_DRIVER_VERSION: i32 = 12_000;

/// CUDA version the installed driver supports, or an error naming the missing library.
///
/// `cudarc` loads the driver on first use and panics when it is absent, so every path
/// that opens a device checks here first.
fn driver_version() -> Result<i32> {
    // SAFETY: probing loads the driver library by name, which runs only its own
    // initialisers and asks nothing of this process.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err(gpu_unusable(
            "the NVIDIA driver library (libcuda.so.1 or nvcuda.dll) did not load; install an \
             NVIDIA driver"
                .to_string(),
        ));
    }
    let mut version = 0;
    // SAFETY: the driver library loads (checked above), and cuDriverGetVersion only writes
    // its out-parameter; it is valid before cuInit.
    unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) }
        .result()
        .map_err(|e| driver_err("driver version", e))?;
    Ok(version)
}

/// Whether the NVRTC library loads under any name `cudarc` searches for.
pub(crate) fn nvrtc_present() -> bool {
    // SAFETY: probing loads the NVRTC library by name, which runs only its own
    // initialisers and asks nothing of this process.
    unsafe { cudarc::nvrtc::sys::is_culib_present() }
}

fn nvrtc_version() -> Option<(i32, i32)> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: NVRTC loads (checked by the caller); nvrtcVersion only writes its
    // out-parameters.
    let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    (status == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS).then_some((major, minor))
}

fn cuda_version(encoded: i32) -> String {
    format!("{}.{}", encoded / 1000, encoded % 1000 / 10)
}

/// Map a failure to load a freshly compiled image, naming the version mismatch when the
/// driver predates the NVRTC that emitted PTX.
fn fresh_module_err(err: DriverError) -> PrismError {
    if err.0 != cudarc::driver::sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION {
        return driver_err("load_module", err);
    }
    let driver = driver_version().map_or_else(|_| "unknown".to_string(), cuda_version);
    let nvrtc = nvrtc_version().map_or_else(|| "unknown".to_string(), |(a, b)| format!("{a}.{b}"));
    gpu_unusable(format!(
        "the NVIDIA driver (CUDA {driver}) is older than the NVRTC that compiled the kernels \
         (CUDA {nvrtc}); update the driver, or use an NVRTC no newer than the driver"
    ))
}

fn gpu_unusable(reason: String) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: "gpu".to_string(),
        reason,
    }
}

/// Virtual architectures the PTX fallback targets, ascending by capability.
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

fn detect_capability(context: &Arc<CudaContext>) -> Result<(i32, i32)> {
    let capability = context
        .compute_capability()
        .map_err(|e| driver_err("compute_capability", e))?;
    match arch_for_capability(capability) {
        Some(_) => Ok(capability),
        None => Err(below_floor(capability)),
    }
}

fn below_floor(capability: (i32, i32)) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!(
            "compute capability {}.{} is below the 6.0 floor the kernels target",
            capability.0, capability.1
        ),
    }
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
}
