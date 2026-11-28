//! CUDA device handle, kernel module loading, and the on-disk kernel image cache.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
#[cfg(feature = "bench-internal")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;

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
    #[cfg(feature = "bench-internal")]
    census: PassCounters,
}

/// Running totals behind [`crate::gpu::GpuPassCensus`].
#[cfg(feature = "bench-internal")]
#[derive(Debug, Default)]
struct PassCounters {
    launches: AtomicU64,
    passes: AtomicU64,
    drains: AtomicU64,
    uploads: AtomicU64,
    dense_2q: AtomicU64,
    tiles: AtomicU64,
}

#[derive(Debug)]
enum DeviceInner {
    Real {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        _module: Arc<CudaModule>,
        functions: HashMap<&'static str, CudaFunction>,
        #[cfg(feature = "parallel")]
        staging: StagingCell<Box<super::memory::HostStaging>>,
        #[cfg(all(test, feature = "parallel"))]
        staged_chunks: std::sync::atomic::AtomicUsize,
    },
    #[cfg_attr(not(test), allow(dead_code))]
    Stub,
}

#[cfg(feature = "parallel")]
#[derive(Debug)]
enum StagingSlotState<T> {
    Empty,
    Idle(T),
    CheckedOut,
}

#[cfg(feature = "parallel")]
#[derive(Debug)]
struct StagingCell<T>(Mutex<StagingSlotState<T>>);

#[cfg(feature = "parallel")]
impl<T> StagingCell<T> {
    fn new() -> Self {
        Self(Mutex::new(StagingSlotState::Empty))
    }

    fn try_checkout_with(
        &self,
        init: impl FnOnce() -> Result<T>,
    ) -> Result<Option<StagingGuard<'_, T>>> {
        let needs_init = {
            let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            match std::mem::replace(&mut *state, StagingSlotState::CheckedOut) {
                StagingSlotState::Empty => true,
                StagingSlotState::Idle(value) => {
                    drop(state);
                    return Ok(Some(StagingGuard {
                        cell: self,
                        value: Some(value),
                    }));
                }
                StagingSlotState::CheckedOut => return Ok(None),
            }
        };
        debug_assert!(needs_init);
        // Guard restores Empty if `init` returns an error or unwinds.
        let mut guard = StagingGuard {
            cell: self,
            value: None,
        };
        let value = init()?;
        guard.value = Some(value);
        Ok(Some(guard))
    }
}

#[cfg(feature = "parallel")]
#[derive(Debug)]
struct StagingGuard<'a, T> {
    cell: &'a StagingCell<T>,
    value: Option<T>,
}

#[cfg(feature = "parallel")]
impl<T> std::ops::Deref for StagingGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().expect("staging guard value present")
    }
}

#[cfg(feature = "parallel")]
impl<T> std::ops::DerefMut for StagingGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().expect("staging guard value present")
    }
}

#[cfg(feature = "parallel")]
impl<T> Drop for StagingGuard<'_, T> {
    fn drop(&mut self) {
        let next = match self.value.take() {
            Some(value) => StagingSlotState::Idle(value),
            None => StagingSlotState::Empty,
        };
        let mut state = self.cell.0.lock().unwrap_or_else(PoisonError::into_inner);
        *state = next;
    }
}

/// Exclusive checkout of a device's pinned readback slots. Dropping restores the slots to
/// the device without holding a lock during CUDA or Rayon transfers.
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub(crate) struct StagingCheckout<'a> {
    guard: StagingGuard<'a, Box<super::memory::HostStaging>>,
    #[cfg(test)]
    staged_chunks: &'a std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "parallel")]
impl std::ops::Deref for StagingCheckout<'_> {
    type Target = super::memory::HostStaging;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

#[cfg(feature = "parallel")]
impl std::ops::DerefMut for StagingCheckout<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

#[cfg(all(test, feature = "parallel"))]
impl Drop for StagingCheckout<'_> {
    fn drop(&mut self) {
        if let Some(staging) = self.guard.value.as_ref() {
            self.staged_chunks
                .store(staging.chunks, std::sync::atomic::Ordering::Release);
        }
    }
}

impl GpuDevice {
    /// Open the device with the given ordinal and load the kernel module.
    ///
    /// NVRTC compiles the kernels to SASS for the device's exact architecture, which any
    /// CUDA 12 driver loads whatever the NVRTC minor version. When the NVRTC predates the
    /// device, it emits PTX for the newest known arch below it instead, which the driver
    /// JITs. A capability below 6.0 is rejected rather than targeted. The image is shared
    /// by every device opened in the process and cached on disk in `PRISM_PTX_CACHE_DIR`
    /// when set (an empty value disables disk caching), or in `prism-q-ptx` under the
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
            if capability.0 >= 7
                && matches!(
                    name,
                    "apply_mixed_tile"
                        | "qft_dif_pass"
                        | "qft_bit_reverse_tiled"
                        | "apply_multi_fused_tiled"
                )
            {
                let _ = func.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    100,
                );
            }
            functions.insert(name, func);
        }
        Ok(Self {
            inner: DeviceInner::Real {
                context,
                stream,
                _module: module,
                functions,
                #[cfg(feature = "parallel")]
                staging: StagingCell::new(),
                #[cfg(all(test, feature = "parallel"))]
                staged_chunks: std::sync::atomic::AtomicUsize::new(0),
            },
            #[cfg(feature = "bench-internal")]
            census: PassCounters::default(),
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
            #[cfg(feature = "bench-internal")]
            census: PassCounters::default(),
        }
    }

    /// Record one kernel launch that sweeps the state `passes` times. Compiles to nothing
    /// without `bench-internal`.
    #[inline(always)]
    pub(crate) fn count_launch(&self, passes: u64) {
        #[cfg(feature = "bench-internal")]
        {
            self.census.launches.fetch_add(1, Ordering::Relaxed);
            self.census.passes.fetch_add(passes, Ordering::Relaxed);
        }
        #[cfg(not(feature = "bench-internal"))]
        let _ = passes;
    }

    /// Record one readback to the host, which waits for the stream to drain. Compiles
    /// to nothing without `bench-internal`.
    #[inline(always)]
    pub(crate) fn count_drain(&self) {
        #[cfg(feature = "bench-internal")]
        self.census.drains.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one upload from the host. An upload from pageable memory is staged and
    /// returns without waiting for the stream. Compiles to nothing without
    /// `bench-internal`.
    #[inline(always)]
    pub(crate) fn count_upload(&self) {
        #[cfg(feature = "bench-internal")]
        self.census.uploads.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one launch of the dense 4x4 kernel, on top of its `count_launch`. Compiles
    /// to nothing without `bench-internal`.
    #[inline(always)]
    pub(crate) fn count_dense_2q(&self) {
        #[cfg(feature = "bench-internal")]
        self.census.dense_2q.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one launch of the mixed tile kernel, on top of its `count_launch`.
    /// Compiles to nothing without `bench-internal`.
    #[inline(always)]
    pub(crate) fn count_tile(&self) {
        #[cfg(feature = "bench-internal")]
        self.census.tiles.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "bench-internal")]
    pub(crate) fn census(&self) -> crate::gpu::GpuPassCensus {
        crate::gpu::GpuPassCensus {
            launches: self.census.launches.load(Ordering::Relaxed),
            passes: self.census.passes.load(Ordering::Relaxed),
            drains: self.census.drains.load(Ordering::Relaxed),
            uploads: self.census.uploads.load(Ordering::Relaxed),
            dense_2q: self.census.dense_2q.load(Ordering::Relaxed),
            tiles: self.census.tiles.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn stream(&self) -> Result<&Arc<CudaStream>> {
        match &self.inner {
            DeviceInner::Real { stream, .. } => Ok(stream),
            DeviceInner::Stub => Err(Self::stub_unsupported("stream access")),
        }
    }

    /// Check out the pinned readback slots for exclusive use without holding a lock across
    /// CUDA or Rayon transfers. Returns `Ok(None)` when another readback holds the slots.
    #[cfg(feature = "parallel")]
    pub(crate) fn checkout_staging(&self) -> Result<Option<StagingCheckout<'_>>> {
        match &self.inner {
            DeviceInner::Real {
                stream,
                staging,
                #[cfg(test)]
                staged_chunks,
                ..
            } => {
                let guard = staging
                    .try_checkout_with(|| Ok(Box::new(super::memory::HostStaging::new(stream)?)))?;
                Ok(guard.map(|guard| StagingCheckout {
                    guard,
                    #[cfg(test)]
                    staged_chunks,
                }))
            }
            DeviceInner::Stub => Err(Self::stub_unsupported("staged readback")),
        }
    }

    /// Run `f` on the pinned readback slots, allocating them on first use, or on a transient
    /// staging buffer when the shared slots are already checked out. A failed pinned
    /// allocation returns its error, and the next call allocates again.
    #[cfg(feature = "parallel")]
    pub(crate) fn with_staging<R>(
        &self,
        f: impl FnOnce(&mut super::memory::HostStaging) -> Result<R>,
    ) -> Result<R> {
        if let Some(mut guard) = self.checkout_staging()? {
            return f(&mut guard);
        }
        let stream = self.stream()?;
        let mut temp = super::memory::HostStaging::new(stream)?;
        f(&mut temp)
    }

    /// Chunks enqueued through the pinned readback slots since the device opened.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) fn staged_chunks(&self) -> usize {
        match &self.inner {
            DeviceInner::Real { staged_chunks, .. } => {
                staged_chunks.load(std::sync::atomic::Ordering::Acquire)
            }
            DeviceInner::Stub => 0,
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
    let cache_dir = ptx_cache_dir();
    let (image, module) = load_or_compile(context, capability, cache_dir.as_deref())?;
    cache.insert(capability, image);
    Ok(module)
}

/// Load the module from an image cached in `cache_dir`, or compile through NVRTC when
/// none loads. Passing `None` skips disk reads and writes. File names bind crate version,
/// device architecture, a hash of the source text, and the NVRTC version that compiled
/// them. A cached cubin matches whichever CUDA 12 NVRTC made it, so it is tried first
/// without opening NVRTC. Cached PTX is tried next, from this NVRTC only when one loads,
/// since a driver older than the NVRTC refuses it. A fresh compile is written back
/// atomically; a write failure is ignored, the module is already loaded.
fn load_or_compile(
    context: &Arc<CudaContext>,
    capability: (i32, i32),
    cache_dir: Option<&Path>,
) -> Result<(Arc<KernelImage>, Arc<CudaModule>)> {
    let source = kernel_source();
    let stem = cache_stem(capability, &source);
    let cached = |extension: &str| -> Vec<PathBuf> {
        let Some(dir) = cache_dir else {
            return Vec::new();
        };
        let prefix = format!("{stem}-nvrtc");
        let suffix = format!(".{extension}");
        std::fs::read_dir(dir)
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
            gpu_unusable(format!(
                "NVRTC, the CUDA runtime compiler, did not load (looked for {}); install a \
                 CUDA 12 or 13 toolkit or put its NVRTC library on the loader path",
                super::nvrtc::library_names()
            ))
        });
    }
    let version =
        nvrtc_version().ok_or_else(|| gpu_unusable("NVRTC did not report its version".into()))?;
    if let Some(dir) = cache_dir
        && let Some(hit) = try_load(&[dir.join(cache_file_name(&stem, version, "ptx"))])
    {
        return Ok(hit);
    }
    let image = compile_kernels(&source, capability)?;
    let module = image.load(context).map_err(fresh_module_err)?;
    if let Some(dir) = cache_dir {
        write_image_cache(
            &dir.join(cache_file_name(&stem, version, image.extension())),
            image.bytes(),
        );
    }
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
    let ptx = super::nvrtc::compile(source, arch, false)
        .map_err(|e| driver_err(&format!("PTX compilation (arch={arch})"), e))?;
    Ok(KernelImage::Ptx(String::from_utf8_lossy(&ptx).into_owned()))
}

/// Compile `source` to a cubin for `arch`.
fn compile_cubin(source: &str, arch: &str) -> Result<Vec<u8>> {
    super::nvrtc::compile(source, arch, true)
        .map_err(|e| driver_err(&format!("SASS compilation (arch={arch})"), e))
}

/// Real architectures this NVRTC emits SASS for, as `10 * major + minor`.
fn nvrtc_sass_archs() -> Result<Vec<i32>> {
    super::nvrtc::supported_archs().map_err(|e| driver_err("NVRTC supported archs", e))
}

/// Per-user cache location: `PRISM_PTX_CACHE_DIR` when set (empty disables disk
/// caching), otherwise `XDG_CACHE_HOME`, `LOCALAPPDATA`, or `HOME/.cache`, falling back
/// to the OS temp dir. A shared temp dir is avoided where a user dir exists so that no
/// other local user can seed the file another process will load.
fn ptx_cache_dir() -> Option<PathBuf> {
    ptx_cache_dir_from(
        std::env::var_os("PRISM_PTX_CACHE_DIR"),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("LOCALAPPDATA"),
        std::env::var_os("HOME"),
    )
}

fn ptx_cache_dir_from(
    override_dir: Option<std::ffi::OsString>,
    xdg: Option<std::ffi::OsString>,
    local_app_data: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    if let Some(val) = override_dir {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    let base = xdg
        .map(PathBuf::from)
        .or_else(|| local_app_data.map(PathBuf::from))
        .or_else(|| home.map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    Some(base.join("prism-q-ptx"))
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

/// Whether NVRTC loads under a CUDA 12 or CUDA 13 library name.
pub(crate) fn nvrtc_present() -> bool {
    super::nvrtc::present()
}

fn nvrtc_version() -> Option<(i32, i32)> {
    super::nvrtc::version()
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
#[path = "device_tests.rs"]
mod tests;
