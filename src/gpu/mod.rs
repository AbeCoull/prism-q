//! Shared GPU execution resource.
//!
//! GPU support in PRISM-Q is not a standalone backend. It is an execution
//! context that CPU backends opt into for hot operations. Today the
//! statevector and stabilizer backends can attach an `Arc<GpuContext>` and
//! route their heavy kernels through this module.
//!
//! # Module layout
//!
//! - [`device`]: cudarc device wrapper, availability checks, VRAM queries
//! - [`memory`]: [`GpuBuffer`] RAII wrapper over device allocations
//! - `kernels::dense`: statevector kernels (CUDA C source plus launch helpers)
//! - `kernels::stabilizer`: stabilizer tableau and measurement kernels
//! - `kernels::bts`: compiled-sampler BTS kernels
//!
//! # Device state layout
//!
//! The statevector lives on device as a `CudaSlice<f64>` of length `2 * 2^n` holding
//! interleaved (re, im) pairs. This matches `num_complex::Complex64` and CUDA's `double2`
//! builtin, allowing zero-cost reinterpretation at kernel boundaries.

pub mod device;
pub(crate) mod kernels;
pub mod memory;

use std::sync::Arc;

use num_complex::Complex64;

use crate::error::Result;

pub use self::device::GpuDevice;
pub use self::memory::GpuBuffer;

/// Default minimum qubit count for routing a sub-circuit to GPU when
/// [`crate::BackendKind::StatevectorGpu`] is selected.
///
/// Below this threshold the dispatch layer builds a plain host-side
/// `StatevectorBackend` instead, keeping PCIe round-trips and kernel launch
/// latency off the critical path for small circuits that fit in L3.
/// Empirically measured at 14 on GTX 1080 Ti; override at runtime via the
/// `PRISM_GPU_MIN_QUBITS` environment variable.
pub const MIN_QUBITS_DEFAULT: usize = 14;

/// Whether a CUDA device is present and usable in this process.
///
/// Safe to call without a [`GpuContext`] and without the `cudarc` dynamic
/// library loaded. Returns `false` if detection fails for any reason.
pub fn is_available() -> bool {
    GpuContext::is_available()
}

/// Effective GPU crossover threshold. Reads `PRISM_GPU_MIN_QUBITS` once per
/// process and caches the result.
pub fn min_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(val) = std::env::var("PRISM_GPU_MIN_QUBITS") {
            if let Ok(n) = val.parse::<usize>() {
                return n;
            }
        }
        MIN_QUBITS_DEFAULT
    })
}

/// Default minimum shot count for routing compiled BTS sampling to the GPU.
///
/// Below this threshold the compiled sampler stays on the CPU BTS path, even
/// when a GPU context is attached, so repeated small shot batches do not pay
/// the device launch and transfer setup cost. Override at runtime via
/// `PRISM_GPU_BTS_MIN_SHOTS`.
pub const BTS_MIN_SHOTS_DEFAULT: usize = 131_072;

/// Default minimum compiled-sampler rank for routing BTS sampling to the GPU.
///
/// Very low-rank circuits such as GHZ or independent H layers are typically
/// faster on the CPU even at large shot counts because the host path can
/// expand each shot from a tiny number of random bits. Override at runtime via
/// `PRISM_GPU_BTS_MIN_RANK`.
pub const BTS_MIN_RANK_DEFAULT: usize = 4;

/// Default minimum average parity-row weight factor for routing compiled BTS
/// sampling to the GPU.
///
/// The effective requirement is `total_weight >= num_measurements * factor`,
/// which filters out low-weight parity maps whose device launch overhead tends
/// to dominate. Override at runtime via `PRISM_GPU_BTS_MIN_WEIGHT_FACTOR`.
pub const BTS_MIN_WEIGHT_FACTOR_DEFAULT: usize = 2;

/// Effective GPU BTS shot threshold. Reads `PRISM_GPU_BTS_MIN_SHOTS` once per
/// process and caches the result.
pub(crate) fn bts_min_shots() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(val) = std::env::var("PRISM_GPU_BTS_MIN_SHOTS") {
            if let Ok(n) = val.parse::<usize>() {
                return n;
            }
        }
        BTS_MIN_SHOTS_DEFAULT
    })
}

/// Effective GPU BTS rank threshold. Reads `PRISM_GPU_BTS_MIN_RANK` once per
/// process and caches the result.
pub(crate) fn bts_min_rank() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(val) = std::env::var("PRISM_GPU_BTS_MIN_RANK") {
            if let Ok(n) = val.parse::<usize>() {
                return n.max(1);
            }
        }
        BTS_MIN_RANK_DEFAULT
    })
}

/// Effective GPU BTS parity-weight threshold factor. Reads
/// `PRISM_GPU_BTS_MIN_WEIGHT_FACTOR` once per process.
pub(crate) fn bts_min_weight_factor() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(val) = std::env::var("PRISM_GPU_BTS_MIN_WEIGHT_FACTOR") {
            if let Ok(n) = val.parse::<usize>() {
                return n.max(1);
            }
        }
        BTS_MIN_WEIGHT_FACTOR_DEFAULT
    })
}

/// Default minimum qubit count for routing a sub-circuit to GPU when
/// [`crate::BackendKind::StabilizerGpu`] is selected.
///
/// Set deliberately high. The stabilizer device path now batches Clifford
/// launches and keeps measurement on device, but the product still defaults to
/// the host path until direct backend benchmarks justify lowering the
/// crossover.
///
/// Until then, opt in explicitly via `PRISM_STABILIZER_GPU_MIN_QUBITS=0` or
/// a similar low value for experimentation. The dispatch is correct; only
/// the performance story is pending.
pub const STABILIZER_MIN_QUBITS_DEFAULT: usize = 100_000;

/// Effective stabilizer GPU crossover threshold. Reads
/// `PRISM_STABILIZER_GPU_MIN_QUBITS` once per process.
pub fn stabilizer_min_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(val) = std::env::var("PRISM_STABILIZER_GPU_MIN_QUBITS") {
            if let Ok(n) = val.parse::<usize>() {
                return n;
            }
        }
        STABILIZER_MIN_QUBITS_DEFAULT
    })
}

/// Shared GPU execution context.
///
/// Holds the device handle and compiled kernel module. Cheap to clone via `Arc`. Pass by
/// `Arc<GpuContext>` so multiple backends or multiple simulations can share one device
/// initialisation.
pub struct GpuContext {
    device: Arc<GpuDevice>,
    launcher_scratch: std::sync::Mutex<kernels::LauncherScratch>,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    /// Initialise the context for the given CUDA device ordinal.
    ///
    /// Compiles the kernel module at construction. Subsequent calls reuse the cached PTX.
    pub fn new(device_id: usize) -> Result<Arc<Self>> {
        let device = Arc::new(GpuDevice::new(device_id)?);
        Ok(Arc::new(Self {
            device,
            launcher_scratch: std::sync::Mutex::new(kernels::LauncherScratch::default()),
        }))
    }

    /// Whether a CUDA device is present and usable.
    pub fn is_available() -> bool {
        GpuDevice::is_available()
    }

    /// Total VRAM on the device bound to this context.
    pub fn vram_bytes(&self) -> Result<usize> {
        self.device.vram_bytes()
    }

    /// Free VRAM currently available on the device bound to this context.
    ///
    /// Reflects allocations by all processes sharing the device, not only those
    /// made through this `GpuContext`. Use this before allocating a large
    /// statevector to avoid trial-and-error out-of-memory failures.
    pub fn vram_available(&self) -> Result<usize> {
        self.device.vram_available()
    }

    /// Maximum qubit count for a dense Complex64 statevector on this device.
    pub fn max_qubits_for_statevector(&self) -> Result<usize> {
        self.device.max_qubits_for_statevector()
    }

    /// Whether the currently-available VRAM can hold a dense Complex64
    /// statevector for `num_qubits` qubits.
    ///
    /// Counts only the main amplitude buffer (`2 * 2^num_qubits` f64s,
    /// 16 bytes per amplitude). Auxiliary scratch (probabilities kernel,
    /// measurement partials) typically adds up to 2^num_qubits additional
    /// f64s; callers expecting many concurrent measurements should leave
    /// headroom by checking against a smaller qubit count.
    pub fn fits_statevector(&self, num_qubits: usize) -> Result<bool> {
        if num_qubits >= usize::BITS as usize - 4 {
            return Ok(false);
        }
        let amplitude_bytes = (1usize << num_qubits).checked_mul(16).ok_or_else(|| {
            crate::error::PrismError::InvalidParameter {
                message: format!("num_qubits={num_qubits} overflows usize"),
            }
        })?;
        let available = self.vram_available()?;
        Ok(amplitude_bytes <= available)
    }

    pub(crate) fn device(&self) -> &GpuDevice {
        &self.device
    }

    pub(crate) fn launcher_scratch(&self) -> std::sync::MutexGuard<'_, kernels::LauncherScratch> {
        self.launcher_scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(test)]
    pub(crate) fn stub_for_tests() -> Arc<Self> {
        Arc::new(Self {
            device: Arc::new(GpuDevice::stub_for_tests()),
            launcher_scratch: std::sync::Mutex::new(kernels::LauncherScratch::default()),
        })
    }
}

/// Per-simulation device-resident state.
///
/// Owns a `GpuBuffer<f64>` holding `2 * 2^num_qubits` f64s (interleaved re/im). Tracks
/// `pending_norm` the same way the CPU statevector backend does, measurement collapse
/// accumulates into this scalar and the final scale is applied at `export_statevector` or
/// `probabilities` time.
#[derive(Debug)]
pub struct GpuState {
    context: Arc<GpuContext>,
    buffer: GpuBuffer<f64>,
    num_qubits: usize,
    pending_norm: f64,
    /// Device-side scratch buffer for `probabilities()` output, reused across calls so
    /// shot-sampling workflows don't re-allocate `2^n` f64s on every read.
    probs_scratch: std::cell::RefCell<Option<GpuBuffer<f64>>>,
}

impl GpuState {
    /// Allocate a fresh |0…0⟩ state on the device bound to `context`.
    pub fn new(context: Arc<GpuContext>, num_qubits: usize) -> Result<Self> {
        let len = 2usize.checked_shl(num_qubits as u32).ok_or_else(|| {
            crate::error::PrismError::InvalidParameter {
                message: format!("num_qubits={num_qubits} overflows addressable memory"),
            }
        })?;
        let buffer = GpuBuffer::<f64>::alloc_zeros(context.device(), len)?;
        let mut state = Self {
            context: context.clone(),
            buffer,
            num_qubits,
            pending_norm: 1.0,
            probs_scratch: std::cell::RefCell::new(None),
        };
        kernels::dense::launch_set_initial_state(&context, &mut state)?;
        Ok(state)
    }

    /// Number of qubits the buffer is sized for.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Multiplicative norm correction deferred from measurement collapse.
    pub fn pending_norm(&self) -> f64 {
        self.pending_norm
    }

    /// Read the raw (unnormalised) amplitude buffer back to host, as interleaved f64 pairs.
    pub fn copy_to_host_raw(&self) -> Result<Vec<f64>> {
        let mut host = vec![0.0_f64; self.buffer.len()];
        self.buffer.copy_to_host(self.context.device(), &mut host)?;
        Ok(host)
    }

    /// Read the amplitude buffer as `Vec<Complex64>` with the deferred `pending_norm`
    /// already applied.
    pub fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let raw = self.copy_to_host_raw()?;
        let norm = self.pending_norm;
        let out = raw
            .chunks_exact(2)
            .map(|p| Complex64::new(p[0] * norm, p[1] * norm))
            .collect();
        Ok(out)
    }

    /// Compute per-basis-state probabilities with `pending_norm²` scaling, via a GPU
    /// reduction kernel. Only `2^n` f64s cross PCIe (vs `2·2^n` for raw amplitudes).
    pub fn probabilities(&self) -> Result<Vec<f64>> {
        kernels::dense::launch_compute_probabilities(&self.context, self)
    }

    pub(crate) fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    pub(crate) fn buffer(&self) -> &GpuBuffer<f64> {
        &self.buffer
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut GpuBuffer<f64> {
        &mut self.buffer
    }

    pub(crate) fn set_pending_norm(&mut self, norm: f64) {
        self.pending_norm = norm;
    }

    /// Access (and lazily allocate) the cached device-side probabilities buffer. Returned as
    /// a `RefMut` so the caller can pass it to a kernel launch for the duration of the call.
    pub(crate) fn probs_scratch(&self) -> std::cell::RefMut<'_, Option<GpuBuffer<f64>>> {
        self.probs_scratch.borrow_mut()
    }
}

/// Per-simulation device-resident stabilizer tableau.
///
/// Owns two buffers that mirror the CPU tableau layout in
/// [`crate::backend::stabilizer::StabilizerBackend`]:
///
/// - `xz`: `(2n+1)` rows × `2 * num_words` u64s per row. Word ordering per row is
///   X-bits in `[0, num_words)` then Z-bits in `[num_words, 2*num_words)`.
/// - `phase`: `(2n+1)` bytes, one per row (0 = +1, 1 = -1).
///
/// `num_words = ceil(n / 64)`. The scratch row at index `2n` is reserved for
/// measurement computations.
#[derive(Debug)]
pub struct GpuTableau {
    #[allow(dead_code)]
    context: Arc<GpuContext>,
    xz: GpuBuffer<u64>,
    #[allow(dead_code)]
    phase: GpuBuffer<u8>,
    measure_pivot: GpuBuffer<i32>,
    measure_outcome: GpuBuffer<u8>,
    num_qubits: usize,
    num_words: usize,
}

impl GpuTableau {
    /// Allocate a fresh identity tableau on the device bound to `context`.
    ///
    /// Both buffers are zero-initialised by `GpuBuffer::alloc_zeros`, then a
    /// `stab_set_initial_tableau` kernel launch sets the destabilizer X-bits
    /// and stabilizer Z-bits in place. Phase stays all zero (identity).
    pub fn new(context: Arc<GpuContext>, num_qubits: usize) -> Result<Self> {
        let num_words = num_qubits.div_ceil(64);
        let total_rows = 2 * num_qubits + 1;
        let xz_len = total_rows * 2 * num_words.max(1);
        let phase_len = total_rows;

        let xz = GpuBuffer::<u64>::alloc_zeros(context.device(), xz_len)?;
        let phase = GpuBuffer::<u8>::alloc_zeros(context.device(), phase_len)?;
        let measure_pivot = GpuBuffer::<i32>::alloc_zeros(context.device(), 1)?;
        let measure_outcome = GpuBuffer::<u8>::alloc_zeros(context.device(), 1)?;

        let mut tableau = Self {
            context: context.clone(),
            xz,
            phase,
            measure_pivot,
            measure_outcome,
            num_qubits,
            num_words,
        };
        kernels::stabilizer::launch_set_initial_tableau(&context, &mut tableau)?;
        Ok(tableau)
    }

    /// Qubit count the tableau is sized for.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of u64 words per bit-packed row half (ceil(n / 64)).
    pub fn num_words(&self) -> usize {
        self.num_words
    }

    pub(crate) fn xz_mut(&mut self) -> &mut GpuBuffer<u64> {
        &mut self.xz
    }

    /// Split-borrow accessor returning both `xz` and `phase` buffers mutably.
    /// Kernel launchers need to pass both buffers as arguments to the same
    /// CUDA function, which requires holding concurrent mutable borrows of
    /// separate fields on the tableau.
    pub(crate) fn xz_phase_mut(&mut self) -> (&mut GpuBuffer<u64>, &mut GpuBuffer<u8>) {
        (&mut self.xz, &mut self.phase)
    }

    /// Split-borrow accessor returning the tableau XZ buffer and the cached
    /// pivot sentinel scratch used by `stab_measure_find_pivot`.
    pub(crate) fn xz_pivot_mut(&mut self) -> (&mut GpuBuffer<u64>, &mut GpuBuffer<i32>) {
        (&mut self.xz, &mut self.measure_pivot)
    }

    /// Split-borrow accessor returning the tableau XZ and phase buffers plus
    /// the cached one-byte deterministic outcome scratch.
    pub(crate) fn xz_phase_outcome_mut(
        &mut self,
    ) -> (&mut GpuBuffer<u64>, &mut GpuBuffer<u8>, &mut GpuBuffer<u8>) {
        (&mut self.xz, &mut self.phase, &mut self.measure_outcome)
    }

    pub(crate) fn total_rows(&self) -> usize {
        2 * self.num_qubits + 1
    }

    /// Copy the full tableau back to host: `xz` as `Vec<u64>`, `phase` as
    /// `Vec<bool>` (0 → false, non-zero → true). Host shape mirrors the CPU
    /// `StabilizerBackend` layout exactly; lets golden tests compare tableau
    /// state byte for byte.
    pub fn copy_to_host(&self) -> Result<(Vec<u64>, Vec<bool>)> {
        let device = self.context.device();
        let mut xz = vec![0u64; self.xz.len()];
        self.xz.copy_to_host(device, &mut xz)?;
        let mut phase_bytes = vec![0u8; self.phase.len()];
        self.phase.copy_to_host(device, &mut phase_bytes)?;
        let phase = phase_bytes.iter().map(|&b| b != 0).collect();
        Ok((xz, phase))
    }

    /// Upload `xz` and `phase` host buffers back into the device tableau.
    /// Used by the GPU measurement path's host copy-back: the CPU measurement
    /// routine mutates the tableau in-place, then this call syncs the result
    /// back to device so subsequent gate kernels see the collapsed state.
    pub fn copy_from_host(&mut self, xz: &[u64], phase: &[bool]) -> Result<()> {
        let device = self.context.device();
        self.xz.copy_from_host(device, xz)?;
        let phase_bytes: Vec<u8> = phase.iter().map(|&b| u8::from(b)).collect();
        self.phase.copy_from_host(device, &phase_bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PrismError;

    #[test]
    fn stub_context_reports_available_false() {
        // Even if a device is present, this test only uses the stub path.
        let ctx = GpuContext::stub_for_tests();
        assert!(ctx.device().is_stub());
    }

    #[test]
    fn state_new_on_stub_returns_unsupported() {
        let ctx = GpuContext::stub_for_tests();
        assert!(matches!(
            GpuState::new(ctx, 4).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn tableau_new_on_stub_returns_unsupported() {
        let ctx = GpuContext::stub_for_tests();
        assert!(matches!(
            GpuTableau::new(ctx, 4).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn min_qubits_default_when_env_unset() {
        // `min_qubits()` caches its result in a OnceLock at first call. Other
        // tests and benchmarks in the same process also read it. The invariant
        // this test enforces is that the cached value is a plausible threshold,
        // not a specific number (the env var may legitimately override it).
        let n = min_qubits();
        assert!(
            (1..=32).contains(&n),
            "implausible gpu crossover threshold: {n}"
        );
    }

    #[test]
    fn stub_vram_available_rejects_cleanly() {
        let ctx = GpuContext::stub_for_tests();
        assert!(matches!(
            ctx.vram_available().unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn stub_fits_statevector_rejects_cleanly() {
        let ctx = GpuContext::stub_for_tests();
        // Any query should surface the underlying unsupported error.
        assert!(matches!(
            ctx.fits_statevector(4).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn fits_statevector_rejects_overflowing_qubit_counts() {
        let ctx = GpuContext::stub_for_tests();
        // usize::BITS - 4 boundary: `1 << 60` times 16 bytes is already 16 EiB
        // which no GPU has. The function clamps these to `Ok(false)` before
        // touching the device, so even the stub context returns cleanly.
        assert!(!ctx.fits_statevector(128).unwrap());
    }
}
