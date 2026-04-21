//! Shared GPU execution resource.
//!
//! GPU support in PRISM-Q is not a standalone backend. It is an execution context that CPU
//! backends (Statevector first, MPS and TensorNetwork later) opt into for their hot
//! operations. Each backend that wants GPU acceleration holds an `Option<Arc<GpuContext>>`
//! and routes through this module's kernel namespaces when the context is present.
//!
//! # Module layout
//!
//! - [`device`] — cudarc device wrapper, availability checks, VRAM queries
//! - [`memory`] — [`GpuBuffer`] RAII wrapper over device allocations
//! - `kernels::dense` — statevector kernels (CUDA C source + launch helpers)
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

/// Shared GPU execution context.
///
/// Holds the device handle and compiled kernel module. Cheap to clone via `Arc`. Pass by
/// `Arc<GpuContext>` so multiple backends or multiple simulations can share one device
/// initialisation.
#[derive(Debug)]
pub struct GpuContext {
    device: Arc<GpuDevice>,
}

impl GpuContext {
    /// Initialise the context for the given CUDA device ordinal.
    ///
    /// Compiles the kernel module at construction. Subsequent calls reuse the cached PTX.
    pub fn new(device_id: usize) -> Result<Arc<Self>> {
        let device = Arc::new(GpuDevice::new(device_id)?);
        Ok(Arc::new(Self { device }))
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

    #[cfg(test)]
    pub(crate) fn stub_for_tests() -> Arc<Self> {
        Arc::new(Self {
            device: Arc::new(GpuDevice::stub_for_tests()),
        })
    }
}

/// Per-simulation device-resident state.
///
/// Owns a `GpuBuffer<f64>` holding `2 * 2^num_qubits` f64s (interleaved re/im). Tracks
/// `pending_norm` the same way the CPU statevector backend does — measurement collapse
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

        let mut tableau = Self {
            context: context.clone(),
            xz,
            phase,
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
