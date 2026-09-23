//! GPU execution context that CPU backends attach to route their hot kernels to a CUDA
//! device, plus the device statevector and tableau types.
//!
//! Device statevectors are `2 * 2^n` interleaved (re, im) `f64` values, the layout of
//! `num_complex::Complex64` and CUDA's `double2`, so neither side converts.

pub mod device;
pub(crate) mod kernels;
pub mod memory;

use std::sync::Arc;

use num_complex::Complex64;

use crate::error::Result;

pub use self::device::GpuDevice;
pub use self::memory::GpuBuffer;

/// Default for `PRISM_GPU_MIN_QUBITS`: below this many qubits,
/// [`crate::BackendKind::StatevectorGpu`] builds a host `StatevectorBackend` instead.
///
/// Measured at 14 on a GTX 1080 Ti, where smaller states fit in L3 and PCIe round trips
/// and launch latency dominate.
pub const MIN_QUBITS_DEFAULT: usize = 14;

/// Whether a CUDA device is present and usable in this process.
///
/// Safe to call without a [`GpuContext`] and without the `cudarc` dynamic
/// library loaded. Returns `false` if detection fails for any reason.
pub fn is_available() -> bool {
    GpuContext::is_available()
}

/// Statevector GPU crossover in qubits, read from `PRISM_GPU_MIN_QUBITS` once per process.
pub fn min_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED
        .get_or_init(|| crate::env_knobs::usize_knob("PRISM_GPU_MIN_QUBITS", MIN_QUBITS_DEFAULT, 0))
}

/// Default for `PRISM_GPU_BTS_MIN_SHOTS`: smaller batches stay on the CPU BTS sampler
/// even with a GPU context attached, skipping the launch and transfer setup.
pub const BTS_MIN_SHOTS_DEFAULT: usize = 131_072;

/// Default for `PRISM_GPU_BTS_MIN_RANK`. Low-rank samplers such as GHZ or independent H
/// layers run faster on the CPU even at large shot counts, since each shot expands from a
/// few random bits.
pub const BTS_MIN_RANK_DEFAULT: usize = 4;

/// Default for `PRISM_GPU_BTS_MIN_WEIGHT_FACTOR`. The GPU path requires
/// `total_weight >= num_measurements * factor`, which filters out parity maps too sparse
/// to cover the launch overhead.
pub const BTS_MIN_WEIGHT_FACTOR_DEFAULT: usize = 2;

pub(crate) fn bts_min_shots() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_knob("PRISM_GPU_BTS_MIN_SHOTS", BTS_MIN_SHOTS_DEFAULT, 0)
    })
}

pub(crate) fn bts_min_rank() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_knob("PRISM_GPU_BTS_MIN_RANK", BTS_MIN_RANK_DEFAULT, 1)
    })
}

pub(crate) fn bts_min_weight_factor() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_knob(
            "PRISM_GPU_BTS_MIN_WEIGHT_FACTOR",
            BTS_MIN_WEIGHT_FACTOR_DEFAULT,
            1,
        )
    })
}

/// Default for `PRISM_STABILIZER_GPU_MIN_QUBITS`, set high so
/// [`crate::BackendKind::StabilizerGpu`] stays on the host path.
///
/// The device path is correct, but no direct backend benchmark yet justifies a lower
/// crossover. Set the variable to `0` to opt in.
pub const STABILIZER_MIN_QUBITS_DEFAULT: usize = 100_000;

/// Stabilizer GPU crossover in qubits, read from `PRISM_STABILIZER_GPU_MIN_QUBITS` once
/// per process.
pub fn stabilizer_min_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_knob(
            "PRISM_STABILIZER_GPU_MIN_QUBITS",
            STABILIZER_MIN_QUBITS_DEFAULT,
            0,
        )
    })
}

/// Device bytes a GPU statevector needs per amplitude: 16 for the interleaved
/// complex buffer, 8 for the probabilities scratch, 1 of margin for the measurement
/// partials and launcher metadata. The soft `Auto` gate and the hard explicit gate
/// both budget with it.
pub(crate) const STATEVECTOR_BYTES_PER_AMPLITUDE: usize = 25;

/// Device handle and compiled kernel module, shared across backends and simulations
/// through an `Arc` so the device initializes once.
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
    /// made through this `GpuContext`.
    pub fn vram_available(&self) -> Result<usize> {
        self.device.vram_available()
    }

    /// Maximum qubit count for a dense Complex64 statevector in the currently
    /// free VRAM of this device.
    pub fn max_qubits_for_statevector(&self) -> Result<usize> {
        self.device.max_qubits_for_statevector()
    }

    /// Whether the currently-available VRAM can hold a dense Complex64
    /// statevector for `num_qubits` qubits.
    ///
    /// Counts only the amplitude buffer at 16 bytes per amplitude;
    /// `fits_statevector_with_scratch` also budgets the probabilities and
    /// measurement scratch.
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

    /// Whether the currently-available VRAM holds a dense statevector for
    /// `num_qubits` plus the reduction scratch `GpuState` keeps live.
    ///
    /// Budgets 25 bytes per amplitude: 16 for amplitudes, 8 for the probabilities
    /// scratch, and 1 of margin for the measurement partials (`ceil(2^n / 512)` f64) and
    /// launcher metadata.
    /// This is the fail-fast gate for the `Auto` GPU statevector leaf; the backend's
    /// soft mode ([`StatevectorBackend::with_gpu_auto`](crate::backend::statevector::StatevectorBackend::with_gpu_auto))
    /// catches any transient allocation that slips past it.
    pub fn fits_statevector_with_scratch(&self, num_qubits: usize) -> Result<bool> {
        if num_qubits >= usize::BITS as usize - 5 {
            return Ok(false);
        }
        let bytes = (1usize << num_qubits)
            .checked_mul(STATEVECTOR_BYTES_PER_AMPLITUDE)
            .ok_or_else(|| crate::error::PrismError::InvalidParameter {
                message: format!("num_qubits={num_qubits} overflows usize"),
            })?;
        Ok(bytes <= self.vram_available()?)
    }

    /// Whether the currently-available VRAM holds a stabilizer tableau for
    /// `num_qubits`, matching the device layout in [`GpuTableau`].
    ///
    /// The tableau stores `(2n+1)` rows of `2 * ceil(n/64)` u64 words plus one
    /// phase byte per row; the fixed measurement scratch is negligible. Used as
    /// the fail-closed gate for the `Auto` GPU stabilizer leaf.
    pub fn fits_tableau(&self, num_qubits: usize) -> Result<bool> {
        let total_rows = num_qubits.checked_mul(2).and_then(|r| r.checked_add(1));
        let num_words = num_qubits.div_ceil(64).max(1);
        let xz_words = total_rows.and_then(|rows| rows.checked_mul(2 * num_words));
        let bytes = xz_words
            .and_then(|w| w.checked_mul(8))
            .and_then(|b| b.checked_add(total_rows.unwrap_or(usize::MAX)));
        match bytes {
            Some(bytes) => Ok(bytes <= self.vram_available()?),
            None => Ok(false),
        }
    }

    /// Block until every kernel queued on this context's stream has finished.
    ///
    /// Launches are asynchronous, so a caller timing device work must call
    /// this before reading the clock; a readback synchronizes on its own.
    pub fn synchronize(&self) -> Result<()> {
        self.device.stream()?.synchronize().map_err(|e| {
            crate::error::PrismError::BackendUnsupported {
                backend: "gpu".to_string(),
                operation: format!("synchronize: {e}"),
            }
        })
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

/// Device statevector of `2 * 2^num_qubits` interleaved (re, im) f64s.
///
/// As on the CPU statevector backend, measurement collapse accumulates into
/// `pending_norm`, applied at `export_statevector` or `probabilities` time.
#[derive(Debug)]
pub struct GpuState {
    context: Arc<GpuContext>,
    buffer: GpuBuffer<f64>,
    num_qubits: usize,
    pending_norm: f64,
    /// Reused across `probabilities()` calls so shot sampling does not reallocate `2^n`
    /// f64s per read.
    probs_scratch: std::cell::RefCell<Option<GpuBuffer<f64>>>,
}

impl GpuState {
    /// Allocate a fresh |0…0⟩ state on the device bound to `context`.
    ///
    /// Rejects `num_qubits` at or above `usize::BITS - 4`, the bound
    /// [`GpuContext::fits_statevector`] uses, where the 16 bytes per amplitude
    /// overflow `usize`.
    pub fn new(context: Arc<GpuContext>, num_qubits: usize) -> Result<Self> {
        if num_qubits >= usize::BITS as usize - 4 {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!("num_qubits={num_qubits} overflows addressable memory"),
            });
        }
        let len = 2usize << num_qubits;
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

    /// Allocate a device state initialized from host amplitudes. The slice
    /// length must be a power of two of at least 2; callers validate before
    /// converting.
    pub fn from_host_amplitudes(context: Arc<GpuContext>, amps: &[Complex64]) -> Result<Self> {
        debug_assert!(amps.len().is_power_of_two() && amps.len() >= 2);
        let num_qubits = amps.len().trailing_zeros() as usize;
        let mut host = Vec::with_capacity(amps.len() * 2);
        for a in amps {
            host.push(a.re);
            host.push(a.im);
        }
        let buffer = GpuBuffer::<f64>::from_host(context.device(), &host)?;
        Ok(Self {
            context,
            buffer,
            num_qubits,
            pending_norm: 1.0,
            probs_scratch: std::cell::RefCell::new(None),
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Multiplicative norm correction deferred from measurement collapse.
    pub fn pending_norm(&self) -> f64 {
        self.pending_norm
    }

    /// Read the amplitudes with `pending_norm` applied. The device copy lands in the
    /// returned vector's own storage, with no second host copy.
    pub fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let mut raw = vec![0.0_f64; self.buffer.len()];
        self.buffer.copy_to_host(self.context.device(), &mut raw)?;
        let mut out = interleaved_into_complex(raw);
        if self.pending_norm != 1.0 {
            scale_in_place(&mut out, self.pending_norm, self.num_qubits);
        }
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

    /// Cached probabilities buffer, `None` until the first caller allocates it. The caller
    /// grows it when `num_qubits` has increased.
    pub(crate) fn probs_scratch(&self) -> std::cell::RefMut<'_, Option<GpuBuffer<f64>>> {
        self.probs_scratch.borrow_mut()
    }
}

/// Reinterpret an interleaved `[re, im, re, im, ...]` vector as complex amplitudes
/// without copying. `raw.len()` must be even.
fn interleaved_into_complex(raw: Vec<f64>) -> Vec<Complex64> {
    let len = raw.len() / 2;
    assert_eq!(raw.len(), 2 * len, "interleaved buffer has an odd length");
    let ptr = Box::into_raw(raw.into_boxed_slice()) as *mut f64;
    // SAFETY: `Complex64` is `repr(C)` of two `f64`, so `len` of them occupy exactly the
    // `2 * len` f64s behind `ptr` at the same 8-byte alignment, and every pair is an
    // initialized value. The boxed slice has length equal to its capacity, so the vector
    // built here owns an allocation whose layout (`16 * len` bytes, align 8) matches the
    // one it will free, and the box was consumed by `into_raw` so nothing else frees it.
    unsafe { Vec::from_raw_parts(ptr.cast::<Complex64>(), len, len) }
}

/// Multiply every amplitude by the real `factor`, in parallel above the statevector
/// threshold so a 27 qubit export does not serialize a 2 GiB pass.
fn scale_in_place(amps: &mut [Complex64], factor: f64, num_qubits: usize) {
    use crate::backend::simd;
    let factor = Complex64::new(factor, 0.0);
    #[cfg(feature = "parallel")]
    if num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
        use rayon::prelude::*;
        amps.par_chunks_mut(crate::backend::MIN_PAR_ELEMS)
            .for_each(|chunk| simd::scale_complex_slice(chunk, factor));
        return;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = num_qubits;
    simd::scale_complex_slice(amps, factor);
}

/// Device stabilizer tableau.
///
/// The `xz` and `phase` buffers mirror the CPU tableau layout in
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
    context: Arc<GpuContext>,
    xz: GpuBuffer<u64>,
    phase: GpuBuffer<u8>,
    measure_pivot: GpuBuffer<i32>,
    measure_outcome: GpuBuffer<u8>,
    num_qubits: usize,
    num_words: usize,
}

impl GpuTableau {
    /// Allocate a fresh identity tableau on the device bound to `context`.
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

    pub(crate) fn xz_phase_mut(&mut self) -> (&mut GpuBuffer<u64>, &mut GpuBuffer<u8>) {
        (&mut self.xz, &mut self.phase)
    }

    /// The second buffer is the pivot sentinel scratch `stab_measure_find_pivot` writes.
    pub(crate) fn xz_pivot_mut(&mut self) -> (&mut GpuBuffer<u64>, &mut GpuBuffer<i32>) {
        (&mut self.xz, &mut self.measure_pivot)
    }

    /// The third buffer is the one-byte deterministic outcome scratch.
    pub(crate) fn xz_phase_outcome_mut(
        &mut self,
    ) -> (&mut GpuBuffer<u64>, &mut GpuBuffer<u8>, &mut GpuBuffer<u8>) {
        (&mut self.xz, &mut self.phase, &mut self.measure_outcome)
    }

    pub(crate) fn total_rows(&self) -> usize {
        2 * self.num_qubits + 1
    }

    /// Copy the tableau to host in the CPU `StabilizerBackend` layout, reading each phase
    /// byte as `b != 0`.
    pub fn copy_to_host(&self) -> Result<(Vec<u64>, Vec<bool>)> {
        let device = self.context.device();
        let mut xz = vec![0u64; self.xz.len()];
        self.xz.copy_to_host(device, &mut xz)?;
        let mut phase_bytes = vec![0u8; self.phase.len()];
        self.phase.copy_to_host(device, &mut phase_bytes)?;
        let phase = phase_bytes.iter().map(|&b| b != 0).collect();
        Ok((xz, phase))
    }

    /// Upload host `xz` and `phase` buffers, in the `copy_to_host` layout,
    /// into the device tableau.
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
        // The value is cached per process and the env var may override it, so only
        // plausibility is checked.
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

    #[test]
    fn state_new_rejects_overflowing_qubit_counts() {
        let ctx = GpuContext::stub_for_tests();
        // 63 is the silent case: `2 << 63` wraps to a zero-length buffer that
        // allocates fine and is then written past. 62 already fails at
        // allocation. The bound runs before allocation, so the stub reaches it.
        assert!(matches!(
            GpuState::new(ctx, 63).unwrap_err(),
            PrismError::InvalidParameter { .. }
        ));
    }

    #[test]
    fn fits_statevector_with_scratch_rejects_cleanly_on_stub() {
        let ctx = GpuContext::stub_for_tests();
        assert!(matches!(
            ctx.fits_statevector_with_scratch(4).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn fits_statevector_with_scratch_clamps_overflow_before_device() {
        let ctx = GpuContext::stub_for_tests();
        assert!(!ctx.fits_statevector_with_scratch(128).unwrap());
    }

    #[test]
    fn fits_tableau_rejects_cleanly_on_stub() {
        let ctx = GpuContext::stub_for_tests();
        assert!(matches!(
            ctx.fits_tableau(64).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn fits_tableau_clamps_overflow_before_device() {
        let ctx = GpuContext::stub_for_tests();
        assert!(!ctx.fits_tableau(usize::MAX / 2).unwrap());
    }
}
