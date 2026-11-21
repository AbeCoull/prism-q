//! Typed device allocation wrapper with host-to-device and device-to-host copies.

#[cfg(feature = "parallel")]
use cudarc::driver::{CudaEvent, CudaStream, DevicePtr, PinnedHostSlice};
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use crate::error::Result;

use super::device::GpuDevice;
use super::driver_err;

/// RAII wrapper over a typed device allocation.
pub struct GpuBuffer<T: DeviceRepr> {
    slice: CudaSlice<T>,
}

impl<T: DeviceRepr + ValidAsZeroBits> GpuBuffer<T> {
    /// Allocate `len` elements of `T` on the given device, zero-initialised.
    pub fn alloc_zeros(device: &GpuDevice, len: usize) -> Result<Self> {
        let stream = device.stream()?;
        let slice = stream
            .alloc_zeros::<T>(len)
            .map_err(|e| driver_err("alloc_zeros", e))?;
        Ok(Self { slice })
    }
}

impl<T: DeviceRepr> GpuBuffer<T> {
    /// Allocate a device buffer initialized from `host` in one transfer,
    /// skipping the zero fill of [`alloc_zeros`](Self::alloc_zeros).
    pub fn from_host(device: &GpuDevice, host: &[T]) -> Result<Self> {
        let stream = device.stream()?;
        let slice = stream
            .clone_htod(host)
            .map_err(|e| driver_err("from_host", e))?;
        device.count_upload();
        Ok(Self { slice })
    }

    /// Copy `host.len()` elements from host to device (in-place).
    pub fn copy_from_host(&mut self, device: &GpuDevice, host: &[T]) -> Result<()> {
        let stream = device.stream()?;
        stream
            .memcpy_htod(host, &mut self.slice)
            .map_err(|e| driver_err("copy_from_host", e))?;
        device.count_upload();
        Ok(())
    }

    /// Copy all of `src` into the start of this buffer on the device.
    pub(crate) fn copy_from_device(
        &mut self,
        device: &GpuDevice,
        src: &GpuBuffer<T>,
    ) -> Result<()> {
        let stream = device.stream()?;
        stream
            .memcpy_dtod(&src.slice, &mut self.slice)
            .map_err(|e| driver_err("copy_from_device", e))
    }

    /// Copy the whole buffer from the device into the start of `host`, leaving the rest
    /// of `host` untouched. With `parallel`, copies of `STAGED_READBACK_MIN_BYTES` or more
    /// go through `HostStaging`, and a failed pinned allocation of its slots returns the
    /// error.
    pub fn copy_to_host(&self, device: &GpuDevice, host: &mut [T]) -> Result<()> {
        #[cfg(feature = "parallel")]
        if self.slice.num_bytes() >= STAGED_READBACK_MIN_BYTES {
            device.with_staging(|staging| staging.copy_to_host(&self.slice, host))?;
            device.count_drain();
            return Ok(());
        }
        let stream = device.stream()?;
        stream
            .memcpy_dtoh(&self.slice, &mut host[..self.slice.len()])
            .map_err(|e| driver_err("copy_to_host", e))?;
        device.count_drain();
        Ok(())
    }
}

impl<T: DeviceRepr> GpuBuffer<T> {
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub(crate) fn raw(&self) -> &CudaSlice<T> {
        &self.slice
    }

    pub(crate) fn raw_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

/// Readbacks of at least this many bytes go through [`HostStaging`]. Into a fresh `Vec`
/// on an RTX 5070 Ti, the median staged copy took 0.58 ms against 1.04 ms direct at
/// 4 MiB and 0.79 ms against 2.17 ms at 8 MiB.
#[cfg(feature = "parallel")]
pub(crate) const STAGED_READBACK_MIN_BYTES: usize = 4 << 20;

#[cfg(feature = "parallel")]
const STAGING_SLOT_BYTES: usize = 64 << 20;

#[cfg(feature = "parallel")]
const STAGING_COPY_GRAIN: usize = 1 << 20;

/// Two pinned host slots that a large readback alternates between, so the DMA of one
/// chunk overlaps the parallel copy of the previous chunk into pageable memory.
///
/// A direct copy into a fresh pageable `Vec` faults its pages in one at a time inside
/// the driver: 2 GiB took 330 to 520 ms against 127 to 138 ms staged.
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub(crate) struct HostStaging {
    stream: std::sync::Arc<CudaStream>,
    slots: [PinnedHostSlice<u8>; 2],
    /// Recorded after each chunk copy into the matching slot. The slots' own events
    /// are recorded only when the context tracks several streams.
    ready: [CudaEvent; 2],
    #[cfg(test)]
    pub(super) chunks: usize,
}

#[cfg(feature = "parallel")]
impl HostStaging {
    pub(crate) fn new(stream: &std::sync::Arc<CudaStream>) -> Result<Self> {
        let context = stream.context();
        let slot = || {
            // SAFETY: a slot is read only after `ready` reports the device copy that
            // wrote the bytes being read.
            unsafe { context.alloc_pinned_with_flags::<u8>(STAGING_SLOT_BYTES, 0) }
                .map_err(|e| driver_err("alloc staging", e))
        };
        let event = || {
            context
                .new_event(None)
                .map_err(|e| driver_err("staging event", e))
        };
        Ok(Self {
            stream: stream.clone(),
            slots: [slot()?, slot()?],
            ready: [event()?, event()?],
            #[cfg(test)]
            chunks: 0,
        })
    }

    fn copy_to_host<T: DeviceRepr>(&mut self, src: &CudaSlice<T>, host: &mut [T]) -> Result<()> {
        use rayon::prelude::*;

        assert!(host.len() >= src.len());
        let bytes = src.num_bytes();
        // SAFETY: `host` holds at least `src.len()` values of the plain data type `T`,
        // so its first `bytes` bytes are in bounds, and the exclusive borrow of `host`
        // covers the byte view for its lifetime. The chunks carry `T`'s bytes verbatim,
        // as a direct `memcpy_dtoh` into `host` would.
        let dst = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) };
        let slot_ptrs = [
            self.slots[0]
                .as_mut_ptr()
                .map_err(|e| driver_err("staging slot", e))?,
            self.slots[1]
                .as_mut_ptr()
                .map_err(|e| driver_err("staging slot", e))?,
        ];
        let stream = &self.stream;
        let (src_ptr, _src_guard) = src.device_ptr(stream);
        let range = |k: usize| k * STAGING_SLOT_BYTES..bytes.min((k + 1) * STAGING_SLOT_BYTES);
        let chunks = bytes.div_ceil(STAGING_SLOT_BYTES);
        for k in 0..=chunks {
            if k < chunks {
                let r = range(k);
                // SAFETY: the slot holds `STAGING_SLOT_BYTES >= r.len()` bytes and no
                // host reference to it is live: the copy out of its previous chunk
                // finished on this thread before this enqueue.
                let slot = unsafe { std::slice::from_raw_parts_mut(slot_ptrs[k % 2], r.len()) };
                // The copy out below may run unrelated rayon jobs on this thread, so bind
                // the context before each raw enqueue rather than once.
                stream
                    .context()
                    .bind_to_thread()
                    .map_err(|e| driver_err("staged dtoh", e))?;
                // SAFETY: `src_ptr + r.start .. src_ptr + r.end` lies inside `src`'s
                // allocation, whose free is queued on this same stream and so runs
                // after the copy, even when an error returns before `ready` is waited.
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_async(
                        slot,
                        src_ptr + r.start as u64,
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| driver_err("staged dtoh", e))?;
                #[cfg(test)]
                {
                    self.chunks += 1;
                }
                self.ready[k % 2]
                    .record(stream)
                    .map_err(|e| driver_err("staging record", e))?;
            }
            if k > 0 {
                let out = range(k - 1);
                self.ready[(k - 1) % 2]
                    .synchronize()
                    .map_err(|e| driver_err("staged readback", e))?;
                // SAFETY: the event recorded after this chunk's copy has completed, so
                // the slot's first `out.len()` bytes are written, and the device writes
                // only the other slot until the next enqueue.
                let staged =
                    unsafe { std::slice::from_raw_parts(slot_ptrs[(k - 1) % 2], out.len()) };
                dst[out]
                    .par_chunks_mut(STAGING_COPY_GRAIN)
                    .zip(staged.par_chunks(STAGING_COPY_GRAIN))
                    .for_each(|(d, s)| d.copy_from_slice(s));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "parallel")]
impl Drop for HostStaging {
    fn drop(&mut self) {
        // A readback that returned early with an error can leave a copy into a slot in
        // flight, possibly with no `ready` record after it, so drain the stream before
        // the slots free.
        let _ = self.stream.synchronize();
    }
}

impl<T: DeviceRepr> std::fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.slice.len())
            .finish()
    }
}
