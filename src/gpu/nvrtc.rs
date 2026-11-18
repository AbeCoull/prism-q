//! NVRTC, the CUDA runtime compiler, loaded at run time under CUDA 12 or CUDA 13 library
//! names. CUDA 12 is tried first: NVRTC 13 refuses architectures older than Turing.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

type Program = *mut c_void;
type Status = c_int;

const SUCCESS: Status = 0;

#[cfg(windows)]
const LIBRARY_NAMES: &[&str] = &["nvrtc64_120_0.dll", "nvrtc64_130_0.dll"];
#[cfg(not(windows))]
const LIBRARY_NAMES: &[&str] = &["libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"];

struct Api {
    _library: libloading::Library,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> Status,
    num_supported_archs: unsafe extern "C" fn(*mut c_int) -> Status,
    supported_archs: unsafe extern "C" fn(*mut c_int) -> Status,
    create_program: unsafe extern "C" fn(
        *mut Program,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> Status,
    destroy_program: unsafe extern "C" fn(*mut Program) -> Status,
    compile_program: unsafe extern "C" fn(Program, c_int, *const *const c_char) -> Status,
    program_log_size: unsafe extern "C" fn(Program, *mut usize) -> Status,
    program_log: unsafe extern "C" fn(Program, *mut c_char) -> Status,
    ptx_size: unsafe extern "C" fn(Program, *mut usize) -> Status,
    ptx: unsafe extern "C" fn(Program, *mut c_char) -> Status,
    cubin_size: unsafe extern "C" fn(Program, *mut usize) -> Status,
    cubin: unsafe extern "C" fn(Program, *mut c_char) -> Status,
    error_string: unsafe extern "C" fn(Status) -> *const c_char,
}

impl Api {
    fn load(name: &str) -> Option<Self> {
        // SAFETY: loading NVRTC by name runs only its own initialisers and asks nothing of
        // this process.
        let library = unsafe { libloading::Library::new(name) }.ok()?;
        macro_rules! symbol {
            ($name:literal) => {
                // SAFETY: each symbol is the NVRTC entry point of that name, whose C
                // signature the field type spells out; NVRTC 12 and 13 share them, and the
                // pointer is used only while `_library` keeps the library loaded.
                *unsafe { library.get($name) }.ok()?
            };
        }
        Some(Api {
            version: symbol!(b"nvrtcVersion\0"),
            num_supported_archs: symbol!(b"nvrtcGetNumSupportedArchs\0"),
            supported_archs: symbol!(b"nvrtcGetSupportedArchs\0"),
            create_program: symbol!(b"nvrtcCreateProgram\0"),
            destroy_program: symbol!(b"nvrtcDestroyProgram\0"),
            compile_program: symbol!(b"nvrtcCompileProgram\0"),
            program_log_size: symbol!(b"nvrtcGetProgramLogSize\0"),
            program_log: symbol!(b"nvrtcGetProgramLog\0"),
            ptx_size: symbol!(b"nvrtcGetPTXSize\0"),
            ptx: symbol!(b"nvrtcGetPTX\0"),
            cubin_size: symbol!(b"nvrtcGetCUBINSize\0"),
            cubin: symbol!(b"nvrtcGetCUBIN\0"),
            error_string: symbol!(b"nvrtcGetErrorString\0"),
            _library: library,
        })
    }

    fn check(&self, status: Status, call: &str) -> Result<(), String> {
        if status == SUCCESS {
            return Ok(());
        }
        // SAFETY: nvrtcGetErrorString returns a static NUL-terminated string for any status.
        let text = unsafe { CStr::from_ptr((self.error_string)(status)) };
        Err(format!("{call}: {}", text.to_string_lossy()))
    }
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| LIBRARY_NAMES.iter().find_map(|name| Api::load(name)))
        .as_ref()
}

/// The library names searched, in order, for error messages.
pub(crate) fn library_names() -> String {
    LIBRARY_NAMES.join(", ")
}

pub(crate) fn present() -> bool {
    api().is_some()
}

pub(crate) fn version() -> Option<(i32, i32)> {
    let api = api()?;
    let (mut major, mut minor) = (0, 0);
    // SAFETY: nvrtcVersion only writes its two out-parameters.
    let status = unsafe { (api.version)(&mut major, &mut minor) };
    (status == SUCCESS).then_some((major, minor))
}

/// Real architectures this NVRTC emits SASS for, as `10 * major + minor`.
pub(crate) fn supported_archs() -> Result<Vec<i32>, String> {
    let api = api().ok_or("NVRTC is not loaded")?;
    let mut count = 0;
    // SAFETY: the call only writes its out-parameter.
    let status = unsafe { (api.num_supported_archs)(&mut count) };
    api.check(status, "nvrtcGetNumSupportedArchs")?;
    let mut archs = vec![0; usize::try_from(count).unwrap_or(0)];
    // SAFETY: `archs` holds the `count` entries nvrtcGetNumSupportedArchs reported.
    let status = unsafe { (api.supported_archs)(archs.as_mut_ptr()) };
    api.check(status, "nvrtcGetSupportedArchs")?;
    Ok(archs)
}

/// NVRTC program handle, destroyed on drop.
struct ProgramHandle<'a> {
    api: &'a Api,
    handle: Program,
}

impl Drop for ProgramHandle<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from nvrtcCreateProgram and is destroyed only here.
        let _ = unsafe { (self.api.destroy_program)(&mut self.handle) };
    }
}

/// Compile `source` for the architecture `arch` (`sm_XY` or `compute_XY`), returning the
/// cubin when `cubin` is set and the PTX text otherwise. A failed compile carries NVRTC's
/// log.
pub(crate) fn compile(source: &str, arch: &str, cubin: bool) -> Result<Vec<u8>, String> {
    let api = api().ok_or("NVRTC is not loaded")?;
    let src = CString::new(source).expect("kernel source contains no NUL byte");
    let mut program = ProgramHandle {
        api,
        handle: std::ptr::null_mut(),
    };
    // SAFETY: `src` is NUL-terminated and outlives the program; no headers are passed.
    let status = unsafe {
        (api.create_program)(
            &mut program.handle,
            src.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    api.check(status, "nvrtcCreateProgram")?;
    let option = CString::new(format!("--gpu-architecture={arch}")).expect("no NUL in arch");
    let options = [option.as_ptr()];
    // SAFETY: `program` is live and `options` holds one NUL-terminated string that outlives
    // the call.
    let status = unsafe { (api.compile_program)(program.handle, 1, options.as_ptr()) };
    if let Err(e) = api.check(status, "nvrtcCompileProgram") {
        return Err(format!("{e}: {}", read_text(api, &program, true).trim()));
    }
    if cubin {
        let mut size = 0;
        // SAFETY: `program` compiled; the call only writes its out-parameter.
        let status = unsafe { (api.cubin_size)(program.handle, &mut size) };
        api.check(status, "nvrtcGetCUBINSize")?;
        let mut bytes = vec![0u8; size];
        // SAFETY: `bytes` holds the `size` bytes nvrtcGetCUBINSize reported.
        let status = unsafe { (api.cubin)(program.handle, bytes.as_mut_ptr().cast()) };
        api.check(status, "nvrtcGetCUBIN")?;
        Ok(bytes)
    } else {
        Ok(read_text(api, &program, false).into_bytes())
    }
}

/// The program's log (`log`) or PTX as text, without the trailing NUL; empty when NVRTC
/// cannot produce it.
fn read_text(api: &Api, program: &ProgramHandle<'_>, log: bool) -> String {
    let (size_fn, read_fn) = if log {
        (api.program_log_size, api.program_log)
    } else {
        (api.ptx_size, api.ptx)
    };
    let mut size = 0;
    // SAFETY: `program` is live; the call only writes its out-parameter.
    if unsafe { size_fn(program.handle, &mut size) } != SUCCESS || size == 0 {
        return String::new();
    }
    let mut buffer = vec![0u8; size];
    // SAFETY: `buffer` holds the `size` bytes the size call reported, NUL included.
    if unsafe { read_fn(program.handle, buffer.as_mut_ptr().cast()) } != SUCCESS {
        return String::new();
    }
    if buffer.last() == Some(&0) {
        buffer.pop();
    }
    String::from_utf8_lossy(&buffer).into_owned()
}
