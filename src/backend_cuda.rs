//! CUDA 后端（骨架）。
//!
//! 通过 `libloading` 动态加载 CUDA 驱动（Windows `nvcuda.dll` / Linux `libcuda.so`），
//! 用 CUDA Driver API 实现平台无关 `TensorId` 的设备内存管理与上传/下载。
//! 设计对齐 [Albatross（信天翁）](https://github.com/BlinkDL/Albatross) 的 CUDA 后端：
//! 设备侧缓冲以裸 device pointer 存放，融合算子（norm_lerp6 / fuse_ka_dplr_norm /
//! gemv_rkv_stage1 等）后续逐一实现为 CUDA kernel。
//!
//! 当前为**骨架**：张量管理（分配/上传/下载/批处理边界）可用并通过单测，
//! 全部算子返回“未实现”错误，待后续按 Albatross 的 kernel 逐一补齐后，
//! `detect_backend()` 才会优先选择 CUDA。

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

use half::f16;

use crate::backend::{ComputeBackend, Int8Handle, TensorDtype, TensorId};
use crate::runtime::R;

/// 检查 CUDA 调用成功，否则返回错误（`op` 为当前操作名）。
macro_rules! cu_check {
    ($e:expr, $op:literal) => {
        let _r = unsafe { $e };
        if _r != CUDA_SUCCESS {
            return Err(format!("CudaBackend: {} failed with CUresult {_r}", $op).into());
        }
    };
}

/// CUDA 驱动函数返回码 `CUresult`。
type CuResult = c_int;
const CUDA_SUCCESS: CuResult = 0;

/// NVRTC 返回码 `nvrtcResult`。
type NvrtcResult = c_int;
const NVRTC_SUCCESS: NvrtcResult = 0;

/// NVRTC 程序句柄（`nvrtcProgram`）。
type NvrtcProgram = *mut c_void;
/// CUDA 模块句柄（`CUmodule`）。
type CuModule = *mut c_void;
/// CUDA 函数句柄（`CUfunction`）。
type CuFunction = *mut c_void;

/// CUDA 驱动 API 函数指针（`cudaDriver.h` 中 `CUresult` 返回的符号）。
type FnCuInit = unsafe extern "C" fn(u32) -> CuResult;
type FnCuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> CuResult;
type FnCuPrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, c_int) -> CuResult;
type FnCuPrimaryCtxRelease = unsafe extern "C" fn(c_int) -> CuResult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> CuResult;
type FnCuDeviceComputeCapability = unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> CuResult;
type FnCuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> CuResult;
type FnCuMemFree = unsafe extern "C" fn(u64) -> CuResult;
type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> CuResult;
type FnCuMemcpyHtoDAsync = unsafe extern "C" fn(u64, *const c_void, usize, CuStream) -> CuResult;
type FnCuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CuResult;
type FnCuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CuResult;
type FnCuModuleLoadDataEx = unsafe extern "C" fn(
    *mut CuModule,
    *const c_void,
    u32,
    *const c_int,
    *const *mut c_void,
) -> CuResult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    usize,
    *mut c_void,
    *const *mut c_void,
    *const *mut c_void,
) -> CuResult;
/// CUDA stream（`CUstream`）。
type CuStream = *mut c_void;
/// CUDA event（`CUevent`）。
type CuEvent = *mut c_void;
type FnCuStreamCreate = unsafe extern "C" fn(*mut CuStream, u32) -> CuResult;
type FnCuStreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;
type FnCuStreamSynchronize = unsafe extern "C" fn(CuStream) -> CuResult;
type FnCuEventCreate = unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult;
type FnCuEventRecord = unsafe extern "C" fn(CuEvent, CuStream) -> CuResult;
type FnCuEventSynchronize = unsafe extern "C" fn(CuEvent) -> CuResult;
type FnCuEventElapsedTime = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
type FnCuEventDestroy = unsafe extern "C" fn(CuEvent) -> CuResult;
/// CUDA graph（`CUgraph`）与可执行 graph（`CUgraphExec`）。
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;
type FnCuGraphBeginCapture = unsafe extern "C" fn(CuStream, u32) -> CuResult;
type FnCuGraphEndCapture = unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult;
type FnCuGraphInstantiate = unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult;
type FnCuGraphLaunch = unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult;
type FnCuGraphDestroy = unsafe extern "C" fn(CuGraph) -> CuResult;
type FnCuGraphExecDestroy = unsafe extern "C" fn(CuGraphExec) -> CuResult;
/// 捕获模式：`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1`（cuda.h `CUstreamCaptureMode_enum`：
/// GLOBAL=0 / THREAD_LOCAL=1 / RELAXED=2）。
/// 本后端只有一条 CUDA 线程（GPU actor 线程），用 THREAD_LOCAL 可避免驱动内部线程
/// 的 CUDA 操作被卷进捕获。此前常量误名为 `..._GLOBAL` 而值为 1，易误导。
const CU_STREAM_CAPTURE_MODE: u32 = 1;

/// NVRTC API 函数指针（`nvrtc.h` 中 `nvrtcResult` 返回的符号）。
type FnNvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
type FnNvrtcCompileProgram =
    unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
type FnNvrtcGetProgramLogSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type FnNvrtcGetProgramLog = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type FnNvrtcGetPTXSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type FnNvrtcGetPTX = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type FnNvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult;
type FnNvrtcGetErrorString = unsafe extern "C" fn(NvrtcResult) -> *const c_char;
type FnCuGetErrorString = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;

/// 加载并持有 CUDA 驱动 + NVRTC 函数指针（一次加载，全进程共享）。
struct CudaDriver {
    _lib: libloading::Library,
    _nvrtc_lib: libloading::Library,
    cu_init: FnCuInit,
    cu_device_get_count: FnCuDeviceGetCount,
    cu_device_get: FnCuDeviceGet,
    cu_primary_ctx_retain: FnCuPrimaryCtxRetain,
    cu_primary_ctx_release: FnCuPrimaryCtxRelease,
    cu_ctx_set_current: FnCuCtxSetCurrent,
    cu_device_compute_capability: FnCuDeviceComputeCapability,
    cu_mem_alloc_v2: FnCuMemAlloc,
    cu_mem_free_v2: FnCuMemFree,
    cu_memcpy_dtoh_v2: FnCuMemcpyDtoH,
    cu_memcpy_htod_async: FnCuMemcpyHtoDAsync,
    cu_mem_host_alloc: FnCuMemHostAlloc,
    cu_mem_free_host: FnCuMemFreeHost,
    cu_module_load_data_ex: FnCuModuleLoadDataEx,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_launch_kernel: FnCuLaunchKernel,
    cu_stream_create: FnCuStreamCreate,
    cu_stream_destroy: FnCuStreamDestroy,
    cu_stream_synchronize: FnCuStreamSynchronize,
    cu_event_create: FnCuEventCreate,
    cu_event_record: FnCuEventRecord,
    cu_event_synchronize: FnCuEventSynchronize,
    cu_event_elapsed_time: FnCuEventElapsedTime,
    cu_event_destroy: FnCuEventDestroy,
    cu_graph_begin_capture: FnCuGraphBeginCapture,
    cu_graph_end_capture: FnCuGraphEndCapture,
    cu_graph_instantiate: FnCuGraphInstantiate,
    cu_graph_launch: FnCuGraphLaunch,
    cu_graph_destroy: FnCuGraphDestroy,
    cu_graph_exec_destroy: FnCuGraphExecDestroy,
    cu_get_error_string: FnCuGetErrorString,
    nvrtc_create_program: FnNvrtcCreateProgram,
    nvrtc_compile_program: FnNvrtcCompileProgram,
    nvrtc_get_program_log_size: FnNvrtcGetProgramLogSize,
    nvrtc_get_program_log: FnNvrtcGetProgramLog,
    nvrtc_get_ptx_size: FnNvrtcGetPTXSize,
    nvrtc_get_ptx: FnNvrtcGetPTX,
    nvrtc_destroy_program: FnNvrtcDestroyProgram,
    nvrtc_get_error_string: FnNvrtcGetErrorString,
    // per-kernel profiling（PROF_CUDA_KERNEL=1）：launch 内用 cuEvent 测每个 kernel 的
    // GPU 执行时间，按 func→name 累计。仅用于诊断，不影响正常路径。
    // 用 Mutex 包裹以保持 CudaDriver: Send+Sync（事件指针为裸指针，仅诊断、单线程访问）。
    prof: Mutex<KernelProfiler>,
}

/// per-kernel profiling 状态（仅在诊断模式下由主线程访问，故 unsafe Send+Sync）。
struct KernelProfiler {
    enabled: bool,
    /// **非同步**事件对池：每次 launch 前后各记一个事件、**不 sync**，
    /// 排空时一次性 `cuStreamSynchronize` 再批量取 elapsed ⇒ 不破坏流水，测到的是真 GPU 时间。
    evs: Vec<(CuEvent, CuEvent)>,
    /// (内核名, 事件池槽位)，按 launch 顺序。
    pending: Vec<(String, usize)>,
    times: HashMap<String, (f64, usize)>,
    names: HashMap<usize, String>,
    /// 启动次数计数（`PROF_CUDA_COUNT=1`）：**不依赖事件、捕获期也照数**，
    /// 因此能穿过 self-loop 图拿到「每段/每步每个内核多少次」。判读用。
    counting: bool,
    counts: HashMap<String, usize>,
}
unsafe impl Send for KernelProfiler {}
unsafe impl Sync for KernelProfiler {}

/// 取符号；`names` 依次尝试（优先 `_v2` 版本化符号，回退到旧名）。
unsafe fn sym<F: Copy>(lib: &libloading::Library, name: &str, names: &[&[u8]]) -> R<F> {
    for n in names {
        if let Ok(s) = unsafe { lib.get::<F>(n) } {
            return Ok(*s);
        }
    }
    Err(format!("CudaDriver: symbol {name} not found").into())
}

impl CudaDriver {
    /// 打开 CUDA 驱动并加载所需函数指针。
    fn open() -> R<Self> {
        let lib = unsafe {
            #[cfg(target_os = "windows")]
            {
                libloading::Library::new("nvcuda.dll")?
            }
            #[cfg(not(target_os = "windows"))]
            {
                libloading::Library::new("libcuda.so.1")
                    .or_else(|_| libloading::Library::new("libcuda.so"))?
            }
        };
        // NVRTC 库（随 CUDA toolkit 分发）：Windows `nvrtc64_120_0.dll`（12.x 固定名），
        // Linux `libnvrtc.so.12`。
        let nvrtc_lib = unsafe {
            #[cfg(target_os = "windows")]
            {
                libloading::Library::new("nvrtc64_120_0.dll")
                    .or_else(|_| libloading::Library::new("nvrtc64_121_0.dll"))?
            }
            #[cfg(not(target_os = "windows"))]
            {
                libloading::Library::new("libnvrtc.so.12")
                    .or_else(|_| libloading::Library::new("libnvrtc.so"))?
            }
        };
        let cu_init = unsafe { sym(&lib, "cuInit", &[b"cuInit\0"]) }?;
        let cu_device_get_count =
            unsafe { sym(&lib, "cuDeviceGetCount", &[b"cuDeviceGetCount\0"]) }?;
        let cu_device_get = unsafe { sym(&lib, "cuDeviceGet", &[b"cuDeviceGet\0"]) }?;
        let cu_primary_ctx_retain = unsafe {
            sym(
                &lib,
                "cuPrimaryCtxRetain",
                &[b"cuPrimaryCtxRetain\0", b"cuDevicePrimaryCtxRetain\0"],
            )?
        };
        let cu_primary_ctx_release = unsafe {
            sym(
                &lib,
                "cuPrimaryCtxRelease",
                &[
                    b"cuPrimaryCtxRelease\0",
                    b"cuDevicePrimaryCtxRelease_v2\0",
                    b"cuDevicePrimaryCtxRelease\0",
                ],
            )?
        };
        let cu_ctx_set_current = unsafe {
            sym(
                &lib,
                "cuCtxSetCurrent",
                &[b"cuCtxSetCurrent\0", b"cuCtxSetCurrent\0"],
            )?
        };
        let cu_device_compute_capability = unsafe {
            sym(
                &lib,
                "cuDeviceComputeCapability",
                &[b"cuDeviceComputeCapability\0"],
            )?
        };
        let cu_mem_alloc_v2 =
            unsafe { sym(&lib, "cuMemAlloc", &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"])? };
        let cu_mem_free_v2 =
            unsafe { sym(&lib, "cuMemFree", &[b"cuMemFree_v2\0", b"cuMemFree\0"])? };
        let cu_memcpy_dtoh_v2 = unsafe {
            sym(
                &lib,
                "cuMemcpyDtoH",
                &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"],
            )?
        };
        let cu_memcpy_htod_async = unsafe {
            sym(
                &lib,
                "cuMemcpyHtoDAsync",
                &[b"cuMemcpyHtoDAsync_v2\0", b"cuMemcpyHtoDAsync\0"],
            )?
        };
        let cu_mem_host_alloc = unsafe {
            sym(
                &lib,
                "cuMemHostAlloc",
                &[b"cuMemHostAlloc\0", b"cuMemHostAlloc_v2\0"],
            )?
        };
        let cu_mem_free_host = unsafe {
            sym(
                &lib,
                "cuMemFreeHost",
                &[b"cuMemFreeHost\0", b"cuMemFreeHost_v2\0"],
            )?
        };
        let cu_module_load_data_ex =
            unsafe { sym(&lib, "cuModuleLoadDataEx", &[b"cuModuleLoadDataEx\0"]) }?;
        let cu_module_get_function =
            unsafe { sym(&lib, "cuModuleGetFunction", &[b"cuModuleGetFunction\0"]) }?;
        let cu_launch_kernel = unsafe { sym(&lib, "cuLaunchKernel", &[b"cuLaunchKernel\0"]) }?;
        let cu_stream_create = unsafe { sym(&lib, "cuStreamCreate", &[b"cuStreamCreate\0"]) }?;
        let cu_stream_destroy = unsafe { sym(&lib, "cuStreamDestroy", &[b"cuStreamDestroy\0"]) }?;
        let cu_stream_synchronize =
            unsafe { sym(&lib, "cuStreamSynchronize", &[b"cuStreamSynchronize\0"]) }?;
        let cu_event_create = unsafe { sym(&lib, "cuEventCreate", &[b"cuEventCreate\0"]) }?;
        let cu_event_record = unsafe { sym(&lib, "cuEventRecord", &[b"cuEventRecord\0"]) }?;
        let cu_event_synchronize =
            unsafe { sym(&lib, "cuEventSynchronize", &[b"cuEventSynchronize\0"]) }?;
        let cu_event_elapsed_time =
            unsafe { sym(&lib, "cuEventElapsedTime", &[b"cuEventElapsedTime\0"]) }?;
        let cu_event_destroy = unsafe { sym(&lib, "cuEventDestroy", &[b"cuEventDestroy\0"]) }?;
        let cu_graph_begin_capture =
            unsafe { sym(&lib, "cuStreamBeginCapture", &[b"cuStreamBeginCapture\0"]) }?;
        let cu_graph_end_capture =
            unsafe { sym(&lib, "cuStreamEndCapture", &[b"cuStreamEndCapture\0"]) }?;
        // ★ 顺序很关键：本工程只用**三参数**签名 `(CUgraphExec*, CUgraph, flags)`。
        // 旧版 `cuGraphInstantiate`（v1）是**五参数**（`phErrorNode`/`errorString` 在后），
        // 驱动里同时导出这两个符号 ⇒ 若先解析到 v1 再按三参数调用，第 3 个实参会被
        // 当成 `phErrorNode`（写 NULL）、第 4/5 个实参是**寄存器垃圾**当 `errorString`
        // ⇒ `0xC0000005`。优先取 `cuGraphInstantiateWithFlags`（三参数、语义完全一致），
        // 再退 `_v2`，最后才是 v1。
        let cu_graph_instantiate = unsafe {
            sym(
                &lib,
                "cuGraphInstantiateWithFlags",
                &[
                    b"cuGraphInstantiateWithFlags\0",
                    b"cuGraphInstantiate_v2\0",
                    b"cuGraphInstantiate\0",
                ],
            )
        }?;
        let cu_graph_launch = unsafe { sym(&lib, "cuGraphLaunch", &[b"cuGraphLaunch\0"]) }?;
        let cu_graph_destroy = unsafe { sym(&lib, "cuGraphDestroy", &[b"cuGraphDestroy\0"]) }?;
        let cu_graph_exec_destroy =
            unsafe { sym(&lib, "cuGraphExecDestroy", &[b"cuGraphExecDestroy\0"]) }?;
        let cu_get_error_string =
            unsafe { sym(&lib, "cuGetErrorString", &[b"cuGetErrorString\0"]) }?;

        let nvrtc_create_program =
            unsafe { sym(&nvrtc_lib, "nvrtcCreateProgram", &[b"nvrtcCreateProgram\0"]) }?;
        let nvrtc_compile_program = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcCompileProgram",
                &[b"nvrtcCompileProgram\0"],
            )
        }?;
        let nvrtc_get_program_log_size = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcGetProgramLogSize",
                &[b"nvrtcGetProgramLogSize\0"],
            )?
        };
        let nvrtc_get_program_log =
            unsafe { sym(&nvrtc_lib, "nvrtcGetProgramLog", &[b"nvrtcGetProgramLog\0"]) }?;
        let nvrtc_get_ptx_size =
            unsafe { sym(&nvrtc_lib, "nvrtcGetPTXSize", &[b"nvrtcGetPTXSize\0"]) }?;
        let nvrtc_get_ptx = unsafe { sym(&nvrtc_lib, "nvrtcGetPTX", &[b"nvrtcGetPTX\0"]) }?;
        let nvrtc_destroy_program = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcDestroyProgram",
                &[b"nvrtcDestroyProgram\0"],
            )
        }?;
        let nvrtc_get_error_string = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcGetErrorString",
                &[b"nvrtcGetErrorString\0"],
            )
        }?;
        Ok(Self {
            _lib: lib,
            _nvrtc_lib: nvrtc_lib,
            cu_init,
            cu_device_get_count,
            cu_device_get,
            cu_primary_ctx_retain,
            cu_primary_ctx_release,
            cu_ctx_set_current,
            cu_device_compute_capability,
            cu_mem_alloc_v2,
            cu_mem_free_v2,
            cu_memcpy_dtoh_v2,
            cu_memcpy_htod_async,
            cu_mem_host_alloc,
            cu_mem_free_host,
            cu_module_load_data_ex,
            cu_module_get_function,
            cu_launch_kernel,
            cu_stream_create,
            cu_stream_destroy,
            cu_stream_synchronize,
            cu_event_create,
            cu_event_record,
            cu_event_synchronize,
            cu_event_elapsed_time,
            cu_event_destroy,
            cu_graph_begin_capture,
            cu_graph_end_capture,
            cu_graph_instantiate,
            cu_graph_launch,
            cu_graph_destroy,
            cu_graph_exec_destroy,
            cu_get_error_string,
            nvrtc_create_program,
            nvrtc_compile_program,
            nvrtc_get_program_log_size,
            nvrtc_get_program_log,
            nvrtc_get_ptx_size,
            nvrtc_get_ptx,
            nvrtc_destroy_program,
            nvrtc_get_error_string,
            prof: Mutex::new(KernelProfiler {
                enabled: false,
                evs: Vec::new(),
                pending: Vec::new(),
                times: HashMap::new(),
                names: HashMap::new(),
                counting: false,
                counts: HashMap::new(),
            }),
        })
    }

    /// 排空计时槽：同步一次 stream，把 pending 里所有事件对的 elapsed 累计进 `times`。
    fn prof_flush(&self, stream: CuStream) {
        let mut pending = {
            let mut p = self.prof.lock().unwrap();
            if p.pending.is_empty() {
                return;
            }
            std::mem::take(&mut p.pending)
        };
        if pending.is_empty() {
            return;
        }
        unsafe { (self.cu_stream_synchronize)(stream) };
        let mut p = self.prof.lock().unwrap();
        for (name, slot) in pending.drain(..) {
            let (a, b) = p.evs[slot];
            let mut ms: f32 = 0.0;
            unsafe { (self.cu_event_elapsed_time)(&mut ms, a, b) };
            let e = p.times.entry(name).or_insert((0.0, 0));
            e.0 += ms as f64;
            e.1 += 1;
        }
    }

    /// 打印内核启动次数（`PROF_CUDA_COUNT=1`）并清零。
    fn dump_counts(&self) {
        let mut p = self.prof.lock().unwrap();
        if !p.counting || p.counts.is_empty() {
            return;
        }
        let mut rows: Vec<_> = p.counts.drain().collect();
        rows.sort_by_key(|b| std::cmp::Reverse(b.1));
        let total: usize = rows.iter().map(|(_, c)| *c).sum();
        log::info!("[内核启动次数] 合计 {total} 次");
        for (name, cnt) in rows {
            log::info!("  {cnt:>8}  {name}");
        }
    }

    /// 启用 per-kernel profiling（惰性创建事件）。
    fn enable_kernel_profiling(&self) {
        let mut p = self.prof.lock().unwrap();
        if p.enabled {
            return;
        }
        p.times.clear();
        p.names.clear();
        p.pending.clear();
        if p.evs.is_empty() {
            // 事件池：开一次 profiling 只建一次（8192 槽 ≈ 15 步的内核数；满了先排空）。
            for _ in 0..8192 {
                let (mut a, mut b): (CuEvent, CuEvent) =
                    (std::ptr::null_mut(), std::ptr::null_mut());
                unsafe {
                    (self.cu_event_create)(&mut a, 0);
                    (self.cu_event_create)(&mut b, 0);
                }
                p.evs.push((a, b));
            }
        }
        p.enabled = true;
    }

    /// 注册 func→kernel 名字映射（由 CudaBackend::kernel 在首次编译后调用）。
    /// 恒注册（不依赖 enabled）：消融探针 `ABLATE=` 需要用名字反查 func，与计时无关。
    fn register_kernel_name(&self, func: CuFunction, name: &str) {
        if func.is_null() {
            return;
        }
        let mut p = self.prof.lock().unwrap();
        p.names.insert(func as usize, name.to_string());
    }

    /// 清空累计的 per-kernel 时间（保留 enabled/names），用于隔离单段剖析。
    fn clear_prof(&self) {
        let mut p = self.prof.lock().unwrap();
        p.times.clear();
    }

    /// 打印累计的 per-kernel 时间并清空。
    fn dump_prof(&self, stream: CuStream) {
        self.prof_flush(stream);
        let p = self.prof.lock().unwrap();
        if !p.enabled {
            return;
        }
        let mut rows: Vec<_> = p.times.iter().collect();
        rows.sort_by(|a, b| {
            b.1.0
                .partial_cmp(&a.1.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total: f64 = p.times.values().map(|(ms, _)| *ms).sum();
        for (name, (ms, cnt)) in rows {
            let avg = *ms / *cnt as f64;
            let pct = *ms / total.max(1e-9) * 100.0;
            log::info!(
                "[PROF_KERNEL] {name:>28} cnt={cnt:>3} total={ms:>9.3}ms avg={avg:>8.4}ms ({pct:>5.1}%)"
            );
        }
        log::info!("[PROF_KERNEL] SUM {total:.3}ms");
    }

    /// NVRTC 将 CUDA C 源码编译为 PTX 字节。
    fn nvrtc_to_ptx(&self, src: &str, name: &str, device: i32) -> R<Vec<u8>> {
        // 统一前置：include CUDA 官方 fp16 头，提供原生 __half/half2/__hfma2 等硬件指令。
        // （NVRTC 编译已加 --include-path 到 CUDA PATH，见 nvrtc_to_ptx。）
        let pre = r#"
#include "cuda_fp16.h"
// 8 字节对齐的 4×fp16 向量加载（对标 Albatross row1_linear 的 half4 加载），
// 提升 gemv 权重读取的内存带宽利用率。
__device__ __forceinline__ void load_half4_f4(const __half* p, float& a, float& b, float& c, float& d) {
    unsigned long long v = *reinterpret_cast<const unsigned long long*>(p);
    a = __half2float(__ushort_as_half((unsigned short)(v & 0xffffu)));
    b = __half2float(__ushort_as_half((unsigned short)((v >> 16) & 0xffffu)));
    c = __half2float(__ushort_as_half((unsigned short)((v >> 32) & 0xffffu)));
    d = __half2float(__ushort_as_half((unsigned short)((v >> 48) & 0xffffu)));
}
"#;
        let full = format!("{pre}{src}");
        let src_c =
            CString::new(full).map_err(|_| format!("CudaBackend: kernel {name} has NUL byte"))?;
        let name_c =
            CString::new(name).map_err(|_| format!("CudaBackend: kernel name {name} has NUL"))?;
        let mut prog: NvrtcProgram = std::ptr::null_mut();
        let r = unsafe {
            (self.nvrtc_create_program)(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if r != NVRTC_SUCCESS {
            return Err(format!(
                "CudaBackend: nvrtcCreateProgram({name}) failed: {}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        // 按设备 compute capability 选 PTX 架构（旧卡如 Turing sm_75 编译 compute_80 会报
        // NO_BINARY_FOR_GPU）。多数 kernel 用 sm_75 也可被 JIT 到更高架构，故取实际能力。
        //
        // ⚠️ 7.x 必须细分到 minor：**int8 IMMA（`mma.m8n8k16`）是 sm_75 起才有的指令**，
        // 若对 cc=7.5 仍按 compute_70 出 PTX，JIT 会以
        // `cuModuleLoadDataEx failed: 218 a PTX JIT compilation failed` 静默失败
        // （PTX 的 .target 低于指令要求）。sm_75 ⊃ sm_70，细分不会丢失兼容性。
        let mut maj: c_int = 0;
        let mut min: c_int = 0;
        cu_check!(
            (self.cu_device_compute_capability)(&mut maj, &mut min, device),
            "cuDeviceComputeCapability"
        );
        let arch = if maj >= 8 {
            "compute_80"
        } else if maj == 7 && min >= 5 {
            "compute_75"
        } else if maj == 7 {
            "compute_70"
        } else {
            return Err(
                format!("CudaBackend: unsupported GPU compute capability {maj}.{min}").into(),
            );
        };
        let opt = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        // 让 NVRTC 使用 CUDA 官方头文件（cuda_fp16.h 等），从而能生成 __hfma2 等硬件半精度指令。
        // 优先取 CUDA_PATH 环境变量，回退到常见安装路径。
        let cuda_inc = std::env::var("CUDA_PATH")
            .map(|p| format!("{p}\\include"))
            .unwrap_or_else(|_| {
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include".to_string()
            });
        let inc_opt = CString::new(format!("--include-path={cuda_inc}")).unwrap();
        let opts = [opt.as_ptr(), inc_opt.as_ptr()];
        let r = unsafe { (self.nvrtc_compile_program)(prog, 2, opts.as_ptr()) };
        if r != NVRTC_SUCCESS {
            let log = self.nvrtc_program_log(prog);
            unsafe { (self.nvrtc_destroy_program)(&mut prog) };
            return Err(format!(
                "CudaBackend: nvrtcCompileProgram({name}) failed ({}): {log}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        let mut size: usize = 0;
        let r = unsafe { (self.nvrtc_get_ptx_size)(prog, &mut size) };
        if r != NVRTC_SUCCESS {
            unsafe { (self.nvrtc_destroy_program)(&mut prog) };
            return Err(format!(
                "CudaBackend: nvrtcGetPTXSize({name}) failed: {}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        let mut ptx = vec![0u8; size];
        let r = unsafe { (self.nvrtc_get_ptx)(prog, ptx.as_mut_ptr() as *mut c_char) };
        unsafe { (self.nvrtc_destroy_program)(&mut prog) };
        if r != NVRTC_SUCCESS {
            return Err(format!(
                "CudaBackend: nvrtcGetPTX({name}) failed: {}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        Ok(ptx)
    }

    /// 读取 NVRTC 编译日志（失败时诊断用）。
    fn nvrtc_program_log(&self, prog: NvrtcProgram) -> String {
        let mut size: usize = 0;
        if unsafe { (self.nvrtc_get_program_log_size)(prog, &mut size) } != NVRTC_SUCCESS {
            return String::new();
        }
        if size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        unsafe { (self.nvrtc_get_program_log)(prog, buf.as_mut_ptr() as *mut c_char) };
        String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string()
    }

    /// 取 NVRTC 错误码对应的可读字符串。
    fn nvrtc_error_str(&self, code: NvrtcResult) -> String {
        let p = unsafe { (self.nvrtc_get_error_string)(code) };
        if p.is_null() {
            format!("nvrtc error {code}")
        } else {
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// 把 PTX 字节加载为 CU 模块。
    fn load_module(&self, ptx: &[u8]) -> R<CuModule> {
        let mut module: CuModule = std::ptr::null_mut();
        let r = unsafe {
            (self.cu_module_load_data_ex)(
                &mut module,
                ptx.as_ptr() as *const c_void,
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if r != CUDA_SUCCESS {
            let msg = self.cuda_error_str(r);
            return Err(format!("CudaBackend: cuModuleLoadDataEx failed: {r} {msg}").into());
        }
        Ok(module)
    }

    /// 取 CUDA 驱动错误码对应的可读字符串。
    fn cuda_error_str(&self, code: c_int) -> String {
        let mut p: *const c_char = std::ptr::null();
        if unsafe { (self.cu_get_error_string)(code, &mut p) } == CUDA_SUCCESS && !p.is_null() {
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("cuda error {code}")
        }
    }

    /// 从模块取指定 kernel 函数句柄。
    fn get_function(&self, module: CuModule, func_name: &str) -> R<CuFunction> {
        let ptx_name = CString::new(func_name)
            .map_err(|_| format!("CudaBackend: func name {func_name} has NUL"))?;
        let mut func: CuFunction = std::ptr::null_mut();
        let r = unsafe { (self.cu_module_get_function)(&mut func, module, ptx_name.as_ptr()) };
        if r != CUDA_SUCCESS {
            return Err(
                format!("CudaBackend: cuModuleGetFunction({func_name}) failed: {r}").into(),
            );
        }
        Ok(func)
    }

    /// 启动 kernel 到 `stream`。`params` 为参数值指针数组（每个元素指向一个参数值）。
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: CuStream,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[*mut c_void],
    ) -> R<()> {
        self.launch_smem(stream, func, grid, block, params, 0)
    }

    /// 启动 kernel，支持动态共享内存大小 `smem_bytes`（字节）。其余同 `launch`。
    #[allow(clippy::too_many_arguments)]
    fn launch_smem(
        &self,
        stream: CuStream,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[*mut c_void],
        smem_bytes: usize,
    ) -> R<()> {
        // 消融探针（ABLATE=name1,name2，诊断用）：命中则**不启动**该 kernel。
        // 用途：逐个"抽掉"内核测总步时间差 → 得到真实的关键路径占比。
        // 比 PROF_CUDA_KERNEL 可靠——后者每个 kernel 后同步，测到的是同步开销。
        if ablate_hit(self, func) {
            return Ok(());
        }
        // 启动计数（`PROF_CUDA_COUNT=1`）：不加同步、捕获期也计数，故能测量图里每段的内核调用次数。
        {
            let mut p = self.prof.lock().unwrap();
            if p.counting {
                let name = p
                    .names
                    .get(&(func as usize))
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                *p.counts.entry(name).or_insert(0) += 1;
            }
        }
        // per-kernel profiling（`PROF_CUDA_KERNEL=1`）：**非同步** —— launch 前后各记一个
        // 事件、**不 sync**，排空时（`dump_prof`）一次性同步再批量取 elapsed。
        // ⇒ 不破坏流水，测到的是真 GPU 执行时间。
        // ⚠️ 必须配 `NO_SELFLOOP_GRAPH=1`：捕获期内核只被记录、不执行，计时全是 0。
        {
            let prof_enabled = self.prof.lock().unwrap().enabled;
            if prof_enabled {
                let (begin, end, name) = {
                    let mut p = self.prof.lock().unwrap();
                    let slot = p.pending.len();
                    if slot >= p.evs.len() {
                        drop(p);
                        self.prof_flush(stream);
                        p = self.prof.lock().unwrap();
                    }
                    let slot = p.pending.len();
                    let name = p
                        .names
                        .get(&(func as usize))
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    p.pending.push((name.clone(), slot));
                    let (a, b) = p.evs[slot];
                    (a, b, name)
                };
                let _ = name;
                unsafe { (self.cu_event_record)(begin, stream) };
                let r = unsafe {
                    (self.cu_launch_kernel)(
                        func,
                        grid.0,
                        grid.1,
                        grid.2,
                        block.0,
                        block.1,
                        block.2,
                        smem_bytes,
                        stream,
                        params.as_ptr(),
                        std::ptr::null(),
                    )
                };
                if r != CUDA_SUCCESS {
                    return Err(format!("CudaBackend: cuLaunchKernel failed: {r}").into());
                }
                unsafe { (self.cu_event_record)(end, stream) };
                return Ok(());
            }
        }
        let r = unsafe {
            (self.cu_launch_kernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                smem_bytes,
                stream,
                params.as_ptr(),
                std::ptr::null(),
            )
        };
        if r != CUDA_SUCCESS {
            return Err(format!("CudaBackend: cuLaunchKernel failed: {r}").into());
        }
        Ok(())
    }
}

/// cuBLAS 句柄（`cublasHandle_t`）。
type CublasHandle = *mut c_void;
type CublasStatus = c_int;
const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
/// `cublasOperation_t`：CUBLAS_OP_N=0, CUBLAS_OP_T=1。
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
/// `cudaDataType_t`：CUDA_R_32F=0, CUDA_R_16F=2。
const CUDA_R_32F: c_int = 0;
const CUDA_R_16F: c_int = 2;
/// `cublasComputeType_t`：CUBLAS_COMPUTE_32F=68（fp16 输入晋升 fp32 累加）。
const CUBLAS_COMPUTE_32F: c_int = 68;
/// `cublasGemmAlgo_t`：CUBLAS_GEMM_DEFAULT=-1。
const CUBLAS_GEMM_DEFAULT: c_int = -1;

type FnCublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type FnCublasDestroy = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
type FnCublasSetStream = unsafe extern "C" fn(CublasHandle, CuStream) -> CublasStatus;
type FnCublasGemmEx = unsafe extern "C" fn(
    CublasHandle,
    c_int,         // transa
    c_int,         // transb
    c_int,         // m
    c_int,         // n
    c_int,         // k
    *const c_void, // alpha
    *const c_void, // A
    c_int,         // Atype
    c_int,         // lda
    *const c_void, // B
    c_int,         // Btype
    c_int,         // ldb
    *const c_void, // beta
    *mut c_void,   // C
    c_int,         // Ctype
    c_int,         // ldc
    c_int,         // computeType
    c_int,         // algo
) -> CublasStatus;

/// 加载并持有 cuBLAS 函数指针（一次加载，全进程共享）。
struct CublasDriver {
    _lib: libloading::Library,
    cublas_create: FnCublasCreate,
    cublas_destroy_v2: FnCublasDestroy,
    cublas_set_stream: FnCublasSetStream,
    cublas_gemm_ex: FnCublasGemmEx,
}

impl CublasDriver {
    fn open() -> R<Self> {
        let lib = unsafe {
            #[cfg(target_os = "windows")]
            {
                libloading::Library::new("cublas64_12.dll")
                    .or_else(|_| libloading::Library::new("cublas64_11.dll"))?
            }
            #[cfg(not(target_os = "windows"))]
            {
                libloading::Library::new("libcublas.so.12")
                    .or_else(|_| libloading::Library::new("libcublas.so.11"))?
            }
        };
        let cublas_create = unsafe {
            sym(
                &lib,
                "cublasCreate",
                &[b"cublasCreate_v2\0", b"cublasCreate\0"],
            )
        }?;
        let cublas_destroy_v2 = unsafe {
            sym(
                &lib,
                "cublasDestroy",
                &[b"cublasDestroy_v2\0", b"cublasDestroy\0"],
            )?
        };
        let cublas_set_stream = unsafe {
            sym(
                &lib,
                "cublasSetStream",
                &[b"cublasSetStream_v2\0", b"cublasSetStream\0"],
            )?
        };
        let cublas_gemm_ex = unsafe { sym(&lib, "cublasGemmEx", &[b"cublasGemmEx\0"]) }?;
        Ok(Self {
            _lib: lib,
            cublas_create,
            cublas_destroy_v2,
            cublas_set_stream,
            cublas_gemm_ex,
        })
    }
}

/// 全局 cuBLAS 驱动（惰性加载一次；库不存在时返回 None，gemm 回退自定义 kernel）。
fn cublas_driver() -> Option<&'static CublasDriver> {
    static D: OnceLock<Option<CublasDriver>> = OnceLock::new();
    D.get_or_init(|| match CublasDriver::open() {
        Ok(d) => Some(d),
        Err(e) => {
            log::warn!("CublasDriver::open failed (fallback to custom gemm): {e}");
            None
        }
    })
    .as_ref()
}

/// 全局 CUDA 驱动（惰性加载一次）。
fn driver() -> Option<&'static CudaDriver> {
    static D: OnceLock<Option<CudaDriver>> = OnceLock::new();
    D.get_or_init(|| match CudaDriver::open() {
        Ok(d) => Some(d),
        Err(e) => {
            log::warn!("CudaDriver::open failed: {e}");
            None
        }
    })
    .as_ref()
}

/// 探测 CUDA 是否可用：驱动可加载、`cuInit` 成功且存在 ≥1 个设备。
pub fn cuda_available() -> bool {
    let Some(d) = driver() else {
        log::warn!("cuda_available: CudaDriver::open failed");
        return false;
    };
    let mut count: i32 = 0;
    if unsafe { (d.cu_init)(0) } != CUDA_SUCCESS {
        log::warn!("cuda_available: cuInit failed");
        return false;
    }
    if unsafe { (d.cu_device_get_count)(&mut count) } != CUDA_SUCCESS {
        log::warn!("cuda_available: cuDeviceGetCount failed");
        return false;
    }
    count > 0
}

/// 设备侧张量（骨架阶段仅需 device pointer + len）。
#[derive(Debug, Clone)]
enum CudaTensor {
    F32 { dptr: u64, len: usize },
    F16 { dptr: u64, len: usize },
    U32 { dptr: u64, len: usize },
}

/// CUDA 后端骨架：持有驱动 + 主上下文 + 张量映射 + kernel 缓存。
pub struct CudaBackend {
    drv: &'static CudaDriver,
    /// 主上下文句柄：`cuPrimaryCtxRetain` 保留其存活，进程退出前由 `Drop` 释放。
    #[allow(dead_code)]
    ctx: *mut c_void,
    device: i32,
    tensors: HashMap<TensorId, CudaTensor>,
    lens: HashMap<TensorId, usize>,
    next_id: u32,
    /// kernel 缓存：kernel 名 → (已加载模块, 函数句柄)。模块需保持存活，故随函数一起存。
    kernels: HashMap<String, (CuModule, CuFunction)>,
    /// 计算 stream：所有 kernel 与异步拷贝都在其上排队，`download` 前 `cuStreamSynchronize`。
    stream: CuStream,
    /// GPU 批剖析（PROF_CUDA_GPU=1）：begin_batch 记 start，end_batch 记 end 并同步打印耗时。
    prof_ev_start: CuEvent,
    prof_ev_end: CuEvent,
    prof_gpu: bool,
    /// per-kernel 剖析（PROF_CUDA_KERNEL=1）：launch 内逐 kernel 计时，end_batch 打印。
    prof_kernel: bool,
    /// 是否正在 CUDA stream 捕获（begin_selfloop_capture..end_selfloop_capture 之间）。
    graph_capturing: bool,
    /// prefill graph：按 token 数 T 缓存，整段 prefill 一次捕获、同 T 重放。
    prefill_graphs: HashMap<usize, CuGraphExec>,
    /// 正在捕获的 prefill 对应 T（end_prefill_capture 用）。
    prefill_t: usize,
    /// 解码 self-loop 图：按形状 key 缓存，**每个形状只捕获一次**，此后长期重放。
    /// 动机：逐段重复 capture/instantiate/destroy 会触发驱动崩溃（实测
    /// nvcuda64.dll 0xC0000005 @ cuGraphInstantiate）。
    selfloop_graphs: HashMap<u64, CuGraphExec>,
    /// 捕获失败被永久禁用的 key（降级为非 graph 逐轮提交，不再重试捕获）。
    selfloop_disabled: std::collections::HashSet<u64>,
    /// 正在捕获的 self-loop 形状 key（end_selfloop_capture 用）。
    selfloop_key: u64,
    /// cuBLAS 句柄（prefill GEMM 用；cuBLAS 不可用时为 None → 回退自定义 kernel）。
    cublas: Option<CublasHandle>,
    /// cuBLAS GEMM 剖析（PROF_GEMM=1）：逐 (m,n,k,op) 累计次数与耗时，end_batch 打印。
    gemm_prof_ev_start: CuEvent,
    gemm_prof_ev_end: CuEvent,
    gemm_prof: bool,
    gemm_times: HashMap<(usize, usize, usize, i32), (u64, f64)>,
    /// pinned host 暂存区（sampler 异步行 + 小上传 scratch）。
    pinned: *mut c_void,
    /// pinned 暂存区总行数（固定 PINNED_ROWS；batch 宽行时按 batch 分组）。
    pinned_rows: usize,
    /// 批量上传环形缓冲（upload_bulk_begin..end 期间启用，消除逐 tensor 全流同步）。
    bulk: std::cell::RefCell<Option<BulkRing>>,
    /// 从其它后端导入的张量（`import_tensors_from` 权重共享）：Drop 时不释放。
    foreign: std::collections::HashSet<TensorId>,
    /// batch 线性层 fp16 激活暂存池（`gemv_variant_mb16` 用）：(容量元素数, 张量)。
    /// 按需增长、**旧块不释放**——图里烘焙了指针，替换后旧块可能仍被已捕获的图引用。
    /// 池尺寸受形状数限制（batch×k 的取值集合很小），泄漏量可忽略。
    x16_pool: Vec<(usize, TensorId)>,
    /// W8A8 激活量化暂存池：int8 `xq`（u32 计量的元素数）与 `xaux`（float4 个数）。
    /// 与 `x16_pool` 同规矩——按需增长、旧块不释放（图里烘焙了指针）。
    xq_pool: Vec<(usize, TensorId)>,
    xaux_pool: Vec<(usize, TensorId)>,
    /// 低秩链 fp16 张量核 GEMM 暂存池（`LOWRANK_GEMM=1`）：同一块内切
    /// 「4×[batch, C] fp16 的 x 降位副本」+「[batch, Σmid_pad] fp16 的 mid16」。
    /// 与 `x16_pool` 同规矩（捕获前建好、按需增长、旧块不释放）。
    lr16_pool: Vec<(usize, TensorId)>,
    /// ffn_value 稠密 GEMM 的 A 侧降位暂存（`r2` fp32 → `r2_16` fp16，batch×fh）。
    /// 与 `x16_pool` 同规矩（捕获前建好、按需增长、旧块不释放）。
    ffn16_pool: Vec<(usize, TensorId)>,
    /// split-K 的**部分和**暂存池（f32，元素数 = `nchain·ksplit·batch·m`）。
    /// 与 `x16_pool` 同规矩（捕获前建好、按需增长、旧块不释放）。
    ipart_pool: Vec<(usize, TensorId)>,
    /// `IMMA_MIN_BATCH` 的**每实例**覆盖（**仅供门禁**：SIMT 路径的门禁
    /// `gemv_variant_int8_matches_cpu` / `gemv_int8_plain_matches_cpu` 必须把阈值顶上去，
    /// 否则 `IMMA_MIN_BATCH = 1` 之后它们会被 IMMA（W8A8）接走，而两者容差口径不同）。
    /// 生产路径恒为 `None`；用实例字段而非全局环境变量是为了**避免测试间串扰**。
    imma_min_batch_override: Option<usize>,
}

/// 批量上传环形缓冲：slots 个 pinned 槽轮转，按槽事件同步（只等即将复用的槽），
/// 替代逐 tensor 的全流 cuStreamSynchronize（模型加载 1255 tensor 时 2500+ 次
/// 全流同步是加载耗时的主要来源）。
struct BulkRing {
    base: *mut u8,
    slot_bytes: usize,
    slots: usize,
    next: usize,
    events: Vec<CuEvent>,
}

impl Drop for BulkRing {
    fn drop(&mut self) {
        // 资源释放由 CudaBackend::upload_bulk_end 驱动（需访问 drv）；
        // 兜底：若直接 drop（未走 end），pinned/事件句柄随进程上下文回收。
        let _ = self;
    }
}

/// 全局 CUDA 上下文锁：串行化 `CudaBackend::new()` 的创建期（cuPrimaryCtxRetain/
/// cuCtxSetCurrent 并发竞争防护）。仅覆盖初始化阶段——创建完成后即释放，
/// 同进程多个后端实例可以共存（同线程顺序使用；见客户端常驻路由小模型场景）。
static CUDA_CTX_LOCK: Mutex<()> = Mutex::new(());

/// pinned 暂存区单行字节数（sampler 参数行 = 10 个 f32 = 40 字节；批量路径的
/// 宽行按 slot 逐个占一行，故设备侧 sampler 步长也必须等于 10 个 f32）。
const PINNED_ROW_BYTES: usize = 40;
/// pinned 暂存区行数（async sampler 路径的上限：selfloop n ≤ 行数）。
const PINNED_ROWS: usize = 8192;
/// pinned 上传 scratch 大小（≤ 此大小的同步上传走常驻 pinned scratch，避免
/// pageable 源在多线程并发 `cuMemcpyHtoDAsync` 时踩踏驱动内部共享 staging）。
const PINNED_UPLOAD_SCRATCH: usize = 64 * 1024 * 1024;

/// 消融探针：`ABLATE=csv` 列出的 kernel 名一律不启动（诊断用，不改变其它行为）。
///
/// **用法**（PowerShell，跑端到端基准时设）：
/// ```powershell
/// $env:ABLATE="gemv_variant_mb16_r4"; cargo run --release --example batch_decode_bench
/// $env:ABLATE="gemv_int8_rkv_stage1_batch,rwkv_sample_batch"; cargo run --release --example batch_decode_bench
/// ```
/// 读数 = 「基线步时间 − 消融后步时间」= 该 kernel 在关键路径上的贡献。
/// ⚠️ **名字必须与 kernel 缓存键完全一致**：`gemv_variant_mb` 系列按几何分名缓存
/// （`gemv_variant_mb16_r{ROWS}`，ROWS 默认 4），写错名字不会报错、只是**一处都不命中**
/// ——曾因此得到「消融后反而更慢」的假读数为 0 收益。
///
/// **为何需要**：`PROF_CUDA_KERNEL=1` 在每个 kernel 后 `cuEventSynchronize`，
/// 总量级被同步开销淹没（实测 batch 段 14400 次 launch 的"内核耗时"≈2.9s 正好等于
/// 同步总开销），无法定位真实热点。消融则保留完整流水与图重放，只用一个 kernel 的
/// 缺席反映它对关键路径的贡献。
///
/// **读数注意**：缺席的内核不写输出 ⇒ 下游读到陈旧数据，**部分内核会因此变快**
/// （如 relu² 全零时稀疏 FFN 几乎免费），故读数是「移除收益上界」，不是干净份额。
/// 需要单内核的干净成本请用 `examples/kernel_bench.rs`。
fn ablate_hit(drv: &CudaDriver, func: CuFunction) -> bool {
    use std::sync::OnceLock;
    static SET: OnceLock<Option<std::collections::HashSet<String>>> = OnceLock::new();
    let set = SET.get_or_init(|| {
        let v = std::env::var("ABLATE").ok()?;
        let names: std::collections::HashSet<String> = v
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if names.is_empty() {
            None
        } else {
            log::warn!("ABLATE 已启用（诊断模式，数值无意义）: {names:?}");
            Some(names)
        }
    });
    let Some(names) = set else { return false };
    // func → name：反查驱动内的注册表（≤ 数十项，且仅诊断时执行）。
    drv.prof
        .lock()
        .unwrap()
        .names
        .get(&(func as usize))
        .is_some_and(|name| names.contains(name.as_str()))
}

impl CudaBackend {
    /// 创建 CUDA 后端：初始化驱动、取首个设备、保留主上下文。
    pub fn new() -> R<Self> {
        // 加锁：串行化创建期（多线程并发 new 的 cuPrimaryCtxRetain/cuCtxSetCurrent 竞争防护）。
        // 守卫在函数返回时释放——后端存活期间不持有锁，多实例可共存。
        let _ctx_lock = CUDA_CTX_LOCK
            .lock()
            .map_err(|_| "CUDA context lock poisoned")?;
        let drv = driver().ok_or("CudaBackend: CUDA driver unavailable")?;
        let mut count: i32 = 0;
        cu_check!((drv.cu_init)(0), "cuInit");
        cu_check!((drv.cu_device_get_count)(&mut count), "cuDeviceGetCount");
        if count <= 0 {
            return Err("CudaBackend: no CUDA device".into());
        }
        let mut device: i32 = 0;
        cu_check!((drv.cu_device_get)(&mut device, 0), "cuDeviceGet");
        let mut ctx: *mut c_void = std::ptr::null_mut();
        cu_check!(
            (drv.cu_primary_ctx_retain)(&mut ctx, device),
            "cuPrimaryCtxRetain"
        );
        // 主上下文 retain 后须绑定到当前线程，否则 cuMemAlloc/cuLaunchKernel 报 INVALID_CONTEXT(201)。
        cu_check!((drv.cu_ctx_set_current)(ctx), "cuCtxSetCurrent");
        // 创建计算 stream（默认标志 0）。
        let mut stream: CuStream = std::ptr::null_mut();
        cu_check!((drv.cu_stream_create)(&mut stream, 0), "cuStreamCreate");
        // 创建剖析事件（PROF_CUDA_GPU=1 时用）。
        let mut ev_start: CuEvent = std::ptr::null_mut();
        let mut ev_end: CuEvent = std::ptr::null_mut();
        cu_check!((drv.cu_event_create)(&mut ev_start, 0), "cuEventCreate");
        cu_check!((drv.cu_event_create)(&mut ev_end, 0), "cuEventCreate");
        let mut gemm_ev_start: CuEvent = std::ptr::null_mut();
        let mut gemm_ev_end: CuEvent = std::ptr::null_mut();
        cu_check!(
            (drv.cu_event_create)(&mut gemm_ev_start, 0),
            "cuEventCreate"
        );
        cu_check!((drv.cu_event_create)(&mut gemm_ev_end, 0), "cuEventCreate");
        // pinned host 暂存区：前 PINNED_ROWS 行作 sampler 异步参数行，尾部作上传 scratch。
        let pinned_bytes = PINNED_ROWS * PINNED_ROW_BYTES + PINNED_UPLOAD_SCRATCH;
        let mut pinned: *mut c_void = std::ptr::null_mut();
        cu_check!(
            (drv.cu_mem_host_alloc)(&mut pinned, pinned_bytes, 0),
            "cuMemHostAlloc(pinned)"
        );
        let prof_gpu = std::env::var("PROF_CUDA_GPU").is_ok();
        let gemm_prof = std::env::var("PROF_GEMM").is_ok();
        // per-kernel profiling（诊断用）：需要可变的 drv 引用来启用。
        let prof_kernel = std::env::var("PROF_CUDA_KERNEL").is_ok();
        if prof_kernel {
            drv.enable_kernel_profiling();
        }
        if std::env::var("PROF_CUDA_COUNT").is_ok() {
            drv.prof.lock().unwrap().counting = true;
        }
        // 创建 cuBLAS 句柄（当前上下文已绑定）。不可用时回退自定义 gemm kernel。
        // CUBLAS=0 可临时禁用 cuBLAS（诊断/对拍用）。
        let cublas = if std::env::var("CUBLAS").as_deref() == Ok("0") {
            None
        } else {
            cublas_driver().and_then(|cd| {
                let mut h: CublasHandle = std::ptr::null_mut();
                if unsafe { (cd.cublas_create)(&mut h) } != CUBLAS_STATUS_SUCCESS || h.is_null() {
                    log::warn!("cublasCreate failed; falling back to custom gemm");
                    return None;
                }
                // 把 cuBLAS 绑定到计算 stream，保证与其它 kernel 的序关系。
                if unsafe { (cd.cublas_set_stream)(h, stream) } != CUBLAS_STATUS_SUCCESS {
                    log::warn!("cublasSetStream failed; falling back to custom gemm");
                    unsafe { (cd.cublas_destroy_v2)(h) };
                    return None;
                }
                Some(h)
            })
        };
        Ok(Self {
            drv,
            ctx,
            device,
            tensors: HashMap::new(),
            lens: HashMap::new(),
            next_id: 0,
            kernels: HashMap::new(),
            stream,
            prof_ev_start: ev_start,
            prof_ev_end: ev_end,
            prof_gpu,
            prof_kernel,
            gemm_prof_ev_start: gemm_ev_start,
            gemm_prof_ev_end: gemm_ev_end,
            gemm_prof,
            gemm_times: HashMap::new(),
            graph_capturing: false,
            prefill_graphs: HashMap::new(),
            prefill_t: 0,
            selfloop_graphs: HashMap::new(),
            selfloop_disabled: std::collections::HashSet::new(),
            selfloop_key: 0,
            cublas,
            pinned,
            pinned_rows: PINNED_ROWS,
            bulk: std::cell::RefCell::new(None),
            foreign: std::collections::HashSet::new(),
            x16_pool: Vec::new(),
            xq_pool: Vec::new(),
            xaux_pool: Vec::new(),
            lr16_pool: Vec::new(),
            ffn16_pool: Vec::new(),
            ipart_pool: Vec::new(),
            imma_min_batch_override: None,
        })
    }

    /// 见 `IMMA_MIN_BATCH`；实例字段（`imma_min_batch_override`）优先，供门禁顶高阈值。
    fn imma_min_batch(&self) -> usize {
        self.imma_min_batch_override.unwrap_or_else(imma_min_batch)
    }

    /// 编译并缓存 kernel（同名复用已编译模块）。`src` 为 CUDA C 源码，`entry` 为 __global__ 函数名。
    fn kernel(&mut self, key: &str, src: &str, entry: &str) -> R<CuFunction> {
        if let Some((_, f)) = self.kernels.get(key) {
            return Ok(*f);
        }
        log::info!("compiling kernel {key} ({entry})");
        let ptx = self.drv.nvrtc_to_ptx(src, key, self.device)?;
        let module = self.drv.load_module(&ptx)?;
        let func = self.drv.get_function(module, entry)?;
        self.kernels.insert(key.to_string(), (module, func));
        self.drv.register_kernel_name(func, key);
        Ok(func)
    }

    fn alloc(&self, bytes: usize) -> R<u64> {
        let mut dptr: u64 = 0;
        cu_check!((self.drv.cu_mem_alloc_v2)(&mut dptr, bytes), "cuMemAlloc");
        Ok(dptr)
    }

    /// 部分上传：只拷贝前置 `n` 个元素（每元素 4 字节），device 其余部分不动。
    /// 对齐 Vulkan 的 `host.copy_from(data, 0)` 语义（允许 data.len() <= 张量 len）。
    ///
    /// 上传源必须为 pinned：pageable 源的 cuMemcpyHtoDAsync 走驱动内部**共享
    /// staging 缓冲**，多线程并发上传互相践踏；同步 cuMemcpyHtoD 则与非阻塞
    /// stream 无顺序关系（kernel 可能先于 DMA 启动）。≤ scratch 的上传用常驻
    /// pinned scratch；更大的上传临时分配 pinned（一次性权重加载）。拷贝以流序
    /// 异步挂到本 stream，前后同步保证：先序于后续 kernel、完成后才返回
    ///（pageable 源可释放、scratch 可复用）。
    fn memcpy_htod_n(&self, dptr: u64, data: &[u8], n: usize) -> R<()> {
        let bytes = n * 4;
        assert!(bytes <= data.len(), "memcpy_htod_n: n 超出 data");
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(htod)"
        );
        self.htod_pinned(dptr, data, bytes)
    }

    /// 部分上传（fp16）：只拷贝前置 `n` 个元素（每元素 2 字节），device 其余部分不动。
    /// 语义同 memcpy_htod_n（pinned 源 + 流序异步 + 前后同步）。
    fn memcpy_htod_n2(&self, dptr: u64, data: &[u8], n: usize) -> R<()> {
        let bytes = n * 2;
        assert!(bytes <= data.len(), "memcpy_htod_n2: n 超出 data");
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(htod2)"
        );
        self.htod_pinned(dptr, data, bytes)
    }

    /// pinned 源流序上传 + 完成同步（调用前须已排空本 stream）。
    /// 批量上传开始：启用环形 pinned 暂存（8 槽 × 32MB）+ 按槽事件同步。
    fn htod_pinned(&self, dptr: u64, data: &[u8], bytes: usize) -> R<()> {
        // 批量加载模式：环形槽 + 按槽事件同步（只等即将复用的槽的上一次拷贝）。
        {
            let mut ring_ref = self.bulk.borrow_mut();
            if let Some(r) = ring_ref.as_mut()
                && bytes <= r.slot_bytes
            {
                let slot = r.next;
                let off = slot * r.slot_bytes;
                // 复用前等该槽上一次拷贝完成（流内 FIFO，等它即等全部更早者）
                cu_check!(
                    (self.drv.cu_event_synchronize)(r.events[slot]),
                    "cuEventSynchronize(bulk slot)"
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), r.base.add(off), bytes);
                }
                let src = unsafe { (r.base as *const u8).add(off) };
                cu_check!(
                    (self.drv.cu_memcpy_htod_async)(dptr, src as *const c_void, bytes, self.stream),
                    "cuMemcpyHtoDAsync(bulk ring)"
                );
                cu_check!(
                    (self.drv.cu_event_record)(r.events[slot], self.stream),
                    "cuEventRecord(bulk)"
                );
                r.next = (slot + 1) % r.slots;
                return Ok(());
            }
            // 超大 tensor 落到下方 tmp-pinned 同步路径（数量极少）
        }
        let scratch_off = PINNED_ROWS * PINNED_ROW_BYTES;
        if bytes <= PINNED_UPLOAD_SCRATCH {
            // 常驻 scratch：host 拷入 → 流序异步 DMA → 同步完成（scratch 可复用）。
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (self.pinned as *mut u8).add(scratch_off),
                    bytes,
                );
            }
            let src = unsafe { (self.pinned as *const u8).add(scratch_off) };
            cu_check!(
                (self.drv.cu_memcpy_htod_async)(dptr, src as *const c_void, bytes, self.stream),
                "cuMemcpyHtoDAsync(scratch)"
            );
        } else {
            // 大上传（权重加载）：临时 pinned 分配，完成后释放。
            let mut tmp: *mut c_void = std::ptr::null_mut();
            cu_check!(
                (self.drv.cu_mem_host_alloc)(&mut tmp, bytes, 0),
                "cuMemHostAlloc(htod tmp)"
            );
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), tmp as *mut u8, bytes);
            }
            cu_check!(
                (self.drv.cu_memcpy_htod_async)(dptr, tmp as *const c_void, bytes, self.stream),
                "cuMemcpyHtoDAsync(tmp pinned)"
            );
            cu_check!(
                (self.drv.cu_stream_synchronize)(self.stream),
                "cuStreamSynchronize(htod tmp-wait)"
            );
            unsafe {
                (self.drv.cu_mem_free_host)(tmp);
            }
            return Ok(());
        }
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(htod-wait)"
        );
        Ok(())
    }

    fn memcpy_dtoh(&self, dptr: u64, out: &mut [u8]) -> R<()> {
        // 同步 stream：确保 kernel 已把结果写入 device 内存，再同步拷回 host。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(dtoh)"
        );
        cu_check!(
            (self.drv.cu_memcpy_dtoh_v2)(out.as_mut_ptr() as *mut c_void, dptr, out.len()),
            "cuMemcpyDtoH"
        );
        Ok(())
    }

    fn get(&self, t: TensorId, op: &str) -> R<CudaTensor> {
        self.tensors
            .get(&t)
            .cloned()
            .ok_or(format!("{op}: unknown tensor {t:?}").into())
    }

    /// 取 f32 张量设备指针。
    fn f32_ptr(&self, t: TensorId, op: &str) -> R<u64> {
        match self.get(t, op)? {
            CudaTensor::F32 { dptr, .. } => Ok(dptr),
            _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
        }
    }

    /// 取 f16 张量设备指针。
    fn f16_ptr(&self, t: TensorId, op: &str) -> R<u64> {
        match self.get(t, op)? {
            CudaTensor::F16 { dptr, .. } => Ok(dptr),
            _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
        }
    }

    /// 取 u32 张量设备指针。
    fn u32_ptr(&self, t: TensorId, op: &str) -> R<u64> {
        match self.get(t, op)? {
            CudaTensor::U32 { dptr, .. } => Ok(dptr),
            _ => Err(format!("{op}: tensor {t:?} must be u32").into()),
        }
    }

    /// 统一调度 gemv_variant kernel（wtype/op 见 GEMV_VARIANT_SRC）。
    /// 传入已解析的设备指针；未用指针传 0。
    /// batch>1 时走 `gemv_variant_mb`（权重复用版：每 block 读一次权重算
    /// BGRP 个 slot，带宽 ≈ 1/ceil(B/BGRP)——信天翁 rows 模型）；batch==1 走原版。
    #[allow(clippy::too_many_arguments)]
    fn gemv_variant_dispatch(
        &mut self,
        af16: u64,
        aidx: u64,
        alut: u64,
        asz: u64,
        xd: u64,
        gd: u64,
        yd: u64,
        m: usize,
        k: usize,
        batch: usize,
        wtype: i32,
        op: i32,
    ) -> R<()> {
        // batch 线性层第三代（`imma_gemm_batch`）：int8 张量核（W8A8）。
        // 开关 `GEMV_IMMA`（**默认开**，2026-09-22 翻转；`=0` 回退 int8 SIMT `gemv_variant_mb16`）。
        // 路径：量化激活（`quant_x_i8`）→ IMMA GEMM。
        // 翻默认依据：§3.3c 贪心 256/256 逐位一致、top50 分叉率 6.44%、端到端 1.44×。
        if batch >= self.imma_min_batch()
            && wtype == 2
            && k.is_multiple_of(QUANT_X_I8_GROUP)
            && env_on("GEMV_IMMA")
        {
            return self.imma_gemm_dispatch(aidx, asz, xd, gd, yd, m, k, batch, op);
        }
        // batch 线性层第二代（`gemv_variant_mb16`）：int8 权重 + fp16 激活，权重读一遍。
        // 开关 `GEMV_F16X`（**默认开**，2026-09-21 已 A/B 通过）；`GEMV_F16X=0` 回退旧路径。
        // 隔离计时（B=8）：relu2 0.2514→0.1625、plain 1.4913→0.9612、mul_add 0.1035→0.0583；
        // 端到端（batch_decode_bench B=8）244.0→287.3 tok/s。测试全绿。
        // 仅 int8 权重（wtype==2）走此路径——fp16 权重另有通路且不走 batch。
        let use_mb16 =
            batch > 1 && wtype == 2 && std::env::var("GEMV_F16X").map(|v| v != "0").unwrap_or(true);
        let (func, grid, block) = if batch > 1 {
            if use_mb16 {
                // 几何：block = 128 线程，全体线程按 k 切分（累加深度不变量，见内核注释），
                // 覆盖 ROWS 行 × **全部 batch 槽**（batch 分块在块内循环，见内核注释）。
                // **每 block 覆盖的行数决定 x 的 L2 重读次数**（x 流量 = (M/ROWS) × batch × K × 2B）。
                // ROWS 以 `#define` 注入源码、并按值分名缓存模块 ⇒ 可不重编译做几何 A/B。
                const GEMV_MB16_THREADS: usize = 128;
                let rows: usize = std::env::var("GEMV_MB16_ROWS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|v| (1..=8).contains(v))
                    .unwrap_or(GEMV_MB16_ROWS_DEFAULT);
                let src = format!("#define MB16_ROWS {rows}\n{GEMV_VARIANT_MB16_SRC}");
                let key = format!("gemv_variant_mb16_r{rows}");
                let f = self.kernel(&key, &src, "gemv_variant_mb16")?;
                // grid.y 恒为 1：块内循环已覆盖全部 batch 槽，权重每块只从 DRAM 读一遍。
                (
                    f,
                    (m.div_ceil(rows) as u32, 1u32, 1u32),
                    (GEMV_MB16_THREADS as u32, 1u32, 1u32),
                )
            } else {
                // 与 kernel 内 ROWS/BGRP 保持一致（ROWS=4、BGRP=4）。
                // 注：block 固定 128 线程——kvi4 = k/4（2560→640、10240→2560）恒为 128
                // 的倍数，行程数整除、线程间工作量均衡。Phase 1a 曾试 160 线程配宽载入，
                // 无改善且更慢，见 kernel 内 int8 段的负结果留档。
                const GEMV_MB_BGRP: usize = 4; // 与 kernel 内 BGRP 同步
                const GEMV_MB_ROWS: usize = 4;
                const GEMV_MB_THREADS: usize = 128;
                let f = self.kernel("gemv_variant_mb", GEMV_VARIANT_MB_SRC, "gemv_variant_mb")?;
                (
                    f,
                    (
                        (m / GEMV_MB_ROWS) as u32,
                        batch.div_ceil(GEMV_MB_BGRP) as u32,
                        1u32,
                    ),
                    (GEMV_MB_THREADS as u32, 1u32, 1u32),
                )
            }
        } else {
            let func = self.kernel("gemv_variant", GEMV_VARIANT_SRC, "gemv_variant")?;
            (func, ((m / 4) as u32, 1u32, 1u32), (128u32, 1u32, 1u32))
        };
        let m_i = m as i32;
        // mb16 路径：先把 fp32 激活（必要时乘门控）降位到常驻 fp16 暂存，再跑 GEMM。
        // 旧路径的 x 是 fp32 且被 m/ROWS 个 block 重复以 float4 读，是 L2 流量最大项。
        let mut xd_eff = xd;
        if use_mb16 {
            let n = batch * k;
            let x16 = self.x16_scratch(n)?;
            let x16d = self.f16_ptr(x16, "gemv_variant_mb16")?;
            let cast = self.kernel("cast_mul_f16", CAST_MUL_F16_SRC, "cast_mul_f16")?;
            let n_i = n as i32;
            let xsrc = xd;
            let gsrc = gd;
            let cparams = [
                &xsrc as *const u64 as *mut c_void,
                &gsrc as *const u64 as *mut c_void,
                &x16d as *const u64 as *mut c_void,
                &n_i as *const i32 as *mut c_void,
            ];
            let cthreads = 256u32;
            let cgrid = ((n as u32).div_ceil(cthreads), 1u32, 1u32);
            self.drv
                .launch_smem(self.stream, cast, cgrid, (cthreads, 1, 1), &cparams, 0)?;
            xd_eff = x16d;
        }
        let k_i = k as i32;
        let b_i = batch as i32;
        // 各分支均不使用动态共享内存（int8 预解量化曾尝试但同步开销反超，已回退）。
        let smem = 0usize;
        if use_mb16 {
            // mb16 内核参数表与旧版不同（无 fp16 权重/门控/lut/wtype，op 语义亦简化）。
            let params = [
                &aidx as *const u64 as *mut c_void,
                &asz as *const u64 as *mut c_void,
                &xd_eff as *const u64 as *mut c_void,
                &yd as *const u64 as *mut c_void,
                &m_i as *const i32 as *mut c_void,
                &k_i as *const i32 as *mut c_void,
                &b_i as *const i32 as *mut c_void,
                &op as *const i32 as *mut c_void,
            ];
            return self
                .drv
                .launch_smem(self.stream, func, grid, block, &params, smem);
        }
        let params = [
            &af16 as *const u64 as *mut c_void,
            &aidx as *const u64 as *mut c_void,
            &alut as *const u64 as *mut c_void,
            &asz as *const u64 as *mut c_void,
            &xd_eff as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
            &wtype as *const i32 as *mut c_void,
            &op as *const i32 as *mut c_void,
        ];
        self.drv
            .launch_smem(self.stream, func, grid, block, &params, smem)
    }

    /// 取（或增长）batch 线性层 fp16 激活暂存。旧块不释放（图内可能仍引用）。
    fn x16_scratch(&mut self, n: usize) -> R<TensorId> {
        if let Some((_, t)) = self.x16_pool.iter().find(|(cap, _)| *cap >= n) {
            return Ok(*t);
        }
        if self.graph_capturing {
            return Err(format!(
                "x16_scratch: 捕获期需要扩容到 {n} 元素（图内禁止分配）；\
                 请调大 X16_SCRATCH_INIT_ELEMS"
            )
            .into());
        }
        let cap = n.next_power_of_two();
        let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::F16)?;
        self.x16_pool.push((cap, id));
        Ok(id)
    }

    /// 取一块容量 ≥ `n` 元素的 fp16 暂存**基址**（供多数组切片复用；同一 stream 上
    /// 各数组「降位紧跟消费」故可安全共享同一块）。
    fn x16_base(&mut self, n: usize) -> R<u64> {
        let t = self.x16_scratch(n)?;
        self.f16_ptr(t, "x16_base")
    }

    /// 启动激活量化内核（W8A8 的 A 侧，见 `QUANT_X_I8_SRC`）。
    /// `xq` 为 u32 张量 `[rows, k/4]`、`xaux` 为 f32 张量 `[rows, G]×4`（float4）。
    #[allow(clippy::too_many_arguments)]
    fn quant_x_i8(
        &mut self,
        xd: u64,
        gd: u64,
        xq: TensorId,
        xaux: TensorId,
        rows: usize,
        k: usize,
    ) -> R<()> {
        let xqd = self.u32_ptr(xq, "quant_x_i8")?;
        let xad = self.f32_ptr(xaux, "quant_x_i8")?;
        self.quant_x_i8_at(xd, gd, xqd, xad, rows, k)
    }

    /// 同上，但 `xq`/`xaux` 直接给裸指针（多链合并时按偏移切同一块暂存）。
    fn quant_x_i8_at(
        &mut self,
        xd: u64,
        gd: u64,
        xqd: u64,
        xad: u64,
        rows: usize,
        k: usize,
    ) -> R<()> {
        assert!(
            k.is_multiple_of(QUANT_X_I8_GROUP),
            "quant_x_i8: k={k} 必须是 {QUANT_X_I8_GROUP} 的倍数"
        );
        let func = self.kernel("quant_x_i8", QUANT_X_I8_SRC, "quant_x_i8")?;
        let rows_i = rows as i32;
        let k_i = k as i32;
        let grid = (
            (k / QUANT_X_I8_GROUP) as u32,
            (rows as u32).div_ceil(8),
            1u32,
        );
        let params = [
            &xd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &xqd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &rows_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
        ];
        self.drv
            .launch_smem(self.stream, func, grid, (256, 1, 1), &params, 0)
    }

    /// 池内查容量足够的块（不可变借用，故与 `create_tensor` 的 `&mut self` 不冲突）。
    fn scratch_take(pool: &[(usize, TensorId)], n: usize) -> Option<TensorId> {
        pool.iter().find(|(cap, _)| *cap >= n).map(|(_, t)| *t)
    }

    /// 取（或增长）int8 激活暂存 `xq`（`n` 为 u32 元素数 = 字节数/4）。
    fn xq_scratch(&mut self, n: usize) -> R<TensorId> {
        if let Some(t) = Self::scratch_take(&self.xq_pool, n) {
            return Ok(t);
        }
        if self.graph_capturing {
            return Err(format!("xq_scratch: 捕获期需要扩容到 {n} 元素（图内禁止分配）").into());
        }
        let cap = n.next_power_of_two();
        let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::U32)?;
        self.xq_pool.push((cap, id));
        Ok(id)
    }

    /// 取（或增长）激活统计暂存 `xaux`（`n` 为 float4 个数）。
    fn xaux_scratch(&mut self, n: usize) -> R<TensorId> {
        if let Some(t) = Self::scratch_take(&self.xaux_pool, n) {
            return Ok(t);
        }
        if self.graph_capturing {
            return Err(format!("xaux_scratch: 捕获期需要扩容到 {n} 组（图内禁止分配）").into());
        }
        let cap = n.next_power_of_two();
        let id = <Self as ComputeBackend>::create_tensor(self, cap * 4, TensorDtype::F32)?;
        self.xaux_pool.push((cap, id));
        Ok(id)
    }

    /// 取（或增长）split-K 的部分和暂存（`n` 为 f32 元素数）。
    fn ipart_scratch(&mut self, n: usize) -> R<TensorId> {
        if let Some(t) = Self::scratch_take(&self.ipart_pool, n) {
            return Ok(t);
        }
        if self.graph_capturing {
            return Err(format!("ipart_scratch: 捕获期需要扩容到 {n} 元素（图内禁止分配）").into());
        }
        let cap = n.next_power_of_two();
        let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::F32)?;
        self.ipart_pool.push((cap, id));
        Ok(id)
    }

    /// 取（或增长）低秩链 GEMM 的暂存**基址**（`n` 为 fp16 元素数）。
    /// 同一块内切 `[0, 4n)` = 4×`[batch, C]` 的 x 降位副本、`[4n, 4n + batch·Σmid_pad)` = mid16。
    fn lr16_base(&mut self, n: usize) -> R<u64> {
        if let Some(t) = Self::scratch_take(&self.lr16_pool, n) {
            return self.f16_ptr(t, "lr16_base");
        }
        if self.graph_capturing {
            return Err(format!(
                "lr16_base: 捕获期需要扩容到 {n} 元素（图内禁止分配）；\
                 请调大 LR16_SCRATCH_INIT_ELEMS"
            )
            .into());
        }
        let cap = n.next_power_of_two();
        let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::F16)?;
        self.lr16_pool.push((cap, id));
        self.f16_ptr(id, "lr16_base")
    }

    /// 取（或增长）`r2_16` 暂存（`n` 为 fp16 元素数）。
    fn ffn16_scratch(&mut self, n: usize) -> R<TensorId> {
        if let Some(t) = Self::scratch_take(&self.ffn16_pool, n) {
            return Ok(t);
        }
        if self.graph_capturing {
            return Err(format!(
                "ffn16_scratch: 捕获期需要扩容到 {n} 元素（图内禁止分配）；\
                 请调大 FFN16_SCRATCH_INIT_ELEMS"
            )
            .into());
        }
        let cap = n.next_power_of_two();
        let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::F16)?;
        self.ffn16_pool.push((cap, id));
        Ok(id)
    }

    /// 取「fp32 或 fp16 皆可」的裸指针 —— WKV 状态在 `WKV_STATE_F16` 下是 fp16，
    /// 两种 dtype 的设备指针取值方式相同（裸 u64），差别只在核内 `stype`。
    fn any_ptr(&self, t: TensorId, op: &str) -> R<u64> {
        match self.get(t, op)? {
            CudaTensor::F32 { dptr, .. } | CudaTensor::F16 { dptr, .. } => Ok(dptr),
            CudaTensor::U32 { .. } => Err(format!("{op}: u32 张量不能当浮点指针用").into()),
        }
    }

    /// WKV 状态内核的 `(key, src)` 变体选择：状态为 fp16 时注入 `#define WKV_S16 1`。
    /// ⚠️ **必须换 key** —— `kernel()` 的缓存只按 key 索引、不校验变体（实施记录 §3d.3）。
    fn dplr_variant(&self, base: &str, src: &str, s: TensorId, op: &str) -> R<(String, String)> {
        let is16 = matches!(self.get(s, op)?, CudaTensor::F16 { .. });
        Ok(if is16 {
            (format!("{base}_s16"), format!("#define WKV_S16 1\n{src}"))
        } else {
            (base.to_string(), src.to_string())
        })
    }

    /// ★ 2026-09-23：`fuse_ka_dplr_norm` 的块内 warp 数。
    ///
    /// 该内核 grid = `(H, batch)`，B=1 时**只有 H=40 个块**（68 个 SM 空转 28 个），
    /// 且每块内部是「KAW 个 warp 各自串行跑 n/KAW 行」——串行链完全暴露。
    /// 实测 B=1 有效带宽仅 **~40 GB/s**（B=256 时同内核 450 GB/s 贴 roofline）
    /// ⇒ 与带宽无关，是**串行链长 + 块数**。
    /// 故小 batch 抬 KAW（块数不足就靠块内并发补），大 batch 保持 4（块已经够多，
    /// 每块 4 warp 时寄存器/占用率最优）。`KAW` 进 kernel key ⇒ 各自编译、可 A/B。
    fn dplr_kaw(batch: usize, n: usize) -> usize {
        // 相位 0 的树归约要求 `KAW·32 ≥ 2n`（否则 `sq[]` 填不满、和会偏小）。
        let min_kaw = (2 * n).div_ceil(32).max(4);
        let want = std::env::var("KAW")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| if batch <= 8 { 16 } else { 4 });
        want.clamp(min_kaw, 32)
    }

    /// W8A8 张量核批量线性层：激活量化 → `imma_gemm_batch`。
    /// 权重仍是 int8 常驻（`aidx`/`asz`），只有 A 侧被量化到 int8。
    #[allow(clippy::too_many_arguments)]
    fn imma_gemm_dispatch(
        &mut self,
        aidx: u64,
        asz: u64,
        xd: u64,
        gd: u64,
        yd: u64,
        m: usize,
        k: usize,
        batch: usize,
        op: i32,
    ) -> R<()> {
        // ★ 2026-09-22：**按 batch 选 BM**（小 batch 时 BM=64 把 grid 抬一倍）。
        // batch ≤ 64 时 `grid.y = batch/BN = 1` ⇒ 整个 launch 只有 `M/BM` 个块：
        // BM=128 时 r/k/v/o 各只有 **20 个块**（68 个 SM 半数空转），BM=64 抬到 40。
        // 实测（同会话交错）：B=16 **546.1 → 588.8** · B=32 **1030.1 → 1072.3** ·
        // B=64 **1801.9 → 1885.7**；而 B=128 **2531.8 → 2396.6（BM=64 变差）**、
        // B=256 同样 BM=128 更优 ⇒ **阈值取 64**。
        let bm = if batch <= 64 {
            env_tile("IMMA_BM_SMALL", 64)
        } else {
            env_tile("IMMA_BM", IMMA_BM_DEFAULT)
        };
        // ★ 同源：小 batch 时 `BN ≈ batch`，且块数不足 68 时按 2 的幂缩 BN（见 `pick_bn`）。
        let bn = if batch <= 64 {
            env_tile("IMMA_BN_SMALL", pick_bn(m, bm, batch))
        } else {
            env_tile("IMMA_BN", IMMA_BN_DEFAULT)
        };
        self.imma_gemm_tiled(aidx, asz, xd, gd, yd, m, k, batch, op, bm, bn)
    }

    /// 同上，但 tile 由调用方指定（形状差异大时用；如 ffn_value 的 m=2560 在 BM=128 下
    /// 只有 80 个 block，喂不满 68 个 SM ⇒ 需要 BM=64 把 grid 抬到 160）。
    #[allow(clippy::too_many_arguments)]
    fn imma_gemm_tiled(
        &mut self,
        aidx: u64,
        asz: u64,
        xd: u64,
        gd: u64,
        yd: u64,
        m: usize,
        k: usize,
        batch: usize,
        op: i32,
        bm: usize,
        bn: usize,
    ) -> R<()> {
        self.imma_gemm_tiled_ks(aidx, asz, xd, gd, yd, m, k, batch, op, bm, bn, None)
    }

    /// `imma_gemm_tiled` 的本体；`ks_force` 供测试门禁强制指定 split-K 分块数。
    #[allow(clippy::too_many_arguments)]
    fn imma_gemm_tiled_ks(
        &mut self,
        aidx: u64,
        asz: u64,
        xd: u64,
        gd: u64,
        yd: u64,
        m: usize,
        k: usize,
        batch: usize,
        op: i32,
        bm: usize,
        bn: usize,
        ks_force: Option<usize>,
    ) -> R<()> {
        assert!(
            k.is_multiple_of(QUANT_X_I8_GROUP),
            "imma_gemm_batch: k 必须是 128 的倍数"
        );
        let xq = self.xq_scratch(batch * (k / 4))?;
        let xaux = self.xaux_scratch(batch * (k / QUANT_X_I8_GROUP))?;
        self.quant_x_i8(xd, gd, xq, xaux, batch, k)?;
        let xqd = self.u32_ptr(xq, "imma_gemm_batch")?;
        let xad = self.f32_ptr(xaux, "imma_gemm_batch")?;
        let lb2 = env_tile("IMMA_LB2", 2);
        // ★ `IMMA_GRID_SWAP`（**默认开**，2026-09-22 A/B：3 轮交错 2445.2 → 2546.5 tok/s，**+4.1%**，
        // token 指纹逐位一致）。显式 `=0` 回退原维序。
        let swap = if std::env::var("IMMA_GRID_SWAP")
            .map(|v| v != "0")
            .unwrap_or(true)
        {
            1
        } else {
            0
        };
        // ★ `IMMA_PIPE`：k-tile 软流水（**已实测 −36%，默认关，留作反例开关**）。
        let pipe = if std::env::var("IMMA_PIPE")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            1
        } else {
            0
        };
        // ★ `IMMA_DB`：**smem 双缓冲**（默认开）。只在 smem 预算够时生效：
        // 2×((BM+BN)·WS + BN·16) ≤ 48KiB ⇒ BM=64/BN=64 时 38.9KiB ✓；BM=128 时 56.3KiB ✗。
        // 小 batch 走 BM=64，正好吃得到；大 batch（BM=128）自动退回单缓冲——那里本来就有
        // 2~3 个块/SM 互相掩盖载入延迟，不需要双缓冲。
        // ⚠️ **反例留档：smem 双缓冲（`IM_NBUF=2`）实测完全无效，默认关。**
        // 同会话 A/B：B=16 **585.0 → 585.8** · B=32 **1080.5 → 1071.9** · B=64 1878.2 → 1877.4 ·
        // B=256 3126.8 → 3129.6（全在噪声内，token 逐位一致）。
        // ⇒ **该内核在小 batch 也**不是**「载入延迟」受限**（否则双缓冲必有效）。
        // 与 §3l.1 的软流水结论一致：`IM_PIPE`/`IM_NBUF` 两条「隐藏延迟」的路都走不通。
        // 真正剩下的是「每块摊到的 k 长度太长」（20 个 k-tile 串行），只有**增加块数**
        // （split-K / 合并多条链的 launch）才能解 —— 见 §3m 的做法。
        let db_want = std::env::var("IMMA_DB").map(|v| v != "0").unwrap_or(false);
        let smem_1buf = (bm + bn) * (128 * 2 + 16) + bn * 16;
        let nbuf = if db_want && smem_1buf * 2 <= 48 * 1024 {
            2
        } else {
            1
        };
        // ★ 2026-09-23：**旧实验的 `lb2=1` 强制是混淆变量**。原判据是「双缓冲吃掉 smem 后
        // 1 个块/SM 都勉强」，但那只在 BM=128/BN=64（38.9KiB×1）成立；小 batch 走
        // BM=64/BN=8 时双缓冲只要 **20.5KiB**，`lb2=2`（2 块/SM）完全放得下。
        // 旧版无条件降到 1 ⇒ `__launch_bounds__(256,1)` 允许 255 个寄存器、并发块数腰斩，
        // 于是「双缓冲 −15%」的结论把并发损失算在了双缓冲头上。
        // 新判据：只有 2×smem 真的超过 24KiB 才降 lb2。
        let lb2 = if nbuf == 2 && smem_1buf * 2 > 24 * 1024 {
            1
        } else {
            lb2
        };
        // ★ split-K：块数不够时把 K 切成 `ks` 段（总流量不变、块数 ×ks），
        // 各段写部分和，再由 `imma_gemm_reduce` 按固定顺序求和。
        let ncol = batch.div_ceil(bn);
        let blocks = (m.div_ceil(bm)) * ncol;
        let ks = imma_ksplit_with(blocks, ncol, k, 1, ks_force);
        let (yd_eff, ipart_ptr) = if ks > 1 {
            let t = self.ipart_scratch(batch * m * ks)?;
            (self.f32_ptr(t, "imma_gemm_partial")?, Some(t))
        } else {
            (yd, None)
        };
        // ★ `IMMA_LDSEP`：载入与落 smem 分离（默认关，实验开关；见 `IM_STAGE_LDSEP`）。
        let ldsep = std::env::var("IMMA_LDSEP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(1);
        // ★ `IMMA_VECLD`：16B/线程向量化 staging（默认关，实验开关；见 `IM_STAGE_VEC`）。
        let vecld = std::env::var("IMMA_VECLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(1);
        let src = format!(
            "#define IM_BM {bm}\n#define IM_BN {bn}\n#define IM_LB2 {lb2}\n#define IM_SWAP {swap}\n\
             #define IM_PIPE {pipe}\n#define IM_NBUF {nbuf}\n#define IM_KSPLIT {ks}\n\
             #define IM_LDSEP {ldsep}\n#define IM_VECLD {vecld}\n{IMMA_GEMM_SRC}"
        );
        // ★ key 里带上 `m`/`k`：源码与几何无关，但**剖析器按 key 名分桶**
        // （`PROF_CUDA_KERNEL`）——不带形状名时「rkv / ffn_key / head」全挤在一个桶里，
        // 小 batch 的账本无法分解。形状只有个位数，多编译几次可忽略。
        let key = format!(
            "imma_gemm_batch_b{bm}n{bn}l{lb2}s{swap}p{pipe}d{nbuf}k{ks}e{ldsep}v{vecld}_m{m}k{k}"
        );
        let func = self.kernel(&key, &src, "imma_gemm_batch")?;
        let m_i = m as i32;
        let k_i = k as i32;
        let b_i = batch as i32;
        let params = [
            &aidx as *const u64 as *mut c_void,
            &aidx as *const u64 as *mut c_void,
            &aidx as *const u64 as *mut c_void,
            &asz as *const u64 as *mut c_void,
            &asz as *const u64 as *mut c_void,
            &asz as *const u64 as *mut c_void,
            &xqd as *const u64 as *mut c_void,
            &xqd as *const u64 as *mut c_void,
            &xqd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &yd_eff as *const u64 as *mut c_void,
            &yd_eff as *const u64 as *mut c_void,
            &yd_eff as *const u64 as *mut c_void,
            &op as *const i32 as *mut c_void,
            &op as *const i32 as *mut c_void,
            &op as *const i32 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        let grid = if swap == 1 {
            (batch.div_ceil(bn) as u32, m.div_ceil(bm) as u32, ks as u32)
        } else {
            (m.div_ceil(bm) as u32, batch.div_ceil(bn) as u32, ks as u32)
        };
        self.drv
            .launch_smem(self.stream, func, grid, (256, 1, 1), &params, 0)?;
        if let Some(t) = ipart_ptr {
            let base = self.f32_ptr(t, "imma_gemm_partial")?;
            self.imma_gemm_reduce(base, yd, ks, batch, m, op)?;
        }
        Ok(())
    }

    /// split-K 的确定性归约：`y = op(Σ_{ksp} partial[ksp])`（ksp 升序相加）。
    fn imma_gemm_reduce(
        &mut self,
        partial: u64,
        y: u64,
        ks: usize,
        batch: usize,
        m: usize,
        op: i32,
    ) -> R<()> {
        let func = self.kernel("imma_gemm_reduce", IMMA_REDUCE_SRC, "imma_gemm_reduce")?;
        let (ks_i, b_i, m_i, op_i) = (ks as i32, batch as i32, m as i32, op);
        let params = [
            &partial as *const u64 as *mut c_void,
            &y as *const u64 as *mut c_void,
            &ks_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &op_i as *const i32 as *mut c_void,
        ];
        let total = batch * m;
        let grid = ((total.div_ceil(256)).min(4096) as u32, 1u32, 1u32);
        self.drv
            .launch_smem(self.stream, func, grid, (256, 1, 1), &params, 0)
    }

    /// **3 条同形状链合并成一次 launch** 的 IMMA（`GEMV_IMMA` 下 r/k/v 用）。
    /// grid = `(M/BM, batch/BN, 3)`：小 batch 时 `batch/BN = 1`，原 3 次串行 launch 各只有
    /// `M/BM` 个块（B=16/BM=64 ⇒ 40 块，68 个 SM 半数空转）；合并后 120 块一次铺开。
    /// 三条链共用一块 `xq`/`xaux` 暂存（按链偏移切），故只需 3 次量化 launch 的**顺序**不变。
    #[allow(clippy::too_many_arguments)]
    fn imma_gemm_dispatch_z3(
        &mut self,
        aidx: [u64; 3],
        asz: [u64; 3],
        xd: [u64; 3],
        yd: [u64; 3],
        op: [i32; 3],
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        assert!(
            k.is_multiple_of(QUANT_X_I8_GROUP),
            "imma_gemm_batch: k 必须是 128 的倍数"
        );
        let kq = k / 4;
        let kg = k / QUANT_X_I8_GROUP;
        let xq = self.xq_scratch(3 * batch * kq)?;
        let xaux = self.xaux_scratch(3 * batch * kg)?;
        let xqd0 = self.u32_ptr(xq, "imma_gemm_dispatch_z3")?;
        let xad0 = self.f32_ptr(xaux, "imma_gemm_dispatch_z3")?;
        let mut xqd = [0u64; 3];
        let mut xad = [0u64; 3];
        for i in 0..3 {
            xqd[i] = xqd0 + (i * batch * kq * 4) as u64;
            xad[i] = xad0 + (i * batch * kg * 16) as u64;
            self.quant_x_i8_at(xd[i], 0, xqd[i], xad[i], batch, k)?;
        }
        let bm = if batch <= 64 {
            env_tile("IMMA_BM_SMALL", 64)
        } else {
            env_tile("IMMA_BM", IMMA_BM_DEFAULT)
        };
        let bn = if batch <= 64 {
            env_tile("IMMA_BN_SMALL", small_batch_bn(batch))
        } else {
            env_tile("IMMA_BN", IMMA_BN_DEFAULT)
        };
        let lb2 = env_tile("IMMA_LB2", 2);
        let swap = if std::env::var("IMMA_GRID_SWAP")
            .map(|v| v != "0")
            .unwrap_or(true)
        {
            1
        } else {
            0
        };
        // ★ split-K：三条链一起切（`grid.z = 3·ks`，部分和布局 `[chain·ks + ksp][batch][m]`）。
        let ncol = batch.div_ceil(bn);
        let blocks = 3 * (m.div_ceil(bm)) * ncol;
        let ks = imma_ksplit(blocks, ncol, k, 3);
        let mut yd_eff = yd;
        let mut ipart_ptr = None;
        if ks > 1 {
            let t = self.ipart_scratch(3 * batch * m * ks)?;
            let base = self.f32_ptr(t, "imma_gemm_partial")?;
            for (i, slot) in yd_eff.iter_mut().enumerate() {
                *slot = base + (i * ks * batch * m * 4) as u64;
            }
            ipart_ptr = Some(t);
        }
        let src = format!(
            "#define IM_BM {bm}\n#define IM_BN {bn}\n#define IM_LB2 {lb2}\n#define IM_SWAP {swap}\n\
             #define IM_PIPE 0\n#define IM_NBUF 1\n#define IM_KSPLIT {ks}\n{IMMA_GEMM_SRC}"
        );
        let key = format!("imma_gemm_batch_z3_b{bm}n{bn}l{lb2}s{swap}p0d1k{ks}_m{m}k{k}");
        let func = self.kernel(&key, &src, "imma_gemm_batch")?;
        let (m_i, k_i, b_i) = (m as i32, k as i32, batch as i32);
        let params = [
            &aidx[0] as *const u64 as *mut c_void,
            &aidx[1] as *const u64 as *mut c_void,
            &aidx[2] as *const u64 as *mut c_void,
            &asz[0] as *const u64 as *mut c_void,
            &asz[1] as *const u64 as *mut c_void,
            &asz[2] as *const u64 as *mut c_void,
            &xqd[0] as *const u64 as *mut c_void,
            &xqd[1] as *const u64 as *mut c_void,
            &xqd[2] as *const u64 as *mut c_void,
            &xad[0] as *const u64 as *mut c_void,
            &xad[1] as *const u64 as *mut c_void,
            &xad[2] as *const u64 as *mut c_void,
            &yd_eff[0] as *const u64 as *mut c_void,
            &yd_eff[1] as *const u64 as *mut c_void,
            &yd_eff[2] as *const u64 as *mut c_void,
            &op[0] as *const i32 as *mut c_void,
            &op[1] as *const i32 as *mut c_void,
            &op[2] as *const i32 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        let grid = if swap == 1 {
            (
                batch.div_ceil(bn) as u32,
                m.div_ceil(bm) as u32,
                (3 * ks) as u32,
            )
        } else {
            (
                m.div_ceil(bm) as u32,
                batch.div_ceil(bn) as u32,
                (3 * ks) as u32,
            )
        };
        self.drv
            .launch_smem(self.stream, func, grid, (256, 1, 1), &params, 0)?;
        if ipart_ptr.is_some() {
            for i in 0..3 {
                self.imma_gemm_reduce(yd_eff[i], yd[i], ks, batch, m, op[i])?;
            }
        }
        Ok(())
    }

    #[allow(dead_code)] // 骨架阶段算子未实现，待补齐后由算子消费
    fn take(&mut self, t: TensorId, op: &str) -> R<CudaTensor> {
        self.tensors
            .remove(&t)
            .ok_or(format!("{op}: unknown tensor {t:?}").into())
    }

    #[allow(dead_code)] // 骨架阶段算子未实现，待补齐后由算子消费
    fn put(&mut self, t: TensorId, v: CudaTensor) {
        self.tensors.insert(t, v);
    }

    /// 统一调度 gemm kernel（op 见 GEMM_SRC：0=plain,1=bias,2=add,3=relu2,4=tanh）。
    /// `bias`/`x` 为可选残差指针；不需要时传空 map 的 None，kernel 内部以 0 表示 null。
    #[allow(clippy::too_many_arguments)]
    fn gemm_dispatch(
        &mut self,
        ad: u64,
        bd: u64,
        bias: Option<u64>,
        xd: Option<u64>,
        cd: u64,
        m: usize,
        n: usize,
        k: usize,
        op: i32,
    ) -> R<()> {
        // cuBLAS 路径：C = A @ B^T（A:[m,k] f16, B:[n,k] f16, C:[m,n] f32）。
        // op 语义：0=纯 GEMM, 1=+bias, 2=+x, 3=relu², 4=tanh。
        // CUBLAS_GEMM=0：保留 cuBLAS 句柄但 gemm 走自定义 kernel（隔离 cuBLAS gemm 与句柄污染）。
        let use_cublas_gemm =
            self.cublas.is_some() && std::env::var("CUBLAS_GEMM").as_deref() != Ok("0");
        if let Some(h) = self.cublas.filter(|_| use_cublas_gemm) {
            if std::env::var("CUBLAS_DIAG").is_ok() {
                log::info!("[CUBLAS_DIAG] gemm m={m} n={n} k={k} op={op}");
            }
            if self.gemm_prof {
                unsafe {
                    (self.drv.cu_event_record)(self.gemm_prof_ev_start, self.stream);
                }
            }
            // op==2 的残差 x 是独立缓冲，不能用 beta=1 累加到 C（C 初始非 x），故 beta=0 后 epilogue 加 x。
            let beta: f32 = 0.0;
            let driver = cublas_driver().ok_or("cublas driver gone")?;
            let alpha: f32 = 1.0;
            // 自定义 kernel 语义 C[m,n] = A[m,k] @ B^T[n,k]（A/B 均行主序 f16，C 行主序 f32）。
            // cuBLAS 输出列主序 C_cm = B @ A^T（形 [n,m]），ldc=n 即得到行主序 C[m,n]；
            // 故 transa=OP_T(权重 B)、transb=OP_N(输入 A)、m=n输出、n=m token、lda=ldb=k。
            let (m_i, n_i, k_i) = (n as c_int, m as c_int, k as c_int);
            let (lda, ldb, ldc) = (k as c_int, k as c_int, n as c_int);
            let r = unsafe {
                (driver.cublas_gemm_ex)(
                    h,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    m_i,
                    n_i,
                    k_i,
                    &alpha as *const f32 as *const c_void,
                    bd as *const c_void,
                    CUDA_R_16F,
                    lda,
                    ad as *const c_void,
                    CUDA_R_16F,
                    ldb,
                    &beta as *const f32 as *const c_void,
                    cd as *mut c_void,
                    CUDA_R_32F,
                    ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                )
            };
            if r != CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasGemmEx failed: {r}").into());
            }
            if self.gemm_prof {
                let mut ms: f32 = 0.0;
                unsafe {
                    (self.drv.cu_event_record)(self.gemm_prof_ev_end, self.stream);
                    (self.drv.cu_event_synchronize)(self.gemm_prof_ev_end);
                    (self.drv.cu_event_elapsed_time)(
                        &mut ms,
                        self.gemm_prof_ev_start,
                        self.gemm_prof_ev_end,
                    );
                }
                let e = self.gemm_times.entry((m, n, k, op)).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += (ms as f64).max(0.0);
            }
            // epilogue：op1 bias / op2 add x / op3 relu² / op4 tanh（op0 无需）。
            if op != 0 {
                let func = self.kernel("gemm_epilogue", GEMM_EPILOGUE_SRC, "rwkv_gemm_epilogue")?;
                let total = (m * n) as u32;
                let grid = (total.div_ceil(256), 1u32, 1u32);
                let block = (256u32, 1u32, 1u32);
                let (m_i, n_i, op_i) = (m as i32, n as i32, op);
                let (bias_v, x_v) = (bias.unwrap_or(0), xd.unwrap_or(0));
                let params = [
                    &cd as *const u64 as *mut c_void,
                    &bias_v as *const u64 as *mut c_void,
                    &x_v as *const u64 as *mut c_void,
                    &m_i as *const i32 as *mut c_void,
                    &n_i as *const i32 as *mut c_void,
                    &op_i as *const i32 as *mut c_void,
                ];
                self.drv.launch(self.stream, func, grid, block, &params)?;
            }
            return Ok(());
        }
        // 回退：自定义 kernel（cuBLAS 不可用时）。
        let func = self.kernel("gemm", GEMM_SRC, "rwkv_gemm")?;
        let grid = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1u32);
        let block = (16u32, 16u32, 1u32);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let (bias_v, x_v) = (bias.unwrap_or(0), xd.unwrap_or(0));
        let params = [
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &bias_v as *const u64 as *mut c_void,
            &x_v as *const u64 as *mut c_void,
            &cd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &op as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // 批量上传环形缓冲清理（若仍在 bulk 模式）
        if let Some(r) = self.bulk.borrow_mut().take() {
            unsafe {
                let _ = (self.drv.cu_stream_synchronize)(self.stream);
            }
            for ev in &r.events {
                unsafe {
                    let _ = (self.drv.cu_event_destroy)(*ev);
                }
            }
            unsafe {
                let _ = (self.drv.cu_mem_free_host)(r.base as *mut c_void);
            }
        }
        for (id, v) in self.tensors.iter() {
            // 导入的共享权重张量（build_shared 权重共享）不释放：归源实例所有。
            if self.foreign.contains(id) {
                continue;
            }
            let dptr = match v {
                CudaTensor::F32 { dptr, .. }
                | CudaTensor::F16 { dptr, .. }
                | CudaTensor::U32 { dptr, .. } => *dptr,
            };
            unsafe {
                (self.drv.cu_mem_free_v2)(dptr);
            }
        }
        for (_, exec) in self.prefill_graphs.drain() {
            unsafe {
                (self.drv.cu_graph_exec_destroy)(exec);
            }
        }
        for (_, exec) in self.selfloop_graphs.drain() {
            unsafe {
                (self.drv.cu_graph_exec_destroy)(exec);
            }
        }
        if let (Some(h), Some(cd)) = (self.cublas, cublas_driver()) {
            unsafe {
                (cd.cublas_destroy_v2)(h);
            }
        }
        unsafe {
            (self.drv.cu_event_destroy)(self.prof_ev_start);
            (self.drv.cu_event_destroy)(self.prof_ev_end);
            (self.drv.cu_event_destroy)(self.gemm_prof_ev_start);
            (self.drv.cu_event_destroy)(self.gemm_prof_ev_end);
            (self.drv.cu_stream_destroy)(self.stream);
            if !self.pinned.is_null() {
                (self.drv.cu_mem_free_host)(self.pinned);
            }
        }
        unsafe {
            (self.drv.cu_primary_ctx_release)(self.device);
        }
    }
}

/// gemv_f16 CUDA kernel：y[m] = Σ_k x[k]·A[m·K + k]。
/// A 为 fp16 行主序 (M,K)，x 为 f32，y 为 f32；batch 走 gridDim.y。
/// 每个 block 处理 4 行输出（对齐 Vulkan `GEMV_ROWS`），block 内 128 线程跨 K 归约。
const GEMV_F16_SRC: &str = r#"
extern "C" __global__ void gemv_f16(
    const __half* __restrict__ A,   // (M, K) row-major fp16
    const float*  __restrict__ x,   // (K * batch)
    float* __restrict__ y,          // (M * batch)
    const int m,
    const int k,
    const int batch)
{
    const int tid   = threadIdx.x;
    const int b     = blockIdx.y;
    const int row0  = blockIdx.x * 4;
    const int k0    = b * k;
    const int m0    = b * m;
    // 半精度向量化累积（对标 Albatross row1_linear_exact4）：x 转 half2、权重按 half2 读，
    // __hfma2 每次迭代做 2×FP16 FMA（吞吐为 FP32 的 2 倍）。4 行分别持有 2 个 half2 累加器。
    half2 hacc[4][2];
    #pragma unroll
    for (int r = 0; r < 4; r++) { hacc[r][0] = __half2half2(0.f); hacc[r][1] = __half2half2(0.f); }

    // 向量化主循环：每线程每次迭代处理 4 个 k（x 按 float4 读、权重按 8B half4 读）。
    const int k4 = k & ~3;
    for (int kq = tid * 4; kq < k4; kq += blockDim.x * 4) {
        const float4 xv = *reinterpret_cast<const float4*>(x + k0 + kq);
        const half2 hx01 = __floats2half2_rn(xv.x, xv.y);
        const half2 hx23 = __floats2half2_rn(xv.z, xv.w);
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const __half* wj = A + (row0 + r) * k + kq;
            const half2 w01 = *reinterpret_cast<const half2*>(wj);
            const half2 w23 = *reinterpret_cast<const half2*>(wj + 2);
            hacc[r][0] = __hfma2(hx01, w01, hacc[r][0]);
            hacc[r][1] = __hfma2(hx23, w23, hacc[r][1]);
        }
    }
    float acc[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const float2 f0 = __half22float2(hacc[r][0]);
        const float2 f1 = __half22float2(hacc[r][1]);
        acc[r] = f0.x + f0.y + f1.x + f1.y;
    }
    // 尾部标量兜底（k 非 4 倍数时）。
    for (int kk = k4 + tid; kk < k; kk += blockDim.x) {
        const float xv = x[k0 + kk];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            acc[r] += __half2float(A[(row0 + r) * k + kk]) * xv;
        }
    }

    // warp shuffle 归约（对齐 Albatross row1_linear_exact4_kernel<128,2>）：
    // 每个 warp（32 线程）先 shfl 归约本 warp 的 4 行，再 3 个 warp 结果由 tid0 汇总。
    // 相比共享内存全归约（7 步 __syncthreads），只同步 1 次，减少 block 内同步开销。
    __shared__ float partial[4 /*warp*/][4 /*row*/];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) partial[warp][r] = v;
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < 4; w++) sum += partial[w][r];
            if (row0 + r < m) y[m0 + row0 + r] = sum;
        }
    }
}
"#;

// ==== batch 并发 kernel（单实例多序列：B slot 共享权重，一次读权重算 B 份）====

/// norm_lerp6 batch CUDA kernel：x/state/or_..og 为 [batch, C]（slot 主序）；
/// gamma/beta 与 **xr..xg（lerp 系数，共享权重）** 跨 slot 共享 [C]——无 slot 偏移。
/// dispatch (ceil(c/BLOCK), batch, 1)：grid.y = slot id。
const NORM_LERP6_BATCH_SRC: &str = r#"
extern "C" __global__ void norm_lerp6_batch(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ xr,
    const float* __restrict__ xw,
    const float* __restrict__ xk,
    const float* __restrict__ xv,
    const float* __restrict__ xa,
    const float* __restrict__ xg,
    float* __restrict__ or_,
    float* __restrict__ ow,
    float* __restrict__ ok,
    float* __restrict__ ov,
    float* __restrict__ oa,
    float* __restrict__ og,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;
    const int off  = blockIdx.y * c;   // slot 基址

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[off + i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off_ = 16; off_ > 0; off_ >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off_);
        sq  += __shfl_down_sync(0xffffffffu, sq, off_);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    const int start = gx * blockDim.x;
    const int end   = min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[off + i];
        const float ln1  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[off + i];
        or_[off + i] = ln1 + xr[i] * (prev - ln1);
        ow[off + i]  = ln1 + xw[i] * (prev - ln1);
        ok[off + i]  = ln1 + xk[i] * (prev - ln1);
        ov[off + i]  = ln1 + xv[i] * (prev - ln1);
        oa[off + i]  = ln1 + xa[i] * (prev - ln1);
        og[off + i]  = ln1 + xg[i] * (prev - ln1);
        state[off + i] = ln1;
    }
}
"#;

/// cmix_norm_lerp batch CUDA kernel：x/state/out_xb 为 [batch, C]，gamma/beta/coeff 共享。
/// dispatch (ceil(c/BLOCK), batch, 1)。
const CMIX_NORM_LERP_BATCH_SRC: &str = r#"
extern "C" __global__ void cmix_norm_lerp_batch(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ coeff,
    float* __restrict__ out_xb,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;
    const int off  = blockIdx.y * c;

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[off + i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off_ = 16; off_ > 0; off_ >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off_);
        sq  += __shfl_down_sync(0xffffffffu, sq, off_);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    const int start = gx * blockDim.x;
    const int end   = min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[off + i];
        const float ln2  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[off + i];
        out_xb[off + i] = ln2 + coeff[i] * (prev - ln2);
        state[off + i] = ln2;
    }
}
"#;

/// gather_rows_f16 batch CUDA kernel：按 tok[b] 各取一行 → dst[b*C + i]（f32）。
/// dispatch (ceil(C/256), batch, 1)。
const GATHER_ROWS_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_gather_rows_f16(
    const unsigned int* __restrict__ in_tok,  // [batch] token 索引（f32 位模式）
    const __half*  __restrict__ in_src,       // [VOCAB, C] fp16
    float* __restrict__ out_dst,              // [batch, C] fp32
    const int c)
{
    const int b = blockIdx.y;
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idx = in_tok[b];
    if (index < c) {
        out_dst[b * c + index] = __half2float(in_src[(size_t)idx * (size_t)c + (size_t)index]);
    }
}
"#;

/// gemv_int8_rkv_stage1 batch CUDA kernel（权重复用版）：每 block 读一次 int8 权重
/// 与 scale/zero，在寄存器累加器中复用给 BGRP 个 slot——带宽 ≈ 1/ceil(B/BGRP)
///（信天翁 rows 模型；旧 grid.y=slot 版每 slot 各读全量权重，带宽 ×B 零增益）。
/// x 输入（xr..xg）与输出为 [batch, ...]（slot 主序）。
/// dispatch (C/ROWS + VM + WM + AM + GM, ceil(batch/BGRP), 1)。
const GEMV_INT8_RKV_STAGE1_BATCH_SRC: &str = r#"
#ifndef MID_GG
#define MID_GG 4
#endif
__device__ __forceinline__ void unpack_int8_sz_batch(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}

extern "C" __global__ void __launch_bounds__(128, 4) gemv_int8_rkv_stage1_batch(
    const unsigned int* __restrict__ R_idx,
    const unsigned int* __restrict__ R_sz,
    const unsigned int* __restrict__ K_idx,
    const unsigned int* __restrict__ K_sz,
    const unsigned int* __restrict__ V_idx,
    const unsigned int* __restrict__ V_sz,
    const float*  __restrict__ V1,
    const float*  __restrict__ W1,
    const float*  __restrict__ A1,
    const float*  __restrict__ G1,
    const float*  __restrict__ xr,            // [batch, C]
    const float*  __restrict__ xk,
    const float*  __restrict__ xv,
    const float*  __restrict__ xw,
    const float*  __restrict__ xa,
    const float*  __restrict__ xg,
    float* __restrict__ out_r,                // [batch, C]
    float* __restrict__ out_k,
    __half* __restrict__ out_v,               // [batch, C] fp16
    float* __restrict__ out_vm,               // [batch, VM]
    float* __restrict__ out_wm,               // [batch, WM]
    float* __restrict__ out_am,               // [batch, AM]
    float* __restrict__ out_gm,               // [batch, GM]
    const __half* __restrict__ xr16,           // r/k/v 分支用 fp16 激活（use16x 时有效）
    const __half* __restrict__ xk16,
    const __half* __restrict__ xv16,
    const int use16x,                          // 1 = r/k/v 走 fp16 激活
    const int c,
    const int vm,
    const int wm,
    const int am,
    const int gm,
    const int batch,
    const int crows)                           // r/k/v 伪行数（= c/ROWS）；**0 = 跳过 r/k/v**（由 IMMA 路径代劳）
{
    // ★ 第三代（2026-09-21）：**r/k/v 三矩阵改用 blockIdx.z 并行**，于是单块只背
    // 一个矩阵，**BGRP 可以抬到 8（权重只读一遍）而累加器仍是 ROWS×BGRP=32**。
    // 旧版三矩阵同块 ⇒ BGRP 只能给到 2（累加器 3×ROWS×BGRP），grid.y=ceil(B/2)=4
    // ⇒ **每层 r/k/v 的 int8 权重（19.65MB）被白读 4 遍（78.6MB/层、2.5GB/步）**。
    // 反例留档：不提 BGRP、只把 ROWS 2 换 BGRP 4 的写法端到端更慢（292.8→262.9），
    // 因为 x 片被 m/ROWS 个 block 重读的流量会翻倍——所以必须**同时**降两侧流量。
    constexpr int ROWS = 4;
    constexpr int KG_MAX = 32;
    constexpr int BGRP = 8;
    const int tid  = threadIdx.x;
    const int flat = blockIdx.x;
    const int b0   = blockIdx.y * (BGRP * MID_GG);
    const int bcnt = min(BGRP * MID_GG, batch - b0);

    if (flat < crows) {
        // 本块负责的矩阵（0=r, 1=k, 2=v）由 blockIdx.z 选
        const int z = blockIdx.z;
        const unsigned int* W_idx = (z == 0) ? R_idx : ((z == 1) ? K_idx : V_idx);
        const unsigned int* W_sz  = (z == 0) ? R_sz  : ((z == 1) ? K_sz  : V_sz);
        const __half*       xh    = (z == 0) ? xr16  : ((z == 1) ? xk16  : xv16);
        const float*        xf    = (z == 0) ? xr    : ((z == 1) ? xk    : xv);
        const int row_base = flat * ROWS;
        const int KV = c / 4;
        const int KG = c / 128;

        __shared__ float s_scale[ROWS][KG_MAX];
        __shared__ float s_zero[ROWS][KG_MAX];

        for (int i = tid; i < ROWS * KG; i += blockDim.x) {
            const int r = i / KG;
            const int g = i % KG;
            const int row = row_base + r;
            float sc, zr;
            unpack_int8_sz_batch(W_sz[row * KG + g], sc, zr);
            s_scale[r][g] = sc;
            s_zero[r][g]  = zr;
        }
        __syncthreads();

        // 累加器：ROWS 行 × BGRP slot（half2，32 个）
        half2 acc[ROWS][BGRP];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) acc[r][b] = __half2half2(0.f);
        // 主循环：int8 idx 读一次 + 反量化一次 → 逐 slot FMA（权重读 1 份算 bcnt 份）。
        for (int kk = tid; kk < KV; kk += blockDim.x) {
            const int g = kk >> 5;
            // x 片按 slot 取一次（2 个 half2）供 ROWS 行复用——旧写法把标量 x 加载
            // 放在 b 循环（b 又在 r 循环内），x 被重复读 ROWS×BGRP 遍，
            // 是本 kernel 随 B 近线性增长的主因（同 gemv_variant_mb 的修法）。
            half2 hx0[BGRP], hx1[BGRP];
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                if (b < bcnt) {
                    const int off = (b0 + b) * c + 4 * kk;
                    if (use16x) {
                        // fp16 激活：直接 half2 读（旧路径本就把 x 舍入到 fp16 再用，
                        // 故数值逐位一致；x 被 m/ROWS 个 block 重读，是最大流量项）。
                        hx0[b] = *reinterpret_cast<const half2*>(xh + off);
                        hx1[b] = *reinterpret_cast<const half2*>(xh + off + 2);
                    } else {
                        hx0[b] = __floats2half2_rn(xf[off], xf[off + 1]);
                        hx1[b] = __floats2half2_rn(xf[off + 2], xf[off + 3]);
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const unsigned int p = W_idx[(row_base + r) * KV + kk];
                const float sc = s_scale[r][g], zr = s_zero[r][g];
                const half2 w0 = __floats2half2_rn(
                    sc * (float)((p >> 0) & 0xFFu) + zr, sc * (float)((p >> 8) & 0xFFu) + zr);
                const half2 w1 = __floats2half2_rn(
                    sc * (float)((p >> 16) & 0xFFu) + zr, sc * (float)((p >> 24) & 0xFFu) + zr);
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    acc[r][b] = __hfma2(hx0[b], w0, acc[r][b]);
                    acc[r][b] = __hfma2(hx1[b], w1, acc[r][b]);
                }
            }
        }
        float l[ROWS][BGRP];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                const float2 f = __half22float2(acc[r][b]);
                l[r][b] = f.x + f.y;
            }

        __shared__ float partial[4 /*warp*/][ROWS][BGRP];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                float v = l[r][b];
                #pragma unroll
                for (int off2 = 16; off2 > 0; off2 >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, off2);
                }
                if (lane == 0) partial[warp][r][b] = v;
            }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row_base + r;
                if (row < c) {
                    #pragma unroll
                    for (int b = 0; b < BGRP; b++) {
                        if (b >= bcnt) break;
                        float s = 0.f;
                        #pragma unroll
                        for (int w = 0; w < 4; w++) s += partial[w][r][b];
                        const int off = (b0 + b) * c + row;
                        if (z == 0) out_r[off] = s;
                        else if (z == 1) out_k[off] = s;
                        else out_v[off] = __float2half(s);
                    }
                }
            }
        }
        return;
    }

    // mid 投影分支只在 z==0 执行（grid.z 是为 r/k/v 复用而设的；z>0 的 mid 块直接退出）。
    if (blockIdx.z != 0) return;
    // mid 投影分支：权重行读一次，逐 slot 累加（bcnt 份 dot 共享权重读取）。
    //
    // ★ 2026-09-21：**槽跨度 ×GG**（`MID_GG`，默认 1 = 每 block BGRP=8 个槽）。
    //   原先这里用 `const float* xsrc[BGRP]` 预存每槽基址，**被编译器落到 local memory**
    //   ⇒ 内层每次 `xsrc[b][kk]` 都是 local 访存，实测 2.55ms/次。
    //   改成「基址 + 常数偏移直接寻址」后降到 **1.03ms/次（2.5×）**——GG=1 就是最优。
    //   反例留档：GG=2/4/8 分别 1.39 / 1.98 / 4.66 ms（累加器变多又把寄存器压回溢出）。
    const int mid_idx = flat - crows;
    constexpr int MSC = BGRP * MID_GG;   // 每 block 覆盖的槽数
    float local_dot[MID_GG][BGRP];
    #pragma unroll
    for (int gg = 0; gg < MID_GG; gg++)
        #pragma unroll
        for (int b = 0; b < BGRP; b++) local_dot[gg][b] = 0.f;
    int chain = 3;
    int row = 0;
    const float* wsrc = nullptr;
    const float* xbase = nullptr;
    if (mid_idx < vm) {
        chain = 0; row = mid_idx;
        wsrc = V1 + (long long)row * c;
        xbase = xv;
    } else if (mid_idx < vm + wm) {
        chain = 1; row = mid_idx - vm;
        wsrc = W1 + (long long)row * c;
        xbase = xw;
    } else if (mid_idx < vm + wm + am) {
        chain = 2; row = mid_idx - vm - wm;
        wsrc = A1 + (long long)row * c;
        xbase = xa;
    } else {
        row = mid_idx - vm - wm - am;
        wsrc = G1 + (long long)row * c;
        xbase = xg;
    }
    for (int kk = tid; kk < c; kk += blockDim.x) {
        const float w = wsrc[kk];
        #pragma unroll
        for (int gg = 0; gg < MID_GG; gg++) {
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                // 越界槽钳到 batch-1（读到合法地址即可，写回时才判边界）
                const int s = min(b0 + gg * BGRP + b, batch - 1);
                local_dot[gg][b] += w * xbase[(long long)s * c + kk];
            }
        }
    }
    __shared__ float sm[128][MSC];
    #pragma unroll
    for (int gg = 0; gg < MID_GG; gg++)
        #pragma unroll
        for (int b = 0; b < BGRP; b++) sm[tid][gg * BGRP + b] = local_dot[gg][b];
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int i = 0; i < MSC; i++) sm[tid][i] += sm[tid + stride][i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int gg = 0; gg < MID_GG; gg++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                const int s = b0 + gg * BGRP + b;
                if (s >= batch) break;
                const float result = sm[0][gg * BGRP + b];
                if (chain == 0) out_vm[s * vm + row] = result;
                else if (chain == 1) out_wm[s * wm + row] = tanhf(result);
                else if (chain == 2) out_am[s * am + row] = result;
                else out_gm[s * gm + row] = result;
            }
    }
}
"#;

/// gemv_lowrank_chain4 batch CUDA kernel（warp-per-row 版）：每 block 8 warp
/// 各算 1 个输出行 × BGRP slot（原版整 block 归约 1 行，M=2560 时 grid 太大
/// 占 14.9%——每 block 只做 512B 功重，syncthreads 6 次为主因）。
/// dispatch (M/8, ceil(batch/BGRP), 1)；block=256（8 warp × 32 thread）。
const GEMV_LOWRANK_CHAIN4_BATCH_SRC: &str = r#"
// g 链在内层每元素一次 sigmoid（= expf）；`expf` 在 NVRTC 默认档是软件展开（~20 条指令），
// 而这是 k 循环里的**逐元素**调用。`CHAIN4_FASTEXP=1` 换 `__expf` 硬件近似做 A/B。
#ifdef CHAIN4_FASTEXP
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
#else
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
#endif

extern "C" __global__ void __launch_bounds__(256, 2) gemv_lowrank_chain4_batch(
    const float*  __restrict__ W2,   // [M, KW] fp32 行主序（共享权重）
    const float*  __restrict__ A2,   // [M, KA]
    const float*  __restrict__ V2,   // [M, KV]
    const float*  __restrict__ G2,   // [M, KG]
    const float*  __restrict__ xw,   // [batch, KW]
    const float*  __restrict__ xa,   // [batch, KA]
    const float*  __restrict__ xv,   // [batch, KV]
    const float*  __restrict__ xg,   // [batch, KG]
    const float*  __restrict__ w0,   // [M]（共享）
    const float*  __restrict__ a0,   // [M]
    const float*  __restrict__ v0,   // [M]
    const float*  __restrict__ scale,// [1]
    const __half* __restrict__ v_first, // [batch, M] fp16
    __half* __restrict__ out_w,      // [batch, M] fp16
    __half* __restrict__ out_a,      // [batch, M] fp16
    __half* __restrict__ out_v,      // [batch, M] fp16（读改写）
    __half* __restrict__ out_g,      // [batch, M] fp16
    const int m,
    const int kw,
    const int ka,
    const int kv,
    const int kg,
    const int batch)
{
    constexpr int BGRP = 4;
    constexpr int WARPS = 8;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row  = blockIdx.x * WARPS + warp;
    if (row >= m) return;
    const int b0   = blockIdx.y * BGRP;
    const int bcnt = min(BGRP, batch - b0);

    const float sc = scale[0];
    // 权重行读一次、逐 slot 累加（旧写法把 slot 放外层，权重行被重复读 bcnt 遍，
    // 是本 kernel 耗时随 B 线性增长的原因——与 gemv_variant_mb 同源）。
    //
    // ★ 2026-09-21 反例留档：曾把这版改成「slot 进块内循环」以削流量
    //   （SPB=64 → 权重只读一遍、x 只读 M/8 遍，流量 717MB→47MB），
    //   实测反而**变慢**（1.99 → 2.82ms；SPB=8 最好也只有 2.10ms）。
    //   原因是 block 数从 4608 掉到 288/2304，**并行度塌了**——本 kernel 在
    //   B=256 已经是带宽受限（717MB / 1.99ms = 360GB/s，roofline 500），
    //   削流量换来的收益抵不上并行度损失。**本结构不要动。**
    float lw[BGRP], la[BGRP], lv[BGRP], lg[BGRP];
    #pragma unroll
    for (int bi = 0; bi < BGRP; bi++) {
        lw[bi] = 0.f;
        la[bi] = 0.f;
        lv[bi] = 0.f;
        lg[bi] = 0.f;
    }
    for (int k = lane; k < kw; k += 32) {
        const float w = W2[row * kw + k];
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            if (bi < bcnt) lw[bi] += xw[(b0 + bi) * kw + k] * w;
        }
    }
    for (int k = lane; k < ka; k += 32) {
        const float w = A2[row * ka + k];
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            if (bi < bcnt) la[bi] += xa[(b0 + bi) * ka + k] * w;
        }
    }
    for (int k = lane; k < kv; k += 32) {
        const float w = V2[row * kv + k];
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            if (bi < bcnt) lv[bi] += xv[(b0 + bi) * kv + k] * w;
        }
    }
    for (int k = lane; k < kg; k += 32) {
        const float w = G2[row * kg + k];
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            if (bi < bcnt) lg[bi] += sigmoidf_(xg[(b0 + bi) * kg + k]) * w;
        }
    }
    #pragma unroll
    for (int bi = 0; bi < BGRP; bi++) {
        if (bi >= bcnt) break;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lw[bi] += __shfl_down_sync(0xffffffffu, lw[bi], off);
            la[bi] += __shfl_down_sync(0xffffffffu, la[bi], off);
            lv[bi] += __shfl_down_sync(0xffffffffu, lv[bi], off);
            lg[bi] += __shfl_down_sync(0xffffffffu, lg[bi], off);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            if (bi >= bcnt) break;
            const int mo = (b0 + bi) * m + row;
            out_w[mo] = __float2half(expf(sc * sigmoidf_(lw[bi] + w0[row])));
            out_a[mo] = __float2half(sigmoidf_(la[bi] + a0[row]));
            const float vcur = __half2float(out_v[mo]);
            out_v[mo] = __float2half(
                vcur + sigmoidf_(lv[bi] + v0[row]) * (__half2float(v_first[mo]) - vcur));
            out_g[mo] = __float2half(lg[bi]);
        }
    }
}
"#;

// fp16 张量核版 lowrank chain4（`CHAIN4_FP16=1`）：内层 4 链 GEMM 全走 `mma.m16n8k8`。
// 与 fp32 版同语义（输出逐位同精度），只是乘加落到 fp16 张量核。
const GEMV_LOWRANK_CHAIN4_BATCH_FP16_SRC: &str = r#"
__device__ __forceinline__ void mma_m16n8k8_f32acc(float* d, const unsigned* a, const unsigned* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void __launch_bounds__(256, 2) gemv_lowrank_chain4_batch(
    const float*  __restrict__ W2, const float*  __restrict__ A2,
    const float*  __restrict__ V2, const float*  __restrict__ G2,
    const float*  __restrict__ xw, const float*  __restrict__ xa,
    const float*  __restrict__ xv, const float*  __restrict__ xg,
    const float*  __restrict__ w0, const float*  __restrict__ a0,
    const float*  __restrict__ v0, const float*  __restrict__ scale,
    const __half* __restrict__ v_first,
    __half* __restrict__ out_w, __half* __restrict__ out_a,
    __half* __restrict__ out_v, __half* __restrict__ out_g,
    const int m, const int kw, const int ka, const int kv, const int kg, const int batch)
{
    constexpr int BGRP = 4;
    constexpr int WARPS = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * WARPS + warp;
    if (row >= m) return;
    const int b0   = blockIdx.y * BGRP;
    const int bcnt = min(BGRP, batch - b0);
    const float sc = scale[0];

    // 每链 f32 累加器，但**乘加在 mma 内走 fp16 张量核**。
    float lw[BGRP], la[BGRP], lv[BGRP], lg[BGRP];
    #pragma unroll
    for (int bi = 0; bi < BGRP; bi++) { lw[bi] = 0.f; la[bi] = 0.f; lv[bi] = 0.f; lg[bi] = 0.f; }

    // w 链：kw ∈ {8 倍数}，每 8 k 一次 mma
    for (int k0 = 0; k0 < kw; k0 += 8) {
        const int k = k0 + (lane & 7);
        const __half wv = __float2half(W2[row * kw + k]);
        const unsigned wa = (unsigned)__half_as_ushort(wv);
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            const __half xv_ = __float2half(xw[(b0 + bi) * kw + k]);
            const unsigned xa_ = (unsigned)__half_as_ushort(xv_);
            float d[4] = {lw[bi], 0.f, 0.f, 0.f};
            mma_m16n8k8_f32acc(d, &wa, &xa_);
            lw[bi] = d[0];
        }
    }
    // a 链
    for (int k0 = 0; k0 < ka; k0 += 8) {
        const int k = k0 + (lane & 7);
        const __half wv = __float2half(A2[row * ka + k]);
        const unsigned wa = (unsigned)__half_as_ushort(wv);
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            const __half xv_ = __float2half(xa[(b0 + bi) * ka + k]);
            const unsigned xa_ = (unsigned)__half_as_ushort(xv_);
            float d[4] = {la[bi], 0.f, 0.f, 0.f};
            mma_m16n8k8_f32acc(d, &wa, &xa_);
            la[bi] = d[0];
        }
    }
    // v 链
    for (int k0 = 0; k0 < kv; k0 += 8) {
        const int k = k0 + (lane & 7);
        const __half wv = __float2half(V2[row * kv + k]);
        const unsigned wa = (unsigned)__half_as_ushort(wv);
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            const __half xv_ = __float2half(xv[(b0 + bi) * kv + k]);
            const unsigned xa_ = (unsigned)__half_as_ushort(xv_);
            float d[4] = {lv[bi], 0.f, 0.f, 0.f};
            mma_m16n8k8_f32acc(d, &wa, &xa_);
            lv[bi] = d[0];
        }
    }
    // g 链（内层 sigmoid 与 fp32 版一致）
    for (int k0 = 0; k0 < kg; k0 += 8) {
        const int k = k0 + (lane & 7);
        const __half wv = __float2half(G2[row * kg + k]);
        const unsigned wa = (unsigned)__half_as_ushort(wv);
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            const __half xv_ = __float2half(sigmoidf_(xg[(b0 + bi) * kg + k]));
            const unsigned xa_ = (unsigned)__half_as_ushort(xv_);
            float d[4] = {lg[bi], 0.f, 0.f, 0.f};
            mma_m16n8k8_f32acc(d, &wa, &xa_);
            lg[bi] = d[0];
        }
    }
    #pragma unroll
    for (int bi = 0; bi < BGRP; bi++) {
        if (bi >= bcnt) break;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lw[bi] += __shfl_down_sync(0xffffffffu, lw[bi], off);
            la[bi] += __shfl_down_sync(0xffffffffu, la[bi], off);
            lv[bi] += __shfl_down_sync(0xffffffffu, lv[bi], off);
            lg[bi] += __shfl_down_sync(0xffffffffu, lg[bi], off);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int bi = 0; bi < BGRP; bi++) {
            if (bi >= bcnt) break;
            const int mo = (b0 + bi) * m + row;
            out_w[mo] = __float2half(expf(sc * sigmoidf_(lw[bi] + w0[row])));
            out_a[mo] = __float2half(sigmoidf_(la[bi] + a0[row]));
            const float vcur = __half2float(out_v[mo]);
            out_v[mo] = __float2half(
                vcur + sigmoidf_(lv[bi] + v0[row]) * (__half2float(v_first[mo]) - vcur));
            out_g[mo] = __float2half(lg[bi]);
        }
    }
}
"#;

/// 低秩链**两级 fp16 张量核 GEMM**（`LOWRANK_GEMM=1`，Turing `mma.m16n8k8`）。
///
/// ★ 与既有 `gemv_*` 系列的结构性区别：**权重 tile 进 smem、整 batch 流过**（真 tiled
/// GEMM），而不是「按 slot 分块、每块重读权重」。后者是 fp32 版本带宽受限的根因
/// （§3.4d 账本：chain4 717MB/层 @360GB/s），也是 `CHAIN4_FP16` 反例失败的原因
/// ——那版把 mma 套在旧分块里，权重仍被重读 ~20 遍 ⇒ 6.52ms（比 fp32 还慢 3.2×）。
///
/// 两级共用同一 staging：A = `[rows, xs]` fp16 行主序、B = `[n, k]` fp16 行主序
/// （本工程权重 `[n][k]` 天然是 col-major 的 B，零转置）；差别只在 epilogue。
///
/// 片段↔线程映射（`m16n8k8.row.col.f32.f16`）：`gID = lane>>2` / `tig = lane&3`；
/// A = 2 个 u32（row = gID 与 gID+8，col = tig*2+{0,1}）、B = 1 个 u32
/// （n = gID，k = tig*2+{0,1}）、D = 4 个 f32（row = gID/gID+8 × col = tig*2+{0,1}）。
///
/// warp 布局：`LR_NWM × LR_NWN = 8` 个 warp（block = 256），每 warp 负责
/// `BM/LR_NWM` 行 × `BN/LR_NWN` 列 ⇒ 要求 `BM % (16·LR_NWM) == 0`、`BN % (8·LR_NWN) == 0`。
const LOWRANK_GEMM_SRC: &str = r#"
#ifndef LR1_BM
#define LR1_BM 64
#endif
#ifndef LR1_BN
#define LR1_BN 64
#endif
#ifndef LR1_BK
#define LR1_BK 128
#endif
#ifndef LR2_BM
#define LR2_BM 64
#endif
#ifndef LR2_BN
#define LR2_BN 64
#endif
#ifndef LR2_BK
#define LR2_BK 64
#endif
#ifndef LR_NWM
#define LR_NWM 4
#endif
#ifndef LR_NWN
#define LR_NWN 2
#endif
#ifndef LR_LB
#define LR_LB 2
#endif
// 第三级（ffn_value 稠密 GEMM）的 tile：n=C=2560、k=fh=10240、batch=256。
// BM 取 128（**整 batch 的一半**）⇒ 权重只被重读 `batch/BM = 2` 遍（BM=64 是 4 遍）。
// BK=64 而非 128 是因为 smem 预算：`(128+64)·(64·2+16) = 27.6KiB`（BK=128 会到 52KiB 超限）。
#ifndef LR3_BM
#define LR3_BM 128
#endif
#ifndef LR3_BN
#define LR3_BN 64
#endif
#ifndef LR3_BK
#define LR3_BK 64
#endif

__device__ __forceinline__ void lr_mma_f16(float* d, const unsigned* a, const unsigned* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}
__device__ __forceinline__ float lr_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// 把 A（[rows, xs]）与 B（[n_pad, k]）的当前 k 分块搬进 smem（u32 粒度 = 2 个 half）。
// 行跨距 AS 留 16 字节余量避 bank 冲突；行越界统一**钳到末行**（只读合法地址，写回时判界）。
__device__ __forceinline__ void lr_load_tiles(
    unsigned char* as_, unsigned char* bs_, int AS,
    const __half* __restrict__ X, const __half* __restrict__ W,
    int xs, int n_pad, int k, int batch, int row0, int col0, int kt,
    int BM, int BN, int BK, int tid, int nthr)
{
    const int a_u32 = BK >> 1;
    for (int i = tid; i < BM * a_u32; i += nthr) {
        const int rr = i / a_u32, cc = i - rr * a_u32;
        const int src = min(row0 + rr, batch - 1);
        *(unsigned int*)(as_ + rr * AS + cc * 4) =
            ((const unsigned int*)(X + (long long)src * xs + kt))[cc];
    }
    for (int i = tid; i < BN * a_u32; i += nthr) {
        const int rr = i / a_u32, cc = i - rr * a_u32;
        const int src = min(col0 + rr, n_pad - 1);
        *(unsigned int*)(bs_ + rr * AS + cc * 4) =
            ((const unsigned int*)(W + (long long)src * k + kt))[cc];
    }
}

// 二级各链的 epilogue（每个输出元素一次；`t` 是 GEMM 的 fp32 累加结果）。
// 语义与 fp32 `gemv_lowrank_chain4_batch` 逐字对应：
//   chain 0 = v：`cur + sigmoid(t + v0[m]) · (v_first[m] − cur)`（**读改写**）
//   chain 1 = w：`exp(sc · sigmoid(t + w0[m]))`   chain 2 = a：`sigmoid(t + a0[m])`
//   chain 3 = g：直出（sigmoid 已在 stage1 epilogue 作用于 mid_g）
__device__ __forceinline__ float lr_chain_epi(
    int chain, float t, int rr, int mm, int n_pad,
    const float* __restrict__ bias, const __half* __restrict__ v_first,
    __half* __restrict__ out, float sc)
{
    if (chain == 0) {
        const long long o = (long long)rr * n_pad + mm;
        const float cur = __half2float(out[o]);
        return cur + lr_sigmoidf(t + bias[mm]) * (__half2float(v_first[o]) - cur);
    }
    if (chain == 1) return expf(sc * lr_sigmoidf(t + bias[mm]));
    if (chain == 2) return lr_sigmoidf(t + bias[mm]);
    return t;
}

// 一级：mid[b, j] = act( Σ_c x[b,c] · W1[j,c] )，出 fp16。
// act：0 = 无（v/a 链）、1 = tanh（w 链）、2 = sigmoid（g 链）。
//
// ★ 2026-09-22：**4 条链合并成一次 launch**（`LR1_MERGE`）。四条链各自的 `n_pad` 是
// 64/128/128/320，BN=16 时 grid.y = 4/8/8/20 ⇒ 四次串行的 launch 分别只有
// 16/32/32/80 个块（**v 链只用 16 个块去喂 68 个 SM**），合计约 5 波。
// 合并后 grid = (batch/BM, max_pad/BN, 4) = 160 个有效块一次铺开 ⇒ 约 3 波。
// 链选择用 `blockIdx.z`；`col0 >= n_pad` 的块直接 return（只浪费空块调度）。
extern "C" __global__ void __launch_bounds__(256, LR_LB) lowrank_stage1_gemm(
    const __half* __restrict__ W0, const __half* __restrict__ W1,
    const __half* __restrict__ W2, const __half* __restrict__ W3,
    const __half* __restrict__ X0, const __half* __restrict__ X1,
    const __half* __restrict__ X2, const __half* __restrict__ X3,
    __half* __restrict__ M0, __half* __restrict__ M1,
    __half* __restrict__ M2, __half* __restrict__ M3,
    const int n0, const int n1, const int n2, const int n3,
    const int a0, const int a1, const int a2, const int a3,
    const int k, const int batch, const int xs, const int ms)
{
    constexpr int BM = LR1_BM, BN = LR1_BN, BK = LR1_BK;
    constexpr int AS = BK * 2 + 16;
    constexpr int MT = BM / (16 * LR_NWM);
    constexpr int NT = BN / (8 * LR_NWN);
    __shared__ unsigned char as_[BM * AS];
    __shared__ unsigned char bs_[BN * AS];

    const int z = blockIdx.z;
    const int n_pad = (z == 0) ? n0 : (z == 1) ? n1 : (z == 2) ? n2 : n3;
    const int act   = (z == 0) ? a0 : (z == 1) ? a1 : (z == 2) ? a2 : a3;
    const __half* __restrict__ W = (z == 0) ? W0 : (z == 1) ? W1 : (z == 2) ? W2 : W3;
    const __half* __restrict__ X = (z == 0) ? X0 : (z == 1) ? X1 : (z == 2) ? X2 : X3;
    __half* __restrict__ mid = (z == 0) ? M0 : (z == 1) ? M1 : (z == 2) ? M2 : M3;

    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int gID = lane >> 2, tig = lane & 3;
    const int row0 = blockIdx.x * BM, col0 = blockIdx.y * BN;
    if (col0 >= n_pad) return;
    const int wrow = (wid / LR_NWN) * (BM / LR_NWM);
    const int wcol = (wid % LR_NWN) * (BN / LR_NWN);

    float acc[MT][NT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][nt][q] = 0.f;

    for (int kt = 0; kt < k; kt += BK) {
        lr_load_tiles(as_, bs_, AS, X, W, xs, n_pad, k, batch, row0, col0, kt,
                      BM, BN, BK, tid, 256);
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < BK / 8; ++ks) {
            const int ko = ks * 16 + tig * 4;   // 字节偏移 = (koff + tig*2) * 2
            unsigned bv[NT];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                bv[nt] = *(const unsigned int*)(bs_ + (wcol + nt * 8 + gID) * AS + ko);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned av[2];
                av[0] = *(const unsigned int*)(as_ + (wrow + mt * 16 + gID) * AS + ko);
                av[1] = *(const unsigned int*)(as_ + (wrow + mt * 16 + gID + 8) * AS + ko);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) lr_mma_f16(acc[mt][nt], av, &bv[nt]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int r  = row0 + wrow + mt * 16 + gID;
            const int cc = col0 + wcol + nt * 8 + tig * 2;
            float v0 = acc[mt][nt][0], v1 = acc[mt][nt][1];
            float v2 = acc[mt][nt][2], v3 = acc[mt][nt][3];
            if (act == 1) {
                v0 = tanhf(v0); v1 = tanhf(v1); v2 = tanhf(v2); v3 = tanhf(v3);
            } else if (act == 2) {
                v0 = lr_sigmoidf(v0); v1 = lr_sigmoidf(v1);
                v2 = lr_sigmoidf(v2); v3 = lr_sigmoidf(v3);
            }
            if (r < batch)
                *(__half2*)&mid[(long long)r * ms + cc] = __floats2half2_rn(v0, v1);
            if (r + 8 < batch)
                *(__half2*)&mid[(long long)(r + 8) * ms + cc] = __floats2half2_rn(v2, v3);
        }
}

// 二级：out[b, m] = epi( Σ_j mid[b,j] · W2[m,j] )，出 fp16（与 chain4 同缓冲区、同舍入）。
extern "C" __global__ void __launch_bounds__(256, LR_LB) lowrank_stage2_gemm(
    const __half* __restrict__ W,        // [n_pad, k] fp16（本链二级权重）
    const __half* __restrict__ X,        // [batch, xs] fp16（mid16 本链列段起点）
    const float*  __restrict__ bias,     // [n_pad] fp32（g 链传 0）
    const __half* __restrict__ v_first,  // [batch, n_pad] fp16
    __half*       __restrict__ out,      // [batch, n_pad] fp16（v 链**读改写**）
    const float*  __restrict__ scale,    // [1]
    const int n_pad, const int k, const int batch, const int chain, const int xs)
{
    constexpr int BM = LR2_BM, BN = LR2_BN, BK = LR2_BK;
    constexpr int AS = BK * 2 + 16;
    constexpr int MT = BM / (16 * LR_NWM);
    constexpr int NT = BN / (8 * LR_NWN);
    __shared__ unsigned char as_[BM * AS];
    __shared__ unsigned char bs_[BN * AS];

    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int gID = lane >> 2, tig = lane & 3;
    const int row0 = blockIdx.x * BM, col0 = blockIdx.y * BN;
    const int wrow = (wid / LR_NWN) * (BM / LR_NWM);
    const int wcol = (wid % LR_NWN) * (BN / LR_NWN);
    const float sc = scale[0];

    float acc[MT][NT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][nt][q] = 0.f;

    for (int kt = 0; kt < k; kt += BK) {
        lr_load_tiles(as_, bs_, AS, X, W, xs, n_pad, k, batch, row0, col0, kt,
                      BM, BN, BK, tid, 256);
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < BK / 8; ++ks) {
            const int ko = ks * 16 + tig * 4;
            unsigned bv[NT];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                bv[nt] = *(const unsigned int*)(bs_ + (wcol + nt * 8 + gID) * AS + ko);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned av[2];
                av[0] = *(const unsigned int*)(as_ + (wrow + mt * 16 + gID) * AS + ko);
                av[1] = *(const unsigned int*)(as_ + (wrow + mt * 16 + gID + 8) * AS + ko);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) lr_mma_f16(acc[mt][nt], av, &bv[nt]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int r  = row0 + wrow + mt * 16 + gID;
            const int cc = col0 + wcol + nt * 8 + tig * 2;
            // v 链必须**先读后写**：先把两个元素读完再落 half2（同一元素只由本线程触及）。
            if (r < batch) {
                const float t0 = lr_chain_epi(chain, acc[mt][nt][0], r, cc,
                                              n_pad, bias, v_first, out, sc);
                const float t1 = lr_chain_epi(chain, acc[mt][nt][1], r, cc + 1,
                                              n_pad, bias, v_first, out, sc);
                *(__half2*)&out[(long long)r * n_pad + cc] = __floats2half2_rn(t0, t1);
            }
            if (r + 8 < batch) {
                const float t2 = lr_chain_epi(chain, acc[mt][nt][2], r + 8, cc,
                                              n_pad, bias, v_first, out, sc);
                const float t3 = lr_chain_epi(chain, acc[mt][nt][3], r + 8, cc + 1,
                                              n_pad, bias, v_first, out, sc);
                *(__half2*)&out[(long long)(r + 8) * n_pad + cc] = __floats2half2_rn(t2, t3);
            }
        }
}

// fp32 → fp16 一进一出（ffn_value 稠密 GEMM 的 A 侧降位）。
extern "C" __global__ void cast_f16(
    const float* __restrict__ a, __half* __restrict__ b, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __float2half_rn(a[i]);
}

// ★ Phase 2-3：ffn_value **稠密** fp16 张量核 GEMM（替代稀疏 SIMT 累加）。
//
// 语义：`x[b, m] += Σ_j r2_16[b, j] · W[m, j]`（W = ffn_value [C, fh] fp16 行主序，
// r2_16 = relu²(ffn_key) 的 fp16 降位，x 为 fp32 **就地累加**）。
//
// 为什么在 batch 大时**放弃稀疏**：稀疏内核按 r2 的非零列 gather 权重，权重流量
// = `(非零列数/fh)·52.4MB·batch`——B=256 时即使只有 23% 非零也是 13.4GB/层；
// 稠密 GEMM 靠 tiled smem 把权重**整 batch 只读 `batch/BM` 遍**（BM=128 ⇒ 2 遍）
// = 105MB/层，**两个数量级之差**。而稠密多算的 FLOP（13.4 GFLOP/层）在 fp16 张量核上
// 只有 0.125ms/层（107 TFLOPS 峰值口径）——**大 batch 下"算"是免费的，"读"才要钱**。
extern "C" __global__ void __launch_bounds__(256, LR_LB) ffn_value_gemm(
    const __half* __restrict__ W,   // [n_pad, k] fp16（ffn_value [C, fh]）
    const __half* __restrict__ X,   // [batch, xs] fp16（r2_16，xs = fh）
    float*        __restrict__ out, // [batch, n_pad] fp32（**就地累加**）
    const int n_pad, const int k, const int batch, const int xs)
{
    constexpr int BM = LR3_BM, BN = LR3_BN, BK = LR3_BK;
    constexpr int AS = BK * 2 + 16;
    constexpr int MT = BM / (16 * LR_NWM);
    constexpr int NT = BN / (8 * LR_NWN);
    __shared__ unsigned char as_[BM * AS];
    __shared__ unsigned char bs_[BN * AS];

    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int gID = lane >> 2, tig = lane & 3;
    const int row0 = blockIdx.x * BM, col0 = blockIdx.y * BN;
    const int wrow = (wid / LR_NWN) * (BM / LR_NWM);
    const int wcol = (wid % LR_NWN) * (BN / LR_NWN);

    float acc[MT][NT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][nt][q] = 0.f;

    for (int kt = 0; kt < k; kt += BK) {
        lr_load_tiles(as_, bs_, AS, X, W, xs, n_pad, k, batch, row0, col0, kt,
                      BM, BN, BK, tid, 256);
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < BK / 8; ++ks) {
            const int ko = ks * 16 + tig * 4;
            unsigned bv[NT];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                bv[nt] = *(const unsigned int*)(bs_ + (wcol + nt * 8 + gID) * AS + ko);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned av[2];
                av[0] = *(const unsigned int*)(as_ + (wrow + mt * 16 + gID) * AS + ko);
                av[1] = *(const unsigned int*)(as_ + (wrow + mt * 16 + gID + 8) * AS + ko);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) lr_mma_f16(acc[mt][nt], av, &bv[nt]);
            }
        }
        __syncthreads();
    }

    // 就地累加：每个 (r, cc) 只被本线程触及（无原子、无竞争）。
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int r  = row0 + wrow + mt * 16 + gID;
            const int cc = col0 + wcol + nt * 8 + tig * 2;
            if (r < batch) {
                float* o = out + (long long)r * n_pad + cc;
                o[0] += acc[mt][nt][0];
                o[1] += acc[mt][nt][1];
            }
            if (r + 8 < batch) {
                float* o = out + (long long)(r + 8) * n_pad + cc;
                o[0] += acc[mt][nt][2];
                o[1] += acc[mt][nt][3];
            }
        }
}
"#;

/// ffn_value_sparse_add batch CUDA kernel：r2 为 [batch, fh]，x 为 [batch, C]。
/// dispatch (fh/TILE, c/C_TILE, batch)。
const FFN_VALUE_SPARSE_BATCH_SRC: &str = r#"
extern "C" __global__ void ffn_value_sparse_add_batch(
    const float*    __restrict__ r2,          // [batch, fh] relu² 输出
    const __half*   __restrict__ value_tiled, // [fh*C] 平铺布局（共享）
    float*          __restrict__ x,           // [batch, C] 就地累加
    const int c,
    const int fh)
{
    // ★ 2026-09-21 重写：**每 (slot, c 分块) 一个 block 覆盖全部 fh**，于是每个 (b,c)
    // 只被一个 block 触及 ⇒ 全局原子操作整体消失。
    // 旧版 grid=(fh/128, c/256, batch)=204800 块，每块只算 128 个 f，
    // **同一 (b,c) 被 fh/128=80 个块以 atomicAdd 争抢**：52.4M 次/层、1.68G 次/步，
    // （B=256）`ABLATE` 实测占整步 33%（#2 瓶颈）。改成 f 维内循环后原子降为 0，
    // 且累加顺序固定 ⇒ 顺带把结果变回**确定性**。
    //
    // ★ 注意「f 索引」与「c 索引」是两个**独立**的循环维度：线程 `tid` 固定持有
    // c 对 `(c0, c0+1)`，必须对**组内每一个非零 f** 都累加（不是只加自己那一个 f）。
    constexpr int C_TILE = 256;
    constexpr int F_TILE = 128;
    __shared__ float r2_slice[F_TILE];
    __shared__ int   nnz_ids[F_TILE];
    __shared__ int   nnz_count;
    __shared__ int   warp_counts[F_TILE / 32];
    __shared__ int   warp_prefix[F_TILE / 32];

    const int c_block  = blockIdx.x;
    const int b        = blockIdx.y;
    const int tid      = threadIdx.x;              // 128 线程 ↔ F_TILE / 每线程 2 个 c
    const int lane     = tid & 31;
    const int warp     = tid >> 5;
    const int c_blocks = c / C_TILE;
    const int nfb      = fh / F_TILE;
    const float* r2b   = r2 + b * fh;
    const int c0       = c_block * C_TILE + tid * 2;
    // f 维分块数：块数太少时（小 batch）靠拆分 f 换并行度，代价是回到 atomicAdd
    // —— 但那时 (b,c) 的争抢方只有 nf 个，远小于旧版的 80 个。
    const int nf  = gridDim.z;
    const int fb0 = (int)((long long)blockIdx.z * nfb / nf);
    const int fb1 = (int)((long long)(blockIdx.z + 1) * nfb / nf);

    float acc0 = 0.f, acc1 = 0.f;
    for (int fb = fb0; fb < fb1; ++fb) {
        const float r2v = r2b[fb * F_TILE + tid];
        const bool  nz  = (r2v != 0.0f);
        const unsigned mask = __ballot_sync(0xffffffffu, nz);
        const int pos = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0) warp_counts[warp] = __popc(mask);
        __syncthreads();
        if (tid == 0) {
            int s = 0;
            #pragma unroll
            for (int w = 0; w < F_TILE / 32; ++w) {
                warp_prefix[w] = s;
                s += warp_counts[w];
            }
            nnz_count = s;
        }
        __syncthreads();
        if (nz) {
            const int dst = warp_prefix[warp] + pos;
            r2_slice[dst] = r2v;
            nnz_ids[dst]  = tid;
        }
        __syncthreads();
        const __half* wt = value_tiled
            + (long long)(fb * c_blocks + c_block) * F_TILE * C_TILE + tid * 2;
        const int nz_n = nnz_count;
        for (int i = 0; i < nz_n; ++i) {
            const float a = r2_slice[i];
            const __half* w = wt + (long long)nnz_ids[i] * C_TILE;
            acc0 += a * __half2float(w[0]);
            acc1 += a * __half2float(w[1]);
        }
        __syncthreads();                           // 复用 smem 前必须同步
    }
    float* xb = x + b * c;
    if (nf == 1) {
        xb[c0]     += acc0;
        xb[c0 + 1] += acc1;
    } else {
        atomicAdd(xb + c0, acc0);
        atomicAdd(xb + c0 + 1, acc1);
    }
}
"#;

/// rwkv_sample batch CUDA kernel：B slot 并行采样（每 slot 独立 logits/参数/seed）。
/// logits/temp/mask/counter 为 [batch, n]；token 为 [batch]；sampler 为 [batch, 10]；
/// hist 为 [batch, hist_stride]（每 slot 实际历史长度取自 sampler[7]，≤ hist_stride）。
/// dispatch (1, batch, 1)，block=112。
const SAMPLE_BATCH_SRC: &str = r#"
__device__ __forceinline__ float u01_batch(unsigned int s) {
    s += 0x9E3779B9u;
    unsigned int z = s;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z ^= z >> 16;
    return (float)z / 4294967296.0f;
}

// 第 i 个词表的「惩罚 → temperature」后的 logit。★ 2026-09-22：抽成函数是为了让
// 快速路径的 **两趟扫描都从 logits 现算**，从而彻底不物化 `temp`（省 1 写 1 读 = 134MB/槽）。
// 同一表达式算两次结果逐位相同（无随机性），故与旧版「先物化再读」等价。
__device__ __forceinline__ float sample_scaled1(
    const float* __restrict__ logits_b,
    const unsigned int* __restrict__ counter_b,
    int i, bool do_pen, float invT,
    float rep, float pres, float freq, float decay)
{
    float l = logits_b[i];
    if (do_pen) {
        const unsigned int cnt = counter_b[i];
        if (cnt > 0u) {
            if (rep != 1.0f) l = l > 0.0f ? l / rep : l * rep;
            if (pres != 0.0f) l -= pres;
            if (freq != 0.0f) l -= freq * powf((float)cnt, decay);
        }
    }
    return l * invT;
}

// 从 temp[0..n)（mask==0 的项）中取最大的 BS 个候选，按 (值降序, 索引升序) 排序后写入
// out_val/out_idx（长度 BS）；无剩余候选的空槽值为 -1e30 且排到末尾。
// c_val/c_idx 为长度 BS 的暂存区，不得与 out_val/out_idx 别名。
// 全块协同：内部含 __syncthreads，必须由所有线程统一调用。
__device__ __forceinline__ void sample_top_candidates(
    const float* __restrict__ temp,
    const float* __restrict__ mask,
    int n,
    float* c_val, int* c_idx,
    float* out_val, int* out_idx)
{
    constexpr int BS = 112;
    const int tid = threadIdx.x;
    float lm = -1e30f; int li = 0;
    for (int i = tid; i < n; i += BS) {
        const float v = temp[i];
        if (mask[i] == 0.0f && v > lm) { lm = v; li = i; }
    }
    c_val[tid] = lm;
    // 空槽给唯一哨兵索引：否则多个 (-1e30, 0) 撞秩，排序结果错乱。
    c_idx[tid] = (lm > -1e29f) ? li : (n + tid);
    __syncthreads();
    // O(BS^2) 并行定秩（每线程只扫 BS 个 shared 元素），避免为 112 元素引入位排序。
    const float mv = c_val[tid];
    const int   mi = c_idx[tid];
    int rank = 0;
    for (int j = 0; j < BS; j++) {
        const float vj = c_val[j];
        const int   ij = c_idx[j];
        if (vj > mv || (vj == mv && ij < mi)) rank++;
    }
    out_val[rank] = mv;
    out_idx[rank] = mi;
    __syncthreads();
}

extern "C" __global__ void rwkv_sample_batch(
    const float*      __restrict__ logits,   // [batch, n]
    float*            __restrict__ token,    // [batch] 写入索引的 f32 位模式
    float*            __restrict__ temp,     // [batch, n] 工作区
    float*            __restrict__ mask,     // [batch, n] 工作区
    unsigned int*     __restrict__ counter,  // [batch, n] 直方图
    const float*      __restrict__ sampler,  // [batch, 10] 参数
    const unsigned int* __restrict__ hist,   // [batch, hist_stride] 历史 token
    const int n,
    const int hist_stride)
{
    const int tid = threadIdx.x;
    const int b   = blockIdx.y;
    const int bo  = b * n;
    const float* sampler_b = sampler + b * 10;
    constexpr int BS = 112;
    // ★ smem 是占用率的唯一瓶颈（2026-09-22 实测）：`s_val/s_idx` 是 **线程私有**的
    // top-K 候选表，体积 = `BS × MAXK × 8B`。MAXK=50 时 44.8KB/块 ⇒ Turing 64KB/SM
    // **只放得下 1 块 = 4 warp = 6.25% 占用率**，而本内核是纯流式扫描（65536 词表 × 3 遍），
    // 靠的是内存级并行 ⇒ 实测仅 39GB/s（8% 带宽）。诊断：MAXK 50→8（smem 44.8→7.2KB）
    // 单次 **10.42 → 5.41ms（−48%）**。故 MAXK 改为 `#define` 注入，host 按 top_k 取最小够用档。
#ifndef SAMPLE_MAXK
#define SAMPLE_MAXK 50
#endif
    constexpr int MAXK = SAMPLE_MAXK;
    __shared__ float s_val[BS][MAXK];
    __shared__ int   s_idx[BS][MAXK];
    __shared__ float s_topval[MAXK];
    __shared__ int   s_topidx[MAXK];
    __shared__ float s_sorted[MAXK];
    __shared__ int   s_sortedidx[MAXK];
    __shared__ float s_fval[BS];
    __shared__ int   s_fidx[BS];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_u;
    __shared__ float s_threshold;
    __shared__ float s_maxkeep;   // 保留集合的最大值（= softmax 的 m，省掉一趟 max 扫描）
    __shared__ float g_cutoff;
    __shared__ float s_cum;
    __shared__ int   s_consumed;
    __shared__ int   s_used;
    __shared__ int   s_done;

    const float temperature = sampler_b[0];
    const unsigned int top_k = __float_as_uint(sampler_b[1]);
    const float top_p = sampler_b[2];
    const unsigned int seed = __float_as_uint(sampler_b[3]);
    const float rep = sampler_b[4];
    const float freq = sampler_b[5];
    const float pres = sampler_b[6];
    const unsigned int hist_len = __float_as_uint(sampler_b[7]);
    const float decay = sampler_b[8];   // 惩罚衰减指数（1.0 = 退化为 freq*cnt）
    const bool do_topk = (top_k > 0u && top_k < (unsigned int)n);
    const int K = do_topk ? (int)top_k : 0;

    float* temp_b = temp + bo;
    float* mask_b = mask + bo;
    unsigned int* counter_b = counter + bo;
    const float* logits_b = logits + bo;
    const unsigned int* hist_b = hist + b * hist_stride;

    // ★ 2026-09-22 融合（Phase 2-②）：旧版把「载入 logits」「惩罚」「乘 temperature」
    // 拆成三趟全词表（每趟 1 读 1 写 temp），加上后面的「掩码」「求 max」「exp+sum」
    // 「归一化」共 **7 趟**，现在压到 **3 趟**。
    // ⚠️ **已实测的两条否定结论（勿重试）**：
    //  ① 继续把「缩放值不物化、两趟都从 logits 现算」压到 2 趟 ⇒ 反而 8.05 → 9.62ms。
    //  ② 只做趟数融合（不动别的）⇒ 10.38 → 10.16ms，仅 −2%。
    //  ⇒ 本内核**不是带宽/趟数受限**，而是**每次迭代的成本（指令数 + 依赖链）受限**
    //    （3 趟共 335MB / 8.05ms = 42GB/s，只有 8% 带宽）。真正有效的是下面那条
    //    「阈值镜像到寄存器」——它砍掉的是主循环的依赖链，而不是流量。
    float invT = 1.0f / temperature;
    if (!(temperature > 0.0f)) invT = 1.0f;
    const bool do_pen = hist_len > 0u && (rep != 1.0f || freq != 0.0f || pres != 0.0f);
    if (do_pen) {
        // 直方图必须先建好（counter_b 与 temp_b 同为 [n] 量级，但只在惩罚开启时付这笔钱）
        for (int i = tid; i < n; i += BS) counter_b[i] = 0u;
        __syncthreads();
        for (int h = tid; h < (int)hist_len; h += BS) {
            // hist_len = 本步之前的历史长度（本步刚采样的 token 尚未计入）；
            // 越界 id 夹到词表末位，避免脏历史写出 counter_b 之外。
            unsigned int hh = hist_b[h];
            if (hh >= (unsigned int)n) hh = (unsigned int)n - 1u;
            atomicAdd(&counter_b[hh], 1u);
        }
        __syncthreads();
    }
    for (int i = tid; i < n; i += BS)
        temp_b[i] = sample_scaled1(logits_b, counter_b, i, do_pen, invT,
                                   rep, pres, freq, decay);
    __syncthreads();
    const bool fast_k = (K > 0 && K <= MAXK);

    if (K > 0 && K <= MAXK) {
        // ================= 快速路径：单遍 top-K =================
        // ★ 2026-09-22：表内最小值 `thr` 镜像到**寄存器**。旧版每轮都读 `s_val[tid][K-1]`
        // 再拿它做分支判据 ⇒ smem 读落在主循环的关键路径上，编译器无法把后续全局载入
        // 提前发出（本内核只有 4 warp/SM，全靠 MLP 掩盖延迟）。改成寄存器判据后主循环
        // 是「纯载入 + 比较」，只有真正插入时才触碰 smem。实测 10.16 → 8.05ms（−21%）。
        for (int j = 0; j < MAXK; j++) { s_val[tid][j] = -1e30f; s_idx[tid][j] = -1; }
        float thr = -1e30f;
        for (int i = tid; i < n; i += BS) {
            const float v = temp_b[i];
            if (v > thr) {
                int pos = K - 1;
                while (pos > 0 && v > s_val[tid][pos - 1]) {
                    s_val[tid][pos] = s_val[tid][pos - 1];
                    s_idx[tid][pos] = s_idx[tid][pos - 1];
                    --pos;
                }
                s_val[tid][pos] = v;
                s_idx[tid][pos] = i;
                thr = s_val[tid][K - 1];
            }
        }
        __syncthreads();

        if (tid == 0) {
            auto sift = [&](int i, int h) {
                while (true) {
                    int l = 2 * i + 1, r = 2 * i + 2, m = i;
                    if (l < h && s_topval[l] < s_topval[m]) m = l;
                    if (r < h && s_topval[r] < s_topval[m]) m = r;
                    if (m == i) break;
                    float tv = s_topval[i]; s_topval[i] = s_topval[m]; s_topval[m] = tv;
                    int ti = s_topidx[i]; s_topidx[i] = s_topidx[m]; s_topidx[m] = ti;
                    i = m;
                }
            };
            for (int j = 0; j < K; j++) { s_topval[j] = s_val[0][j]; s_topidx[j] = s_idx[0][j]; }
            for (int j = K / 2 - 1; j >= 0; j--) sift(j, K);
            for (int th = 1; th < BS; th++) {
                for (int j = 0; j < K; j++) {
                    const float v = s_val[th][j];
                    if (v <= -1e29f) break;
                    if (v > s_topval[0]) {
                        s_topval[0] = v; s_topidx[0] = s_idx[th][j];
                        sift(0, K);
                    }
                }
            }
            for (int r = K; r > 0; r--) {
                s_sorted[r - 1] = s_topval[0];
                s_sortedidx[r - 1] = s_topidx[0];
                s_topval[0] = s_topval[r - 1];
                s_topidx[0] = s_topidx[r - 1];
                sift(0, r - 1);
            }
            s_threshold = s_sorted[K - 1];  // 第 K 大（降序末位）= 保留边界
            s_maxkeep   = s_sorted[0];      // 保留集合的最大值 = softmax 的 m（top-K 必含全局最大）
        }
        __syncthreads();
        // ★ 这里原本还有一趟「低于阈值写 -1e30」的掩码（1 读 1 写全词表）——已并入下面的
        // 融合 softmax 一趟（`(v < s_threshold) ? 0 : exp(v - m)`），语义逐位等价。
    } else {
        // ================= 兜底路径 =================
        if (tid == 0) s_threshold = -1e30f;   // 无 top-k 时阈值不生效（旧版此时 s_threshold 未初始化）
        __syncthreads();
        if (do_topk) {
            for (int i = tid; i < n; i += BS) mask_b[i] = 0.0f;
            __syncthreads();
            for (unsigned int round = 0u; round < top_k; round++) {
                float lm = -1e30f; int li = 0;
                for (int i = tid; i < n; i += BS) {
                    if (mask_b[i] == 0.0f && temp_b[i] > lm) { lm = temp_b[i]; li = i; }
                }
                s_fval[tid] = lm; s_fidx[tid] = li;
                __syncthreads();
                // BS=112 非 2 的幂：step 序列 56,28,14,7,3,1 会在 14→7→3→1 段孤儿化
                // 索引 6 与 2 的结果（s_fval[0] 只是约 1/16 元素的最大值 → 阈值偏低）。
                // 修法同 softmax：先把尾部 [P2, BS) 折进 [0, BS-P2)，再 2 幂树归约。
                {
                    constexpr int P2 = 64;
                    if (tid >= P2 && tid < BS) {
                        const float bv = s_fval[tid];
                        const int   bi = s_fidx[tid];
                        const float av = s_fval[tid - P2];
                        const int   ai = s_fidx[tid - P2];
                        if (bv > av || (bv == av && bi < ai)) {
                            s_fval[tid - P2] = bv; s_fidx[tid - P2] = bi;
                        }
                    }
                    __syncthreads();
                    for (int step = P2 >> 1; step > 0; step >>= 1) {
                        if (tid < step) {
                            const float bv = s_fval[tid + step];
                            const int   bi = s_fidx[tid + step];
                            if (bv > s_fval[tid] || (bv == s_fval[tid] && bi < s_fidx[tid])) {
                                s_fval[tid] = bv; s_fidx[tid] = bi;
                            }
                        }
                        __syncthreads();
                    }
                }
                if (tid == 0) { s_threshold = s_fval[0]; mask_b[s_fidx[0]] = 1.0f; }
                __syncthreads();
            }
        }
        // 兜底路径的 m：单趟归约。旧版是「先写 -1e30 掩码（1 读 1 写）再扫 max（1 读）」
        // 两趟；掩码本身不改变最大值（阈值 ≤ 全局最大），故一趟求 max 即可，
        // 掩码已并入下面融合 softmax 的那一趟。
        {
            float lm = -1e30f;
            for (int i = tid; i < n; i += BS) lm = fmaxf(lm, temp_b[i]);
            s_fval[tid] = lm;
            __syncthreads();
            constexpr int P2 = 64;  // BS=112 → 64 + 48（非 2 幂安全归约，同下方说明）
            if (tid >= P2 && tid < BS) s_fval[tid - P2] = fmaxf(s_fval[tid - P2], s_fval[tid]);
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] = fmaxf(s_fval[tid], s_fval[tid + step]);
                __syncthreads();
            }
            if (tid == 0) s_maxkeep = s_fval[0];
            __syncthreads();
        }
    }

    // 7. softmax（★ 2026-09-22 融合）：**掩码 + exp + 求和 一趟完成**。
    // 旧版是「掩码（1 读 1 写）+ 求 max（1 读）+ exp/sum（1 读 1 写）+ 归一化（1 读 1 写）」
    // 四趟；现在 m 由上面两条路径的 top-K 归并直接给出（保留集合的最大值），
    // 低于阈值不再写 -1e30 而是当场取 0（`exp(-1e30 - m) == 0`，逐位等价）。
    // 归约为非幂 block 安全版：BS=112 非 2 的幂，纯树归约（step 减半）会让
    // 部分 warp 的结果成为"孤儿"（如 step=7 时 s_fval[5..6] 不再被合并），
    // sum 漏加 → 概率偏小。修法：先把尾部 [P2, BS) 并入 [0, BS-P2)（P2 = ≤BS 的最大 2 幂）。
    const float m = s_maxkeep;
    float ssum = 0.0f;
    for (int i = tid; i < n; i += BS) {
        const float v = temp_b[i];
        const float e = (v < s_threshold) ? 0.0f : expf(v - m);
        temp_b[i] = e;
        ssum += e;
    }
    s_fval[tid] = ssum;
    __syncthreads();
    {
        constexpr int P2 = 64;
        if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
        __syncthreads();
        for (int step = P2 >> 1; step > 0; step >>= 1) {
            if (tid < step) s_fval[tid] += s_fval[tid + step];
            __syncthreads();
        }
    }
    const float total = s_fval[0];
    // 归一化（★ 省掉一整趟全词表）：只有 §8 的**兜底分支**（`sample_top_candidates` 走全表
    // 并按概率累计到 top_p）才需要整表归一化值；快速分支与 §9 都只碰 top-K 那 K 个索引
    // ⇒ 只归一化这 K 个即可。`need_full_norm` 与线程无关，故分支内不出现 `__syncthreads`。
    {
        const bool need_full_norm = (top_p > 0.0f && top_p < 1.0f) && !fast_k;
        if (total > 0.0f) {
            const float it = 1.0f / total;
            if (need_full_norm) {
                for (int i = tid; i < n; i += BS) temp_b[i] *= it;
            } else if (fast_k && tid == 0) {
                for (int j = 0; j < K; j++) temp_b[s_sortedidx[j]] *= it;
            }
        }
    }
    __syncthreads();

    // 8. top-p
    if (top_p > 0.0f && top_p < 1.0f) {
        if (K > 0 && K <= MAXK) {
            if (tid == 0) {
                // s_sortedidx 为降序（[0] 最大），必须从 [0] 起累积——旧版从 K-1
                // （最小）起累积，与其自身注释「从最大概率起累积」相悖：截断阈值
                // 偏低 → 保留集合偏大（多峰分布下与兜底路径结果不一致）。
                float cum = 0.0f, cutoffv = -1e30f;
                for (int j = 0; j < K; j++) {
                    const int idx = s_sortedidx[j];
                    cum += temp_b[idx];
                    cutoffv = temp_b[idx];
                    if (cum >= top_p) break;
                }
                g_cutoff = cutoffv;
            }
        } else {
            // 兜底（无 top-k 或 top_k > MAXK）：每轮取 BS 个候选，而非旧版每轮 1 个。
            // 旧版在平坦分布下要跑满 512 轮 × 全词表扫描（实测 51.8 ms/token）。
            // 暂存复用快速路径的 s_val/s_idx（此分支下二者未被使用）。
            float* c_val = (float*)s_val;
            int*   c_idx = (int*)s_idx;
            float* o_val = (float*)s_val + BS;
            int*   o_idx = (int*)s_idx + BS;
            for (int i = tid; i < n; i += BS) mask_b[i] = 0.0f;
            __syncthreads();
            if (tid == 0) { g_cutoff = 0.0f; s_cum = 0.0f; s_consumed = 0; s_done = 0; }
            __syncthreads();
            // 退出条件全块统一（shared），避免 __syncthreads 分歧死锁（同单流版）
            while (!s_done) {
                sample_top_candidates(temp_b, mask_b, n, c_val, c_idx, o_val, o_idx);
                if (tid == 0) {
                    int used = 0;
                    for (int j = 0; j < BS && s_consumed < 512; j++) {
                        const float v = o_val[j];
                        if (v <= -1e29f) break;  // 候选耗尽
                        s_cum += v;
                        g_cutoff = v;
                        ++used; ++s_consumed;
                        if (s_cum >= top_p) break;
                    }
                    s_used = used;
                    if (used == 0 || s_cum >= top_p || s_consumed >= 512) s_done = 1;
                }
                __syncthreads();
                // 标记本轮已消费候选（即将退出时多标也无害：mask 之后不再使用）
                if (tid < s_used) mask_b[o_idx[tid]] = 1.0f;
                __syncthreads();
            }
            __syncthreads();
        }
        for (int i = tid; i < n; i += BS) if (temp_b[i] < g_cutoff) temp_b[i] = 0.0f;
        __syncthreads();
    }

    // 9. 采样
    if (K > 0 && K <= MAXK) {
        if (tid == 0) {
            float total = 0.0f;
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp_b[idx] > 0.0f) total += temp_b[idx];
            }
            const float u = u01_batch(seed) * total;
            float acc = 0.0f;
            int chosen = s_sortedidx[K - 1];
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp_b[idx] > 0.0f) {
                    acc += temp_b[idx];
                    if (acc > u) { chosen = idx; break; }
                }
            }
            token[b] = __int_as_float(chosen);
        }
    } else {
        float ts = 0.0f;
        for (int i = tid; i < n; i += BS) ts += temp_b[i];
        s_fval[tid] = ts;
        __syncthreads();
        {
            constexpr int P2 = 64;
            if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] += s_fval[tid + step];
                __syncthreads();
            }
        }
        const float total = s_fval[0];
        __syncthreads();
        if (tid == 0) s_u = u01_batch(seed) * total;
        __syncthreads();
        // ★ 2026-09-23：与单流 `rwkv_sample` 同源的**并行定位采样索引**（见那里的完整说明）。
        // 旧版是 `if (tid == 0) for (i = 0; i < n; i++) { acc += temp_b[i]; if (acc > s_u) break; }`
        // —— 单线程按 i 升序串行读全表，top-p 掩码后非零项只有几百个、散布全词表
        // ⇒ 要空跑约 n/2 次依赖延迟的标量载入。单流版实测 **1.28ms/token**
        // （`SAMP_AB=1` 消融 1.534 → 0.257ms）；批量版每槽一次、B 个槽并行。
        // 两段式：① 每线程一段连续区间（4 路 ILP）→ s_blk；② tid0 定位跨块；
        // ③ 全块对该块再切分（CH2），tid0 只扫 ≤CH2 个。
        // ⚠️ 求和顺序与旧版不同（相对误差 ~1e-6），刀锋处才可能换 token（~1e-6/步）。
        {
            __shared__ int   s_cross_b;
            __shared__ float s_base_b;
            const int CH = (n + BS - 1) / BS;
            {
                const int b0 = tid * CH;
                const int b1 = min(b0 + CH, n);
                float cs = 0.0f;
                int i = b0;
                for (; i + 3 < b1; i += 4)
                    cs += (temp_b[i] + temp_b[i + 1]) + (temp_b[i + 2] + temp_b[i + 3]);
                for (; i < b1; i++) cs += temp_b[i];
                s_fval[tid] = cs;
            }
            __syncthreads();
            if (tid == 0) {
                float acc = 0.0f;
                int cross = BS - 1;
                for (int t = 0; t < BS; t++) {
                    const float nxt = acc + s_fval[t];
                    if (nxt > s_u) { cross = t; break; }
                    acc = nxt;
                }
                s_cross_b = cross;
                s_base_b = acc;
            }
            __syncthreads();
            const int cb0 = s_cross_b * CH;
            const int cb1 = min(cb0 + CH, n);
            const int CH2 = (CH + BS - 1) / BS;
            {
                const int q0 = cb0 + tid * CH2;
                const int q1 = min(q0 + CH2, cb1);
                float cs = 0.0f;
                for (int i = q0; i < q1; i++) cs += temp_b[i];
                s_fval[tid] = cs;
            }
            __syncthreads();
            if (tid == 0) {
                float acc = s_base_b;
                int chosen = n - 1;
                for (int t = 0; t < BS; t++) {
                    const float nxt = acc + s_fval[t];
                    if (nxt > s_u) {
                        const int r0 = cb0 + t * CH2;
                        const int r1 = min(r0 + CH2, cb1);
                        for (int i = r0; i < r1; i++) {
                            acc += temp_b[i];
                            if (acc > s_u) { chosen = i; break; }
                        }
                        break;
                    }
                    acc = nxt;
                }
                token[b] = __int_as_float(chosen);
            }
        }
    }
}
"#;

/// record_tokens batch CUDA kernel：把 in_tok[b] 各自追加到
/// out_seq[b*stride + atomicAdd(&cnt[b])]（每 slot 独立段独立计数）。
/// dispatch (batch, 1, 1)，每 block 单线程。
const RECORD_TOKENS_SRC: &str = r#"
extern "C" __global__ void rwkv_record_tokens(
    const unsigned int* __restrict__ in_tok,  // [batch] token（f32 位模式）
    unsigned int* __restrict__ out_seq,       // [batch, stride] 序列缓冲
    unsigned int* __restrict__ cnt,           // [batch] 计数器（各自原子自增）
    const int stride)
{
    const int b = blockIdx.x;
    const unsigned int i = atomicAdd(&cnt[b], 1u);
    out_seq[b * stride + i] = in_tok[b];
}
"#;

/// norm_lerp6 CUDA kernel：单 token 深度融合。
/// 语义对齐 Vulkan `norm_lerp6.comp`（f32 张量）：
///   ln1 = (x - mean) * inv_std * gamma + beta；mean/inv_std 对全 C 归约。
///   o_*[i] = ln1 + x_*[i] * (prev[i] - ln1)；state[i] = ln1。
/// 多 block 并行：grid = ceil(c/BLOCK)。每个 block 先对全 C 做**冗余归约**
/// （x 仅 ~10KB，命中 L2，各 block 独立算出同一 mean/inv_std），再并行 apply
/// 本 block 负责的 C 片段。避免单 block 时 67/68 个 SM 空闲的延迟瓶颈。
const NORM_LERP6_SRC: &str = r#"
extern "C" __global__ void norm_lerp6(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ xr,
    const float* __restrict__ xw,
    const float* __restrict__ xk,
    const float* __restrict__ xv,
    const float* __restrict__ xa,
    const float* __restrict__ xg,
    float* __restrict__ or_,
    float* __restrict__ ow,
    float* __restrict__ ok,
    float* __restrict__ ov,
    float* __restrict__ oa,
    float* __restrict__ og,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;

    // Phase 1：每个 block 独立对全 C 做冗余归约（x 命中 L2，开销小）。
    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        sq  += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    // Phase 2：每个 block apply 自己负责的 C 片段（gx*BLOCK .. min+BLOCK）。
    const int start = gx * blockDim.x;
    const int end   = min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[i];
        const float ln1  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[i];
        or_[i] = ln1 + xr[i] * (prev - ln1);
        ow[i]  = ln1 + xw[i] * (prev - ln1);
        ok[i]  = ln1 + xk[i] * (prev - ln1);
        ov[i]  = ln1 + xv[i] * (prev - ln1);
        oa[i]  = ln1 + xa[i] * (prev - ln1);
        og[i]  = ln1 + xg[i] * (prev - ln1);
        state[i] = ln1;
    }
}
"#;

/// cmix_norm_lerp CUDA kernel：channel-mix 深度融合。
/// 语义对齐 Vulkan `cmix_norm_lerp.comp`（f32 张量）：
///   ln2 = (x - mean) * inv_std * gamma + beta；out_xb[i] = ln2 + coeff[i]*(prev-ln2)；state[i]=ln2。
/// 多 block 并行：grid = ceil(c/BLOCK)，每个 block 冗余归约全 C 后分段 apply。
const CMIX_NORM_LERP_SRC: &str = r#"
extern "C" __global__ void cmix_norm_lerp(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ coeff,
    float* __restrict__ out_xb,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        sq  += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    // ★ 2026-09-23：`gridDim.x == 1` 时（单流路径，`CMIX_NORM_BLK` 默认 1024）
    // 本块独自覆盖整个 [0,c)；多块时才按 `gx` 分段。
    // 旧版恒为 `end = min(start + blockDim.x, c)`，grid=1 时会漏掉 c > blockDim.x 的部分。
    const int start = gx * blockDim.x;
    const int end   = (gridDim.x == 1u) ? c : min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[i];
        const float ln2  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[i];
        out_xb[i] = ln2 + coeff[i] * (prev - ln2);
        state[i] = ln2;
    }
}
"#;

/// norm CUDA kernel：per-row layer norm + affine。
/// 语义对齐 Vulkan `norm.comp`（f32 输入/输出，affine）：
///   layout x[b][head][c]，gamma/beta 跨 batch 共享（[head][c]）。
///   每个 block（256 线程）归一化一个 (head,batch) 行。
const NORM_SRC: &str = r#"
extern "C" __global__ void rwkv_norm(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    const int c,
    const int h,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;

    const int row   = blockIdx.x;
    const int b     = row / h;
    const int hh    = row - b * h;
    const int x_base = b * c * h + hh * c;
    const int g_base = hh * c;

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[x_base + i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        sq  += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[x_base + i];
        y[x_base + i] = (v - mean) * inv_std * gamma[g_base + i] + beta[g_base + i];
    }
}
"#;

/// fuse_ka_dplr_norm CUDA kernel：fuse_ka + dplr(S 更新) + group_norm + sum_rk_rk 一次 dispatch。
/// 语义对齐 Vulkan `fuse_ka_dplr_norm.comp`（单 token 路径）：
///   kk_l2_i = normalize(k_i * k_k_i)；b_i = -kk_l2_i * a_i
///   k_mod_i = k_i * (1 + k_a_i * (a_i - 1))
///   S 更新：S[row,j] = S[row,j]*w[j] + sa[row]*b[j] + v[row]*k_mod[j]；y[row] = S@r
///   y_norm[row] = group_norm(y) + sum(r*k_mod*r_k) * v[row]
/// 每个 block 处理一个 (head,batch)；128 线程 = 4 warp。
/// warp-per-row 映射：warp 串行处理 row ≡ warp (mod 4) 的状态行，lane ↔ 连续列
/// （j0=lane、j1=lane+32）——state 行访问每条 warp 指令 128B 连续（100% coalesced；
/// 旧映射 row=t/2 隔行分片时每 2 lane 才 8B 连续，仅 25% sector 利用率）。
/// sa/y 行内归约用 warp 蝶形 shuffle（替代 shared 树归约，行循环内零 barrier）。
/// a/v/w 为 fp16，其余 f32。
const FUSE_KA_DPRL_NORM_SRC: &str = r#"
// ★ 2026-09-23：块内 warp 数（host 按 batch 注入；缺省 4 = 旧行为）。
// 状态行循环 `row = warp; row += KAW` ⇒ 每 warp 串行行数 = n/KAW。
// 小 batch（块数只有 H=40）时抬到 8/16 以缩短串行链、提高块内并发。
#ifndef KAW
#define KAW 4
#endif
// ★ Phase 2-2（`WKV_STATE_F16`）：WKV 状态 `s` 可存 fp16 以砍半其读写流量
// （状态是 [batch][H][N][N]，本模型 H=40/N=64/B=256 ⇒ 84MB/层、读+写 168MB）。
// **计算全程仍是 fp32**，只在载入/回存时降位。
// ⚠️ **修正 2026-09-22**：旧注释说「fp32 档 450GB/s 已贴 roofline」——对 fp32 成立
// （336MB/0.75ms = 448GB/s），但**换 fp16 后只跑到 301GB/s（168MB/0.557ms）**，
// 说明减半字节后**已从带宽受限转为「每块固定开销」受限**（每块 8KB 状态、4 warp、
// 13 个 `__syncthreads`、每行 2 条 5 级蝶形链）。⇒ 再砍字节收益递减，
// 要提速得动「每块摊到的状态量」。**已实测无效的两条路**：状态存取向量化
// （`half2`/`float2`，只 +0.55% 且改了归约顺序）、把 `asz` 类一次性读搬 smem。
// 用 typedef 让同一份源码出两个变体（host 侧按 `#define` 注入 + 各自独立 key 缓存）。
#ifdef WKV_S16
typedef __half stype;
__device__ __forceinline__ float st_ld(const stype* p) { return __half2float(*p); }
__device__ __forceinline__ void  st_st(stype* p, float v) { *p = __float2half_rn(v); }
#else
typedef float stype;
__device__ __forceinline__ float st_ld(const stype* p) { return *p; }
__device__ __forceinline__ void  st_st(stype* p, float v) { *p = v; }
#endif

extern "C" __global__ void fuse_ka_dplr_norm(
    stype* __restrict__ s,          // [batch][head][n][n] 状态（in-place 更新）
    const float* __restrict__ k,    // [batch][head][n]
    const float* __restrict__ kk,   // k_k [head][n]
    const __half* __restrict__ a,   // [batch][head][n]
    const float* __restrict__ ka,   // k_a [head][n]
    const float* __restrict__ r,    // [batch][head][n]
    const __half* __restrict__ v,   // [batch][head][n]
    const __half* __restrict__ w,   // [batch][head][n]
    const float* __restrict__ gamma,// [head][n]
    const float* __restrict__ beta, // [head][n]
    const float* __restrict__ rk,   // r_k [head][n]
    float* __restrict__ km,         // k_mod [batch][head][n]
    float* __restrict__ /*y*/,      // [batch][head][n]（本融合 kernel 不再写 y）
    float* __restrict__ yn,         // y_norm [batch][head][n]
    const int h,
    const int n,
    const float eps,
    const float gn_eps)
{
    constexpr int N_MAX = 64;

    __shared__ float sh_a[N_MAX];
    __shared__ float sh_b[N_MAX];
    __shared__ float sh_k[N_MAX];
    __shared__ float sh_w[N_MAX];
    __shared__ float sh_r[N_MAX];
    __shared__ float yv[N_MAX];
    // ★ 2026-09-23：`sq` 按块内线程数定尺（`KAW·32`），相位 0 的树归约起点随之改变。
    __shared__ float sq[KAW * 32];
    __shared__ float sqY[N_MAX];
    __shared__ float sqY2[N_MAX];
    __shared__ float ssRed[N_MAX];
    __shared__ float mean;
    __shared__ float inv_std;
    __shared__ float s_acc;

    const int head  = blockIdx.x;
    const int batch = blockIdx.y;
    const int t     = threadIdx.x;
    const int warp  = t >> 5;    // 0..KAW-1
    const int lane  = t & 31;    // 0..31
    // ⚠️ 旧版这里是 `if (t >= 2*n) return;`（KAW=4、n=64 时恒 false）。
    // 加大 KAW 后该早退会**把参与相位 0 树归约的线程砍掉**，故改为就地归零：
    // 未落到 [0,2n) 的槽写 0，归约仍是「每项算两遍 ⇒ 2 倍和」，与 CPU 参考口径不变。

    const int v_base = batch * (h * n) + head * n;
    const int w_base = head * n;
    const int s_base = batch * (h * n * n) + head * (n * n);

    // Phase 0：L2 范数（保持旧版冗余归约：每行 2 线程各算一次 → 2 倍和，与 CPU 参考一致）
    {
        float kk2 = 0.0f;
        if (t < 2 * n) {
            const int r2 = t >> 1;
            const float k_i  = k[v_base + r2];
            const float kk_i = k_i * kk[w_base + r2];
            kk2 = kk_i * kk_i;
        }
        sq[t] = kk2;
    }
    __syncthreads();
    #pragma unroll
    for (int step = (KAW * 32) >> 1; step > 0; step >>= 1) {
        if (t < step) sq[t] += sq[t + step];
        __syncthreads();
    }
    const float inv_norm = 1.0f / fmaxf(sqrtf(sq[0]), eps);

    // Phase 1：t < n 的线程一人一列填充按列 shared（lane↔列，coalesced）
    if (t < n) {
        const float kc  = k[v_base + t];
        const float kkc = kc * kk[w_base + t];
        const float ac  = __half2float(a[v_base + t]);
        const float kl2 = kkc * inv_norm;
        sh_a[t] = kl2;
        sh_b[t] = -kl2 * ac;
        sh_k[t] = kc * (1.0f + ka[w_base + t] * (ac - 1.0f));
        sh_w[t] = __half2float(w[v_base + t]);
        sh_r[t] = r[v_base + t];
    }
    __syncthreads();
    if (t < n) km[v_base + t] = sh_k[t];

    // Phase 2+3：warp-per-row —— 每 warp 串行处理 row ≡ warp (mod 4) 的状态行，
    // lane ↔ 连续列（j0=lane、j1=lane+32），S 行元素寄存器化（每 lane 2 个），
    // 一遍读一遍写；sa/y 行内归约用 warp 蝶形 shuffle（行循环内零 barrier）。
    // ⚠️ **反例留档（勿重试）**：曾把列映射改成 `j0 = 2·lane / j1 = j0+1` 以便
    // `half2` 向量化存取（指令数减半、128B 全合并）⇒ 端到端只 **+0.55%**
    // （2923.2 → 2939.4），但**改变了蝶形归约的求和顺序**（token 指纹 `0xd182bf8/0x7c02`
    // → `0xd13f69a/0x1b44`）。**收益与「动数值口径」不成比例，已回退。**
    const int j0 = lane;
    const int j1 = lane + 32;
    const bool has_j1 = j1 < n;
    // `#pragma unroll 2`：让两行（相互独立）的蝶形链交错，隐藏 5 级 shfl 的延迟。
    // 逐行的算术完全不变 ⇒ 数值逐位一致（只改调度）。
    // ★ 2026-09-23：块内 warp 数（`KAW`）。**B=1 时从 4 抬到 8/16**。
    //
    // 病灶：本内核 grid = (H, batch)，B=1 时**只有 40 个块**（68 个 SM 有 28 个空转），
    // 而每块内部是「4 warp × 16 行串行」的状态行循环，每行 2 条 5 级蝶形链 +
    // 2 载入 2 存回 —— 整条链的延迟**没有任何东西掩盖**。
    // 实测 B=1：0.51 ms/token、每层 21.6 µs；而状态流量只有 640KB/层（读+写）
    // ⇒ 有效带宽 **~40 GB/s**（B=256 时同内核 450 GB/s 贴 roofline）。
    // 即：不是带宽问题，是**每块串行链长 + 块数太少**。
    //
    // 改法：`row = warp; row < n; row += KAW` ⇒ 每 warp 的串行行数从 `n/4` 降到 `n/KAW`
    // （KAW=8 时 16→8，KAW=16 时 →4），块内并发 warp 数翻倍/翻四倍。
    // 状态行访问模式不变（lane ↔ 连续列，每 warp 指令 128B 全合并）。
    // 数值口径不变：相位 0 的「每项算两遍 ⇒ 2 倍和」与各相位的归约结构逐字保留。
    #pragma unroll 2
    for (int row = warp; row < n; row += KAW) {
        const int s_row_base = s_base + row * n;
        const float v_i = __half2float(v[v_base + row]);
        float s0 = st_ld(&s[s_row_base + j0]);
        float s1 = has_j1 ? st_ld(&s[s_row_base + j1]) : 0.0f;
        // sa[row] = sum_j S[row,j] * kk_l2[j]：本 lane 2 列部分和 + 蝶形归约（全 lane 得全和）
        float sa_part = s0 * sh_a[j0];
        if (has_j1) sa_part = fmaf(s1, sh_a[j1], sa_part);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sa_part += __shfl_xor_sync(0xffffffffu, sa_part, mask);
        }
        // S[row,j] = S[row,j]*w[j] + sa[row]*b[j] + v[row]*k_mod[j]
        s0 = s0 * sh_w[j0] + sa_part * sh_b[j0] + v_i * sh_k[j0];
        if (has_j1) s1 = s1 * sh_w[j1] + sa_part * sh_b[j1] + v_i * sh_k[j1];
        // y[row] = sum_j S_new[row,j] * r[j]
        float y_part = s0 * sh_r[j0];
        if (has_j1) y_part = fmaf(s1, sh_r[j1], y_part);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            y_part += __shfl_xor_sync(0xffffffffu, y_part, mask);
        }
        if (lane == 0) yv[row] = y_part;
        st_st(&s[s_row_base + j0], s0);
        if (has_j1) st_st(&s[s_row_base + j1], s1);
    }
    __syncthreads();

    // Phase 4+5：group-norm(y) 与 s 归约（t < n 一人一行，树归约同旧版结构）
    if (t < n) {
        const float y_i = yv[t];
        sqY[t]   = y_i;
        sqY2[t]  = y_i * y_i;
        ssRed[t] = sh_r[t] * sh_k[t] * rk[w_base + t];
    }
    __syncthreads();
    for (int step = n >> 1; step > 0; step >>= 1) {
        if (t < step) {
            sqY[t]   += sqY[t + step];
            sqY2[t]  += sqY2[t + step];
            ssRed[t] += ssRed[t + step];
        }
        __syncthreads();
    }
    if (t == 0) {
        const float ssum = sqY[0];
        const float ssq  = sqY2[0];
        mean    = ssum / (float)n;
        const float variance = ssq / (float)n - mean * mean;
        inv_std = rsqrtf(variance + gn_eps);
        s_acc   = ssRed[0];
    }
    __syncthreads();

    // Phase 6：y_norm[row] = (y[row]-mean)*inv_std*gamma[row]+beta[row] + s*v[row]
    if (t < n) {
        const float v_i = __half2float(v[v_base + t]);
        const float normalized =
            (yv[t] - mean) * inv_std * gamma[w_base + t] + beta[w_base + t];
        yn[v_base + t] = normalized + s_acc * v_i;
    }
}
"#;

/// gemv_rkv_stage1 CUDA kernel：r/k/v 三个 C×C 投影 + v1/w1/a1/g1 四个 mid 投影，一次 dispatch。
/// 语义对齐 Vulkan `gemv_rkv_stage1.comp`：
///   r = xr @ R^T；k = xk @ K^T；v = xv @ V^T（fp16 权重，f32 输入/累加，v 输出 fp16）
///   v_mid = xv @ V1；w_mid = tanh(xw @ W1)；a_mid = xa @ A1；g_mid = xg @ G1（fp32 权重）
/// dispatch (C/ROWS + VM + WM + AM + GM, 1, 1)：每 block 128 线程。
///   前 C/ROWS 个 block 各算 ROWS=4 行 r/k/v；后各算一个 mid 输出。
const GEMV_RKV_STAGE1_SRC: &str = r#"
extern "C" __global__ void gemv_rkv_stage1(
    const __half* __restrict__ R,   // [C,C] fp16
    const __half* __restrict__ K,   // [C,C] fp16
    const __half* __restrict__ V,   // [C,C] fp16
    const float*  __restrict__ V1,  // [VM,C] fp32
    const float*  __restrict__ W1,  // [WM,C] fp32
    const float*  __restrict__ A1,  // [AM,C] fp32
    const float*  __restrict__ G1,  // [GM,C] fp32
    const float*  __restrict__ xr,  // [C]
    const float*  __restrict__ xk,  // [C]
    const float*  __restrict__ xv,  // [C]
    const float*  __restrict__ xw,  // [C]
    const float*  __restrict__ xa,  // [C]
    const float*  __restrict__ xg,  // [C]
    float* __restrict__ out_r,      // [C]
    float* __restrict__ out_k,      // [C]
    __half* __restrict__ out_v,     // [C] fp16
    float* __restrict__ out_vm,     // [VM]
    float* __restrict__ out_wm,     // [WM]
    float* __restrict__ out_am,     // [AM]
    float* __restrict__ out_gm,     // [GM]
    const int c,
    const int vm,
    const int wm,
    const int am,
    const int gm)
{
    constexpr int ROWS = 4;
    const int tid = threadIdx.x;
    const int flat = blockIdx.x;

    if (flat < c / ROWS) {
        const int row_base = flat * ROWS;
        // 半精度累加器（half2），对齐 Albatross h2stage_hfma2 版本：__hfma2 输出 half2，
        // 累加保持在 half2，最终一次转 float 归约。
        half2 lr2[ROWS];
        half2 lk2[ROWS];
        half2 lv2[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            lr2[r] = __half2half2(0.f);
            lk2[r] = __half2half2(0.f);
            lv2[r] = __half2half2(0.f);
        }
        // 向量化主循环：对齐 Albatross rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe。
        // 每线程每次迭代处理 2 个 k（x 转 half2、权重按 half2 读），用 __hfma2 半精度乘加
        // （FP16 FMA 吞吐为 FP32 的 2 倍），权重与 x 均只读一次即可贡献给 ROWS 行。
        // kq 步进 2 且 blockDim 为偶数，保证 half2 的 4 字节对齐。
        const int c2 = c & ~1;
        for (int kq = tid * 2; kq < c2; kq += blockDim.x * 2) {
            const half2 hxr = __floats2half2_rn(xr[kq], xr[kq + 1]);
            const half2 hxk = __floats2half2_rn(xk[kq], xk[kq + 1]);
            const half2 hxv = __floats2half2_rn(xv[kq], xv[kq + 1]);
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const half2 wr = *reinterpret_cast<const half2*>(R + (long long)(row_base + r) * c + kq);
                const half2 wk = *reinterpret_cast<const half2*>(K + (long long)(row_base + r) * c + kq);
                const half2 wv = *reinterpret_cast<const half2*>(V + (long long)(row_base + r) * c + kq);
                lr2[r] = __hfma2(hxr, wr, lr2[r]);
                lk2[r] = __hfma2(hxk, wk, lk2[r]);
                lv2[r] = __hfma2(hxv, wv, lv2[r]);
            }
        }
        // 累加器 half2 → float（每行 2 分量求和），供后续 warp 归约。
        float lr[ROWS];
        float lk[ROWS];
        float lv[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float2 rf = __half22float2(lr2[r]);
            const float2 kf = __half22float2(lk2[r]);
            const float2 vf = __half22float2(lv2[r]);
            lr[r] = rf.x + rf.y;
            lk[r] = kf.x + kf.y;
            lv[r] = vf.x + vf.y;
        }
        // 尾部标量兜底（c 为奇数时）。
        if ((c & 1) && tid == 0) {
            const int kk = c - 1;
            const float xrv = xr[kk];
            const float xkv = xk[kk];
            const float xvv = xv[kk];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int a = (row_base + r) * c + kk;
                if (row_base + r < c) {
                    lr[r] += __half2float(R[a]) * xrv;
                    lk[r] += __half2float(K[a]) * xkv;
                    lv[r] += __half2float(V[a]) * xvv;
                }
            }
        }
        // warp shuffle 归约 sr/sk/sv（对齐 Albatross row1_linear_exact4_kernel）。
        __shared__ float partial_r[4 /*warp*/][ROWS];
        __shared__ float partial_k[4 /*warp*/][ROWS];
        __shared__ float partial_v[4 /*warp*/][ROWS];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float vr = lr[r];
            float vk = lk[r];
            float vv = lv[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vr += __shfl_down_sync(0xffffffffu, vr, off);
                vk += __shfl_down_sync(0xffffffffu, vk, off);
                vv += __shfl_down_sync(0xffffffffu, vv, off);
            }
            if (lane == 0) { partial_r[warp][r] = vr; partial_k[warp][r] = vk; partial_v[warp][r] = vv; }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float sr_ = 0.f, sk_ = 0.f, sv_ = 0.f;
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    sr_ += partial_r[w][r]; sk_ += partial_k[w][r]; sv_ += partial_v[w][r];
                }
                const int row = row_base + r;
                if (row < c) {
                    out_r[row] = sr_;
                    out_k[row] = sk_;
                    out_v[row] = __float2half(sv_);
                }
            }
        }
        return;
    }

    // mid 投影分支
    const int mid_idx = flat - c / ROWS;
    float local_dot = 0.f;
    int chain = 3;
    int row = 0;
    if (mid_idx < vm) {
        chain = 0; row = mid_idx;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += V1[row * c + kk] * xv[kk];
    } else if (mid_idx < vm + wm) {
        chain = 1; row = mid_idx - vm;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += W1[row * c + kk] * xw[kk];
    } else if (mid_idx < vm + wm + am) {
        chain = 2; row = mid_idx - vm - wm;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += A1[row * c + kk] * xa[kk];
    } else {
        row = mid_idx - vm - wm - am;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += G1[row * c + kk] * xg[kk];
    }
    __shared__ float sm[128];
    sm[tid] = local_dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float result = sm[0];
        if (chain == 0) out_vm[row] = result;
        else if (chain == 1) out_wm[row] = tanhf(result);
        else if (chain == 2) out_am[row] = result;
        else out_gm[row] = result;
    }
}
"#;

/// gemv_int8_rkv_stage1 CUDA kernel：r/k/v 三个 C×C int8 量化投影 + 四个 mid fp32 投影，一次 dispatch。
/// 语义对齐 Vulkan `gemv_int8_rkv_stage1.comp`：
///   r = xr @ R^T；k = xk @ K^T；v = xv @ V^T（int8 权重，f32 输入/累加，v 输出 fp16）
///   v_mid = xv @ V1；w_mid = tanh(xw @ W1)；a_mid = xa @ A1；g_mid = xg @ G1（fp32 权重）
/// int8 格式（每矩阵 W[C,C] 行主序，K=C 收缩，group=128）：
///   idx: uint32 [C, C/4]（每 uint32 打包 4 个 uint8 权重，字节序 = 权重索引低位优先）
///   sz:  uint32 [C, C/128]（每元素 = (scale: fp16 低16位 | zero: fp16 高16位)）
/// 反量化 w[m,k] = scale[m,k/128] * idx[m,k] + zero[m,k/128]（无 LUT，直接字节提取）。
/// dispatch (C/ROWS + VM + WM + AM + GM, 1, 1)：每 block 128 线程。
///   前 C/ROWS 个 block 各算 ROWS=4 行 r/k/v；后各算一个 mid 输出（与 fp16 版一致）。
const GEMV_INT8_RKV_STAGE1_SRC: &str = r#"
// ★ 2026-09-23：r/k/v 分支每 block 的行数（host 注入；缺省 4 = 旧行为）。
// 本内核是 **B=1 的关键路径 #1**（ABLATE 移除收益 17%）：grid = `C/ROWS + mid`。
// ROWS 越小 ⇒ 块数越多（ROWS=4 → 640 块，ROWS=2 → 1280 块），
// 每个 block 的串行 k 循环轮数不变（`KV/blockDim` 与 ROWS 无关），
// 但**每 block 摊到的权重字节减半** ⇒ 并发块数翻倍。
// ⚠️ 需 `C % ROWS == 0`；ROWS 越大寄存器/累加器与 smem（`3·ROWS·KG_MAX`）越吃紧。
#ifndef RKV_ROWS
#define RKV_ROWS 2
#endif
/// int8 量化权重解包辅助：`sz` 每元素 = (scale: fp16 低16位 | zero: fp16 高16位)。
__device__ __forceinline__ void unpack_int8_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}

extern "C" __global__ void gemv_int8_rkv_stage1(
    const unsigned int* __restrict__ R_idx,   // int8 idx [C, C/4]（4 字节/uint32）
    const unsigned int* __restrict__ R_sz,    // int8 sz  [C, C/128]
    const unsigned int* __restrict__ K_idx,
    const unsigned int* __restrict__ K_sz,
    const unsigned int* __restrict__ V_idx,
    const unsigned int* __restrict__ V_sz,
    const float*  __restrict__ V1,            // [VM,C] fp32
    const float*  __restrict__ W1,            // [WM,C]
    const float*  __restrict__ A1,            // [AM,C]
    const float*  __restrict__ G1,            // [GM,C]
    const float*  __restrict__ xr,            // [C]
    const float*  __restrict__ xk,            // [C]
    const float*  __restrict__ xv,            // [C]
    const float*  __restrict__ xw,            // [C]
    const float*  __restrict__ xa,            // [C]
    const float*  __restrict__ xg,            // [C]
    float* __restrict__ out_r,                // [C]
    float* __restrict__ out_k,                // [C]
    __half* __restrict__ out_v,               // [C] fp16
    float* __restrict__ out_vm,               // [VM]
    float* __restrict__ out_wm,               // [WM]
    float* __restrict__ out_am,               // [AM]
    float* __restrict__ out_gm,               // [GM]
    const int c,
    const int vm,
    const int wm,
    const int am,
    const int gm)
{
    constexpr int ROWS = RKV_ROWS;
    constexpr int KG_MAX = 32;   // C/128 上限（C ≤ 4096）。C=2560 → KG=20。
    const int tid  = threadIdx.x;
    const int flat = blockIdx.x;

    // r/k/v 分支：每个 block 处理 ROWS=4 行，int8 反量化 + 归约。
    if (flat < c / ROWS) {
        const int row_base = flat * ROWS;
        const int KV = c / 4;    // 每行 uint32 idx 数（4 字节/uint32）
        const int KG = c / 128;  // 每行 group 数

        __shared__ float s_scale[3][ROWS][KG_MAX];
        __shared__ float s_zero[3][ROWS][KG_MAX];

        // Phase 0：协作加载 3 矩阵 × ROWS 行的 scale/zero。无 LUT。
        for (int i = tid; i < 3 * ROWS * KG; i += blockDim.x) {
            const int mat = i / (ROWS * KG);
            const int rem = i % (ROWS * KG);
            const int r   = rem / KG;
            const int g   = rem % KG;
            const int row = row_base + r;
            const unsigned int* szp = (mat == 0) ? R_sz : ((mat == 1) ? K_sz : V_sz);
            float sc, zr;
            unpack_int8_sz(szp[row * KG + g], sc, zr);
            s_scale[mat][r][g] = sc;
            s_zero[mat][r][g]  = zr;
        }
        __syncthreads();

        // Phase 1：主循环，每 iter 反量化 4 权重/矩阵/行（1 uint32），半精度 __hfma2 累加。
        // 对齐 fp16 版：反量化结果转 __half 拼 half2，与 x 的 half2 用 __hfma2 乘加
        // （FP16 FMA 吞吐为 FP32 的 2 倍），累加保持在 half2，最终一次转 float 归约。
        half2 acc_r[ROWS], acc_k[ROWS], acc_v[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            acc_r[r] = __half2half2(0.f);
            acc_k[r] = __half2half2(0.f);
            acc_v[r] = __half2half2(0.f);
        }
        // ⚠️ 反例留档（2026-09-23）：这里加 `#pragma unroll 2` 试过提高 MLP ⇒ **反而更慢**
        // （96.2 → 91.2~93.4 tok/s），因为 ROWS=4 已经占了 12 个 half2 累加器，展开后
        // 寄存器压力上升、占用率下降。**本内核已 ~511 GB/s（可达上限 528），不要再动。**
        for (int kk = tid; kk < KV; kk += blockDim.x) {
            const half2 hxr0 = __floats2half2_rn(xr[4 * kk],     xr[4 * kk + 1]);
            const half2 hxr1 = __floats2half2_rn(xr[4 * kk + 2], xr[4 * kk + 3]);
            const half2 hxk0 = __floats2half2_rn(xk[4 * kk],     xk[4 * kk + 1]);
            const half2 hxk1 = __floats2half2_rn(xk[4 * kk + 2], xk[4 * kk + 3]);
            const half2 hxv0 = __floats2half2_rn(xv[4 * kk],     xv[4 * kk + 1]);
            const half2 hxv1 = __floats2half2_rn(xv[4 * kk + 2], xv[4 * kk + 3]);
            const int g = kk >> 5;   // 32 个 uint32/组
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int irow = (row_base + r) * KV + kk;
                const unsigned int pr = R_idx[irow];
                const unsigned int pk = K_idx[irow];
                const unsigned int pv = V_idx[irow];
                const float scr = s_scale[0][r][g], zrr = s_zero[0][r][g];
                const float sck = s_scale[1][r][g], zrk = s_zero[1][r][g];
                const float scv = s_scale[2][r][g], zrv = s_zero[2][r][g];
                __align__(16) __half wr[4], wk[4], wv[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int nbr = (pr >> (8 * j)) & 0xFF;
                    const int nbk = (pk >> (8 * j)) & 0xFF;
                    const int nbv = (pv >> (8 * j)) & 0xFF;
                    wr[j] = __float2half(scr * (float)nbr + zrr);
                    wk[j] = __float2half(sck * (float)nbk + zrk);
                    wv[j] = __float2half(scv * (float)nbv + zrv);
                }
                const half2 wr0 = *reinterpret_cast<const half2*>(&wr[0]);
                const half2 wr1 = *reinterpret_cast<const half2*>(&wr[2]);
                const half2 wk0 = *reinterpret_cast<const half2*>(&wk[0]);
                const half2 wk1 = *reinterpret_cast<const half2*>(&wk[2]);
                const half2 wv0 = *reinterpret_cast<const half2*>(&wv[0]);
                const half2 wv1 = *reinterpret_cast<const half2*>(&wv[2]);
                acc_r[r] = __hfma2(hxr0, wr0, acc_r[r]);
                acc_r[r] = __hfma2(hxr1, wr1, acc_r[r]);
                acc_k[r] = __hfma2(hxk0, wk0, acc_k[r]);
                acc_k[r] = __hfma2(hxk1, wk1, acc_k[r]);
                acc_v[r] = __hfma2(hxv0, wv0, acc_v[r]);
                acc_v[r] = __hfma2(hxv1, wv1, acc_v[r]);
            }
        }
        // 累加器 half2 → float（每行 2 分量求和），供后续 warp 归约。
        float lr[ROWS], lk[ROWS], lv[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float2 rrf = __half22float2(acc_r[r]);
            const float2 rkf = __half22float2(acc_k[r]);
            const float2 rvf = __half22float2(acc_v[r]);
            lr[r] = rrf.x + rrf.y;
            lk[r] = rkf.x + rkf.y;
            lv[r] = rvf.x + rvf.y;
        }

        // warp shuffle 归约（3 矩阵 × ROWS 行）。
        __shared__ float partial_r[4 /*warp*/][ROWS];
        __shared__ float partial_k[4 /*warp*/][ROWS];
        __shared__ float partial_v[4 /*warp*/][ROWS];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float vr = lr[r];
            float vk = lk[r];
            float vv = lv[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vr += __shfl_down_sync(0xffffffffu, vr, off);
                vk += __shfl_down_sync(0xffffffffu, vk, off);
                vv += __shfl_down_sync(0xffffffffu, vv, off);
            }
            if (lane == 0) { partial_r[warp][r] = vr; partial_k[warp][r] = vk; partial_v[warp][r] = vv; }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float sr_ = 0.f, sk_ = 0.f, sv_ = 0.f;
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    sr_ += partial_r[w][r]; sk_ += partial_k[w][r]; sv_ += partial_v[w][r];
                }
                const int row = row_base + r;
                if (row < c) {
                    out_r[row] = sr_;
                    out_k[row] = sk_;
                    out_v[row] = __float2half(sv_);
                }
            }
        }
        return;
    }

    // mid 投影分支 ★ 2026-09-23：**改成 warp-per-row**（每 warp 一行、4 行/块）。
    //
    // 病灶（与单流 `gemv_lowrank_chain4` 同源）：旧版每个 block 用 128 线程算**一行**，
    // 再做 7 轮 `__syncthreads` 的块级树归约。4 个 mid 矩阵共 640 行 ⇒ 640 个 block，
    // 每 block 只做 2560 个 MAC 却付整套块级同步。实测本内核 **1.96 ms/token**
    // （`gemv_int8_rkv_stage1`，单流 B=1 的 #3 项），而流量只有 ~26MB/次
    // ⇒ 约 13 GB/s，**纯延迟/占用率瓶颈**，不是带宽。
    //
    // 归约改 `__shfl_down_sync`（warp 内、无 smem、无 barrier）。数值口径随之改变
    // （块级树 → warp 树，相对差 ~1e-7；门禁 `gemv_int8_rkv_stage1_matches_cpu` 容差 1e-2）。
    const int MIDW = 4;   // 每 block 的 mid 行数 = warp 数
    const int mid_base = (flat - c / ROWS) * MIDW;
    {
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const int mrow = mid_base + warp;
        const int midtot = vm + wm + am + gm;
        if (mrow < midtot) {
            const float* __restrict__ wp;
            const float* __restrict__ xp;
            int chain = 3;
            int row = 0;
            if (mrow < vm) {
                chain = 0; row = mrow; wp = V1; xp = xv;
            } else if (mrow < vm + wm) {
                chain = 1; row = mrow - vm; wp = W1; xp = xw;
            } else if (mrow < vm + wm + am) {
                chain = 2; row = mrow - vm - wm; wp = A1; xp = xa;
            } else {
                row = mrow - vm - wm - am; wp = G1; xp = xg;
            }
            const float* wrow = wp + (long)row * c;
            float d = 0.f;
            for (int kk = lane; kk < c; kk += 32) d += wrow[kk] * xp[kk];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffffu, d, off);
            if (lane == 0) {
                if (chain == 0) out_vm[row] = d;
                else if (chain == 1) out_wm[row] = tanhf(d);
                else if (chain == 2) out_am[row] = d;
                else out_gm[row] = d;
            }
        }
    }
}
"#;

/// gemv_lowrank_chain4 CUDA kernel：融合 4 条低秩链第二级（w/a/g/v 的 w2/a2/g2/v2），
/// 一次 dispatch。语义对齐 Vulkan `gemv_lowrank_chain4.comp`：
///   w：out = exp(scale[0] * sigmoid(xw @ W2[row] + w0[row]))          [K_W, scale, w0]
///   a：out = sigmoid(xa @ A2[row] + a0[row])                          [K_A, a0]
///   v：out_v[row] += sigmoid(xv @ V2[row] + v0[row]) * (v_first[row] - out_v[row])（原地）
///   g：out = sum_k sigmoid(xg[k]) * G2[row,k]（g 链 sigmoid 作用于 mid，无 bias）
/// 权重矩阵行主序 [M, K] fp32；x 为 mid 向量 [K] fp32；v_first 与四个输出为 fp16。
///
/// ★ 2026-09-23：**改成 warp-per-row**（dispatch `(ceil(M/8), 1, 1)`、block=256、每 warp 一行）。
///
/// 病灶：旧版是「整 block 归约 1 行」（dispatch `(M,1,1)`）——M=2560 时 2560 个 block，
/// 每 block 只做 640 个 MAC（2.5 个/线程），却要付 **9 次 `__syncthreads`** 的块级树归约。
/// 实测 0.054 ms/次 = 6.55MB / 53.7µs = **122 GB/s**（同机 f32 流式可达 ~500）。
/// 探针排除：把内层 `sigmoidf(xg[k])` 换成 `xg[k]`（错数）**零变化** ⇒ 不是 expf 的问题。
/// 同时 `IMMA_BM_SMALL=128` 探针（块数减半、k-tile 数减半）**反而更慢** ⇒ 不是「每 k-tile 固定开销」。
/// 批量版 `gemv_lowrank_chain4_batch` 早已是这个结构（其注释同样点名「整 block 归约 1 行、
/// grid 太大、syncthreads 6 次为主因」）——单流版一直没跟上。
///
/// 归约改用 `__shfl_down_sync`（warp 内，无 smem、无 barrier）；数值口径与
/// `gemv_lowrank_chain4_batch` **一致**（同一条 warp 树），但与该单流旧版的块级树不同
/// （相对差 ~1e-7，见门禁 `gemv_lowrank_chain4_matches_cpu` 的 1e-2 容差）。
const GEMV_LOWRANK_CHAIN4_SRC: &str = r#"
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void gemv_lowrank_chain4(
    const float*  __restrict__ W2,   // [M, KW] fp32 行主序
    const float*  __restrict__ A2,   // [M, KA]
    const float*  __restrict__ V2,   // [M, KV]
    const float*  __restrict__ G2,   // [M, KG]
    const float*  __restrict__ xw,   // [KW]
    const float*  __restrict__ xa,   // [KA]
    const float*  __restrict__ xv,   // [KV]
    const float*  __restrict__ xg,   // [KG]
    const float*  __restrict__ w0,   // [M]
    const float*  __restrict__ a0,   // [M]
    const float*  __restrict__ v0,   // [M]
    const float*  __restrict__ scale,// [1]
    const __half* __restrict__ v_first, // [M] fp16
    __half* __restrict__ out_w,      // [M] fp16
    __half* __restrict__ out_a,      // [M] fp16
    __half* __restrict__ out_v,      // [M] fp16（读改写）
    __half* __restrict__ out_g,      // [M] fp16
    const int m,
    const int kw,
    const int ka,
    const int kv,
    const int kg)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * 8 + warp;
    if (row >= m) return;

    float lw = 0.f, la = 0.f, lv = 0.f, lg = 0.f;
    // ★ 2026-09-23：四条链的内层循环加 `#pragma unroll 4`。
    // 每行总读取量 = `wm+am+vm+gm` 列 × 4B（本模型 640×4 = 2560B），
    // 而 warp 内是 `k = lane; k += 32` 的**串行 128B 载入**（每链 ceil(k/32) 次）。
    // 展开后每个线程 4 条独立载入在飞 ⇒ 提高 MLP，直接攻「269 GB/s vs 可达 500」的缺口。
    // 逐行算术与求和顺序完全不变 ⇒ 数值逐位一致（只改调度）。
    #pragma unroll 4
    for (int k = lane; k < kw; k += 32) lw += xw[k] * W2[row * kw + k];
    #pragma unroll 4
    for (int k = lane; k < ka; k += 32) la += xa[k] * A2[row * ka + k];
    #pragma unroll 4
    for (int k = lane; k < kv; k += 32) lv += xv[k] * V2[row * kv + k];
    #pragma unroll 4
    for (int k = lane; k < kg; k += 32) lg += sigmoidf(xg[k]) * G2[row * kg + k];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lw += __shfl_down_sync(0xffffffffu, lw, off);
        la += __shfl_down_sync(0xffffffffu, la, off);
        lv += __shfl_down_sync(0xffffffffu, lv, off);
        lg += __shfl_down_sync(0xffffffffu, lg, off);
    }
    if (lane == 0) {
        out_w[row] = __float2half(expf(scale[0] * sigmoidf(lw + w0[row])));
        out_a[row] = __float2half(sigmoidf(la + a0[row]));
        const float vcur = __half2float(out_v[row]);
        out_v[row] = __float2half(
            vcur + sigmoidf(lv + v0[row]) * (__half2float(v_first[row]) - vcur));
        out_g[row] = __float2half(lg);
    }
}
"#;

/// gemv_variant CUDA kernel：统一处理 9 个 gemv 变体（权重类型 × 输出变换）。
/// 语义对齐 Vulkan `gemv_f32io_relu2` / `gemv_f32io_add_mul` / `gemv_f32io_add`：
///   relu2   ：y = relu²(x @ A)  = max(0, dot)²
///   mul_add ：y += (x .* g) @ A（g 为 fp16 门控，逐元素作用于 x）
///   add     ：y += x @ A
/// 权重反量化由 `wtype` 选择（0=f16、2=int8），`op` 选择输出变换
/// （0=relu2、1=mul_add、2=add）。dispatch (M/4, batch, 1)，每 block 128 线程处理 4 行。
/// 与 gemv_f16 同构：每 block 处理 ROWS=4 行，跨线程归约后 tid==0 写输出。
const GEMV_VARIANT_SRC: &str = r#"
__device__ __forceinline__ void unpack_variant_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
__device__ __forceinline__ float relu2f(float x) { return x > 0.f ? x * x : 0.f; }

extern "C" __global__ void gemv_variant(
    const __half*         __restrict__ Af16,  // fp16 [M*K]（wtype==0 用）
    const unsigned int*   __restrict__ aidx,  // int8 idx [M,K/4]
    const __half*         __restrict__ alut,  // 保留（当前未使用）
    const unsigned int*   __restrict__ asz,   // int8 sz [M,K/128]
    const float*          __restrict__ x,     // [K*batch]
    const __half*         __restrict__ g,     // [K*batch] fp16 门控（op==1 用）
    float*                __restrict__ y,     // [M*batch]（累加式读改写）
    const int m,
    const int k,
    const int batch,
    const int wtype,   // 0=f16, 2=int8
    const int op)      // 0=relu2, 1=mul_add, 2=add
{
    const int tid  = threadIdx.x;
    const int b    = blockIdx.y;
    const int row0 = blockIdx.x * 4;
    const int k0   = b * k;
    const int m0   = b * m;
    const int kvi  = k / 4;   // int8 每行 uint32 数
    const int kg   = k / 128; // int8 每行 group 数
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    if (wtype == 0) {
        // fp16 向量化主循环：每线程每次迭代处理 4 个 k（x 按 float4、权重按 8B half4 读）。
        // 半精度累积（__hfma2，吞吐为 FP32 2 倍），4 行各持 2 个 half2 累加器。
        half2 hacc[4][2];
        #pragma unroll
        for (int r = 0; r < 4; r++) { hacc[r][0] = __half2half2(0.f); hacc[r][1] = __half2half2(0.f); }
        const int k4 = k & ~3;
        for (int kq = tid * 4; kq < k4; kq += blockDim.x * 4) {
            const float4 xv = *reinterpret_cast<const float4*>(x + k0 + kq);
            float gx = 1.f, gy = 1.f, gz = 1.f, gw = 1.f;
            if (op == 1) {
                // g 为 fp16，按 8B half4 加载（勿用 float4，避免 16B 对齐越界）。
                load_half4_f4(g + k0 + kq, gx, gy, gz, gw);
            }
            const half2 hx01 = __floats2half2_rn(xv.x * gx, xv.y * gy);
            const half2 hx23 = __floats2half2_rn(xv.z * gz, xv.w * gw);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const __half* wj = Af16 + (row0 + r) * k + kq;
                hacc[r][0] = __hfma2(hx01, *reinterpret_cast<const half2*>(wj), hacc[r][0]);
                hacc[r][1] = __hfma2(hx23, *reinterpret_cast<const half2*>(wj + 2), hacc[r][1]);
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const float2 f0 = __half22float2(hacc[r][0]);
            const float2 f1 = __half22float2(hacc[r][1]);
            acc[r] = f0.x + f0.y + f1.x + f1.y;
        }
        // 尾部标量兜底（k 非 4 倍数时）。
        for (int kk = k4 + tid; kk < k; kk += blockDim.x) {
            const float xv = x[k0 + kk];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = row0 + r;
                acc[r] += __half2float(Af16[row * k + kk]) * xv;
            }
        }
    } else {
        // int8 向量化路径：每线程每次迭代处理 4 行 × 4 个 k（1 uint32 = 4 字节权重），
        // 反量化后拼 half2，与 x 的 half2 用 __hfma2 累加（FP16 FMA 吞吐为 FP32 的 2 倍）。
        // scale/zero 按 group 内循环解一次，字节提取兼内联。
        half2 hacc[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) hacc[r] = __half2half2(0.f);
        const int kvi4 = k / 4;   // int8 每行 uint32 数（4 字节/uint32）
        for (int kq = tid; kq < kvi4; kq += blockDim.x) {
            const int kbase = kq * 4;
            const int gr = kg > 0 ? (kbase / 128) : 0;
            float gv0 = 1.f, gv1 = 1.f, gv2 = 1.f, gv3 = 1.f;
            if (op == 1) {
                const __half* gq = g + k0 + kbase;
                gv0 = __half2float(gq[0]); gv1 = __half2float(gq[1]);
                gv2 = __half2float(gq[2]); gv3 = __half2float(gq[3]);
            }
            const half2 hx01 = __floats2half2_rn(x[k0 + kbase] * gv0, x[k0 + kbase + 1] * gv1);
            const half2 hx23 = __floats2half2_rn(x[k0 + kbase + 2] * gv2, x[k0 + kbase + 3] * gv3);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                const unsigned int p = aidx[row * kvi4 + kq];
                float sc, zr;
                unpack_variant_sz(asz[row * kg + gr], sc, zr);
                __align__(16) __half w[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int byte = (int)((p >> (j * 8)) & 0xFFu);
                    w[j] = __float2half(sc * (float)byte + zr);
                }
                const half2 w01 = *reinterpret_cast<const half2*>(&w[0]);
                const half2 w23 = *reinterpret_cast<const half2*>(&w[2]);
                hacc[r] = __hfma2(hx01, w01, hacc[r]);
                hacc[r] = __hfma2(hx23, w23, hacc[r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const float2 f = __half22float2(hacc[r]);
            acc[r] = f.x + f.y;
        }
    }

    // warp shuffle 归约（对齐 Albatross row1_linear_exact4_kernel<128,2>）：只有 1 次 __syncthreads。
    __shared__ float partial[4 /*warp*/][4 /*row*/];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) partial[warp][r] = v;
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < 4; w++) sum += partial[w][r];
            const int row = row0 + r;
            if (row < m) {
                if (op == 0) y[m0 + row] = relu2f(sum);
                else if (op == 3) y[m0 + row] = sum;
                else y[m0 + row] += sum;
            }
        }
    }
}
"#;

/// fp32 激活 → fp16 激活的降位内核（可选乘门控）。batch 线性层第二代的输入准备。
///
/// 语义与旧内核内联的 `__floats2half2_rn(x[i] * g[i])` 完全一致（同样在 fp32 域
/// 相乘、同样 round-to-nearest），故 op!=1 时逐位等同旧路径，op==1 亦不变。
const CAST_MUL_F16_SRC: &str = r#"
extern "C" __global__ void cast_mul_f16(
    const float*  __restrict__ x,
    const __half* __restrict__ g,   // 可为 0：不乘门控
    __half*       __restrict__ y,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (g != 0) v *= __half2float(g[i]);
    y[i] = __float2half_rn(v);
}
"#;

/// 三路 fp32 → fp16 降位（r/k/v 激活一次搞定，省两次 launch）。
/// 语义 = 逐个 `__float2half_rn`，与 `gemv_int8_rkv_stage1_batch` 内联的
/// `__floats2half2_rn` 同轮次舍入 ⇒ 数值与旧路径逐位一致。
const CAST3_F16_SRC: &str = r#"
extern "C" __global__ void cast3_f16(
    const float* __restrict__ a0,
    const float* __restrict__ a1,
    const float* __restrict__ a2,
    __half* __restrict__ b0,
    __half* __restrict__ b1,
    __half* __restrict__ b2,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    b0[i] = __float2half_rn(a0[i]);
    b1[i] = __float2half_rn(a1[i]);
    b2[i] = __float2half_rn(a2[i]);
}
"#;

/// `cast3_f16` 的四路版：xw/xa/xv/xg `[batch, C]` fp32 → fp16（低秩 GEMM 的 A 操作数）。
/// 与 `cast3_f16` 同为**纯降位**（不含 sigmoid）——g 链的 sigmoid 落在 stage1 epilogue 上，
/// 与 fp32 路径「sigmoid 作用在 mid_g」逐位同序（见 `LOWRANK_GEMM_SRC`）。
const CAST4_F16_SRC: &str = r#"
extern "C" __global__ void cast4_f16(
    const float* __restrict__ a0,
    const float* __restrict__ a1,
    const float* __restrict__ a2,
    const float* __restrict__ a3,
    __half* __restrict__ b0,
    __half* __restrict__ b1,
    __half* __restrict__ b2,
    __half* __restrict__ b3,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    b0[i] = __float2half_rn(a0[i]);
    b1[i] = __float2half_rn(a1[i]);
    b2[i] = __float2half_rn(a2[i]);
    b3[i] = __float2half_rn(a3[i]);
}
"#;

/// `gemv_variant_mb16` 默认几何：每 block 覆盖的行数（每线程累加器 = ROWS×8 个 half2）。
/// 隔离计时扫描（B=8）：ROWS=1 → 0.2208/1.3596/0.0638，**2 → 0.1602/0.9302/0.0570**，
/// **4 → 0.1577/0.8840/0.0522**（relu2/plain/mul_add），8 → 0.2000/1.1107/0.0792
/// （64 个累加器 → 寄存器溢出）。**取 4**。
/// 可用 `GEMV_MB16_ROWS` 覆盖做几何 A/B。
const GEMV_MB16_ROWS_DEFAULT: usize = 4;

/// batch 线性层 fp16 激活暂存的初始容量（元素数，f16 → 4 MiB）。
/// 该块在 `begin_batch` 于**捕获之前**一次性建好，之后所有 `batch*k` 形状复用同一指针
/// （图安全）；4 MiB 覆盖 batch=64 × k=10240 仍有余量。
const X16_SCRATCH_INIT_ELEMS: usize = 2 * 1024 * 1024;

/// W8A8 激活量化暂存初值：`xq`（u32 元素数 = 字节数/4）与 `xaux`（float4 个数）。
/// 覆盖 batch=256 × k=10240（xq 2.6 MB→1M u32 / xaux 20480 float4）仍有余量。
const XQ_SCRATCH_INIT_U32: usize = 1 << 20;
const XAUX_SCRATCH_INIT_F4: usize = 1 << 15;
/// split-K 部分和暂存的初值（f32 元素数）。按默认规则的最大需求 = B=128 的
/// r/k/v 三链 `3×2×128×2560 = 1.97M` 留一倍余量 ⇒ 4 Mi（16 MiB）。
const IPART_SCRATCH_INIT_ELEMS: usize = 1 << 22;

/// 组宽（激活量化的分组粒度，必须与权重的 `k/128` 分组对齐）。
const QUANT_X_I8_GROUP: usize = 128;

/// 激活量化内核（W8A8 的 A 侧）：fp32 `x[rows,k]`（可选乘 fp16 门控）→
/// 对称 int8 `xq[rows,k/4]` + 每组统计 `xaux[rows,G]`。
///
/// 粒度 **per-(token, k 组)**，组宽 128 —— 与权重 `s[m,k/128]` 的分组同构，
/// 于是 IMMA 的「每组 flush」天然对齐（见 [int8 IMMA 实施记录](../../参考/2026-09-21-int8-IMMA实施记录.md) §2.1）。
///
/// `xaux[b,g] = {sx, 128·sx·colsum_xq, rowsum_x, 0}`，三项各有用途：
/// - `sx` = amax/127：把 int32 的 IMMA 结果还原回 fp32 的尺度因子；
/// - `128·sx·colsum_xq`：权重侧把 byte 当作**有符号** int8 解释（IMMA 只吃 s8），
///   而磁盘上的 `idx` 是 0..255 的无符号 byte ⇒ 做 `byte ^ 0x80` 把它整体下移 128，
///   补偿项 `Σ xq·128` 在此预先乘好 `sx`；
/// - `rowsum_x` = Σ 组内 **fp32** 原值：零点项 `z[m,g]·Σ_k x` 用原值算比用 `sx·Σxq`
///   算更准（量化误差相消），且省掉一次 rank-1 修正 GEMM。
///
/// grid = (G, ceil(rows/8))，block = 256（8 warp，每 warp 一个 (b,g) 组）；
/// 组内 128 元素按 lane 均分 4 个，尺度用 warp shuffle 归约。
const QUANT_X_I8_SRC: &str = r#"
extern "C" __global__ void __launch_bounds__(256) quant_x_i8(
    const float*  __restrict__ x,
    const __half* __restrict__ gp,    // 可为 0：不乘门控
    unsigned int* __restrict__ xq,    // [rows, k/4]，byte i = k%4==i
    float4*       __restrict__ xaux,  // [rows, G]
    const int rows,
    const int k)
{
    const int G    = k >> 7;
    const int g    = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int b    = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (b >= rows) return;
    const int k0 = (g << 7) + (lane << 2);
    const float* xp = x + b * k + k0;
    float v0 = xp[0], v1 = xp[1], v2 = xp[2], v3 = xp[3];
    if (gp != 0) {
        const __half* gp2 = gp + b * k + k0;
        v0 *= __half2float(gp2[0]);
        v1 *= __half2float(gp2[1]);
        v2 *= __half2float(gp2[2]);
        v3 *= __half2float(gp2[3]);
    }
    // 组内 amax → 对称尺度（整组共享，避免逐线程尺度破坏 IMMA 的 k 分组语义）
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float sx  = amax > 0.f ? amax * (1.f / 127.f) : 1.f;
    const float inv = 1.f / sx;
    // 量化：round-to-nearest，钳到 [-127,127]（对称量化不用 -128，免得 ± 不对称）
    const int q0 = __float2int_rn(fminf(fmaxf(v0 * inv, -127.f), 127.f));
    const int q1 = __float2int_rn(fminf(fmaxf(v1 * inv, -127.f), 127.f));
    const int q2 = __float2int_rn(fminf(fmaxf(v2 * inv, -127.f), 127.f));
    const int q3 = __float2int_rn(fminf(fmaxf(v3 * inv, -127.f), 127.f));
    xq[b * (k >> 2) + (g << 5) + lane] =
        ((unsigned int)(q0 & 0xFF)) | ((unsigned int)(q1 & 0xFF) << 8)
        | ((unsigned int)(q2 & 0xFF) << 16) | ((unsigned int)(q3 & 0xFF) << 24);
    int   cs = q0 + q1 + q2 + q3;
    float rs = v0 + v1 + v2 + v3;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cs += __shfl_xor_sync(0xffffffffu, cs, off);
        rs += __shfl_xor_sync(0xffffffffu, rs, off);
    }
    if (lane == 0)
        xaux[b * G + g] = make_float4(sx, 128.f * sx * (float)cs, rs, 0.f);
}
"#;

/// IMMA 批量 GEMM 几何（`#define` 注入，按值分名缓存模块 ⇒ 可不重编译做 A/B）。
/// 权重流量 = `M·K·ceil(batch/BN)`（BN 越大越省）、x 流量 = `batch·K·(M/BM)`。
///
/// ★ 2026-09-22 实测扫描（B=256，10304 次启动，ms/次）：
/// `BM=64/BN=64` 0.2662 · **`BM=128/BN=64` 0.2295** · `BM=128/BN=128` 0.5412 ·
/// `BM=256/BN=64` 0.9304。⇒ **BM=128 / BN=64 最优**（BM=256 时 MT=4 把累加器压到寄存器溢出）。
const IMMA_BM_DEFAULT: usize = 128;
const IMMA_BN_DEFAULT: usize = 64;
/// 走 IMMA 路径的最小 batch（`IMMA_MIN_BATCH` 环境变量可覆盖，做小 batch 路径 A/B 用）。
///
/// ★ 2026-09-22 由 16 → 8 → **1**。原先设 16/8 都是担心「batch 小于 mma 的 n 维
/// （8 槽）填不满」，但两次实测都证明**填不满也远胜 int8 SIMT** —— 小 batch 的瓶颈是
/// 「每 block 摊到的权重流量」而非张量核利用率（`BN` 由 `small_batch_bn` 保证 ≥8，
/// 槽位浪费**不影响权重读取量**）：
///
/// | 阈值 | 场景 | 改前 | 改后 | 收益 |
/// |---|---|---|---|---|
/// | 16 → 8 | 批量 B=8 | 354.0 | **462.5** | +30.7% |
/// | 8 → 1 | 批量 B=1 | 65.1 / 67.5 | **78.8 / 78.8** | **+18%** |
/// | 8 → 1 | 单流 selfloop（`gemv_variant_dispatch` 的 batch=1 分支） | 75.4 | **80.4** | **+6.6%** |
///
/// 单流 prefill（`forward_seq_with_state`）**不受影响**（`WARMUP=0` 隔离实测在噪声内）；
/// 首段多出的 ~200ms 是**一次性的图捕获**（多编译了几个 batch=1 的 IMMA 变体）。
/// B≥8 的路径不受影响（阈值只决定「低于它才回退」）。
const IMMA_MIN_BATCH: usize = 1;

/// 见 `IMMA_MIN_BATCH`。
fn imma_min_batch() -> usize {
    std::env::var("IMMA_MIN_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(IMMA_MIN_BATCH)
}

/// 小 batch 时的 `BN`：让 `grid.y = batch/BN = 1`，同时把**每块载入量 `(BM+BN)·K` 压到最小**。
///
/// ★ 机理（2026-09-22 实测钉死）：小 batch 时块数 = `M/BM` 且**每 SM 只摊到 1 块**
/// ⇒ 每次 launch 的时间 ≈ `(BM+BN)·K / 单SM可达带宽`，**与权重总量无关**。
/// 这解释了「BN 变小 ⇒ 权重重读变多」却仍然更快这个反直觉结果。
///
/// 实测（同会话，B=16/32/64）：
/// `BN=64 → 16` 得 **634.5 → 703.0（+10.8%）**；`BN=64 → 32`（B=32）得
/// **1162.0 → 1375.2（+18.3%）**；B=64 两者持平（2010.3 vs 1993.9，grid.y 都为 1 或 2）。
fn small_batch_bn(batch: usize) -> usize {
    batch.next_multiple_of(8).clamp(8, 64)
}

/// **形状感知**的 `BN`：在 `small_batch_bn` 基础上，若块数 `(M/BM)·(batch/BN)` 不足 68
/// （68 个 SM 喂不满），按 2 的幂把 `BN` 缩到能凑够 68 块为止，**下限 32**。
///
/// 实测（B=64，2026-09-22）：`att_output` 与 `ffn_value`（M=2560）各只有 `2560/64 = 40` 块，
/// 缩到 `BN=32` 后变 80 块 ⇒ B=64 **1994.7 → 2026.9（+1.6%）**；
/// 而 `ffn_key`（M=10240）本来就有 160 块，**不受影响**——这正是「一刀切改 `IMMA_BN`
/// 测不出增益、必须按形状取」的原因。
/// ⚠️ **下限必须停在 32**：再往下缩（BN=16/8）虽然块数更多，但**权重重读倍数同步翻倍**，
/// 实测 B=16 `BN=16 → 8` 反而 **703.0 → 686.1**、B=32 若缩到 16 也会变差。
fn pick_bn(m: usize, bm: usize, batch: usize) -> usize {
    let mut bn = small_batch_bn(batch);
    while bn > 32 && (m / bm) * batch.div_ceil(bn) < 68 {
        bn /= 2;
    }
    bn
}

/// ★ split-K 的分块数（`IM_KSPLIT`）选择 —— 2026-09-22，**小 batch 加块的唯一手段**。
///
/// **依据**（`PROF_CUDA_KERNEL` 按形状分解的 B=64 账本，见实施记录 §3p）：
/// `imma_gemm_batch` 的**每 block 吞吐被钉死在 ~90 GMAC/s**（`att_output` / `ffn_key` /
/// `ffn_value` 三个不同 K 的形状实测一致），而**聚合吞吐随并发块数单调上升**：
/// 80 块 → 102、160 块 → 172、1024 块 → 232 GMAC/s/SM。
/// 小 batch 时 `grid.y = batch/BN = 1`，块数只剩 `M/BM`（`att_output` 的 M=2560
/// ⇒ 只有 40~80 块，68 个 SM 大半空转）。**不增加流量的加块手段只有 split-K**：
/// 每个 k-段由独立 block 读**不同的** k 区间 ⇒ 块数 ×ksplit 而**总流量不变**
/// （代价：部分和写+读，以及一条确定性归约内核）。
///
/// 规则：把**总块数**抬到 `IMMA_KSPLIT_TARGET`（默认 **400 ≈ 6×SM 数**），
/// 分块数上限 `IMMA_KSPLIT_MAX`（默认 **5**），并要求 `(k/128) % ksplit == 0`
/// （k 段按量化组对齐）。
///
/// 目标块数的标定（B=64 同会话交错 A/B，ms/step 越大越好）：
/// `2×SM`(136) **2164** · `4×SM`(272) **2219** · **`6×SM`(400) 2234** ·
/// `8×SM`(544) 2185 · `12×SM`(800) 2144 —— **400 是峰值**（再多时部分和流量与归约
/// 开销开始反超并行收益）；400 与 272 的差主要在 `att_output`/`ffn_value`
/// 的 ks 从 4 抬到 5（块数 320→400），`ffn_key` 保持 ks=2。
/// 该组参数在 **B=8/16/32/64/128 全面优于** 272/4（+1.8/+1.1/+1.8/+0.7/+0.8%）。
///
/// 两条**闸门**（都由同会话交错 A/B 钉死）：
///
/// 1. `batch/BN ≤ 2`：`batch/BN` 就是「同一份权重 slab 被几批 batch 列块重读」，
///    它 ≥3 时 grid 的 batch 维本身已提供足够块数，实测 B=256 的 `att_output`/`ffn_value`
///    （`batch/BN = 4`、80 块）加 split-K 反而 **3250.4 → 3244.8（−0.2%）**。
/// 2. **多链合并的 launch（`z3`）不自动切**：`imma_gemm_dispatch_z3` 已经用 `grid.z = 3`
///    把 r/k/v 三条链并成一次 launch（块数 ×3），再切 K 的**收益小、代价却是 3 份**
///    （每条链一次归约）。实测 B=64：全切 **2090** vs 只切 `att_output`/`ffn_value` **2131**
///    ——r/k/v 那一刀净亏 **约 2%**。
///
/// `IMMA_KSPLIT=<n>` 强制指定（`=0` 关闭，可强行让 z3 也切）；`IMMA_KSPLIT_TARGET` /
/// `IMMA_KSPLIT_MAX` 覆盖目标块数与上限。
fn imma_ksplit(blocks: usize, ncol: usize, k: usize, chains: usize) -> usize {
    imma_ksplit_with(blocks, ncol, k, chains, None)
}

/// 同上，但可**强制**指定分块数（`forced`，测试门禁用；`None` 时读 `IMMA_KSPLIT`）。
/// 强制值同样要过「按 128 组整除」的收敛，否则 k 段会对不齐。
fn imma_ksplit_with(
    blocks: usize,
    ncol: usize,
    k: usize,
    chains: usize,
    forced: Option<usize>,
) -> usize {
    let max_ks = std::env::var("IMMA_KSPLIT_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5)
        .clamp(1, 16);
    let forced = forced.or_else(|| {
        std::env::var("IMMA_KSPLIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    });
    let mut ks = match forced {
        Some(v) => v.clamp(1, 8),
        None => {
            let target = std::env::var("IMMA_KSPLIT_TARGET")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(400);
            if blocks == 0 || blocks >= target || ncol > 2 || chains > 1 {
                1
            } else {
                target.div_ceil(blocks).min(max_ks)
            }
        }
    };
    let ktiles = k / 128;
    while ks > 1 && (!ktiles.is_multiple_of(ks) || (k / ks) < 128) {
        ks -= 1;
    }
    ks
}

/// 低秩链 GEMM 暂存初值（fp16 元素数）：`4×batch×C`（x 降位）+ `batch×Σmid_pad`（mid16）。
/// 按 batch=256 / C=2560 / Σmid_pad=640 计需 2,785,280 ⇒ 取 4 Mi ⇒ 余量充足。
const LR16_SCRATCH_INIT_ELEMS: usize = 1 << 22;

/// ffn_value 稠密 GEMM 的 `r2_16` 暂存初值（fp16 元素数）：`batch × fh`。
/// 按 batch=256 / fh=10240 计需 2,621,440 ⇒ 取 4 Mi（8 MiB）⇒ 余量充足。
const FFN16_SCRATCH_INIT_ELEMS: usize = 1 << 22;

/// 低秩 GEMM 的 tile（`LR1_*` 一级、`LR2_*` 二级；env 可覆盖做几何 A/B）。
/// 合法性约束（不满足则回落默认值）：
/// - `BM` 为 `16·LR_NWM = 64` 的倍数（warp 行布局）、`BN` 为 `8·LR_NWN = 16` 的倍数；
/// - `BK ∈ {64, 128}`（除尽 C=2560 与各 mid_pad）且 smem 预算 `(BM+BN)·(BK·2+16) ≤ 48KiB`。
///
/// ★ **一级默认 BN=16 而非 64**（2026-09-22 实测，B=256 / 8192 次启动）：
/// 一级的 `n`（各链 mid_pad）只有 64/128/128/320，BN=64 时 `grid.y=1` ⇒
/// 整个 launch 只有 `batch/BM = 4` 个 block（68 个 SM 上只跑 4 个块，**并行度塌**）。
/// BN=16 把 grid 抬到 16~80 个块：0.1292 → 0.0900 ms/次（**1.44×**）。
/// 代价是 x 片被重读 `n_pad/BN` 遍，但一级 k=2560 很长、n 很短 ⇒ 并行度远比这点重读值钱。
fn lr_tiles() -> (usize, usize, usize, usize, usize, usize) {
    let pick = |bmv: &str, bnv: &str, bkv: &str, bmd: usize, bnd: usize, bkd: usize| {
        let (bm, bn, bk) = (env_tile(bmv, bmd), env_tile(bnv, bnd), env_tile(bkv, bkd));
        let ok = bm.is_multiple_of(64)
            && bm <= 128
            && bn.is_multiple_of(16)
            && bn <= 128
            && (bk == 64 || bk == 128)
            && (bm + bn) * (bk * 2 + 16) <= 48 * 1024;
        if ok { (bm, bn, bk) } else { (bmd, bnd, bkd) }
    };
    let (b1, n1, k1) = pick("LR1_BM", "LR1_BN", "LR1_BK", 64, 16, 128);
    let (b2, n2, k2) = pick("LR2_BM", "LR2_BN", "LR2_BK", 64, 64, 64);
    (b1, n1, k1, b2, n2, k2)
}

/// ffn_value 稠密 GEMM 的 tile（`LR3_*`）。与 `lr_tiles()` 同规矩，
/// 额外约束 `BK` 必须除尽 `k = fh`（10240 对 64/128 都成立）。
///
/// ★ 2026-09-22 实测扫描（B=256，2048 次启动，ms/次）：**`BM=64` 0.8471** ·
/// `BM=128` 0.9983 · `BM=256` 1.5212。**BM 越小越快**——虽然 BM 小会让权重多读
/// `batch/BM` 遍（BM=64 ⇒ 4 遍 = 210MB/层），但 grid 从 80 块涨到 160 块，
/// **并行度又一次压过了重读**（同 §3d.4 的一级 tile、§3.4c 的反例）。
/// BM=32 不可用（`MT = BM/(16·LR_NWM) < 1`）。
fn lr3_tile(fh: usize, c: usize) -> (usize, usize, usize) {
    let (bm, bn, bk) = (
        env_tile("LR3_BM", 64),
        env_tile("LR3_BN", 64),
        env_tile("LR3_BK", 64),
    );
    let ok = bm.is_multiple_of(64)
        && bm <= 256
        && bn.is_multiple_of(16)
        && bn <= 256
        && (bk == 64 || bk == 128)
        && fh.is_multiple_of(bk)
        && c.is_multiple_of(bn)
        && (bm + bn) * (bk * 2 + 16) <= 48 * 1024;
    if ok { (bm, bn, bk) } else { (64, 64, 64) }
}

/// 从环境变量读 tile 尺寸（合法值 = 8 的倍数且 ≤256），非法/缺失则用默认值。
fn env_tile(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 8 && *v % 8 == 0 && *v <= 256)
        .unwrap_or(default)
}

/// 「默认开」的布尔开关：**显式 `0` 才关**（未设置 = 开）。
/// 与 `is_ok_and(|v| v != "0")`（默认关）区分开——后者用于实验性开关。
/// 已翻默认的：`GEMV_IMMA`（W8A8 张量核，§3.3c 贪心逐位一致）、
/// `LOWRANK_GEMM`（低秩链 fp16 tiled GEMM，1.43×，精度档与信天翁一致）。
fn env_on(var: &str) -> bool {
    std::env::var(var).map(|v| v != "0").unwrap_or(true)
}

/// `imma_gemm_batch`：int8×int8→int32 张量核批量 GEMM（W8A8）。
///
/// `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`：D(8 槽×8 行) = A(8 槽×16 k) · B(8 行×16 k)。
/// 片段↔线程映射已由 `imma_m8n8k16_probe` 逐位证实（见 IMMA 实施记录 §1）：
/// thread `lane`：`gID = lane>>2` / `tig = lane&3`；A 一个 u32（row=gID, col=tig*4+{0..3}）、
/// B 一个 u32（行=gID, k=tig*4+{0..3}，本工程权重 `[m][k]` 行主序天然就是 col-major B）、
/// D 两个寄存器（row=gID, col=tig*2+{0,1}）。
///
/// ★ **每组（128 k）flush 而非全程 int32 累加**：权重是**非对称**分组量化
/// `w = s[m,g]·q_u + z[m,g]`，而 IMMA 只吃 s8 ⇒ 组内把 byte `^0x80` 下移到有符号域、
/// 组末按 `sx[b,g]·s[m,g]` 还原 fp32，再加零点项 `z[m,g]·rowsum_x[b,g]`：
/// ```text
/// y[b,m] += s[m,g]·( sx[b,g]·S_g(b,m) + 128·sx[b,g]·colsum_xq[b,g] ) + z[m,g]·rowsum_x[b,g]
/// ```
/// `S_g` 是 IMMA 的 int32 输出；后两项随 `xaux`/`asz` 逐组取，每输出每组的额外开销
/// 只有 3 个 FMA（K=2560 ⇒ 20 组，相对 2560 个 MAC 可忽略）。
///
/// 共享内存布局：行跨距 `BK+16` 字节（= 36 words ≡ 4 mod 32），使片段读（线程
/// `(gID,tig)` 读 `row·WS + ks·16 + tig·4`）与落盘写都恰好铺满 32 个 bank，零冲突。
const IMMA_GEMM_SRC: &str = r#"
#ifndef IM_BM
#define IM_BM 64
#endif
#ifndef IM_BN
#define IM_BN 64
#endif
#ifndef IM_LB2
#define IM_LB2 2
#endif
#ifndef IM_PIPE
#define IM_PIPE 1
#endif
// smem 双缓冲的缓冲数（1 = 关，2 = 开）。host 只在 smem 预算够时注入 2。
#ifndef IM_NBUF
#define IM_NBUF 1
#endif
// ★ 2026-09-23：两个已实测「无效」的 staging 变体开关（默认关，留作反例，见各自注释）。
#ifndef IM_LDSEP
#define IM_LDSEP 0
#endif
#ifndef IM_VECLD
#define IM_VECLD 0
#endif
// ★ split-K：把 K 均分成 IM_KSPLIT 段，每段由**独立的一批 block** 计算，
// 各段只写自己的**部分和**（`op` 语义交给归约内核），再由 `imma_gemm_reduce` 按
// 固定顺序求和（确定性）。动机见 host 侧 `imma_ksplit`。
// `blockIdx.z` 的语义随之变为 `z = chain · IM_KSPLIT + ksp`；`IM_KSPLIT == 1` 时
// 退化为原来的 `z = chain`（单链调用恒 z = 0）。
#ifndef IM_KSPLIT
#define IM_KSPLIT 1
#endif

__device__ __forceinline__ void im_unpack_sz(unsigned int sz, float& s, float& z)
{
    s = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    z = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
__device__ __forceinline__ float im_relu2(float x) { return x > 0.f ? x * x : 0.f; }

extern "C" __global__ void __launch_bounds__(256, IM_LB2) imma_gemm_batch(
    const unsigned int* __restrict__ A0,
    const unsigned int* __restrict__ A1,
    const unsigned int* __restrict__ A2,
    const unsigned int* __restrict__ S0,
    const unsigned int* __restrict__ S1,
    const unsigned int* __restrict__ S2,
    const unsigned int* __restrict__ X0,
    const unsigned int* __restrict__ X1,
    const unsigned int* __restrict__ X2,
    const float4*       __restrict__ U0,
    const float4*       __restrict__ U1,
    const float4*       __restrict__ U2,
    float*              __restrict__ Y0,
    float*              __restrict__ Y1,
    float*              __restrict__ Y2,
    const int o0, const int o1, const int o2,
    const int m, const int k, const int batch)
{
    // ★ 2026-09-22：**3 条同形状链可合并成一次 launch**（`blockIdx.z` 选链）。
    // 动机：小 batch 时 `grid.y = batch/BN = 1`，每次 launch 只有 `M/BM` 个块
    // （B=16/BM=64 ⇒ 40 块，68 个 SM 半数空转）。r/k/v 三条链形状完全相同
    // （m=k=C、同 batch），合并后 `grid = (M/BM, batch/BN, 3)` = 120 块一次铺开。
    // 单链调用时把同一组指针传 3 遍、`grid.z = 1`（z 恒为 0）。
    // ★ split-K（`IM_KSPLIT > 1`）把 z 拆成 `z = chain · IM_KSPLIT + ksp`。
    const int z     = blockIdx.z;
    const int chain = z / IM_KSPLIT;
    const int ksp   = z % IM_KSPLIT;
    const unsigned int* __restrict__ aidx = (chain == 0) ? A0 : (chain == 1) ? A1 : A2;
    const unsigned int* __restrict__ asz  = (chain == 0) ? S0 : (chain == 1) ? S1 : S2;
    const unsigned int* __restrict__ xq   = (chain == 0) ? X0 : (chain == 1) ? X1 : X2;
    const float4*       __restrict__ xaux = (chain == 0) ? U0 : (chain == 1) ? U1 : U2;
    float*              __restrict__ y    = (chain == 0) ? Y0 : (chain == 1) ? Y1 : Y2;
    const int op = (chain == 0) ? o0 : (chain == 1) ? o1 : o2;
    // split-K 时本块负责的 k 区间（按 BK 对齐；host 保证 k/BK 能被 IM_KSPLIT 整除）
    const int kb = k / IM_KSPLIT;
    // split-K 时各段写**部分和**到 `y + ksp·batch·m`（布局 [KSPLIT][batch][m]），
    // `op` 一律不施加（归约内核负责）；`IM_KSPLIT == 1` 时偏移恒 0、语义不变。
    float* __restrict__ yp = (IM_KSPLIT > 1)
        ? y + (unsigned long long)ksp * (unsigned long long)batch * (unsigned long long)m
        : y;

    constexpr int BM = IM_BM;   // 每 block 的输出行数
    constexpr int BN = IM_BN;   // 每 block 的 batch 槽数
    constexpr int BK = 128;     // k 分块 = 量化组宽（flush 粒度与之对齐）
    constexpr int WS = BK + 16; // 行跨距（字节），见上方 bank 冲突说明
    constexpr int NST = BN / 8; // 每 warp 的 slot 组数（每 mma 吃 8 槽）
    constexpr int RG  = BM / 8; // 每 warp 的行数
    constexpr int NT  = RG / 8; // 每 warp 的行组数（每 mma 吃 8 行）

    __shared__ unsigned char ws_all[IM_NBUF * BM * WS];
    __shared__ unsigned char xs_all[IM_NBUF * BN * WS];
    // ★ 2026-09-22：`xaux` 的**本 k 组列**缓存（每 k-tile 只需 BN 个 float4 = 1KB）。
    // 旧版在 flush 里直接 `xaux[sb*G+gr]` 全局读：每线程 `NT×NST = 16` 次（CSE 后 8 次）
    // ⇒ 每块每 k-tile 32KB、每块 640KB、全步 13GB —— 比 W/X 的 staging 流量（10GB）还大。
    // 落 smem 后 flush 只读 smem，全局读降到 **每块每 k-tile 64 个 float4（1KB）**。
    __shared__ float4 xa_all[IM_NBUF * BN];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int gID  = lane >> 2;
    const int tig  = lane & 3;
    const int K4   = k >> 2;
    const int G    = k >> 7;
    // ★ 2026-09-22：**grid 维序可交换**（`IM_SWAP`）。CUDA 的线性块号 = `x + y*gridDim.x`
    // ⇒ 交换后**连续 4 个块共享同一份权重 slab**（只是 batch 槽块不同），
    // 权重被 `batch/BN` 重读时能命中 L2（slab = BM·K·1B，BM=64/K=2560 时 164KB，
    // 34 个 slab 组 ≈ 5.6MB ≈ L2 5.5MB）。默认维序下连续块各读不同 slab ⇒ 零 L2 复用。
#ifndef IM_SWAP
#define IM_SWAP 0
#endif
#if IM_SWAP
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
#else
    const int row0 = blockIdx.x * BM;
    const int col0 = blockIdx.y * BN;
#endif
    const int wrow0 = row0 + wid * RG;

    int   ai[NT][NST][2];
    float af[NT][NST][2];
    #pragma unroll
    for (int rg = 0; rg < NT; rg++)
        #pragma unroll
        for (int st = 0; st < NST; st++) {
            ai[rg][st][0] = 0; ai[rg][st][1] = 0;
            af[rg][st][0] = 0.f; af[rg][st][1] = 0.f;
        }

#if IM_PIPE
    // ⚠️ 软流水反例（见下方 k 循环内的注释）：实测 −36%，已废弃，仅留作实验开关。
    unsigned int wr[BM / 8], xr[BN / 8];
    #pragma unroll
    for (int it = 0; it < BM / 8; it++) {
        const int src = min(row0 + it * 8 + wid, m - 1);
        wr[it] = aidx[src * K4 + lane] ^ 0x80808080u;
    }
    #pragma unroll
    for (int it = 0; it < BN / 8; it++) {
        const int src = min(col0 + it * 8 + wid, batch - 1);
        xr[it] = xq[src * K4 + lane];
    }
#endif

    // —— 把某个 k-tile 的 W / X / xaux 搬进 `BUF` 号 smem 缓冲 ——
    // ★ 2026-09-22：抽成宏是为了做 **smem 双缓冲**（`IM_NBUF=2`）：在算当前块之前就把
    // 下一块的全局载入**发出去**，让 ~600ns 的载入延迟与 mma/flush 重叠。
    // 这解决的是**小 batch** 的病：`grid.y = batch/BN = 1` 时整个 launch 只有 `M/BM`
    // 个块（r/k/v/o 各 40 个），每 SM 只摊到 1 个块 ⇒ 「载入→同步→计算」完全串行，
    // 实测每 k-tile 4.25μs（其中载入带宽只占 2.2μs）。
    // ⚠️ 寄存器预取版（`IM_PIPE`）在 BM=128 下溢出 −36%，**但 smem 双缓冲不占寄存器**。
    #define IM_STAGE(BUF, KT)                                                    \
    do {                                                                         \
        unsigned char* _w = ws_all + (BUF) * (BM * WS);                          \
        unsigned char* _x = xs_all + (BUF) * (BN * WS);                          \
        float4*        _a = xa_all + (BUF) * BN;                                 \
        const int _kt = (KT);                                                    \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < BM / 8; it++) {                                    \
            const int r = it * 8 + wid;                                          \
            const int src = min(row0 + r, m - 1);                                \
            *(unsigned int*)(_w + r * WS + lane * 4) =                           \
                aidx[src * K4 + (_kt >> 2) + lane] ^ 0x80808080u;                \
        }                                                                        \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < BN / 8; it++) {                                    \
            const int r = it * 8 + wid;                                          \
            const int src = min(col0 + r, batch - 1);                            \
            *(unsigned int*)(_x + r * WS + lane * 4) = xq[src * K4 + (_kt >> 2) + lane]; \
        }                                                                        \
        {                                                                        \
            const int _gr = _kt >> 7;                                            \
            for (int r = tid; r < BN; r += 256) {                                \
                const int src = min(col0 + r, batch - 1);                        \
                _a[r] = xaux[src * G + _gr];                                     \
            }                                                                    \
        }                                                                        \
    } while (0)

    // ★ 2026-09-23：**载入与落 smem 分离**（`IMMA_LDSEP=1`）。
    // 动机：实测每 SM 的有效带宽只有 **4.4 GB/s**（≈ 每 22 cycle 才发出 1 条 warp 级
    // 128B 请求），而本机纯流式读可达 7.4 GB/s/SM。怀疑编译器把 `LDG → STS` 逐对发出
    // （每对要等一次 ~600 cycle 的全局延迟）⇒ 9 次串行。
    // 修法：先把 BM/8 + BN/8 个 32 位载入全部取到寄存器，再统一落 smem。
    // 代价：+9 个寄存器（BM=64/BN=8）；BM=128/BN=64 时 +24，可能压回溢出。
    // ⚠️ **实测：收益在噪声内**（ffn_key 0.0935→0.0924、ffn_value 0.0719→0.0720、
    // att_output 0.0262→0.0243、head 0.5243→0.5147），**默认关**。⇒ 载入延迟本来
    // 就被多块并发掩盖了，不靠单线程 ILP 救。
    #define IM_STAGE_LDSEP(BUF, KT)                                              \
    do {                                                                         \
        unsigned char* _w = ws_all + (BUF) * (BM * WS);                          \
        unsigned char* _x = xs_all + (BUF) * (BN * WS);                          \
        float4*        _a = xa_all + (BUF) * BN;                                 \
        const int _kt = (KT);                                                    \
        unsigned int _wr[BM / 8], _xr[BN / 8];                                   \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < BM / 8; it++) {                                    \
            const int src = min(row0 + it * 8 + wid, m - 1);                     \
            _wr[it] = aidx[src * K4 + (_kt >> 2) + lane] ^ 0x80808080u;          \
        }                                                                        \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < BN / 8; it++) {                                    \
            const int src = min(col0 + it * 8 + wid, batch - 1);                 \
            _xr[it] = xq[src * K4 + (_kt >> 2) + lane];                          \
        }                                                                        \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < BM / 8; it++)                                      \
            *(unsigned int*)(_w + (it * 8 + wid) * WS + lane * 4) = _wr[it];     \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < BN / 8; it++)                                      \
            *(unsigned int*)(_x + (it * 8 + wid) * WS + lane * 4) = _xr[it];     \
        {                                                                        \
            const int _gr = _kt >> 7;                                            \
            for (int r = tid; r < BN; r += 256) {                                \
                const int src = min(col0 + r, batch - 1);                        \
                _a[r] = xaux[src * G + _gr];                                     \
            }                                                                    \
        }                                                                        \
    } while (0)

// ★ 2026-09-23：**16 字节/线程的向量化 staging**（`IMMA_VECLD=1`）。
    //
    // 动机：消融探针（临时 `IMMA_NOMMA=1`，已删）显示本内核**去掉全部 mma/flush 只省 9%**
    // （ffn_key 0.0935 → 0.0811 ms），即它 **~91% 是「载入受限」**。
    // 而旧 staging 每 warp 一条指令只搬**一行 128B**（32 lane × 4B）。
    //
    // 修法：改成每 lane 一个 `uint4`（16B）⇒ **一条 warp 指令搬 4 行**（32×16B = 512B）。
    // 行内偏移 `c = (lane & 7) * 4`（4 个 u32 = 16B，8 lane 覆盖 128B 行）；
    // 行号 `r = 基址 + (lane >> 3)`（4 组 lane 各负责一行）。
    // 越界行用 `min(..., BM-1)` **夹取**而非分支——重复写同一行同值是无害的。
    //
    // ⚠️ **实测：零收益**（ffn_key 0.0912→0.0911、ffn_value 0.0719→0.0713、
    // head 0.4873→0.4859），**默认关**。⇒ 瓶颈不是「指令发射速率」，而是
    // **128B 事务本身的数量/带宽**（换算后 IMMA 的聚合已达 **370~420 GB/s**，
    // 即本机 500 GB/s 可达上限的 74~84%）——**该内核已接近内存 roofline**。
    #define IM_STAGE_VEC(BUF, KT)                                                \
    do {                                                                         \
        unsigned char* _w = ws_all + (BUF) * (BM * WS);                          \
        unsigned char* _x = xs_all + (BUF) * (BN * WS);                          \
        float4*        _a = xa_all + (BUF) * BN;                                 \
        const int _kt = (KT);                                                    \
        const int _ro = (lane >> 3), _co = (lane & 7) * 4;                       \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < BM / 32; it++) {                                   \
            const int r = min(wid * 4 + it * 32 + _ro, BM - 1);                  \
            const int src = min(row0 + r, m - 1);                                \
            const uint4 v = *(const uint4*)(aidx + src * K4 + (_kt >> 2) + _co);  \
            const uint4 o = make_uint4(v.x ^ 0x80808080u, v.y ^ 0x80808080u,      \
                                       v.z ^ 0x80808080u, v.w ^ 0x80808080u);     \
            *(uint4*)(_w + r * WS + _co * 4) = o;                                 \
        }                                                                        \
        _Pragma("unroll")                                                        \
        for (int it = 0; it < (BN + 31) / 32; it++) {                            \
            const int r = min(wid * 4 + it * 32 + _ro, BN - 1);                  \
            const int src = min(col0 + r, batch - 1);                            \
            const uint4 v = *(const uint4*)(xq + src * K4 + (_kt >> 2) + _co);    \
            *(uint4*)(_x + r * WS + _co * 4) = v;                                 \
        }                                                                        \
        {                                                                        \
            const int _gr = _kt >> 7;                                            \
            for (int r = tid; r < BN; r += 256) {                                \
                const int src = min(col0 + r, batch - 1);                        \
                _a[r] = xaux[src * G + _gr];                                     \
            }                                                                    \
        }                                                                        \
    } while (0)

#if IM_LDSEP
#define IM_STAGE_USE(BUF, KT) IM_STAGE_LDSEP(BUF, KT)
#elif IM_VECLD
#define IM_STAGE_USE(BUF, KT) IM_STAGE_VEC(BUF, KT)
#else
#define IM_STAGE_USE(BUF, KT) IM_STAGE(BUF, KT)
#endif

#if IM_NBUF > 1
    IM_STAGE_USE(0, 0);
    __syncthreads();
#endif
    for (int kt = ksp * kb; kt < (ksp + 1) * kb; kt += BK) {
        // —— 搬 W/X 到 smem。每 warp 一次搬一整行（32 个 u32 = 128B 连续）⇒ 完全合并。
        //    权重落盘时做 `^0x80`：磁盘 idx 是 0..255 无符号，IMMA 只吃有符号 s8。
        // ⚠️ **已实测的反例（勿重试）**：把这块改成「寄存器预取下一块 + 与 mma 重叠」的
        // 软流水（`IM_PIPE`，每线程 +24 个 u32 预取寄存器）⇒ 端到端 **2550 → 1636 tok/s
        // （−36%）**。原因是累加器已占 `NT×NST×4 = 64` 个寄存器，再加 24 个直接溢出到
        // local memory。**寄存器是硬约束 ⇒ 只能走 smem 双缓冲（`IM_NBUF=2`）。**
#if IM_NBUF > 1
        // 先发下一块的载入（写另一号缓冲，与下面的 mma/flush 重叠），本块数据上一轮已就绪
        const int cur = ((kt / BK) & 1);
        if (kt + BK < (ksp + 1) * kb) IM_STAGE_USE(cur ^ 1, kt + BK);
#else
        const int cur = 0;
        IM_STAGE_USE(0, kt);
        __syncthreads();
#endif
        unsigned char* ws = ws_all + cur * (BM * WS);
        unsigned char* xs = xs_all + cur * (BN * WS);
        float4*      xa_s = xa_all + cur * BN;

        // s/z 只随行变 ⇒ 同一 warp 的 NT 个行组各取一次，供组内全部 slot 组复用
        const int gr = kt >> 7;
        float sm[NT][2], zm[NT][2];
        #pragma unroll
        for (int rg = 0; rg < NT; rg++) {
            const int r0 = wrow0 + rg * 8 + tig * 2;
            const unsigned int sz0 = (r0 < m) ? asz[r0 * G + gr] : 0u;
            const unsigned int sz1 = (r0 + 1 < m) ? asz[(r0 + 1) * G + gr] : 0u;
            im_unpack_sz(sz0, sm[rg][0], zm[rg][0]);
            im_unpack_sz(sz1, sm[rg][1], zm[rg][1]);
        }

        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            const int ko = ks * 16 + tig * 4;
            unsigned int bf[NT], afr[NST];
            #pragma unroll
            for (int rg = 0; rg < NT; rg++)
                bf[rg] = *(const unsigned int*)(ws + (wid * RG + rg * 8 + gID) * WS + ko);
            #pragma unroll
            for (int st = 0; st < NST; st++)
                afr[st] = *(const unsigned int*)(xs + (st * 8 + gID) * WS + ko);
            #pragma unroll
            for (int rg = 0; rg < NT; rg++)
                #pragma unroll
                for (int st = 0; st < NST; st++)
                    asm volatile(
                        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                        : "+r"(ai[rg][st][0]), "+r"(ai[rg][st][1])
                        : "r"(afr[st]), "r"(bf[rg]));
        }

        // —— 组末 flush：int32 → fp32 尺度还原 + 零点修正，然后清零给下一组用
        #pragma unroll
        for (int rg = 0; rg < NT; rg++)
            #pragma unroll
            for (int st = 0; st < NST; st++) {
                const float4 au = xa_s[st * 8 + gID];   // smem（旧版是全局读，见上方注释）
                af[rg][st][0] += sm[rg][0] * (au.x * (float)ai[rg][st][0] + au.y) + zm[rg][0] * au.z;
                af[rg][st][1] += sm[rg][1] * (au.x * (float)ai[rg][st][1] + au.y) + zm[rg][1] * au.z;
                ai[rg][st][0] = 0;
                ai[rg][st][1] = 0;
            }
        __syncthreads();
    }

    #pragma unroll
    for (int rg = 0; rg < NT; rg++)
        #pragma unroll
        for (int st = 0; st < NST; st++) {
            const int sb = col0 + st * 8 + gID;
            if (sb >= batch) continue;
            const int r0 = wrow0 + rg * 8 + tig * 2;
            float v0 = af[rg][st][0], v1 = af[rg][st][1];
#if IM_KSPLIT > 1
            // split-K：只写部分和（不施加 op、不读旧值），归约内核负责 op。
            if (r0 < m) yp[sb * m + r0] = v0;
            if (r0 + 1 < m) yp[sb * m + r0 + 1] = v1;
#else
            if (op == 0) { v0 = im_relu2(v0); v1 = im_relu2(v1); }
            if (op == 4) {
                // 覆盖写、但落 fp16（rkv_stage1 的 out_v 是 fp16 语义，舍入与
                // `__float2half` 一致——SIMT 路径当年就这么写的）
                __half* yh = reinterpret_cast<__half*>(y);
                if (r0 < m) yh[sb * m + r0] = __float2half(v0);
                if (r0 + 1 < m) yh[sb * m + r0 + 1] = __float2half(v1);
            } else if (op == 0 || op == 3) {
                if (r0 < m) y[sb * m + r0] = v0;
                if (r0 + 1 < m) y[sb * m + r0 + 1] = v1;
            } else {
                if (r0 < m) y[sb * m + r0] += v0;
                if (r0 + 1 < m) y[sb * m + r0 + 1] += v1;
            }
#endif
        }
}
"#;

/// split-K 的确定性归约内核（独立源码 —— 只编这一条，避免把大 GEMM 再编一遍）。
/// 语义与 `imma_gemm_batch` 的 epilogue 一致，见 `IMMA_GEMM_SRC` 里的注释。
const IMMA_REDUCE_SRC: &str = r#"
__device__ __forceinline__ float im_relu2_r(float x) { return x > 0.f ? x * x : 0.f; }

extern "C" __global__ void imma_gemm_reduce(
    const float* __restrict__ partial,   // [ksplit][batch][m]
    float*       __restrict__ y,         // [batch][m]（op==4 时按 fp16 视图写）
    const int ksplit, const int batch, const int m, const int op)
{
    const long long total = (long long)batch * (long long)m;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        const float* p = partial + i;
        float acc = p[0];
        for (int s = 1; s < ksplit; s++) acc += p[(long long)s * total];
        if (op == 0) {
            y[i] = im_relu2_r(acc);
        } else if (op == 4) {
            reinterpret_cast<__half*>(y)[i] = __float2half(acc);
        } else if (op == 3) {
            y[i] = acc;
        } else {
            y[i] += acc;
        }
    }
}
"#;

/// gemv_variant_mb16：batch 线性层第二代内核（2026-09-21 隔离计时择优）。
///
/// 相对 `gemv_variant_mb` 的两处结构性改动，均在 `probe3.cu` 隔离计时中定位：
///   ① **BGRP=8 使权重只读一遍**（旧版 BGRP=4，B=8 时 grid.y=2 把权重读了两遍）；
///   ② **激活 x 改 fp16**（旧版 x 为 fp32，每 block 要以 float4 重读 BGRP 份
///      x；x 被 m/ROWS 个 block 重复读，是 L2 流量的最大项）。
/// 实测（ffn_key 形状 m=10240/k=2560/B=8，权重一遍 = 26.2MB）：
///   旧版 0.2275 ms → 本内核 0.1494 ms（**1.52×**）；对照 fp16 常驻权重 0.1508 ms
///   ——即**不必放弃 int8 常驻**（8GB 卡约束，计划 D1）也能拿到 fp16 常驻的收益。
///
/// 几何（第三代，2026-09-21）：**block = 128 线程；全体线程一起按 k 维切分，
/// 同时覆盖 `MB16_ROWS` 行 × BGRP 个 slot**（每线程累加器 = ROWS×BGRP 个 half2）。
///
/// ★ **不变量：k 维恒由整块 128 线程切分 ⇒ 每累加器的 fp16 串行深度 = kvi4/128。**
/// 这条不是风格问题，而是数值口径：`gemv_variant_int8_matches_cpu` /
/// `gemv_variant_mb_matches_single` 两个门禁都按「batch 与单流同累加顺序」收紧容差。
/// 2026-09-21 实测反例：若把行按 warp 切分、k 只在 32 lane 内切（深度 5→20），
/// 两个门禁立刻越界（CPU 参照 0.0211 > 2e-2）；把线程数抬到 512×4 行组虽能保住
/// 深度，却因每线程迭代数太少而**完全吃不到收益**（relu2 0.1625→0.1663，反慢）。
/// ⇒ 提高「每 block 行数」只能走 **加大 ROWS**（代价是累加器寄存器）。
///
/// 收益来源（与 x 重读流量相关但不完全等价）：ROWS 越大，x 片被越少的 block 重读
/// （x 的 L2 流量 = (M/ROWS) × batch × K × 2B，head 形状 M=65536 时曾达 1.3 GB/步），
/// 同时每线程「每轮载入 x 一次、喂 ROWS 行」的算术强度也随之提高。
///
/// ROWS 由调用方以 `#define MB16_ROWS` 注入（`GEMV_MB16_ROWS` 环境变量，
/// 便于不重编译地做几何 A/B）。
/// dispatch (ceil(M/ROWS), ceil(batch/8), 1)；x 为 [batch, K] fp16，
/// 门控已由 cast 内核预乘。
const GEMV_VARIANT_MB16_SRC: &str = r#"
#ifndef MB16_ROWS
#define MB16_ROWS 2
#endif
__device__ __forceinline__ void unpack_mb_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
__device__ __forceinline__ float relu2_mb16(float x) { return x > 0.f ? x * x : 0.f; }
// 绕过 L1 的 half2 载入（x 是流式数据，不能让它挤掉要靠 L1 命中的权重 tile）
__device__ __forceinline__ half2 ldcg_h2(const __half* p)
{
    const unsigned int u = __ldcg(reinterpret_cast<const unsigned int*>(p));
    half2 r;
    r.x = __ushort_as_half((unsigned short)(u & 0xFFFFu));
    r.y = __ushort_as_half((unsigned short)(u >> 16));
    return r;
}

extern "C" __global__ void __launch_bounds__(128, 4) gemv_variant_mb16(
    const unsigned int* __restrict__ aidx,  // int8 idx [M, K/4]
    const unsigned int* __restrict__ asz,   // int8 sz  [M, K/128]
    const __half*       __restrict__ x,     // [batch, K] fp16（门控已预乘）
    float*              __restrict__ y,     // [batch, M]
    const int m,
    const int k,
    const int batch,
    const int op)   // 0=relu2 覆盖, 1/2=累加, 3=覆盖
{
    constexpr int ROWS = MB16_ROWS;
    constexpr int BGRP = 8;
    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x * ROWS;
    const int kvi4 = k / 4;
    const int kg   = k / 128;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float partial[4 /*warp*/][ROWS][BGRP];

    // ★ 一个 block 覆盖**全部** batch 槽（batch 分块在块内循环），
    // 于是权重（本 block 的 ROWS 行 × K）**从 DRAM 只读一遍**，
    // 第 1 块之后的批次分块全部命中 L1。
    // 旧版 grid.y = ceil(B/BGRP)，**每个 grid.y 切片都要把全部权重再读一遍**
    // ⇒ 权重总流量 = ceil(B/8) × 权重，永远摊薄不到 8 槽以上：
    // 实测 B=8→128 每槽成本只降 1.3×（B=8 3.30 ms/槽 → B=128 2.52 ms/槽），
    // 而按权重主导本应接近线性摊薄。这正是「并发上去吞吐不涨」的根因。
    const int nchunk = (batch + BGRP - 1) / BGRP;
    for (int c = 0; c < nchunk; ++c) {
        if (c > 0) __syncthreads();   // 复用 partial 前必须同步（c 全块一致）
        const int b0   = c * BGRP;
        const int bcnt = min(BGRP, batch - b0);

        half2 hacc[ROWS][BGRP];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) hacc[r][b] = __half2half2(0.f);

        // k 维由整块 128 线程切分（不变量：累加深度 = kvi4/128）
        for (int kq = tid; kq < kvi4; kq += 128) {
            const int kbase = kq * 4;
            const int gr = kg > 0 ? (kbase / 128) : 0;
            half2 hx01[BGRP], hx23[BGRP];
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                if (b < bcnt) {
                    const half2* q = reinterpret_cast<const half2*>(x + (b0 + b) * k + kbase);
                    hx01[b] = q[0];
                    hx23[b] = q[1];
                }
            }
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                const unsigned int p = aidx[row * kvi4 + kq];
                float sc, zr;
                unpack_mb_sz(asz[row * kg + gr], sc, zr);
                const half2 w01 = __floats2half2_rn(
                    sc * (float)((p >> 0) & 0xFFu) + zr, sc * (float)((p >> 8) & 0xFFu) + zr);
                const half2 w23 = __floats2half2_rn(
                    sc * (float)((p >> 16) & 0xFFu) + zr, sc * (float)((p >> 24) & 0xFFu) + zr);
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    hacc[r][b] = __hfma2(hx01[b], w01, hacc[r][b]);
                    hacc[r][b] = __hfma2(hx23[b], w23, hacc[r][b]);
                }
            }
        }

        // 两级归约：warp 内 shuffle → 4 个 warp 经 smem 合并（与单流版同序）
        float acc[ROWS][BGRP];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                const float2 f = __half22float2(hacc[r][b]);
                acc[r][b] = f.x + f.y;
            }
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                float v = acc[r][b];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, off);
                }
                if (lane == 0) partial[warp][r][b] = v;
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    float sum = 0.f;
                    #pragma unroll
                    for (int w2 = 0; w2 < 4; w2++) sum += partial[w2][r][b];
                    const int yb = (b0 + b) * m + row;
                    if (op == 0) y[yb] = relu2_mb16(sum);
                    else if (op == 3) y[yb] = sum;
                    else y[yb] += sum;
                }
            }
        }
    }
}
"#;

/// gemv_variant_mb CUDA kernel：batch 并发的**权重复用**版（信天翁 rows 模型）。
/// 与 gemv_variant 的区别：不是 grid.y=slot 各读全量权重（带宽 ×B），而是
/// 每 block 一次读权重（4 行 × K），在寄存器累加器中复用给 BGRP 个 slot——
/// weight-bound 场景带宽 ≈ 1/ceil(B/BGRP)。
/// dispatch (M/4, ceil(batch/BGRP), 1)；x/y 为 [batch, ...]（slot 主序）。
/// op 语义与 gemv_variant 一致（0=relu2、1=mul_add、2=add、3=plain）。
const GEMV_VARIANT_MB_SRC: &str = r#"
__device__ __forceinline__ void unpack_mb_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
__device__ __forceinline__ float relu2_mb(float x) { return x > 0.f ? x * x : 0.f; }
__device__ __forceinline__ void load_half4_f4_mb(
    const __half* p, float& x0, float& x1, float& x2, float& x3)
{
    const half2 h01 = *reinterpret_cast<const half2*>(p);
    const half2 h23 = *reinterpret_cast<const half2*>(p + 2);
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    x0 = f01.x; x1 = f01.y; x2 = f23.x; x3 = f23.y;
}

extern "C" __global__ void __launch_bounds__(128, 4) gemv_variant_mb(
    const __half*         __restrict__ Af16,  // fp16 [M*K]（wtype==0 用）
    const unsigned int*   __restrict__ aidx,  // int8 idx [M,K/4]
    const __half*         __restrict__ alut,  // 保留（当前未使用）
    const unsigned int*   __restrict__ asz,   // int8 sz [M,K/128]
    const float*          __restrict__ x,     // [batch, K]
    const __half*         __restrict__ g,     // [batch, K] fp16 门控（op==1 用）
    float*                __restrict__ y,     // [batch, M]（op==2 累加式读改写）
    const int m,
    const int k,
    const int batch,
    const int wtype,   // 0=f16, 2=int8
    const int op)      // 0=relu2, 1=mul_add, 2=add, 3=plain
{
    // 块内几何：ROWS 行 × BGRP slot。两者乘积（16 个 half2 累加器）决定每线程
    // FMA 量与寄存器占用。
    // 2026-09-21 隔离计时实测（B=8，单次调用）：
    //   ROWS=4,BGRP=4：0.075ms(att_output) / 0.247ms(ffn_key)
    //   ROWS=4,BGRP=8：无改善
    //   ROWS=2,BGRP=8：无改善（0.070 / 0.252）——权重流量减半未换来收益
    // Phase 1a 追加（2026-09-21）：对 `gemv_variant_mb` 做过 **7 组隔离计时实验**
    // （kernel_bench，跨轮偏差 ≤4.5%），**全部无改善或更慢**：
    //   ① BGRP 4→8（权重读 2 遍→1 遍）        relu2 B=8 0.2497→0.2914（+17%）
    //   ② ROWS=2/BGRP=8（流量减半、每线程 FMA 不变）  0.070/0.252 平
    //   ③ blockDim 128→256                    端到端更差（217 vs 260 tok/s）
    //   ④ 宽载入 uint4（16 int8/次，载入指令 28→8）  0.3451（+38%）
    //   ⑤ ④ + blockDim=160 保证行程整除         0.3416（+37%，⇒ 失衡非主因）
    //   ⑥ ④ + 4 组独立累加器                   1.2092（**+384%**，寄存器溢出）
    //   ⑦ BGRP=8 复核（kernel_bench）          0.2914（+17%）
    // ★ 结论：**本内核处于局部最优，参数/微观结构无法改进**。瓶颈既非权重带宽
    //   （①⑦ 否证）、也非 FMA（12% 峰值）、也非 issue（~9%），而是**寄存器受限**
    //   —— hacc[ROWS][BGRP]=16 个 half2 + hx[8] 已是 __launch_bounds__(128,4)
    //   上限内的甜点，任何增加并行维度/缓冲/载入宽度的改动都会把它推过临界点。
    //   另有未验证观察：x 被 grid.x（2560）个 block 重复读，x 流量 ≈210MB 是权重
    //   54MB 的 4 倍；但减少它只能靠增大 ROWS/BGRP（＝加寄存器），同样撞墙。
    constexpr int ROWS = 4;
    constexpr int BGRP = 4;
    const int tid  = threadIdx.x;
    const int b0   = blockIdx.y * BGRP;
    const int bcnt = min(BGRP, batch - b0);
    const int row0 = blockIdx.x * ROWS;
    const int kvi4 = k / 4;
    const int kg   = k / 128;

    // half2 累加器：ROWS 行 × BGRP slot（fp16 FMA 吞吐路径，与单序列版一致）。
    half2 hacc[ROWS][BGRP];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < BGRP; b++) hacc[r][b] = __half2half2(0.f);

    if (wtype == 0) {
        // ===== fp16 路径：权重 half4 读一次，逐 slot FMA =====
        const int k4 = k & ~3;
        for (int kq = tid * 4; kq < k4; kq += blockDim.x * 4) {
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                if (b >= bcnt) break;
                const float4 xv = *reinterpret_cast<const float4*>(x + (b0 + b) * k + kq);
                float gx = 1.f, gy = 1.f, gz = 1.f, gw = 1.f;
                if (op == 1) {
                    load_half4_f4_mb(g + (b0 + b) * k + kq, gx, gy, gz, gw);
                }
                const half2 hx01 = __floats2half2_rn(xv.x * gx, xv.y * gy);
                const half2 hx23 = __floats2half2_rn(xv.z * gz, xv.w * gw);
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    const __half* wj = Af16 + (row0 + r) * k + kq;
                    hacc[r][b] = __hfma2(hx01, *reinterpret_cast<const half2*>(wj), hacc[r][b]);
                    hacc[r][b] = __hfma2(hx23, *reinterpret_cast<const half2*>(wj + 2), hacc[r][b]);
                }
            }
        }
    } else {
        // ===== int8 路径：反量化一次 + x 片预转 half2 一次，逐行/槽 FMA =====
        //
        // 【Phase 1a 负结果留档，勿重试】曾试「宽载入 / ILP 改造」两种形态，均更慢：
        //   ① 权重改 uint4（16 个 int8/次，载入指令数 28→8）+ 门控向量化：
        //      relu2 B=8 0.2497→0.3451ms（+38%）；blockDim 128→160 保证行程整除后
        //      仍 0.3416ms ⇒ **失衡不是主因**。
        //   ② 再加 4 组独立累加器打破依赖链（hacc[4][ROWS][BGRP]）：
        //      relu2 B=8 0.2497→**1.2092ms（+384%）** ⇒ 典型**寄存器溢出到 local memory**。
        // 结论：本内核是**寄存器受限**的（hacc[ROWS][BGRP]=16 个 half2 已是甜点），
        // 加宽载入/加累加器都会把它推过临界点。ILP 不是可用的杠杆。
        // 真正的结构性方案见计划 Phase 1c（cp.async 多级流水，sm_80+）与 Phase 2/3。
        for (int kq = tid; kq < kvi4; kq += blockDim.x) {
            const int kbase = kq * 4;
            const int gr = kg > 0 ? (kbase / 128) : 0;
            // x 片按 slot 取一次（float4 向量化加载），预乘门控后转 half2，供 4 行复用。
            // 旧写法把 4 次标量 x 加载 + 4 次 __floats2half2_rn 放在 r 循环内，
            // 使 x 被重复读 4 行（每行 4 次标量读）——这是 batch GEMV 耗时随 B
            // 严格线性（权重摊薄收益为零）的主因。
            half2 hx01[BGRP];
            half2 hx23[BGRP];
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                if (b < bcnt) {
                    const float4 xv =
                        *reinterpret_cast<const float4*>(x + (b0 + b) * k + kbase);
                    float g0 = 1.f, g1 = 1.f, g2 = 1.f, g3 = 1.f;
                    if (op == 1) {
                        const __half* gq = g + (b0 + b) * k + kbase;
                        g0 = __half2float(gq[0]);
                        g1 = __half2float(gq[1]);
                        g2 = __half2float(gq[2]);
                        g3 = __half2float(gq[3]);
                    }
                    hx01[b] = __floats2half2_rn(xv.x * g0, xv.y * g1);
                    hx23[b] = __floats2half2_rn(xv.z * g2, xv.w * g3);
                }
            }
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                const unsigned int p = aidx[row * kvi4 + kq];
                float sc, zr;
                unpack_mb_sz(asz[row * kg + gr], sc, zr);
                const float f0 = sc * (float)((p >> 0) & 0xFFu) + zr;
                const float f1 = sc * (float)((p >> 8) & 0xFFu) + zr;
                const float f2 = sc * (float)((p >> 16) & 0xFFu) + zr;
                const float f3 = sc * (float)((p >> 24) & 0xFFu) + zr;
                const half2 w01 = __floats2half2_rn(f0, f1);
                const half2 w23 = __floats2half2_rn(f2, f3);
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    hacc[r][b] = __hfma2(hx01[b], w01, hacc[r][b]);
                    hacc[r][b] = __hfma2(hx23[b], w23, hacc[r][b]);
                }
            }
        }
    }

    // half2 → float（尾部标量并入用 float 累加）。
    float acc[ROWS][BGRP];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < BGRP; b++) {
            const float2 f = __half22float2(hacc[r][b]);
            acc[r][b] = f.x + f.y;
        }

    // 尾部标量兜底（fp16 路径 k 非 4 倍数时；int8 路径 k 恒为 4 倍数）。
    if (wtype == 0) {
        const int k4 = k & ~3;
        for (int kk = k4 + tid; kk < k; kk += blockDim.x) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                const float wv = __half2float(Af16[row * k + kk]);
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    acc[r][b] += wv * x[(b0 + b) * k + kk];
                }
            }
        }
    }

    // warp shuffle 归约（4 行 × BGRP slot；只有 1 次 __syncthreads）。
    // partial 按最大 8 warp（256 线程）开，warp 数与求和上界都取运行时 blockDim.x>>5
    // ——原先硬编码 4 warp：blockDim=256 时 partial[4..7] 越界写共享内存 → 访问违例。
    __shared__ float partial[8 /*warp*/][ROWS /*row*/][BGRP];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = blockDim.x >> 5;
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        #pragma unroll
        for (int b = 0; b < BGRP; b++) {
            float v = acc[r][b];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) partial[warp][r][b] = v;
        }
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= m) continue;
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                if (b >= bcnt) break;
                float sum = 0.f;
                for (int w = 0; w < nwarp; w++) sum += partial[w][r][b];
                const int yb = (b0 + b) * m + row;
                if (op == 0) y[yb] = relu2_mb(sum);
                else if (op == 3) y[yb] = sum;
                else y[yb] += sum;
            }
        }
    }
}
"#;

/// 稀疏 FFN value 投影内核：x += r2 @ ffn_value，r2 已 relu²（约 96% 稀疏）。
/// 对齐 Albatross `cmix_sparse_down_relu_one_vtile_hfma2_split2_kernel` 的平铺布局与稀疏遍历：
/// 每 block 处理一个 f 片（TILE=128）和一个 c 片（C_TILE=256），只读取 r2 非零 f 对应的权重列，
/// 把 52MB 权重读取降到 ~2MB（带宽 ~17× 削减）。
/// value_tiled 布局：元素 (f,c) → [f_block][c_block][f_local][c_local]，
///   f_block=f/128, c_block=c/256, tile_base=((f_block*c_blocks+c_block)*128)*256。
/// dispatch (fh/128, c/256, 1)，每 block 128 线程。x 已含残差，跨 f_block 用原子累加。
const FFN_VALUE_SPARSE_SRC: &str = r#"
extern "C" __global__ void ffn_value_sparse_add(
    const float*    __restrict__ r2,          // [fh] relu² 输出
    const __half*   __restrict__ value_tiled, // [fh*C] 平铺布局
    float*          __restrict__ x,           // [C] 就地原子累加（已含残差）
    const int c,
    const int fh)
{
    constexpr int TILE    = 128;
    constexpr int C_TILE  = 256; // 2 * TILE
    __shared__ float r2_slice[TILE];
    __shared__ int   nnz_ids[TILE];
    __shared__ int   nnz_count;
    __shared__ int   warp_counts[TILE / 32];
    __shared__ int   warp_prefix[TILE / 32];

    const int f_block = blockIdx.x;
    const int c_block = blockIdx.y;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const int start_f = f_block * TILE;

    // 读 r2 片并统计非零（r2 已 relu²，非零即 r2v != 0）。
    float r2v = 0.f;
    bool  nonzero = false;
    int   local_pos = 0;
    if (tid < TILE) {
        r2v = r2[start_f + tid];
        r2_slice[tid] = r2v;
        nonzero = (r2v != 0.0f);
        unsigned mask = __ballot_sync(0xffffffffu, nonzero);
        local_pos = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0) warp_counts[warp] = __popc(mask);
    }
    __syncthreads();
    if (tid == 0) {
        int s = 0;
#pragma unroll
        for (int w = 0; w < TILE / 32; ++w) {
            warp_prefix[w] = s;
            s += warp_counts[w];
        }
        nnz_count = s;
    }
    __syncthreads();
    if (tid < TILE && nonzero) {
        nnz_ids[warp_prefix[warp] + local_pos] = tid;
    }
    __syncthreads();

    const int c_blocks = c / C_TILE;
    const int tile_base = ((f_block * c_blocks + c_block) * TILE) * C_TILE;
    const int c0 = c_block * C_TILE + tid * 2;
    float acc0 = 0.f, acc1 = 0.f;
    for (int i = 0; i < nnz_count; i += 2) {
        const int f0 = nnz_ids[i];
        const __half* w0 = value_tiled + (long long)tile_base + f0 * C_TILE + tid * 2;
        const float a0 = r2_slice[f0];
        acc0 += a0 * __half2float(w0[0]);
        acc1 += a0 * __half2float(w0[1]);
        if (i + 1 < nnz_count) {
            const int f1 = nnz_ids[i + 1];
            const __half* w1 = value_tiled + (long long)tile_base + f1 * C_TILE + tid * 2;
            const float a1 = r2_slice[f1];
            acc0 += a1 * __half2float(w1[0]);
            acc1 += a1 * __half2float(w1[1]);
        }
    }
    atomicAdd(x + c0, acc0);
    atomicAdd(x + c0 + 1, acc1);
}
"#;

/// argmax CUDA kernel：在 logits [N] 中找最大值索引，写入 token[0]（f32 位模式存 uint）。
/// 语义对齐 Vulkan `argmax.comp`：单 block（256 线程）协作扫描，shared 树归约取全局 argmax，
/// 平局取更小索引（与 CPU 严格大于 argmax 一致）。dispatch (1,1,1)。
const ARGMAX_SRC: &str = r#"
extern "C" __global__ void rwkv_argmax(
    const float* __restrict__ logits,   // [N]
    float* __restrict__ token,          // [1] 写入 argmax 索引的 f32 位模式
    const int n)
{
    const int tid = threadIdx.x;
    constexpr int BS = 256;
    __shared__ float s_max[BS];
    __shared__ int   s_idx[BS];

    // 每线程沿 stride 扫描，取局部最大（严格大于，平局取小索引）。
    float lm = -1e30f;
    int   li = 0;
    for (int i = tid; i < n; i += BS) {
        const float v = logits[i];
        if (v > lm) { lm = v; li = i; }
    }
    s_max[tid] = lm;
    s_idx[tid] = li;
    __syncthreads();

    // shared 树归约，平局取更小索引。
    for (int step = BS >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            const float av = s_max[tid];
            const int   ai = s_idx[tid];
            const float bv = s_max[tid + step];
            const int   bi = s_idx[tid + step];
            if (bv > av || (bv == av && bi < ai)) {
                s_max[tid] = bv;
                s_idx[tid] = bi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        token[0] = __int_as_float(s_idx[0]);
    }
}
"#;

/// 统一 sample CUDA kernel：penalty(repetition/frequency/presence) + temperature + top-k + top-p
/// 过滤后按概率采样，写入 token[0]（f32 位模式存 uint）。语义对齐 Vulkan `sample.comp`。
/// 单 block（256 线程）协作，dispatch (1,1,1)。采样参数从 sampler 缓冲读取（f32 位模式存 uint）：
///   sampler[0]=temperature  sampler[1]=top_k(uint)  sampler[2]=top_p  sampler[3]=seed(uint)
///   sampler[4]=repetition_penalty  sampler[5]=frequency_penalty  sampler[6]=presence_penalty
///   sampler[7]=hist_len(uint)
/// 流程：载入 logits → 惩罚（counter 直方图统计历史 token）→ temperature → top-k（迭代求第 k 大
/// 阈值）→ softmax（max 归约→exp→归一化）→ top-p（累积截断）→ splitmix 采样定位 token。
const SAMPLE_SRC: &str = r#"
// ★ 2026-09-23：扫描趟展开倍数（host 注入覆盖；缺省 4 = 旧行为）。
#ifndef SAMP_U
#define SAMP_U 4
#endif
__device__ __forceinline__ float u01(unsigned int s) {
    s += 0x9E3779B9u;
    unsigned int z = s;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z ^= z >> 16;
    return (float)z / 4294967296.0f;
}

// ★ 2026-09-23：**扫描趟的统一展开原语**（`SAMP_U` 个元素/线程/轮，默认 4）。
//
// 病灶：整个单流采样器是 `grid=(1,1,1)`、`block=(112,)` —— 一个块 3.5 个 warp 独占一个
// SM，`for (i = tid; i < n; i += BS)` 是 **585 次/线程的串行访存**（无 ILP ⇒ 每轮只有
// 1 个载入在飞，~600 cycle 的延迟完全暴露）。一趟 ≈ 585×600 ≈ 0.22ms，而 TOPK=0 路径
// 有 7 趟全词表扫描 + 若干轮 top-p 候选扫描。
//
// 修法：把步长从 `bs` 拉到 `bs*U`，一轮内展开 U 个**独立**载入 ⇒ U 个在飞。
// **语义逐位不变**：同一元素仍由同一 lane 按**同一顺序**处理（lane `tid` 依次吃
// `tid, tid+bs, tid+2bs, ...`），只是把 U 轮并成 1 轮；U 个累加按 k 升序串行执行，
// 编译器不重结合浮点 ⇒ `s += v` 的求和顺序与旧版完全一致。
// 尾部另起标量循环收尾（不能把守卫写进展开体——实测那样会把 U 条载入又钉成顺序）。
//
// ⚠️ 展开倍数由 host 注入的 `SAMP_U` 决定（kernel key 里带该值，便于 A/B）。
template <int U, class F>
__device__ __forceinline__ void samp_scan(int tid, int bs, int n, F&& f)
{
    int i = tid;
    for (; i + (U - 1) * bs < n; i += bs * U) {
        _Pragma("unroll")
        for (int k = 0; k < U; k++) f(i + k * bs);
    }
    for (; i < n; i += bs) f(i);
}

// 从 temp[0..n)（mask==0 的项）中取最大的 BS 个候选，按 (值降序, 索引升序) 排序后写入
// out_val/out_idx（长度 BS）；无剩余候选的空槽值为 -1e30 且排到末尾。
// c_val/c_idx 为长度 BS 的暂存区，不得与 out_val/out_idx 别名。
// 全块协同：内部含 __syncthreads，必须由所有线程统一调用。
__device__ __forceinline__ void sample_top_candidates(
    const float* __restrict__ temp,
    const float* __restrict__ mask,
    int n,
    float* c_val, int* c_idx,
    float* out_val, int* out_idx)
{
    constexpr int BS = 112;
    const int tid = threadIdx.x;
    float lm = -1e30f; int li = 0;
    samp_scan<SAMP_U>(tid, BS, n, [&](int i) {
        const float v = temp[i];
        if (mask[i] == 0.0f && v > lm) { lm = v; li = i; }
    });
    c_val[tid] = lm;
    // 空槽给唯一哨兵索引：否则多个 (-1e30, 0) 撞秩，排序结果错乱。
    c_idx[tid] = (lm > -1e29f) ? li : (n + tid);
    __syncthreads();
    // O(BS^2) 并行定秩（每线程只扫 BS 个 shared 元素），避免为 112 元素引入位排序。
    const float mv = c_val[tid];
    const int   mi = c_idx[tid];
    int rank = 0;
    for (int j = 0; j < BS; j++) {
        const float vj = c_val[j];
        const int   ij = c_idx[j];
        if (vj > mv || (vj == mv && ij < mi)) rank++;
    }
    out_val[rank] = mv;
    out_idx[rank] = mi;
    __syncthreads();
}

// ★ 2026-09-22：扫描趟展开（原版固定 4×）。**2026-09-23 改为由 `samp_scan<SAMP_U>` 承载**
// （见其定义处的完整病灶说明），倍数可调（默认 4，`SAMP_U` 由 host 注入）。
//
// 病灶：`grid=(1,1,1)`、`block=(112,)` —— 整段 65536 词表只有**一个块 112 线程**，
// 而每个趟都是 `for (i = tid; i < n; i += BS)` ⇒ **585 次/线程的串行访存**
// （无 ILP，每轮 ~600 cycle 延迟）；这样的趟有 ~9 个（copy → temperature →
// mask 清零 → top-p 候选扫描 → softmax 的 max/exp+sum/normalize → cutoff 清零 →
// 采样扫描）⇒ 585×9×600 ≈ 3.2M cycle ≈ 2.1ms，**与实测 1.74ms/token 吻合**
// （占单流 decode 的 ~11%）。
//
// ⚠️ **不能靠加大 block 解决**：`s_val[tid][j]`/`s_idx[tid][j]` 按 `BS=112` 定义，
// `tid >= 112` 会越界（sticky error 700，host 侧有注释）。
//
// ⚠️ **第一版把守卫写在展开体内**（`if (_i + k*BS < n) {...}`）⇒ **实测零收益**
// （`rwkv_sample` 69.64 → 69.22 ms/32tok），原因是每轮的分支把 4 条载入又钉成了
// 顺序执行。`samp_scan` 用**「无守卫主循环 + 尾部标量循环」**，让编译器能自由调度。
// ⚠️ 替换列表自带结尾 `;`（旧宏是 `{...}` 块语句，调用点不写分号；新宏是函数调用表达式）。
#define SAMP_SCAN4(BODY) samp_scan<SAMP_U>(tid, BS, n, [&](int i) { BODY });

extern "C" __global__ void rwkv_sample(
    const float*      __restrict__ logits,   // [n]
    float*            __restrict__ token,    // [1] 写入索引的 f32 位模式
    float*            __restrict__ temp,     // [n] 工作区
    float*            __restrict__ mask,     // [n] 工作区
    unsigned int*     __restrict__ counter,  // [n] 直方图
    const float*      __restrict__ sampler,  // [10] 参数（[8]=penalty_decay，[9] 保留）
    const unsigned int* __restrict__ hist,   // [hist_len] 历史 token
    const int n)
{
    const int tid = threadIdx.x;
    constexpr int BS = 112;
    constexpr int MAXK = 50;   // 快速路径支持的最大 top_k（覆盖常见 50）
    // 单遍 top-K 快速路径共享缓冲：每线程一个局部有序 top-K。
    __shared__ float s_val[BS][MAXK];
    __shared__ int   s_idx[BS][MAXK];
    // 全局 top-K 结果（s_topval/s_topidx 为最小堆，s_sorted/s_sortedidx 为降序结果）。
    __shared__ float s_topval[MAXK];
    __shared__ int   s_topidx[MAXK];
    __shared__ float s_sorted[MAXK];
    __shared__ int   s_sortedidx[MAXK];
    // 兜底路径（top_k 未设或 > MAXK）用的小块归约缓冲。
    __shared__ float s_fval[BS];
    __shared__ int   s_fidx[BS];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_u;
    __shared__ float s_threshold;
    __shared__ float g_cutoff;
    __shared__ float s_cum;
    __shared__ int   s_consumed;
    __shared__ int   s_used;
    __shared__ int   s_done;

    const float temperature = sampler[0];
    const unsigned int top_k = __float_as_uint(sampler[1]);
    const float top_p = sampler[2];
    const unsigned int seed = __float_as_uint(sampler[3]);
    const float rep = sampler[4];
    const float freq = sampler[5];
    const float pres = sampler[6];
    const unsigned int hist_len = __float_as_uint(sampler[7]);
    const float decay = sampler[8];   // 惩罚衰减指数（1.0 = 退化为 freq*cnt）
    const bool do_topk = (top_k > 0u && top_k < (unsigned int)n);
    const int K = do_topk ? (int)top_k : 0;

    // 1. 载入 logits
    SAMP_SCAN4({ temp[i] = logits[i]; })

    // 2. 惩罚
    if (hist_len > 0u && (rep != 1.0f || freq != 0.0f || pres != 0.0f)) {
        for (int i = tid; i < n; i += BS) counter[i] = 0u;
        __syncthreads();
        for (int h = tid; h < (int)hist_len; h += BS) {
            atomicAdd(&counter[hist[h]], 1u);
        }
        __syncthreads();
        for (int i = tid; i < n; i += BS) {
            const unsigned int cnt = counter[i];
            float l = temp[i];
            if (cnt > 0u) {
                if (rep != 1.0f) l = l > 0.0f ? l / rep : l * rep;
                if (pres != 0.0f) l -= pres;
                if (freq != 0.0f) l -= freq * powf((float)cnt, decay);
            }
            temp[i] = l;
        }
        __syncthreads();
    }

    // 3. temperature
    float invT = 1.0f / temperature;
    if (!(temperature > 0.0f)) invT = 1.0f;
    SAMP_SCAN4({ temp[i] *= invT; })
    __syncthreads();

    if (K > 0 && K <= MAXK) {
        // ================= 快速路径：单遍 top-K =================
        // 4. 每线程维护局部有序 top-K（降序），一次扫描完成。
        // ★ 2026-09-23：`SAMP_U` 个载入先批量发出，再按 **i 升序**逐个插入
        // （插入顺序与旧版逐元素循环逐位一致，只把访存并起来 ⇒ 打掉串行延迟）。
        for (int j = 0; j < MAXK; j++) { s_val[tid][j] = -1e30f; s_idx[tid][j] = -1; }
        {
            int i = tid;
            for (; i + (SAMP_U - 1) * BS < n; i += BS * SAMP_U) {
                float vv[SAMP_U];
                _Pragma("unroll")
                for (int k = 0; k < SAMP_U; k++) vv[k] = temp[i + k * BS];
                _Pragma("unroll")
                for (int k = 0; k < SAMP_U; k++) {
                    const float v = vv[k];
                    if (v > s_val[tid][K - 1]) {
                        int pos = K - 1;
                        while (pos > 0 && v > s_val[tid][pos - 1]) {
                            s_val[tid][pos] = s_val[tid][pos - 1];
                            s_idx[tid][pos] = s_idx[tid][pos - 1];
                            --pos;
                        }
                        s_val[tid][pos] = v;
                        s_idx[tid][pos] = i + k * BS;
                    }
                }
            }
            for (; i < n; i += BS) {
                const float v = temp[i];
                if (v > s_val[tid][K - 1]) {
                    int pos = K - 1;
                    while (pos > 0 && v > s_val[tid][pos - 1]) {
                        s_val[tid][pos] = s_val[tid][pos - 1];
                        s_idx[tid][pos] = s_idx[tid][pos - 1];
                        --pos;
                    }
                    s_val[tid][pos] = v;
                    s_idx[tid][pos] = i;
                }
            }
        }
        __syncthreads();

        // 5. tid==0 用最小堆合并 BS*K 个候选 → 全局 top-K，再降序提取。
        if (tid == 0) {
            auto sift = [&](int i, int h) {
                while (true) {
                    int l = 2 * i + 1, r = 2 * i + 2, m = i;
                    if (l < h && s_topval[l] < s_topval[m]) m = l;
                    if (r < h && s_topval[r] < s_topval[m]) m = r;
                    if (m == i) break;
                    float tv = s_topval[i]; s_topval[i] = s_topval[m]; s_topval[m] = tv;
                    int ti = s_topidx[i]; s_topidx[i] = s_topidx[m]; s_topidx[m] = ti;
                    i = m;
                }
            };
            // 用第 0 行初始化最小堆
            for (int j = 0; j < K; j++) { s_topval[j] = s_val[0][j]; s_topidx[j] = s_idx[0][j]; }
            for (int j = K / 2 - 1; j >= 0; j--) sift(j, K);
            // 插入其余行候选
            for (int th = 1; th < BS; th++) {
                for (int j = 0; j < K; j++) {
                    const float v = s_val[th][j];
                    if (v <= -1e29f) break; // 空槽
                    if (v > s_topval[0]) {
                        s_topval[0] = v; s_topidx[0] = s_idx[th][j];
                        sift(0, K);
                    }
                }
            }
            // 降序提取到 s_sorted
            for (int r = K; r > 0; r--) {
                s_sorted[r - 1] = s_topval[0];
                s_sortedidx[r - 1] = s_topidx[0];
                s_topval[0] = s_topval[r - 1];
                s_topidx[0] = s_topidx[r - 1];
                sift(0, r - 1);
            }
            s_threshold = s_sorted[K - 1]; // 第 K 大（降序末位）= 保留边界
            // 历史注：曾误取 s_sorted[0]（最大值），top-K 名义保留 50 实际只留
            // top-1 + 并列——温度较高时采样分布被错误坍缩到单点。
        }
        __syncthreads();

        // 6. 低于阈值置 -inf（保留 top-K 及阈值并列项）
        for (int i = tid; i < n; i += BS) if (temp[i] < s_threshold) temp[i] = -1e30f;
        __syncthreads();
    } else {
        // ================= 兜底路径：top_k 未设或 > MAXK，保持原逻辑 =================
        if (do_topk) {
            SAMP_SCAN4({ mask[i] = 0.0f; })
            __syncthreads();
            if (tid == 0) s_threshold = -1e30f;
            __syncthreads();
            for (unsigned int round = 0u; round < top_k; round++) {
                float lm = -1e30f; int li = 0;
                for (int i = tid; i < n; i += BS) {
                    if (mask[i] == 0.0f && temp[i] > lm) { lm = temp[i]; li = i; }
                }
                s_fval[tid] = lm; s_fidx[tid] = li;
                __syncthreads();
                // BS=112 非 2 的幂：step 序列 56,28,14,7,3,1 会在 14→7→3→1 段孤儿化
                // 索引 6 与 2 的结果（s_fval[0] 只是约 1/16 元素的最大值 → 阈值偏低）。
                // 修法同 softmax：先把尾部 [P2, BS) 折进 [0, BS-P2)，再 2 幂树归约。
                {
                    constexpr int P2 = 64;
                    if (tid >= P2 && tid < BS) {
                        const float bv = s_fval[tid];
                        const int   bi = s_fidx[tid];
                        const float av = s_fval[tid - P2];
                        const int   ai = s_fidx[tid - P2];
                        if (bv > av || (bv == av && bi < ai)) {
                            s_fval[tid - P2] = bv; s_fidx[tid - P2] = bi;
                        }
                    }
                    __syncthreads();
                    for (int step = P2 >> 1; step > 0; step >>= 1) {
                        if (tid < step) {
                            const float bv = s_fval[tid + step];
                            const int   bi = s_fidx[tid + step];
                            if (bv > s_fval[tid] || (bv == s_fval[tid] && bi < s_fidx[tid])) {
                                s_fval[tid] = bv; s_fidx[tid] = bi;
                            }
                        }
                        __syncthreads();
                    }
                }
                if (tid == 0) { s_threshold = s_fval[0]; mask[s_fidx[0]] = 1.0f; }
                __syncthreads();
            }
            for (int i = tid; i < n; i += BS) if (temp[i] < s_threshold) temp[i] = -1e30f;
            __syncthreads();
        }
    }

    // 7. softmax：max -> exp -> normalize
    // 归约为非幂 block 安全版（BS=112 非 2 的幂，纯树归约会孤儿化部分 warp 的
    // 结果——同 batch 版注释，max 漏读/sum 漏加 → softmax 全错）。
    {
        float lm = -1e30f;
        SAMP_SCAN4({ lm = fmaxf(lm, temp[i]); })
        s_fval[tid] = lm;
        __syncthreads();
        {
            constexpr int P2 = 64;  // BS=112 → 64 + 48
            if (tid >= P2 && tid < BS) s_fval[tid - P2] = fmaxf(s_fval[tid - P2], s_fval[tid]);
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] = fmaxf(s_fval[tid], s_fval[tid + step]);
                __syncthreads();
            }
        }
        const float m = s_fval[0];
        __syncthreads();
        float s = 0.0f;
        SAMP_SCAN4({
            const float v = expf(temp[i] - m);
            temp[i] = v;
            s += v;
        })
        s_fval[tid] = s;
        __syncthreads();
        {
            constexpr int P2 = 64;
            if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] += s_fval[tid + step];
                __syncthreads();
            }
        }
        const float total = s_fval[0];
        __syncthreads();
        if (total > 0.0f) {
            SAMP_SCAN4({ temp[i] /= total; })
        }
        __syncthreads();
    }

    // 8. top-p：从最大概率起累积达 top_p 后截断（在全局 top-K 降序列表上单遍完成）
    if (top_p > 0.0f && top_p < 1.0f) {
        if (K > 0 && K <= MAXK) {
            if (tid == 0) {
                // s_sortedidx 为降序（[0] 最大），必须从 [0] 起累积——旧版从 K-1
                // （最小）起累积，与其自身注释「从最大概率起累积」相悖：截断阈值
                // 偏低 → 保留集合偏大（多峰分布下与兜底路径结果不一致）。
                float cum = 0.0f, cutoffv = -1e30f;
                for (int j = 0; j < K; j++) {
                    const int idx = s_sortedidx[j];
                    cum += temp[idx];
                    cutoffv = temp[idx];
                    if (cum >= top_p) break;
                }
                g_cutoff = cutoffv;
            }
        } else {
            // 兜底（无 top-k 或 top_k > MAXK）：每轮取 BS 个候选，而非旧版每轮 1 个。
            // 旧版在平坦分布下要跑满 512 轮 × 全词表扫描（实测 51.8 ms/token，单流
            // selfloop 从 85 掉到 20 tok/s）。512 上限现在只需 5 轮。
            // 暂存复用快速路径的 s_val/s_idx（此分支下二者未被使用）。
            float* c_val = (float*)s_val;
            int*   c_idx = (int*)s_idx;
            float* o_val = (float*)s_val + BS;
            int*   o_idx = (int*)s_idx + BS;
            SAMP_SCAN4({ mask[i] = 0.0f; })
            __syncthreads();
            if (tid == 0) { g_cutoff = 0.0f; s_cum = 0.0f; s_consumed = 0; s_done = 0; }
            __syncthreads();
            // 退出条件必须全块统一（shared）：否则 tid0 先越 top_p 退出循环、
            // 其余线程继续跑，__syncthreads 分歧 = 块内死锁（kernel 永久自旋 99% SM）
            while (!s_done) {
                sample_top_candidates(temp, mask, n, c_val, c_idx, o_val, o_idx);
                if (tid == 0) {
                    int used = 0;
                    for (int j = 0; j < BS && s_consumed < 512; j++) {
                        const float v = o_val[j];
                        if (v <= -1e29f) break;  // 候选耗尽
                        s_cum += v;
                        g_cutoff = v;
                        ++used; ++s_consumed;
                        if (s_cum >= top_p) break;
                    }
                    s_used = used;
                    if (used == 0 || s_cum >= top_p || s_consumed >= 512) s_done = 1;
                }
                __syncthreads();
                // 标记本轮已消费候选（即将退出时多标也无害：mask 之后不再使用）
                if (tid < s_used) mask[o_idx[tid]] = 1.0f;
                __syncthreads();
            }
            __syncthreads();
        }
        SAMP_SCAN4({ if (temp[i] < g_cutoff) temp[i] = 0.0f; })
        __syncthreads();
    }

    // 9. 采样：在 top-K（降序）上做前缀和定位（兜底路径用全量扫描）。
    if (K > 0 && K <= MAXK) {
        if (tid == 0) {
            float total = 0.0f;
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp[idx] > 0.0f) total += temp[idx];
            }
            const float u = u01(seed) * total;
            float acc = 0.0f;
            int chosen = s_sortedidx[K - 1];
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp[idx] > 0.0f) {
                    acc += temp[idx];
                    if (acc > u) { chosen = idx; break; }
                }
            }
            token[0] = __int_as_float(chosen);
        }
    } else {
        float ts = 0.0f;
        SAMP_SCAN4({ ts += temp[i]; })
        s_fval[tid] = ts;
        __syncthreads();
        {
            constexpr int P2 = 64;
            if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] += s_fval[tid + step];
                __syncthreads();
            }
        }
        const float total = s_fval[0];
        __syncthreads();
        if (tid == 0) s_u = u01(seed) * total;
        __syncthreads();
        // ★ 2026-09-23：**并行定位采样索引**。
        //
        // 旧版是 `if (tid == 0) { for (i = 0; i < n; i++) { acc += temp[i]; if (acc > s_u) break; } }`
        // —— **单线程**按 i 升序串行读全表，而 `temp` 经过 top-p 掩码后非零项只有
        // 几百个、散布在 65536 个位置上 ⇒ 命中位置期望在词表**中部**，要空跑约 n/2 次
        // 依赖 L2 延迟的标量载入。实测 **1.28 ms/token，占采样器 83%、整步 10%**
        // （`SAMP_AB=1` 消融：1.534 → 0.257 ms）。
        //
        // 新版两段式（全块协同）：
        //   ① 每线程负责**连续**一段 [tid·CH, (tid+1)·CH)，段内 4 路 ILP 求和 → s_blk[]；
        //   ② tid0 在块和上定位跨块 → s_cross/s_base；
        //   ③ 全块对**该块**再做同样切分（粒度 CH2），tid0 只扫 ≤CH2 个元素。
        // 单线程串行步数从 ~n/2 降到 `BS + BS + CH2`（n=65536 时 112+112+6 ≈ 230 步）。
        //
        // ⚠️ 求和顺序与旧版不同（旧版是严格 i 升序的单累加器），相对误差 ~1e-6。
        // 只有在 `|s_u − 前缀和| < 1e-6·total` 的刀锋处才可能改变所选 token
        // （概率 ~1e-6/步，可用 token 指纹 sum/xor 验证）。
        {
            constexpr int BS_ = 112;
            __shared__ float s_blk[BS_];
            __shared__ int   s_cross;
            __shared__ float s_base;
            const int CH = (n + BS_ - 1) / BS_;
            {
                const int b0 = tid * CH;
                const int b1 = min(b0 + CH, n);
                float cs = 0.0f;
                int i = b0;
                for (; i + 3 < b1; i += 4)
                    cs += (temp[i] + temp[i + 1]) + (temp[i + 2] + temp[i + 3]);
                for (; i < b1; i++) cs += temp[i];
                s_blk[tid] = cs;
            }
            __syncthreads();
            if (tid == 0) {
                float acc = 0.0f;
                int cross = BS_ - 1;
                for (int t = 0; t < BS_; t++) {
                    const float nxt = acc + s_blk[t];
                    if (nxt > s_u) { cross = t; break; }
                    acc = nxt;
                }
                s_cross = cross;
                s_base = acc;
            }
            __syncthreads();
            const int cb0 = s_cross * CH;
            const int cb1 = min(cb0 + CH, n);
            const int CH2 = (CH + BS_ - 1) / BS_;
            {
                const int q0 = cb0 + tid * CH2;
                const int q1 = min(q0 + CH2, cb1);
                float cs = 0.0f;
                for (int i = q0; i < q1; i++) cs += temp[i];
                s_blk[tid] = cs;
            }
            __syncthreads();
            if (tid == 0) {
                float acc = s_base;
                int chosen = n - 1;
                for (int t = 0; t < BS_; t++) {
                    const float nxt = acc + s_blk[t];
                    if (nxt > s_u) {
                        const int r0 = cb0 + t * CH2;
                        const int r1 = min(r0 + CH2, cb1);
                        for (int i = r0; i < r1; i++) {
                            acc += temp[i];
                            if (acc > s_u) { chosen = i; break; }
                        }
                        break;
                    }
                    acc = nxt;
                }
                token[0] = __int_as_float(chosen);
            }
        }
    }
}
"#;

/// record_token CUDA kernel：把 in_tok[0]（f32 位模式存 uint 的 token 索引）追加到
/// out_seq[atomicAdd(cnt)]，随后 cnt 自增。语义对齐 Vulkan `record_token.comp`。
/// 单线程（dispatch (1,1,1)），供 GPU self-loop 记录每轮生成的 token。
const RECORD_TOKEN_SRC: &str = r#"
extern "C" __global__ void rwkv_record_token(
    const unsigned int* __restrict__ in_tok,  // [1] token（f32 位模式）
    unsigned int* __restrict__ out_seq,       // [n] 序列缓冲
    unsigned int* __restrict__ cnt)           // [1] 计数器（原子自增）
{
    const unsigned int i = atomicAdd(&cnt[0], 1u);
    out_seq[i] = in_tok[0];
}
"#;

/// gather_row_device_f16 CUDA kernel：从 fp16 表 src[VOCAB, C] 按 token 索引读一行，
/// 转 fp32 写入 dst[C]。索引来自 tok[0]（f32 位模式存 uint）。
/// 语义对齐 Vulkan `gather_row_f16.comp`：dispatch (ceil(C/256), 1, 1)。
const GATHER_ROW_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_gather_row_f16(
    const unsigned int* __restrict__ in_tok,  // [1] token 索引（f32 位模式）
    const __half*  __restrict__ in_src,       // [VOCAB, C] fp16
    float* __restrict__ out_dst,              // [C] fp32
    const int c)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idx = in_tok[0];
    if (index < c) {
        out_dst[index] = __half2float(in_src[(size_t)idx * (size_t)c + (size_t)index]);
    }
}
"#;

/// copy_device_f16 CUDA kernel：f16 设备到设备全量拷贝（v_first 快照用）。
/// 语义对齐 Vulkan `copy_token.comp` 的设备侧拷贝分支；len 为元素数，一维拷贝。
const COPY_DEVICE_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_device_f16(
    const __half* __restrict__ src,  // [len]
    __half* __restrict__ dst,        // [len]
    const int len)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) dst[i] = src[i];
}
"#;

/// copy_device（f32）CUDA kernel：f32 设备到设备全量拷贝（v_first 快照 / 状态缓冲用）。
const COPY_DEVICE_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_device(
    const float* __restrict__ src,  // [len]
    float* __restrict__ dst,        // [len]
    const int len)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) dst[i] = src[i];
}
"#;

/// copy_range：定长区间设备内拷贝（dst[dst_off..] = src[src_off..]）。
/// batch State 单行迁移用（src/dst 通常同一张量的不同 slot 段，区间不重叠）。
const COPY_RANGE_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_range(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int src_off,
    const int dst_off,
    const int len)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) dst[dst_off + i] = src[src_off + i];
}
"#;

/// copy_range_f16：f16 定长区间设备内拷贝（v_first 等 f16 状态缓冲的单行迁移）。
const COPY_RANGE_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_range_f16(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    const int src_off,
    const int dst_off,
    const int len)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) dst[dst_off + i] = src[src_off + i];
}
"#;

/// copy_token CUDA kernel：y[i] = x[token*stride + i]（sequence-parallel 状态更新用）。
/// 语义对齐 Vulkan `copy_token.comp`。
const COPY_TOKEN_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_token(
    const float* __restrict__ x,   // [T, C]
    float* __restrict__ y,         // [C]
    const int c,
    const int stride,              // token 行步长
    const int token)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < c) y[i] = x[(size_t)token * (size_t)stride + (size_t)i];
}
"#;

/// elementwise_sigmoid CUDA kernel：y = sigmoid(a) = 1/(1+exp(-a))。
/// 语义对齐 Vulkan `elementwise_f32_f32_sigmoid`（OP=1）；grid.y = batch。
const ELEMENTWISE_SIGMOID_SRC: &str = r#"
extern "C" __global__ void rwkv_elementwise_sigmoid(
    const float* __restrict__ a,
    float* __restrict__ y,
    const int c,
    const int batch)
{
    const int b = blockIdx.y;
    const int base = b * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        y[base + i] = 1.0f / (1.0f + __expf(-a[base + i]));
    }
}
"#;

/// elementwise_scale_exp CUDA kernel：y = exp(a * b[0])（b 为共享 f32 标量）。
/// 语义对齐 Vulkan `elementwise_f32_f32_scale_exp`（OP=9）；grid.y = batch。
const ELEMENTWISE_SCALE_EXP_SRC: &str = r#"
extern "C" __global__ void rwkv_elementwise_scale_exp(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    const int c,
    const int batch)
{
    const float sc = b[0]; // 全局共享标量（与 Vulkan elementwise.comp OP9 一致）
    const int b0 = blockIdx.y;
    const int base = b0 * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        y[base + i] = __expf(a[base + i] * sc);
    }
}
"#;

/// elementwise_mul CUDA kernel：y = a * b（逐元素，grid.y = batch）。
/// 语义对齐 Vulkan `elementwise_f32_f32_mul`（OP=5）。
const ELEMENTWISE_MUL_SRC: &str = r#"
extern "C" __global__ void rwkv_elementwise_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    const int c,
    const int batch)
{
    const int b0 = blockIdx.y;
    const int base = b0 * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        y[base + i] = a[base + i] * b[base + i];
    }
}
"#;

/// to_f16 CUDA kernel：f32 → f16（token 并行，sequence-parallel）。
/// 语义对齐 Vulkan `to_f16.comp`：非对齐 token（token>=T）写 0，供 GEMM 填充行输出为 0。
const TO_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_to_f16(
    const float* __restrict__ x,   // [T, C] f32
    __half* __restrict__ y,        // [M_PAD, C] f16
    const int c,
    const int t,
    const int x_stride,
    const int y_stride)
{
    const int token = blockIdx.x;
    const size_t yb = (size_t)token * (size_t)y_stride;
    if (token >= t) {
        for (int i = threadIdx.x; i < c; i += blockDim.x) y[yb + i] = __float2half(0.0f);
        return;
    }
    const size_t xb = (size_t)token * (size_t)x_stride;
    for (int i = threadIdx.x; i < c; i += blockDim.x) y[yb + i] = __float2half(x[xb + i]);
}
"#;

/// to_f16_triple CUDA kernel：一次把 xr/xk/xv 三个 [T,C] f32 转成 [M_PAD,C] f16。
/// 语义对齐 Vulkan `to_f16_triple.comp`。
const TO_F16_TRIPLE_SRC: &str = r#"
extern "C" __global__ void rwkv_to_f16_triple(
    const float* __restrict__ xr,
    const float* __restrict__ xk,
    const float* __restrict__ xv,
    __half* __restrict__ yr,
    __half* __restrict__ yk,
    __half* __restrict__ yv,
    const int c,
    const int t,
    const int x_stride,
    const int y_stride)
{
    const int token = blockIdx.x;
    const size_t yb = (size_t)token * (size_t)y_stride;
    if (token >= t) {
        for (int i = threadIdx.x; i < c; i += blockDim.x) {
            yr[yb + i] = __float2half(0.0f);
            yk[yb + i] = __float2half(0.0f);
            yv[yb + i] = __float2half(0.0f);
        }
        return;
    }
    const size_t xb = (size_t)token * (size_t)x_stride;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        yr[yb + i] = __float2half(xr[xb + i]);
        yk[yb + i] = __float2half(xk[xb + i]);
        yv[yb + i] = __float2half(xv[xb + i]);
    }
}
"#;

/// dequant_int8_to_f16 CUDA kernel：int8 [M,K] 反量化为 fp16 [M,K]。
/// 语义对齐 Vulkan `dequant_int8_f16.comp`：idx uint32[M,K/4]（4 uint8/uint32）、sz uint32[M,K/128]。
const DEQUANT_INT8_SRC: &str = r#"
__device__ __forceinline__ void unpack_quant_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
extern "C" __global__ void rwkv_dequant_int8(
    const unsigned int* __restrict__ idx,  // [M, K/4]
    const unsigned int* __restrict__ sz,   // [M, K/128]
    __half* __restrict__ w,                // [M, K] f16 输出
    const int m,
    const int k)
{
    const int kv = k / 4;
    const int kg = k / 128;
    const long total = (long)m * kv;
    const long linear = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int mm = (int)(linear / kv);
    const int kk = (int)(linear % kv);
    const int g = kk / 32;
    float sc, zr;
    unpack_quant_sz(sz[(size_t)mm * kg + g], sc, zr);
    const unsigned int ipack = idx[linear];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int byte = (int)((ipack >> (8 * j)) & 0xFFu);
        const float wv = sc * (float)byte + zr;
        w[(size_t)mm * k + kk * 4 + j] = __float2half(wv);
    }
}
"#;

/// fuse_ka CUDA kernel：k_mod = k*(1+k_a*(a-1))、kk_l2 = normalize(k*k_k)、b = -kk_l2*a。
/// 语义对齐 Vulkan `fuse_ka.comp`（EPSILON=1e-12）；grid = (H, batch)、block 256。
const FUSE_KA_SRC: &str = r#"
extern "C" __global__ void rwkv_fuse_ka(
    const float* __restrict__ k,    // [batch, H*N]
    const float* __restrict__ kk_w, // [H*N] 共享
    const float* __restrict__ a,    // [batch, H*N]
    const float* __restrict__ ka_w, // [H*N] 共享
    float* __restrict__ km,         // [batch, H*N]
    float* __restrict__ kl,         // [batch, H*N]
    float* __restrict__ b,          // [batch, H*N]
    const int h,
    const int n,
    const int batch)
{
    const int head = blockIdx.x;
    const int bidx = blockIdx.y;
    const int tid  = threadIdx.x;
    const int base   = bidx * (h * n) + head * n;
    const int wbase  = head * n;
    float local_sq = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        const float kkv = k[base + j] * kk_w[wbase + j];
        local_sq += kkv * kkv;
    }
    __shared__ float sdata[256];
    sdata[tid] = local_sq;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float inv_norm = 1.0f / fmaxf(sqrtf(sdata[0]), 1e-12f);
    for (int j = tid; j < n; j += blockDim.x) {
        const int addr  = base + j;
        const int waddr = wbase + j;
        const float kv_  = k[addr];
        const float kkv  = kv_ * kk_w[waddr];
        const float k_l2 = kkv * inv_norm;
        const float av   = a[addr];
        km[addr] = kv_ * (1.0f + ka_w[waddr] * (av - 1.0f));
        kl[addr] = k_l2;
        b[addr]  = -k_l2 * av;
    }
}
"#;

/// sum_rk_rk CUDA kernel：s = Σ_j r[j]*k_mod[j]*r_k[j]（head 归约），y[j] += s*v[j]。
/// 语义对齐 Vulkan `sum_rk_rk.comp`；grid = (H, batch)、block 256。
const SUM_RK_RK_SRC: &str = r#"
extern "C" __global__ void rwkv_sum_rk_rk(
    const float* __restrict__ r,    // [batch, H*N]
    const float* __restrict__ km,   // k_mod [batch, H*N]
    const float* __restrict__ rk,   // [H*N] 共享
    const float* __restrict__ v,    // [batch, H*N]
    float* __restrict__ y,          // [batch, H*N] 累加
    const int h,
    const int n,
    const int batch)
{
    const int head = blockIdx.x;
    const int bidx = blockIdx.y;
    const int tid  = threadIdx.x;
    const int base = bidx * (h * n) + head * n;
    float local = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        local += r[base + j] * km[base + j] * rk[head * n + j];
    }
    __shared__ float sdata[256];
    sdata[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float s = sdata[0];
    for (int j = tid; j < n; j += blockDim.x) {
        y[base + j] += s * v[base + j];
    }
}
"#;

/// seq_shift CUDA kernel：result[t] = x[t] + tm*(prev - x[t])，prev 为 x[t-1] 或 token-shift state（t=0）。
/// 语义对齐 Vulkan `seq_shift.comp`；grid = (T, 1)、block 256。
const SEQ_SHIFT_SRC: &str = r#"
extern "C" __global__ void rwkv_seq_shift(
    const float* __restrict__ x,   // [T, C]
    const float* __restrict__ s,   // token-shift state [C]
    const float* __restrict__ tm,  // [C]
    float* __restrict__ y,         // [T, C]
    const int c,
    const int t,
    const int stride_x,
    const int stride_y)
{
    const int token = blockIdx.x;
    if (token >= t) return;
    const size_t xbase = (size_t)token * stride_x;
    const size_t ybase = (size_t)token * stride_y;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float cur  = x[xbase + i];
        const float prev = (token == 0) ? s[i] : x[xbase - stride_x + i];
        const float tmv  = tm[i];
        y[ybase + i] = cur + tmv * (prev - cur);
    }
}
"#;

/// v_first_lerp CUDA kernel：v[t] = v[t] + gate[t]*(v_first[t] - v[t])。
/// 语义对齐 Vulkan `v_first_lerp.comp`；grid = (T, 1)、block 256。
const V_FIRST_LERP_SRC: &str = r#"
extern "C" __global__ void rwkv_v_first_lerp(
    float* __restrict__ v,          // [T, C] in/out
    const float* __restrict__ g,    // [T, C] gate
    const float* __restrict__ vf,   // [T, C] v_first
    const int c,
    const int t,
    const int stride)
{
    const int token = blockIdx.x;
    if (token >= t) return;
    const size_t base = (size_t)token * stride;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float vv = v[base + i];
        const float gv = g[base + i];
        const float fv = vf[base + i];
        v[base + i] = vv + gv * (fv - vv);
    }
}
"#;

/// seq_shift_batch CUDA kernel：batch prefill 的 token shift（slot 边界 t=0 读该 slot 的 state）。
/// x/y 为 [batch, T, C]（slot 主序），s 为 [batch, C]。dispatch (T, batch, 1)。
const SEQ_SHIFT_BATCH_SRC: &str = r#"
extern "C" __global__ void rwkv_seq_shift_batch(
    const float* __restrict__ x,   // [batch, T, C]
    const float* __restrict__ s,   // token-shift state [batch, C]
    const float* __restrict__ tm,  // [C]（共享）
    float* __restrict__ y,         // [batch, T, C]
    const int c,
    const int t,
    const int stride_x,
    const int stride_y)
{
    const int token = blockIdx.x;
    const int b      = blockIdx.y;
    if (token >= t) return;
    const size_t base = ((size_t)b * t + token) * stride_x;
    const size_t ybase = ((size_t)b * t + token) * stride_y;
    const float* sprev = (token == 0) ? (s + (size_t)b * c) : (x + base - stride_x);
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float cur = x[base + i];
        const float prev = sprev[i];
        const float tmv = tm[i];
        y[ybase + i] = cur + tmv * (prev - cur);
    }
}
"#;

/// copy_token_batch CUDA kernel：每 slot 把 x 的第 lens[b]-1 行拷到 state[b]。
/// x 为 [batch, T, C]，state 为 [batch, C]。dispatch (batch, 1, 1)，block 256。
const COPY_TOKEN_BATCH_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_token_batch(
    const float* __restrict__ x,      // [batch, T, C]
    float* __restrict__ state,        // [batch, C]
    const int* __restrict__ lens,     // [batch]（实际 prompt 长度，>=1）
    const int c,
    const int t)
{
    const int b = blockIdx.x;
    const int last = lens[b] - 1;
    const float* src = x + ((size_t)b * t + last) * c;
    float* dst = state + (size_t)b * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        dst[i] = src[i];
    }
}
"#;

/// segmean CUDA kernel：分段均值——x [batch, T_pad, C] 每 slot 对前 lens[b] 行求均值
/// → out [batch, C]（pad 行不计入）。批量 mean-hidden 特征提取用（语义对齐单序列
/// forward_seq_mean_hidden 的 T 维均值，仅累加顺序不同）。
/// dispatch (batch, 1, 1)，block 256（跨 C stride，循环 token 累加）。
const SEGMEAN_SRC: &str = r#"
extern "C" __global__ void rwkv_segmean(
    const float* __restrict__ x,     // [batch, t_pad, c]
    float* __restrict__ out,         // [batch, c]
    const int* __restrict__ lens,    // [batch]（实际 token 数，>=1）
    const int c,
    const int t_pad)
{
    const int b = blockIdx.x;
    const int len = lens[b];
    const float* xb = x + (size_t)b * t_pad * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < len; t++) {
            acc += xb[(size_t)t * c + i];
        }
        out[(size_t)b * c + i] = acc / (float)len;
    }
}
"#;

/// dplr_seq_batch CUDA kernel：batch prefill 的 DPLR 状态更新。
/// s 为 [batch, H, N*N]（batch State 布局），r/w/k/v/a/b/y 为 [batch, T, C]，
/// lens[b] 截断实际长度（padding 段不进 state）。dispatch (ceil(H*N/8), batch, 1)。
const DPLR_SEQ_BATCH_SRC: &str = r#"
// Phase 2-2：状态 `s` 的 dtype 由 `WKV_S16` 选择（与 FUSE_KA_DPRL_NORM_SRC 同规矩，
// 三处状态内核必须同 dtype）。计算全程 fp32，只在载入/回存时降位。
#ifdef WKV_S16
typedef __half stype;
__device__ __forceinline__ float st_ld(const stype* p) { return __half2float(*p); }
__device__ __forceinline__ void  st_st(stype* p, float v) { *p = __float2half_rn(v); }
#else
typedef float stype;
__device__ __forceinline__ float st_ld(const stype* p) { return *p; }
__device__ __forceinline__ void  st_st(stype* p, float v) { *p = v; }
#endif
__device__ __forceinline__ float dplr_b_halfwarp_sum_all_xor(float v) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask, 16);
    }
    return v;
}
extern "C" __global__ void rwkv_dplr_seq_batch(
    stype* __restrict__ s,          // [batch, H, N*N] 状态（in/out）
    const float* __restrict__ r,    // [batch, T, C]
    const float* __restrict__ w,    // [batch, T, C]
    const float* __restrict__ k,    // [batch, T, C]
    const float* __restrict__ v,    // [batch, T, C]
    const float* __restrict__ a,    // [batch, T, C]
    const float* __restrict__ b,    // [batch, T, C]
    float* __restrict__ y,          // [batch, T, C] 输出
    const int* __restrict__ lens,   // [batch]
    const int h,
    const int n,
    const int t,
    const int c)
{
    const int bslot = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int half = lane >> 4;
    const int subl = lane & 15;
    const int row  = (int)(blockIdx.x * 8 + warp * 2 + half);
    if (row >= h * n) return;
    const int head = row / n;
    const int i    = row % n;
    const int j0 = subl, j1 = subl + 16, j2 = subl + 32, j3 = subl + 48;
    // slot 内偏移：state 段 + token 基址。
    const size_t s_base = (size_t)bslot * h * n * n + (size_t)head * n * n + (size_t)i * n;
    const size_t slot_tok = (size_t)bslot * t;   // 该 slot 在 [batch,T,C] 中的 token 基址
    const size_t tok_off = (size_t)head * n;      // 该 head 在 [C] 内的列偏移
    float s0 = st_ld(&s[s_base + j0]);
    float s1 = st_ld(&s[s_base + j1]);
    float s2 = st_ld(&s[s_base + j2]);
    float s3 = st_ld(&s[s_base + j3]);
    const int len = lens[bslot];
    for (int tt = 0; tt < len; tt++) {
        const size_t e = (slot_tok + tt) * c + tok_off;
        const float vv = v[e + i];
        const float a0 = a[e + j0], a1 = a[e + j1], a2 = a[e + j2], a3 = a[e + j3];
        float sa = s0 * a0 + s1 * a1 + s2 * a2 + s3 * a3;
        sa = dplr_b_halfwarp_sum_all_xor(sa);
        const float w0 = w[e + j0], w1 = w[e + j1], w2 = w[e + j2], w3 = w[e + j3];
        const float k0 = k[e + j0], k1 = k[e + j1], k2 = k[e + j2], k3 = k[e + j3];
        const float b0 = b[e + j0], b1 = b[e + j1], b2 = b[e + j2], b3 = b[e + j3];
        const float r0 = r[e + j0], r1 = r[e + j1], r2 = r[e + j2], r3 = r[e + j3];
        s0 = s0 * w0 + k0 * vv + sa * b0;
        s1 = s1 * w1 + k1 * vv + sa * b1;
        s2 = s2 * w2 + k2 * vv + sa * b2;
        s3 = s3 * w3 + k3 * vv + sa * b3;
        float yv = s0 * r0 + s1 * r1 + s2 * r2 + s3 * r3;
        yv = dplr_b_halfwarp_sum_all_xor(yv);
        if (subl == 0) y[e + i] = yv;
    }
    s[s_base + j0] = s0;
    s[s_base + j1] = s1;
    s[s_base + j2] = s2;
    s[s_base + j3] = s3;
}
"#;

/// dplr_seq CUDA kernel：sequence-parallel DPLR 状态更新（内部循环 T）。
/// 语义对齐 Vulkan `dplr_seq.comp`：每个 block 一个 head，64 线程（==N），S 行存寄存器跨 token 传递。
/// 要求 n <= 64（RWKV-7 恒 N=64）。
const DPLR_SEQ_SRC: &str = r#"
// Phase 2-2：状态 dtype 见 `WKV_S16`（三处状态内核必须同 dtype）。
#ifdef WKV_S16
typedef __half stype;
__device__ __forceinline__ float st_ld(const stype* p) { return __half2float(*p); }
__device__ __forceinline__ void  st_st(stype* p, float v) { *p = __float2half_rn(v); }
#else
typedef float stype;
__device__ __forceinline__ float st_ld(const stype* p) { return *p; }
__device__ __forceinline__ void  st_st(stype* p, float v) { *p = v; }
#endif
// 每个状态行（head, i）用一个 half-warp（16 线程，每线程管 4 个 j 列）并行处理，
// 整块 GPU 并行处理全部 h*n 个状态行（旧版仅 h*n 线程、仅 h 个 block，占用率极低）。
// 行间独立：sa[i] = sum_j a[j]*s[i][j] 由 half-warp 归约，随后逐列更新状态。
__device__ __forceinline__ float dplr_halfwarp_sum_all_xor(float v) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask, 16);
    }
    return v;
}
extern "C" __global__ void rwkv_dplr_seq(
    stype* __restrict__ s,          // [H, N*N] 状态（in/out）
    const float* __restrict__ r,    // [T, C]
    const float* __restrict__ w,    // [T, C]
    const float* __restrict__ k,    // [T, C]
    const float* __restrict__ v,    // [T, C]
    const float* __restrict__ a,    // [T, C]
    const float* __restrict__ b,    // [T, C]
    float* __restrict__ y,          // [T, C] 输出
    const int h,
    const int n,
    const int t,
    const int c)
{
    // 128 线程/block = 4 warp = 8 half-warp，每 half-warp 管一个状态行。
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;      // 0..3
    const int lane = tid & 31;
    const int half = lane >> 4;     // 0/1
    const int subl = lane & 15;     // 0..15
    const int row  = (int)(blockIdx.x * 8 + warp * 2 + half); // 全局状态行 index
    if (row >= h * n) return;
    const int head = row / n;
    const int i    = row % n;
    // 每线程管 4 列：j0, j0+16, j0+32, j0+48（n=64）。
    const int j0 = subl, j1 = subl + 16, j2 = subl + 32, j3 = subl + 48;
    const size_t s_base = (size_t)head * n * n + (size_t)i * n; // 该行状态首地址
    const size_t tok_off = (size_t)head * n;                    // 该 head 在 [C] 内的列偏移
    float s0 = st_ld(&s[s_base + j0]);
    float s1 = st_ld(&s[s_base + j1]);
    float s2 = st_ld(&s[s_base + j2]);
    float s3 = st_ld(&s[s_base + j3]);
    for (int tt = 0; tt < t; tt++) {
        const size_t e = tok_off + (size_t)tt * c;
        const float vv = v[e + i];
        // 本线程持有的 4 列 a/w/k/b/r
        const float a0 = a[e + j0], a1 = a[e + j1], a2 = a[e + j2], a3 = a[e + j3];
        // sa = sum_j a[j]*s[j]，先算本线程 4 列 partial 再 half-warp 归约
        float sa = s0 * a0 + s1 * a1 + s2 * a2 + s3 * a3;
        sa = dplr_halfwarp_sum_all_xor(sa);
        const float w0 = w[e + j0], w1 = w[e + j1], w2 = w[e + j2], w3 = w[e + j3];
        const float k0 = k[e + j0], k1 = k[e + j1], k2 = k[e + j2], k3 = k[e + j3];
        const float b0 = b[e + j0], b1 = b[e + j1], b2 = b[e + j2], b3 = b[e + j3];
        const float r0 = r[e + j0], r1 = r[e + j1], r2 = r[e + j2], r3 = r[e + j3];
        s0 = s0 * w0 + k0 * vv + sa * b0;
        s1 = s1 * w1 + k1 * vv + sa * b1;
        s2 = s2 * w2 + k2 * vv + sa * b2;
        s3 = s3 * w3 + k3 * vv + sa * b3;
        float yv = s0 * r0 + s1 * r1 + s2 * r2 + s3 * r3;
        yv = dplr_halfwarp_sum_all_xor(yv);
        if (subl == 0) y[e + i] = yv;
    }
    st_st(&s[s_base + j0], s0);
    st_st(&s[s_base + j1], s1);
    st_st(&s[s_base + j2], s2);
    st_st(&s[s_base + j3], s3);
}
"#;

/// gemm 统一 CUDA kernel：C[M,N] = A[M,K] @ B[N,K]^T（A/B fp16，C fp32）。
/// op: 0=plain, 1=+bias[n], 2=+x[M,N], 3=relu2, 4=tanh。每线程计算一个 C 元素。
/// 语义对齐 Vulkan `gemm*.comp`（tensor-core 版的结果）。
const GEMM_SRC: &str = r#"
__device__ __forceinline__ float gemm_dplr_relu2f(float x) { return x > 0.f ? x * x : 0.f; }
extern "C" __global__ void rwkv_gemm(
    const __half* __restrict__ a,    // [M, K] f16
    const __half* __restrict__ b,    // [N, K] f16
    const float* __restrict__ bias,  // [N]（op==1 用）
    const float* __restrict__ x,     // [M, N]（op==2 用）
    float* __restrict__ c,           // [M, N] f32
    const int m,
    const int n,
    const int k,
    const int op)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    const __half* arow = a + (size_t)row * k;
    const __half* bcol = b + (size_t)col * k;
    float acc = 0.0f;
    for (int kk = 0; kk < k; kk++) {
        acc += __half2float(arow[kk]) * __half2float(bcol[kk]);
    }
    float v;
    if (op == 1) v = acc + bias[col];
    else if (op == 2) v = acc + x[(size_t)row * n + col];
    else if (op == 3) v = gemm_dplr_relu2f(acc);
    else if (op == 4) v = tanhf(acc);
    else v = acc;
    c[(size_t)row * n + col] = v;
}
"#;

/// cuBLAS GEMM 的 epilogue：对 C[m,n]（f32）就地补充 op==1 bias / op==2 加 x / op==3 relu2 / op==4 tanh。
const GEMM_EPILOGUE_SRC: &str = r#"
extern "C" __global__ void rwkv_gemm_epilogue(
    float* __restrict__ c,          // [m, n]
    const float* __restrict__ bias, // [n]（op==1 用）
    const float* __restrict__ x,    // [m, n]（op==2 用）
    const int m,
    const int n,
    const int op)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * n;
    if (linear >= total) return;
    const int col = linear % n;
    float v = c[linear];
    if (op == 1) v += bias[col];
    else if (op == 2) v += x[linear];
    else if (op == 3) v = v > 0.f ? v * v : 0.f;
    else if (op == 4) v = tanhf(v);
    c[linear] = v;
}
"#;

/// gemv_seq CUDA kernel：y[b, m] = Σ_k x[b*x_stride + k] * A[m*k + k]（A f32 权重，跨步批量）。
/// 语义对齐 Vulkan `gemv_f32_f32`（gemv_seq_impl）；grid = (m, batch)、每 block 计算一行。
const GEMV_SEQ_SRC: &str = r#"
extern "C" __global__ void rwkv_gemv_seq(
    const float* __restrict__ a,      // [m, k] f32
    const float* __restrict__ x,      // [batch, x_stride]
    float* __restrict__ y,            // [batch, y_stride]
    const int m,
    const int k,
    const int x_stride,
    const int y_stride,
    const int batch)
{
    const int row = blockIdx.x;
    const int bidx = blockIdx.y;
    if (row >= m || bidx >= batch) return;
    const size_t xb = (size_t)bidx * x_stride;
    const float* arow = a + (size_t)row * k;
    float acc = 0.0f;
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
        acc += arow[kk] * x[xb + kk];
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)bidx * y_stride + row] = sdata[0];
}
"#;

impl ComputeBackend for CudaBackend {
    fn create_tensor(&mut self, len: usize, dtype: TensorDtype) -> R<TensorId> {
        self.next_id += 1;
        let id = TensorId(self.next_id);
        let bytes = match dtype {
            TensorDtype::F32 | TensorDtype::U32 => len * 4,
            TensorDtype::F16 => len * 2,
        };
        let dptr = self.alloc(bytes)?;
        let t = match dtype {
            TensorDtype::F32 => CudaTensor::F32 { dptr, len },
            TensorDtype::F16 => CudaTensor::F16 { dptr, len },
            TensorDtype::U32 => CudaTensor::U32 { dptr, len },
        };
        self.tensors.insert(id, t);
        self.lens.insert(id, len);
        Ok(id)
    }

    fn upload(&self, t: TensorId, data: &[f32]) -> R<()> {
        match self.get(t, "upload")? {
            CudaTensor::F32 { dptr, len } => {
                // 对齐 Vulkan：允许部分上传（data.len() <= len），只写前置元素，余下不动。
                // 模型会用 padded 缓冲（如 seq x 为 m_pad*C）只上传实际 t*C，其余不读。
                if data.len() > len {
                    return Err(format!("upload: len mismatch ({} > {len})", data.len()).into());
                }
                let bytes = bytemuck::cast_slice::<f32, u8>(data);
                self.memcpy_htod_n(dptr, bytes, data.len())?;
                Ok(())
            }
            CudaTensor::F16 { dptr, len } => {
                if data.len() > len {
                    return Err(
                        format!("upload(f16): len mismatch ({} > {len})", data.len()).into(),
                    );
                }
                let f16s: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                let bytes = bytemuck::cast_slice::<f16, u8>(&f16s);
                self.memcpy_htod_n2(dptr, bytes, f16s.len())?;
                Ok(())
            }
            CudaTensor::U32 { .. } => Err("upload: u32 tensor requires upload_u32".into()),
        }
    }

    fn upload_part(&self, t: TensorId, offset: usize, data: &[f32]) -> R<()> {
        // 部分上传：只写 [offset, offset+len) 段（元素偏移），其余不动。
        // F32 直拷；F16 先转半精度再按 2 字节元素偏移上传（v_first 用）。
        let (dptr, bytes, off_bytes, src_buf): (u64, usize, usize, Vec<u8>) =
            match self.get(t, "upload_part")? {
                CudaTensor::F32 { dptr, len } => {
                    if offset + data.len() > len {
                        return Err(format!(
                            "upload_part: range {}..{} exceeds len {len}",
                            offset,
                            offset + data.len()
                        )
                        .into());
                    }
                    (
                        dptr,
                        data.len() * 4,
                        offset * 4,
                        bytemuck::cast_slice::<f32, u8>(data).to_vec(),
                    )
                }
                CudaTensor::F16 { dptr, len } => {
                    if offset + data.len() > len {
                        return Err(format!(
                            "upload_part(f16): range {}..{} exceeds len {len}",
                            offset,
                            offset + data.len()
                        )
                        .into());
                    }
                    let f16s: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                    (
                        dptr,
                        f16s.len() * 2,
                        offset * 2,
                        bytemuck::cast_slice::<f16, u8>(&f16s).to_vec(),
                    )
                }
                _ => return Err("upload_part: tensor must be f32 or f16".into()),
            };
        // 源数据走 pinned scratch（同 htod_pinned 语义），拷贝到张量偏移段。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(upload_part)"
        );
        let scratch_off = PINNED_ROWS * PINNED_ROW_BYTES;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_buf.as_ptr(),
                (self.pinned as *mut u8).add(scratch_off),
                bytes,
            );
        }
        let src = unsafe { (self.pinned as *const u8).add(scratch_off) };
        cu_check!(
            (self.drv.cu_memcpy_htod_async)(
                dptr + off_bytes as u64,
                src as *const c_void,
                bytes,
                self.stream
            ),
            "cuMemcpyHtoDAsync(upload_part)"
        );
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(upload_part-wait)"
        );
        Ok(())
    }

    fn upload_u32_part(&self, t: TensorId, offset: usize, data: &[u32]) -> R<()> {
        // 部分上传（u32）：只写 [offset, offset+len) 段（元素偏移），其余不动。
        // 惩罚历史逐 token 追加用——整行重传是 O(hist_len)，追加是 O(1)。
        let (dptr, len) = match self.get(t, "upload_u32_part")? {
            CudaTensor::U32 { dptr, len } => (dptr, len),
            _ => return Err("upload_u32_part: tensor must be u32".into()),
        };
        if offset + data.len() > len {
            return Err(format!(
                "upload_u32_part: range {}..{} exceeds len {len}",
                offset,
                offset + data.len()
            )
            .into());
        }
        let bytes = data.len() * 4;
        if bytes == 0 {
            return Ok(());
        }
        let src_buf = bytemuck::cast_slice::<u32, u8>(data);
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(upload_u32_part)"
        );
        let scratch_off = PINNED_ROWS * PINNED_ROW_BYTES;
        if bytes > PINNED_UPLOAD_SCRATCH {
            return Err(format!("upload_u32_part: 段 {bytes} 字节超过 pinned scratch").into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_buf.as_ptr(),
                (self.pinned as *mut u8).add(scratch_off),
                bytes,
            );
        }
        let src = unsafe { (self.pinned as *const u8).add(scratch_off) };
        cu_check!(
            (self.drv.cu_memcpy_htod_async)(
                dptr + (offset * 4) as u64,
                src as *const c_void,
                bytes,
                self.stream
            ),
            "cuMemcpyHtoDAsync(upload_u32_part)"
        );
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(upload_u32_part-wait)"
        );
        Ok(())
    }

    fn upload_u32(&self, t: TensorId, data: &[u32]) -> R<()> {
        match self.get(t, "upload_u32")? {
            CudaTensor::U32 { dptr, len } => {
                // 对齐 Vulkan：允许部分上传（data.len() <= len）。
                if data.len() > len {
                    return Err(format!("upload_u32: len mismatch ({} > {len})", data.len()).into());
                }
                let bytes = bytemuck::cast_slice::<u32, u8>(data);
                self.memcpy_htod_n(dptr, bytes, data.len())?;
                Ok(())
            }
            _ => Err("upload_u32: t must be u32".into()),
        }
    }

    /// 加载大量 tensor 时消除逐 tensor 全流同步（原 19s 加载的主要开销）。
    fn upload_bulk_begin(&mut self) {
        const RING_SLOTS: usize = 4;
        const RING_SLOT_BYTES: usize = 128 * 1024 * 1024;
        if self.bulk.borrow().is_some() {
            return;
        }
        unsafe {
            let mut base: *mut c_void = std::ptr::null_mut();
            let r = (self.drv.cu_mem_host_alloc)(&mut base, RING_SLOTS * RING_SLOT_BYTES, 0);
            if r != 0 {
                log::warn!("bulk ring pinned 分配失败（cuResult {r}），回退逐 tensor 同步上传");
                return;
            }
            let mut events = Vec::with_capacity(RING_SLOTS);
            for _ in 0..RING_SLOTS {
                let mut ev: CuEvent = std::ptr::null_mut();
                let er = (self.drv.cu_event_create)(&mut ev, 0);
                if er != 0 {
                    log::warn!("cuEventCreate 失败（{er}），回退逐 tensor 同步上传");
                    let _ = (self.drv.cu_mem_free_host)(base);
                    return;
                }
                events.push(ev);
            }
            self.bulk.borrow_mut().replace(BulkRing {
                base: base as *mut u8,
                slot_bytes: RING_SLOT_BYTES,
                slots: RING_SLOTS,
                next: 0,
                events,
            });
            log::info!(
                "bulk upload mode: {RING_SLOTS} × {} MB pinned ring",
                RING_SLOT_BYTES / 1024 / 1024
            );
        }
    }

    /// 批量上传结束：排空全部 DMA、释放环形缓冲。
    fn upload_bulk_end(&mut self) {
        let ring = self.bulk.borrow_mut().take();
        if let Some(r) = ring {
            unsafe {
                let _ = (self.drv.cu_stream_synchronize)(self.stream);
            }
            for ev in &r.events {
                unsafe {
                    let _ = (self.drv.cu_event_destroy)(*ev);
                }
            }
            unsafe {
                let _ = (self.drv.cu_mem_free_host)(r.base as *mut c_void);
            }
            log::info!("bulk upload mode ended");
        }
    }

    fn download(&self, t: TensorId) -> R<Vec<f32>> {
        match self.get(t, "download")? {
            CudaTensor::F32 { dptr, len } => {
                let mut out = vec![0u8; len * 4];
                self.memcpy_dtoh(dptr, &mut out)?;
                Ok(bytemuck::cast_slice::<u8, f32>(&out).to_vec())
            }
            CudaTensor::F16 { dptr, len } => {
                let mut bytes = vec![0u8; len * 2];
                self.memcpy_dtoh(dptr, &mut bytes)?;
                let f16s: &[f16] = bytemuck::cast_slice(&bytes);
                Ok(f16s.iter().map(|&v| v.to_f32()).collect())
            }
            CudaTensor::U32 { .. } => Err("download: u32 tensor unsupported here".into()),
        }
    }

    fn download_part(&self, t: TensorId, offset: usize, len: usize) -> R<Vec<f32>> {
        // 只拷回 [offset, offset+len) 段：batch State 取单 slot 一行时避免整表
        // 下载（batch=16 的 tmix_rnn 全表 ~260MB，单行仅 ~15MB）。
        if len == 0 {
            return Ok(Vec::new());
        }
        match self.get(t, "download_part")? {
            CudaTensor::F32 { dptr, len: total } => {
                if offset + len > total {
                    return Err(format!("download_part: {offset}+{len} 超过 {total}").into());
                }
                let mut out = vec![0u8; len * 4];
                self.memcpy_dtoh(dptr + (offset * 4) as u64, &mut out)?;
                Ok(bytemuck::cast_slice::<u8, f32>(&out).to_vec())
            }
            CudaTensor::F16 { dptr, len: total } => {
                if offset + len > total {
                    return Err(format!("download_part(f16): {offset}+{len} 超过 {total}").into());
                }
                let mut bytes = vec![0u8; len * 2];
                self.memcpy_dtoh(dptr + (offset * 2) as u64, &mut bytes)?;
                let f16s: &[f16] = bytemuck::cast_slice(&bytes);
                Ok(f16s.iter().map(|&v| v.to_f32()).collect())
            }
            CudaTensor::U32 { .. } => Err("download_part: u32 tensor unsupported here".into()),
        }
    }

    fn download_u32(&self, t: TensorId) -> R<Vec<u32>> {
        let len = *self
            .lens
            .get(&t)
            .ok_or("download_u32: unknown tensor len")?;
        match self.get(t, "download_u32")? {
            CudaTensor::U32 { dptr, .. } => {
                let mut bytes = vec![0u8; len * 4];
                self.memcpy_dtoh(dptr, &mut bytes)?;
                Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
            }
            _ => Err("download_u32: t must be u32".into()),
        }
    }

    fn begin_batch(&mut self) -> R<()> {
        // batch 线性层 fp16 激活暂存：在**进入捕获之前**备好（捕获期内 cuMemAlloc
        // 非法）。begin_batch 恒先于 ensure_selfloop_graph/捕获调用，故此处预置一块
        // 足够大的常驻缓冲即可让后续所有 batch*k 形状复用同一指针（图安全）。
        if !self.graph_capturing && self.x16_pool.is_empty() {
            let cap = X16_SCRATCH_INIT_ELEMS;
            let t = TensorDtype::F16;
            let id = <Self as ComputeBackend>::create_tensor(self, cap, t)?;
            self.x16_pool.push((cap, id));
        }
        // W8A8 激活量化暂存（`quant_x_i8`/`imma_gemm_batch`）：同样必须在捕获前备好，
        // 否则第一次走 IMMA 路径就撞上「图内禁止分配」。容量按 batch=256 × k=10240 留余量。
        if !self.graph_capturing && self.xq_pool.is_empty() {
            let cap = XQ_SCRATCH_INIT_U32;
            let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::U32)?;
            self.xq_pool.push((cap, id));
        }
        if !self.graph_capturing && self.xaux_pool.is_empty() {
            let cap = XAUX_SCRATCH_INIT_F4;
            let id = <Self as ComputeBackend>::create_tensor(self, cap * 4, TensorDtype::F32)?;
            self.xaux_pool.push((cap, id));
        }
        // split-K 部分和暂存（`IM_KSPLIT > 1`）：同上，捕获前备好。
        if !self.graph_capturing && self.ipart_pool.is_empty() {
            let cap = IPART_SCRATCH_INIT_ELEMS;
            let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::F32)?;
            self.ipart_pool.push((cap, id));
        }
        // 低秩链 GEMM 暂存（`LOWRANK_GEMM=1`）：同样必须在捕获前备好。
        if !self.graph_capturing && self.lr16_pool.is_empty() {
            let cap = LR16_SCRATCH_INIT_ELEMS;
            let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::F16)?;
            self.lr16_pool.push((cap, id));
        }
        // ffn_value 稠密 GEMM 的 r2_16 暂存（`FFN_VALUE_GEMM=1`）。
        if !self.graph_capturing && self.ffn16_pool.is_empty() {
            let cap = FFN16_SCRATCH_INIT_ELEMS;
            let id = <Self as ComputeBackend>::create_tensor(self, cap, TensorDtype::F16)?;
            self.ffn16_pool.push((cap, id));
        }
        // 剖析：record start 事件到 stream（测本批纯 GPU 执行时间）。
        // 捕获期间不允许 cuEventRecord（stream 处于 capture mode），跳过。
        if self.prof_gpu && !self.graph_capturing {
            cu_check!(
                (self.drv.cu_event_record)(self.prof_ev_start, self.stream),
                "cuEventRecord(start)"
            );
        }
        Ok(())
    }

    fn clear_kernel_prof(&mut self) {
        self.drv.clear_prof();
    }

    fn dump_kernel_prof(&mut self) {
        self.drv.dump_prof(self.stream);
    }

    fn end_batch(&mut self) -> R<()> {
        if self.prof_kernel && !self.graph_capturing {
            self.drv.dump_prof(self.stream);
        }
        if self.gemm_prof && !self.graph_capturing && !self.gemm_times.is_empty() {
            let mut total = 0.0f64;
            let mut rows: Vec<_> = self.gemm_times.iter().collect();
            rows.sort_by(|a, b| {
                b.1.1
                    .partial_cmp(&a.1.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for ((m, n, k, op), (cnt, ms)) in &rows {
                total += ms;
                log::info!(
                    "[PROF_GEMM] m={m:>5} n={n:>5} k={k:>5} op={op} cnt={cnt:>3} total={ms:>8.2}ms avg={:>7.3}ms",
                    ms / *cnt as f64
                );
            }
            log::info!("[PROF_GEMM] SUM {total:.2}ms");
            self.gemm_times.clear();
        }
        if self.prof_gpu && !self.graph_capturing {
            cu_check!(
                (self.drv.cu_event_record)(self.prof_ev_end, self.stream),
                "cuEventRecord(end)"
            );
            unsafe {
                (self.drv.cu_event_synchronize)(self.prof_ev_end);
            }
            let mut ms: f32 = 0.0;
            unsafe {
                (self.drv.cu_event_elapsed_time)(&mut ms, self.prof_ev_start, self.prof_ev_end);
            }
            log::info!("[CUDA_GPU] batch: {ms:.3} ms");
        }
        Ok(())
    }

    fn supports_graph_capture(&self) -> bool {
        // 兼容开关：RWKV_GRAPH=0 时禁用 CUDA graph（降级为批量记录）。
        // 背景：驱动 610.47 上图捕获+重放在个别请求上非确定性挂死（2026-09-19，
        // town-model-server 实测）；修复前可用此开关绕过。结果缓存避免每调用读 env。
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            let enabled = std::env::var("RWKV_GRAPH")
                .map(|v| v != "0")
                .unwrap_or(true);
            if !enabled {
                log::info!("RWKV_GRAPH=0：禁用 CUDA graph capture（降级批量记录路径）");
            }
            enabled
        })
    }

    /// 登记在册张量的设备字节总数（f16 2B / f32 4B / u32 4B）。
    /// 用途：比较「模型加载 / 建 state / prefill / 解码」各阶段的增量，定位显存大户。
    fn device_bytes(&self) -> usize {
        self.tensors
            .iter()
            .filter(|(id, _)| !self.foreign.contains(id))
            .map(|(_, t)| match t {
                CudaTensor::F16 { len, .. } => 2 * len,
                CudaTensor::F32 { len, .. } | CudaTensor::U32 { len, .. } => 4 * len,
            })
            .sum()
    }

    fn prefill_graph_valid(&mut self, t: usize) -> R<bool> {
        // RWKV_GRAPH=0 兼容开关同时禁用 prefill 图捕获（见 supports_graph_capture）。
        if !self.supports_graph_capture() {
            return Ok(false);
        }
        Ok(self.prefill_graphs.contains_key(&t))
    }

    fn begin_prefill_capture(&mut self, t: usize) -> R<()> {
        // 确保 stream 空闲（无排队 kernel），再开始捕获。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(before prefill capture)"
        );
        cu_check!(
            (self.drv.cu_graph_begin_capture)(self.stream, CU_STREAM_CAPTURE_MODE),
            "cuStreamBeginCapture(prefill)"
        );
        self.prefill_t = t;
        self.graph_capturing = true;
        Ok(())
    }

    fn end_prefill_capture(&mut self) -> R<()> {
        let mut graph: CuGraph = std::ptr::null_mut();
        cu_check!(
            (self.drv.cu_graph_end_capture)(self.stream, &mut graph),
            "cuGraphEndCapture(prefill)"
        );
        self.graph_capturing = false;
        let mut exec: CuGraphExec = std::ptr::null_mut();
        cu_check!(
            (self.drv.cu_graph_instantiate)(&mut exec, graph, 0),
            "cuGraphInstantiate(prefill)"
        );
        cu_check!(
            (self.drv.cu_graph_destroy)(graph),
            "cuGraphDestroy(prefill)"
        );
        // 绑定到当前 T 的 prefill graph；若同 T 已存在则销毁旧的。
        if let Some(old) = self.prefill_graphs.insert(self.prefill_t, exec) {
            cu_check!(
                (self.drv.cu_graph_exec_destroy)(old),
                "cuGraphExecDestroy(old prefill)"
            );
        }
        Ok(())
    }

    fn prefill_graph_replay(&mut self) -> R<()> {
        let exec = self
            .prefill_graphs
            .get(&self.prefill_t)
            .copied()
            .ok_or("prefill_graph_replay: no captured graph for this T")?;
        cu_check!(
            (self.drv.cu_graph_launch)(exec, self.stream),
            "cuGraphLaunch(prefill)"
        );
        Ok(())
    }

    // —— 解码 self-loop 图（按形状 key 长期持有）——
    //
    // 与无 key 的 `begin/end_graph_capture` 单槽 API 的区别：这里每个形状各留一张
    // instanced graph，捕获一次后永久重放。原先每段（32 token）都重新
    // capture→instantiate→destroy，持续压测累计数百次 churn，实测在
    // `cuGraphInstantiate` 驱动代码内部触发 0xC0000005（nvcuda64.dll+0x297ca0）。
    // ★ 2026-09-22：那个崩溃的**根因已定位** = 符号解析取到了 v1（五参数）而按三参数
    // 调用（见 `cu_graph_instantiate` 处的注释）。按 key 长期持有仍是好设计。

    fn selfloop_graph_ready(&mut self, key: u64) -> bool {
        self.selfloop_graphs.contains_key(&key)
    }

    fn selfloop_graph_skip(&mut self, key: u64) -> bool {
        // 诊断开关：`NO_SELFLOOP_GRAPH=1` 强制走非 graph 逐轮提交 —— 只有这条路才能用
        // cuEvent 给每个内核计时（捕获期内核只被记录、不执行，计时拿不到数）。
        std::env::var("NO_SELFLOOP_GRAPH").is_ok() || self.selfloop_disabled.contains(&key)
    }

    fn begin_selfloop_capture(&mut self, key: u64) -> R<()> {
        // 先确保 stream 空闲（无排队 kernel），再开始捕获。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(before selfloop capture)"
        );
        cu_check!(
            (self.drv.cu_graph_begin_capture)(self.stream, CU_STREAM_CAPTURE_MODE),
            "cuStreamBeginCapture(selfloop)"
        );
        self.selfloop_key = key;
        self.graph_capturing = true;
        // 计数清零：dump 出来的就是**这张图里**的内核清单（不含之前的 prefill/warmup）。
        self.drv.prof.lock().unwrap().counts.clear();
        Ok(())
    }

    fn end_selfloop_capture(&mut self) -> R<()> {
        let key = self.selfloop_key;
        let mut graph: CuGraph = std::ptr::null_mut();
        let end_res = unsafe { (self.drv.cu_graph_end_capture)(self.stream, &mut graph) };
        self.graph_capturing = false;
        if end_res != CUDA_SUCCESS || graph.is_null() {
            // 捕获本身就失败：stream 已退出捕获态（end capture 成功），标记 key 禁用。
            self.selfloop_disabled.insert(key);
            self.selfloop_key = 0;
            log::warn!(
                "self-loop 图捕获失败（key=0x{key:x}，cuStreamEndCapture={end_res}）：\
                 该形状永久降级为非 graph 逐轮提交"
            );
            return Err(format!("cuStreamEndCapture(selfloop) failed: {end_res}").into());
        }
        let mut exec: CuGraphExec = std::ptr::null_mut();
        let inst_res = unsafe { (self.drv.cu_graph_instantiate)(&mut exec, graph, 0) };
        // 源 graph 用完即销毁（失败路径也要销毁，避免泄漏）。
        unsafe {
            (self.drv.cu_graph_destroy)(graph);
        }
        if inst_res != CUDA_SUCCESS || exec.is_null() {
            self.selfloop_disabled.insert(key);
            self.selfloop_key = 0;
            log::warn!(
                "self-loop 图实例化失败（key=0x{key:x}，cuGraphInstantiate={inst_res}）：\
                 该形状永久降级为非 graph 逐轮提交"
            );
            return Err(format!("cuGraphInstantiate(selfloop) failed: {inst_res}").into());
        }
        if let Some(old) = self.selfloop_graphs.insert(key, exec) {
            // 同 key 重捕获（本不应发生）：销毁旧的，避免泄漏。
            cu_check!(
                (self.drv.cu_graph_exec_destroy)(old),
                "cuGraphExecDestroy(selfloop old)"
            );
            log::warn!("self-loop 图同 key 重复捕获（key=0x{key:x}）：已替换旧图");
        }
        self.selfloop_key = 0;
        log::info!("self-loop 图已捕获并实例化（key=0x{key:x}）：该形状后续只重放，不再重捕获");
        // 捕获这一趟把图里所有 launch 都走了一遍 `launch_smem` ⇒ 计数就是「每段的内核次数」。
        self.drv.dump_counts();
        Ok(())
    }

    fn abort_selfloop_capture(&mut self) {
        // 捕获区间内某步出错：必须 end capture 让 stream 退出捕获态，否则后续所有
        // CUDA 调用都会失败（甚至永久损坏上下文）。丢弃半成品 graph。
        if !self.graph_capturing {
            return;
        }
        let mut graph: CuGraph = std::ptr::null_mut();
        let res = unsafe { (self.drv.cu_graph_end_capture)(self.stream, &mut graph) };
        self.graph_capturing = false;
        if res == CUDA_SUCCESS && !graph.is_null() {
            unsafe {
                (self.drv.cu_graph_destroy)(graph);
            }
        }
        let key = self.selfloop_key;
        self.selfloop_disabled.insert(key);
        self.selfloop_key = 0;
        unsafe {
            (self.drv.cu_stream_synchronize)(self.stream);
        }
        log::warn!("self-loop 捕获中断（key=0x{key:x}）：清理并永久降级该形状");
    }

    fn selfloop_graph_replay(&mut self, key: u64) -> R<()> {
        let exec = self
            .selfloop_graphs
            .get(&key)
            .copied()
            .ok_or("selfloop_graph_replay: no captured graph for this key")?;
        cu_check!(
            (self.drv.cu_graph_launch)(exec, self.stream),
            "cuGraphLaunch(selfloop)"
        );
        Ok(())
    }

    fn clear_selfloop_graphs(&mut self, kind: Option<u64>) {
        // key 高 2 bit 即 kind；`None` 匹配全部。
        let matches = |k: u64| kind.is_none_or(|want| (k >> 62) == want);
        let doomed: Vec<u64> = self
            .selfloop_graphs
            .keys()
            .copied()
            .filter(|k| matches(*k))
            .collect();
        for k in doomed {
            if let Some(exec) = self.selfloop_graphs.remove(&k) {
                unsafe {
                    (self.drv.cu_graph_exec_destroy)(exec);
                }
            }
        }
        // 禁用标记单独过滤：被禁用的形状本就**没有**图，不会被上面的循环覆盖。
        if kind.is_none() {
            self.selfloop_disabled.clear();
        } else {
            self.selfloop_disabled.retain(|k| !matches(*k));
        }
    }

    fn store_token_host(&self, tok: TensorId, token: u32) -> R<()> {
        // token 张量为 F32（f32 位模式存 uint 索引），直接上传单个元素。
        self.upload(tok, &[f32::from_bits(token)])
    }

    fn store_sampler_async(
        &self,
        sampler: TensorId,
        row: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
    ) -> R<()> {
        if row >= self.pinned_rows {
            return Err(format!(
                "store_sampler_async: row {row} >= pinned rows {}",
                self.pinned_rows
            )
            .into());
        }
        let dptr = match self.get(sampler, "store_sampler_async")? {
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("store_sampler_async: sampler must be f32".into()),
        };
        let data = [
            temperature,
            f32::from_bits(top_k),
            top_p,
            f32::from_bits(seed),
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            f32::from_bits(hist_len),
            1.0, // penalty_decay（单流自循环不用衰减；index 8）
            0.0, // 保留（index 9，对齐 40 字节行）
        ];
        // 写 pinned 行 + 流序异步拷贝（零 host 同步；拷贝在 stream 中排在
        // 此前已提交的 kernel 之后，供下一轮 graph replay 读取）。
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                (self.pinned as *mut u8).add(row * PINNED_ROW_BYTES),
                PINNED_ROW_BYTES,
            );
        }
        let src = unsafe { (self.pinned as *const u8).add(row * PINNED_ROW_BYTES) };
        cu_check!(
            (self.drv.cu_memcpy_htod_async)(
                dptr,
                src as *const c_void,
                PINNED_ROW_BYTES,
                self.stream
            ),
            "cuMemcpyHtoDAsync(sampler)"
        );
        Ok(())
    }

    fn sampler_async_rows(&self) -> usize {
        self.pinned_rows
    }

    fn import_tensors_from(&mut self, src: &dyn ComputeBackend) -> R<()> {
        let src = src
            .as_any()
            .downcast_ref::<CudaBackend>()
            .ok_or("import_tensors_from: src is not a CudaBackend")?;
        if src.device != self.device {
            return Err("import_tensors_from: device mismatch".into());
        }
        // 全量复制张量表（同 TensorId → 同设备指针；同一 primary ctx 下有效）。
        for (id, ct) in &src.tensors {
            if self.tensors.insert(*id, ct.clone()).is_some() {
                return Err(format!("import_tensors_from: tensor id {id:?} collision").into());
            }
            let len = *src
                .lens
                .get(id)
                .ok_or("import_tensors_from: src lens missing")?;
            self.lens.insert(*id, len);
            self.foreign.insert(*id);
        }
        // 后续新张量 id 接续源后端计数（新实例自建工作缓冲/状态不与共享权重冲突）。
        self.next_id = self.next_id.max(src.next_id);
        // kernel 缓存同样共享：CuModule/CuFunction 句柄在同一 primary ctx 下跨
        // 实例有效（省每实例 ~30s 的 nvrtc 编译）。
        for (k, v) in &src.kernels {
            self.kernels.insert(k.clone(), *v);
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // ==== batch 并发算子（单实例多序列：B slot 共享权重，一次读权重算 B 份）====

    fn gather_rows_device_f16(
        &mut self,
        s: TensorId,
        d: TensorId,
        t: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        // in_src 为 fp16 表 [VOCAB, C]；out_dst 为 f32 [batch, C]；in_tok 为 [batch]（f32 位模式存 uint）。
        let src_d = self.f16_ptr(s, "gather_rows_device_f16")?;
        let dst_d = self.f32_ptr(d, "gather_rows_device_f16")?;
        let tok_d = match self.get(t, "gather_rows_device_f16")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("gather_rows_device_f16: t must be u32 or f32".into()),
        };
        let func = self.kernel(
            "gather_rows_f16",
            GATHER_ROWS_F16_SRC,
            "rwkv_gather_rows_f16",
        )?;
        let grid = ((c as u32).div_ceil(256), batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &tok_d as *const u64 as *mut c_void,
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn norm_lerp6_batch(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        xr: TensorId,
        xw: TensorId,
        xk: TensorId,
        xv: TensorId,
        xa: TensorId,
        xg: TensorId,
        or: TensorId,
        ow: TensorId,
        ok: TensorId,
        ov: TensorId,
        oa: TensorId,
        og: TensorId,
        c: usize,
        eps: f32,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, xr) = (
            f32(x, "norm_lerp6_batch")?,
            f32(s, "norm_lerp6_batch")?,
            f32(g, "norm_lerp6_batch")?,
            f32(b, "norm_lerp6_batch")?,
            f32(xr, "norm_lerp6_batch")?,
        );
        let (xw, xk, xv, xa, xg) = (
            f32(xw, "norm_lerp6_batch")?,
            f32(xk, "norm_lerp6_batch")?,
            f32(xv, "norm_lerp6_batch")?,
            f32(xa, "norm_lerp6_batch")?,
            f32(xg, "norm_lerp6_batch")?,
        );
        let (or_, ow, ok) = (
            f32(or, "norm_lerp6_batch")?,
            f32(ow, "norm_lerp6_batch")?,
            f32(ok, "norm_lerp6_batch")?,
        );
        let (ov, oa, og) = (
            f32(ov, "norm_lerp6_batch")?,
            f32(oa, "norm_lerp6_batch")?,
            f32(og, "norm_lerp6_batch")?,
        );
        let func = self.kernel("norm_lerp6_batch", NORM_LERP6_BATCH_SRC, "norm_lerp6_batch")?;
        let grid = (c.div_ceil(256).max(1) as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &xr as *const u64 as *mut c_void,
            &xw as *const u64 as *mut c_void,
            &xk as *const u64 as *mut c_void,
            &xv as *const u64 as *mut c_void,
            &xa as *const u64 as *mut c_void,
            &xg as *const u64 as *mut c_void,
            &or_ as *const u64 as *mut c_void,
            &ow as *const u64 as *mut c_void,
            &ok as *const u64 as *mut c_void,
            &ov as *const u64 as *mut c_void,
            &oa as *const u64 as *mut c_void,
            &og as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn cmix_norm_lerp_batch(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        coeff: TensorId,
        out_xb: TensorId,
        c: usize,
        eps: f32,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, coeff, out_xb) = (
            f32(x, "cmix_norm_lerp_batch")?,
            f32(s, "cmix_norm_lerp_batch")?,
            f32(g, "cmix_norm_lerp_batch")?,
            f32(b, "cmix_norm_lerp_batch")?,
            f32(coeff, "cmix_norm_lerp_batch")?,
            f32(out_xb, "cmix_norm_lerp_batch")?,
        );
        let func = self.kernel(
            "cmix_norm_lerp_batch",
            CMIX_NORM_LERP_BATCH_SRC,
            "cmix_norm_lerp_batch",
        )?;
        let grid = (c.div_ceil(256).max(1) as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &coeff as *const u64 as *mut c_void,
            &out_xb as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    /// batch 版：kernel 与单序列版同一份（内部已带 batch=blockIdx.y 维），仅 grid.y=batch。
    #[allow(clippy::too_many_arguments)]
    fn fuse_ka_dplr_norm_batch(
        &mut self,
        s: TensorId,
        k: TensorId,
        k_k: TensorId,
        a: TensorId,
        k_a: TensorId,
        r: TensorId,
        v: TensorId,
        w: TensorId,
        gamma: TensorId,
        beta: TensorId,
        r_k: TensorId,
        k_mod: TensorId,
        y: TensorId,
        y_norm: TensorId,
        h: usize,
        n: usize,
        eps: f32,
        gn_eps: f32,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let sd = self.any_ptr(s, "fuse_ka_dplr_norm_batch")?;
        let kd = f32(k, "fuse_ka_dplr_norm_batch")?;
        let kkd = f32(k_k, "fuse_ka_dplr_norm_batch")?;
        let ad = f16(a, "fuse_ka_dplr_norm_batch")?;
        let kad = f32(k_a, "fuse_ka_dplr_norm_batch")?;
        let rd = f32(r, "fuse_ka_dplr_norm_batch")?;
        let vd = f16(v, "fuse_ka_dplr_norm_batch")?;
        let wd = f16(w, "fuse_ka_dplr_norm_batch")?;
        let gd = f32(gamma, "fuse_ka_dplr_norm_batch")?;
        let bd = f32(beta, "fuse_ka_dplr_norm_batch")?;
        let rkd = f32(r_k, "fuse_ka_dplr_norm_batch")?;
        let kmd = f32(k_mod, "fuse_ka_dplr_norm_batch")?;
        let yd = f32(y, "fuse_ka_dplr_norm_batch")?;
        let ynd = f32(y_norm, "fuse_ka_dplr_norm_batch")?;

        let (key, src) = self.dplr_variant(
            "fuse_ka_dplr_norm",
            FUSE_KA_DPRL_NORM_SRC,
            s,
            "fuse_ka_dplr_norm_batch",
        )?;
        // ★ 2026-09-23：小 batch 抬块内 warp 数（见 `dplr_kaw`）。大 batch 走 4（块已够多）。
        let kaw = Self::dplr_kaw(batch, n);
        let key = format!("{key}_w{kaw}");
        let src = format!("#define KAW {kaw}\n{src}");
        let func = self.kernel(&key, &src, "fuse_ka_dplr_norm")?;
        // 每个 block 处理一个 (head, slot)；kernel 内 batch=blockIdx.y。
        let grid = (h as u32, batch as u32, 1u32);
        let block = ((kaw * 32) as u32, 1u32, 1u32);
        let h_i = h as i32;
        let n_i = n as i32;
        let params = [
            &sd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &kkd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &kad as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &rkd as *const u64 as *mut c_void,
            &kmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &ynd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
            &gn_eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_rkv_stage1_batch(
        &mut self,
        r: &Int8Handle,
        k: &Int8Handle,
        v: &Int8Handle,
        v1: TensorId,
        w1: TensorId,
        a1: TensorId,
        g1: TensorId,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        xw: TensorId,
        xa: TensorId,
        xg: TensorId,
        out_r: TensorId,
        out_k: TensorId,
        out_v: TensorId,
        out_vm: TensorId,
        out_wm: TensorId,
        out_am: TensorId,
        out_gm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
        batch: usize,
    ) -> R<()> {
        let ridx = self.u32_ptr(r.idx, "gemv_int8_rkv_stage1_batch")?;
        let rsz = self.u32_ptr(r.sz, "gemv_int8_rkv_stage1_batch")?;
        let kidx = self.u32_ptr(k.idx, "gemv_int8_rkv_stage1_batch")?;
        let ksz = self.u32_ptr(k.sz, "gemv_int8_rkv_stage1_batch")?;
        let vidx = self.u32_ptr(v.idx, "gemv_int8_rkv_stage1_batch")?;
        let vsz = self.u32_ptr(v.sz, "gemv_int8_rkv_stage1_batch")?;
        let v1d = self.f32_ptr(v1, "gemv_int8_rkv_stage1_batch")?;
        let w1d = self.f32_ptr(w1, "gemv_int8_rkv_stage1_batch")?;
        let a1d = self.f32_ptr(a1, "gemv_int8_rkv_stage1_batch")?;
        let g1d = self.f32_ptr(g1, "gemv_int8_rkv_stage1_batch")?;
        let xrd = self.f32_ptr(xr, "gemv_int8_rkv_stage1_batch")?;
        let xkd = self.f32_ptr(xk, "gemv_int8_rkv_stage1_batch")?;
        let xvd = self.f32_ptr(xv, "gemv_int8_rkv_stage1_batch")?;
        let xwd = self.f32_ptr(xw, "gemv_int8_rkv_stage1_batch")?;
        let xad = self.f32_ptr(xa, "gemv_int8_rkv_stage1_batch")?;
        let xgd = self.f32_ptr(xg, "gemv_int8_rkv_stage1_batch")?;
        let ord = self.f32_ptr(out_r, "gemv_int8_rkv_stage1_batch")?;
        let okd = self.f32_ptr(out_k, "gemv_int8_rkv_stage1_batch")?;
        let ovd = self.f16_ptr(out_v, "gemv_int8_rkv_stage1_batch")?;
        let ovmd = self.f32_ptr(out_vm, "gemv_int8_rkv_stage1_batch")?;
        let owmd = self.f32_ptr(out_wm, "gemv_int8_rkv_stage1_batch")?;
        let oamd = self.f32_ptr(out_am, "gemv_int8_rkv_stage1_batch")?;
        let ogmd = self.f32_ptr(out_gm, "gemv_int8_rkv_stage1_batch")?;

        // r/k/v 分支的 fp16 激活：整批 x 被 grid.x 的每个 block 重读，字节数减半直接
        // 反映到端到端（ABLATE：本 kernel 是 decode 第二大项）。开关 `RKV_F16X`
        // （默认开，2026-09-21 A/B 通过）；`RKV_F16X=0` 回退 fp32 内联降位。
        let use16x = std::env::var("RKV_F16X").map(|v| v != "0").unwrap_or(true);
        // grid：(r/k/v 行段 + mid 行段, slot 分组数, 3 个矩阵)。
        // 与 kernel 内 ROWS/BGRP 同步：ROWS=4 → grid.x 的 r/k/v 段 = c/4；
        // BGRP=8 → B=8 时 grid.y=1（**权重只读一遍**）；grid.z = r/k/v 三矩阵。
        // z>0 的 mid 行段块在 kernel 内直接 return（只浪费空块调度）。
        const RKV_MB_ROWS: usize = 4;
        const RKV_MB_BGRP: usize = 8;
        // ★ 第四代（2026-09-21）：r/k/v 三个 [c,c] int8 投影改走 **IMMA 张量核**
        // （`imma_gemm_dispatch`，与主线性层同一条路径），本 kernel 只保留 mid 投影分支。
        // 依据：同形状 (2560,2560) 隔离计时 SIMT 1.3777 ms → IMMA 0.1919 ms（7.18×），
        // 而 B=256 时 `ABLATE=gemv_int8_rkv_stage1_batch` 移除收益占整步 43%（头号瓶颈）。
        // 开关沿用 `GEMV_IMMA`（**默认开**）；`=0` 即完全回退到原 r/k/v 段。
        let use_imma = batch >= self.imma_min_batch()
            && c.is_multiple_of(QUANT_X_I8_GROUP)
            && env_on("GEMV_IMMA");
        // mid 分支的「每 block 槽跨度」= BGRP × mid_gg。r/k/v 段还在时（IMMA 关）必须为 1
        // （那段的分块几何是 BGRP=8 硬编码的）；IMMA 打开后只剩 mid，可以放大换 x 复用。
        let mid_gg = if use_imma {
            std::env::var("MID_GG")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| (1..=16).contains(v))
                .unwrap_or(1)
        } else {
            1
        };
        let crows = if use_imma { 0usize } else { c / RKV_MB_ROWS };
        let func = self.kernel(
            &format!("gemv_int8_rkv_stage1_batch_g{mid_gg}"),
            &format!("#define MID_GG {mid_gg}\n{GEMV_INT8_RKV_STAGE1_BATCH_SRC}"),
            "gemv_int8_rkv_stage1_batch",
        )?;
        let (xr16d, xk16d, xv16d) = if use_imma {
            // out_r / out_k 覆盖写 fp32；out_v 是 fp16 语义 ⇒ op=4 让 IMMA 内核直接落半精度
            // ★ 2026-09-22：**r/k/v 三条链合并成一次 launch**（`RKV_MERGE`，默认开）。
            // 小 batch 时 `grid.y = batch/BN = 1`，3 次串行 launch 各只有 `C/BM` 个块
            // （B=16/BM=64 ⇒ 40 块，68 个 SM 半数空转）；合并后 120 块一次铺开。
            if std::env::var("RKV_MERGE").map(|v| v != "0").unwrap_or(true) {
                self.imma_gemm_dispatch_z3(
                    [ridx, kidx, vidx],
                    [rsz, ksz, vsz],
                    [xrd, xkd, xvd],
                    [ord, okd, ovd],
                    [3, 3, 4],
                    c,
                    c,
                    batch,
                )?;
            } else {
                self.imma_gemm_dispatch(ridx, rsz, xrd, 0, ord, c, c, batch, 3)?;
                self.imma_gemm_dispatch(kidx, ksz, xkd, 0, okd, c, c, batch, 3)?;
                self.imma_gemm_dispatch(vidx, vsz, xvd, 0, ovd, c, c, batch, 4)?;
            }
            (0u64, 0u64, 0u64)
        } else if use16x && batch > 1 {
            let n = batch * c;
            let base = self.x16_base(3 * n)?;
            let (p0, p1, p2) = (base, base + (n * 2) as u64, base + (2 * n * 2) as u64);
            let cast = self.kernel("cast3_f16", CAST3_F16_SRC, "cast3_f16")?;
            let n_i = n as i32;
            let (s0, s1, s2) = (xrd, xkd, xvd);
            let cparams = [
                &s0 as *const u64 as *mut c_void,
                &s1 as *const u64 as *mut c_void,
                &s2 as *const u64 as *mut c_void,
                &p0 as *const u64 as *mut c_void,
                &p1 as *const u64 as *mut c_void,
                &p2 as *const u64 as *mut c_void,
                &n_i as *const i32 as *mut c_void,
            ];
            let ct = 256u32;
            let cg = ((n as u32).div_ceil(ct), 1u32, 1u32);
            self.drv
                .launch_smem(self.stream, cast, cg, (ct, 1, 1), &cparams, 0)?;
            (p0, p1, p2)
        } else {
            (0u64, 0u64, 0u64)
        };
        // ★ `LOWRANK_GEMM=1` 时 mid 伪行数（vm+wm+am+gm）全 0 ⇒ mid 段由 fp16 张量核
        // GEMM（`lowrank_gemm_batch`）代劳，本函数只剩 r/k/v（上面已走 IMMA）——
        // 整段 mid launch 省掉（否则 grid.x 退化为 0，launch 非法）。
        if use_imma && vm + wm + am + gm == 0 {
            return Ok(());
        }
        let use16x_i = i32::from(use16x && batch > 1 && !use_imma);
        // grid：(r/k/v 行段 + mid 行段, slot 分组数, 3 个矩阵)。
        // 与 kernel 内 ROWS/BGRP 同步：ROWS=4 → grid.x 的 r/k/v 段 = c/4；
        // BGRP=8 → B=8 时 grid.y=1（**权重只读一遍**）；grid.z = r/k/v 三矩阵。
        // z>0 的 mid 行段块在 kernel 内直接 return（只浪费空块调度）。
        let grid = (
            (crows + vm + wm + am + gm) as u32,
            batch.div_ceil(RKV_MB_BGRP * mid_gg) as u32,
            if use_imma { 1u32 } else { 3u32 },
        );
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let vm_i = vm as i32;
        let wm_i = wm as i32;
        let am_i = am as i32;
        let gm_i = gm as i32;
        let batch_i = batch as i32;
        let crows_i = crows as i32;
        let params = [
            &ridx as *const u64 as *mut c_void,
            &rsz as *const u64 as *mut c_void,
            &kidx as *const u64 as *mut c_void,
            &ksz as *const u64 as *mut c_void,
            &vidx as *const u64 as *mut c_void,
            &vsz as *const u64 as *mut c_void,
            &v1d as *const u64 as *mut c_void,
            &w1d as *const u64 as *mut c_void,
            &a1d as *const u64 as *mut c_void,
            &g1d as *const u64 as *mut c_void,
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &xwd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xgd as *const u64 as *mut c_void,
            &ord as *const u64 as *mut c_void,
            &okd as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ovmd as *const u64 as *mut c_void,
            &owmd as *const u64 as *mut c_void,
            &oamd as *const u64 as *mut c_void,
            &ogmd as *const u64 as *mut c_void,
            &xr16d as *const u64 as *mut c_void,
            &xk16d as *const u64 as *mut c_void,
            &xv16d as *const u64 as *mut c_void,
            &use16x_i as *const i32 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &vm_i as *const i32 as *mut c_void,
            &wm_i as *const i32 as *mut c_void,
            &am_i as *const i32 as *mut c_void,
            &gm_i as *const i32 as *mut c_void,
            &batch_i as *const i32 as *mut c_void,
            &crows_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    /// 低秩链两级 fp16 张量核 GEMM（`LOWRANK_GEMM=1`，见 `LOWRANK_GEMM_SRC`）。
    ///
    /// 调用序（同一 stream，顺序即依赖）：
    /// `cast4_f16` → 4×`lowrank_stage1_gemm`（v/w/a/g，act 各异）→ 4×`lowrank_stage2_gemm`
    /// （chain4 epilogue）。权重用**加载时已常驻**的 `*_16`（prefill 也在用，零转换成本）。
    ///
    /// mid16 布局：单块 `[batch, Σmid_pad]`，各链占 `[off, off + mid_pad)` 列段
    /// （与 `W1_16 [mid_pad, C]` / `W2_16 [C, mid_pad]` 的行/列宽一一对应）。
    #[allow(clippy::too_many_arguments)]
    fn lowrank_gemm_batch(
        &mut self,
        v1_16: TensorId,
        w1_16: TensorId,
        a1_16: TensorId,
        g1_16: TensorId,
        w2_16: TensorId,
        a2_16: TensorId,
        v2_16: TensorId,
        g2_16: TensorId,
        xw: TensorId,
        xa: TensorId,
        xv: TensorId,
        xg: TensorId,
        w0: TensorId,
        a0: TensorId,
        v0: TensorId,
        scale: TensorId,
        v_first: TensorId,
        out_w: TensorId,
        out_a: TensorId,
        out_v: TensorId,
        out_g: TensorId,
        c: usize,
        batch: usize,
        wm: usize,
        am: usize,
        vm: usize,
        gm: usize,
        wmp: usize,
        amp: usize,
        vmp: usize,
        gmp: usize,
    ) -> R<()> {
        assert!(batch >= IMMA_MIN_BATCH, "lowrank_gemm_batch: batch 过小");
        assert!(c.is_multiple_of(8), "lowrank_gemm_batch: C 必须是 8 的倍数");
        let total = vmp + wmp + amp + gmp;
        assert!(
            [wmp, amp, vmp, gmp].iter().all(|p| p.is_multiple_of(64)),
            "lowrank_gemm_batch: 各 mid_pad 必须是 64 的倍数"
        );
        // 暂存切分（捕获前已建好）：x16 段 4×[batch, C] + mid16 段 [batch, total]。
        let n = batch * c;
        let base = self.lr16_base(4 * n + batch * total)?;
        let (xw16, xa16, xv16, xg16) = (
            base,
            base + (n * 2) as u64,
            base + (2 * n * 2) as u64,
            base + (3 * n * 2) as u64,
        );
        let mid_base = base + (4 * n * 2) as u64;
        // x 降位（纯 cast，不含 sigmoid —— g 的 sigmoid 在 stage1 epilogue）。
        let (xwd, xad, xvd, xgd) = (
            self.f32_ptr(xw, "lowrank_gemm_batch")?,
            self.f32_ptr(xa, "lowrank_gemm_batch")?,
            self.f32_ptr(xv, "lowrank_gemm_batch")?,
            self.f32_ptr(xg, "lowrank_gemm_batch")?,
        );
        {
            let cast = self.kernel("cast4_f16", CAST4_F16_SRC, "cast4_f16")?;
            let n_i = n as i32;
            let cparams = [
                &xwd as *const u64 as *mut c_void,
                &xad as *const u64 as *mut c_void,
                &xvd as *const u64 as *mut c_void,
                &xgd as *const u64 as *mut c_void,
                &xw16 as *const u64 as *mut c_void,
                &xa16 as *const u64 as *mut c_void,
                &xv16 as *const u64 as *mut c_void,
                &xg16 as *const u64 as *mut c_void,
                &n_i as *const i32 as *mut c_void,
            ];
            let ct = 256u32;
            let cg = ((n as u32).div_ceil(ct), 1u32, 1u32);
            self.drv
                .launch_smem(self.stream, cast, cg, (ct, 1, 1), &cparams, 0)?;
        }
        let (b1, n1, k1d, b2, n2, k2d) = lr_tiles();
        // BK 必须除尽 k：一级 k=C、二级 k=mid_pad（g 链 320 ⇒ 取 64 而非 128）。
        let k1 = if c.is_multiple_of(k1d) {
            k1d
        } else if c.is_multiple_of(64) {
            64
        } else {
            0
        };
        assert!(k1 != 0, "lowrank_gemm_batch: C={c} 需能被 64 整除");
        let pads_all_div = |d: usize| [wmp, amp, vmp, gmp].iter().all(|p| p.is_multiple_of(d));
        let k2 = if pads_all_div(k2d) { k2d } else { 64 };
        assert!(
            pads_all_div(k2),
            "lowrank_gemm_batch: 各 mid_pad 需能被 {k2} 整除"
        );
        let kkey = format!("lowrank_gemm_1b{b1}n{n1}k{k1}_2b{b2}n{n2}k{k2}");
        let src = format!(
            "#define LR1_BM {b1}\n#define LR1_BN {n1}\n#define LR1_BK {k1}\n\
             #define LR2_BM {b2}\n#define LR2_BN {n2}\n#define LR2_BK {k2}\n{LOWRANK_GEMM_SRC}"
        );
        // ⚠️ `kernel()` 的缓存**只按 `key` 索引、不校验 `entry`** ⇒ 两个入口必须用不同的 key，
        // 否则第二次调用会拿到第一次缓存的函数句柄（实测：stage2 拿到了 stage1 的句柄，
        // 于是 stage1 用 stage2 的参数布局读参 → 把指针当 int 读，地址野飞到 1GB 外）。
        let f1 = self.kernel(&format!("{kkey}_s1"), &src, "lowrank_stage1_gemm")?;
        let f2 = self.kernel(&format!("{kkey}_s2"), &src, "lowrank_stage2_gemm")?;
        // 链序固定 [v, w, a, g]（与 chain4 的 chain 编号一致）。
        let chains = [
            (v1_16, v2_16, xv16, vmp, vm, 0i32, 0i32),
            (w1_16, w2_16, xw16, wmp, wm, 1i32, 1i32),
            (a1_16, a2_16, xa16, amp, am, 0i32, 2i32),
            (g1_16, g2_16, xg16, gmp, gm, 2i32, 3i32),
        ];
        let scd = self.f32_ptr(scale, "lowrank_gemm_batch")?;
        let vfd = self.f16_ptr(v_first, "lowrank_gemm_batch")?;
        // —— 一级：把 4 条链的 (W, X, mid, n_pad, act) 收齐 ——
        // 合并 launch 的收益见 `lowrank_stage1_gemm` 上方注释（v 链单独跑只有 16 个块）。
        let mut s1 = [(0u64, 0u64, 0u64, 0usize, 0i32); 4];
        let mut offs = [0usize; 4];
        {
            let mut o = 0usize;
            for (i, (w1t, _w2t, x16, pad, _real, act, _chain)) in chains.iter().enumerate() {
                s1[i] = (
                    self.f16_ptr(*w1t, "lowrank_gemm_batch")?,
                    *x16,
                    mid_base + (o * 2) as u64,
                    *pad,
                    *act,
                );
                offs[i] = o;
                o += *pad;
            }
        }
        // `LR1_MERGE=0` 回退「4 次串行 launch」（每次 grid.z=1）。
        let merge = std::env::var("LR1_MERGE").map(|v| v != "0").unwrap_or(true);
        if merge {
            let max_pad = s1.iter().map(|x| x.3).max().unwrap_or(0);
            let (np0, np1, np2, np3) = (
                s1[0].3 as i32,
                s1[1].3 as i32,
                s1[2].3 as i32,
                s1[3].3 as i32,
            );
            let (ac0, ac1, ac2, ac3) = (s1[0].4, s1[1].4, s1[2].4, s1[3].4);
            let k_i = c as i32;
            let b_i = batch as i32;
            let xs_i = c as i32;
            let ms_i = total as i32;
            let p1 = [
                &s1[0].0 as *const u64 as *mut c_void,
                &s1[1].0 as *const u64 as *mut c_void,
                &s1[2].0 as *const u64 as *mut c_void,
                &s1[3].0 as *const u64 as *mut c_void,
                &s1[0].1 as *const u64 as *mut c_void,
                &s1[1].1 as *const u64 as *mut c_void,
                &s1[2].1 as *const u64 as *mut c_void,
                &s1[3].1 as *const u64 as *mut c_void,
                &s1[0].2 as *const u64 as *mut c_void,
                &s1[1].2 as *const u64 as *mut c_void,
                &s1[2].2 as *const u64 as *mut c_void,
                &s1[3].2 as *const u64 as *mut c_void,
                &np0 as *const i32 as *mut c_void,
                &np1 as *const i32 as *mut c_void,
                &np2 as *const i32 as *mut c_void,
                &np3 as *const i32 as *mut c_void,
                &ac0 as *const i32 as *mut c_void,
                &ac1 as *const i32 as *mut c_void,
                &ac2 as *const i32 as *mut c_void,
                &ac3 as *const i32 as *mut c_void,
                &k_i as *const i32 as *mut c_void,
                &b_i as *const i32 as *mut c_void,
                &xs_i as *const i32 as *mut c_void,
                &ms_i as *const i32 as *mut c_void,
            ];
            let g1 = (batch.div_ceil(b1) as u32, max_pad.div_ceil(n1) as u32, 4u32);
            self.drv
                .launch_smem(self.stream, f1, g1, (256, 1, 1), &p1, 0)?;
        }
        for (ci, (w1t, w2t, x16, pad, real, act, chain)) in chains.into_iter().enumerate() {
            assert!(pad >= real, "lowrank_gemm_batch: mid_pad < mid");
            let off = offs[ci];
            // 一级（仅 `LR1_MERGE=0` 的回退路径）：每次只跑一条链，grid.z=1 ⇒ z 恒为 0。
            if !merge {
                let (w1d, _, mid, _, _) = s1[ci];
                let _ = w1t;
                let (np_i, k_i, b_i, act_i) = (pad as i32, c as i32, batch as i32, act);
                let (xs_i, ms_i) = (c as i32, total as i32);
                let z0 = 0i32;
                let (x0, x1, x2, x3) = (x16, x16, x16, x16);
                let (m1, m2, m3) = (mid, mid, mid);
                let p1 = [
                    &w1d as *const u64 as *mut c_void,
                    &w1d as *const u64 as *mut c_void,
                    &w1d as *const u64 as *mut c_void,
                    &w1d as *const u64 as *mut c_void,
                    &x0 as *const u64 as *mut c_void,
                    &x1 as *const u64 as *mut c_void,
                    &x2 as *const u64 as *mut c_void,
                    &x3 as *const u64 as *mut c_void,
                    &mid as *const u64 as *mut c_void,
                    &m1 as *const u64 as *mut c_void,
                    &m2 as *const u64 as *mut c_void,
                    &m3 as *const u64 as *mut c_void,
                    &np_i as *const i32 as *mut c_void,
                    &z0 as *const i32 as *mut c_void,
                    &z0 as *const i32 as *mut c_void,
                    &z0 as *const i32 as *mut c_void,
                    &act_i as *const i32 as *mut c_void,
                    &z0 as *const i32 as *mut c_void,
                    &z0 as *const i32 as *mut c_void,
                    &z0 as *const i32 as *mut c_void,
                    &k_i as *const i32 as *mut c_void,
                    &b_i as *const i32 as *mut c_void,
                    &xs_i as *const i32 as *mut c_void,
                    &ms_i as *const i32 as *mut c_void,
                ];
                let g1 = (batch.div_ceil(b1) as u32, pad.div_ceil(n1) as u32, 1u32);
                self.drv
                    .launch_smem(self.stream, f1, g1, (256, 1, 1), &p1, 0)?;
            }
            // 二级：k = mid_pad，n = C，chain4 epilogue（v 链读改写 out_v）。
            let _ = w1t;
            let mut bd: u64 = 0;
            if chain != 3 {
                let b = if chain == 0 {
                    v0
                } else if chain == 1 {
                    w0
                } else {
                    a0
                };
                bd = self.f32_ptr(b, "lowrank_gemm_batch")?;
            }
            let out = match chain {
                0 => self.f16_ptr(out_v, "lowrank_gemm_batch")?,
                1 => self.f16_ptr(out_w, "lowrank_gemm_batch")?,
                2 => self.f16_ptr(out_a, "lowrank_gemm_batch")?,
                _ => self.f16_ptr(out_g, "lowrank_gemm_batch")?,
            };
            let w2d = self.f16_ptr(w2t, "lowrank_gemm_batch")?;
            let xoff = mid_base + (off * 2) as u64;
            let (cnp_i, ck_i, cb_i, ch_i, cxs_i) =
                (c as i32, pad as i32, batch as i32, chain, total as i32);
            let p2 = [
                &w2d as *const u64 as *mut c_void,
                &xoff as *const u64 as *mut c_void,
                &bd as *const u64 as *mut c_void,
                &vfd as *const u64 as *mut c_void,
                &out as *const u64 as *mut c_void,
                &scd as *const u64 as *mut c_void,
                &cnp_i as *const i32 as *mut c_void,
                &ck_i as *const i32 as *mut c_void,
                &cb_i as *const i32 as *mut c_void,
                &ch_i as *const i32 as *mut c_void,
                &cxs_i as *const i32 as *mut c_void,
            ];
            let g2 = (batch.div_ceil(b2) as u32, c.div_ceil(n2) as u32, 1u32);
            self.drv
                .launch_smem(self.stream, f2, g2, (256, 1, 1), &p2, 0)?;
            let _ = x16;
        }
        Ok(())
    }

    /// ffn_value 稠密 fp16 张量核 GEMM（`FFN_VALUE_GEMM`，见 `ffn_value_gemm`）。
    /// 序：`cast_f16`（r2 fp32 → r2_16）→ `ffn_value_gemm`（就地累加到 x）。
    fn ffn_value_gemm_batch(
        &mut self,
        w16: TensorId,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
        batch: usize,
    ) -> R<()> {
        assert!(batch >= IMMA_MIN_BATCH, "ffn_value_gemm_batch: batch 过小");
        let (b3, n3, k3) = lr3_tile(fh, c);
        let kkey = format!("ffn_value_gemm_b{b3}n{n3}k{k3}");
        let src = format!(
            "#define LR3_BM {b3}\n#define LR3_BN {n3}\n#define LR3_BK {k3}\n{LOWRANK_GEMM_SRC}"
        );
        // ⚠️ 每个入口用不同 key（`kernel()` 的缓存不校验 entry，见 §3d.3）。
        let cast = self.kernel(&format!("{kkey}_cast"), &format!("{src}\n"), "cast_f16")?;
        let gemm = self.kernel(&format!("{kkey}_g"), &src, "ffn_value_gemm")?;
        let r2d = self.f32_ptr(r2, "ffn_value_gemm_batch")?;
        let r2t = self.ffn16_scratch(batch * fh)?;
        let r2t_d = self.f16_ptr(r2t, "ffn_value_gemm_batch")?;
        {
            let n_i = (batch * fh) as i32;
            let p = [
                &r2d as *const u64 as *mut c_void,
                &r2t_d as *const u64 as *mut c_void,
                &n_i as *const i32 as *mut c_void,
            ];
            let g = (((batch * fh) as u32).div_ceil(256), 1u32, 1u32);
            self.drv
                .launch_smem(self.stream, cast, g, (256, 1, 1), &p, 0)?;
        }
        let wd = self.f16_ptr(w16, "ffn_value_gemm_batch")?;
        let xd = self.f32_ptr(x, "ffn_value_gemm_batch")?;
        let (np_i, k_i, b_i, xs_i) = (c as i32, fh as i32, batch as i32, fh as i32);
        let p = [
            &wd as *const u64 as *mut c_void,
            &r2t_d as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &np_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
            &xs_i as *const i32 as *mut c_void,
        ];
        let grid = (batch.div_ceil(b3) as u32, c.div_ceil(n3) as u32, 1u32);
        self.drv
            .launch_smem(self.stream, gemm, grid, (256, 1, 1), &p, 0)
    }

    /// ffn_value 稠密 **int8 IMMA** GEMM（`FFN_VALUE_IMMA`，见 `ffn_value_gemm`）。
    /// 与 fp16 版同语义（`x[b,c] += Σ_f r2[b,f]·W[c,f]`），两点差别：
    /// ① 权重是 int8 常驻（`{key}.int8_idx` 本身就是 `[c, fh/4]` u32，**零转换**），
    ///    字节数只有 fp16 版的一半 ⇒ 权重流量 105MB/层（fp16 版 210MB）；
    /// ② A 侧由 `imma_gemm_dispatch` 在核内量化成 int8（复用 `quant_x_i8`），
    ///    连带把 `cast_f16`（r2 fp32 → fp16，10.5MB 读 + 5.2MB 写）整段省掉。
    /// 形状：m = c = 2560（输出行）、k = fh = 10240（归约维）、op=1（就地累加 fp32）。
    fn ffn_value_imma_batch(
        &mut self,
        a: &Int8Handle,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
        batch: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "ffn_value_imma_batch")?;
        let asz = self.u32_ptr(a.sz, "ffn_value_imma_batch")?;
        let rd = self.f32_ptr(r2, "ffn_value_imma_batch")?;
        let xd = self.f32_ptr(x, "ffn_value_imma_batch")?;
        // ★ 本形状 m = C = 2560（比 r/k/v 的 2560 同量级但 k = fh = 10240 长 4 倍）：
        // 全局默认 BM=128 时 grid 只有 `(2560/128)×(256/64) = 80` 个 block，
        // 喂不满 68 个 SM（1.18 波，尾部空转）。BM=64 ⇒ 160 个 block（2.35 波）。
        let bm = env_tile("FFN_IMMA_BM", 64);
        // ★ 小 batch 时 `BN ≈ batch`，块数不足 68 时按 2 的幂缩 BN（见 `pick_bn`）——
        // 实测 B=16 **634.5 → 703.0**；B=64 的 `ffn_value`（40 块）缩到 BN=32 变 80 块。
        let bn = if batch <= 64 {
            env_tile("FFN_IMMA_BN_SMALL", pick_bn(c, bm, batch))
        } else {
            env_tile("FFN_IMMA_BN", 64)
        };
        self.imma_gemm_tiled(aidx, asz, rd, 0, xd, c, fh, batch, 1, bm, bn)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_lowrank_chain4_batch(
        &mut self,
        w2: TensorId,
        a2: TensorId,
        v2: TensorId,
        g2: TensorId,
        w_mid: TensorId,
        a_mid: TensorId,
        v_mid: TensorId,
        g_mid: TensorId,
        w0: TensorId,
        a0: TensorId,
        v0: TensorId,
        scale: TensorId,
        v_first: TensorId,
        out_w: TensorId,
        out_a: TensorId,
        out_v: TensorId,
        out_g: TensorId,
        m: usize,
        kw: usize,
        ka: usize,
        kv: usize,
        kg: usize,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let w2d = f32(w2, "gemv_lowrank_chain4_batch")?;
        let a2d = f32(a2, "gemv_lowrank_chain4_batch")?;
        let v2d = f32(v2, "gemv_lowrank_chain4_batch")?;
        let g2d = f32(g2, "gemv_lowrank_chain4_batch")?;
        let wmd = f32(w_mid, "gemv_lowrank_chain4_batch")?;
        let amd = f32(a_mid, "gemv_lowrank_chain4_batch")?;
        let vmd = f32(v_mid, "gemv_lowrank_chain4_batch")?;
        let gmd = f32(g_mid, "gemv_lowrank_chain4_batch")?;
        let w0d = f32(w0, "gemv_lowrank_chain4_batch")?;
        let a0d = f32(a0, "gemv_lowrank_chain4_batch")?;
        let v0d = f32(v0, "gemv_lowrank_chain4_batch")?;
        let scaled = f32(scale, "gemv_lowrank_chain4_batch")?;
        let vfd = f16(v_first, "gemv_lowrank_chain4_batch")?;
        let owd = f16(out_w, "gemv_lowrank_chain4_batch")?;
        let oad = f16(out_a, "gemv_lowrank_chain4_batch")?;
        let ovd = f16(out_v, "gemv_lowrank_chain4_batch")?;
        let ogd = f16(out_g, "gemv_lowrank_chain4_batch")?;

        // warp-per-row：grid.x = ceil(M/8)（每 block 8 warp 各 1 行），grid.y = slot 分组。
        // ⚠️ 见 kernel 内的反例留档：**不要把 slot 挪进块内循环**（削流量但塌并行度，实测更慢）。
        const CHAIN4_WARPS: usize = 8;
        // fp16 张量核版（`CHAIN4_FP16=1`）：内层 4 链 GEMM 走 `mma.m16n8k8`。
        let fp16 = std::env::var("CHAIN4_FP16").is_ok_and(|v| v != "0");
        let fast = std::env::var("CHAIN4_FASTEXP").is_ok_and(|v| v != "0");
        let (key, src) = if fp16 {
            (
                "gemv_lowrank_chain4_batch_fp16",
                GEMV_LOWRANK_CHAIN4_BATCH_FP16_SRC.to_string(),
            )
        } else if fast {
            (
                "gemv_lowrank_chain4_batch_fx",
                format!("#define CHAIN4_FASTEXP 1\n{GEMV_LOWRANK_CHAIN4_BATCH_SRC}"),
            )
        } else {
            (
                "gemv_lowrank_chain4_batch",
                GEMV_LOWRANK_CHAIN4_BATCH_SRC.to_string(),
            )
        };
        let func = self.kernel(key, &src, "gemv_lowrank_chain4_batch")?;
        const CHAIN4_BGRP: usize = 4;
        let grid = (
            m.div_ceil(CHAIN4_WARPS) as u32,
            batch.div_ceil(CHAIN4_BGRP) as u32,
            1u32,
        );
        let block = (256u32, 1u32, 1u32);
        let m_i = m as i32;
        let kw_i = kw as i32;
        let ka_i = ka as i32;
        let kv_i = kv as i32;
        let kg_i = kg as i32;
        let batch_i = batch as i32;
        let params = [
            &w2d as *const u64 as *mut c_void,
            &a2d as *const u64 as *mut c_void,
            &v2d as *const u64 as *mut c_void,
            &g2d as *const u64 as *mut c_void,
            &wmd as *const u64 as *mut c_void,
            &amd as *const u64 as *mut c_void,
            &vmd as *const u64 as *mut c_void,
            &gmd as *const u64 as *mut c_void,
            &w0d as *const u64 as *mut c_void,
            &a0d as *const u64 as *mut c_void,
            &v0d as *const u64 as *mut c_void,
            &scaled as *const u64 as *mut c_void,
            &vfd as *const u64 as *mut c_void,
            &owd as *const u64 as *mut c_void,
            &oad as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ogd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &kw_i as *const i32 as *mut c_void,
            &ka_i as *const i32 as *mut c_void,
            &kv_i as *const i32 as *mut c_void,
            &kg_i as *const i32 as *mut c_void,
            &batch_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn ffn_value_sparse_add_batch(
        &mut self,
        value_tiled: TensorId,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
        batch: usize,
    ) -> R<()> {
        let vt = self.f16_ptr(value_tiled, "ffn_value_sparse_add_batch")?;
        let rd = self.f32_ptr(r2, "ffn_value_sparse_add_batch")?;
        let xd = self.f32_ptr(x, "ffn_value_sparse_add_batch")?;
        let func = self.kernel(
            "ffn_value_sparse_add_batch",
            FFN_VALUE_SPARSE_BATCH_SRC,
            "ffn_value_sparse_add_batch",
        )?;
        // grid = (c/C_TILE, batch, nf)：f 维进块内循环（消除 80× 原子争抢）。
        // nf：**大 batch 一个 block 吃完全部 fh（零原子）**；小 batch 块数不够时拆 f 换并行度
        // （那时争抢方只有 nf 个）。实测 B=256 → nf=1 最优；B=8 不拆会掉 ~11%。
        let nf = std::env::var("FFN_NF")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| (1..=fh / 128).contains(v))
            .unwrap_or(if batch >= 64 {
                1
            } else {
                (512 / batch.max(1)).clamp(1, fh / 128)
            });
        let grid = ((c / 256) as u32, batch as u32, nf as u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let fh_i = fh as i32;
        let params = [
            &rd as *const u64 as *mut c_void,
            &vt as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &fh_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_into_host_seeded_batch(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temp: TensorId,
        mask: TensorId,
        counter: TensorId,
        sampler: TensorId,
        hist: TensorId,
        batch: usize,
        hist_stride: usize,
    ) -> R<()> {
        let logits_d = self.f32_ptr(logits, "sample_into_host_seeded_batch")?;
        let token_d = self.f32_ptr(token, "sample_into_host_seeded_batch")?;
        let temp_d = self.f32_ptr(temp, "sample_into_host_seeded_batch")?;
        let mask_d = self.f32_ptr(mask, "sample_into_host_seeded_batch")?;
        let counter_d = self.u32_ptr(counter, "sample_into_host_seeded_batch")?;
        let sampler_d = self.f32_ptr(sampler, "sample_into_host_seeded_batch")?;
        let hist_d = match self.get(hist, "sample_into_host_seeded_batch")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("sample_into_host_seeded_batch: hist must be u32 or f32".into()),
        };
        let func = self.kernel("rwkv_sample_batch", SAMPLE_BATCH_SRC, "rwkv_sample_batch")?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (112u32, 1u32, 1u32);
        let n_i = n as i32;
        let hs_i = hist_stride as i32;
        let params = [
            &logits_d as *const u64 as *mut c_void,
            &token_d as *const u64 as *mut c_void,
            &temp_d as *const u64 as *mut c_void,
            &mask_d as *const u64 as *mut c_void,
            &counter_d as *const u64 as *mut c_void,
            &sampler_d as *const u64 as *mut c_void,
            &hist_d as *const u64 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &hs_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    fn record_tokens(
        &mut self,
        in_tok: TensorId,
        out_seq: TensorId,
        cnt: TensorId,
        stride: usize,
        batch: usize,
    ) -> R<()> {
        let ptr = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::U32 { dptr, .. } => Ok(dptr),
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be u32 or f32").into()),
            }
        };
        let in_tok_d = ptr(in_tok, "record_tokens")?;
        let out_d = ptr(out_seq, "record_tokens")?;
        let cnt_d = ptr(cnt, "record_tokens")?;
        let func = self.kernel(
            "rwkv_record_tokens",
            RECORD_TOKENS_SRC,
            "rwkv_record_tokens",
        )?;
        let grid = (batch as u32, 1u32, 1u32);
        let block = (1u32, 1u32, 1u32);
        let stride_i = stride as i32;
        let params = [
            &in_tok_d as *const u64 as *mut c_void,
            &out_d as *const u64 as *mut c_void,
            &cnt_d as *const u64 as *mut c_void,
            &stride_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    /// batch 版异步 sampler 上传：宽行（batch*10 f32 = batch*40 字节）。
    /// pinned 区按宽行切分：可用轮数 = PINNED_ROWS / batch（行宽随 batch 增大）。
    #[allow(clippy::too_many_arguments)]
    fn store_sampler_async_batch(
        &self,
        sampler: TensorId,
        row: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seeds: &[u32],
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
        penalty_decay: f32,
    ) -> R<()> {
        let batch = seeds.len();
        let row_bytes = batch * PINNED_ROW_BYTES;
        let max_rows = self.pinned_rows / batch.max(1);
        if row >= max_rows {
            return Err(format!(
                "store_sampler_async_batch: row {row} >= max rows {max_rows} (batch {batch})"
            )
            .into());
        }
        let dptr = match self.get(sampler, "store_sampler_async_batch")? {
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("store_sampler_async_batch: sampler must be f32".into()),
        };
        // 每 slot 10 个 f32：temperature/top_k/top_p/seed/rep/freq/pres/hist_len/decay/保留。
        let mut data = Vec::with_capacity(batch * 10);
        for &seed in seeds {
            data.extend_from_slice(&[
                temperature,
                f32::from_bits(top_k),
                top_p,
                f32::from_bits(seed),
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                f32::from_bits(hist_len),
                penalty_decay,
                0.0,
            ]);
        }
        // 写 pinned 宽行 + 流序异步拷贝（零 host 同步）。
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                (self.pinned as *mut u8).add(row * row_bytes),
                row_bytes,
            );
        }
        let src = unsafe { (self.pinned as *const u8).add(row * row_bytes) };
        cu_check!(
            (self.drv.cu_memcpy_htod_async)(dptr, src as *const c_void, row_bytes, self.stream),
            "cuMemcpyHtoDAsync(sampler batch)"
        );
        Ok(())
    }

    fn gather_row_device_f16(&mut self, s: TensorId, d: TensorId, t: TensorId, c: usize) -> R<()> {
        // in_src 为 fp16 表 [VOCAB, C]；out_dst 为 f32 [C]；in_tok 为 f32 位模式存 uint 索引。
        let src_d = self.f16_ptr(s, "gather_row_device_f16")?;
        let dst_d = self.f32_ptr(d, "gather_row_device_f16")?;
        // tok 为 F32（current_token 用 f32 位模式存 uint）或 U32，位模式相同，取 dptr。
        let tok_d = match self.get(t, "gather_row_device_f16")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("gather_row_device_f16: t must be u32 or f32".into()),
        };
        let func = self.kernel("gather_row_f16", GATHER_ROW_F16_SRC, "rwkv_gather_row_f16")?;
        let grid = ((c as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &tok_d as *const u64 as *mut c_void,
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn copy_device_f16(&mut self, s: TensorId, d: TensorId) -> R<()> {
        let src_d = self.f16_ptr(s, "copy_device_f16")?;
        let dst_d = self.f16_ptr(d, "copy_device_f16")?;
        let len = *self
            .lens
            .get(&s)
            .ok_or("copy_device_f16: unknown src len")?;
        let func = self.kernel(
            "copy_device_f16",
            COPY_DEVICE_F16_SRC,
            "rwkv_copy_device_f16",
        )?;
        let grid = ((len as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let len_i = len as i32;
        let params = [
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &len_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_f16(
        &mut self,
        w: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        n: usize,
    ) -> R<()> {
        // 与 kernel 每 block 4 行对齐：M 需为 4 的倍数（GEMV_ROWS）。
        if !m.is_multiple_of(4) {
            return Err(format!("gemv_f16: M={m} must be divisible by 4 (GEMV_ROWS)").into());
        }
        // w 为 fp16 权重 (M,K)；x 为 f32 (K·batch)；y 为 f32 (M·batch)。
        let (a, x, y) = match self.get(w, "gemv_f16")? {
            CudaTensor::F16 { dptr, .. } => {
                let x = match self.get(x, "gemv_f16")? {
                    CudaTensor::F32 { dptr, .. } => dptr,
                    _ => return Err("gemv_f16: x must be f32 tensor".into()),
                };
                let y = match self.get(y, "gemv_f16")? {
                    CudaTensor::F32 { dptr, .. } => dptr,
                    _ => return Err("gemv_f16: y must be f32 tensor".into()),
                };
                (dptr, x, y)
            }
            _ => return Err("gemv_f16: w must be f16 tensor".into()),
        };
        // 编译并缓存 kernel。
        let func = self.kernel("gemv_f16", GEMV_F16_SRC, "gemv_f16")?;
        // grid.x = M/4，grid.y = batch；block = 128 线程。
        let grid = ((m / 4) as u32, n as u32, 1);
        let block = (128u32, 1u32, 1u32);
        // 按值传参（需取地址，指针参数为 u64 设备地址）。
        let m_i = m as i32;
        let k_i = k as i32;
        let batch_i = n as i32;
        let params = [
            &a as *const u64 as *mut c_void,
            &x as *const u64 as *mut c_void,
            &y as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &batch_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn norm(
        &mut self,
        x: TensorId,
        g: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        h: usize,
        eps: f32,
        rows: usize,
    ) -> R<()> {
        // 全部为 f32 张量，取 device 指针。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (xd, gd, bd, yd) = (
            f32(x, "norm")?,
            f32(g, "norm")?,
            f32(b, "norm")?,
            f32(y, "norm")?,
        );
        let func = self.kernel("norm", NORM_SRC, "rwkv_norm")?;
        // 每个 block 归一化一个 (head,batch) 行：grid.x = rows*h（对齐 Vulkan 的 (h, batch) 网格）。
        // h=1 时退化为 rows（layer norm）；h>1 时覆盖全部 head（group norm），
        // 否则非首 head 输出不被写入，残留脏数据导致跨 run 非确定。
        let grid = ((rows * h) as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let h_i = h as i32;
        let params = [
            &xd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn norm_lerp6(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        xr: TensorId,
        xw: TensorId,
        xk: TensorId,
        xv: TensorId,
        xa: TensorId,
        xg: TensorId,
        or: TensorId,
        ow: TensorId,
        ok: TensorId,
        ov: TensorId,
        oa: TensorId,
        og: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()> {
        // 全部为 f32 张量，取 device 指针。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, xr) = (
            f32(x, "norm_lerp6")?,
            f32(s, "norm_lerp6")?,
            f32(g, "norm_lerp6")?,
            f32(b, "norm_lerp6")?,
            f32(xr, "norm_lerp6")?,
        );
        let (xw, xk, xv, xa, xg) = (
            f32(xw, "norm_lerp6")?,
            f32(xk, "norm_lerp6")?,
            f32(xv, "norm_lerp6")?,
            f32(xa, "norm_lerp6")?,
            f32(xg, "norm_lerp6")?,
        );
        let (or, ow, ok) = (
            f32(or, "norm_lerp6")?,
            f32(ow, "norm_lerp6")?,
            f32(ok, "norm_lerp6")?,
        );
        let (ov, oa, og) = (
            f32(ov, "norm_lerp6")?,
            f32(oa, "norm_lerp6")?,
            f32(og, "norm_lerp6")?,
        );
        let func = self.kernel("norm_lerp6", NORM_LERP6_SRC, "norm_lerp6")?;
        // 多 block 并行：每个 block 负责 256 个 C 片段（冗余归约 + 分段 apply）。
        let grid = (c.div_ceil(256).max(1) as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &xr as *const u64 as *mut c_void,
            &xw as *const u64 as *mut c_void,
            &xk as *const u64 as *mut c_void,
            &xv as *const u64 as *mut c_void,
            &xa as *const u64 as *mut c_void,
            &xg as *const u64 as *mut c_void,
            &or as *const u64 as *mut c_void,
            &ow as *const u64 as *mut c_void,
            &ok as *const u64 as *mut c_void,
            &ov as *const u64 as *mut c_void,
            &oa as *const u64 as *mut c_void,
            &og as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn cmix_norm_lerp(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        coeff: TensorId,
        out_xb: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, coeff, out_xb) = (
            f32(x, "cmix_norm_lerp")?,
            f32(s, "cmix_norm_lerp")?,
            f32(g, "cmix_norm_lerp")?,
            f32(b, "cmix_norm_lerp")?,
            f32(coeff, "cmix_norm_lerp")?,
            f32(out_xb, "cmix_norm_lerp")?,
        );
        let func = self.kernel("cmix_norm_lerp", CMIX_NORM_LERP_SRC, "cmix_norm_lerp")?;
        // ★ 2026-09-23：**单块覆盖整个 C**（`CMIX_NORM_BLK`，默认 1024）。
        //
        // 病灶（B=1 单流实测 0.50 ms/token、18.6 µs/次）：旧版 `grid = ceil(c/256) = 10` 个块，
        // 而**每个块都把整个 [0,c) 归约一遍**（mean / inv_std 必须全向量）⇒ 归约流量
        // 10×2560 = 25600 次载入，实际只需 2560；有效带宽仅 ~3.8 GB/s。
        // 大 batch 走的是另一个内核（`cmix_norm_lerp_batch`，按 batch 维分块），
        // **单流路径只有这一串块**，冗余被完全暴露。
        //
        // 改法：`grid = 1`、`block = 1024` ⇒ 归约**只做一遍**（每线程 2~3 个元素），
        // apply 段也由同一批线程扫完整个 C。归约结构（每线程部分和 → warp 内 shfl →
        // tid0 顺序累加 warp 和）不变，只是每线程负责的元素集合变了 ⇒ 浮点求和顺序变、
        // 相对差 ~1e-7（门禁 `cmix_norm_lerp_matches_cpu`）。
        // 注意：`s_val[32]/s_sq[32]` 是 32 个 warp 槽 ⇒ block ≤ 1024 恒安全。
        let blk = std::env::var("CMIX_NORM_BLK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            .clamp(32, 1024);
        let grid = (c.div_ceil(blk).max(1) as u32, 1u32, 1u32);
        let block = (blk as u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &coeff as *const u64 as *mut c_void,
            &out_xb as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn fuse_ka_dplr_norm(
        &mut self,
        s: TensorId,
        k: TensorId,
        k_k: TensorId,
        a: TensorId,
        k_a: TensorId,
        r: TensorId,
        v: TensorId,
        w: TensorId,
        gamma: TensorId,
        beta: TensorId,
        r_k: TensorId,
        k_mod: TensorId,
        y: TensorId,
        y_norm: TensorId,
        h: usize,
        n: usize,
        eps: f32,
        gn_eps: f32,
    ) -> R<()> {
        // s 为 f32（in-place 更新，写 device 指针）；a/v/w 为 fp16；其余 f32。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let sd = self.any_ptr(s, "fuse_ka_dplr_norm")?;
        let kd = f32(k, "fuse_ka_dplr_norm")?;
        let kkd = f32(k_k, "fuse_ka_dplr_norm")?;
        let ad = f16(a, "fuse_ka_dplr_norm")?;
        let kad = f32(k_a, "fuse_ka_dplr_norm")?;
        let rd = f32(r, "fuse_ka_dplr_norm")?;
        let vd = f16(v, "fuse_ka_dplr_norm")?;
        let wd = f16(w, "fuse_ka_dplr_norm")?;
        let gd = f32(gamma, "fuse_ka_dplr_norm")?;
        let bd = f32(beta, "fuse_ka_dplr_norm")?;
        let rkd = f32(r_k, "fuse_ka_dplr_norm")?;
        let kmd = f32(k_mod, "fuse_ka_dplr_norm")?;
        let yd = f32(y, "fuse_ka_dplr_norm")?;
        let ynd = f32(y_norm, "fuse_ka_dplr_norm")?;

        let (key, src) = self.dplr_variant(
            "fuse_ka_dplr_norm",
            FUSE_KA_DPRL_NORM_SRC,
            s,
            "fuse_ka_dplr_norm",
        )?;
        // ★ 2026-09-23：小 batch 时抬块内 warp 数（见 `dplr_kaw`）——单流只有 H=40 个块。
        let kaw = Self::dplr_kaw(1, n);
        let key = format!("{key}_w{kaw}");
        let src = format!("#define KAW {kaw}\n{src}");
        let func = self.kernel(&key, &src, "fuse_ka_dplr_norm")?;
        // 每个 block 处理一个 (head,batch)；block = KAW 个 warp。
        let grid = (h as u32, 1u32, 1u32);
        let block = ((kaw * 32) as u32, 1u32, 1u32);
        let h_i = h as i32;
        let n_i = n as i32;
        let params = [
            &sd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &kkd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &kad as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &rkd as *const u64 as *mut c_void,
            &kmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &ynd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
            &gn_eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_rkv_stage1(
        &mut self,
        r: TensorId,
        k: TensorId,
        v: TensorId,
        v1: TensorId,
        w1: TensorId,
        a1: TensorId,
        g1: TensorId,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        xw: TensorId,
        xa: TensorId,
        xg: TensorId,
        or: TensorId,
        ok: TensorId,
        ov: TensorId,
        ovm: TensorId,
        owm: TensorId,
        oam: TensorId,
        ogm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        // r/k/v 与 out_v 为 fp16；x 与其余 out 为 f32；v1/w1/a1/g1 权重为 f32。
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let rd = f16(r, "gemv_rkv_stage1")?;
        let kd = f16(k, "gemv_rkv_stage1")?;
        let vd = f16(v, "gemv_rkv_stage1")?;
        let v1d = f32(v1, "gemv_rkv_stage1")?;
        let w1d = f32(w1, "gemv_rkv_stage1")?;
        let a1d = f32(a1, "gemv_rkv_stage1")?;
        let g1d = f32(g1, "gemv_rkv_stage1")?;
        let xrd = f32(xr, "gemv_rkv_stage1")?;
        let xkd = f32(xk, "gemv_rkv_stage1")?;
        let xvd = f32(xv, "gemv_rkv_stage1")?;
        let xwd = f32(xw, "gemv_rkv_stage1")?;
        let xad = f32(xa, "gemv_rkv_stage1")?;
        let xgd = f32(xg, "gemv_rkv_stage1")?;
        let ord = f32(or, "gemv_rkv_stage1")?;
        let okd = f32(ok, "gemv_rkv_stage1")?;
        let ovd = f16(ov, "gemv_rkv_stage1")?;
        let ovmd = f32(ovm, "gemv_rkv_stage1")?;
        let owmd = f32(owm, "gemv_rkv_stage1")?;
        let oamd = f32(oam, "gemv_rkv_stage1")?;
        let ogmd = f32(ogm, "gemv_rkv_stage1")?;

        let func = self.kernel("gemv_rkv_stage1", GEMV_RKV_STAGE1_SRC, "gemv_rkv_stage1")?;
        // dispatch (C/ROWS + VM + WM + AM + GM, 1, 1)，block=128。
        let grid = ((c / 4 + vm + wm + am + gm) as u32, 1u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let vm_i = vm as i32;
        let wm_i = wm as i32;
        let am_i = am as i32;
        let gm_i = gm as i32;
        let params = [
            &rd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &v1d as *const u64 as *mut c_void,
            &w1d as *const u64 as *mut c_void,
            &a1d as *const u64 as *mut c_void,
            &g1d as *const u64 as *mut c_void,
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &xwd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xgd as *const u64 as *mut c_void,
            &ord as *const u64 as *mut c_void,
            &okd as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ovmd as *const u64 as *mut c_void,
            &owmd as *const u64 as *mut c_void,
            &oamd as *const u64 as *mut c_void,
            &ogmd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &vm_i as *const i32 as *mut c_void,
            &wm_i as *const i32 as *mut c_void,
            &am_i as *const i32 as *mut c_void,
            &gm_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_int8_rkv_stage1(
        &mut self,
        r: &Int8Handle,
        k: &Int8Handle,
        v: &Int8Handle,
        v1: TensorId,
        w1: TensorId,
        a1: TensorId,
        g1: TensorId,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        xw: TensorId,
        xa: TensorId,
        xg: TensorId,
        or: TensorId,
        ok: TensorId,
        ov: TensorId,
        ovm: TensorId,
        owm: TensorId,
        oam: TensorId,
        ogm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        // int8 句柄：idx/sz 均为 u32；mid 权重与 x 为 f32；out_v 为 f16。
        let u32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::U32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be u32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let r_idx = u32(r.idx, "gemv_int8_rkv_stage1")?;
        let r_sz = u32(r.sz, "gemv_int8_rkv_stage1")?;
        let k_idx = u32(k.idx, "gemv_int8_rkv_stage1")?;
        let k_sz = u32(k.sz, "gemv_int8_rkv_stage1")?;
        let v_idx = u32(v.idx, "gemv_int8_rkv_stage1")?;
        let v_sz = u32(v.sz, "gemv_int8_rkv_stage1")?;
        let v1d = f32(v1, "gemv_int8_rkv_stage1")?;
        let w1d = f32(w1, "gemv_int8_rkv_stage1")?;
        let a1d = f32(a1, "gemv_int8_rkv_stage1")?;
        let g1d = f32(g1, "gemv_int8_rkv_stage1")?;
        let xrd = f32(xr, "gemv_int8_rkv_stage1")?;
        let xkd = f32(xk, "gemv_int8_rkv_stage1")?;
        let xvd = f32(xv, "gemv_int8_rkv_stage1")?;
        let xwd = f32(xw, "gemv_int8_rkv_stage1")?;
        let xad = f32(xa, "gemv_int8_rkv_stage1")?;
        let xgd = f32(xg, "gemv_int8_rkv_stage1")?;
        let ord = f32(or, "gemv_int8_rkv_stage1")?;
        let okd = f32(ok, "gemv_int8_rkv_stage1")?;
        let ovd = f16(ov, "gemv_int8_rkv_stage1")?;
        let ovmd = f32(ovm, "gemv_int8_rkv_stage1")?;
        let owmd = f32(owm, "gemv_int8_rkv_stage1")?;
        let oamd = f32(oam, "gemv_int8_rkv_stage1")?;
        let ogmd = f32(ogm, "gemv_int8_rkv_stage1")?;

        // ★ 2026-09-23：r/k/v 每 block 行数可调（见 `RKV_ROWS`，**默认 2**）。
        // 同会话 B=1 交错 A/B（各 2 轮）：`ROWS=1` 87.4/87.3 · **`ROWS=2` 94.9/94.5** ·
        // `ROWS=4` 93.9/94.2 · `ROWS=8` 91.3/91.0 ⇒ **2 最优**（1 时每块只 1 行、
        // 归约/载入摊不开；8 时累加器与 smem 吃满）。
        // `RKV_ROWS` 进 key ⇒ 各自编译、可 A/B；`C % ROWS != 0` 则回落 2。
        let rows = std::env::var("RKV_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|r| matches!(*r, 1 | 2 | 4 | 8) && c.is_multiple_of(*r))
            .unwrap_or(2);
        let key = format!("gemv_int8_rkv_stage1_r{rows}");
        let src = format!("#define RKV_ROWS {rows}\n{GEMV_INT8_RKV_STAGE1_SRC}");
        let func = self.kernel(&key, &src, "gemv_int8_rkv_stage1")?;
        // mid 分支为 warp-per-row（4 行/块）；r/k/v 段每块 `rows` 行。
        let midtot = (vm + wm + am + gm) as u32;
        let grid = ((c / rows) as u32 + midtot.div_ceil(4), 1u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let vm_i = vm as i32;
        let wm_i = wm as i32;
        let am_i = am as i32;
        let gm_i = gm as i32;
        let params = [
            &r_idx as *const u64 as *mut c_void,
            &r_sz as *const u64 as *mut c_void,
            &k_idx as *const u64 as *mut c_void,
            &k_sz as *const u64 as *mut c_void,
            &v_idx as *const u64 as *mut c_void,
            &v_sz as *const u64 as *mut c_void,
            &v1d as *const u64 as *mut c_void,
            &w1d as *const u64 as *mut c_void,
            &a1d as *const u64 as *mut c_void,
            &g1d as *const u64 as *mut c_void,
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &xwd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xgd as *const u64 as *mut c_void,
            &ord as *const u64 as *mut c_void,
            &okd as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ovmd as *const u64 as *mut c_void,
            &owmd as *const u64 as *mut c_void,
            &oamd as *const u64 as *mut c_void,
            &ogmd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &vm_i as *const i32 as *mut c_void,
            &wm_i as *const i32 as *mut c_void,
            &am_i as *const i32 as *mut c_void,
            &gm_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_lowrank_chain4(
        &mut self,
        w2: TensorId,
        a2: TensorId,
        v2: TensorId,
        g2: TensorId,
        wm: TensorId,
        am: TensorId,
        vm: TensorId,
        gm: TensorId,
        w0: TensorId,
        a0: TensorId,
        v0: TensorId,
        scale: TensorId,
        vf: TensorId,
        ow: TensorId,
        oa: TensorId,
        ov: TensorId,
        og: TensorId,
        m: usize,
        kw: usize,
        ka: usize,
        kv: usize,
        kg: usize,
    ) -> R<()> {
        // 权重、mid、bias、scale 为 f32；v_first 与 4 个输出为 f16。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let w2d = f32(w2, "gemv_lowrank_chain4")?;
        let a2d = f32(a2, "gemv_lowrank_chain4")?;
        let v2d = f32(v2, "gemv_lowrank_chain4")?;
        let g2d = f32(g2, "gemv_lowrank_chain4")?;
        let wmd = f32(wm, "gemv_lowrank_chain4")?;
        let amd = f32(am, "gemv_lowrank_chain4")?;
        let vmd = f32(vm, "gemv_lowrank_chain4")?;
        let gmd = f32(gm, "gemv_lowrank_chain4")?;
        let w0d = f32(w0, "gemv_lowrank_chain4")?;
        let a0d = f32(a0, "gemv_lowrank_chain4")?;
        let v0d = f32(v0, "gemv_lowrank_chain4")?;
        let scaled = f32(scale, "gemv_lowrank_chain4")?;
        let vfd = f16(vf, "gemv_lowrank_chain4")?;
        let owd = f16(ow, "gemv_lowrank_chain4")?;
        let oad = f16(oa, "gemv_lowrank_chain4")?;
        let ovd = f16(ov, "gemv_lowrank_chain4")?;
        let ogd = f16(og, "gemv_lowrank_chain4")?;

        let func = self.kernel(
            "gemv_lowrank_chain4",
            GEMV_LOWRANK_CHAIN4_SRC,
            "gemv_lowrank_chain4",
        )?;
        // ★ 2026-09-23：warp-per-row ⇒ grid = ceil(M/8)（每 block 8 个 warp、每 warp 一行）。
        let grid = (m.div_ceil(8) as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let m_i = m as i32;
        let kw_i = kw as i32;
        let ka_i = ka as i32;
        let kv_i = kv as i32;
        let kg_i = kg as i32;
        let params = [
            &w2d as *const u64 as *mut c_void,
            &a2d as *const u64 as *mut c_void,
            &v2d as *const u64 as *mut c_void,
            &g2d as *const u64 as *mut c_void,
            &wmd as *const u64 as *mut c_void,
            &amd as *const u64 as *mut c_void,
            &vmd as *const u64 as *mut c_void,
            &gmd as *const u64 as *mut c_void,
            &w0d as *const u64 as *mut c_void,
            &a0d as *const u64 as *mut c_void,
            &v0d as *const u64 as *mut c_void,
            &scaled as *const u64 as *mut c_void,
            &vfd as *const u64 as *mut c_void,
            &owd as *const u64 as *mut c_void,
            &oad as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ogd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &kw_i as *const i32 as *mut c_void,
            &ka_i as *const i32 as *mut c_void,
            &kv_i as *const i32 as *mut c_void,
            &kg_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_f16_relu2(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let af16 = self.f16_ptr(a, "gemv_f16_relu2")?;
        let xd = self.f32_ptr(x, "gemv_f16_relu2")?;
        let yd = self.f32_ptr(y, "gemv_f16_relu2")?;
        self.gemv_variant_dispatch(af16, 0, 0, 0, xd, 0, yd, m, k, b, 0, 0)
    }
    fn gemv_int8_relu2(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_relu2")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_relu2")?;
        let xd = self.f32_ptr(x, "gemv_int8_relu2")?;
        let yd = self.f32_ptr(y, "gemv_int8_relu2")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, 0, yd, m, k, b, 2, 0)
    }
    fn gemv_f16_mul_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let af16 = self.f16_ptr(a, "gemv_f16_mul_add")?;
        let xd = self.f32_ptr(x, "gemv_f16_mul_add")?;
        let gd = self.f16_ptr(g, "gemv_f16_mul_add")?;
        let yd = self.f32_ptr(y, "gemv_f16_mul_add")?;
        self.gemv_variant_dispatch(af16, 0, 0, 0, xd, gd, yd, m, k, b, 0, 1)
    }
    fn gemv_int8_mul_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_mul_add")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_mul_add")?;
        let xd = self.f32_ptr(x, "gemv_int8_mul_add")?;
        let gd = self.f16_ptr(g, "gemv_int8_mul_add")?;
        let yd = self.f32_ptr(y, "gemv_int8_mul_add")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, gd, yd, m, k, b, 2, 1)
    }
    fn gemv_f16_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let af16 = self.f16_ptr(a, "gemv_f16_add")?;
        let xd = self.f32_ptr(x, "gemv_f16_add")?;
        let yd = self.f32_ptr(y, "gemv_f16_add")?;
        self.gemv_variant_dispatch(af16, 0, 0, 0, xd, 0, yd, m, k, b, 0, 2)
    }
    fn gemv_int8_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_add")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_add")?;
        let xd = self.f32_ptr(x, "gemv_int8_add")?;
        let yd = self.f32_ptr(y, "gemv_int8_add")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, 0, yd, m, k, b, 2, 2)
    }
    /// y = x @ A（int8 量化权重，f32 输出，覆盖写）——head 用。
    fn gemv_int8_plain(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_plain")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_plain")?;
        let xd = self.f32_ptr(x, "gemv_int8_plain")?;
        let yd = self.f32_ptr(y, "gemv_int8_plain")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, 0, yd, m, k, b, 2, 3)
    }

    fn ffn_value_sparse_add(
        &mut self,
        value_w16: Option<TensorId>,
        value_tiled: TensorId,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
    ) -> R<()> {
        let _ = value_w16; // CudaBackend 走稀疏内核，稠密权重仅作回退占位
        let vt = self.f16_ptr(value_tiled, "ffn_value_sparse_add")?;
        let rd = self.f32_ptr(r2, "ffn_value_sparse_add")?;
        let xd = self.f32_ptr(x, "ffn_value_sparse_add")?;
        let func = self.kernel(
            "ffn_value_sparse_add",
            FFN_VALUE_SPARSE_SRC,
            "ffn_value_sparse_add",
        )?;
        let grid = ((fh / 128) as u32, (c / 256) as u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let fh_i = fh as i32;
        let params = [
            &rd as *const u64 as *mut c_void,
            &vt as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &fh_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    fn supports_sparse_ffn(&self) -> bool {
        true
    }

    /// `ffn_value_imma_batch` 的形状门控：`imma_gemm_batch` 要求
    /// `k = fh` 是量化分组 128 的倍数，且 m = c 是 mma 行块 8 的倍数。
    fn supports_ffn_value_imma(&self, c: usize, fh: usize, batch: usize) -> bool {
        batch >= imma_min_batch()
            && fh.is_multiple_of(QUANT_X_I8_GROUP)
            && c.is_multiple_of(8)
            && env_on("FFN_VALUE_IMMA")
    }

    fn argmax(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()> {
        let logits_d = self.f32_ptr(logits, "argmax")?;
        let token_d = self.f32_ptr(token, "argmax")?;
        let func = self.kernel("rwkv_argmax", ARGMAX_SRC, "rwkv_argmax")?;
        let grid = (1u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let n_i = n as i32;
        let params = [
            &logits_d as *const u64 as *mut c_void,
            &token_d as *const u64 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn sample(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        history: &[u32],
    ) -> R<()> {
        // 自建采样临时缓冲（对齐 Vulkan/CUDA 统一 kernel 语义）。
        let temp = self.create_tensor(n, TensorDtype::F32)?;
        let mask = self.create_tensor(n, TensorDtype::F32)?;
        let counter = self.create_tensor(n, TensorDtype::U32)?;
        let sampler = self.create_tensor(10, TensorDtype::F32)?;
        self.store_sampler_host(
            sampler,
            temperature,
            top_k,
            top_p,
            seed,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            history.len() as u32,
        )?;
        // 历史 token 缓冲（hist_len=0 时 kernel 跳过直方图，缓冲仅需合法地址）
        let hist = self.create_tensor(history.len().max(1), TensorDtype::U32)?;
        if !history.is_empty() {
            self.upload_u32(hist, history)?;
        }
        self.sample_into_host_seeded(logits, token, n, temp, mask, counter, sampler, hist)
    }
    fn clear_cache(&mut self) {
        // CUDA kernel 与缓冲地址/形状无关（地址全部为启动参数，PTX 由静态源码
        // 编译），跨 T 变化可安全复用，无需清空 kernels 重编译。
        // 仅销毁 prefill graph：graph 内 bake 了缓冲指针，T 变化重建 seq 缓冲后失效。
        for (_, exec) in self.prefill_graphs.drain() {
            unsafe {
                (self.drv.cu_graph_exec_destroy)(exec);
            }
        }
    }
    fn drop_host(&mut self, _t: TensorId) {
        // CUDA 后端张量仅持有设备指针，无 host 镜像缓冲，无需释放。
    }

    fn free_tensor(&mut self, t: TensorId) {
        // 移除注册表条目并释放设备内存（防 seq 缓冲按 T 重建时泄漏）。
        if let Some(tensor) = self.tensors.remove(&t) {
            self.lens.remove(&t);
            let dptr = match tensor {
                CudaTensor::F32 { dptr, .. }
                | CudaTensor::F16 { dptr, .. }
                | CudaTensor::U32 { dptr, .. } => dptr,
            };
            unsafe {
                (self.drv.cu_mem_free_v2)(dptr);
            }
        }
    }
    fn copy_device(&mut self, src: TensorId, dst: TensorId) -> R<()> {
        let src_d = self.f32_ptr(src, "copy_device")?;
        let dst_d = self.f32_ptr(dst, "copy_device")?;
        let len = *self.lens.get(&src).ok_or("copy_device: unknown src len")?;
        let func = self.kernel("copy_device", COPY_DEVICE_SRC, "rwkv_copy_device")?;
        let grid = ((len as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let len_i = len as i32;
        let params = [
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &len_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn copy_range(
        &mut self,
        src: TensorId,
        src_off: usize,
        dst: TensorId,
        dst_off: usize,
        len: usize,
    ) -> R<()> {
        if len == 0 {
            return Ok(());
        }
        // 支持 f32 与 f16（v_first 为 f16）：按 src dtype 选 kernel。
        let is_f16 = matches!(self.get(src, "copy_range")?, CudaTensor::F16 { .. });
        let (sf16, df16, src_d, dst_d) = if is_f16 {
            (
                self.f16_ptr(src, "copy_range")?,
                self.f16_ptr(dst, "copy_range")?,
                0u64,
                0u64,
            )
        } else {
            (
                0u64,
                0u64,
                self.f32_ptr(src, "copy_range")?,
                self.f32_ptr(dst, "copy_range")?,
            )
        };
        let src_len = *self.lens.get(&src).ok_or("copy_range: unknown src len")?;
        let dst_len = *self.lens.get(&dst).ok_or("copy_range: unknown dst len")?;
        if src_off + len > src_len || dst_off + len > dst_len {
            return Err(format!(
                "copy_range: 越界 (src {src_off}+{len}/{src_len}, dst {dst_off}+{len}/{dst_len})"
            )
            .into());
        }
        let (func, isrc, idst) = if is_f16 {
            (
                self.kernel("copy_range_f16", COPY_RANGE_F16_SRC, "rwkv_copy_range_f16")?,
                sf16,
                df16,
            )
        } else {
            (
                self.kernel("copy_range", COPY_RANGE_SRC, "rwkv_copy_range")?,
                src_d,
                dst_d,
            )
        };
        let grid = ((len as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (so, d_o, l) = (src_off as i32, dst_off as i32, len as i32);
        let params = [
            &isrc as *const u64 as *mut c_void,
            &idst as *const u64 as *mut c_void,
            &so as *const i32 as *mut c_void,
            &d_o as *const i32 as *mut c_void,
            &l as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn copy_token(
        &mut self,
        x: TensorId,
        y: TensorId,
        c: usize,
        stride: usize,
        token: usize,
    ) -> R<()> {
        let xd = self.f32_ptr(x, "copy_token")?;
        let yd = self.f32_ptr(y, "copy_token")?;
        let func = self.kernel("copy_token", COPY_TOKEN_SRC, "rwkv_copy_token")?;
        let grid = ((c as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, stride_i, token_i) = (c as i32, stride as i32, token as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &stride_i as *const i32 as *mut c_void,
            &token_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemm(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let (ad, bd, cd) = (
            self.f16_ptr(a, "gemm")?,
            self.f16_ptr(b, "gemm")?,
            self.f32_ptr(c, "gemm")?,
        );
        self.gemm_dispatch(ad, bd, None, None, cd, m, n, k, 0)
    }
    fn gemm_bias(
        &mut self,
        a: TensorId,
        b: TensorId,
        bias: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let (ad, bd, biasd, cd) = (
            self.f16_ptr(a, "gemm_bias")?,
            self.f16_ptr(b, "gemm_bias")?,
            self.f32_ptr(bias, "gemm_bias")?,
            self.f32_ptr(c, "gemm_bias")?,
        );
        self.gemm_dispatch(ad, bd, Some(biasd), None, cd, m, n, k, 1)
    }
    fn gemm_add(
        &mut self,
        a: TensorId,
        b: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let (ad, bd, xd, yd) = (
            self.f16_ptr(a, "gemm_add")?,
            self.f16_ptr(b, "gemm_add")?,
            self.f32_ptr(x, "gemm_add")?,
            self.f32_ptr(y, "gemm_add")?,
        );
        self.gemm_dispatch(ad, bd, None, Some(xd), yd, m, n, k, 2)
    }
    fn to_f16(
        &mut self,
        x: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()> {
        let xd = self.f32_ptr(x, "to_f16")?;
        let yd = self.f16_ptr(y, "to_f16")?;
        let func = self.kernel("to_f16", TO_F16_SRC, "rwkv_to_f16")?;
        let grid = (m_pad as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, xs, ys) = (c as i32, t as i32, x_stride as i32, y_stride as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &xs as *const i32 as *mut c_void,
            &ys as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn to_f16_triple(
        &mut self,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        yr: TensorId,
        yk: TensorId,
        yv: TensorId,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()> {
        let (xrd, xkd, xvd) = (
            self.f32_ptr(xr, "to_f16_triple")?,
            self.f32_ptr(xk, "to_f16_triple")?,
            self.f32_ptr(xv, "to_f16_triple")?,
        );
        let (yrd, ykd, yvd) = (
            self.f16_ptr(yr, "to_f16_triple")?,
            self.f16_ptr(yk, "to_f16_triple")?,
            self.f16_ptr(yv, "to_f16_triple")?,
        );
        let func = self.kernel("to_f16_triple", TO_F16_TRIPLE_SRC, "rwkv_to_f16_triple")?;
        let grid = (m_pad as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, xs, ys) = (c as i32, t as i32, x_stride as i32, y_stride as i32);
        let params = [
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &yrd as *const u64 as *mut c_void,
            &ykd as *const u64 as *mut c_void,
            &yvd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &xs as *const i32 as *mut c_void,
            &ys as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn dequant_int8_to_f16(&mut self, a: &Int8Handle, out: TensorId, m: usize, k: usize) -> R<()> {
        let (idx, sz) = (
            self.u32_ptr(a.idx, "dequant_int8_to_f16")?,
            self.u32_ptr(a.sz, "dequant_int8_to_f16")?,
        );
        let w = self.f16_ptr(out, "dequant_int8_to_f16")?;
        let func = self.kernel("dequant_int8", DEQUANT_INT8_SRC, "rwkv_dequant_int8")?;
        let total = m * (k / 4);
        let grid = ((total as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (m_i, k_i) = (m as i32, k as i32);
        let params = [
            &idx as *const u64 as *mut c_void,
            &sz as *const u64 as *mut c_void,
            &w as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn elementwise_sigmoid(&mut self, a: TensorId, y: TensorId, c: usize, batch: usize) -> R<()> {
        let ad = self.f32_ptr(a, "elementwise_sigmoid")?;
        let yd = self.f32_ptr(y, "elementwise_sigmoid")?;
        let func = self.kernel(
            "elementwise_sigmoid",
            ELEMENTWISE_SIGMOID_SRC,
            "rwkv_elementwise_sigmoid",
        )?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, b_i) = (c as i32, batch as i32);
        let params = [
            &ad as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn elementwise_sigmoid_inplace(&mut self, y: TensorId, c: usize, batch: usize) -> R<()> {
        // 原地 sigmoid：a 与 y 指向同一张量。
        self.elementwise_sigmoid(y, y, c, batch)
    }
    fn fuse_ka(
        &mut self,
        k: TensorId,
        kk_w: TensorId,
        a: TensorId,
        ka_w: TensorId,
        k_mod: TensorId,
        kk_l2: TensorId,
        b: TensorId,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (kd, kk_d, ad, ka_d, km_d, kl_d, bd) = (
            f32(k, "fuse_ka")?,
            f32(kk_w, "fuse_ka")?,
            f32(a, "fuse_ka")?,
            f32(ka_w, "fuse_ka")?,
            f32(k_mod, "fuse_ka")?,
            f32(kk_l2, "fuse_ka")?,
            f32(b, "fuse_ka")?,
        );
        let func = self.kernel("fuse_ka", FUSE_KA_SRC, "rwkv_fuse_ka")?;
        let grid = (h as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (h_i, n_i, b_i) = (h as i32, n as i32, batch as i32);
        let params = [
            &kd as *const u64 as *mut c_void,
            &kk_d as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &ka_d as *const u64 as *mut c_void,
            &km_d as *const u64 as *mut c_void,
            &kl_d as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn sum_rk_rk(
        &mut self,
        r: TensorId,
        k_mod: TensorId,
        r_k: TensorId,
        v: TensorId,
        y: TensorId,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (rd, km_d, rk_d, vd, yd) = (
            f32(r, "sum_rk_rk")?,
            f32(k_mod, "sum_rk_rk")?,
            f32(r_k, "sum_rk_rk")?,
            f32(v, "sum_rk_rk")?,
            f32(y, "sum_rk_rk")?,
        );
        let func = self.kernel("sum_rk_rk", SUM_RK_RK_SRC, "rwkv_sum_rk_rk")?;
        let grid = (h as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (h_i, n_i, b_i) = (h as i32, n as i32, batch as i32);
        let params = [
            &rd as *const u64 as *mut c_void,
            &km_d as *const u64 as *mut c_void,
            &rk_d as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn seq_shift(
        &mut self,
        x: TensorId,
        state: TensorId,
        tm: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> R<()> {
        let (xd, sd, tmd, yd) = (
            self.f32_ptr(x, "seq_shift")?,
            self.f32_ptr(state, "seq_shift")?,
            self.f32_ptr(tm, "seq_shift")?,
            self.f32_ptr(y, "seq_shift")?,
        );
        let func = self.kernel("seq_shift", SEQ_SHIFT_SRC, "rwkv_seq_shift")?;
        let grid = (t as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, sx, sy) = (c as i32, t as i32, stride_x as i32, stride_y as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &sd as *const u64 as *mut c_void,
            &tmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &sx as *const i32 as *mut c_void,
            &sy as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    #[allow(clippy::too_many_arguments)]
    fn seq_shift_batch(
        &mut self,
        x: TensorId,
        state: TensorId,
        tm: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        stride_x: usize,
        stride_y: usize,
        batch: usize,
    ) -> R<()> {
        let (xd, sd, tmd, yd) = (
            self.f32_ptr(x, "seq_shift_batch")?,
            self.f32_ptr(state, "seq_shift_batch")?,
            self.f32_ptr(tm, "seq_shift_batch")?,
            self.f32_ptr(y, "seq_shift_batch")?,
        );
        let func = self.kernel(
            "seq_shift_batch",
            SEQ_SHIFT_BATCH_SRC,
            "rwkv_seq_shift_batch",
        )?;
        let grid = (t as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, sx, sy) = (c as i32, t as i32, stride_x as i32, stride_y as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &sd as *const u64 as *mut c_void,
            &tmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &sx as *const i32 as *mut c_void,
            &sy as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn copy_token_batch(
        &mut self,
        x: TensorId,
        state: TensorId,
        lens: TensorId,
        c: usize,
        t: usize,
        batch: usize,
    ) -> R<()> {
        let xd = self.f32_ptr(x, "copy_token_batch")?;
        let sd = self.f32_ptr(state, "copy_token_batch")?;
        let ld = self.u32_ptr(lens, "copy_token_batch")?;
        let func = self.kernel(
            "copy_token_batch",
            COPY_TOKEN_BATCH_SRC,
            "rwkv_copy_token_batch",
        )?;
        let grid = (batch as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i) = (c as i32, t as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &sd as *const u64 as *mut c_void,
            &ld as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    #[allow(clippy::too_many_arguments)]
    fn dplr_seq_batch(
        &mut self,
        s: TensorId,
        r: TensorId,
        w: TensorId,
        k: TensorId,
        v: TensorId,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        lens: TensorId,
        h: usize,
        n: usize,
        t: usize,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (sd, rd, wd, kd, vd, ad, bd, yd) = (
            self.any_ptr(s, "dplr_seq_batch")?,
            f32(r, "dplr_seq_batch")?,
            f32(w, "dplr_seq_batch")?,
            f32(k, "dplr_seq_batch")?,
            f32(v, "dplr_seq_batch")?,
            f32(a, "dplr_seq_batch")?,
            f32(b, "dplr_seq_batch")?,
            f32(y, "dplr_seq_batch")?,
        );
        let ld = self.u32_ptr(lens, "dplr_seq_batch")?;
        let (key, src) =
            self.dplr_variant("dplr_seq_batch", DPLR_SEQ_BATCH_SRC, s, "dplr_seq_batch")?;
        let func = self.kernel(&key, &src, "rwkv_dplr_seq_batch")?;
        let blocks = h * n;
        let grid = ((blocks as u32).div_ceil(8), batch as u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let (h_i, n_i, t_i, c_i) = (h as i32, n as i32, t as i32, c as i32);
        let params = [
            &sd as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &ld as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn segmean(
        &mut self,
        x: TensorId,
        out: TensorId,
        lens: TensorId,
        c: usize,
        t_pad: usize,
        batch: usize,
    ) -> R<()> {
        let xd = self.f32_ptr(x, "segmean")?;
        let od = self.f32_ptr(out, "segmean")?;
        let ld = self.u32_ptr(lens, "segmean")?;
        let func = self.kernel("segmean", SEGMEAN_SRC, "rwkv_segmean")?;
        let grid = (batch as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i) = (c as i32, t_pad as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &od as *const u64 as *mut c_void,
            &ld as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn dplr_seq(
        &mut self,
        s: TensorId,
        r: TensorId,
        w: TensorId,
        k: TensorId,
        v: TensorId,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        h: usize,
        n: usize,
        t: usize,
        c: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (sd, rd, wd, kd, vd, ad, bd, yd) = (
            self.any_ptr(s, "dplr_seq")?,
            f32(r, "dplr_seq")?,
            f32(w, "dplr_seq")?,
            f32(k, "dplr_seq")?,
            f32(v, "dplr_seq")?,
            f32(a, "dplr_seq")?,
            f32(b, "dplr_seq")?,
            f32(y, "dplr_seq")?,
        );
        let (key, src) = self.dplr_variant("dplr_seq", DPLR_SEQ_SRC, s, "dplr_seq")?;
        let func = self.kernel(&key, &src, "rwkv_dplr_seq")?;
        // 128 线程/block 处理 8 个状态行（4 warp × 2 half-warp），grid.x = ceil(h*n/8)。
        let blocks = h * n;
        let grid = ((blocks as u32).div_ceil(8), 1u32, 1u32);
        let block = (128u32, 1u32, 1u32); // 要求 n<=64
        let (h_i, n_i, t_i, c_i) = (h as i32, n as i32, t as i32, c as i32);
        let params = [
            &sd as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemm_relu2(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let (ad, bd, cd) = (
            self.f16_ptr(a, "gemm_relu2")?,
            self.f16_ptr(b, "gemm_relu2")?,
            self.f32_ptr(c, "gemm_relu2")?,
        );
        self.gemm_dispatch(ad, bd, None, None, cd, m, n, k, 3)
    }
    fn gemm_tanh(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let (ad, bd, cd) = (
            self.f16_ptr(a, "gemm_tanh")?,
            self.f16_ptr(b, "gemm_tanh")?,
            self.f32_ptr(c, "gemm_tanh")?,
        );
        self.gemm_dispatch(ad, bd, None, None, cd, m, n, k, 4)
    }
    fn elementwise_scale_exp(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let (ad, bd, yd) = (
            self.f32_ptr(a, "elementwise_scale_exp")?,
            self.f32_ptr(b, "elementwise_scale_exp")?,
            self.f32_ptr(y, "elementwise_scale_exp")?,
        );
        let func = self.kernel(
            "elementwise_scale_exp",
            ELEMENTWISE_SCALE_EXP_SRC,
            "rwkv_elementwise_scale_exp",
        )?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, b_i) = (c as i32, batch as i32);
        let params = [
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn elementwise_mul(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let (ad, bd, yd) = (
            self.f32_ptr(a, "elementwise_mul")?,
            self.f32_ptr(b, "elementwise_mul")?,
            self.f32_ptr(y, "elementwise_mul")?,
        );
        let func = self.kernel(
            "elementwise_mul",
            ELEMENTWISE_MUL_SRC,
            "rwkv_elementwise_mul",
        )?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, b_i) = (c as i32, batch as i32);
        let params = [
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn v_first_lerp(
        &mut self,
        v: TensorId,
        gate: TensorId,
        v_first: TensorId,
        c: usize,
        t: usize,
        stride: usize,
    ) -> R<()> {
        let (vd, gd, fvd) = (
            self.f32_ptr(v, "v_first_lerp")?,
            self.f32_ptr(gate, "v_first_lerp")?,
            self.f32_ptr(v_first, "v_first_lerp")?,
        );
        let func = self.kernel("v_first_lerp", V_FIRST_LERP_SRC, "rwkv_v_first_lerp")?;
        let grid = (t as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, s_i) = (c as i32, t as i32, stride as i32);
        let params = [
            &vd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &fvd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &s_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_seq(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        x_stride: usize,
        y_stride: usize,
        batch: usize,
    ) -> R<()> {
        let (ad, xd, yd) = (
            self.f32_ptr(a, "gemv_seq")?,
            self.f32_ptr(x, "gemv_seq")?,
            self.f32_ptr(y, "gemv_seq")?,
        );
        let func = self.kernel("gemv_seq", GEMV_SEQ_SRC, "rwkv_gemv_seq")?;
        let grid = (m as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (m_i, k_i, xs, ys, b_i) = (
            m as i32,
            k as i32,
            x_stride as i32,
            y_stride as i32,
            batch as i32,
        );
        let params = [
            &ad as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &xs as *const i32 as *mut c_void,
            &ys as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn store_sampler_host(
        &self,
        sampler: TensorId,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
    ) -> R<()> {
        let data = [
            temperature,
            f32::from_bits(top_k),
            top_p,
            f32::from_bits(seed),
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            f32::from_bits(hist_len),
            1.0, // penalty_decay：单流采样恒为线性（衰减仅批量 API 支持）
            0.0, // 保留（index 9，对齐 40 字节行）
        ];
        self.upload(sampler, &data)
    }
    fn sample_into_host_seeded(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temp: TensorId,
        mask: TensorId,
        counter: TensorId,
        sampler: TensorId,
        hist: TensorId,
    ) -> R<()> {
        let logits_d = self.f32_ptr(logits, "sample_into_host_seeded")?;
        let token_d = self.f32_ptr(token, "sample_into_host_seeded")?;
        let temp_d = self.f32_ptr(temp, "sample_into_host_seeded")?;
        let mask_d = self.f32_ptr(mask, "sample_into_host_seeded")?;
        let counter_d = self.u32_ptr(counter, "sample_into_host_seeded")?;
        let sampler_d = self.f32_ptr(sampler, "sample_into_host_seeded")?;
        // hist 可能是 U32（sample 自建）或 F32（self-loop 的 token_seq，位模式存索引）。
        let hist_d = match self.get(hist, "sample_into_host_seeded")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("sample_into_host_seeded: hist must be u32 or f32".into()),
        };
        // ★ 2026-09-23：扫描趟展开倍数（`SAMP_U`，默认 4 = 旧行为）。
        // 单流采样器是 `grid=(1,1,1)`、一个块 112 线程独占一个 SM ⇒ 每轮只有 1 个载入
        // 在飞、~600 cycle 延迟全暴露；把倍数抬到 8/16 可让 U 个载入并行。
        // 倍数进 kernel key（不同 U 各自编译），便于同会话交错 A/B。
        let samp_u = env_tile("SAMP_U", 4).clamp(1, 32);
        let src = format!("#define SAMP_U {samp_u}\n{SAMPLE_SRC}");
        let key = format!("rwkv_sample_u{samp_u}");
        let func = self.kernel(&key, &src, "rwkv_sample")?;
        let grid = (1u32, 1u32, 1u32);
        // 注意：kernel 内共享内存数组按 BS=112 定义（s_val/s_idx），block 必须与 BS 一致，
        // 否则 tid>=112 的线程访问 s_val[tid] 越界 → 非法内存访问（sticky error 700）。
        let block = (112u32, 1u32, 1u32);
        let n_i = n as i32;
        let params = [
            &logits_d as *const u64 as *mut c_void,
            &token_d as *const u64 as *mut c_void,
            &temp_d as *const u64 as *mut c_void,
            &mask_d as *const u64 as *mut c_void,
            &counter_d as *const u64 as *mut c_void,
            &sampler_d as *const u64 as *mut c_void,
            &hist_d as *const u64 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn record_token(&mut self, in_tok: TensorId, out_seq: TensorId, cnt: TensorId) -> R<()> {
        // in_tok/out_seq/cnt 存 token 位模式，以 u32 读取（F32/U32 均可，位模式相同）。
        let ptr = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::U32 { dptr, .. } => Ok(dptr),
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be u32 or f32").into()),
            }
        };
        let in_tok_d = ptr(in_tok, "record_token")?;
        let out_d = ptr(out_seq, "record_token")?;
        let cnt_d = ptr(cnt, "record_token")?;
        let func = self.kernel("rwkv_record_token", RECORD_TOKEN_SRC, "rwkv_record_token")?;
        let grid = (1u32, 1u32, 1u32);
        let block = (1u32, 1u32, 1u32);
        let params = [
            &in_tok_d as *const u64 as *mut c_void,
            &out_d as *const u64 as *mut c_void,
            &cnt_d as *const u64 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn argmax_into_host(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()> {
        // CUDA 后端 token 即为设备 F32 缓冲，与 argmax 一致（语义相同，写位模式）。
        self.argmax(logits, token, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试辅助：创建张量（避免闭包捕获 `&mut b` 造成借用冲突）。
    fn mk_tensor(b: &mut CudaBackend, len: usize, dtype: TensorDtype) -> TensorId {
        b.create_tensor(len, dtype).expect("create tensor")
    }

    /// 探测 CUDA（驱动可加载 + ≥1 设备）。无 CUDA 环境下跳过。
    #[test]
    fn detect_cuda_available() {
        // 仅验证探测函数可调用且返回 bool；具体真值取决于本机硬件。
        let _ = cuda_available();
    }

    /// 骨架张量管理：create/upload/download 往返（f32 与 f16）。
    /// 无 CUDA 设备时跳过（cuda_available() 为 false）。
    #[test]
    fn tensor_upload_download_roundtrip() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping roundtrip test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        // f32 往返
        let n = 256usize;
        let t = b.create_tensor(n, TensorDtype::F32).expect("create f32");
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
        b.upload(t, &data).unwrap();
        let got = b.download(t).unwrap();
        let mut max_diff = 0.0f32;
        for (a, g) in data.iter().zip(got.iter()) {
            max_diff = max_diff.max((a - g).abs());
        }
        assert!(
            max_diff == 0.0,
            "f32 roundtrip mismatch, max_diff={max_diff}"
        );

        // f16 往返（经 half 转换，允许舍入误差）
        let t16 = b.create_tensor(n, TensorDtype::F16).expect("create f16");
        b.upload(t16, &data).unwrap();
        let got16 = b.download(t16).unwrap();
        let mut max_diff16 = 0.0f32;
        for (a, g) in data.iter().zip(got16.iter()) {
            max_diff16 = max_diff16.max((a - g).abs());
        }
        assert!(
            max_diff16 < 1e-2,
            "f16 roundtrip mismatch, max_diff={max_diff16}"
        );
        log::info!(
            "CUDA tensor upload/download roundtrip OK (f32 max_diff={max_diff}, f16 max_diff={max_diff16})"
        );
    }

    /// gemv_f16 与 CPU 参考对比：y[m] = Σ_k x[k]·A[m·K+k]（fp16 权重，f32 累加）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let m = 8usize;
        let k = 256usize;
        let batch = 2usize;

        // 随机权重（NaiveXorshift，避免依赖外部 rand 种子）。
        let mut seed = 0x9E3779B9u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let a: Vec<f32> = (0..m * k).map(|_| rng()).collect();
        let x: Vec<f32> = (0..k * batch).map(|_| rng()).collect();

        // CPU 参考（fp16 权重量化后计算，与 GPU 一致）。
        let mut expect = vec![0.0f32; m * batch];
        for bb in 0..batch {
            for mm in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let w = f16::from_f32(a[mm * k + kk]).to_f32();
                    acc += w * x[bb * k + kk];
                }
                expect[bb * m + mm] = acc;
            }
        }

        let w = b.create_tensor(m * k, TensorDtype::F16).expect("create w");
        let xt = b
            .create_tensor(k * batch, TensorDtype::F32)
            .expect("create x");
        let yt = b
            .create_tensor(m * batch, TensorDtype::F32)
            .expect("create y");
        b.upload(w, &a).unwrap();
        b.upload(xt, &x).unwrap();
        b.gemv_f16(w, xt, yt, m, k, batch).expect("gemv_f16");
        // 同步：cuMemcpyDtoH 隐式同步，kernel 已完成。
        let got = b.download(yt).unwrap();

        let mut max_diff = 0.0f32;
        for (e, g) in expect.iter().zip(got.iter()) {
            max_diff = max_diff.max((e - g).abs());
        }
        assert!(
            max_diff < 1e-2,
            "gemv_f16 mismatch, max_diff={max_diff}\nexpect={expect:?}\ngot={got:?}"
        );
        log::info!("gemv_f16 vs CPU reference OK (max_diff={max_diff})");
    }

    /// norm_lerp6 与 CPU 参考对比：ln1 = LN(x) + 6 次 lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn norm_lerp6_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping norm_lerp6 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let eps = 1e-5f32;

        let mut seed = 0x9E3779B9u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let xd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let coeffs: Vec<Vec<f32>> = (0..6).map(|_| (0..c).map(|_| rng()).collect()).collect();

        // CPU 参考
        let mean: f32 = xd.iter().sum::<f32>() / c as f32;
        let var: f32 = xd.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let ln1: Vec<f32> = (0..c)
            .map(|i| (xd[i] - mean) * inv_std * gd[i] + bd[i])
            .collect();
        let mut expect_o = vec![Vec::new(); 6];
        let mut expect_s = vec![0.0f32; c];
        for i in 0..c {
            for j in 0..6 {
                expect_o[j].push(ln1[i] + coeffs[j][i] * (sd[i] - ln1[i]));
            }
            expect_s[i] = ln1[i];
        }

        let x = mk_tensor(&mut b, c, TensorDtype::F32);
        let s = mk_tensor(&mut b, c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, c, TensorDtype::F32);
        let ct: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, c, TensorDtype::F32))
            .collect();
        let outs: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, c, TensorDtype::F32))
            .collect();
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        for j in 0..6 {
            b.upload(ct[j], &coeffs[j]).unwrap();
        }
        b.norm_lerp6(
            x, s, g, bb, ct[0], ct[1], ct[2], ct[3], ct[4], ct[5], outs[0], outs[1], outs[2],
            outs[3], outs[4], outs[5], c, eps,
        )
        .expect("norm_lerp6");
        let got_s = b.download(s).unwrap();
        for j in 0..6 {
            let got = b.download(outs[j]).unwrap();
            let mut diff = 0.0f32;
            for (e, gr) in expect_o[j].iter().zip(got.iter()) {
                diff = diff.max((e - gr).abs());
            }
            assert!(diff < 1e-3, "norm_lerp6 out[{j}] mismatch, max_diff={diff}");
        }
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(
            s_diff < 1e-3,
            "norm_lerp6 state mismatch, max_diff={s_diff}"
        );
        log::info!("norm_lerp6 vs CPU reference OK (out/state max_diff<1e-3)");
    }

    /// cmix_norm_lerp 与 CPU 参考对比：ln2 + lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn cmix_norm_lerp_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping cmix_norm_lerp test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let eps = 1e-5f32;

        let mut seed = 0x12345678u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let xd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let cd: Vec<f32> = (0..c).map(|_| rng()).collect();

        let mean: f32 = xd.iter().sum::<f32>() / c as f32;
        let var: f32 = xd.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let mut expect_xb = vec![0.0f32; c];
        let mut expect_s = vec![0.0f32; c];
        for i in 0..c {
            let ln2 = (xd[i] - mean) * inv_std * gd[i] + bd[i];
            expect_xb[i] = ln2 + cd[i] * (sd[i] - ln2);
            expect_s[i] = ln2;
        }

        let x = mk_tensor(&mut b, c, TensorDtype::F32);
        let s = mk_tensor(&mut b, c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, c, TensorDtype::F32);
        let co = mk_tensor(&mut b, c, TensorDtype::F32);
        let o = mk_tensor(&mut b, c, TensorDtype::F32);
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        b.upload(co, &cd).unwrap();
        b.cmix_norm_lerp(x, s, g, bb, co, o, c, eps)
            .expect("cmix_norm_lerp");
        let got_xb = b.download(o).unwrap();
        let got_s = b.download(s).unwrap();
        let mut xb_diff = 0.0f32;
        for (e, gr) in expect_xb.iter().zip(got_xb.iter()) {
            xb_diff = xb_diff.max((e - gr).abs());
        }
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(xb_diff < 1e-3, "cmix xb mismatch, max_diff={xb_diff}");
        assert!(s_diff < 1e-3, "cmix state mismatch, max_diff={s_diff}");
        log::info!("cmix_norm_lerp vs CPU reference OK (xb/state max_diff<1e-3)");
    }

    /// norm 与 CPU 参考对比：y = LN(x) * gamma + beta，逐 (head,batch) 行，跨 batch 共享 gamma/beta。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn norm_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping norm test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let h = 4usize;
        let batch = 3usize;
        let rows = batch * h;
        let eps = 1e-5f32;

        let mut seed = 0x0F0F0F0Fu32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // x 布局 [batch][head][c]；gamma/beta 布局 [head][c]（跨 batch 共享）。
        let xd: Vec<f32> = (0..batch * h * c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..h * c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..h * c).map(|_| rng()).collect();

        // CPU 参考
        let mut expect = vec![0.0f32; batch * h * c];
        for bb in 0..batch {
            for hh in 0..h {
                let x_base = bb * h * c + hh * c;
                let g_base = hh * c;
                let mean: f32 = (0..c).map(|i| xd[x_base + i]).sum::<f32>() / c as f32;
                let var: f32 = (0..c)
                    .map(|i| {
                        let v = xd[x_base + i] - mean;
                        v * v
                    })
                    .sum::<f32>()
                    / c as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for i in 0..c {
                    expect[x_base + i] =
                        (xd[x_base + i] - mean) * inv_std * gd[g_base + i] + bd[g_base + i];
                }
            }
        }

        let x = mk_tensor(&mut b, batch * h * c, TensorDtype::F32);
        let g = mk_tensor(&mut b, h * c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, h * c, TensorDtype::F32);
        let y = mk_tensor(&mut b, batch * h * c, TensorDtype::F32);
        b.upload(x, &xd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        b.norm(x, g, bb, y, c, h, eps, rows).expect("norm");
        let got = b.download(y).unwrap();
        let mut max_diff = 0.0f32;
        for (e, gr) in expect.iter().zip(got.iter()) {
            max_diff = max_diff.max((e - gr).abs());
        }
        assert!(max_diff < 1e-3, "norm mismatch, max_diff={max_diff}");
        log::info!("norm vs CPU reference OK (max_diff<1e-3)");
    }

    /// xorshift rng（batch 测试共享）。
    fn test_rng(seed: u32) -> impl FnMut() -> f32 {
        let mut seed = seed;
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// norm_lerp6_batch 与 CPU 参考对比：B slot 各自独立归一化 + lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn norm_lerp6_batch_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping norm_lerp6_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let batch = 4usize;
        let eps = 1e-5f32;
        let mut rng = test_rng(0xABCDEF01);

        let xd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        // lerp 系数为共享权重 [C]（跨 slot 共享，与线上 x_r..x_g 一致）。
        let coeffs: Vec<Vec<f32>> = (0..6).map(|_| (0..c).map(|_| rng()).collect()).collect();

        // CPU 参考（每 slot 独立归一化，系数共享）
        let mut expect_o = vec![vec![0.0f32; batch * c]; 6];
        let mut expect_s = vec![0.0f32; batch * c];
        for bi in 0..batch {
            let base = bi * c;
            let mean: f32 = (0..c).map(|i| xd[base + i]).sum::<f32>() / c as f32;
            let var: f32 = (0..c)
                .map(|i| {
                    let v = xd[base + i] - mean;
                    v * v
                })
                .sum::<f32>()
                / c as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..c {
                let ln1 = (xd[base + i] - mean) * inv_std * gd[i] + bd[i];
                for j in 0..6 {
                    expect_o[j][base + i] = ln1 + coeffs[j][i] * (sd[base + i] - ln1);
                }
                expect_s[base + i] = ln1;
            }
        }

        let x = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let s = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let beta = mk_tensor(&mut b, c, TensorDtype::F32);
        let ct: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, c, TensorDtype::F32))
            .collect();
        let outs: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, batch * c, TensorDtype::F32))
            .collect();
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(beta, &bd).unwrap();
        for j in 0..6 {
            b.upload(ct[j], &coeffs[j]).unwrap();
        }
        b.norm_lerp6_batch(
            x, s, g, beta, ct[0], ct[1], ct[2], ct[3], ct[4], ct[5], outs[0], outs[1], outs[2],
            outs[3], outs[4], outs[5], c, eps, batch,
        )
        .expect("norm_lerp6_batch");
        let got_s = b.download(s).unwrap();
        for j in 0..6 {
            let got = b.download(outs[j]).unwrap();
            let mut diff = 0.0f32;
            for (e, gr) in expect_o[j].iter().zip(got.iter()) {
                diff = diff.max((e - gr).abs());
            }
            assert!(
                diff < 1e-3,
                "norm_lerp6_batch out[{j}] mismatch, max_diff={diff}"
            );
        }
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(
            s_diff < 1e-3,
            "norm_lerp6_batch state mismatch, max_diff={s_diff}"
        );
        log::info!("norm_lerp6_batch vs CPU reference OK (batch={batch})");
    }

    /// cmix_norm_lerp_batch 与 CPU 参考对比：B slot 各自独立 ln2 + lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn cmix_norm_lerp_batch_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping cmix_norm_lerp_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let batch = 4usize;
        let eps = 1e-5f32;
        let mut rng = test_rng(0x12349999);

        let xd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let cd: Vec<f32> = (0..c).map(|_| rng()).collect();

        let mut expect_xb = vec![0.0f32; batch * c];
        let mut expect_s = vec![0.0f32; batch * c];
        for bi in 0..batch {
            let base = bi * c;
            let mean: f32 = (0..c).map(|i| xd[base + i]).sum::<f32>() / c as f32;
            let var: f32 = (0..c)
                .map(|i| {
                    let v = xd[base + i] - mean;
                    v * v
                })
                .sum::<f32>()
                / c as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..c {
                let ln2 = (xd[base + i] - mean) * inv_std * gd[i] + bd[i];
                expect_xb[base + i] = ln2 + cd[i] * (sd[base + i] - ln2);
                expect_s[base + i] = ln2;
            }
        }

        let x = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let s = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let beta = mk_tensor(&mut b, c, TensorDtype::F32);
        let co = mk_tensor(&mut b, c, TensorDtype::F32);
        let o = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(beta, &bd).unwrap();
        b.upload(co, &cd).unwrap();
        b.cmix_norm_lerp_batch(x, s, g, beta, co, o, c, eps, batch)
            .expect("cmix_norm_lerp_batch");
        let got_xb = b.download(o).unwrap();
        let got_s = b.download(s).unwrap();
        let mut diff = 0.0f32;
        for (e, gr) in expect_xb.iter().zip(got_xb.iter()) {
            diff = diff.max((e - gr).abs());
        }
        assert!(
            diff < 1e-3,
            "cmix_norm_lerp_batch xb mismatch, max_diff={diff}"
        );
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(
            s_diff < 1e-3,
            "cmix_norm_lerp_batch state mismatch, max_diff={s_diff}"
        );
        log::info!("cmix_norm_lerp_batch vs CPU reference OK (batch={batch})");
    }

    /// gather_rows_device_f16 与 CPU 参考对比：B slot 各自按 tok[b] 取行。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gather_rows_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gather_rows test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 256usize;
        let vocab = 1024usize;
        let batch = 4usize;
        let mut rng = test_rng(0x5A5A5A5A);

        let src: Vec<f32> = (0..vocab * c).map(|_| rng()).collect();
        let toks: Vec<u32> = vec![0, 17, 512, vocab as u32 - 1];
        // 期望值做 f32→f16→f32 round-trip（upload 时后端转 f16 存储）。
        let mut expect = Vec::with_capacity(batch * c);
        for &t in &toks {
            let t = t as usize;
            for i in 0..c {
                expect.push(src[t * c + i]);
            }
        }

        let src_t = mk_tensor(&mut b, vocab * c, TensorDtype::F16);
        let dst = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let tok = mk_tensor(&mut b, batch, TensorDtype::F32);
        b.upload(src_t, &src).unwrap();
        b.upload(
            tok,
            &toks.iter().map(|t| f32::from_bits(*t)).collect::<Vec<_>>(),
        )
        .unwrap();
        b.gather_rows_device_f16(src_t, dst, tok, c, batch)
            .expect("gather_rows");
        let got = b.download(dst).unwrap();
        let mut max_diff = 0.0f32;
        for (e, gr) in expect.iter().zip(got.iter()) {
            max_diff = max_diff.max((e - gr).abs());
        }
        assert!(max_diff < 1e-2, "gather_rows mismatch, max_diff={max_diff}");
        log::info!("gather_rows vs CPU reference OK (batch={batch})");
    }

    /// sample_into_host_seeded_batch 与 CPU argmax 一致性：temperature≈0 时退化为
    /// argmax（确定性），验证每 slot 的 top-1 选择互不干扰。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn sample_batch_matches_single() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping sample_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let n = 1024usize;
        let batch = 4usize;
        let mut rng = test_rng(0xFEED1234);

        let logits: Vec<f32> = (0..batch * n).map(|_| rng() * 10.0).collect();
        // 每 slot 的期望 argmax（CPU 参考）
        let expect: Vec<usize> = logits
            .chunks(n)
            .map(|chunk| {
                let mut best = 0usize;
                for i in 1..n {
                    if chunk[i] > chunk[best] {
                        best = i;
                    }
                }
                best
            })
            .collect();

        let logits_t = mk_tensor(&mut b, batch * n, TensorDtype::F32);
        let token_t = mk_tensor(&mut b, batch, TensorDtype::F32);
        let temp_t = mk_tensor(&mut b, batch * n, TensorDtype::F32);
        let mask_t = mk_tensor(&mut b, batch * n, TensorDtype::F32);
        let counter_t = mk_tensor(&mut b, batch * n, TensorDtype::U32);
        let sampler_t = mk_tensor(&mut b, batch * 10, TensorDtype::F32);
        let hist_t = mk_tensor(&mut b, batch, TensorDtype::U32);
        b.upload(logits_t, &logits).unwrap();

        // sampler 参数：temperature=0.0001（≈argmax）、top_k=50、top_p=1.0、seed 逐 slot。
        let mut sampler_data = Vec::with_capacity(batch * 10);
        for bi in 0..batch {
            sampler_data.extend_from_slice(&[
                0.0001,
                f32::from_bits(50),
                1.0,
                f32::from_bits(42 + bi as u32),
                1.0,
                0.0,
                0.0,
                f32::from_bits(0u32),
                1.0, // penalty_decay
                0.0, // 保留
            ]);
        }
        b.upload(sampler_t, &sampler_data).unwrap();

        b.sample_into_host_seeded_batch(
            logits_t, token_t, n, temp_t, mask_t, counter_t, sampler_t, hist_t, batch, 1,
        )
        .expect("sample_batch");
        let got = b.download(token_t).unwrap();
        for bi in 0..batch {
            let tok = got[bi].to_bits();
            assert_eq!(
                tok, expect[bi] as u32,
                "sample_batch slot {bi}: got {tok} expect {}",
                expect[bi]
            );
        }
        log::info!("sample_batch vs CPU argmax OK (batch={batch}, temp≈0)");
    }

    /// record_tokens 与 CPU 参考对比：B slot 各自独立计数追加。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn record_tokens_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping record_tokens test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let batch = 4usize;
        let stride = 8usize;
        let rounds = 3usize;

        // 每 slot 预置计数（验证原子追加起点）。
        let init_cnt: Vec<u32> = vec![1, 0, 5, 2];
        let mut expect_seq = vec![0u32; batch * stride];
        let mut expect_cnt = init_cnt.clone();

        let tok = mk_tensor(&mut b, batch, TensorDtype::F32);
        let seq = mk_tensor(&mut b, batch * stride, TensorDtype::F32);
        let cnt = mk_tensor(&mut b, batch, TensorDtype::F32);
        // cnt 存 u32 位模式（f32 缓冲），初值必须按位写入而非数值转换。
        b.upload(
            cnt,
            &init_cnt
                .iter()
                .map(|c| f32::from_bits(*c))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        // 逐轮：round r 每 slot 追加 token = slot*100 + r。
        for r in 0..rounds {
            let toks: Vec<u32> = (0..batch).map(|bi| bi as u32 * 100 + r as u32).collect();
            b.upload(
                tok,
                &toks.iter().map(|t| f32::from_bits(*t)).collect::<Vec<_>>(),
            )
            .unwrap();
            b.record_tokens(tok, seq, cnt, stride, batch)
                .expect("record_tokens");
            for bi in 0..batch {
                let pos = expect_cnt[bi];
                expect_seq[bi * stride + pos as usize] = toks[bi];
                expect_cnt[bi] += 1;
            }
        }

        let got_seq = b.download(seq).unwrap();
        for (e, gr) in expect_seq.iter().zip(got_seq.iter()) {
            assert_eq!(e, &gr.to_bits(), "record_tokens seq mismatch");
        }
        let got_cnt = b.download(cnt).unwrap();
        for (e, gr) in expect_cnt.iter().zip(got_cnt.iter()) {
            assert_eq!(e, &gr.to_bits(), "record_tokens cnt mismatch");
        }
        log::info!("record_tokens vs CPU reference OK (batch={batch})");
    }

    /// gemv_variant_mb（batch 权重复用版）vs 逐 slot 单序列版数值一致性：
    /// int8 relu2 / mul_add / plain 三种 op，batch=6（含 BGRP 分组边界 4 的跨界）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_variant_mb_matches_single() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_variant_mb test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let m = 64usize; // M 为 4 的倍数（GEMV_ROWS）
        let k = 256usize; // K 为 128 的倍数（int8 group）
        let batch = 6usize; // 跨 BGRP=4 分组边界
        let mut rng = test_rng(0x77AA33CC);

        // int8 量化权重（scale/zero 打包进 sz；idx 打包 4×uint8）。
        let mut w_host = vec![0.0f32; m * k];
        for v in w_host.iter_mut() {
            *v = rng() * 2.0;
        }
        let (idx, sz) = {
            let mut idx = vec![0u32; m * (k / 4)];
            let mut sz = vec![0u32; m * (k / 128)];
            for row in 0..m {
                for g in 0..k / 128 {
                    let scale = 0.01f32;
                    let zero = 0.5f32;
                    let s16 = half::f16::from_f32(scale).to_bits() as u32;
                    let z16 = half::f16::from_f32(zero).to_bits() as u32;
                    sz[row * (k / 128) + g] = (s16 & 0xFFFF) | (z16 << 16);
                    for j in 0..k / 4 {
                        let mut packed = 0u32;
                        for q in 0..4 {
                            let wv = w_host[row * k + j * 4 + q];
                            let qv = ((wv - zero) / scale).round().clamp(-128.0, 127.0) as i32;
                            let qv = (qv as u32) & 0xFF;
                            packed |= qv << (8 * q);
                        }
                        idx[row * (k / 4) + j] = packed;
                    }
                }
            }
            (idx, sz)
        };

        // 激活 [batch, K] + 门控 fp16 + 残差初值。
        let x: Vec<f32> = (0..batch * k).map(|_| rng()).collect();
        let g: Vec<f32> = (0..batch * k).map(|_| 0.5 + 0.5 * rng()).collect();
        let y_init: Vec<f32> = (0..batch * m).map(|_| rng()).collect();

        for op in [0i32, 1i32, 3i32] {
            // 逐 slot 单序列基准（batch=1 原版 kernel，分 slot 调）。
            let mut expect = vec![0.0f32; batch * m];
            for bi in 0..batch {
                let a8 = crate::backend::Int8Handle {
                    idx: mk_tensor(&mut b, m * (k / 4), TensorDtype::U32),
                    sz: mk_tensor(&mut b, m * (k / 128), TensorDtype::U32),
                    m,
                    k,
                };
                let xt = mk_tensor(&mut b, k, TensorDtype::F32);
                let gt = mk_tensor(&mut b, k, TensorDtype::F16);
                let yt = mk_tensor(&mut b, m, TensorDtype::F32);
                b.upload_u32(a8.idx, &idx).unwrap();
                b.upload_u32(a8.sz, &sz).unwrap();
                b.upload(xt, &x[bi * k..(bi + 1) * k]).unwrap();
                b.upload(gt, &g[bi * k..(bi + 1) * k]).unwrap();
                // op==2(add) 用残差初值；其余覆盖写（mb 版同语义，覆盖即可对比）。
                b.upload(yt, &y_init[bi * m..(bi + 1) * m]).unwrap();
                match op {
                    0 => b.gemv_int8_relu2(&a8, xt, yt, m, k, 1).unwrap(),
                    1 => b.gemv_int8_mul_add(&a8, xt, gt, yt, m, k, 1).unwrap(),
                    _ => b.gemv_int8_plain(&a8, xt, yt, m, k, 1).unwrap(),
                }
                let got = b.download(yt).unwrap();
                expect[bi * m..(bi + 1) * m].copy_from_slice(&got);
                // 释放临时张量（避免注册表膨胀）。
                for t in [a8.idx, a8.sz, xt, gt, yt] {
                    b.free_tensor(t);
                }
            }

            // batch mb 版（一次算 6 slot，跨 BGRP=4 分组）。
            let a8 = crate::backend::Int8Handle {
                idx: mk_tensor(&mut b, m * (k / 4), TensorDtype::U32),
                sz: mk_tensor(&mut b, m * (k / 128), TensorDtype::U32),
                m,
                k,
            };
            let xt = mk_tensor(&mut b, batch * k, TensorDtype::F32);
            let gt = mk_tensor(&mut b, batch * k, TensorDtype::F16);
            let yt = mk_tensor(&mut b, batch * m, TensorDtype::F32);
            b.upload_u32(a8.idx, &idx).unwrap();
            b.upload_u32(a8.sz, &sz).unwrap();
            b.upload(xt, &x).unwrap();
            b.upload(gt, &g).unwrap();
            b.upload(yt, &y_init).unwrap();
            match op {
                0 => b.gemv_int8_relu2(&a8, xt, yt, m, k, batch).unwrap(),
                1 => b.gemv_int8_mul_add(&a8, xt, gt, yt, m, k, batch).unwrap(),
                _ => b.gemv_int8_plain(&a8, xt, yt, m, k, batch).unwrap(),
            }
            let got = b.download(yt).unwrap();

            // half2 累加顺序与单序列版一致（per slot 独立），容差收紧到 fp16 级。
            let mut diff = 0.0f32;
            for (e, gv) in expect.iter().zip(got.iter()) {
                diff = diff.max((e - gv).abs());
            }
            let tol = if op == 0 { 5e-2 } else { 2e-1 };
            assert!(
                diff < tol,
                "gemv_variant_mb op={op} mismatch, max_diff={diff} (batch={batch})"
            );
            log::info!("gemv_variant_mb op={op} vs single OK (batch={batch}, max_diff={diff:.5})");
            for t in [a8.idx, a8.sz, xt, gt, yt] {
                b.free_tensor(t);
            }
        }
    }

    /// gemv_int8_rkv_stage1_batch（权重复用版）vs 逐 slot 单序列版数值一致性：
    /// r/k/v + 4 mid 投影，batch=6（跨 BGRP=4 分组边界）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_int8_rkv_stage1_batch_matches_single() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping rkv_stage1_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 256usize;
        let vm = 32usize;
        let wm = 32usize;
        let am = 32usize;
        let gm = 32usize;
        let batch = 6usize;
        let mut rng = test_rng(0x1357BEEF);

        // int8 量化权重制造（与 mb 测试同一打包格式）。
        fn mk_a8(m: usize, k: usize, rng: &mut dyn FnMut() -> f32) -> (Vec<u32>, Vec<u32>) {
            let mut idx = vec![0u32; m * (k / 4)];
            let mut sz = vec![0u32; m * (k / 128)];
            for row in 0..m {
                for g in 0..k / 128 {
                    let scale = 0.01f32;
                    let zero = 0.5f32;
                    let s16 = half::f16::from_f32(scale).to_bits() as u32;
                    let z16 = half::f16::from_f32(zero).to_bits() as u32;
                    sz[row * (k / 128) + g] = (s16 & 0xFFFF) | (z16 << 16);
                    for j in 0..k / 4 {
                        let mut packed = 0u32;
                        for q in 0..4 {
                            let wv = rng() * 2.0;
                            let qv = ((wv - zero) / scale).round().clamp(-128.0, 127.0) as i32;
                            packed |= ((qv as u32) & 0xFF) << (8 * q);
                        }
                        idx[row * (k / 4) + j] = packed;
                    }
                }
            }
            (idx, sz)
        }
        let (r_idx, r_sz) = mk_a8(c, c, &mut rng);
        let (k_idx, k_sz) = mk_a8(c, c, &mut rng);
        let (v_idx, v_sz) = mk_a8(c, c, &mut rng);
        // mid 权重（fp32 [mid, C]）。
        let v1: Vec<f32> = (0..vm * c).map(|_| rng()).collect();
        let w1: Vec<f32> = (0..wm * c).map(|_| rng()).collect();
        let a1: Vec<f32> = (0..am * c).map(|_| rng()).collect();
        let g1: Vec<f32> = (0..gm * c).map(|_| rng()).collect();
        // 激活 [batch, C]。
        let xr: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xk: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xv: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xw: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xa: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xg: Vec<f32> = (0..batch * c).map(|_| rng()).collect();

        // 逐 slot 单序列基准。
        let mut expect_r = vec![0.0f32; batch * c];
        let mut expect_k = vec![0.0f32; batch * c];
        let mut expect_v = vec![0.0f32; batch * c];
        let mut expect_vm = vec![0.0f32; batch * vm];
        let mut expect_wm = vec![0.0f32; batch * wm];
        let mut expect_am = vec![0.0f32; batch * am];
        let mut expect_gm = vec![0.0f32; batch * gm];
        {
            let (r_i, r_s) = (
                mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
                mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
            );
            let (k_i, k_s) = (
                mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
                mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
            );
            let (v_i, v_s) = (
                mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
                mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
            );
            let rh = crate::backend::Int8Handle {
                idx: r_i,
                sz: r_s,
                m: c,
                k: c,
            };
            let kh = crate::backend::Int8Handle {
                idx: k_i,
                sz: k_s,
                m: c,
                k: c,
            };
            let vh = crate::backend::Int8Handle {
                idx: v_i,
                sz: v_s,
                m: c,
                k: c,
            };
            let (v1t, w1t, a1t, g1t) = (
                mk_tensor(&mut b, vm * c, TensorDtype::F32),
                mk_tensor(&mut b, wm * c, TensorDtype::F32),
                mk_tensor(&mut b, am * c, TensorDtype::F32),
                mk_tensor(&mut b, gm * c, TensorDtype::F32),
            );
            b.upload_u32(rh.idx, &r_idx).unwrap();
            b.upload_u32(rh.sz, &r_sz).unwrap();
            b.upload_u32(kh.idx, &k_idx).unwrap();
            b.upload_u32(kh.sz, &k_sz).unwrap();
            b.upload_u32(vh.idx, &v_idx).unwrap();
            b.upload_u32(vh.sz, &v_sz).unwrap();
            b.upload(v1t, &v1).unwrap();
            b.upload(w1t, &w1).unwrap();
            b.upload(a1t, &a1).unwrap();
            b.upload(g1t, &g1).unwrap();
            for bi in 0..batch {
                let (xrt, xkt, xvt, xwt, xat, xgt) = (
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                );
                let (ort, okt, ovt, ovmt, owmt, oamt, ogmt) = (
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F16),
                    mk_tensor(&mut b, vm, TensorDtype::F32),
                    mk_tensor(&mut b, wm, TensorDtype::F32),
                    mk_tensor(&mut b, am, TensorDtype::F32),
                    mk_tensor(&mut b, gm, TensorDtype::F32),
                );
                b.upload(xrt, &xr[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xkt, &xk[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xvt, &xv[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xwt, &xw[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xat, &xa[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xgt, &xg[bi * c..(bi + 1) * c]).unwrap();
                b.gemv_int8_rkv_stage1(
                    &rh, &kh, &vh, v1t, w1t, a1t, g1t, xrt, xkt, xvt, xwt, xat, xgt, ort, okt, ovt,
                    ovmt, owmt, oamt, ogmt, c, vm, wm, am, gm,
                )
                .unwrap();
                expect_r[bi * c..(bi + 1) * c].copy_from_slice(&b.download(ort).unwrap());
                expect_k[bi * c..(bi + 1) * c].copy_from_slice(&b.download(okt).unwrap());
                expect_v[bi * c..(bi + 1) * c].copy_from_slice(&b.download(ovt).unwrap());
                expect_vm[bi * vm..(bi + 1) * vm].copy_from_slice(&b.download(ovmt).unwrap());
                expect_wm[bi * wm..(bi + 1) * wm].copy_from_slice(&b.download(owmt).unwrap());
                expect_am[bi * am..(bi + 1) * am].copy_from_slice(&b.download(oamt).unwrap());
                expect_gm[bi * gm..(bi + 1) * gm].copy_from_slice(&b.download(ogmt).unwrap());
                for t in [
                    xrt, xkt, xvt, xwt, xat, xgt, ort, okt, ovt, ovmt, owmt, oamt, ogmt,
                ] {
                    b.free_tensor(t);
                }
            }
            for t in [
                rh.idx, rh.sz, kh.idx, kh.sz, vh.idx, vh.sz, v1t, w1t, a1t, g1t,
            ] {
                b.free_tensor(t);
            }
        }

        // batch mb 版。
        let (r_i, r_s) = (
            mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
            mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
        );
        let (k_i, k_s) = (
            mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
            mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
        );
        let (v_i, v_s) = (
            mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
            mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
        );
        let rh = crate::backend::Int8Handle {
            idx: r_i,
            sz: r_s,
            m: c,
            k: c,
        };
        let kh = crate::backend::Int8Handle {
            idx: k_i,
            sz: k_s,
            m: c,
            k: c,
        };
        let vh = crate::backend::Int8Handle {
            idx: v_i,
            sz: v_s,
            m: c,
            k: c,
        };
        let (v1t, w1t, a1t, g1t) = (
            mk_tensor(&mut b, vm * c, TensorDtype::F32),
            mk_tensor(&mut b, wm * c, TensorDtype::F32),
            mk_tensor(&mut b, am * c, TensorDtype::F32),
            mk_tensor(&mut b, gm * c, TensorDtype::F32),
        );
        let (xrt, xkt, xvt, xwt, xat, xgt) = (
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
        );
        let (ort, okt, ovt, ovmt, owmt, oamt, ogmt) = (
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F16),
            mk_tensor(&mut b, batch * vm, TensorDtype::F32),
            mk_tensor(&mut b, batch * wm, TensorDtype::F32),
            mk_tensor(&mut b, batch * am, TensorDtype::F32),
            mk_tensor(&mut b, batch * gm, TensorDtype::F32),
        );
        b.upload_u32(rh.idx, &r_idx).unwrap();
        b.upload_u32(rh.sz, &r_sz).unwrap();
        b.upload_u32(kh.idx, &k_idx).unwrap();
        b.upload_u32(kh.sz, &k_sz).unwrap();
        b.upload_u32(vh.idx, &v_idx).unwrap();
        b.upload_u32(vh.sz, &v_sz).unwrap();
        b.upload(v1t, &v1).unwrap();
        b.upload(w1t, &w1).unwrap();
        b.upload(a1t, &a1).unwrap();
        b.upload(g1t, &g1).unwrap();
        b.upload(xrt, &xr).unwrap();
        b.upload(xkt, &xk).unwrap();
        b.upload(xvt, &xv).unwrap();
        b.upload(xwt, &xw).unwrap();
        b.upload(xat, &xa).unwrap();
        b.upload(xgt, &xg).unwrap();
        b.gemv_int8_rkv_stage1_batch(
            &rh, &kh, &vh, v1t, w1t, a1t, g1t, xrt, xkt, xvt, xwt, xat, xgt, ort, okt, ovt, ovmt,
            owmt, oamt, ogmt, c, vm, wm, am, gm, batch,
        )
        .unwrap();

        let got_r = b.download(ort).unwrap();
        let got_k = b.download(okt).unwrap();
        let got_v = b.download(ovt).unwrap();
        let got_vm = b.download(ovmt).unwrap();
        let got_wm = b.download(owmt).unwrap();
        let got_am = b.download(oamt).unwrap();
        let got_gm = b.download(ogmt).unwrap();

        for (name, e, g, tol) in [
            ("r", &expect_r, &got_r, 2e-1f32),
            ("k", &expect_k, &got_k, 2e-1),
            ("v", &expect_v, &got_v, 2e-1),
            ("vm", &expect_vm, &got_vm, 1e-2),
            ("wm", &expect_wm, &got_wm, 1e-2),
            ("am", &expect_am, &got_am, 1e-2),
            ("gm", &expect_gm, &got_gm, 1e-2),
        ] {
            let mut diff = 0.0f32;
            for (ev, gv) in e.iter().zip(g.iter()) {
                diff = diff.max((ev - gv).abs());
            }
            assert!(
                diff < tol,
                "rkv_stage1_batch {name} mismatch, max_diff={diff} (batch={batch})"
            );
            log::info!("rkv_stage1_batch {name} vs single OK (max_diff={diff:.5})");
        }
    }

    /// segmean 与 CPU 参考对比：x [batch, t_pad, c] 每 slot 前 lens[b] 行均值。
    /// 含 lens < t_pad（padding 段必须不计入）与 batch=1 边界。无 CUDA 设备时跳过。
    #[test]
    fn segmean_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping segmean test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let t_pad = 8usize;
        let batch = 5usize;
        let lens: Vec<u32> = vec![8, 1, 5, 3, 8]; // 覆盖全 pad / 全长 / 中间档
        let mut rng = test_rng(0x5EED5EED);
        let x: Vec<f32> = (0..batch * t_pad * c).map(|_| rng()).collect();

        // CPU 参考（只对前 lens[b] 行求均值）。
        let mut expect = vec![vec![0.0f32; c]; batch];
        for bi in 0..batch {
            let l = lens[bi] as usize;
            for t in 0..l {
                for i in 0..c {
                    expect[bi][i] += x[(bi * t_pad + t) * c + i];
                }
            }
            for v in expect[bi].iter_mut() {
                *v /= l as f32;
            }
        }

        let xt = mk_tensor(&mut b, batch * t_pad * c, TensorDtype::F32);
        let ot = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let lt = mk_tensor(&mut b, batch, TensorDtype::U32);
        b.upload(xt, &x).unwrap();
        b.upload_u32(lt, &lens).unwrap();
        b.segmean(xt, ot, lt, c, t_pad, batch).unwrap();
        let got = b.download(ot).unwrap();
        let mut max_diff = 0.0f32;
        for bi in 0..batch {
            for i in 0..c {
                max_diff = max_diff.max((expect[bi][i] - got[bi * c + i]).abs());
            }
        }
        assert!(max_diff < 1e-4, "segmean mismatch, max_diff={max_diff}");
        log::info!("segmean vs CPU reference OK (batch={batch}, max_diff={max_diff:.7})");
    }

    /// fuse_ka_dplr_norm 与 CPU 参考对比：fuse_ka + dplr(S 更新) + group_norm + sum_rk_rk。
    /// 无 CUDA 设备时跳过。batch=1（单 token 路径）。
    #[test]
    fn fuse_ka_dplr_norm_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping fuse_ka_dplr_norm test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let h = 3usize;
        let n = 64usize;
        let eps = 1e-12f32;
        let gn_eps = 1e-6f32;

        let mut seed = 0xABCDEF01u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let s_size = h * n * n;
        let k_size = h * n;
        let sd: Vec<f32> = (0..s_size).map(|_| rng() * 0.1).collect();
        let kd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let kkd: Vec<f32> = (0..k_size).map(|_| 0.5 + rng()).collect();
        let ad: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let kad: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let rd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let vd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let wd: Vec<f32> = (0..k_size).map(|_| 0.5 + rng()).collect();
        let gd: Vec<f32> = (0..k_size).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let rkd: Vec<f32> = (0..k_size).map(|_| rng()).collect();

        // CPU 参考（batch=1，逐 head）
        let mut expect_s = sd.clone();
        let mut expect_kmod = vec![0.0f32; k_size];
        let mut expect_yn = vec![0.0f32; k_size];
        for head in 0..h {
            let v_b = head * n;
            let w_b = head * n;
            let s_b = head * n * n;
            // L2 范数（与 shader/CUDA 相同的全块 2 倍归约）
            let mut sq_sum = 0.0f32;
            for row in 0..n {
                let kk = kd[v_b + row] * kkd[w_b + row];
                sq_sum += kk * kk;
            }
            sq_sum *= 2.0; // 128 线程冗余归约（每行 ct=0/1 各算一次）
            let inv_norm = 1.0 / sq_sum.sqrt().max(eps);

            // Phase 1: 按列 kk_l2 / b / k_mod / w / r
            let mut sh_a = vec![0.0f32; n];
            let mut sh_b = vec![0.0f32; n];
            let mut sh_k = vec![0.0f32; n];
            let mut sh_w = vec![0.0f32; n];
            let mut sh_r = vec![0.0f32; n];
            for j in 0..n {
                let kc = kd[v_b + j];
                let kkc = kc * kkd[w_b + j];
                let ac = f16::from_f32(ad[v_b + j]).to_f32();
                let kl2 = kkc * inv_norm;
                sh_a[j] = kl2;
                sh_b[j] = -kl2 * ac;
                sh_k[j] = kc * (1.0 + kad[w_b + j] * (ac - 1.0));
                sh_w[j] = f16::from_f32(wd[v_b + j]).to_f32();
                sh_r[j] = rd[v_b + j];
            }
            expect_kmod[v_b..v_b + n].copy_from_slice(&sh_k[..n]);

            // Phase 2: sa[row] = sum_j S[row,j] * sh_a[j]
            let mut sa_val = vec![0.0f32; n];
            for row in 0..n {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += expect_s[s_b + row * n + j] * sh_a[j];
                }
                sa_val[row] = acc;
            }

            // Phase 3: S 更新 + y[row] = S@r
            let mut yv = vec![0.0f32; n];
            for row in 0..n {
                let vi = f16::from_f32(vd[v_b + row]).to_f32();
                let sv = sa_val[row];
                let mut yp = 0.0f32;
                for j in 0..n {
                    let s_ij = expect_s[s_b + row * n + j];
                    let new_s = s_ij * sh_w[j] + sv * sh_b[j] + vi * sh_k[j];
                    expect_s[s_b + row * n + j] = new_s;
                    yp += new_s * sh_r[j];
                }
                yv[row] = yp;
            }

            // Phase 4+5: group-norm(y) + s = sum(r*k_mod*r_k)
            let mean: f32 = yv.iter().sum::<f32>() / n as f32;
            let var: f32 = yv.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv_std = 1.0 / (var + gn_eps).sqrt();
            let mut s_acc = 0.0f32;
            for row in 0..n {
                s_acc += sh_r[row] * sh_k[row] * rkd[w_b + row];
            }

            // Phase 6: y_norm
            for row in 0..n {
                let vi = f16::from_f32(vd[v_b + row]).to_f32();
                let normalized = (yv[row] - mean) * inv_std * gd[w_b + row] + bd[w_b + row];
                expect_yn[v_b + row] = normalized + s_acc * vi;
            }
        }

        let s = mk_tensor(&mut b, s_size, TensorDtype::F32);
        let k = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let kk = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let a = mk_tensor(&mut b, k_size, TensorDtype::F16);
        let ka = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let r = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let v = mk_tensor(&mut b, k_size, TensorDtype::F16);
        let w = mk_tensor(&mut b, k_size, TensorDtype::F16);
        let g = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let bb = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let rk = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let km = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let y = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let yn = mk_tensor(&mut b, k_size, TensorDtype::F32);
        b.upload(s, &sd).unwrap();
        b.upload(k, &kd).unwrap();
        b.upload(kk, &kkd).unwrap();
        b.upload(a, &ad).unwrap();
        b.upload(ka, &kad).unwrap();
        b.upload(r, &rd).unwrap();
        b.upload(v, &vd).unwrap();
        b.upload(w, &wd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        b.upload(rk, &rkd).unwrap();
        b.fuse_ka_dplr_norm(
            s, k, kk, a, ka, r, v, w, g, bb, rk, km, y, yn, h, n, eps, gn_eps,
        )
        .expect("fuse_ka_dplr_norm");
        let got_s = b.download(s).unwrap();
        let got_km = b.download(km).unwrap();
        let got_yn = b.download(yn).unwrap();

        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        let mut km_diff = 0.0f32;
        for (e, gr) in expect_kmod.iter().zip(got_km.iter()) {
            km_diff = km_diff.max((e - gr).abs());
        }
        let mut yn_diff = 0.0f32;
        for (e, gr) in expect_yn.iter().zip(got_yn.iter()) {
            yn_diff = yn_diff.max((e - gr).abs());
        }
        assert!(s_diff < 1e-2, "fuse_ka s mismatch, max_diff={s_diff}");
        assert!(km_diff < 1e-3, "fuse_ka k_mod mismatch, max_diff={km_diff}");
        assert!(
            yn_diff < 1e-2,
            "fuse_ka y_norm mismatch, max_diff={yn_diff}"
        );
        log::info!("fuse_ka_dplr_norm vs CPU reference OK (s/km/yn max_diff<1e-2)");
    }

    /// 生成 int8 量化权重：`w[m,k] = scale[m,k/128] * idx_byte + zero[m,k/128]`。
    /// 返回 `(idx 打包 uint32 [m,k/4], sz [m,k/128], 参考 w [m*k])`。
    /// 每行 scale/zero 随机；CPU 参考按 fp16 舍入后的 scale/zero 反量化以对齐 GPU。
    fn make_int8_weights(
        m: usize,
        k: usize,
        rng: &mut impl FnMut() -> f32,
    ) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let kv = k / 4;
        let kg = k / 128;
        let mut idx = vec![0u32; m * kv];
        let mut w = vec![0.0f32; m * k];
        // 每行一份 fp16 舍入的 scale/zero。
        // 真实 int8 量化：scale = range/128（byte∈[0,255]），反量化权重与原始 fp16 同值域。
        // 测试用 rng()~[-1,1] 直接作 scale 会得到 ±128 的权重，超出 fp16 累加精度，故缩小 128 倍。
        let mut scale = vec![0.0f32; m];
        let mut zero = vec![0.0f32; m];
        for s in scale.iter_mut() {
            *s = f16::from_f32(rng() / 128.0).to_f32();
        }
        for z in zero.iter_mut() {
            *z = f16::from_f32(rng() / 128.0).to_f32();
        }
        for mm in 0..m {
            for kk in 0..k {
                let byte = (rng() * 256.0) as u32 % 256;
                idx[mm * kv + kk / 4] |= byte << ((kk % 4) * 8);
                w[mm * k + kk] = scale[mm] * (byte as f32) + zero[mm];
            }
        }
        let mut sz = vec![0u32; m * kg];
        for mm in 0..m {
            let sc = f16::from_f32(scale[mm]);
            let zr = f16::from_f32(zero[mm]);
            let pack = (sc.to_bits() as u32) | ((zr.to_bits() as u32) << 16);
            for g in 0..kg {
                sz[mm * kg + g] = pack;
            }
        }
        (idx, sz, w)
    }

    /// gemv_int8_rkv_stage1 与 CPU 参考对比：int8 量化 r/k/v 投影 + mid fp32 投影。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_int8_rkv_stage1_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_int8_rkv_stage1 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 256usize; // 整除 128(group)/4(打包)/4(ROWS)
        let vm = 2usize;
        let wm = 3usize;
        let am = 4usize;
        let gm = 5usize;

        let mut seed = 0x55667788u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // r/k/v int8 权重
        let (r_idx, r_sz, r_w) = make_int8_weights(c, c, &mut rng);
        let (k_idx, k_sz, k_w) = make_int8_weights(c, c, &mut rng);
        let (v_idx, v_sz, v_w) = make_int8_weights(c, c, &mut rng);
        // mid 权重（fp32）
        let v1d: Vec<f32> = (0..vm * c).map(|_| rng()).collect();
        let w1d: Vec<f32> = (0..wm * c).map(|_| rng()).collect();
        let a1d: Vec<f32> = (0..am * c).map(|_| rng()).collect();
        let g1d: Vec<f32> = (0..gm * c).map(|_| rng()).collect();
        // 输入
        let xr: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xk: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xv: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xw: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xa: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xg: Vec<f32> = (0..c).map(|_| rng()).collect();

        // CPU 参考
        let dot = |w: &[f32], x: &[f32], m: usize, k: usize| -> Vec<f32> {
            let mut y = vec![0.0f32; m];
            for mm in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += w[mm * k + kk] * x[kk];
                }
                y[mm] = acc;
            }
            y
        };
        let expect_r = dot(&r_w, &xr, c, c);
        let expect_k = dot(&k_w, &xk, c, c);
        let expect_v = dot(&v_w, &xv, c, c);
        // out_v 为 fp16 张量：GPU 输出 = f16 舍入后的 CPU 参考，比较前对齐精度。
        let expect_v: Vec<f32> = expect_v
            .iter()
            .map(|&x| f16::from_f32(x).to_f32())
            .collect();
        let mut expect_vm = vec![0.0f32; vm];
        let mut expect_wm = vec![0.0f32; wm];
        let mut expect_am = vec![0.0f32; am];
        let mut expect_gm = vec![0.0f32; gm];
        for i in 0..vm {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += v1d[i * c + kk] * xv[kk];
            }
            expect_vm[i] = acc;
        }
        for i in 0..wm {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += w1d[i * c + kk] * xw[kk];
            }
            expect_wm[i] = acc.tanh();
        }
        for i in 0..am {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += a1d[i * c + kk] * xa[kk];
            }
            expect_am[i] = acc;
        }
        for i in 0..gm {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += g1d[i * c + kk] * xg[kk];
            }
            expect_gm[i] = acc;
        }

        let make_handle = |b: &mut CudaBackend, idx: &[u32], sz: &[u32], m: usize, k: usize| {
            let it = b
                .create_tensor(idx.len(), TensorDtype::U32)
                .expect("create");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("create");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let rh = make_handle(&mut b, &r_idx, &r_sz, c, c);
        let kh = make_handle(&mut b, &k_idx, &k_sz, c, c);
        let vh = make_handle(&mut b, &v_idx, &v_sz, c, c);

        let v1 = mk_tensor(&mut b, vm * c, TensorDtype::F32);
        let w1 = mk_tensor(&mut b, wm * c, TensorDtype::F32);
        let a1 = mk_tensor(&mut b, am * c, TensorDtype::F32);
        let g1 = mk_tensor(&mut b, gm * c, TensorDtype::F32);
        let xr_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xk_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xv_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xw_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xa_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xg_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let or_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let ok_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let ov_t = mk_tensor(&mut b, c, TensorDtype::F16);
        let ovm_t = mk_tensor(&mut b, vm, TensorDtype::F32);
        let owm_t = mk_tensor(&mut b, wm, TensorDtype::F32);
        let oam_t = mk_tensor(&mut b, am, TensorDtype::F32);
        let ogm_t = mk_tensor(&mut b, gm, TensorDtype::F32);
        b.upload(v1, &v1d).unwrap();
        b.upload(w1, &w1d).unwrap();
        b.upload(a1, &a1d).unwrap();
        b.upload(g1, &g1d).unwrap();
        b.upload(xr_t, &xr).unwrap();
        b.upload(xk_t, &xk).unwrap();
        b.upload(xv_t, &xv).unwrap();
        b.upload(xw_t, &xw).unwrap();
        b.upload(xa_t, &xa).unwrap();
        b.upload(xg_t, &xg).unwrap();

        b.gemv_int8_rkv_stage1(
            &rh, &kh, &vh, v1, w1, a1, g1, xr_t, xk_t, xv_t, xw_t, xa_t, xg_t, or_t, ok_t, ov_t,
            ovm_t, owm_t, oam_t, ogm_t, c, vm, wm, am, gm,
        )
        .expect("gemv_int8_rkv_stage1");

        let got_r = b.download(or_t).unwrap();
        let got_k = b.download(ok_t).unwrap();
        let got_v = b.download(ov_t).unwrap();
        let got_vm = b.download(ovm_t).unwrap();
        let got_wm = b.download(owm_t).unwrap();
        let got_am = b.download(oam_t).unwrap();
        let got_gm = b.download(ogm_t).unwrap();

        let maxd = |a: &[f32], bv: &[f32]| -> f32 {
            a.iter()
                .zip(bv.iter())
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        let dr = maxd(&expect_r, &got_r);
        let dk = maxd(&expect_k, &got_k);
        let dv = maxd(&expect_v, &got_v);
        let dvm = maxd(&expect_vm, &got_vm);
        let dwm = maxd(&expect_wm, &got_wm);
        let dam = maxd(&expect_am, &got_am);
        let dgm = maxd(&expect_gm, &got_gm);
        assert!(dr < 1e-2, "int8 r mismatch, max_diff={dr}");
        assert!(dk < 1e-2, "int8 k mismatch, max_diff={dk}");
        assert!(dv < 1e-2, "int8 v mismatch, max_diff={dv}");
        assert!(dvm < 1e-3, "int8 vm mismatch, max_diff={dvm}");
        assert!(dwm < 1e-3, "int8 wm mismatch, max_diff={dwm}");
        assert!(dam < 1e-3, "int8 am mismatch, max_diff={dam}");
        assert!(dgm < 1e-3, "int8 gm mismatch, max_diff={dgm}");
        log::info!("gemv_int8_rkv_stage1 vs CPU OK (r/k/v<1e-2, vm/wm/am/gm<1e-3)");
    }

    /// gemv_variant int8（wtype=2）路径与 CPU 参考对比：覆盖 relu2/mul_add/add 三个 op。
    /// int8 idx [M,K/4]（每 uint32 4 字节）+ sz [M,K/128]（scale/zero 各 half）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_variant_int8_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_variant int8 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        // ★ 本门禁压的是 **int8 SIMT** 内核，容差 2e-2 按「fp16 激活」标定。
        // `IMMA_MIN_BATCH = 1` 之后 batch=2 会被 IMMA（W8A8，激活量化误差更大）接走
        // ⇒ 必须把阈值顶回 8 才测得到目标内核（实例字段，不影响其它测试）。
        b.imma_min_batch_override = Some(8);
        let (m, k, batch) = (8usize, 256usize, 2usize);

        let mut seed = 0xA11CE8u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let (idx, sz, _w) = make_int8_weights(m, k, &mut rng);
        let x: Vec<f32> = (0..k * batch).map(|_| rng()).collect();
        let g: Vec<f32> = (0..k * batch).map(|_| rng()).collect();

        // CPU 参考（int8 反量化权重）：w[m,k] = scale[m,k/128]*byte + zero[m,..]
        let kv = k / 4;
        let kg = k / 128;
        let dequant = |mm: usize, kk: usize| -> f32 {
            let byte = (idx[mm * kv + kk / 4] >> ((kk % 4) * 8)) & 0xFF;
            let sc = half::f16::from_bits((sz[mm * kg + kk / 128] & 0xFFFF) as u16).to_f32();
            let zr = half::f16::from_bits((sz[mm * kg + kk / 128] >> 16) as u16).to_f32();
            sc * (byte as f32) + zr
        };
        let relu2 = |v: f32| if v > 0.0 { v * v } else { 0.0 };
        let gemv = |op_sel: usize| -> Vec<f32> {
            let mut y = vec![0.0f32; m * batch];
            for bb in 0..batch {
                for mm in 0..m {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        let wv = dequant(mm, kk);
                        let gv = if op_sel == 1 {
                            half::f16::from_f32(g[bb * k + kk]).to_f32()
                        } else {
                            1.0
                        };
                        acc += wv * x[bb * k + kk] * gv;
                    }
                    y[bb * m + mm] = if op_sel == 0 { relu2(acc) } else { acc };
                }
            }
            y
        };
        let expect_relu2 = gemv(0);
        let expect_mul = gemv(1);
        let expect_add = gemv(2);

        let make_handle = |b: &mut CudaBackend, idx: &[u32], sz: &[u32]| {
            let it = b.create_tensor(idx.len(), TensorDtype::U32).expect("c");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("c");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let h = make_handle(&mut b, &idx, &sz);
        let xt = mk_tensor(&mut b, k * batch, TensorDtype::F32);
        let gt = mk_tensor(&mut b, k * batch, TensorDtype::F16);
        let yt = mk_tensor(&mut b, m * batch, TensorDtype::F32);
        b.upload(xt, &x).unwrap();
        let g16: Vec<f32> = g.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        b.upload(gt, &g16).unwrap();

        // relu2 (op=0)
        b.upload(yt, &vec![0.0f32; m * batch]).unwrap();
        b.gemv_int8_relu2(&h, xt, yt, m, k, batch).unwrap();
        let got_r2 = b.download(yt).unwrap();
        let md_r2 = got_r2
            .iter()
            .zip(&expect_relu2)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md_r2 < 2e-2, "int8 relu2 mismatch max_diff={md_r2}");

        // mul_add (op=1)：y 累加式
        b.upload(yt, &vec![0.0f32; m * batch]).unwrap();
        b.gemv_int8_mul_add(&h, xt, gt, yt, m, k, batch).unwrap();
        let got_mul = b.download(yt).unwrap();
        let md_mul = got_mul
            .iter()
            .zip(&expect_mul)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md_mul < 2e-2, "int8 mul_add mismatch max_diff={md_mul}");

        // add (op=2)：y 累加式
        b.upload(yt, &vec![0.0f32; m * batch]).unwrap();
        b.gemv_int8_add(&h, xt, yt, m, k, batch).unwrap();
        let got_add = b.download(yt).unwrap();
        let md_add = got_add
            .iter()
            .zip(&expect_add)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md_add < 2e-2, "int8 add mismatch max_diff={md_add}");

        log::info!(
            "gemv_variant int8 vs CPU OK (relu2={md_r2:.2e}, mul_add={md_mul:.2e}, add={md_add:.2e})"
        );
    }

    /// gemv_int8_plain（op=3，覆盖写）与 CPU 参考对比：y[m] = Σ_k x[k]·w[m,k]。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_int8_plain_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_int8_plain test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        // 同 `gemv_variant_int8_matches_cpu`：把阈值顶回 8，压 int8 SIMT 内核本身。
        b.imma_min_batch_override = Some(8);
        let (m, k, batch) = (8usize, 256usize, 2usize);

        let mut seed = 0x219F3DA7u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let (idx, sz, _w) = make_int8_weights(m, k, &mut rng);
        let x: Vec<f32> = (0..k * batch).map(|_| rng()).collect();

        // CPU 参考（int8 反量化权重）：w[m,k] = scale[m,k/128]*byte + zero[m,..]
        let kv = k / 4;
        let kg = k / 128;
        let dequant = |mm: usize, kk: usize| -> f32 {
            let byte = (idx[mm * kv + kk / 4] >> ((kk % 4) * 8)) & 0xFF;
            let sc = half::f16::from_bits((sz[mm * kg + kk / 128] & 0xFFFF) as u16).to_f32();
            let zr = half::f16::from_bits((sz[mm * kg + kk / 128] >> 16) as u16).to_f32();
            sc * (byte as f32) + zr
        };
        let mut expect = vec![0.0f32; m * batch];
        for bb in 0..batch {
            for mm in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += dequant(mm, kk) * x[bb * k + kk];
                }
                expect[bb * m + mm] = acc;
            }
        }

        let make_handle = |b: &mut CudaBackend, idx: &[u32], sz: &[u32]| {
            let it = b.create_tensor(idx.len(), TensorDtype::U32).expect("c");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("c");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let h = make_handle(&mut b, &idx, &sz);
        let xt = mk_tensor(&mut b, k * batch, TensorDtype::F32);
        let yt = mk_tensor(&mut b, m * batch, TensorDtype::F32);
        b.upload(xt, &x).unwrap();
        // 覆盖写：y 预填脏数据，验证 op=3 确实覆盖而非累加。
        b.upload(yt, &vec![123.0f32; m * batch]).unwrap();
        b.gemv_int8_plain(&h, xt, yt, m, k, batch).unwrap();
        let got = b.download(yt).unwrap();
        let md = got
            .iter()
            .zip(&expect)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md < 2e-2, "int8 plain mismatch max_diff={md}");
        log::info!("gemv_int8_plain vs CPU OK (max_diff={md:.2e})");
    }

    /// ffn_value_sparse_add 与 CPU 参考对比：x += r2 @ value，r2（relu²）约 96% 稀疏。
    /// 反量化 int8 后的 fp16 平铺权重走同一内核，故此测试覆盖 int8 稀疏 FFN 路径。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn ffn_value_sparse_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping ffn_value_sparse test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, fh) = (512usize, 256usize);
        const TILE: usize = 128;
        const C_TILE: usize = 256;

        let mut seed = 0x51A2C4E6u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        // value: [c, fh] 行主序（解码 gemv 按 [C, fh]）。
        let value: Vec<f32> = (0..c * fh).map(|_| rng()).collect();
        // r2: [fh]，只保留 ~6% 非零（模拟 relu² 稀疏；内核按非零列只读）。
        let mut r2 = vec![0.0f32; fh];
        for v in r2.iter_mut() {
            if rng() > 0.94 {
                *v = rng().abs();
            }
        }
        // 初始 x（含残差）与 CPU 参考。
        let mut x = vec![0.0f32; c];
        let mut expect = vec![0.0f32; c];
        for cc in 0..c {
            let v = rng();
            x[cc] = v;
            expect[cc] = v;
        }
        for f in 0..fh {
            if r2[f] != 0.0 {
                for cc in 0..c {
                    expect[cc] += r2[f] * value[cc * fh + f];
                }
            }
        }
        // 构建平铺布局（与 gpu_model::load_ffn_value_tiled 一致）。
        let c_blocks = c / C_TILE;
        let mut tiled = vec![0.0f32; fh * c];
        for f in 0..fh {
            let f_block = f / TILE;
            let f_local = f % TILE;
            for cc in 0..c {
                let c_block = cc / C_TILE;
                let c_local = cc % C_TILE;
                tiled[((f_block * c_blocks + c_block) * TILE) * C_TILE
                    + f_local * C_TILE
                    + c_local] = value[cc * fh + f];
            }
        }
        let vt = b.create_tensor(tiled.len(), TensorDtype::F16).unwrap();
        b.upload(vt, &tiled).unwrap();
        let rt = b.create_tensor(r2.len(), TensorDtype::F32).unwrap();
        let xt = b.create_tensor(x.len(), TensorDtype::F32).unwrap();
        b.upload(rt, &r2).unwrap();
        b.upload(xt, &x).unwrap();
        // int8 无稠密 fp16，value_w16 传 None；CUDA 内核忽略该参数。
        b.ffn_value_sparse_add(None, vt, rt, xt, c, fh).unwrap();
        let got = b.download(xt).unwrap();
        let md = got
            .iter()
            .zip(&expect)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md < 2e-2, "ffn_value_sparse mismatch max_diff={md}");
        log::info!("ffn_value_sparse vs CPU OK (max_diff={md:.2e})");
    }

    /// gemv_lowrank_chain4 与 CPU 参考对比：融合 4 条低秩链第二级。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_lowrank_chain4_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_lowrank_chain4 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let m = 64usize;
        let kw = 96usize;
        let ka = 96usize;
        let kv = 64usize;
        let kg = 128usize;

        let mut seed = 0x90abcdefu32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let w2: Vec<f32> = (0..m * kw).map(|_| rng()).collect();
        let a2: Vec<f32> = (0..m * ka).map(|_| rng()).collect();
        let v2: Vec<f32> = (0..m * kv).map(|_| rng()).collect();
        let g2: Vec<f32> = (0..m * kg).map(|_| rng()).collect();
        let wm: Vec<f32> = (0..kw).map(|_| rng()).collect();
        let am: Vec<f32> = (0..ka).map(|_| rng()).collect();
        let vm: Vec<f32> = (0..kv).map(|_| rng()).collect();
        let gm: Vec<f32> = (0..kg).map(|_| rng()).collect();
        let w0: Vec<f32> = (0..m).map(|_| rng()).collect();
        let a0: Vec<f32> = (0..m).map(|_| rng()).collect();
        let v0: Vec<f32> = (0..m).map(|_| rng()).collect();
        let scale = vec![rng()];
        let vf: Vec<f32> = (0..m).map(|_| rng()).collect();
        let ov_init: Vec<f32> = (0..m).map(|_| rng()).collect();

        // CPU 参考（sigmoid = 1/(1+exp(-x))）
        let sig = |x: f32| 1.0f32 / (1.0 + (-x).exp());
        let mut ew = vec![0.0f32; m];
        let mut ea = vec![0.0f32; m];
        let mut ev = vec![0.0f32; m];
        let mut eg = vec![0.0f32; m];
        for r in 0..m {
            let mut lw = 0.0f32;
            let mut la = 0.0f32;
            let mut lv = 0.0f32;
            let mut lg = 0.0f32;
            for k in 0..kw {
                lw += wm[k] * w2[r * kw + k];
            }
            for k in 0..ka {
                la += am[k] * a2[r * ka + k];
            }
            for k in 0..kv {
                lv += vm[k] * v2[r * kv + k];
            }
            for k in 0..kg {
                lg += sig(gm[k]) * g2[r * kg + k];
            }
            ew[r] = (scale[0] * sig(lw + w0[r])).exp();
            ea[r] = sig(la + a0[r]);
            ev[r] = ov_init[r] + sig(lv + v0[r]) * (vf[r] - ov_init[r]);
            eg[r] = lg;
        }

        let w2t = mk_tensor(&mut b, m * kw, TensorDtype::F32);
        let a2t = mk_tensor(&mut b, m * ka, TensorDtype::F32);
        let v2t = mk_tensor(&mut b, m * kv, TensorDtype::F32);
        let g2t = mk_tensor(&mut b, m * kg, TensorDtype::F32);
        let wmt = mk_tensor(&mut b, kw, TensorDtype::F32);
        let amt = mk_tensor(&mut b, ka, TensorDtype::F32);
        let vmt = mk_tensor(&mut b, kv, TensorDtype::F32);
        let gmt = mk_tensor(&mut b, kg, TensorDtype::F32);
        let w0t = mk_tensor(&mut b, m, TensorDtype::F32);
        let a0t = mk_tensor(&mut b, m, TensorDtype::F32);
        let v0t = mk_tensor(&mut b, m, TensorDtype::F32);
        let st = mk_tensor(&mut b, 1, TensorDtype::F32);
        let vft = mk_tensor(&mut b, m, TensorDtype::F16);
        let ow = mk_tensor(&mut b, m, TensorDtype::F16);
        let oa = mk_tensor(&mut b, m, TensorDtype::F16);
        let ov = mk_tensor(&mut b, m, TensorDtype::F16);
        let og = mk_tensor(&mut b, m, TensorDtype::F16);
        b.upload(w2t, &w2).unwrap();
        b.upload(a2t, &a2).unwrap();
        b.upload(v2t, &v2).unwrap();
        b.upload(g2t, &g2).unwrap();
        b.upload(wmt, &wm).unwrap();
        b.upload(amt, &am).unwrap();
        b.upload(vmt, &vm).unwrap();
        b.upload(gmt, &gm).unwrap();
        b.upload(w0t, &w0).unwrap();
        b.upload(a0t, &a0).unwrap();
        b.upload(v0t, &v0).unwrap();
        b.upload(st, &scale).unwrap();
        b.upload(vft, &vf).unwrap();
        b.upload(ow, &ov_init).unwrap();
        b.upload(oa, &ev).unwrap();
        b.upload(ov, &ov_init).unwrap();
        b.upload(og, &ev).unwrap();

        b.gemv_lowrank_chain4(
            w2t, a2t, v2t, g2t, wmt, amt, vmt, gmt, w0t, a0t, v0t, st, vft, ow, oa, ov, og, m, kw,
            ka, kv, kg,
        )
        .expect("gemv_lowrank_chain4");

        let gw = b.download(ow).unwrap();
        let ga = b.download(oa).unwrap();
        let gv = b.download(ov).unwrap();
        let gg = b.download(og).unwrap();
        let maxd = |a: &[f32], bv: &[f32]| -> f32 {
            a.iter()
                .zip(bv.iter())
                .fold(0.0f32, |mm, (x, y)| mm.max((x - y).abs()))
        };
        let dw = maxd(&ew, &gw);
        let da = maxd(&ea, &ga);
        let dv = maxd(&ev, &gv);
        let dg = maxd(&eg, &gg);
        assert!(dw < 1e-2, "lowrank w mismatch, max_diff={dw}");
        assert!(da < 1e-2, "lowrank a mismatch, max_diff={da}");
        assert!(dv < 1e-2, "lowrank v mismatch, max_diff={dv}");
        assert!(dg < 1e-2, "lowrank g mismatch, max_diff={dg}");
        log::info!("gemv_lowrank_chain4 vs CPU OK (<1e-2)");
    }

    /// argmax 与 CPU 参考对比：logits [N] 的最大值索引（f32 位模式存 token[0]）。
    /// 覆盖平局取小索引的语义。无 CUDA 设备时跳过。
    #[test]
    fn argmax_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping argmax test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        // 用例 1：唯一最大值。
        let n = 65536usize;
        let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, c)| a.partial_cmp(c).unwrap().then(ib.cmp(ia)))
            .map(|(i, _)| i)
            .unwrap();

        let lt = b.create_tensor(n, TensorDtype::F32).expect("create logits");
        let tok_t = b.create_tensor(1, TensorDtype::F32).expect("create token");
        b.upload(lt, &logits).unwrap();
        b.argmax(lt, tok_t, n).expect("argmax");
        let got = b.download(tok_t).unwrap();
        let got_idx = f32::to_bits(got[0]) as usize;
        assert_eq!(
            got_idx, max_idx,
            "argmax unique max mismatch: got {got_idx}, expect {max_idx}"
        );

        // 用例 2：平局取更小索引。构造两个相同最大值。
        let n2 = 1024usize;
        let mut logits2 = vec![0.0f32; n2];
        logits2[0] = 5.0;
        logits2[999] = 5.0;
        let lt2 = b
            .create_tensor(n2, TensorDtype::F32)
            .expect("create logits2");
        let tok2 = b.create_tensor(1, TensorDtype::F32).expect("create token2");
        b.upload(lt2, &logits2).unwrap();
        b.argmax(lt2, tok2, n2).expect("argmax tie");
        let got2 = b.download(tok2).unwrap();
        let got_idx2 = f32::to_bits(got2[0]) as usize;
        assert_eq!(
            got_idx2, 0,
            "argmax tie mismatch: got {got_idx2}, expect 0 (smaller index)"
        );
        log::info!("argmax vs CPU OK (unique={max_idx}, tie=0)");
    }

    /// sample 确定性验证：构造一个 logits 使某 token 概率≈1，采样必返回该 token。
    /// 覆盖空 history（无惩罚）、temperature/top-k/top-p/seed 路径。无 CUDA 设备时跳过。
    #[test]
    fn sample_deterministic() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping sample_deterministic test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        let sample_tok = |b: &mut CudaBackend,
                          logits: &[f32],
                          temp: f32,
                          top_k: u32,
                          top_p: f32,
                          seed: u32,
                          freq: f32,
                          history: &[u32]|
         -> usize {
            let lt = b
                .create_tensor(logits.len(), TensorDtype::F32)
                .expect("create logits");
            let tok = b.create_tensor(1, TensorDtype::F32).expect("create token");
            b.upload(lt, logits).unwrap();
            b.sample(
                lt,
                tok,
                logits.len(),
                temp,
                top_k,
                top_p,
                seed,
                1.0,
                freq,
                0.0,
                history,
            )
            .expect("sample");
            let got = b.download(tok).unwrap();
            f32::to_bits(got[0]) as usize
        };

        // 用例 1：logits[5] 独占极大值 → 任何 seed/参数下必返回 5。
        let n = 4096usize;
        let mut logits = vec![0.0f32; n];
        logits[5] = 100.0;
        for seed in [1u32, 42u32, 0xdeadbeefu32] {
            assert_eq!(
                sample_tok(&mut b, &logits, 1.0, 0, 1.0, seed, 0.0, &[]),
                5,
                "sample dominant-token mismatch (seed={seed})"
            );
        }

        // 用例 2：top_k=1 强制取全局最大（即使有多个相近值）。
        let mut logits2 = vec![1.0f32; n];
        logits2[7] = 2.0;
        assert_eq!(sample_tok(&mut b, &logits2, 0.5, 1, 1.0, 7, 0.0, &[]), 7);

        // 用例 3：frequency 惩罚路径。token0/token1 等大，history 含 token0 多次 →
        // freq=1.0 使 token0 logit -= 出现次数，必选 token1。其余 token 设极低 logit，
        // 避免零 logit 的大量 token 在 softmax 中淹没 token1。
        let mut logits3 = vec![-100.0f32; n];
        logits3[0] = 5.0;
        logits3[1] = 5.0;
        let history = vec![0u32, 0, 0, 0, 0, 0]; // token0 出现 6 次
        let got = sample_tok(&mut b, &logits3, 1.0, 0, 1.0, 99, 1.0, &history);
        assert_eq!(
            got, 1,
            "sample frequency-penalty mismatch: got {got}, expect 1"
        );
        log::info!("sample_deterministic OK (dominant=5, topk=7, freq-penalty=1)");
    }

    /// record_token 与 store_token_host 验证：连续写入 token 索引，序列缓冲按序累积。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn record_token_accumulates() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping record_token test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let n = 8usize;
        let in_tok = b.create_tensor(1, TensorDtype::F32).expect("create in_tok");
        let out_seq = b
            .create_tensor(n, TensorDtype::F32)
            .expect("create out_seq");
        let cnt = b.create_tensor(1, TensorDtype::F32).expect("create cnt");
        b.upload(out_seq, &vec![0.0; n]).unwrap();
        b.upload(cnt, &[0.0; 1]).unwrap();

        let tokens = [7u32, 3u32, 5u32, 1u32];
        for tk in tokens {
            b.store_token_host(in_tok, tk).expect("store_token_host");
            b.record_token(in_tok, out_seq, cnt).expect("record_token");
        }

        // 按位解释序列缓冲为 u32，验证顺序累积。
        let seq = b.download(out_seq).unwrap();
        let got: Vec<u32> = seq.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            &got[..tokens.len()],
            &tokens,
            "record_token sequence mismatch"
        );
        let cnt_got = b.download(cnt).unwrap();
        assert_eq!(cnt_got[0].to_bits(), tokens.len() as u32, "cnt mismatch");
        log::info!("record_token vs store_token_host OK (seq={tokens:?})");
    }

    /// gather_row_device_f16 与 CPU 参考对比：从 fp16 表 src[VOCAB,C] 按 token 索引取一行转 f32。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gather_row_device_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gather_row_device_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        let vocab = 64usize;
        let c = 512usize;
        // 构造 fp16 表（f32 值，上传时经 f16 舍入）。
        let table_f32: Vec<f32> = (0..vocab * c).map(|i| ((i as f32) * 0.13).sin()).collect();
        let src = b
            .create_tensor(vocab * c, TensorDtype::F16)
            .expect("create src");
        b.upload(src, &table_f32).unwrap();

        // 取 token 行：idx=42。
        let token = 42u32;
        let tok = b.create_tensor(1, TensorDtype::F32).expect("create tok");
        b.store_token_host(tok, token).unwrap();
        let dst = b.create_tensor(c, TensorDtype::F32).expect("create dst");
        b.gather_row_device_f16(src, dst, tok, c)
            .expect("gather_row");

        let got = b.download(dst).unwrap();
        // 参考：f16 舍入后的该行。
        let ref_row: Vec<f32> = table_f32[token as usize * c..(token as usize + 1) * c].to_vec();
        let mut max_diff = 0.0f32;
        for (a, g) in ref_row.iter().zip(got.iter()) {
            max_diff = max_diff.max((a - g).abs());
        }
        assert!(
            max_diff < 1e-3,
            "gather_row_device_f16 mismatch: max_diff={max_diff}"
        );
        log::info!("gather_row_device_f16 vs CPU OK (token={token}, max_diff={max_diff})");
    }

    /// copy_device_f16 与 CPU 参考对比：f16 设备到设备全量拷贝。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn copy_device_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping copy_device_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        let n = 1024usize;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).cos() * 3.0).collect();
        let src = b.create_tensor(n, TensorDtype::F16).expect("create src");
        b.upload(src, &data).unwrap();
        let dst = b.create_tensor(n, TensorDtype::F16).expect("create dst");
        b.copy_device_f16(src, dst).expect("copy_device_f16");

        let src_got = b.download(src).unwrap();
        let dst_got = b.download(dst).unwrap();
        let mut max_diff = 0.0f32;
        for (a, g) in src_got.iter().zip(dst_got.iter()) {
            max_diff = max_diff.max((a - g).abs());
        }
        assert_eq!(
            max_diff, 0.0,
            "copy_device_f16 mismatch: max_diff={max_diff}"
        );
        log::info!("copy_device_f16 OK (max_diff=0.0, n={n})");
    }

    /// 统一的 gemm 参考实现：C[i*n+j] = sum_kk A[i*k+kk]*B[j*k+kk] + op。
    fn ref_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, op: i32) -> Vec<f32> {
        // 输入先经 f16 舍入（kernel 以 fp16 读取）。
        let f16r = |v: f32| {
            let h = f16::from_f32(v);
            h.to_f32()
        };
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += f16r(a[i * k + kk]) * f16r(b[j * k + kk]);
                }
                c[i * n + j] = match op {
                    1 => acc + 1.0, // bias 常量（测试上传 bias 全 1）
                    2 => acc + 1.0, // x 常量
                    3 => {
                        if acc > 0.0 {
                            acc * acc
                        } else {
                            0.0
                        }
                    }
                    4 => acc.tanh(),
                    _ => acc,
                };
            }
        }
        c
    }

    /// 诊断：直接用 cuBLAS 驱动做一次最小 fp16 gemm（C = A@B^T），
    /// 隔离 cuBLAS 库/驱动/参数问题（700 异步非法地址排查）。
    #[test]
    fn cublas_direct_gemm_probe() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping cublas probe");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let drv = match cublas_driver() {
            Some(d) => d,
            None => {
                log::info!("cublas driver unavailable");
                return;
            }
        };
        let m = 8usize;
        let n = 6usize;
        let k = 16usize;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.13).sin()).collect();
        let wt: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.07).cos()).collect();
        let ad = b.create_tensor(m * k, TensorDtype::F16).unwrap();
        let bd = b.create_tensor(n * k, TensorDtype::F16).unwrap();
        let cd = b.create_tensor(m * n, TensorDtype::F32).unwrap();
        b.upload(ad, &a).unwrap();
        b.upload(bd, &wt).unwrap();
        let (ad_, bd_, cd_) = (
            b.f16_ptr(ad, "probe").unwrap(),
            b.f16_ptr(bd, "probe").unwrap(),
            b.f32_ptr(cd, "probe").unwrap(),
        );
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (m_i, n_i, k_i) = (n as c_int, m as c_int, k as c_int);
        let (lda, ldb, ldc) = (k as c_int, k as c_int, n as c_int);
        let r = unsafe {
            (drv.cublas_gemm_ex)(
                b.cublas.unwrap(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                m_i,
                n_i,
                k_i,
                &alpha as *const f32 as *const c_void,
                bd_ as *const c_void,
                CUDA_R_16F,
                lda,
                ad_ as *const c_void,
                CUDA_R_16F,
                ldb,
                &beta as *const f32 as *const c_void,
                cd_ as *mut c_void,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT,
            )
        };
        log::info!("[CUBLAS_PROBE] gemm_ex status={r}");
        let got = b.download(cd).unwrap();
        let exp = ref_gemm(&a, &wt, m, n, k, 0);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        log::info!("[CUBLAS_PROBE] max_diff={md}");
    }

    /// gemm 系列（plain/bias/add/relu2/tanh）与 CPU 参考对比。
    #[test]
    fn gemm_variants_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemm_variants test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, n, k) = (8usize, 6usize, 16usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.13).sin()).collect();
        let bb: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.07).cos()).collect();
        let at = b.create_tensor(m * k, TensorDtype::F16).expect("a");
        let bt = b.create_tensor(n * k, TensorDtype::F16).expect("b");
        b.upload(at, &a).unwrap();
        b.upload(bt, &bb).unwrap();

        // plain
        let ct = b.create_tensor(m * n, TensorDtype::F32).expect("c");
        b.gemm(at, bt, ct, m, n, k).expect("gemm");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 0);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm mismatch max_diff={md}");

        // bias
        let bias_t = b.create_tensor(n, TensorDtype::F32).expect("bias");
        b.upload(bias_t, &vec![1.0f32; n]).unwrap();
        b.gemm_bias(at, bt, bias_t, ct, m, n, k).expect("gemm_bias");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 1);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_bias mismatch max_diff={md}");

        // add（x 全 1）
        let xt = b.create_tensor(m * n, TensorDtype::F32).expect("x");
        b.upload(xt, &vec![1.0f32; m * n]).unwrap();
        b.gemm_add(at, bt, xt, ct, m, n, k).expect("gemm_add");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 2);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_add mismatch max_diff={md}");

        // relu2
        b.gemm_relu2(at, bt, ct, m, n, k).expect("gemm_relu2");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 3);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_relu2 mismatch max_diff={md}");

        // tanh
        b.gemm_tanh(at, bt, ct, m, n, k).expect("gemm_tanh");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 4);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_tanh mismatch max_diff={md}");
        log::info!("gemm variants (plain/bias/add/relu2/tanh) vs CPU OK");
    }

    /// copy_device（f32）与 copy_token 与 CPU 参考对比。
    #[test]
    fn copy_device_token_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping copy_device_token test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        // copy_device：全量 f32 拷贝。
        let n = 512usize;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).cos()).collect();
        let src = b.create_tensor(n, TensorDtype::F32).expect("src");
        b.upload(src, &data).unwrap();
        let dst = b.create_tensor(n, TensorDtype::F32).expect("dst");
        b.copy_device(src, dst).expect("copy_device");
        let got = b.download(dst).unwrap();
        let md = got
            .iter()
            .zip(&data)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert_eq!(md, 0.0, "copy_device mismatch max_diff={md}");

        // copy_token：取 x 的第 token 行。
        let (c, t) = (16usize, 8usize);
        let x: Vec<f32> = (0..t * c).map(|i| (i as f32) * 0.5).collect();
        let xt = b.create_tensor(t * c, TensorDtype::F32).expect("x");
        b.upload(xt, &x).unwrap();
        let token = 3usize;
        let yt = b.create_tensor(c, TensorDtype::F32).expect("y");
        b.copy_token(xt, yt, c, c, token).expect("copy_token");
        let got = b.download(yt).unwrap();
        for i in 0..c {
            assert!(
                (got[i] - x[token * c + i]).abs() < 1e-6,
                "copy_token mismatch at {i}"
            );
        }
        log::info!("copy_device & copy_token vs CPU OK");
    }

    /// to_f16 / to_f16_triple 与 CPU 参考对比。
    #[test]
    fn to_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping to_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, t, m_pad) = (8usize, 4usize, 6usize); // m_pad > t，验证填充行写 0
        let x: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.9).sin()).collect();
        let xt = b.create_tensor(t * c, TensorDtype::F32).expect("x");
        b.upload(xt, &x).unwrap();
        let yt = b.create_tensor(m_pad * c, TensorDtype::F16).expect("y");
        b.to_f16(xt, yt, c, t, m_pad, c, c).expect("to_f16");
        let got = b.download(yt).unwrap();
        for tok in 0..m_pad {
            for i in 0..c {
                let expect = if tok < t {
                    f16::from_f32(x[tok * c + i]).to_f32()
                } else {
                    0.0
                };
                assert!(
                    (got[tok * c + i] - expect).abs() < 1e-6,
                    "to_f16 mismatch at tok={tok} i={i}"
                );
            }
        }

        // to_f16_triple：三输入一次转换。
        let (xr_t, xk_t, xv_t) = (
            b.create_tensor(t * c, TensorDtype::F32).expect("xr"),
            b.create_tensor(t * c, TensorDtype::F32).expect("xk"),
            b.create_tensor(t * c, TensorDtype::F32).expect("xv"),
        );
        let xk: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.4).cos()).collect();
        let xv: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.2).tan()).collect();
        b.upload(xr_t, &x).unwrap();
        b.upload(xk_t, &xk).unwrap();
        b.upload(xv_t, &xv).unwrap();
        let (yr_t, yk_t, yv_t) = (
            b.create_tensor(m_pad * c, TensorDtype::F16).expect("yr"),
            b.create_tensor(m_pad * c, TensorDtype::F16).expect("yk"),
            b.create_tensor(m_pad * c, TensorDtype::F16).expect("yv"),
        );
        b.to_f16_triple(xr_t, xk_t, xv_t, yr_t, yk_t, yv_t, c, t, m_pad, c, c)
            .expect("to_f16_triple");
        let gr = b.download(yr_t).unwrap();
        let gk = b.download(yk_t).unwrap();
        let gv = b.download(yv_t).unwrap();
        for tok in 0..m_pad {
            for i in 0..c {
                let (er, ek, ev) = if tok < t {
                    (
                        f16::from_f32(x[tok * c + i]).to_f32(),
                        f16::from_f32(xk[tok * c + i]).to_f32(),
                        f16::from_f32(xv[tok * c + i]).to_f32(),
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };
                assert!((gr[tok * c + i] - er).abs() < 1e-6, "triple r mismatch");
                assert!((gk[tok * c + i] - ek).abs() < 1e-6, "triple k mismatch");
                assert!((gv[tok * c + i] - ev).abs() < 1e-6, "triple v mismatch");
            }
        }
        log::info!("to_f16 & to_f16_triple vs CPU OK");
    }

    /// elementwise 系列（sigmoid/inplace/scale_exp/mul）与 CPU 参考对比。
    #[test]
    fn elementwise_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping elementwise test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, batch) = (64usize, 3usize);
        let a: Vec<f32> = (0..c * batch).map(|i| ((i as f32) * 0.11).sin()).collect();
        let bb: Vec<f32> = (0..c * batch).map(|i| ((i as f32) * 0.05) + 0.5).collect();
        let at = b.create_tensor(c * batch, TensorDtype::F32).expect("a");
        let bt = b.create_tensor(c * batch, TensorDtype::F32).expect("b");
        let yt = b.create_tensor(c * batch, TensorDtype::F32).expect("y");
        b.upload(at, &a).unwrap();
        b.upload(bt, &bb).unwrap();

        b.elementwise_sigmoid(at, yt, c, batch).expect("sigmoid");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            let e = 1.0 / (1.0 + (-a[i]).exp());
            assert!((got[i] - e).abs() < 1e-5, "sigmoid mismatch at {i}");
        }

        b.elementwise_sigmoid_inplace(yt, c, batch)
            .expect("sigmoid_inplace");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            let e = 1.0 / (1.0 + (-(1.0 / (1.0 + (-a[i]).exp()))).exp());
            assert!((got[i] - e).abs() < 1e-5, "sigmoid_inplace mismatch at {i}");
        }

        b.elementwise_scale_exp(at, bt, yt, c, batch)
            .expect("scale_exp");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            // kernel 语义：y = exp(a * b[0])，b 为全局共享标量（与 Vulkan elementwise.comp OP9 一致）。
            let e = (a[i] * bb[0]).exp();
            assert!((got[i] - e).abs() < 1e-4, "scale_exp mismatch at {i}");
        }

        b.elementwise_mul(at, bt, yt, c, batch).expect("mul");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            let e = a[i] * bb[i];
            assert!((got[i] - e).abs() < 1e-6, "mul mismatch at {i}");
        }
        log::info!("elementwise (sigmoid/inplace/scale_exp/mul) vs CPU OK");
    }

    /// fuse_ka 与 sum_rk_rk 与 CPU 参考对比。
    #[test]
    fn fuse_ka_sum_rk_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping fuse_ka_sum_rk test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (h, n, batch) = (2usize, 8usize, 3usize);
        let hn = h * n;

        let k = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let kk = mk_tensor(&mut b, hn, TensorDtype::F32);
        let a = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let ka = mk_tensor(&mut b, hn, TensorDtype::F32);
        let k_mod = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let kk_l2 = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let bb = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let kd: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.3).cos()).collect();
        let kkd: Vec<f32> = (0..hn).map(|i| ((i as f32) * 0.1) + 0.5).collect();
        let ad: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.2).sin()).collect();
        let kad: Vec<f32> = (0..hn).map(|i| ((i as f32) * 0.4) + 1.0).collect();
        b.upload(k, &kd).unwrap();
        b.upload(kk, &kkd).unwrap();
        b.upload(a, &ad).unwrap();
        b.upload(ka, &kad).unwrap();
        b.fuse_ka(k, kk, a, ka, k_mod, kk_l2, bb, h, n, batch)
            .expect("fuse_ka");

        // 参考：对每个 (bidx, head) 计算。
        let mut exp_km = vec![0.0f32; batch * hn];
        let mut exp_kl = vec![0.0f32; batch * hn];
        let mut exp_b = vec![0.0f32; batch * hn];
        for bidx in 0..batch {
            for head in 0..h {
                let base = bidx * hn + head * n;
                let wbase = head * n;
                let mut sq = 0.0f32;
                for j in 0..n {
                    let kkv = kd[base + j] * kkd[wbase + j];
                    sq += kkv * kkv;
                }
                let inv = 1.0 / sq.sqrt().max(1e-12);
                for j in 0..n {
                    let kv_ = kd[base + j];
                    let kkv = kv_ * kkd[wbase + j];
                    let k_l2 = kkv * inv;
                    let av = ad[base + j];
                    exp_km[base + j] = kv_ * (1.0 + kad[wbase + j] * (av - 1.0));
                    exp_kl[base + j] = k_l2;
                    exp_b[base + j] = -k_l2 * av;
                }
            }
        }
        let got_km = b.download(k_mod).unwrap();
        let got_kl = b.download(kk_l2).unwrap();
        let got_b = b.download(bb).unwrap();
        let md = |g: &[f32], e: &[f32]| {
            g.iter()
                .zip(e)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        assert!(md(&got_km, &exp_km) < 1e-5, "fuse_ka km mismatch");
        assert!(md(&got_kl, &exp_kl) < 1e-5, "fuse_ka kl mismatch");
        assert!(md(&got_b, &exp_b) < 1e-5, "fuse_ka b mismatch");

        // sum_rk_rk：y[b,h*n+j] += sum_j r*km*rk * v。
        let r = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let rk = mk_tensor(&mut b, hn, TensorDtype::F32);
        let v = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let y = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let rd: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.6).cos()).collect();
        let rkd: Vec<f32> = (0..hn).map(|i| ((i as f32) * 0.7) + 0.1).collect();
        let vd: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.8).sin()).collect();
        let yd: Vec<f32> = (0..batch * hn).map(|i| (i as f32) * 0.01).collect();
        b.upload(r, &rd).unwrap();
        b.upload(rk, &rkd).unwrap();
        b.upload(v, &vd).unwrap();
        b.upload(y, &yd).unwrap();
        b.sum_rk_rk(r, k_mod, rk, v, y, h, n, batch)
            .expect("sum_rk_rk");
        let got = b.download(y).unwrap();
        let mut exp_y = yd.clone();
        for bidx in 0..batch {
            for head in 0..h {
                let base = bidx * hn + head * n;
                let mut s = 0.0f32;
                for j in 0..n {
                    s += rd[base + j] * exp_km[base + j] * rkd[head * n + j];
                }
                for j in 0..n {
                    exp_y[base + j] += s * vd[base + j];
                }
            }
        }
        let md = got
            .iter()
            .zip(&exp_y)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-5, "sum_rk_rk mismatch max_diff={md}");
        log::info!("fuse_ka & sum_rk_rk vs CPU OK");
    }

    /// seq_shift 与 v_first_lerp 与 CPU 参考对比。
    #[test]
    fn seq_shift_v_first_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping seq_shift_v_first test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, t) = (16usize, 8usize);

        let x = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let s = mk_tensor(&mut b, c, TensorDtype::F32);
        let tm = mk_tensor(&mut b, c, TensorDtype::F32);
        let y = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let xd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.2).cos()).collect();
        let sd: Vec<f32> = (0..c).map(|i| ((i as f32) * 0.5) + 1.0).collect();
        let tmd: Vec<f32> = (0..c).map(|i| ((i as f32) * 0.1) + 0.3).collect();
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(tm, &tmd).unwrap();
        b.seq_shift(x, s, tm, y, c, t, c, c).expect("seq_shift");
        let got = b.download(y).unwrap();
        for tok in 0..t {
            for i in 0..c {
                let cur = xd[tok * c + i];
                let prev = if tok == 0 {
                    sd[i]
                } else {
                    xd[(tok - 1) * c + i]
                };
                let e = cur + tmd[i] * (prev - cur);
                assert!((got[tok * c + i] - e).abs() < 1e-5, "seq_shift mismatch");
            }
        }

        // v_first_lerp：v = v + gate*(v_first - v)（原地 v）。
        let g = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let vf = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let gd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.3).sin()).collect();
        let vfd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.4).cos()).collect();
        b.upload(g, &gd).unwrap();
        b.upload(vf, &vfd).unwrap();
        b.v_first_lerp(x, g, vf, c, t, c).expect("v_first_lerp");
        let got = b.download(x).unwrap();
        for i in 0..t * c {
            let e = xd[i] + gd[i] * (vfd[i] - xd[i]);
            assert!((got[i] - e).abs() < 1e-5, "v_first_lerp mismatch at {i}");
        }
        log::info!("seq_shift & v_first_lerp vs CPU OK");
    }

    /// dplr_seq 与 CPU 参考对比（n<=64，逐线程独立状态）。
    /// 注：kernel 按 RWKV-7 的 N=64 设计（half-warp 16 线程 × 4 列），故用 n=64 测试。
    #[test]
    fn dplr_seq_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping dplr_seq test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (h, n, t) = (2usize, 64usize, 6usize);
        let c = h * n;

        let s = mk_tensor(&mut b, h * n * n, TensorDtype::F32);
        let r = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let w = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let k = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let v = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let a = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let y = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let sd: Vec<f32> = (0..h * n * n).map(|i| ((i as f32) * 0.1).cos()).collect();
        let rd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.2).sin()).collect();
        let wd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.3) + 0.5).collect();
        let kd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.4).cos()).collect();
        let vd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.5).sin()).collect();
        let ad: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.6) + 1.0).collect();
        let bd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.7).cos()).collect();
        b.upload(s, &sd).unwrap();
        b.upload(r, &rd).unwrap();
        b.upload(w, &wd).unwrap();
        b.upload(k, &kd).unwrap();
        b.upload(v, &vd).unwrap();
        b.upload(a, &ad).unwrap();
        b.upload(bb, &bd).unwrap();
        b.dplr_seq(s, r, w, k, v, a, bb, y, h, n, t, c)
            .expect("dplr_seq");
        let got_y = b.download(y).unwrap();
        let got_s = b.download(s).unwrap();

        let mut exp_y = vec![0.0f32; t * c];
        let mut exp_s = sd.clone();
        for head in 0..h {
            // 每线程 i 独立 sreg。
            let mut sreg = vec![vec![0.0f32; n]; n];
            for i in 0..n {
                for j in 0..n {
                    sreg[i][j] = sd[head * n * n + i * n + j];
                }
            }
            for tt in 0..t {
                for i in 0..n {
                    let vv = vd[head * n + i + tt * c];
                    let mut sa = 0.0f32;
                    for j in 0..n {
                        sa += ad[head * n + j + tt * c] * sreg[i][j];
                    }
                    let mut yv = 0.0f32;
                    for j in 0..n {
                        sreg[i][j] = sreg[i][j] * wd[head * n + j + tt * c]
                            + kd[head * n + j + tt * c] * vv
                            + sa * bd[head * n + j + tt * c];
                        yv += sreg[i][j] * rd[head * n + j + tt * c];
                    }
                    exp_y[head * n + i + tt * c] = yv;
                }
            }
            for i in 0..n {
                for j in 0..n {
                    exp_s[head * n * n + i * n + j] = sreg[i][j];
                }
            }
        }
        // 测试数据（wd/ad 递增）导致递归状态数值指数爆炸（可达 1e8），
        // 绝对容差不适用，改为相对容差（kernel 与参考逐位一致）。
        let rel = |got: &[f32], exp: &[f32]| -> f32 {
            let md = got
                .iter()
                .zip(exp)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            let scale = exp.iter().fold(0.0f32, |m, y| m.max(y.abs())).max(1.0);
            md / scale
        };
        let rel_y = rel(&got_y, &exp_y);
        let rel_s = rel(&got_s, &exp_s);
        assert!(rel_y < 1e-4, "dplr_seq y relative mismatch={rel_y}");
        assert!(rel_s < 1e-4, "dplr_seq s relative mismatch={rel_s}");
        log::info!("dplr_seq vs CPU OK (rel_y={rel_y} rel_s={rel_s})");
    }

    /// gemv_seq 与 CPU 参考对比。
    #[test]
    fn gemv_seq_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_seq test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, k, batch) = (6usize, 16usize, 3usize);
        let x_stride = k;
        let y_stride = m;

        let a = mk_tensor(&mut b, m * k, TensorDtype::F32);
        let x = mk_tensor(&mut b, batch * x_stride, TensorDtype::F32);
        let y = mk_tensor(&mut b, batch * y_stride, TensorDtype::F32);
        let ad_: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.3).sin()).collect();
        let xd: Vec<f32> = (0..batch * k).map(|i| ((i as f32) * 0.4).cos()).collect();
        b.upload(a, &ad_).unwrap();
        b.upload(x, &xd).unwrap();
        b.gemv_seq(a, x, y, m, k, x_stride, y_stride, batch)
            .expect("gemv_seq");
        let got = b.download(y).unwrap();
        for bidx in 0..batch {
            for row in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += ad_[row * k + kk] * xd[bidx * k + kk];
                }
                assert!(
                    (got[bidx * y_stride + row] - acc).abs() < 1e-5,
                    "gemv_seq mismatch bidx={bidx} row={row}"
                );
            }
        }
        log::info!("gemv_seq vs CPU OK");
    }

    /// dequant_int8_to_f16 与 CPU 参考对比。
    #[test]
    fn dequant_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping dequant test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, k) = (4usize, 128usize); // k 为 128 倍数，满足 int8 分组
        let make_int8 = |b: &mut CudaBackend, idx: &[u32], sz: &[u32]| {
            let it = b.create_tensor(idx.len(), TensorDtype::U32).expect("idx");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("sz");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let kg = k / 128;
        let sz: Vec<u32> = (0..m * kg)
            .map(|i| {
                let scale = ((i as f32) * 0.05) + 1.0;
                let zero = ((i as f32) * 0.01) + 0.1;
                scale.to_bits().wrapping_add(zero.to_bits())
            })
            .collect();
        // 参考反量化。
        let sz_scale = |i: usize| -> (f32, f32) {
            let s = sz[i];
            let scale = half::f16::from_bits((s & 0xFFFF) as u16).to_f32();
            let zero = half::f16::from_bits((s >> 16) as u16).to_f32();
            (scale, zero)
        };

        // int8：每个 uint32 装 4 个字节。
        let kv8 = k / 4;
        let mut i_idx = vec![0u32; m * kv8];
        for (i, v) in i_idx.iter_mut().enumerate() {
            *v = ((i as u32) & 0xFF)
                | (((i as u32) & 0xFF) << 8)
                | (((i as u32) & 0xFF) << 16)
                | (((i as u32) & 0xFF) << 24);
        }
        let hi = make_int8(&mut b, &i_idx, &sz);
        let out8 = b.create_tensor(m * k, TensorDtype::F16).expect("out8");
        b.dequant_int8_to_f16(&hi, out8, m, k)
            .expect("dequant_int8");
        let got = b.download(out8).unwrap();
        let mut exp8 = vec![0.0f32; m * k];
        for mm in 0..m {
            for kk in 0..kv8 {
                let ipack = i_idx[mm * kv8 + kk];
                let g = kk / 32;
                let (sc, zr) = sz_scale(mm * kg + g);
                for j in 0..4 {
                    let byte = (ipack >> (8 * j)) & 0xFF;
                    let wv = sc * (byte as f32) + zr;
                    exp8[mm * k + kk * 4 + j] = half::f16::from_f32(wv).to_f32();
                }
            }
        }
        let md = got
            .iter()
            .zip(&exp8)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-5, "dequant_int8 mismatch max_diff={md}");
        log::info!("dequant_int8_to_f16 vs CPU OK");
    }

    /// ★ **临时探针**：验 `mma.sync.aligned.m8n8k16`（int8 IMMA）在 NVRTC + sm_75 上
    /// ①能否编译、②片段→线程的地址映射是否如假设。验完即删。
    ///
    /// 假设的映射（PTX m8n8k16 .s8）：
    ///   groupID = lane>>2，tig = lane&3
    ///   A（row-major，8×16）：线程持 row=groupID、col=tig*4+{0..3} ⇒ **一个 u32**
    ///   B（col-major，16×8，本工程权重即 [n][k] 行主序）：线程持 k=tig*4+{0..3}、n=groupID
    ///   D（8×8 int32）：线程持 row=groupID、col=tig*2+{0,1}
    /// 语义：D[batch, n] = Σ_k xq[batch,k] * wq[n,k]（int32 精确累加）。
    #[test]
    fn imma_m8n8k16_probe() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping imma probe");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (batch, m_dim, k) = (8usize, 8usize, 256usize);
        assert_eq!(k % 16, 0);

        // int8 数据（有符号，[-64,63]）；4 个连续 k 打包进一个 u32（byte i = k%4==i）。
        let mut seed = 0x1BAD_5EEDu32;
        let mut next_i8 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ((seed >> 8) % 128) as i32 - 64
        };
        let xq: Vec<i8> = (0..batch * k).map(|_| next_i8() as i8).collect();
        let wq: Vec<i8> = (0..m_dim * k).map(|_| next_i8() as i8).collect();
        let pack = |v: &[i8]| -> Vec<u32> {
            v.chunks(4)
                .map(|c| {
                    (c[0] as u8 as u32)
                        | ((c[1] as u8 as u32) << 8)
                        | ((c[2] as u8 as u32) << 16)
                        | ((c[3] as u8 as u32) << 24)
                })
                .collect()
        };
        let xpack = pack(&xq);
        let wpack = pack(&wq);

        const SRC: &str = r#"
extern "C" __global__ void imma_probe(
    const unsigned int* __restrict__ xq,
    const unsigned int* __restrict__ wq,
    unsigned int* __restrict__ d,
    const int k)
{
    const int lane = threadIdx.x & 31;
    const int row  = lane >> 2;
    const int tig  = lane & 3;
    const int kv   = k >> 2;
    const unsigned int* arow = xq + row * kv + tig;
    const unsigned int* brow = wq + row * kv + tig;
    int acc0 = 0, acc1 = 0;
    for (int q = 0; q < kv; q += 4) {
        const unsigned int a = arow[q];
        const unsigned int c = brow[q];
        asm volatile(
            "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
            : "+r"(acc0), "+r"(acc1) : "r"(a), "r"(c));
    }
    d[row * 8 + tig * 2 + 0] = (unsigned int)acc0;
    d[row * 8 + tig * 2 + 1] = (unsigned int)acc1;
}
"#;
        let func = b
            .kernel("imma_probe", SRC, "imma_probe")
            .expect("NVRTC 编译 mma.m8n8k16 失败（sm_75 不支持？）");

        let xt = b.create_tensor(xpack.len(), TensorDtype::U32).expect("xq");
        let wt = b.create_tensor(wpack.len(), TensorDtype::U32).expect("wq");
        let dt = b.create_tensor(batch * m_dim, TensorDtype::U32).expect("d");
        b.upload_u32(xt, &xpack).unwrap();
        b.upload_u32(wt, &wpack).unwrap();
        b.upload_u32(dt, &vec![0u32; batch * m_dim]).unwrap();

        let xd = b.u32_ptr(xt, "imma_probe").expect("xq ptr");
        let wd = b.u32_ptr(wt, "imma_probe").expect("wq ptr");
        let dd = b.u32_ptr(dt, "imma_probe").expect("d ptr");
        let k_i = k as i32;
        let params = [
            &xd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &dd as *const u64 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
        ];
        b.drv
            .launch_smem(b.stream, func, (1, 1, 1), (32, 1, 1), &params, 0)
            .expect("launch imma_probe");
        let got = b.download_u32(dt).expect("download d");

        let mut worst = 0i32;
        for bb in 0..batch {
            for nn in 0..m_dim {
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += xq[bb * k + kk] as i32 * wq[nn * k + kk] as i32;
                }
                let g = got[bb * m_dim + nn] as i32;
                worst = worst.max((g - acc).abs());
                assert_eq!(g, acc, "imma mismatch b={bb} n={nn} got={g} expect={acc}");
            }
        }
        log::info!("★ IMMA m8n8k16 在 sm_75 可用且片段映射正确（imma_probe worst={worst}）");
    }

    /// 数值门禁：`quant_x_i8` 对 CPU 参考（int8 字节 / 每组尺度 / 零点修正统计）。
    /// 门控与非门控两条路径都测（门控是 op==1 的 `mul_add` 用的）。
    #[test]
    fn quant_x_i8_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping quant_x_i8 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (rows, k) = (8usize, 256usize);
        let g = k / QUANT_X_I8_GROUP;
        assert_eq!(k % QUANT_X_I8_GROUP, 0);

        let mut seed = 0x51ED_2B17u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        // 幅度逐组放大，覆盖 amax 差异明显的组（尺度分组是否真的按组独立）
        let x: Vec<f32> = (0..rows * k)
            .map(|i| rng() * (1.0 + (i / 128) as f32))
            .collect();
        let gate: Vec<f32> = (0..rows * k)
            .map(|_| 0.25 + 0.5 * (rng() * 0.5 + 0.5))
            .collect();
        // GPU 侧门控是 fp16，参考也必须先降到 fp16 再乘，否则容差不成立
        let gate16: Vec<f32> = gate.iter().map(|v| f16::from_f32(*v).to_f32()).collect();

        let xt = b
            .create_tensor(rows * k, TensorDtype::F32)
            .expect("create x");
        let gt = b
            .create_tensor(rows * k, TensorDtype::F16)
            .expect("create gate");
        b.upload(xt, &x).unwrap();
        b.upload(gt, &gate).unwrap();
        let xd = b.f32_ptr(xt, "quant_x_i8").unwrap();
        let gd = b.f16_ptr(gt, "quant_x_i8").unwrap();

        for (use_gate, gptr) in [(false, 0u64), (true, gd)] {
            let xqt = b
                .create_tensor(rows * k / 4, TensorDtype::U32)
                .expect("create xq");
            let xat = b
                .create_tensor(rows * g * 4, TensorDtype::F32)
                .expect("create xaux");
            b.upload_u32(xqt, &vec![0u32; rows * k / 4]).unwrap();
            b.upload(xat, &vec![0.0f32; rows * g * 4]).unwrap();
            b.quant_x_i8(xd, gptr, xqt, xat, rows, k)
                .expect("quant_x_i8");

            let got_q = b.download_u32(xqt).unwrap();
            let got_aux = b.download(xat).unwrap();

            let mut worst_q = 0i32;
            let mut worst_sx = 0.0f32;
            let mut worst_corr = 0.0f32;
            let mut worst_rs = 0.0f32;
            for bb in 0..rows {
                for gg in 0..g {
                    let base = bb * k + gg * QUANT_X_I8_GROUP;
                    let mut amax = 0.0f32;
                    let mut vals = [0.0f32; QUANT_X_I8_GROUP];
                    for (i, slot) in vals.iter_mut().enumerate() {
                        let v = x[base + i] * if use_gate { gate16[base + i] } else { 1.0 };
                        *slot = v;
                        amax = amax.max(v.abs());
                    }
                    let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    let mut cs = 0i32;
                    let mut rs = 0.0f32;
                    for (i, v) in vals.iter().enumerate() {
                        let q = (v / sx).round().clamp(-127.0, 127.0) as i32;
                        cs += q;
                        rs += *v;
                        let wi = (base + i) / 4;
                        let sh = (((base + i) % 4) * 8) as u32;
                        let byte = (got_q[wi] >> sh) & 0xFF;
                        // int8 回读是补码
                        let got_byte = (byte as i32) << 24 >> 24;
                        // `round`（半数远离零）与 `__float2int_rn`（半数取偶）在恰好 .5
                        // 时差 1，随机数据不该命中；给 1 LSB 余量以防万一。
                        worst_q = worst_q.max((got_byte - q).abs());
                        assert!(
                            (got_byte - q).abs() <= 1,
                            "xq mismatch b={bb} g={gg} i={i}: got={got_byte} expect={q}"
                        );
                    }
                    let au = (bb * g + gg) * 4;
                    worst_sx = worst_sx.max((got_aux[au] - sx).abs());
                    let corr = 128.0 * sx * cs as f32;
                    worst_corr = worst_corr.max((got_aux[au + 1] - corr).abs());
                    // rs 归约顺序不同（GPU 是 shuffle 树、CPU 是顺序），按相对量判
                    worst_rs = worst_rs.max((got_aux[au + 2] - rs).abs() / (1.0 + rs.abs()));
                    assert!(
                        (got_aux[au] - sx).abs() <= 1e-6,
                        "sx mismatch b={bb} g={gg}: got={} expect={sx}",
                        got_aux[au]
                    );
                    assert!(
                        (got_aux[au + 1] - corr).abs() <= 1e-3 * (1.0 + corr.abs()),
                        "corr mismatch b={bb} g={gg}: got={} expect={corr}",
                        got_aux[au + 1]
                    );
                    assert!(
                        (got_aux[au + 2] - rs).abs() <= 1e-4 * (1.0 + rs.abs()),
                        "rowsum mismatch b={bb} g={gg}: got={} expect={rs}",
                        got_aux[au + 2]
                    );
                }
            }
            log::info!(
                "quant_x_i8 use_gate={use_gate} OK（worst_q={worst_q} sx={worst_sx:e} \
                 corr={worst_corr:e} rs_rel={worst_rs:e}）"
            );
        }
    }

    /// 数值门禁：`imma_gemm_batch`（int8 张量核 W8A8）对 CPU 参考。
    ///
    /// 参考口径 = 「**用重建的 x**（`sx·xq`）乘精确反量化权重」，再放宽到
    /// int8 激活量化的**最坏误差上界**：
    /// `Σ_k 0.5·sx·|w|`（主项舍入） + `Σ_g |z|·128·sx/2`（零点项里用精确 rowsum 而非
    /// `sx·Σxq` 带来的差异）。公式/片段映射任何一处错位都会远超这个界。
    ///
    /// **覆盖 split-K 三条路径**：`ks=None`（自动，实际取 1）+ 强制 `ks=2`/`ks=3`
    /// ——后者把 `imma_gemm_batch` 的部分和写 + `imma_gemm_reduce` 的确定性归约一起压到。
    #[test]
    fn imma_gemm_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping imma_gemm test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        // m 不取 tile 整数倍、batch 跨过 BN 边界的档位：把尾部 guard 与 grid.y 一并压到
        imma_gemm_gate(&mut b, 100, 640, 80, 64, None);
        // k=768 = 6 个 k-tile ⇒ 可被 2/3 整除，专门走 split-K（部分和 + 归约）
        imma_gemm_gate(&mut b, 100, 768, 80, 64, Some(2));
        imma_gemm_gate(&mut b, 100, 768, 80, 64, Some(3));
        // ★ `IMMA_MIN_BATCH = 1` 之后 batch=1/3 也走 IMMA，且 `small_batch_bn` 给出 **BN=8**
        //（mma n 维刚好填满、槽位浪费最多）—— 这两档必须单独压住。
        imma_gemm_gate(&mut b, 100, 768, 1, 8, None);
        imma_gemm_gate(&mut b, 100, 768, 1, 8, Some(2));
        imma_gemm_gate(&mut b, 100, 768, 3, 8, None);
        imma_gemm_gate(&mut b, 100, 768, 3, 8, Some(3));
    }

    fn imma_gemm_gate(
        b: &mut CudaBackend,
        m: usize,
        k: usize,
        batch: usize,
        bn: usize,
        ks: Option<usize>,
    ) {
        let kv = k / 4;
        let kg = k / QUANT_X_I8_GROUP;
        assert_eq!(k % QUANT_X_I8_GROUP, 0);

        let mut seed = 0xC0FF_EE11u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // 权重：per-(行, 128 组) 的 scale/zero（真实磁盘格式），byte 铺满 0..255
        let mut idx = vec![0u32; m * kv];
        let mut sz = vec![0u32; m * kg];
        let mut w = vec![0.0f32; m * k];
        let mut z_abs_max = 0.0f32;
        for mm in 0..m {
            for gg in 0..kg {
                let sc16 = f16::from_f32(0.002 + 0.004 * rng().abs());
                let zr16 = f16::from_f32(0.01 * rng());
                let (sc, zr) = (sc16.to_f32(), zr16.to_f32());
                sz[mm * kg + gg] = (sc16.to_bits() as u32) | ((zr16.to_bits() as u32) << 16);
                z_abs_max = z_abs_max.max(zr.abs());
                for kk in gg * QUANT_X_I8_GROUP..(gg + 1) * QUANT_X_I8_GROUP {
                    let byte = ((rng() * 0.5 + 0.5) * 256.0) as u32 & 0xFF;
                    idx[mm * kv + kk / 4] |= byte << ((kk % 4) * 8);
                    w[mm * k + kk] = sc * byte as f32 + zr;
                }
            }
        }
        let x: Vec<f32> = (0..k * batch).map(|_| rng()).collect();

        // CPU 参考：重建 x 的量化 + 精确权重
        let mut ref_y = vec![0.0f32; batch * m];
        let mut tol = vec![0.0f32; batch * m];
        for bb in 0..batch {
            for gg in 0..kg {
                let g0 = gg * QUANT_X_I8_GROUP;
                let mut amax = 0.0f32;
                for kk in g0..g0 + QUANT_X_I8_GROUP {
                    amax = amax.max(x[bb * k + kk].abs());
                }
                let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                for kk in g0..g0 + QUANT_X_I8_GROUP {
                    let q = (x[bb * k + kk] / sx).round().clamp(-127.0, 127.0);
                    for mm in 0..m {
                        let wv = w[mm * k + kk];
                        ref_y[bb * m + mm] += sx * q * wv;
                        tol[bb * m + mm] += 0.5 * sx * wv.abs();
                    }
                }
                // 零点项走精确 rowsum 而非 sx·Σxq 的差异上界
                let extra = 0.5 * sx * QUANT_X_I8_GROUP as f32 * z_abs_max;
                for slot in tol[bb * m..(bb + 1) * m].iter_mut() {
                    *slot += extra;
                }
            }
        }

        let it = b.create_tensor(m * kv, TensorDtype::U32).expect("idx");
        let st = b.create_tensor(m * kg, TensorDtype::U32).expect("sz");
        let xt = b.create_tensor(k * batch, TensorDtype::F32).expect("x");
        let yt = b.create_tensor(m * batch, TensorDtype::F32).expect("y");
        let yr = b.create_tensor(m * batch, TensorDtype::F32).expect("y_ref");
        b.upload_u32(it, &idx).unwrap();
        b.upload_u32(st, &sz).unwrap();
        b.upload(xt, &x).unwrap();
        b.upload(yt, &vec![0.0f32; m * batch]).unwrap();
        b.upload(yr, &vec![0.0f32; m * batch]).unwrap();
        let aidx = b.u32_ptr(it, "imma_gemm_batch").unwrap();
        let asz = b.u32_ptr(st, "imma_gemm_batch").unwrap();
        let xd = b.f32_ptr(xt, "imma_gemm_batch").unwrap();
        let yd = b.f32_ptr(yt, "imma_gemm_batch").unwrap();
        let yrd = b.f32_ptr(yr, "imma_gemm_batch").unwrap();
        b.imma_gemm_tiled_ks(aidx, asz, xd, 0, yd, m, k, batch, 3, 64, bn, ks)
            .expect("imma_gemm_batch");
        let got = b.download(yt).unwrap();

        // ★ split-K 专项：同输入下 `ks>1` 与 `ks=1` 必须**只差 fp32 重结合**。
        // 这条比「对 CPU 参考」灵敏得多 —— 部分和写/读的偏移、归约顺序、k 段边界
        // 任何一处错位都会给出 O(1) 级偏差，而重结合只有 1e-6 级。
        if ks.unwrap_or(1) > 1 {
            b.imma_gemm_tiled_ks(aidx, asz, xd, 0, yrd, m, k, batch, 3, 64, bn, Some(1))
                .expect("imma_gemm_batch ks=1");
            let base = b.download(yr).unwrap();
            let mut worst = 0.0f32;
            let mut worst_i = 0usize;
            for i in 0..batch * m {
                let d = (got[i] - base[i]).abs() / base[i].abs().max(1.0);
                if d > worst {
                    worst = d;
                    worst_i = i;
                }
            }
            assert!(
                worst < 1e-5,
                "split-K(ks={}) 与 ks=1 偏差过大：rel={worst:e} @b={} m={}（got={} base={}）",
                ks.unwrap(),
                worst_i / m,
                worst_i % m,
                got[worst_i],
                base[worst_i]
            );
            log::info!(
                "★ split-K(ks={}) vs ks=1 最大相对偏差 {worst:e}",
                ks.unwrap()
            );
        }

        let mut worst_ratio = 0.0f32;
        let mut worst_abs = 0.0f32;
        let mut sq_err = 0.0f64;
        let mut sq_ref = 0.0f64;
        for i in 0..batch * m {
            let e = (got[i] - ref_y[i]).abs();
            worst_abs = worst_abs.max(e);
            worst_ratio = worst_ratio.max(e / (tol[i] + 1e-4));
            sq_err += (e as f64) * (e as f64);
            sq_ref += (ref_y[i] as f64) * (ref_y[i] as f64);
            assert!(
                e <= tol[i] + 1e-4,
                "imma_gemm mismatch at b={} m={}: got={} expect={} |e|={e} tol={}",
                i / m,
                i % m,
                got[i],
                ref_y[i],
                tol[i]
            );
        }
        let rel_rms = (sq_err / sq_ref.max(1e-30)).sqrt();
        log::info!(
            "★ imma_gemm_batch vs CPU OK（m={m} k={k} batch={batch} bn={bn} ks={:?} \
             worst_abs={worst_abs:e} worst_ratio={worst_ratio:.3} rel_rms={rel_rms:.3e}）",
            ks.unwrap_or(1)
        );
    }

    /// ★ Phase 2 门禁：低秩链两级 fp16 张量核 GEMM（`lowrank_gemm_batch`）对 CPU 参考。
    ///
    /// 参考口径 = **与内核同源的 fp32 计算**：权重/x/mid 全按 `f16` 舍入（内核的 A/B/D 就是
    /// fp16 语义），再按 chain4 的逐字语义在 CPU 上算一遍。两级累加顺序差异与 `expf` 的
    /// libm 差异都只有 1e-6 级 ⇒ 容差取 `2e-3·max(1,|ref|)`；片段映射、A/B 布局、
    /// mid16 列段偏移、链序、v 链读改写任何一处错位都会给出 O(1) 误差。
    #[test]
    fn lowrank_gemm_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping lowrank_gemm test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        // 小尺寸：C=64（≥8）、batch=40（≥16）、四链 mid/pad 均 64（满足「pad 为 64 倍数」）。
        let (c, batch, pad) = (64usize, 40usize, 64usize);
        let mut seed = 0x5EED_2026u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        // 模拟落盘/降位的 f16 舍入（与 `upload` 的 `f16::from_f32`、内核 `__float2half_rn` 同）
        let r16 = |v: f32| f16::from_f32(v).to_f32();
        // 链序固定 [v, w, a, g]
        let w1: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..pad * c).map(|_| r16(0.1 * rng())).collect())
            .collect();
        let w2: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..c * pad).map(|_| r16(0.1 * rng())).collect())
            .collect();
        let xs: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..batch * c).map(|_| r16(rng())).collect())
            .collect();
        // [v0, w0, a0]
        let bias: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..c).map(|_| r16(0.2 * rng())).collect())
            .collect();
        let vf: Vec<f32> = (0..batch * c).map(|_| r16(rng())).collect();
        let ov0: Vec<f32> = (0..batch * c).map(|_| r16(0.5 * rng())).collect();
        let sc = -1.0f32;

        // —— CPU 参考：一级（act: v/a 无、w=tanh、g=sigmoid）→ 二级（chain4 epilogue）
        let sig = |t: f32| 1.0 / (1.0 + (-t).exp());
        let mut mid = vec![vec![0.0f32; batch * pad]; 4];
        for (ch, act) in [(0usize, 0u8), (1, 1), (2, 0), (3, 2)] {
            for bb in 0..batch {
                for j in 0..pad {
                    let mut s = 0.0f32;
                    for kk in 0..c {
                        s += xs[ch][bb * c + kk] * w1[ch][j * c + kk];
                    }
                    mid[ch][bb * pad + j] = r16(match act {
                        1 => s.tanh(),
                        2 => sig(s),
                        _ => s,
                    });
                }
            }
        }
        let mut ref_out = vec![vec![0.0f32; batch * c]; 4];
        for bb in 0..batch {
            for m in 0..c {
                let dot = |ch: usize| -> f32 {
                    let mut s = 0.0f32;
                    for j in 0..pad {
                        s += mid[ch][bb * pad + j] * w2[ch][m * pad + j];
                    }
                    s
                };
                let (dv, dw, da, dg) = (dot(0), dot(1), dot(2), dot(3));
                let i = bb * c + m;
                let cur = ov0[i];
                ref_out[0][i] = cur + sig(dv + bias[0][m]) * (vf[i] - cur);
                ref_out[1][i] = (sc * sig(dw + bias[1][m])).exp();
                ref_out[2][i] = sig(da + bias[2][m]);
                ref_out[3][i] = dg;
            }
        }

        // —— 上传（张量创建/上传的 f16 舍入与参考口径一致）
        let up16 = |b: &mut CudaBackend, data: &[f32]| -> TensorId {
            let t = b.create_tensor(data.len(), TensorDtype::F16).unwrap();
            b.upload(t, data).unwrap();
            t
        };
        let up32 = |b: &mut CudaBackend, data: &[f32]| -> TensorId {
            let t = b.create_tensor(data.len(), TensorDtype::F32).unwrap();
            b.upload(t, data).unwrap();
            t
        };
        let (v1t, w1t, a1t, g1t) = (
            up16(&mut b, &w1[0]),
            up16(&mut b, &w1[1]),
            up16(&mut b, &w1[2]),
            up16(&mut b, &w1[3]),
        );
        let (w2t, a2t, v2t, g2t) = (
            up16(&mut b, &w2[1]),
            up16(&mut b, &w2[2]),
            up16(&mut b, &w2[0]),
            up16(&mut b, &w2[3]),
        );
        let (xwt, xat, xvt, xgt) = (
            up32(&mut b, &xs[1]),
            up32(&mut b, &xs[2]),
            up32(&mut b, &xs[0]),
            up32(&mut b, &xs[3]),
        );
        let (w0t, a0t, v0t) = (
            up32(&mut b, &bias[1]),
            up32(&mut b, &bias[2]),
            up32(&mut b, &bias[0]),
        );
        let sct = up32(&mut b, &[sc]);
        let vft = up16(&mut b, &vf);
        let owt = up16(&mut b, &vec![0.0f32; batch * c]);
        let oat = up16(&mut b, &vec![0.0f32; batch * c]);
        let ovt = up16(&mut b, &ov0); // v 链**读改写**：初值即 v 投影
        let ogt = up16(&mut b, &vec![0.0f32; batch * c]);

        #[allow(clippy::too_many_arguments)]
        b.lowrank_gemm_batch(
            v1t, w1t, a1t, g1t, w2t, a2t, v2t, g2t, xwt, xat, xvt, xgt, w0t, a0t, v0t, sct, vft,
            owt, oat, ovt, ogt, c, batch, pad, pad, pad, pad, pad, pad, pad, pad,
        )
        .expect("lowrank_gemm_batch");

        let mut worst = 0.0f32;
        for (name, t, ch) in [
            ("w", owt, 1usize),
            ("a", oat, 2),
            ("v", ovt, 0),
            ("g", ogt, 3),
        ] {
            let got = b.download(t).unwrap();
            for i in 0..batch * c {
                let e = (got[i] - ref_out[ch][i]).abs();
                worst = worst.max(e);
                assert!(
                    e <= 2e-3 * ref_out[ch][i].abs().max(1.0),
                    "lowrank_gemm {name} 链错位 at b={} m={}: got={} expect={} |e|={e}",
                    i / c,
                    i % c,
                    got[i],
                    ref_out[ch][i]
                );
            }
        }
        log::info!("★ lowrank_gemm_batch vs CPU OK（四链 worst_abs={worst:e}）");
    }

    /// ★ Phase 2-3 门禁：ffn_value 稠密 fp16 张量核 GEMM 对 CPU 参考。
    ///
    /// 参考口径与内核同源：`W`/`r2` 都按 fp16 舍入（内核的 B/A 就是 fp16），
    /// 累加在 CPU 上走 fp64 以隔离"累加顺序"差异，再比 fp16 输入舍入本身的误差。
    /// 容差 `2e-3·max(1,|ref|)`：k=10240 时 fp16 输入舍入的随机游走误差约
    /// `sqrt(k)·2⁻¹¹·term_rms ≈ 5e-4·|ref|`，留 4× 余量；A/B 布局错位或
    /// 「就地累加」写错地址都会给出 O(1) 误差。
    #[test]
    fn ffn_value_gemm_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping ffn_value_gemm test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        // 非 tile 整数倍的 batch（40）+ k 为 BK 整数倍（256）；n 取 128。
        let (c, fh, batch) = (128usize, 256usize, 40usize);
        let mut seed = 0xF0F0_2026u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let r16 = |v: f32| f16::from_f32(v).to_f32();
        // r2 = relu² 的输出语义（非负、可稀疏）；权重 [c, fh] 行主序。
        let r2: Vec<f32> = (0..batch * fh)
            .map(|_| {
                let v = rng();
                r16(if v > 0.2 { v * v } else { 0.0 })
            })
            .collect();
        let w: Vec<f32> = (0..c * fh).map(|_| r16(0.05 * rng())).collect();
        let x0: Vec<f32> = (0..batch * c).map(|_| r16(rng())).collect();

        // CPU 参考：x += Σ_f r2[b,f]·W[m,f]（fp64 累加，隔离顺序差异）
        let mut refx = x0.clone();
        for bb in 0..batch {
            for m in 0..c {
                let mut s = 0.0f64;
                for f in 0..fh {
                    s += r2[bb * fh + f] as f64 * w[m * fh + f] as f64;
                }
                refx[bb * c + m] = (refx[bb * c + m] as f64 + s) as f32;
            }
        }

        let wt = b.create_tensor(c * fh, TensorDtype::F16).unwrap();
        b.upload(wt, &w).unwrap();
        let rt = b.create_tensor(batch * fh, TensorDtype::F32).unwrap();
        b.upload(rt, &r2).unwrap();
        let xt = b.create_tensor(batch * c, TensorDtype::F32).unwrap();
        b.upload(xt, &x0).unwrap();

        b.ffn_value_gemm_batch(wt, rt, xt, c, fh, batch)
            .expect("ffn_value_gemm_batch");
        let got = b.download(xt).unwrap();

        let mut worst = 0.0f32;
        let mut sq = 0.0f64;
        for i in 0..batch * c {
            let e = (got[i] - refx[i]).abs();
            worst = worst.max(e);
            sq += (e as f64) * (e as f64);
            assert!(
                e <= 2e-3 * refx[i].abs().max(1.0),
                "ffn_value_gemm mismatch at b={} m={}: got={} expect={} |e|={e}",
                i / c,
                i % c,
                got[i],
                refx[i]
            );
        }
        log::info!(
            "★ ffn_value_gemm vs CPU OK（worst_abs={worst:e} rel_rms={:.3e}）",
            (sq / (batch * c) as f64).sqrt() / 10.0
        );
    }
}
