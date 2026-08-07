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
use std::ffi::c_void;
use std::sync::OnceLock;

use half::f16;

use crate::backend::{Any4Handle, ComputeBackend, Int8Handle, TensorDtype, TensorId};
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
type CuResult = i32;
const CUDA_SUCCESS: CuResult = 0;

/// 加载并持有 CUDA 驱动函数指针（一次加载，全进程共享）。
struct CudaDriver {
    _lib: libloading::Library,
    cu_init: unsafe extern "C" fn(u32) -> CuResult,
    cu_device_get_count: unsafe extern "C" fn(*mut i32) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> CuResult,
    cu_primary_ctx_retain: unsafe extern "C" fn(*mut *mut c_void, i32) -> CuResult,
    cu_primary_ctx_release: unsafe extern "C" fn(i32) -> CuResult,
    cu_mem_alloc_v2: unsafe extern "C" fn(*mut u64, usize) -> CuResult,
    cu_mem_free_v2: unsafe extern "C" fn(u64) -> CuResult,
    cu_memcpy_htod_v2: unsafe extern "C" fn(u64, *const c_void, usize) -> CuResult,
    cu_memcpy_dtoh_v2: unsafe extern "C" fn(*mut c_void, u64, usize) -> CuResult,
}

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
        let cu_init = unsafe { sym(&lib, "cuInit", &[b"cuInit\0"]) }?;
        let cu_device_get_count =
            unsafe { sym(&lib, "cuDeviceGetCount", &[b"cuDeviceGetCount\0"]) }?;
        let cu_device_get = unsafe { sym(&lib, "cuDeviceGet", &[b"cuDeviceGet\0"]) }?;
        let cu_primary_ctx_retain =
            unsafe { sym(&lib, "cuPrimaryCtxRetain", &[b"cuPrimaryCtxRetain\0"]) }?;
        let cu_primary_ctx_release =
            unsafe { sym(&lib, "cuPrimaryCtxRelease", &[b"cuPrimaryCtxRelease\0"]) }?;
        let cu_mem_alloc_v2 =
            unsafe { sym(&lib, "cuMemAlloc", &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"])? };
        let cu_mem_free_v2 =
            unsafe { sym(&lib, "cuMemFree", &[b"cuMemFree_v2\0", b"cuMemFree\0"])? };
        let cu_memcpy_htod_v2 = unsafe {
            sym(
                &lib,
                "cuMemcpyHtoD",
                &[b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"],
            )?
        };
        let cu_memcpy_dtoh_v2 = unsafe {
            sym(
                &lib,
                "cuMemcpyDtoH",
                &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"],
            )?
        };
        Ok(Self {
            _lib: lib,
            cu_init,
            cu_device_get_count,
            cu_device_get,
            cu_primary_ctx_retain,
            cu_primary_ctx_release,
            cu_mem_alloc_v2,
            cu_mem_free_v2,
            cu_memcpy_htod_v2,
            cu_memcpy_dtoh_v2,
        })
    }
}

/// 全局 CUDA 驱动（惰性加载一次）。
fn driver() -> Option<&'static CudaDriver> {
    static D: OnceLock<Option<CudaDriver>> = OnceLock::new();
    D.get_or_init(|| CudaDriver::open().ok()).as_ref()
}

/// 探测 CUDA 是否可用：驱动可加载、`cuInit` 成功且存在 ≥1 个设备。
pub fn cuda_available() -> bool {
    let Some(d) = driver() else {
        return false;
    };
    let mut count: i32 = 0;
    if unsafe { (d.cu_init)(0) } != CUDA_SUCCESS {
        return false;
    }
    if unsafe { (d.cu_device_get_count)(&mut count) } != CUDA_SUCCESS {
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

/// CUDA 后端骨架：持有驱动 + 主上下文 + 张量映射。
pub struct CudaBackend {
    drv: &'static CudaDriver,
    /// 主上下文句柄：`cuPrimaryCtxRetain` 保留其存活，进程退出前由 `Drop` 释放。
    #[allow(dead_code)]
    ctx: *mut c_void,
    device: i32,
    tensors: HashMap<TensorId, CudaTensor>,
    lens: HashMap<TensorId, usize>,
    next_id: u32,
}

impl CudaBackend {
    /// 创建 CUDA 后端：初始化驱动、取首个设备、保留主上下文。
    pub fn new() -> R<Self> {
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
        Ok(Self {
            drv,
            ctx,
            device,
            tensors: HashMap::new(),
            lens: HashMap::new(),
            next_id: 0,
        })
    }

    fn alloc(&self, bytes: usize) -> R<u64> {
        let mut dptr: u64 = 0;
        cu_check!((self.drv.cu_mem_alloc_v2)(&mut dptr, bytes), "cuMemAlloc");
        Ok(dptr)
    }

    fn memcpy_htod(&self, dptr: u64, data: &[u8]) -> R<()> {
        cu_check!(
            (self.drv.cu_memcpy_htod_v2)(dptr, data.as_ptr() as *const c_void, data.len()),
            "cuMemcpyHtoD"
        );
        Ok(())
    }

    fn memcpy_dtoh(&self, dptr: u64, out: &mut [u8]) -> R<()> {
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

    /// 算子未实现（骨架）：返回统一错误，注明对齐 Albatross 后续实现。
    fn unimplemented(&self, op: &str) -> R<()> {
        Err(format!("CudaBackend: {op} not yet implemented (对齐 Albatross 后续补齐)").into())
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        for v in self.tensors.values() {
            let dptr = match v {
                CudaTensor::F32 { dptr, .. }
                | CudaTensor::F16 { dptr, .. }
                | CudaTensor::U32 { dptr, .. } => *dptr,
            };
            unsafe {
                (self.drv.cu_mem_free_v2)(dptr);
            }
        }
        unsafe {
            (self.drv.cu_primary_ctx_release)(self.device);
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn create_tensor(&mut self, len: usize, dtype: TensorDtype) -> R<TensorId> {
        let id = TensorId(self.next_id);
        self.next_id += 1;
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
                if data.len() != len {
                    return Err(format!("upload: len mismatch ({} != {len})", data.len()).into());
                }
                let bytes = bytemuck::cast_slice::<f32, u8>(data);
                self.memcpy_htod(dptr, bytes)
            }
            CudaTensor::F16 { dptr, len } => {
                if data.len() != len {
                    return Err(
                        format!("upload(f16): len mismatch ({} != {len})", data.len()).into(),
                    );
                }
                let f16s: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                let bytes = bytemuck::cast_slice::<f16, u8>(&f16s);
                self.memcpy_htod(dptr, bytes)
            }
            CudaTensor::U32 { .. } => Err("upload: u32 tensor requires upload_u32".into()),
        }
    }

    fn upload_u32(&self, t: TensorId, data: &[u32]) -> R<()> {
        match self.get(t, "upload_u32")? {
            CudaTensor::U32 { dptr, len } => {
                if data.len() != len {
                    return Err(
                        format!("upload_u32: len mismatch ({} != {len})", data.len()).into(),
                    );
                }
                let bytes = bytemuck::cast_slice::<u32, u8>(data);
                self.memcpy_htod(dptr, bytes)
            }
            _ => Err("upload_u32: t must be u32".into()),
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
        // 骨架：CUDA stream 语义下为空操作（后续接 stream）。
        Ok(())
    }

    fn end_batch(&mut self) -> R<()> {
        Ok(())
    }

    fn store_token_host(&self, _tok: TensorId, _token: u32) -> R<()> {
        self.unimplemented("store_token_host")
    }
    fn gather_row_device_f16(
        &mut self,
        _s: TensorId,
        _d: TensorId,
        _t: TensorId,
        _c: usize,
    ) -> R<()> {
        self.unimplemented("gather_row_device_f16")
    }
    fn copy_device_f16(&mut self, _s: TensorId, _d: TensorId) -> R<()> {
        self.unimplemented("copy_device_f16")
    }
    fn gemv_f16(
        &mut self,
        _w: TensorId,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> R<()> {
        self.unimplemented("gemv_f16")
    }
    fn norm(
        &mut self,
        _x: TensorId,
        _g: TensorId,
        _b: TensorId,
        _y: TensorId,
        _c: usize,
        _h: usize,
        _eps: f32,
        _rows: usize,
    ) -> R<()> {
        self.unimplemented("norm")
    }
    fn norm_lerp6(
        &mut self,
        _x: TensorId,
        _s: TensorId,
        _g: TensorId,
        _b: TensorId,
        _xr: TensorId,
        _xw: TensorId,
        _xk: TensorId,
        _xv: TensorId,
        _xa: TensorId,
        _xg: TensorId,
        _or: TensorId,
        _ow: TensorId,
        _ok: TensorId,
        _ov: TensorId,
        _oa: TensorId,
        _og: TensorId,
        _c: usize,
        _eps: f32,
    ) -> R<()> {
        self.unimplemented("norm_lerp6")
    }
    fn cmix_norm_lerp(
        &mut self,
        _x: TensorId,
        _s: TensorId,
        _g: TensorId,
        _b: TensorId,
        _coeff: TensorId,
        _oxb: TensorId,
        _c: usize,
        _eps: f32,
    ) -> R<()> {
        self.unimplemented("cmix_norm_lerp")
    }
    fn fuse_ka_dplr_norm(
        &mut self,
        _s: TensorId,
        _k: TensorId,
        _kk: TensorId,
        _a: TensorId,
        _ka: TensorId,
        _r: TensorId,
        _v: TensorId,
        _w: TensorId,
        _g: TensorId,
        _b: TensorId,
        _rk: TensorId,
        _km: TensorId,
        _y: TensorId,
        _yn: TensorId,
        _h: usize,
        _n: usize,
        _eps: f32,
        _ge: f32,
    ) -> R<()> {
        self.unimplemented("fuse_ka_dplr_norm")
    }
    fn gemv_rkv_stage1(
        &mut self,
        _r: TensorId,
        _k: TensorId,
        _v: TensorId,
        _v1: TensorId,
        _w1: TensorId,
        _a1: TensorId,
        _g1: TensorId,
        _xr: TensorId,
        _xk: TensorId,
        _xv: TensorId,
        _xw: TensorId,
        _xa: TensorId,
        _xg: TensorId,
        _or: TensorId,
        _ok: TensorId,
        _ov: TensorId,
        _ovm: TensorId,
        _owm: TensorId,
        _oam: TensorId,
        _ogm: TensorId,
        _c: usize,
        _vm: usize,
        _wm: usize,
        _am: usize,
        _gm: usize,
    ) -> R<()> {
        self.unimplemented("gemv_rkv_stage1")
    }
    fn gemv_any4_rkv_stage1(
        &mut self,
        _r: &Any4Handle,
        _k: &Any4Handle,
        _v: &Any4Handle,
        _v1: TensorId,
        _w1: TensorId,
        _a1: TensorId,
        _g1: TensorId,
        _xr: TensorId,
        _xk: TensorId,
        _xv: TensorId,
        _xw: TensorId,
        _xa: TensorId,
        _xg: TensorId,
        _or: TensorId,
        _ok: TensorId,
        _ov: TensorId,
        _ovm: TensorId,
        _owm: TensorId,
        _oam: TensorId,
        _ogm: TensorId,
        _c: usize,
        _vm: usize,
        _wm: usize,
        _am: usize,
        _gm: usize,
    ) -> R<()> {
        self.unimplemented("gemv_any4_rkv_stage1")
    }
    fn gemv_int8_rkv_stage1(
        &mut self,
        _r: &Int8Handle,
        _k: &Int8Handle,
        _v: &Int8Handle,
        _v1: TensorId,
        _w1: TensorId,
        _a1: TensorId,
        _g1: TensorId,
        _xr: TensorId,
        _xk: TensorId,
        _xv: TensorId,
        _xw: TensorId,
        _xa: TensorId,
        _xg: TensorId,
        _or: TensorId,
        _ok: TensorId,
        _ov: TensorId,
        _ovm: TensorId,
        _owm: TensorId,
        _oam: TensorId,
        _ogm: TensorId,
        _c: usize,
        _vm: usize,
        _wm: usize,
        _am: usize,
        _gm: usize,
    ) -> R<()> {
        self.unimplemented("gemv_int8_rkv_stage1")
    }
    fn gemv_lowrank_chain4(
        &mut self,
        _w2: TensorId,
        _a2: TensorId,
        _v2: TensorId,
        _g2: TensorId,
        _wm: TensorId,
        _am: TensorId,
        _vm: TensorId,
        _gm: TensorId,
        _w0: TensorId,
        _a0: TensorId,
        _v0: TensorId,
        _scale: TensorId,
        _vf: TensorId,
        _ow: TensorId,
        _oa: TensorId,
        _ov: TensorId,
        _og: TensorId,
        _m: usize,
        _kw: usize,
        _ka: usize,
        _kv: usize,
        _kg: usize,
    ) -> R<()> {
        self.unimplemented("gemv_lowrank_chain4")
    }
    fn gemv_f16_relu2(
        &mut self,
        _a: TensorId,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_f16_relu2")
    }
    fn gemv_any4_relu2(
        &mut self,
        _a: &Any4Handle,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_any4_relu2")
    }
    fn gemv_int8_relu2(
        &mut self,
        _a: &Int8Handle,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_int8_relu2")
    }
    fn gemv_f16_mul_add(
        &mut self,
        _a: TensorId,
        _x: TensorId,
        _g: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_f16_mul_add")
    }
    fn gemv_any4_mul_add(
        &mut self,
        _a: &Any4Handle,
        _x: TensorId,
        _g: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_any4_mul_add")
    }
    fn gemv_int8_mul_add(
        &mut self,
        _a: &Int8Handle,
        _x: TensorId,
        _g: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_int8_mul_add")
    }
    fn gemv_f16_add(
        &mut self,
        _a: TensorId,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_f16_add")
    }
    fn gemv_any4_add(
        &mut self,
        _a: &Any4Handle,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_any4_add")
    }
    fn gemv_int8_add(
        &mut self,
        _a: &Int8Handle,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _b: usize,
    ) -> R<()> {
        self.unimplemented("gemv_int8_add")
    }
    fn argmax(&mut self, _logits: TensorId, _token: TensorId, _n: usize) -> R<()> {
        self.unimplemented("argmax")
    }
    fn sample(
        &mut self,
        _logits: TensorId,
        _token: TensorId,
        _n: usize,
        _t: f32,
        _tk: u32,
        _tp: f32,
        _seed: u32,
        _rp: f32,
        _fp: f32,
        _pp: f32,
        _hist: &[u32],
    ) -> R<()> {
        self.unimplemented("sample")
    }
    fn clear_cache(&mut self) {}
    fn drop_host(&mut self, _t: TensorId) {}
    fn copy_device(&mut self, _src: TensorId, _dst: TensorId) -> R<()> {
        self.unimplemented("copy_device")
    }
    fn copy_token(
        &mut self,
        _x: TensorId,
        _y: TensorId,
        _c: usize,
        _stride: usize,
        _token: usize,
    ) -> R<()> {
        self.unimplemented("copy_token")
    }
    fn gemm(
        &mut self,
        _a: TensorId,
        _b: TensorId,
        _c: TensorId,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> R<()> {
        self.unimplemented("gemm")
    }
    fn gemm_bias(
        &mut self,
        _a: TensorId,
        _b: TensorId,
        _bias: TensorId,
        _c: TensorId,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> R<()> {
        self.unimplemented("gemm_bias")
    }
    fn gemm_add(
        &mut self,
        _a: TensorId,
        _b: TensorId,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> R<()> {
        self.unimplemented("gemm_add")
    }
    fn to_f16(
        &mut self,
        _x: TensorId,
        _y: TensorId,
        _c: usize,
        _t: usize,
        _mp: usize,
        _xs: usize,
        _ys: usize,
    ) -> R<()> {
        self.unimplemented("to_f16")
    }
    fn to_f16_triple(
        &mut self,
        _xr: TensorId,
        _xk: TensorId,
        _xv: TensorId,
        _yr: TensorId,
        _yk: TensorId,
        _yv: TensorId,
        _c: usize,
        _t: usize,
        _mp: usize,
        _xs: usize,
        _ys: usize,
    ) -> R<()> {
        self.unimplemented("to_f16_triple")
    }
    fn dequant_any4_to_f16(
        &mut self,
        _a: &Any4Handle,
        _out: TensorId,
        _m: usize,
        _k: usize,
    ) -> R<()> {
        self.unimplemented("dequant_any4_to_f16")
    }
    fn dequant_int8_to_f16(
        &mut self,
        _a: &Int8Handle,
        _out: TensorId,
        _m: usize,
        _k: usize,
    ) -> R<()> {
        self.unimplemented("dequant_int8_to_f16")
    }
    fn elementwise_sigmoid(&mut self, _a: TensorId, _y: TensorId, _c: usize, _b: usize) -> R<()> {
        self.unimplemented("elementwise_sigmoid")
    }
    fn elementwise_sigmoid_inplace(&mut self, _y: TensorId, _c: usize, _b: usize) -> R<()> {
        self.unimplemented("elementwise_sigmoid_inplace")
    }
    fn fuse_ka(
        &mut self,
        _k: TensorId,
        _kk: TensorId,
        _a: TensorId,
        _ka: TensorId,
        _km: TensorId,
        _kkl: TensorId,
        _b: TensorId,
        _h: usize,
        _n: usize,
        _batch: usize,
    ) -> R<()> {
        self.unimplemented("fuse_ka")
    }
    fn sum_rk_rk(
        &mut self,
        _r: TensorId,
        _km: TensorId,
        _rk: TensorId,
        _v: TensorId,
        _y: TensorId,
        _h: usize,
        _n: usize,
        _batch: usize,
    ) -> R<()> {
        self.unimplemented("sum_rk_rk")
    }
    fn seq_shift(
        &mut self,
        _x: TensorId,
        _s: TensorId,
        _tm: TensorId,
        _y: TensorId,
        _c: usize,
        _t: usize,
        _sx: usize,
        _sy: usize,
    ) -> R<()> {
        self.unimplemented("seq_shift")
    }
    fn dplr_seq(
        &mut self,
        _s: TensorId,
        _r: TensorId,
        _w: TensorId,
        _k: TensorId,
        _v: TensorId,
        _a: TensorId,
        _b: TensorId,
        _y: TensorId,
        _h: usize,
        _n: usize,
        _t: usize,
        _c: usize,
    ) -> R<()> {
        self.unimplemented("dplr_seq")
    }
    fn gemm_any4(
        &mut self,
        _a: &Any4Handle,
        _x: TensorId,
        _res: Option<TensorId>,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _t: usize,
        _act: bool,
    ) -> R<()> {
        self.unimplemented("gemm_any4")
    }
    fn gemm_relu2(
        &mut self,
        _a: TensorId,
        _b: TensorId,
        _c: TensorId,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> R<()> {
        self.unimplemented("gemm_relu2")
    }
    fn gemm_tanh(
        &mut self,
        _a: TensorId,
        _b: TensorId,
        _c: TensorId,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> R<()> {
        self.unimplemented("gemm_tanh")
    }
    fn elementwise_scale_exp(
        &mut self,
        _a: TensorId,
        _b: TensorId,
        _y: TensorId,
        _c: usize,
        _batch: usize,
    ) -> R<()> {
        self.unimplemented("elementwise_scale_exp")
    }
    fn elementwise_mul(
        &mut self,
        _a: TensorId,
        _b: TensorId,
        _y: TensorId,
        _c: usize,
        _batch: usize,
    ) -> R<()> {
        self.unimplemented("elementwise_mul")
    }
    fn v_first_lerp(
        &mut self,
        _v: TensorId,
        _g: TensorId,
        _vf: TensorId,
        _c: usize,
        _t: usize,
        _s: usize,
    ) -> R<()> {
        self.unimplemented("v_first_lerp")
    }
    fn gemv_seq(
        &mut self,
        _a: TensorId,
        _x: TensorId,
        _y: TensorId,
        _m: usize,
        _k: usize,
        _xs: usize,
        _ys: usize,
        _batch: usize,
    ) -> R<()> {
        self.unimplemented("gemv_seq")
    }
    fn store_sampler_host(
        &self,
        _sampler: TensorId,
        _t: f32,
        _tk: u32,
        _tp: f32,
        _seed: u32,
        _rp: f32,
        _fp: f32,
        _pp: f32,
        _hl: u32,
    ) -> R<()> {
        self.unimplemented("store_sampler_host")
    }
    fn sample_into_host_seeded(
        &mut self,
        _logits: TensorId,
        _token: TensorId,
        _n: usize,
        _temp: TensorId,
        _mask: TensorId,
        _counter: TensorId,
        _sampler: TensorId,
        _hist: TensorId,
    ) -> R<()> {
        self.unimplemented("sample_into_host_seeded")
    }
    fn record_token(&mut self, _in_tok: TensorId, _out_seq: TensorId, _cnt: TensorId) -> R<()> {
        self.unimplemented("record_token")
    }
    fn argmax_into_host(&mut self, _logits: TensorId, _token: TensorId, _n: usize) -> R<()> {
        self.unimplemented("argmax_into_host")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
