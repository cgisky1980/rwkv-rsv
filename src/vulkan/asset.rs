use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "assets"]
#[include = "shaders/spv/*.spv"]
pub struct Asset;

fn spv_dir() -> &'static Option<PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("VK_SPV_DIR").map(PathBuf::from))
}

/// 取着色器字节。
///
/// `VK_SPV_DIR=<dir>` 时**只**从 `<dir>/<basename>` 读盘（缺文件即报错，不静默回落），
/// 否则返回构建期嵌入的字节。
///
/// ★ 为什么必须有这个开关：release 构建把 `.spv` 编进可执行文件（上面的 `#[include]`），
/// 所以「用 glslang 重编译一个 `.spv` 再拷进 `assets/shaders/spv/`」在运行时**不生效**——
/// 那样做的 A/B 其实是同一份二进制自比，两臂的时间与数值恒等（2026-10-11 实测踩坑，
/// 一批「探针臂与基线逐格相同」的结论皆源于此）。
/// shader A/B 只有两条可信路径：每臂各自 `cargo build --release`，或本开关。
pub fn shader_bytes(name: &str) -> Result<Cow<'static, [u8]>, String> {
    if let Some(dir) = spv_dir() {
        let file = Path::new(name)
            .file_name()
            .ok_or_else(|| format!("bad shader name: {name}"))?;
        let path = dir.join(file);
        return std::fs::read(&path)
            .map(Cow::Owned)
            .map_err(|e| format!("VK_SPV_DIR shader not found: {} ({})", path.display(), e));
    }
    Asset::get(name)
        .map(|f| f.data)
        .ok_or_else(|| format!("embedded shader not found: {name}"))
}
