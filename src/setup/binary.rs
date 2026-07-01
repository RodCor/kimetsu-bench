//! Linux kimetsu binary resolution for +km runs.
//!
//! Terminal-Bench runs in Linux Docker containers, so kimetsu MUST be
//! a Linux ELF binary bind-mounted into the container. This module
//! finds (or builds) one with no user intervention.
//!
//! Strategy, in order:
//!   1. Use `cache/linux-build/release/kimetsu` if it exists and is
//!      newer than every `*.rs` / `*.toml` under `<repo_root>/crates/`.
//!   2. If `no_build` is set, return cached (even if stale) or None.
//!   3. WSL2 cargo build from current source (Windows only).
//!   4. Download latest GitHub release asset.
//!   5. Fall back to stale cache if all else fails.
//!
//! On Linux/macOS hosts the binary can be built natively, but the
//! cross-platform story is messier (you'd want musl-static on macOS).
//! Linux users typically run the bench from a Linux host where the
//! kimetsu binary built by `cargo build -p kimetsu-cli` is already
//! Linux-ELF, so we surface a clear error there instead of trying
//! to be clever.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub enum BinarySource {
    Cached(PathBuf),
    WslBuilt(PathBuf),
    GithubRelease { path: PathBuf, tag: String },
    StaleCache(PathBuf),
}

impl BinarySource {
    pub fn path(&self) -> &Path {
        match self {
            BinarySource::Cached(p)
            | BinarySource::WslBuilt(p)
            | BinarySource::StaleCache(p)
            | BinarySource::GithubRelease { path: p, .. } => p,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            BinarySource::Cached(p) => format!("cached (fresh): {}", p.display()),
            BinarySource::WslBuilt(p) => format!("WSL2 build: {}", p.display()),
            BinarySource::StaleCache(p) => format!("cached (stale): {}", p.display()),
            BinarySource::GithubRelease { path, tag } => {
                format!("GitHub release {tag}: {}", path.display())
            }
        }
    }
}

pub fn resolve(cache_dir: &Path, repo_root: &Path, no_build: bool) -> Option<BinarySource> {
    let cached_path = cache_dir
        .join("linux-build")
        .join("release")
        .join("kimetsu");

    if let Some(fresh) = cached_if_fresh(&cached_path, repo_root) {
        return Some(BinarySource::Cached(fresh));
    }

    if no_build {
        if cached_path.is_file() {
            return Some(BinarySource::StaleCache(cached_path));
        }
        return None;
    }

    // Windows: build via WSL2 (need a Linux ELF for the Docker container).
    #[cfg(windows)]
    if let Some(p) = try_wsl_build(cache_dir, repo_root) {
        return Some(BinarySource::WslBuilt(p));
    }
    // Linux/macOS: build natively. The output IS the Linux binary we need
    // (assuming we're running on Linux — on macOS the user should use the
    // GitHub release path instead, since the binary needs to be Linux ELF).
    #[cfg(target_os = "linux")]
    if let Some(p) = try_native_build(cache_dir, repo_root) {
        return Some(BinarySource::WslBuilt(p));
    }

    if let Some((p, tag)) = try_github_release(cache_dir) {
        return Some(BinarySource::GithubRelease { path: p, tag });
    }

    if cached_path.is_file() {
        return Some(BinarySource::StaleCache(cached_path));
    }
    None
}

fn cached_if_fresh(cached: &Path, repo_root: &Path) -> Option<PathBuf> {
    if !cached.is_file() {
        return None;
    }
    let cached_mtime = std::fs::metadata(cached).ok()?.modified().ok()?;

    let crates_dir = repo_root.join("crates");
    if !crates_dir.is_dir() {
        return Some(cached.to_path_buf());
    }

    match newest_source_mtime(&crates_dir) {
        Some(src) if cached_mtime >= src => Some(cached.to_path_buf()),
        Some(_) => None,
        None => Some(cached.to_path_buf()),
    }
}

fn newest_source_mtime(dir: &Path) -> Option<std::time::SystemTime> {
    let mut newest: Option<std::time::SystemTime> = None;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&p) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip build artifact dirs to keep this fast on large repos.
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(name, "target" | "node_modules" | ".git") {
                    continue;
                }
                stack.push(path);
            } else {
                let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("");
                if matches!(ext, "rs" | "toml")
                    && let Ok(meta) = std::fs::metadata(&path)
                    && let Ok(m) = meta.modified()
                {
                    newest = Some(match newest {
                        Some(prev) if prev >= m => prev,
                        _ => m,
                    });
                }
            }
        }
    }
    newest
}

#[cfg(windows)]
fn try_wsl_build(cache_dir: &Path, repo_root: &Path) -> Option<PathBuf> {
    if Command::new("wsl").arg("--status").output().is_err() {
        eprintln!("kbench: setup: wsl not available, skipping WSL2 build");
        return None;
    }
    let wsl_repo = to_wsl_path(repo_root)?;
    let wsl_target = format!("{}/linux-build", to_wsl_path(cache_dir)?);

    // Build lean (no ORT/fastembed) for Linux: embeddings requires glibc ≥2.38
    // (__isoc23_strtoll etc.) which ORT prebuilts depend on, but many WSL2
    // distros ship glibc 2.35 (Ubuntu 22.04) or older. The brain still works
    // inside the container — semantic dedup degrades to exact-only dedup.
    let script = format!(
        "cargo build --release -p kimetsu-cli \
         --no-default-features \
         --manifest-path '{wsl_repo}/Cargo.toml' \
         --target-dir '{wsl_target}'"
    );

    eprintln!("kbench: setup: building Linux kimetsu via WSL2 (a few minutes on first run)...");
    let status = Command::new("wsl")
        .args(["--", "bash", "-lc", &script])
        .status()
        .ok()?;
    if !status.success() {
        eprintln!(
            "kbench: setup: WSL2 build failed (exit {:?}); falling back",
            status.code()
        );
        return None;
    }

    let built = cache_dir
        .join("linux-build")
        .join("release")
        .join("kimetsu");
    if built.is_file() {
        Some(built)
    } else {
        eprintln!(
            "kbench: setup: WSL2 reported success but binary missing at {}",
            built.display()
        );
        None
    }
}

/// Native Linux build (used when host is Linux, including inside WSL2 Ubuntu).
/// Produces a Linux ELF directly — exactly what the Docker container needs.
#[cfg(target_os = "linux")]
fn try_native_build(cache_dir: &Path, repo_root: &Path) -> Option<PathBuf> {
    eprintln!(
        "kbench: setup: building Linux kimetsu via native cargo (a few minutes on first run)..."
    );
    let target_dir = cache_dir.join("linux-build");
    let manifest = repo_root.join("Cargo.toml");
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("-p")
        .arg("kimetsu-cli")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .status()
        .ok()?;
    if !status.success() {
        eprintln!(
            "kbench: setup: native cargo build failed (exit {:?})",
            status.code()
        );
        return None;
    }
    let built = target_dir.join("release").join("kimetsu");
    if built.is_file() {
        Some(built)
    } else {
        eprintln!(
            "kbench: setup: cargo reported success but binary missing at {}",
            built.display()
        );
        None
    }
}

#[cfg(windows)]
fn to_wsl_path(p: &Path) -> Option<String> {
    let canon = std::fs::canonicalize(p).ok()?;
    let raw = canon.to_string_lossy().to_string();
    let raw = raw.strip_prefix(r"\\?\").unwrap_or(&raw).to_string();
    let raw = raw.replace('\\', "/");
    let bytes = raw.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        let rest = &raw[2..];
        Some(format!("/mnt/{drive}{rest}"))
    } else {
        Some(raw)
    }
}

fn try_github_release(cache_dir: &Path) -> Option<(PathBuf, String)> {
    eprintln!("kbench: setup: trying latest GitHub release download...");

    let api_url = "https://api.github.com/repos/RodCor/kimetsu/releases/latest";
    let response = ureq::get(api_url)
        .set("User-Agent", "kimetsu-kbench")
        .set("Accept", "application/vnd.github+json")
        .call();

    let body: serde_json::Value = match response {
        Ok(r) => match r.into_json() {
            Ok(j) => j,
            Err(e) => {
                eprintln!("kbench: setup: GitHub response not JSON: {e}");
                return None;
            }
        },
        Err(e) => {
            eprintln!("kbench: setup: GitHub API call failed: {e}");
            return None;
        }
    };

    let tag = body
        .get("tag_name")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let assets = body.get("assets")?.as_array()?;

    // Prefer the "lean" build (no embeddings) — smaller, faster to download.
    let pick = assets
        .iter()
        .find(|a| {
            a.get("name")
                .and_then(|n| n.as_str())
                .map(|n| n.ends_with("x86_64-unknown-linux-gnu-lean.tar.gz"))
                .unwrap_or(false)
        })
        .or_else(|| {
            assets.iter().find(|a| {
                a.get("name")
                    .and_then(|n| n.as_str())
                    .map(|n| n.contains("x86_64-unknown-linux-gnu") && n.ends_with(".tar.gz"))
                    .unwrap_or(false)
            })
        })?;
    let asset_name = pick.get("name")?.as_str()?;
    let url = pick.get("browser_download_url")?.as_str()?;

    eprintln!("kbench: setup: downloading {asset_name} (tag {tag})...");

    let tar_path = cache_dir.join("kimetsu-release.tar.gz");
    let extract_dir = cache_dir.join("linux-extract");
    std::fs::create_dir_all(&extract_dir).ok()?;

    let resp = match ureq::get(url).call() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("kbench: setup: download failed: {e}");
            return None;
        }
    };
    let mut file = std::fs::File::create(&tar_path).ok()?;
    if let Err(e) = std::io::copy(&mut resp.into_reader(), &mut file) {
        eprintln!("kbench: setup: writing release tarball failed: {e}");
        return None;
    }
    drop(file);

    let extract_status = Command::new("tar")
        .args([
            "-xzf",
            &tar_path.to_string_lossy(),
            "-C",
            &extract_dir.to_string_lossy(),
        ])
        .status()
        .ok()?;
    if !extract_status.success() {
        eprintln!("kbench: setup: tar extract failed");
        return None;
    }

    let kimetsu_path = find_kimetsu_binary(&extract_dir)?;
    Some((kimetsu_path, tag))
}

fn find_kimetsu_binary(dir: &Path) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&p) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("kimetsu") {
                return Some(path);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"x").unwrap();
    }

    #[test]
    fn cached_fresh_when_newer_than_sources() {
        let root = std::env::temp_dir().join(format!("kbench-bin-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let crates = root.join("crates").join("foo").join("src");
        let bin = root
            .join("cache")
            .join("linux-build")
            .join("release")
            .join("kimetsu");
        touch(&crates.join("lib.rs"));
        // Sleep is unreliable; just touch bin LAST so it's newer.
        std::thread::sleep(std::time::Duration::from_millis(20));
        touch(&bin);

        let resolved = cached_if_fresh(&bin, &root);
        assert!(resolved.is_some(), "binary should be fresh");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cached_stale_when_older_than_sources() {
        let root = std::env::temp_dir().join(format!("kbench-bin-test2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let crates = root.join("crates").join("foo").join("src");
        let bin = root
            .join("cache")
            .join("linux-build")
            .join("release")
            .join("kimetsu");
        touch(&bin);
        std::thread::sleep(std::time::Duration::from_millis(20));
        touch(&crates.join("lib.rs"));

        let resolved = cached_if_fresh(&bin, &root);
        assert!(resolved.is_none(), "binary should be stale");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cached_returns_none_when_missing() {
        let root = std::env::temp_dir().join(format!("kbench-bin-test3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert!(cached_if_fresh(&root.join("missing"), &root).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
