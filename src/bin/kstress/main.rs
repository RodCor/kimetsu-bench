//! `kstress` — kimetsu brain stress test (local v1.0.0 only).
//!
//! Seeds the brain at 100 → 1,000,000 memories and profiles time, db size,
//! concurrency (local) and HTTP throughput (remote `kimetsu-remote`), across
//! the lean (FTS) and embeddings (vec0 ANN) matrices. Everything builds from
//! the local `../` workspace — never a published artifact. Output:
//! `runs/stress/<run-id>/<mode>-<matrix>/{summary.json,report.md,data.csv}`.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};

mod corpus;
mod local;
mod remote;
mod report;
mod seed;

use report::{HostInfo, StressReport};

#[derive(Parser)]
#[command(name = "kstress", about = "kimetsu brain stress test (local v1.0.0)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// In-process local profiler (insert/query latency, db size, concurrency).
    Local(LocalArgs),
    /// Remote HTTP profiler against a locally-built `kimetsu-remote serve`.
    Remote(RemoteArgs),
}

#[derive(Args)]
struct Common {
    /// "lean" (FTS-only / NoopEmbedder) or "emb" (fastembed BGE-small + vec0).
    #[arg(long, default_value = "lean")]
    matrix: String,
    /// Comma-separated memory-count checkpoints.
    #[arg(
        long,
        default_value = "100,500,5000,50000,100000,1000000",
        value_delimiter = ','
    )]
    scales: Vec<usize>,
    /// Drop checkpoints above this (cap the long pole, e.g. emb to 100000).
    #[arg(long)]
    max_scale: Option<usize>,
    /// Shared run id (timestamp) so sibling invocations group under one dir.
    #[arg(long)]
    run_id: Option<String>,
    /// Deterministic seed for the synthetic corpus.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Bulk-insert batch size (rows per transaction). Larger = fewer commits =
    /// less FTS5 re-merge overhead during seeding.
    #[arg(long, default_value_t = 25_000)]
    batch_size: usize,

    /// Base dir for the throwaway working brains. Defaults to `.cache/stress/`
    /// on the bench drive (E:, honors no-C:). On WSL2 that's a 9p mount where
    /// large-DB query latency is dominated by filesystem overhead, NOT the
    /// brain — point this at a native ext4 path (e.g. `/tmp/kstress` or
    /// `~/.cache/kstress`) for realistic numbers at high scales.
    #[arg(long)]
    work: Option<PathBuf>,

    /// Keep the seeded working brains after the run (default: delete them;
    /// reports are unaffected).
    #[arg(long)]
    keep_work: bool,
}

#[derive(Args)]
struct LocalArgs {
    #[command(flatten)]
    common: Common,
    /// Above this scale, skip the realistic add_memory sample + writer scenario.
    #[arg(long, default_value_t = 100_000)]
    realistic_max_scale: usize,
    /// Reader-thread levels for the concurrency probe.
    #[arg(long, default_value = "1,4,16", value_delimiter = ',')]
    readers: Vec<usize>,
    /// Concurrency window per scenario, in milliseconds.
    #[arg(long, default_value_t = 1500)]
    window_ms: u64,
}

#[derive(Args)]
struct RemoteArgs {
    #[command(flatten)]
    common: Common,
    /// Path to the locally-built kimetsu-remote binary.
    #[arg(long, default_value = "../target/release/kimetsu-remote")]
    remote_bin: PathBuf,
    /// HTTP client concurrency levels.
    #[arg(long, default_value = "1,4,16,64", value_delimiter = ',')]
    concurrency: Vec<usize>,
    /// Server --max-blocking-threads.
    #[arg(long, default_value_t = 64)]
    max_blocking_threads: usize,
    /// Load window per concurrency level, in milliseconds.
    #[arg(long, default_value_t = 2000)]
    window_ms: u64,
}

fn main() {
    let cli = Cli::parse();
    let code = match &cli.cmd {
        Cmd::Local(a) => run_local(a),
        Cmd::Remote(a) => run_remote(a),
    };
    if let Err(e) = code {
        eprintln!("kstress: error: {e}");
        std::process::exit(1);
    }
}

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    bench_dir()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(bench_dir)
}

/// Default embedding model for the emb matrix (overridable by pre-setting
/// `KIMETSU_BRAIN_EMBEDDER` to another built-in id, e.g. `bge-m3`).
const EMB_DEFAULT_MODEL: &str = "bge-small-en-v1.5";

/// Pin the matrix's embedder via `KIMETSU_BRAIN_EMBEDDER` BEFORE any embedder
/// is built (`open_default_embedder` caches on first call). The env override
/// is authoritative over the project's `[embedder] enabled` config — a freshly
/// init'd project defaults that to off, so without this the emb query path
/// would silently fall back to FTS-only and never build the vec0 ANN index.
fn apply_matrix_env(matrix: &str) {
    if std::env::var("KIMETSU_BRAIN_EMBEDDER").is_ok() {
        return; // caller pinned a model explicitly — respect it.
    }
    let value = if matrix == "emb" {
        EMB_DEFAULT_MODEL
    } else {
        "noop"
    };
    // SAFETY: set before any thread spawns / embedder initializes.
    unsafe { std::env::set_var("KIMETSU_BRAIN_EMBEDDER", value) };
}

fn scales_of(common: &Common) -> Vec<usize> {
    let mut s: Vec<usize> = common
        .scales
        .iter()
        .copied()
        .filter(|&n| common.max_scale.map(|m| n <= m).unwrap_or(true))
        .collect();
    s.sort_unstable();
    s.dedup();
    s
}

fn now_stamp() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "now".into())
        .replace(':', "-")
}

fn git_sha() -> String {
    Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn host_info(matrix: &str) -> HostInfo {
    let embedder = kimetsu_brain::embeddings::open_default_embedder();
    HostInfo {
        embedder_model: if matrix == "emb" {
            embedder.model_id().to_string()
        } else {
            "noop".into()
        },
        embedder_dim: embedder.dim(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

fn out_dir(common: &Common, mode: &str) -> PathBuf {
    let run_id = common.run_id.clone().unwrap_or_else(now_stamp);
    bench_dir()
        .join("local")
        .join("runs")
        .join("stress")
        .join(run_id)
        .join(format!("{mode}-{}", common.matrix))
}

fn work_dir(common: &Common, mode: &str) -> PathBuf {
    // Working brains default to the bench drive (E:, honors no-C:). `--work`
    // overrides with a native ext4 path for realistic high-scale numbers.
    let run_id = common.run_id.clone().unwrap_or_else(now_stamp);
    let base = common
        .work
        .clone()
        .unwrap_or_else(|| bench_dir().join(".cache").join("stress"));
    base.join(run_id).join(format!("{mode}-{}", common.matrix))
}

/// Warn when the working brains land on a WSL drvfs/9p mount (`/mnt/...`) at a
/// scale where the 9p I/O overhead dominates the measurement (it's ~12-33x
/// slower than ext4). Points the operator at `--work <ext4 path>`.
fn warn_if_slow_fs(work: &std::path::Path, scales: &[usize]) {
    let on_drvfs = work.components().any(|c| c.as_os_str() == "mnt")
        && work.to_string_lossy().starts_with("/mnt/");
    let max = scales.iter().copied().max().unwrap_or(0);
    if on_drvfs && max >= 50_000 {
        eprintln!(
            "kstress: WARNING — working brains are on a 9p drvfs mount ({}). At {max} \
             memories, query/seed latency reflects the FILESYSTEM (~12-33x slower than \
             ext4), not the brain. For realistic numbers pass --work <native ext4 path> \
             (e.g. --work $HOME/.cache/kstress); reports still go to runs/stress on E:.",
            work.display()
        );
    }
}

fn finalize(report: StressReport, common: &Common, mode: &str) -> Result<(), String> {
    let out = out_dir(common, mode);
    let json = report.write_all(&out).map_err(|e| e.to_string())?;
    println!("{}", report.to_markdown());
    eprintln!("kstress: report saved -> {}", json.display());
    Ok(())
}

fn run_local(a: &LocalArgs) -> Result<(), String> {
    apply_matrix_env(&a.common.matrix);
    let scales = scales_of(&a.common);
    let cfg = local::LocalCfg {
        root: work_dir(&a.common, "local"),
        scales,
        matrix: a.common.matrix.clone(),
        seed: a.common.seed,
        batch_size: a.common.batch_size,
        realistic_max_scale: a.realistic_max_scale,
        realistic_sample: 50,
        query_sample: 50,
        reader_levels: a.readers.clone(),
        window: Duration::from_millis(a.window_ms),
    };
    eprintln!(
        "kstress local [{}] scales={:?} -> {}",
        cfg.matrix,
        cfg.scales,
        cfg.root.display()
    );
    warn_if_slow_fs(&cfg.root, &cfg.scales);
    let work_root = cfg.root.clone();
    let checkpoints = local::run(&cfg)?;
    let report = StressReport {
        kimetsu_version: "1.0.0".into(),
        kimetsu_git_sha: git_sha(),
        generated_at: now_stamp(),
        matrix: a.common.matrix.clone(),
        mode: "local".into(),
        host: host_info(&a.common.matrix),
        checkpoints,
    };
    finalize(report, &a.common, "local")?;
    if !a.common.keep_work {
        match std::fs::remove_dir_all(&work_root) {
            Ok(()) => eprintln!(
                "kstress: removed working brains at {} (pass --keep-work to retain)",
                work_root.display()
            ),
            Err(e) => eprintln!(
                "kstress: WARNING — could not remove working brains at {}: {e}",
                work_root.display()
            ),
        }
    }
    Ok(())
}

fn run_remote(a: &RemoteArgs) -> Result<(), String> {
    apply_matrix_env(&a.common.matrix);
    let scales = scales_of(&a.common);
    let cfg = remote::RemoteCfg {
        remote_bin: a.remote_bin.clone(),
        data_dir: work_dir(&a.common, "remote"),
        repo: "kstress".into(),
        token: "kstress-token".into(),
        scales,
        matrix: a.common.matrix.clone(),
        seed: a.common.seed,
        batch_size: a.common.batch_size,
        concurrency: a.concurrency.clone(),
        max_blocking_threads: a.max_blocking_threads,
        window: Duration::from_millis(a.window_ms),
    };
    eprintln!(
        "kstress remote [{}] scales={:?} bin={} -> {}",
        cfg.matrix,
        cfg.scales,
        cfg.remote_bin.display(),
        cfg.data_dir.display()
    );
    warn_if_slow_fs(&cfg.data_dir, &cfg.scales);
    let work_root = cfg.data_dir.clone();
    let checkpoints = remote::run(&cfg)?;
    let report = StressReport {
        kimetsu_version: "1.0.0".into(),
        kimetsu_git_sha: git_sha(),
        generated_at: now_stamp(),
        matrix: a.common.matrix.clone(),
        mode: "remote".into(),
        host: host_info(&a.common.matrix),
        checkpoints,
    };
    finalize(report, &a.common, "remote")?;
    if !a.common.keep_work {
        match std::fs::remove_dir_all(&work_root) {
            Ok(()) => eprintln!(
                "kstress: removed working brains at {} (pass --keep-work to retain)",
                work_root.display()
            ),
            Err(e) => eprintln!(
                "kstress: WARNING — could not remove working brains at {}: {e}",
                work_root.display()
            ),
        }
    }
    Ok(())
}
