//! Remote HTTP profiler: spawn the locally-built `kimetsu-remote serve`
//! (v1.0.0, never a release), pre-seed its per-repo brain, then drive it with
//! concurrent blocking `ureq` workers. Measures throughput + latency
//! percentiles vs concurrency, plus rate-limit and multi-repo isolation probes.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rusqlite::Connection;

use kimetsu_brain::embeddings::open_default_embedder;
use kimetsu_brain::project;
use kimetsu_core::memory::{MemoryKind, MemoryScope};

use crate::report::*;
use crate::seed::{SeedOpts, bulk_seed};

pub struct RemoteCfg {
    pub remote_bin: PathBuf,
    pub data_dir: PathBuf,
    pub repo: String,
    pub token: String,
    pub scales: Vec<usize>,
    pub matrix: String,
    pub seed: u64,
    pub batch_size: usize,
    pub concurrency: Vec<usize>,
    pub max_blocking_threads: usize,
    pub window: Duration,
}

/// A spawned server that is killed on drop.
struct Server {
    child: Child,
    base: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn run(cfg: &RemoteCfg) -> Result<Vec<Checkpoint>, String> {
    if !cfg.remote_bin.exists() {
        return Err(format!(
            "kimetsu-remote binary not found at {} — build it from ../ first \
             (cargo build -p kimetsu-remote --release [--features embeddings])",
            cfg.remote_bin.display()
        ));
    }
    let embedder = open_default_embedder();
    let with_emb = cfg.matrix == "emb" && !embedder.is_noop();
    let opts = SeedOpts {
        scope: MemoryScope::Project,
        kind: MemoryKind::Fact,
        batch_size: cfg.batch_size,
        with_embeddings: with_emb,
        seed: cfg.seed,
    };

    let repo_root = cfg.data_dir.join(&cfg.repo);
    std::fs::create_dir_all(&repo_root).map_err(|e| e.to_string())?;
    kimetsu_core::paths::git_init_boundary(&repo_root);
    project::init_project_at_root(&repo_root, true).map_err(|e| format!("init repo brain: {e}"))?;
    let db = repo_root.join(".kimetsu").join("brain.db");

    let mut checkpoints = Vec::new();
    let mut bulk_total = 0usize;

    for &scale in &cfg.scales {
        // Seed up to this checkpoint while no server holds the DB.
        let delta = scale.saturating_sub(bulk_total);
        {
            let mut conn = Connection::open(&db).map_err(|e| e.to_string())?;
            conn.pragma_update(None, "busy_timeout", 15_000).ok();
            bulk_seed(&mut conn, bulk_total, delta, &opts, embedder)?;
        }
        bulk_total = scale;

        // Throughput sweep.
        let server = spawn_server(cfg, cfg.max_blocking_threads, None)?;
        wait_healthz(&server.base)?;
        let kw_space = scale.min(crate::corpus::KW_BUCKETS);
        let mut throughput = Vec::new();
        for &c in &cfg.concurrency {
            throughput.push(load(&server, cfg, c, kw_space));
        }
        drop(server);

        // Rate-limit probe: restart with a tight limit and burst.
        let rate_limit_429s = probe_rate_limit(cfg).ok();
        // Multi-repo isolation probe (per-repo token rejected on another repo).
        let isolation_ok = probe_isolation(cfg).ok();

        eprintln!(
            "  [{}] remote {scale}: peak QPS {:.0}",
            cfg.matrix,
            throughput.iter().map(|t| t.qps).fold(0.0, f64::max)
        );

        checkpoints.push(Checkpoint {
            scale,
            local: None,
            remote: Some(RemoteMetrics {
                throughput,
                rate_limit_429s,
                isolation_ok,
            }),
        });
    }
    Ok(checkpoints)
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
        .unwrap_or(8799)
}

fn spawn_server(
    cfg: &RemoteCfg,
    blocking_threads: usize,
    rate_limit: Option<u32>,
) -> Result<Server, String> {
    let port = free_port();
    let mut cmd = Command::new(&cfg.remote_bin);
    cmd.arg("serve")
        .arg("--addr")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--data")
        .arg(&cfg.data_dir)
        .arg("--token")
        .arg(&cfg.token)
        .arg("--max-blocking-threads")
        .arg(blocking_threads.to_string());
    if let Some(r) = rate_limit {
        cmd.arg("--rate-limit").arg(r.to_string());
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let child = cmd
        .spawn()
        .map_err(|e| format!("spawn kimetsu-remote: {e}"))?;
    Ok(Server {
        child,
        base: format!("http://127.0.0.1:{port}"),
    })
}

fn wait_healthz(base: &str) -> Result<(), String> {
    let url = format!("{base}/healthz");
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if ureq::get(&url)
            .timeout(Duration::from_millis(500))
            .call()
            .is_ok()
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err("server did not become healthy within 15s".into())
}

fn mcp_body(query: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"kimetsu_brain_context","arguments":{{"query":{}}}}}}}"#,
        serde_json::to_string(query).unwrap()
    )
}

/// One throughput run at concurrency `c` for `cfg.window`.
fn load(server: &Server, cfg: &RemoteCfg, c: usize, kw_space: usize) -> RemoteConc {
    let url = format!("{}/mcp/{}", server.base, cfg.repo);
    let total = AtomicU64::new(0);
    let errors = AtomicU64::new(0);
    let deadline = Instant::now() + cfg.window;
    let mut all: Vec<f64> = Vec::new();

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for tid in 0..c {
            let url = &url;
            let token = &cfg.token;
            let total = &total;
            let errors = &errors;
            handles.push(s.spawn(move || {
                let mut g = crate::corpus::Gen::new(0x9001 ^ tid as u64);
                let mut lat = Vec::new();
                while Instant::now() < deadline {
                    let body = mcp_body(&g.query(kw_space));
                    let t = Instant::now();
                    let resp = ureq::post(url)
                        .set("Authorization", &format!("Bearer {token}"))
                        .set("Content-Type", "application/json")
                        .timeout(Duration::from_secs(30))
                        .send_string(&body);
                    lat.push(t.elapsed().as_secs_f64() * 1000.0);
                    match resp {
                        Ok(_) => {
                            total.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                lat
            }));
        }
        for h in handles {
            if let Ok(lat) = h.join() {
                all.extend(lat);
            }
        }
    });

    let secs = cfg.window.as_secs_f64().max(0.001);
    RemoteConc {
        concurrency: c,
        max_blocking_threads: cfg.max_blocking_threads,
        qps: total.load(Ordering::Relaxed) as f64 / secs,
        p50_ms: percentile(&mut all.clone(), 50.0),
        p95_ms: percentile(&mut all.clone(), 95.0),
        p99_ms: percentile(&mut all, 99.0),
        errors: errors.load(Ordering::Relaxed) as usize,
    }
}

/// Burst more requests than `--rate-limit` allows in a minute and count 429s.
fn probe_rate_limit(cfg: &RemoteCfg) -> Result<usize, String> {
    let limit = 60u32; // 60/min
    let server = spawn_server(cfg, cfg.max_blocking_threads, Some(limit))?;
    wait_healthz(&server.base)?;
    let url = format!("{}/mcp/{}", server.base, cfg.repo);
    let mut rejected = 0usize;
    for _ in 0..(limit as usize + 30) {
        let resp = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", cfg.token))
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(10))
            .send_string(&mcp_body("rate probe"));
        if let Err(ureq::Error::Status(429, _)) = resp {
            rejected += 1;
        }
    }
    Ok(rejected)
}

/// Verify a per-repo token is rejected on a DIFFERENT repo (isolation).
fn probe_isolation(cfg: &RemoteCfg) -> Result<bool, String> {
    // Second repo brain so the path resolves; auth should still reject.
    let other = "kstress-other";
    let other_root = cfg.data_dir.join(other);
    project::init_project_at_root(&other_root, true).map_err(|e| e.to_string())?;

    // tokens file: the token is valid ONLY for cfg.repo.
    let tokens_file = cfg.data_dir.join("tokens.toml");
    std::fs::write(
        &tokens_file,
        format!(
            "global = []\n[per_repo]\n{} = [\"{}\"]\n",
            cfg.repo, cfg.token
        ),
    )
    .map_err(|e| e.to_string())?;

    let port = free_port();
    let mut child = Command::new(&cfg.remote_bin)
        .arg("serve")
        .arg("--addr")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--data")
        .arg(&cfg.data_dir)
        .arg("--tokens-file")
        .arg(&tokens_file)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    let base = format!("http://127.0.0.1:{port}");
    let guard = scopeguard(&mut child);
    wait_healthz(&base)?;

    // Allowed on its own repo:
    let ok_self = ureq::post(&format!("{base}/mcp/{}", cfg.repo))
        .set("Authorization", &format!("Bearer {}", cfg.token))
        .set("Content-Type", "application/json")
        .timeout(Duration::from_secs(10))
        .send_string(&mcp_body("self"))
        .is_ok();
    // Rejected on the other repo:
    let rejected_other = matches!(
        ureq::post(&format!("{base}/mcp/{other}"))
            .set("Authorization", &format!("Bearer {}", cfg.token))
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(10))
            .send_string(&mcp_body("cross")),
        Err(ureq::Error::Status(401, _)) | Err(ureq::Error::Status(403, _))
    );
    drop(guard);
    Ok(ok_self && rejected_other)
}

/// Minimal kill-on-drop guard for a borrowed child.
fn scopeguard(child: &mut Child) -> impl Drop + '_ {
    struct G<'a>(&'a mut Child);
    impl Drop for G<'_> {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    G(child)
}
