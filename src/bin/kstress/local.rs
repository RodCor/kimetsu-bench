//! Local in-process profiler: seed the brain incrementally and, at each scale
//! checkpoint, measure insert/query latency, db size, read/write concurrency,
//! and RSS. Talks to the local v1.0.0 brain through its PUBLIC API plus a
//! direct `rusqlite` handle for bulk-seed + size/concurrency probes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rusqlite::{Connection, OpenFlags};

use kimetsu_brain::embeddings::open_default_embedder;
use kimetsu_brain::project;
use kimetsu_core::memory::{MemoryKind, MemoryScope};

use crate::report::*;
use crate::seed::{SeedOpts, bulk_seed};

pub struct LocalCfg {
    pub root: PathBuf,
    pub scales: Vec<usize>,
    pub matrix: String,
    pub seed: u64,
    pub batch_size: usize,
    /// Above this scale, skip the realistic `add_memory` sample + reader+writer
    /// scenario (the emb conflict scan is O(N) per add → too slow at 1M).
    pub realistic_max_scale: usize,
    pub realistic_sample: usize,
    pub query_sample: usize,
    pub reader_levels: Vec<usize>,
    pub window: Duration,
}

pub fn run(cfg: &LocalCfg) -> Result<Vec<Checkpoint>, String> {
    // Make the stress root a standalone git boundary so kimetsu's project
    // discovery (`git rev-parse` from `start`) resolves HERE instead of
    // climbing to the enclosing bench repo — otherwise add_memory /
    // search_memories / retrieve_context all fail to find a project.
    std::fs::create_dir_all(&cfg.root).map_err(|e| e.to_string())?;
    kimetsu_core::paths::git_init_boundary(&cfg.root);
    project::init_project_at_root(&cfg.root, true).map_err(|e| format!("init project: {e}"))?;
    let db = cfg.root.join(".kimetsu").join("brain.db");
    let embedder = open_default_embedder();
    let with_emb = cfg.matrix == "emb" && !embedder.is_noop();

    let opts = SeedOpts {
        scope: MemoryScope::Project,
        kind: MemoryKind::Fact,
        batch_size: cfg.batch_size,
        with_embeddings: with_emb,
        seed: cfg.seed,
    };

    let mut checkpoints = Vec::new();
    let mut bulk_total = 0usize;

    for &scale in &cfg.scales {
        let delta = scale.saturating_sub(bulk_total);
        // --- bulk seed up to this checkpoint (sample CPU across the seed) ---
        let seed_cpu0 = cpu_secs();
        let seed_cpu_wall = Instant::now();
        let bulk = {
            let mut conn = open_rw(&db)?;
            let stats = bulk_seed(&mut conn, bulk_total, delta, &opts, embedder)?;
            let secs = stats.gen_secs + stats.embed_secs + stats.sql_secs;
            BulkInsert {
                rows: stats.inserted,
                rows_per_sec: if secs > 0.0 {
                    stats.inserted as f64 / secs
                } else {
                    0.0
                },
                gen_secs: stats.gen_secs,
                embed_secs: stats.embed_secs,
                sql_secs: stats.sql_secs,
            }
        };
        let seed_cpu_cores = cpu_cores_since(seed_cpu0, &seed_cpu_wall);
        bulk_total = scale;
        eprintln!(
            "  [{}] seeded {scale} memories ({:.0} rows/s, embed {:.1}s)",
            cfg.matrix, bulk.rows_per_sec, bulk.embed_secs
        );

        // Queries target a POPULATED keyword bucket so FTS stays selective;
        // sample fewer at large scales (each call opens a fresh session over a
        // multi-hundred-MB DB on the slow 9p mount).
        let kw_space = scale.min(crate::corpus::KW_BUCKETS);
        let qn = if scale >= 50_000 {
            cfg.query_sample.min(15)
        } else {
            cfg.query_sample
        };

        // --- realistic add_memory latency (sampled, small scales only) ---
        let realistic = if scale <= cfg.realistic_max_scale {
            Some(measure_realistic_inserts(
                &cfg.root,
                cfg.realistic_sample,
                cfg.seed ^ scale as u64,
            ))
        } else {
            None
        };

        // --- FTS query latency (real public path) ---
        let fts_query = measure_fts(&cfg.root, qn, cfg.seed, kw_space);

        // --- context retrieval: cold (builds the ANN index on emb) then warm ---
        let (context_query_cold_ms, context_query_warm, query_cpu_cores) =
            measure_context(&cfg.root, qn, cfg.seed, kw_space);

        // --- size + concurrency + memory (current + peak RSS) ---
        let size = db_size(&db);
        let concurrency = measure_concurrency(
            &db,
            &cfg.root,
            &cfg.reader_levels,
            cfg.window,
            scale <= cfg.realistic_max_scale,
            kw_space,
        );
        let rss_mb = rss_mb();
        let rss_peak_mb = rss_peak_mb();

        eprintln!(
            "  [{}] {scale}: db {:.1} MB · RSS {:.0}/{:.0} MB · seed/query CPU {:.1}/{:.1} cores · ctx cold {:.0}ms warm p99 {:.1}ms",
            cfg.matrix,
            size.db_bytes as f64 / 1e6,
            rss_mb,
            rss_peak_mb,
            seed_cpu_cores,
            query_cpu_cores,
            context_query_cold_ms,
            context_query_warm.p99_ms,
        );

        checkpoints.push(Checkpoint {
            scale,
            local: Some(LocalMetrics {
                bulk_insert: bulk,
                realistic_insert: realistic,
                fts_query,
                context_query_cold_ms,
                context_query_warm,
                size,
                concurrency,
                rss_mb,
                rss_peak_mb,
                seed_cpu_cores,
                query_cpu_cores,
            }),
            remote: None,
        });
    }
    Ok(checkpoints)
}

fn open_rw(db: &Path) -> Result<Connection, String> {
    let conn = Connection::open(db).map_err(|e| e.to_string())?;
    conn.pragma_update(None, "busy_timeout", 15_000).ok();
    Ok(conn)
}

fn open_ro(db: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| e.to_string())?;
    conn.pragma_update(None, "busy_timeout", 15_000).ok();
    Ok(conn)
}

fn measure_realistic_inserts(root: &Path, n: usize, seed: u64) -> Latency {
    let mut g = crate::corpus::Gen::new(seed);
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let text = g.sentence(1_000_000_000 + i);
        let t = Instant::now();
        let _ = project::add_memory(root, MemoryScope::Project, MemoryKind::Fact, &text);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    latency(samples)
}

fn measure_fts(root: &Path, n: usize, seed: u64, kw_space: usize) -> Latency {
    let mut g = crate::corpus::Gen::new(seed ^ 0xF7);
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let q = g.query(kw_space);
        let t = Instant::now();
        let _ = project::search_memories(root, &q, 20, 0, None, None);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    latency(samples)
}

/// Returns `(cold_ms, warm_latency, warm_cpu_cores)` — the last is the average
/// CPU cores the process used across the warm query loop (query embedding +
/// ANN search + hydration + rerank).
fn measure_context(root: &Path, n: usize, seed: u64, kw_space: usize) -> (f64, Latency, f64) {
    let mut g = crate::corpus::Gen::new(seed ^ 0xC0);
    // Cold call (first emb query builds the ANN index).
    let q0 = g.query(kw_space);
    let t = Instant::now();
    if let Err(e) = project::retrieve_context(root, "recall", &q0, 1024) {
        eprintln!("  kstress: retrieve_context error: {e}");
    }
    let cold = t.elapsed().as_secs_f64() * 1000.0;
    // Warm calls — sample CPU across the whole loop.
    let cpu0 = cpu_secs();
    let cpu_wall = Instant::now();
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let q = g.query(kw_space);
        let t = Instant::now();
        let _ = project::retrieve_context(root, "recall", &q, 1024);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let cpu_cores = cpu_cores_since(cpu0, &cpu_wall);
    (cold, latency(samples), cpu_cores)
}

/// Raw FTS read used by the concurrency threads (no project/session overhead —
/// pure SQLite read concurrency under WAL).
fn raw_fts(conn: &Connection, query: &str) -> usize {
    let sql = "SELECT m.memory_id FROM memories_fts \
               JOIN memories m ON m.memory_id = memories_fts.memory_id \
               WHERE m.invalidated_at IS NULL AND memories_fts MATCH ?1 \
               ORDER BY bm25(memories_fts) LIMIT 20";
    let Ok(mut stmt) = conn.prepare_cached(sql) else {
        return 0;
    };
    match stmt.query_map([sanitize_fts(query)], |r| r.get::<_, String>(0)) {
        Ok(rows) => rows.filter_map(Result::ok).count(),
        Err(_) => 0,
    }
}

fn sanitize_fts(q: &str) -> String {
    q.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn measure_concurrency(
    db: &Path,
    root: &Path,
    levels: &[usize],
    window: Duration,
    allow_writer: bool,
    kw_space: usize,
) -> Vec<ConcResult> {
    let mut out = Vec::new();
    for &readers in levels {
        out.push(run_concurrency(db, root, readers, window, false, kw_space));
    }
    // One readers+writer scenario at the largest reader level (small scales).
    if allow_writer && let Some(&max) = levels.iter().max() {
        out.push(run_concurrency(db, root, max, window, true, kw_space));
    }
    out
}

fn run_concurrency(
    db: &Path,
    root: &Path,
    readers: usize,
    window: Duration,
    writer: bool,
    kw_space: usize,
) -> ConcResult {
    let total = AtomicU64::new(0);
    let deadline = Instant::now() + window;
    let mut all_latencies: Vec<f64> = Vec::new();
    let mut writer_p99 = 0.0;

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for tid in 0..readers {
            let total = &total;
            handles.push(s.spawn(move || {
                let Ok(conn) = open_ro(db) else {
                    return Vec::new();
                };
                let mut g = crate::corpus::Gen::new(0xABCD ^ tid as u64);
                let mut lat = Vec::new();
                while Instant::now() < deadline {
                    let q = g.query(kw_space);
                    let t = Instant::now();
                    let _ = raw_fts(&conn, &q);
                    lat.push(t.elapsed().as_secs_f64() * 1000.0);
                    total.fetch_add(1, Ordering::Relaxed);
                }
                lat
            }));
        }
        // Optional single writer (measures lock-serialized write latency).
        let writer_handle = if writer {
            Some(s.spawn(|| {
                let mut g = crate::corpus::Gen::new(0x5151);
                let mut lat = Vec::new();
                let mut i = 0usize;
                while Instant::now() < deadline {
                    let text = g.sentence(2_000_000_000 + i);
                    i += 1;
                    let t = Instant::now();
                    let _ =
                        project::add_memory(root, MemoryScope::Project, MemoryKind::Fact, &text);
                    lat.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                lat
            }))
        } else {
            None
        };

        for h in handles {
            if let Ok(lat) = h.join() {
                all_latencies.extend(lat);
            }
        }
        if let Some(h) = writer_handle
            && let Ok(mut lat) = h.join()
        {
            writer_p99 = percentile(&mut lat, 99.0);
        }
    });

    let secs = window.as_secs_f64().max(0.001);
    ConcResult {
        readers,
        writer,
        qps: total.load(Ordering::Relaxed) as f64 / secs,
        p99_ms: percentile(&mut all_latencies, 99.0),
        writer_lock_wait_ms: writer_p99,
    }
}

fn db_size(db: &Path) -> DbSize {
    let f = |p: PathBuf| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let mut size = DbSize {
        db_bytes: f(db.to_path_buf()),
        wal_bytes: f(with_suffix(db, "-wal")),
        ..Default::default()
    };
    // Per-table sizes via dbstat (best-effort: not all SQLite builds enable it).
    if let Ok(conn) = open_ro(db) {
        // Authoritative logical size from SQLite itself — `fs::metadata` on the
        // 9p mount has been observed to return 0 for a large, freshly-written
        // file. page_count*page_size is never wrong.
        let pages: i64 = conn
            .query_row("PRAGMA page_count", [], |r| r.get(0))
            .unwrap_or(0);
        let page_size: i64 = conn
            .query_row("PRAGMA page_size", [], |r| r.get(0))
            .unwrap_or(0);
        let logical = (pages * page_size) as u64;
        if logical > size.db_bytes {
            size.db_bytes = logical;
        }
        let q = |name_like: &str| -> u64 {
            conn.query_row(
                "SELECT COALESCE(SUM(pgsize),0) FROM dbstat WHERE name LIKE ?1",
                [name_like],
                |r| r.get::<_, i64>(0),
            )
            .map(|v| v as u64)
            .unwrap_or(0)
        };
        size.memories_bytes = q("memories");
        size.fts_bytes = q("memories\\_fts%").max(q("memories_fts%"));
        size.vec_bytes = q("memory\\_vec%").max(q("memory_vec%"));
    }
    size
}

fn with_suffix(db: &Path, suffix: &str) -> PathBuf {
    let mut s = db.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

/// Resident set size in MB (Linux/WSL2 via /proc; 0 elsewhere).
fn rss_mb() -> f64 {
    rss_field_mb("VmRSS:")
}

/// Peak resident set size (high-water mark) in MB — the most memory the process
/// has held at any point. More meaningful than the instantaneous RSS for "how
/// much RAM does the brain need at this scale".
fn rss_peak_mb() -> f64 {
    rss_field_mb("VmHWM:")
}

fn rss_field_mb(field: &str) -> f64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kb: f64 = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

/// Cumulative process CPU time (user + system) in seconds, Linux/WSL2 via
/// `/proc/self/stat`; 0 elsewhere. Sample before/after a window and divide the
/// delta by wall time to get "CPU cores utilized" during that window.
fn cpu_secs() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    // Field 2 (comm) is parenthesized and may itself contain spaces/parens —
    // split AFTER the last ')' so the whitespace fields line up reliably.
    let Some((_, after)) = stat.rsplit_once(')') else {
        return 0.0;
    };
    let f: Vec<&str> = after.split_whitespace().collect();
    // After ')': f[0] = state (field 3). utime = field 14 -> f[11];
    // stime = field 15 -> f[12]. Clock ticks; _SC_CLK_TCK = 100 on Linux/WSL.
    let utime: f64 = f.get(11).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let stime: f64 = f.get(12).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    (utime + stime) / 100.0
}

/// CPU cores utilized over a window: `(cpu_secs now - cpu0) / wall_secs`.
/// ~1.0 = one core saturated (single-threaded); ~N = N cores busy in parallel.
fn cpu_cores_since(cpu0: f64, wall: &Instant) -> f64 {
    let elapsed = wall.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        (cpu_secs() - cpu0) / elapsed
    } else {
        0.0
    }
}
