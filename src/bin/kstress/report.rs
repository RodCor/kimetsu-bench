//! Stress-test result types + JSON / Markdown / CSV writers.
//!
//! All metrics are plain numbers (latency ms, throughput, bytes) — the
//! pass/fail-shaped `kbench` report doesn't fit, so `kstress` carries its own.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct StressReport {
    /// Workspace version of the local v1.0.0 kimetsu under test.
    pub kimetsu_version: String,
    /// `git rev-parse HEAD` of the `../` workspace (reproducibility stamp).
    pub kimetsu_git_sha: String,
    pub generated_at: String,
    /// "lean" (FTS-only / NoopEmbedder) or "emb" (fastembed BGE-small + vec0).
    pub matrix: String,
    /// "local" or "remote".
    pub mode: String,
    pub host: HostInfo,
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Serialize)]
pub struct HostInfo {
    pub embedder_model: String,
    pub embedder_dim: usize,
    pub cpus: usize,
}

#[derive(Debug, Serialize)]
pub struct Checkpoint {
    pub scale: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteMetrics>,
}

#[derive(Debug, Serialize)]
pub struct LocalMetrics {
    pub bulk_insert: BulkInsert,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realistic_insert: Option<Latency>,
    pub fts_query: Latency,
    pub context_query_cold_ms: f64,
    pub context_query_warm: Latency,
    pub size: DbSize,
    pub concurrency: Vec<ConcResult>,
    /// Instantaneous resident set size (MB) at the end of this checkpoint.
    pub rss_mb: f64,
    /// Peak resident set size (high-water mark, MB) — max RAM held so far.
    #[serde(default)]
    pub rss_peak_mb: f64,
    /// CPU cores utilized during the bulk seed (embedding is the heavy part).
    #[serde(default)]
    pub seed_cpu_cores: f64,
    /// CPU cores utilized during the warm context-query loop.
    #[serde(default)]
    pub query_cpu_cores: f64,
}

#[derive(Debug, Serialize)]
pub struct BulkInsert {
    pub rows: usize,
    pub rows_per_sec: f64,
    pub gen_secs: f64,
    pub embed_secs: f64,
    pub sql_secs: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct Latency {
    pub n: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct DbSize {
    pub db_bytes: u64,
    pub wal_bytes: u64,
    pub memories_bytes: u64,
    pub fts_bytes: u64,
    pub vec_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct ConcResult {
    pub readers: usize,
    pub writer: bool,
    pub qps: f64,
    pub p99_ms: f64,
    pub writer_lock_wait_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct RemoteMetrics {
    pub throughput: Vec<RemoteConc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit_429s: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isolation_ok: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct RemoteConc {
    pub concurrency: usize,
    pub max_blocking_threads: usize,
    pub qps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub errors: usize,
}

/// Percentile from an UNSORTED slice of millisecond samples (nearest-rank).
pub fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p / 100.0 * (samples.len() as f64 - 1.0)).round() as usize;
    samples[rank.min(samples.len() - 1)]
}

/// Build a `Latency` from raw ms samples.
pub fn latency(mut samples: Vec<f64>) -> Latency {
    let n = samples.len();
    let max = samples.iter().copied().fold(0.0f64, f64::max);
    Latency {
        n,
        p50_ms: percentile(&mut samples, 50.0),
        p95_ms: percentile(&mut samples, 95.0),
        p99_ms: percentile(&mut samples, 99.0),
        max_ms: max,
    }
}

impl StressReport {
    /// Write `summary.json`, `report.md`, and `data.csv` into `dir`.
    /// Returns the JSON path.
    pub fn write_all(&self, dir: &Path) -> std::io::Result<PathBuf> {
        fs::create_dir_all(dir)?;
        let json = dir.join("summary.json");
        fs::write(&json, serde_json::to_string_pretty(self).unwrap())?;
        fs::write(dir.join("report.md"), self.to_markdown())?;
        fs::write(dir.join("data.csv"), self.to_csv())?;
        Ok(json)
    }

    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "# kimetsu stress test — {} / {}\n\n",
            self.mode, self.matrix
        ));
        s.push_str(&format!(
            "kimetsu `{}` ({})  ·  embedder `{}` (dim {})  ·  {} CPUs  ·  {}\n\n",
            self.kimetsu_version,
            self.kimetsu_git_sha,
            self.host.embedder_model,
            self.host.embedder_dim,
            self.host.cpus,
            self.generated_at,
        ));

        if self.mode == "local" {
            s.push_str("## Local — time & size vs scale\n\n");
            s.push_str("| memories | bulk ins/s | add p99 (ms) | FTS p50/p99 (ms) | ctx cold (ms) | ctx warm p50/p99 (ms) | db (MB) | wal (MB) |\n");
            s.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|\n");
            for c in &self.checkpoints {
                if let Some(m) = &c.local {
                    let add = m
                        .realistic_insert
                        .as_ref()
                        .map(|l| format!("{:.1}", l.p99_ms))
                        .unwrap_or_else(|| "—".into());
                    s.push_str(&format!(
                        "| {} | {:.0} | {} | {:.1}/{:.1} | {:.1} | {:.1}/{:.1} | {:.1} | {:.1} |\n",
                        c.scale,
                        m.bulk_insert.rows_per_sec,
                        add,
                        m.fts_query.p50_ms,
                        m.fts_query.p99_ms,
                        m.context_query_cold_ms,
                        m.context_query_warm.p50_ms,
                        m.context_query_warm.p99_ms,
                        m.size.db_bytes as f64 / 1e6,
                        m.size.wal_bytes as f64 / 1e6,
                    ));
                }
            }

            // Memory + CPU. RSS is the whole in-process footprint (brain +
            // embedder + the in-RAM usearch index). CPU cores: ~1 = single
            // core saturated, ~N = N cores busy in parallel (host has N CPUs).
            s.push_str("\n## Local — resource usage (RAM + CPU)\n\n");
            s.push_str(
                "| memories | RSS (MB) | peak RSS (MB) | seed CPU (cores) | query CPU (cores) |\n",
            );
            s.push_str("|---:|---:|---:|---:|---:|\n");
            for c in &self.checkpoints {
                if let Some(m) = &c.local {
                    s.push_str(&format!(
                        "| {} | {:.0} | {:.0} | {:.1} | {:.1} |\n",
                        c.scale, m.rss_mb, m.rss_peak_mb, m.seed_cpu_cores, m.query_cpu_cores,
                    ));
                }
            }

            s.push_str("\n## Local — read concurrency (QPS / reader p99)\n\n");
            s.push_str(
                "| memories | scenario | readers | QPS | p99 (ms) | writer lock wait (ms) |\n",
            );
            s.push_str("|---:|:--|---:|---:|---:|---:|\n");
            for c in &self.checkpoints {
                if let Some(m) = &c.local {
                    for r in &m.concurrency {
                        s.push_str(&format!(
                            "| {} | {} | {} | {:.0} | {:.1} | {:.1} |\n",
                            c.scale,
                            if r.writer {
                                "readers+writer"
                            } else {
                                "readers"
                            },
                            r.readers,
                            r.qps,
                            r.p99_ms,
                            r.writer_lock_wait_ms,
                        ));
                    }
                }
            }
        } else {
            s.push_str("## Remote — HTTP throughput\n\n");
            s.push_str(
                "| memories | concurrency | blocking threads | QPS | p50/p95/p99 (ms) | errors |\n",
            );
            s.push_str("|---:|---:|---:|---:|---:|---:|\n");
            for c in &self.checkpoints {
                if let Some(m) = &c.remote {
                    for t in &m.throughput {
                        s.push_str(&format!(
                            "| {} | {} | {} | {:.0} | {:.1}/{:.1}/{:.1} | {} |\n",
                            c.scale,
                            t.concurrency,
                            t.max_blocking_threads,
                            t.qps,
                            t.p50_ms,
                            t.p95_ms,
                            t.p99_ms,
                            t.errors,
                        ));
                    }
                    if let Some(n) = m.rate_limit_429s {
                        s.push_str(&format!(
                            "\n- rate-limit probe: **{n}** HTTP 429 responses\n"
                        ));
                    }
                    if let Some(ok) = m.isolation_ok {
                        s.push_str(&format!(
                            "- multi-repo isolation: **{}**\n",
                            if ok {
                                "enforced (cross-repo token rejected)"
                            } else {
                                "NOT enforced"
                            }
                        ));
                    }
                }
            }
        }
        s
    }

    pub fn to_csv(&self) -> String {
        let mut s = String::new();
        s.push_str("mode,matrix,scale,metric,key,value\n");
        let row = |s: &mut String, scale: usize, metric: &str, key: &str, value: f64| {
            s.push_str(&format!(
                "{},{},{},{},{},{}\n",
                self.mode, self.matrix, scale, metric, key, value
            ));
        };
        for c in &self.checkpoints {
            if let Some(m) = &c.local {
                row(
                    &mut s,
                    c.scale,
                    "bulk_insert",
                    "rows_per_sec",
                    m.bulk_insert.rows_per_sec,
                );
                row(
                    &mut s,
                    c.scale,
                    "bulk_insert",
                    "embed_secs",
                    m.bulk_insert.embed_secs,
                );
                if let Some(l) = &m.realistic_insert {
                    row(&mut s, c.scale, "add_memory", "p99_ms", l.p99_ms);
                }
                row(&mut s, c.scale, "fts_query", "p50_ms", m.fts_query.p50_ms);
                row(&mut s, c.scale, "fts_query", "p99_ms", m.fts_query.p99_ms);
                row(
                    &mut s,
                    c.scale,
                    "context_query",
                    "cold_ms",
                    m.context_query_cold_ms,
                );
                row(
                    &mut s,
                    c.scale,
                    "context_query",
                    "warm_p99_ms",
                    m.context_query_warm.p99_ms,
                );
                row(&mut s, c.scale, "size", "db_bytes", m.size.db_bytes as f64);
                row(
                    &mut s,
                    c.scale,
                    "size",
                    "wal_bytes",
                    m.size.wal_bytes as f64,
                );
                row(
                    &mut s,
                    c.scale,
                    "size",
                    "memories_bytes",
                    m.size.memories_bytes as f64,
                );
                row(
                    &mut s,
                    c.scale,
                    "size",
                    "fts_bytes",
                    m.size.fts_bytes as f64,
                );
                row(
                    &mut s,
                    c.scale,
                    "size",
                    "vec_bytes",
                    m.size.vec_bytes as f64,
                );
                row(&mut s, c.scale, "rss", "mb", m.rss_mb);
                row(&mut s, c.scale, "rss", "peak_mb", m.rss_peak_mb);
                row(&mut s, c.scale, "cpu", "seed_cores", m.seed_cpu_cores);
                row(&mut s, c.scale, "cpu", "query_cores", m.query_cpu_cores);
                for r in &m.concurrency {
                    let tag = if r.writer { "rw" } else { "ro" };
                    row(
                        &mut s,
                        c.scale,
                        "concurrency",
                        &format!("{tag}_{}_qps", r.readers),
                        r.qps,
                    );
                    row(
                        &mut s,
                        c.scale,
                        "concurrency",
                        &format!("{tag}_{}_p99_ms", r.readers),
                        r.p99_ms,
                    );
                }
            }
            if let Some(m) = &c.remote {
                for t in &m.throughput {
                    row(
                        &mut s,
                        c.scale,
                        "remote",
                        &format!("c{}_qps", t.concurrency),
                        t.qps,
                    );
                    row(
                        &mut s,
                        c.scale,
                        "remote",
                        &format!("c{}_p99_ms", t.concurrency),
                        t.p99_ms,
                    );
                    row(
                        &mut s,
                        c.scale,
                        "remote",
                        &format!("c{}_errors", t.concurrency),
                        t.errors as f64,
                    );
                }
            }
        }
        s
    }
}
