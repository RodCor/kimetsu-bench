//! LoCoMo driver: `kbench locomo`.
//!
//! [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al.) is the
//! long-conversation memory benchmark mem0, Letta, and Honcho report on: 10
//! two-speaker conversations (~600 turns each across ~27 dated sessions) with
//! ~2,000 QA pairs in five categories:
//!
//!   1 = multi-hop · 2 = temporal · 3 = open-domain · 4 = single-hop ·
//!   5 = adversarial (unanswerable; the correct behaviour is to abstain)
//!
//! The driver mirrors `kbench longmemeval`: each conversation is ingested into
//! a fresh Kimetsu brain (one memory per turn, tagged with speaker + session
//! date), each question retrieves through `kimetsu brain context`, and an LLM
//! reader answers + an LLM judge grades. Parallel from day one: conversations
//! ingest concurrently, then questions run through a worker pool (each has its
//! own reader/judge subprocess; retrieval on a shared per-conversation brain is
//! read-mostly and safe under the v3.0 concurrency work).
//!
//! Dataset: `data/locomo10.json` from the LoCoMo repo. Fetch with:
//!   curl -L -o locomo10.json \
//!     https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::longmemeval::{
    LlmBackend, LmeError, claude_call, codex_call, codex_judge_prompt, codex_reader_prompt,
    heuristic_judge, reader_effort, require_embeddings_build,
};

/// Retrieval token budget for `kimetsu brain context`. LoCoMo conversations
/// are ~600 turns, so 48k (the LongMemEval setting) comfortably covers the
/// relevant span while still exercising ranking.
fn budget_tokens() -> String {
    std::env::var("LOCOMO_BUDGET_TOKENS").unwrap_or_else(|_| "48000".to_string())
}

pub fn category_name(cat: u8) -> &'static str {
    match cat {
        1 => "multi-hop",
        2 => "temporal",
        3 => "open-domain",
        4 => "single-hop",
        5 => "adversarial",
        _ => "unknown",
    }
}

// ─── Dataset model ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LocomoQa {
    pub question: String,
    /// None for category 5 (adversarial): the correct behaviour is abstention.
    pub answer: Option<String>,
    pub category: u8,
}

#[derive(Debug, Clone)]
pub struct LocomoSession {
    pub index: usize,
    pub date_time: String,
    /// (speaker, text)
    pub turns: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct LocomoSample {
    pub sample_id: String,
    pub sessions: Vec<LocomoSession>,
    pub qa: Vec<LocomoQa>,
}

/// Parse `locomo10.json`. Sessions live under dynamic keys (`session_1`,
/// `session_1_date_time`, ...), so this walks a `serde_json::Value`.
pub fn load_dataset(path: &Path) -> Result<Vec<LocomoSample>, LmeError> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        LmeError::Other(format!(
            "could not read {} ({e}). Fetch it with:\n  curl -L -o locomo10.json \
             https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            path.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| LmeError::Other(format!("invalid LoCoMo JSON: {e}")))?;
    let arr = json
        .as_array()
        .ok_or_else(|| LmeError::Other("LoCoMo dataset: expected a top-level array".into()))?;

    let mut samples = Vec::new();
    for (si, item) in arr.iter().enumerate() {
        let sample_id = item
            .get("sample_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("sample-{}", si + 1));

        let conv = item
            .get("conversation")
            .and_then(|v| v.as_object())
            .ok_or_else(|| LmeError::Other(format!("{sample_id}: missing conversation")))?;

        let mut sessions = Vec::new();
        for i in 1..=200 {
            let key = format!("session_{i}");
            let Some(turns_v) = conv.get(&key).and_then(|v| v.as_array()) else {
                continue;
            };
            let date_time = conv
                .get(&format!("session_{i}_date_time"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut turns = Vec::new();
            for t in turns_v {
                let speaker = t
                    .get("speaker")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Speaker");
                let text = t.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if !text.trim().is_empty() {
                    turns.push((speaker.to_string(), text.trim().to_string()));
                }
            }
            if !turns.is_empty() {
                sessions.push(LocomoSession {
                    index: i,
                    date_time,
                    turns,
                });
            }
        }

        let mut qa = Vec::new();
        for q in item
            .get("qa")
            .and_then(|v| v.as_array())
            .unwrap_or(&Vec::new())
        {
            let question = q
                .get("question")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if question.is_empty() {
                continue;
            }
            let category = q.get("category").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
            // `answer` can be a string or a number; category 5 has none.
            let answer = q.get("answer").map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            });
            qa.push(LocomoQa {
                question,
                answer,
                category,
            });
        }

        samples.push(LocomoSample {
            sample_id,
            sessions,
            qa,
        });
    }
    Ok(samples)
}

// ─── Config + report ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LocomoConfig {
    pub dataset_path: PathBuf,
    /// Max questions (0 = all). Sampled round-robin per category, so a small
    /// run still covers every category.
    pub limit: usize,
    /// Only these categories (empty = all). Values 1-5.
    pub categories: Vec<u8>,
    pub dry_run: bool,
    pub kimetsu_bin: Option<PathBuf>,
    pub llm_backend: LlmBackend,
    pub llm_model: Option<String>,
    /// Question-level worker-pool size (0 = auto: KBENCH_PARALLEL or 3).
    pub parallel: usize,
    /// Run the full question set this many times against the SAME brains.
    /// With `learn`, accuracy per iteration charts the learning loop.
    pub iterations: usize,
    /// Close the ground-truth loop between iterations: correctly-answered
    /// TRAIN questions cite their top retrieved memories (`kimetsu brain
    /// cite`, the same signal live usage records), and `brain tune --apply`
    /// self-tunes retrieval between iterations. Questions are split
    /// deterministically into train/holdout halves; feedback only ever comes
    /// from the train half, so a rising holdout curve demonstrates
    /// generalization rather than memorizing the test.
    pub learn: bool,
    /// Persistent per-sample workspaces (survive across iterations and
    /// process restarts; ingest is skipped when a brain already exists).
    /// None = temp dirs (single-iteration behaviour).
    pub workspace_root: Option<PathBuf>,
    /// Full-power mode: the reader itself reports which memories it used
    /// (a CITED: line in its answer) instead of the harness citing top-k
    /// retrieved. Real cite_memory semantics, zero extra calls.
    pub self_cite: bool,
    /// Distill each session into declarative fact memories at ingest (one
    /// reader call per session), stored alongside the raw turns.
    pub distill_ingest: bool,
}

#[derive(Debug, Serialize)]
pub struct LocomoResult {
    pub sample_id: String,
    pub category: u8,
    pub question: String,
    pub predicted: String,
    pub gold: String,
    pub score: f32,
    /// Train half (feedback source in --learn mode) vs holdout half.
    pub train: bool,
    /// Top retrieved memory ids (cited on CORRECT train answers in --learn).
    pub top_memory_ids: Vec<String>,
}

/// Per-iteration accuracy summary (overall + train/holdout split).
#[derive(Debug, Clone, Serialize)]
pub struct IterationSummary {
    pub iteration: usize,
    pub correct: usize,
    pub total: usize,
    pub train_correct: usize,
    pub train_total: usize,
    pub holdout_correct: usize,
    pub holdout_total: usize,
}

#[derive(Debug, Serialize)]
pub struct LocomoReport {
    pub total: usize,
    pub correct: usize,
    pub results: Vec<LocomoResult>,
    /// One entry per iteration in --iterations mode (learning curve).
    pub iterations: Vec<IterationSummary>,
}

// ─── Run ─────────────────────────────────────────────────────────────────────

fn resolve_bin(cfg: &LocomoConfig) -> String {
    if let Some(p) = &cfg.kimetsu_bin {
        return p.to_string_lossy().to_string();
    }
    std::env::var("KIMETSU_BIN").unwrap_or_else(|_| "kimetsu".to_string())
}

/// One work unit: a question bound to its (already-ingested) sample workspace.
struct WorkItem {
    sample_idx: usize,
    qa: LocomoQa,
    /// Deterministic split: even work-index = train (feedback source in
    /// --learn), odd = holdout (never receives feedback).
    train: bool,
}

/// A per-sample workspace: throwaway temp dir, or a persistent dir that
/// survives across iterations (and process restarts) in --learn mode.
enum Ws {
    Temp(tempfile::TempDir),
    Fixed(PathBuf),
}

impl Ws {
    fn path(&self) -> &Path {
        match self {
            Ws::Temp(t) => t.path(),
            Ws::Fixed(p) => p,
        }
    }
}

/// Extract the top distinct `memory:<ULID>` ids from a `brain context` output,
/// in ranked order (the output lists candidates best-first).
fn parse_top_memory_ids(context: &str, max: usize) -> Vec<String> {
    let mut ids = Vec::new();
    let mut rest = context;
    while let Some(pos) = rest.find("memory:") {
        let tail = &rest[pos + 7..];
        let id: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect();
        if id.len() == 26 && !ids.contains(&id) {
            ids.push(id.clone());
            if ids.len() >= max {
                break;
            }
        }
        rest = &rest[pos + 7..];
    }
    ids
}

pub fn run_locomo(cfg: &LocomoConfig) -> Result<LocomoReport, LmeError> {
    let samples = load_dataset(&cfg.dataset_path)?;

    // Select questions: category filter, then round-robin per category limit.
    let mut by_cat: std::collections::BTreeMap<u8, Vec<(usize, LocomoQa)>> = Default::default();
    for (si, s) in samples.iter().enumerate() {
        for q in &s.qa {
            if !cfg.categories.is_empty() && !cfg.categories.contains(&q.category) {
                continue;
            }
            by_cat.entry(q.category).or_default().push((si, q.clone()));
        }
    }
    let mut work: Vec<WorkItem> = Vec::new();
    if cfg.limit == 0 {
        for list in by_cat.values() {
            for (si, qa) in list {
                work.push(WorkItem {
                    sample_idx: *si,
                    qa: qa.clone(),
                    train: false,
                });
            }
        }
    } else {
        // round-robin across categories until the limit is reached
        let mut iters: Vec<_> = by_cat.values().map(|v| v.iter()).collect();
        'outer: loop {
            let mut any = false;
            for it in iters.iter_mut() {
                if let Some((si, qa)) = it.next() {
                    work.push(WorkItem {
                        sample_idx: *si,
                        qa: qa.clone(),
                        train: false,
                    });
                    any = true;
                    if work.len() >= cfg.limit {
                        break 'outer;
                    }
                }
            }
            if !any {
                break;
            }
        }
    }
    // Deterministic train/holdout split: even work-index = train. The
    // round-robin selection interleaves categories, so both halves keep the
    // same category mix. Only the train half ever produces feedback.
    for (i, w) in work.iter_mut().enumerate() {
        w.train = i % 2 == 0;
    }

    eprintln!(
        "locomo: {} sample(s), {} question(s) selected (backend: {})",
        samples.len(),
        work.len(),
        cfg.llm_backend.as_str()
    );

    if cfg.dry_run {
        for (i, s) in samples.iter().enumerate() {
            let turns: usize = s.sessions.iter().map(|x| x.turns.len()).sum();
            eprintln!(
                "  [dry-run] {} | sessions={} turns={} qa={}",
                s.sample_id,
                s.sessions.len(),
                turns,
                samples[i].qa.len()
            );
        }
        return Ok(LocomoReport {
            total: work.len(),
            correct: 0,
            results: vec![],
            iterations: vec![],
        });
    }

    let kimetsu_bin = resolve_bin(cfg);
    require_embeddings_build(&kimetsu_bin)?;

    let workers = if cfg.parallel == 0 {
        std::env::var("KBENCH_PARALLEL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
    } else {
        cfg.parallel
    };

    // Which samples does the selected work actually touch?
    let needed: std::collections::BTreeSet<usize> = work.iter().map(|w| w.sample_idx).collect();

    // ── Phase A: ingest each needed conversation (parallel) ────────────────
    eprintln!(
        "locomo: ingesting {} conversation(s) ({} workers)...",
        needed.len(),
        workers
    );
    let ws_slots: Arc<Mutex<Vec<Option<Ws>>>> =
        Arc::new(Mutex::new((0..samples.len()).map(|_| None).collect()));
    let ingest_list: Vec<usize> = needed.iter().copied().collect();
    let next = Arc::new(AtomicUsize::new(0));
    let ingest_err: Arc<Mutex<Option<LmeError>>> = Arc::new(Mutex::new(None));
    std::thread::scope(|scope| {
        for _ in 0..workers.min(ingest_list.len().max(1)) {
            let ingest_list = ingest_list.clone();
            let next = next.clone();
            let ws_slots = ws_slots.clone();
            let ingest_err = ingest_err.clone();
            let samples = &samples;
            let kimetsu_bin = kimetsu_bin.clone();
            let root = cfg.workspace_root.clone();
            let cfg = cfg.clone();
            scope.spawn(move || {
                loop {
                    let k = next.fetch_add(1, Ordering::SeqCst);
                    if k >= ingest_list.len() {
                        break;
                    }
                    let si = ingest_list[k];
                    match ingest_sample(&samples[si], &kimetsu_bin, root.as_deref(), &cfg) {
                        Ok(ws) => {
                            eprintln!(
                                "  [ingest {}/{}] {} ok",
                                k + 1,
                                ingest_list.len(),
                                samples[si].sample_id
                            );
                            ws_slots.lock().unwrap()[si] = Some(ws);
                        }
                        Err(e) => {
                            eprintln!("  [ingest] {} FAILED: {e}", samples[si].sample_id);
                            *ingest_err.lock().unwrap() = Some(e);
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = ingest_err.lock().unwrap().take() {
        return Err(e);
    }

    // ── Phase B: answer + judge every question, once per iteration ─────────
    let total = work.len();
    let work = Arc::new(work);
    let iterations = cfg.iterations.max(1);
    let mut iteration_summaries: Vec<IterationSummary> = Vec::new();
    let mut last_results: Vec<LocomoResult> = Vec::new();

    for iter in 1..=iterations {
        if iterations > 1 {
            eprintln!("locomo: ── iteration {iter}/{iterations} ──");
            // Quota gate: a k-run makes thousands of reader calls and the
            // codex window WILL die mid-run. Waiting here (instead of
            // grinding a dead reader) is what keeps every iteration's
            // numbers valid — two k=5 runs were tail-poisoned before this.
            if matches!(cfg.llm_backend, LlmBackend::Codex)
                && !super::longmemeval::wait_for_codex(cfg.llm_model.as_deref(), 8)
            {
                eprintln!("locomo: codex never recovered; stopping before iteration {iter}");
                break;
            }
        }
        let results: Arc<Mutex<Vec<Option<LocomoResult>>>> =
            Arc::new(Mutex::new((0..total).map(|_| None).collect()));
        let next = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let work = work.clone();
                let results = results.clone();
                let next = next.clone();
                let ws_slots = ws_slots.clone();
                let samples = &samples;
                let cfg = cfg.clone();
                let kimetsu_bin = kimetsu_bin.clone();
                scope.spawn(move || {
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= total {
                            break;
                        }
                        let item = &work[i];
                        let sample = &samples[item.sample_idx];
                        eprintln!(
                            "  [{}/{}] {} | {} ...",
                            i + 1,
                            total,
                            sample.sample_id,
                            category_name(item.qa.category)
                        );
                        let res = answer_one(item, sample, &ws_slots, &cfg, &kimetsu_bin);
                        let entry = match res {
                            Ok(r) => r,
                            Err(e) => LocomoResult {
                                sample_id: sample.sample_id.clone(),
                                category: item.qa.category,
                                question: item.qa.question.clone(),
                                predicted: format!("[error: {e}]"),
                                gold: item
                                    .qa
                                    .answer
                                    .clone()
                                    .unwrap_or_else(|| "(unanswerable)".into()),
                                score: 0.0,
                                train: item.train,
                                top_memory_ids: vec![],
                            },
                        };
                        eprintln!(
                            "    -> [{}] {} {}",
                            sample.sample_id,
                            category_name(entry.category),
                            if entry.score >= 1.0 {
                                "CORRECT"
                            } else {
                                "INCORRECT"
                            }
                        );
                        results.lock().unwrap()[i] = Some(entry);
                    }
                });
            }
        });

        let results: Vec<LocomoResult> = Arc::try_unwrap(results)
            .expect("workers joined")
            .into_inner()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        // Iteration summary (overall + train/holdout split).
        let mut s = IterationSummary {
            iteration: iter,
            correct: 0,
            total: results.len(),
            train_correct: 0,
            train_total: 0,
            holdout_correct: 0,
            holdout_total: 0,
        };
        for r in &results {
            let ok = r.score >= 1.0;
            if ok {
                s.correct += 1;
            }
            if r.train {
                s.train_total += 1;
                if ok {
                    s.train_correct += 1;
                }
            } else {
                s.holdout_total += 1;
                if ok {
                    s.holdout_correct += 1;
                }
            }
        }
        let pct = |c: usize, t: usize| {
            if t == 0 {
                0.0
            } else {
                100.0 * c as f64 / t as f64
            }
        };
        eprintln!(
            "locomo: iteration {iter}: overall {:.1}% ({}/{}) | train {:.1}% | holdout {:.1}%",
            pct(s.correct, s.total),
            s.correct,
            s.total,
            pct(s.train_correct, s.train_total),
            pct(s.holdout_correct, s.holdout_total),
        );
        iteration_summaries.push(s);

        // ── Learning feedback: cite top memories of CORRECT train answers,
        //    then self-tune retrieval. Applied AFTER the iteration completes
        //    so every iteration is internally static (no mid-run reshaping).
        if cfg.learn && iter < iterations {
            let mut cites = 0usize;
            for r in results.iter().filter(|r| r.train && r.score >= 1.0) {
                // sample_idx by id (results carry the sample_id).
                let Some(si) = samples.iter().position(|s| s.sample_id == r.sample_id) else {
                    continue;
                };
                let ws_path = {
                    let guard = ws_slots.lock().unwrap();
                    guard[si].as_ref().map(|w| w.path().to_path_buf())
                };
                let Some(ws_path) = ws_path else { continue };
                if r.top_memory_ids.is_empty() {
                    continue;
                }
                // ONE grouped call: the group is the co-citation signal that
                // `brain reinforce --staple` consolidates, and --query feeds
                // the routing index.
                let mut cmd = Command::new(&kimetsu_bin);
                cmd.current_dir(&ws_path).args(["brain", "cite"]);
                for id in r.top_memory_ids.iter().take(4) {
                    cmd.args(["--memory-id", id]);
                }
                cmd.args(["--note", "locomo: answered correctly"]);
                cmd.args(["--query", &r.question]);
                let ok = cmd.output().map(|o| o.status.success()).unwrap_or(false);
                if ok {
                    cites += r.top_memory_ids.len().min(4);
                }
            }
            eprintln!("locomo: feedback applied ({cites} citations from train half)");
            // Consolidation v1: staple co-cited memories + rebuild the
            // query-routing index before the next iteration retrieves.
            let mut staples = 0usize;
            let mut routes = 0usize;
            for si in needed.iter() {
                let ws_path = {
                    let guard = ws_slots.lock().unwrap();
                    guard[*si].as_ref().map(|w| w.path().to_path_buf())
                };
                if let Some(ws_path) = ws_path
                    && let Ok(out) = Command::new(&kimetsu_bin)
                        .current_dir(&ws_path)
                        .args(["brain", "reinforce", "--staple", "--routes"])
                        .output()
                {
                    let s = String::from_utf8_lossy(&out.stdout);
                    // "reinforce: N staple candidate(s), M staple(s) created, K route(s) built"
                    for (label, sink) in [
                        ("staple(s) created", &mut staples),
                        ("route(s) built", &mut routes),
                    ] {
                        if let Some(pos) = s.find(label) {
                            let head = &s[..pos];
                            if let Some(n) = head
                                .split_whitespace()
                                .last()
                                .and_then(|w| w.parse::<usize>().ok())
                            {
                                *sink += n;
                            }
                        }
                    }
                }
            }
            eprintln!("locomo: reinforce applied ({staples} staples, {routes} routes)");
            for si in needed.iter() {
                let ws_path = {
                    let guard = ws_slots.lock().unwrap();
                    guard[*si].as_ref().map(|w| w.path().to_path_buf())
                };
                if let Some(ws_path) = ws_path {
                    let _ = Command::new(&kimetsu_bin)
                        .current_dir(&ws_path)
                        .args(["brain", "tune", "--apply"])
                        .output();
                }
            }
            eprintln!(
                "locomo: brain tune --apply run on {} workspace(s)",
                needed.len()
            );
        }

        last_results = results;
    }

    let correct = last_results.iter().filter(|r| r.score >= 1.0).count();
    Ok(LocomoReport {
        total: last_results.len(),
        correct,
        results: last_results,
        iterations: iteration_summaries,
    })
}

/// Ingest one conversation: one memory per turn, speaker + session date tags,
/// single add-batch call, graph edges skipped (flat is the published setup).
///
/// With a `root`, the workspace is persistent (`<root>/<sample_id>`) and
/// ingest is SKIPPED when a brain already exists — this is what lets
/// `--iterations` re-run the same questions against the same evolving brain.
fn ingest_sample(
    sample: &LocomoSample,
    kimetsu_bin: &str,
    root: Option<&Path>,
    cfg: &LocomoConfig,
) -> Result<Ws, LmeError> {
    let holder = match root {
        Some(root) => {
            let dir = root.join(&sample.sample_id);
            std::fs::create_dir_all(&dir)
                .map_err(|e| LmeError::Other(format!("workspace dir: {e}")))?;
            if dir.join(".kimetsu").join("brain.db").exists() {
                eprintln!("  [ingest] {} reusing existing brain", sample.sample_id);
                return Ok(Ws::Fixed(dir));
            }
            Ws::Fixed(dir)
        }
        None => Ws::Temp(
            tempfile::Builder::new()
                .prefix("kbench-locomo-")
                .tempdir()
                .map_err(|e| LmeError::Other(format!("temp dir: {e}")))?,
        ),
    };
    let ws = holder.path();

    // git init pins kimetsu's project discovery to THIS workspace. Without the
    // boundary, discovery walks up from the temp dir and can hit a stray
    // project.toml above it (a legacy ~/.kimetsu/project.toml broke every
    // ingest with a schema-version mismatch). Non-fatal if git is missing.
    let _ = Command::new("git")
        .current_dir(ws)
        .args(["init", "--quiet"])
        .output();

    let out = Command::new(kimetsu_bin)
        .current_dir(ws)
        .args(["init"])
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn init: {e}")))?;
    if !out.status.success() {
        return Err(LmeError::KimetsuError(format!(
            "init failed for {}: {}",
            sample.sample_id,
            String::from_utf8_lossy(&out.stderr)
        )));
    }

    let mut entries = Vec::new();
    for sess in &sample.sessions {
        for (speaker, text) in &sess.turns {
            let tagged = if sess.date_time.is_empty() {
                format!("[session {}] {speaker}: {text}", sess.index)
            } else {
                format!(
                    "[session {} | {}] {speaker}: {text}",
                    sess.index, sess.date_time
                )
            };
            entries.push(
                serde_json::json!({ "text": tagged, "scope": "project", "kind": "fact" })
                    .to_string(),
            );
        }
    }
    // v2.5.2 --distill-ingest: the representation fix. Raw chat turns force
    // the reader to reverse-engineer dialogue; the 90%+ systems store
    // LLM-extracted atomic facts instead. One reader call per SESSION
    // distills declarative facts, stored ALONGSIDE the raw turns (two-tier,
    // same keep-the-originals discipline as staples). This mirrors real
    // usage where the host agent distills via `brain record` — the agent's
    // tokens, not a memory-pipeline LLM. Per-session failures (quota etc.)
    // are logged and skipped: the raw turns remain as the floor.
    if cfg.distill_ingest {
        let mut distilled = 0usize;
        for sess in &sample.sessions {
            let convo = sess
                .turns
                .iter()
                .map(|(sp, tx)| format!("{sp}: {tx}"))
                .collect::<Vec<_>>()
                .join("\n");
            let prompt = format!(
                "Extract the durable facts from this single session of a two-person \
                 conversation (dated {date}).\n\nRules:\n\
                 - One fact per line, declarative, self-contained.\n\
                 - Always name the person the fact is about.\n\
                 - Include events with their dates, plans, preferences, possessions, \
                 relationships, and changes to earlier facts.\n\
                 - No commentary, no numbering, no headers. Facts only.\n\n\
                 Conversation:\n{convo}",
                date = if sess.date_time.is_empty() {
                    "unknown date".to_string()
                } else {
                    sess.date_time.clone()
                },
            );
            let reply = match cfg.llm_backend {
                LlmBackend::Codex => {
                    codex_call(&prompt, cfg.llm_model.as_deref(), Some(&reader_effort()))
                }
                LlmBackend::Claude => claude_call(&prompt, cfg.llm_model.as_deref()),
                LlmBackend::Http => Err(LmeError::LlmError("distill needs codex|claude".into())),
            };
            match reply {
                Ok(facts) => {
                    for line in facts.lines() {
                        let fact = line.trim().trim_start_matches(['-', '*', ' ']).trim();
                        if fact.len() < 8 {
                            continue;
                        }
                        let tagged = if sess.date_time.is_empty() {
                            format!("[session {}] {fact}", sess.index)
                        } else {
                            format!("[session {} | {}] {fact}", sess.index, sess.date_time)
                        };
                        entries.push(
                            serde_json::json!({ "text": tagged, "scope": "project", "kind": "fact" })
                                .to_string(),
                        );
                        distilled += 1;
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  [distill] {} session {}: skipped ({e})",
                        sample.sample_id, sess.index
                    );
                }
            }
        }
        eprintln!(
            "  [distill] {}: {} fact(s) extracted across {} session(s)",
            sample.sample_id,
            distilled,
            sample.sessions.len()
        );
    }

    let batch = ws.join("locomo-batch.jsonl");
    std::fs::write(&batch, entries.join("\n"))
        .map_err(|e| LmeError::KimetsuError(format!("write batch: {e}")))?;
    let out = Command::new(kimetsu_bin)
        .current_dir(ws)
        .args(["brain", "memory", "add-batch"])
        .arg(&batch)
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn add-batch: {e}")))?;
    if !out.status.success() {
        return Err(LmeError::KimetsuError(format!(
            "add-batch failed for {}: {}",
            sample.sample_id,
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(holder)
}

fn answer_one(
    item: &WorkItem,
    sample: &LocomoSample,
    ws_slots: &Arc<Mutex<Vec<Option<Ws>>>>,
    cfg: &LocomoConfig,
    kimetsu_bin: &str,
) -> Result<LocomoResult, LmeError> {
    // Retrieve from the sample's shared workspace (read-mostly; concurrent
    // kimetsu processes on one brain are safe post-v3.0).
    let ws_path = {
        let guard = ws_slots.lock().unwrap();
        guard[item.sample_idx]
            .as_ref()
            .map(|t| t.path().to_path_buf())
            .ok_or_else(|| LmeError::Other("workspace missing (ingest failed?)".into()))?
    };
    let out = Command::new(kimetsu_bin)
        .current_dir(&ws_path)
        .args([
            "brain",
            "context",
            &item.qa.question,
            "--no-ambient",
            "--budget-tokens",
            &budget_tokens(),
        ])
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn brain context: {e}")))?;
    let context = String::from_utf8_lossy(&out.stdout).to_string();

    // The reader anchors "today" to the conversation's last session date.
    let question_date = sample
        .sessions
        .last()
        .map(|s| s.date_time.clone())
        .unwrap_or_default();
    let mut prompt = codex_reader_prompt(&item.qa.question, &context, &question_date);
    if cfg.self_cite {
        // Full-power mode: the AGENT decides which memories it leaned on —
        // the real cite_memory semantics instead of the harness guessing
        // "top-2 retrieved". Same call, zero extra invocations.
        prompt.push_str(
            "\n\nAfter your answer, output ONE final line of the form\n\
             CITED: memory:<ID>, memory:<ID>\n\
             listing ONLY the memory ids (shown in the context lines) whose \
             content you actually used to answer. If you used none, output \
             exactly: CITED: none",
        );
    }
    let raw_predicted = match cfg.llm_backend {
        LlmBackend::Codex => codex_call(&prompt, cfg.llm_model.as_deref(), Some(&reader_effort()))?,
        LlmBackend::Claude => claude_call(&prompt, cfg.llm_model.as_deref())?,
        LlmBackend::Http => {
            return Err(LmeError::LlmError(
                "locomo supports --reader-backend codex|claude".to_string(),
            ));
        }
    };
    // Split the CITED line off so the judge never sees it.
    let (predicted, self_cited_ids) = if cfg.self_cite {
        match raw_predicted.rsplit_once("CITED:") {
            Some((answer, cited)) => (answer.trim().to_string(), parse_top_memory_ids(cited, 4)),
            None => (raw_predicted, vec![]),
        }
    } else {
        (raw_predicted, vec![])
    };

    // Judge. Category 5 is adversarial: unanswerable, so abstention is correct.
    let is_abstention = item.qa.category == 5;
    let gold = item
        .qa
        .answer
        .clone()
        .unwrap_or_else(|| "(unanswerable: the conversation does not contain this)".into());
    let judge_prompt = codex_judge_prompt(
        &item.qa.question,
        &gold,
        &predicted,
        is_abstention,
        category_name(item.qa.category),
    );
    let verdict = match cfg.llm_backend {
        LlmBackend::Codex => codex_call(&judge_prompt, cfg.llm_model.as_deref(), None),
        LlmBackend::Claude => claude_call(&judge_prompt, cfg.llm_model.as_deref()),
        LlmBackend::Http => unreachable!(),
    };
    let score = match verdict {
        Ok(reply) => {
            let up = reply.to_uppercase();
            if up.contains("CORRECT") && !up.contains("INCORRECT") {
                1.0
            } else {
                0.0
            }
        }
        Err(e) => {
            // Infra death is NOT a grading problem: a quota-exhausted judge
            // must surface as a question-level error (visible, retryable,
            // gated) — the heuristic fallback exists only for judges that
            // answered but unparseably. Falling back on quota death silently
            // deflated iteration 4 of the v3 learning run by ~7 points.
            let msg = e.to_string();
            if msg.contains("usage limit") {
                return Err(e);
            }
            eprintln!("    [judge] warn: {e} — heuristic fallback");
            heuristic_judge(&predicted, &gold, is_abstention)
        }
    };

    Ok(LocomoResult {
        sample_id: sample.sample_id.clone(),
        category: item.qa.category,
        question: item.qa.question.clone(),
        predicted,
        gold,
        score,
        train: item.train,
        // Self-cite mode trusts the agent completely: "CITED: none" means
        // cite nothing (no fallback to top-retrieved).
        top_memory_ids: if cfg.self_cite {
            self_cited_ids
        } else {
            parse_top_memory_ids(&context, 3)
        },
    })
}

// ─── Report rendering ────────────────────────────────────────────────────────

pub fn render_markdown(report: &LocomoReport, dataset: &Path) -> String {
    use std::fmt::Write as _;
    let mut md = String::new();
    let _ = writeln!(md, "# LoCoMo Report\n");
    let _ = writeln!(md, "Dataset: {}\n", dataset.display());
    let pct = |c: usize, t: usize| {
        if t == 0 {
            0.0
        } else {
            100.0 * c as f64 / t as f64
        }
    };
    let _ = writeln!(
        md,
        "**Overall accuracy: {:.1}% ({}/{})**\n",
        pct(report.correct, report.total),
        report.correct,
        report.total
    );

    // Learning curve (only in --iterations mode): feedback comes exclusively
    // from the train half, so the holdout column is the generalization signal.
    if report.iterations.len() > 1 {
        let _ = writeln!(md, "## Learning curve\n");
        let _ = writeln!(md, "| iteration | overall | train half | holdout half |");
        let _ = writeln!(md, "|---|---|---|---|");
        for s in &report.iterations {
            let _ = writeln!(
                md,
                "| {} | {:.1}% ({}/{}) | {:.1}% ({}/{}) | {:.1}% ({}/{}) |",
                s.iteration,
                pct(s.correct, s.total),
                s.correct,
                s.total,
                pct(s.train_correct, s.train_total),
                s.train_correct,
                s.train_total,
                pct(s.holdout_correct, s.holdout_total),
                s.holdout_correct,
                s.holdout_total,
            );
        }
        let _ = writeln!(
            md,
            "\nFeedback (citations + retrieval self-tuning) is applied only from the \
             train half between iterations. A rising holdout column means the \
             ranking improvements generalize beyond the questions that produced \
             the feedback.\n"
        );
    }

    let mut cats: std::collections::BTreeMap<u8, (usize, usize)> = Default::default();
    for r in &report.results {
        let e = cats.entry(r.category).or_default();
        e.1 += 1;
        if r.score >= 1.0 {
            e.0 += 1;
        }
    }
    let _ = writeln!(md, "## By category\n");
    let _ = writeln!(md, "| category | correct | total | accuracy |");
    let _ = writeln!(md, "|---|---|---|---|");
    let mut non_adv = (0usize, 0usize);
    for (cat, (c, t)) in &cats {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {:.1}% |",
            category_name(*cat),
            c,
            t,
            pct(*c, *t)
        );
        if *cat != 5 {
            non_adv.0 += c;
            non_adv.1 += t;
        }
    }
    let _ = writeln!(
        md,
        "\nNon-adversarial overall (cats 1-4, the slice most vendors report): \
         **{:.1}% ({}/{})**",
        pct(non_adv.0, non_adv.1),
        non_adv.0,
        non_adv.1
    );
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"[{
          "sample_id": "t1",
          "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "1:00 pm on 1 May, 2023",
            "session_1": [
              {"speaker": "A", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
              {"speaker": "B", "dia_id": "D1:2", "text": "Congrats!"}
            ],
            "session_2_date_time": "2:00 pm on 8 May, 2023",
            "session_2": [
              {"speaker": "A", "dia_id": "D2:1", "text": "Rex learned to fetch."}
            ]
          },
          "qa": [
            {"question": "What is the dog's name?", "answer": "Rex", "evidence": ["D1:1"], "category": 4},
            {"question": "When did A adopt Rex?", "answer": "1 May 2023", "evidence": ["D1:1"], "category": 2},
            {"question": "What color is Rex?", "adversarial_answer": "brown", "evidence": [], "category": 5}
          ]
        }]"#
    }

    #[test]
    fn parses_sessions_turns_and_qa() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("l.json");
        std::fs::write(&p, sample_json()).unwrap();
        let samples = load_dataset(&p).unwrap();
        assert_eq!(samples.len(), 1);
        let s = &samples[0];
        assert_eq!(s.sessions.len(), 2);
        assert_eq!(s.sessions[0].turns.len(), 2);
        assert_eq!(s.sessions[0].date_time, "1:00 pm on 1 May, 2023");
        assert_eq!(s.qa.len(), 3);
        // category 5 has no `answer` field
        assert!(s.qa[2].answer.is_none());
        assert_eq!(s.qa[2].category, 5);
    }

    #[test]
    fn category_names_cover_paper_taxonomy() {
        assert_eq!(category_name(1), "multi-hop");
        assert_eq!(category_name(2), "temporal");
        assert_eq!(category_name(3), "open-domain");
        assert_eq!(category_name(4), "single-hop");
        assert_eq!(category_name(5), "adversarial");
    }

    #[test]
    fn limit_samples_round_robin_across_categories() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("l.json");
        std::fs::write(&p, sample_json()).unwrap();
        let cfg = LocomoConfig {
            dataset_path: p,
            limit: 2,
            categories: vec![],
            dry_run: true,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Codex,
            llm_model: None,
            parallel: 1,
            iterations: 1,
            learn: false,
            workspace_root: None,
            self_cite: false,
            distill_ingest: false,
        };
        // dry_run: selection happens before any model call; total reflects it
        let report = run_locomo(&cfg).unwrap();
        assert_eq!(report.total, 2);
    }

    #[test]
    fn render_markdown_reports_non_adversarial_slice() {
        let report = LocomoReport {
            total: 2,
            correct: 1,
            results: vec![
                LocomoResult {
                    sample_id: "t1".into(),
                    category: 4,
                    question: "q".into(),
                    predicted: "Rex".into(),
                    gold: "Rex".into(),
                    score: 1.0,
                    train: true,
                    top_memory_ids: vec![],
                },
                LocomoResult {
                    sample_id: "t1".into(),
                    category: 5,
                    question: "q2".into(),
                    predicted: "I don't know".into(),
                    gold: "(unanswerable)".into(),
                    score: 0.0,
                    train: false,
                    top_memory_ids: vec![],
                },
            ],
            iterations: vec![],
        };
        let md = render_markdown(&report, Path::new("x.json"));
        assert!(md.contains("Non-adversarial overall"));
        assert!(md.contains("single-hop"));
        assert!(md.contains("adversarial"));
    }
}
