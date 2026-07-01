//! LongMemEval benchmark driver — Pillar 0.2.
//!
//! Runs the public LongMemEval benchmark (github.com/xiaowu0162/LongMemEval)
//! against Kimetsu's memory layer so we can publish a comparable accuracy
//! number vs mem0 / Zep.
//!
//! ## Dataset
//!
//! LongMemEval ships as one of three JSON files:
//!   - `longmemeval_s.json` — ~40 sessions, ~115k tokens (fastest)
//!   - `longmemeval_m.json` — ~500 sessions per question (hardest)
//!   - `longmemeval_oracle.json` — only evidence sessions (retrieval oracle)
//!
//! Each JSON file is a flat array of `LmeInstance` objects.  The schema is
//! documented at https://github.com/xiaowu0162/LongMemEval and confirmed by
//! the paper (arXiv 2410.10813).
//!
//! ## Flow per question instance
//!
//!   1. Create a fresh temp Kimetsu brain workspace.
//!   2. Ingest the haystack sessions into it via `kimetsu brain memory add`,
//!      preserving session dates for temporal/knowledge-update questions.
//!   3. Retrieve context for the question via `kimetsu brain context "<q>"`.
//!   4. Feed retrieved context + question to a configurable LLM answerer.
//!   5. Score the answer vs the gold `answer` via an LLM judge (with a
//!      substring heuristic fallback when no model is available).
//!   6. Accumulate per-question-type accuracy and overall accuracy.
//!
//! ## Model config — HTTP backend (default)
//!
//!   KBENCH_LLM_MODEL      — model id (e.g. "gpt-4o-mini")
//!   KBENCH_LLM_API_KEY    — API key for the provider
//!   KBENCH_LLM_BASE_URL   — base URL (default: https://api.openai.com/v1)
//!
//! ## Model config — Codex backend (no API key required)
//!
//!   KBENCH_LLM_BACKEND=codex   — select the codex CLI backend
//!   KBENCH_LLM_MODEL           — optional model id passed as `-m <model>`;
//!                                when unset codex picks its default (gpt-5.5)
//!
//! The codex backend shells out to:
//!   codex exec --ignore-user-config --skip-git-repo-check --ephemeral \
//!              --color never -C <fresh_tmp> -o <answer_file> [-m <model>] "<prompt>"
//!
//!   --ignore-user-config is CRITICAL: without it codex loads ~/.codex/config.toml,
//!   which spins up MCP servers (incl. kimetsu) and hangs.  Auth still works with
//!   this flag because codex reads its OAuth token separately.
//!   -o <file>  writes ONLY the model's final message — no transcript parsing needed.
//!   -C <tmp>   + --skip-git-repo-check + --ephemeral: neutral throwaway dir.
//!
//! When HTTP vars are unset the driver errors with a clear message on every
//! step that requires a model call.  The `--dry-run` path parses + plans the
//! ingest without making any model calls, so it works offline.
//!
//! ## Kimetsu CLI surface used
//!
//! The driver shells out to the `kimetsu` binary (auto-discovered via
//! `DriverContext.overrides["kimetsu_bin"]` or the `KIMETSU_BIN` env var
//! or falling back to `kimetsu` on PATH):
//!
//!   kimetsu brain memory add --workspace <tmp> --scope project --text "<turn>"
//!       — ingest one memory into a fresh workspace
//!
//!   kimetsu brain context "<question>" --workspace <tmp> --output json
//!       — retrieve the top-k context snippets for a question
//!
//! Both calls are non-interactive (output to stdout) and safe to shell out to.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

// ─── Dataset types ──────────────────────────────────────────────────────────

/// One turn in a haystack session.  The `has_answer` flag marks the turn(s)
/// that contain the evidence required to answer the question — used for
/// turn-level recall metrics in the paper; the driver uses it to optionally
/// prioritise those turns during ingest.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LmeTurn {
    pub role: String,
    pub content: String,
    /// Present (true) only on turns that contain the gold answer evidence.
    #[serde(default)]
    pub has_answer: bool,
}

/// One evaluation instance from the LongMemEval JSON array.
///
/// Field names match the dataset exactly as published in the repo README and
/// confirmed by arXiv paper 2410.10813:
///
///   question_id          — unique question identifier
///   question_type        — category (see `LmeQuestionType`)
///   question             — the question text
///   answer               — gold answer string
///   question_date        — ISO-8601 date string for the question
///   haystack_session_ids — session IDs for the full haystack (chronological)
///   haystack_dates       — date strings per haystack session (parallel)
///   haystack_sessions    — [[{role,content,has_answer?}]] — the actual turns
///   answer_session_ids   — session IDs that contain the gold evidence
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LmeInstance {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    /// Gold answer. In LongMemEval the answer is usually a string, but some
    /// questions (e.g. counting) have a numeric or boolean answer, so coerce
    /// any JSON scalar to its string form.
    #[serde(deserialize_with = "de_scalar_to_string")]
    pub answer: String,
    /// ISO-8601 date when the question is posed.
    pub question_date: String,
    /// Session identifiers, parallel to `haystack_sessions`.
    #[serde(default)]
    pub haystack_session_ids: Vec<String>,
    /// Per-session dates, parallel to `haystack_sessions`.
    #[serde(default)]
    pub haystack_dates: Vec<String>,
    /// The full chat history: list of sessions, each a list of turns.
    #[serde(default)]
    pub haystack_sessions: Vec<Vec<LmeTurn>>,
    /// Session IDs that contain the answer evidence (for recall metrics).
    #[serde(default)]
    pub answer_session_ids: Vec<String>,
}

/// Coerce any JSON scalar (string / number / bool / array / null) to a String.
/// LongMemEval answers are usually strings but counting questions can be ints.
fn de_scalar_to_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    Ok(match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::String(s) => s,
        serde_json::Value::Null => String::new(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => other.to_string(),
    })
}

impl LmeInstance {
    /// True when the question type is an abstention variant (the model should
    /// answer "I don't know" — the answer field is typically "unknown"/"N/A").
    pub fn is_abstention(&self) -> bool {
        self.question_type.ends_with("_abs")
    }

    /// The base question type without the `_abs` suffix.
    pub fn base_type(&self) -> &str {
        self.question_type
            .strip_suffix("_abs")
            .unwrap_or(&self.question_type)
    }
}

/// Canonical question types from the LongMemEval paper.
/// Used for --help text and validation in downstream tooling.
#[allow(dead_code)]
pub const QUESTION_TYPES: &[&str] = &[
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
    // Abstention variants (same base types with _abs suffix):
    "single-session-user_abs",
    "single-session-assistant_abs",
    "single-session-preference_abs",
    "temporal-reasoning_abs",
    "knowledge-update_abs",
    "multi-session_abs",
];

// ─── LLM backend selector ────────────────────────────────────────────────────

/// Which LLM backend to use for answering and judging.
///
/// `Http`  — OpenAI-compatible HTTP API (default).  Requires
///           `KBENCH_LLM_MODEL` + `KBENCH_LLM_API_KEY`.
///
/// `Codex` — `codex exec` CLI (ChatGPT login, no API key needed).
///           Requires codex to be on PATH and authenticated.
///           Controlled by `KBENCH_LLM_BACKEND=codex` or `--reader-backend codex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmBackend {
    #[default]
    Http,
    Codex,
}

impl LlmBackend {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "http" => Some(Self::Http),
            "codex" => Some(Self::Codex),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Codex => "codex",
        }
    }
}

// ─── Driver config ───────────────────────────────────────────────────────────

/// Configuration for the LongMemEval run, built from CLI args + env vars.
#[derive(Debug, Clone)]
pub struct LmeConfig {
    /// Path to the LongMemEval JSON file (e.g. `longmemeval_s.json`).
    pub dataset_path: PathBuf,
    /// Maximum number of instances to evaluate (0 = all).
    pub limit: usize,
    /// If non-empty, only these question_type values are evaluated.
    pub question_types: Vec<String>,
    /// Dry-run: parse + plan ingest, but make no model or kimetsu calls.
    pub dry_run: bool,
    /// Path to the kimetsu binary (auto-resolved when None).
    pub kimetsu_bin: Option<PathBuf>,
    /// LLM backend selector (http or codex).
    pub llm_backend: LlmBackend,
    /// LLM model id for answering + judging.  None = not configured.
    /// For codex backend: passed as `-m <model>`; None = let codex pick.
    pub llm_model: Option<String>,
    /// API key for the LLM provider (http backend only).
    pub llm_api_key: Option<String>,
    /// Base URL for the LLM provider API (http backend only).
    pub llm_base_url: Option<String>,
}

impl LmeConfig {
    /// Build from env vars.  Call after parsing CLI args; CLI wins over env.
    pub fn with_env_overlay(mut self) -> Self {
        // Backend selector: env var wins if CLI hasn't set it explicitly.
        if self.llm_backend == LlmBackend::Http
            && let Ok(v) = std::env::var("KBENCH_LLM_BACKEND")
            && let Some(b) = LlmBackend::from_str(&v)
        {
            self.llm_backend = b;
        }
        if self.llm_model.is_none() {
            self.llm_model = std::env::var("KBENCH_LLM_MODEL").ok();
        }
        if self.llm_api_key.is_none() {
            self.llm_api_key = std::env::var("KBENCH_LLM_API_KEY").ok();
        }
        if self.llm_base_url.is_none() {
            self.llm_base_url = std::env::var("KBENCH_LLM_BASE_URL").ok();
        }
        if self.kimetsu_bin.is_none()
            && let Ok(v) = std::env::var("KIMETSU_BIN")
        {
            self.kimetsu_bin = Some(PathBuf::from(v));
        }
        self
    }

    /// Error if no LLM model is configured for the HTTP backend.
    fn require_http_llm(&self) -> Result<(&str, &str), LmeError> {
        match (&self.llm_model, &self.llm_api_key) {
            (Some(model), Some(key)) => Ok((model.as_str(), key.as_str())),
            _ => Err(LmeError::NoModelConfigured),
        }
    }

    /// Legacy alias — used by tests and http-backend callers.
    #[allow(dead_code)]
    pub fn require_llm(&self) -> Result<(&str, &str), LmeError> {
        self.require_http_llm()
    }

    fn base_url(&self) -> &str {
        self.llm_base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
    }

    /// Validate that the configured backend is ready for real calls.
    /// For http: model + api_key must be set.
    /// For codex: `codex` must be on PATH (we check at call time).
    pub fn validate_llm_config(&self) -> Result<(), LmeError> {
        match self.llm_backend {
            LlmBackend::Http => {
                self.require_http_llm()?;
                Ok(())
            }
            LlmBackend::Codex => {
                // codex auth is validated at first call; nothing to check here.
                Ok(())
            }
        }
    }
}

// ─── Per-instance result ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmeInstanceResult {
    pub question_id: String,
    pub question_type: String,
    /// The answer produced by the answerer (or "[dry-run]" / "[error]").
    pub predicted: String,
    /// The gold answer from the dataset.
    pub gold: String,
    /// 1.0 = correct, 0.0 = incorrect.
    pub score: f32,
    /// Judge explanation or heuristic label.
    pub judge_reason: String,
    /// Seconds spent on this instance.
    pub duration_secs: f64,
}

// ─── Aggregate report ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmeReport {
    /// ISO-8601 timestamp.
    pub generated_at: String,
    /// Path to the dataset file.
    pub dataset: String,
    /// Total number of instances evaluated.
    pub total: usize,
    /// Number of correct answers.
    pub correct: usize,
    /// Overall accuracy (0.0–1.0).
    pub overall_accuracy: f32,
    /// Per question-type breakdown.
    pub by_type: BTreeMap<String, LmeTypeStats>,
    /// Per-instance results (full detail).
    pub instances: Vec<LmeInstanceResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmeTypeStats {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f32,
}

impl LmeReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# LongMemEval Report\n\n");
        out.push_str(&format!("Generated: {}\n", self.generated_at));
        out.push_str(&format!("Dataset: {}\n\n", self.dataset));
        out.push_str(&format!(
            "**Overall accuracy: {:.1}% ({}/{})**\n\n",
            self.overall_accuracy * 100.0,
            self.correct,
            self.total
        ));

        out.push_str("## By question type\n\n");
        out.push_str("| Question type | Correct | Total | Accuracy |\n");
        out.push_str("|---|---|---|---|\n");
        for (qtype, stats) in &self.by_type {
            out.push_str(&format!(
                "| {} | {} | {} | {:.1}% |\n",
                qtype,
                stats.correct,
                stats.total,
                stats.accuracy * 100.0
            ));
        }
        out.push('\n');

        out.push_str("## Per-instance results\n\n");
        out.push_str("| ID | Type | Score | Predicted (truncated) | Gold (truncated) |\n");
        out.push_str("|---|---|---|---|---|\n");
        for r in &self.instances {
            let pred_trunc = truncate_str(&r.predicted, 60);
            let gold_trunc = truncate_str(&r.gold, 60);
            out.push_str(&format!(
                "| {} | {} | {:.0} | {} | {} |\n",
                r.question_id, r.question_type, r.score, pred_trunc, gold_trunc
            ));
        }
        out
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize LmeReport")
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.replace('|', "\\|")
    } else {
        format!("{}…", &s[..max]).replace('|', "\\|")
    }
}

// ─── Error type ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LmeError {
    /// Dataset file not found or unreadable.
    DatasetLoad(String),
    /// JSON parse failed.
    DatasetParse(String),
    /// No LLM model is configured (set KBENCH_LLM_* env vars).
    NoModelConfigured,
    /// Kimetsu CLI call failed.
    KimetsuError(String),
    /// LLM API call failed.
    LlmError(String),
    /// Other error.
    Other(String),
}

impl std::fmt::Display for LmeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LmeError::DatasetLoad(msg) => write!(f, "dataset load error: {msg}"),
            LmeError::DatasetParse(msg) => write!(f, "dataset parse error: {msg}"),
            LmeError::NoModelConfigured => write!(
                f,
                "no LLM model configured — for the HTTP backend set KBENCH_LLM_MODEL and \
                 KBENCH_LLM_API_KEY (optionally KBENCH_LLM_BASE_URL, default: \
                 https://api.openai.com/v1); for the codex backend set \
                 KBENCH_LLM_BACKEND=codex (no API key required — uses `codex exec` with \
                 your ChatGPT login; optionally KBENCH_LLM_MODEL to pin the model)"
            ),
            LmeError::KimetsuError(msg) => write!(f, "kimetsu error: {msg}"),
            LmeError::LlmError(msg) => write!(f, "llm error: {msg}"),
            LmeError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LmeError {}

// ─── Dataset loading ─────────────────────────────────────────────────────────

/// Load and parse a LongMemEval JSON file into a `Vec<LmeInstance>`.
pub fn load_dataset(path: &Path) -> Result<Vec<LmeInstance>, LmeError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| LmeError::DatasetLoad(format!("could not read {}: {e}", path.display())))?;
    serde_json::from_str::<Vec<LmeInstance>>(&content).map_err(|e| {
        LmeError::DatasetParse(format!(
            "JSON parse failed for {}: {e}. \
             Confirm this is a LongMemEval JSON file (flat array of instances).",
            path.display()
        ))
    })
}

/// Filter and limit instances according to `LmeConfig`.
pub fn filter_instances(instances: Vec<LmeInstance>, cfg: &LmeConfig) -> Vec<LmeInstance> {
    let mut out: Vec<LmeInstance> = instances
        .into_iter()
        .filter(|inst| {
            if cfg.question_types.is_empty() {
                return true;
            }
            cfg.question_types.iter().any(|t| t == &inst.question_type)
        })
        .collect();

    if cfg.limit > 0 && out.len() > cfg.limit {
        out = stratified_take(out, cfg.limit);
    }
    out
}

/// Take `limit` instances spread evenly across question types via round-robin,
/// instead of truncating to the first `limit` (the LongMemEval file is grouped
/// by type, so a plain truncate would sample only the first one or two types —
/// e.g. all single-session-user — and miss multi-session/temporal/knowledge-
/// update entirely). Round-robin is deterministic and reproducible: it walks the
/// types in first-appearance order, taking one not-yet-taken instance from each
/// per round until `limit` is reached, preserving original order within a type.
fn stratified_take(instances: Vec<LmeInstance>, limit: usize) -> Vec<LmeInstance> {
    // Bucket by type, preserving first-appearance order of the types.
    let mut type_order: Vec<String> = Vec::new();
    let mut buckets: std::collections::HashMap<String, std::collections::VecDeque<LmeInstance>> =
        std::collections::HashMap::new();
    for inst in instances {
        let ty = inst.question_type.clone();
        if !buckets.contains_key(&ty) {
            type_order.push(ty.clone());
        }
        buckets.entry(ty).or_default().push_back(inst);
    }

    let mut out: Vec<LmeInstance> = Vec::with_capacity(limit);
    while out.len() < limit {
        let mut took_any = false;
        for ty in &type_order {
            if out.len() >= limit {
                break;
            }
            if let Some(inst) = buckets.get_mut(ty).and_then(|b| b.pop_front()) {
                out.push(inst);
                took_any = true;
            }
        }
        if !took_any {
            break; // all buckets drained
        }
    }
    out
}

// ─── Kimetsu interaction ─────────────────────────────────────────────────────

/// Discover the kimetsu binary: override > env > PATH.
fn resolve_kimetsu_bin(cfg: &LmeConfig) -> String {
    if let Some(p) = &cfg.kimetsu_bin {
        return p.to_string_lossy().to_string();
    }
    if let Ok(v) = std::env::var("KIMETSU_BIN") {
        return v;
    }
    "kimetsu".to_string()
}

/// Result of a per-instance ingest: the live temp workspace plus the number of
/// memory-add subprocess calls performed (one per ingested turn). The count is
/// reported so we can estimate full-run cost (per-turn ingest is many more
/// subprocess calls than the old per-session ingest).
struct IngestOutcome {
    tmp: tempfile::TempDir,
    turns_ingested: usize,
}

/// Ingest all haystack sessions for one instance into a fresh temp workspace,
/// **one memory per TURN** (not one per session).
///
/// Per-turn granularity is the retrieval fix: a whole-session embedding is a
/// muddy average that surfaces semantically-similar-but-wrong sessions, whereas
/// a single answer-bearing turn embeds cleanly and matches the query, so it
/// lands in the retrieved top-k under the bigger budget.
///
/// Each turn is stored as `"User: <content>"` / `"Assistant: <content>"`. The
/// originating session index is preserved as a light prefix tag so multi-session
/// reasoning can still attribute a turn, without bloating the embedding.
///
/// Creates an isolated git repo + kimetsu brain in a temp dir, then ingests
/// each turn.  Using `current_dir(tmp)` routes kimetsu's workspace discovery to
/// the temp dir instead of climbing to an enclosing repo.
///
/// Returns the temp dir (kept alive so the workspace persists for retrieval)
/// plus the per-instance turn-ingest count.
fn ingest_sessions(instance: &LmeInstance, kimetsu_bin: &str) -> Result<IngestOutcome, LmeError> {
    let tmp = tempfile::Builder::new()
        .prefix("kbench-lme-")
        .tempdir()
        .map_err(|e| LmeError::Other(format!("could not create temp dir: {e}")))?;

    let workspace = tmp.path();

    // 1. `git init` in the temp dir — this makes kimetsu's discover_repo_root()
    //    stop here instead of climbing to the enclosing repo on the drive.
    let git_out = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(workspace)
        .output()
        .map_err(|e| LmeError::Other(format!("could not spawn git init: {e}")))?;
    if !git_out.status.success() {
        // Non-fatal: without git init, kimetsu will still work but may walk up
        // to a parent repo.  Warn and continue.
        let stderr = String::from_utf8_lossy(&git_out.stderr);
        eprintln!("    [lme] warn: git init in temp workspace failed (non-fatal): {stderr}");
    }

    // 2. `kimetsu init` — creates .kimetsu/project.toml + brain.db in workspace.
    let init_out = Command::new(kimetsu_bin)
        .args(["init"])
        .current_dir(workspace)
        .output()
        .map_err(|e| {
            LmeError::KimetsuError(format!("could not spawn `{kimetsu_bin} init`: {e}"))
        })?;
    if !init_out.status.success() {
        let stderr = String::from_utf8_lossy(&init_out.stderr);
        return Err(LmeError::KimetsuError(format!(
            "kimetsu init failed in temp workspace: {stderr}"
        )));
    }

    // One memory per TURN, but ingested via a SINGLE `add-batch` call (one
    // process, embedder loaded once) instead of a subprocess per turn — turns
    // ~13 min/instance of process-spawn overhead into seconds.
    let mut entries: Vec<String> = Vec::new();
    for (sess_idx, session) in instance.haystack_sessions.iter().enumerate() {
        for turn in session.iter() {
            let body = turn.content.trim();
            if body.is_empty() {
                continue;
            }
            // Prefix the session index AND its date so multi-session attribution
            // and time-based reasoning survive (temporal-reasoning questions need
            // to know WHEN each session happened). Without the date the reader has
            // no anchor for "how long ago"/"which came first" — the schema has the
            // dates (`haystack_dates`), we just have to surface them to the reader.
            let date = instance
                .haystack_dates
                .get(sess_idx)
                .map(|s| s.trim())
                .unwrap_or("");
            let turn_text = if date.is_empty() {
                format!(
                    "[session {sess_idx}] {}: {body}",
                    capitalize_role(&turn.role)
                )
            } else {
                format!(
                    "[session {sess_idx} | {date}] {}: {body}",
                    capitalize_role(&turn.role)
                )
            };
            entries.push(
                serde_json::json!({ "text": turn_text, "scope": "project", "kind": "fact" })
                    .to_string(),
            );
        }
    }
    let turns_ingested = entries.len();

    let batch_file = workspace.join("lme-batch.jsonl");
    std::fs::write(&batch_file, entries.join("\n"))
        .map_err(|e| LmeError::KimetsuError(format!("could not write batch ingest file: {e}")))?;

    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .args(["brain", "memory", "add-batch"])
        .arg(&batch_file)
        .output()
        .map_err(|e| {
            LmeError::KimetsuError(format!(
                "could not spawn `{kimetsu_bin} brain memory add-batch`: {e}"
            ))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(LmeError::KimetsuError(format!(
            "memory add-batch failed for instance {}: {stderr}",
            instance.question_id
        )));
    }

    Ok(IngestOutcome {
        tmp,
        turns_ingested,
    })
}

/// Retrieval token budget for `kimetsu brain context` (default CLI is 6000).
///
/// The capsule render budget is `budget_tokens / 2`, so 48000 yields a ~24000-
/// token capsule budget. With small per-turn memories (~40-120 tokens each)
/// that admits ~100-300 candidate turns. This is the recall lever: ~42% of the
/// misses at the old 12000 budget were "I don't know" — the answer turn existed
/// in the store but never reached the reader's window. gpt-5.5 has a large
/// context window, so we trade a fatter prompt for recall and let the reader
/// (run at high reasoning effort) sift. `max_capsules` stays at the CLI default
/// (0 = no cap), so the budget alone governs how many turns surface.
const LME_BUDGET_TOKENS: &str = "48000";

/// Retrieve context for a question from an ingested workspace.
/// Returns the raw stdout from `kimetsu brain context`.
fn retrieve_context(
    question: &str,
    workspace: &Path,
    kimetsu_bin: &str,
) -> Result<String, LmeError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .args([
            "brain",
            "context",
            question,
            "--no-ambient",
            "--budget-tokens",
            LME_BUDGET_TOKENS,
        ])
        .output()
        .map_err(|e| {
            LmeError::KimetsuError(format!(
                "could not spawn `{kimetsu_bin} brain context`: {e}"
            ))
        })?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(LmeError::KimetsuError(format!(
            "brain context failed: {stderr}"
        )));
    }

    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn capitalize_role(role: &str) -> &str {
    match role {
        "user" => "User",
        "assistant" => "Assistant",
        other => other,
    }
}

// ─── Codex backend ───────────────────────────────────────────────────────────

/// Per-call timeout for codex exec (seconds).
const CODEX_TIMEOUT_SECS: u64 = 120;

/// Call `codex exec` with the verified flags.
///
/// Exact argv constructed:
///   codex exec --ignore-user-config --skip-git-repo-check --ephemeral \
///              --color never -C <fresh_tmp> -o <answer_file> [-m <model>] "<prompt>"
///
/// The `-o` file receives only the model's final message — no transcript
/// parsing needed.  Returns the content of that file on success.
///
/// On timeout or non-zero exit, returns an `Err(LmeError::LlmError)`.
pub fn codex_call(
    prompt: &str,
    model: Option<&str>,
    effort: Option<&str>,
) -> Result<String, LmeError> {
    // Fresh temp dir: -C arg (codex runs in this neutral dir).
    let tmp_dir = tempfile::Builder::new()
        .prefix("kbench-codex-")
        .tempdir()
        .map_err(|e| LmeError::Other(format!("could not create codex temp dir: {e}")))?;

    // Output file: codex writes the final message here via -o.
    let answer_file = tmp_dir.path().join("answer.txt");

    let mut argv: Vec<String> = vec![
        "exec".to_string(),
        "--ignore-user-config".to_string(),
        "--skip-git-repo-check".to_string(),
        "--ephemeral".to_string(),
        "--color".to_string(),
        "never".to_string(),
        "-C".to_string(),
        tmp_dir.path().to_string_lossy().to_string(),
        "-o".to_string(),
        answer_file.to_string_lossy().to_string(),
    ];

    if let Some(m) = model
        && !m.is_empty()
    {
        argv.push("-m".to_string());
        argv.push(m.to_string());
    }

    // Reasoning effort override. Codex defaults to "none" for the reader, which
    // tanks the reasoning-bound categories (temporal arithmetic, multi-session
    // counting). `-c model_reasoning_effort=<effort>` works even under
    // --ignore-user-config (it overrides on top of defaults, not the file).
    if let Some(e) = effort
        && !e.is_empty()
    {
        argv.push("-c".to_string());
        argv.push(format!("model_reasoning_effort=\"{e}\""));
    }

    // Pass the prompt via STDIN (argv has a hard length limit on Windows, and
    // LongMemEval contexts are large). `-` tells codex to read instructions
    // from stdin.
    argv.push("-".to_string());

    eprintln!(
        "    [codex] argv: codex {}",
        argv.iter()
            .map(|a| {
                // Truncate the prompt arg in the log to keep it readable.
                if a.len() > 120 {
                    format!("\"{}…\"", &a[..120])
                } else {
                    format!("\"{a}\"")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    );

    // On Windows, `codex` is typically installed as a `.cmd` script (e.g. via
    // Volta/nvm/npm).  `Command::new("codex")` will not find `.cmd` files on
    // Windows — we must invoke it through `cmd /c codex <args>`.
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.args(["/c", "codex"]);
        c.args(&argv);
        c
    };
    #[cfg(not(target_os = "windows"))]
    let mut cmd = {
        let mut c = Command::new("codex");
        c.args(&argv);
        c
    };
    // Feed the prompt over stdin (see the `-` arg above).
    cmd.stdin(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        LmeError::LlmError(format!(
            "could not spawn `codex exec`: {e} \
             (is `codex` on PATH and authenticated? \
             On Windows, codex must be installed as a .cmd script via npm/volta)"
        ))
    })?;

    // Write the prompt to stdin from a thread so a large prompt can't deadlock
    // on a full pipe buffer; dropping the handle sends EOF.
    if let Some(mut stdin) = child.stdin.take() {
        let prompt_bytes = prompt.as_bytes().to_vec();
        std::thread::spawn(move || {
            use std::io::Write;
            let _ = stdin.write_all(&prompt_bytes);
        });
    }

    // Poll with a timeout.  std::process::Command has no built-in timeout;
    // we use try_wait in a sleep loop and kill on expiry.
    let timeout = std::time::Duration::from_secs(CODEX_TIMEOUT_SECS);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited.
                if !status.success() {
                    let code = status
                        .code()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "signal".to_string());
                    return Err(LmeError::LlmError(format!(
                        "codex exec exited with code {code}"
                    )));
                }
                break;
            }
            Ok(None) => {
                // Still running.
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(LmeError::LlmError(format!(
                        "codex exec timed out after {}s",
                        CODEX_TIMEOUT_SECS
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(e) => {
                return Err(LmeError::LlmError(format!("codex exec wait failed: {e}")));
            }
        }
    }

    // Read the answer file.
    let answer = std::fs::read_to_string(&answer_file).map_err(|e| {
        LmeError::LlmError(format!(
            "codex exec succeeded but answer file is missing/unreadable \
             ({}): {e}",
            answer_file.display()
        ))
    })?;

    let trimmed = answer.trim().to_string();
    if trimmed.is_empty() {
        return Err(LmeError::LlmError(
            "codex exec returned an empty answer".to_string(),
        ));
    }

    // tmp_dir dropped here — temp dir cleaned up automatically.
    Ok(trimmed)
}

// ─── LLM calls (OpenAI-compatible) ──────────────────────────────────────────

/// Call an OpenAI-compatible chat completions endpoint.
/// Returns the assistant message content on success.
fn llm_chat(
    base_url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
) -> Result<String, LmeError> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": 512,
        "temperature": 0.0
    });

    let response = ureq::post(&url)
        .set("Authorization", &format!("Bearer {api_key}"))
        .set("Content-Type", "application/json")
        .send_json(&body)
        .map_err(|e| LmeError::LlmError(format!("HTTP request failed: {e}")))?;

    let v: serde_json::Value = response
        .into_json()
        .map_err(|e| LmeError::LlmError(format!("response parse failed: {e}")))?;

    v["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| LmeError::LlmError(format!("unexpected response shape: {v}")))
}

// ─── Reader prompt ───────────────────────────────────────────────────────────

/// Build the reader prompt (combined system+user) for the codex backend.
/// The codex exec takes a single prompt string, not system+user separately.
fn codex_reader_prompt(question: &str, context: &str, question_date: &str) -> String {
    let date_line = if question_date.trim().is_empty() {
        String::new()
    } else {
        format!(
            "Today's date (when the question is asked) is {question_date}. Each memory is \
             tagged with the date of its session as [session N | DATE]. Use these dates for \
             any time-based reasoning — durations, ordering, and \"how long ago\" — counting \
             from today's date. IMPORTANT: if the memories give conflicting values for the \
             same fact (a fact that changed over time — someone moved, a price was updated, a \
             preference changed), the value from the MOST RECENT session date is the current, \
             correct answer — report that one, even if an older value appears more often.\n\n"
        )
    };
    format!(
        "You are a memory assistant. Answer the question using ONLY the provided \
         memory context below. Be concise — output just the answer text, nothing else. \
         If the answer is not in the context, output exactly: I don't know\n\n\
         {date_line}\
         Memory context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
}

/// Build the judge prompt for the codex backend.
fn codex_judge_prompt(
    question: &str,
    gold: &str,
    predicted: &str,
    is_abstention: bool,
    base_type: &str,
) -> String {
    // Preference questions are graded by rubric, not exact match: the gold answer
    // DESCRIBES the user's preference, and a correct response is one that honors
    // it (LongMemEval's preference grading). A strict string match wrongly fails
    // the reader for giving good advice consistent with the preference.
    let guidance = if is_abstention {
        "Note: this is an abstention question. The model should have said it does not know. \
         Reply CORRECT only if the predicted answer indicates the model doesn't know.\n\n"
    } else if base_type == "single-session-preference" {
        "Note: this is a PREFERENCE question. The gold answer DESCRIBES the user's preference \
         (a rubric), not a literal expected string. Reply CORRECT if the predicted answer is \
         consistent with and honors that preference; reply INCORRECT only if it ignores or \
         contradicts it.\n\n"
    } else {
        ""
    };
    format!(
        "You are a strict answer judge. Given a question, a gold (correct) answer, \
         and a predicted answer, decide if the predicted answer is correct. \
         Reply with exactly one word: CORRECT or INCORRECT.\n\n\
         {guidance}\
         Question: {question}\n\
         Gold answer: {gold}\n\
         Predicted answer: {predicted}\n\n\
         Is the predicted answer correct? (CORRECT/INCORRECT)"
    )
}

// ─── Answer + Judge dispatch ─────────────────────────────────────────────────

/// Generate an answer for a question given retrieved context.
fn answer_question(
    question: &str,
    context: &str,
    question_date: &str,
    cfg: &LmeConfig,
) -> Result<String, LmeError> {
    match cfg.llm_backend {
        LlmBackend::Codex => {
            let prompt = codex_reader_prompt(question, context, question_date);
            // Reader runs at high reasoning effort: it must do date arithmetic and
            // cross-session counting, which collapse at the default "none".
            codex_call(&prompt, cfg.llm_model.as_deref(), Some("high"))
        }
        LlmBackend::Http => {
            let (model, api_key) = cfg.require_http_llm()?;
            let system = "You are a helpful assistant. Answer the question based on the provided \
                          memory context. Be concise. If the answer is not in the context, say \
                          \"I don't know\".";
            let date_line = if question_date.trim().is_empty() {
                String::new()
            } else {
                format!(
                    "Today's date is {question_date}. Memories are tagged [session N | DATE]; \
                     use those dates for time-based reasoning.\n\n"
                )
            };
            let user =
                format!("{date_line}Memory context:\n{context}\n\nQuestion: {question}\n\nAnswer:");
            llm_chat(cfg.base_url(), api_key, model, system, &user)
        }
    }
}

/// Judge whether a predicted answer is correct given the gold answer.
/// Returns (score, reason).
///
/// Uses an LLM judge (LongMemEval paper uses GPT-4-style judging).
/// Falls back to a substring heuristic when LLM is unavailable or for
/// abstention questions.
fn judge_answer(
    question: &str,
    predicted: &str,
    gold: &str,
    is_abstention: bool,
    base_type: &str,
    cfg: &LmeConfig,
) -> (f32, String) {
    // Heuristic fallback: try substring match first (fast, offline).
    let heuristic_score = heuristic_judge(predicted, gold, is_abstention);

    match cfg.llm_backend {
        LlmBackend::Codex => {
            // Codex backend: use codex_call for judging. The judge is a simple
            // CORRECT/INCORRECT call, so leave it at the default effort (fast) —
            // the reader is where reasoning effort pays off.
            let prompt = codex_judge_prompt(question, gold, predicted, is_abstention, base_type);
            match codex_call(&prompt, cfg.llm_model.as_deref(), None) {
                Ok(reply) => {
                    let reply_up = reply.trim().to_uppercase();
                    let score = if reply_up.contains("CORRECT") && !reply_up.contains("INCORRECT") {
                        1.0
                    } else {
                        0.0
                    };
                    (score, format!("codex judge: {}", reply.trim()))
                }
                Err(e) => {
                    eprintln!("    [codex judge] warn: {e} — falling back to heuristic");
                    (
                        heuristic_score,
                        format!("heuristic fallback (codex judge failed: {e})"),
                    )
                }
            }
        }
        LlmBackend::Http => {
            // If no LLM is configured, use the heuristic.
            if cfg.llm_model.is_none() || cfg.llm_api_key.is_none() {
                return (heuristic_score, "heuristic: substring match".to_string());
            }

            match llm_judge(question, predicted, gold, is_abstention, cfg) {
                Ok((score, reason)) => (score, reason),
                Err(e) => {
                    // LLM judging failed — fall back to heuristic and note it.
                    (
                        heuristic_score,
                        format!("heuristic fallback (llm judge failed: {e}): substring match"),
                    )
                }
            }
        }
    }
}

/// LLM-based judge (HTTP backend).  Mirrors the LongMemEval paper's judging prompt.
fn llm_judge(
    question: &str,
    predicted: &str,
    gold: &str,
    is_abstention: bool,
    cfg: &LmeConfig,
) -> Result<(f32, String), LmeError> {
    let (model, api_key) = cfg.require_http_llm()?;

    let system = "You are a strict answer judge. Given a question, a gold (correct) answer, \
                  and a predicted answer, decide if the predicted answer is correct. \
                  Respond with exactly one word: CORRECT or INCORRECT.";

    let abstention_note = if is_abstention {
        "Note: this is an abstention question. The model should say it does not know. \
         Mark CORRECT only if the predicted answer indicates the model doesn't know.\n\n"
    } else {
        ""
    };

    let user = format!(
        "{abstention_note}\
         Question: {question}\n\
         Gold answer: {gold}\n\
         Predicted answer: {predicted}\n\n\
         Is the predicted answer correct? (CORRECT/INCORRECT)"
    );

    let reply = llm_chat(cfg.base_url(), api_key, model, system, &user)?;
    let reply_up = reply.trim().to_uppercase();
    let score = if reply_up.starts_with("CORRECT") {
        1.0
    } else {
        0.0
    };
    Ok((score, format!("llm judge: {}", reply.trim())))
}

/// Heuristic judge: substring match with some normalization.
fn heuristic_judge(predicted: &str, gold: &str, is_abstention: bool) -> f32 {
    let pred_low = predicted.to_lowercase();
    let gold_low = gold.to_lowercase();

    if is_abstention {
        // For abstention questions: model should say it doesn't know.
        let unsure_phrases = [
            "don't know",
            "do not know",
            "not sure",
            "cannot find",
            "no information",
            "i don't have",
            "unknown",
            "i'm not sure",
        ];
        let signals_abstention = unsure_phrases.iter().any(|p| pred_low.contains(p));
        return if signals_abstention { 1.0 } else { 0.0 };
    }

    // For factual questions: gold answer substring present in prediction.
    if gold_low.is_empty() {
        return 0.0;
    }
    if pred_low.contains(&gold_low) || gold_low.contains(&pred_low) {
        1.0
    } else {
        0.0
    }
}

// ─── Ingest plan (dry-run) ───────────────────────────────────────────────────

/// Describes what the driver *would* do for one instance, without executing it.
#[derive(Debug, Serialize)]
pub struct IngestPlan {
    pub question_id: String,
    pub question_type: String,
    pub total_sessions: usize,
    pub total_turns: usize,
    pub uses_dates: bool,
    pub answer_session_count: usize,
}

impl IngestPlan {
    pub fn from_instance(inst: &LmeInstance) -> Self {
        let total_turns: usize = inst.haystack_sessions.iter().map(|s| s.len()).sum();
        let uses_dates = matches!(inst.base_type(), "temporal-reasoning" | "knowledge-update");
        Self {
            question_id: inst.question_id.clone(),
            question_type: inst.question_type.clone(),
            total_sessions: inst.haystack_sessions.len(),
            total_turns,
            uses_dates,
            answer_session_count: inst.answer_session_ids.len(),
        }
    }
}

// ─── Main run entry point ────────────────────────────────────────────────────

/// Run the LongMemEval benchmark with the given config.
///
/// In `--dry-run` mode: loads and filters instances, produces an ingest plan
/// for each, asserts structural validity, returns a report with all scores 0
/// and notes `[dry-run]`.
///
/// In real mode: ingests, retrieves, answers, and judges each instance,
/// accumulating per-type accuracy.
pub fn run_longmemeval(cfg: &LmeConfig) -> Result<LmeReport, LmeError> {
    let instances = load_dataset(&cfg.dataset_path)?;
    let instances = filter_instances(instances, cfg);

    eprintln!(
        "longmemeval: {} instance(s) to evaluate{} (backend: {})",
        instances.len(),
        if cfg.dry_run { " (dry-run)" } else { "" },
        cfg.llm_backend.as_str(),
    );

    if cfg.dry_run {
        return run_dry(instances, cfg);
    }

    // Real run: validate LLM config upfront so we fail fast.
    cfg.validate_llm_config()?;

    let kimetsu_bin = resolve_kimetsu_bin(cfg);
    run_real(instances, cfg, &kimetsu_bin)
}

fn run_dry(instances: Vec<LmeInstance>, cfg: &LmeConfig) -> Result<LmeReport, LmeError> {
    let mut results: Vec<LmeInstanceResult> = Vec::new();
    let mut plans: Vec<IngestPlan> = Vec::new();

    for inst in &instances {
        let plan = IngestPlan::from_instance(inst);
        eprintln!(
            "  [dry-run] {} | type={} | sessions={} turns={} uses_dates={}",
            plan.question_id,
            plan.question_type,
            plan.total_sessions,
            plan.total_turns,
            plan.uses_dates
        );
        plans.push(plan);

        results.push(LmeInstanceResult {
            question_id: inst.question_id.clone(),
            question_type: inst.question_type.clone(),
            predicted: "[dry-run]".to_string(),
            gold: inst.answer.clone(),
            score: 0.0,
            judge_reason: "dry-run: no model calls made".to_string(),
            duration_secs: 0.0,
        });
    }

    // Validate plans (structural assertions).
    for plan in &plans {
        if plan.total_sessions == 0 {
            eprintln!(
                "  [dry-run] warn: instance {} has no haystack_sessions",
                plan.question_id
            );
        }
    }

    Ok(build_report(results, &cfg.dataset_path))
}

fn run_real(
    instances: Vec<LmeInstance>,
    cfg: &LmeConfig,
    kimetsu_bin: &str,
) -> Result<LmeReport, LmeError> {
    let mut results: Vec<LmeInstanceResult> = Vec::new();

    for (i, inst) in instances.iter().enumerate() {
        let start = std::time::Instant::now();
        eprintln!(
            "  [{}/{}] {} | type={} ...",
            i + 1,
            instances.len(),
            inst.question_id,
            inst.question_type
        );

        let result = run_single_instance(inst, cfg, kimetsu_bin);
        let duration_secs = start.elapsed().as_secs_f64();

        match result {
            Ok((predicted, score, judge_reason)) => {
                eprintln!("    -> score={score:.0} judge={judge_reason}");
                results.push(LmeInstanceResult {
                    question_id: inst.question_id.clone(),
                    question_type: inst.question_type.clone(),
                    predicted,
                    gold: inst.answer.clone(),
                    score,
                    judge_reason,
                    duration_secs,
                });
            }
            Err(e) => {
                eprintln!("    -> ERROR (counted as incorrect): {e}");
                results.push(LmeInstanceResult {
                    question_id: inst.question_id.clone(),
                    question_type: inst.question_type.clone(),
                    predicted: format!("[error: {e}]"),
                    gold: inst.answer.clone(),
                    score: 0.0,
                    judge_reason: format!("error: {e}"),
                    duration_secs,
                });
            }
        }
    }

    Ok(build_report(results, &cfg.dataset_path))
}

fn run_single_instance(
    inst: &LmeInstance,
    cfg: &LmeConfig,
    kimetsu_bin: &str,
) -> Result<(String, f32, String), LmeError> {
    // 1. Ingest sessions into a fresh workspace, one memory per turn.
    let ingest_start = std::time::Instant::now();
    let outcome = ingest_sessions(inst, kimetsu_bin)?;
    let tmp = &outcome.tmp;
    eprintln!(
        "    [ingest] {} turns in {:.1}s ({:.0} ms/turn)",
        outcome.turns_ingested,
        ingest_start.elapsed().as_secs_f64(),
        if outcome.turns_ingested > 0 {
            ingest_start.elapsed().as_millis() as f64 / outcome.turns_ingested as f64
        } else {
            0.0
        },
    );

    // 2. Retrieve context.
    let context = retrieve_context(&inst.question, tmp.path(), kimetsu_bin)?;

    // 3. Answer.
    let predicted = answer_question(&inst.question, &context, &inst.question_date, cfg)?;

    // 4. Judge.
    let (score, reason) = judge_answer(
        &inst.question,
        &predicted,
        &inst.answer,
        inst.is_abstention(),
        inst.base_type(),
        cfg,
    );

    Ok((predicted, score, reason))
    // tmp dropped here — temp dir cleaned up automatically.
}

fn build_report(results: Vec<LmeInstanceResult>, dataset_path: &Path) -> LmeReport {
    let now = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());

    let total = results.len();
    let correct = results.iter().filter(|r| r.score >= 1.0).count();
    let overall_accuracy = if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    };

    // Per-type breakdown.
    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in &results {
        let e = by_type.entry(r.question_type.clone()).or_insert((0, 0));
        e.0 += 1;
        if r.score >= 1.0 {
            e.1 += 1;
        }
    }
    let by_type: BTreeMap<String, LmeTypeStats> = by_type
        .into_iter()
        .map(|(k, (total, correct))| {
            let accuracy = if total == 0 {
                0.0
            } else {
                correct as f32 / total as f32
            };
            (
                k,
                LmeTypeStats {
                    total,
                    correct,
                    accuracy,
                },
            )
        })
        .collect();

    LmeReport {
        generated_at: now,
        dataset: dataset_path.to_string_lossy().to_string(),
        total,
        correct,
        overall_accuracy,
        by_type,
        instances: results,
    }
}

// ─── Synthetic fixture for testing ──────────────────────────────────────────

/// Build a tiny synthetic LongMemEval dataset covering all major question
/// types.  Used by the dry-run smoke test and unit tests — no dataset file
/// required.
pub fn synthetic_fixture() -> Vec<LmeInstance> {
    vec![
        LmeInstance {
            question_id: "syn-001".to_string(),
            question_type: "single-session-user".to_string(),
            question: "What is Alice's favorite color?".to_string(),
            answer: "blue".to_string(),
            question_date: "2024-06-01".to_string(),
            haystack_session_ids: vec!["s1".to_string()],
            haystack_dates: vec!["2024-05-15".to_string()],
            haystack_sessions: vec![vec![
                LmeTurn {
                    role: "user".to_string(),
                    content: "I really love the color blue.".to_string(),
                    has_answer: true,
                },
                LmeTurn {
                    role: "assistant".to_string(),
                    content: "That's a great color!".to_string(),
                    has_answer: false,
                },
            ]],
            answer_session_ids: vec!["s1".to_string()],
        },
        LmeInstance {
            question_id: "syn-002".to_string(),
            question_type: "temporal-reasoning".to_string(),
            question: "What did Bob buy before June 2024?".to_string(),
            answer: "a laptop".to_string(),
            question_date: "2024-07-01".to_string(),
            haystack_session_ids: vec!["s2".to_string(), "s3".to_string()],
            haystack_dates: vec!["2024-03-10".to_string(), "2024-06-20".to_string()],
            haystack_sessions: vec![
                vec![
                    LmeTurn {
                        role: "user".to_string(),
                        content: "I just bought a laptop last week.".to_string(),
                        has_answer: true,
                    },
                    LmeTurn {
                        role: "assistant".to_string(),
                        content: "Nice purchase!".to_string(),
                        has_answer: false,
                    },
                ],
                vec![LmeTurn {
                    role: "user".to_string(),
                    content: "I'm thinking of buying a car.".to_string(),
                    has_answer: false,
                }],
            ],
            answer_session_ids: vec!["s2".to_string()],
        },
        LmeInstance {
            question_id: "syn-003".to_string(),
            question_type: "knowledge-update".to_string(),
            question: "What is Carol's current job?".to_string(),
            answer: "software engineer".to_string(),
            question_date: "2024-08-01".to_string(),
            haystack_session_ids: vec!["s4".to_string(), "s5".to_string()],
            haystack_dates: vec!["2024-01-01".to_string(), "2024-07-01".to_string()],
            haystack_sessions: vec![
                vec![LmeTurn {
                    role: "user".to_string(),
                    content: "I work as a teacher.".to_string(),
                    has_answer: false,
                }],
                vec![LmeTurn {
                    role: "user".to_string(),
                    content: "I switched careers! Now I'm a software engineer.".to_string(),
                    has_answer: true,
                }],
            ],
            answer_session_ids: vec!["s5".to_string()],
        },
        LmeInstance {
            question_id: "syn-004".to_string(),
            question_type: "multi-session".to_string(),
            question: "What hobbies does Dave have?".to_string(),
            answer: "hiking and photography".to_string(),
            question_date: "2024-09-01".to_string(),
            haystack_session_ids: vec!["s6".to_string(), "s7".to_string()],
            haystack_dates: vec!["2024-04-01".to_string(), "2024-05-01".to_string()],
            haystack_sessions: vec![
                vec![LmeTurn {
                    role: "user".to_string(),
                    content: "I love hiking in the mountains on weekends.".to_string(),
                    has_answer: true,
                }],
                vec![LmeTurn {
                    role: "user".to_string(),
                    content: "I've been getting into photography lately.".to_string(),
                    has_answer: true,
                }],
            ],
            answer_session_ids: vec!["s6".to_string(), "s7".to_string()],
        },
        LmeInstance {
            question_id: "syn-005".to_string(),
            question_type: "single-session-preference_abs".to_string(),
            question: "What is Eve's favorite movie?".to_string(),
            answer: "unknown".to_string(),
            question_date: "2024-09-15".to_string(),
            haystack_session_ids: vec!["s8".to_string()],
            haystack_dates: vec!["2024-08-01".to_string()],
            haystack_sessions: vec![vec![LmeTurn {
                role: "user".to_string(),
                content: "I mostly watch documentaries.".to_string(),
                has_answer: false,
            }]],
            answer_session_ids: vec![],
        },
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    // ── Dataset parsing ───────────────────────────────────────────────────────

    #[test]
    fn parse_minimal_instance_from_json() {
        let json = r#"[{
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What color does Alice like?",
            "answer": "blue",
            "question_date": "2024-06-01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024-05-15"],
            "haystack_sessions": [[
                {"role": "user", "content": "I love blue.", "has_answer": true},
                {"role": "assistant", "content": "Great!"}
            ]],
            "answer_session_ids": ["s1"]
        }]"#;
        let instances: Vec<LmeInstance> = serde_json::from_str(json).expect("should parse");
        assert_eq!(instances.len(), 1);
        let inst = &instances[0];
        assert_eq!(inst.question_id, "q1");
        assert_eq!(inst.question_type, "single-session-user");
        assert_eq!(inst.answer, "blue");
        assert_eq!(inst.haystack_sessions.len(), 1);
        assert_eq!(inst.haystack_sessions[0].len(), 2);
        assert!(inst.haystack_sessions[0][0].has_answer);
        assert!(!inst.haystack_sessions[0][1].has_answer);
    }

    #[test]
    fn parse_instance_missing_optional_fields() {
        // answer_session_ids, haystack_dates, haystack_session_ids are all optional
        let json = r#"[{
            "question_id": "q2",
            "question_type": "multi-session",
            "question": "Test?",
            "answer": "yes",
            "question_date": "2024-01-01",
            "haystack_sessions": []
        }]"#;
        let instances: Vec<LmeInstance> =
            serde_json::from_str(json).expect("should parse with missing optional fields");
        assert_eq!(instances[0].haystack_session_ids, Vec::<String>::new());
        assert_eq!(instances[0].answer_session_ids, Vec::<String>::new());
    }

    #[test]
    fn parse_instance_without_has_answer_defaults_to_false() {
        let json = r#"[{
            "question_id": "q3",
            "question_type": "temporal-reasoning",
            "question": "When?",
            "answer": "yesterday",
            "question_date": "2024-01-02",
            "haystack_sessions": [[{"role": "user", "content": "Hi"}]]
        }]"#;
        let instances: Vec<LmeInstance> = serde_json::from_str(json).expect("parse");
        assert!(!instances[0].haystack_sessions[0][0].has_answer);
    }

    // ── LmeInstance helpers ───────────────────────────────────────────────────

    #[test]
    fn is_abstention_detects_abs_suffix() {
        let inst = LmeInstance {
            question_id: "x".to_string(),
            question_type: "single-session-user_abs".to_string(),
            question: "q".to_string(),
            answer: "unknown".to_string(),
            question_date: "2024-01-01".to_string(),
            haystack_session_ids: vec![],
            haystack_dates: vec![],
            haystack_sessions: vec![],
            answer_session_ids: vec![],
        };
        assert!(inst.is_abstention());
        assert_eq!(inst.base_type(), "single-session-user");
    }

    #[test]
    fn is_abstention_false_for_normal_types() {
        let inst = LmeInstance {
            question_id: "y".to_string(),
            question_type: "temporal-reasoning".to_string(),
            question: "q".to_string(),
            answer: "a".to_string(),
            question_date: "2024-01-01".to_string(),
            haystack_session_ids: vec![],
            haystack_dates: vec![],
            haystack_sessions: vec![],
            answer_session_ids: vec![],
        };
        assert!(!inst.is_abstention());
        assert_eq!(inst.base_type(), "temporal-reasoning");
    }

    // ── LlmBackend ────────────────────────────────────────────────────────────

    #[test]
    fn llm_backend_from_str_roundtrip() {
        assert_eq!(LlmBackend::from_str("http"), Some(LlmBackend::Http));
        assert_eq!(LlmBackend::from_str("codex"), Some(LlmBackend::Codex));
        assert_eq!(LlmBackend::from_str("HTTP"), Some(LlmBackend::Http));
        assert_eq!(LlmBackend::from_str("CODEX"), Some(LlmBackend::Codex));
        assert_eq!(LlmBackend::from_str("unknown"), None);
    }

    #[test]
    fn llm_backend_default_is_http() {
        assert_eq!(LlmBackend::default(), LlmBackend::Http);
    }

    // ── Codex argv construction ───────────────────────────────────────────────

    /// Verify the argv we'd pass to codex exec is correct without actually
    /// spawning codex.  We do this by constructing the same argv vector the
    /// real codex_call() constructs and asserting its shape.
    #[test]
    fn codex_argv_construction() {
        let model = Some("gpt-5.5");
        let prompt = "What is the capital of France?";

        // Reproduce the argv construction from codex_call().
        let tmp_dir_path = std::path::Path::new("/tmp/kbench-codex-test");
        let answer_file = tmp_dir_path.join("answer.txt");

        let mut argv: Vec<String> = vec![
            "exec".to_string(),
            "--ignore-user-config".to_string(),
            "--skip-git-repo-check".to_string(),
            "--ephemeral".to_string(),
            "--color".to_string(),
            "never".to_string(),
            "-C".to_string(),
            tmp_dir_path.to_string_lossy().to_string(),
            "-o".to_string(),
            answer_file.to_string_lossy().to_string(),
        ];

        if let Some(m) = model {
            argv.push("-m".to_string());
            argv.push(m.to_string());
        }
        // Reasoning-effort override (reader uses "high").
        let effort = Some("high");
        if let Some(e) = effort {
            argv.push("-c".to_string());
            argv.push(format!("model_reasoning_effort=\"{e}\""));
        }
        // Prompt goes over stdin, not argv; the arg is `-`.
        let _ = prompt;
        argv.push("-".to_string());

        // Assertions.
        assert_eq!(argv[0], "exec");
        assert!(argv.contains(&"--ignore-user-config".to_string()));
        assert!(argv.contains(&"--skip-git-repo-check".to_string()));
        assert!(argv.contains(&"--ephemeral".to_string()));
        assert!(argv.contains(&"--color".to_string()));
        assert!(argv.contains(&"never".to_string()));
        assert!(argv.contains(&"-C".to_string()));
        assert!(argv.contains(&"-o".to_string()));
        assert!(argv.contains(&"-m".to_string()));
        assert!(argv.contains(&"gpt-5.5".to_string()));
        assert!(argv.contains(&"-c".to_string()));
        assert!(argv.contains(&"model_reasoning_effort=\"high\"".to_string()));
        // Prompt is read from stdin; the argv prompt slot is `-`.
        assert_eq!(argv.last().unwrap(), "-");
    }

    #[test]
    fn codex_argv_no_model_omits_m_flag() {
        let model: Option<&str> = None;
        let prompt = "test";
        let tmp_dir_path = std::path::Path::new("/tmp/kbench-codex-test2");
        let answer_file = tmp_dir_path.join("answer.txt");

        let mut argv: Vec<String> = vec![
            "exec".to_string(),
            "--ignore-user-config".to_string(),
            "--skip-git-repo-check".to_string(),
            "--ephemeral".to_string(),
            "--color".to_string(),
            "never".to_string(),
            "-C".to_string(),
            tmp_dir_path.to_string_lossy().to_string(),
            "-o".to_string(),
            answer_file.to_string_lossy().to_string(),
        ];

        if let Some(m) = model
            && !m.is_empty()
        {
            argv.push("-m".to_string());
            argv.push(m.to_string());
        }
        let _ = prompt;
        argv.push("-".to_string());

        assert!(!argv.contains(&"-m".to_string()));
        assert_eq!(argv.last().unwrap(), "-");
    }

    // ── Codex prompts ─────────────────────────────────────────────────────────

    #[test]
    fn codex_reader_prompt_contains_key_instructions() {
        let p = codex_reader_prompt("What color?", "Alice likes blue.", "");
        assert!(p.contains("ONLY the provided"));
        assert!(p.contains("Alice likes blue."));
        assert!(p.contains("What color?"));
        assert!(p.contains("I don't know"));
        // No date line when question_date is empty.
        assert!(!p.contains("Today's date"));
    }

    #[test]
    fn codex_reader_prompt_includes_date_when_present() {
        let p = codex_reader_prompt("When?", "ctx", "2023/05/20 (Sat) 02:21");
        assert!(p.contains("Today's date"));
        assert!(p.contains("2023/05/20 (Sat) 02:21"));
        assert!(p.contains("time-based reasoning"));
        // Recency tie-break rule for changed facts (knowledge-update).
        assert!(p.contains("MOST RECENT session date"));
    }

    #[test]
    fn codex_judge_prompt_correct_incorrect_instruction() {
        let p = codex_judge_prompt("Q?", "gold", "pred", false, "single-session-user");
        assert!(p.contains("CORRECT"));
        assert!(p.contains("INCORRECT"));
        assert!(!p.contains("abstention")); // no abstention note for non-abs
        assert!(!p.contains("PREFERENCE")); // no preference note for non-preference
    }

    #[test]
    fn codex_judge_prompt_abstention_note() {
        let p = codex_judge_prompt("Q?", "gold", "pred", true, "single-session-user");
        assert!(p.contains("abstention"));
        assert!(p.contains("does not know"));
    }

    #[test]
    fn codex_judge_prompt_preference_rubric() {
        let p = codex_judge_prompt("Q?", "gold", "pred", false, "single-session-preference");
        assert!(p.contains("PREFERENCE"));
        assert!(p.contains("rubric"));
        assert!(p.contains("honors that preference"));
    }

    // ── Filtering ─────────────────────────────────────────────────────────────

    fn make_cfg(dry_run: bool) -> LmeConfig {
        LmeConfig {
            dataset_path: PathBuf::from("dummy.json"),
            limit: 0,
            question_types: vec![],
            dry_run,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Http,
            llm_model: None,
            llm_api_key: None,
            llm_base_url: None,
        }
    }

    #[test]
    fn filter_by_question_type() {
        let instances = synthetic_fixture();
        let mut cfg = make_cfg(true);
        cfg.question_types = vec!["temporal-reasoning".to_string()];
        let filtered = filter_instances(instances, &cfg);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].question_type, "temporal-reasoning");
    }

    #[test]
    fn filter_by_limit() {
        let instances = synthetic_fixture();
        let mut cfg = make_cfg(true);
        cfg.limit = 2;
        let filtered = filter_instances(instances, &cfg);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_limit_stratifies_across_types() {
        // A fixture grouped by type (all of type A first, then B) must NOT be
        // sampled as "first N of type A" — the limit should spread across types.
        let mut instances = synthetic_fixture();
        // Force a grouped-by-type layout: two of one type, then others.
        instances.sort_by(|a, b| a.question_type.cmp(&b.question_type));
        let mut cfg = make_cfg(true);
        cfg.limit = 3;
        let filtered = filter_instances(instances, &cfg);
        assert_eq!(filtered.len(), 3);
        let distinct: std::collections::HashSet<_> =
            filtered.iter().map(|i| i.question_type.clone()).collect();
        assert!(
            distinct.len() >= 3,
            "stratified take should cover >=3 distinct types, got {distinct:?}"
        );
    }

    #[test]
    fn filter_zero_limit_returns_all() {
        let instances = synthetic_fixture();
        let n = instances.len();
        let cfg = make_cfg(true);
        let filtered = filter_instances(instances, &cfg);
        assert_eq!(filtered.len(), n);
    }

    // ── Ingest plan ───────────────────────────────────────────────────────────

    #[test]
    fn ingest_plan_uses_dates_for_temporal_types() {
        let instances = synthetic_fixture();
        // syn-002 = temporal-reasoning
        let plan = IngestPlan::from_instance(&instances[1]);
        assert_eq!(plan.question_type, "temporal-reasoning");
        assert!(plan.uses_dates);
        assert_eq!(plan.total_sessions, 2);
        assert_eq!(plan.total_turns, 3);
    }

    #[test]
    fn ingest_plan_no_dates_for_single_session() {
        let instances = synthetic_fixture();
        // syn-001 = single-session-user
        let plan = IngestPlan::from_instance(&instances[0]);
        assert!(!plan.uses_dates);
    }

    // ── Heuristic judge ───────────────────────────────────────────────────────

    #[test]
    fn heuristic_judge_substring_match() {
        assert_eq!(heuristic_judge("I think it's blue.", "blue", false), 1.0);
    }

    #[test]
    fn heuristic_judge_mismatch() {
        assert_eq!(heuristic_judge("I think it's red.", "blue", false), 0.0);
    }

    #[test]
    fn heuristic_judge_abstention_correct() {
        assert_eq!(
            heuristic_judge("I don't know the answer to that.", "unknown", true),
            1.0
        );
    }

    #[test]
    fn heuristic_judge_abstention_incorrect() {
        assert_eq!(
            heuristic_judge("The answer is definitely blue.", "unknown", true),
            0.0
        );
    }

    // ── Dry-run smoke test ────────────────────────────────────────────────────

    /// Full dry-run end-to-end: loads the synthetic fixture, runs the dry-run
    /// path, asserts structural validity of the report — no model or kimetsu
    /// calls made.
    #[test]
    fn dry_run_smoke_test_with_synthetic_fixture() {
        // Write the synthetic fixture to a temp file.
        let fixture = synthetic_fixture();
        let json = serde_json::to_string(&fixture).expect("serialize fixture");
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::io::BufWriter::new(tmp.as_file())
            .write_all(json.as_bytes())
            .expect("write fixture");
        let path = tmp.path().to_path_buf();

        let cfg = LmeConfig {
            dataset_path: path.clone(),
            limit: 0,
            question_types: vec![],
            dry_run: true,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Http,
            llm_model: None,
            llm_api_key: None,
            llm_base_url: None,
        };

        let report = run_longmemeval(&cfg).expect("dry-run should not error");

        // Structural assertions.
        assert_eq!(report.total, fixture.len(), "total instances mismatch");
        assert_eq!(
            report.instances.len(),
            fixture.len(),
            "per-instance results count mismatch"
        );

        // Dry-run: all scores should be 0 (no model calls).
        for r in &report.instances {
            assert_eq!(r.score, 0.0, "dry-run scores should all be 0");
            assert_eq!(r.predicted, "[dry-run]");
        }

        // All question types from the fixture should appear in by_type.
        for inst in &fixture {
            assert!(
                report.by_type.contains_key(&inst.question_type),
                "missing type {} in by_type",
                inst.question_type
            );
        }

        // Markdown and JSON rendering should not panic.
        let md = report.to_markdown();
        assert!(md.contains("LongMemEval"), "markdown missing title");
        let _json = report.to_json();
    }

    #[test]
    fn dry_run_smoke_test_with_codex_backend() {
        // Dry-run with codex backend selected: no codex calls made in dry-run.
        let fixture = synthetic_fixture();
        let json = serde_json::to_string(&fixture).expect("serialize fixture");
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::io::BufWriter::new(tmp.as_file())
            .write_all(json.as_bytes())
            .expect("write fixture");
        let path = tmp.path().to_path_buf();

        let cfg = LmeConfig {
            dataset_path: path.clone(),
            limit: 0,
            question_types: vec![],
            dry_run: true,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Codex,
            llm_model: None,
            llm_api_key: None,
            llm_base_url: None,
        };

        // Codex dry-run must not error (no codex calls).
        let report = run_longmemeval(&cfg).expect("codex dry-run should not error");
        assert_eq!(report.total, fixture.len());
        for r in &report.instances {
            assert_eq!(r.predicted, "[dry-run]");
        }
    }

    #[test]
    fn no_model_configured_error_message_is_actionable() {
        let cfg = LmeConfig {
            dataset_path: PathBuf::from("dummy.json"),
            limit: 1,
            question_types: vec![],
            dry_run: false,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Http,
            llm_model: None,
            llm_api_key: None,
            llm_base_url: None,
        };
        let err = cfg.require_llm().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("KBENCH_LLM_MODEL"),
            "error should mention KBENCH_LLM_MODEL; got: {msg}"
        );
        assert!(
            msg.contains("KBENCH_LLM_API_KEY"),
            "error should mention KBENCH_LLM_API_KEY; got: {msg}"
        );
    }

    #[test]
    fn codex_backend_validate_config_ok() {
        let cfg = LmeConfig {
            dataset_path: PathBuf::from("dummy.json"),
            limit: 0,
            question_types: vec![],
            dry_run: false,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Codex,
            llm_model: None,
            llm_api_key: None,
            llm_base_url: None,
        };
        // Codex backend: no API key required — validate_llm_config() must not error.
        assert!(
            cfg.validate_llm_config().is_ok(),
            "codex backend should not require API key"
        );
    }

    #[test]
    fn build_report_computes_accuracy_correctly() {
        let results = vec![
            LmeInstanceResult {
                question_id: "a".to_string(),
                question_type: "single-session-user".to_string(),
                predicted: "x".to_string(),
                gold: "x".to_string(),
                score: 1.0,
                judge_reason: "correct".to_string(),
                duration_secs: 1.0,
            },
            LmeInstanceResult {
                question_id: "b".to_string(),
                question_type: "single-session-user".to_string(),
                predicted: "y".to_string(),
                gold: "z".to_string(),
                score: 0.0,
                judge_reason: "incorrect".to_string(),
                duration_secs: 1.0,
            },
            LmeInstanceResult {
                question_id: "c".to_string(),
                question_type: "temporal-reasoning".to_string(),
                predicted: "ok".to_string(),
                gold: "ok".to_string(),
                score: 1.0,
                judge_reason: "correct".to_string(),
                duration_secs: 1.0,
            },
        ];
        let report = build_report(results, Path::new("test.json"));
        assert_eq!(report.total, 3);
        assert_eq!(report.correct, 2);
        assert!((report.overall_accuracy - 2.0 / 3.0).abs() < 1e-6);
        let ssu = &report.by_type["single-session-user"];
        assert_eq!(ssu.total, 2);
        assert_eq!(ssu.correct, 1);
        assert!((ssu.accuracy - 0.5).abs() < 1e-6);
    }
}
