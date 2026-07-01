//! BrainBench — reader-free benchmark for the Kimetsu brain's OWN behavior.
//!
//! Where LongMemEval scores an LLM *reader* answering questions over memories
//! retrieved by the brain, BrainBench scores the **brain itself**: it drives the
//! real `kimetsu` binary against authored fixtures and measures the quality of
//! the memory layer directly, with NO reader/LLM in the loop. Every metric is a
//! pure function of what the brain returns — deterministic and offline-friendly.
//!
//! ## Dimensions
//!
//! Three are content-driven:
//! - **retrieval** — does `brain context "<q>"` surface the relevant memories
//!   (recall@k, MRR) and, for knowledge-update/contradiction cases, does the
//!   *current* memory outrank the stale one (resolution correctness)?
//! - **importance** — does an authored-salient memory rank within top-k for a
//!   query that should privilege it?
//! - **dedup** — does `brain memory conflicts` flag planted near-duplicate groups?
//!
//! Two more are outcome-driven, exercising the brain's `cite`/`regret` surface:
//! - **forgetting** — `forget --dry-run --json` proposes low-usefulness,
//!   unprotected, stale memories; in outcome mode, cited "signal" memories must
//!   be protected from pruning.
//! - **calibration** — after applying `cite`/`regret` outcomes, the brain's
//!   per-memory confidence (via `memory list --json`) must respect the authored
//!   gold ordering (pairwise accuracy).
//!
//! ## Flow per scenario
//!
//!   1. Create a fresh, isolated temp Kimetsu brain workspace (temp dir +
//!      `git init --quiet` + `kimetsu init`), exactly like the LongMemEval
//!      driver. EVERY `kimetsu` subprocess sets `KIMETSU_USER_BRAIN=0` so the
//!      global cross-project brain cannot leak memories into measurements.
//!   2. Ingest the scenario's fixture memories via a single
//!      `kimetsu brain memory add-batch <file>` (one JSONL line per memory).
//!   3. Drive the dimension-specific kimetsu command(s) and score the output
//!      with pure metric functions over the fixture keys.
//!
//! ## Kimetsu CLI surface used
//!
//!   kimetsu init
//!       — create .kimetsu/ + brain.db in the temp workspace
//!   kimetsu brain memory add-batch <file.jsonl>
//!       — ingest all fixture memories in one process
//!   kimetsu brain context "<query>" --no-ambient --json --budget-tokens <N>
//!       — retrieve ranked capsules for a query
//!   kimetsu brain memory conflicts --json
//!       — list detected near-duplicate / contradiction conflicts

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

// ─── Tier + Dimension enums ──────────────────────────────────────────────────

/// Difficulty tier of a scenario. Used to bucket aggregate scores so we can see,
/// e.g., that retrieval is solid on `easy` recall but weaker on `hard` oblique
/// queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Easy,
    Medium,
    Hard,
    Complex,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Easy => "easy",
            Tier::Medium => "medium",
            Tier::Hard => "hard",
            Tier::Complex => "complex",
        }
    }
}

impl FromStr for Tier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "easy" => Ok(Tier::Easy),
            "medium" => Ok(Tier::Medium),
            "hard" => Ok(Tier::Hard),
            "complex" => Ok(Tier::Complex),
            other => Err(format!(
                "unknown tier `{other}`; expected one of: easy, medium, hard, complex"
            )),
        }
    }
}

/// Which behavior of the brain a scenario exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Dimension {
    Retrieval,
    Dedup,
    Importance,
    Forgetting,
    Calibration,
    WritePrecision,
    Graph,
}

impl Dimension {
    pub fn as_str(self) -> &'static str {
        match self {
            Dimension::Retrieval => "retrieval",
            Dimension::Dedup => "dedup",
            Dimension::Importance => "importance",
            Dimension::Forgetting => "forgetting",
            Dimension::Calibration => "calibration",
            Dimension::WritePrecision => "write-precision",
            Dimension::Graph => "graph",
        }
    }
}

impl FromStr for Dimension {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "retrieval" => Ok(Dimension::Retrieval),
            "dedup" => Ok(Dimension::Dedup),
            "importance" => Ok(Dimension::Importance),
            "forgetting" => Ok(Dimension::Forgetting),
            "calibration" => Ok(Dimension::Calibration),
            "write-precision" | "writeprecision" => Ok(Dimension::WritePrecision),
            "graph" => Ok(Dimension::Graph),
            other => Err(format!(
                "unknown dimension `{other}`; expected one of: retrieval, dedup, \
                 importance, forgetting, calibration, write-precision, graph"
            )),
        }
    }
}

// ─── Dataset types ───────────────────────────────────────────────────────────

fn default_scope() -> String {
    "project".to_string()
}
fn default_kind() -> String {
    "fact".to_string()
}
fn default_top_k() -> usize {
    4
}

/// A single fixture memory ingested into the brain. `key` is a stable handle the
/// scenario uses to refer to this memory in `relevant`/`stale`/`expect_key`/dedup
/// groups; it is NOT sent to the brain — only `text`/`scope`/`kind` are ingested.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Memory {
    pub key: String,
    pub text: String,
    #[serde(default = "default_scope")]
    pub scope: String,
    #[serde(default = "default_kind")]
    pub kind: String,
}

/// A retrieval/importance probe against an ingested scenario.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Query {
    pub query: String,
    /// Keys of memories that SHOULD surface for this query (recall/MRR targets).
    #[serde(default)]
    pub relevant: Vec<String>,
    /// Keys of stale/superseded memories that should NOT outrank the relevant
    /// ones (knowledge-update / contradiction cases).
    #[serde(default)]
    pub stale: Vec<String>,
    /// For importance scenarios: the single key expected to rank within `top_k`.
    #[serde(default)]
    pub expect_key: Option<String>,
    /// Cut-off rank for importance / top-k scoring.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

/// Dedup expectations: groups of fixture keys that are near-duplicates of one
/// another and should be flagged by `brain memory conflicts`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DedupSpec {
    pub near_duplicate_groups: Vec<Vec<String>>,
    /// Groups of DISTINCT memory keys that should NOT be flagged as a conflict.
    /// Each group tests precision: a false positive occurs when >=2 of its
    /// members' texts appear in the conflicts payload.
    #[serde(default)]
    pub must_not_flag: Vec<Vec<String>>,
}

/// Forgetting expectations. The brain's write-time importance scoring assigns a
/// usefulness by memory `kind`; `brain forget --dry-run` then proposes the
/// low-usefulness, unprotected, stale memories. With a `usefulness_floor`
/// between the "signal" and "noise" memories, the proposed set should equal
/// `expect_forgotten` — i.e. noise is pruned, signal is kept.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ForgettingSpec {
    /// Usefulness floor passed to `forget --dry-run` (memories at/below are
    /// candidates, given use_count 0 and min-age 0).
    pub usefulness_floor: f32,
    /// Keys that SHOULD be proposed for forgetting at this floor.
    #[serde(default)]
    pub expect_forgotten: Vec<String>,
    /// Minimum age (days) a memory must have to be a forget candidate. Combine
    /// with scenario `ages` to exercise the time dimension. Default 0.
    #[serde(default)]
    pub min_age_days: u32,
}

/// Calibration expectations: keys ordered MOST→LEAST trustworthy/valuable — the
/// gold confidence ordering. After applying `cite`/`regret` outcomes, the brain's
/// per-memory confidence should respect this ordering.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CalibrationSpec {
    pub ranked_keys: Vec<String>,
}

/// One turn of a transcript fed to `brain distill` for the write-precision
/// dimension. `role` is "user" or "assistant"; `text` is the message body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptTurn {
    pub role: String,
    pub text: String,
}

/// A gold lesson the distiller SHOULD capture from a transcript. It is
/// considered "captured" when ALL of its `keywords` appear (case-insensitive
/// substring) across the distilled lessons. Keep keywords short, lowercase
/// stems a paraphrase is likely to contain.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoldLesson {
    pub keywords: Vec<String>,
}

/// One authored scenario.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Scenario {
    pub id: String,
    pub dimension: Dimension,
    pub tier: Tier,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub memories: Vec<Memory>,
    #[serde(default)]
    pub queries: Vec<Query>,
    #[serde(default)]
    pub dedup: Option<DedupSpec>,
    #[serde(default)]
    pub forgetting: Option<ForgettingSpec>,
    /// Fixture keys to `brain cite` (raise usefulness/confidence) before scoring.
    #[serde(default)]
    pub cite: Vec<String>,
    /// Fixture keys to `brain regret` (lower usefulness/confidence) before scoring.
    #[serde(default)]
    pub regret: Vec<String>,
    /// Gold confidence ordering for calibration scenarios.
    #[serde(default)]
    pub calibration: Option<CalibrationSpec>,
    /// Backdate memories before scoring: key -> days-ago (exercises age-sensitive
    /// forgetting). Applied via `brain memory set-age`.
    #[serde(default)]
    pub ages: std::collections::HashMap<String, u32>,
    /// Transcript fed to `brain distill` for write-precision scenarios.
    #[serde(default)]
    pub transcript: Vec<TranscriptTurn>,
    /// Gold lessons the distiller should capture (write-precision scenarios).
    #[serde(default)]
    pub write_gold: Vec<GoldLesson>,
}

/// A reference to an external EvalFixture file (LongMemEval-style) to import as
/// retrieval scenarios. `path` is resolved relative to the dataset file's
/// directory; `id_prefix` (or the file stem) prefixes the synthesized scenario
/// ids.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalFixtureRef {
    pub path: String,
    #[serde(default)]
    pub id_prefix: String,
}

/// Directive to SYNTHESIZE calibration scenarios from a memory pool, so the
/// calibration dimension reaches release-grade case counts (tight CI) without
/// hand-authoring hundreds of near-identical JSON blocks. Each synthesized
/// scenario draws three well-separated memories from `source` (an EvalFixture
/// pool), cites one (confidence ↑), regrets one (confidence ↓), leaves one
/// neutral, and asserts the gold ordering cited > neutral > regretted.
/// Generation is fully deterministic (index-derived) so the dataset is stable
/// run-to-run.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CalibrationGenSpec {
    /// Pool fixture file (EvalFixtureFile shape) supplying realistic memory text.
    /// Resolved relative to the dataset file's directory.
    pub source: String,
    /// Number of scenarios to synthesize.
    pub count: usize,
    /// id prefix for synthesized scenarios (default "calib-gen").
    #[serde(default)]
    pub id_prefix: String,
}

/// On-disk EvalFixture file: a bag of memories plus gold-labeled retrieval
/// cases. Mirrors `dataset-correctness.json` / `dataset-100.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalFixtureFile {
    pub memories: Vec<EvalFixMemory>,
    pub cases: Vec<EvalFixCase>,
}

/// One memory in an EvalFixture file. Extra fields (e.g. `valid_to`) are ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalFixMemory {
    pub key: String,
    pub text: String,
}

/// One gold-labeled retrieval case in an EvalFixture file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalFixCase {
    pub query: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub relevant: Vec<String>,
    #[serde(default)]
    pub stale: Vec<String>,
}

/// A loaded BrainBench dataset.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BrainBenchDataset {
    pub scenarios: Vec<Scenario>,
    /// External EvalFixture files to expand into retrieval scenarios at load
    /// time. Empty (the default) preserves the original behavior exactly.
    #[serde(default)]
    pub eval_fixtures: Vec<EvalFixtureRef>,
    /// Optional directive to synthesize calibration scenarios from a memory pool
    /// at load time (release-grade case counts). None (the default) is a no-op.
    #[serde(default)]
    pub calibration_gen: Option<CalibrationGenSpec>,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Default retrieval budget for `brain context` (mirrors the CLI default well
/// enough for small fixtures; overridable via `--budget-tokens`).
const DEFAULT_BUDGET_TOKENS: usize = 12000;

/// Configuration for a BrainBench run, built from CLI args + env vars.
#[derive(Debug, Clone)]
pub struct BrainBenchConfig {
    /// Path to the dataset JSON file (a `{ "scenarios": [...] }` object).
    pub dataset_path: PathBuf,
    /// Override for the kimetsu binary (auto-resolved when None).
    pub kimetsu_bin: Option<PathBuf>,
    /// Retrieval token budget for `brain context`.
    pub budget_tokens: usize,
    /// If non-empty, only scenarios with these tiers are run.
    pub tiers: Vec<Tier>,
    /// If non-empty, only scenarios with these dimensions are run.
    pub dimensions: Vec<Dimension>,
    /// Truncate to this many scenarios after filtering (0 = no limit).
    pub limit: usize,
    /// Cheap-model provider written into the workspace `[cheap_model]` table for
    /// write-precision scenarios (drives `brain distill`).
    pub distill_provider: String,
    /// Cheap-model id written into the workspace `[cheap_model]` table for
    /// write-precision scenarios.
    pub distill_model: String,
}

impl BrainBenchConfig {
    /// Overlay environment variables. CLI values win; env fills the gaps.
    /// `KIMETSU_BIN` is consulted at resolve time (see `resolve_kimetsu_bin`),
    /// so it is not copied here — this keeps a single source of truth.
    pub fn with_env_overlay(mut self) -> Self {
        if self.budget_tokens == 0 {
            self.budget_tokens = DEFAULT_BUDGET_TOKENS;
        }
        if self.distill_provider.is_empty() {
            self.distill_provider = "ollama".to_string();
        }
        if self.distill_model.is_empty() {
            self.distill_model = "qwen2.5:3b".to_string();
        }
        self
    }
}

// ─── Error type ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum BrainBenchError {
    /// Dataset file not found or unreadable.
    DatasetLoad(String),
    /// JSON parse failed.
    DatasetParse(String),
    /// Kimetsu CLI call failed.
    KimetsuError(String),
    /// Other error.
    Other(String),
}

impl std::fmt::Display for BrainBenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainBenchError::DatasetLoad(msg) => write!(f, "dataset load error: {msg}"),
            BrainBenchError::DatasetParse(msg) => write!(f, "dataset parse error: {msg}"),
            BrainBenchError::KimetsuError(msg) => write!(f, "kimetsu error: {msg}"),
            BrainBenchError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BrainBenchError {}

// ─── Result + Report types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub id: String,
    pub dimension: Dimension,
    pub tier: Tier,
    /// 0.0–1.0 score for this scenario (0.0 when skipped or errored).
    pub score: f64,
    /// True when the dimension is a Phase-2 stub (not measured).
    pub skipped: bool,
    /// Human-readable detail: sub-metrics, achieved ranks, or an error message.
    pub detail: String,
}

/// Per-dimension significance summary: mean score, case count, and the 95% CI
/// half-width (None when n < 2). Lets the scorecard show how trustworthy each
/// dimension's number is so a thin dimension is always visible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionStat {
    pub mean: f64,
    pub n: usize,
    /// 95% CI half-width (±), or None when n < 2.
    pub ci95: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainBenchReport {
    /// RFC-3339 timestamp.
    pub generated_at: String,
    /// Path to the dataset file.
    pub dataset: String,
    /// Per-scenario results (full detail).
    pub scenarios: Vec<ScenarioResult>,
    /// "dimension/tier" -> (sum_of_scores, count). Skipped scenarios excluded.
    pub by_dimension_tier: BTreeMap<String, (f64, usize)>,
    /// Per-dimension mean + n + 95% CI (skipped scenarios excluded).
    #[serde(default)]
    pub by_dimension: BTreeMap<String, DimensionStat>,
    /// Mean score over NON-skipped scenarios (0.0–1.0).
    pub overall_index: f64,
}

impl BrainBenchReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# BrainBench Report\n\n");
        out.push_str(&format!("Generated: {}\n", self.generated_at));
        out.push_str(&format!("Dataset: {}\n\n", self.dataset));

        let total = self.scenarios.len();
        let skipped = self.scenarios.iter().filter(|s| s.skipped).count();
        out.push_str(&format!(
            "**Overall Brain Quality Index: {:.1}% ({} scenarios, {} skipped)**\n\n",
            self.overall_index * 100.0,
            total,
            skipped
        ));

        out.push_str("## By dimension (n, 95% CI)\n\n");
        out.push_str("| dimension | score | n | 95% CI |\n");
        out.push_str("|---|---|---|---|\n");
        for (dim, stat) in &self.by_dimension {
            let ci = match stat.ci95 {
                Some(h) => format!("±{:.1}%", h * 100.0),
                None => "n/a".to_string(),
            };
            out.push_str(&format!(
                "| {} | {:.1}% | {} | {} |\n",
                dim,
                stat.mean * 100.0,
                stat.n,
                ci
            ));
        }
        out.push('\n');

        out.push_str("## By dimension × tier\n\n");
        out.push_str("| dimension | tier | score | n |\n");
        out.push_str("|---|---|---|---|\n");
        for (key, (sum, count)) in &self.by_dimension_tier {
            let (dim, tier) = key.split_once('/').unwrap_or((key.as_str(), ""));
            let mean = if *count == 0 {
                0.0
            } else {
                sum / *count as f64
            };
            out.push_str(&format!(
                "| {} | {} | {:.1}% | {} |\n",
                dim,
                tier,
                mean * 100.0,
                count
            ));
        }
        out.push('\n');

        out.push_str("## Per-scenario results\n\n");
        out.push_str("| id | dim | tier | score | detail |\n");
        out.push_str("|---|---|---|---|---|\n");
        for r in &self.scenarios {
            let score_cell = if r.skipped {
                "skipped".to_string()
            } else {
                format!("{:.1}%", r.score * 100.0)
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                r.id,
                r.dimension.as_str(),
                r.tier.as_str(),
                score_cell,
                truncate_str(&r.detail, 80)
            ));
        }
        out
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize BrainBenchReport")
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max {
        s.replace('|', "\\|")
    } else {
        // Truncate on a char boundary to avoid slicing inside a multibyte char.
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end]).replace('|', "\\|")
    }
}

// ─── Metric helpers (pure functions over fixture keys) ───────────────────────

/// Fraction of distinct `relevant` keys found within the first `k` of `ranked`.
/// Returns 1.0 when `relevant` is empty (nothing to recall = perfect), and 0.0
/// when `k == 0` or `ranked` is empty.
pub fn recall_at_k(ranked: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    if k == 0 || ranked.is_empty() {
        return 0.0;
    }
    let top: &[String] = &ranked[..k.min(ranked.len())];
    let mut seen = std::collections::HashSet::new();
    let mut found = 0usize;
    for rel in relevant {
        if seen.contains(rel) {
            continue;
        }
        seen.insert(rel.clone());
        if top.iter().any(|r| r == rel) {
            found += 1;
        }
    }
    let distinct = seen.len();
    if distinct == 0 {
        return 1.0;
    }
    found as f64 / distinct as f64
}

/// Mean reciprocal rank: 1/(1-based rank) of the first `relevant` key in
/// `ranked`. 0.0 when no relevant key appears.
pub fn mrr(ranked: &[String], relevant: &[String]) -> f64 {
    if relevant.is_empty() || ranked.is_empty() {
        return 0.0;
    }
    for (i, r) in ranked.iter().enumerate() {
        if relevant.iter().any(|rel| rel == r) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// 1.0 if any `stale` key appears within the first `k` of `ranked`, else 0.0.
/// 0.0 when `stale` is empty (no stale memory planted = nothing to hit).
pub fn stale_hit_rate(ranked: &[String], stale: &[String], k: usize) -> f64 {
    if stale.is_empty() || k == 0 || ranked.is_empty() {
        return 0.0;
    }
    let top: &[String] = &ranked[..k.min(ranked.len())];
    if top.iter().any(|r| stale.iter().any(|s| s == r)) {
        1.0
    } else {
        0.0
    }
}

/// True iff the best (lowest-index) `relevant` key outranks the best `stale`
/// key. If no stale key is present, resolution trivially holds (true). If no
/// relevant key is present, resolution fails (false).
pub fn resolution_correct(ranked: &[String], relevant: &[String], stale: &[String]) -> bool {
    let best_relevant = ranked
        .iter()
        .position(|r| relevant.iter().any(|rel| rel == r));
    let best_stale = ranked.iter().position(|r| stale.iter().any(|s| s == r));
    match (best_relevant, best_stale) {
        (None, _) => false,      // relevant absent => fail
        (Some(_), None) => true, // stale absent => trivially correct
        (Some(rel), Some(sta)) => rel < sta,
    }
}

// ─── Dataset loading + filtering ─────────────────────────────────────────────

/// Load and parse a BrainBench dataset JSON file.
pub fn load_dataset(path: &Path) -> Result<BrainBenchDataset, BrainBenchError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        BrainBenchError::DatasetLoad(format!("could not read {}: {e}", path.display()))
    })?;
    serde_json::from_str::<BrainBenchDataset>(&content).map_err(|e| {
        BrainBenchError::DatasetParse(format!(
            "JSON parse failed for {}: {e}. \
             Confirm this is a BrainBench dataset (object with a `scenarios` array).",
            path.display()
        ))
    })
}

/// Tier for a synthesized EvalFixture scenario, keyed by case `kind`.
fn tier_for_kind(kind: &str) -> Tier {
    match kind {
        "recall" => Tier::Easy,
        "knowledge_update" => Tier::Medium,
        "contradiction" => Tier::Medium,
        "temporal" => Tier::Hard,
        "multi_session" => Tier::Hard,
        _ => Tier::Medium,
    }
}

/// Expand external EvalFixture references into retrieval Scenarios. For each
/// referenced file, the cases are grouped by `kind` and one Scenario is
/// synthesized per distinct kind, carrying ALL of the file's memories and the
/// kind's cases as queries. Paths are resolved relative to `base_dir` (the
/// dataset file's directory).
pub fn expand_eval_fixtures(
    refs: &[EvalFixtureRef],
    base_dir: &Path,
) -> Result<Vec<Scenario>, BrainBenchError> {
    let mut out: Vec<Scenario> = Vec::new();
    for r in refs {
        let path = base_dir.join(&r.path);
        let content = std::fs::read_to_string(&path).map_err(|e| {
            BrainBenchError::DatasetLoad(format!(
                "could not read eval fixture {}: {e}",
                path.display()
            ))
        })?;
        let fixture: EvalFixtureFile = serde_json::from_str(&content).map_err(|e| {
            BrainBenchError::DatasetParse(format!(
                "JSON parse failed for eval fixture {}: {e}",
                path.display()
            ))
        })?;

        // Prefix: explicit id_prefix, else the file stem.
        let prefix = if r.id_prefix.is_empty() {
            path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "fixture".to_string())
        } else {
            r.id_prefix.clone()
        };

        // Memories carry over to every synthesized scenario (default scope/kind).
        let memories: Vec<Memory> = fixture
            .memories
            .iter()
            .map(|m| Memory {
                key: m.key.clone(),
                text: m.text.clone(),
                scope: default_scope(),
                kind: default_kind(),
            })
            .collect();

        // Group cases by kind, preserving first-seen order for stable ids.
        let mut order: Vec<String> = Vec::new();
        let mut grouped: BTreeMap<String, Vec<&EvalFixCase>> = BTreeMap::new();
        for c in &fixture.cases {
            let entry = grouped.entry(c.kind.clone()).or_default();
            if entry.is_empty() {
                order.push(c.kind.clone());
            }
            entry.push(c);
        }

        for kind in &order {
            let cases = &grouped[kind];
            let queries: Vec<Query> = cases
                .iter()
                .map(|c| Query {
                    query: c.query.clone(),
                    relevant: c.relevant.clone(),
                    stale: c.stale.clone(),
                    expect_key: None,
                    top_k: default_top_k(),
                })
                .collect();
            out.push(Scenario {
                id: format!("{prefix}-{kind}"),
                dimension: Dimension::Retrieval,
                tier: tier_for_kind(kind),
                description: format!(
                    "imported eval fixture {} ({} cases of kind `{}`)",
                    r.path,
                    queries.len(),
                    kind
                ),
                memories: memories.clone(),
                queries,
                dedup: None,
                forgetting: None,
                cite: vec![],
                regret: vec![],
                calibration: None,
                ages: std::collections::HashMap::new(),
                transcript: vec![],
                write_gold: vec![],
            });
        }
    }
    Ok(out)
}

/// Synthesize calibration scenarios from a memory pool (see `CalibrationGenSpec`).
/// Each scenario draws three well-separated pool memories — one cited (good),
/// one untouched (neutral), one regretted (bad) — and asserts the gold ordering
/// good > neutral > bad. Deterministic: scenario `i` uses pool offset `i`, plus
/// thirds of the pool for the neutral/bad picks, so the set is identical every
/// run. Scenario-local keys (`good`/`neutral`/`bad`) are always distinct; if the
/// three drawn texts are not pairwise distinct the bad pick is nudged forward.
pub fn expand_calibration_gen(
    spec: &CalibrationGenSpec,
    base_dir: &Path,
) -> Result<Vec<Scenario>, BrainBenchError> {
    let path = base_dir.join(&spec.source);
    let content = std::fs::read_to_string(&path).map_err(|e| {
        BrainBenchError::DatasetLoad(format!(
            "could not read calibration pool {}: {e}",
            path.display()
        ))
    })?;
    let fixture: EvalFixtureFile = serde_json::from_str(&content).map_err(|e| {
        BrainBenchError::DatasetParse(format!(
            "JSON parse failed for calibration pool {}: {e}",
            path.display()
        ))
    })?;
    let pool = &fixture.memories;
    let n = pool.len();
    if n < 3 {
        return Err(BrainBenchError::DatasetLoad(format!(
            "calibration pool {} has {n} memories; need at least 3",
            path.display()
        )));
    }
    let prefix = if spec.id_prefix.is_empty() {
        "calib-gen"
    } else {
        spec.id_prefix.as_str()
    };
    let tiers = [Tier::Easy, Tier::Medium, Tier::Hard, Tier::Complex];
    let third = (n / 3).max(1);

    let mut out: Vec<Scenario> = Vec::with_capacity(spec.count);
    for i in 0..spec.count {
        let gi = i % n;
        let ni = (gi + third) % n;
        let mut bi = (gi + 2 * third) % n;
        // Guarantee three distinct pool texts (dedup at ingest would otherwise
        // merge identical memories and break the key→id mapping). Nudge the bad
        // pick forward until all three texts differ, or give up after n tries.
        let mut tries = 0;
        while tries < n
            && (pool[bi].text == pool[gi].text
                || pool[bi].text == pool[ni].text
                || bi == gi
                || bi == ni)
        {
            bi = (bi + 1) % n;
            tries += 1;
        }
        out.push(Scenario {
            id: format!("{prefix}-{i:03}"),
            dimension: Dimension::Calibration,
            tier: tiers[i % tiers.len()],
            description: format!(
                "synthesized calibration: cite good > neutral > regret bad (pool {})",
                spec.source
            ),
            memories: vec![
                Memory {
                    key: "good".to_string(),
                    text: pool[gi].text.clone(),
                    scope: default_scope(),
                    kind: default_kind(),
                },
                Memory {
                    key: "neutral".to_string(),
                    text: pool[ni].text.clone(),
                    scope: default_scope(),
                    kind: default_kind(),
                },
                Memory {
                    key: "bad".to_string(),
                    text: pool[bi].text.clone(),
                    scope: default_scope(),
                    kind: default_kind(),
                },
            ],
            queries: vec![],
            dedup: None,
            forgetting: None,
            cite: vec!["good".to_string()],
            regret: vec!["bad".to_string()],
            calibration: Some(CalibrationSpec {
                ranked_keys: vec!["good".to_string(), "neutral".to_string(), "bad".to_string()],
            }),
            ages: std::collections::HashMap::new(),
            transcript: vec![],
            write_gold: vec![],
        });
    }
    Ok(out)
}

/// Keep scenarios whose tier ∈ cfg.tiers (empty = all) and dimension ∈
/// cfg.dimensions (empty = all); then truncate to cfg.limit if > 0.
pub fn filter_scenarios(scenarios: Vec<Scenario>, cfg: &BrainBenchConfig) -> Vec<Scenario> {
    let mut out: Vec<Scenario> = scenarios
        .into_iter()
        .filter(|s| cfg.tiers.is_empty() || cfg.tiers.contains(&s.tier))
        .filter(|s| cfg.dimensions.is_empty() || cfg.dimensions.contains(&s.dimension))
        .collect();
    if cfg.limit > 0 && out.len() > cfg.limit {
        out.truncate(cfg.limit);
    }
    out
}

// ─── Kimetsu interaction ─────────────────────────────────────────────────────

/// Discover the kimetsu binary: override > env > PATH.
fn resolve_kimetsu_bin(cfg: &BrainBenchConfig) -> String {
    if let Some(p) = &cfg.kimetsu_bin {
        return p.to_string_lossy().to_string();
    }
    if let Ok(v) = std::env::var("KIMETSU_BIN") {
        return v;
    }
    "kimetsu".to_string()
}

/// Create an isolated brain workspace: temp dir + `git init --quiet` +
/// `kimetsu init`. Returns the live temp dir (kept alive for the scenario).
///
/// EVERY kimetsu call sets `KIMETSU_USER_BRAIN=0` so the global cross-project
/// brain cannot leak pre-existing memories into measurements.
fn setup_brain(kimetsu_bin: &str) -> Result<tempfile::TempDir, BrainBenchError> {
    let tmp = tempfile::Builder::new()
        .prefix("kbench-brain-")
        .tempdir()
        .map_err(|e| BrainBenchError::Other(format!("could not create temp dir: {e}")))?;
    let workspace = tmp.path();

    // git init so kimetsu's repo discovery stops here instead of climbing to an
    // enclosing repo on the drive.
    let git_out = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(workspace)
        .output()
        .map_err(|e| BrainBenchError::Other(format!("could not spawn git init: {e}")))?;
    if !git_out.status.success() {
        let stderr = String::from_utf8_lossy(&git_out.stderr);
        eprintln!("    [brainbench] warn: git init failed (non-fatal): {stderr}");
    }

    let init_out = Command::new(kimetsu_bin)
        .args(["init"])
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .output()
        .map_err(|e| {
            BrainBenchError::KimetsuError(format!("could not spawn `{kimetsu_bin} init`: {e}"))
        })?;
    if !init_out.status.success() {
        let stderr = String::from_utf8_lossy(&init_out.stderr);
        return Err(BrainBenchError::KimetsuError(format!(
            "kimetsu init failed in temp workspace: {stderr}"
        )));
    }

    Ok(tmp)
}

/// Ingest fixture memories via a single `brain memory add-batch` call. Each
/// Memory becomes one JSONL line of `{"text","scope","kind"}`.
fn ingest(workspace: &Path, kimetsu_bin: &str, memories: &[Memory]) -> Result<(), BrainBenchError> {
    let entries: Vec<String> = memories
        .iter()
        .map(|m| {
            serde_json::json!({
                "text": m.text,
                "scope": m.scope,
                "kind": m.kind,
            })
            .to_string()
        })
        .collect();

    let batch_file = workspace.join("brainbench-batch.jsonl");
    std::fs::write(&batch_file, entries.join("\n")).map_err(|e| {
        BrainBenchError::KimetsuError(format!("could not write batch ingest file: {e}"))
    })?;

    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args(["brain", "memory", "add-batch"])
        .arg(&batch_file)
        .output()
        .map_err(|e| {
            BrainBenchError::KimetsuError(format!(
                "could not spawn `{kimetsu_bin} brain memory add-batch`: {e}"
            ))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(BrainBenchError::KimetsuError(format!(
            "memory add-batch failed: {stderr}"
        )));
    }
    Ok(())
}

/// Normalize a string for fuzzy text matching: trim, lowercase, collapse all
/// internal whitespace runs to single spaces.
fn normalize(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Strip an optional leading `"<scope>:<kind> - "` summary prefix: return the
/// text after the FIRST " - " if present, else the whole string.
fn strip_prefix_summary(summary: &str) -> &str {
    match summary.find(" - ") {
        Some(idx) => &summary[idx + 3..],
        None => summary,
    }
}

/// Retrieve the ranked fixture keys for a query.
///
/// Runs `brain context "<query>" --no-ambient --json --budget-tokens <N>`,
/// parses the `capsules` array (already score-sorted), strips each capsule
/// `summary`'s prefix, normalizes it, and matches against the normalized text of
/// each fixture Memory to recover its `key`. Capsules that match no fixture
/// memory are dropped. Order is preserved.
fn retrieve_ranked_keys(
    workspace: &Path,
    kimetsu_bin: &str,
    query: &str,
    budget: usize,
    memories: &[Memory],
) -> Result<Vec<String>, BrainBenchError> {
    let budget_str = budget.to_string();
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args([
            "brain",
            "context",
            query,
            "--no-ambient",
            "--json",
            "--budget-tokens",
            &budget_str,
        ])
        .output()
        .map_err(|e| {
            BrainBenchError::KimetsuError(format!(
                "could not spawn `{kimetsu_bin} brain context`: {e}"
            ))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(BrainBenchError::KimetsuError(format!(
            "brain context failed: {stderr}"
        )));
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
        BrainBenchError::KimetsuError(format!(
            "brain context returned non-JSON output: {e}\n{stdout}"
        ))
    })?;

    // Precompute normalized fixture text -> key.
    let norm_to_key: Vec<(String, String)> = memories
        .iter()
        .map(|m| (normalize(&m.text), m.key.clone()))
        .collect();

    let mut ranked: Vec<String> = Vec::new();
    if let Some(capsules) = v.get("capsules").and_then(|c| c.as_array()) {
        for cap in capsules {
            let summary = cap.get("summary").and_then(|s| s.as_str()).unwrap_or("");
            let body = normalize(strip_prefix_summary(summary));
            if body.is_empty() {
                continue;
            }
            // Match the capsule body against fixture memory texts. Prefer exact
            // normalized equality; fall back to substring containment either way
            // (the summary may truncate or lightly reword the stored text).
            let matched = norm_to_key
                .iter()
                .find(|(norm, _)| *norm == body)
                .or_else(|| {
                    norm_to_key
                        .iter()
                        .find(|(norm, _)| body.contains(norm.as_str()) || norm.contains(&body))
                });
            if let Some((_, key)) = matched {
                ranked.push(key.clone());
            }
        }
    }
    Ok(ranked)
}

/// Run `brain memory conflicts --json` and return its stdout. Non-fatal: returns
/// "" on any failure (the conflicts surface is optional).
fn detect_conflicts_payload(workspace: &Path, kimetsu_bin: &str) -> String {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args(["brain", "memory", "conflicts", "--json"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!("    [brainbench] warn: conflicts call failed (non-fatal): {stderr}");
            String::new()
        }
        Err(e) => {
            eprintln!("    [brainbench] warn: could not spawn conflicts (non-fatal): {e}");
            String::new()
        }
    }
}

// ─── Per-dimension runners ───────────────────────────────────────────────────

/// Retrieval correctness: recall@4 + MRR + resolution for knowledge-update.
fn run_retrieval(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
    budget: usize,
) -> Result<ScenarioResult, BrainBenchError> {
    ingest(workspace, kimetsu_bin, &scenario.memories)?;

    if scenario.queries.is_empty() {
        return Ok(skeleton_result(
            scenario,
            0.0,
            "no queries in scenario".to_string(),
        ));
    }

    let mut per_query_scores: Vec<f64> = Vec::new();
    let mut recalls: Vec<f64> = Vec::new();
    let mut mrrs: Vec<f64> = Vec::new();
    let mut stale_hits: Vec<f64> = Vec::new();
    let mut resolutions: Vec<f64> = Vec::new();

    for q in &scenario.queries {
        let ranked =
            retrieve_ranked_keys(workspace, kimetsu_bin, &q.query, budget, &scenario.memories)?;
        let r4 = recall_at_k(&ranked, &q.relevant, 4);
        let m = mrr(&ranked, &q.relevant);
        let sh = stale_hit_rate(&ranked, &q.stale, 4);
        let res = resolution_correct(&ranked, &q.relevant, &q.stale);
        recalls.push(r4);
        mrrs.push(m);
        stale_hits.push(sh);
        resolutions.push(if res { 1.0 } else { 0.0 });

        // Per-query blended score: recall@4, gated by resolution when the query
        // plants a stale memory (a "current value" must outrank the old one).
        let blended = if q.stale.is_empty() {
            r4
        } else {
            r4 * if res { 1.0 } else { 0.0 }
        };
        per_query_scores.push(blended);
    }

    let score = mean(&per_query_scores);
    let detail = format!(
        "recall@4={:.2} mrr={:.2} stale-hit={:.2} resolution={:.2} ({} quer{})",
        mean(&recalls),
        mean(&mrrs),
        mean(&stale_hits),
        mean(&resolutions),
        scenario.queries.len(),
        if scenario.queries.len() == 1 {
            "y"
        } else {
            "ies"
        }
    );
    Ok(skeleton_result(scenario, score, detail))
}

/// Importance ranking: expect_key must land within top_k.
fn run_importance(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
    budget: usize,
) -> Result<ScenarioResult, BrainBenchError> {
    ingest(workspace, kimetsu_bin, &scenario.memories)?;

    // Distinct-importance mode: cite the "important" memories so their read-time
    // usefulness multiplier rises — does the important one then outrank an
    // equally-relevant peer? Tests importance scoring as a ranking factor, not
    // just semantic recall.
    if !scenario.cite.is_empty() {
        let (id_by_key, _) = list_memory_rows(workspace, kimetsu_bin, &scenario.memories)?;
        for key in &scenario.cite {
            if let Some(id) = id_by_key.get(key) {
                apply_outcome(workspace, kimetsu_bin, "cite", id)?;
            }
        }
    }

    let probes: Vec<&Query> = scenario
        .queries
        .iter()
        .filter(|q| q.expect_key.is_some())
        .collect();
    if probes.is_empty() {
        return Ok(skeleton_result(
            scenario,
            0.0,
            "no queries with expect_key".to_string(),
        ));
    }

    let mut scores: Vec<f64> = Vec::new();
    let mut ranks: Vec<String> = Vec::new();
    for q in probes {
        let expect = q.expect_key.as_ref().unwrap();
        let ranked =
            retrieve_ranked_keys(workspace, kimetsu_bin, &q.query, budget, &scenario.memories)?;
        let pos = ranked.iter().position(|k| k == expect);
        let hit = pos.map(|p| p < q.top_k).unwrap_or(false);
        scores.push(if hit { 1.0 } else { 0.0 });
        ranks.push(match pos {
            Some(p) => format!("{expect}@{}", p + 1),
            None => format!("{expect}@miss"),
        });
    }

    let score = mean(&scores);
    let detail = format!("ranks: {}", ranks.join(", "));
    Ok(skeleton_result(scenario, score, detail))
}

/// Dedup detection: how many planted near-duplicate groups does the brain flag?
fn run_dedup(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
) -> Result<ScenarioResult, BrainBenchError> {
    ingest(workspace, kimetsu_bin, &scenario.memories)?;

    let spec = match &scenario.dedup {
        Some(d) if !d.near_duplicate_groups.is_empty() || !d.must_not_flag.is_empty() => d,
        _ => {
            return Ok(skeleton_result(
                scenario,
                0.0,
                "no dedup.near_duplicate_groups defined".to_string(),
            ));
        }
    };

    let payload = detect_conflicts_payload(workspace, kimetsu_bin);
    let payload_norm = normalize(&payload);

    // Build key -> normalized text lookup for the scenario's memories.
    let key_text: std::collections::HashMap<&str, String> = scenario
        .memories
        .iter()
        .map(|m| (m.key.as_str(), normalize(&m.text)))
        .collect();

    let (score, detail) = score_dedup(spec, &key_text, &payload_norm);
    Ok(skeleton_result(scenario, score, detail))
}

/// Count how many members of a group have their (non-empty) normalized text
/// present as a substring of the conflicts payload.
fn group_text_hits(
    group: &[String],
    key_text: &std::collections::HashMap<&str, String>,
    payload_norm: &str,
) -> usize {
    let mut hits = 0usize;
    for key in group {
        if let Some(text) = key_text.get(key.as_str())
            && !text.is_empty()
            && payload_norm.contains(text.as_str())
        {
            hits += 1;
        }
    }
    hits
}

/// Balanced dedup score over true-positive (near_duplicate) and true-negative
/// (must_not_flag) groups:
///   score = (TP detected + TN correct) / (total TP groups + total TN groups)
/// A near_duplicate group is "detected" when >=2 members appear in the payload;
/// a must_not_flag group is "correct" when it does NOT (i.e. <2 members appear).
/// Falls back to 0.0 when there are no groups at all.
fn score_dedup(
    spec: &DedupSpec,
    key_text: &std::collections::HashMap<&str, String>,
    payload_norm: &str,
) -> (f64, String) {
    let total_tp = spec.near_duplicate_groups.len();
    let total_tn = spec.must_not_flag.len();

    let mut detected = 0usize;
    let mut detected_ids: Vec<String> = Vec::new();
    for (gi, group) in spec.near_duplicate_groups.iter().enumerate() {
        if group_text_hits(group, key_text, payload_norm) >= 2 {
            detected += 1;
            detected_ids.push(format!("dup{}", gi + 1));
        }
    }

    let mut tn_correct = 0usize;
    let mut fp_ids: Vec<String> = Vec::new();
    for (gi, group) in spec.must_not_flag.iter().enumerate() {
        if group_text_hits(group, key_text, payload_norm) >= 2 {
            // False positive: a distinct-but-related group got flagged.
            fp_ids.push(format!("fp{}", gi + 1));
        } else {
            tn_correct += 1;
        }
    }

    let denom = total_tp + total_tn;
    let score = if denom == 0 {
        0.0
    } else {
        (detected + tn_correct) as f64 / denom as f64
    };

    let detected_str = if detected_ids.is_empty() {
        "none".to_string()
    } else {
        detected_ids.join(",")
    };
    let fp_str = if fp_ids.is_empty() {
        "none".to_string()
    } else {
        fp_ids.join(",")
    };
    let detail = format!(
        "TP {detected}/{total_tp} detected ({detected_str}); TN {tn_correct}/{total_tn} clean (FP: {fp_str})"
    );
    (score, detail)
}

/// Forgetting: does write-time importance scoring + the forget policy prune the
/// low-value "noise" while keeping high-value "signal"? Memory `kind` drives the
/// initial usefulness; `forget --dry-run --json` at the scenario floor proposes
/// the low-usefulness, unprotected, stale memories. Score = F1 of the proposed
/// set against `expect_forgotten`.
fn run_forgetting(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
    budget: usize,
) -> Result<ScenarioResult, BrainBenchError> {
    ingest(workspace, kimetsu_bin, &scenario.memories)?;

    // Resolve key -> id once if we need to cite or age memories.
    let need_ids = !scenario.cite.is_empty() || !scenario.ages.is_empty();
    let id_by_key = if need_ids {
        list_memory_rows(workspace, kimetsu_bin, &scenario.memories)?.0
    } else {
        std::collections::HashMap::new()
    };
    // Outcome-driven mode: cite the signal memories so the forget policy protects
    // them. When `cite` is empty, the kind-based mode is unchanged.
    for key in &scenario.cite {
        if let Some(id) = id_by_key.get(key) {
            apply_outcome(workspace, kimetsu_bin, "cite", id)?;
        }
    }
    // Time dimension: backdate selected memories so the age filter engages.
    for (key, days) in &scenario.ages {
        if let Some(id) = id_by_key.get(key) {
            set_age(workspace, kimetsu_bin, id, *days)?;
        }
    }

    // Recall-preservation mode: if the scenario has signal queries, FORGET for
    // real and verify retrieval still surfaces the signal — the honest test that
    // we forgot noise, not signal (penalizes over-aggressive pruning).
    if scenario.queries.iter().any(|q| !q.relevant.is_empty()) {
        return run_forget_recall_preservation(scenario, workspace, kimetsu_bin, budget);
    }

    let spec = match &scenario.forgetting {
        Some(s) => s,
        None => {
            return Ok(skeleton_result(
                scenario,
                0.0,
                "no forgetting spec defined".to_string(),
            ));
        }
    };

    let previews = run_forget_candidates(workspace, kimetsu_bin, spec.usefulness_floor)?;

    // Map each candidate preview (a <=80-char prefix of the stored text) to a key.
    let mut proposed: Vec<String> = Vec::new();
    for prev in &previews {
        let prev_norm = normalize(prev);
        if prev_norm.is_empty() {
            continue;
        }
        if let Some(m) = scenario
            .memories
            .iter()
            .find(|m| normalize(&m.text).contains(&prev_norm))
            && !proposed.contains(&m.key)
        {
            proposed.push(m.key.clone());
        }
    }

    let score = forget_f1(&proposed, &spec.expect_forgotten);
    let detail = format!(
        "proposed [{}] vs expected [{}] @floor {:.2}",
        proposed.join(","),
        spec.expect_forgotten.join(","),
        spec.usefulness_floor
    );
    Ok(skeleton_result(scenario, score, detail))
}

/// Recall-preservation: forget for real, then verify retrieval still surfaces the
/// signal. Score = mean recall@4 on the signal queries AFTER forgetting — a good
/// policy keeps recall high (forgot only noise); an over-aggressive one prunes a
/// signal memory and recall drops. Hard to game (we don't grade a forget-list).
fn run_forget_recall_preservation(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
    budget: usize,
) -> Result<ScenarioResult, BrainBenchError> {
    let (floor, min_age) = scenario
        .forgetting
        .as_ref()
        .map(|s| (s.usefulness_floor, s.min_age_days))
        .unwrap_or((0.1, 0));

    let signal: Vec<&Query> = scenario
        .queries
        .iter()
        .filter(|q| !q.relevant.is_empty())
        .collect();

    let mut before: Vec<f64> = Vec::new();
    for q in &signal {
        let ranked =
            retrieve_ranked_keys(workspace, kimetsu_bin, &q.query, budget, &scenario.memories)?;
        before.push(recall_at_k(&ranked, &q.relevant, 4));
    }

    apply_forget(workspace, kimetsu_bin, floor, min_age)?;

    let mut after: Vec<f64> = Vec::new();
    for q in &signal {
        let ranked =
            retrieve_ranked_keys(workspace, kimetsu_bin, &q.query, budget, &scenario.memories)?;
        after.push(recall_at_k(&ranked, &q.relevant, 4));
    }

    let score = mean(&after);
    let detail = format!(
        "signal recall before={:.2} after={:.2} (forgot @floor {:.2} min-age {}d; signal must survive)",
        mean(&before),
        score,
        floor,
        min_age
    );
    Ok(skeleton_result(scenario, score, detail))
}

/// Backdate a memory's age via `brain memory set-age`.
fn set_age(
    workspace: &Path,
    kimetsu_bin: &str,
    memory_id: &str,
    days_ago: u32,
) -> Result<(), BrainBenchError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args([
            "brain",
            "memory",
            "set-age",
            "--memory-id",
            memory_id,
            "--days-ago",
        ])
        .arg(days_ago.to_string())
        .output()
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not spawn set-age: {e}")))?;
    if !out.status.success() {
        return Err(BrainBenchError::KimetsuError(format!(
            "set-age failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(())
}

/// Apply a real forget pass (`forget --yes --force-enabled`) that archives the
/// candidates so subsequent retrieval no longer surfaces them.
fn apply_forget(
    workspace: &Path,
    kimetsu_bin: &str,
    floor: f32,
    min_age: u32,
) -> Result<(), BrainBenchError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args([
            "brain",
            "forget",
            "--yes",
            "--force-enabled",
            "--min-age-days",
        ])
        .arg(min_age.to_string())
        .arg("--usefulness-floor")
        .arg(format!("{floor}"))
        .output()
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not spawn forget: {e}")))?;
    if !out.status.success() {
        return Err(BrainBenchError::KimetsuError(format!(
            "forget --yes failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(())
}

/// Outcome-driven confidence calibration. Ingest the fixture, apply `cite`/`regret`
/// outcomes to nudge per-memory confidence, then re-read confidence and score how
/// well the brain's confidence ordering respects the authored gold ordering
/// (`calibration.ranked_keys`, most→least trustworthy) via pairwise accuracy.
fn run_calibration(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
) -> Result<ScenarioResult, BrainBenchError> {
    ingest(workspace, kimetsu_bin, &scenario.memories)?;
    let spec = match &scenario.calibration {
        Some(s) if s.ranked_keys.len() >= 2 => s,
        _ => {
            return Ok(skeleton_result(
                scenario,
                0.0,
                "no calibration spec (need >=2 ranked_keys)".to_string(),
            ));
        }
    };
    let (id_by_key, _) = list_memory_rows(workspace, kimetsu_bin, &scenario.memories)?;
    for key in &scenario.cite {
        if let Some(id) = id_by_key.get(key) {
            apply_outcome(workspace, kimetsu_bin, "cite", id)?;
        }
    }
    for key in &scenario.regret {
        if let Some(id) = id_by_key.get(key) {
            apply_outcome(workspace, kimetsu_bin, "regret", id)?;
        }
    }
    // Re-read confidence AFTER outcomes.
    let (_, conf_by_key) = list_memory_rows(workspace, kimetsu_bin, &scenario.memories)?;
    let score = pairwise_order_accuracy(&spec.ranked_keys, &conf_by_key);
    let detail = format!(
        "confidence order {} | {}",
        spec.ranked_keys.join(">"),
        spec.ranked_keys
            .iter()
            .map(|k| format!("{k}={:.2}", conf_by_key.get(k).copied().unwrap_or(f64::NAN)))
            .collect::<Vec<_>>()
            .join(" "),
    );
    Ok(skeleton_result(scenario, score, detail))
}

/// Write precision: what does the distiller choose to remember from a
/// transcript? Append a `[cheap_model]` table to the workspace config, write the
/// scenario transcript as JSONL, run `brain distill <transcript> --json` (which
/// distills WITHOUT recording), then score the distilled lessons against the
/// authored `write_gold` via keyword matching (a small model paraphrases, so we
/// match short stems, not exact text).
///
/// Degrades gracefully: on a distill failure (non-zero exit / non-JSON) it
/// returns a 0.0 result whose detail starts with "error: " — it never propagates
/// a hard error, mirroring the conflicts/forget runners.
fn run_write_precision(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
    provider: &str,
    model: &str,
) -> Result<ScenarioResult, BrainBenchError> {
    // Append a cheap_model config so `brain distill` has a model to call. The
    // project.toml exists after `kimetsu init`.
    let config_path = workspace.join(".kimetsu").join("project.toml");
    let cheap_model = format!(
        "\n[cheap_model]\nenabled = true\nprovider = \"{provider}\"\nmodel = \"{model}\"\n"
    );
    use std::io::Write as _;
    match std::fs::OpenOptions::new().append(true).open(&config_path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(cheap_model.as_bytes()) {
                return Ok(skeleton_result(
                    scenario,
                    0.0,
                    format!("error: could not append cheap_model config: {e}"),
                ));
            }
        }
        Err(e) => {
            return Ok(skeleton_result(
                scenario,
                0.0,
                format!("error: could not open {}: {e}", config_path.display()),
            ));
        }
    }

    // Write the transcript as JSONL, one message per line.
    let lines: Vec<String> = scenario
        .transcript
        .iter()
        .map(|t| {
            serde_json::json!({
                "message": {
                    "role": t.role,
                    "content": [{ "type": "text", "text": t.text }],
                }
            })
            .to_string()
        })
        .collect();
    let transcript_path = workspace.join("transcript.jsonl");
    if let Err(e) = std::fs::write(&transcript_path, lines.join("\n")) {
        return Ok(skeleton_result(
            scenario,
            0.0,
            format!("error: could not write transcript: {e}"),
        ));
    }

    // Run `brain distill <transcript> --json`. This distills without recording.
    let out = match Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args(["brain", "distill"])
        .arg(&transcript_path)
        .arg("--json")
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            return Ok(skeleton_result(
                scenario,
                0.0,
                format!("error: could not spawn distill: {e}"),
            ));
        }
    };
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Ok(skeleton_result(
            scenario,
            0.0,
            format!("error: {}", stderr.trim()),
        ));
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(e) => {
            return Ok(skeleton_result(
                scenario,
                0.0,
                format!("error: distill returned non-JSON: {e}"),
            ));
        }
    };

    // Collect each element's "lesson" string.
    let distilled: Vec<String> = v
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    e.get("lesson")
                        .and_then(|l| l.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default();

    let (precision, recall, captured, on_target) =
        score_write_precision(&distilled, &scenario.write_gold);
    let score = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let detail = format!(
        "recall {captured}/{total_gold}, precision {on_target}/{total_extracted} ({n} lessons distilled)",
        total_gold = scenario.write_gold.len(),
        total_extracted = distilled.len(),
        n = distilled.len(),
    );
    Ok(skeleton_result(scenario, score, detail))
}

/// Pure keyword scoring for write-precision. Returns
/// `(precision, recall, captured_gold, on_target_extracted)`.
///
/// - A gold lesson is CAPTURED iff every (lowercased) keyword is a substring of
///   the haystack = all distilled lesson texts joined by " " (lowercased).
///   recall = captured / total_gold (1.0 when total_gold == 0).
/// - An extracted lesson is ON-TARGET iff its own lowercased text contains all
///   keywords of at least one gold lesson. precision = on_target / total_extracted
///   (1.0 when total_extracted == 0).
fn score_write_precision(distilled: &[String], gold: &[GoldLesson]) -> (f64, f64, usize, usize) {
    let haystack = distilled
        .iter()
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    // Recall: how many gold lessons are fully captured across all distilled text.
    let captured = gold
        .iter()
        .filter(|g| {
            g.keywords
                .iter()
                .all(|kw| haystack.contains(&kw.to_lowercase()))
        })
        .count();
    let recall = if gold.is_empty() {
        1.0
    } else {
        captured as f64 / gold.len() as f64
    };

    // Precision: how many extracted lessons match (all keywords of) some gold.
    let on_target = distilled
        .iter()
        .filter(|lesson| {
            let lc = lesson.to_lowercase();
            gold.iter().any(|g| {
                !g.keywords.is_empty()
                    && g.keywords.iter().all(|kw| lc.contains(&kw.to_lowercase()))
            })
        })
        .count();
    let precision = if distilled.is_empty() {
        1.0
    } else {
        on_target as f64 / distilled.len() as f64
    };

    (precision, recall, captured, on_target)
}

/// (key -> memory_id, key -> confidence) by matching normalized text via `memory list --json`.
type MemoryRows = (HashMap<String, String>, HashMap<String, f64>);

fn list_memory_rows(
    workspace: &Path,
    kimetsu_bin: &str,
    memories: &[Memory],
) -> Result<MemoryRows, BrainBenchError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args(["brain", "memory", "list", "--json"])
        .output()
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not spawn memory list: {e}")))?;
    if !out.status.success() {
        return Err(BrainBenchError::KimetsuError(format!(
            "memory list --json failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let rows: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
        BrainBenchError::KimetsuError(format!("memory list returned non-JSON: {e}\n{stdout}"))
    })?;
    let norm_to_key: Vec<(String, String)> = memories
        .iter()
        .map(|m| (normalize(&m.text), m.key.clone()))
        .collect();
    let mut id_by_key = HashMap::new();
    let mut conf_by_key = HashMap::new();
    if let Some(arr) = rows.as_array() {
        for r in arr {
            let text = r.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let id = r.get("memory_id").and_then(|v| v.as_str()).unwrap_or("");
            let conf = r.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let tn = normalize(text);
            if let Some((_, key)) = norm_to_key.iter().find(|(n, _)| *n == tn) {
                id_by_key.insert(key.clone(), id.to_string());
                conf_by_key.insert(key.clone(), conf);
            }
        }
    }
    Ok((id_by_key, conf_by_key))
}

/// Run `brain cite` or `brain regret` for one memory id.
fn apply_outcome(
    workspace: &Path,
    kimetsu_bin: &str,
    sub: &str,
    id: &str,
) -> Result<(), BrainBenchError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args(["brain", sub, "--memory-id", id])
        .output()
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not spawn {sub}: {e}")))?;
    if !out.status.success() {
        return Err(BrainBenchError::KimetsuError(format!(
            "{sub} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(())
}

/// Fraction of gold-ordered pairs whose confidence respects the order (tie = 0.5).
/// `ranked` is most→least trustworthy; key[i] should have >= confidence than key[j>i].
fn pairwise_order_accuracy(ranked: &[String], score_by_key: &HashMap<String, f64>) -> f64 {
    let mut total = 0.0;
    let mut correct = 0.0;
    for i in 0..ranked.len() {
        for j in (i + 1)..ranked.len() {
            if let (Some(a), Some(b)) = (score_by_key.get(&ranked[i]), score_by_key.get(&ranked[j]))
            {
                total += 1.0;
                if a > b {
                    correct += 1.0;
                } else if (a - b).abs() < 1e-9 {
                    correct += 0.5;
                }
            }
        }
    }
    if total == 0.0 { 1.0 } else { correct / total }
}

/// Run `brain forget --dry-run --json` at `floor` and return candidate previews.
fn run_forget_candidates(
    workspace: &Path,
    kimetsu_bin: &str,
    floor: f32,
) -> Result<Vec<String>, BrainBenchError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args([
            "brain",
            "forget",
            "--dry-run",
            "--json",
            "--min-age-days",
            "0",
            "--usefulness-floor",
        ])
        .arg(format!("{floor}"))
        .output()
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not spawn forget: {e}")))?;
    if !out.status.success() {
        return Err(BrainBenchError::KimetsuError(format!(
            "forget --dry-run --json failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
        BrainBenchError::KimetsuError(format!("forget returned non-JSON: {e}\n{stdout}"))
    })?;
    let mut previews = Vec::new();
    if let Some(cands) = v.get("candidates").and_then(|c| c.as_array()) {
        for c in cands {
            if let Some(p) = c.get("text_preview").and_then(|s| s.as_str()) {
                previews.push(p.to_string());
            }
        }
    }
    Ok(previews)
}

/// F1 of a proposed key set against a gold set. Empty/empty => 1.0.
fn forget_f1(proposed: &[String], gold: &[String]) -> f64 {
    if proposed.is_empty() && gold.is_empty() {
        return 1.0;
    }
    let hits = proposed.iter().filter(|k| gold.contains(k)).count() as f64;
    let precision = if proposed.is_empty() {
        1.0
    } else {
        hits / proposed.len() as f64
    };
    let recall = if gold.is_empty() {
        1.0
    } else {
        hits / gold.len() as f64
    };
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn skeleton_result(scenario: &Scenario, score: f64, detail: String) -> ScenarioResult {
    ScenarioResult {
        id: scenario.id.clone(),
        dimension: scenario.dimension,
        tier: scenario.tier,
        score,
        skipped: false,
        detail,
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Half-width of the 95% confidence interval for the mean of `xs` (scores in
/// [0,1]), using the normal approximation `1.96 * sample_sd / sqrt(n)`. Returns
/// `None` when `n < 2` (CI undefined). For a single dimension this answers
/// "how tight is this number?" so a release claim is not made on thin data.
fn ci95_half_width(xs: &[f64]) -> Option<f64> {
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    Some(1.96 * (var / n as f64).sqrt())
}

// ─── Graph dimension (#2): does graph-lite beat flat on multi-hop? ───────────

/// Run `kimetsu brain graph build` to populate `relates_to` edges over the
/// ingested memories. Best-effort: returns the edges-written count from the JSON
/// summary (0 on any non-fatal issue).
fn graph_build(workspace: &Path, kimetsu_bin: &str) -> Result<usize, BrainBenchError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(workspace)
        .env("KIMETSU_USER_BRAIN", "0")
        .args(["brain", "graph", "build", "--json"])
        .output()
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not spawn graph build: {e}")))?;
    if !out.status.success() {
        return Err(BrainBenchError::KimetsuError(format!(
            "graph build failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
        BrainBenchError::KimetsuError(format!("graph build non-JSON: {e}\n{stdout}"))
    })?;
    Ok(v.get("written").and_then(|w| w.as_u64()).unwrap_or(0) as usize)
}

/// Rewrite the workspace `project.toml` to select a retrieval `backend`
/// ("flat" | "graph-lite" | "graph"). `base` is the post-init config text with
/// any prior `[storage]` block stripped, so switching backends never duplicates
/// the table. Returns an error string on write failure.
fn set_storage_backend(workspace: &Path, base: &str, backend: &str) -> Result<(), BrainBenchError> {
    let config_path = workspace.join(".kimetsu").join("project.toml");
    let body = format!("{}\n[storage]\nbackend = \"{backend}\"\n", base.trim_end());
    std::fs::write(&config_path, body)
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not set backend: {e}")))
}

/// Read the post-init `project.toml`, stripping any existing `[storage]` table so
/// the backend can be set cleanly. A line-oriented strip: drop the `[storage]`
/// header and the contiguous key lines that follow it, up to the next table
/// header or blank-section boundary.
fn config_base_without_storage(workspace: &Path) -> Result<String, BrainBenchError> {
    let config_path = workspace.join(".kimetsu").join("project.toml");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| BrainBenchError::KimetsuError(format!("could not read project.toml: {e}")))?;
    let mut out: Vec<&str> = Vec::new();
    let mut in_storage = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') {
            // A new table header: enter/exit the storage skip region.
            in_storage = trimmed.starts_with("[storage]");
            if in_storage {
                continue;
            }
        }
        if in_storage {
            continue; // skip keys under [storage]
        }
        out.push(line);
    }
    Ok(out.join("\n"))
}

/// #2 multi-hop graph eval. Seed memories where the `relevant` answer memory is
/// reachable only via a `relates_to` edge from a directly-matching bridge memory,
/// build the graph, then measure recall@k under `flat` vs `graph-lite`. Score =
/// graph-lite recall (retained answer recall); detail reports the lift over flat.
/// This is the proof gate for whether the knowledge graph adds retrieval value.
fn run_graph(
    scenario: &Scenario,
    workspace: &Path,
    kimetsu_bin: &str,
    budget: usize,
) -> Result<ScenarioResult, BrainBenchError> {
    if scenario.queries.is_empty() {
        return Ok(skeleton_result(
            scenario,
            0.0,
            "no queries for graph scenario".to_string(),
        ));
    }
    ingest(workspace, kimetsu_bin, &scenario.memories)?;
    let edges = graph_build(workspace, kimetsu_bin)?;
    let base = config_base_without_storage(workspace)?;

    // Measure mean recall@k for a backend across all queries.
    let measure = |backend: &str| -> Result<f64, BrainBenchError> {
        set_storage_backend(workspace, &base, backend)?;
        let mut recalls: Vec<f64> = Vec::new();
        for q in &scenario.queries {
            let ranked =
                retrieve_ranked_keys(workspace, kimetsu_bin, &q.query, budget, &scenario.memories)?;
            recalls.push(recall_at_k(&ranked, &q.relevant, q.top_k));
        }
        Ok(mean(&recalls))
    };

    let flat = measure("flat")?;
    let graph = measure("graph-lite")?;
    let lift = graph - flat;
    let detail =
        format!("edges={edges} graph-lite recall@k={graph:.2} vs flat={flat:.2} (lift {lift:+.2})");
    // Score is the graph-lite recall: how much of the multi-hop answer the graph
    // backend recovers. A flat-only brain scores `flat` here, so a positive lift
    // is exactly the graph's contribution.
    Ok(skeleton_result(scenario, graph, detail))
}

// ─── Scenario dispatch ───────────────────────────────────────────────────────

/// Run one scenario end-to-end against a fresh isolated brain. On any error,
/// returns a 0.0-score result with `detail = "error: <e>"` instead of panicking.
pub fn run_single_scenario(
    scenario: &Scenario,
    cfg: &BrainBenchConfig,
    kimetsu_bin: &str,
) -> ScenarioResult {
    let tmp = match setup_brain(kimetsu_bin) {
        Ok(t) => t,
        Err(e) => return skeleton_result(scenario, 0.0, format!("error: {e}")),
    };
    let workspace = tmp.path();

    let result = match scenario.dimension {
        Dimension::Retrieval => run_retrieval(scenario, workspace, kimetsu_bin, cfg.budget_tokens),
        Dimension::Importance => {
            run_importance(scenario, workspace, kimetsu_bin, cfg.budget_tokens)
        }
        Dimension::Dedup => run_dedup(scenario, workspace, kimetsu_bin),
        Dimension::Forgetting => {
            run_forgetting(scenario, workspace, kimetsu_bin, cfg.budget_tokens)
        }
        Dimension::Calibration => run_calibration(scenario, workspace, kimetsu_bin),
        Dimension::WritePrecision => run_write_precision(
            scenario,
            workspace,
            kimetsu_bin,
            &cfg.distill_provider,
            &cfg.distill_model,
        ),
        Dimension::Graph => run_graph(scenario, workspace, kimetsu_bin, cfg.budget_tokens),
    };

    match result {
        Ok(r) => r,
        Err(e) => skeleton_result(scenario, 0.0, format!("error: {e}")),
    }
    // tmp dropped here — workspace cleaned up automatically.
}

// ─── Main run entry point ────────────────────────────────────────────────────

/// Run BrainBench with the given config.
pub fn run_brainbench(cfg: &BrainBenchConfig) -> Result<BrainBenchReport, BrainBenchError> {
    let dataset = load_dataset(&cfg.dataset_path)?;

    // Expand any external EvalFixture references (paths relative to the dataset
    // file's directory) and append them to the authored scenarios.
    let base_dir = cfg
        .dataset_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let mut all_scenarios = dataset.scenarios;
    let imported = expand_eval_fixtures(&dataset.eval_fixtures, &base_dir)?;
    if !imported.is_empty() {
        eprintln!(
            "brainbench: imported {} scenario(s) from {} eval fixture(s)",
            imported.len(),
            dataset.eval_fixtures.len()
        );
    }
    all_scenarios.extend(imported);

    // Synthesize calibration scenarios from a pool (release-grade case counts).
    if let Some(calib_gen) = &dataset.calibration_gen {
        let synthesized = expand_calibration_gen(calib_gen, &base_dir)?;
        eprintln!(
            "brainbench: synthesized {} calibration scenario(s) from pool {}",
            synthesized.len(),
            calib_gen.source
        );
        all_scenarios.extend(synthesized);
    }

    let scenarios = filter_scenarios(all_scenarios, cfg);

    eprintln!("brainbench: {} scenario(s) to run", scenarios.len());

    let kimetsu_bin = resolve_kimetsu_bin(cfg);

    let mut results: Vec<ScenarioResult> = Vec::new();
    for (i, scenario) in scenarios.iter().enumerate() {
        eprintln!(
            "  [{}/{}] {} | dim={} tier={} ...",
            i + 1,
            scenarios.len(),
            scenario.id,
            scenario.dimension.as_str(),
            scenario.tier.as_str()
        );
        let r = run_single_scenario(scenario, cfg, &kimetsu_bin);
        if r.skipped {
            eprintln!("    -> skipped (Phase 2)");
        } else {
            eprintln!("    -> score={:.2} | {}", r.score, r.detail);
        }
        results.push(r);
    }

    Ok(build_report(results, &cfg.dataset_path))
}

fn build_report(results: Vec<ScenarioResult>, dataset_path: &Path) -> BrainBenchReport {
    let now = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());

    // Per dimension×tier aggregate (skipped scenarios excluded).
    let mut by_dimension_tier: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for r in &results {
        if r.skipped {
            continue;
        }
        let key = format!("{}/{}", r.dimension.as_str(), r.tier.as_str());
        let e = by_dimension_tier.entry(key).or_insert((0.0, 0));
        e.0 += r.score;
        e.1 += 1;
    }

    // Per-dimension scores (skipped excluded) → mean + n + 95% CI.
    let mut scores_by_dim: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in &results {
        if r.skipped {
            continue;
        }
        scores_by_dim
            .entry(r.dimension.as_str().to_string())
            .or_default()
            .push(r.score);
    }
    let by_dimension: BTreeMap<String, DimensionStat> = scores_by_dim
        .into_iter()
        .map(|(dim, xs)| {
            let stat = DimensionStat {
                mean: mean(&xs),
                n: xs.len(),
                ci95: ci95_half_width(&xs),
            };
            (dim, stat)
        })
        .collect();

    // Overall index = mean score over NON-skipped scenarios.
    let scored: Vec<f64> = results
        .iter()
        .filter(|r| !r.skipped)
        .map(|r| r.score)
        .collect();
    let overall_index = mean(&scored);

    BrainBenchReport {
        generated_at: now,
        dataset: dataset_path.to_string_lossy().to_string(),
        scenarios: results,
        by_dimension_tier,
        by_dimension,
        overall_index,
    }
}

// ─── Synthetic fixture ───────────────────────────────────────────────────────

/// Build a small in-memory BrainBench dataset covering retrieval (incl. a
/// knowledge-update / stale case), importance, and dedup across easy + medium
/// tiers. Used by the `--synthetic` flag and unit tests — no dataset file
/// required.
pub fn synthetic_fixture() -> BrainBenchDataset {
    BrainBenchDataset {
        eval_fixtures: vec![],
        calibration_gen: None,
        scenarios: vec![
            Scenario {
                id: "syn-retrieval-easy".to_string(),
                dimension: Dimension::Retrieval,
                tier: Tier::Easy,
                description: "plain recall of a build command".to_string(),
                memories: vec![
                    Memory {
                        key: "build-cmd".to_string(),
                        text: "The project is built with `cargo build --release` from the bench directory.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                    Memory {
                        key: "test-cmd".to_string(),
                        text: "Run the test suite with `cargo test` from the workspace root.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                    Memory {
                        key: "lint-cmd".to_string(),
                        text: "Lint the codebase with `cargo clippy --all-targets`.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                ],
                queries: vec![Query {
                    query: "How do I build the release binary?".to_string(),
                    relevant: vec!["build-cmd".to_string()],
                    stale: vec![],
                    expect_key: None,
                    top_k: 4,
                }],
                dedup: None,
                forgetting: None,
                cite: vec![],
                regret: vec![],
                calibration: None,
                ages: std::collections::HashMap::new(),
                transcript: vec![],
                write_gold: vec![],
            },
            Scenario {
                id: "syn-retrieval-update".to_string(),
                dimension: Dimension::Retrieval,
                tier: Tier::Medium,
                description: "knowledge update: env var supersedes config.toml".to_string(),
                memories: vec![
                    Memory {
                        key: "cheap-model-old".to_string(),
                        text: "The cheap model is configured in config.toml under the [models] section.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                    Memory {
                        key: "cheap-model-new".to_string(),
                        text: "The cheap model is now set via the KIMETSU_CHEAP_MODEL environment variable, not config.toml.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                ],
                queries: vec![Query {
                    query: "How is the cheap model configured currently?".to_string(),
                    relevant: vec!["cheap-model-new".to_string()],
                    stale: vec!["cheap-model-old".to_string()],
                    expect_key: None,
                    top_k: 4,
                }],
                dedup: None,
                forgetting: None,
                cite: vec![],
                regret: vec![],
                calibration: None,
                ages: std::collections::HashMap::new(),
                transcript: vec![],
                write_gold: vec![],
            },
            Scenario {
                id: "syn-importance-medium".to_string(),
                dimension: Dimension::Importance,
                tier: Tier::Medium,
                description: "salient security memory should rank within top-4".to_string(),
                memories: vec![
                    Memory {
                        key: "secret-rule".to_string(),
                        text: "Never commit API keys or secrets to the repository; use the .env file which is gitignored.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                    Memory {
                        key: "format-pref".to_string(),
                        text: "The team prefers 4-space indentation in Python files.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                    Memory {
                        key: "ci-note".to_string(),
                        text: "CI runs on GitHub Actions on every push to main.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                ],
                queries: vec![Query {
                    query: "Where should secrets and API keys go?".to_string(),
                    relevant: vec![],
                    stale: vec![],
                    expect_key: Some("secret-rule".to_string()),
                    top_k: 4,
                }],
                dedup: None,
                forgetting: None,
                cite: vec![],
                regret: vec![],
                calibration: None,
                ages: std::collections::HashMap::new(),
                transcript: vec![],
                write_gold: vec![],
            },
            Scenario {
                id: "syn-dedup-easy".to_string(),
                dimension: Dimension::Dedup,
                tier: Tier::Easy,
                description: "two paraphrases of the same DB-path fact".to_string(),
                memories: vec![
                    Memory {
                        key: "db-path-a".to_string(),
                        text: "The brain database lives at .kimetsu/brain.db in the workspace.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                    Memory {
                        key: "db-path-b".to_string(),
                        text: "The workspace stores its brain database at .kimetsu/brain.db.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                    Memory {
                        key: "editor-pref".to_string(),
                        text: "The default editor for commit messages is vim.".to_string(),
                        scope: "project".to_string(),
                        kind: "fact".to_string(),
                    },
                ],
                queries: vec![],
                dedup: Some(DedupSpec {
                    near_duplicate_groups: vec![vec![
                        "db-path-a".to_string(),
                        "db-path-b".to_string(),
                    ]],
                    must_not_flag: vec![],
                }),
                forgetting: None,
                cite: vec![],
                regret: vec![],
                calibration: None,
                ages: std::collections::HashMap::new(),
                transcript: vec![],
                write_gold: vec![],
            },
        ],
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    // ── Metric: recall_at_k ───────────────────────────────────────────────────

    #[test]
    fn recall_empty_relevant_is_one() {
        assert_eq!(recall_at_k(&s(&["a", "b"]), &[], 4), 1.0);
    }

    #[test]
    fn recall_zero_k_or_empty_ranked_is_zero() {
        assert_eq!(recall_at_k(&s(&["a"]), &s(&["a"]), 0), 0.0);
        assert_eq!(recall_at_k(&[], &s(&["a"]), 4), 0.0);
    }

    #[test]
    fn recall_full_and_partial() {
        // both relevant in top-4
        assert_eq!(recall_at_k(&s(&["a", "b", "c"]), &s(&["a", "b"]), 4), 1.0);
        // only one of two relevant present
        assert_eq!(recall_at_k(&s(&["a", "x", "y"]), &s(&["a", "b"]), 4), 0.5);
        // relevant present but beyond k
        assert_eq!(
            recall_at_k(&s(&["x", "y", "z", "w", "a"]), &s(&["a"]), 4),
            0.0
        );
    }

    // ── Metric: mrr ───────────────────────────────────────────────────────────

    #[test]
    fn mrr_first_position() {
        assert_eq!(mrr(&s(&["a", "b"]), &s(&["a"])), 1.0);
    }

    #[test]
    fn mrr_second_position() {
        assert!((mrr(&s(&["x", "a"]), &s(&["a"])) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mrr_no_relevant_is_zero() {
        assert_eq!(mrr(&s(&["x", "y"]), &s(&["a"])), 0.0);
        assert_eq!(mrr(&s(&["x"]), &[]), 0.0);
    }

    // ── Metric: forget_f1 ─────────────────────────────────────────────────────

    #[test]
    fn forget_f1_perfect_and_empty() {
        assert_eq!(forget_f1(&s(&["a", "b"]), &s(&["a", "b"])), 1.0);
        assert_eq!(forget_f1(&[], &[]), 1.0);
    }

    #[test]
    fn forget_f1_partial_wrong_and_overproposed() {
        // proposed {a,c}, gold {a,b}: precision .5, recall .5 -> F1 .5
        assert!((forget_f1(&s(&["a", "c"]), &s(&["a", "b"])) - 0.5).abs() < 1e-9);
        // disjoint -> 0
        assert_eq!(forget_f1(&s(&["x"]), &s(&["a", "b"])), 0.0);
        // over-proposing (forgetting signal) drops precision below 1.0
        assert!(forget_f1(&s(&["a", "b", "c"]), &s(&["a", "b"])) < 1.0);
    }

    // ── Metric: pairwise_order_accuracy ───────────────────────────────────────

    fn conf(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn pairwise_perfect_order_is_one() {
        // ranked most→least: a > b > c, confidences strictly descending.
        let c = conf(&[("a", 0.9), ("b", 0.6), ("c", 0.3)]);
        assert_eq!(pairwise_order_accuracy(&s(&["a", "b", "c"]), &c), 1.0);
    }

    #[test]
    fn pairwise_reversed_order_is_zero() {
        // gold says a>b>c but confidences ascend => every pair wrong.
        let c = conf(&[("a", 0.3), ("b", 0.6), ("c", 0.9)]);
        assert_eq!(pairwise_order_accuracy(&s(&["a", "b", "c"]), &c), 0.0);
    }

    #[test]
    fn pairwise_one_swapped_pair_is_two_thirds() {
        // 3 keys => 3 pairs (a,b)(a,c)(b,c). b and c swapped: (b,c) wrong only.
        let c = conf(&[("a", 0.9), ("b", 0.3), ("c", 0.6)]);
        assert!((pairwise_order_accuracy(&s(&["a", "b", "c"]), &c) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn pairwise_missing_keys_are_skipped() {
        // b absent from scores => all pairs involving b skipped; only (a,c) counts
        // and it respects order => 1.0.
        let c = conf(&[("a", 0.9), ("c", 0.3)]);
        assert_eq!(pairwise_order_accuracy(&s(&["a", "b", "c"]), &c), 1.0);
    }

    #[test]
    fn pairwise_tie_is_half() {
        let c = conf(&[("a", 0.5), ("b", 0.5)]);
        assert!((pairwise_order_accuracy(&s(&["a", "b"]), &c) - 0.5).abs() < 1e-9);
    }

    // ── Metric: stale_hit_rate ────────────────────────────────────────────────

    #[test]
    fn stale_empty_is_zero() {
        assert_eq!(stale_hit_rate(&s(&["a"]), &[], 4), 0.0);
    }

    #[test]
    fn stale_hit_within_and_beyond_k() {
        assert_eq!(stale_hit_rate(&s(&["old", "a"]), &s(&["old"]), 4), 1.0);
        assert_eq!(
            stale_hit_rate(&s(&["a", "b", "c", "d", "old"]), &s(&["old"]), 4),
            0.0
        );
    }

    // ── Metric: resolution_correct ────────────────────────────────────────────

    #[test]
    fn resolution_stale_absent_is_true() {
        assert!(resolution_correct(&s(&["new", "x"]), &s(&["new"]), &[]));
        // stale planted but not retrieved => absent => true
        assert!(resolution_correct(
            &s(&["new", "x"]),
            &s(&["new"]),
            &s(&["old"])
        ));
    }

    #[test]
    fn resolution_relevant_absent_is_false() {
        assert!(!resolution_correct(
            &s(&["old", "x"]),
            &s(&["new"]),
            &s(&["old"])
        ));
        assert!(!resolution_correct(&s(&["x"]), &[], &[]));
    }

    #[test]
    fn resolution_relevant_outranks_stale() {
        assert!(resolution_correct(
            &s(&["new", "old"]),
            &s(&["new"]),
            &s(&["old"])
        ));
        assert!(!resolution_correct(
            &s(&["old", "new"]),
            &s(&["new"]),
            &s(&["old"])
        ));
    }

    // ── FromStr ───────────────────────────────────────────────────────────────

    #[test]
    fn tier_from_str() {
        assert_eq!(Tier::from_str("easy").unwrap(), Tier::Easy);
        assert_eq!(Tier::from_str("MEDIUM").unwrap(), Tier::Medium);
        assert_eq!(Tier::from_str(" hard ").unwrap(), Tier::Hard);
        assert_eq!(Tier::from_str("complex").unwrap(), Tier::Complex);
        assert!(Tier::from_str("nope").is_err());
    }

    #[test]
    fn dimension_from_str() {
        assert_eq!(
            Dimension::from_str("retrieval").unwrap(),
            Dimension::Retrieval
        );
        assert_eq!(Dimension::from_str("dedup").unwrap(), Dimension::Dedup);
        assert_eq!(
            Dimension::from_str("Importance").unwrap(),
            Dimension::Importance
        );
        assert_eq!(
            Dimension::from_str("forgetting").unwrap(),
            Dimension::Forgetting
        );
        assert_eq!(
            Dimension::from_str("calibration").unwrap(),
            Dimension::Calibration
        );
        assert_eq!(
            Dimension::from_str("write-precision").unwrap(),
            Dimension::WritePrecision
        );
        assert_eq!(
            Dimension::from_str("WritePrecision").unwrap(),
            Dimension::WritePrecision
        );
        assert!(Dimension::from_str("bogus").is_err());
    }

    // ── Metric: score_write_precision ─────────────────────────────────────────

    fn gold(groups: &[&[&str]]) -> Vec<GoldLesson> {
        groups
            .iter()
            .map(|kws| GoldLesson {
                keywords: kws.iter().map(|k| k.to_string()).collect(),
            })
            .collect()
    }

    #[test]
    fn write_precision_all_captured_all_on_target() {
        // One gold lesson; the single distilled lesson contains both keywords.
        let g = gold(&[&["test", "isolat"]]);
        let distilled = s(&["Always isolate the TEST environment per process"]);
        let (precision, recall, captured, on_target) = score_write_precision(&distilled, &g);
        assert!((recall - 1.0).abs() < 1e-9);
        assert!((precision - 1.0).abs() < 1e-9);
        assert_eq!(captured, 1);
        assert_eq!(on_target, 1);
    }

    #[test]
    fn write_precision_partial_recall() {
        // Two gold lessons; only the first is captured.
        let g = gold(&[&["serial", "sweep"], &["docker", "mount"]]);
        let distilled = s(&["Run sweeps serially to avoid crashes"]);
        let (_precision, recall, captured, _on_target) = score_write_precision(&distilled, &g);
        assert_eq!(captured, 1);
        assert!((recall - 0.5).abs() < 1e-9);
    }

    #[test]
    fn write_precision_empty_gold_recall_one() {
        let distilled = s(&["some unrelated lesson"]);
        let (_precision, recall, captured, _on_target) = score_write_precision(&distilled, &[]);
        assert_eq!(captured, 0);
        assert!((recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn write_precision_offtarget_lesson_lowers_precision() {
        // One gold lesson captured by the first distilled lesson; the second
        // distilled lesson matches no gold -> precision = 1/2.
        let g = gold(&[&["cache", "invalidat"]]);
        let distilled = s(&[
            "Remember to invalidate the cache on write",
            "The sky is blue and unrelated",
        ]);
        let (precision, recall, captured, on_target) = score_write_precision(&distilled, &g);
        assert_eq!(captured, 1);
        assert!((recall - 1.0).abs() < 1e-9);
        assert_eq!(on_target, 1);
        assert!((precision - 0.5).abs() < 1e-9);
    }

    #[test]
    fn write_precision_empty_distilled_precision_one() {
        // No lessons distilled: precision defined as 1.0, recall 0 (gold missed).
        let g = gold(&[&["foo"]]);
        let (precision, recall, captured, on_target) = score_write_precision(&[], &g);
        assert_eq!(captured, 0);
        assert_eq!(on_target, 0);
        assert!((precision - 1.0).abs() < 1e-9);
        assert!((recall - 0.0).abs() < 1e-9);
    }

    // ── normalize + strip_prefix_summary ──────────────────────────────────────

    #[test]
    fn normalize_collapses_whitespace_and_case() {
        assert_eq!(
            normalize("  Hello   WORLD\tfoo\n bar "),
            "hello world foo bar"
        );
        assert_eq!(normalize("Single"), "single");
    }

    #[test]
    fn strip_prefix_summary_strips_scope_kind() {
        assert_eq!(
            strip_prefix_summary("project:fact - the actual memory text"),
            "the actual memory text"
        );
        // No prefix -> returned whole.
        assert_eq!(strip_prefix_summary("no prefix here"), "no prefix here");
        // Only strips the FIRST " - ".
        assert_eq!(strip_prefix_summary("project:fact - a - b"), "a - b");
    }

    // ── Dataset round-trip ────────────────────────────────────────────────────

    #[test]
    fn load_dataset_roundtrips_synthetic_fixture() {
        let fixture = synthetic_fixture();
        let n = fixture.scenarios.len();
        let json = serde_json::to_string(&fixture).expect("serialize fixture");
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), json.as_bytes()).expect("write fixture");

        let loaded = load_dataset(tmp.path()).expect("load");
        assert_eq!(loaded.scenarios.len(), n);
        assert_eq!(loaded.scenarios[0].id, "syn-retrieval-easy");
        assert_eq!(loaded.scenarios[0].dimension, Dimension::Retrieval);
        assert_eq!(loaded.scenarios[0].tier, Tier::Easy);
    }

    // ── filter_scenarios ──────────────────────────────────────────────────────

    fn make_cfg() -> BrainBenchConfig {
        BrainBenchConfig {
            dataset_path: PathBuf::from("dummy.json"),
            kimetsu_bin: None,
            budget_tokens: DEFAULT_BUDGET_TOKENS,
            tiers: vec![],
            dimensions: vec![],
            limit: 0,
            distill_provider: "ollama".to_string(),
            distill_model: "qwen2.5:3b".to_string(),
        }
    }

    #[test]
    fn filter_by_tier() {
        let mut cfg = make_cfg();
        cfg.tiers = vec![Tier::Easy];
        let filtered = filter_scenarios(synthetic_fixture().scenarios, &cfg);
        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|s| s.tier == Tier::Easy));
    }

    #[test]
    fn filter_by_dimension() {
        let mut cfg = make_cfg();
        cfg.dimensions = vec![Dimension::Dedup];
        let filtered = filter_scenarios(synthetic_fixture().scenarios, &cfg);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].dimension, Dimension::Dedup);
    }

    #[test]
    fn filter_by_limit() {
        let mut cfg = make_cfg();
        cfg.limit = 2;
        let filtered = filter_scenarios(synthetic_fixture().scenarios, &cfg);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_empty_returns_all() {
        let cfg = make_cfg();
        let n = synthetic_fixture().scenarios.len();
        let filtered = filter_scenarios(synthetic_fixture().scenarios, &cfg);
        assert_eq!(filtered.len(), n);
    }

    // ── Report building ───────────────────────────────────────────────────────

    #[test]
    fn build_report_excludes_skipped_from_index() {
        let results = vec![
            ScenarioResult {
                id: "r1".to_string(),
                dimension: Dimension::Retrieval,
                tier: Tier::Easy,
                score: 1.0,
                skipped: false,
                detail: "ok".to_string(),
            },
            ScenarioResult {
                id: "r2".to_string(),
                dimension: Dimension::Retrieval,
                tier: Tier::Easy,
                score: 0.0,
                skipped: false,
                detail: "miss".to_string(),
            },
            ScenarioResult {
                id: "f1".to_string(),
                dimension: Dimension::Forgetting,
                tier: Tier::Medium,
                score: 0.0,
                skipped: true,
                detail: "skipped".to_string(),
            },
        ];
        let report = build_report(results, Path::new("test.json"));
        // overall = mean over non-skipped = (1.0 + 0.0) / 2 = 0.5
        assert!((report.overall_index - 0.5).abs() < 1e-9);
        let agg = &report.by_dimension_tier["retrieval/easy"];
        assert_eq!(agg.1, 2);
        assert!((agg.0 - 1.0).abs() < 1e-9);
        // Forgetting excluded from aggregates.
        assert!(!report.by_dimension_tier.contains_key("forgetting/medium"));
        // Markdown + JSON render without panicking.
        let md = report.to_markdown();
        assert!(md.contains("BrainBench"));
        assert!(md.contains("Overall Brain Quality Index"));
        let _ = report.to_json();
    }

    // ── expand_eval_fixtures ──────────────────────────────────────────────────

    #[test]
    fn expand_eval_fixtures_one_scenario_per_kind() {
        let fixture = EvalFixtureFile {
            memories: vec![
                EvalFixMemory {
                    key: "m-old".to_string(),
                    text: "old value".to_string(),
                },
                EvalFixMemory {
                    key: "m-new".to_string(),
                    text: "new value".to_string(),
                },
                EvalFixMemory {
                    key: "m-plain".to_string(),
                    text: "a plain fact".to_string(),
                },
            ],
            cases: vec![
                EvalFixCase {
                    query: "recall q1".to_string(),
                    kind: "recall".to_string(),
                    relevant: vec!["m-plain".to_string()],
                    stale: vec![],
                },
                EvalFixCase {
                    query: "recall q2".to_string(),
                    kind: "recall".to_string(),
                    relevant: vec!["m-plain".to_string()],
                    stale: vec![],
                },
                EvalFixCase {
                    query: "update q1".to_string(),
                    kind: "knowledge_update".to_string(),
                    relevant: vec!["m-new".to_string()],
                    stale: vec!["m-old".to_string()],
                },
                EvalFixCase {
                    query: "temporal q1".to_string(),
                    kind: "temporal".to_string(),
                    relevant: vec!["m-new".to_string()],
                    stale: vec!["m-old".to_string()],
                },
            ],
        };

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fix.json");
        std::fs::write(&path, serde_json::to_string(&fixture).unwrap()).unwrap();

        let refs = vec![EvalFixtureRef {
            path: "fix.json".to_string(),
            id_prefix: "p".to_string(),
        }];
        let scenarios = expand_eval_fixtures(&refs, dir.path()).expect("expand");

        // One scenario per distinct kind: recall, knowledge_update, temporal.
        assert_eq!(scenarios.len(), 3);

        let by_id: std::collections::HashMap<&str, &Scenario> =
            scenarios.iter().map(|s| (s.id.as_str(), s)).collect();

        let recall = by_id["p-recall"];
        assert_eq!(recall.dimension, Dimension::Retrieval);
        assert_eq!(recall.tier, Tier::Easy);
        assert_eq!(recall.queries.len(), 2);
        // All memories carry over to every synthesized scenario.
        assert_eq!(recall.memories.len(), 3);
        assert_eq!(recall.queries[0].top_k, 4);
        assert_eq!(recall.queries[0].relevant, vec!["m-plain".to_string()]);

        assert_eq!(by_id["p-knowledge_update"].tier, Tier::Medium);
        assert_eq!(by_id["p-temporal"].tier, Tier::Hard);
        // Stale carries through.
        assert_eq!(
            by_id["p-knowledge_update"].queries[0].stale,
            vec!["m-old".to_string()]
        );
    }

    #[test]
    fn expand_eval_fixtures_empty_refs_is_empty() {
        let scenarios = expand_eval_fixtures(&[], Path::new(".")).expect("expand");
        assert!(scenarios.is_empty());
    }

    // ── expand_calibration_gen ────────────────────────────────────────────────

    fn write_pool(dir: &Path, name: &str, n: usize) -> PathBuf {
        let memories: Vec<EvalFixMemory> = (0..n)
            .map(|i| EvalFixMemory {
                key: format!("m{i}"),
                text: format!("distinct fact number {i} about subsystem {i}"),
            })
            .collect();
        let fixture = EvalFixtureFile {
            memories,
            cases: vec![],
        };
        let path = dir.join(name);
        std::fs::write(&path, serde_json::to_string(&fixture).unwrap()).unwrap();
        path
    }

    #[test]
    fn expand_calibration_gen_is_deterministic_and_well_formed() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_pool(dir.path(), "pool.json", 30);
        let spec = CalibrationGenSpec {
            source: "pool.json".to_string(),
            count: 12,
            id_prefix: String::new(),
        };
        let a = expand_calibration_gen(&spec, dir.path()).expect("gen a");
        let b = expand_calibration_gen(&spec, dir.path()).expect("gen b");

        // Deterministic: same count, same ids, same memory texts.
        assert_eq!(a.len(), 12);
        let ids_a: Vec<&str> = a.iter().map(|s| s.id.as_str()).collect();
        let ids_b: Vec<&str> = b.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids_a, ids_b);
        assert_eq!(a[0].memories[0].text, b[0].memories[0].text);

        for s in &a {
            assert_eq!(s.dimension, Dimension::Calibration);
            // good/neutral/bad, all three distinct keys and distinct texts.
            assert_eq!(s.memories.len(), 3);
            let texts: std::collections::HashSet<&str> =
                s.memories.iter().map(|m| m.text.as_str()).collect();
            assert_eq!(texts.len(), 3, "scenario {} has duplicate texts", s.id);
            assert_eq!(s.cite, vec!["good".to_string()]);
            assert_eq!(s.regret, vec!["bad".to_string()]);
            assert_eq!(
                s.calibration.as_ref().unwrap().ranked_keys,
                vec!["good".to_string(), "neutral".to_string(), "bad".to_string()]
            );
        }
        // id prefix applies and ids are zero-padded + sequential.
        assert_eq!(a[0].id, "calib-gen-000");
        assert_eq!(a[11].id, "calib-gen-011");
    }

    #[test]
    fn config_base_strips_storage_table_for_clean_backend_switch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let kdir = dir.path().join(".kimetsu");
        std::fs::create_dir_all(&kdir).unwrap();
        std::fs::write(
            kdir.join("project.toml"),
            "[project]\nid = \"demo\"\n\n[storage]\nbackend = \"flat\"\n\n[lifecycle]\nforget_enabled = false\n",
        )
        .unwrap();

        let base = config_base_without_storage(dir.path()).expect("strip");
        assert!(
            !base.contains("[storage]"),
            "storage table must be stripped"
        );
        assert!(!base.contains("backend ="), "backend key must be stripped");
        assert!(base.contains("[project]"), "other tables preserved");
        assert!(base.contains("[lifecycle]"), "trailing tables preserved");

        // Setting a backend yields exactly one [storage] table.
        set_storage_backend(dir.path(), &base, "graph-lite").expect("set");
        let written = std::fs::read_to_string(kdir.join("project.toml")).unwrap();
        assert_eq!(
            written.matches("[storage]").count(),
            1,
            "exactly one [storage] table after set"
        );
        assert!(written.contains("backend = \"graph-lite\""));
    }

    #[test]
    fn expand_calibration_gen_rejects_tiny_pool() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_pool(dir.path(), "tiny.json", 2);
        let spec = CalibrationGenSpec {
            source: "tiny.json".to_string(),
            count: 5,
            id_prefix: String::new(),
        };
        assert!(expand_calibration_gen(&spec, dir.path()).is_err());
    }

    // ── ci95_half_width ───────────────────────────────────────────────────────

    #[test]
    fn ci95_half_width_undefined_for_small_n() {
        assert!(ci95_half_width(&[]).is_none());
        assert!(ci95_half_width(&[1.0]).is_none());
    }

    #[test]
    fn ci95_half_width_zero_for_constant_scores() {
        // All-1.0 → zero variance → CI half-width 0.
        let h = ci95_half_width(&[1.0, 1.0, 1.0, 1.0]).expect("ci");
        assert!(h.abs() < 1e-12, "expected ~0, got {h}");
    }

    #[test]
    fn ci95_half_width_shrinks_with_n() {
        // Same spread, more samples → tighter interval.
        let small = ci95_half_width(&[1.0, 0.0, 1.0, 0.0]).expect("ci small");
        let big: Vec<f64> = (0..40).map(|i| (i % 2) as f64).collect();
        let large = ci95_half_width(&big).expect("ci large");
        assert!(large < small, "expected {large} < {small}");
    }

    // ── score_dedup balanced scoring ──────────────────────────────────────────

    fn dedup_key_text<'a>(
        pairs: &'a [(&'a str, &'a str)],
    ) -> std::collections::HashMap<&'a str, String> {
        pairs.iter().map(|(k, t)| (*k, normalize(t))).collect()
    }

    #[test]
    fn score_dedup_detects_true_positive() {
        // One dup group, both texts present in payload => detected.
        let kt = dedup_key_text(&[
            ("a", "the brain db lives here"),
            ("b", "the brain db is here"),
        ]);
        let spec = DedupSpec {
            near_duplicate_groups: vec![vec!["a".to_string(), "b".to_string()]],
            must_not_flag: vec![],
        };
        let payload = normalize("conflict: the brain db lives here ~~ the brain db is here");
        let (score, _detail) = score_dedup(&spec, &kt, &payload);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn score_dedup_balances_precision_and_recall() {
        // One dup group (should flag) + one must_not_flag group (should NOT).
        let kt = dedup_key_text(&[
            ("a", "alpha duplicate one"),
            ("b", "alpha duplicate two"),
            ("c", "distinct config key foo"),
            ("d", "distinct config key bar"),
        ]);
        let spec = DedupSpec {
            near_duplicate_groups: vec![vec!["a".to_string(), "b".to_string()]],
            must_not_flag: vec![vec!["c".to_string(), "d".to_string()]],
        };
        // Payload flags BOTH the real dup AND (wrongly) the distinct pair.
        let payload = normalize(
            "alpha duplicate one ~~ alpha duplicate two ~~ distinct config key foo ~~ distinct config key bar",
        );
        let (score, detail) = score_dedup(&spec, &kt, &payload);
        // TP detected = 1, TN correct = 0 (false positive) => 1/2.
        assert!((score - 0.5).abs() < 1e-9);
        assert!(detail.contains("FP: fp1"));
    }

    #[test]
    fn score_dedup_perfect_precision_no_false_positive() {
        let kt = dedup_key_text(&[
            ("a", "alpha duplicate one"),
            ("b", "alpha duplicate two"),
            ("c", "distinct config key foo"),
            ("d", "distinct config key bar"),
        ]);
        let spec = DedupSpec {
            near_duplicate_groups: vec![vec!["a".to_string(), "b".to_string()]],
            must_not_flag: vec![vec!["c".to_string(), "d".to_string()]],
        };
        // Payload flags only the true duplicate; distinct pair untouched.
        let payload = normalize("alpha duplicate one ~~ alpha duplicate two");
        let (score, _detail) = score_dedup(&spec, &kt, &payload);
        // TP 1/1 + TN 1/1 => 2/2 = 1.0.
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn score_dedup_no_groups_falls_back_to_zero() {
        let kt = dedup_key_text(&[]);
        let spec = DedupSpec {
            near_duplicate_groups: vec![],
            must_not_flag: vec![],
        };
        let (score, _) = score_dedup(&spec, &kt, "");
        assert_eq!(score, 0.0);
    }
}
