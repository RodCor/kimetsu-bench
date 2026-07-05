//! `kbench` — one-command benchmark runner for kimetsu.
//!
//! Pass task IDs as positional args; everything else is auto-discovered:
//!
//! ```bash
//! cd bench
//! cargo run --release -- hello-world                 # claude+km vs claude
//! cargo run --release -- task1,task2 --codex         # codex+km vs codex
//! cargo run --release -- task1 --both                # all 4 agents
//! cargo run --release -- --full-dataset              # whole Terminal-Bench
//! cargo run --release -- --dry-run task1,task2       # synthetic, no Harbor
//! ```
//!
//! Auto-discovered:
//!   * Claude Code OAuth token  (env / .env / ~/.claude/auth.json)
//!   * Codex auth                (env / .env / ~/.codex/ bind-mount)
//!   * Linux kimetsu binary      (cache / WSL2 build / GitHub release)
//!   * Brain workspace           (the parent kimetsu repo)
//!
//! Escape hatches remain (`--agents`, `--kimetsu-binary`,
//! `--brain-workspace`, `--harbor-arg`, `--no-build`, `--output json`).

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Parser, Subcommand, ValueEnum};

mod driver;
mod drivers;
mod report;
mod setup;

use driver::{AgentConfig, BenchmarkDriver, DriverContext, Grade, RunResult, TaskId};
use drivers::brainbench::{BrainBenchConfig, Dimension, Tier, run_brainbench};
use drivers::longmemeval::{LlmBackend, LmeConfig, run_longmemeval};
use drivers::terminal_bench::{TerminalBenchDriver, parse_tasks_from_dataset_path};
use kimetsu_core::memory::{MemoryKind, MemoryScope};
use report::Report;

// ─── LongMemEval subcommand ──────────────────────────────────────────────────

/// Args for `kbench longmemeval`.
#[derive(Debug, Parser)]
#[command(
    name = "longmemeval",
    about = "Run the LongMemEval benchmark against Kimetsu's memory layer.\n\
             \n\
             Requires a downloaded LongMemEval JSON dataset and an LLM backend.\n\
             Use --dry-run to validate parsing + ingest planning without a model.\n\
             Use --synthetic to run the 5 built-in synthetic instances in REAL mode\n\
             without downloading the dataset — useful for end-to-end validation.\n\
             \n\
             LLM backends:\n\
             \n\
             --reader-backend codex  (recommended — no API key required)\n  \
             Uses `codex exec` with your ChatGPT login.  Set KIMETSU_BIN to\n  \
             the kimetsu binary.  Optionally set KBENCH_LLM_MODEL to pin the model.\n  \
             Example: kbench longmemeval --synthetic --reader-backend codex\n\
             \n\
             --reader-backend http  (default)\n  \
             Calls an OpenAI-compatible HTTP endpoint. Requires:\n  \
             KBENCH_LLM_MODEL      — model id (e.g. gpt-4o-mini)\n  \
             KBENCH_LLM_API_KEY    — API key\n  \
             KBENCH_LLM_BASE_URL   — base URL (default: https://api.openai.com/v1)\n\
             \n\
             Dataset download (for real V1 run):\n  \
             git clone https://github.com/xiaowu0162/LongMemEval\n  \
             # then point --dataset at longmemeval_s.json (fastest) or longmemeval_m.json"
)]
pub struct LongmemevalArgs {
    /// Path to the LongMemEval JSON file.
    /// One of: longmemeval_s.json (~40 sessions), longmemeval_m.json (~500 sessions),
    ///         longmemeval_oracle.json (oracle retrieval).
    /// Not required when --dry-run or --synthetic is set.
    #[arg(long, required_unless_present_any = ["dry_run", "synthetic"])]
    dataset: Option<PathBuf>,

    /// Only evaluate this many instances (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Only evaluate instances with these question types (comma-separated).
    /// Choices: single-session-user, single-session-assistant, single-session-preference,
    ///          temporal-reasoning, knowledge-update, multi-session (+ _abs variants).
    #[arg(long, value_delimiter = ',')]
    question_types: Vec<String>,

    /// Dry-run: parse + plan ingest on the synthetic built-in fixture without
    /// making any model or kimetsu calls.  Validates the harness end-to-end.
    #[arg(long, conflicts_with = "synthetic")]
    dry_run: bool,

    /// Real-mode synthetic run: run the 5 built-in synthetic instances through
    /// the full ingest→retrieve→answer→judge loop without the real dataset.
    /// Useful for end-to-end validation with the codex backend (~10 codex calls).
    /// Requires KIMETSU_BIN to be set (or --kimetsu-binary).
    /// Example: kbench longmemeval --synthetic --reader-backend codex
    #[arg(long, conflicts_with = "dry_run")]
    synthetic: bool,

    /// Override the kimetsu binary path (default: KIMETSU_BIN env or `kimetsu` on PATH).
    #[arg(long)]
    kimetsu_binary: Option<PathBuf>,

    /// LLM backend to use for answering and judging.
    /// `http`  — OpenAI-compatible API (requires KBENCH_LLM_MODEL + KBENCH_LLM_API_KEY).
    /// `codex` — `codex exec` CLI (no API key; uses ChatGPT login).
    /// Also controlled by KBENCH_LLM_BACKEND env var.
    #[arg(long, default_value = "http")]
    reader_backend: String,

    /// LLM model id. Overrides KBENCH_LLM_MODEL env var.
    /// For codex backend: passed as `-m <model>`; omit to let codex pick (default gpt-5.5).
    #[arg(long)]
    llm_model: Option<String>,

    /// LLM API key. Overrides KBENCH_LLM_API_KEY env var. (http backend only)
    #[arg(long)]
    llm_api_key: Option<String>,

    /// LLM base URL. Overrides KBENCH_LLM_BASE_URL env var. (http backend only)
    #[arg(long)]
    llm_base_url: Option<String>,

    /// Instances to run concurrently (each has its own workspace + reader
    /// calls). 0 = auto (KBENCH_PARALLEL env or 3). 1 = sequential.
    #[arg(long, default_value_t = 0)]
    parallel: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    output: OutputFormat,
}

// ─── BrainBench subcommand ───────────────────────────────────────────────────

/// Args for `kbench brainbench`.
#[derive(Debug, Parser)]
#[command(
    name = "brainbench",
    about = "Reader-free benchmark for the Kimetsu brain's OWN behavior.\n\
             \n\
             Drives the real `kimetsu` binary against authored fixtures and\n\
             scores the memory layer directly (no LLM reader in the loop):\n  \
             retrieval correctness, importance ranking, and dedup detection are\n  \
             fully implemented; forgetting + calibration are scaffolded as\n  \
             \"skipped (Phase 2)\" pending new CLI surface.\n\
             \n\
             Every kimetsu subprocess sets KIMETSU_USER_BRAIN=0 so the global\n\
             cross-project brain cannot leak into measurements.\n\
             \n\
             Use --synthetic to run the built-in fixtures without a dataset file,\n\
             or --dataset <path> to run an authored dataset (see\n\
             brainbench-data/starter.json). Set KIMETSU_BIN (or --kimetsu-binary)\n\
             to the kimetsu binary."
)]
pub struct BrainbenchArgs {
    /// Path to a BrainBench dataset JSON file (`{ "scenarios": [...] }`).
    /// Not required when --synthetic is set.
    #[arg(long, required_unless_present = "synthetic")]
    dataset: Option<PathBuf>,

    /// Run the built-in synthetic fixtures instead of a dataset file.
    #[arg(long)]
    synthetic: bool,

    /// Override the kimetsu binary path (default: KIMETSU_BIN env or `kimetsu`).
    #[arg(long)]
    kimetsu_binary: Option<PathBuf>,

    /// Retrieval token budget for `brain context`.
    #[arg(long, default_value_t = 12000)]
    budget_tokens: usize,

    /// Only run scenarios with these tiers (comma-separated): easy, medium, hard, complex.
    #[arg(long, value_delimiter = ',')]
    tiers: Vec<String>,

    /// Only run scenarios with these dimensions (comma-separated):
    /// retrieval, dedup, importance, forgetting, calibration.
    #[arg(long, value_delimiter = ',')]
    dimensions: Vec<String>,

    /// Truncate to this many scenarios after filtering (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Cheap-model provider used by write-precision scenarios (`brain distill`).
    #[arg(long, default_value = "ollama")]
    distill_provider: String,

    /// Cheap-model id used by write-precision scenarios (`brain distill`).
    #[arg(long, default_value = "qwen2.5:3b")]
    distill_model: String,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    output: OutputFormat,
}

/// Top-level kbench subcommands.  The default (no subcommand) is the
/// Terminal-Bench runner, kept backward-compatible as the outer `Cli`.
#[derive(Debug, Subcommand)]
enum KbenchCmd {
    /// Run the LongMemEval memory benchmark against Kimetsu.
    #[command(name = "longmemeval")]
    Longmemeval(LongmemevalArgs),
    /// Run the reader-free BrainBench benchmark against Kimetsu's own brain.
    #[command(name = "brainbench")]
    Brainbench(BrainbenchArgs),
    /// Run the BEAM long-term-memory benchmark (rubric LLM-judge, 10 abilities).
    #[command(name = "beam")]
    Beam(BeamArgs),
    /// Run the LoCoMo long-conversation memory benchmark (5 QA categories).
    #[command(name = "locomo")]
    Locomo(LocomoArgs),
}

/// Args for `kbench locomo` (github.com/snap-research/locomo).
#[derive(Debug, Parser)]
#[command(
    name = "locomo",
    about = "Run the LoCoMo long-conversation memory benchmark against Kimetsu.\n\
             10 conversations, ~2,000 QA pairs across 5 categories (multi-hop,\n\
             temporal, open-domain, single-hop, adversarial). Parallel by default.\n\
             Dataset: curl -L -o locomo10.json https://raw.githubusercontent.com/\
             snap-research/locomo/main/data/locomo10.json"
)]
pub struct LocomoArgs {
    /// Path to locomo10.json.
    #[arg(long)]
    dataset: PathBuf,
    /// Max questions (0 = all ~2,000). Sampled round-robin per category.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Only these categories, comma-separated (1=multi-hop, 2=temporal,
    /// 3=open-domain, 4=single-hop, 5=adversarial). Empty = all.
    #[arg(long, value_delimiter = ',')]
    categories: Vec<u8>,
    /// Parse + plan only; no kimetsu or model calls.
    #[arg(long)]
    dry_run: bool,
    /// Override the kimetsu binary path (default: KIMETSU_BIN env or `kimetsu`).
    #[arg(long)]
    kimetsu_binary: Option<PathBuf>,
    /// LLM backend for answering + judging (`codex` or `claude`).
    #[arg(long, default_value = "codex")]
    reader_backend: String,
    /// LLM model id (codex: `-m`; claude: `--model`).
    #[arg(long)]
    llm_model: Option<String>,
    /// Questions to run concurrently. 0 = auto (KBENCH_PARALLEL or 3).
    #[arg(long, default_value_t = 0)]
    parallel: usize,
    /// Run the question set N times against the SAME brains (learning curve).
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    /// Between iterations, cite the top memories of correctly-answered TRAIN
    /// questions and self-tune retrieval (`brain tune --apply`). The holdout
    /// half never produces feedback, so its curve shows generalization.
    #[arg(long)]
    learn: bool,
    /// Persistent per-sample workspace root (brains survive across iterations
    /// and restarts; ingest skipped when a brain exists). Default: temp dirs.
    #[arg(long)]
    workspace_root: Option<PathBuf>,
    /// Full-power learning: the reader reports which memories it used
    /// (CITED: line) instead of the harness citing top-k retrieved.
    #[arg(long)]
    self_cite: bool,
}

/// Args for `kbench beam` (github.com/mohammadtavakoli78/BEAM).
#[derive(Debug, Parser)]
#[command(
    name = "beam",
    about = "Run the BEAM long-term-memory benchmark against Kimetsu.\n\
             Probes 10 memory abilities over long multi-session conversations,\n\
             scoring answers with a rubric-based LLM-as-judge (codex backend).\n\
             The HF dataset (Mohammadta/BEAM-10M) ships as parquet — convert to\n\
             the JSON shape {\"conversations\":[{id,chat,probing}]} first, or use\n\
             --synthetic / --dry-run for end-to-end validation."
)]
pub struct BeamArgs {
    /// Path to the BEAM JSON dataset. Not required with --synthetic / --dry-run.
    #[arg(long)]
    dataset: Option<PathBuf>,
    /// Truncate to N conversations (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Only these ability categories (comma-separated; empty = all).
    #[arg(long, value_delimiter = ',')]
    categories: Vec<String>,
    /// Parse + plan, no kimetsu/model calls.
    #[arg(long)]
    dry_run: bool,
    /// Real-mode synthetic run (built-in fixture through the full loop).
    #[arg(long)]
    synthetic: bool,
    /// Override the kimetsu binary path (default: KIMETSU_BIN env or `kimetsu`).
    #[arg(long)]
    kimetsu_binary: Option<PathBuf>,
    /// LLM backend for answering + judging (`codex` only for now).
    #[arg(long, default_value = "codex")]
    reader_backend: String,
    /// LLM model id (codex: passed as -m; omit to let codex pick gpt-5.5).
    #[arg(long)]
    llm_model: Option<String>,
    /// Output format.
    #[arg(long, default_value = "markdown")]
    output: OutputFormat,
}

// ─── Terminal-Bench CLI ──────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(name = "kbench")]
#[command(about = "One-command Terminal-Bench runner for kimetsu (with/without kimetsu MCP).")]
#[command(version)]
struct Cli {
    /// Subcommand (e.g. `longmemeval`). When absent, kbench runs Terminal-Bench.
    #[command(subcommand)]
    subcommand: Option<KbenchCmd>,

    /// Task IDs (comma-separated or repeated). When omitted, use
    /// `--full-dataset` or `--dry-run`. Real runs without any tasks
    /// surface a friendly error.
    #[arg(value_delimiter = ',', num_args = 0..)]
    tasks: Vec<String>,

    /// Run codex+km vs codex instead of the default claude+km vs claude.
    #[arg(long, conflicts_with_all = ["both", "agents"])]
    codex: bool,

    /// Run all 4 agents (claude+km, claude, codex+km, codex).
    #[arg(long, conflicts_with_all = ["codex", "agents"])]
    both: bool,

    /// Explicit agent list (escape hatch). Comma-separated values from:
    /// `claude+km`, `claude`, `codex+km`, `codex`.
    #[arg(long, value_delimiter = ',')]
    agents: Option<Vec<String>>,

    /// Alias for the positional tasks arg. Both forms work; the
    /// positional form is preferred.
    #[arg(long = "tasks", value_delimiter = ',', hide = true)]
    tasks_flag: Option<Vec<String>>,

    /// Walk a downloaded Terminal-Bench dataset and run every task in it.
    /// Defaults to `~/.cache/harbor/tasks/packages/terminal-bench` when
    /// no path is given.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    full_dataset: Option<String>,

    /// Synthetic run: skips Harbor + Docker + auth + binary entirely.
    /// Use to validate orchestrator wiring on a fresh machine.
    #[arg(long)]
    dry_run: bool,

    /// Model name forwarded to Harbor via `--model`. Required by Harbor's
    /// codex agent. Optional for claude-code (defaults to its built-in).
    #[arg(long)]
    model: Option<String>,

    /// Skip WSL2 build; use the cached binary (even if stale) or the
    /// GitHub release download.
    #[arg(long)]
    no_build: bool,

    /// Override the auto-resolved Linux kimetsu binary path.
    #[arg(long)]
    kimetsu_binary: Option<PathBuf>,

    /// Override the auto-resolved brain workspace (host-side dir
    /// bind-mounted at `/kimetsu-workspace` in the container).
    #[arg(long)]
    brain_workspace: Option<PathBuf>,

    /// Extra args forwarded verbatim to `harbor run` (repeatable).
    /// Escape hatch for Harbor flags kbench doesn't expose directly.
    #[arg(long = "harbor-arg", allow_hyphen_values = true)]
    harbor_args: Vec<String>,

    /// Output format. `markdown` prints a readable table and saves it
    /// to `runs/auto/<timestamp>.md`. `json` does the same but in JSON.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    output: OutputFormat,

    /// Run tasks from a named programming-language family defined in the
    /// families manifest (see --families-manifest). Example: `--family python`.
    #[arg(long, conflicts_with_all = ["full_dataset"])]
    family: Option<String>,

    /// Path to a families manifest JSON file.
    /// Defaults to `<bench-dir>/datasets/prog-families-v1.json`.
    #[arg(long)]
    families_manifest: Option<PathBuf>,

    /// Print the available families with task counts and exit.
    #[arg(long)]
    list_families: bool,

    /// Hidden alias (only one driver exists). Kept for backward-compat
    /// with scripts that pass `--driver tb`.
    #[arg(long, hide = true, default_value = "tb")]
    driver: String,

    /// Internal: single-trial worker mode. The orchestrator re-execs
    /// kbench once per (task, agent) with this flag so each Harbor
    /// invocation runs in a FRESH process — and thus a fresh, valid cwd.
    /// (Harbor 0.8 on WSL2/DrvFs crashes in pyiceberg's os.getcwd() on
    /// the 2nd invocation within one process: the inherited cwd handle
    /// goes stale after the 1st run's Docker churn.) The worker runs
    /// exactly one trial, writes {run, grade} JSON to this path, and
    /// exits; the orchestrator reads it back.
    #[arg(long, hide = true)]
    worker_result: Option<PathBuf>,

    /// Internal: shared absolute run directory for this invocation's
    /// Harbor artifacts. The orchestrator computes it once
    /// (`bench/runs/<run-ts>`) and passes it to every worker so all trials
    /// group under one run dir on the bench drive — NOT the worker's `/tmp`
    /// cwd, which lives on the WSL ext4 vhdx (C:). Keeping the cwd on `/tmp`
    /// dodges the DrvFs `getcwd` staleness crash; an absolute output dir on
    /// the bench drive keeps the bulky per-trial artifacts off C:.
    #[arg(long, hide = true)]
    run_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Markdown,
    Json,
}

fn parse_agent(label: &str) -> Result<AgentConfig, String> {
    match label.trim() {
        "claude+km" | "claude+kimetsu" => Ok(AgentConfig::ClaudePlusKimetsu),
        "claude" => Ok(AgentConfig::ClaudeAlone),
        "codex+km" | "codex+kimetsu" => Ok(AgentConfig::CodexPlusKimetsu),
        "codex" => Ok(AgentConfig::CodexAlone),
        other => Err(format!(
            "unknown agent `{other}`; expected: claude+km, claude, codex+km, codex"
        )),
    }
}

fn resolve_agents(cli: &Cli) -> Result<Vec<AgentConfig>, String> {
    if let Some(list) = &cli.agents {
        return list.iter().map(|s| parse_agent(s)).collect();
    }
    if cli.both {
        return Ok(vec![
            AgentConfig::ClaudePlusKimetsu,
            AgentConfig::ClaudeAlone,
            AgentConfig::CodexPlusKimetsu,
            AgentConfig::CodexAlone,
        ]);
    }
    if cli.codex {
        return Ok(vec![AgentConfig::CodexPlusKimetsu, AgentConfig::CodexAlone]);
    }
    Ok(vec![
        AgentConfig::ClaudePlusKimetsu,
        AgentConfig::ClaudeAlone,
    ])
}

fn collect_tasks(cli: &Cli) -> Vec<String> {
    let mut all = cli.tasks.clone();
    if let Some(flag) = &cli.tasks_flag {
        all.extend(flag.iter().cloned());
    }
    all
}

fn print_no_task_error() {
    eprintln!();
    eprintln!("kbench: ERROR — no tasks specified for a real run.");
    eprintln!();
    eprintln!("  Examples:");
    eprintln!("    cargo run --release -- hello-world");
    eprintln!("    cargo run --release -- task1,task2 --codex --model gpt-5-codex-2025-08-19");
    eprintln!("    cargo run --release -- task1 --both --model gpt-5-codex-2025-08-19");
    eprintln!("    cargo run --release -- --full-dataset");
    eprintln!("    cargo run --release -- --dry-run hello-world,git-bisect");
    eprintln!();
}

fn detect_paths() -> (PathBuf, PathBuf, PathBuf) {
    let bench_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = bench_dir
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| bench_dir.clone());
    let cache_dir = bench_dir.join(".cache");
    let _ = std::fs::create_dir_all(&cache_dir);
    (bench_dir, repo_root, cache_dir)
}

fn resolve_dataset_path(input: &str) -> PathBuf {
    if !input.is_empty() {
        return PathBuf::from(input);
    }
    let default = dirs::cache_dir().or_else(dirs::home_dir).map(|h| {
        h.join(if dirs::cache_dir().is_some() {
            "harbor"
        } else {
            ".cache/harbor"
        })
        .join("tasks")
        .join("packages")
        .join("terminal-bench")
    });
    // On Windows dirs::cache_dir() returns AppData\Local which isn't where
    // harbor stores datasets. Harbor uses ~/.cache/harbor everywhere.
    if let Some(home) = dirs::home_dir() {
        let harbor_default = home
            .join(".cache")
            .join("harbor")
            .join("tasks")
            .join("packages")
            .join("terminal-bench");
        if harbor_default.is_dir() {
            return harbor_default;
        }
    }
    default.unwrap_or_else(|| PathBuf::from("terminal-bench"))
}

fn save_report(runs_dir: &Path, body: &str, format: OutputFormat) -> Option<PathBuf> {
    let ext = match format {
        OutputFormat::Markdown => "md",
        OutputFormat::Json => "json",
    };
    let stamp = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "now".to_string())
        .replace(':', "-");
    let path = runs_dir.join(format!("{stamp}.{ext}"));
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, body) {
        Ok(_) => Some(path),
        Err(e) => {
            eprintln!(
                "kbench: warn: could not save report to {}: {e}",
                path.display()
            );
            None
        }
    }
}

fn default_families_manifest(bench_dir: &Path) -> PathBuf {
    bench_dir.join("datasets").join("prog-families-v1.json")
}

fn resolve_family_tasks(manifest_path: &Path, family: &str) -> Result<Vec<TaskId>, String> {
    let content = std::fs::read_to_string(manifest_path).map_err(|e| {
        format!(
            "could not read families manifest {}: {e}",
            manifest_path.display()
        )
    })?;
    let manifest: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("invalid JSON in manifest: {e}"))?;
    let families = manifest["families"]
        .as_object()
        .ok_or("manifest missing 'families' object")?;
    let entry = families.get(family).ok_or_else(|| {
        let names: Vec<&str> = families.keys().map(|k| k.as_str()).collect();
        let mut sorted = names;
        sorted.sort_unstable();
        format!(
            "unknown family `{family}`; available: {}",
            sorted.join(", ")
        )
    })?;
    let tasks = entry["tasks"]
        .as_array()
        .ok_or("family entry missing 'tasks' array")?;
    Ok(tasks
        .iter()
        .filter_map(|v| v.as_str())
        .map(|s| TaskId(s.to_string()))
        .collect())
}

fn print_families(manifest_path: &Path) {
    let content = match std::fs::read_to_string(manifest_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("kbench: could not read {}: {e}", manifest_path.display());
            std::process::exit(1);
        }
    };
    let manifest: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("kbench: invalid JSON in manifest: {e}");
            std::process::exit(1);
        }
    };
    let Some(families) = manifest["families"].as_object() else {
        eprintln!("kbench: manifest missing 'families' object");
        std::process::exit(1);
    };
    let desc = manifest["description"].as_str().unwrap_or("");
    if !desc.is_empty() {
        println!("{desc}");
        println!();
    }
    let mut names: Vec<&str> = families.keys().map(|k| k.as_str()).collect();
    names.sort_unstable();
    for name in names {
        let entry = &families[name];
        let count = entry["tasks"].as_array().map(|a| a.len()).unwrap_or(0);
        let fdesc = entry["description"].as_str().unwrap_or("");
        println!("  {name:<16} {count:>2} tasks  —  {fdesc}");
    }
}

/// Auto-accept every pending proposal in the shared brain workspace.
/// `kimetsu_benchmark_record_outcome` lands `semantic_operator` and
/// `anti_pattern` proposals in PENDING status — the broker only
/// retrieves ACCEPTED memories, so without this step Claude's rich
/// transferable lessons stay invisible to subsequent trials.
///
/// In production use, a human reviews proposals before accepting. For
/// the bench gauntlet we trust Claude's `record_outcome` calls and
/// accept everything so transfer learning can actually transfer.
fn auto_accept_proposals(workspace: &Path) {
    use kimetsu_brain::project::{
        AcceptOverrides, ProposalFilter, accept_proposal, list_proposals,
    };
    let filter = ProposalFilter {
        status: Some("pending".to_string()),
        limit: 200,
        ..Default::default()
    };
    let proposals = match list_proposals(workspace, filter) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("    -> auto-accept: list_proposals failed: {e}");
            return;
        }
    };
    if proposals.is_empty() {
        return;
    }
    let mut accepted = 0usize;
    for p in &proposals {
        match accept_proposal(workspace, &p.proposal_id, AcceptOverrides::default()) {
            Ok(_) => accepted += 1,
            Err(e) => eprintln!("    -> auto-accept {}: {e}", p.proposal_id),
        }
    }
    eprintln!(
        "    -> auto-accepted {accepted}/{} pending proposals",
        proposals.len()
    );
}

/// Post-trial structured ingest: write a memory.accepted event into
/// the shared brain workspace summarizing what happened. Runs for
/// EVERY trial (both +km and baseline) so the brain accumulates an
/// outcome log even when the agent didn't call kimetsu itself.
///
/// The text is short and includes: task id, agent label, pass/fail,
/// duration, and the failure reason (if any). Kept terse so retrieval
/// later doesn't dump huge episodic blobs into the context window.
fn ingest_trial_outcome(workspace: &Path, run: &RunResult, grade: &Grade) {
    let verdict = if grade.score >= 1.0 {
        "passed"
    } else if grade.score > 0.0 {
        "partial"
    } else {
        "failed"
    };
    let reason = grade
        .reason
        .as_deref()
        .map(|r| format!(" reason=\"{}\"", r.replace('"', "'")))
        .unwrap_or_default();
    let text = format!(
        "terminal-bench task `{}` under `{}` {} in {:.0}s (exit={}).{}",
        run.task.0,
        run.agent.label(),
        verdict,
        run.duration_secs,
        run.exit_code,
        reason,
    );
    // Kimetsu's add_memory walks UP from `start` looking for `.kimetsu/`,
    // so pass the brain workspace directly. Failures here are non-fatal:
    // the bench report is the source of truth; this is a side effect.
    match kimetsu_brain::project::add_memory(
        workspace,
        MemoryScope::Project,
        MemoryKind::Fact,
        &text,
    ) {
        Ok(id) => eprintln!("    -> ingested as memory {id}"),
        Err(e) => eprintln!("    -> brain ingest failed (non-fatal): {e}"),
    }
}

/// Run one trial in-process: run + grade, then (for real runs only)
/// ingest the outcome into the shared brain and auto-accept proposals so
/// the NEXT trial's broker query can see them. Used directly for dry-runs
/// and inside the single-trial worker.
fn run_and_grade_trial(
    driver: &mut dyn BenchmarkDriver,
    task: &TaskId,
    agent: AgentConfig,
    dry_run: bool,
    workspace: &Path,
) -> Result<(RunResult, Grade), String> {
    let run = driver
        .run(task, agent)
        .map_err(|e| format!("run failed: {e}"))?;
    let grade = driver
        .grade(&run)
        .map_err(|e| format!("grade failed: {e}"))?;
    if !dry_run {
        ingest_trial_outcome(workspace, &run, &grade);
        auto_accept_proposals(workspace);
    }
    Ok((run, grade))
}

/// Run one trial in an isolated subprocess (a re-exec of kbench in
/// `--worker-result` mode). Each real Harbor invocation thus gets a fresh
/// process — and a fresh, valid cwd. Harbor 0.8 on WSL2/DrvFs otherwise
/// crashes in pyiceberg's `os.getcwd()` on the 2nd invocation within one
/// process (the inherited cwd handle goes stale after the 1st run's Docker
/// churn). The worker writes {run, grade} JSON which we read back here.
///
/// Auth is NOT forwarded on the command line — the worker re-derives it
/// from the environment / `.env` via its own `setup::discover`, so the
/// OAuth token never lands in argv.
fn run_trial_isolated(
    cli: &Cli,
    setup: &setup::Setup,
    task: &TaskId,
    agent: AgentConfig,
    cache_dir: &Path,
    run_root: &Path,
) -> Result<(RunResult, Grade), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe failed: {e}"))?;
    let results_dir = cache_dir.join("worker-results");
    std::fs::create_dir_all(&results_dir)
        .map_err(|e| format!("could not create {}: {e}", results_dir.display()))?;
    let result_file = results_dir.join(format!("{}-{}.json", task.0, agent.label()));
    let _ = std::fs::remove_file(&result_file);

    let mut cmd = Command::new(&exe);
    cmd.arg(&task.0);
    cmd.arg("--agents").arg(agent.label());
    // Pin the worker to the SAME shared brain workspace + resolved binary so
    // transfer learning accumulates across trials and no rebuild happens.
    cmd.arg("--brain-workspace").arg(&setup.brain_workspace);
    if let Some(bin) = &setup.kimetsu_binary {
        cmd.arg("--kimetsu-binary").arg(bin);
    }
    if let Some(model) = &cli.model {
        cmd.arg("--model").arg(model);
    }
    for ha in &cli.harbor_args {
        cmd.arg("--harbor-arg").arg(ha);
    }
    cmd.arg("--no-build");
    cmd.arg("--worker-result").arg(&result_file);
    // Pin every worker to the SAME absolute run dir on the bench drive so all
    // trials group under one runs/<run-ts>/ on E:, not under the worker's
    // /tmp cwd (the WSL ext4 vhdx, i.e. C:). The cwd stays on /tmp below to
    // avoid the DrvFs getcwd staleness crash; only the OUTPUT moves to E:.
    cmd.arg("--run-dir").arg(run_root);

    // Run the worker with a cwd on a REAL Linux filesystem (/tmp = ext4 in
    // WSL2), never the DrvFs bench dir on /mnt/e. DrvFs directory handles go
    // stale after Docker churn — the very thing that crashes Harbor's
    // os.getcwd() AND (observed) breaks the parent's ability to posix_spawn
    // the 2nd worker (ENOENT). Harbor's relative `runs/` paths resolve
    // self-consistently under this cwd; everything else we pass is absolute.
    let worker_cwd = std::env::temp_dir().join(format!("kbench-{}-{}", task.0, agent.label()));
    std::fs::create_dir_all(&worker_cwd)
        .map_err(|e| format!("could not create worker cwd {}: {e}", worker_cwd.display()))?;
    cmd.current_dir(&worker_cwd);

    // Inherit stdio so the worker's setup/Harbor progress streams live.
    let status = cmd
        .status()
        .map_err(|e| format!("could not spawn trial worker: {e}"))?;
    if !status.success() {
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string());
        return Err(format!(
            "trial worker failed (exit {code}); see output above"
        ));
    }

    let body = std::fs::read_to_string(&result_file).map_err(|e| {
        format!(
            "worker produced no result file {}: {e}",
            result_file.display()
        )
    })?;
    let _ = std::fs::remove_file(&result_file);
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("bad worker result JSON: {e}"))?;
    let run: RunResult = serde_json::from_value(v["run"].clone())
        .map_err(|e| format!("bad run in worker result: {e}"))?;
    let grade: Grade = serde_json::from_value(v["grade"].clone())
        .map_err(|e| format!("bad grade in worker result: {e}"))?;
    Ok((run, grade))
}

fn run_longmemeval_cmd(args: LongmemevalArgs, bench_dir: &Path) {
    use drivers::longmemeval::synthetic_fixture;

    // Parse the backend selector early so we can emit useful errors.
    let llm_backend = LlmBackend::from_str(&args.reader_backend).unwrap_or_else(|| {
        eprintln!(
            "kbench longmemeval: unknown --reader-backend `{}`; expected `http` or `codex`",
            args.reader_backend
        );
        std::process::exit(1);
    });

    // Determine dataset path.
    // --dry-run: synthetic fixture, no real calls.
    // --synthetic: synthetic fixture, real calls (ingest + LLM).
    // otherwise: --dataset path is required.
    let (dataset_path, dry_run) = if args.dry_run {
        // Dry-run: write synthetic fixture to temp file.
        let fixture = synthetic_fixture();
        let json = serde_json::to_string(&fixture).expect("serialize synthetic fixture");
        let tmp_path = std::env::temp_dir().join("kbench-lme-synthetic.json");
        std::fs::write(&tmp_path, &json).unwrap_or_else(|e| {
            eprintln!("kbench longmemeval: could not write synthetic fixture: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "longmemeval: using built-in synthetic fixture for dry-run ({})",
            tmp_path.display()
        );
        (tmp_path, true)
    } else if args.synthetic {
        // Real-mode synthetic: write fixture + run the full loop.
        let fixture = synthetic_fixture();
        let json = serde_json::to_string(&fixture).expect("serialize synthetic fixture");
        let tmp_path = std::env::temp_dir().join("kbench-lme-synthetic-real.json");
        std::fs::write(&tmp_path, &json).unwrap_or_else(|e| {
            eprintln!("kbench longmemeval: could not write synthetic fixture: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "longmemeval: REAL-MODE synthetic run ({} instances, backend={}) — ({})",
            fixture.len(),
            llm_backend.as_str(),
            tmp_path.display()
        );
        (tmp_path, false)
    } else {
        let path = args.dataset.clone().unwrap_or_else(|| {
            eprintln!("kbench longmemeval: --dataset is required for real runs.");
            std::process::exit(1);
        });
        (path, false)
    };

    let cfg = LmeConfig {
        dataset_path,
        limit: args.limit,
        question_types: args.question_types.clone(),
        dry_run,
        kimetsu_bin: args.kimetsu_binary.clone(),
        llm_backend,
        llm_model: args.llm_model.clone(),
        llm_api_key: args.llm_api_key.clone(),
        llm_base_url: args.llm_base_url.clone(),
        parallel: args.parallel,
    }
    .with_env_overlay();

    let report = match run_longmemeval(&cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("kbench longmemeval: {e}");
            std::process::exit(1);
        }
    };

    let body = match args.output {
        OutputFormat::Markdown => report.to_markdown(),
        OutputFormat::Json => report.to_json(),
    };
    println!("{body}");

    // Save report.
    let ext = match args.output {
        OutputFormat::Markdown => "md",
        OutputFormat::Json => "json",
    };
    let stamp = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "now".to_string())
        .replace(':', "-");
    let runs_dir = bench_dir.join("local").join("runs").join("longmemeval");
    let out_path = runs_dir.join(format!("{stamp}.{ext}"));
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&out_path, &body) {
        Ok(_) => eprintln!("kbench longmemeval: report saved -> {}", out_path.display()),
        Err(e) => eprintln!("kbench longmemeval: warn: could not save report: {e}"),
    }
}

fn run_beam_cmd(args: BeamArgs, bench_dir: &Path) {
    use drivers::beam::{BeamConfig, load_dataset, run_beam, synthetic_fixture};

    let llm_backend = LlmBackend::from_str(&args.reader_backend).unwrap_or_else(|| {
        eprintln!(
            "kbench beam: unknown --reader-backend `{}`; expected `codex`",
            args.reader_backend
        );
        std::process::exit(1);
    });

    // Source selection:
    //   --synthetic        → built-in fixture, REAL calls (end-to-end smoke).
    //   --dataset (+/-dry) → load the real dataset; --dry-run counts probes w/o calls.
    //   --dry-run alone    → built-in fixture, no calls.
    let (conversations, dataset_path, dry_run) = if args.synthetic {
        eprintln!(
            "beam: REAL-MODE synthetic fixture (backend={})",
            llm_backend.as_str()
        );
        (synthetic_fixture(), None, args.dry_run)
    } else if let Some(path) = args.dataset.clone() {
        let convs = load_dataset(&path).unwrap_or_else(|e| {
            eprintln!("kbench beam: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "beam: loaded {} conversation(s) from {}{}",
            convs.len(),
            path.display(),
            if args.dry_run {
                " (dry-run: counting probes, no calls)"
            } else {
                ""
            }
        );
        (convs, Some(path), args.dry_run)
    } else if args.dry_run {
        eprintln!(
            "beam: dry-run synthetic fixture (backend={})",
            llm_backend.as_str()
        );
        (synthetic_fixture(), None, true)
    } else {
        eprintln!(
            "kbench beam: --dataset is required for real runs (or use --synthetic/--dry-run)."
        );
        std::process::exit(1);
    };

    let cfg = BeamConfig {
        dataset_path,
        kimetsu_bin: args.kimetsu_binary.clone(),
        llm_backend,
        llm_model: args.llm_model.clone(),
        limit: args.limit,
        categories: args.categories.clone(),
        dry_run,
    };

    eprintln!("beam: {} conversation(s) to evaluate", conversations.len());
    let mut report = match run_beam(&cfg, conversations) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("kbench beam: {e}");
            std::process::exit(1);
        }
    };
    report.generated_at = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "now".to_string());

    let body = match args.output {
        OutputFormat::Markdown => report.to_markdown(),
        OutputFormat::Json => report.to_json(),
    };
    println!("{body}");

    let ext = match args.output {
        OutputFormat::Markdown => "md",
        OutputFormat::Json => "json",
    };
    let stamp = report.generated_at.replace(':', "-");
    let out_path = bench_dir
        .join("local")
        .join("runs")
        .join("beam")
        .join(format!("{stamp}.{ext}"));
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&out_path, &body).is_ok() {
        eprintln!("kbench beam: report saved -> {}", out_path.display());
    }
}

fn run_locomo_cmd(args: LocomoArgs, bench_dir: &Path) {
    use drivers::locomo::{LocomoConfig, render_markdown, run_locomo};

    let backend = match drivers::longmemeval::LlmBackend::from_str(&args.reader_backend) {
        Some(b) => b,
        None => {
            eprintln!(
                "kbench locomo: unknown --reader-backend `{}`; expected `codex` or `claude`",
                args.reader_backend
            );
            std::process::exit(1);
        }
    };
    let cfg = LocomoConfig {
        dataset_path: args.dataset.clone(),
        limit: args.limit,
        categories: args.categories.clone(),
        dry_run: args.dry_run,
        kimetsu_bin: args.kimetsu_binary.clone(),
        llm_backend: backend,
        llm_model: args.llm_model.clone(),
        parallel: args.parallel,
        iterations: args.iterations,
        learn: args.learn,
        workspace_root: args.workspace_root.clone(),
        self_cite: args.self_cite,
    };

    let report = match run_locomo(&cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("kbench locomo: {e}");
            std::process::exit(1);
        }
    };
    let body = render_markdown(&report, &args.dataset);
    println!("{body}");

    let stamp = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "now".to_string())
        .replace(':', "-");
    let out_path = bench_dir
        .join("local")
        .join("runs")
        .join("locomo")
        .join(format!("{stamp}.md"));
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&out_path, &body).is_ok() {
        eprintln!("kbench locomo: report saved -> {}", out_path.display());
    }
}

fn run_brainbench_cmd(args: BrainbenchArgs, bench_dir: &Path) {
    use drivers::brainbench::synthetic_fixture;

    // Resolve dataset path: --synthetic writes the built-in fixture to a temp
    // file; otherwise --dataset is required.
    let dataset_path = if args.synthetic {
        let fixture = synthetic_fixture();
        let json = serde_json::to_string(&fixture).expect("serialize synthetic fixture");
        let tmp_path = std::env::temp_dir().join("kbench-brainbench-synthetic.json");
        std::fs::write(&tmp_path, &json).unwrap_or_else(|e| {
            eprintln!("kbench brainbench: could not write synthetic fixture: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "brainbench: using built-in synthetic fixture ({})",
            tmp_path.display()
        );
        tmp_path
    } else {
        args.dataset.clone().unwrap_or_else(|| {
            eprintln!("kbench brainbench: --dataset is required (or pass --synthetic).");
            std::process::exit(1);
        })
    };

    // Parse tier / dimension filters via FromStr, with friendly errors.
    let tiers: Vec<Tier> = args
        .tiers
        .iter()
        .map(|s| {
            s.parse::<Tier>().unwrap_or_else(|e| {
                eprintln!("kbench brainbench: {e}");
                std::process::exit(1);
            })
        })
        .collect();
    let dimensions: Vec<Dimension> = args
        .dimensions
        .iter()
        .map(|s| {
            s.parse::<Dimension>().unwrap_or_else(|e| {
                eprintln!("kbench brainbench: {e}");
                std::process::exit(1);
            })
        })
        .collect();

    let cfg = BrainBenchConfig {
        dataset_path,
        kimetsu_bin: args.kimetsu_binary.clone(),
        budget_tokens: args.budget_tokens,
        tiers,
        dimensions,
        limit: args.limit,
        distill_provider: args.distill_provider.clone(),
        distill_model: args.distill_model.clone(),
    }
    .with_env_overlay();

    let report = match run_brainbench(&cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("kbench brainbench: {e}");
            std::process::exit(1);
        }
    };

    let body = match args.output {
        OutputFormat::Markdown => report.to_markdown(),
        OutputFormat::Json => report.to_json(),
    };
    println!("{body}");

    // Save report.
    let ext = match args.output {
        OutputFormat::Markdown => "md",
        OutputFormat::Json => "json",
    };
    let stamp = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "now".to_string())
        .replace(':', "-");
    let runs_dir = bench_dir.join("local").join("runs").join("brainbench");
    let out_path = runs_dir.join(format!("{stamp}.{ext}"));
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&out_path, &body) {
        Ok(_) => eprintln!("kbench brainbench: report saved -> {}", out_path.display()),
        Err(e) => eprintln!("kbench brainbench: warn: could not save report: {e}"),
    }
}

fn main() {
    let cli = Cli::parse();
    let (bench_dir, repo_root, cache_dir) = detect_paths();

    // Dispatch subcommands before Terminal-Bench logic.
    if let Some(sub) = cli.subcommand {
        match sub {
            KbenchCmd::Longmemeval(args) => {
                run_longmemeval_cmd(args, &bench_dir);
                return;
            }
            KbenchCmd::Brainbench(args) => {
                run_brainbench_cmd(args, &bench_dir);
                return;
            }
            KbenchCmd::Beam(args) => {
                run_beam_cmd(args, &bench_dir);
                return;
            }
            KbenchCmd::Locomo(args) => {
                run_locomo_cmd(args, &bench_dir);
                return;
            }
        }
    }

    if cli.list_families {
        let manifest = cli
            .families_manifest
            .clone()
            .unwrap_or_else(|| default_families_manifest(&bench_dir));
        print_families(&manifest);
        return;
    }

    let agents: Vec<AgentConfig> = match resolve_agents(&cli) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("kbench: {msg}");
            std::process::exit(2);
        }
    };
    let needs_kimetsu = agents.iter().any(|a| a.uses_kimetsu());

    let tasks_input = collect_tasks(&cli);
    let no_tasks = tasks_input.is_empty() && cli.full_dataset.is_none() && cli.family.is_none();
    if !cli.dry_run && no_tasks {
        print_no_task_error();
        std::process::exit(1);
    }

    // Auto-discover auth + binary + workspace BEFORE building the driver
    // so the discovered values flow into the driver context.
    let s = setup::discover(setup::SetupOptions {
        bench_dir: &bench_dir,
        repo_root: &repo_root,
        cache_dir: &cache_dir,
        dry_run: cli.dry_run,
        need_binary: needs_kimetsu,
        no_build: cli.no_build,
        brain_workspace_override: cli.brain_workspace.clone(),
        kimetsu_binary_override: cli.kimetsu_binary.clone(),
        quiet: cli.worker_result.is_some(),
    });

    // Capture the brain workspace for post-trial ingest (the driver takes
    // ownership of the path when we hand it Setup).
    let brain_workspace_for_ingest = s.brain_workspace.clone();

    // Final harbor arg list: auto-discovered first, then explicit overrides
    // so the user's `--harbor-arg` always wins on duplicate keys.
    let mut all_harbor_args = s.harbor_args.clone();
    all_harbor_args.extend(cli.harbor_args.clone());

    // Shared run directory for this whole invocation. The orchestrator
    // computes it once (bench/runs/<run-ts>) and hands the SAME absolute path
    // to every worker via --run-dir, so all (task × agent) trials land under
    // one run dir on the bench drive (E:). Without this, the driver defaults
    // to a relative `./runs/<stamp>` resolved against the worker's /tmp cwd,
    // which lives on the WSL ext4 vhdx (C:) — see `run_trial_isolated`.
    let run_root = cli.run_dir.clone().unwrap_or_else(|| {
        let stamp = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "now".to_string())
            .replace(':', "-");
        bench_dir.join("local").join("runs").join(stamp)
    });
    let ctx = DriverContext {
        work_dir: Some(run_root.clone()),
        ..DriverContext::default()
    };
    let mut driver: Box<dyn BenchmarkDriver> = Box::new(TerminalBenchDriver::with_options(
        ctx,
        cli.dry_run,
        cli.model.clone(),
        all_harbor_args,
        s.kimetsu_binary.clone(),
        Some(s.brain_workspace.clone()),
        s.kimetsu_nudge_path.clone(),
    ));

    // Resolve tasks. Priority: explicit list > --full-dataset > driver.list_tasks().
    let tasks: Vec<TaskId> = if !tasks_input.is_empty() {
        tasks_input.iter().map(|s| TaskId(s.clone())).collect()
    } else if let Some(family) = cli.family.as_deref() {
        let manifest = cli
            .families_manifest
            .clone()
            .unwrap_or_else(|| default_families_manifest(&bench_dir));
        match resolve_family_tasks(&manifest, family) {
            Ok(v) if v.is_empty() => {
                eprintln!("kbench: --family {family} has no tasks in manifest.");
                std::process::exit(1);
            }
            Ok(v) => v,
            Err(e) => {
                eprintln!("kbench: --family {family}: {e}");
                std::process::exit(1);
            }
        }
    } else if let Some(dataset_input) = cli.full_dataset.as_ref() {
        let dataset_path = resolve_dataset_path(dataset_input);
        match parse_tasks_from_dataset_path(&dataset_path) {
            Ok(v) if v.is_empty() => {
                eprintln!(
                    "kbench: --full-dataset {} matched 0 tasks. \
                     Confirm the dataset is downloaded (`harbor dataset download terminal-bench/terminal-bench-2`).",
                    dataset_path.display()
                );
                std::process::exit(1);
            }
            Ok(v) => v,
            Err(e) => {
                eprintln!("kbench: --full-dataset failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match driver.list_tasks() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("kbench: list_tasks failed: {e}");
                std::process::exit(1);
            }
        }
    };

    // Single-trial worker mode: run exactly one (task, agent), write
    // {run, grade} as JSON, and exit. No aggregate report. The
    // orchestrator re-execs us like this so each real Harbor invocation
    // gets a fresh process (see `run_trial_isolated`).
    if let Some(result_path) = cli.worker_result.clone() {
        let task = tasks[0].clone();
        let agent = agents[0];
        match run_and_grade_trial(
            driver.as_mut(),
            &task,
            agent,
            cli.dry_run,
            &brain_workspace_for_ingest,
        ) {
            Ok((run, grade)) => {
                let payload = serde_json::json!({ "run": run, "grade": grade });
                let write = serde_json::to_string(&payload)
                    .map_err(|e| e.to_string())
                    .and_then(|s| std::fs::write(&result_path, s).map_err(|e| e.to_string()));
                if let Err(e) = write {
                    eprintln!("    worker: could not write result: {e}");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("    {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    eprintln!(
        "kbench: {} task(s) × {} agent(s) = {} run(s)",
        tasks.len(),
        agents.len(),
        tasks.len() * agents.len()
    );

    // Real runs isolate each trial in its own subprocess so Harbor's
    // per-invocation cwd staleness on WSL2 can't cascade. Dry-runs stay
    // in-process (no Harbor, and CI exercises this path directly).
    let isolate = !cli.dry_run;

    // Move the orchestrator's own cwd off the DrvFs bench dir onto a stable
    // real-fs dir before spawning workers. After the 1st worker's Docker
    // churn the inherited DrvFs cwd handle goes stale and posix_spawn of the
    // 2nd worker fails with ENOENT. Every path kbench uses is absolute
    // (CARGO_MANIFEST_DIR-rooted), so the orchestrator's cwd is irrelevant.
    if isolate {
        let _ = std::env::set_current_dir(std::env::temp_dir());
    }

    let mut runs_and_grades = Vec::new();
    let mut error_count = 0usize;
    for task in &tasks {
        for agent in &agents {
            eprintln!("  -> {} | {} ...", task, agent.label());
            let outcome = if isolate {
                run_trial_isolated(&cli, &s, task, *agent, &cache_dir, &run_root)
            } else {
                run_and_grade_trial(
                    driver.as_mut(),
                    task,
                    *agent,
                    cli.dry_run,
                    &brain_workspace_for_ingest,
                )
            };
            match outcome {
                Ok((run, grade)) => runs_and_grades.push((run, grade)),
                Err(e) => {
                    eprintln!("    {e}");
                    error_count += 1;
                }
            }
        }
    }

    if runs_and_grades.is_empty() {
        eprintln!(
            "kbench: no graded runs to report ({error_count} errors). \
             Try --dry-run if Harbor isn't installed."
        );
        std::process::exit(1);
    }

    let report = Report::build(driver.name(), runs_and_grades);
    let body = match cli.output {
        OutputFormat::Markdown => report.to_markdown(),
        OutputFormat::Json => report.to_json(),
    };

    println!("{body}");

    let runs_dir = bench_dir.join("local").join("runs").join("auto");
    if let Some(p) = save_report(&runs_dir, &body, cli.output) {
        eprintln!("kbench: report saved -> {}", p.display());
    }

    if error_count > 0 {
        eprintln!("kbench: completed with {error_count} per-run error(s).");
    }
}
