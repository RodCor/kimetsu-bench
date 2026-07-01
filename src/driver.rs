//! The `BenchmarkDriver` trait — the plug-in surface for swapping
//! benchmark backends without rewriting the orchestrator.
//!
//! Adding a new backend (SWE-bench, custom internal corpus, etc.) means
//! impl-ing these four methods. The `kbench` binary doesn't know about
//! Terminal-Bench specifically; it only knows the trait.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Stable identifier for a benchmark task. Backend-specific — for
/// Terminal-Bench it's the task slug (`"hello-world"`, `"git-bisect"`),
/// for a future SWE-bench driver it would be `"pylint:foo-1234"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for TaskId {
    fn from(s: &str) -> Self {
        TaskId(s.to_string())
    }
}

/// Which agent stack runs each task. The bench compares these in
/// pairs (or trios) to measure kimetsu's impact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AgentConfig {
    /// Claude Code (host) with kimetsu's MCP server attached.
    ClaudePlusKimetsu,
    /// Claude Code (host) with no kimetsu attached — baseline.
    ClaudeAlone,
    /// Codex (host) with kimetsu's MCP server attached.
    CodexPlusKimetsu,
    /// Codex (host) with no kimetsu attached — baseline.
    CodexAlone,
}

impl AgentConfig {
    pub fn label(&self) -> &'static str {
        match self {
            AgentConfig::ClaudePlusKimetsu => "claude+km",
            AgentConfig::ClaudeAlone => "claude",
            AgentConfig::CodexPlusKimetsu => "codex+km",
            AgentConfig::CodexAlone => "codex",
        }
    }

    /// True when this config should attach the kimetsu MCP server to
    /// the host. The driver uses this to decide whether to set
    /// `MCP_KIMETSU_SOCKET` / spawn the sidecar / etc.
    pub fn uses_kimetsu(&self) -> bool {
        matches!(
            self,
            AgentConfig::ClaudePlusKimetsu | AgentConfig::CodexPlusKimetsu
        )
    }

    /// "claude" or "codex" — the host harness label. Used by the
    /// real Terminal-Bench driver (not yet wired) to pick the
    /// harbor `--agent-import-path`. Marked allow until then.
    #[allow(dead_code)]
    pub fn host(&self) -> &'static str {
        match self {
            AgentConfig::ClaudePlusKimetsu | AgentConfig::ClaudeAlone => "claude",
            AgentConfig::CodexPlusKimetsu | AgentConfig::CodexAlone => "codex",
        }
    }
}

/// What the driver got back from running a single task with a single
/// agent config. Drivers populate this; `grade` consumes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub task: TaskId,
    pub agent: AgentConfig,
    /// Raw exit status — 0 means the agent run completed (graded
    /// separately for pass/fail); non-zero means the run *itself*
    /// errored (model timeout, docker died, network).
    pub exit_code: i32,
    /// Wallclock seconds.
    pub duration_secs: f64,
    /// Total cost surfaced by the host agent (USD). 0 when the host
    /// doesn't expose cost (Codex / Claude Code OAuth subscription).
    pub cost_usd: f32,
    /// Free-form notes — driver-specific telemetry, error messages,
    /// log paths.
    pub notes: String,
}

/// Pass/fail + score for a graded RunResult. Numeric so multi-step
/// tasks (Terminal-Bench partial credit, SWE-bench resolved-fraction)
/// can express degree of success.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grade {
    pub task: TaskId,
    pub agent: AgentConfig,
    /// 1.0 = full pass; 0.0 = full fail; (0.0, 1.0) = partial credit
    /// for backends that support it.
    pub score: f32,
    /// Driver-specific reason string (for failed-runs surfacing).
    pub reason: Option<String>,
}

/// The plug-in surface. Anything that can list a corpus, run a task,
/// and grade the result is a `BenchmarkDriver`.
pub trait BenchmarkDriver {
    /// Human-readable identifier shown in reports + on the CLI
    /// (`--driver tb`, `--driver swebench`, etc.).
    fn name(&self) -> &str;

    /// Enumerate all tasks the backend knows about. The orchestrator
    /// filters / slices this client-side based on `--tasks <list>`.
    fn list_tasks(&self) -> Result<Vec<TaskId>, DriverError>;

    /// Run a single task with the given agent config. Drivers are
    /// responsible for: spinning up the sandbox, invoking the host
    /// agent, attaching kimetsu via MCP when `agent.uses_kimetsu()`,
    /// capturing exit code + duration + cost.
    fn run(&mut self, task: &TaskId, agent: AgentConfig) -> Result<RunResult, DriverError>;

    /// Grade a completed run. Read-only — for most backends this
    /// inspects a verdict file written by the sandbox.
    fn grade(&self, run: &RunResult) -> Result<Grade, DriverError>;
}

/// Errors a driver can surface. We don't blow up the whole run for a
/// single-task failure — the orchestrator catches the error, records
/// it in the report, and moves on to the next task.
///
/// The dry-run stub only emits `InfraUnavailable`; the other variants
/// land when the real Terminal-Bench impl wires them up.
#[derive(Debug)]
#[allow(dead_code)]
pub enum DriverError {
    /// Backend infrastructure unavailable (Docker not running, harbor
    /// CLI not installed, etc.). User-actionable.
    InfraUnavailable(String),
    /// The task itself errored out (process crashed, sandbox lost).
    RunFailed(String),
    /// The grade file was missing or malformed.
    GradingFailed(String),
    /// Anything else.
    Other(String),
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriverError::InfraUnavailable(msg) => write!(f, "infra unavailable: {msg}"),
            DriverError::RunFailed(msg) => write!(f, "run failed: {msg}"),
            DriverError::GradingFailed(msg) => write!(f, "grading failed: {msg}"),
            DriverError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for DriverError {}

/// Driver-construction context. Holds host config a driver might need
/// (paths to harbor CLI, env vars, etc.) without polluting the trait.
///
/// Fields land here as concrete drivers need them. The v0.5.5 dry-run
/// stub doesn't consume any of these yet — `#[allow(dead_code)]`
/// silences the dead-field warning until the real Terminal-Bench
/// driver wires them in.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct DriverContext {
    /// Path to a working dir for sandboxes / log files. Defaults to
    /// `./runs/<timestamp>/`.
    pub work_dir: Option<PathBuf>,
    /// Free-form key-value overrides (e.g. `harbor_bin = /usr/local/bin/harbor`).
    pub overrides: BTreeMap<String, String>,
}
