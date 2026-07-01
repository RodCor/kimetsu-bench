//! Terminal-Bench driver — Layer 2 v1.
//!
//! Wraps the Harbor CLI to drive Terminal-Bench tasks under different
//! agent configurations. Uses Harbor's built-in `claude-code` / `codex`
//! host agents; for `+km` configs, attaches kimetsu's MCP server via
//! `--mcp-config` rather than running a custom host agent.
//!
//! ## How it talks to Harbor
//!
//! Harbor is a Python CLI installed via `pip install harbor`. We
//! shell out to it once per task using `std::process::Command`:
//!
//!   `harbor list-tasks --dataset <dataset> --json`
//!       — enumerate the corpus. We parse a JSON array of objects
//!       with a `"task_id"` field.
//!
//!   `harbor run --agent-import-path <path> --task <id>
//!       --output-dir <dir> --env <docker|daytona|...> [--env-file <f>]`
//!       — run one task. Harbor writes per-task output to
//!       `<output-dir>/<task-id>/`, including `verdict.json` which
//!       the grading step reads.
//!
//! The exact CLI surface here reflects Harbor 0.6.x; if Harbor
//! moves the verdict file or renames flags, the constants at the
//! top of this module + the JSON keys in `parse_verdict` are the
//! one place to change.
//!
//! ## Agent argument shapes
//!
//! All 4 AgentConfigs use Harbor's built-in host agents — the
//! difference between `+km` and baseline is whether we also
//! attach kimetsu's MCP server via `--mcp-config`:
//!
//!   ClaudePlusKimetsu  → `--agent claude-code --mcp-config <path>`
//!                        where <path> is a per-run
//!                        `kimetsu.mcp.json` we write describing
//!                        how Harbor should spawn
//!                        `kimetsu mcp serve --workspace <dir>`.
//!   ClaudeAlone        → `--agent claude-code` (no MCP config).
//!   CodexPlusKimetsu   → `--agent codex --mcp-config <path>`
//!                        with the same kimetsu.mcp.json shape.
//!                        Codex CLI handles its own auth via the
//!                        local login state, so no env var is
//!                        required from us.
//!   CodexAlone         → `--agent codex` (no MCP config).
//!
//! The MCP-attachment path measures what users actually do — install
//! kimetsu, attach it to their host harness, run their normal workflow.
//! (The pre-0.5 driver routed +km configs through a Python shim that
//! always spawned a Claude-based binary, so `codex+km` measured Claude.
//! That shim has been deleted.)
//!
//! Override the built-in agent names via:
//!   * `DriverContext.overrides["claude_alone_agent_name"]`
//!   * `DriverContext.overrides["codex_alone_agent_name"]`
//!   * `DriverContext.overrides["kimetsu_bin"]`           (path to `kimetsu`)
//!   * `DriverContext.overrides["kimetsu_mcp_workspace"]` (per-run brain workspace)
//!
//! When Harbor isn't installed or the listed tasks aren't reachable,
//! `--dry-run` keeps the orchestrator + report formatter
//! exercisable without any external dependencies. That's also what
//! CI uses to validate the orchestrator wiring without paying for
//! API calls.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use serde::Deserialize;

use crate::driver::{
    AgentConfig, BenchmarkDriver, DriverContext, DriverError, Grade, RunResult, TaskId,
};

/// Default Harbor binary name. Overridden via
/// `DriverContext.overrides["harbor_bin"]` when the user has a
/// non-standard install location.
const DEFAULT_HARBOR_BIN: &str = "harbor";

/// Default Terminal-Bench dataset slice. Override via
/// `DriverContext.overrides["tb_dataset"]`. Matches the slice the
/// v0.4 MP-* benchmarks were calibrated against.
const DEFAULT_DATASET: &str = "terminal-bench/terminal-bench-2";

/// Default sandbox environment. Override via
/// `DriverContext.overrides["tb_env"]`. `docker` works everywhere
/// with Docker installed; cloud sandboxes (`daytona`, `e2b`,
/// `modal`, `runloop`) work with the matching API keys.
const DEFAULT_ENV: &str = "docker";

/// Nudge content appended to the task instruction for `+km` agents.
/// Kimetsu's MCP server returns rich `instructions` in its initialize
/// response, but Claude Code's `--print` mode drops them. Without
/// this nudge, Claude sees 21 kimetsu_* tools as decoration and never
/// calls them (verified empirically: 12-task gauntlet 2026-05-25 → 0
/// invocations across all +km trials).
///
/// We write this to a file at startup and pass it via Harbor's
/// `--extra-instruction-path` (which Harbor properly handles, unlike
/// `--agent-kwarg append_system_prompt=...` which it fails to
/// shell-quote in the inner bash command).
///
/// We point Claude at `kimetsu_benchmark_context` rather than the
/// generic `kimetsu_brain_context` because the benchmark variant
/// biases retrieval toward `semantic_operator` and `anti_pattern`
/// memory roles over raw `episodic` run summaries — which accumulate
/// fast in gauntlets and would otherwise crowd retrieval. TODO: once
/// kimetsu's generic brain_context can apply the same role-bias on
/// demand (see TODO in kimetsu-chat/src/mcp_server.rs near
/// BENCHMARK_CONTEXT_DESCRIPTION), switch this nudge to
/// `kimetsu_brain_context` so the bench stops depending on
/// bench-aware MCP surface.
pub const KIMETSU_AGENT_NUDGE: &str = "\n\n---\n\n\
# Kimetsu brain (MCP) — transfer learning across tasks\n\n\
You have the Kimetsu brain attached as an MCP server (tools prefixed \
`mcp__kimetsu__`). Use it as follows:\n\n\
1. **BEFORE planning**, call `mcp__kimetsu__kimetsu_benchmark_context` with a concise \
description of the current task. It returns prior lessons, anti-patterns, and outcome \
signals from related tasks. Apply any relevant findings to your plan.\n\n\
2. **AFTER finishing** (success OR failure), call \
`mcp__kimetsu__kimetsu_benchmark_record_outcome` to log the attempt. If you spotted a \
transferable lesson (a useful command, a gotcha, an anti-pattern), include it in \
`generalized_memory` with `memory_role=\"semantic_operator\"` or `anti_pattern` so \
future tasks benefit.\n\n\
The brain is empty on the very first task; capsules accumulate as you finish each one. \
Don't skip these steps — they're how kimetsu earns its keep over the gauntlet.\n";

pub struct TerminalBenchDriver {
    ctx: DriverContext,
    /// True when running in --dry-run mode: list/run/grade all return
    /// synthetic data instead of touching Harbor. Used by the bench
    /// CLI's `--dry-run` flag to exercise the orchestration path
    /// without an installed Harbor.
    dry_run: bool,
    /// Per-driver-construction work dir. Each `run()` invocation
    /// uses a subdirectory under this so per-task artifacts don't
    /// collide.
    work_dir: PathBuf,
    /// Optional `--model <name>` forwarded to `harbor run`. Required
    /// by Harbor's codex agent (errors "Model name is required"
    /// without it).
    model: Option<String>,
    /// Extra args appended verbatim to every `harbor run` invocation.
    /// Source: kbench's `--harbor-arg` flag (repeatable).
    extra_args: Vec<String>,
    /// Host-side path to a Linux `kimetsu` binary. When set, every
    /// `+km` run bind-mounts it at `/usr/local/bin/kimetsu` inside
    /// the container so the MCP config can spawn it. When unset,
    /// `+km` runs degrade silently — the host agent can't find
    /// `kimetsu` in the container's PATH and runs as if no MCP was
    /// attached.
    kimetsu_binary: Option<PathBuf>,
    /// Host-side path to a kimetsu workspace (with `.kimetsu/brain.db`).
    /// Bind-mounted at `/kimetsu-workspace` inside the container.
    /// When unset, the per-task output_dir is used as a fresh
    /// per-run workspace.
    brain_workspace: Option<PathBuf>,
    /// Host-side path to a markdown file containing the kimetsu workflow
    /// nudge. Passed to Harbor as `--extra-instruction-path` for +km
    /// agents so Claude sees the instruction inside its prompt. When
    /// `None`, kimetsu tools are attached but Claude isn't told to use
    /// them (and historically never does).
    kimetsu_nudge_path: Option<PathBuf>,
}

impl TerminalBenchDriver {
    /// Construct with default options (no --model, no extra args, no
    /// kimetsu mounts). Kept for backwards compat; new callers should
    /// prefer `with_options` so the passthroughs are explicit at the
    /// call site.
    #[allow(dead_code)]
    pub fn new(ctx: DriverContext, dry_run: bool) -> Self {
        Self::with_options(ctx, dry_run, None, Vec::new(), None, None, None)
    }

    /// Full constructor with model + extra-args + kimetsu mount paths.
    pub fn with_options(
        ctx: DriverContext,
        dry_run: bool,
        model: Option<String>,
        extra_args: Vec<String>,
        kimetsu_binary: Option<PathBuf>,
        brain_workspace: Option<PathBuf>,
        kimetsu_nudge_path: Option<PathBuf>,
    ) -> Self {
        // Resolve work dir: caller-supplied wins; otherwise
        // ./runs/<rfc3339-ish>/ so concurrent invocations don't
        // step on each other.
        let work_dir = ctx.work_dir.clone().unwrap_or_else(|| {
            let stamp = time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "now".to_string())
                .replace(':', "-");
            PathBuf::from("./local/runs").join(stamp)
        });
        Self {
            ctx,
            dry_run,
            work_dir,
            model,
            extra_args,
            kimetsu_binary,
            brain_workspace,
            kimetsu_nudge_path,
        }
    }

    /// Tasks the dry-run path surfaces. Real `list_tasks` returns
    /// whatever Harbor enumerates.
    fn dry_run_corpus() -> Vec<TaskId> {
        vec![
            "hello-world".into(),
            "git-bisect".into(),
            "fix-import".into(),
            "rename-symbol".into(),
        ]
    }

    /// Resolve the Harbor binary path. Order: explicit override,
    /// then PATH lookup of `harbor`.
    fn harbor_bin(&self) -> String {
        self.ctx
            .overrides
            .get("harbor_bin")
            .cloned()
            .unwrap_or_else(|| DEFAULT_HARBOR_BIN.to_string())
    }

    /// Resolve the Terminal-Bench dataset slug.
    fn dataset(&self) -> String {
        self.ctx
            .overrides
            .get("tb_dataset")
            .cloned()
            .unwrap_or_else(|| DEFAULT_DATASET.to_string())
    }

    /// Resolve the sandbox environment flag.
    fn env_flag(&self) -> String {
        self.ctx
            .overrides
            .get("tb_env")
            .cloned()
            .unwrap_or_else(|| DEFAULT_ENV.to_string())
    }

    /// Pick the Harbor `--agent <name>` for an AgentConfig.
    ///
    /// All 4 configs use Harbor's *built-in* host agents
    /// (`claude-code`, `codex`). The difference between
    /// `claude+km` and `claude` is whether we also pass
    /// `--mcp-config <kimetsu.mcp.json>` to attach kimetsu's
    /// MCP server.
    fn host_agent_name(&self, agent: AgentConfig) -> String {
        match agent {
            AgentConfig::ClaudePlusKimetsu | AgentConfig::ClaudeAlone => self
                .ctx
                .overrides
                .get("claude_alone_agent_name")
                .cloned()
                .unwrap_or_else(|| "claude-code".to_string()),
            AgentConfig::CodexPlusKimetsu | AgentConfig::CodexAlone => self
                .ctx
                .overrides
                .get("codex_alone_agent_name")
                .cloned()
                .unwrap_or_else(|| "codex".to_string()),
        }
    }

    /// Write a per-run `kimetsu.mcp.json` with CONTAINER-side
    /// paths. The host agent (Claude Code / Codex) runs inside the
    /// Terminal-Bench Docker container and spawns the MCP server
    /// from this config — `command` + the workspace path must
    /// resolve inside the container, not on the host.
    ///
    /// We use:
    ///   command = `/usr/local/bin/kimetsu`  (driver bind-mounts the
    ///                                        host binary here; see
    ///                                        `kimetsu_mounts_json`)
    ///   args[3] = `/kimetsu-workspace`      (also bind-mounted from
    ///                                        host)
    ///
    /// The file itself lives on the host (next to the per-task
    /// output dir) and is referenced by `--mcp-config <host-path>`.
    /// Harbor reads it on the host then propagates the spawn-command
    /// into the container.
    fn write_mcp_config(&self, output_dir: &std::path::Path) -> Result<PathBuf, DriverError> {
        let config = serde_json::json!({
            "mcpServers": {
                "kimetsu": {
                    "command": "/usr/local/bin/kimetsu",
                    "args": ["mcp", "serve", "--workspace", "/kimetsu-workspace"],
                }
            }
        });
        let path = output_dir.join("kimetsu.mcp.json");
        let body = serde_json::to_string_pretty(&config)
            .map_err(|e| DriverError::Other(format!("serialize mcp config: {e}")))?;
        std::fs::write(&path, body).map_err(|e| {
            DriverError::Other(format!(
                "could not write mcp config to {}: {e}",
                path.display()
            ))
        })?;
        Ok(path)
    }

    /// Build the JSON value for Harbor's `--mounts` flag covering
    /// the kimetsu bind-mounts (binary + workspace).
    ///
    /// Returns `None` when no Linux kimetsu binary path is
    /// configured — caller skips passing `--mounts` entirely so
    /// the +km run degrades silently (host agent can't find
    /// `/usr/local/bin/kimetsu` and runs without MCP). A warning
    /// is printed by the caller so the operator knows the
    /// comparison won't be meaningful.
    fn kimetsu_mounts_json(
        &self,
        per_run_workspace: &std::path::Path,
    ) -> Result<Option<String>, DriverError> {
        let Some(binary) = self.kimetsu_binary.as_deref() else {
            return Ok(None);
        };
        if !binary.is_file() {
            return Err(DriverError::Other(format!(
                "--kimetsu-binary points at {} which is not a regular file",
                binary.display()
            )));
        }
        let workspace = match self.brain_workspace.as_deref() {
            Some(p) => {
                if !p.is_dir() {
                    return Err(DriverError::Other(format!(
                        "--brain-workspace points at {} which is not a directory",
                        p.display()
                    )));
                }
                p.to_path_buf()
            }
            None => per_run_workspace.to_path_buf(),
        };

        // Docker rejects relative bind-mount sources. Always
        // canonicalize to absolute before normalizing for the
        // mount JSON.
        let binary_abs = std::fs::canonicalize(binary).map_err(|e| {
            DriverError::Other(format!(
                "could not canonicalize --kimetsu-binary {}: {e}",
                binary.display()
            ))
        })?;
        let workspace_abs = std::fs::canonicalize(&workspace).map_err(|e| {
            DriverError::Other(format!(
                "could not canonicalize brain workspace {}: {e}",
                workspace.display()
            ))
        })?;
        let bin_src = normalize_for_mount(&binary_abs);
        let ws_src = normalize_for_mount(&workspace_abs);

        // Note: we DON'T include `read_only: true` on the binary
        // mount even though semantically it should be read-only.
        // Harbor 0.7.1's mount schema only accepts the three core
        // docker-compose long-form keys (source/target/type) and
        // silently drops the whole mount entry when extras appear.
        // The container has no reason to modify the binary anyway,
        // and read-only at the OS level is the user's responsibility.
        let mounts = serde_json::json!([
            {
                "source": bin_src,
                "target": "/usr/local/bin/kimetsu",
                "type": "bind",
            },
            {
                "source": ws_src,
                "target": "/kimetsu-workspace",
                "type": "bind",
            }
        ]);
        Ok(Some(serde_json::to_string(&mounts).map_err(|e| {
            DriverError::Other(format!("serialize kimetsu mounts: {e}"))
        })?))
    }

    /// Per-task output dir under `work_dir`. Includes the agent
    /// label so the same task running under multiple configs writes
    /// to distinct directories.
    fn task_output_dir(&self, task: &TaskId, agent: AgentConfig) -> PathBuf {
        self.work_dir.join(format!("{}-{}", task.0, agent.label()))
    }
}

impl BenchmarkDriver for TerminalBenchDriver {
    fn name(&self) -> &str {
        "terminal-bench"
    }

    fn list_tasks(&self) -> Result<Vec<TaskId>, DriverError> {
        if self.dry_run {
            return Ok(Self::dry_run_corpus());
        }
        // Harbor 0.7.1 does NOT expose a top-level `list-tasks`
        // subcommand. Tasks live inside datasets, which are either
        // downloaded locally (`harbor dataset download`) or fetched
        // from a registry on demand by `harbor run`. For now this
        // driver requires the operator to pass `--tasks <list>`
        // explicitly to kbench. Future work: shell out to
        // `harbor dataset download` + walk the downloaded directory
        // structure to enumerate tasks.
        Err(DriverError::InfraUnavailable(
            "Harbor 0.7.1 has no top-level `list-tasks` command. \
             Pass --tasks <id,id,...> to kbench, or run with --dry-run \
             to exercise the orchestrator. (Auto-enumeration via \
             `harbor dataset download` + directory walk is a future \
             improvement.)"
                .to_string(),
        ))
    }

    fn run(&mut self, task: &TaskId, agent: AgentConfig) -> Result<RunResult, DriverError> {
        if self.dry_run {
            return Ok(synth_dry_run_result(task, agent));
        }

        let harbor = self.harbor_bin();
        let dataset = self.dataset();
        let env_flag = self.env_flag();
        let agent_name = self.host_agent_name(agent);
        let output_dir = self.task_output_dir(task, agent);

        // Make sure the output dir exists; Harbor writes per-task
        // artifacts here.
        fs::create_dir_all(&output_dir).map_err(|e| {
            DriverError::RunFailed(format!(
                "could not create output dir {}: {e}",
                output_dir.display()
            ))
        })?;

        // For kimetsu-attached configs we write a per-run
        // mcp.json describing how Harbor should spawn the kimetsu
        // MCP server, AND build the docker --mounts JSON for the
        // kimetsu binary + workspace bind-mounts. For baselines,
        // both stay None.
        let (mcp_config_path, kimetsu_mounts) = if agent.uses_kimetsu() {
            let mcp = self.write_mcp_config(&output_dir)?;
            let mounts = self.kimetsu_mounts_json(&output_dir)?;
            if mounts.is_none() {
                eprintln!(
                    "kbench: warning: agent {} requested but --kimetsu-binary not set. \
                     The container won't have a `kimetsu` binary; the MCP attach will fail \
                     silently and the host agent will run as if no MCP was attached. \
                     Pass --kimetsu-binary <path-to-linux-kimetsu> for a meaningful run.",
                    agent.label()
                );
            }
            (Some(mcp), mounts)
        } else {
            (None, None)
        };

        let start = Instant::now();
        let mut cmd = Command::new(&harbor);
        // Harbor 0.7.1 flag surface:
        //   --dataset DATASET@VERSION       -d
        //   --include-task-name PATTERN     -i  (single-task glob)
        //   --agent AGENT                    -a  (built-in host name)
        //   --mcp-config PATH                    (attach an MCP server)
        //   --jobs-dir DIR                   -o
        //   --env ENV                        -e
        //   --yes                                (auto-confirm prompts)
        //   --quiet                          -q
        // Harbor stores task ids as `<registry>/<task-name>` (e.g.
        // `terminal-bench/adaptive-rejection-sampler`) and matches
        // `--include-task-name` against the full name with glob
        // semantics. If the operator passed a short, unqualified id
        // (`adaptive-rejection-sampler`), wrap it as `*/<name>` so
        // Harbor matches it across whatever registry the dataset
        // belongs to. Full names pass through unchanged.
        let include_pattern = if task.0.contains('/') {
            task.0.clone()
        } else {
            format!("*/{}", task.0)
        };
        cmd.args([
            "run",
            "--dataset",
            &dataset,
            "--include-task-name",
            &include_pattern,
            "--agent",
            &agent_name,
            "--jobs-dir",
            &output_dir.to_string_lossy(),
            "--env",
            &env_flag,
            "--yes",
            "--quiet",
        ]);
        // On Windows, Harbor's auto-generated docker-compose volume
        // mounts contain Windows paths like `E:/Kimetsu/...`. Docker
        // Compose's short-form parser splits on `:`, sees the drive
        // colon, and rejects the spec as malformed. Setting
        // `COMPOSE_CONVERT_WINDOWS_PATHS=1` makes Compose rewrite
        // `E:/...` to `/e/...` before parsing, which the WSL2-backed
        // engine accepts. No-op on Linux/macOS where the value is
        // ignored.
        cmd.env("COMPOSE_CONVERT_WINDOWS_PATHS", "1");
        if let Some(ref mcp_path) = mcp_config_path {
            cmd.args(["--mcp-config", &mcp_path.to_string_lossy()]);
        }
        // Harbor only honors ONE --mounts flag per invocation (later wins).
        // The driver may need to attach 2 mounts (kimetsu binary + workspace
        // for +km configs) AND the caller may pass additional mounts via
        // --harbor-arg --mounts=[...] (typically ~/.codex for codex auth).
        // Merge them into a single --mounts=[...] so nothing gets dropped.
        let merged_mounts = merge_mounts(kimetsu_mounts.as_deref(), &self.extra_args);
        if let Some(merged) = merged_mounts {
            cmd.arg(format!("--mounts={merged}"));
        }
        if let Some(ref model) = self.model {
            cmd.args(["--model", model.as_str()]);
        }
        // For +km configs, append the kimetsu workflow nudge to the task
        // instruction. The MCP server itself returns rich `instructions` in
        // its initialize response, but Claude Code's --print mode silently
        // drops that field. Without this nudge, Claude sees 21 kimetsu_*
        // tools as unfamiliar decoration and never calls them.
        //
        // We use Harbor's `--extra-instruction-path PATH` rather than the
        // `--agent-kwarg append_system_prompt=<text>` route because Harbor
        // 0.8 fails to shell-quote the kwarg value when assembling the
        // inner bash command — the nudge text breaks on its first space
        // and claude errors with `Got unexpected extra arguments`.
        // Passing a file path sidesteps the quoting bug entirely.
        if agent.uses_kimetsu()
            && let Some(nudge) = self.kimetsu_nudge_path.as_deref()
        {
            cmd.arg("--extra-instruction-path");
            cmd.arg(nudge);
        }
        // Verbatim passthrough from kbench --harbor-arg (repeatable), MINUS
        // any --mounts= entries already merged above.
        for arg in &self.extra_args {
            if arg.starts_with("--mounts=") {
                continue;
            }
            cmd.arg(arg);
        }

        let result = cmd.output().map_err(|e| {
            DriverError::InfraUnavailable(format!(
                "could not spawn `{harbor} run`: {e}. \
                 Install Harbor (`pip install harbor`) or set \
                 `harbor_bin` override; run with --dry-run to \
                 exercise the orchestrator instead."
            ))
        })?;

        let duration_secs = start.elapsed().as_secs_f64();
        let exit_code = result.status.code().unwrap_or(-1);

        // Harbor may print a JSON line like `{"cost_usd": 0.42}` to
        // stdout on success. We grep for it; missing = 0.
        let cost_usd = parse_cost_from_stdout(&result.stdout).unwrap_or(0.0);

        // Free-form notes: capture stderr tail + the output dir for
        // post-mortem inspection.
        let stderr_tail = tail_lossy(&result.stderr, 1024);
        let notes = format!(
            "exit={exit_code} output_dir={} stderr_tail={stderr_tail}",
            output_dir.display()
        );

        Ok(RunResult {
            task: task.clone(),
            agent,
            exit_code,
            duration_secs,
            cost_usd,
            notes,
        })
    }

    fn grade(&self, run: &RunResult) -> Result<Grade, DriverError> {
        if self.dry_run {
            let score = if run.exit_code == 0 { 1.0 } else { 0.0 };
            return Ok(Grade {
                task: run.task.clone(),
                agent: run.agent,
                score,
                reason: if score < 1.0 {
                    Some("[dry-run] simulated failure".to_string())
                } else {
                    None
                },
            });
        }

        // Harbor 0.7.1's actual output layout under `--jobs-dir`:
        //   <jobs-dir>/<our-task-agent-subdir>/<harbor-timestamp>/result.json
        //
        // Harbor creates a timestamped subdir per run; the result lives
        // inside it. We find the single child directory and read its
        // result.json.
        let output_dir = self.task_output_dir(&run.task, run.agent);
        let result_path = match find_harbor_result_path(&output_dir) {
            Ok(p) => p,
            Err(e) => {
                let underlying = if run.exit_code != 0 {
                    format!("Harbor exited {}; {}", run.exit_code, run.notes)
                } else {
                    format!(
                        "Harbor exit 0 but no result.json found. {e}. \
                         Inspect {} manually.",
                        output_dir.display()
                    )
                };
                return Err(DriverError::GradingFailed(format!(
                    "result.json missing under {}; {underlying}",
                    output_dir.display()
                )));
            }
        };
        let bytes = fs::read(&result_path).map_err(|e| {
            DriverError::GradingFailed(format!("could not read {}: {e}", result_path.display()))
        })?;
        parse_harbor_result(&bytes, run.task.clone(), run.agent)
    }
}

/// Locate the single Harbor-written `result.json` inside our
/// per-task output dir. Harbor nests one timestamp subdir between
/// our `<jobs-dir>/<our-task-agent>/` and the actual result file:
///
/// ```text
/// <our-task-agent>/2026-05-24__20-08-13/result.json
/// ```
///
/// We walk one level down looking for any subdir containing
/// `result.json`. If there's exactly one, return its path.
/// Multiple matches → ambiguity surface, error out (caller should
/// inspect the directory). Zero matches → "Harbor ran without
/// writing a result," a useful surface for the caller's failure
/// notes.
fn find_harbor_result_path(output_dir: &std::path::Path) -> Result<PathBuf, String> {
    let entries = std::fs::read_dir(output_dir)
        .map_err(|e| format!("could not read {}: {e}", output_dir.display()))?;
    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let candidate = p.join("result.json");
        if candidate.is_file() {
            candidates.push(candidate);
        }
    }
    match candidates.len() {
        0 => Err("no <subdir>/result.json found".to_string()),
        1 => Ok(candidates.remove(0)),
        n => Err(format!(
            "expected exactly one result.json subdir; found {n}"
        )),
    }
}

// ---------- --mounts merge: combine kimetsu mounts with --harbor-arg mounts ----------

/// Harbor's CLI accepts a single `--mounts=<json-array>` per invocation.
/// When the +km driver wants to mount the kimetsu binary + workspace AND
/// the caller passes additional mounts via `--harbor-arg --mounts=...`,
/// passing both flags would let the last one win (silently dropping the
/// others). This merges them into one JSON array.
///
/// Returns `None` when there are no mounts to pass at all.
fn merge_mounts(driver_mounts: Option<&str>, extra_args: &[String]) -> Option<String> {
    let mut combined: Vec<serde_json::Value> = Vec::new();
    if let Some(raw) = driver_mounts
        && let Ok(serde_json::Value::Array(arr)) = serde_json::from_str(raw)
    {
        combined.extend(arr);
    }
    for arg in extra_args {
        let Some(json_part) = arg.strip_prefix("--mounts=") else {
            continue;
        };
        if let Ok(serde_json::Value::Array(arr)) = serde_json::from_str(json_part) {
            combined.extend(arr);
        }
    }
    if combined.is_empty() {
        return None;
    }
    serde_json::to_string(&serde_json::Value::Array(combined)).ok()
}

// ---------- Path normalization for docker bind-mount sources ----------

/// Normalize a host path for use as a `bind` mount `source` field.
///
/// Handles three Windows-specific quirks:
///   1. Backslashes → forward slashes (`E:\foo` → `E:/foo`).
///   2. UNC prefix from `fs::canonicalize` (`\\?\C:\foo` →
///      `C:/foo`). Windows' canonical path form embeds `\\?\` to
///      bypass MAX_PATH, but Docker rejects those.
///   3. Drive-colon (`E:/foo` → `/e/foo`, Git-Bash form). Even
///      with `COMPOSE_CONVERT_WINDOWS_PATHS=1`, the long-form
///      `--mounts` JSON we pass goes straight to Docker without
///      the conversion pass.
///
/// On Linux/macOS, just replaces backslashes with forward slashes.
pub fn normalize_for_mount(path: &std::path::Path) -> String {
    let mut raw = path.to_string_lossy().to_string().replace('\\', "/");
    // Strip UNC prefixes produced by Windows fs::canonicalize.
    for prefix in ["//?/UNC/", "//?/"] {
        if let Some(rest) = raw.strip_prefix(prefix) {
            raw = rest.to_string();
            break;
        }
    }
    if cfg!(windows) && raw.len() >= 2 {
        let bytes = raw.as_bytes();
        if bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
            let drive = (bytes[0] as char).to_ascii_lowercase();
            let rest = &raw[2..];
            let rest = if rest.starts_with('/') {
                rest.to_string()
            } else {
                format!("/{rest}")
            };
            return format!("/{drive}{rest}");
        }
    }
    raw
}

// ---------- Pure parsing helpers (testable without Harbor) ----------

/// Walk a locally-downloaded dataset directory and return each
/// task as a registry-qualified `TaskId`. Harbor's downloaded
/// layout is:
///
/// ```text
/// <registry-root>/<task-name>/<content-hash>/task.toml
/// ```
///
/// e.g. `~/.cache/harbor/tasks/packages/terminal-bench/adaptive-rejection-sampler/abc.../task.toml`.
/// The directory's name is the *registry*; each subdir is one
/// task. We return `TaskId("<registry>/<task-name>")` so the ids
/// match what Harbor's `--include-task-name` expects exactly
/// (Harbor's stored task ids are registry-prefixed; passing the
/// short form would miss them).
///
/// We filter to subdirs that contain at least one `task.toml` two
/// levels deep, so accidentally pointing at a wrong directory
/// (`~/.cache/harbor/tasks/packages/`, or some non-dataset dir)
/// returns empty instead of garbage.
pub fn parse_tasks_from_dataset_path(path: &std::path::Path) -> Result<Vec<TaskId>, DriverError> {
    if !path.is_dir() {
        return Err(DriverError::Other(format!(
            "tasks-from-dataset path is not a directory: {}",
            path.display()
        )));
    }
    let registry = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        DriverError::Other(format!(
            "tasks-from-dataset path has no readable basename: {}",
            path.display()
        ))
    })?;
    let entries = std::fs::read_dir(path).map_err(|e| {
        DriverError::Other(format!(
            "could not read dataset dir {}: {e}",
            path.display()
        ))
    })?;
    let mut tasks = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        // Confirm at least one `*/task.toml` exists under this dir
        // so we ignore stray non-task subdirs (READMEs, .git, etc).
        let has_task_toml = std::fs::read_dir(&p)
            .map(|inner| {
                inner.flatten().any(|hash_entry| {
                    let mut tp = hash_entry.path();
                    tp.push("task.toml");
                    tp.is_file()
                })
            })
            .unwrap_or(false);
        if !has_task_toml {
            continue;
        }
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            tasks.push(TaskId(format!("{registry}/{name}")));
        }
    }
    tasks.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(tasks)
}

/// Parse a JSON array of task entries into `Vec<TaskId>`. Currently
/// only exercised by tests — `list_tasks()` returns an
/// `InfraUnavailable` error because Harbor 0.7.1 has no
/// `list-tasks` subcommand. Reserved for the future
/// `harbor dataset download` + JSON-manifest path.
#[allow(dead_code)]
fn parse_list_tasks(stdout: &[u8]) -> Result<Vec<TaskId>, DriverError> {
    #[derive(Deserialize)]
    struct ListEntry {
        #[serde(alias = "id", alias = "name")]
        task_id: String,
    }
    let entries: Vec<ListEntry> = serde_json::from_slice(stdout).map_err(|e| {
        DriverError::Other(format!(
            "could not parse `harbor list-tasks --json` output as JSON array of objects with `task_id` (or `id`/`name`): {e}. \
             First 256 bytes: {}",
            String::from_utf8_lossy(&stdout[..stdout.len().min(256)])
        ))
    })?;
    Ok(entries.into_iter().map(|e| TaskId(e.task_id)).collect())
}

/// Scan stdout for a JSON line containing `"cost_usd"`. Harbor's
/// output format isn't fully spec'd; this is best-effort. Returns
/// None when no cost line is found (caller treats as 0).
fn parse_cost_from_stdout(stdout: &[u8]) -> Option<f32> {
    let text = std::str::from_utf8(stdout).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if !line.contains("cost_usd") {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line)
            && let Some(cost) = v.get("cost_usd").and_then(|v| v.as_f64())
        {
            return Some(cost as f32);
        }
    }
    None
}

/// Parse a Harbor 0.7.1 `result.json` into a `Grade`. The schema
/// (only the fields we need are listed; extras are tolerated):
///
/// ```json
/// {
///   "n_total_trials": 1,
///   "stats": {
///     "n_errored_trials": 0,
///     "n_cancelled_trials": 0,
///     "evals": {
///       "<agent>__<dataset>": {
///         "n_trials": 1,
///         "n_errors": 0,
///         "metrics": [{"mean": 1.0}],
///         "exception_stats": {"RuntimeError": ["..."]}
///       }
///     },
///     "cost_usd": 0.42
///   }
/// }
/// ```
///
/// Score: `stats.evals.<first-key>.metrics[0].mean`. When there
/// are multiple eval keys (rare; would mean Harbor ran the task
/// under multiple datasets) we average the means. When no metrics
/// are present, we treat the run as a hard failure (score=0).
///
/// Reason on failure: surface a summary of `exception_stats` so
/// the report shows which exception kind drove the score down.
fn parse_harbor_result(
    bytes: &[u8],
    task: TaskId,
    agent: AgentConfig,
) -> Result<Grade, DriverError> {
    #[derive(Deserialize)]
    struct HarborResult {
        #[serde(default)]
        stats: Option<Stats>,
    }
    #[derive(Deserialize)]
    struct Stats {
        #[serde(default)]
        n_errored_trials: u32,
        #[serde(default)]
        n_cancelled_trials: u32,
        #[serde(default)]
        evals: std::collections::BTreeMap<String, EvalEntry>,
    }
    #[derive(Deserialize)]
    struct EvalEntry {
        #[serde(default)]
        metrics: Vec<MetricEntry>,
        #[serde(default)]
        exception_stats: std::collections::BTreeMap<String, Vec<String>>,
    }
    #[derive(Deserialize)]
    struct MetricEntry {
        #[serde(default)]
        mean: Option<f64>,
    }

    let r: HarborResult = serde_json::from_slice(bytes).map_err(|e| {
        DriverError::GradingFailed(format!(
            "result.json was not valid JSON: {e}. First 256 bytes: {}",
            String::from_utf8_lossy(&bytes[..bytes.len().min(256)])
        ))
    })?;

    let stats = match r.stats {
        Some(s) => s,
        None => {
            return Ok(Grade {
                task,
                agent,
                score: 0.0,
                reason: Some("result.json had no `stats` block".to_string()),
            });
        }
    };

    // Score = average of all eval means. Missing / empty metrics → 0.
    let mut means: Vec<f64> = Vec::new();
    for entry in stats.evals.values() {
        if let Some(m) = entry.metrics.first().and_then(|m| m.mean) {
            means.push(m);
        }
    }
    let score: f32 = if means.is_empty() {
        0.0
    } else {
        (means.iter().sum::<f64>() / (means.len() as f64)) as f32
    };

    // Reason: if anything errored or got cancelled, summarize the
    // exception kinds across eval entries.
    let reason = if stats.n_errored_trials > 0 || stats.n_cancelled_trials > 0 || score == 0.0 {
        let exc_summary: Vec<String> = stats
            .evals
            .values()
            .flat_map(|entry| entry.exception_stats.keys().cloned())
            .collect();
        if exc_summary.is_empty() {
            Some(format!(
                "errored={} cancelled={} score={}",
                stats.n_errored_trials, stats.n_cancelled_trials, score
            ))
        } else {
            Some(format!(
                "exceptions=[{}] errored={} cancelled={}",
                exc_summary.join(","),
                stats.n_errored_trials,
                stats.n_cancelled_trials
            ))
        }
    } else {
        None
    };

    Ok(Grade {
        task,
        agent,
        score,
        reason,
    })
}

/// Trim trailing bytes off stderr for the notes field so we don't
/// embed multi-megabyte logs in the report. Lossy UTF-8 decode is
/// fine — these are debug strings.
fn tail_lossy(bytes: &[u8], max: usize) -> String {
    if bytes.len() <= max {
        return String::from_utf8_lossy(bytes).trim().to_string();
    }
    let cut = bytes.len() - max;
    let mut s = String::from_utf8_lossy(&bytes[cut..]).to_string();
    s = format!("...{}", s.trim_start());
    s
}

/// Dry-run synthesizer. Plausibly-skewed results so the report
/// formatter has something to render.
fn synth_dry_run_result(task: &TaskId, agent: AgentConfig) -> RunResult {
    let start = Instant::now();
    let score_jitter = match agent {
        AgentConfig::ClaudePlusKimetsu => 0,
        AgentConfig::ClaudeAlone => 1,
        AgentConfig::CodexPlusKimetsu => 2,
        AgentConfig::CodexAlone => 3,
    };
    let exit_code = if (task.0.len() + score_jitter).is_multiple_of(4) {
        1
    } else {
        0
    };
    RunResult {
        task: task.clone(),
        agent,
        exit_code,
        duration_secs: start.elapsed().as_secs_f64() + 12.5,
        cost_usd: if agent.uses_kimetsu() { 0.18 } else { 0.42 },
        notes: format!("[dry-run] simulated {} on {}", agent.label(), task),
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_list_tasks_accepts_task_id_field() {
        let stdout = br#"[{"task_id":"hello-world"},{"task_id":"git-bisect"}]"#;
        let tasks = parse_list_tasks(stdout).expect("parse");
        assert_eq!(
            tasks,
            vec![TaskId("hello-world".into()), TaskId("git-bisect".into())]
        );
    }

    #[test]
    fn parse_list_tasks_accepts_id_alias() {
        // Harbor older releases might use `id` instead of `task_id`.
        let stdout = br#"[{"id":"alpha"},{"id":"beta"}]"#;
        let tasks = parse_list_tasks(stdout).expect("parse");
        assert_eq!(tasks, vec![TaskId("alpha".into()), TaskId("beta".into())]);
    }

    #[test]
    fn parse_list_tasks_surfaces_useful_error_on_garbage() {
        let stdout = b"not-json-at-all";
        let err = parse_list_tasks(stdout).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("could not parse"),
            "error should mention parsing context; got: {msg}"
        );
    }

    #[test]
    fn merge_mounts_returns_none_when_nothing_to_mount() {
        assert_eq!(merge_mounts(None, &[]), None);
        assert_eq!(
            merge_mounts(None, &["--ae=FOO=bar".to_string()]),
            None,
            "non-mount extra args ignored"
        );
    }

    #[test]
    fn merge_mounts_combines_driver_and_extra_args() {
        let driver = r#"[{"source":"/e/kimetsu","target":"/usr/local/bin/kimetsu","type":"bind"}]"#;
        let extras = vec![
            "--ae=CLAUDE_CODE_OAUTH_TOKEN=sk-x".to_string(),
            r#"--mounts=[{"source":"/c/Users/r/.codex","target":"/root/.codex","type":"bind"}]"#
                .to_string(),
        ];
        let merged = merge_mounts(Some(driver), &extras).expect("merged");
        let parsed: serde_json::Value = serde_json::from_str(&merged).expect("valid json");
        let arr = parsed.as_array().expect("array");
        assert_eq!(arr.len(), 2, "should contain both mounts; got {merged}");
        let targets: Vec<&str> = arr
            .iter()
            .filter_map(|v| v.get("target")?.as_str())
            .collect();
        assert!(targets.contains(&"/usr/local/bin/kimetsu"));
        assert!(targets.contains(&"/root/.codex"));
    }

    #[test]
    fn merge_mounts_uses_only_extra_when_driver_none() {
        let extras = vec![r#"--mounts=[{"source":"/x","target":"/y","type":"bind"}]"#.to_string()];
        let merged = merge_mounts(None, &extras).expect("merged");
        let parsed: serde_json::Value = serde_json::from_str(&merged).expect("valid json");
        assert_eq!(parsed.as_array().unwrap().len(), 1);
    }

    #[test]
    fn parse_cost_from_stdout_finds_embedded_cost_line() {
        let stdout = b"some leading text\n{\"event\":\"done\",\"cost_usd\":0.42}\ntrailing\n";
        let cost = parse_cost_from_stdout(stdout).expect("parse cost");
        assert!((cost - 0.42).abs() < 1e-6);
    }

    #[test]
    fn parse_cost_from_stdout_returns_none_when_absent() {
        assert!(parse_cost_from_stdout(b"nothing about cost in here").is_none());
        assert!(parse_cost_from_stdout(b"").is_none());
    }

    /// Real-world fixture: this is what Harbor 0.7.1 wrote when a
    /// task errored at the Docker compose step. The driver should
    /// surface score=0 + the RuntimeError kind in the reason.
    #[test]
    fn parse_harbor_result_errored_task_surfaces_exception_kinds() {
        let bytes = br#"{
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
                "n_errored_trials": 1,
                "evals": {
                    "codex__terminal-bench/terminal-bench-2": {
                        "n_trials": 0,
                        "n_errors": 1,
                        "metrics": [{"mean": 0.0}],
                        "exception_stats": {"RuntimeError": ["adaptive-rejection-sampler__abc"]}
                    }
                },
                "cost_usd": null
            }
        }"#;
        let g = parse_harbor_result(
            bytes,
            TaskId("adaptive-rejection-sampler".into()),
            AgentConfig::CodexAlone,
        )
        .expect("parse");
        assert_eq!(g.score, 0.0);
        let reason = g.reason.unwrap_or_default();
        assert!(
            reason.contains("RuntimeError"),
            "expected RuntimeError in reason; got: {reason}"
        );
    }

    /// Happy path: a passing task surfaces score=1.0 and no reason.
    #[test]
    fn parse_harbor_result_pass_surfaces_clean_grade() {
        let bytes = br#"{
            "stats": {
                "n_errored_trials": 0,
                "n_cancelled_trials": 0,
                "evals": {
                    "claude-code__terminal-bench/terminal-bench-2": {
                        "n_trials": 1,
                        "n_errors": 0,
                        "metrics": [{"mean": 1.0}],
                        "exception_stats": {}
                    }
                }
            }
        }"#;
        let g = parse_harbor_result(bytes, TaskId("t".into()), AgentConfig::ClaudePlusKimetsu)
            .expect("parse");
        assert!((g.score - 1.0).abs() < 1e-6);
        assert!(g.reason.is_none());
    }

    /// Partial-credit path: a multi-trial mean lands between 0 and 1.
    #[test]
    fn parse_harbor_result_partial_credit_passes_through() {
        let bytes = br#"{
            "stats": {
                "evals": {
                    "claude-code__ds": {
                        "metrics": [{"mean": 0.75}]
                    }
                }
            }
        }"#;
        let g = parse_harbor_result(bytes, TaskId("t".into()), AgentConfig::ClaudeAlone)
            .expect("parse");
        assert!((g.score - 0.75).abs() < 1e-6);
    }

    /// Multiple eval keys (rare) → average their means.
    #[test]
    fn parse_harbor_result_averages_multiple_evals() {
        let bytes = br#"{
            "stats": {
                "evals": {
                    "a__x": {"metrics": [{"mean": 1.0}]},
                    "b__y": {"metrics": [{"mean": 0.0}]}
                }
            }
        }"#;
        let g = parse_harbor_result(bytes, TaskId("t".into()), AgentConfig::ClaudeAlone)
            .expect("parse");
        assert!((g.score - 0.5).abs() < 1e-6);
    }

    /// No stats block → score=0 + helpful reason. Shouldn't error.
    #[test]
    fn parse_harbor_result_missing_stats_grades_zero() {
        let bytes = br#"{"id": "abc"}"#;
        let g = parse_harbor_result(bytes, TaskId("t".into()), AgentConfig::ClaudeAlone)
            .expect("parse");
        assert_eq!(g.score, 0.0);
        assert!(g.reason.as_deref().unwrap_or("").contains("no `stats`"));
    }

    #[test]
    fn parse_harbor_result_surfaces_useful_error_on_garbage() {
        let err = parse_harbor_result(b"not-json", TaskId("t".into()), AgentConfig::ClaudeAlone)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("result.json was not valid JSON"),
            "expected helpful context; got: {msg}"
        );
    }

    #[test]
    fn tail_lossy_trims_to_max_with_ellipsis() {
        let bytes = b"abcdefghijklmnopqrstuvwxyz";
        let out = tail_lossy(bytes, 5);
        assert!(out.starts_with("..."), "got: {out}");
        // The last 5 chars should be present after the ellipsis.
        assert!(out.contains("vwxyz"), "got: {out}");
    }

    #[test]
    fn tail_lossy_returns_whole_string_when_under_limit() {
        let out = tail_lossy(b"short stderr", 1024);
        assert_eq!(out, "short stderr");
    }

    /// Directory walker: each immediate subdir containing `*/task.toml`
    /// counts as one task. Returns registry-prefixed ids so they match
    /// Harbor's `--include-task-name` filter exactly. Subdirs without
    /// nested task.toml are skipped (READMEs, .git, the wrong path
    /// level all produce empty or filtered results instead of garbage).
    #[test]
    fn parse_tasks_from_dataset_path_returns_registry_prefixed_ids() {
        // Use a unique-per-run leaf so concurrent test invocations don't
        // collide. The leaf becomes the registry name in returned ids.
        let root = std::env::temp_dir()
            .join(format!("kbench-test-{}", std::process::id()))
            .join("terminal-bench");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create root");

        // Two valid task dirs (with the <hash>/task.toml layout).
        for name in ["alpha", "beta"] {
            let hash_dir = root.join(name).join("deadbeef");
            std::fs::create_dir_all(&hash_dir).expect("mkdir");
            std::fs::write(hash_dir.join("task.toml"), b"version='1.0'\n").expect("write");
        }
        // A stray non-task dir without any task.toml under it.
        std::fs::create_dir_all(root.join("notes")).expect("mkdir notes");
        std::fs::write(root.join("README.md"), b"# dataset").expect("write readme");

        let tasks = parse_tasks_from_dataset_path(&root).expect("walk");
        assert_eq!(
            tasks,
            vec![
                TaskId("terminal-bench/alpha".to_string()),
                TaskId("terminal-bench/beta".to_string()),
            ],
            "expected registry-prefixed ids, sorted; got {tasks:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_tasks_from_dataset_path_errors_on_non_directory() {
        let bogus = std::env::temp_dir().join("kbench-not-a-real-dataset-dir");
        let _ = std::fs::remove_dir_all(&bogus);
        let err = parse_tasks_from_dataset_path(&bogus).unwrap_err();
        assert!(
            format!("{err}").contains("not a directory"),
            "expected 'not a directory' error; got: {err}"
        );
    }
}
