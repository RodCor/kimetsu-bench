//! Pre-run setup: discover everything kbench needs to run a real
//! Terminal-Bench comparison so the operator only types task names.
//!
//! What this module discovers:
//!   * Claude Code auth token  (env / .env / ~/.claude/auth.json)
//!   * Codex auth              (env OPENAI_API_KEY / .env / ~/.codex/)
//!   * Linux kimetsu binary    (cache / WSL2 build / GitHub release)
//!   * Brain workspace path    (defaults to the parent kimetsu repo)
//!
//! Outputs:
//!   * `harbor_args: Vec<String>` — to merge with `--harbor-arg` passthrough
//!   * `kimetsu_binary: Option<PathBuf>` — for kbench's `--kimetsu-binary`
//!   * `brain_workspace: PathBuf`
//!
//! A short setup banner is printed to stderr so the operator can see
//! what got picked up. Tokens are masked (`***`).

pub mod auth;
pub mod binary;

use std::path::{Path, PathBuf};

use auth::{ClaudeAuth, CodexAuth};
use binary::BinarySource;

pub struct Setup {
    pub harbor_args: Vec<String>,
    pub kimetsu_binary: Option<PathBuf>,
    pub brain_workspace: PathBuf,
    /// Path to the kimetsu workflow nudge markdown file (written into
    /// cache_dir). Passed to Harbor via `--extra-instruction-path` for
    /// +km agents so Claude is told to actually call kimetsu tools.
    /// `None` in dry-run mode or when there are no +km agents.
    pub kimetsu_nudge_path: Option<PathBuf>,
}

pub struct SetupOptions<'a> {
    pub bench_dir: &'a Path,
    pub repo_root: &'a Path,
    pub cache_dir: &'a Path,
    pub dry_run: bool,
    pub need_binary: bool,
    pub no_build: bool,
    pub brain_workspace_override: Option<PathBuf>,
    pub kimetsu_binary_override: Option<PathBuf>,
    /// Suppress the setup banner. Set by the single-trial worker so a
    /// large sweep doesn't reprint the banner once per (task, agent).
    pub quiet: bool,
}

pub fn discover(opts: SetupOptions) -> Setup {
    let claude = auth::discover_claude(opts.bench_dir, opts.repo_root);
    let codex = auth::discover_codex(opts.bench_dir, opts.repo_root);
    let harbor_args = auth::to_harbor_args(claude.as_ref(), &codex);

    // Harbor's claude_code.py reads CLAUDE_CODE_OAUTH_TOKEN / OPENAI_API_KEY
    // from os.environ.get(...) at agent-construction time. Without these in
    // the host process env, Harbor's agent records the value passed via --ae
    // but Harbor's own auth chain (e.g. ANTHROPIC_API_KEY fallback) sees
    // empties. Export into our process env so harbor sees both paths.
    //
    // SAFETY: single-threaded at this point; no other thread reads env.
    if let Some(ref c) = claude {
        unsafe {
            std::env::set_var(&c.var_name, &c.token);
        }
    }
    if let CodexAuth::Env { ref token, .. } = codex {
        unsafe {
            std::env::set_var("OPENAI_API_KEY", token);
        }
    }

    // Brain workspace defaults to a SHARED persistent dir under bench/.cache/.
    // Same brain.db across every trial in a gauntlet — and across gauntlets —
    // so memories accumulate and transfer learning has a chance to show up.
    // (Defaulting to the parent kimetsu repo would mix bench memories with
    // the user's dev work, which is wrong.)
    let brain_workspace = opts.brain_workspace_override.clone().unwrap_or_else(|| {
        let p = opts.cache_dir.join("brain-workspace");
        let _ = std::fs::create_dir_all(&p);
        p
    });

    // Make the brain workspace a git boundary AND an initialized kimetsu
    // project. Two reasons this matters:
    //
    //   1. `kimetsu_brain` uses `git rev-parse --show-toplevel` to find
    //      the "project root" that anchors `.kimetsu/`. Without a local
    //      .git/, walking up from `bench/.cache/brain-workspace/` lands
    //      in `bench/.kimetsu/` (the OUTER bench git repo) — wrong dir,
    //      and divergent from what the in-container MCP server sees.
    //   2. The MCP server inside the Docker container calls
    //      `kimetsu_benchmark_context`, which requires
    //      `<workspace>/.kimetsu/project.toml` to exist. Without it the
    //      tool returns an `initialized: false` error and Claude gets
    //      no signal even when it does ask for context.
    //
    // Both contexts (host kbench + in-container MCP) now converge on
    // `<brain_workspace>/.kimetsu/brain.db` regardless of git context.
    init_brain_workspace_for_bench(&brain_workspace);

    let (binary_source, kimetsu_binary) = if opts.dry_run || !opts.need_binary {
        (None, opts.kimetsu_binary_override.clone())
    } else if let Some(p) = opts.kimetsu_binary_override.clone() {
        // User-supplied path wins; no auto-resolution.
        (None, Some(p))
    } else {
        let src = binary::resolve(opts.cache_dir, opts.repo_root, opts.no_build);
        let path = src.as_ref().map(|s| s.path().to_path_buf());
        (src, path)
    };

    // Write the kimetsu nudge to a file once; driver passes it via
    // --extra-instruction-path for every +km invocation. Skip in dry-run
    // and when no +km agents are requested.
    let kimetsu_nudge_path = if opts.dry_run || !opts.need_binary {
        None
    } else {
        let p = opts.cache_dir.join("kimetsu-nudge.md");
        match std::fs::write(&p, crate::drivers::terminal_bench::KIMETSU_AGENT_NUDGE) {
            Ok(_) => Some(p),
            Err(e) => {
                eprintln!(
                    "kbench: setup: could not write kimetsu nudge to {}: {e}",
                    p.display()
                );
                None
            }
        }
    };

    if !opts.quiet {
        print_banner(
            &claude,
            &codex,
            &brain_workspace,
            binary_source.as_ref(),
            kimetsu_binary.as_deref(),
            kimetsu_nudge_path.as_deref(),
            opts.dry_run,
            opts.need_binary,
        );
    }

    Setup {
        harbor_args,
        kimetsu_binary,
        brain_workspace,
        kimetsu_nudge_path,
    }
}

/// Set up the brain workspace so the host kbench process and the
/// in-container MCP server both find the SAME `.kimetsu/brain.db`.
/// Idempotent — safe to call on every kbench invocation.
fn init_brain_workspace_for_bench(workspace: &Path) {
    // Anchor `git rev-parse` here, not at the outer bench/ git repo.
    // Plain `git init` (creates .git/). Silently swallow errors; if git
    // isn't installed we'll just write to the wrong .kimetsu/ but the
    // run will still complete (and the user will see the symptom in
    // the +km trial's tool-result error).
    if !workspace.join(".git").exists()
        && let Err(e) = std::process::Command::new("git")
            .arg("init")
            .arg("-q")
            .current_dir(workspace)
            .status()
    {
        eprintln!(
            "kbench: setup: could not git-init brain workspace {}: {e} (continuing)",
            workspace.display()
        );
    }
    // Create .kimetsu/project.toml + brain.db so the MCP server's
    // kimetsu_benchmark_context doesn't error with `initialized: false`.
    if let Err(e) = kimetsu_brain::project::init_project(workspace, false) {
        eprintln!(
            "kbench: setup: kimetsu init at {} failed: {e} (continuing)",
            workspace.display()
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn print_banner(
    claude: &Option<ClaudeAuth>,
    codex: &CodexAuth,
    brain_workspace: &Path,
    binary_source: Option<&BinarySource>,
    kimetsu_binary: Option<&Path>,
    kimetsu_nudge: Option<&Path>,
    dry_run: bool,
    need_binary: bool,
) {
    eprintln!("kbench: setup");
    match claude {
        Some(c) => eprintln!("  claude auth      : {} (token: ***)", c.source),
        None => eprintln!("  claude auth      : (none — claude runs will fail)"),
    }
    match codex {
        CodexAuth::Env { source, .. } => eprintln!("  codex auth       : {source} (token: ***)"),
        CodexAuth::DirMount { source, .. } => eprintln!("  codex auth       : {source}"),
        CodexAuth::None => eprintln!("  codex auth       : (none — codex runs will fail)"),
    }
    eprintln!("  brain workspace  : {}", brain_workspace.display());
    if dry_run {
        eprintln!("  linux binary     : (dry-run, skipped)");
    } else if !need_binary {
        eprintln!("  linux binary     : (not needed — no +km agents requested)");
    } else if let Some(src) = binary_source {
        eprintln!("  linux binary     : {}", src.describe());
    } else if let Some(p) = kimetsu_binary {
        eprintln!("  linux binary     : user-supplied {}", p.display());
    } else {
        eprintln!("  linux binary     : (not found — +km agents will degrade silently)");
    }
    if let Some(p) = kimetsu_nudge {
        eprintln!(
            "  kimetsu nudge    : {} (via --extra-instruction-path)",
            p.display()
        );
    }
}
