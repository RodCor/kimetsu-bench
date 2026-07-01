# kimetsu-bench

The benchmark harness for [kimetsu](https://github.com/RodCor/kimetsu) — a
local-first memory sidecar for AI coding agents. It answers one question: does
kimetsu actually help the agents it attaches to, and how does the brain itself
perform?

**Datasets and raw results are never committed here** (see `.gitignore`) — this
repo ships *how to run* the benchmarks: the drivers, converters, and docs. The
drivers download or convert public datasets and write all artifacts to a
gitignored `local/` folder, so you reproduce the numbers yourself rather than
taking ours on faith.

## What it measures

`kbench` runs one or more Terminal-Bench tasks under multiple agent
configurations and produces a side-by-side comparison:

| Agent       | What it is                                           |
|-------------|------------------------------------------------------|
| `claude+km` | Claude Code + kimetsu attached via `--mcp-config`    |
| `claude`    | Claude Code (baseline)                               |
| `codex+km`  | Codex + kimetsu attached via `--mcp-config`          |
| `codex`     | Codex (baseline)                                     |

The interesting metric is the diff between the `+km` column and its
baseline — does kimetsu earn its keep?

## How it works (~30 seconds of reading)

```
You: make bench TASK=fix-git
 │
 ▼
kbench auto-discovers
  • Claude OAuth token  (env / .env / ~/.claude/auth.json)
  • Codex auth          (env / .env / ~/.codex bind-mount)
  • Linux kimetsu binary (cache / WSL2 build / GitHub release)
  • Brain workspace     (parent kimetsu repo)
 │
 ▼
For each (task × agent) pair:
  harbor run --dataset terminal-bench/terminal-bench-2
             --include-task-name */<task>
             --agent claude-code | codex
             --mcp-config <kimetsu.mcp.json>   # +km only
             --mounts=[<kimetsu binary + workspace + codex>]
             --ae=CLAUDE_CODE_OAUTH_TOKEN=***
 │
 ▼
Harbor spins up Docker, runs the host agent inside,
captures stdout + result.json
 │
 ▼
kbench reads result.json, builds markdown table:
  | Task    | claude+km          | claude             |
  | fix-git | ✓ 1.00 (328s, $0)  | ✗ 0.00 (118s, $0)  |
```

Report is printed to stdout AND saved to `local/runs/auto/<timestamp>.md`.

## Prerequisites

- **Docker** running.
- **Harbor 0.8+** (`uv tool install harbor` or `pip install harbor`).
- **Rust toolchain** with the kimetsu repo at `../` (path deps).
- **WSL2 on Windows** — see note below. Linux/macOS hosts just need the above.

### Windows hosts MUST run from WSL2

Harbor's per-trial `/logs/{agent,verifier,artifacts}` bind-mounts use raw
Windows paths (`E:/Kimetsu/...`) which Docker Desktop's WSL2 backend silently
drops in long-form mounts — every real run dies with `RewardFileNotFoundError`.
From WSL2, Harbor's paths become `/mnt/e/...` and Docker handles them cleanly.

One-time WSL2 setup:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install harbor          # installs to /root/.local/bin/
```

Then run from WSL2 (the `Makefile` puts `/root/.local/bin` on PATH for you):

```bash
wsl -d Ubuntu
$ cd /mnt/e/Kimetsu/bench
$ make bench TASK=fix-git
```

Dry-runs (`make dry`) work from PowerShell directly since they don't touch
Harbor or Docker.

### Keeping benchmark writes off the C: drive

On WSL2 the distro's root filesystem (`/`, `/tmp`, `/root/.cache`) is a virtual
disk (`ext4.vhdx`) that by default lives on **C:**, while the bench repo is on
**E:** (`/mnt/e`). Anything written outside `/mnt/e` grows that C: vhdx. There
are three sources of benchmark writes, in decreasing order of how much the repo
controls them:

1. **Per-trial Harbor artifacts** — *handled in code.* kbench passes Harbor an
   absolute `--jobs-dir` under `bench/local/runs/<run-ts>/`, so `result.json`, agent
   logs, and verifier output land on E:. (Each trial's *working directory*
   stays on `/tmp` — an empty dir, no artifacts — to dodge a DrvFs `getcwd`
   staleness crash; only the output moves to E:.)
2. **Harbor's dataset cache** (`~/.cache/harbor`) — run **`make setup-cache`**
   once. It symlinks `~/.cache/harbor → bench/.cache/harbor` (migrating any
   existing data), so downloaded datasets live on E:. The real-run targets
   (`bench`/`family`/`sweep`/`full`) depend on it, so it's enforced.
3. **Docker's image/container/build storage** — *Docker Desktop config, not a
   repo change.* This is the largest remaining C: consumer during a run. Move it
   in **Docker Desktop → Settings → Resources → Advanced → "Disk image
   location"** (point it at a folder on E:), or for the WSL2 backend:

   ```powershell
   wsl --manage docker-desktop-data --move E:\wsl\docker-desktop-data
   ```

   To take **everything** off C: (so even `/tmp` is on E:), relocate the whole
   Ubuntu distro's vhdx — export, unregister, re-import to E::

   ```powershell
   wsl --shutdown
   wsl --export Ubuntu E:\wsl\ubuntu.tar
   wsl --unregister Ubuntu
   wsl --import Ubuntu E:\wsl\ubuntu E:\wsl\ubuntu.tar
   ```

   (Back up first; this rewrites where the distro lives.)

## Quick start

The `Makefile` is the single entry point — a thin wrapper over `kbench` (build +
PATH + dispatch). Run `make help` to see every target.

```bash
cd bench   # in WSL2 on Windows; native shell on Linux/macOS

# Smoke test — no Harbor, no API, no Docker
make dry TASK=fix-git

# Real run: claude+km vs claude on one task
make bench TASK=fix-git

# Real run: codex+km vs codex (Codex requires MODEL=)
make bench TASK=fix-git AGENTS=codex+km,codex MODEL=gpt-5-codex-2025-08-19

# Real run: all 4 agents on multiple tasks
make bench TASK=fix-git,cobol-modernization AGENTS=claude+km,claude,codex+km,codex MODEL=gpt-5-codex-2025-08-19

# One programming-language family (see: make list-families)
make family FAM=python

# All families, sequentially (the full gauntlet)
make sweep

# Full Terminal-Bench dataset (must be downloaded: harbor dataset download terminal-bench/terminal-bench-2)
make full
```

> The underlying binary is still there if you need it: every `make` target maps
> to a `./target/release/kbench …` invocation. Run `cargo run --release -- --help`
> for the raw flag surface.

## Auth

`kbench` finds credentials automatically. Order:

| Provider | Looks in (first hit wins) |
|----------|----------------------------|
| Claude   | `CLAUDE_CODE_OAUTH_TOKEN` env → `ANTHROPIC_API_KEY` env → `bench/.env` → `../.env` → `~/.claude/auth.json` (oauthToken / accessToken / token / claudeAiOAuth fields). On WSL2 also `/mnt/c/Users/*/.claude/auth.json`. |
| Codex    | `OPENAI_API_KEY` env → `bench/.env` → `../.env` → `~/.codex/` bind-mount. On WSL2 also `/mnt/c/Users/*/.codex/`. |

Easiest: drop your token in `bench/.env` (gitignored):

```
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

The setup banner at the top of every run shows what got picked up.

## Escape hatches

Set these as `make` variables (they map to the underlying `kbench` flags):

| Variable               | Purpose                                                                |
|------------------------|------------------------------------------------------------------------|
| `AGENTS=claude+km,codex` | Explicit agent list (any subset of the 4)                            |
| `MODEL=<name>`         | Model forwarded to Harbor (required for codex)                         |
| `KBIN=<path>`          | Override the auto-resolved Linux kimetsu binary                        |
| `NB=1`                 | Skip WSL2 cargo build; use cached binary or GitHub release             |
| `HARBOR_ARGS="<arg>"`  | Forward an extra arg to `harbor run`                                   |
| `OUTPUT=json`          | JSON report instead of markdown                                        |

Example: bump per-task timeout:

```bash
make bench TASK=adaptive-rejection-sampler HARBOR_ARGS="--agent-timeout-multiplier=2"
```

## Housekeeping

| Target              | What it does                                                       |
|---------------------|-------------------------------------------------------------------|
| `make cost`         | Token-usage → cost report from `/tmp/kbench-*` (families from the manifest) |
| `make list-families`| Print the programming-language families and task counts           |
| `make setup-cache`  | Symlink harbor's dataset cache onto the bench drive, off the C: vhdx (see below) |
| `make clean-cache`  | Delete `.cache` scratch/logs; keep `linux-build*`, `brain-workspace*`, `warm-*`, `harbor/` |
| `make prune-runs`   | Delete per-trial run dirs under `local/runs/`, keep `local/runs/auto` + `local/runs/stress` reports |
| `make clean`        | `cargo clean`                                                      |

## Output

- **Markdown report** → stdout + `bench/local/runs/auto/<rfc3339-timestamp>.md`
- **Per-task Harbor artifacts** → `bench/local/runs/<run-ts>/<task>-<agent>/<harbor-ts>/<trial>/`
  with `result.json`, `trial.log`, `verifier/reward.txt`, `agent/claude-code.txt`.
  Useful for post-mortem when a task fails.

`bench/local/runs/` is gitignored. `make prune-runs` clears the bulky per-trial dirs
while keeping the `local/runs/auto` summary reports.

## Brain stress test (`kstress`)

A second binary, `kstress`, profiles the **brain itself** (not agent tasks) at
scale — seed 100 → 1,000,000 memories and measure insert/query latency, db size,
read/write concurrency (local), and HTTP throughput against `kimetsu-remote`
(remote), across two matrices: **lean** (FTS-only) and **embeddings**
(fastembed BGE-small + `vec0` ANN). It builds from the local v1.0.0 `../`
workspace only — never a release.

```bash
make stress-smoke     # fast wiring check (scales 100,500; local + remote)
make stress-local     # local sweep, both matrices
make stress-remote    # remote HTTP sweep, both matrices
make stress           # everything, into one local/runs/stress/<ts>/ dir
```

Output → `local/runs/stress/<ts>/<mode>-<matrix>/{summary.json,report.md,data.csv}`,
stamped with the kimetsu version + `../` git SHA. Working brains live under
`.cache/stress/` (on E:, never C:).

> **Build host:** the embeddings matrix links the ONNX runtime, which needs
> **glibc ≥ 2.38** — run the stress targets from a recent distro (e.g.
> **Ubuntu-24.04** on WSL2), not Ubuntu 20.04. One `--features embeddings` build
> runs both matrices: the lean matrix pins `KIMETSU_BRAIN_EMBEDDER=noop` (FTS-only
> path); the emb matrix uses the real embedder + `vec0` index.

Key knobs (Makefile vars): `STRESS_SCALES`, `STRESS_MAX_EMB` (caps the slow
embeddings tier; default 50000 — emb seeding is embed-bound), `STRESS_MATRICES`,
`STRESS_REMOTE_SCALES`, `STRESS_CONCURRENCY`.

> **Filesystem matters a LOT at scale.** Working brains default to `.cache/stress`
> on **E:**, which from WSL2 is a **9p drvfs** mount. 9p random reads over a large
> DB are punishingly slow — at 100k memories, seed and query latency run **12–33×
> slower than native ext4** (measured), so high-scale numbers reflect the
> *filesystem*, not the brain. For realistic numbers above ~50k, put the
> *transient* working DB on a native ext4 path:
>
> ```bash
> make stress STRESS_WORK=$HOME/.cache/kstress STRESS_MAX_LEAN=1000000
> ```
>
> Reports still land in `local/runs/stress/` on E:; only the throwaway seed DB moves.
> Note `$HOME/.cache` is on the WSL distro's ext4 vhdx (on C: unless you relocated
> the distro to E: per "Keeping benchmark writes off the C: drive" above). The
> default (E:/9p) honors the no-C: rule but is only practical to ~50k; `kstress`
> prints a warning when you run ≥50k on a 9p mount.

## Memory benchmarks (LongMemEval, BEAM, BrainBench)

Three drivers measure the **brain's memory quality** on public and authored
tasks. Each spins up a fresh, isolated Kimetsu brain per item, ingests the
conversation, retrieves per question via `kimetsu brain context`, answers with an
LLM **reader**, and scores per ability/category. **No datasets or raw results are
committed** — you bring (or convert) the data, and every artifact is written to
the gitignored `local/` folder.

### Prerequisites

- **Build once:** `cargo build --release` (or run each driver via
  `cargo run --release --bin kbench -- <driver> …`).
- **A kimetsu binary** — set `KIMETSU_BIN=/path/to/kimetsu` (or pass
  `--kimetsu-binary`). Build it from the parent repo (`cargo build --release -p
  kimetsu-cli`) or use an installed `kimetsu` on `PATH`.
- **A reader model.** Default is **Codex** (`--reader-backend codex`) — no API
  key, driven via `codex exec` with your ChatGPT login. LongMemEval also supports
  any OpenAI-compatible endpoint (`--reader-backend http` + `KBENCH_LLM_MODEL` /
  `KBENCH_LLM_API_KEY` / `KBENCH_LLM_BASE_URL`).
- **Node 18+** — only to build the BEAM dataset (the converter uses global `fetch`).

Flags shared by all three: **`--dry-run`** (parse + count, no model calls) and
**`--synthetic`** (tiny built-in fixture, full loop) validate the pipeline before
a real run; **`--limit N`** caps items; **`--output json|markdown`** picks the
report format. Reports are saved to `local/runs/<driver>/<timestamp>.{md,json}`.

### LongMemEval — long-term-memory QA

[LongMemEval](https://github.com/xiaowu0162/LongMemEval): single/multi-session,
temporal-reasoning, knowledge-update, and preference questions over a long chat
haystack. Download `longmemeval_s.json` from the LongMemEval repo into
`local/lme-data/`, then:

```bash
# Stratified 200-question slice (round-robin across all six question types):
cargo run --release --bin kbench -- longmemeval \
  --dataset local/lme-data/longmemeval_s.json --limit 200 --reader-backend codex

# No dataset — 5 built-in synthetic instances, real reader:
cargo run --release --bin kbench -- longmemeval --synthetic --reader-backend codex
```

Scores per question type + overall. `--question-types a,b` filters types; omit
`--limit` for the full 500.

### BEAM — ten memory abilities over long conversations

[BEAM](https://github.com/mohammadtavakoli78/BEAM): information extraction,
multi-session reasoning, knowledge update, temporal reasoning, abstention,
contradiction resolution, event ordering, instruction following, preference
following, and summarization — each probe graded by an LLM judge against a
rubric. Build a dataset from the BEAM repo's JSON with the bundled converter (it
fetches the chosen token bucket from GitHub):

```bash
# 100K bucket (20 conversations)  -> local/beam-data/beam-100k.json
node scripts/convert_beam.js --repo-path chats/100K --convs 1-20 --bucket 100k
# 1M bucket (35 conversations)    -> local/beam-data/beam-1m.json
node scripts/convert_beam.js --repo-path chats/1M --convs 1-35 --bucket 1m

cargo run --release --bin kbench -- beam \
  --dataset local/beam-data/beam-100k.json --reader-backend codex
cargo run --release --bin kbench -- beam --dry-run     # count probes, no calls
```

`--categories a,b` runs only some abilities; `--limit N` caps conversations.
Global-aggregation abilities (summarization, event ordering, …) need a wide
retrieval budget to see enough of the conversation — see the driver's module docs.

### BrainBench — reader-free brain capability

Drives the real `kimetsu` binary and scores the brain **directly**, with no LLM
reader in the loop: retrieval correctness, dedup, importance ranking, forgetting,
and calibration. Runs from a built-in fixture or an authored dataset:

```bash
cargo run --release --bin kbench -- brainbench --synthetic
cargo run --release --bin kbench -- brainbench \
  --dataset <scenarios>.json --tiers easy,medium,hard
```

## Troubleshooting

- **`No reward file found` on Windows** — you ran from PowerShell. Re-run from
  WSL2 (see Prerequisites).
- **`claude auth: (none)`** in the banner — set `CLAUDE_CODE_OAUTH_TOKEN` in
  `bench/.env` or shell env.
- **`linux binary: (not found)`** — install Rust in WSL2 (`curl https://sh.rustup.rs -sSf | sh`),
  or run with `KBIN=<path>` pointing at a pre-built Linux ELF.
- **`Model name is required`** — pass `MODEL=gpt-5-codex-2025-08-19`
  (Harbor's codex agent always needs it).

## Repo layout

```
bench/
├── Makefile               # ← single workflow entry point (make help)
├── README.md              # this file (operator surface)
├── Cargo.toml             # standalone Rust project (not in kimetsu workspace)
├── datasets/
│   └── prog-families-v1.json   # task → language-family manifest (--family / sweep)
├── scripts/
│   ├── cost.sh            # token-usage → cost report (reads the manifest)
│   └── convert_beam.js    # build a BEAM dataset from the BEAM repo's JSON
├── local/                 # gitignored: ALL datasets, results, and run reports
└── src/
    ├── main.rs            # `kbench` CLI + orchestrator
    ├── driver.rs          # BenchmarkDriver trait + types
    ├── drivers/
    │   ├── terminal_bench.rs   # Harbor + Terminal-Bench impl
    │   ├── longmemeval.rs      # LongMemEval memory benchmark
    │   ├── beam.rs             # BEAM (ten memory abilities)
    │   └── brainbench.rs       # reader-free brain-capability bench
    ├── report.rs          # JSON + markdown comparison output
    ├── setup/             # auto-discovery: auth + Linux binary + workspace
    │   ├── mod.rs
    │   ├── auth.rs
    │   └── binary.rs
    └── bin/kstress/       # `kstress` brain stress test (local + remote)
        ├── main.rs        # CLI (local/remote), matrix env, report glue
        ├── corpus.rs      # deterministic synthetic-memory generator
        ├── seed.rs        # fast bulk-seed (direct SQL + batch embed)
        ├── local.rs       # in-process profiler (latency/size/concurrency)
        ├── remote.rs      # spawn kimetsu-remote + concurrent HTTP load
        └── report.rs      # StressReport → JSON / Markdown / CSV
```

`CHANGELOG.md` has the version history of the bench tool itself plus the
headline impact results from each gauntlet (MP-4 through MP-14).

## Working tree

Clone this repo INSIDE the kimetsu working tree at `./bench/`:

```
kimetsu/                # public repo (github.com/RodCor/kimetsu)
├── crates/
├── .gitignore          # contains /bench/
└── bench/              # ← THIS REPO (private)
```

Cargo path deps (`../crates/kimetsu-*`) resolve from this layout.

## Commit convention

Same as kimetsu: short imperative subject lines, `Co-Authored-By` trailers.

## Contributing & conduct

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to build, the quality bar, and
what never gets committed (datasets, results, secrets). All participation is
under the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option. Unless you explicitly state otherwise, any contribution you submit
for inclusion shall be dual-licensed as above, without additional terms.
