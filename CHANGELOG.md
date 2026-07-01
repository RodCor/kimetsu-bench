# kimetsu-bench changelog

What shipped in the bench tool itself and the headline impact results we
measured against each kimetsu version. For kimetsu's API/feature
changelog, see the public [kimetsu CHANGELOG](../CHANGELOG.md).

## Bench tool

### v0.5 — 2026-06-05 — `kstress` brain stress test

New second binary `kstress` profiles the brain (not agent tasks) at scale —
100 → 1,000,000 memories — across lean (FTS) and embeddings (BGE-small + `vec0`
ANN) matrices, local and remote. All code lives in `bench/`; it consumes only
kimetsu's PUBLIC v1.0.0 API plus a direct `rusqlite` handle, with ZERO changes
to `crates/`.

- **local**: incremental seed; per-checkpoint insert (bulk rows/s + realistic
  `add_memory` p50/p95/p99), FTS + `retrieve_context` latency (cold/warm), db
  size (file + per-table via `dbstat` + `vec0`), read concurrency + writer-lock
  contention, RSS.
- **remote**: spawns the locally-built `kimetsu-remote serve`, drives it with
  concurrent `ureq` workers — throughput + p50/p95/p99 vs concurrency, plus
  rate-limit (HTTP 429) and multi-repo isolation probes.
- Fast bulk-seed bypasses the O(N²) per-insert conflict scan (direct SQL +
  batch embed); `vec0` is built lazily by kimetsu's first query. Seeding uses
  `git_init_boundary` so project discovery resolves to the stress root, not the
  enclosing bench repo.
- Output `runs/stress/<ts>/<mode>-<matrix>/{summary.json,report.md,data.csv}`,
  version/SHA-stamped. Working DBs under `.cache/stress/` (E:, never C:).
- `make stress{,-local,-remote,-smoke}`. The embeddings matrix links ONNX
  (needs glibc ≥ 2.38); the Makefile auto-detects glibc and falls back to a
  lean-only build on older distros (run from Ubuntu-24.04 for emb).

Scale/perf notes:
- Bulk seed tuned for 1M: `synchronous=OFF` + 256MB cache + 25k-row batches +
  periodic WAL checkpoint (≈15× faster seeding).
- Synthetic corpus carries a high-cardinality `kw<bucket>` token so queries stay
  selective (FTS matches ~tens of rows, not half the corpus) at any scale.
- DB size read via `PRAGMA page_count` (authoritative — `fs::metadata` returns 0
  for a large freshly-written file on the 9p mount).
- **Filesystem dominates at scale**: on WSL2, `.cache/stress` is a 9p drvfs mount
  where large-DB I/O is ~12–33× slower than ext4 — so high-scale numbers measure
  the filesystem, not the brain. `--work <ext4 path>` (Makefile `STRESS_WORK`)
  puts the throwaway seed DB on a native fs; `kstress` warns at ≥50k on 9p.
  Default stays on E: (honors no-C:), practical to ~50k.

### v0.4 — 2026-06-05 — single Makefile workflow + repo hygiene

One tracked, shareable entry point. The benchmark workflow used to live in
loose `cargo run --release -- …` invocations plus two scripts stranded inside
gitignored `.cache/` (`sweep-all.sh`, `cost.sh`) with hardcoded absolute paths.

- **`Makefile`** at the repo root is now the single workflow: `make bench`,
  `make family`, `make sweep` (the full per-family gauntlet, replaces
  `sweep-all.sh`), `make full`, `make dry`, `make cost`, plus housekeeping
  (`make clean-cache`, `make prune-runs`). Path-self-resolving, exports PATH
  for WSL2's harbor. Still a thin wrapper — every target shells out to
  `./target/release/kbench`.
- **`scripts/cost.sh`** promoted out of `.cache/` and rewired to read the
  task→family map from `datasets/prog-families-v1.json` instead of duplicating
  it inline.
- **`.cache/` hygiene:** dropped throwaway session scratch and regenerable
  artifacts; kept the heavy caches (`linux-build*`, `brain-workspace*`,
  `warm-*`). `make clean-cache` makes this repeatable.

**Keep benchmark writes off C:.** On WSL2 the distro root (`/tmp`, `~/.cache`)
is a vhdx on C:; the bench repo is on E:. Two fixes pull writes back to E::

- Per-trial Harbor artifacts: the orchestrator now computes one absolute run
  dir (`bench/runs/<run-ts>/`) and plumbs it to every worker via a hidden
  `--run-dir`, so Harbor's `--jobs-dir` is absolute on E:. The worker cwd stays
  on `/tmp` (empty dir) to keep dodging the DrvFs `getcwd` staleness crash —
  only the bulky output moved. Trials regroup under one run dir like pre-`/tmp`.
- `make setup-cache` symlinks `~/.cache/harbor → bench/.cache/harbor`; the
  real-run targets depend on it. Docker's data-root + (optionally) the whole
  WSL vhdx are manual moves — documented in the README.

Deleted: `docs/` (historical planning notes), `kimetsu_harbor/` (dead Python
shim). README quick-start now documents the `make` surface.

### v0.3 — 2026-05-25 — one-command pipeline

`cargo run -- <tasks>` is now the whole interface. Everything else is
auto-discovered:

- Claude / Codex auth from env vars, `.env`, `~/.claude/auth.json`,
  `~/.codex/` (and `/mnt/c/Users/*/` when running from WSL2 as root).
- Linux kimetsu binary: cache → WSL2 cargo build → GitHub release fallback.
- Brain workspace defaults to the parent kimetsu repo.

New flags: `--codex`, `--both`, `--full-dataset`, `--no-build`. Tasks are
positional. Setup banner shows what got picked up.

Driver fixes:
- Harbor only honors one `--mounts` flag — driver now merges kimetsu's
  binary + workspace mounts with the auth-discovered codex mount into a
  single JSON array. Without this, `+km` runs ran with no kimetsu attached.
- Auth tokens exported to host env so Harbor's `claude_code.py` agent
  (which reads `os.environ.get(...)`) actually receives them.

Deleted: `auto-bench.ps1` (440 LOC PowerShell wrapper, logic moved into
Rust), `kimetsu_harbor/` (Python shim — driver uses Harbor's built-in
`claude-code` / `codex` agents + `--mcp-config` instead).

Operator surface collapsed to one self-contained `README.md`.

**Known limitation:** real runs on Windows must be invoked from WSL2.
Harbor's per-trial `/logs/{agent,verifier,artifacts}` bind-mounts emit
raw `E:/...` paths in docker-compose long-form mounts, which Docker
Desktop's WSL2 backend silently drops → every run dies with
`RewardFileNotFoundError`. From WSL2 the paths become `/mnt/e/...` and
Docker accepts them. See README for the WSL2 invocation.

### v0.2 — 2026-05-23 — Harbor refactor (kimetsu v0.5.3-v0.5.5)

Cloned out of the kimetsu repo into a separate git tree at `./bench/`.

- `BenchmarkDriver` trait + Terminal-Bench impl wrapping Harbor 0.7+.
- JSON / markdown comparison reports.
- Python Harbor adapter moved over from `kimetsu/kimetsu_harbor/`.
- First end-to-end CLI: `cargo run -- --driver tb --tasks ... --agents claude+km,claude`.

### v0.1 — 2026-05-11 — scaffold

`kbench` binary stub + dry-run path for orchestrator development without
a live Harbor.

## Impact gauntlets (Terminal-Bench)

| Date       | Run    | Setup                                                  | Headline                                                        |
|------------|--------|--------------------------------------------------------|-----------------------------------------------------------------|
| 2026-05-11 | MP-4   | 16 tasks × 5 modes, $7.38                              | All modes green; personal-memory pipeline deprioritized for v0.1. |
| 2026-05-13 | MP-8   | v0.2 Terminal-Bench gauntlet                           | Claude Code CLI deemed wrong tool surface for v0.2.             |
| 2026-05-13 | MP-10  | 16 tasks, Opus 4.7, n=2 k=1                            | First clean kimetsu-no-brain baseline.                          |
| 2026-05-14 | MP-11  | 3 modes (bare CC, no-brain, brain) × 16, Opus 4.7      | Three-mode comparison; brain leg measured.                      |
| 2026-05-15 | MP-12  | 7-tool wider surface, n=16                             | Tool widening alone didn't move the needle.                     |
| 2026-05-16 | MP-13  | + retry-on-5xx + auto-orient + persistence gate        | Fixed brittle Anthropic 5xx handling.                           |
| 2026-05-16 | MP-13g | Brain-leg re-run with retry-on-5xx                     | Stabilized brain leg.                                           |
| 2026-05-17 | MP-14  | Wider tool surface, gate-1 re-run                      | First clean brain > no-brain margin (+6.25pp).                  |
| 2026-05-25 | smoke  | `fix-git`, v0.3 pipeline                               | `claude+km` ✓ 1.0; `claude` ✗ 0.0 (n=1).                        |

## Released kimetsu versions covered

- **v0.1** — initial scaffold, MVP.
- **v0.2** — Terminal-Bench validation; brain budget control.
- **v0.3.x** — chat client + bridge plugin + prompt-cache visibility
  (v0.3.4); perf passes (v0.3.5).
- **v0.4.x** — distribution arc: user-scope brain, embeddings,
  ambient context, redaction, `kimetsu doctor`, SecretString tokens,
  automated crates.io publish (v0.4.1–v0.4.11).
- **v0.5.x** — outcome learning: citations + blame (v0.5.0), decay
  (v0.5.1), conflicts (v0.5.2); harbor refactor (v0.5.3–v0.5.5).

Per-version detail: see [kimetsu CHANGELOG](../CHANGELOG.md).
