# kimetsu-bench — single workflow entry point.
#
# Thin wrapper over the `kbench` binary (it already auto-discovers auth, the
# Linux kimetsu binary, and the brain workspace, then writes a report to
# local/runs/auto/<timestamp>.md). This Makefile only adds: build, PATH setup for
# WSL2's harbor, command dispatch, the per-family sweep loop, and housekeeping.
#
# Run real benchmarks from WSL2 on Windows (see README "Windows hosts MUST run
# from WSL2"). `make dry ...` works from any shell since it never touches Docker.
#
#   make bench TASK=fix-git              # claude+km vs claude, one task
#   make bench TASK=fix-git,git-leak     # several tasks (comma-separated)
#   make family FAM=python               # every task in a family
#   make sweep                           # all families, sequentially
#   make full                            # the whole Terminal-Bench dataset
#   make dry TASK=fix-git                # synthetic, no Docker/Harbor
#   make cost                            # token-cost report from /tmp/kbench-*
#   make clean-cache                     # drop .cache scratch (keep heavy caches)
#   make prune-runs                      # drop per-trial run dirs (keep local/runs/auto)
#
# Variables (override on the command line):
#   TASK=...         task id(s), comma-separated      (bench/dry)
#   FAM=...          family name                       (family)
#   AGENTS=...       explicit agent list, e.g. claude+km,codex   (overrides default)
#   MODEL=...        model name forwarded to harbor (required for codex)
#   NB=1             pass --no-build (use cached Linux binary)
#   KBIN=/path       override the auto-resolved Linux kimetsu binary
#   HARBOR_ARGS=...  one extra arg forwarded to `harbor run`
#   OUTPUT=json      report format (default: markdown)

SHELL := /bin/bash
BENCH := $(CURDIR)
KBENCH := $(BENCH)/target/release/kbench

# Keep ALL benchmark writes on the bench drive (E:), never the WSL/Docker
# vhdx on C:. Harbor caches downloaded datasets under ~/.cache/harbor (which
# lives on the C: vhdx); `setup-cache` symlinks it onto the bench drive. The
# per-trial artifacts already go to E: via kbench's absolute --run-dir.
HARBOR_CACHE_SRC := $(HOME)/.cache/harbor
HARBOR_CACHE_DST := $(BENCH)/.cache/harbor

# --- kstress: brain stress test (local v1.0.0) -----------------------------
# Built WITH embeddings so ONE binary runs BOTH matrices (lean via the noop
# embedder, emb via fastembed/vec0). The embeddings build links the ONNX
# runtime, which needs glibc >= 2.38. We AUTO-DETECT: on glibc >= 2.38 we build
# both matrices; on older glibc (e.g. Ubuntu 20.04, glibc 2.31) we fall back to
# a LEAN-ONLY build so `make stress` still works — for the emb matrix, run from
# a recent distro (e.g. Ubuntu-24.04 on WSL2).
KSTRESS         := $(BENCH)/target/release/kstress
REMOTE_BIN      := $(BENCH)/../target/release/kimetsu-remote
STRESS_MATRICES ?= lean emb
STRESS_SCALES   ?= 100,500,5000,50000,500000,1000000
STRESS_MAX_LEAN ?= 1000000
STRESS_MAX_EMB  ?= 1000000
STRESS_REMOTE_SCALES ?= 100,500,5000,50000,500000,1000000
STRESS_CONCURRENCY   ?= 1,4,16,64
# Working-brain location. Defaults to $(HOME)/.cache/kstress (native ext4) so
# `make stress` gets realistic I/O numbers out of the box (~12-33x faster than
# the 9p drvfs E: mount at high scales). Override with STRESS_WORK=<path> to
# point at any directory, or STRESS_WORK= (empty) to fall back to .cache/stress
# on E: (honors no-C:, fine to ~50k rows). Reports always go to local/runs/stress on
# E: regardless of STRESS_WORK. kstress deletes the working brains after each
# run; pass --keep-work to retain them.
STRESS_WORK     ?= $(HOME)/.cache/kstress
STRESS_WORK_FLAG := $(if $(STRESS_WORK),--work $(STRESS_WORK),)

# glibc gate for the embeddings (ONNX) link. EMB_OK=yes when glibc >= 2.38.
STRESS_GLIBC  := $(shell ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$$')
STRESS_EMB_OK := $(shell printf '2.38\n%s\n' "$(STRESS_GLIBC)" | sort -V -C >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(STRESS_EMB_OK),yes)
  STRESS_FEATURES      := --features embeddings
  STRESS_MATRICES_EFF  := $(STRESS_MATRICES)
else
  STRESS_FEATURES      :=
  STRESS_MATRICES_EFF  := $(filter-out emb,$(STRESS_MATRICES))
endif

# WSL2 installs harbor under /root/.local/bin and cargo under /root/.cargo/bin;
# putting them on PATH lets `make` work without the caller pre-exporting them.
export PATH := /root/.local/bin:/root/.cargo/bin:$(PATH)

# Optional-flag plumbing: empty vars expand to nothing.
AGENTS_FLAG  := $(if $(AGENTS),--agents $(AGENTS),)
MODEL_FLAG   := $(if $(MODEL),--model $(MODEL),)
NB_FLAG      := $(if $(NB),--no-build,)
KBIN_FLAG    := $(if $(KBIN),--kimetsu-binary $(KBIN),)
HARBOR_FLAG  := $(if $(HARBOR_ARGS),--harbor-arg "$(HARBOR_ARGS)",)
OUTPUT       ?= markdown
COMMON_FLAGS := $(AGENTS_FLAG) $(MODEL_FLAG) $(NB_FLAG) $(KBIN_FLAG) $(HARBOR_FLAG) --output $(OUTPUT)

.DEFAULT_GOAL := help

.PHONY: help build bench family sweep full dry cost list-families clean-cache prune-runs clean setup-cache \
        stress stress-build stress-local stress-remote stress-smoke

help: ## Show this help.
	@echo "kimetsu-bench workflow — targets:"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | sed -E 's/^([a-zA-Z_-]+):.*## /  \1\t/' \
	  | sort | column -t -s $$'\t'
	@echo ""
	@echo "Examples:"
	@echo "  make bench TASK=fix-git"
	@echo "  make family FAM=python"
	@echo "  make sweep"
	@echo "  make dry TASK=fix-git           # no Docker"
	@echo ""
	@echo "Vars: TASK FAM AGENTS MODEL NB KBIN HARBOR_ARGS OUTPUT (see Makefile header)"

build: ## Build the kbench binary (release).
	cargo build --release

setup-cache: ## Symlink harbor's dataset cache (~/.cache/harbor) onto the bench drive (E:), off the C: vhdx. One-time; idempotent.
	@mkdir -p $(HARBOR_CACHE_DST) $(HOME)/.cache
	@if [ -L "$(HARBOR_CACHE_SRC)" ]; then \
	  :; \
	elif [ -e "$(HARBOR_CACHE_SRC)" ]; then \
	  echo "migrating $(HARBOR_CACHE_SRC) -> $(HARBOR_CACHE_DST) ..."; \
	  cp -a "$(HARBOR_CACHE_SRC)/." "$(HARBOR_CACHE_DST)/" 2>/dev/null || true; \
	  rm -rf "$(HARBOR_CACHE_SRC)"; \
	  ln -s "$(HARBOR_CACHE_DST)" "$(HARBOR_CACHE_SRC)"; \
	  echo "linked $(HARBOR_CACHE_SRC) -> $(HARBOR_CACHE_DST)"; \
	else \
	  ln -s "$(HARBOR_CACHE_DST)" "$(HARBOR_CACHE_SRC)"; \
	  echo "linked $(HARBOR_CACHE_SRC) -> $(HARBOR_CACHE_DST)"; \
	fi

bench: build setup-cache ## Run one or more tasks (TASK=a,b). Default agents: claude+km vs claude.
	@test -n "$(TASK)" || { echo "ERROR: set TASK=<task-id[,task-id...]>"; exit 2; }
	$(KBENCH) $(TASK) $(COMMON_FLAGS)

family: build setup-cache ## Run every task in a family (FAM=python). See `make list-families`.
	@test -n "$(FAM)" || { echo "ERROR: set FAM=<family>  (try: make list-families)"; exit 2; }
	$(KBENCH) --family $(FAM) $(COMMON_FLAGS)

sweep: build setup-cache ## Run all families sequentially (concurrent runs crash Docker). Logs to .cache/sweep-logs/.
	@mkdir -p $(BENCH)/.cache/sweep-logs
	@echo "=== SWEEP STARTED $$(date -u) ==="
	@for f in $$($(KBENCH) --list-families | awk '$$3=="tasks"{print $$1}'); do \
	  echo ""; echo "=== FAMILY $$f  $$(date -u) ==="; \
	  $(KBENCH) --family $$f --no-build $(AGENTS_FLAG) $(MODEL_FLAG) $(KBIN_FLAG) $(HARBOR_FLAG) \
	    > $(BENCH)/.cache/sweep-logs/$$f.log 2>&1; \
	  echo "family $$f finished rc=$$? -> .cache/sweep-logs/$$f.log"; \
	  grep -E "Summary|wins|per-run error|report saved" $(BENCH)/.cache/sweep-logs/$$f.log | tail -8 || true; \
	done
	@echo ""; echo "=== SWEEP COMPLETE $$(date -u) ==="
	@echo "per-family logs in .cache/sweep-logs ; reports in local/runs/auto"

full: build setup-cache ## Run the whole downloaded Terminal-Bench dataset.
	$(KBENCH) --full-dataset $(COMMON_FLAGS)

dry: build ## Synthetic run — no Docker/Harbor/auth (TASK=fix-git). Works from any shell.
	@test -n "$(TASK)" || { echo "ERROR: set TASK=<task-id[,task-id...]>"; exit 2; }
	$(KBENCH) --dry-run $(TASK) --output $(OUTPUT)

cost: ## Token-usage -> cost report from /tmp/kbench-* (families read from the manifest).
	@bash $(BENCH)/scripts/cost.sh

list-families: build ## List the programming-language families and task counts.
	@$(KBENCH) --list-families

clean-cache: ## Delete .cache scratch (logs/scratch/regenerable); keep linux-build*, brain-workspace*, warm-*.
	@if [ -d $(BENCH)/.cache ]; then \
	  find $(BENCH)/.cache -maxdepth 1 -type f -delete; \
	  rm -rf $(BENCH)/.cache/report $(BENCH)/.cache/sweep-logs $(BENCH)/.cache/worker-results; \
	  echo "cleaned .cache scratch; kept:"; \
	  ls -1 $(BENCH)/.cache 2>/dev/null | sed 's/^/  /'; \
	else echo "no .cache dir"; fi

prune-runs: ## Delete per-trial run dirs under local/runs/, keep local/runs/auto + local/runs/stress reports.
	@if [ -d $(BENCH)/local/runs ]; then \
	  find $(BENCH)/local/runs -maxdepth 1 -mindepth 1 -type d ! -name auto ! -name stress -exec rm -rf {} +; \
	  echo "pruned per-trial run dirs; kept local/runs/auto + local/runs/stress"; \
	else echo "no local/runs dir"; fi

stress-build: ## Build kstress + kimetsu-remote from local v1.0.0 source. Embeddings auto-enabled on glibc>=2.38; else lean-only.
ifeq ($(STRESS_EMB_OK),yes)
	@echo "glibc $(STRESS_GLIBC) >= 2.38 — building WITH embeddings (lean + emb matrices)"
else
	@echo "WARNING: glibc '$(STRESS_GLIBC)' < 2.38 — ONNX runtime won't link here."
	@echo "         Building LEAN-ONLY (FTS matrix). For the embeddings matrix, run"
	@echo "         these targets from a glibc>=2.38 distro (e.g. Ubuntu-24.04 on WSL2)."
endif
	cargo build --release --bin kstress $(STRESS_FEATURES)
	cargo build --release --manifest-path ../Cargo.toml -p kimetsu-remote $(STRESS_FEATURES)

stress-local: stress-build ## Local sweep (insert/query/size/concurrency) per matrix. Vars: STRESS_SCALES, STRESS_MAX_*, STRESS_MATRICES.
	@ts=$$(date -u +%Y-%m-%dT%H-%M-%SZ); \
	for m in $(STRESS_MATRICES_EFF); do \
	  max=$(STRESS_MAX_LEAN); [ "$$m" = "emb" ] && max=$(STRESS_MAX_EMB); \
	  echo "=== local/$$m (max $$max) ==="; \
	  $(KSTRESS) local --matrix $$m --scales $(STRESS_SCALES) --max-scale $$max $(STRESS_WORK_FLAG) --run-id $$ts || true; \
	done; \
	echo "reports in local/runs/stress/$$ts (matrices: $(STRESS_MATRICES_EFF))"

stress-remote: stress-build ## Remote HTTP sweep (throughput/rate-limit/isolation) per matrix. Vars: STRESS_REMOTE_SCALES, STRESS_CONCURRENCY.
	@ts=$$(date -u +%Y-%m-%dT%H-%M-%SZ); \
	for m in $(STRESS_MATRICES_EFF); do \
	  echo "=== remote/$$m ==="; \
	  $(KSTRESS) remote --matrix $$m --scales $(STRESS_REMOTE_SCALES) --concurrency $(STRESS_CONCURRENCY) \
	    --remote-bin $(REMOTE_BIN) $(STRESS_WORK_FLAG) --run-id $$ts || true; \
	done; \
	echo "reports in local/runs/stress/$$ts (matrices: $(STRESS_MATRICES_EFF))"

stress: stress-build ## Full stress test: local + remote × available matrices, one run dir.
	@ts=$$(date -u +%Y-%m-%dT%H-%M-%SZ); \
	for m in $(STRESS_MATRICES_EFF); do \
	  max=$(STRESS_MAX_LEAN); [ "$$m" = "emb" ] && max=$(STRESS_MAX_EMB); \
	  echo "=== local/$$m (max $$max) ==="; \
	  $(KSTRESS) local --matrix $$m --scales $(STRESS_SCALES) --max-scale $$max $(STRESS_WORK_FLAG) --run-id $$ts || true; \
	  echo "=== remote/$$m ==="; \
	  $(KSTRESS) remote --matrix $$m --scales $(STRESS_REMOTE_SCALES) --concurrency $(STRESS_CONCURRENCY) \
	    --remote-bin $(REMOTE_BIN) $(STRESS_WORK_FLAG) --run-id $$ts || true; \
	done; \
	echo "reports in local/runs/stress/$$ts (matrices: $(STRESS_MATRICES_EFF))"

stress-smoke: stress-build ## Fast wiring check: scales 100,500 (local + remote), available matrices.
	@ts=smoke-$$(date -u +%H-%M-%SZ); \
	for m in $(STRESS_MATRICES_EFF); do \
	  $(KSTRESS) local  --matrix $$m --scales 100,500 --readers 1,4 --window-ms 500 $(STRESS_WORK_FLAG) --run-id $$ts || true; \
	  $(KSTRESS) remote --matrix $$m --scales 500 --concurrency 1,8 --window-ms 800 \
	    --remote-bin $(REMOTE_BIN) $(STRESS_WORK_FLAG) --run-id $$ts || true; \
	done; \
	echo "reports in local/runs/stress/$$ts (matrices: $(STRESS_MATRICES_EFF))"

clean: ## cargo clean (drop target/).
	cargo clean
