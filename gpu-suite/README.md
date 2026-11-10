# Kimetsu memory benchmark on DESKTOP-R97ML0H

All bulky files are on `D:\kimetsu-bench` (`/mnt/d/kimetsu-bench` in WSL). C: holds only tiny transfer scripts; CUDA compiler/runtime packages are installed inside WSL. No Windows driver, firewall, router, or system setting was changed.

## Launch only when Rodrigo asks

From PowerShell:

```powershell
wsl -e bash /mnt/d/kimetsu-bench/launch_full.sh --model 0.5b
# Or, separately (never concurrently):
wsl -e bash /mnt/d/kimetsu-bench/launch_full.sh --model 7b
```

The full runs are staged, not started. A GPU watchdog (`gpu_watchdog.sh`) logs temperature every 10 s to `runs/<model>/gpu-*.csv` and stops the run if the GPU passes 85 C or nvidia-smi fails. The launcher holds a GPU lock, binds llama-server only to 127.0.0.1, and stops its own server on exit. It resumes saved per-question predictions/contexts. Logs/results are under `runs/0.5b` and `runs/7b`; each launch also writes a dated log. D: has a 20 GiB safety floor during LongMemEval preparation and hybrid processing. Inspect errors before claiming completion; a finished launcher is not proof all optional arms succeeded.

## Conditions

LoCoMo: all 1,986 questions, official category-aware scorer (overall includes adversarial abstention, not pure F1). No memory, single lexical, directory sync, split-isolated/merged, real signed-pack iroh QUIC, truncated full-history reference, hybrid BGE embeddings. Supervised cite experiment deliberately excluded.

LongMemEval: all 500 oracle and 500 s-cleaned questions. Same memory/network conditions plus truncated-history reference and hybrid. Scores are normalized token-F1 proxies, NOT the official LLM judge. Supersede is unavailable through MCP and is reported as blocked, never emulated. Strict signed-pack export can reject private-looking strings in public data; rejected rows remain exclusions/errors, not invented successes.

One physical Windows PC under WSL. Every network arm is loopback, NOT LAN/WAN. Qwen2.5 Instruct Q4_K_M, llama.cpp b12189 CUDA sm_120, context 8192, 4 CPU inference/batch threads, 1 slot, GPU layers 99, temperature 0, seed 42, max output 96, cache_prompt=false, `-lm none` (mmap over the D: drive made the GPU run at CPU speed; weights are now read into RAM, then offloaded). Hybrid uses bge-small-en-v1.5 float32 ONNX (384 dimensions), one ONNX thread.

## Calibration

`timing50/summary.json`, `timing50/predictions-*.jsonl`, and `timing50/logs/` contain exactly the first 50 oracle questions: 0.5B no-memory + single lexical, and 7B single lexical. Brain preparation and inference are separate. This is one planning calibration per condition, not a repeatability/performance study. Token throughput includes request/prefill unless explicitly labeled server decode throughput. The CPU reference uses the same 50 IDs; GPU numerics and random memory IDs may change answers/context ranking.

## Provenance and storage

`setup/download-sha256.txt`, `setup/llama-commit.txt`, `setup/kimetsu-commit.txt`, `setup/binary-sha256.txt`, `setup/python-freeze.txt`; original harness archive under the C: staging folder and extracted under `harness/`. Per-model `STAGED.json` records operational adaptations. No production/shared brain is accessed; all benchmark brains have the user brain disabled.

WSL exposes approximately 15 GiB RAM despite 32 GiB physically installed. Memory limits were not changed; long-history hybrid reindex may still require care. Stop/resume errors rather than silently relaxing validation.

## Calibration results, 2026-11-09 (driver 621.14, after the 00:22 nvlddmkm incident)

`timing50/summary.json`. First 50 LongMemEval oracle questions, one pass each, GPU (RTX 5060 Ti), no nvlddmkm errors during the run, max GPU temp 62 C.

| Condition | n | mean s/question (inference only) | proxy F1 % | box CPU F1 % (same 50) | exact answer match vs box |
|---|---|---|---|---|---|
| Qwen 0.5B no-memory | 50 | 0.067 | 19.25 | 19.21 | 43/50 |
| Qwen 0.5B single lexical | 50 | 0.079 | 26.08 | 24.00 | 45/50 |
| Qwen 7B single lexical | 50 | 0.359 | 24.44 | (box has 0.5B only) | 1/50 |

Brain preparation on D: 1,583 s for 2,214 record calls = 0.715 s/record (per question 0.45-1.33 s/record), retrieval 0.54 s/question. One-question I/O probe (60 records): tmpfs 0.037 s/record, WSL ext4 0.51 s/record, D: 1.22 s/record. Preparation is bound by synchronous disk writes, not CPU or GPU.

### Full-suite estimate per model (lower bounds)

The suite makes about 1.19 M fresh record calls (3 passes over 398,252 raw chunks: lexical A, split halves, hybrid brain). At 0.715 s/record on D: that is about 237 h (~10 days) per model, before directory/QUIC imports, hybrid ONNX reindex and full-context prefill, which were not timed. LongMemEval s-cleaned is 94% of it (374,067 chunks, ~223 h). LoCoMo + LongMemEval oracle alone is ~14 h of record calls on D:. Inference is small: all 23,888 rows ~0.5 h (0.5B) or ~2.4 h (7B) at the measured single-lexical rate. Each model directory prepares its own brains, so running both models doubles the preparation.

## Full suite with RAM scratch (launched 2026-11-09 14:25 ART)

Approved by Rodrigo: "Yes, go with RAM scratch and save each question to D: as it finishes."

- Code: `suite.py`, `launch_full.sh`, `gpu_watchdog.sh`, `status.sh` (masters in `C:\Users\rodri\kimetsu-bench\suite\`, copies here). The old staged launcher is in `old-staging/`; the old `runs/` dirs are no longer used except `runs/0.5b/full-locomo/full-dataset.json` and `common.py` (official LoCoMo scorer).
- Each unit (LoCoMo conversation or LongMemEval question) is prepared in `/dev/shm/kimetsu-scratch/suite/`. When finished, the whole scratch tree (brains + operation log) is written to `suite/shared/brains/<dataset>/<id>.tar` (temp name, fsync, rename), then `suite/shared/contexts/<dataset>/<id>.json` (atomic; the completion marker), then scratch is deleted. A crash loses at most the in-progress units; stale scratch and `*.tmp` are cleaned on start.
- Retrieval is model-independent, so 0.5B and 7B both read the same contexts (no second preparation). Both llama-servers run at once (ports 18082/18083, `-lm none`, 900 s startup wait).
- Predictions: `suite/runs/<model>/predictions.jsonl`, appended and fsynced per row; restart skips done (dataset, id, condition).
- Hybrid runs after preparation, one unit at a time, only when MemAvailable >= 11 GiB, with the reindex process marked OOM-first (oom_score_adj 1000).
- Watchdog: stops everything and writes `suite/STOP` if GPU > 85 C, nvidia-smi fails, or a new nvlddmkm System event appears (checked every 60 s).

Status: `wsl -e bash /mnt/d/kimetsu-bench/status.sh`
Resume after a crash/reboot (same command; refuses if already running or if STOP exists):
`wsl -e bash /mnt/c/Users/rodri/kimetsu-bench/suite/run_full.sh`


## Hybrid paused 2026-11-09 20:22 ART (America/Buenos_Aires)

RodCor stopped the hybrid embeddings stage on purpose. Those runs measure the **pre-#67 buggy build**; hybrid will be rerun after the **#68** fixes.

- Stopped only: `suite.py hybrid` (and its `kimetsu-embeddings` reindex child). Infer 0.5B/7B, llama-servers, prep outputs, and the GPU watchdog were left running.
- Partial hybrid outputs kept intact as a **'before the fix, partial n' baseline** (`suite/shared/hybrid-contexts/`, `suite/shared/brains-hybrid/`, hybrid rows already in `suite/runs/*/predictions.jsonl`). Do not delete them.
- Marker: `suite/SKIP_HYBRID`. While this file exists, `suite.py hybrid` no-ops (also via `SKIP_HYBRID=1`), and `launch_full.sh` / `run_full.sh` export that env so a later resume **will not continue hybrid**.
- `suite/shared/hybrid-complete.json` was written with `skipped` so the infer processes can finish remaining non-hybrid work (e.g. signed-pack QUIC) without waiting forever on hybrid.
- **Do not remove `suite/SKIP_HYBRID` until #68 lands.** After #68: delete the marker, delete or rename `suite/shared/hybrid-complete.json` if you need hybrid to run again, then `wsl -e bash /mnt/c/Users/rodri/kimetsu-bench/suite/run_full.sh` (or `launch_full.sh`). Completed prediction keys are still skipped on resume.
