# Network-condition memory benchmark (LoCoMo + LongMemEval)

Harness used to measure Kimetsu memory conditions on one machine (Qwen2.5-0.5B-Instruct Q4_K_M, llama.cpp b12189, CPU, ctx 8192, temp 0, seed 42, 96 output tokens):
no-memory, single lexical MCP, directory A→B sync, split-B alone vs merged, signed-pack QUIC (iroh, **single-machine loopback**), tail-truncated full-history reference.

- `locomo/` : official category-aware LoCoMo scorer (`common.py`, `prepare.py`, `infer.py`, `verify.py`).
- `longmemeval/` : oracle + s-cleaned; scores are a **proxy token F1, not the official LLM judge** (`run_full.py`, `prepare_only.py`, `infer_only.py`, `quic_extension.py`, `report_full.py`).
- `commands.txt` : launch/resume commands.
- Paths in the scripts are those of the original box (`/workspace/kimetsu-memory-bench`); adjust before reuse. No datasets, models, brains or secrets are included: fetch LoCoMo / LongMemEval and the GGUF yourself.
- Hybrid-embedding arms are pre-#67 and excluded from published numbers.

Final results snapshots are under `../results/` (LongMemEval at completion; the LoCoMo RESULTS.md is the last checkpoint before the mandatory arms were verified, 1986/1986 for the six core arms).
