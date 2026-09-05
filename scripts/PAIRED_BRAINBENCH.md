# Compare two Kimetsu builds without a reader model

Build the current `kbench` once and use that identical harness for both binaries:

```powershell
cargo build --release --bin kbench --locked --offline
python scripts/compare_brainbench.py --kbench target/release/kbench.exe --baseline path/to/baseline/kimetsu.exe --candidate path/to/candidate/kimetsu.exe --dataset datasets/brainbench/agent-memory-contract.json --budget-tokens 512 --repeats 3 --out local/paired-memory
```

Use an existing shared `FASTEMBED_CACHE_DIR` to avoid model downloads. The binaries must support embeddings. The runner disables the user brain and permits only non-generative dimensions. Every scenario uses a temporary isolated project. It never uses your live brain.

The runner alternates baseline/candidate order, fixes one scenario worker, retains every report and stderr log, and records SHA-256 fingerprints for binaries, the harness, the dataset and referenced fixture files. It does not force identical model defaults: a binary-default comparison intentionally includes changes to those defaults. Pin relevant model/environment settings for an algorithm-only experiment; recorded overrides make that distinction reviewable.

Each scenario is paired by dimension and ID. Repeats are averaged within a scenario, not counted as additional independent cases. A scenario with a skipped or errored observation on either side is reported as unpaired and excluded from quality deltas; execution failures are not scores. The exploratory confidence interval bootstraps scenario IDs. Related scenarios are still correlated: a release claim requires held-out task/repository families and real task-success measurements. Changed/duplicate scenario identities are errors.

`comparison.json` is written as the run progresses. A command failure, timeout, invalid JSON/report response, or final identity-validation failure leaves it with `status: "incomplete"`, structured failure evidence, and all completed run records; the process exits nonzero and does not write a completed Markdown comparison.

The whole-run timeout terminates descendants before reaping `kbench` (Windows `taskkill /T`, Unix process group). This prevents a hung MCP inference child from surviving a timed-out comparison and competing with the next run. The deadline covers seeding and queries together; it is not a per-query latency cutoff.

BrainBench's headline now weights measured dimensions equally; the old scenario-weighted average remains a diagnostic. No-answer queries score abstention, not the vacuous recall of an empty relevant set. Positive recall and negative injection rates use separate denominators, and stale correctness is reported as unavailable when no stale cases exist. Unmatched or ambiguous returned capsules retain their rank and count as injected material.

Retrieval and workflow scenarios query the production `kimetsu_brain_context` tool through a persistent stdio MCP process. Workflow writes happen through a separate process while MCP stays alive, exercising index freshness. The render-contract dimension retains its explicit CLI rendering check. Invalid MCP responses fail the scenario instead of masquerading as abstention. Current-context retrieval scores become zero if any explicitly stale gold item is delivered in the top four, even below the correct answer; the older ordering-only resolution metric remains diagnostic.

Query observations retain hit@4, fraction recall@4, MRR, negative injection and stale injection separately. Repeats are averaged within the same query for quality summaries. Timing reports distinguish the first query in each process from subsequent queries. First-query timing is not a disk-cache-cold measurement; process startup is recorded separately. Subsequent p50/p95 are descriptive pooled measurements, not independent trials or a confidence interval. Full-run timings also include seeding and process startup.

Response sizes include UTF-8 model text, the serialized MCP result, and the full JSON-RPC response line. `reported_used_tokens` is retained as a diagnostic: old heuristic estimates and new conservative byte bounds are not directly comparable. These byte measurements are not provider tokenizer counts or billed-token measurements.

For runtime experiments use `--baseline-threads 0 --candidate-threads 4` with the same binary on both sides (`0` removes `KIMETSU_INTRA_THREADS`; omission inherits it). `KBENCH_RERANKER` sets `embedder.reranker` only inside temporary benchmark projects and is recorded; the binary must actually honor that setting on MCP for a model comparison to be valid. Keep other settings fixed and avoid concurrent builds or other inference during timing runs.

The small checked-in fixture is a regression and exploratory language track, not a comprehensive held-out benchmark. It includes temporal validity fields and persistent-process write visibility. Final host token consumption and complete-agent success require the corresponding serving and agent evaluations.
