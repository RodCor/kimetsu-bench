# Compare two Kimetsu builds without a reader model

Build the current `kbench` once and use that identical harness for both binaries:

```powershell
cargo build --release --bin kbench --locked --offline
python scripts/compare_brainbench.py --kbench target/release/kbench.exe --baseline path/to/baseline/kimetsu.exe --candidate path/to/candidate/kimetsu.exe --dataset datasets/brainbench/agent-memory-contract.json --budget-tokens 512 --repeats 3 --out local/paired-memory
```

Use an existing shared `FASTEMBED_CACHE_DIR` to avoid model downloads. The binaries must support embeddings. The runner disables the user brain and permits only non-generative dimensions. Every scenario uses a temporary isolated project. It never uses your live brain.

The runner alternates baseline/candidate order, fixes one scenario worker, retains every report and stderr log, and records SHA-256 fingerprints for binaries, the harness, the dataset and referenced fixture files. It does not force identical model defaults: a binary-default comparison intentionally includes changes to those defaults. Pin relevant model/environment settings for an algorithm-only experiment; recorded overrides make that distinction reviewable.

Each scenario is paired by dimension and ID. Repeats are averaged within a scenario, not counted as additional independent cases. A scenario with a skipped or errored observation on either side is reported as unpaired and excluded from quality deltas; execution failures are not scores. The exploratory confidence interval bootstraps scenario IDs. Related scenarios are still correlated: a release claim requires held-out task/repository families and real task-success measurements. Changed/duplicate scenario identities are errors.

`comparison.json` is written as the run progresses. A command failure, timeout, or invalid JSON response leaves it with `status: "incomplete"`, structured failure evidence, and all completed run records; the process exits nonzero and does not write a completed Markdown comparison.

BrainBench's headline now weights measured dimensions equally; the old scenario-weighted average remains a diagnostic. No-answer queries score abstention, not the vacuous recall of an empty relevant set. Positive recall and negative injection rates use separate denominators, and stale correctness is reported as unavailable when no stale cases exist. Unmatched or ambiguous returned capsules retain their rank and count as injected material.

The small checked-in fixture is a regression and exploratory language track, not a comprehensive held-out benchmark. Full-run timings include model/process startup, seeding, and queries; do not call them warm inference latency. Final host token consumption and complete-agent success require the corresponding serving and agent evaluations.
