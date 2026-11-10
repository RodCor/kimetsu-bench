# Full LoCoMo × Kimetsu — checkpoint

Updated 2026-11-09T13:18:33.388398-03:00 America/Buenos_Aires (UTC−3). **RUN IN PROGRESS unless all applicable conditions explicitly say complete.**

10 conversations, 1986 released QA questions, including all 446 category-5 adversarial questions. No sampling. Values below use only completed predictions; partial prefixes are not full-dataset results and must not be compared unless matched.

| Condition | n / 1986 | Overall official score % | F1 cats1–4 only | C1 | C2 | C3 | C4 | C5 | Retrieval median ms | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| No memory | 1986 | 19.12 | 2.12 | 2.22 | 0.52 | 7.46 | 2.08 | 77.80 | — | complete |
| Single MCP lexical | 1986 | 17.65 | 22.05 | 12.57 | 27.83 | 12.92 | 24.06 | 2.47 | 1197.15 | complete |
| A→B directory sync | 1986 | 18.19 | 22.87 | 13.84 | 29.49 | 11.27 | 24.69 | 2.02 | 833.54 | complete |
| B isolated half-sessions | 1986 | 13.63 | 16.21 | 12.15 | 25.98 | 13.02 | 14.20 | 4.71 | 1197.38 | complete |
| B merged halves / directory | 1986 | 17.55 | 21.92 | 12.63 | 28.49 | 13.25 | 23.52 | 2.47 | 1221.91 | complete |
| Signed pack / iroh QUIC loopback | 1986 | 17.90 | 22.57 | 13.60 | 28.77 | 12.29 | 24.38 | 1.79 | 1166.95 | complete |
| Full history, tail-truncated | 94 | 11.27 | 11.27 | 9.30 | 12.43 | 13.93 | 10.07 | — | — | queued after priority conditions |
| Single MCP hybrid embeddings | 0 | — | — | — | — | — | — | — | — | unavailable: embeddings feature build failed; exact compiler/dependency error in logs/embeddings-build-final.log |
| Cite helped, supervised rerun (exploratory) | 0 | — | — | — | — | — | — | — | — | contexts ready; supervised test-label feedback, not generalization |

## Metric and method

Official snap-research/locomo category-aware `eval_question_answering` scorer, same frozen source/hash as pilot. Only the unused top-level bert_score import is removed via AST; scorer logic unchanged. C1 multi-answer F1; C2 temporal F1; C3 open-domain F1 with official semicolon rule; C4 single-hop F1; **C5 official binary abstention rule, not token F1**. Overall is the official mean across all five categories, thus a mixed category-aware score, not pure token F1 or accuracy. The separate cats1–4 F1 column excludes the 446 adversarial items and is comparable in metric (not sample composition) to the pilot. Category-specific n are in summary.json. For category 5 the absent answer field is an empty placeholder ignored by upstream scoring.

Same Qwen2.5-0.5B-Instruct Q4_K_M, llama.cpp b12189, CPU four inference threads and four batch threads, context 8192, temperature 0, seed 42, 96 output tokens, cache_prompt=false; same prompt as pilot. Dedicated loopback port 18082. Baseline never gets history. MCP conditions get only returned capsules, max 8, budget_tokens=16000 (UTF-8 byte upper-bound units), min_score/min_lexical_coverage/abstain_evidence=0; embeddings explicitly off for lexical arms. Raw dated turns/image captions chunked ≤1200 characters, no answer labels, generated summaries or evidence supplied to the brain. Gold/evidence labels are used only for scoring, except the explicitly supervised exploratory cite arm.

All workspaces are isolated below full-locomo/workspaces, user brain disabled; /home/box/kimetsu-team is never opened. Directory arms use actual memory-event sync, not SQLite copies. B starts empty and A/B active text equality is asserted. Split arms divide whole sessions by alternating session number; B's half is tested before importing A's other half. This tests complementary knowledge, not simply replication. Signed-pack arm must use real kimetsu-sdk request_pack, admin-signed roster, BLAKE3+Ed25519 verification, iroh QUIC bound to 127.0.0.1 and relay disabled. **Every transport is one machine, not LAN.**

Full-context reference takes latest history tokens (up to 7000) via the model tokenizer, leaving room for the fixed system/question and completion; actual prompt tokens logged.

One greedy QA pass per condition. This is one tiny quantized model, not a result about frontier models or general agent quality. Public-data pretraining contamination cannot be ruled out; within-conversation question dependence is not addressed by this single run. Retrieval median describes the calls producing those contexts, no speedup claim; not a five-run perf benchmark. Two public-benchmark workers share the 8-vCPU host, with an occasional Rust build; contention makes latency comparisons inconclusive. Checkpoints are flushed per question; interrupted/resumed runs skip completed IDs, no fabricated scores. Full-context and optional arms are deliberately scheduled after priority arms 1,3,5,7,6.

## Artifacts

summary.json; scores.png; logs/predictions.jsonl; logs/operations.jsonl; logs/preparation.jsonl; logs/errors.jsonl; logs/llama-server.log; logs/prepare.log; logs/run.log; logs/embeddings-build.log; logs/rust-install.log; full-dataset.json; contexts/ (retrieval payloads); workspaces/ (auditable local brains); conditions.json. Pilot manifest.json pins model, dataset and scorer. Missing optional-arm reasons live in conditions.json and logs.

## Paired diagnostics (matched IDs only)

- single minus baseline: -1.46 points, 1986 matched questions (not a significance claim).

- directory minus single: +0.53 points, 1986 matched questions (not a significance claim).

- split_merged minus split_isolated: +3.93 points, 1986 matched questions (not a significance claim).

- quic minus single: +0.25 points, 1986 matched questions (not a significance claim).
