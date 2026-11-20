# LongMemEval × Kimetsu — live full-run checkpoint

Updated 2026-11-10T14:04:32.404175-03:00 ART (America/Buenos_Aires, UTC−3). **Not complete unless n equals expected.** All numbers are proxy metrics, NOT official LongMemEval accuracy. Oracle has 500 questions (470 answerable, 30 abstention). The s-cleaned variant is the real long-haystack retrieval test; oracle is evidence-session-only.

## oracle

| Condition | n / expected | Proxy F1 % | Exact % | Contains % | Retrieval median ms | Actual inference median ms | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| no-memory | 500/500 | 5.73 | 0.60 | 3.40 | 0.00 | 752.89 | complete |
| single-mcp-lexical | 500/500 | 21.38 | 6.40 | 20.60 | 224.95 | 4087.62 | complete |
| directory-A-to-B | 500/500 | 23.03 | 8.20 | 21.40 | 192.34 | 9377.20 | complete |
| split-B-alone | 500/500 | 14.12 | 4.40 | 13.80 | 56.82 | 5770.24 | complete |
| split-B-merged-directory | 500/500 | 23.36 | 8.20 | 21.20 | 218.33 | 6163.61 | complete |
| supersede-knowledge-update | 0/78 | — | — | — | — | — | Unavailable in installed host edge: tools/list exposes 31 tools but no supersede; CLI brain supersede is unrecognized and brain memory has no supersede. Not substituting invalidate/edit for actual supersede. SDK helper would be needed; no gold-derived update selection was attempted. |
| signed-pack-QUIC-loopback | 488/500 | 22.71 | 7.38 | 21.72 | 228.13 | 8687.84 | Running via sibling-built kimetsu-sdk request_pack helper; real admin-signed roster, BLAKE3/Ed25519 pack verification, iroh QUIC on 127.0.0.1, BusRelay::Disabled. Waits for each directory prediction so core arms have priority. |
| truncated-history | 499/500 | 17.01 | 2.40 | 24.65 | 0.00 | 14843.80 | in progress / queued |
| hybrid-embeddings | 0/500 | — | — | — | — | — | Running as a separate checkpointed arm: see HYBRID_RESULTS.md and summary-hybrid.json; sibling logs/predictions-hybrid.jsonl is intentionally not counted by this main report. Embeddings release build succeeded. |

### By type — proxy F1 % (n)

| Condition | single-session-user | single-session-assistant | multi-session | temporal-reasoning | knowledge-update | single-session-preference | abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-memory | 2.58 (64) | 9.70 (56) | 0.54 (121) | 11.83 (127) | 0.00 (72) | 8.48 (30) | 11.13 (30) |
| single-mcp-lexical | 44.60 (64) | 33.79 (56) | 8.12 (121) | 17.74 (127) | 26.42 (72) | 12.33 (30) | 14.49 (30) |
| directory-A-to-B | 47.11 (64) | 34.54 (56) | 9.88 (121) | 18.58 (127) | 30.52 (72) | 12.17 (30) | 14.96 (30) |
| split-B-alone | 2.58 (64) | 9.70 (56) | 6.69 (121) | 16.54 (127) | 39.48 (72) | 8.48 (30) | 11.46 (30) |
| split-B-merged-directory | 47.11 (64) | 34.54 (56) | 10.14 (121) | 20.75 (127) | 28.29 (72) | 11.86 (30) | 15.84 (30) |
| signed-pack-QUIC-loopback | 47.17 (64) | 35.41 (55) | 9.97 (118) | 16.93 (123) | 29.46 (69) | 13.71 (30) | 14.29 (29) |
| truncated-history | 23.83 (64) | 45.53 (56) | 4.45 (121) | 15.44 (126) | 14.26 (72) | 13.83 (30) | 16.20 (30) |

Abstention F1 compares the response with an explanatory reference and is not a semantic abstention score; fixed-phrase refusal rate is provided separately in summary.json. Original type stratification (including abstention IDs) is also retained.

## s-cleaned

| Condition | n / expected | Proxy F1 % | Exact % | Contains % | Retrieval median ms | Actual inference median ms | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| no-memory | 500/500 | 5.35 | 0.60 | 3.00 | 0.00 | 1379.62 | complete |
| single-mcp-lexical | 500/500 | 20.27 | 7.00 | 17.60 | 2653.53 | 4598.08 | complete |
| directory-A-to-B | 500/500 | 21.82 | 8.00 | 19.40 | 1770.57 | 4873.90 | complete |
| split-B-alone | 499/500 | 14.80 | 4.21 | 12.63 | 2570.93 | 3441.24 | in progress / queued |
| split-B-merged-directory | 499/500 | 21.32 | 8.22 | 18.04 | 2673.97 | 2436.51 | in progress / queued |
| supersede-knowledge-update | 0/78 | — | — | — | — | — | Unavailable in installed host edge: tools/list exposes 31 tools but no supersede; CLI brain supersede is unrecognized and brain memory has no supersede. Not substituting invalidate/edit for actual supersede. SDK helper would be needed; no gold-derived update selection was attempted. |
| signed-pack-QUIC-loopback | 306/500 | 21.89 | 9.48 | 19.61 | 2914.77 | 4714.28 | Running via sibling-built kimetsu-sdk request_pack helper; real admin-signed roster, BLAKE3/Ed25519 pack verification, iroh QUIC on 127.0.0.1, BusRelay::Disabled. Waits for each directory prediction so core arms have priority. |
| truncated-history | 500/500 | 7.22 | 0.20 | 4.80 | 0.00 | 15590.06 | complete |
| hybrid-embeddings | 0/500 | — | — | — | — | — | Running as a separate checkpointed arm: see HYBRID_RESULTS.md and summary-hybrid.json; sibling logs/predictions-hybrid.jsonl is intentionally not counted by this main report. Embeddings release build succeeded. |

### By type — proxy F1 % (n)

| Condition | single-session-user | single-session-assistant | multi-session | temporal-reasoning | knowledge-update | single-session-preference | abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-memory | 0.49 (64) | 9.29 (56) | 0.37 (121) | 11.72 (127) | 0.12 (72) | 8.31 (30) | 11.15 (30) |
| single-mcp-lexical | 36.84 (64) | 21.72 (56) | 7.35 (121) | 22.48 (127) | 30.09 (72) | 10.20 (30) | 11.48 (30) |
| directory-A-to-B | 43.54 (64) | 24.11 (56) | 10.34 (121) | 20.96 (127) | 30.58 (72) | 10.34 (30) | 11.62 (30) |
| split-B-alone | 22.95 (64) | 18.12 (56) | 6.79 (120) | 18.66 (127) | 15.22 (72) | 8.84 (30) | 11.80 (30) |
| split-B-merged-directory | 43.53 (64) | 22.62 (56) | 7.92 (121) | 21.30 (127) | 31.72 (72) | 9.90 (29) | 11.65 (30) |
| signed-pack-QUIC-loopback | 44.34 (40) | 27.29 (28) | 8.89 (79) | 19.56 (77) | 35.46 (43) | 9.25 (19) | 12.59 (20) |
| truncated-history | 1.15 (64) | 10.91 (56) | 1.09 (121) | 12.62 (127) | 4.72 (72) | 10.81 (30) | 17.60 (30) |

Abstention F1 compares the response with an explanatory reference and is not a semantic abstention score; fixed-phrase refusal rate is provided separately in summary.json. Original type stratification (including abstention IDs) is also retained.

## Matched-question comparisons
- oracle: single-mcp-lexical − no-memory: +15.65 F1 points on 500 matched questions; identical contexts 0, identical answers 52.
- oracle: directory-A-to-B − single-mcp-lexical: +1.65 F1 points on 500 matched questions; identical contexts 189, identical answers 325.
- oracle: split-B-merged-directory − split-B-alone: +9.24 F1 points on 500 matched questions; identical contexts 3, identical answers 99.
- s-cleaned: single-mcp-lexical − no-memory: +14.92 F1 points on 500 matched questions; identical contexts 0, identical answers 46.
- s-cleaned: directory-A-to-B − single-mcp-lexical: +1.55 F1 points on 500 matched questions; identical contexts 146, identical answers 283.
- s-cleaned: split-B-merged-directory − split-B-alone: +6.55 F1 points on 498 matched questions; identical contexts 14, identical answers 139.

## Design, interpretation, and blockers

# Full extension, frozen before inference
Claim: compare the same tiny model with no history, timestamped raw-turn lexical MCP retrieval, and genuinely isolated directory replicas; test whether merging complementary writer brains helps the answering bot. Oracle is evidence-session-only, not a long-haystack retrieval benchmark. Full cleaned s variant follows all 500 oracle questions, in dataset order, as runtime allows.
Same pilot Qwen2.5-0.5B-Instruct Q4_K_M, llama.cpp b12189 CPU, 4 threads, 8192 context, temperature 0, seed 42, max_tokens 96, exact pilot system and user prompt. Every raw turn split at 1200 characters, MCP context budget 16000/max 8, embedder off. Truncated-history reference takes last 20000 characters (not full-context inference). No gold answers/questions are stored in brains. Gold labels used only after predictions.
All brains isolated below this directory. No user brain or production team brain. Full writer A to initially empty B uses actual event export/replay with text equality assertions. Complementary writers receive even/odd sessions respectively; B answers before and after actual directory merge. One physical host, not LAN. No model pretending to be an independent agent conversation.
Conditions rotate per question; ingestion producer overlaps inference. A sibling LoCoMo worker shares this 8-vCPU machine. QA one greedy pass; byte-identical prompts may reuse a logged inference (explicit cache flag); do not interpret duplicated rows as independent inference trials. Latencies describe this noisy run, not a tuned production speed comparison. Model CPU prefill/decode is the expected limiter. All failures are logged and missing cells must stay missing.
Official GPT-4o judge not available; report normalized token-overlap F1, exact-match and reference-contained proxies, all NOT official accuracy. Include abstention separately (IDs ending _abs), and retain original types too. A tiny model's answer-to-explanation overlap is not a useful measure of correct abstention; separately report fixed refusal phrase rate.
Supersede, embedding, and signed-pack conditions require real supported interfaces/compilation. Do not replace them with ad-hoc deletion, SQLite mutation, or relabel directory sync as QUIC. Record blockers if unavailable. Existing compiled pack drivers did not survive restoration. Local scratch harness extensions are necessary because the model, binary, and dataset are machine-local; production code is untouched.

Scheduling amendment (10:41 ART): after confirming slow CPU prefill, separate preparation and inference into independent checkpointed processes. No data/model/prompt/retrieval/scoring choices changed. All 500 no-memory questions run first, then complete single, directory, isolated-half, merged-half and truncated reference arms in that order. Preparation proceeds ahead of inference rather than blocking after 20 queued questions. Existing exact prediction keys are resumed, never silently overwritten. Prior partial interleaved rows are retained. This maximizes usable full-condition checkpoints; latency remains noisy and not a controlled speed comparison. `prepare_only.py` plus `infer_only.py` now supersede `run_full.py` as the resume commands.

Priority clarification (10:53 ART): oracle core arms, then s-cleaned core arms, then expensive truncated-history references. Restart resumes finished keys. Writer bots are harness-controlled independent MCP clients, not independently reasoning model agents. The two variants have the same 500 questions/answers but all question dates differ; never reuse a response across differing full prompts or interpret cross-variant raw score changes as only retrieval effects.


Observed inference rows: 6791; non-cached scored prediction rows: 5405; errors: 1681. Summary includes individual error tracebacks. Directory replay verifies active text equality, but context order can differ due to distinct event metadata/ranking; never assume equal QA. Matched comparisons, not unmatched partial prefixes, support diagnostic deltas. One greedy pass, no significance or latency speedup claims. Retrieval times include the first context lookup in each brain: A and isolated B follow writes; synced/QUIC B use fresh MCP processes. These are not uniform warm-cache latency measurements. Resumption duplicates are deduplicated by dataset/question/condition; duplicate answer/prompt identities were audited.

- supersede-knowledge-update: Unavailable in installed host edge: tools/list exposes 31 tools but no supersede; CLI brain supersede is unrecognized and brain memory has no supersede. Not substituting invalidate/edit for actual supersede. SDK helper would be needed; no gold-derived update selection was attempted.
- signed-pack-QUIC-loopback: Running via sibling-built kimetsu-sdk request_pack helper; real admin-signed roster, BLAKE3/Ed25519 pack verification, iroh QUIC on 127.0.0.1, BusRelay::Disabled. Waits for each directory prediction so core arms have priority.
- hybrid-embeddings: Running as a separate checkpointed arm: see HYBRID_RESULTS.md and summary-hybrid.json; sibling logs/predictions-hybrid.jsonl is intentionally not counted by this main report. Embeddings release build succeeded.
- official-GPT4o-judge: No judge API keys available. F1, exact-match, contains, and refusal-rate are proxies, not official accuracy. A 7B local judge is deferred: only ~4.9 GiB available RAM at preflight while both 0.5B workers run, with 8 shared vCPUs; not feasible without disrupting primary full runs.

Artifacts: summary.json; scores.png; logs/predictions.jsonl (all contexts and raw answers); logs/tasks.jsonl; logs/operations.jsonl (MCP requests/responses and CLI calls); logs/events.jsonl (ingestion/sync assertions/errors); logs/llama-server.log; logs/run*.log; brains/; model/scorer checksums in ../manifest.json. Resume commands: `python3 prepare_only.py`, `python3 infer_only.py`, `python3 quic_extension.py`; it skips completed prediction keys and preserves abandoned incomplete brains under -attemptN.
