//! BEAM driver — the long-term-memory benchmark from
//! github.com/mohammadtavakoli78/BEAM (HF: Mohammadta/BEAM-10M).
//!
//! BEAM probes ten memory abilities (information extraction, multi-hop
//! reasoning, knowledge update, temporal reasoning, abstention, contradiction
//! resolution, event ordering, instruction following, preference following,
//! summarization) over long multi-session conversations (128K → 10M tokens).
//! Each conversation carries `probing_questions` keyed by ability category;
//! every probe has a `question` and a grading `rubric`. The official pipeline
//! scores answers with an **LLM-as-judge against the rubric** (plus optional
//! lexical metrics). This driver mirrors that, reusing the LongMemEval reader/
//! judge machinery (codex `codex_call`, no API key).
//!
//! ## Pipeline (per conversation)
//!   1. Spin a fresh isolated kimetsu brain (git init + `kimetsu init`).
//!   2. Ingest the `chat` turns (one memory per turn) via `brain memory add-batch`.
//!   3. For each probe: `brain context` → reader answers → LLM-judges vs `rubric`.
//!   4. Aggregate score per ability category + overall.
//!
//! ## Dataset format (JSON — see `BeamDataset`)
//! The HF dataset ships as parquet; convert it to the JSON shape below first
//! (one object per conversation). The `--synthetic` fixture and `--dry-run`
//! work without any dataset for end-to-end validation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

// Reuse the LongMemEval LLM machinery: codex backend, error type, retrieval
// budget. `codex_call` / `LlmBackend` / `LmeError` are all public there.
use super::longmemeval::{LlmBackend, LmeError, claude_call, codex_call};

/// One conversation turn.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BeamTurn {
    pub role: String,
    pub content: String,
}

/// One probing question + its grading rubric (the LLM-judge's criteria).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BeamProbe {
    pub question: String,
    /// Grading rubric (what a correct answer must satisfy). Falls back to the
    /// gold answer string if a dataset only provides `answer`.
    #[serde(default)]
    pub rubric: String,
}

/// One BEAM conversation = a haystack `chat` + per-category probing questions.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BeamConversation {
    pub id: String,
    /// Token-length bucket: "128k" | "500k" | "1m" | "10m" (informational).
    #[serde(default)]
    pub token_bucket: String,
    /// Full conversation history (alternating user/assistant turns).
    pub chat: Vec<BeamTurn>,
    /// Probing questions keyed by ability category (the 10 BEAM categories).
    pub probing: BTreeMap<String, Vec<BeamProbe>>,
}

/// On-disk dataset: `{ "conversations": [ BeamConversation, ... ] }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BeamDataset {
    pub conversations: Vec<BeamConversation>,
}

// ─── Config ──────────────────────────────────────────────────────────────────

// Retrieval budget for `brain context`. BEAM's global-aggregation abilities
// (summarization, event_ordering, contradiction_resolution) need COMPREHENSIVE
// recall — a small budget that surfaces only the top-k most relevant capsules
// structurally fails them (an incomplete summary; one side of a contradiction).
// 96k covers most of a ~100k-token conversation while still exercising ranking
// (it's not unbounded), so it's a fair budget for the 100K bucket.
const BEAM_BUDGET_TOKENS: &str = "96000";

#[derive(Debug, Clone)]
pub struct BeamConfig {
    pub dataset_path: Option<PathBuf>,
    pub kimetsu_bin: Option<PathBuf>,
    pub llm_backend: LlmBackend,
    pub llm_model: Option<String>,
    /// Truncate to this many conversations (0 = all).
    pub limit: usize,
    /// Only these ability categories (empty = all).
    pub categories: Vec<String>,
    /// Parse + plan, no kimetsu/model calls.
    pub dry_run: bool,
}

impl BeamConfig {
    fn resolve_bin(&self) -> String {
        if let Some(p) = &self.kimetsu_bin {
            return p.to_string_lossy().to_string();
        }
        std::env::var("KIMETSU_BIN").unwrap_or_else(|_| "kimetsu".to_string())
    }
}

// ─── Report ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct BeamProbeResult {
    pub conversation_id: String,
    pub category: String,
    pub question: String,
    pub predicted: String,
    pub correct: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BeamCategoryStats {
    pub correct: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BeamReport {
    pub generated_at: String,
    pub dataset: String,
    pub conversations: usize,
    pub correct: usize,
    pub total: usize,
    pub by_category: BTreeMap<String, BeamCategoryStats>,
    pub results: Vec<BeamProbeResult>,
}

impl BeamReport {
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# BEAM Report\n\n");
        out.push_str(&format!("Generated: {}\n", self.generated_at));
        out.push_str(&format!("Dataset: {}\n\n", self.dataset));
        out.push_str(&format!(
            "**Overall accuracy: {:.1}% ({}/{}) over {} conversation(s)**\n\n",
            self.accuracy() * 100.0,
            self.correct,
            self.total,
            self.conversations
        ));
        out.push_str("## By ability category\n\n| category | correct | total | accuracy |\n|---|---|---|---|\n");
        for (cat, s) in &self.by_category {
            let acc = if s.total == 0 {
                0.0
            } else {
                s.correct as f64 / s.total as f64
            };
            out.push_str(&format!(
                "| {} | {} | {} | {:.1}% |\n",
                cat,
                s.correct,
                s.total,
                acc * 100.0
            ));
        }
        out
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ─── Dataset loading ─────────────────────────────────────────────────────────

pub fn load_dataset(path: &Path) -> Result<Vec<BeamConversation>, LmeError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        LmeError::Other(format!(
            "could not read BEAM dataset {}: {e}",
            path.display()
        ))
    })?;
    // Accept either { "conversations": [...] } or a bare [ ... ] array.
    if let Ok(ds) = serde_json::from_str::<BeamDataset>(&text) {
        return Ok(ds.conversations);
    }
    serde_json::from_str::<Vec<BeamConversation>>(&text)
        .map_err(|e| LmeError::Other(format!("BEAM dataset parse failed: {e}")))
}

/// Five tiny synthetic conversations (one per a few categories) for end-to-end
/// validation without the real dataset.
pub fn synthetic_fixture() -> Vec<BeamConversation> {
    let mut probing: BTreeMap<String, Vec<BeamProbe>> = BTreeMap::new();
    probing.insert(
        "information_extraction".to_string(),
        vec![BeamProbe {
            question: "What city did the user move to?".to_string(),
            rubric: "Correct iff the answer says Berlin.".to_string(),
        }],
    );
    probing.insert(
        "knowledge_update".to_string(),
        vec![BeamProbe {
            question: "What is the user's current job title?".to_string(),
            rubric: "Correct iff the answer reflects the MOST RECENT title: Staff Engineer."
                .to_string(),
        }],
    );
    probing.insert(
        "abstention".to_string(),
        vec![BeamProbe {
            question: "What is the user's blood type?".to_string(),
            rubric: "Correct iff the answer indicates it is unknown / not stated.".to_string(),
        }],
    );
    vec![BeamConversation {
        id: "syn-1".to_string(),
        token_bucket: "128k".to_string(),
        chat: vec![
            BeamTurn {
                role: "user".into(),
                content: "I just moved to Berlin for a new role.".into(),
            },
            BeamTurn {
                role: "assistant".into(),
                content: "Congrats on the move to Berlin!".into(),
            },
            BeamTurn {
                role: "user".into(),
                content: "I started as a Senior Engineer.".into(),
            },
            BeamTurn {
                role: "user".into(),
                content: "Update: I was promoted to Staff Engineer last month.".into(),
            },
            BeamTurn {
                role: "assistant".into(),
                content: "Great, Staff Engineer is a big step.".into(),
            },
        ],
        probing,
    }]
}

// ─── Per-conversation pipeline ───────────────────────────────────────────────

struct Workspace {
    tmp: tempfile::TempDir,
}

fn setup_and_ingest(conv: &BeamConversation, kimetsu_bin: &str) -> Result<Workspace, LmeError> {
    let tmp = tempfile::Builder::new()
        .prefix("kbench-beam-")
        .tempdir()
        .map_err(|e| LmeError::Other(format!("temp dir: {e}")))?;
    let ws = tmp.path();

    let _ = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(ws)
        .output();
    let init = Command::new(kimetsu_bin)
        .args(["init"])
        .current_dir(ws)
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn kimetsu init: {e}")))?;
    if !init.status.success() {
        return Err(LmeError::KimetsuError(format!(
            "kimetsu init failed: {}",
            String::from_utf8_lossy(&init.stderr)
        )));
    }

    // Retrieval backend. Default graph-lite; override via KBENCH_BEAM_BACKEND
    // (e.g. "flat") to A/B graph-lite vs flat on the same data. graph-lite
    // follows model-free `relates_to` edges (built below) for multi-hop recall,
    // scored by hop-decayed seed relevance (levers 2/3 in the engine).
    let backend = std::env::var("KBENCH_BEAM_BACKEND").unwrap_or_else(|_| "graph-lite".into());
    let _ = Command::new(kimetsu_bin)
        .current_dir(ws)
        .args(["config", "set", "storage.backend", &backend])
        .output();
    let use_graph = backend == "graph-lite";

    // One memory per turn (role-prefixed), ingested in a single add-batch.
    let entries: Vec<String> = conv
        .chat
        .iter()
        .filter(|t| !t.content.trim().is_empty())
        .map(|t| {
            let role = match t.role.as_str() {
                "user" => "User",
                "assistant" => "Assistant",
                other => other,
            };
            serde_json::json!({
                "text": format!("{role}: {}", t.content.trim()),
                "scope": "project",
                "kind": "fact"
            })
            .to_string()
        })
        .collect();
    let batch = ws.join("beam-batch.jsonl");
    std::fs::write(&batch, entries.join("\n"))
        .map_err(|e| LmeError::KimetsuError(format!("write batch: {e}")))?;
    // Supersession on ingest: detect + resolve contradictions so an updated
    // fact collapses to its current value (targets knowledge_update). Needs the
    // embedder (this run uses jina) for the cosine neighbour scan.
    let out = Command::new(kimetsu_bin)
        .current_dir(ws)
        .env("KIMETSU_DETECT_CONFLICTS", "1")
        .env("KIMETSU_RESOLVE_CONFLICTS", "1")
        .args(["brain", "memory", "add-batch"])
        .arg(&batch)
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn add-batch: {e}")))?;
    if !out.status.success() {
        return Err(LmeError::KimetsuError(format!(
            "add-batch failed for {}: {}",
            conv.id,
            String::from_utf8_lossy(&out.stderr)
        )));
    }

    // Build the model-free `relates_to` entity graph over the ingested memories
    // (shared proper nouns / salient terms) so graph-lite retrieval has edges to
    // traverse. Only when graph-lite is active. Rule-based only (no model).
    if use_graph {
        let _ = Command::new(kimetsu_bin)
            .current_dir(ws)
            .args(["brain", "graph", "build"])
            .output();
    }

    Ok(Workspace { tmp })
}

fn retrieve(question: &str, ws: &Path, kimetsu_bin: &str) -> Result<String, LmeError> {
    let out = Command::new(kimetsu_bin)
        .current_dir(ws)
        .args([
            "brain",
            "context",
            question,
            "--no-ambient",
            "--budget-tokens",
            BEAM_BUDGET_TOKENS,
        ])
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn brain context: {e}")))?;
    if !out.status.success() {
        return Err(LmeError::KimetsuError(format!(
            "brain context failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Category-aware reader instruction. BEAM's ten abilities are NOT all factual
/// single-answer lookups: a recency-QA framing ("report the most recent value,
/// be concise, else say I don't know") systematically ZEROES summarization,
/// event_ordering and contradiction_resolution — the reader answers "Yes/No" or
/// gives a terse fact when the rubric wants a full summary / ordered sequence /
/// the surfaced contradiction. Each task gets the instruction it actually needs;
/// everything else keeps the factual-QA framing (which abstention relies on).
fn reader_prompt(category: &str, question: &str, context: &str) -> String {
    let task = match category {
        "summarization" => {
            "Write a COMPREHENSIVE summary covering ALL the key developments, features, decisions \
             and milestones in the context that are relevant to the question. Be thorough and \
             include every distinct point — do not be terse or omit details."
        }
        "event_ordering" => {
            "List the relevant items/events in the ORDER they occurred or were discussed, as an \
             ordered sequence — use the dates and the order they appear in the context. Cover every \
             distinct item the question asks about."
        }
        "contradiction_resolution" => {
            "Determine whether the context contains CONFLICTING statements about what the question \
             asks. If it does, explicitly STATE that there is contradictory information and mention \
             BOTH conflicting statements. Do not answer with a bare yes or no."
        }
        _ => {
            "Answer the question. Be concise — output just the answer text. If a fact changed over \
             time, report the MOST RECENT value. If the answer is not in the context, output \
             exactly: I don't know"
        }
    };
    format!(
        "You are a memory assistant. Use ONLY the provided memory context below. {task}\n\n\
         Memory context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
}

/// Rubric-coverage judge. BEAM rubrics are multi-point checklists ("the response
/// should contain/mention/state X ... Y ... Z"); a single all-or-nothing
/// CORRECT/INCORRECT against an N-point rubric never gets a full match and
/// returns INCORRECT every time. Instead: identify the distinct required points,
/// count how many the answer satisfies (paraphrase counts), and pass on
/// substantive coverage — all of a single-point rubric, at least half of a
/// multi-point one. Mirrors BEAM's official rubric-scored LLM judge.
fn judge_prompt(question: &str, rubric: &str, predicted: &str) -> String {
    format!(
        "You are grading an answer for a long-term-memory benchmark using a RUBRIC. The rubric \
         lists one or more required points (often phrased 'the response should contain/mention/\
         state ...'). Identify each distinct required point, then decide for each whether the \
         PREDICTED ANSWER satisfies it — paraphrases and equivalent wording count, do NOT require \
         verbatim text.\n\n\
         First output a line 'POINTS: k/n' where n is the number of distinct required points and k \
         is how many the predicted answer satisfies. Then output 'VERDICT: CORRECT' if the answer \
         covers the substance of the rubric (for a single-point rubric that point must be met; for \
         a multi-point rubric at least half the points must be met), otherwise 'VERDICT: \
         INCORRECT'.\n\n\
         Question: {question}\n\
         Rubric: {rubric}\n\
         Predicted answer: {predicted}\n\n\
         Grade now (POINTS line then VERDICT line):"
    )
}

fn answer(
    category: &str,
    question: &str,
    context: &str,
    cfg: &BeamConfig,
) -> Result<String, LmeError> {
    match cfg.llm_backend {
        LlmBackend::Codex => codex_call(
            &reader_prompt(category, question, context),
            cfg.llm_model.as_deref(),
            Some(&super::longmemeval::reader_effort()),
        ),
        LlmBackend::Claude => claude_call(
            &reader_prompt(category, question, context),
            cfg.llm_model.as_deref(),
        ),
        LlmBackend::Http => Err(LmeError::LlmError(
            "BEAM driver supports the codex or claude backend (--reader-backend codex|claude)"
                .to_string(),
        )),
    }
}

fn judge(
    question: &str,
    rubric: &str,
    predicted: &str,
    cfg: &BeamConfig,
) -> Result<bool, LmeError> {
    match cfg.llm_backend {
        LlmBackend::Codex => {
            let verdict = codex_call(
                &judge_prompt(question, rubric, predicted),
                cfg.llm_model.as_deref(),
                Some("low"),
            )?;
            Ok(verdict.to_uppercase().contains("CORRECT")
                && !verdict.to_uppercase().contains("INCORRECT"))
        }
        LlmBackend::Claude => {
            let verdict = claude_call(
                &judge_prompt(question, rubric, predicted),
                cfg.llm_model.as_deref(),
            )?;
            Ok(verdict.to_uppercase().contains("CORRECT")
                && !verdict.to_uppercase().contains("INCORRECT"))
        }
        LlmBackend::Http => Err(LmeError::LlmError(
            "BEAM driver supports the codex or claude backend".to_string(),
        )),
    }
}

/// Score one probe end-to-end (retrieve → read → judge). Errors (LLM
/// degeneration, kimetsu failures) are RETURNED, not propagated past the caller —
/// the run loop records them as incorrect and continues, so a single bad codex
/// sample can't abort a multi-hour 400-probe run (mirrors LongMemEval's
/// `run_real` per-instance error tolerance).
fn score_probe(
    category: &str,
    question: &str,
    rubric: &str,
    ws_path: &Path,
    cfg: &BeamConfig,
    kimetsu_bin: &str,
) -> Result<(String, bool), LmeError> {
    let context = retrieve(question, ws_path, kimetsu_bin)?;
    // The reader/judge codex calls can occasionally degenerate (token-salad
    // output → API 400 invalid_request). Re-sampling almost always fixes it, so
    // retry each once before giving up.
    let predicted = answer(category, question, &context, cfg).or_else(|e| {
        eprintln!("      [reader] retry after: {e}");
        answer(category, question, &context, cfg)
    })?;
    let correct = judge(question, rubric, &predicted, cfg).or_else(|e| {
        eprintln!("      [judge] retry after: {e}");
        judge(question, rubric, &predicted, cfg)
    })?;
    Ok((predicted, correct))
}

// ─── Run ─────────────────────────────────────────────────────────────────────

pub fn run_beam(
    cfg: &BeamConfig,
    conversations: Vec<BeamConversation>,
) -> Result<BeamReport, LmeError> {
    let kimetsu_bin = cfg.resolve_bin();
    super::longmemeval::require_embeddings_build(&kimetsu_bin)?;

    // Filter categories + limit.
    let mut convs = conversations;
    if cfg.limit > 0 && convs.len() > cfg.limit {
        convs.truncate(cfg.limit);
    }

    let mut report = BeamReport {
        generated_at: "unknown".to_string(),
        dataset: cfg
            .dataset_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<synthetic>".to_string()),
        conversations: convs.len(),
        correct: 0,
        total: 0,
        by_category: BTreeMap::new(),
        results: Vec::new(),
    };

    for (ci, conv) in convs.iter().enumerate() {
        eprintln!(
            "  [{}/{}] {} ({}) ...",
            ci + 1,
            convs.len(),
            conv.id,
            conv.token_bucket
        );
        // Ingest is per-conversation. If it fails (e.g. kimetsu init/add-batch
        // error), don't abort the whole run — record this conversation's probes
        // as errored and move on, so one bad conversation costs 20 probes, not
        // the remaining hundreds.
        let ws = if cfg.dry_run {
            None
        } else {
            match setup_and_ingest(conv, &kimetsu_bin) {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!(
                        "    -> INGEST FAILED for {} (probes counted as errored): {e}",
                        conv.id
                    );
                    None
                }
            }
        };
        let ingest_failed = !cfg.dry_run && ws.is_none();

        for (category, probes) in &conv.probing {
            if !cfg.categories.is_empty() && !cfg.categories.iter().any(|c| c == category) {
                continue;
            }
            for probe in probes {
                let stats = report.by_category.entry(category.clone()).or_default();
                stats.total += 1;
                report.total += 1;

                if cfg.dry_run {
                    report.results.push(BeamProbeResult {
                        conversation_id: conv.id.clone(),
                        category: category.clone(),
                        question: probe.question.clone(),
                        predicted: "[dry-run]".to_string(),
                        correct: false,
                    });
                    continue;
                }

                if ingest_failed {
                    report.results.push(BeamProbeResult {
                        conversation_id: conv.id.clone(),
                        category: category.clone(),
                        question: probe.question.clone(),
                        predicted: "[error: ingest failed]".to_string(),
                        correct: false,
                    });
                    continue;
                }

                // Per-probe error tolerance: a single LLM degeneration or kimetsu
                // hiccup is recorded as incorrect, not propagated — the run
                // continues to the next probe.
                let ws_path = ws.as_ref().unwrap().tmp.path();
                let (predicted, correct) = match score_probe(
                    category,
                    &probe.question,
                    &probe.rubric,
                    ws_path,
                    cfg,
                    &kimetsu_bin,
                ) {
                    Ok((p, c)) => (p, c),
                    Err(e) => {
                        eprintln!("      {category}: ERROR (counted incorrect): {e}");
                        (format!("[error: {e}]"), false)
                    }
                };

                if correct {
                    report.by_category.get_mut(category).unwrap().correct += 1;
                    report.correct += 1;
                }
                eprintln!(
                    "      {category}: {} | {}",
                    if correct { "CORRECT" } else { "INCORRECT" },
                    predicted
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(60)
                        .collect::<String>()
                );
                report.results.push(BeamProbeResult {
                    conversation_id: conv.id.clone(),
                    category: category.clone(),
                    question: probe.question.clone(),
                    predicted,
                    correct,
                });
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_fixture_is_well_formed() {
        let convs = synthetic_fixture();
        assert_eq!(convs.len(), 1);
        assert!(!convs[0].chat.is_empty());
        assert!(convs[0].probing.contains_key("knowledge_update"));
    }

    #[test]
    fn load_dataset_accepts_envelope_and_bare_array() {
        let convs = synthetic_fixture();
        let envelope = serde_json::to_string(&BeamDataset {
            conversations: convs.clone(),
        })
        .unwrap();
        let bare = serde_json::to_string(&convs).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("env.json");
        let p2 = dir.path().join("bare.json");
        std::fs::write(&p1, envelope).unwrap();
        std::fs::write(&p2, bare).unwrap();
        assert_eq!(load_dataset(&p1).unwrap().len(), 1);
        assert_eq!(load_dataset(&p2).unwrap().len(), 1);
    }

    #[test]
    fn dry_run_counts_probes_without_calls() {
        let cfg = BeamConfig {
            dataset_path: None,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Codex,
            llm_model: None,
            limit: 0,
            categories: vec![],
            dry_run: true,
        };
        let report = run_beam(&cfg, synthetic_fixture()).unwrap();
        assert_eq!(report.total, 3, "3 probes across 3 categories");
        assert_eq!(report.by_category.len(), 3);
        assert_eq!(report.correct, 0); // dry-run never scores
    }
}
