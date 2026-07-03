//! LoCoMo driver: `kbench locomo`.
//!
//! [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al.) is the
//! long-conversation memory benchmark mem0, Letta, and Honcho report on: 10
//! two-speaker conversations (~600 turns each across ~27 dated sessions) with
//! ~2,000 QA pairs in five categories:
//!
//!   1 = multi-hop · 2 = temporal · 3 = open-domain · 4 = single-hop ·
//!   5 = adversarial (unanswerable; the correct behaviour is to abstain)
//!
//! The driver mirrors `kbench longmemeval`: each conversation is ingested into
//! a fresh Kimetsu brain (one memory per turn, tagged with speaker + session
//! date), each question retrieves through `kimetsu brain context`, and an LLM
//! reader answers + an LLM judge grades. Parallel from day one: conversations
//! ingest concurrently, then questions run through a worker pool (each has its
//! own reader/judge subprocess; retrieval on a shared per-conversation brain is
//! read-mostly and safe under the v3.0 concurrency work).
//!
//! Dataset: `data/locomo10.json` from the LoCoMo repo. Fetch with:
//!   curl -L -o locomo10.json \
//!     https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::longmemeval::{
    LlmBackend, LmeError, claude_call, codex_call, codex_judge_prompt, codex_reader_prompt,
    heuristic_judge, reader_effort, require_embeddings_build,
};

/// Retrieval token budget for `kimetsu brain context`. LoCoMo conversations
/// are ~600 turns, so 48k (the LongMemEval setting) comfortably covers the
/// relevant span while still exercising ranking.
fn budget_tokens() -> String {
    std::env::var("LOCOMO_BUDGET_TOKENS").unwrap_or_else(|_| "48000".to_string())
}

pub fn category_name(cat: u8) -> &'static str {
    match cat {
        1 => "multi-hop",
        2 => "temporal",
        3 => "open-domain",
        4 => "single-hop",
        5 => "adversarial",
        _ => "unknown",
    }
}

// ─── Dataset model ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LocomoQa {
    pub question: String,
    /// None for category 5 (adversarial): the correct behaviour is abstention.
    pub answer: Option<String>,
    pub category: u8,
}

#[derive(Debug, Clone)]
pub struct LocomoSession {
    pub index: usize,
    pub date_time: String,
    /// (speaker, text)
    pub turns: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct LocomoSample {
    pub sample_id: String,
    pub sessions: Vec<LocomoSession>,
    pub qa: Vec<LocomoQa>,
}

/// Parse `locomo10.json`. Sessions live under dynamic keys (`session_1`,
/// `session_1_date_time`, ...), so this walks a `serde_json::Value`.
pub fn load_dataset(path: &Path) -> Result<Vec<LocomoSample>, LmeError> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        LmeError::Other(format!(
            "could not read {} ({e}). Fetch it with:\n  curl -L -o locomo10.json \
             https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            path.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| LmeError::Other(format!("invalid LoCoMo JSON: {e}")))?;
    let arr = json
        .as_array()
        .ok_or_else(|| LmeError::Other("LoCoMo dataset: expected a top-level array".into()))?;

    let mut samples = Vec::new();
    for (si, item) in arr.iter().enumerate() {
        let sample_id = item
            .get("sample_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("sample-{}", si + 1));

        let conv = item
            .get("conversation")
            .and_then(|v| v.as_object())
            .ok_or_else(|| LmeError::Other(format!("{sample_id}: missing conversation")))?;

        let mut sessions = Vec::new();
        for i in 1..=200 {
            let key = format!("session_{i}");
            let Some(turns_v) = conv.get(&key).and_then(|v| v.as_array()) else {
                continue;
            };
            let date_time = conv
                .get(&format!("session_{i}_date_time"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut turns = Vec::new();
            for t in turns_v {
                let speaker = t.get("speaker").and_then(|v| v.as_str()).unwrap_or("Speaker");
                let text = t.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if !text.trim().is_empty() {
                    turns.push((speaker.to_string(), text.trim().to_string()));
                }
            }
            if !turns.is_empty() {
                sessions.push(LocomoSession { index: i, date_time, turns });
            }
        }

        let mut qa = Vec::new();
        for q in item.get("qa").and_then(|v| v.as_array()).unwrap_or(&Vec::new()) {
            let question = q.get("question").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if question.is_empty() {
                continue;
            }
            let category = q.get("category").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
            // `answer` can be a string or a number; category 5 has none.
            let answer = q.get("answer").map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            });
            qa.push(LocomoQa { question, answer, category });
        }

        samples.push(LocomoSample { sample_id, sessions, qa });
    }
    Ok(samples)
}

// ─── Config + report ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LocomoConfig {
    pub dataset_path: PathBuf,
    /// Max questions (0 = all). Sampled round-robin per category, so a small
    /// run still covers every category.
    pub limit: usize,
    /// Only these categories (empty = all). Values 1-5.
    pub categories: Vec<u8>,
    pub dry_run: bool,
    pub kimetsu_bin: Option<PathBuf>,
    pub llm_backend: LlmBackend,
    pub llm_model: Option<String>,
    /// Question-level worker-pool size (0 = auto: KBENCH_PARALLEL or 3).
    pub parallel: usize,
}

#[derive(Debug, Serialize)]
pub struct LocomoResult {
    pub sample_id: String,
    pub category: u8,
    pub question: String,
    pub predicted: String,
    pub gold: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct LocomoReport {
    pub total: usize,
    pub correct: usize,
    pub results: Vec<LocomoResult>,
}

// ─── Run ─────────────────────────────────────────────────────────────────────

fn resolve_bin(cfg: &LocomoConfig) -> String {
    if let Some(p) = &cfg.kimetsu_bin {
        return p.to_string_lossy().to_string();
    }
    std::env::var("KIMETSU_BIN").unwrap_or_else(|_| "kimetsu".to_string())
}

/// One work unit: a question bound to its (already-ingested) sample workspace.
struct WorkItem {
    sample_idx: usize,
    qa: LocomoQa,
}

pub fn run_locomo(cfg: &LocomoConfig) -> Result<LocomoReport, LmeError> {
    let samples = load_dataset(&cfg.dataset_path)?;

    // Select questions: category filter, then round-robin per category limit.
    let mut by_cat: std::collections::BTreeMap<u8, Vec<(usize, LocomoQa)>> = Default::default();
    for (si, s) in samples.iter().enumerate() {
        for q in &s.qa {
            if !cfg.categories.is_empty() && !cfg.categories.contains(&q.category) {
                continue;
            }
            by_cat.entry(q.category).or_default().push((si, q.clone()));
        }
    }
    let mut work: Vec<WorkItem> = Vec::new();
    if cfg.limit == 0 {
        for list in by_cat.values() {
            for (si, qa) in list {
                work.push(WorkItem { sample_idx: *si, qa: qa.clone() });
            }
        }
    } else {
        // round-robin across categories until the limit is reached
        let mut iters: Vec<_> = by_cat.values().map(|v| v.iter()).collect();
        'outer: loop {
            let mut any = false;
            for it in iters.iter_mut() {
                if let Some((si, qa)) = it.next() {
                    work.push(WorkItem { sample_idx: *si, qa: qa.clone() });
                    any = true;
                    if work.len() >= cfg.limit {
                        break 'outer;
                    }
                }
            }
            if !any {
                break;
            }
        }
    }

    eprintln!(
        "locomo: {} sample(s), {} question(s) selected (backend: {})",
        samples.len(),
        work.len(),
        cfg.llm_backend.as_str()
    );

    if cfg.dry_run {
        for (i, s) in samples.iter().enumerate() {
            let turns: usize = s.sessions.iter().map(|x| x.turns.len()).sum();
            eprintln!(
                "  [dry-run] {} | sessions={} turns={} qa={}",
                s.sample_id,
                s.sessions.len(),
                turns,
                samples[i].qa.len()
            );
        }
        return Ok(LocomoReport { total: work.len(), correct: 0, results: vec![] });
    }

    let kimetsu_bin = resolve_bin(cfg);
    require_embeddings_build(&kimetsu_bin)?;

    let workers = if cfg.parallel == 0 {
        std::env::var("KBENCH_PARALLEL").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
    } else {
        cfg.parallel
    };

    // Which samples does the selected work actually touch?
    let needed: std::collections::BTreeSet<usize> = work.iter().map(|w| w.sample_idx).collect();

    // ── Phase A: ingest each needed conversation (parallel) ────────────────
    eprintln!("locomo: ingesting {} conversation(s) ({} workers)...", needed.len(), workers);
    let ws_slots: Arc<Mutex<Vec<Option<tempfile::TempDir>>>> =
        Arc::new(Mutex::new((0..samples.len()).map(|_| None).collect()));
    let ingest_list: Vec<usize> = needed.iter().copied().collect();
    let next = Arc::new(AtomicUsize::new(0));
    let ingest_err: Arc<Mutex<Option<LmeError>>> = Arc::new(Mutex::new(None));
    std::thread::scope(|scope| {
        for _ in 0..workers.min(ingest_list.len().max(1)) {
            let ingest_list = ingest_list.clone();
            let next = next.clone();
            let ws_slots = ws_slots.clone();
            let ingest_err = ingest_err.clone();
            let samples = &samples;
            let kimetsu_bin = kimetsu_bin.clone();
            scope.spawn(move || loop {
                let k = next.fetch_add(1, Ordering::SeqCst);
                if k >= ingest_list.len() {
                    break;
                }
                let si = ingest_list[k];
                match ingest_sample(&samples[si], &kimetsu_bin) {
                    Ok(tmp) => {
                        eprintln!("  [ingest {}/{}] {} ok", k + 1, ingest_list.len(), samples[si].sample_id);
                        ws_slots.lock().unwrap()[si] = Some(tmp);
                    }
                    Err(e) => {
                        eprintln!("  [ingest] {} FAILED: {e}", samples[si].sample_id);
                        *ingest_err.lock().unwrap() = Some(e);
                    }
                }
            });
        }
    });
    if let Some(e) = ingest_err.lock().unwrap().take() {
        return Err(e);
    }

    // ── Phase B: answer + judge every question (parallel) ──────────────────
    let total = work.len();
    let work = Arc::new(work);
    let results: Arc<Mutex<Vec<Option<LocomoResult>>>> =
        Arc::new(Mutex::new((0..total).map(|_| None).collect()));
    let next = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let work = work.clone();
            let results = results.clone();
            let next = next.clone();
            let ws_slots = ws_slots.clone();
            let samples = &samples;
            let cfg = cfg.clone();
            let kimetsu_bin = kimetsu_bin.clone();
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= total {
                    break;
                }
                let item = &work[i];
                let sample = &samples[item.sample_idx];
                eprintln!(
                    "  [{}/{}] {} | {} ...",
                    i + 1,
                    total,
                    sample.sample_id,
                    category_name(item.qa.category)
                );
                let res = answer_one(item, sample, &ws_slots, &cfg, &kimetsu_bin);
                let entry = match res {
                    Ok(r) => r,
                    Err(e) => LocomoResult {
                        sample_id: sample.sample_id.clone(),
                        category: item.qa.category,
                        question: item.qa.question.clone(),
                        predicted: format!("[error: {e}]"),
                        gold: item.qa.answer.clone().unwrap_or_else(|| "(unanswerable)".into()),
                        score: 0.0,
                    },
                };
                eprintln!(
                    "    -> [{}] {} {}",
                    sample.sample_id,
                    category_name(entry.category),
                    if entry.score >= 1.0 { "CORRECT" } else { "INCORRECT" }
                );
                results.lock().unwrap()[i] = Some(entry);
            });
        }
    });

    let results: Vec<LocomoResult> = Arc::try_unwrap(results)
        .expect("workers joined")
        .into_inner()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    let correct = results.iter().filter(|r| r.score >= 1.0).count();
    Ok(LocomoReport { total: results.len(), correct, results })
}

/// Ingest one conversation: one memory per turn, speaker + session date tags,
/// single add-batch call, graph edges skipped (flat is the published setup).
fn ingest_sample(sample: &LocomoSample, kimetsu_bin: &str) -> Result<tempfile::TempDir, LmeError> {
    let tmp = tempfile::Builder::new()
        .prefix("kbench-locomo-")
        .tempdir()
        .map_err(|e| LmeError::Other(format!("temp dir: {e}")))?;
    let ws = tmp.path();

    let out = Command::new(kimetsu_bin)
        .current_dir(ws)
        .args(["init"])
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn init: {e}")))?;
    if !out.status.success() {
        return Err(LmeError::KimetsuError(format!(
            "init failed for {}: {}",
            sample.sample_id,
            String::from_utf8_lossy(&out.stderr)
        )));
    }

    let mut entries = Vec::new();
    for sess in &sample.sessions {
        for (speaker, text) in &sess.turns {
            let tagged = if sess.date_time.is_empty() {
                format!("[session {}] {speaker}: {text}", sess.index)
            } else {
                format!("[session {} | {}] {speaker}: {text}", sess.index, sess.date_time)
            };
            entries.push(
                serde_json::json!({ "text": tagged, "scope": "project", "kind": "fact" })
                    .to_string(),
            );
        }
    }
    let batch = ws.join("locomo-batch.jsonl");
    std::fs::write(&batch, entries.join("\n"))
        .map_err(|e| LmeError::KimetsuError(format!("write batch: {e}")))?;
    let out = Command::new(kimetsu_bin)
        .current_dir(ws)
        .args(["brain", "memory", "add-batch"])
        .arg(&batch)
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn add-batch: {e}")))?;
    if !out.status.success() {
        return Err(LmeError::KimetsuError(format!(
            "add-batch failed for {}: {}",
            sample.sample_id,
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(tmp)
}

fn answer_one(
    item: &WorkItem,
    sample: &LocomoSample,
    ws_slots: &Arc<Mutex<Vec<Option<tempfile::TempDir>>>>,
    cfg: &LocomoConfig,
    kimetsu_bin: &str,
) -> Result<LocomoResult, LmeError> {
    // Retrieve from the sample's shared workspace (read-mostly; concurrent
    // kimetsu processes on one brain are safe post-v3.0).
    let ws_path = {
        let guard = ws_slots.lock().unwrap();
        guard[item.sample_idx]
            .as_ref()
            .map(|t| t.path().to_path_buf())
            .ok_or_else(|| LmeError::Other("workspace missing (ingest failed?)".into()))?
    };
    let out = Command::new(kimetsu_bin)
        .current_dir(&ws_path)
        .args([
            "brain",
            "context",
            &item.qa.question,
            "--no-ambient",
            "--budget-tokens",
            &budget_tokens(),
        ])
        .output()
        .map_err(|e| LmeError::KimetsuError(format!("spawn brain context: {e}")))?;
    let context = String::from_utf8_lossy(&out.stdout).to_string();

    // The reader anchors "today" to the conversation's last session date.
    let question_date = sample
        .sessions
        .last()
        .map(|s| s.date_time.clone())
        .unwrap_or_default();
    let prompt = codex_reader_prompt(&item.qa.question, &context, &question_date);
    let predicted = match cfg.llm_backend {
        LlmBackend::Codex => {
            codex_call(&prompt, cfg.llm_model.as_deref(), Some(&reader_effort()))?
        }
        LlmBackend::Claude => claude_call(&prompt, cfg.llm_model.as_deref())?,
        LlmBackend::Http => {
            return Err(LmeError::LlmError(
                "locomo supports --reader-backend codex|claude".to_string(),
            ));
        }
    };

    // Judge. Category 5 is adversarial: unanswerable, so abstention is correct.
    let is_abstention = item.qa.category == 5;
    let gold = item
        .qa
        .answer
        .clone()
        .unwrap_or_else(|| "(unanswerable: the conversation does not contain this)".into());
    let judge_prompt = codex_judge_prompt(
        &item.qa.question,
        &gold,
        &predicted,
        is_abstention,
        category_name(item.qa.category),
    );
    let verdict = match cfg.llm_backend {
        LlmBackend::Codex => codex_call(&judge_prompt, cfg.llm_model.as_deref(), None),
        LlmBackend::Claude => claude_call(&judge_prompt, cfg.llm_model.as_deref()),
        LlmBackend::Http => unreachable!(),
    };
    let score = match verdict {
        Ok(reply) => {
            let up = reply.to_uppercase();
            if up.contains("CORRECT") && !up.contains("INCORRECT") { 1.0 } else { 0.0 }
        }
        Err(e) => {
            eprintln!("    [judge] warn: {e} — heuristic fallback");
            heuristic_judge(&predicted, &gold, is_abstention)
        }
    };

    Ok(LocomoResult {
        sample_id: sample.sample_id.clone(),
        category: item.qa.category,
        question: item.qa.question.clone(),
        predicted,
        gold,
        score,
    })
}

// ─── Report rendering ────────────────────────────────────────────────────────

pub fn render_markdown(report: &LocomoReport, dataset: &Path) -> String {
    use std::fmt::Write as _;
    let mut md = String::new();
    let _ = writeln!(md, "# LoCoMo Report\n");
    let _ = writeln!(md, "Dataset: {}\n", dataset.display());
    let pct = |c: usize, t: usize| if t == 0 { 0.0 } else { 100.0 * c as f64 / t as f64 };
    let _ = writeln!(
        md,
        "**Overall accuracy: {:.1}% ({}/{})**\n",
        pct(report.correct, report.total),
        report.correct,
        report.total
    );

    let mut cats: std::collections::BTreeMap<u8, (usize, usize)> = Default::default();
    for r in &report.results {
        let e = cats.entry(r.category).or_default();
        e.1 += 1;
        if r.score >= 1.0 {
            e.0 += 1;
        }
    }
    let _ = writeln!(md, "## By category\n");
    let _ = writeln!(md, "| category | correct | total | accuracy |");
    let _ = writeln!(md, "|---|---|---|---|");
    let mut non_adv = (0usize, 0usize);
    for (cat, (c, t)) in &cats {
        let _ = writeln!(md, "| {} | {} | {} | {:.1}% |", category_name(*cat), c, t, pct(*c, *t));
        if *cat != 5 {
            non_adv.0 += c;
            non_adv.1 += t;
        }
    }
    let _ = writeln!(
        md,
        "\nNon-adversarial overall (cats 1-4, the slice most vendors report): \
         **{:.1}% ({}/{})**",
        pct(non_adv.0, non_adv.1),
        non_adv.0,
        non_adv.1
    );
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"[{
          "sample_id": "t1",
          "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "1:00 pm on 1 May, 2023",
            "session_1": [
              {"speaker": "A", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
              {"speaker": "B", "dia_id": "D1:2", "text": "Congrats!"}
            ],
            "session_2_date_time": "2:00 pm on 8 May, 2023",
            "session_2": [
              {"speaker": "A", "dia_id": "D2:1", "text": "Rex learned to fetch."}
            ]
          },
          "qa": [
            {"question": "What is the dog's name?", "answer": "Rex", "evidence": ["D1:1"], "category": 4},
            {"question": "When did A adopt Rex?", "answer": "1 May 2023", "evidence": ["D1:1"], "category": 2},
            {"question": "What color is Rex?", "adversarial_answer": "brown", "evidence": [], "category": 5}
          ]
        }]"#
    }

    #[test]
    fn parses_sessions_turns_and_qa() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("l.json");
        std::fs::write(&p, sample_json()).unwrap();
        let samples = load_dataset(&p).unwrap();
        assert_eq!(samples.len(), 1);
        let s = &samples[0];
        assert_eq!(s.sessions.len(), 2);
        assert_eq!(s.sessions[0].turns.len(), 2);
        assert_eq!(s.sessions[0].date_time, "1:00 pm on 1 May, 2023");
        assert_eq!(s.qa.len(), 3);
        // category 5 has no `answer` field
        assert!(s.qa[2].answer.is_none());
        assert_eq!(s.qa[2].category, 5);
    }

    #[test]
    fn category_names_cover_paper_taxonomy() {
        assert_eq!(category_name(1), "multi-hop");
        assert_eq!(category_name(2), "temporal");
        assert_eq!(category_name(3), "open-domain");
        assert_eq!(category_name(4), "single-hop");
        assert_eq!(category_name(5), "adversarial");
    }

    #[test]
    fn limit_samples_round_robin_across_categories() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("l.json");
        std::fs::write(&p, sample_json()).unwrap();
        let cfg = LocomoConfig {
            dataset_path: p,
            limit: 2,
            categories: vec![],
            dry_run: true,
            kimetsu_bin: None,
            llm_backend: LlmBackend::Codex,
            llm_model: None,
            parallel: 1,
        };
        // dry_run: selection happens before any model call; total reflects it
        let report = run_locomo(&cfg).unwrap();
        assert_eq!(report.total, 2);
    }

    #[test]
    fn render_markdown_reports_non_adversarial_slice() {
        let report = LocomoReport {
            total: 2,
            correct: 1,
            results: vec![
                LocomoResult {
                    sample_id: "t1".into(),
                    category: 4,
                    question: "q".into(),
                    predicted: "Rex".into(),
                    gold: "Rex".into(),
                    score: 1.0,
                },
                LocomoResult {
                    sample_id: "t1".into(),
                    category: 5,
                    question: "q2".into(),
                    predicted: "I don't know".into(),
                    gold: "(unanswerable)".into(),
                    score: 0.0,
                },
            ],
        };
        let md = render_markdown(&report, Path::new("x.json"));
        assert!(md.contains("Non-adversarial overall"));
        assert!(md.contains("single-hop"));
        assert!(md.contains("adversarial"));
    }
}
