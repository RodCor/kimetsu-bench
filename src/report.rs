//! Comparison report — turns a set of graded RunResults into something
//! readable. Two formats:
//!
//! - JSON (raw, for piping into other tools / persistence).
//! - Markdown table (for paste-into-notes ergonomics).
//!
//! Both keyed by task × agent; the markdown emits one row per task
//! with side-by-side scores across configs so the impact diff is
//! visible at a glance.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::driver::{AgentConfig, Grade, RunResult, TaskId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub driver: String,
    pub generated_at: String,
    pub agents: Vec<AgentConfig>,
    pub tasks: Vec<TaskId>,
    pub entries: Vec<ReportEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportEntry {
    pub task: TaskId,
    pub agent: AgentConfig,
    pub run: RunResult,
    pub grade: Grade,
}

impl Report {
    pub fn build(driver_name: &str, runs_and_grades: Vec<(RunResult, Grade)>) -> Self {
        let now = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "unknown".to_string());

        let mut agents: Vec<AgentConfig> = Vec::new();
        let mut tasks: Vec<TaskId> = Vec::new();
        let mut entries: Vec<ReportEntry> = Vec::new();

        for (run, grade) in runs_and_grades {
            if !agents.contains(&run.agent) {
                agents.push(run.agent);
            }
            if !tasks.contains(&run.task) {
                tasks.push(run.task.clone());
            }
            entries.push(ReportEntry {
                task: run.task.clone(),
                agent: run.agent,
                run,
                grade,
            });
        }

        Self {
            driver: driver_name.to_string(),
            generated_at: now,
            agents,
            tasks,
            entries,
        }
    }

    /// JSON-pretty for CI / piping.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("serialize report")
    }

    /// Markdown table: one row per task, one column per agent config.
    /// Cells show score (and exit code, when non-zero). The summary
    /// line at the bottom totals each agent's wins.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Benchmark report — {}\n\n", self.driver));
        out.push_str(&format!("Generated: {}\n\n", self.generated_at));

        // Header row.
        out.push_str("| Task |");
        for agent in &self.agents {
            out.push_str(&format!(" {} |", agent.label()));
        }
        out.push('\n');
        out.push_str("|------|");
        for _ in &self.agents {
            out.push_str("------|");
        }
        out.push('\n');

        // Index entries by (task, agent) for fast lookup.
        let mut by_pair: BTreeMap<(String, AgentConfig), &ReportEntry> = BTreeMap::new();
        for entry in &self.entries {
            by_pair.insert((entry.task.0.clone(), entry.agent), entry);
        }

        // One row per task.
        for task in &self.tasks {
            out.push_str(&format!("| {} |", task));
            for agent in &self.agents {
                if let Some(entry) = by_pair.get(&(task.0.clone(), *agent)) {
                    let mark = if entry.grade.score >= 1.0 {
                        "✓"
                    } else if entry.grade.score > 0.0 {
                        "~"
                    } else {
                        "✗"
                    };
                    out.push_str(&format!(
                        " {mark} {:.2} ({:.1}s, ${:.2}) |",
                        entry.grade.score, entry.run.duration_secs, entry.run.cost_usd
                    ));
                } else {
                    out.push_str(" – |");
                }
            }
            out.push('\n');
        }

        // Summary row: wins / total per agent.
        out.push_str("\n## Summary\n\n");
        for agent in &self.agents {
            let wins = self
                .entries
                .iter()
                .filter(|e| e.agent == *agent && e.grade.score >= 1.0)
                .count();
            let total = self.entries.iter().filter(|e| e.agent == *agent).count();
            let cost: f32 = self
                .entries
                .iter()
                .filter(|e| e.agent == *agent)
                .map(|e| e.run.cost_usd)
                .sum();
            out.push_str(&format!(
                "- `{}`: {wins}/{total} wins, ${cost:.2} total\n",
                agent.label()
            ));
        }
        out
    }
}
