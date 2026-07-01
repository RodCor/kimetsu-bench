//! Per-backend implementations of the `BenchmarkDriver` trait.
//!
//! Each driver is independent. The orchestrator picks one by name
//! via the `--driver` CLI flag; future drivers (SWE-bench, custom
//! internal corpora) drop in as new files here.

pub mod beam;
pub mod brainbench;
pub mod longmemeval;
pub mod terminal_bench;
