//! Deterministic synthetic memory-text generator.
//!
//! Seeded xorshift64 so a run is reproducible (no `rand` dep, no wall-clock).
//! Produces varied, non-degenerate sentences so FTS tokenization and
//! embeddings behave like real corpus data rather than one repeated string.

const SUBJECTS: &[&str] = &[
    "the build",
    "the parser",
    "the async runtime",
    "the migration",
    "the cache",
    "the embedder",
    "the broker",
    "the verifier",
    "the scheduler",
    "the allocator",
    "the docker layer",
    "the wal checkpoint",
    "the index",
    "the tokenizer",
    "the http server",
];
const VERBS: &[&str] = &[
    "fails when",
    "stalls if",
    "panics after",
    "leaks memory while",
    "deadlocks on",
    "regresses once",
    "recovers after",
    "speeds up when",
    "blocks until",
    "skips",
];
const OBJECTS: &[&str] = &[
    "the lock is held",
    "the dataset is cold",
    "the column is null",
    "the socket times out",
    "the vector dim mismatches",
    "the cwd goes stale",
    "the token expires",
    "the queue drains",
    "concurrent writers contend",
    "the page cache is evicted",
    "the model is reloaded",
    "the batch exceeds the budget",
];
const TAILS: &[&str] = &[
    "use a bounded retry",
    "pin the connection",
    "checkpoint eagerly",
    "raise the busy timeout",
    "batch the inserts",
    "warm the index first",
    "shard by repo",
    "fall back to lexical",
    "redact at ingest",
    "measure before tuning",
    "prefer the cached path",
    "isolate the subprocess",
];

/// Distinct keyword buckets. Each memory gets one `kw<bucket>` token so a query
/// targeting a bucket matches only ~N/KW_BUCKETS rows — selective FTS, like real
/// memories. Without this the tiny word pools make every query match ~half the
/// corpus and FTS5 ranks hundreds of thousands of rows (tens of seconds/query at
/// 1M). 40k buckets → ~25 matches/query at 1M, ~2 at 50k, ~1 at 100.
pub const KW_BUCKETS: usize = 40_000;

#[derive(Clone)]
pub struct Gen {
    state: u64,
}

impl Gen {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn pick<'a>(&mut self, pool: &'a [&'a str]) -> &'a str {
        pool[(self.next() as usize) % pool.len()]
    }

    /// A unique, varied sentence carrying one `kw<bucket>` selectivity token
    /// (`bucket = idx % KW_BUCKETS`). `idx` is also folded into the suffix so
    /// even an unlucky PRNG collision yields distinct `text`.
    pub fn sentence(&mut self, idx: usize) -> String {
        format!(
            "{} {} {}; {} kw{:05} (#{idx})",
            self.pick(SUBJECTS),
            self.pick(VERBS),
            self.pick(OBJECTS),
            self.pick(TAILS),
            idx % KW_BUCKETS,
        )
    }

    /// A short query that hits a POPULATED keyword bucket so it matches a small,
    /// realistic number of rows. `kw_space` = how many buckets are populated at
    /// the current scale (`min(scale, KW_BUCKETS)`); pass `scale`.
    pub fn query(&mut self, kw_space: usize) -> String {
        let bucket = (self.next() as usize) % kw_space.max(1);
        let subj = self.pick(SUBJECTS).trim_start_matches("the ");
        format!("kw{bucket:05} {subj}")
    }
}
