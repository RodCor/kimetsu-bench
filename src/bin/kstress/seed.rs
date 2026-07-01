//! Fast bulk-seed: insert N synthetic memories straight into `brain.db`.
//!
//! Bypasses the public `add_memory` path (file lock + event sourcing + O(N)
//! per-insert conflict scan) — that path is ~O(N²) with embeddings and can't
//! reach 1M. We write `memories` + `memories_fts` (and, for the emb matrix,
//! the `embedding` BLOB) directly in one transaction per batch, exactly the
//! columns kimetsu's own writer fills. The `vec0` ANN index is left for
//! kimetsu's `retrieve_context` to build lazily on first query.
//!
//! ALL of this lives in `bench/` and only calls kimetsu's PUBLIC API
//! (`encode_embedding`, `normalize_memory_text`, the `Embedder` trait).

use std::time::Instant;

use rusqlite::{Connection, params};

use kimetsu_brain::embeddings::{Embedder, encode_embedding};
use kimetsu_core::memory::{MemoryKind, MemoryScope, normalize_memory_text};

use crate::corpus::Gen;

pub struct SeedOpts {
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub batch_size: usize,
    pub with_embeddings: bool,
    pub seed: u64,
}

#[derive(Default)]
pub struct SeedStats {
    pub inserted: usize,
    pub gen_secs: f64,
    pub embed_secs: f64,
    pub sql_secs: f64,
}

/// Insert `count` rows starting at `start_index` (so callers can grow a DB
/// incrementally without colliding `memory_id`s). `embedder` is consulted only
/// when `opts.with_embeddings` is set AND the embedder is real (not Noop).
pub fn bulk_seed(
    conn: &mut Connection,
    start_index: usize,
    count: usize,
    opts: &SeedOpts,
    embedder: &dyn Embedder,
) -> Result<SeedStats, String> {
    let scope = opts.scope.to_string();
    let kind = opts.kind.to_string();
    let created = now_rfc3339();
    let do_embed = opts.with_embeddings && !embedder.is_noop();
    let model_id = embedder.model_id().to_string();

    // This is a throwaway benchmark DB and the bulk seed is SETUP, not a
    // measured path — so trade durability for speed. Without this, seeding 1M
    // rows on the slow /mnt/e 9p mount (one fsync per commit) crawls to a halt.
    // synchronous=OFF removes the per-commit fsync; a big page cache keeps the
    // index + FTS B-trees hot; we checkpoint the WAL periodically below.
    conn.execute_batch(
        "PRAGMA synchronous=OFF; PRAGMA cache_size=-262144; PRAGMA temp_store=MEMORY;",
    )
    .map_err(|e| e.to_string())?;

    let mut corpus = Gen::new(opts.seed.wrapping_add(start_index as u64));
    let mut stats = SeedStats::default();
    let mut done = 0usize;
    let mut since_checkpoint = 0usize;

    while done < count {
        let n = opts.batch_size.min(count - done);

        // 1. Generate this batch's text up front (so we can batch-embed).
        let t = Instant::now();
        let mut rows: Vec<(String, String, String)> = Vec::with_capacity(n);
        for i in 0..n {
            let idx = start_index + done + i;
            let text = corpus.sentence(idx);
            let norm = normalize_memory_text(&text);
            rows.push((format!("kstress-{idx:013}"), text, norm));
        }
        stats.gen_secs += t.elapsed().as_secs_f64();

        // 2. Embeddings (emb matrix only) — batch the whole chunk through ONNX
        //    in one pass (fastembed override); fall back to per-row on error so
        //    a single malformed text can't fail the whole batch.
        let embeddings: Vec<Option<Vec<u8>>> = if do_embed {
            let t = Instant::now();
            let texts: Vec<&str> = rows.iter().map(|(_, text, _)| text.as_str()).collect();
            let v = match embedder.embed_batch(&texts) {
                Ok(vecs) => vecs.iter().map(|vec| Some(encode_embedding(vec))).collect(),
                Err(_) => rows
                    .iter()
                    .map(|(_, text, _)| embedder.embed(text).ok().map(|vec| encode_embedding(&vec)))
                    .collect(),
            };
            stats.embed_secs += t.elapsed().as_secs_f64();
            v
        } else {
            vec![None; n]
        };

        // 3. One transaction for the whole batch.
        let t = Instant::now();
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        {
            let mut mem = tx
                .prepare_cached(
                    "INSERT INTO memories
                       (memory_id, scope, kind, text, normalized_text, confidence,
                        provenance_snapshot_json, created_at, use_count, usefulness_score,
                        embedding, embedding_model)
                     VALUES (?1,?2,?3,?4,?5,1.0,'{}',?6,0,0.0,?7,?8)",
                )
                .map_err(|e| e.to_string())?;
            let mut fts = tx
                .prepare_cached(
                    "INSERT INTO memories_fts (memory_id, text, kind, scope)
                     VALUES (?1,?2,?3,?4)",
                )
                .map_err(|e| e.to_string())?;
            for (i, (id, text, norm)) in rows.iter().enumerate() {
                let emb = embeddings[i].as_deref();
                let model = emb.map(|_| model_id.as_str());
                mem.execute(params![id, scope, kind, text, norm, created, emb, model])
                    .map_err(|e| e.to_string())?;
                fts.execute(params![id, text, kind, scope])
                    .map_err(|e| e.to_string())?;
            }
        }
        tx.commit().map_err(|e| e.to_string())?;
        stats.sql_secs += t.elapsed().as_secs_f64();

        done += n;
        stats.inserted += n;

        // Fold the WAL back into the main db file periodically so it can't grow
        // unbounded on a long seed (keeps reads fast + db-size measurement sane).
        since_checkpoint += n;
        if since_checkpoint >= 200_000 {
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);").ok();
            since_checkpoint = 0;
        }
    }
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);").ok();

    Ok(stats)
}

fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}
