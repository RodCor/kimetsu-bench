#!/usr/bin/env node
// Convert the BEAM repo's raw per-conversation JSON (github.com/mohammadtavakoli78/BEAM,
// chats/100K/{N}/) into the kbench BEAM driver's dataset shape:
//   { conversations: [ { id, token_bucket, chat:[{role,content}], probing:{cat:[{question,rubric}]} } ] }
//
// Source schemas (locked against the real files):
//   chat.json                 = [ { batch_number, time_anchor, turns:[ [ {role,content,time_anchor?}, ... ] ] } ]
//   probing_questions.json    = { <category>: [ { question, rubric:[str], ideal_response, ... } ] }  (10 cats x 2)
//
// The batch/message `time_anchor` (e.g. "March-15-2024") is prefixed into each
// message's content so the retrieved memory text carries the date — required for
// temporal-reasoning / event-ordering (LongMemEval lesson: the date must be
// VISIBLE in the capsule the reader sees, not just in metadata).
//
// No npm deps: uses Node 18+ global fetch. Pure JSON in, pure JSON out — the
// parquet 10M bucket on HF is a separate, heavier path (deferred).
//
// Usage:
//   node convert_beam.js [--convs 1-20] [--bucket 100k] [--out beam-100k.json] [--raw-dir ./raw]

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const a = { convs: "1-20", bucket: "100k", out: null, rawDir: null, repoPath: "chats/100K" };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (k === "--convs") a.convs = argv[++i];
    else if (k === "--bucket") a.bucket = argv[++i];
    else if (k === "--out") a.out = argv[++i];
    else if (k === "--raw-dir") a.rawDir = argv[++i];
    else if (k === "--repo-path") a.repoPath = argv[++i];
  }
  return a;
}

function expandRange(spec) {
  // "1-20" | "1,3,5" | "1-5,8"
  const out = [];
  for (const part of spec.split(",")) {
    const m = part.trim().match(/^(\d+)-(\d+)$/);
    if (m) {
      for (let i = +m[1]; i <= +m[2]; i++) out.push(i);
    } else if (/^\d+$/.test(part.trim())) {
      out.push(+part.trim());
    }
  }
  return out;
}

const RAW_BASE = "https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/main";

async function fetchText(url) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url);
      if (res.status === 404) return null; // missing file → skip
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    } catch (e) {
      if (attempt === 2) throw e;
      await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
    }
  }
}

// Load a JSON file either from a local raw dir (if --raw-dir given and present)
// or by fetching the GitHub raw URL.
async function loadJson(rawDir, repoPath, n, relPath) {
  if (rawDir) {
    const local = path.join(rawDir, `${n}`, relPath);
    if (fs.existsSync(local)) return JSON.parse(fs.readFileSync(local, "utf8"));
  }
  const url = `${RAW_BASE}/${repoPath}/${n}/${relPath}`;
  const txt = await fetchText(url);
  return txt == null ? null : JSON.parse(txt);
}

function prefixDate(content, date) {
  const c = (content || "").trim();
  if (!c) return c;
  return date ? `[${date}] ${c}` : c;
}

// Flatten chat.json batches -> turns -> messages into [{role, content}],
// baking the (message|batch) time_anchor into the content.
function flattenChat(chatJson) {
  const out = [];
  if (!Array.isArray(chatJson)) return out;
  for (const batch of chatJson) {
    const batchDate = batch && batch.time_anchor;
    const turns = (batch && batch.turns) || [];
    for (const turn of turns) {
      const msgs = Array.isArray(turn) ? turn : [turn];
      for (const m of msgs) {
        if (!m || typeof m.content !== "string") continue;
        const role = m.role === "assistant" ? "assistant" : m.role === "user" ? "user" : String(m.role || "user");
        const date = m.time_anchor || batchDate || "";
        const content = prefixDate(m.content, date);
        if (content) out.push({ role, content });
      }
    }
  }
  return out;
}

// Build probing:{cat:[{question,rubric}]} from probing_questions.json.
function buildProbing(pqJson) {
  const probing = {};
  if (!pqJson || typeof pqJson !== "object") return probing;
  for (const cat of Object.keys(pqJson)) {
    const entries = pqJson[cat];
    if (!Array.isArray(entries)) continue;
    const probes = [];
    for (const e of entries) {
      if (!e || typeof e.question !== "string") continue;
      let rubric = "";
      if (Array.isArray(e.rubric)) rubric = e.rubric.filter(Boolean).join(" ");
      else if (typeof e.rubric === "string") rubric = e.rubric;
      if (!rubric) rubric = e.ideal_response || e.ideal_answer || e.answer || "";
      probes.push({ question: e.question.trim(), rubric: String(rubric).trim() });
    }
    if (probes.length) probing[cat] = probes;
  }
  return probing;
}

async function main() {
  const args = parseArgs(process.argv);
  const ns = expandRange(args.convs);
  // Default output goes under the gitignored local/ artifacts folder (run from
  // the bench root); override with --out. Datasets are never committed.
  const outPath = args.out || path.join("local", "beam-data", `beam-${args.bucket}.json`);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });

  const conversations = [];
  let probeCount = 0;
  for (const n of ns) {
    let chatJson, pqJson;
    try {
      // chat.json (fall back to chat_trunecated.json — the repo's truncated variant).
      chatJson =
        (await loadJson(args.rawDir, args.repoPath, n, "chat.json")) ||
        (await loadJson(args.rawDir, args.repoPath, n, "chat_trunecated.json"));
      pqJson = await loadJson(args.rawDir, args.repoPath, n, "probing_questions/probing_questions.json");
    } catch (e) {
      console.error(`  conv ${n}: fetch/parse error: ${e.message} — skipped`);
      continue;
    }
    if (!chatJson || !pqJson) {
      console.error(`  conv ${n}: missing chat.json or probing_questions.json — skipped`);
      continue;
    }
    const chat = flattenChat(chatJson);
    const probing = buildProbing(pqJson);
    const nProbes = Object.values(probing).reduce((s, a) => s + a.length, 0);
    if (!chat.length || !nProbes) {
      console.error(`  conv ${n}: empty chat (${chat.length}) or probes (${nProbes}) — skipped`);
      continue;
    }
    probeCount += nProbes;
    conversations.push({ id: `beam-${args.bucket}-${n}`, token_bucket: args.bucket, chat, probing });
    console.error(`  conv ${n}: ${chat.length} turns, ${Object.keys(probing).length} categories, ${nProbes} probes`);
  }

  fs.writeFileSync(outPath, JSON.stringify({ conversations }, null, 2));
  console.error(
    `\nWrote ${conversations.length} conversation(s), ${probeCount} probes -> ${outPath} (${(fs.statSync(outPath).size / 1024).toFixed(0)} KB)`
  );
}

main().catch((e) => {
  console.error("convert_beam failed:", e);
  process.exit(1);
});
