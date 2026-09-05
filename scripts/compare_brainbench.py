"""Paired, reader-free BrainBench runs through the same harness and fixture.

Standard library only. Alternates baseline/candidate order and averages repeats
within each scenario before estimating uncertainty. Never starts a reader or
write-precision generation task. Timings include CLI startup, seeding and queries.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

OFFLINE_DIMENSIONS = {
    "retrieval", "dedup", "importance", "forgetting", "calibration",
    "poisoning", "render-contract", "graph", "workflow",
}


def indexed(report):
    result = {}
    for row in report["scenarios"]:
        key = f"{row['dimension']}/{row['id']}"
        if key in result:
            raise ValueError(f"duplicate scenario identity: {key}")
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError(f"invalid score: {key}")
        result[key] = row
    return result


def compare_reports(baseline, candidate):
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("equal nonempty repeat counts are required")
    bases, candidates = list(map(indexed, baseline)), list(map(indexed, candidate))
    keys = set(bases[0])
    if not keys or any(set(run) != keys for run in bases + candidates):
        raise ValueError("scenario sets differ or are empty; comparisons must use identical fixtures")
    rows, unpaired = [], []
    for key in sorted(keys):
        if any(run[key]["skipped"] for run in bases + candidates):
            unpaired.append(key)
            continue
        a = statistics.mean(run[key]["score"] for run in bases)
        b = statistics.mean(run[key]["score"] for run in candidates)
        rows.append(dict(identity=key, dimension=bases[0][key]["dimension"],
                         baseline=a, candidate=b, delta=b-a))
    dimensions = {}
    for dimension in sorted({row["dimension"] for row in rows}):
        group = [row for row in rows if row["dimension"] == dimension]
        deltas = [row["delta"] for row in group]
        interval = None
        if len(deltas) >= 2:
            rng = random.Random(20260904)
            boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(5000))
            interval = [boot[125], boot[4874]]
        dimensions[dimension] = dict(
            n_scenarios=len(group), baseline=statistics.mean(row["baseline"] for row in group),
            candidate=statistics.mean(row["candidate"] for row in group),
            mean_delta=statistics.mean(deltas), ci95=interval,
            wins=sum(d > 0 for d in deltas), ties=sum(d == 0 for d in deltas),
            losses=sum(d < 0 for d in deltas))
    errors = lambda reports: sum(row["detail"].startswith("error:")
                                 for report in reports for row in report["scenarios"])
    return dict(by_dimension=dimensions, scenarios=rows, unpaired_scenarios=unpaired,
                baseline_errors=errors(baseline), candidate_errors=errors(candidate),
                repeats=len(baseline),
                uncertainty_note="Exploratory paired bootstrap over scenario IDs after averaging repeats; correlated task families require a separate grouped holdout.")


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def dataset_fingerprints(path):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = list(data.get("eval_fixtures", [])) + list(data.get("workflow_gen", []))
    if data.get("calibration_gen"):
        sources.append(data["calibration_gen"])
    files = {path}
    for source in sources:
        if source.get("source"):
            files.add((path.parent / source["source"]).resolve(strict=True))
        elif source.get("path"):
            files.add((path.parent / source["path"]).resolve(strict=True))
    return [fingerprint(p) for p in sorted(files)]


def markdown(result):
    lines = ["# Paired BrainBench comparison", "",
             "Same harness and fixture; run order alternates. Positive delta favors the candidate.", "",
             "| Dimension | Scenarios | Baseline | Candidate | Delta | Exploratory 95% interval |",
             "|---|---:|---:|---:|---:|---|"]
    for name, row in result["comparison"]["by_dimension"].items():
        ci = row["ci95"]
        interval = "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(f"| {name} | {row['n_scenarios']} | {row['baseline']:.3f} | {row['candidate']:.3f} | {row['mean_delta']:+.3f} | {interval} |")
    compare = result["comparison"]
    lines += ["", f"Errors: baseline {compare['baseline_errors']}, candidate {compare['candidate_errors']}.",
              f"Unpaired/skipped scenarios: {len(compare['unpaired_scenarios'])}.", "",
              compare["uncertainty_note"], "",
              "Wall times include process/model startup, corpus seeding and queries; they are not warm inference latency.", ""]
    for label in ["baseline", "candidate"]:
        values = [run["wall_seconds"] for run in result["runs"] if run["label"] == label]
        lines.append(f"{label}: mean complete-run time {statistics.mean(values):.2f} s ({len(values)} repeats).")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["kbench", "baseline", "candidate", "dataset", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--dimensions", default="retrieval,workflow,render-contract,poisoning")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    dimensions = set(args.dimensions.split(","))
    if not dimensions or not dimensions <= OFFLINE_DIMENSIONS:
        parser.error("only reader-free, non-generative dimensions are supported")
    if args.repeats < 1 or args.budget_tokens < 1 or args.timeout_seconds < 1:
        parser.error("repeats, budget and timeout must be positive")
    binaries = {label: getattr(args, label).resolve(strict=True) for label in ["baseline", "candidate"]}
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KIMETSU_USER_BRAIN="0")
    # Record only relevant non-secret overrides, never the full environment.
    overrides = {key: env[key] for key in ["KIMETSU_BRAIN_EMBEDDER", "KIMETSU_ABSTAIN_EVIDENCE",
                 "KIMETSU_DETECT_CONFLICTS", "KIMETSU_RESOLVE_CONFLICTS", "KIMETSU_INTRA_THREADS",
                 "FASTEMBED_CACHE_DIR"] if key in env}
    result = dict(schema_version=1, harness=fingerprint(args.kbench),
                  binaries={k: fingerprint(v) for k,v in binaries.items()},
                  datasets=dataset_fingerprints(args.dataset),
                  settings=dict(budget_tokens=args.budget_tokens, dimensions=sorted(dimensions),
                                jobs=1, overrides=overrides), runs=[])
    reports = {"baseline": [], "candidate": []}
    for repeat in range(args.repeats):
        for label in (["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]):
            cmd = [str(args.kbench.resolve()), "brainbench", "--dataset", str(args.dataset.resolve()),
                   "--kimetsu-binary", str(binaries[label]), "--budget-tokens", str(args.budget_tokens),
                   "--dimensions", ",".join(sorted(dimensions)), "--jobs", "1", "--output", "json"]
            print(f"repeat {repeat+1}/{args.repeats}: {label}", flush=True)
            start = time.perf_counter()
            completed = subprocess.run(cmd, env=env, capture_output=True, timeout=args.timeout_seconds,
                                       encoding="utf-8", errors="strict")
            elapsed = time.perf_counter() - start
            stem = args.out / f"{repeat+1}-{label}"
            stem.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"{label} exited {completed.returncode}; see {stem}.stderr.log")
            report = json.loads(completed.stdout)
            stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            reports[label].append(report)
            result["runs"].append(dict(label=label, repeat=repeat+1, wall_seconds=elapsed,
                                       report_file=stem.with_suffix(".json").name))
    result["comparison"] = compare_reports(reports["baseline"], reports["candidate"])
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "comparison.md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
