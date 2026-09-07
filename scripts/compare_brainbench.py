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
import signal
import statistics
import subprocess
import time


def run_owned_tree(cmd, *, env, timeout):
    """Kill descendants while the parent still exists, before reaping it.

    subprocess.run kills only the parent on timeout. Inference children can
    otherwise survive, including on Windows where process groups do not die
    with their parent. The timeout applies to the whole benchmark invocation.
    """
    options = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else dict(start_new_session=True)
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               encoding="utf-8", errors="strict", **options)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:
        if os.name == "nt":
            # /T walks descendants before /F terminates their parent; do not
            # call process.kill first, which loses Windows tree ancestry.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        stdout, stderr = process.communicate(timeout=15)
        if isinstance(error, subprocess.TimeoutExpired):
            raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr) from error
        raise
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

OFFLINE_DIMENSIONS = {
    "retrieval", "dedup", "importance", "forgetting", "calibration",
    "poisoning", "render-contract", "graph", "workflow",
}

REPORT_ROW_FIELDS = {"id", "dimension", "score", "skipped", "detail"}


def validate_report(report):
    if not isinstance(report, dict) or not isinstance(report.get("scenarios"), list):
        raise ValueError("report must be an object with a scenarios array")
    configuration = report.get("session_configuration")
    if (not isinstance(configuration, dict) or configuration.get("warm_start") is not False
            or configuration.get("include_ambient") is not False):
        raise ValueError("session_configuration must confirm warm_start=false and include_ambient=false")
    for index, row in enumerate(report["scenarios"]):
        if not isinstance(row, dict):
            raise ValueError(f"scenario {index} must be an object")
        missing = REPORT_ROW_FIELDS - set(row)
        if missing:
            raise ValueError(f"scenario {index} lacks fields: {', '.join(sorted(missing))}")
        if not isinstance(row["id"], str) or not isinstance(row["dimension"], str):
            raise ValueError(f"scenario {index} identity must contain strings")
        if (isinstance(row["score"], bool) or not isinstance(row["score"], (int, float))
                or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1):
            raise ValueError(f"scenario {index} has an invalid score")
        if not isinstance(row["skipped"], bool) or not isinstance(row["detail"], str):
            raise ValueError(f"scenario {index} has invalid status fields")


def indexed(report):
    validate_report(report)
    result = {}
    for row in report["scenarios"]:
        key = f"{row['dimension']}/{row['id']}"
        if key in result:
            raise ValueError(f"duplicate scenario identity: {key}")
        result[key] = row
    return result


def measurement_summary(reports, paired_keys):
    groups = {}
    for run in reports:
        for scenario in run["scenarios"]:
            identity = f"{scenario['dimension']}/{scenario['id']}"
            if identity not in paired_keys:
                continue
            observations = scenario.get("observations", [])
            if not isinstance(observations, list):
                raise ValueError("observations must be an array")
            for index, observation in enumerate(observations):
                if not isinstance(observation, dict) or not isinstance(observation.get("query"), str):
                    raise ValueError("query observation requires a query string")
                key = (identity, index, observation["query"])
                groups.setdefault(key, []).append(observation)
    if not groups:
        return None
    metric_names = ["positive_recall_at_4", "positive_hit_at_4", "positive_mrr", "negative_injection", "stale_injection"]
    metrics = {name: [] for name in metric_names}
    first, subsequent, text_bytes, result_bytes = [], [], [], []
    working_sets, peak_working_sets = [], []
    count = 0
    for observations in groups.values():
        for name in metric_names:
            values = [o.get(name) for o in observations if o.get(name) is not None]
            if any(not isinstance(v, (int, float, bool)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f"invalid query metric {name}")
            if values:
                metrics[name].append(statistics.mean(values))
        for observation in observations:
            count += 1
            if not isinstance(observation.get("first_query"), bool):
                raise ValueError("query observation requires first-query classification")
            for field in ["latency_ms", "model_text_bytes", "mcp_result_bytes"]:
                value = observation.get(field)
                if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid query measurement {field}")
            (first if observation["first_query"] else subsequent).append(observation["latency_ms"])
            text_bytes.append(observation["model_text_bytes"])
            result_bytes.append(observation["mcp_result_bytes"])
            for field, values in [("working_set_bytes", working_sets), ("peak_working_set_bytes", peak_working_sets)]:
                value = observation.get(field)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid memory measurement {field}")
                    values.append(value)
    avg = lambda values: statistics.mean(values) if values else None
    def percentile(values, p):
        return sorted(values)[max(0, math.ceil(len(values)*p)-1)] if values else None
    return dict(unique_queries=len(groups), query_observations=count,
                positive_queries=len(metrics["positive_recall_at_4"]), negative_queries=len(metrics["negative_injection"]),
                stale_queries=len(metrics["stale_injection"]),
                positive_recall_at_4=avg(metrics["positive_recall_at_4"]),
                positive_hit_at_4=avg(metrics["positive_hit_at_4"]), positive_mrr=avg(metrics["positive_mrr"]),
                negative_injection_rate=avg(metrics["negative_injection"]), stale_injection_rate=avg(metrics["stale_injection"]),
                first_query_mean_ms=avg(first), subsequent_query_p50_ms=percentile(subsequent,.5),
                subsequent_query_p95_ms=percentile(subsequent,.95), subsequent_observations=len(subsequent),
                mean_model_text_bytes=avg(text_bytes), mean_mcp_result_bytes=avg(result_bytes),
                memory_observations=len(working_sets), mean_mcp_working_set_bytes=avg(working_sets),
                max_mcp_peak_working_set_bytes=max(peak_working_sets) if peak_working_sets else None,
                note="Quality averages repeats per query; latency percentiles pool repeated observations descriptively, not as independent evidence. First query includes model loading where applicable; server/process initialization is recorded separately.")


def compare_reports(baseline, candidate):
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("equal nonempty repeat counts are required")
    bases, candidates = list(map(indexed, baseline)), list(map(indexed, candidate))
    keys = set(bases[0])
    if not keys or any(set(run) != keys for run in bases + candidates):
        raise ValueError("scenario sets differ or are empty; comparisons must use identical fixtures")
    rows, unpaired, unpaired_details = [], [], []
    for key in sorted(keys):
        reasons = []
        for label, runs in (("baseline", bases), ("candidate", candidates)):
            for repeat, run in enumerate(runs, 1):
                row = run[key]
                if row["skipped"]:
                    reasons.append(dict(side=label, repeat=repeat, kind="skipped",
                                        detail=row["detail"]))
                elif row["detail"].startswith("error:"):
                    reasons.append(dict(side=label, repeat=repeat, kind="error",
                                        detail=row["detail"]))
        if reasons:
            unpaired.append(key)
            unpaired_details.append(dict(identity=key, reasons=reasons))
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
    paired_keys = {row["identity"] for row in rows}
    measurements = {"baseline": measurement_summary(baseline, paired_keys), "candidate": measurement_summary(candidate, paired_keys)}
    return dict(measurement_summary=measurements, by_dimension=dimensions, scenarios=rows, unpaired_scenarios=unpaired,
                unpaired_details=unpaired_details,
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
    if any(compare["measurement_summary"].values()):
        lines += ["", "Query measurements through persistent MCP (subsequent queries reuse the process):", "",
                  "| Build | Positive hit@4 | Positive recall@4 | False injection | Subsequent p50 / p95 ms | Mean MCP result bytes | Peak MCP working set MiB |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"
        for label, summary in compare["measurement_summary"].items():
            if summary is not None:
                peak = summary['max_mcp_peak_working_set_bytes']
                lines.append(f"| {label} | {fmt(summary['positive_hit_at_4'])} | {fmt(summary['positive_recall_at_4'])} | {fmt(summary['negative_injection_rate'])} | {fmt(summary['subsequent_query_p50_ms'])} / {fmt(summary['subsequent_query_p95_ms'])} | {fmt(summary['mean_mcp_result_bytes'])} | {fmt(peak / 1048576 if peak is not None else None)} |")
        lines.append("\nMeasured bytes include JSON escaping; reported token estimates are retained per query but may use different accounting rules across builds. Query timing excludes the separately recorded MCP initialization and corpus seeding.")
    return "\n".join(lines) + "\n"


def text_output(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def persist_result(path, result):
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def environment_for_side(base, threads, reranker=None, rerank_floor=None):
    result = dict(base)
    if threads == 0:
        result.pop("KIMETSU_INTRA_THREADS", None)
    elif threads is not None:
        result["KIMETSU_INTRA_THREADS"] = str(threads)
    if reranker is not None:
        result["KBENCH_RERANKER"] = reranker
    if rerank_floor is not None:
        if not 0 <= rerank_floor <= 1:
            raise ValueError("rerank floor must be finite and between 0 and 1")
        result["KBENCH_RERANK_FLOOR"] = str(rerank_floor)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["kbench", "baseline", "candidate", "dataset", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-tokens", type=int, default=512)
    parser.add_argument("--dimensions", default="retrieval,workflow,render-contract,poisoning")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--baseline-threads", type=int, help="0 unsets the override; omitted inherits environment")
    parser.add_argument("--candidate-threads", type=int, help="0 unsets the override; omitted inherits environment")
    parser.add_argument("--baseline-reranker", help="Override reranker only in baseline temporary projects")
    parser.add_argument("--candidate-reranker", help="Override reranker only in candidate temporary projects")
    parser.add_argument("--baseline-rerank-floor", type=float)
    parser.add_argument("--candidate-rerank-floor", type=float)
    args = parser.parse_args()
    if any(v is not None and not 0 <= v <= 1 for v in [args.baseline_rerank_floor, args.candidate_rerank_floor]):
        parser.error("rerank floors must be finite and between 0 and 1")
    dimensions = set(args.dimensions.split(","))
    if not dimensions or not dimensions <= OFFLINE_DIMENSIONS:
        parser.error("only reader-free, non-generative dimensions are supported")
    if args.repeats < 1 or args.budget_tokens < 1 or args.timeout_seconds < 1:
        parser.error("repeats, budget and timeout must be positive")
    if any(value is not None and not 0 <= value <= 1024 for value in [args.baseline_threads, args.candidate_threads]):
        parser.error("thread overrides must be between 0 (unset) and 1024")
    binaries = {label: getattr(args, label).resolve(strict=True) for label in ["baseline", "candidate"]}
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KIMETSU_USER_BRAIN="0")
    # Record only relevant non-secret overrides, never the full environment.
    overrides = {key: env[key] for key in ["KIMETSU_BRAIN_EMBEDDER", "KIMETSU_ABSTAIN_EVIDENCE",
                 "KIMETSU_DETECT_CONFLICTS", "KIMETSU_RESOLVE_CONFLICTS", "KIMETSU_INTRA_THREADS",
                 "FASTEMBED_CACHE_DIR", "HF_HOME", "KBENCH_RERANKER", "KBENCH_RERANK_FLOOR"] if key in env}
    result = dict(schema_version=1, status="running", harness=fingerprint(args.kbench), runner=fingerprint(Path(__file__)),
                  binaries={k: fingerprint(v) for k,v in binaries.items()},
                  datasets=dataset_fingerprints(args.dataset),
                  settings=dict(budget_tokens=args.budget_tokens, dimensions=sorted(dimensions),
                                jobs=1, warm_start=False, include_ambient=False, overrides=overrides,
                                baseline_threads=args.baseline_threads, candidate_threads=args.candidate_threads,
                                baseline_reranker=args.baseline_reranker, candidate_reranker=args.candidate_reranker,
                                baseline_rerank_floor=args.baseline_rerank_floor, candidate_rerank_floor=args.candidate_rerank_floor), runs=[])
    reports = {"baseline": [], "candidate": []}
    for repeat in range(args.repeats):
        for label in (["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]):
            cmd = [str(args.kbench.resolve()), "brainbench", "--dataset", str(args.dataset.resolve()),
                   "--kimetsu-binary", str(binaries[label]), "--budget-tokens", str(args.budget_tokens),
                   "--dimensions", ",".join(sorted(dimensions)), "--jobs", "1", "--output", "json"]
            print(f"repeat {repeat+1}/{args.repeats}: {label}", flush=True)
            start = time.perf_counter()
            stem = args.out / f"{repeat+1}-{label}"
            run_record = dict(label=label, repeat=repeat+1)
            run_env = environment_for_side(env, getattr(args, f"{label}_threads"), getattr(args, f"{label}_reranker"), getattr(args, f"{label}_rerank_floor"))
            run_record["intra_threads_override"] = run_env.get("KIMETSU_INTRA_THREADS")
            run_record["rerank_floor_override"] = run_env.get("KBENCH_RERANK_FLOOR")
            run_record["reranker_override"] = run_env.get("KBENCH_RERANKER")
            try:
                completed = run_owned_tree(cmd, env=run_env, timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                run_record["wall_seconds"] = time.perf_counter() - start
                run_record["failure"] = dict(kind="timeout", timeout_seconds=args.timeout_seconds,
                                              message=str(error))
                stem.with_suffix(".stdout.log").write_text(text_output(error.output), encoding="utf-8")
                stem.with_suffix(".stderr.log").write_text(text_output(error.stderr), encoding="utf-8")
                result["runs"].append(run_record)
                result["status"] = "incomplete"
                persist_result(args.out / "comparison.json", result)
                return 1
            elapsed = time.perf_counter() - start
            stem.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
            stem.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
            run_record["wall_seconds"] = elapsed
            if completed.returncode:
                run_record["failure"] = dict(kind="nonzero_exit",
                                              returncode=completed.returncode,
                                              message=f"{label} exited {completed.returncode}")
                result["runs"].append(run_record)
                result["status"] = "incomplete"
                persist_result(args.out / "comparison.json", result)
                return 1
            try:
                report = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                run_record["failure"] = dict(kind="invalid_json", message=str(error))
                result["runs"].append(run_record)
                result["status"] = "incomplete"
                persist_result(args.out / "comparison.json", result)
                return 1
            report_path = stem.with_suffix(".json")
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            run_record["report_file"] = report_path.name
            try:
                validate_report(report)
            except ValueError as error:
                run_record["failure"] = dict(kind="invalid_report", message=str(error))
                result["runs"].append(run_record)
                result["status"] = "incomplete"
                persist_result(args.out / "comparison.json", result)
                return 1
            reports[label].append(report)
            result["runs"].append(run_record)
            persist_result(args.out / "comparison.json", result)
    try:
        result["comparison"] = compare_reports(reports["baseline"], reports["candidate"])
    except ValueError as error:
        result["status"] = "incomplete"
        result["failure"] = dict(kind="comparison_validation", message=str(error))
        persist_result(args.out / "comparison.json", result)
        return 1
    result["status"] = "complete"
    persist_result(args.out / "comparison.json", result)
    (args.out / "comparison.md").write_text(markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
