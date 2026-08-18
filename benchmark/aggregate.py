#!/usr/bin/env python3
"""Aggregate per-case benchmark result JSONs into report.md + report.json.

Results are grouped by their `suite` field. Each suite gets its own table and
summary block; an overall cross-suite summary closes the report.

Usage: aggregate.py <results_dir>
"""

import json
import statistics
import sys
from pathlib import Path


def load_results(directory: Path):
    results = []
    for path in sorted(directory.glob("result-*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            results.append({"case": {"id": path.stem}, "error": f"unreadable: {error}"})
    return results


# metric key -> (container, field) inside a v2 result JSON.
METRIC_PATHS = {
    "completion_score": ("metrics", "completion_score"),
    "duration_ms": ("metrics", "duration_ms"),
    "ttft_ms": ("metrics", "ttft_ms"),
    "tokens_total": ("metrics", "tokens", "total"),
    "tokens_prompt": ("metrics", "tokens", "prompt"),
    "tokens_output": ("metrics", "tokens", "output"),
    "tool_calls_count": ("metrics", "tool_calls", "count"),
    "tool_calls_success": ("metrics", "tool_calls", "success"),
    "tool_calls_error": ("metrics", "tool_calls", "error"),
}


def metric_of(result, key):
    path = METRIC_PATHS.get(key)
    if path is None:
        return result.get(key)
    value = result
    for field in path:
        if not isinstance(value, dict):
            return None
        value = value.get(field)
    if value is None:
        return None
    if (result.get("outcome") or {}).get("error"):
        # A failed case still reports wall-clock style durations when present,
        # but quality metrics are meaningless.
        if key not in ("duration_ms",):
            return None
    return value


def stats_line(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "p95": round(
            statistics.quantiles(values, n=20)[-1] if len(values) >= 2 else max(values), 2
        ),
    }


def fmt(value):
    return "-" if value is None else f"{value:g}"


def _ms_to_s(value):
    return None if value is None else round(value / 1000, 2)


def _ok_percent(result):
    calls = (result.get("metrics") or {}).get("tool_calls") or {}
    count = calls.get("count") or 0
    success = calls.get("success") or 0
    return None if count == 0 else round(success / count * 100, 1)


COLUMNS = [
    ("case", lambda r: r.get("case", {}).get("id", "?")),
    ("score", lambda r: fmt(metric_of(r, "completion_score"))),
    ("grade", lambda r: (r.get("metrics") or {}).get("completion_grade") or "-"),
    ("time_s", lambda r: fmt(_ms_to_s(metric_of(r, "duration_ms")))),
    ("ttft_s", lambda r: fmt(_ms_to_s(metric_of(r, "ttft_ms")))),
    ("tokens", lambda r: fmt(metric_of(r, "tokens_total"))),
    ("tools", lambda r: fmt(metric_of(r, "tool_calls_count"))),
    ("tool_ok%", lambda r: fmt(_ok_percent(r))),
    ("wall_s", lambda r: fmt(r.get("wall_seconds"))),
    ("status", lambda r: "ok" if not (r.get("outcome") or {}).get("error")
        else f"ERR: {r['outcome']['error'][:40]}"),
]


def collect(results, key, transform=lambda v: v):
    values = []
    for result in results:
        value = metric_of(result, key)
        if value is not None:
            values.append(transform(value))
    return values


def summarize(results):
    summary = {
        "n_cases": len(results),
        "completion_score": stats_line(collect(results, "completion_score")),
        "duration_ms": stats_line(collect(results, "duration_ms")),
        "ttft_ms": stats_line(collect(results, "ttft_ms")),
        "tokens_total": stats_line(collect(results, "tokens_total")),
        "tokens_prompt": stats_line(collect(results, "tokens_prompt")),
        "tokens_output": stats_line(collect(results, "tokens_output")),
        "tool_calls_count": stats_line(collect(results, "tool_calls_count")),
    }
    total_calls = sum(v for v in collect(results, "tool_calls_count") if v) or 0
    total_ok = sum(v for v in collect(results, "tool_calls_success") if v) or 0
    summary["tool_call_success_rate_overall"] = (
        round(total_ok / total_calls * 100, 1) if total_calls else None
    )
    return summary


def summary_lines(results, indent=""):
    lines = []
    for key in ("completion_score", "duration_ms", "ttft_ms",
                "tokens_total", "tool_calls_count"):
        values = collect(results, key)
        lines.append(
            f"{indent}- **{key}**: n={len(values)}"
            + (f" mean={statistics.fmean(values):.2f}"
               f" median={statistics.median(values):.2f}"
               f" min={min(values):g} max={max(values):g}" if values else ""))
    total_calls = sum(v for v in collect(results, "tool_calls_count") if v) or 0
    total_ok = sum(v for v in collect(results, "tool_calls_success") if v) or 0
    rate = round(total_ok / total_calls * 100, 1) if total_calls else None
    lines.append(f"{indent}- **tool_call_success_rate_overall**: "
                 f"{rate if rate is not None else '-'}%")
    return lines


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    directory = Path(sys.argv[1])
    results = load_results(directory)
    if not results:
        print("no result-*.json files found", file=sys.stderr)
        return 1

    suites = {}
    for result in results:
        suites.setdefault(result.get("suite") or "default", []).append(result)

    lines = ["# Napaxi Benchmark Report", "",
             f"- results: {len(results)} ({len(suites)} suite(s))",
             f"- source: {directory.name}", ""]

    report = {"summary": {}, "suites": {}, "results": results}
    for suite, suite_results in suites.items():
        lines.append(f"## Suite: {suite} ({len(suite_results)} cases)")
        lines.append("")
        lines.append("| " + " | ".join(name for name, _ in COLUMNS) + " |")
        lines.append("|" + "|".join("---" for _ in COLUMNS) + "|")
        for result in suite_results:
            lines.append("| " + " | ".join(str(fn(result)) for _, fn in COLUMNS) + " |")
        lines.append("")
        lines.append(f"### {suite} summary (successful cases only)")
        lines.append("")
        lines.extend(summary_lines(suite_results))
        lines.append("")
        report["suites"][suite] = summarize(suite_results)

    if len(suites) > 1:
        lines.append("## Overall (all suites)")
        lines.append("")
        lines.extend(summary_lines(results))
        report["summary"] = summarize(results)

    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
