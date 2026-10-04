"""Readable summaries of complete or partial JSON/JSON.gz benchmark evidence."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def load(path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def summary(paths):
    lines = [
        "# Archived chirp evidence",
        "",
        "Rows retain all failures and expected case IDs. Stress characterization is separate from recovery acceptance.",
        "",
    ]
    for path in paths:
        report = load(path)
        rows = report["results"]
        standard = [r for r in rows if r.get("acceptance_class") == "standard"]
        lines += [
            f"## {path.name}",
            "",
            f"Completed {len(rows)}/{len(report['expected_case_ids'])} expected rows; complete={report['complete']}.",
            f"Recorded source revision: `{report.get('revision', 'unrecorded')}`.",
            "",
        ]
        if standard:
            lines += [
                f"Standard recovery: {sum(bool(r['accepted']) for r in standard)}/{len(standard)} pass.",
                "",
            ]
        counts = Counter((r.get("acceptance_class", "stress"), r["status"]) for r in rows)
        lines += ["| Class | Status | Count |", "| --- | --- | ---: |"]
        lines += [
            f"| {family} | {status} | {count} |"
            for (family, status), count in sorted(counts.items())
        ]
        failures = [r for r in rows if r.get("accepted") is False]
        if failures:
            lines += [
                "",
                "Unsuccessful required cases:",
                "",
                "| Case | Status | Mic RMS (m) | Source RMS (m) | Unresolved |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for r in failures:

                def metric(name):
                    v = r.get(name)
                    return "unavailable" if v is None else f"{v:.6g}"

                lines.append(
                    f"| `{r['case_id']}` | {r['status']} | {metric('microphone_rms_error_m')} | {metric('source_rms_error_m')} | {r.get('unresolved_events', 'unreported')} |"
                )
        lines += [""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = summary(args.reports)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
