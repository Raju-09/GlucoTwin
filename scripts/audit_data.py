"""Dataset audit script for GlucoTwin.

Usage
-----
    python scripts/audit_data.py --help
    python scripts/audit_data.py --data-dir data/raw --out reports/audit.json
    python scripts/audit_data.py --data-dir data/raw --ts-col timestamp \
        --glucose-col glucose --patient-col patient_id

This script:
1. Discovers all supported files in the given data directory.
2. Loads each file and runs all validation checks.
3. Produces a machine-readable JSON report and a human-readable summary.
4. NEVER prints or exports identifiable patient data.
5. NEVER modifies the raw files.

All column names are configurable; the script does not assume a schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the src package is importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from glucotwin.data.load import discover_files, load_raw
from glucotwin.data.validate import run_full_validation

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _audit_file(
    path: Path,
    ts_col: str,
    glucose_col: str,
    patient_col: str | None,
) -> dict[str, Any]:
    """Load and validate a single file; return a JSON-serialisable report dict."""
    report: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
    }

    try:
        df = load_raw(path)
    except Exception as exc:
        report["status"] = "load_error"
        report["error"] = str(exc)
        logging.error("Failed to load %s: %s", path.name, exc)
        return report

    report["sha256"] = df.attrs.get("sha256", "unknown")
    report["format"] = df.attrs.get("format", "unknown")
    report["n_rows"] = len(df)
    report["n_cols"] = len(df.columns)
    report["columns"] = list(df.columns)

    # Patient count (if patient_col exists)
    if patient_col and patient_col in df.columns:
        report["n_patients"] = int(df[patient_col].nunique())
    else:
        report["n_patients"] = None

    # Timestamp range (if ts_col exists) — reported as strings, not raw values
    if ts_col in df.columns:
        try:
            ts = pd.to_datetime(df[ts_col], utc=True)
            report["timestamp_min"] = str(ts.min())
            report["timestamp_max"] = str(ts.max())
        except Exception:
            report["timestamp_min"] = None
            report["timestamp_max"] = None

    # Run validation
    val = run_full_validation(
        df,
        ts_col=ts_col,
        glucose_col=glucose_col,
        patient_col=patient_col,
    )
    report["validation_passed"] = val.passed
    report["issues"] = [
        {
            "check": i.check,
            "severity": i.severity,
            "message": i.message,
            "count": i.count,
        }
        for i in val.issues
    ]

    report["status"] = "ok" if val.passed else "issues_found"
    return report


def _print_summary(reports: list[dict[str, Any]]) -> None:
    """Print a human-readable summary table to the console."""
    table = Table(title="GlucoTwin Dataset Audit", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Patients", justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Warnings", justify="right", style="yellow")
    table.add_column("Status", style="green")

    for r in reports:
        issues = r.get("issues", [])
        n_errors = sum(1 for i in issues if i["severity"] == "error")
        n_warnings = sum(1 for i in issues if i["severity"] == "warning")
        status = r.get("status", "unknown")
        status_color = "green" if status == "ok" else "red"
        table.add_row(
            r.get("file", "?"),
            str(r.get("n_rows", "?")),
            str(r.get("n_patients", "?")),
            str(n_errors),
            str(n_warnings),
            f"[{status_color}]{status}[/{status_color}]",
        )

    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit raw dataset files for GlucoTwin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing raw data files.",
    )
    parser.add_argument(
        "--out",
        default="reports/audit.json",
        help="Path for the JSON audit report (default: reports/audit.json).",
    )
    parser.add_argument(
        "--ts-col",
        default="timestamp",
        help="Name of the timestamp column (default: timestamp).",
    )
    parser.add_argument(
        "--glucose-col",
        default="glucose",
        help="Name of the glucose value column (default: glucose).",
    )
    parser.add_argument(
        "--patient-col",
        default=None,
        help="Name of the patient identifier column (optional).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]GlucoTwin — Dataset Audit")
    console.print(f"Data directory : {data_dir}")
    console.print(f"Report output  : {out_path}")
    console.print()

    if not data_dir.exists():
        console.print(
            f"[red]Data directory does not exist: {data_dir}[/red]\n"
            "Place your downloaded dataset files in data/raw/ and try again."
        )
        return 1

    files = discover_files(data_dir)
    if not files:
        console.print(
            "[yellow]No supported files found.[/yellow] "
            "If data/raw/ is empty, download your dataset first."
        )
        return 0

    reports: list[dict[str, Any]] = []
    for path in files:
        console.print(f"Auditing: [cyan]{path.name}[/cyan]")
        report = _audit_file(
            path,
            ts_col=args.ts_col,
            glucose_col=args.glucose_col,
            patient_col=args.patient_col,
        )
        reports.append(report)

    # Write JSON report
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"files": reports}, fh, indent=2)

    # Write text summary alongside the JSON
    summary_path = out_path.with_suffix(".txt")
    with summary_path.open("w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(f"File: {r.get('file')}\n")
            fh.write(f"  Rows: {r.get('n_rows')}\n")
            fh.write(f"  Patients: {r.get('n_patients')}\n")
            fh.write(f"  Status: {r.get('status')}\n")
            for issue in r.get("issues", []):
                fh.write(f"  [{issue['severity'].upper()}] {issue['message']}\n")
            fh.write("\n")

    _print_summary(reports)
    console.print(f"\n[green]Report saved to:[/green] {out_path}")
    console.print(f"[green]Summary saved to:[/green] {summary_path}")

    # Exit 1 if any file has errors
    any_errors = any(
        any(i["severity"] == "error" for i in r.get("issues", []))
        for r in reports
    )
    return 1 if any_errors else 0


if __name__ == "__main__":
    sys.exit(main())
