#!/usr/bin/env python3
"""
HackerRank Orchestrate — Support Triage Agent
Entry point.

Usage:
    python code/main.py                          # process support_tickets/support_tickets.csv
    python code/main.py --input path/to/in.csv  # custom input
    python code/main.py --sample                # run on sample_support_tickets.csv for validation

Output: support_tickets/output.csv
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Ensure code/ is on the path regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from agent import triage
from retriever import get_retriever
from corpus import get_domain_for_company

REPO_ROOT = Path(__file__).parent.parent
TICKETS_DIR = REPO_ROOT / "support_tickets"
OUTPUT_CSV = TICKETS_DIR / "output.csv"
DEFAULT_INPUT = TICKETS_DIR / "support_tickets.csv"
SAMPLE_INPUT  = TICKETS_DIR / "sample_support_tickets.csv"

OUTPUT_COLUMNS = ["issue", "subject", "company", "response", "product_area",
                  "status", "request_type", "justification"]


def _warm_retrievers() -> None:
    """Pre-build BM25 indexes for all three domains to avoid first-query lag."""
    print("  Building BM25 indexes...", end=" ", flush=True)
    for domains in [["hackerrank"], ["claude"], ["visa"], ["hackerrank", "claude", "visa"]]:
        get_retriever(domains)
    print("done.")


def _process_csv(input_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _write_output(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in OUTPUT_COLUMNS})


def run(input_path: Path, output_path: Path, verbose: bool = True) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[ERROR] ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    print(f"\nSupport Triage Agent")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print()

    _warm_retrievers()

    rows = _process_csv(input_path)
    print(f"  Processing {len(rows)} ticket(s)...\n")

    results: list[dict] = []
    for i, row in enumerate(rows, 1):
        issue   = row.get("Issue", row.get("issue", "")).strip()
        subject = row.get("Subject", row.get("subject", "")).strip()
        company = row.get("Company", row.get("company", "None")).strip()

        label = (subject or issue)[:55]
        print(f"  [{i:02d}/{len(rows)}] {company:12s} | {label}")
        t0 = time.monotonic()
        result = triage(issue, subject, company)
        elapsed = time.monotonic() - t0

        status_icon = "v" if result.status == "replied" else "^"
        print(f"           {status_icon} {result.status:10s} | {result.product_area:30s} | {result.request_type} ({elapsed:.1f}s)")

        results.append({
            "issue":         issue,
            "subject":       subject,
            "company":       company,
            "response":      result.response,
            "product_area":  result.product_area,
            "status":        result.status,
            "request_type":  result.request_type,
            "justification": result.justification,
        })

    _write_output(results, output_path)
    print(f"\n  Done. Output written to {output_path}")

    replied   = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"  Replied: {replied} | Escalated: {escalated}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--sample", action="store_true",
                        help="Run on sample_support_tickets.csv for validation")
    args = parser.parse_args()

    input_path = SAMPLE_INPUT if args.sample else args.input
    run(input_path, args.output)


if __name__ == "__main__":
    main()
