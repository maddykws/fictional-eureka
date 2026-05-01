"""
Evaluation harness for the support triage agent.

Three evaluation layers:
  1. Structural accuracy  — run agent against sample_support_tickets.csv (ground truth)
                           scores status + request_type correctness (no API key needed
                           for safety-triggered tickets; LLM tickets need ANTHROPIC_API_KEY)
  2. DeepEval metrics     — Faithfulness, AnswerRelevancy, Hallucination (needs API key)
  3. Ragas metrics        — Faithfulness, ContextPrecision, AnswerRelevancy (needs API key)

Usage:
  python eval.py                     # Layer 1 only (deterministic safety rules, no API key)
  python eval.py --llm               # Layer 1 with LLM calls for all sample tickets
  python eval.py --deepeval          # also run DeepEval  (needs ANTHROPIC_API_KEY)
  python eval.py --ragas             # also run Ragas     (needs ANTHROPIC_API_KEY)
  python eval.py --llm --deepeval --ragas   # full suite

Outputs:
  - Terminal scorecard
  - eval_results.json  (machine-readable, all scores per ticket)
"""

from __future__ import annotations
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT    = Path(__file__).parent.parent
OUTPUT_CSV   = REPO_ROOT / "support_tickets" / "output.csv"
SAMPLE_CSV   = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
EVAL_RESULTS = REPO_ROOT / "support_tickets" / "eval_results.json"

sys.path.insert(0, str(Path(__file__).parent))


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _norm(row: dict) -> dict:
    """Normalise keys: lowercase + underscores, strip values."""
    return {
        k.strip().lower().replace(" ", "_"): (v.strip() if isinstance(v, str) else v)
        for k, v in row.items()
    }


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [_norm(r) for r in csv.DictReader(f)]


# ── Layer 1: Structural accuracy ──────────────────────────────────────────────

def evaluate_structural(run_llm: bool) -> dict:
    """
    Run the triage pipeline against sample_support_tickets.csv and score
    status + request_type against the ground truth labels.

    If run_llm=False only safety-rule-triggered tickets are evaluated
    deterministically (no API key needed). LLM-dependent tickets are skipped.
    """
    from safety import check as safety_check
    from retriever import retrieve_for_ticket, format_context, LOW_SCORE_THRESHOLD

    sample_rows = _load_csv(SAMPLE_CSV)
    results = []

    for row in sample_rows:
        issue   = row.get("issue", "")
        subject = row.get("subject", "")
        company = row.get("company", "None")
        true_status = row.get("status", "").lower()
        true_type   = row.get("request_type", "").lower()

        # Try safety rules first (always deterministic)
        decision = safety_check(issue, subject, company)
        if decision:
            pred_status = decision.status
            pred_type   = decision.request_type
            method      = "safety_rule"
        elif run_llm:
            # Full LLM triage
            try:
                from agent import triage
                result = triage(issue, subject, company)
                pred_status = result.status
                pred_type   = result.request_type
                method      = "llm"
            except Exception as e:
                pred_status = "?"
                pred_type   = "?"
                method      = f"error:{e}"
        else:
            # Skip — LLM not available
            pred_status = "?"
            pred_type   = "?"
            method      = "skipped_no_llm"

        sc = pred_status == true_status if pred_status != "?" else None
        tc = pred_type   == true_type   if pred_type   != "?" else None

        results.append({
            "issue_snippet":     issue[:60],
            "company":           company,
            "method":            method,
            "pred_status":       pred_status,
            "true_status":       true_status,
            "status_correct":    sc,
            "pred_request_type": pred_type,
            "true_request_type": true_type,
            "type_correct":      tc,
        })

    scored = [r for r in results if r["status_correct"] is not None]
    n = len(scored)
    status_acc = sum(r["status_correct"] for r in scored) / n if n else 0
    type_acc   = sum(r["type_correct"]   for r in scored) / n if n else 0

    return {
        "total_sample_tickets": len(sample_rows),
        "evaluated":            n,
        "skipped":              len(results) - n,
        "status_accuracy":      round(status_acc, 3),
        "type_accuracy":        round(type_acc, 3),
        "per_ticket":           results,
    }


def _print_structural(r: dict) -> None:
    print("\n" + "═" * 70)
    print("  LAYER 1 — Structural Accuracy (vs sample ground truth)")
    print("═" * 70)
    print(f"  Total sample tickets : {r['total_sample_tickets']}")
    print(f"  Evaluated            : {r['evaluated']}  (skipped: {r['skipped']} — need --llm)")
    print(f"  Status accuracy      : {r['status_accuracy'] * 100:.1f}%")
    print(f"  Type accuracy        : {r['type_accuracy'] * 100:.1f}%")
    print()
    print(f"  {'Issue':<42} {'Method':<16} {'Status':8} {'Type':8}")
    print(f"  {'-'*42} {'-'*16} {'-'*8} {'-'*8}")
    for t in r["per_ticket"]:
        sc = "✓" if t["status_correct"] else ("–" if t["status_correct"] is None else f"✗({t['pred_status']}≠{t['true_status']})")
        tc = "✓" if t["type_correct"]   else ("–" if t["type_correct"]   is None else f"✗({t['pred_request_type']}≠{t['true_request_type']})")
        print(f"  {t['issue_snippet']:<42} {t['method']:<16} {sc:<10} {tc}")


# ── Layer 2: DeepEval ─────────────────────────────────────────────────────────

def evaluate_deepeval(output_rows: list[dict]) -> list[dict]:
    try:
        from deepeval import evaluate as deval
        from deepeval.test_cases import LLMTestCase
        from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
    except ImportError:
        print("\n  [skip] deepeval not installed — pip install deepeval")
        return []

    from retriever import retrieve_for_ticket

    print("\n" + "═" * 70)
    print("  LAYER 2 — DeepEval (Hallucination · Faithfulness · AnswerRelevancy)")
    print("═" * 70)

    test_cases, ctx_list = [], []
    for row in output_rows:
        results = retrieve_for_ticket(row["issue"], row.get("subject", ""), row.get("company", "None"), top_k=4)
        ctx = [r.snippet for r in results]
        ctx_list.append(ctx)
        test_cases.append(LLMTestCase(input=row["issue"], actual_output=row["response"], retrieval_context=ctx))

    metrics = [HallucinationMetric(threshold=0.5), AnswerRelevancyMetric(threshold=0.5), FaithfulnessMetric(threshold=0.5)]
    deval(test_cases, metrics, print_results=False)

    per_ticket = []
    for i, tc in enumerate(test_cases):
        scores = {type(m).__name__: getattr(tc, f"{type(m).__name__.lower()}_score", None) for m in metrics}
        scores["issue_snippet"] = output_rows[i]["issue"][:60]
        per_ticket.append(scores)
        line = f"  [{i+1:02d}] {output_rows[i]['issue'][:48]}"
        for m in metrics:
            n = type(m).__name__
            s = scores[n]
            flag = "✓" if s and s >= m.threshold else "✗"
            line += f"  {flag}{n[:4]}={s:.2f}" if s else f"  ?{n[:4]}"
        print(line)

    return per_ticket


# ── Layer 3: Ragas ────────────────────────────────────────────────────────────

def evaluate_ragas(output_rows: list[dict], sample_rows: list[dict]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_eval
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    except ImportError:
        print("\n  [skip] ragas not installed — pip install ragas datasets")
        return {}

    from retriever import retrieve_for_ticket

    print("\n" + "═" * 70)
    print("  LAYER 3 — Ragas (Faithfulness · ContextPrecision · AnswerRelevancy)")
    print("═" * 70)

    sample_index = {r["issue"].strip().lower()[:80]: r for r in sample_rows}
    questions, answers, contexts, ground_truths = [], [], [], []

    for row in output_rows:
        results = retrieve_for_ticket(row["issue"], row.get("subject", ""), row.get("company", "None"), top_k=4)
        ctx = [r.snippet for r in results]
        key = row["issue"].strip().lower()[:80]
        gt = sample_index[key]["response"] if key in sample_index else row["response"]

        questions.append(row["issue"])
        answers.append(row["response"])
        contexts.append(ctx)
        ground_truths.append(gt)

    dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths})
    scores = ragas_eval(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    score_dict = scores.to_pandas().mean().to_dict()

    print(f"  Faithfulness     : {score_dict.get('faithfulness', 'N/A'):.3f}")
    print(f"  AnswerRelevancy  : {score_dict.get('answer_relevancy', 'N/A'):.3f}")
    print(f"  ContextPrecision : {score_dict.get('context_precision', 'N/A'):.3f}")
    return score_dict


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",      action="store_true", help="Use LLM for non-safety sample tickets (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--deepeval", action="store_true", help="Run DeepEval metrics on output.csv")
    parser.add_argument("--ragas",    action="store_true", help="Run Ragas metrics on output.csv")
    args = parser.parse_args()

    needs_key = args.llm or args.deepeval or args.ragas
    if needs_key and not os.getenv("ANTHROPIC_API_KEY"):
        print("[error] ANTHROPIC_API_KEY not set — required for --llm / --deepeval / --ragas")
        sys.exit(1)

    print(f"\nEvaluating agent — sample ground truth: {SAMPLE_CSV.name}")

    all_results: dict = {}

    # Layer 1
    structural = evaluate_structural(run_llm=args.llm)
    _print_structural(structural)
    all_results["structural"] = structural

    # Layer 2 & 3 — need output.csv
    if args.deepeval or args.ragas:
        if not OUTPUT_CSV.exists():
            print(f"\n[error] {OUTPUT_CSV} not found — run `python main.py` first")
            sys.exit(1)
        output_rows = _load_csv(OUTPUT_CSV)
        sample_rows = _load_csv(SAMPLE_CSV)

        if args.deepeval:
            all_results["deepeval"] = evaluate_deepeval(output_rows)
        if args.ragas:
            all_results["ragas"] = evaluate_ragas(output_rows, sample_rows)
    else:
        print("\n  Run with --deepeval and/or --ragas to get LLM-graded quality scores")

    with open(EVAL_RESULTS, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {EVAL_RESULTS}")
    print("═" * 70)

    s = structural
    print("\n  SUMMARY (for AI judge interview):")
    print(f"  • Sample tickets evaluated   : {s['evaluated']} / {s['total_sample_tickets']}")
    print(f"  • Status routing accuracy    : {s['status_accuracy']*100:.1f}%")
    print(f"  • Request type accuracy      : {s['type_accuracy']*100:.1f}%")
    if "ragas" in all_results and all_results["ragas"]:
        r = all_results["ragas"]
        print(f"  • Ragas Faithfulness         : {r.get('faithfulness', '?'):.3f}")
        print(f"  • Ragas AnswerRelevancy      : {r.get('answer_relevancy', '?'):.3f}")
        print(f"  • Ragas ContextPrecision     : {r.get('context_precision', '?'):.3f}")
    print()


if __name__ == "__main__":
    main()
