# Support Triage Agent

Terminal-based agent for the HackerRank Orchestrate hackathon.

Triages support tickets across HackerRank, Claude, and Visa using only the provided corpus in `data/`.

## Architecture

```
main.py        Entry point — reads CSV, runs pipeline, writes output.csv
agent.py       Triage orchestrator (safety → retrieval → Claude → validate)
safety.py      Hard escalation rules (pre-LLM, non-overridable)
retriever.py   BM25 corpus retrieval (domain-filtered)
corpus.py      Corpus loader + tokeniser
```

### Pipeline per ticket

1. **Hard safety rules** — prompt injection, score manipulation, billing, fraud, identity theft, non-owner access, infrastructure outages, assessment rescheduling, InfoSec form requests. These fire before any LLM call and cannot be overridden.
2. **Domain-filtered BM25 retrieval** — only searches the relevant company's corpus (HackerRank / Claude / Visa). Retrieves top-6 chunks using BM25Okapi.
3. **Claude structured triage** — few-shot prompted to classify and respond using only the retrieved corpus. Outputs JSON with all 5 required fields.
4. **Output contract** — validates field values, enforces allowed enum sets, falls back to escalation on parse errors.

## Setup

```bash
pip install -r code/requirements.txt
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# Process the main support_tickets.csv → output.csv
python code/main.py

# Validate against sample tickets (has expected outputs)
python code/main.py --sample

# Custom input
python code/main.py --input path/to/tickets.csv --output path/to/output.csv
```

## Design decisions

- **BM25 over embeddings**: no embedding API key required; fast; fully offline retrieval.
- **Domain filtering**: restricts retrieval to the company's own corpus so answers are never cross-contaminated.
- **Hard safety rules first**: avoids LLM judgment on high-stakes cases (fraud, score disputes, prompt injection). Rules are regex-based, deterministic, and auditable.
- **Few-shot examples**: drawn directly from `sample_support_tickets.csv` so the model has calibrated examples.
- **temperature=0**: deterministic outputs across runs.
- **Prompt injection detection**: multilingual patterns (English, French, Spanish) covering the Visa "blocked card" ticket that contains a prompt injection attempt.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
