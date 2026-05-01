"""
Corpus loader — reads all markdown files from data/ and returns
structured chunks indexed by domain and subdomain.
"""

from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass, field

DATA_ROOT = Path(__file__).parent.parent / "data"

DOMAIN_MAP = {
    "hackerrank": "HackerRank",
    "claude":     "Claude",
    "visa":       "Visa",
}


@dataclass
class CorpusChunk:
    domain:    str          # "HackerRank" | "Claude" | "Visa"
    subdomain: str          # e.g. "screen", "privacy-and-legal"
    path:      str          # relative path for traceability
    title:     str
    content:   str          # cleaned text
    tokens:    list[str] = field(default_factory=list)  # for BM25


def _strip_frontmatter(text: str) -> tuple[str, str]:
    """Return (title, body) with YAML frontmatter removed."""
    title = ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end]
            body = text[end + 4:].strip()
            m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)
            if m:
                title = m.group(1).strip()
            return title, body
    return title, text


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus(domains: list[str] | None = None) -> list[CorpusChunk]:
    """
    Load all corpus chunks.  Pass domains=["hackerrank"] to restrict.
    domain values: "hackerrank", "claude", "visa"
    """
    chunks: list[CorpusChunk] = []

    for domain_dir in sorted(DATA_ROOT.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain_key = domain_dir.name.lower()
        if domains and domain_key not in domains:
            continue
        domain_label = DOMAIN_MAP.get(domain_key, domain_key.title())

        for md_file in sorted(domain_dir.rglob("*.md")):
            rel = md_file.relative_to(DATA_ROOT)
            parts = rel.parts  # e.g. ("hackerrank", "screen", "file.md")
            subdomain = parts[1] if len(parts) > 2 else domain_key

            try:
                raw = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            title, body = _strip_frontmatter(raw)
            if not title:
                # Try to extract from first heading
                m = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
                title = m.group(1).strip() if m else md_file.stem

            # Remove image tags, URLs in markdown, HTML comments
            body = re.sub(r'!\[.*?\]\(.*?\)', '', body)
            body = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL)
            body = body.strip()

            if not body:
                continue

            chunk = CorpusChunk(
                domain=domain_label,
                subdomain=subdomain,
                path=str(rel),
                title=title,
                content=body,
                tokens=_tokenise(title + " " + body),
            )
            chunks.append(chunk)

    return chunks


def get_domain_for_company(company: str) -> list[str]:
    """Map ticket company field to corpus domain keys."""
    mapping = {
        "hackerrank": ["hackerrank"],
        "claude":     ["claude"],
        "visa":       ["visa"],
        "none":       ["hackerrank", "claude", "visa"],
    }
    return mapping.get(company.strip().lower(), ["hackerrank", "claude", "visa"])
