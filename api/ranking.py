"""
FTS5 query construction + helpers — ported from the ECHR app's
`_build_fts_query` pattern. Sanitises user input so a stray paren or
quote can never break the SQL, and lets users compose AND / OR / NOT
boolean expressions naturally.

Public API
----------
build_fts_query(raw: str) -> str | None
    Return an FTS5 MATCH expression, or None if the input is empty
    after sanitisation.

parse_csv(value: str | None) -> list[str]
    Split a comma-separated query parameter, trimming + de-duping.

Note on tokenisation:
The DB uses `tokenize=porter unicode61`, so `discriminat*` (prefix
wildcard), `"reasonable accommodation"` (phrase), and `women OR girls`
(OR) all work natively. We forward those constructs untouched and
escape everything else.
"""
from __future__ import annotations

import re

# Booleans + parentheses + asterisk + quoted phrases pass through.
# Everything else gets wrapped in double quotes so FTS5 treats it as a
# literal phrase (single-token sequence). This is what we want for
# user input — they don't think in MATCH grammar.
_BOOLEAN_TOKENS = {"AND", "OR", "NOT", "NEAR"}
_TOKEN_RE = re.compile(
    r'"[^"]*"'                               # phrase, including the quotes
    r'|\([^()]*\)'                           # simple paren group
    r'|[A-Za-z][A-Za-z0-9*]*\*'              # prefix wildcard token
    r'|[A-Za-z][A-Za-z0-9_]*'                # bare word
    r'|\d+'                                  # bare number
    r'|[()]'                                 # paren on its own (we'll keep)
)

# FTS5 reserved single chars we never want raw in user input.
_DANGEROUS_CHARS = re.compile(r'[^\w*"() :\-]+')


def _sanitise_token(tok: str) -> str:
    """Take one token from the query, return a safe FTS5 fragment."""
    if not tok:
        return ""
    # Booleans pass through (uppercased).
    if tok.upper() in _BOOLEAN_TOKENS:
        return tok.upper()
    # Parens pass through.
    if tok in ("(", ")"):
        return tok
    # Already-quoted phrase: keep as-is (quotes balance was matched by regex).
    if tok.startswith('"') and tok.endswith('"'):
        # Strip embedded quotes inside the phrase to avoid breaking it.
        inner = tok[1:-1].replace('"', "")
        return f'"{inner}"' if inner else ""
    # Single-level paren group: recurse on its contents so booleans inside
    # (like "NOT (sexual)" → "NOT ( \"sexual\" )") keep their grouping.
    if tok.startswith("(") and tok.endswith(")"):
        inner = build_fts_query(tok[1:-1])
        return f"( {inner} )" if inner else ""
    # Prefix wildcard: keep the trailing *.
    if tok.endswith("*"):
        body = tok[:-1]
        body = _DANGEROUS_CHARS.sub("", body)
        return f"{body}*" if body else ""
    # Bare word: scrub then quote-and-prefix so FTS5 stems the term.
    # v19.17 (recommendation B): bare words emit `"word"*` so a query
    # like `women NOT girl` stems both sides (matches women/womens AND
    # excludes girl/girls/girlfriend). Without the trailing `*` the
    # NOT side missed plurals — a real UX gap for boolean searches.
    # Users who want strict literal match can still quote the term:
    # `"AI"` stays literal; `AI` stems to AI/AID/AIM.
    safe = _DANGEROUS_CHARS.sub("", tok)
    return f'"{safe}"*' if safe else ""


def build_fts_query(raw: str | None) -> str | None:
    """Sanitise + return an FTS5 MATCH expression, or None if empty.

    Examples
    --------
    >>> build_fts_query('reasonable accommodation')
    '"reasonable"* "accommodation"*'
    >>> build_fts_query('"AI bias"')
    '"AI bias"'
    >>> build_fts_query('trafficking AND children NOT (sexual)')
    '"trafficking"* AND "children"* NOT ( "sexual"* )'
    >>> build_fts_query('discriminat*')
    'discriminat*'
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    tokens = _TOKEN_RE.findall(raw)
    parts = [s for s in (_sanitise_token(t) for t in tokens) if s]
    if not parts:
        return None
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers used by /api/search to parse comma-separated query params.
# ---------------------------------------------------------------------------
def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def column_weights() -> str:
    """BM25F column weights for paragraphs_fts.
    title = 8, treaty_committee = 4, text = 1 — the heavier the column,
    the more a hit there pushes the row up the ranking. ECHR uses 10/4/1;
    we tone the title weight down because our titles are shorter
    ("GC6: equality and non-discrimination") so they would otherwise win
    every match in their own document."""
    return "8.0, 4.0, 1.0"
