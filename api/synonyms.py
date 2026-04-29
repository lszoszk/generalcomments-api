"""
Synonym table — bridges 'modern terminology' to UN treaty body vocabulary.

UN documents pre-2024 don't always use the words researchers (especially
journalists / NGOs / students new to the corpus) reach for first. Phrases
like "AI bias" or "deepfake" are 0-result searches even though the
treaty bodies discuss the underlying concepts under different names.

When a search returns 0 hits and the query matches a key (case-insensitive,
whitespace-normalised), the API surfaces an `also_try` hint with the
mapped UN-vocabulary terms. The frontend renders these as one-click
"Did you also try…" links.

Keep this list short and high-precision. Add a new row only when:
  1. We've seen the modern term return 0 results in user testing, AND
  2. The mapped terms appear in our corpus with reasonable hit counts.

We treat each value as an OR group. The frontend constructs its own
search URL.
"""
from __future__ import annotations

import re

# Lowercase keys (normalised), each maps to a list of suggested queries.
_SYNONYMS: dict[str, list[str]] = {
    # AI / automated decision-making cluster
    "ai bias": [
        "algorithmic discrimination",
        "automated decision-making",
        "profiling",
    ],
    "ai discrimination": [
        "algorithmic discrimination",
        "automated decision-making",
    ],
    "artificial intelligence bias": [
        "algorithmic discrimination",
        "automated decision-making",
    ],
    "deepfake": [
        "image-based abuse",
        "digital violence",
        "manipulated media",
    ],
    "facial recognition": [
        "biometric surveillance",
        "remote biometric identification",
        "biometric data",
    ],
    "generative ai": [
        "automated decision-making",
        "algorithmic systems",
    ],
    "machine learning": [
        "algorithmic systems",
        "automated decision-making",
    ],

    # Surveillance / privacy cluster
    "online tracking": [
        "surveillance",
        "interception of communications",
        "right to privacy",
    ],
    "spyware": [
        "surveillance",
        "interception of communications",
    ],
    "data broker": [
        "personal data",
        "data protection",
    ],

    # Online violence / safety cluster
    "doxing": [
        "online harassment",
        "image-based abuse",
        "digital violence",
    ],
    "cyberbullying": [
        "online harassment",
        "violence against children",
        "digital environment",
    ],
    "online hate": [
        "incitement to hatred",
        "racist hate speech",
        "online harassment",
    ],
    "revenge porn": [
        "image-based abuse",
        "violence against women",
        "digital violence",
    ],

    # Misc
    "gig worker": [
        "informal economy",
        "platform workers",
        "decent work",
    ],
    "platform worker": [
        "informal economy",
        "decent work",
    ],
    "climate refugee": [
        "climate change",
        "internally displaced persons",
        "non-refoulement",
    ],
}


def _normalise(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).strip("\"'")


def lookup_synonyms(query: str) -> list[str]:
    """Return alternate-vocabulary suggestions, or [] if none.

    Match is exact (case- and whitespace-insensitive) on the user query
    after stripping leading/trailing quotes. We don't fuzzy-match because
    a near-miss would surface the wrong cluster (e.g. "facial scan" → the
    biometric list is right; "facial cream" → wrong).
    """
    if not query:
        return []
    return _SYNONYMS.get(_normalise(query), [])
