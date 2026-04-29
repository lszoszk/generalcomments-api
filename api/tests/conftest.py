"""
Pytest fixtures.

We build a tiny in-memory SQLite from the same schema as build_db.py and
hand it to FastAPI's TestClient via env-var override. Real corpus data
isn't needed — we synthesise a few documents and paragraphs so the
endpoints exercise the full code path without dragging the 218 MB
production DB into CI.

The full-build smoke test is separate (run_smoke.sh) and uses the real
unhrdb.sqlite3 when available.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest


# Make the api/ package importable when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# IMPORTANT: set env vars BEFORE importing main, because main reads them
# at import time. Use a temp dir that the test session cleans up.
_TMP = tempfile.mkdtemp(prefix="unhrdb-test-")
_DB_PATH = os.path.join(_TMP, "test.sqlite3")
_FB_PATH = os.path.join(_TMP, "feedback")
os.makedirs(_FB_PATH, exist_ok=True)
os.environ["UNHRDB_DB_PATH"] = _DB_PATH
os.environ["UNHRDB_FEEDBACK_DIR"] = _FB_PATH


SCHEMA = """
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY, type TEXT NOT NULL, treaty TEXT, committee TEXT,
    mandate TEXT, name TEXT, name_short TEXT, signature TEXT, country TEXT,
    year INTEGER, adoption_date TEXT, outcome TEXT, link TEXT,
    paragraph_count INTEGER, case_labels TEXT, abstract TEXT, status TEXT,
    superseded_by TEXT, report_type TEXT, presented TEXT, articles_cited TEXT,
    last_verified TEXT, word_count INTEGER, label_count INTEGER
);
CREATE TABLE paragraphs (
    rowid INTEGER PRIMARY KEY, para_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL, idx INTEGER, n TEXT, section TEXT,
    text TEXT NOT NULL, title TEXT, treaty_committee TEXT
);
CREATE VIRTUAL TABLE paragraphs_fts USING fts5(
    title, treaty_committee, text,
    content='paragraphs', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER paragraphs_ai AFTER INSERT ON paragraphs BEGIN
    INSERT INTO paragraphs_fts(rowid, title, treaty_committee, text)
        VALUES (new.rowid, new.title, new.treaty_committee, new.text);
END;
CREATE TABLE paragraph_label (rowid INTEGER, value TEXT);
CREATE TABLE document_committee (doc_id TEXT, value TEXT);
"""

# Minimal corpus: one of each scope type with a couple of paragraphs.
DOCS = [
    dict(doc_id="crpd-c-gc-6", type="gc", treaty="CRPD", committee="CRPD",
         name="General Comment No. 6 (2018) on equality and non-discrimination",
         name_short="GC6: equality and non-discrimination", signature="CRPD/C/GC/6",
         year=2018, adoption_date="26 April 2018", paragraph_count=2),
    dict(doc_id="crpd-c-34-d-103-2022", type="jur", treaty="CRPD", committee="CRPD",
         country="Spain", outcome="violation_found",
         name="Communication No. 103/2022", name_short="103/2022",
         signature="CRPD/C/34/D/103/2022", year=2022, paragraph_count=2),
    dict(doc_id="a-76-380", type="sp", mandate="SR Freedom of Religion or Belief",
         name="The Freedom of Thought", name_short="Freedom of Thought",
         signature="A/76/380", year=2021, paragraph_count=1),
]
PARAS = [
    ("crpd-c-gc-6-0001", "crpd-c-gc-6", 1, "1", None,
     "Reasonable accommodation is required to secure equality."),
    ("crpd-c-gc-6-0002", "crpd-c-gc-6", 2, "2", None,
     "Discrimination on grounds of disability includes denial of reasonable accommodation."),
    ("crpd-c-34-d-103-2022-0001", "crpd-c-34-d-103-2022", 1, "1", None,
     "The author argues that Spain failed to provide reasonable accommodation under article 12."),
    ("crpd-c-34-d-103-2022-0002", "crpd-c-34-d-103-2022", 2, "2", None,
     "The Committee finds a violation of article 12 of the Convention."),
    ("a-76-380-0001", "a-76-380", 1, "1", None,
     "The right to freedom of thought has long been overlooked in algorithmic discrimination contexts."),
]


@pytest.fixture(scope="session")
def fixture_db():
    conn = sqlite3.connect(_DB_PATH)
    conn.executescript(SCHEMA)
    cur = conn.cursor()
    for d in DOCS:
        cols = ",".join(d.keys())
        ph   = ",".join("?" for _ in d)
        cur.execute(f"INSERT INTO documents ({cols}) VALUES ({ph})", tuple(d.values()))
    for para_id, doc_id, idx, n, section, text in PARAS:
        doc = next(x for x in DOCS if x["doc_id"] == doc_id)
        cur.execute(
            "INSERT INTO paragraphs (para_id, doc_id, idx, n, section, text, title, treaty_committee) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (para_id, doc_id, idx, n, section, text,
             doc["name_short"], f"{doc.get('treaty') or doc.get('mandate')} · {doc['signature']}"),
        )
    conn.commit()
    conn.close()
    yield _DB_PATH


@pytest.fixture(scope="session")
def client(fixture_db):
    from fastapi.testclient import TestClient
    import importlib

    # Late-import so env vars are honoured.
    import main as main_module
    importlib.reload(main_module)
    return TestClient(main_module.app)
