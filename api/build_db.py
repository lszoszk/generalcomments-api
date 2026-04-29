#!/usr/bin/env python3
"""
Build the unhrdb.sqlite3 database from the static-frontend corpus.

Reads:
  ../../generalcomments-repo/docs/documents.json     (GC + SP doc metadata)
  ../../generalcomments-repo/docs/corpus.json        (GC + SP paragraph bodies)
  ../../generalcomments-repo/docs/jur/documents.json (JUR case metadata)
  ../../generalcomments-repo/docs/jur/shards/*.json  (JUR paragraph shards)

Writes:
  ../data/unhrdb.sqlite3
  ../data/manifest.json   {builtAt, version, counts, source_sha}

Schema mirrors VM_DEPLOY_PLAN.md §"Target architecture" — three columns
in the FTS5 virtual table (title, treaty_committee, text) so BM25F can
weight them differently at query time.

Usage:
  python3 build_db.py                    # default paths
  python3 build_db.py --src ../../generalcomments-repo/docs --out ../data
  python3 build_db.py --pretty           # pretty-print manifest.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema  (kept in this file so deployment is self-contained — no separate
# .sql migrations to ship; the DB is rebuilt from the source-of-truth JSON
# on every deploy anyway).
# ---------------------------------------------------------------------------
SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    type            TEXT NOT NULL,        -- 'gc' | 'jur' | 'sp'
    treaty          TEXT,                 -- CRPD, CEDAW, CCPR, …
    committee       TEXT,                 -- "CAT", "CRC" or "SR Freedom of …"
    mandate         TEXT,                 -- SP only — mandate-holder name
    name            TEXT,                 -- full title
    name_short      TEXT,                 -- catalogue-row title
    signature       TEXT,                 -- UN doc symbol (CRPD/C/GC/6, A/76/380…)
    country         TEXT,                 -- JUR only — respondent state
    year            INTEGER,
    adoption_date   TEXT,                 -- "21 Nov 1997" or YYYY where unknown
    outcome         TEXT,                 -- JUR only (violation, inadmissible…)
    link            TEXT,                 -- canonical un.org / ohchr.org URL
    paragraph_count INTEGER,
    case_labels     TEXT,                 -- JSON array (JUR concerned groups)
    abstract        TEXT,                 -- short summary if present
    status          TEXT,                 -- "final" | "superseded" | "revised"
    superseded_by   TEXT,
    report_type     TEXT,                 -- SP only (annual, thematic…)
    presented       TEXT,                 -- SP only (GA session string)
    articles_cited  TEXT,                 -- JSON array
    last_verified   TEXT,                 -- YYYY-MM-DD
    word_count      INTEGER,
    label_count     INTEGER
);

CREATE TABLE IF NOT EXISTS paragraphs (
    rowid              INTEGER PRIMARY KEY,
    para_id            TEXT NOT NULL UNIQUE,    -- "<doc_id>-NNNN"
    doc_id             TEXT NOT NULL REFERENCES documents(doc_id),
    idx                INTEGER,                  -- order within doc (1-based)
    n                  TEXT,                     -- displayed paragraph number ("12", "12(a)")
    section            TEXT,                     -- sectional heading if any
    text               TEXT NOT NULL,
    -- Denormalised columns powering FTS5 BM25F:
    title              TEXT,                     -- copy of documents.name_short
    treaty_committee   TEXT                      -- "CRPD · GC6" / "CEDAW · 103/2022"
);

-- FTS5 virtual table — three columns ranked by built-in BM25 with
-- column weights set at query time via bm25(title=8, treaty=4, text=1).
CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs_fts USING fts5(
    title,
    treaty_committee,
    text,
    content       = 'paragraphs',
    content_rowid = 'rowid',
    tokenize      = 'porter unicode61'
);

-- Trigger pair: keep paragraphs_fts in sync with the underlying table.
-- (We rebuild the DB from scratch each deploy, so the after-update trigger
-- isn't strictly needed; the after-insert/delete pair is what matters
-- when we later move to incremental updates.)
CREATE TRIGGER IF NOT EXISTS paragraphs_ai AFTER INSERT ON paragraphs BEGIN
  INSERT INTO paragraphs_fts(rowid, title, treaty_committee, text)
       VALUES (new.rowid, new.title, new.treaty_committee, new.text);
END;
CREATE TRIGGER IF NOT EXISTS paragraphs_ad AFTER DELETE ON paragraphs BEGIN
  INSERT INTO paragraphs_fts(paragraphs_fts, rowid, title, treaty_committee, text)
       VALUES ('delete', old.rowid, old.title, old.treaty_committee, old.text);
END;
CREATE TRIGGER IF NOT EXISTS paragraphs_au AFTER UPDATE ON paragraphs BEGIN
  INSERT INTO paragraphs_fts(paragraphs_fts, rowid, title, treaty_committee, text)
       VALUES ('delete', old.rowid, old.title, old.treaty_committee, old.text);
  INSERT INTO paragraphs_fts(rowid, title, treaty_committee, text)
       VALUES (new.rowid, new.title, new.treaty_committee, new.text);
END;

-- Many-to-many lookups for filter chips.
CREATE TABLE IF NOT EXISTS paragraph_label (
    rowid  INTEGER NOT NULL REFERENCES paragraphs(rowid) ON DELETE CASCADE,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_committee (
    doc_id  TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    value   TEXT NOT NULL
);

-- Indexes that match the common WHERE clauses in /api/search.
CREATE INDEX IF NOT EXISTS idx_paragraphs_doc_id     ON paragraphs(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_type        ON documents(type);
CREATE INDEX IF NOT EXISTS idx_documents_year        ON documents(year);
CREATE INDEX IF NOT EXISTS idx_documents_committee   ON documents(committee);
CREATE INDEX IF NOT EXISTS idx_documents_treaty      ON documents(treaty);
CREATE INDEX IF NOT EXISTS idx_documents_country     ON documents(country);
CREATE INDEX IF NOT EXISTS idx_documents_outcome     ON documents(outcome);
CREATE INDEX IF NOT EXISTS idx_label_value           ON paragraph_label(value, rowid);
CREATE INDEX IF NOT EXISTS idx_doc_committee_value   ON document_committee(value, doc_id);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sniff_doc_committee_string(d: dict) -> str:
    """One-line label that goes into paragraphs.treaty_committee — the FTS5
    column boosted moderately for BM25F. Format mirrors the v16 result-row
    headline so the index "agrees" with what users see in result cards."""
    if d.get("type") == "jur":
        return f"{d.get('treaty') or d.get('committee') or ''} · {d.get('signature', '')}"
    if d.get("type") == "sp":
        # Mandate is the salient signal for SP rows.
        return f"{d.get('mandate', '')} · {d.get('signature', '')}".strip(" ·")
    return f"{d.get('committee') or ''} · {d.get('signature', '')}"


def insert_document(cur, d: dict):
    cur.execute(
        """INSERT OR REPLACE INTO documents (
            doc_id, type, treaty, committee, mandate, name, name_short, signature,
            country, year, adoption_date, outcome, link, paragraph_count,
            case_labels, abstract, status, superseded_by, report_type, presented,
            articles_cited, last_verified, word_count, label_count
        ) VALUES (?,?,?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?)""",
        (
            d.get("docId"),
            d.get("type"),
            d.get("treaty"),
            d.get("committee"),
            d.get("mandate"),
            d.get("name"),
            d.get("nameShort"),
            d.get("signature") or d.get("symbol"),
            d.get("country"),
            d.get("year"),
            d.get("adoptionDate"),
            d.get("outcome"),
            d.get("link"),
            d.get("paragraphCount"),
            json.dumps(d.get("caseLabels") or [], ensure_ascii=False) if d.get("caseLabels") else None,
            d.get("abstract"),
            d.get("status"),
            d.get("supersededBy"),
            d.get("reportType"),
            d.get("presented"),
            json.dumps(d.get("articlesCited") or [], ensure_ascii=False) if d.get("articlesCited") else None,
            d.get("lastVerifiedAt"),
            d.get("wordCount"),
            d.get("labelCount"),
        ),
    )
    # Many-to-many committee list (mainly defensive — most docs have a
    # single committee, but SP can have e.g. ["SR Privacy", "SR Religion"]).
    for committee in d.get("committees") or []:
        cur.execute(
            "INSERT INTO document_committee (doc_id, value) VALUES (?, ?)",
            (d.get("docId"), committee),
        )


def insert_paragraph(cur, p: dict, doc: dict):
    """Insert one paragraph + its labels. Returns rowid."""
    cur.execute(
        """INSERT INTO paragraphs (
            para_id, doc_id, idx, n, section, text, title, treaty_committee
        ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            p.get("id") or p.get("paragraphId"),
            p.get("docId"),
            p.get("idx"),
            None if p.get("n") is None else str(p.get("n")),
            p.get("section"),
            p.get("text") or "",
            doc.get("nameShort") or doc.get("name") or doc.get("docId"),
            sniff_doc_committee_string(doc),
        ),
    )
    rowid = cur.lastrowid
    for label in p.get("labels") or []:
        cur.execute(
            "INSERT INTO paragraph_label (rowid, value) VALUES (?, ?)",
            (rowid, label),
        )
    return rowid


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------
def build(src: Path, out_db: Path) -> dict:
    """Build the SQLite database. Returns counts dict for the manifest."""
    if out_db.exists():
        out_db.unlink()
    out_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(out_db)
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    # 1) GC + SP documents
    print("[build] reading documents.json …")
    gc_sp_docs = read_json(src / "documents.json")
    docs_by_id: dict[str, dict] = {}
    for d in gc_sp_docs:
        if not d.get("docId"):
            continue
        docs_by_id[d["docId"]] = d
        insert_document(cur, d)

    # 2) JUR documents
    jur_docs_path = src / "jur" / "documents.json"
    if jur_docs_path.exists():
        print(f"[build] reading {jur_docs_path.relative_to(src)} …")
        for d in read_json(jur_docs_path):
            if not d.get("docId"):
                continue
            d.setdefault("type", "jur")
            docs_by_id[d["docId"]] = d
            insert_document(cur, d)

    # 3) GC + SP paragraphs (corpus.json — single big array)
    print("[build] reading corpus.json …")
    gc_sp_paragraphs = read_json(src / "corpus.json")
    inserted_paragraphs = 0
    for p in gc_sp_paragraphs:
        doc = docs_by_id.get(p.get("docId"))
        if not doc:
            continue
        insert_paragraph(cur, p, doc)
        inserted_paragraphs += 1
        if inserted_paragraphs % 5000 == 0:
            print(f"  …inserted {inserted_paragraphs} paragraphs")

    # 4) JUR paragraphs (one shard at a time so memory stays flat)
    jur_shards_dir = src / "jur" / "shards"
    if jur_shards_dir.exists():
        shard_paths = sorted(jur_shards_dir.glob("*.json"))
        print(f"[build] reading {len(shard_paths)} jurisprudence shards …")
        for shard_path in shard_paths:
            shard = read_json(shard_path)
            for p in shard.get("paragraphs", []):
                doc = docs_by_id.get(p.get("docId"))
                if not doc:
                    continue
                # JUR paragraphs in shards don't carry doc-level fields; the
                # insert_paragraph helper denormalises from `doc` so the FTS5
                # title/treaty columns get filled correctly.
                insert_paragraph(cur, p, doc)
                inserted_paragraphs += 1
            print(f"  {shard_path.name}: {inserted_paragraphs} paragraphs total")

    conn.commit()

    # 5) Optimise. The FTS5 'optimize' command merges segments — slower
    # writes after this, but read latency drops by 30-50 % on big corpora.
    print("[build] optimising FTS5 index …")
    cur.execute("INSERT INTO paragraphs_fts(paragraphs_fts) VALUES('optimize')")
    cur.execute("ANALYZE")
    conn.commit()
    conn.close()

    return {
        "documents": len(docs_by_id),
        "paragraphs": inserted_paragraphs,
        "size_bytes": out_db.stat().st_size,
    }


def write_manifest(out_dir: Path, counts: dict, db_path: Path, *, pretty: bool):
    manifest = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "builtAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "schema": "1.0",
        "tokenizer": "porter unicode61",
        "fts_columns": ["title", "treaty_committee", "text"],
        "counts": counts,
        "files": {
            "unhrdb.sqlite3": {
                "sha": sha256_file(db_path),
                "bytes": db_path.stat().st_size,
            }
        },
    }
    target = out_dir / "manifest.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2 if pretty else None, ensure_ascii=False)
    return manifest


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=here.parent / "generalcomments-repo" / "docs",
        help="Path to generalcomments-repo/docs (the static frontend's data root).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "data",
        help="Output directory; unhrdb.sqlite3 + manifest.json will land here.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print manifest.json")
    args = parser.parse_args()

    if not args.src.exists():
        print(f"❌ src not found: {args.src}", file=sys.stderr)
        return 2

    out_db = args.out / "unhrdb.sqlite3"
    print(f"[build] src = {args.src}")
    print(f"[build] out = {out_db}")
    counts = build(args.src, out_db)
    manifest = write_manifest(args.out, counts, out_db, pretty=args.pretty)

    print()
    print("✓ build complete")
    print(f"  documents:  {counts['documents']:>7,}")
    print(f"  paragraphs: {counts['paragraphs']:>7,}")
    print(f"  size:       {counts['size_bytes']/1024/1024:>7.1f} MB")
    print(f"  built at:   {manifest['builtAt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
