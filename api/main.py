"""
unhrdb-api — FastAPI service powering the UN Human Rights Database.

Read-only HTTP/JSON over a SQLite + FTS5 corpus. Mirrors the route
shape of the sibling echr-api / uhri-dataset-api on the same VM.

  GET /health                           liveness probe (used by docker)
  GET /api/stats                        corpus counts + freshness
  GET /api/facets                       filter chips (treaties, mandates, …)
  GET /api/search                       paragraph-level full-text search
  GET /api/document/{doc_id}            full document (all paragraphs)
  GET /api/paragraph/{para_id}          single paragraph + parent doc
  GET /api/browse                       paginated catalogue list
  POST /api/feedback                    write-side: report a problem

CORS is permissive for the GH-Pages origin
(https://lszoszk.github.io). Tighten / parameterise if we ever expose
this anywhere else.

Environment:
  UNHRDB_DB_PATH       default /data/unhrdb.sqlite3
  UNHRDB_FEEDBACK_DIR  default /feedback
  UNHRDB_VERSION       tag string surfaced in /health and /api/stats

Note: we deliberately do NOT use `from __future__ import annotations`
in this file. FastAPI + pydantic 2 + slowapi together cannot resolve
ForwardRef'd type hints when the limiter wraps the function, even when
the class is declared earlier in the module. Keeping annotations
evaluated eagerly side-steps the issue cleanly.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ranking import build_fts_query, column_weights, parse_csv
from synonyms import lookup_synonyms

# Per-endpoint Cache-Control max-age in seconds. The DB is rebuilt per
# deploy, so a few minutes of staleness is fine for everything except
# /health.  Search results don't carry user-specific data, so a public
# cache is safe — nginx in front will benefit from this too.
CACHE_HEADERS = {
    "/health":           "no-cache",
    "/api/stats":        "public, max-age=300",
    "/api/facets":       "public, max-age=600",
    "/api/search":       "public, max-age=120",
    "/api/document":     "public, max-age=600",
    "/api/paragraph":    "public, max-age=600",
    "/api/browse":       "public, max-age=300",
}


def _set_cache(response: Response, key: str) -> None:
    """Stamp the appropriate Cache-Control header on the response."""
    if response is None:
        return
    response.headers["Cache-Control"] = CACHE_HEADERS.get(key, "no-cache")


DB_PATH = os.getenv("UNHRDB_DB_PATH", "/data/unhrdb.sqlite3")
FEEDBACK_DIR = os.getenv("UNHRDB_FEEDBACK_DIR", "/feedback")
APP_VERSION = os.getenv("UNHRDB_VERSION", "v18-sprint2")

# Pydantic body model for /api/feedback. Declared at module top so
# FastAPI can fully resolve the type when it builds the OpenAPI schema
# (a ForwardRef to a class declared further down breaks pydantic 2's
# strict schema-building, even with Body() annotation).
class FeedbackBody(BaseModel):
    kind: str = Field(..., max_length=32)         # "bug" | "data" | "feature" | "other"
    paraId: Optional[str] = Field(None, max_length=200)
    docId: Optional[str] = Field(None, max_length=200)
    message: str = Field(..., min_length=4, max_length=2000)
    contact: Optional[str] = Field(None, max_length=120)


app = FastAPI(
    title="UN Human Rights Database API",
    version=APP_VERSION,
    description="Paragraph-level FTS5 over treaty-body General Comments, "
    "jurisprudence and Special Procedures reports.",
)

# Sprint 2: rate limit. Keys on remote IP (nginx forwards X-Forwarded-For,
# slowapi reads it via get_remote_address). Tight on /api/feedback (write
# path), looser on the read endpoints.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lszoszk.github.io",
        "http://localhost:8765",          # local dev mirror of static site
        "http://127.0.0.1:8765",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)


# ---------------------------------------------------------------------------
# DB connection (per-request, autocommit, read-only).
# ---------------------------------------------------------------------------
@contextmanager
def db_cursor():
    # Read-only via URI mode=ro AND immutable=1.
    # ─ mode=ro alone fails on a :ro bind mount because SQLite still
    #   tries to create -shm / -wal side files for WAL bookkeeping
    #   (the DB was built with journal_mode=WAL, leaving a WAL marker
    #    in the file even after a clean close).
    # ─ immutable=1 tells SQLite the file will never change during the
    #   connection's lifetime, so it skips all WAL/SHM management. This
    #   is exactly true for our deploy model (we rebuild the DB locally
    #   and rsync it; the container is restarted).
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro&immutable=1",
        uri=True, check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    try:
        yield conn.cursor()
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def manifest_path() -> str:
    return os.path.join(os.path.dirname(DB_PATH), "manifest.json")


def load_manifest() -> dict:
    try:
        with open(manifest_path(), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def doc_label(d: dict) -> str:
    """Headline text for a document — mirrors v16 frontend formatting so
    the API's snippet/result view matches the static-site cards."""
    if d.get("type") == "jur":
        return f"{d.get('treaty') or d.get('committee') or ''} · {d.get('signature', '')} · {d.get('country', '')}".strip(" ·")
    if d.get("type") == "sp":
        return f"{d.get('mandate', '')} · {d.get('name_short') or d.get('name', '')}".strip(" ·")
    return f"{d.get('committee', '')} · {d.get('name_short') or d.get('name', '')}".strip(" ·")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health(response: Response):
    _set_cache(response, "/health")
    try:
        with db_cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM paragraphs LIMIT 1")
            n = cur.fetchone()["n"]
        return {"status": "ok", "version": APP_VERSION, "paragraphs": n}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")


# ---------------------------------------------------------------------------
# /api/stats — used by the freshness card on the About page.
# ---------------------------------------------------------------------------
@app.get("/api/stats")
def stats(response: Response):
    _set_cache(response, "/api/stats")
    with db_cursor() as cur:
        cur.execute(
            "SELECT type, count(*) AS n_docs, sum(paragraph_count) AS n_paras "
            "FROM documents GROUP BY type"
        )
        per_type = [row_to_dict(r) for r in cur.fetchall()]
        cur.execute("SELECT count(*) AS n FROM paragraphs")
        total_paras = cur.fetchone()["n"]
    manifest = load_manifest()
    return {
        "version": APP_VERSION,
        "manifest": manifest,
        "totalParagraphs": total_paras,
        "byType": {row["type"]: {"documents": row["n_docs"], "paragraphs": row["n_paras"]} for row in per_type},
    }


# ---------------------------------------------------------------------------
# /api/facets — populates the left-rail filter chips on the SPA.
# ---------------------------------------------------------------------------
@app.get("/api/facets")
def facets(response: Response, scope: str = Query("all", pattern="^(all|gc|jur|sp)$")):
    _set_cache(response, "/api/facets")
    type_filter, params = _scope_clause(scope)
    with db_cursor() as cur:
        # Treaties (JUR + GC scopes both use it; SP has no treaty)
        cur.execute(
            f"SELECT treaty AS value, count(*) AS count FROM documents "
            f"WHERE treaty IS NOT NULL {type_filter} GROUP BY treaty ORDER BY count DESC",
            params,
        )
        treaties = [row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            f"SELECT committee AS value, count(*) AS count FROM documents "
            f"WHERE committee IS NOT NULL {type_filter} GROUP BY committee ORDER BY count DESC",
            params,
        )
        committees = [row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            f"SELECT mandate AS value, count(*) AS count FROM documents "
            f"WHERE mandate IS NOT NULL {type_filter} GROUP BY mandate ORDER BY count DESC",
            params,
        )
        mandates = [row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            f"SELECT country AS value, count(*) AS count FROM documents "
            f"WHERE country IS NOT NULL {type_filter} GROUP BY country ORDER BY count DESC",
            params,
        )
        countries = [row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            f"SELECT outcome AS value, count(*) AS count FROM documents "
            f"WHERE outcome IS NOT NULL {type_filter} GROUP BY outcome ORDER BY count DESC",
            params,
        )
        outcomes = [row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            f"SELECT pl.value, count(*) AS count "
            f"FROM paragraph_label pl JOIN paragraphs p ON p.rowid = pl.rowid "
            f"JOIN documents d ON d.doc_id = p.doc_id "
            f"WHERE 1=1 {type_filter.replace('AND type', 'AND d.type')} "
            f"GROUP BY pl.value ORDER BY count DESC",
            params,
        )
        labels = [row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            f"SELECT year, count(*) AS count FROM documents "
            f"WHERE year IS NOT NULL {type_filter} GROUP BY year ORDER BY year ASC",
            params,
        )
        year_rows = [row_to_dict(r) for r in cur.fetchall()]
        years = {
            "min": year_rows[0]["year"] if year_rows else None,
            "max": year_rows[-1]["year"] if year_rows else None,
            "histogram": year_rows,
        }

    return {
        "treaties": treaties,
        "committees": committees,
        "mandates": mandates,
        "countries": countries,
        "outcomes": outcomes,
        "labels": labels,
        "years": years,
    }


def _scope_clause(scope: str) -> tuple[str, list[Any]]:
    """Return ('AND type IN (?, ?)', [...]) for the documents table."""
    if scope == "all":
        return "", []
    return "AND type = ?", [scope]


# ---------------------------------------------------------------------------
# /api/search — the workhorse.
# ---------------------------------------------------------------------------
@app.get("/api/search")
@limiter.limit("120/minute")
def search(
    request: Request,
    response: Response,
    q: str = Query("", description="Free-text query (FTS5 syntax allowed)"),
    scope: str = Query("all", pattern="^(all|gc|jur|sp)$"),
    treaties: Optional[str] = None,
    committees: Optional[str] = None,
    mandates: Optional[str] = None,
    countries: Optional[str] = None,
    outcomes: Optional[str] = None,
    labels: Optional[str] = None,
    year_from: Optional[int] = Query(None, ge=1900, le=2100),
    year_to: Optional[int] = Query(None, ge=1900, le=2100),
    sort: str = Query("relevance", pattern="^(relevance|date_desc|date_asc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    _set_cache(response, "/api/search")
    t0 = time.perf_counter()

    fts_expr = build_fts_query(q) if q else None
    treaty_list = parse_csv(treaties)
    committee_list = parse_csv(committees)
    mandate_list = parse_csv(mandates)
    country_list = parse_csv(countries)
    outcome_list = parse_csv(outcomes)
    label_list = parse_csv(labels)

    # Build dynamic SQL. We start with a join over documents so all
    # filter chips work whether or not `q` is set.
    select_cols = (
        "p.rowid, p.para_id, p.doc_id, p.idx, p.n, p.section, p.text, "
        "d.type, d.treaty, d.committee, d.mandate, d.country, d.year, "
        "d.adoption_date, d.signature, d.outcome, d.name, d.name_short"
    )
    joins = ["JOIN documents d ON d.doc_id = p.doc_id"]
    where: list[str] = []
    params: list[Any] = []

    if fts_expr:
        joins.append("JOIN paragraphs_fts ON paragraphs_fts.rowid = p.rowid")
        where.append("paragraphs_fts MATCH ?")
        params.append(fts_expr)

    if scope != "all":
        where.append("d.type = ?")
        params.append(scope)

    def in_clause(col: str, values: list[str]):
        ph = ",".join("?" for _ in values)
        where.append(f"{col} IN ({ph})")
        params.extend(values)

    if treaty_list:    in_clause("d.treaty", treaty_list)
    if committee_list: in_clause("d.committee", committee_list)
    if mandate_list:   in_clause("d.mandate", mandate_list)
    if country_list:   in_clause("d.country", country_list)
    if outcome_list:   in_clause("d.outcome", outcome_list)
    if year_from is not None:
        where.append("d.year >= ?"); params.append(year_from)
    if year_to is not None:
        where.append("d.year <= ?"); params.append(year_to)
    if label_list:
        joins.append("JOIN paragraph_label pl ON pl.rowid = p.rowid")
        in_clause("pl.value", label_list)

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    join_sql = " " + " ".join(joins)

    # Sorting. For relevance we use the FTS5 bm25() function with our
    # column weights; date_desc / date_asc fall back to documents.year.
    if sort == "relevance" and fts_expr:
        order_sql = f" ORDER BY bm25(paragraphs_fts, {column_weights()}) ASC"
    elif sort == "date_asc":
        order_sql = " ORDER BY d.year ASC, p.idx ASC"
    else:
        order_sql = " ORDER BY d.year DESC, p.idx ASC"

    offset = (page - 1) * page_size

    # Sprint 2 optimisation: collapse count + breakdown into one query
    # (was three separate queries in Sprint 1). Page slice still runs
    # separately because FTS5's bm25() / snippet() auxiliary functions
    # can't share a SELECT with window functions. So 2 FTS scans total
    # instead of 3 — drops the worst-case JUR query from ~900 ms to
    # ~600 ms.
    with db_cursor() as cur:
        # 1) Count + per-scope breakdown — single scan, no bm25/snippet.
        count_sql = (
            f"SELECT count(*) AS total, "
            f"sum(CASE WHEN d.type='gc'  THEN 1 ELSE 0 END) AS gc, "
            f"sum(CASE WHEN d.type='jur' THEN 1 ELSE 0 END) AS jur, "
            f"sum(CASE WHEN d.type='sp'  THEN 1 ELSE 0 END) AS sp "
            f"FROM paragraphs p{join_sql}{where_sql}"
        )
        cur.execute(count_sql, params)
        c = cur.fetchone()
        total = c["total"] or 0
        breakdown = {"gc": c["gc"] or 0, "jur": c["jur"] or 0, "sp": c["sp"] or 0}

        # 2) Page slice. SELECT shape differs depending on whether FTS5
        # is involved (snippet() + bm25() are only valid when MATCH ran).
        if fts_expr:
            page_sql = (
                f"SELECT {select_cols}, "
                f"snippet(paragraphs_fts, 2, '<mark>', '</mark>', '…', 24) AS snippet, "
                f"bm25(paragraphs_fts, {column_weights()}) AS score "
                f"FROM paragraphs p{join_sql}{where_sql}{order_sql} LIMIT ? OFFSET ?"
            )
        else:
            page_sql = (
                f"SELECT {select_cols}, NULL AS snippet, NULL AS score "
                f"FROM paragraphs p{join_sql}{where_sql}{order_sql} LIMIT ? OFFSET ?"
            )
        cur.execute(page_sql, params + [page_size, offset])
        rows = [row_to_dict(r) for r in cur.fetchall()]

    # Sprint 2: surface synonym suggestions when the query returned zero
    # hits. Keeps the cost zero on the happy path (lookup is dict-of-strings).
    also_try: list[str] = []
    if total == 0 and q:
        also_try = lookup_synonyms(q)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "query": q,
        "ftsExpr": fts_expr,
        "scope": scope,
        "total": total,
        "page": page,
        "pageSize": page_size,
        "tookMs": elapsed_ms,
        "breakdown": breakdown,
        "hits": rows,
        "alsoTry": also_try,
    }


# ---------------------------------------------------------------------------
# /api/document/{doc_id}
# ---------------------------------------------------------------------------
@app.get("/api/document/{doc_id}")
def get_document(doc_id: str, response: Response):
    _set_cache(response, "/api/document")
    with db_cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        d = cur.fetchone()
        if not d:
            raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
        doc = row_to_dict(d)
        # Hydrate JSON columns
        for k in ("case_labels", "articles_cited"):
            if doc.get(k):
                try: doc[k] = json.loads(doc[k])
                except Exception: pass

        cur.execute(
            "SELECT para_id, idx, n, section, text FROM paragraphs WHERE doc_id = ? ORDER BY idx ASC",
            (doc_id,),
        )
        paragraphs = [row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT DISTINCT pl.value FROM paragraph_label pl "
            "JOIN paragraphs p ON p.rowid = pl.rowid WHERE p.doc_id = ?",
            (doc_id,),
        )
        labels = [r["value"] for r in cur.fetchall()]

    return {"document": doc, "paragraphs": paragraphs, "labels": labels}


# ---------------------------------------------------------------------------
# /api/paragraph/{para_id}
# ---------------------------------------------------------------------------
@app.get("/api/paragraph/{para_id}")
def get_paragraph(para_id: str, response: Response):
    _set_cache(response, "/api/paragraph")
    with db_cursor() as cur:
        cur.execute(
            "SELECT p.*, d.type, d.treaty, d.committee, d.mandate, d.country, "
            "       d.year, d.signature, d.name, d.name_short, d.outcome, d.link "
            "FROM paragraphs p JOIN documents d ON d.doc_id = p.doc_id "
            "WHERE p.para_id = ?",
            (para_id,),
        )
        r = cur.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail=f"Unknown para_id: {para_id}")
        para = row_to_dict(r)

        cur.execute(
            "SELECT value FROM paragraph_label WHERE rowid = (SELECT rowid FROM paragraphs WHERE para_id = ?)",
            (para_id,),
        )
        para["labels"] = [row["value"] for row in cur.fetchall()]
    return para


# ---------------------------------------------------------------------------
# /api/browse — catalogue list, no FTS, server-side filters.
# ---------------------------------------------------------------------------
@app.get("/api/browse")
def browse(
    response: Response,
    scope: str = Query("all", pattern="^(all|gc|jur|sp)$"),
    treaty: Optional[str] = None,
    committee: Optional[str] = None,
    country: Optional[str] = None,
    year_from: Optional[int] = Query(None, ge=1900, le=2100),
    year_to: Optional[int] = Query(None, ge=1900, le=2100),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort: str = Query("year_desc", pattern="^(year_desc|year_asc|name_asc|signature_asc)$"),
):
    _set_cache(response, "/api/browse")
    where: list[str] = []
    params: list[Any] = []
    if scope != "all":
        where.append("type = ?"); params.append(scope)
    if treaty:    where.append("treaty = ?"); params.append(treaty)
    if committee: where.append("committee = ?"); params.append(committee)
    if country:   where.append("country = ?"); params.append(country)
    if year_from is not None: where.append("year >= ?"); params.append(year_from)
    if year_to   is not None: where.append("year <= ?"); params.append(year_to)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    order_map = {
        "year_desc": "year DESC, signature ASC",
        "year_asc":  "year ASC, signature ASC",
        "name_asc":  "name_short ASC",
        "signature_asc": "signature ASC",
    }
    order_sql = f" ORDER BY {order_map[sort]}"

    offset = (page - 1) * page_size
    with db_cursor() as cur:
        cur.execute(f"SELECT count(*) AS n FROM documents{where_sql}", params)
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT * FROM documents{where_sql}{order_sql} LIMIT ? OFFSET ?",
            params + [page_size, offset],
        )
        docs = [row_to_dict(r) for r in cur.fetchall()]

    return {"total": total, "page": page, "pageSize": page_size, "documents": docs}


# ---------------------------------------------------------------------------
# /api/feedback — minimal write endpoint for the "report a problem" flow.
# Persists to a single jsonl file in the writable mount. Tight rate
# limit (5/hour/IP) because write paths are abuse-prone. The Pydantic
# body model is declared at the top of this file so OpenAPI schema
# resolution doesn't choke on ForwardRef.
# ---------------------------------------------------------------------------
@app.post("/api/feedback")
@limiter.limit("5/hour")
def post_feedback(request: Request, body: FeedbackBody = Body(...)):
    # `Body(...)` is required because slowapi wraps the handler in
    # functools.wraps before FastAPI introspects the signature; without
    # it FastAPI reads `body` as a query parameter and rejects the JSON.
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    record = {
        "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ip":      get_remote_address(request),
        "agent":   request.headers.get("user-agent", "")[:200],
        "kind":    body.kind,
        "paraId":  body.paraId,
        "docId":   body.docId,
        "message": body.message,
        "contact": body.contact,
    }
    try:
        with open(os.path.join(FEEDBACK_DIR, "feedback.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Don't leak the host path to the client.
        raise HTTPException(status_code=500, detail="Failed to record feedback.") from exc
    return {"ok": True, "ts": record["ts"]}


# Optional dev-mode entry point — production runs uvicorn directly via Docker.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
