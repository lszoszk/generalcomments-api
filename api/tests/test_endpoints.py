"""Happy-path tests for every public endpoint."""
from __future__ import annotations


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["paragraphs"] == 5         # fixture has 5 paragraphs
    assert r.headers["cache-control"] == "no-cache"


def test_stats_returns_per_type(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["totalParagraphs"] == 5
    assert body["byType"]["gc"]["paragraphs"]  == 2
    assert body["byType"]["jur"]["paragraphs"] == 2
    assert body["byType"]["sp"]["paragraphs"]  == 1
    assert "max-age=300" in r.headers["cache-control"]


def test_facets_filtered_by_scope(client):
    r = client.get("/api/facets?scope=jur")
    assert r.status_code == 200
    body = r.json()
    treaties = [t["value"] for t in body["treaties"]]
    assert "CRPD" in treaties
    assert any(c["value"] == "Spain" for c in body["countries"])
    assert any(o["value"] == "violation_found" for o in body["outcomes"])


def test_search_keyword(client):
    r = client.get("/api/search?q=reasonable+accommodation")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2
    assert all("<mark>" in (h["snippet"] or "") for h in body["hits"])
    assert body["breakdown"]["gc"] >= 1
    assert body["breakdown"]["jur"] >= 1


def test_search_boolean(client):
    r = client.get("/api/search?q=reasonable AND accommodation NOT (Spain)")
    body = r.json()
    assert body["ftsExpr"].startswith('"reasonable"')
    assert "( " in body["ftsExpr"]                 # paren group preserved
    # The Spain paragraph should be excluded.
    ids = [h["para_id"] for h in body["hits"]]
    assert "crpd-c-34-d-103-2022-0001" not in ids


def test_search_scope_filter(client):
    r = client.get("/api/search?q=violation&scope=jur")
    body = r.json()
    assert body["total"] >= 1
    assert body["breakdown"]["gc"] == 0
    assert body["breakdown"]["sp"] == 0
    assert all(h["type"] == "jur" for h in body["hits"])


def test_search_year_filter(client):
    r = client.get("/api/search?q=accommodation&year_from=2018&year_to=2018")
    body = r.json()
    assert body["total"] >= 1
    assert all(h["year"] == 2018 for h in body["hits"])


def test_search_synonyms_when_no_hits(client):
    r = client.get("/api/search?q=AI+bias")
    body = r.json()
    assert body["total"] == 0
    assert body["alsoTry"]                         # not empty
    assert "algorithmic discrimination" in body["alsoTry"]


def test_document_full(client):
    r = client.get("/api/document/crpd-c-gc-6")
    assert r.status_code == 200
    body = r.json()
    assert body["document"]["doc_id"] == "crpd-c-gc-6"
    assert len(body["paragraphs"]) == 2
    assert body["paragraphs"][0]["idx"] == 1


def test_document_404(client):
    r = client.get("/api/document/does-not-exist")
    assert r.status_code == 404


def test_paragraph_lookup(client):
    r = client.get("/api/paragraph/crpd-c-gc-6-0002")
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "crpd-c-gc-6"
    assert body["text"].startswith("Discrimination")


def test_browse_by_scope(client):
    r = client.get("/api/browse?scope=gc")
    body = r.json()
    assert body["total"] == 1
    assert body["documents"][0]["doc_id"] == "crpd-c-gc-6"


def test_feedback_writes(client, tmp_path, monkeypatch):
    r = client.post(
        "/api/feedback",
        json={"kind": "bug", "message": "Cite button missing on mobile"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_feedback_validates(client):
    # message too short
    r = client.post("/api/feedback", json={"kind": "bug", "message": "no"})
    assert r.status_code == 422
