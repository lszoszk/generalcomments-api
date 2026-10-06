from __future__ import annotations

import json
import sqlite3

from build_db import SCHEMA, build, insert_document, insert_paragraph


def test_footnotes_are_stored_for_display():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    insert_document(conn.cursor(), {"docId": "x", "type": "jur", "name": "X"})
    insert_paragraph(
        conn.cursor(),
        {"id": "x-0001", "docId": "x", "idx": 1, "n": "8.2", "text": "Body.[[fn:8]]",
         "footnotes": [{"n": 8, "text": "See general comment No. 16 (1988), para. 4."}]},
        {"docId": "x", "name": "X"},
    )
    stored, flat = conn.execute("SELECT footnotes_json, footnotes_text FROM paragraphs").fetchone()
    assert json.loads(stored) == [{"n": 8, "text": "See general comment No. 16 (1988), para. 4."}]
    assert flat == "See general comment No. 16 (1988), para. 4."


def test_jurisprudence_insert_uses_adoption_year():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    insert_document(
        conn.cursor(),
        {
            "docId": "crpd-c-31-d-94-2021",
            "type": "jur",
            "signature": "CRPD/C/31/D/94/2021",
            "year": 2021,
            "communicationYear": 2021,
            "adoptionYear": 2024,
            "adoptionDate": "29 August 2024",
        },
    )

    row = conn.execute(
        "SELECT year, communication_year, adoption_date FROM documents"
    ).fetchone()
    assert row == (2024, 2021, "29 August 2024")


def test_ohchr_juris_date_overrides_bad_extracted_date():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    insert_document(
        conn.cursor(),
        {
            "docId": "ccpr-c-139-d-2933-2017",
            "type": "jur",
            "year": 2017,
            "communicationYear": 2017,
            "adoptionYear": 2027,
            "adoptionDate": "31 Oct 2027",
            "jurisDecisionDate": "31 Oct 2023",
        },
    )

    row = conn.execute(
        "SELECT year, communication_year, adoption_date FROM documents"
    ).fetchone()
    assert row == (2023, 2017, "31 Oct 2023")


def test_future_adoption_year_falls_back_to_plausible_full_date():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    insert_document(
        conn.cursor(),
        {
            "docId": "ccpr-c-140-d-3011-2018",
            "type": "jur",
            "year": 2018,
            "communicationYear": 2018,
            "adoptionYear": 2029,
            "adoptionDate": "18 March 2024",
        },
    )

    row = conn.execute(
        "SELECT year, communication_year, adoption_date FROM documents"
    ).fetchone()
    assert row == (2024, 2018, "18 March 2024")


def test_build_reads_special_procedures_shards(tmp_path):
    src = tmp_path / "docs"
    shards = src / "sp" / "shards"
    shards.mkdir(parents=True)
    (src / "documents.json").write_text(
        json.dumps(
            [
                {
                    "docId": "a-73-348",
                    "type": "sp",
                    "name": "Artificial intelligence and freedom of opinion",
                    "nameShort": "Artificial intelligence",
                    "signature": "A/73/348",
                    "paragraphCount": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    (src / "corpus.json").write_text("[]", encoding="utf-8")
    (shards / "sp_SR_Freedom_of_Expression.json").write_text(
        json.dumps(
            [
                {
                    "id": "a-73-348-0023",
                    "docId": "a-73-348",
                    "idx": 23,
                    "n": 23,
                    "text": "The right to form an opinion includes reasoning.",
                    "type": "sp",
                }
            ]
        ),
        encoding="utf-8",
    )

    out_db = tmp_path / "unhrdb.sqlite3"
    counts = build(src, out_db)
    assert counts["paragraphs"] == 1
    with sqlite3.connect(out_db) as conn:
        assert conn.execute("SELECT count(*) FROM paragraphs").fetchone()[0] == 1


def test_jurisprudence_taxonomies_become_filter_rows():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    doc = {
        "docId": "crpd-c-31-d-94-2021",
        "type": "jur",
        "year": 2024,
        "jurisDecisionType": "Views",
        "jurisSubstantiveIssues": ["Equal recognition before the law", ""],
        "jurisProceduralIssues": ["Exhaustion of domestic remedies"],
        "articlesCited": ["Art. 12", "Art. 13"],
    }
    insert_document(conn.cursor(), doc)
    # Re-inserting the same document replaces its rows rather than doubling them.
    insert_document(conn.cursor(), doc)

    assert conn.execute("SELECT decision_type FROM documents").fetchone() == ("Views",)
    assert conn.execute(
        "SELECT kind, value FROM document_issue ORDER BY kind"
    ).fetchall() == [
        ("procedural", "Exhaustion of domestic remedies"),
        ("substantive", "Equal recognition before the law"),
    ]
    assert conn.execute(
        "SELECT value FROM document_article ORDER BY value"
    ).fetchall() == [("Art. 12",), ("Art. 13",)]
