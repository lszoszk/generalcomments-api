# unhrdb-api

FastAPI service powering the UN Human Rights Database — a paragraph-
level search interface for UN Treaty Body General Comments,
jurisprudence and Special Procedures reports.

Designed to **complement** the static GH-Pages frontend, not replace it.
The frontend keeps the **General Comments** corpus in-browser
(7 103 ¶, ~6 MB gzipped — fits in a single FlexSearch index, runs
faster than any network round-trip). This API takes over for the
heavier scopes — **Jurisprudence** (≈107 k ¶ today, projected 180 k by
2030) and eventually **Special Procedures** when its full crawl pushes
past ~30 k ¶.

> Sibling apps on the same VM, in case you need to compare:
> `echr-api` (8000), `uhri-dataset-api` (8001).
> This service runs on `127.0.0.1:8002` and lives at `/unhrdb-api/`.

## Repo layout

```
generalcomments-api/
  api/
    Dockerfile          python:3.11-slim, copies app, runs uvicorn
    main.py             FastAPI routes (/health /api/* )
    ranking.py          FTS5 query sanitiser + BM25F column weights
    build_db.py         reads ../generalcomments-repo/docs/*.json → SQLite
    requirements.txt    fastapi + uvicorn pinned to ECHR's versions
  data/
    unhrdb.sqlite3      built artefact, mounted into the container ro
    manifest.json       version + sha + counts (consumed by /api/stats)
  docker-compose.yml    binds 127.0.0.1:8002, 512 MB mem cap, restart=unless-stopped
  README.md             this file
```

## Build the database

```bash
cd api
python3 build_db.py --pretty
```

Reads from `../../generalcomments-repo/docs/` by default; override with
`--src` and `--out`. Output:

```
✓ build complete
  documents:    3,296
  paragraphs: 132,711
  size:         205.9 MB
  built at:   2026-04-29T12:19:09+00:00
```

The DB is ~6× smaller than ECHR's because we only have ~1/4 of its
paragraphs and don't store HTML mirrors.

## Run locally

```bash
cd api
UNHRDB_DB_PATH=$(pwd)/../data/unhrdb.sqlite3 \
  python3 -m uvicorn main:app --host 127.0.0.1 --port 8002

# Smoke
curl -s http://127.0.0.1:8002/health | jq
curl -s 'http://127.0.0.1:8002/api/search?q=reasonable%20accommodation&page_size=3' | jq '.total, .tookMs, .hits[0].para_id'
```

Or via Docker:

```bash
docker compose up -d --build
curl -s http://127.0.0.1:8002/health | jq
docker compose logs -f
```

## Deploy on the VM

The deploy plan lives in `../GC_Database/VM_DEPLOY_PLAN.md`. Sprint 3
of that plan covers the actual cutover; below is the short version.

1. **Build the DB on your laptop**, scp the artefact to the VM:
   ```bash
   cd api && python3 build_db.py
   scp ../data/unhrdb.sqlite3 ../data/manifest.json \
       amuvmuser@150.254.115.204:/home/amuvmuser/unhrdb/data/
   ```

2. **scp the app**:
   ```bash
   rsync -avz --exclude=__pycache__ --exclude='.git' \
       /Users/lszoszk/Desktop/generalcomments-api/ \
       amuvmuser@150.254.115.204:/home/amuvmuser/unhrdb/
   ```

3. **Boot the container** (already-installed Docker on the VM):
   ```bash
   ssh amuvmuser@150.254.115.204 'cd /home/amuvmuser/unhrdb && docker compose up -d --build'
   ```

4. **Add the nginx route** (see `nginx-snippet.conf` for the block).
   Append it inside the existing `server { listen 443 ssl http2; }` in
   `/etc/nginx/sites-enabled/default` and reload:
   ```bash
   sudo nginx -t && sudo systemctl reload nginx
   curl https://150.254.115.204/unhrdb-api/health
   ```

## Endpoints

| Method | Path                            | Purpose                                   |
|--------|---------------------------------|-------------------------------------------|
| GET    | `/health`                       | Liveness, paragraph count                 |
| GET    | `/api/stats`                    | Manifest + per-type counts                |
| GET    | `/api/facets?scope=…`           | Filter chip data (treaties, mandates, …)  |
| GET    | `/api/search?q=…&scope=…&…`     | Paginated FTS hits + snippet + score      |
| GET    | `/api/document/{doc_id}`        | Full document with all paragraphs         |
| GET    | `/api/paragraph/{para_id}`      | Single paragraph + parent-doc fields      |
| GET    | `/api/browse?scope=…&…`         | Catalogue list, no FTS                    |

All read-only; CORS allows `https://lszoszk.github.io` and localhost.
Gzip middleware enabled. Cache-Control headers can be added per
endpoint in Sprint 2.

## Performance numbers (Sprint 1, Apple M-series local)

| Scenario                                              | tookMs |
|-------------------------------------------------------|-------:|
| `/health`                                             |    <1  |
| `/api/stats`                                          |    ~5  |
| `q=reasonable accommodation`, all scopes              |    72  |
| `q=trafficking AND children NOT (sexual)`, all scopes |    31  |
| `q=disability`, scope=jur, no other filter            |   125  |
| `q=violation`, scope=jur, treaty=CRPD, year 2020-24   |   908* |
| `q=discriminat*` (prefix wildcard), all scopes        |    46  |

(* the long one is the count + page + breakdown each re-running the
FTS scan; Sprint 2 will collapse this to a single CTE.)

## Known limitations (Sprint 2 backlog)

- **Counts run twice.** Filtered queries do `count(*)` and the page
  slice as separate queries. Replace with a `WITH hits AS (...)` CTE.
- **No phrase suggestions.** `"AI bias"` returns 0 because UN docs
  pre-2023 don't use that phrase. We need a synonyms table → empty
  state with "did you also try…" hints.
- **No Cache-Control headers.** Add `public, max-age=300` per endpoint.
- **No request rate limit.** Will need slowapi if we expose
  `/api/feedback` or anything write-shaped later.
- **Browse endpoint sort options.** Only year_desc is heavily used in
  the static frontend; we have 4 server-side; verify the FE actually
  needs them.

## Testing

```bash
cd api
pip install pytest httpx
# (tests will land in Sprint 2 — TestClient-based, no DB mocking,
#  uses a small fixture DB built in tmp_path.)
```

## Versioning

`UNHRDB_VERSION` env var is surfaced in `/health` and `/api/stats`.
Bump it on each deploy. The DB itself is rebuilt fresh per deploy
(no migrations); the manifest's `sha` column lets the frontend
invalidate its cached facet/year-histogram state when the corpus
changes.

## Feedback / GitHub Issues forwarding (v19.14)

`/api/feedback` accepts paragraph-scoped reports from the frontend's
flag-button affordance and persists them to
`/feedback/feedback.jsonl` (durable). When two env vars are set, it
ALSO files a GitHub Issue per submission for triage convenience:

```bash
# In the VM operator's shell before `docker compose up -d`:
export GITHUB_FEEDBACK_TOKEN="ghp_…"      # PAT, scope: repo (issues:write)
export GITHUB_FEEDBACK_REPO="lszoszk/generalcomments-feedback"
docker compose up -d
```

The token never appears in container env via the image — it's
injected via the host shell, read once at startup. Failure to file
an issue (token expired, GitHub down, network blocked) does NOT fail
the user-facing request: the jsonl log is the source of truth.

To rotate: revoke the old token on GitHub, update the env var on the
VM, `docker compose up -d` (recreates with fresh env). Issues filed
during the gap stay in the local jsonl — replay them from the log if
you want them to land on GitHub retroactively.

## License

Same as the parent project.
