FROM python:3.11-slim

# Match the sibling apps on the VM (echr-api, uhri-dataset-api).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so re-builds skip this layer when only the
# Python sources change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App sources. We copy all .py files at the api/ root (main, ranking,
# synonyms, build_db) — keeping the surface explicit prevents tests/
# and __pycache__ from sneaking in. build_db.py isn't needed in the
# image but it's tiny; leaving it lets us re-build the DB inside the
# running container if we ever want to (`docker exec unhrdb-api python
# build_db.py`).
COPY main.py ranking.py synonyms.py build_db.py ./

# The DB is mounted in via docker-compose at /data; the env var below is
# the same default the code uses.  We set it here for clarity / docs.
ENV UNHRDB_DB_PATH=/data/unhrdb.sqlite3

# Health check pings /health every 30s so docker reports the container as
# healthy in `docker ps`. Three failed checks → unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/health',timeout=3).status==200 else 1)"

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
