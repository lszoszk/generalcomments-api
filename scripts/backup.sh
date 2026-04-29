#!/usr/bin/env bash
# Daily backup of unhrdb.sqlite3 with 14-day rotation.
#
# Install (run on the VM, no sudo needed):
#   crontab -e
#   # add the line:
#   17 3 * * *  /home/amuvmuser/unhrdb/scripts/backup.sh >> /home/amuvmuser/unhrdb/backups/backup.log 2>&1
#
# Why sqlite3 .backup instead of cp:
#   The DB is mounted ro into the container, but the file on disk is
#   "live" — nothing writes during normal operation, but a future
#   incremental-update path might. .backup is the only safe online
#   copy method that won't corrupt a partial transaction.

set -eu

SRC=/home/amuvmuser/unhrdb/data/unhrdb.sqlite3
DST_DIR=/home/amuvmuser/unhrdb/backups
RETAIN_DAYS=14

mkdir -p "$DST_DIR"

stamp=$(date -u +%Y%m%d-%H%M%S)
out="$DST_DIR/unhrdb-${stamp}.sqlite3"

# .backup creates a consistent copy even if a writer is mid-transaction.
sqlite3 "$SRC" ".backup '$out'"

# Compress it — backups compress to ~30 % of their source.
gzip -f "$out"

echo "[$stamp] backup ok · $(ls -1sh ${out}.gz | awk '{print $1}')"

# Rotate: anything older than RETAIN_DAYS days goes.
find "$DST_DIR" -name 'unhrdb-*.sqlite3.gz' -mtime "+${RETAIN_DAYS}" -print -delete
