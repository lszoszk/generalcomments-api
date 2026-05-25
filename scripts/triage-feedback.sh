#!/usr/bin/env bash
# Pretty-print the user-feedback log for weekly triage.
#
# The /api/feedback endpoint appends one JSON object per report to
# feedback.jsonl (fields: ts, ip, agent, kind, paraId, docId, message,
# contact). This script reads that log and prints a scannable, grouped
# view so fixes can be batched.
#
# Run on the VM:
#   ~/unhrdb/scripts/triage-feedback.sh            # everything
#   ~/unhrdb/scripts/triage-feedback.sh --since 7  # last 7 days
#
# Or from a laptop without installing anything on the VM:
#   ssh amuvmuser@150.254.115.204 'bash -s -- --since 7' < scripts/triage-feedback.sh
#
# Override the log path with UNHRDB_FEEDBACK_LOG if needed.
set -eu

LOG="${UNHRDB_FEEDBACK_LOG:-/home/amuvmuser/unhrdb/feedback/feedback.jsonl}"

SINCE_DAYS=""
if [ "${1:-}" = "--since" ] && [ -n "${2:-}" ]; then
  SINCE_DAYS="$2"
fi

if [ ! -f "$LOG" ]; then
  echo "No feedback log at $LOG"
  exit 0
fi

UNHRDB_FEEDBACK_LOG="$LOG" SINCE_DAYS="$SINCE_DAYS" python3 - <<'PY'
import json, os, sys
from collections import Counter
from datetime import datetime, timezone, timedelta

log = os.environ["UNHRDB_FEEDBACK_LOG"]
since_days = os.environ.get("SINCE_DAYS") or ""
cutoff = None
if since_days.strip():
    cutoff = datetime.now(timezone.utc) - timedelta(days=float(since_days))

def parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None

rows = []
with open(log, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        ts = parse_ts(d.get("ts"))
        if cutoff and ts and ts < cutoff:
            continue
        rows.append(d)

rows.sort(key=lambda d: d.get("ts") or "")

window = f" (last {since_days} days)" if cutoff else ""
print(f"\n{len(rows)} report(s){window} — {log}\n" + "=" * 64)

for d in rows:
    ts = (d.get("ts") or "?")[:19].replace("T", " ")
    kind = (d.get("kind") or "other").upper()
    loc = d.get("paraId") or d.get("docId") or "(no doc)"
    msg = (d.get("message") or "").replace("\n", " ").strip()
    contact = d.get("contact")
    print(f"\n[{ts}]  {kind:<7}  {loc}")
    if msg:
        print(f"    {msg}")
    if contact:
        print(f"    ↳ contact: {contact}")

print("\n" + "=" * 64)
by_kind = Counter((d.get("kind") or "other") for d in rows)
by_doc = Counter(d.get("docId") for d in rows if d.get("docId"))
print("by kind:  " + (", ".join(f"{k}={n}" for k, n in by_kind.most_common()) or "—"))
top = by_doc.most_common(8)
if top:
    print("top docs: " + ", ".join(f"{k}={n}" for k, n in top))
print()
PY
