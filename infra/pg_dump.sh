#!/usr/bin/env bash
# Dump the control-plane database to the VOLUME.
#
# The datadir lives on the container disk (Postgres over MooseFS risks corruption), so
# a pod stop destroys it. This dump is the only thing that carries accounts across a
# stop, and pg_setup.sh restores the newest one into an empty cluster automatically.
#
# Run before stopping the pod, and on a timer while it's up:
#   bash /workspace/infra/pg_dump.sh
set -euo pipefail
cd /tmp
export LC_ALL=C LANG=C
BACKUP=/workspace/.pgbackup
mkdir -p "$BACKUP"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$BACKUP/wemend-$STAMP.sql.gz"

su postgres -c "pg_dump -h 127.0.0.1 --clean --if-exists -d wemend" | gzip > "$OUT"
# A dump smaller than a few hundred bytes means pg_dump failed and we gzipped an
# error; better to fail loudly than to keep a useless "backup".
SIZE=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
if [ "$SIZE" -lt 300 ]; then
  rm -f "$OUT"
  echo "dump looked empty ($SIZE bytes) — removed, NOT treating this as a backup" >&2
  exit 1
fi
chmod 600 "$OUT"

# Keep the 10 most recent; the volume is huge but unbounded growth is still a bug.
ls -1t "$BACKUP"/wemend-*.sql.gz | tail -n +11 | xargs -r rm -f

echo "  dumped $(numfmt --to=iec "$SIZE" 2>/dev/null || echo "${SIZE}B") -> $(basename "$OUT")"
echo "  kept:  $(ls -1 "$BACKUP"/wemend-*.sql.gz | wc -l | tr -d ' ') dumps"
