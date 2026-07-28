#!/usr/bin/env bash
# Postgres for the control plane, on the pod as a stopgap until a VPS exists.
#
# PGDATA is on the CONTAINER disk on purpose: Postgres over MooseFS risks corruption
# because advisory locks and WAL shared memory don't behave on a network filesystem —
# the same reason SQLite there was ruled out. Durability across pod stops comes from
# pg_dump into /workspace instead (see pg_dump.sh / resume.sh).
set -euo pipefail
cd /tmp                       # su postgres cannot read /root
export LC_ALL=C LANG=C        # image has no generated locales; initdb needs a valid one
PGBIN=/usr/lib/postgresql/14/bin
PGDATA=/opt/pgdata
BACKUP=/workspace/.pgbackup

mkdir -p "$PGDATA" "$BACKUP"
chown -R postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

if [ ! -f "$PGDATA/PG_VERSION" ]; then
  su postgres -c "$PGBIN/initdb -D $PGDATA -E UTF8 --locale=C -A trust" >/dev/null
  echo "  cluster initialised"
else
  echo "  cluster already present"
fi

# 127.0.0.1 only: nothing outside the pod should reach the database.
if ! su postgres -c "$PGBIN/pg_isready -h 127.0.0.1 -q"; then
  su postgres -c "$PGBIN/pg_ctl -D $PGDATA -l $PGDATA/pg.log -o '-c listen_addresses=127.0.0.1 -p 5432' -w start" >/dev/null
fi
su postgres -c "$PGBIN/pg_isready -h 127.0.0.1" | sed 's/^/  /'

# Idempotent role + database, via heredoc so quoting survives.
su postgres -c "psql -h 127.0.0.1 -X -q -v ON_ERROR_STOP=1 -d postgres" <<'SQL'
DO $do$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'wemend') THEN
    CREATE ROLE wemend LOGIN SUPERUSER;
  END IF;
END
$do$;
SQL
echo "  role wemend ready"

if ! su postgres -c "psql -h 127.0.0.1 -X -tAc \"SELECT 1 FROM pg_database WHERE datname='wemend'\"" | grep -q 1; then
  su postgres -c "createdb -h 127.0.0.1 -O wemend wemend"
  echo "  database wemend created"
else
  echo "  database wemend already present"
fi

count_tables() {
  su postgres -c "psql -h 127.0.0.1 -X -tAc \"SELECT count(*) FROM information_schema.tables WHERE table_schema='public'\" -d wemend"
}

# Restore the newest dump if the cluster came back empty (i.e. after a container wipe).
TABLES=$(count_tables || echo 0)
LATEST=$(ls -1t "$BACKUP"/wemend-*.sql.gz 2>/dev/null | head -1 || true)
if [ "${TABLES//[[:space:]]/}" = "0" ] && [ -n "$LATEST" ]; then
  echo "  empty cluster + dump present -> restoring $(basename "$LATEST")"
  gunzip -c "$LATEST" | su postgres -c "psql -h 127.0.0.1 -X -q -d wemend" >/dev/null
  echo "  restored"
fi
echo "  tables in wemend: $(count_tables)"
