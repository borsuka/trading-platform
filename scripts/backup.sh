#!/usr/bin/env bash
#
# Database backup.
#
# Backs up the database only. `ENCRYPTION_KEY` is deliberately NOT included: storing the key
# alongside the ciphertext it protects defeats the point of encrypting credentials at rest.
# Back the key up separately, somewhere the database backups are not.
set -euo pipefail

cd "$(dirname "$0")/.."

BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
STAMP="$(date +%Y-%m-%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

if [ -n "${DATABASE_URL:-}" ] && [[ "$DATABASE_URL" == postgresql* ]]; then
    target="$BACKUP_DIR/db-$STAMP.sql.gz"
    echo "Backing up PostgreSQL to $target"
    docker compose exec -T postgres pg_dump \
        -U "${POSTGRES_USER:-trading}" \
        "${POSTGRES_DB:-trading}" | gzip > "$target"
else
    src="${SQLITE_PATH:-./data/trading.db}"
    if [ ! -f "$src" ]; then
        echo "No database at $src; nothing to back up." >&2
        exit 1
    fi
    target="$BACKUP_DIR/db-$STAMP.sqlite"
    echo "Backing up SQLite to $target"
    # Python's sqlite3.Connection.backup() is the online-backup API: safe while the database
    # is in use, unlike copying the file. Using Python rather than the sqlite3 CLI because the
    # CLI is not installed everywhere, whereas Python is a hard requirement of this platform.
    python - "$src" "$target" <<'PY'
import sqlite3
import sys

source, destination = sys.argv[1], sys.argv[2]
with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
    src.backup(dst)
PY
    gzip -f "$target"
    target="$target.gz"
fi

size=$(du -h "$target" | cut -f1)
echo "Wrote $target ($size)"

echo "Pruning backups older than $RETENTION_DAYS days"
find "$BACKUP_DIR" -name 'db-*' -type f -mtime "+$RETENTION_DAYS" -print -delete

cat <<'REMINDER'

Reminder: this backup does NOT include ENCRYPTION_KEY.
Without that key, the encrypted exchange credentials in this dump cannot be decrypted.
Store it separately - and test a restore, because an untested backup is a hypothesis.
REMINDER
