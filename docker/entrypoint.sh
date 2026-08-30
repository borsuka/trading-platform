#!/usr/bin/env bash
#
# Container entrypoint.
#
# Waits for the database, applies migrations, then starts the requested process. Migrations run
# here rather than at application startup so that a rolling deploy applies them once, not once
# per replica.
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }

wait_for_database() {
    local attempts=${DB_WAIT_ATTEMPTS:-30}
    local delay=${DB_WAIT_DELAY:-2}

    if [[ "${DATABASE_URL:-}" == sqlite* ]]; then
        log "SQLite database; no wait required"
        return 0
    fi

    log "waiting for the database"
    for ((i = 1; i <= attempts; i++)); do
        if python -c "
import asyncio, sys
from app.database.session import check_connection
sys.exit(0 if asyncio.run(check_connection()) else 1)
" 2>/dev/null; then
            log "database is reachable"
            return 0
        fi
        sleep "$delay"
    done
    log "database did not become reachable after $((attempts * delay))s"
    return 1
}

announce_mode() {
    if [[ "${TRADING_MODE:-paper}" == "live" && "${LIVE_TRADING_ENABLED:-false}" == "true" ]]; then
        log "*******************************************************"
        log "*  LIVE TRADING ENABLED - REAL ORDERS, REAL MONEY     *"
        log "*******************************************************"
    else
        log "mode: ${TRADING_MODE:-paper} (no real money at risk)"
    fi
}

announce_mode

case "${1:-serve}" in
    serve)
        wait_for_database
        log "applying migrations"
        alembic upgrade head
        # Access logs are disabled: structlog owns logging and the lifespan re-applies
        # its configuration (including secret redaction) after uvicorn installs its own.
        log "starting API server"
        exec uvicorn app.main:app \
            --host "${HOST:-0.0.0.0}" \
            --port "${PORT:-8000}" \
            --workers "${WEB_CONCURRENCY:-1}" \
            --no-access-log
        ;;
    worker)
        wait_for_database
        log "starting worker"
        exec python -m app.worker
        ;;
    migrate)
        wait_for_database
        exec alembic upgrade head
        ;;
    shell)
        exec python
        ;;
    *)
        exec "$@"
        ;;
esac
