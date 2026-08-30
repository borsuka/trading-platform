#!/usr/bin/env bash
#
# Start a local development environment: migrations, then the API.
set -euo pipefail

cd "$(dirname "$0")/../backend"

if [ ! -f ../.env ]; then
    echo "No .env found. Copy .env.example and generate keys:"
    echo "  cp .env.example .env"
    echo "  cd backend && python -m app.cli generate-key"
    exit 1
fi

echo "Applying migrations..."
python -m app.cli migrate up

echo "Checking the installation..."
python -m app.cli check || true

echo
python -m app.cli serve --reload
