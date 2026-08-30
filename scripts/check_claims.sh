#!/usr/bin/env bash
#
# Fail if the repository contains a claim that is false and, in most jurisdictions, a
# regulatory problem to make about a trading product.
#
# Lives in a script rather than inline in CI so it can be run locally before pushing, and so
# the pattern list has one home.
#
# Excluded from the scan:
#   docs/legal/README.md - documents the banned phrases so they can be recognised
#   .github, .venv, node_modules, site-packages - not prose we author
set -euo pipefail

cd "$(dirname "$0")/.."

PATTERNS='guaranteed (profit|return|monthly)'
PATTERNS+='|risk[- ]free'
PATTERNS+='|cannot lose'
PATTERNS+='|no risk of loss'
PATTERNS+='|(ai|algorithm) knows where the market'

matches=$(
    grep -rniE "$PATTERNS" \
        --include="*.py" --include="*.md" --include="*.ts" --include="*.tsx" \
        --exclude-dir=node_modules \
        --exclude-dir=.github \
        --exclude-dir=.venv \
        --exclude-dir=venv \
        --exclude-dir=site-packages \
        --exclude-dir=.next \
        . 2>/dev/null | grep -v 'docs/legal/README.md' || true
)

if [ -n "$matches" ]; then
    echo "Forbidden claim found:"
    echo "$matches"
    echo
    echo "These claims are false. Remove them."
    exit 1
fi

echo "No forbidden claims found"
