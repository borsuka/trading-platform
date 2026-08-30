# Build everything the desktop application needs.
#
# Run once after cloning, and again whenever the dashboard source changes. The result is a
# static export in frontend/out that the API serves from its own port - there is no Node
# process at runtime, so Node is needed here and nowhere else.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

Write-Host "Building the Trading Platform desktop application" -ForegroundColor Cyan
Write-Host ""

# --------------------------------------------------------------------------- #
# Python side
# --------------------------------------------------------------------------- #
$venvPython = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating the Python environment..." -ForegroundColor Yellow
    python -m venv (Join-Path $repo ".venv")
}

Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -e (Join-Path $repo "backend") --quiet
# pywebview gives the application a real window instead of a browser tab. Without it the
# launcher still runs, it just falls back to the default browser.
& $venvPython -m pip install pywebview --quiet
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed" }

# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
$frontend = Join-Path $repo "frontend"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js 20 or newer, then run this again."
}

Push-Location $frontend
try {
    if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
        Write-Host "Installing dashboard dependencies..." -ForegroundColor Yellow
        npm ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    }

    Write-Host "Building the dashboard..." -ForegroundColor Yellow
    $env:NEXT_OUTPUT = "export"
    # Empty on purpose: the dashboard and the API share an origin in the desktop build, so
    # every request is relative and no host is baked into the bundle.
    $env:NEXT_PUBLIC_API_URL = ""
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "The dashboard build failed" }
}
finally {
    Pop-Location
    Remove-Item Env:\NEXT_OUTPUT -ErrorAction SilentlyContinue
    Remove-Item Env:\NEXT_PUBLIC_API_URL -ErrorAction SilentlyContinue
}

if (-not (Test-Path (Join-Path $frontend "out\index.html"))) {
    throw "The build reported success but produced no output in frontend/out"
}

Write-Host ""
Write-Host "Done. Start the application with TradingPlatform.bat" -ForegroundColor Green
Write-Host "It starts in PAPER mode. No real money is involved until you" -ForegroundColor Green
Write-Host "deliberately change that, and the platform will not let you do it by accident." -ForegroundColor Green
