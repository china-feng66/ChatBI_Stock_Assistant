$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "frontend"

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Push-Location $frontend
    try { npm install } finally { Pop-Location }
}

if (-not (Test-Path (Join-Path $frontend "dist\index.html"))) {
    Push-Location $frontend
    try { npm run build } finally { Pop-Location }
}

$env:PYTHONPATH = Join-Path $root "src"
Set-Location $root
python -m uvicorn quantquery_a.api:app --host 127.0.0.1 --port 8000
