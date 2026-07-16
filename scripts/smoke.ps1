# Minimal end-to-end smoke on Windows: init -> dry-run plan -> gate, checking
# artifacts. Proves pathlib/Windows-path handling (R-ARCH-5). PowerShell counterpart
# of scripts/smoke.sh. Run after: pip install -e ".[dev]"
$ErrorActionPreference = "Stop"

$work = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("docusearch-smoke-" + [guid]::NewGuid()))
try {
    Set-Location $work

    python -m docusearch init
    python -m docusearch ingest --dry-run
    python -m docusearch gate 1

    if (-not (Test-Path "docusearch.yaml")) { throw "FAIL: config not written" }
    if (-not (Test-Path "tmp/gates/GATE-1-phase-1.md")) { throw "FAIL: gate file not written" }
    if (-not (Get-ChildItem "tmp/logs/*.jsonl" -ErrorAction SilentlyContinue)) { throw "FAIL: log not written" }

    Write-Host "smoke OK"
}
finally {
    Set-Location $env:TEMP
    Remove-Item -Recurse -Force $work
}
