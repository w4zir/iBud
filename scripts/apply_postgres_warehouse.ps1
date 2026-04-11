# Applies all Postgres migrations (001–006) via infra/postgres/migrations/all_migrations.sql.
# New installs: warehouse + analytics (003–004) are also inlined in infra/postgres/init.sql;
# run this script to align older volumes or to add 001/002/005/006 without re-creating the DB.
# Usage (from repo root): .\scripts\apply_postgres_warehouse.ps1

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$db = $env:POSTGRES_DB
if (-not $db) { $db = "ecom_support" }
$user = $env:POSTGRES_USER
if (-not $user) { $user = "admin" }

Push-Location $root
try {
    Get-Content -Path "infra/postgres/migrations/all_migrations.sql" -Raw `
        | docker compose exec -T postgres psql -U $user -d $db -v ON_ERROR_STOP=1
} finally {
    Pop-Location
}
