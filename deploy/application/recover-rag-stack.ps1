# Recover and start the full RAG Builder stack (infrastructure + application).
# Single env: deploy/application/.env
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InfraDir = Join-Path $ScriptDir "..\infrastructure"
$EnvFile = Join-Path $ScriptDir ".env"

function Invoke-DockerCompose {
    param([string]$WorkingDir, [string[]]$ComposeArgs)
    Push-Location $WorkingDir
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker @ComposeArgs 2>&1 | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    Pop-Location
    return $code
}

function Wait-DockerEngine {
    param([int]$MaxAttempts = 24)
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        $v = docker version --format "{{.Server.Version}}" 2>$null
        if ($v) {
            Write-Host "Docker engine ready ($v)"
            return $true
        }
        if ($i -eq 6) {
            Write-Host "Docker engine not responding; attempting WSL reset..."
            Get-Process "Docker Desktop","com.docker.backend" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
            wsl --shutdown 2>$null | Out-Null
            Start-Sleep 6
            Start-Service com.docker.service -ErrorAction SilentlyContinue
            Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        }
        Start-Sleep 10
        Write-Host "  waiting for Docker ($i/$MaxAttempts)..."
    }
    throw "Docker engine did not become ready."
}

Write-Host "=== RAG Builder stack recovery ==="
Wait-DockerEngine | Out-Null

if (-not (docker network ls --format "{{.Name}}" | Select-String -Pattern "^docnet$")) {
    docker network create docnet | Out-Null
    Write-Host "Created docnet network"
}

& (Join-Path $ScriptDir "ensure-rag-secrets.ps1")

Write-Host "Starting infrastructure..."
Invoke-DockerCompose $InfraDir @("compose", "--env-file", $EnvFile, "up", "-d") | Out-Null

Write-Host "Starting application tier..."
Invoke-DockerCompose $ScriptDir @("compose", "-f", "docker-compose.application.yml", "--env-file", ".env", "up", "-d", "--remove-orphans") | Out-Null

Write-Host "Waiting for DMS health..."
$healthy = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8088/health" -TimeoutSec 5
        if ($r.status -eq "healthy") { $healthy = $true; break }
    } catch {}
    Start-Sleep 4
}
if (-not $healthy) {
    Write-Host "DMS not healthy yet. Check: docker logs rag-dms-service --tail 40"
    exit 1
}

Write-Host "DMS healthy."
$gxpSync = "C:\Users\DELL\Downloads\gxp-doc-ai (1)\gxp-doc-ai\backend\scripts\sync_rag_credentials.ps1"
if (Test-Path $gxpSync) {
    Write-Host "Syncing GxP credentials..."
    & $gxpSync
}

docker ps --format "table {{.Names}}\t{{.Status}}" | Select-String -Pattern "rag-|reference|dependencies|weaviate|mysql"
