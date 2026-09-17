# Build rag-dms-service:latest and rag-scheduler-service:latest on Windows / OneDrive.
# OneDrive cloud-only files break "docker compose ... --build". This script hydrates
# source into %TEMP% then runs docker build for both application images.
#
# Usage (from deploy/application):
#   powershell -File build-application.ps1
#
# Then start containers (no --build):
#   docker compose -f docker-compose.application.yml --env-file .env up -d

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$StageRoot = Join-Path $env:TEMP ("rag-application-build-" + [guid]::NewGuid().ToString("n"))

function Copy-HydratedTree {
    param(
        [string]$RelativePath,
        [string[]]$ExcludeNames = @(".env", ".venv", ".gitignore")
    )
    $Source = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path $Source)) {
        throw "Missing path: $Source"
    }
    Get-ChildItem -Path $Source -Recurse -File | ForEach-Object {
        if ($ExcludeNames -contains $_.Name) { return }
        if ($_.Extension -eq ".log") { return }
        if ($_.FullName -match "\\\.venv\\|\\__pycache__\\|\\\.pytest_cache\\") { return }
        $Rel = $_.FullName.Substring($RepoRoot.Length).TrimStart("\", "/")
        $Target = Join-Path $StageRoot $Rel
        $TargetDir = Split-Path $Target -Parent
        if (-not (Test-Path $TargetDir)) {
            New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
        }
        [System.IO.File]::WriteAllBytes($Target, [System.IO.File]::ReadAllBytes($_.FullName))
    }
}

function Build-Image {
    param(
        [string]$DockerfileRel,
        [string]$ImageTag
    )
    $Dockerfile = Join-Path $RepoRoot $DockerfileRel
    Write-Host "Building $ImageTag ..."
    docker build -f $Dockerfile -t $ImageTag $StageRoot
    if ($LASTEXITCODE -ne 0) {
        throw "docker build failed for $ImageTag"
    }
}

Write-Host "Staging application build context at $StageRoot"
New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null
Copy-HydratedTree "src\dms_service"
Copy-HydratedTree "src\scheduler_service"
Copy-HydratedTree "src\shared"
Copy-HydratedTree "src\processor_service"

try {
    Build-Image "src\scheduler_service\Dockerfile" "rag-scheduler-service:latest"
    Build-Image "src\dms_service\Dockerfile" "rag-dms-service:latest"
}
finally {
    Remove-Item -Recurse -Force $StageRoot -ErrorAction SilentlyContinue
}

Write-Host "Done. Images:"
docker image ls "rag-scheduler-service"
docker image ls "rag-dms-service"
