# Build rag-processor:latest on Windows / OneDrive hosts.
# OneDrive "online-only" files are reparse points that Docker BuildKit cannot COPY.
# This script materializes source into %TEMP% then runs docker build.
#
# Usage (from repo root):
#   powershell -File deploy/application/build-processor-image.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$StageRoot = Join-Path $env:TEMP ("rag-processor-build-" + [guid]::NewGuid().ToString("n"))
$ImageName = if ($env:PROCESSOR_IMAGE_NAME) { $env:PROCESSOR_IMAGE_NAME } else { "rag-processor:latest" }

function Copy-HydratedTree {
    param(
        [string]$RelativePath,
        [string[]]$ExcludeNames = @(".env", ".venv", "__pycache__")
    )
    $Source = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path $Source)) {
        throw "Missing path: $Source"
    }
    Get-ChildItem -Path $Source -Recurse -File | ForEach-Object {
        if ($ExcludeNames -contains $_.Name) { return }
        if ($_.FullName -match "\\\.venv\\|\\__pycache__\\") { return }
        $Rel = $_.FullName.Substring($RepoRoot.Length).TrimStart("\", "/")
        $Target = Join-Path $StageRoot $Rel
        $TargetDir = Split-Path $Target -Parent
        if (-not (Test-Path $TargetDir)) {
            New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
        }
        [System.IO.File]::WriteAllBytes($Target, [System.IO.File]::ReadAllBytes($_.FullName))
    }
}

Write-Host "Staging processor build context at $StageRoot"
New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null
Copy-HydratedTree "src"

$Dockerfile = Join-Path $RepoRoot "deploy\application\Dockerfile.processor"
Write-Host "Building $ImageName ..."
docker build -f $Dockerfile -t $ImageName $StageRoot
if ($LASTEXITCODE -ne 0) {
    Remove-Item -Recurse -Force $StageRoot -ErrorAction SilentlyContinue
    exit $LASTEXITCODE
}

Remove-Item -Recurse -Force $StageRoot -ErrorAction SilentlyContinue
Write-Host "Done: $ImageName"
docker image ls $ImageName.Split(":")[0]
