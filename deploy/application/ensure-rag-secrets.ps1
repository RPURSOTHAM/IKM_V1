# Ensures PLATFORM_BOOTSTRAP_ADMIN_PASSWORD and JWT_SIGNING_KEY exist in .env.
# Run once before first `docker compose ... up` (does not print secrets).
param(
    [string]$EnvFile = (Join-Path $PSScriptRoot ".env")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EnvFile)) {
    throw "Missing $EnvFile — create deploy/application/.env (single stack env)."
}

function Read-EnvMap([string]$Path) {
    $map = @{}
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { return }
        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1)
        $map[$key] = $value
    }
    return $map
}

function Write-EnvMap([string]$Path, [hashtable]$Map, [string[]]$Order) {
    $lines = Get-Content $Path
    $seen = @{}
    $output = New-Object System.Collections.Generic.List[string]
    foreach ($line in $lines) {
        $trim = $line.Trim()
        if ($trim -and -not $trim.StartsWith("#") -and $trim.Contains("=")) {
            $key = $trim.Split("=", 2)[0].Trim()
            if ($Map.ContainsKey($key)) {
                $output.Add("$key=$($Map[$key])")
                $seen[$key] = $true
                continue
            }
        }
        $output.Add($line)
    }
    foreach ($key in $Order) {
        if (-not $seen.ContainsKey($key) -and $Map.ContainsKey($key)) {
            $output.Add("$key=$($Map[$key])")
        }
    }
    Set-Content -Path $Path -Value $output -Encoding UTF8
}

$map = Read-EnvMap $EnvFile
$changed = $false

if (-not $map["REPO_ROOT"]) {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path -replace "\\", "/"
    $map["REPO_ROOT"] = $repoRoot
    $changed = $true
}

if (-not $map["PLATFORM_BOOTSTRAP_ADMIN_USER"]) {
    $map["PLATFORM_BOOTSTRAP_ADMIN_USER"] = "admin"
    $changed = $true
}

if (-not $map["PLATFORM_BOOTSTRAP_ADMIN_PASSWORD"]) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $map["PLATFORM_BOOTSTRAP_ADMIN_PASSWORD"] = [Convert]::ToBase64String($bytes)
    $changed = $true
    Write-Host "Generated PLATFORM_BOOTSTRAP_ADMIN_PASSWORD in $EnvFile"
}

if (-not $map["JWT_SIGNING_KEY"]) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $map["JWT_SIGNING_KEY"] = [Convert]::ToBase64String($bytes)
    $changed = $true
    Write-Host "Generated JWT_SIGNING_KEY in $EnvFile"
}

if ($changed) {
    Write-EnvMap $EnvFile $map @(
        "PLATFORM_BOOTSTRAP_ADMIN_USER",
        "PLATFORM_BOOTSTRAP_ADMIN_PASSWORD",
        "JWT_SIGNING_KEY"
    )
    Write-Host "Updated $EnvFile"
} else {
    Write-Host "Secrets already present in $EnvFile"
}
