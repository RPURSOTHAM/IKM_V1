<#
.SYNOPSIS
  Download Pharma CIH models into rag-builder/models/cih for cih-service.

.DESCRIPTION
  Makes it obvious which models users need:
    Required     - faster-whisper small + tiny (ASR)
    Recommended  - Required + MiniLM for MLR / key-message embeddings
    Full         - Recommended + prints steps for optional OCR/VLM

.PARAMETER Profile
  Required | Recommended | Full  (default: Recommended)

.EXAMPLE
  powershell -File rag-builder/scripts/download-cih-models.ps1
  powershell -File rag-builder/scripts/download-cih-models.ps1 -Profile Required
#>
[CmdletBinding()]
param(
    [ValidateSet("Required", "Recommended", "Full")]
    [string]$Profile = "Recommended"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$CacheRoot = Join-Path $RepoRoot "models\cih"
$HfHome = Join-Path $CacheRoot "huggingface"
$WhisperDir = Join-Path $CacheRoot "whisper"

New-Item -ItemType Directory -Force -Path $CacheRoot, $HfHome, $WhisperDir | Out-Null

Write-Host ""
Write-Host "Pharma CIH model download" -ForegroundColor Cyan
Write-Host "  Profile : $Profile"
Write-Host "  Cache   : $CacheRoot"
Write-Host "  Feature : src/features/content_intelligence_hub"
Write-Host "  Docs    : src/features/content_intelligence_hub/MODELS.md"
Write-Host "  UI      : http://localhost:8088/api/v1/cih/"
Write-Host "  Swagger : http://localhost:8088/docs  (Content Intelligence Hub)"
Write-Host ""

function Ensure-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        throw "Python is required on PATH to download models. Install Python 3.11+ and retry."
    }
    return $py.Source
}

function Ensure-PipPackage([string]$Package) {
    python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$($Package.Replace('-', '_').Split('[')[0])') else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing $Package ..." -ForegroundColor Yellow
        python -m pip install --upgrade $Package
    }
}

$python = Ensure-Python
Write-Host "Using Python: $python"

# --- Required: faster-whisper ---
Write-Host ""
Write-Host "[1/2] Required — faster-whisper ASR (small + tiny)" -ForegroundColor Green
Ensure-PipPackage "faster-whisper"

$env:HF_HOME = $HfHome
$env:XDG_CACHE_HOME = $CacheRoot
$downloadWhisper = @"
from faster_whisper import WhisperModel
import os
cache = os.environ.get('XDG_CACHE_HOME', '.')
print('Downloading faster-whisper model: small')
WhisperModel('small', device='cpu', compute_type='int8', download_root=os.path.join(cache, 'whisper'))
print('Downloading faster-whisper model: tiny')
WhisperModel('tiny', device='cpu', compute_type='int8', download_root=os.path.join(cache, 'whisper'))
print('ASR models ready.')
"@
python -c $downloadWhisper
if ($LASTEXITCODE -ne 0) { throw "ASR model download failed." }

if ($Profile -eq "Required") {
    Write-Host ""
    Write-Host "Required models done." -ForegroundColor Green
    Write-Host "Next: docker compose up mongodb dms-service  (UI http://localhost:8088/cih)"
    exit 0
}

# --- Recommended: MiniLM ---
Write-Host ""
Write-Host "[2/2] Recommended — MiniLM embeddings for MLR / key messages" -ForegroundColor Green
Ensure-PipPackage "sentence-transformers"

$downloadMiniLM = @"
from sentence_transformers import SentenceTransformer
import os
os.environ.setdefault('HF_HOME', os.environ.get('HF_HOME', ''))
print('Downloading sentence-transformers/all-MiniLM-L6-v2')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('MiniLM ready.')
"@
python -c $downloadMiniLM
if ($LASTEXITCODE -ne 0) { throw "MiniLM download failed." }

Write-Host ""
Write-Host "Set in deploy/application/.env after this download:" -ForegroundColor Yellow
Write-Host "  MLR_EMBEDDING_LOCAL_FILES_ONLY=true"

if ($Profile -eq "Full") {
    Write-Host ""
    Write-Host "Optional (not auto-downloaded here):" -ForegroundColor Cyan
    Write-Host "  OCR   — PaddleOCR downloads on first image/PDF OCR in DMS CIH feature"
    Write-Host "          (cache: data/documents/cih/outputs/paddle_cache). IMAGE_PADDLEOCR_ENABLED=true"
    Write-Host "  VLM   — Install Ollama, pull e.g. qwen2.5vl:3b, then set:"
    Write-Host "          OPEN_WEIGHT_VLM_ENABLED=true"
    Write-Host "          OPEN_WEIGHT_VLM_API_URL=http://host.docker.internal:11434/api/chat"
    Write-Host "  Azure — Set QWEN_ACCURACY_ENABLED / AZURE_OPENAI_TRANSLATION_ENABLED (keys already in .env)"
    Write-Host ""
    Write-Host "Full details: src/features/content_intelligence_hub/MODELS.md"
}

Write-Host ""
Write-Host "Done. Start DMS (CIH at /cih):" -ForegroundColor Green
Write-Host "  cd rag-builder/deploy/application"
Write-Host "  docker compose -f docker-compose.application.yml --env-file .env up -d mongodb dms-service"
Write-Host "  Open http://localhost:8088/docs  (tag: Content Intelligence Hub)"
Write-Host "  UI   http://localhost:8088/api/v1/cih/"
