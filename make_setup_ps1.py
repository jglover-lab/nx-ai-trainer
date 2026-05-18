#!/usr/bin/env python3
"""Generate installer/windows/setup.ps1 from current source files.

Embeds server.py, web files, requirements.txt, and the generic config.json
(with placeholder credentials) into a self-contained PowerShell installer.

Run after any change to server.py, web/, requirements.txt, or config.json.
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


server_b64  = b64(HERE / "server.py")
index_b64   = b64(HERE / "web" / "index.html")
appjs_b64   = b64(HERE / "web" / "app.js")
css_b64     = b64(HERE / "web" / "style.css")
req_b64     = b64(HERE / "requirements.txt")
cfg_b64     = b64(HERE / "config.json")

PS = r"""
# nx-ai-trainer Windows Setup Script
# Run as Administrator in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup.ps1

$ErrorActionPreference = "Stop"
$InstallDir = "$env:ProgramData\nx-ai-trainer"

Write-Host ""
Write-Host "=== nx-ai-trainer Setup ===" -ForegroundColor Cyan
Write-Host "Install directory: $InstallDir"
Write-Host ""

# -- [1/5] Create directories
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\web" | Out-Null
Write-Host "[1/5] Directories created" -ForegroundColor Green

# -- [2/5] Write application files
""".lstrip()

# Build file-write block
def write_block(varname, dest, b64_data):
    return (
        f'$b64 = "{b64_data}"\n'
        f'[System.IO.File]::WriteAllBytes("{dest}",'
        f' [System.Convert]::FromBase64String($b64))\n'
    )

PS += write_block("b64_server",  r"$InstallDir\server.py",       server_b64)
PS += write_block("b64_index",   r"$InstallDir\web\index.html",  index_b64)
PS += write_block("b64_appjs",   r"$InstallDir\web\app.js",      appjs_b64)
PS += write_block("b64_css",     r"$InstallDir\web\style.css",   css_b64)
PS += write_block("b64_req",     r"$InstallDir\requirements.txt", req_b64)
PS += 'Write-Host "[2/5] Application files written" -ForegroundColor Green\n'

PS += r"""
# -- [3/5] Write config.json (only if it doesn't exist - never overwrite user settings)
$configPath = "$InstallDir\config.json"
if (-not (Test-Path $configPath)) {
"""
PS += f'    $b64cfg = "{cfg_b64}"\n'
PS += r"""    [System.IO.File]::WriteAllBytes($configPath, [System.Convert]::FromBase64String($b64cfg))
    Write-Host "[3/5] config.json written - edit it with your Nx server credentials!" -ForegroundColor Yellow
    Write-Host "      File: $configPath" -ForegroundColor Yellow
} else {
    Write-Host "[3/5] config.json already exists - keeping your settings" -ForegroundColor Green
}

# -- [4/5] Install Python packages
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") { $pythonCmd = $cmd; break }
    } catch {}
}
if (-not $pythonCmd) {
    Write-Host "ERROR: Python 3 not found. Install from https://python.org and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "Found Python: $(& $pythonCmd --version 2>&1) ($pythonCmd)" -ForegroundColor Green

Write-Host "Installing Python packages from requirements.txt..." -ForegroundColor Cyan
& $pythonCmd -m pip install --upgrade pip --quiet
& $pythonCmd -m pip install -r "$InstallDir\requirements.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed." -ForegroundColor Red
    exit 1
}

# -- Optional: PyTorch for CNN / MobileNetV2 training
Write-Host ""
$installTorch = Read-Host "Install PyTorch for CNN/MobileNetV2 training? (Y/N)"
if ($installTorch -match "^[Yy]") {
    $hasCuda = $false
    try { nvidia-smi 2>&1 | Out-Null; $hasCuda = ($LASTEXITCODE -eq 0) } catch {}
    if ($hasCuda) {
        $torchUrl = "https://download.pytorch.org/whl/cu121"
        Write-Host "NVIDIA GPU detected - installing CUDA 12.1 build..." -ForegroundColor Green
    } else {
        $torchUrl = "https://download.pytorch.org/whl/cpu"
        Write-Host "No NVIDIA GPU - installing CPU build..." -ForegroundColor Yellow
    }
    & $pythonCmd -m pip install torch torchvision --index-url $torchUrl
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PyTorch installed." -ForegroundColor Green
    } else {
        Write-Host "WARNING: PyTorch install failed. CNN/MobileNetV2 methods will not be available." -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping PyTorch. Basic training method will still work." -ForegroundColor Gray
}
Write-Host "[4/5] Python packages installed" -ForegroundColor Green

# -- [5/5] Install Windows service via NSSM
$nssmPath = $null
# Check PATH first, then fixed locations
$nssmInPath = Get-Command nssm -ErrorAction SilentlyContinue
if ($nssmInPath) { $nssmPath = $nssmInPath.Source }
if (-not $nssmPath) {
    foreach ($p in @(
        "C:\tools\nssm.exe",
        "$env:ProgramData\nx-sip-client\nssm.exe",
        "$env:ProgramData\nssm\nssm.exe",
        "$env:ProgramData\nssm-2.24\win64\nssm.exe",
        "$env:ProgramFiles\nssm\nssm.exe"
    )) {
        if (Test-Path $p) { $nssmPath = $p; break }
    }
}

$ServiceName = "nx-ai-trainer"
if ($nssmPath) {
    # Remove legacy service name (install.bat used "NxAITrainer" before standardisation)
    $legacySvc = Get-Service -Name "NxAITrainer" -ErrorAction SilentlyContinue
    if ($legacySvc) {
        Write-Host "Removing legacy NxAITrainer service..." -ForegroundColor Yellow
        & $nssmPath stop "NxAITrainer" confirm 2>&1 | Out-Null
        & $nssmPath remove "NxAITrainer" confirm 2>&1 | Out-Null
    }
    $existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($existingService) {
        Write-Host "Stopping existing service..." -ForegroundColor Yellow
        & $nssmPath stop $ServiceName confirm 2>&1 | Out-Null
        & $nssmPath remove $ServiceName confirm 2>&1 | Out-Null
    }
    $pythonExe = (Get-Command $pythonCmd).Source
    & $nssmPath install $ServiceName $pythonExe
    & $nssmPath set $ServiceName AppParameters "$InstallDir\server.py"
    & $nssmPath set $ServiceName AppDirectory $InstallDir
    & $nssmPath set $ServiceName DisplayName "nx-ai-trainer"
    & $nssmPath set $ServiceName Description "Train AI models from Nx camera streams"
    # Delayed auto-start: waits until network/session services are ready (avoids Nx connection failure at boot)
    & $nssmPath set $ServiceName Start SERVICE_DELAYED_AUTO_START
    & $nssmPath set $ServiceName AppStdout "$InstallDir\service.log"
    & $nssmPath set $ServiceName AppStderr "$InstallDir\service.log"
    & $nssmPath set $ServiceName AppRotateFiles 1
    & $nssmPath set $ServiceName AppRotateBytes 1048576
    # Restart automatically if server.py crashes (5 s delay before restart)
    & $nssmPath set $ServiceName AppExit Default Restart
    & $nssmPath set $ServiceName AppRestartDelay 5000
    & $nssmPath start $ServiceName
    Write-Host "[5/5] Service installed and started" -ForegroundColor Green
} else {
    Write-Host "[5/5] NSSM not found - service NOT registered." -ForegroundColor Yellow
    Write-Host "      Download NSSM: https://nssm.cc/download" -ForegroundColor Yellow
    Write-Host "      Then re-run this script, or register the service manually." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To run without a service (for testing):" -ForegroundColor Cyan
    Write-Host "  python $InstallDir\server.py" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: Edit your Nx server credentials in:" -ForegroundColor Yellow
Write-Host "  $InstallDir\config.json" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then add a Web Page item in Nx client pointing to:" -ForegroundColor White
Write-Host "  http://localhost:8767" -ForegroundColor Cyan
Write-Host ""
"""

# Validate no non-ASCII slips through
raw = PS.encode("ascii")
non_ascii = [(i, b) for i, b in enumerate(raw) if b > 127]
if non_ascii:
    raise ValueError(f"Non-ASCII bytes at positions: {non_ascii[:5]}")

out = HERE / "installer" / "windows" / "setup.ps1"
bom = b"\xef\xbb\xbf"
out.write_bytes(bom + raw)
print(f"Written {len(bom + raw):,} bytes to {out}")
