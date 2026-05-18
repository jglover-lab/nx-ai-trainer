#!/usr/bin/env python3
"""Generate installer/windows/update_server.ps1 from current source files.

Run after any change to server.py or web/ files.
Also regenerates setup.ps1 via make_setup_ps1.py.
"""
import base64
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def b64file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


server_b64  = b64file(HERE / "server.py")
index_b64   = b64file(HERE / "web" / "index.html")
appjs_b64   = b64file(HERE / "web" / "app.js")
css_b64     = b64file(HERE / "web" / "style.css")

ps_lines = [
    r"# update_server.ps1",
    r'$ErrorActionPreference = "Stop"',
    r'$dest    = Join-Path $env:ProgramData "nx-ai-trainer"',
    r'$webDest = Join-Path $dest "web"',
    r'$svcName = "nx-ai-trainer"',
    r'Write-Host "[update] Stopping service..."',
    r'$svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue',
    r'if ($svc -and $svc.Status -eq "Running") {',
    r'    Stop-Service -Name $svcName -Force -ErrorAction SilentlyContinue',
    r'    Start-Sleep -Seconds 2',
    r'}',
    r'if (-not (Test-Path $dest))    { New-Item -ItemType Directory -Path $dest    | Out-Null }',
    r'if (-not (Test-Path $webDest)) { New-Item -ItemType Directory -Path $webDest | Out-Null }',
    r'Write-Host "[update] Writing server.py..."',
    r"$b64 = @'",
    server_b64,
    r"'@",
    r'$bytes = [System.Convert]::FromBase64String($b64)',
    r'[System.IO.File]::WriteAllBytes((Join-Path $dest "server.py"), $bytes)',
    r'Write-Host "[update] server.py written ($($bytes.Length) bytes)"',
    r'Write-Host "[update] Writing web/index.html..."',
    r"$b64 = @'",
    index_b64,
    r"'@",
    r'$bytes = [System.Convert]::FromBase64String($b64)',
    r'[System.IO.File]::WriteAllBytes((Join-Path $webDest "index.html"), $bytes)',
    r'Write-Host "[update] Writing web/app.js..."',
    r"$b64 = @'",
    appjs_b64,
    r"'@",
    r'$bytes = [System.Convert]::FromBase64String($b64)',
    r'[System.IO.File]::WriteAllBytes((Join-Path $webDest "app.js"), $bytes)',
    r'Write-Host "[update] Writing web/style.css..."',
    r"$b64 = @'",
    css_b64,
    r"'@",
    r'$bytes = [System.Convert]::FromBase64String($b64)',
    r'[System.IO.File]::WriteAllBytes((Join-Path $webDest "style.css"), $bytes)',
    r'Write-Host "[update] Restarting service..."',
    r'$svc2 = Get-Service -Name $svcName -ErrorAction SilentlyContinue',
    r'if ($svc2) {',
    r'    Start-Service -Name $svcName -ErrorAction SilentlyContinue',
    r'    Start-Sleep -Seconds 2',
    r'    $svc3 = Get-Service -Name $svcName -ErrorAction SilentlyContinue',
    r'    Write-Host "[update] Service status: $($svc3.Status)"',
    r'} else {',
    r'    Write-Host "[update] Service not found - start manually"',
    r'}',
    r'Write-Host "[update] Done."',
]

content = "\r\n".join(ps_lines) + "\r\n"

out_path = HERE / "installer" / "windows" / "update_server.ps1"
bom = b"\xef\xbb\xbf"
raw = content.encode("ascii")

non_ascii = [(i, b) for i, b in enumerate(raw) if b > 127]
if non_ascii:
    raise ValueError(f"Non-ASCII bytes at positions: {non_ascii[:5]}")

out_path.write_bytes(bom + raw)
print(f"Written {len(bom + raw):,} bytes to {out_path}")

# Also regenerate setup.ps1 and Linux installers so everything stays in sync
for gen in ("make_setup_ps1.py", "make_setup_sh.py"):
    script = HERE / gen
    if script.exists():
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"WARNING: {gen} failed: {result.stderr.strip()}")
