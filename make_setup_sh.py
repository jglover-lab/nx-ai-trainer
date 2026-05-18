#!/usr/bin/env python3
"""Generate installer/linux/setup.sh and installer/linux/update_server.sh
from current source files.

Run after any change to server.py, web/, requirements.txt, or config.json.
Called automatically by make_update_ps1.py.
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

# ── Shared helper ──────────────────────────────────────────────────────────────
def write_block(dest, b64_data):
    """Emit bash that base64-decodes b64_data into dest."""
    return (
        f"base64 -d > {dest} << 'B64EOF'\n"
        f"{b64_data}\n"
        "B64EOF\n"
    )

# ── setup.sh ───────────────────────────────────────────────────────────────────
SETUP = """\
#!/bin/bash
# nx-ai-trainer Linux Setup Script
# Run as root:  sudo bash setup.sh

set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run as root (sudo bash setup.sh)" >&2
    exit 1
fi

INSTALL_DIR="/opt/nx-ai-trainer"
SERVICE_NAME="nx-ai-trainer"

echo ""
echo "=== nx-ai-trainer Setup ==="
echo "Install directory: $INSTALL_DIR"
echo ""

# -- [1/5] Create directories
mkdir -p "$INSTALL_DIR/web"
echo "[1/5] Directories created"

# -- [2/5] Write application files
"""

SETUP += write_block('"$INSTALL_DIR/server.py"',       server_b64)
SETUP += write_block('"$INSTALL_DIR/web/index.html"',  index_b64)
SETUP += write_block('"$INSTALL_DIR/web/app.js"',      appjs_b64)
SETUP += write_block('"$INSTALL_DIR/web/style.css"',   css_b64)
SETUP += write_block('"$INSTALL_DIR/requirements.txt"', req_b64)
SETUP += 'echo "[2/5] Application files written"\n'

SETUP += """
# -- [3/5] Write config.json (only if it doesn't exist - never overwrite user settings)
if [ ! -f "$INSTALL_DIR/config.json" ]; then
"""
SETUP += "    " + write_block('"$INSTALL_DIR/config.json"', cfg_b64).replace("\n", "\n    ").rstrip() + "\n"
SETUP += """\
    echo "[3/5] config.json written - edit it with your Nx server credentials!"
    echo "      File: $INSTALL_DIR/config.json"
else
    echo "[3/5] config.json already exists - keeping your settings"
fi

# -- [4/5] Install Python packages (isolated venv avoids system-Python conflicts)
echo ""
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install with: apt install python3 python3-venv" >&2
    exit 1
fi

echo "Creating Python venv at $INSTALL_DIR/venv..."
python3 -m venv "$INSTALL_DIR/venv"

echo "Installing Python packages from requirements.txt..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip --quiet
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# -- Optional: PyTorch for CNN / MobileNetV2 training
echo ""
read -rp "Install PyTorch for CNN/MobileNetV2 training? (Y/N): " install_torch
if [[ "$install_torch" =~ ^[Yy]$ ]]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        TORCH_URL="https://download.pytorch.org/whl/cu121"
        echo "NVIDIA GPU detected - installing CUDA 12.1 build..."
    else
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        echo "No NVIDIA GPU - installing CPU build..."
    fi
    "$INSTALL_DIR/venv/bin/pip" install torch torchvision --index-url "$TORCH_URL"
    if [ $? -eq 0 ]; then
        echo "PyTorch installed."
    else
        echo "WARNING: PyTorch install failed. CNN/MobileNetV2 methods will not be available."
    fi
else
    echo "Skipping PyTorch. Basic training method will still work."
fi
echo "[4/5] Python packages installed"

# -- [5/5] Install systemd service
PYTHON_EXE="$INSTALL_DIR/venv/bin/python"

cat > /etc/systemd/system/${SERVICE_NAME}.service << UNIT
[Unit]
Description=nx-ai-trainer - Train AI models from Nx camera streams
After=network.target

[Service]
Type=simple
ExecStart=${PYTHON_EXE} ${INSTALL_DIR}/server.py
WorkingDirectory=${INSTALL_DIR}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
echo "[5/5] Service installed and started"

echo ""
echo "=== Setup complete ==="
echo ""
echo "IMPORTANT: Edit your Nx server credentials in:"
echo "  $INSTALL_DIR/config.json"
echo ""
echo "Then add a Web Page item in Nx client pointing to:"
echo "  http://$(hostname -I | awk '{print $1}'):8767"
echo "  or http://localhost:8767"
echo ""
echo "To view logs:  journalctl -u $SERVICE_NAME -f"
echo "To restart:    systemctl restart $SERVICE_NAME"
echo ""
"""

# ── update_server.sh ───────────────────────────────────────────────────────────
UPDATE = """\
#!/bin/bash
# nx-ai-trainer Linux Update Script
# Run as root:  sudo bash update_server.sh

set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run as root (sudo bash update_server.sh)" >&2
    exit 1
fi

INSTALL_DIR="/opt/nx-ai-trainer"
SERVICE_NAME="nx-ai-trainer"

echo "[update] Stopping service..."
systemctl stop "$SERVICE_NAME" 2>/dev/null || true

mkdir -p "$INSTALL_DIR/web"

echo "[update] Writing server.py..."
"""

UPDATE += write_block('"$INSTALL_DIR/server.py"', server_b64)
UPDATE += 'echo "[update] Writing web files..."\n'
UPDATE += write_block('"$INSTALL_DIR/web/index.html"', index_b64)
UPDATE += write_block('"$INSTALL_DIR/web/app.js"',     appjs_b64)
UPDATE += write_block('"$INSTALL_DIR/web/style.css"',  css_b64)

UPDATE += """\
echo "[update] Restarting service..."
systemctl start "$SERVICE_NAME" 2>/dev/null && echo "[update] Service status: $(systemctl is-active $SERVICE_NAME)" || echo "[update] Service not found - start manually: python3 $INSTALL_DIR/server.py"
echo "[update] Done."
"""

# ── Write files ────────────────────────────────────────────────────────────────
out_dir = HERE / "installer" / "linux"
out_dir.mkdir(parents=True, exist_ok=True)

setup_path  = out_dir / "setup.sh"
update_path = out_dir / "update_server.sh"

setup_path.write_text(SETUP,  encoding="utf-8")
update_path.write_text(UPDATE, encoding="utf-8")

# Make executable
import stat
for p in (setup_path, update_path):
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

print(f"Written {setup_path.stat().st_size:,} bytes to {setup_path}")
print(f"Written {update_path.stat().st_size:,} bytes to {update_path}")
