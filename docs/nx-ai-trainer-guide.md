# nx-ai-trainer User Guide

**Version 1.0 — May 2026**

---

## Overview

nx-ai-trainer is a locally-hosted web application that lets you train custom AI image classification models directly from live Nx camera streams, then automatically deploy them to Nx AI Manager — no cloud services, no coding, no external tools required.

Open the app as a Web Page item inside the Nx client, capture frames from any camera, label them, train a model, and deploy it — all in a single browser session.

### What you can build

- Door open / door closed detection
- Object present / object absent detection
- Light on / light off detection
- Any scene-change or object classification task with 2–8 distinct visual states

---

## How it works

```
Nx Camera  →  nx-ai-trainer  →  Nx AI Manager  →  Live inference on camera
(captures)     (trains model)     (runs model)       (bounding box overlays)
```

nx-ai-trainer does not need to run on the same machine as the Nx server or Nx AI Manager. It only needs network access to the Nx server REST API (port 7001) and internet access to upload models to the Network Optix cloud.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.9 or later | 3.13 recommended; 3.14 supported for Basic training only |
| Nx Witness / Network Optix VMS | v6.0 or later |
| Nx AI Manager plugin | Installed and enabled on the Nx server |
| Network Optix cloud account | For model upload and deployment |
| NSSM (Windows service installs) | Included download instructions below |
| systemd (Linux) | Standard on Ubuntu 20.04+ |

> **PyTorch is optional.** The Basic training method works without it. CNN and MobileNetV2 methods require PyTorch, which the installer will offer to install.

---

## Installation

### Windows

1. Copy `setup.ps1` to the target machine (any location).
2. Open **PowerShell as Administrator**.
3. Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

4. When prompted, choose whether to install PyTorch for CNN/MobileNetV2 training.
5. If NSSM is not found, download `nssm.exe` to `C:\tools\nssm.exe` and re-run `setup.ps1`.

**Files are installed to:** `C:\ProgramData\nx-ai-trainer\`

**Service name:** `nx-ai-trainer` (delayed auto-start, restarts automatically on crash)

**Logs:** `C:\ProgramData\nx-ai-trainer\service.log`

---

### Linux (Ubuntu)

1. Copy `setup.sh` to the target machine.
2. Run as root:

```bash
sudo bash setup.sh
```

3. When prompted, choose whether to install PyTorch.

**Files are installed to:** `/opt/nx-ai-trainer/`

**Service name:** `nx-ai-trainer` (systemd, starts after network, restarts on crash)

**Logs:** `journalctl -u nx-ai-trainer -f`

---

## First-time configuration

After installation, edit the configuration file with your Nx server credentials before starting the service.

**Windows:** `C:\ProgramData\nx-ai-trainer\config.json`

**Linux:** `/opt/nx-ai-trainer/config.json`

```json
{
  "nx": {
    "host": "192.168.1.100",
    "port": 7001,
    "username": "admin",
    "password": "your_password"
  },
  "port": 8767
}
```

| Field | Description |
|-------|-------------|
| `host` | IP address or hostname of your Nx server |
| `port` | Nx server REST API port (default: 7001) |
| `username` | Nx administrator username |
| `password` | Nx administrator password |

After editing, restart the service:

```powershell
# Windows
Restart-Service nx-ai-trainer
```

```bash
# Linux
sudo systemctl restart nx-ai-trainer
```

You can also update the Nx server settings at any time from within the app using the **⚙ Settings** button in the top-right corner.

---

## Adding to Nx client

1. In the Nx client, add a new **Web Page** item to a layout.
2. Set the URL to `http://<machine-ip>:8767`
   - If nx-ai-trainer is on the same machine as the Nx client: `http://localhost:8767`
   - If on a separate machine: use that machine's IP address, e.g. `http://192.168.1.50:8767`
3. The app will open directly in the Nx layout panel.

---

## Using the app

The workflow follows four numbered steps across three panels.

### Step 1 — Camera

Select a camera from the dropdown. Three stream modes are available:

| Mode | Use case |
|------|----------|
| **Live** | Capture frames from the live stream |
| **Recorded** | Navigate to a specific time in recorded footage |
| **Bookmarks** | Jump to a saved bookmark and capture from there |

**Resolution:** Use **Lo** for faster capture and training. Use **Hi** if fine detail is needed for classification.

**Recorded playback controls:**
- Enter a date/time and press **▶** to jump to that moment
- Use **‹ ›** to step ±5 seconds, **« »** to step ±1 minute
- Press **▶ Play** to auto-advance through recorded footage

---

### Step 2 — Classes

Add a class for each visual state you want to detect.

**Examples:**
- `Door Open` / `Door Closed`
- `Person Present` / `No Person`
- `Light On` / `Light Off`

You need at least **2 classes** with **5 frames each** to train. More is better — aim for 50–75 frames per class.

**Capturing frames:**

1. Select a class tab at the bottom of the camera panel.
2. Click and hold **● Capture Frame** — frames are captured continuously while held.
3. Release to stop.

The class tab shows a running count of captured frames.

> **Tip:** Capture variety. Move the subject left, right, close, far. Vary the background where possible. A model trained on identical frames will overfit and fail in real use.

---

### Step 3 — Train

Choose a training method and click **▶ Train Model**.

| Method | Speed | Accuracy | Best for |
|--------|-------|----------|----------|
| **Basic** | ~5 sec | Good for static scenes | Light on/off, door open/closed — consistent backgrounds |
| **CNN** | ~4 min (CPU) | Better with variation | Moving subjects, varying angles |
| **MobileNetV2** | ~8 min (CPU) | Highest accuracy | Complex scenes, subtle differences |

> CNN and MobileNetV2 require PyTorch. If not installed, only Basic will be available.

**Reading the result:**

```
✓ [CNN] Accuracy: 94.0% | 246 samples | 367.9KB | Classes: Door Open, Door Closed
```

| Value | Meaning |
|-------|---------|
| ~85–95% | Healthy — model generalised well |
| 100% | Possible overfitting — may not perform well on new footage |
| ~50% | Underfitting — add more varied captures and retrain |

---

### Step 4 — Deploy

1. Enter a **model name** (used in Nx AI Manager).
2. Select the **camera** to assign the model to (defaults to the camera you captured from).
3. Click **↑ Upload & Assign**.

The app will:
1. Upload the ONNX model to Network Optix cloud
2. Wait for the model to finish compiling (~30 seconds)
3. Assign it to the selected camera in Nx AI Manager

Once deployed, the camera will show bounding box overlays for each detected class in the Nx client.

> **Note:** The Network Optix cloud account used for Sign In must have write permission on Nx AI Manager. A read-only OAuth token will produce a 403 error on upload.

---

## Sign in to Network Optix

Click **Sign In** in the top-right corner. A browser window will open to the Network Optix cloud login page. After signing in, the window closes and the status bar shows your email.

**Token expiry:** The app automatically refreshes your session token in the background. If the token cannot be refreshed, the status bar will show **"token expires in Xm — re-sign in"** as a warning. Click Sign Out then Sign In to get a fresh token.

---

## Training tips

- **50–75 original captures per class** is the sweet spot. The trainer doubles the dataset automatically with horizontal flip augmentation.
- **Vary your captures.** The most common cause of poor inference is a model trained on one angle/position. Capture the subject from left, right, center, close, and far.
- **Keep the scene consistent** between training and deployment. If the camera moves or lighting changes significantly after training, retrain.
- **Basic method is position-sensitive.** It detects pixel-level changes. Ideal for static objects (a door, a light switch) where the background is fixed.
- **CNN handles variation better.** Use it when the subject moves or appears at different positions.
- **100% training accuracy is a warning sign**, not a goal. It usually means the model memorised the training data. Add more varied captures and retrain.

---

## Updating

When a new version of nx-ai-trainer is released, run the update script as Administrator. It stops the service, replaces server.py and the web files, and restarts the service. Your `config.json` and training data are preserved.

**Windows:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\update_server.ps1
```

**Linux:**

```bash
sudo bash update_server.sh
```

---

## Service management

### Windows

```powershell
# Status
Get-Service nx-ai-trainer

# Start / Stop / Restart
Start-Service nx-ai-trainer
Stop-Service nx-ai-trainer
Restart-Service nx-ai-trainer

# View logs (last 50 lines)
Get-Content C:\ProgramData\nx-ai-trainer\service.log -Tail 50
```

### Linux

```bash
# Status
systemctl status nx-ai-trainer

# Start / Stop / Restart
sudo systemctl start nx-ai-trainer
sudo systemctl stop nx-ai-trainer
sudo systemctl restart nx-ai-trainer

# Live logs
journalctl -u nx-ai-trainer -f
```

---

## Troubleshooting

### Nx status shows red / "unreachable"

- Verify the host, port, username, and password in **⚙ Settings** or `config.json`.
- Confirm the Nx server is running and reachable on port 7001 from the nx-ai-trainer machine.
- Check the service log for `Cannot reach Nx server` errors.

### "Not signed in to Nx AI Manager"

- Click **Sign In** and complete the Network Optix cloud login.
- If the popup closes without completing, try again — a previous login attempt may have left a stale session.

### Upload blocked (403 Forbidden)

- Your cloud account may have read-only permissions on Nx AI Manager.
- Contact your Network Optix administrator to grant upload permissions.

### Training accuracy ~50%

- This usually means the model learned nothing — check that captures for each class look visually distinct.
- Ensure you have at least 20–30 captures per class with visual variety.
- Try the CNN method if using Basic.

### Service fails to start after reboot

- On Windows, verify the service is set to delayed auto-start: `sc.exe qc nx-ai-trainer` should show `START_TYPE: AUTO_START (DELAYED)`.
- On Linux, verify the service is enabled: `systemctl is-enabled nx-ai-trainer` should return `enabled`.

### PyTorch not available (CNN/MobileNetV2 greyed out)

- Run the PyTorch install command manually:

**Windows:**
```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Linux:**
```bash
sudo /opt/nx-ai-trainer/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Reset training data

Click **↺ Reset** in the top-right corner to delete all captured frames and start over. The trained model and deployment history in Nx AI Manager are not affected.

---

*nx-ai-trainer — Network Optix internal tool*
*Built with Flask · PyTorch · scikit-learn · ONNX*
