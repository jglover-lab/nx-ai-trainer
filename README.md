# nx-ai-trainer

Train custom AI image classification models directly from live Nx camera streams, then automatically deploy them to Nx AI Manager — no cloud services, no coding required.

Open as a **Web Page** item inside the Nx client, capture frames, label them, train, and deploy — all in one browser session.

## Quick start

See [docs/nx-ai-trainer-guide.md](docs/nx-ai-trainer-guide.md) for full installation and usage instructions.

### Windows

1. Download `installer/windows/setup.ps1`
2. Run as Administrator in PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

### Linux (Ubuntu)

1. Download `installer/linux/setup.sh`
2. Run as root:
```bash
sudo bash setup.sh
```

## What you can build

- Door open / door closed detection
- Object present / absent detection
- Light on / off detection
- Any scene-change classification task with 2–8 visual states

## Requirements

- Python 3.9+
- Nx Witness or Nx Meta v6.0+
- Nx AI Manager plugin installed on the Nx server
- Network Optix cloud account

## Updating

```powershell
# Windows — run from C:\Windows\Temp\
.\update_server.ps1
```

```bash
# Linux
sudo bash update_server.sh
```

---

*Built with Flask · PyTorch · scikit-learn · ONNX*
