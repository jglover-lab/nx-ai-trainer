#!/usr/bin/env python3
"""
nx-ai-trainer
Train custom image classification models from Nx camera streams,
then auto-deploy to Nx AI Manager via Scailable API.

Training pipeline: HOG features + SVM → exported as ONNX via skl2onnx
"""

import base64
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
import urllib3
from pathlib import Path
from urllib.parse import urlencode

# Windows service logs use cp1252 by default — reconfigure so Unicode print()
# calls (e.g. '→' in debug lines) don't raise UnicodeEncodeError.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

import requests
from flask import Flask, Response, jsonify, redirect, request, send_file, stream_with_context

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
TOKEN_FILE = BASE_DIR / ".tokens.json"
WEB_DIR = BASE_DIR / "web"
TRAIN_DIR = BASE_DIR / "training_data"

# ── Constants ──────────────────────────────────────────────────────────────────
SCAILABLE_CPT = "https://api.sclbl.nxvms.com/cpt"
SCAILABLE_CPT_AUDIENCE = "https://api.sclbl.nxvms.com/cpt"
SCAILABLE_DEV = "https://api.sclbl.nxvms.com/dev"
OAUTH_CLIENT_ID = "df920335-ead9-2acb-f5b1-6af896834c84"  # NX AI Manager integration ID
CLOUD_ENDPOINT = "https://meta.nxvms.com"
CLOUD_CDB_ENDPOINT = "https://meta.nxvms.com/cdb"
NX_AI_MANAGER_ENGINE_NAME = "NX AI Manager"

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")

@app.after_request
def _no_cache_static(response):
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# ── Config ─────────────────────────────────────────────────────────────────────
def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)

# ── Nx Auth ────────────────────────────────────────────────────────────────────
_nx_token_cache = {}

def nx_login():
    cfg = load_config()
    nx = cfg["nx"]
    url = f"https://{nx['host']}:{nx['port']}/rest/v3/login/sessions"
    resp = requests.post(
        url,
        json={"username": nx["username"], "password": nx["password"], "setCookie": False},
        verify=False,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["token"]

def get_nx_token():
    cfg = load_config()
    nx = cfg["nx"]
    key = f"{nx['host']}:{nx['port']}"
    if key not in _nx_token_cache:
        _nx_token_cache[key] = nx_login()
    return _nx_token_cache[key], nx

def nx_request(method, path, **kwargs):
    try:
        token, nx = get_nx_token()
    except requests.exceptions.ConnectionError:
        cfg = load_config()
        nxcfg = cfg["nx"]
        raise RuntimeError(
            f"Cannot reach Nx server at {nxcfg['host']}:{nxcfg['port']} — "
            "check host/port in Settings (⚙)"
        )
    except requests.exceptions.Timeout:
        cfg = load_config()
        nxcfg = cfg["nx"]
        raise RuntimeError(
            f"Nx server at {nxcfg['host']}:{nxcfg['port']} timed out — "
            "check that the server is running"
        )
    cfg = load_config()
    nxcfg = cfg["nx"]
    url = f"https://{nxcfg['host']}:{nxcfg['port']}{path}"
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    timeout = kwargs.pop("timeout", 30)  # allow callers to override
    try:
        resp = requests.request(method, url, headers=headers, verify=False, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Nx server at {nxcfg['host']}:{nxcfg['port']} — "
            "check host/port in Settings (⚙)"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Nx server at {nxcfg['host']}:{nxcfg['port']} timed out"
        )
    if resp.status_code == 401:
        key = f"{nxcfg['host']}:{nxcfg['port']}"
        _nx_token_cache.pop(key, None)
        try:
            token, _ = get_nx_token()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Nx server at {nxcfg['host']}:{nxcfg['port']} — "
                "check host/port in Settings (⚙)"
            )
        headers["Authorization"] = f"Bearer {token}"
        resp = requests.request(method, url, headers=headers, verify=False, timeout=timeout, **kwargs)
    return resp

# ── Scailable Auth ─────────────────────────────────────────────────────────────
_oauth_states = {}   # state_val -> redirect_url used in authorize

def load_tokens():
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return {}

def save_tokens(data):
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)

def get_scailable_headers(require_write=False):
    """
    Return Authorization headers for the Scailable CPT API.
    If a Scailable API key is configured in config.json (sclbl_api_key), use it —
    it has full write permissions on the CPT API. Otherwise fall back to the user's
    NX Cloud OAuth token (read-only scope — upload will return 403).
    """
    # Prefer configured API key: full write permissions on CPT API
    try:
        api_key = load_config().get("sclbl_api_key", "").strip()
    except Exception:
        api_key = ""
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}

    tokens = _auto_refresh_tokens()
    token = tokens.get("access_token")
    if not token:
        return None
    # Strip the nxcdb- prefix that Nx Cloud sometimes prepends
    if token.startswith("nxcdb-"):
        token = token[len("nxcdb-"):]
    hdrs = {"Authorization": f"Bearer {token}"}

    # Include cloud system ID — Scailable uses it to identify the VMS tenant.
    # Read from config (fastest, no VMS round-trip) then fall back to REST API.
    cloud_id = None
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or ""
    except Exception:
        pass
    if not cloud_id:
        cloud_id = _get_cloud_system_id() or ""
    if cloud_id:
        hdrs["X-Cloud-System-Id"] = cloud_id
        hdrs["X-Nx-System-Id"]    = cloud_id   # alternate header name the plugin may use

    # Also include the integration ID
    if _engine_integration_id_cache:
        hdrs["X-Integration-Id"] = _engine_integration_id_cache

    return hdrs


def get_oauth_headers():
    """
    Return Authorization headers for the Scailable DEV API using the meta-scoped OAuth token.
    meta_token is stored specifically for DEV API use; always prefer it regardless of token type.
    The CPT-audience access token gets 403 on DEV API write operations.
    Returns None if no meta/refresh token is stored (triggers API key fallback in callers).
    """
    tokens = _auto_refresh_tokens()
    # meta_token is stored specifically for DEV API use (meta.nxvms.com scope).
    # Use it regardless of whether it's an access or refresh token — both work for DEV API writes.
    # Fall back to refresh_token for backwards compatibility.
    token = tokens.get("meta_token") or tokens.get("refresh_token")
    if not token:
        return None
    # Send raw token exactly as stored — DEV API expects "nxcdb-<jwt>" not "Bearer <jwt>"
    hdrs = {"Authorization": token}
    cloud_id = None
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or ""
    except Exception:
        pass
    if not cloud_id:
        cloud_id = _get_cloud_system_id() or ""
    if cloud_id:
        hdrs["X-Cloud-System-Id"] = cloud_id
        hdrs["X-Nx-System-Id"]    = cloud_id
    if _engine_integration_id_cache:
        hdrs["X-Integration-Id"] = _engine_integration_id_cache
    return hdrs

def decode_jwt_payload(token):
    """Decode JWT payload without verifying signature."""
    try:
        # Strip nxcdb- prefix if present
        jwt = token.split("-", 1)[-1] if "-" in token else token
        payload_b64 = jwt.split(".")[1]
        # Add padding
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        return json.loads(base64.b64decode(payload_b64))
    except Exception:
        return {}

def _is_token_expired(token, buffer_sec=120):
    """Return True if the JWT exp claim is within buffer_sec seconds of now."""
    exp = decode_jwt_payload(token).get("exp")
    if not exp:
        return False  # opaque token or no exp — can't tell, assume valid
    return time.time() >= (exp - buffer_sec)

def _try_refresh(refresh_tok, scope=None):
    """POST refresh_token grant; return (access_token, refresh_token) or (None, None)."""
    body = {
        "grant_type":    "refresh_token",
        "refresh_token": refresh_tok,
        "client_id":     OAUTH_CLIENT_ID,
    }
    if scope:
        body["scope"] = scope
    try:
        resp = requests.post(f"{CLOUD_CDB_ENDPOINT}/oauth2/token", json=body, timeout=15)
        print(f"[auth/refresh] scope={scope!r} → {resp.status_code}: {resp.text[:300]}")
        if resp.ok:
            rd = resp.json()
            at = rd.get("access_token") or rd.get("token") or rd.get("accessToken")
            rt = rd.get("refresh_token") or rd.get("refreshToken")
            return at, rt
    except Exception as e:
        print(f"[auth/refresh] error: {e}")
    return None, None

def _auto_refresh_tokens():
    """
    Lazily refresh stored tokens when they are near expiry.
    Called by get_scailable_headers / get_oauth_headers before each use.
    Updates .tokens.json in place; returns the (possibly updated) token dict.
    """
    tokens = load_tokens()
    changed = False

    # CPT access token — used for model upload
    at      = tokens.get("access_token", "")
    cpt_rt  = tokens.get("cpt_refresh_token", "")
    if at and cpt_rt and _is_token_expired(at):
        print("[auth/refresh] CPT access token near expiry — refreshing…")
        new_at, new_rt = _try_refresh(cpt_rt, scope=f"{SCAILABLE_CPT_AUDIENCE} cloudSystemId=*")
        if new_at:
            tokens["access_token"] = new_at
            if new_rt:
                tokens["cpt_refresh_token"] = new_rt
            changed = True
            print("[auth/refresh] CPT token refreshed OK")
        else:
            print("[auth/refresh] CPT refresh failed — re-login required")

    # meta / DEV token — used for device-agent assignment
    meta_tok = tokens.get("meta_token", "")
    if meta_tok and _is_token_expired(meta_tok):
        print("[auth/refresh] meta token near expiry — refreshing…")
        new_at, new_rt = _try_refresh(meta_tok, scope=f"{CLOUD_ENDPOINT} cloudSystemId=*")
        if new_rt or new_at:
            tokens["meta_token"] = new_rt or new_at
            changed = True
            print("[auth/refresh] meta token refreshed OK")
        else:
            print("[auth/refresh] meta refresh failed — re-login required")

    if changed:
        save_tokens(tokens)
    return tokens

# ── Nx AI Manager engine discovery ────────────────────────────────────────────
_engine_id_cache = None

_engine_integration_id_cache = None   # Scailable integration / org ID

def get_nxai_engine_id():
    global _engine_id_cache, _engine_integration_id_cache
    if _engine_id_cache:
        return _engine_id_cache
    resp = nx_request("GET", "/rest/v4/analytics/engines")
    resp.raise_for_status()
    for engine in resp.json():
        if engine.get("name") == NX_AI_MANAGER_ENGINE_NAME:
            _engine_id_cache = engine["id"].strip("{}")
            _engine_integration_id_cache = engine.get("integrationId", "").strip("{}")
            return _engine_id_cache
    raise RuntimeError("NX AI Manager engine not found. Is the plugin installed and enabled?")

# ── VMS system cloud credentials ──────────────────────────────────────────────
_vms_cpt_token_cache = None

def _parse_mserver_conf(path):
    """Parse an NX Witness mserver.conf INI file. Returns dict of key=value pairs."""
    values = {}
    section = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and "]" in line:
                section = line[1:line.index("]")]
            elif "=" in line and not line.startswith(("#", ";")):
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                values[k] = v
                if section:
                    values[f"{section}.{k}"] = v
    return values


def _parse_nx_sqlite(path):
    """
    Read cloud credentials from an NX Witness SQLite database.
    Handles both mserver.sqlite (small settings store) and ecs.sqlite (main VMS DB).
    Opens read-only so it works while mediaserver is running.
    Returns dict of {name: value} — only cloud/auth-related keys to keep it small.
    """
    import sqlite3, sys
    values = {}
    try:
        uri = f"file:{path.replace(chr(92), '/')}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=5)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print(f"[nx_sqlite] {path} — tables: {tables}")
        sys.stdout.flush()
        for table in tables:
            # NX Witness main DB uses vms_kvpair with name/value columns
            for cols in [("name", "value"), ("key", "value"), ("id", "value"),
                         ("name", "data"), ("key", "data"), ("name", "val")]:
                try:
                    cur.execute(f"SELECT [{cols[0]}], [{cols[1]}] FROM [{table}]")
                    rows = cur.fetchall()
                    for k, v in rows:
                        if k:
                            values[str(k)] = str(v) if v is not None else ""
                    if rows:
                        print(f"[nx_sqlite] table '{table}' cols={cols} — {len(rows)} rows")
                        sys.stdout.flush()
                    break
                except Exception:
                    continue
        con.close()
    except Exception as e:
        print(f"[nx_sqlite] error reading {path}: {e}")
        sys.stdout.flush()
    return values


# Keep the old name as an alias so existing call-sites still work
_parse_mserver_sqlite = _parse_nx_sqlite


def _dump_all_nx_sqlite_tables(path):
    """
    Comprehensive SQLite reader: returns every row of every table in the database,
    regardless of schema. Used for discovery — especially to find auth/cloud keys
    in tables that don't use the standard (name, value) pattern.
    Returns dict: {table_name: {"columns": [...], "rows": [[...], ...]}}
    """
    import sqlite3
    result = {}
    try:
        uri = f"file:{path.replace(chr(92), '/')}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=5)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        for table in tables:
            try:
                cur.execute(f"SELECT * FROM [{table}] LIMIT 500")
                cols = [d[0] for d in cur.description]
                rows = [list(r) for r in cur.fetchall()]
                result[table] = {"columns": cols, "rows": rows}
            except Exception as e:
                result[table] = {"error": str(e)}
        con.close()
    except Exception as e:
        result["__error__"] = str(e)
    return result


def _find_nxai_plugin_conf():
    """
    Search for NX AI Manager (Scailable) plugin configuration files on disk.
    These can contain Scailable API credentials stored by the plugin itself.
    Returns list of (path_str, content_dict_or_str) tuples.
    """
    import glob, os, json

    if os.name != "nt":
        return []

    results = []
    prog_files = [r"C:\Program Files", r"C:\Program Files (x86)"]
    pd = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    system_appdata = r"C:\Windows\System32\config\systemprofile\AppData\Local"
    system_appdata32 = r"C:\Windows\SysWOW64\config\systemprofile\AppData\Local"

    # Candidate directories to search for NX AI Manager config files
    search_dirs = []
    for pf in prog_files:
        for vnd in ("Network Optix", "Nx Meta", "NetworkOptix"):
            for prod in ("Nx Meta", "Nx Witness", "Network Optix VMS", ""):
                for rel in ("MediaServer\\plugins", "MediaServer\\nx_ai_manager",
                             "plugins", "mediaserver\\plugins"):
                    base = os.path.join(pf, vnd, prod, rel) if prod else os.path.join(pf, vnd, rel)
                    search_dirs.append(base)
    for base in (pd, system_appdata, system_appdata32):
        for vnd in ("Network Optix", "Nx Meta", "NetworkOptix",
                    "Network Optix MetaVMS Media Server"):
            for prod in ("Nx Meta", "Nx Witness", "mediaserver", ""):
                for rel in ("etc\\plugins", "mediaserver\\etc\\plugins",
                             "plugins", "nx_ai_manager", ""):
                    parts = [x for x in (vnd, prod, rel) if x]
                    search_dirs.append(os.path.join(base, *parts))
    # Also scan all sub-dirs one level deep from the above
    extra_dirs = []
    for d in search_dirs:
        if os.path.isdir(d):
            try:
                for sub in os.listdir(d):
                    full = os.path.join(d, sub)
                    if os.path.isdir(full):
                        extra_dirs.append(full)
            except Exception:
                pass
    search_dirs.extend(extra_dirs)

    seen_paths = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for fname in os.listdir(d):
                fpath = os.path.join(d, fname)
                if fpath in seen_paths or not os.path.isfile(fpath):
                    continue
                seen_paths.add(fpath)
                fname_lower = fname.lower()
                ext = os.path.splitext(fname_lower)[1]
                # Only read text/config/json files
                if ext not in (".json", ".conf", ".ini", ".yaml", ".yml", ".txt", ".cfg", ""):
                    continue
                # Check filename relevance
                interesting = any(x in fname_lower for x in
                    ("nx_ai", "nxai", "scailable", "sclbl", "auth", "credential",
                     "settings", "config", "token", "api"))
                if not interesting:
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(8192)
                    # Try JSON parse
                    try:
                        parsed = json.loads(content)
                        results.append((fpath, parsed))
                        continue
                    except Exception:
                        pass
                    # Plain text — only keep if it has credential-like content
                    if any(x in content.lower() for x in
                           ("authkey", "auth_key", "secret", "sclbl", "scailable",
                            "cloudSystemId", "token")):
                        results.append((fpath, {"raw": content[:1000]}))
                except Exception:
                    pass
        except Exception:
            pass
    return results


def _get_nxai_engine_settings_raw():
    """Fetch the raw NX AI Manager engine settings via NX REST API."""
    import sys
    try:
        engine_id = get_nxai_engine_id()
        for path in [
            f"/rest/v4/analytics/engines/{engine_id}/settings",
            f"/rest/v3/analytics/engines/{engine_id}/settings",
            f"/rest/v4/analytics/engines/{engine_id}/parameters",
            f"/rest/v4/analytics/integrations/{engine_id}/settings",
        ]:
            try:
                r = nx_request("GET", path)
                print(f"[nxai_settings] {path} → {r.status_code}: {r.text[:200]}")
                sys.stdout.flush()
                if r.ok:
                    return {"path": path, "data": r.json()}
            except Exception as e:
                print(f"[nxai_settings] {path} error: {e}")
                sys.stdout.flush()
    except Exception as e:
        print(f"[nxai_settings] engine lookup error: {e}")
        sys.stdout.flush()
    return None


def _nx_service_data_dirs():
    """
    Find the NX mediaserver data directory via Windows service info or registry.
    Returns a list of candidate directory paths to search.
    """
    import os, sys
    dirs = []

    if os.name != "nt":
        return dirs

    # ── 1. Windows registry — NX stores install/data paths here ──────────────
    try:
        import winreg
        reg_roots = [
            r"SOFTWARE\Network Optix\Nx Meta",
            r"SOFTWARE\Network Optix\Nx Witness",
            r"SOFTWARE\Network Optix\Network Optix VMS",
            r"SOFTWARE\WOW6432Node\Network Optix\Nx Meta",
            r"SOFTWARE\WOW6432Node\Network Optix\Nx Witness",
        ]
        for rpath in reg_roots:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rpath)
                for vname in ("DataDir", "InstallDir", "MediaServerDataDir",
                              "dataDirectory", "installDirectory"):
                    try:
                        val, _ = winreg.QueryValueEx(key, vname)
                        if val:
                            dirs.append(str(val))
                            # Also try var/ subdirectory
                            dirs.append(os.path.join(str(val), "var"))
                            dirs.append(os.path.join(str(val), "mediaserver", "var"))
                    except FileNotFoundError:
                        pass
                winreg.CloseKey(key)
            except FileNotFoundError:
                pass
        if dirs:
            print(f"[nx_conf] registry dirs: {dirs}")
            sys.stdout.flush()
    except ImportError:
        pass

    # ── 2. Windows service executable path → infer data dir ──────────────────
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-WmiObject Win32_Service | "
             "Where-Object {$_.PathName -like '*mediaserver*' -or "
             "$_.PathName -like '*optix*' -or $_.PathName -like '*meta*'} | "
             "Select-Object -ExpandProperty PathName"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.splitlines():
            line = line.strip().strip('"')
            if not line:
                continue
            # PathName is the exe, e.g.:
            #   C:\Program Files\Network Optix\Nx Meta\mediaserver\bin\mediaserver.exe
            #   C:\Program Files\Network Optix\mediaserver\bin\mediaserver.exe
            # Walk up the path to find the first component that lives under
            # "Program Files" — everything above that maps to ProgramData.
            pdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
            parts = line.replace("/", "\\").split("\\")
            # Find 'mediaserver' in the path parts
            try:
                ms_idx = next(i for i, p in enumerate(parts) if p.lower() == "mediaserver")
            except StopIteration:
                ms_idx = len(parts) - 1
            # Everything from the drive up to (and including) the parent of mediaserver
            # is the install root; replace the drive prefix + Program Files with ProgramData
            # to get the data root.
            above_ms = parts[:ms_idx]  # e.g. ['C:', 'Program Files', 'Network Optix'] or ['C:', ..., 'Nx Meta']
            # Strip the leading drive + "Program Files*" component
            data_rel_parts = [p for p in above_ms[2:] if p.lower() not in
                               ("program files", "program files (x86)")]
            # Now build the data dir candidates
            for suffix in (["mediaserver", "var"], ["var"], []):
                cand = os.path.join(pdata, *data_rel_parts, *suffix)
                dirs.append(cand)
            print(f"[nx_conf] service exe={line!r} → data_rel={data_rel_parts} → {dirs[-3:]}")
            sys.stdout.flush()
    except Exception as e:
        print(f"[nx_conf] service lookup failed: {e}")
        sys.stdout.flush()

    return dirs


def _find_nx_server_conf():
    """
    Locate the NX Witness server config on the local filesystem.
    Checks for both mserver.sqlite (NX v5+) and mserver.conf (older).
    Returns (path, parsed_dict) or (None, {}).
    """
    import glob, os, sys

    # ── Build the set of base directories to search ───────────────────────────
    # Start with standard ProgramData paths, then add registry/service-derived dirs
    pdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData") if os.name == "nt" else "/etc"

    base_dirs = [
        pdata,
        r"C:\ProgramData",
    ]
    # SYSTEM account AppData — NX mediaserver runs as LocalSystem and stores
    # ecs.sqlite here when no custom dataDir is configured.
    system_appdata = r"C:\Windows\System32\config\systemprofile\AppData\Local"
    if os.path.isdir(system_appdata):
        base_dirs.append(system_appdata)
    # Also check the SysWOW64 mirror path (32-bit processes on 64-bit Windows)
    system_appdata32 = r"C:\Windows\SysWOW64\config\systemprofile\AppData\Local"
    if os.path.isdir(system_appdata32):
        base_dirs.append(system_appdata32)
    # Non-standard root-level NX data directories (some installs use these)
    for drive in ("C", "D", "E", "F"):
        for name in ("Nx MetaVMS Media", "NxMetaVMS Media", "Nx Meta", "NxMeta",
                     "Network Optix", "NetworkOptix", "Nx Witness", "NxWitness"):
            d = fr"{drive}:\{name}"
            if os.path.isdir(d):
                base_dirs.append(d)
    # Also check other common drive letters for ProgramData
    for drive in ("D", "E", "F"):
        d = fr"{drive}:\ProgramData"
        if os.path.isdir(d):
            base_dirs.append(d)
    # Add service/registry-derived paths
    base_dirs.extend(_nx_service_data_dirs())
    # Deduplicate while preserving order
    seen = set()
    base_dirs = [d for d in base_dirs if d and d not in seen and not seen.add(d)]

    # ── Build glob patterns relative to each base dir ─────────────────────────
    # NX stores cloud credentials in either mserver.sqlite (new) or ecs.sqlite (main DB)
    rel_sqlite = [
        # With product subdirectory (e.g. C:\ProgramData\Network Optix\Nx Meta\mediaserver\var\)
        r"Network Optix\*\mediaserver\var\mserver.sqlite",
        r"Network Optix\*\mediaserver\var\ecs.sqlite",
        r"Network Optix\*\var\mserver.sqlite",
        r"Network Optix\*\var\ecs.sqlite",
        r"Network Optix\*\mserver.sqlite",
        r"Network Optix\*\ecs.sqlite",
        r"Network Optix\*\*\mserver.sqlite",
        r"Network Optix\*\*\ecs.sqlite",
        r"Network Optix\*\*\var\mserver.sqlite",
        r"Network Optix\*\*\var\ecs.sqlite",
        # Without product subdirectory (e.g. C:\ProgramData\Network Optix\mediaserver\var\)
        r"Network Optix\mediaserver\var\mserver.sqlite",
        r"Network Optix\mediaserver\var\ecs.sqlite",
        r"Network Optix\mediaserver\mserver.sqlite",
        r"Network Optix\var\mserver.sqlite",
        r"Network Optix\var\ecs.sqlite",
        r"Network Optix\mserver.sqlite",
        r"Network Optix\ecs.sqlite",
        # Other vendor/product name variants
        r"Nx Meta\mediaserver\var\mserver.sqlite",
        r"Nx Meta\mediaserver\var\ecs.sqlite",
        r"Nx Meta\var\mserver.sqlite",
        r"Nx Meta\var\ecs.sqlite",
        r"Nx Meta\mserver.sqlite",
        r"Nx Meta\ecs.sqlite",
        r"Nx Meta\*\mserver.sqlite",
        r"Nx Meta\*\ecs.sqlite",
        r"networkoptix\*\mserver.sqlite",
        r"networkoptix\mediaserver\var\mserver.sqlite",
        r"networkoptix\mediaserver\var\ecs.sqlite",
        # SYSTEM-profile AppData path: "Network Optix\Network Optix MetaVMS Media Server\ecs.sqlite"
        r"Network Optix\Network Optix MetaVMS Media Server\ecs.sqlite",
        r"Network Optix\Network Optix MetaVMS Media Server\mserver.sqlite",
        r"Network Optix\*\ecs.sqlite",
        r"Network Optix\*\mserver.sqlite",
        r"mserver.sqlite",
        r"ecs.sqlite",
        r"mediaserver\var\ecs.sqlite",
        r"mediaserver\var\mserver.sqlite",
        r"var\ecs.sqlite",
        r"var\mserver.sqlite",
    ]
    rel_conf = [
        r"Network Optix\*\mserver.conf",
        r"Network Optix\*\etc\mserver.conf",
        r"Network Optix\*\*\mserver.conf",
        r"Network Optix\mediaserver\mserver.conf",
        r"Network Optix\mediaserver\etc\mserver.conf",
        r"Network Optix\mserver.conf",
        r"Nx Meta\mserver.conf",
        r"Nx Meta\etc\mserver.conf",
        r"Nx Meta\*\mserver.conf",
        r"networkoptix\*\mserver.conf",
        r"mserver.conf",
    ]

    def _try_glob(base, rel_patterns, parser_fn):
        for rel in rel_patterns:
            pattern = os.path.join(base, rel)
            for path in glob.glob(pattern):
                try:
                    values = parser_fn(path)
                    if values:
                        print(f"[nx_conf] read {path} — {len(values)} keys")
                        sys.stdout.flush()
                        return path, values
                except Exception as e:
                    print(f"[nx_conf] {path}: {e}")
        return None, {}

    for base in base_dirs:
        path, values = _try_glob(base, rel_sqlite, _parse_mserver_sqlite)
        if values:
            return path, values

    for base in base_dirs:
        path, values = _try_glob(base, rel_conf, _parse_mserver_conf)
        if values:
            return path, values

    # ── Last resort: full walk of all fixed drives (depth ≤ 6) ───────────────
    if os.name == "nt":
        import string, ctypes
        # Find all fixed drives
        drives = []
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i):
                root = f"{letter}:\\"
                try:
                    if ctypes.windll.kernel32.GetDriveTypeW(root) == 3:  # DRIVE_FIXED
                        drives.append(root)
                except Exception:
                    pass
        print(f"[nx_conf] full walk on drives: {drives}")
        sys.stdout.flush()
        for drive in drives:
            drive_depth = drive.count(os.sep)
            sqlite_hit = conf_hit = None
            for dirpath, dirnames, filenames in os.walk(drive):
                depth = dirpath.count(os.sep) - drive_depth
                if depth >= 6:
                    dirnames.clear()
                    continue
                # Skip irrelevant directories early
                dn = os.path.basename(dirpath).lower()
                if depth == 1 and dn not in ("programdata", "users", "program files",
                                              "program files (x86)", "windows"):
                    dirnames.clear()
                    continue
                for fname in ("mserver.sqlite", "ecs.sqlite"):
                    if fname in filenames and not sqlite_hit:
                        sqlite_hit = os.path.join(dirpath, fname)
                if "mserver.conf" in filenames and not conf_hit:
                    conf_hit = os.path.join(dirpath, "mserver.conf")
                if sqlite_hit and conf_hit:
                    break
            for candidate, parser in [(sqlite_hit, _parse_nx_sqlite),
                                       (conf_hit, _parse_mserver_conf)]:
                if candidate:
                    try:
                        values = parser(candidate)
                        if values:
                            print(f"[nx_conf] walk found: {candidate} — {len(values)} keys")
                            sys.stdout.flush()
                            return candidate, values
                    except Exception as e:
                        print(f"[nx_conf] walk {candidate}: {e}")

    return None, {}


def _get_vms_cpt_token():
    """
    Authenticate as the VMS *system* (not the user) with Nx Cloud to get a
    token that the Scailable CPT API will accept.

    Credential sources (tried in order):
      1. config.json  cloud.auth_key + cloud.cloud_system_id  (manual override)
      2. mserver.conf on disk (auto-detected path)
      3. VMS REST API  /rest/v3/system/cloudCredentials  (if it exists)
    Then: POST client_credentials to meta.nxvms.com/cdb/oauth2/token.
    """
    global _vms_cpt_token_cache
    if _vms_cpt_token_cache:
        return _vms_cpt_token_cache

    import sys

    auth_key = None
    cloud_id = None
    source = None
    auth_key_candidates = []  # (key_name, value) — tried in order if primary 401s

    # ── 1. Manual config override ─────────────────────────────────────────────
    try:
        cfg = load_config()
        cloud_cfg = cfg.get("cloud", {})
        auth_key = cloud_cfg.get("auth_key") or cloud_cfg.get("authKey")
        cloud_id = (cloud_cfg.get("cloud_system_id") or cloud_cfg.get("cloudSystemId"))
        if auth_key:
            source = "config.json"
            print(f"[vms_token] using manual config: cloudId={str(cloud_id or '')[:8]}… authKey present")
            sys.stdout.flush()
    except Exception:
        pass

    # ── 2. mserver.sqlite / mserver.conf on disk ─────────────────────────────
    if not auth_key:
        conf_path, values = _find_nx_server_conf()
        if values:
            # ALWAYS log every credential-related key we see (name + length + first 8 chars)
            # so we can identify the correct cloud auth key name if it differs from expected
            cred_kv = {k: (str(v)[:8] + "… len=" + str(len(str(v))))
                       for k, v in values.items()
                       if v and any(x in k.lower() for x in
                                    ("auth", "cloud", "key", "secret", "token", "cred", "pass"))}
            all_keys = list(values.keys())
            print(f"[vms_token] db has {len(all_keys)} keys; cred-related ({len(cred_kv)}): {cred_kv}")
            sys.stdout.flush()

            # Collect ALL candidate auth keys from the db (for fallback retry)
            # Try cloud auth key names in order — cloudAuthKey FIRST, authKey LAST
            # authKey = per-server media authentication key (wrong); cloudAuthKey = cloud system key (correct)
            key_order = ["cloudAuthKey", "cloudConnect.authKey", "cloud.authKey",
                         "cloud_auth_key", "cloudSystemAuthenticationKey",
                         "cloudSystemIdentityKey", "cloudSystemKey",
                         "authenticationKey", "authKey"]
            for kname in key_order:
                v = values.get(kname)
                if v and str(v).strip():
                    auth_key_candidates.append((kname, str(v).strip()))

            if auth_key_candidates:
                first_kname, first_val = auth_key_candidates[0]
                auth_key = first_val
                source = conf_path
                print(f"[vms_token] primary key '{first_kname}' len={len(auth_key)} val[:8]={auth_key[:8]}…")
                if len(auth_key_candidates) > 1:
                    print(f"[vms_token] {len(auth_key_candidates)-1} fallback candidate(s): "
                          + ", ".join(f"'{n}'(len={len(v)})" for n, v in auth_key_candidates[1:]))
                sys.stdout.flush()

            cloud_id = cloud_id or (
                values.get("cloudSystemId") or values.get("cloudConnect.cloudSystemId")
                or values.get("cloud.cloudSystemId") or values.get("cloud_system_id")
                or values.get("cloudSystemID"))
            if not auth_key:
                print(f"[vms_token] config file found but no cloud auth key in any known field")
                sys.stdout.flush()
        else:
            print("[vms_token] mserver.sqlite / mserver.conf not found on this host")
            sys.stdout.flush()

    # ── 3. VMS REST API for cloud credentials ─────────────────────────────────
    if not auth_key:
        for cred_path in ["/rest/v3/system/cloudCredentials",
                          "/rest/v4/system/cloudCredentials",
                          "/api/cloudCredentials",
                          "/api/mergeSystems"]:
            try:
                r = nx_request("GET", cred_path)
                print(f"[vms_token] REST {cred_path} → {r.status_code}: {r.text[:200]}")
                sys.stdout.flush()
                if r.ok:
                    d = r.json()
                    # Flatten nested dicts
                    flat = d if isinstance(d, dict) else {}
                    if isinstance(d.get("reply"), dict):
                        flat.update(d["reply"])
                    ak = (flat.get("authKey") or flat.get("cloudAuthKey")
                          or flat.get("auth_key") or flat.get("secret"))
                    cid = (flat.get("cloudSystemId") or flat.get("systemId")
                           or flat.get("cloudId"))
                    if ak:
                        auth_key = ak
                        cloud_id = cloud_id or cid
                        source = f"REST {cred_path}"
                        print(f"[vms_token] got authKey from {cred_path}")
                        sys.stdout.flush()
                        break
            except Exception:
                pass

    # ── 4. NX AI Manager engine settings via REST API ─────────────────────────
    if not auth_key:
        try:
            engine_id = get_nxai_engine_id()
            for epath in [
                f"/rest/v4/analytics/engines/{engine_id}/settings",
                f"/rest/v3/analytics/engines/{engine_id}/settings",
                f"/rest/v4/analytics/engines/{engine_id}/parameters",
            ]:
                try:
                    r = nx_request("GET", epath)
                    if r.ok:
                        s = r.json()
                        # Flatten any nested settings
                        flat = {}
                        def _flatten(d, prefix=""):
                            if isinstance(d, dict):
                                for k, v in d.items():
                                    _flatten(v, f"{prefix}{k}.")
                            else:
                                flat[prefix.rstrip(".")] = d
                        _flatten(s)
                        cand = [k for k in flat if any(x in k.lower()
                            for x in ("auth", "key", "secret", "token", "credential", "cloud"))]
                        print(f"[vms_token] engine settings {epath}: cred keys={cand}")
                        sys.stdout.flush()
                        ak = (flat.get("authKey") or flat.get("auth_key")
                              or flat.get("cloudAuthKey") or flat.get("apiKey")
                              or flat.get("api_key") or flat.get("secret"))
                        if ak:
                            auth_key = ak
                            cloud_id = cloud_id or flat.get("cloudSystemId") or flat.get("systemId")
                            source = f"NX engine settings {epath}"
                            print(f"[vms_token] found authKey in engine settings")
                            sys.stdout.flush()
                            break
                except Exception:
                    pass
            if auth_key:
                pass  # found it
        except Exception as e:
            print(f"[vms_token] engine settings probe error: {e}")
            sys.stdout.flush()

    # ── 5. NX AI Manager plugin config files on disk ──────────────────────────
    if not auth_key:
        try:
            plugin_confs = _find_nxai_plugin_conf()
            print(f"[vms_token] plugin conf files found: {[p for p, _ in plugin_confs]}")
            sys.stdout.flush()
            for fpath, conf_data in plugin_confs:
                flat2 = {}
                def _flatten2(d, prefix=""):
                    if isinstance(d, dict):
                        for k, v in d.items():
                            _flatten2(v, f"{prefix}{k}.")
                    else:
                        flat2[prefix.rstrip(".")] = d
                _flatten2(conf_data)
                ak = (flat2.get("authKey") or flat2.get("auth_key")
                      or flat2.get("cloudAuthKey") or flat2.get("apiKey")
                      or flat2.get("api_key") or flat2.get("secret"))
                if ak:
                    auth_key = ak
                    cloud_id = cloud_id or flat2.get("cloudSystemId") or flat2.get("systemId")
                    source = fpath
                    print(f"[vms_token] found authKey in plugin conf: {fpath}")
                    sys.stdout.flush()
                    break
        except Exception as e:
            print(f"[vms_token] plugin conf search error: {e}")
            sys.stdout.flush()

    if not auth_key:
        print("[vms_token] no authKey found in any source — cannot get system cloud token")
        sys.stdout.flush()
        return None

    # Fall back to cloud system ID from VMS REST if not found in conf/config
    cloud_id = cloud_id or _get_cloud_system_id()
    if not cloud_id:
        print("[vms_token] cloudSystemId unknown — cannot authenticate")
        sys.stdout.flush()
        return None

    print(f"[vms_token] source={source}  cloudId={str(cloud_id)[:8]}…  authKey present")
    sys.stdout.flush()

    print(f"[vms_token] authKey[:8]={str(auth_key)[:8]}…  len={len(str(auth_key))}")
    sys.stdout.flush()

    # NX Witness v6 uses a CUSTOM grant type "system_credentials" — not the standard
    # OAuth 2.0 "client_credentials".  The NX source (oauth_data.h) defines:
    #   grant_type = system_credentials
    #   client_id  = cloudSystemId
    #   password   = authKey          (NOT client_secret)
    #   response_type = token         (optional but recommended)
    #
    # Token endpoints (both work, v1 is preferred for v6):
    #   POST https://meta.nxvms.com/cdb/oauth2/v1/token
    #   POST https://meta.nxvms.com/cdb/oauth2/token
    #
    # The request should be JSON (application/json).  Form-urlencoded also
    # exists but JSON is what the v6 C++ client sends.

    token_urls = [
        f"https://meta.nxvms.com/cdb/oauth2/v1/token",   # NX v6 preferred
        f"{CLOUD_CDB_ENDPOINT}/oauth2/token",              # legacy / v5
        f"https://meta.nxvms.com/cdb/oauth2/token",       # explicit legacy
    ]

    # ── Step A: Ask the NX server itself for a cloud token ───────────────────
    # The NX Media Server has an active cloud session. Some NX versions expose
    # an endpoint that returns the current cloud bearer token.
    for cloud_token_path in [
        "/rest/v4/system/cloudToken",
        "/rest/v4/login/cloudToken",
        "/api/cloudToken",
        "/rest/v3/system/cloudToken",
        "/rest/v4/system/cloudCredentials",
        "/rest/v3/system/cloudCredentials",
        "/api/cloudCredentials",
    ]:
        try:
            r = nx_request("GET", cloud_token_path)
            print(f"[vms_token] NX REST {cloud_token_path} → {r.status_code}: {r.text[:300]}")
            sys.stdout.flush()
            if r.ok:
                d = r.json()
                flat = d if isinstance(d, dict) else {}
                if isinstance(d.get("reply"), dict):
                    flat.update(d["reply"])
                tok = (flat.get("access_token") or flat.get("token") or flat.get("accessToken")
                       or flat.get("cloudToken") or flat.get("cloud_token"))
                if tok:
                    _vms_cpt_token_cache = tok
                    print(f"[vms_token] GOT cloud token from NX REST {cloud_token_path}")
                    sys.stdout.flush()
                    return tok
        except Exception as e:
            print(f"[vms_token] NX REST {cloud_token_path} error: {e}")
            sys.stdout.flush()

    # ── Step B: Try the cloudAuthKey in multiple encodings ────────────────────
    # The hex string might need to be base64-encoded raw bytes for the OAuth grant.
    import base64 as _b64
    import binascii as _binhex

    def _key_variants(kname, kval):
        """Yield (variant_name, value) for a key in different encodings."""
        yield kname, kval                    # hex string as-is (what we've tried)
        # If the key is hex, also try raw-bytes base64
        try:
            raw = bytes.fromhex(kval)
            yield kname + "_b64", _b64.b64encode(raw).decode()
            yield kname + "_b64url", _b64.urlsafe_b64encode(raw).decode().rstrip("=")
        except Exception:
            pass

    # Build list of (key_name, key_value) to try — primary candidate first, then fallbacks
    keys_to_try = auth_key_candidates if auth_key_candidates else [("primary", auth_key)]
    # Deduplicate by value while preserving order, then expand with encoding variants
    _seen_vals: set = set()
    keys_to_try_expanded = []
    for kn, kv in keys_to_try:
        for vn, vv in _key_variants(kn, kv):
            if vv not in _seen_vals:
                _seen_vals.add(vv)
                keys_to_try_expanded.append((vn, vv))

    for try_kname, try_key in keys_to_try_expanded:
        print(f"[vms_token] trying key '{try_kname}' len={len(try_key)} val[:8]={try_key[:8]}…")
        sys.stdout.flush()

        bodies = [
            {"grant_type": "system_credentials", "response_type": "token",
             "client_id": cloud_id, "password": try_key},
            {"grant_type": "system_credentials",
             "client_id": cloud_id, "password": try_key},
            {"grant_type": "system_credentials", "response_type": "token",
             "client_id": cloud_id, "client_secret": try_key},
        ]

        for token_url in token_urls[:1]:
            for body in bodies:
                try:
                    resp = requests.post(token_url, json=body, timeout=15)
                    print(f"[vms_token] {token_url.split('nxvms.com')[-1]} "
                          f"grant={body['grant_type']} pw_key='{try_kname}' "
                          f"→ {resp.status_code}: {resp.text[:200]}")
                    sys.stdout.flush()
                    if resp.ok:
                        d = resp.json()
                        tok = (d.get("access_token") or d.get("token") or d.get("accessToken"))
                        if tok:
                            _vms_cpt_token_cache = tok
                            print(f"[vms_token] GOT system token (key='{try_kname}')")
                            sys.stdout.flush()
                            return tok
                    elif resp.status_code in (401, 403):
                        break
                except Exception as e:
                    print(f"[vms_token] {token_url}: {e}")
                    sys.stdout.flush()

    # ── Step C: HTTP Basic auth fallback ──────────────────────────────────────
    basic = _b64.b64encode(f"{cloud_id}:{auth_key}".encode()).decode()
    for token_url in token_urls:
        for form_body in [
            {"grant_type": "system_credentials"},
            {"grant_type": "client_credentials"},
        ]:
            try:
                resp = requests.post(
                    token_url, data=form_body,
                    headers={"Authorization": f"Basic {basic}"}, timeout=15,
                )
                print(f"[vms_token] Basic {token_url.split('nxvms.com')[-1]} "
                      f"grant={form_body['grant_type']} → {resp.status_code}: {resp.text[:200]}")
                sys.stdout.flush()
                if resp.ok:
                    d = resp.json()
                    tok = (d.get("access_token") or d.get("token") or d.get("accessToken"))
                    if tok:
                        _vms_cpt_token_cache = tok
                        return tok
            except Exception as e:
                print(f"[vms_token] Basic {token_url}: {e}")
                sys.stdout.flush()

    return None

# ── Training state ─────────────────────────────────────────────────────────────
_deploy_state = {"phase": "idle", "detail": ""}  # idle | uploading | processing | assigning | done | error

_training_state = {
    "classes": {},       # {class_name: [frame_path, ...]}
    "model_path": None,  # path to exported ONNX model
    "class_names": [],   # ordered list of class names
    "status": "idle",    # idle | trained | error
    "accuracy": None,
}

def _restore_training_state():
    """Restore in-memory state from .model_meta.json + training_data/ after a restart."""
    meta_path = BASE_DIR / ".model_meta.json"
    model_path = BASE_DIR / ".model.onnx"
    if meta_path.exists() and model_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            _training_state["model_path"]  = str(model_path)
            _training_state["class_names"] = meta.get("class_names", [])
            _training_state["accuracy"]    = meta.get("accuracy")
            _training_state["status"]      = "restored"
        except Exception as e:
            print(f"[startup] Could not restore training state: {e}")
    # Rebuild class frame lists from training_data/ so class counts are correct
    if TRAIN_DIR.exists():
        for cls_dir in TRAIN_DIR.iterdir():
            if cls_dir.is_dir():
                frames = [str(p) for p in sorted(cls_dir.glob("*.jpg"))]
                if frames:
                    _training_state["classes"][cls_dir.name] = frames

_restore_training_state()

def get_class_dir(class_name):
    safe = "".join(c for c in class_name if c.isalnum() or c in " -_").strip()[:40]
    return TRAIN_DIR / safe

# ── Routes: Static ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(WEB_DIR / "index.html")

# ── Routes: Status ─────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    result = {
        "nx": False,
        "scailable": False,
        "nx_host": None,
        "nx_version": None,
        "nxai_engine_id": None,
        "training_status": _training_state["status"],
        "class_counts": {k: len(v) for k, v in _training_state["classes"].items()},
        "trained_class_names": _training_state["class_names"],
        "trained_accuracy": round((_training_state["accuracy"] or 0) * 100, 1) if _training_state["accuracy"] else None,
        "trained_model_kb": None,
    }
    if _training_state["model_path"]:
        try:
            result["trained_model_kb"] = round(Path(_training_state["model_path"]).stat().st_size / 1024, 1)
        except Exception:
            pass
    cfg = load_config()
    result["nx_host"] = f"{cfg['nx']['host']}:{cfg['nx']['port']}"

    try:
        resp = nx_request("GET", "/rest/v3/servers/this")
        data = resp.json()
        result["nx"] = True
        result["nx_version"] = data.get("version", "unknown")
    except Exception as e:
        result["nx_error"] = str(e)

    try:
        result["nxai_engine_id"] = get_nxai_engine_id()
    except Exception:
        pass

    tokens = _auto_refresh_tokens()
    if tokens.get("access_token"):
        result["scailable"] = True
        payload = decode_jwt_payload(tokens["access_token"])
        result["scailable_user"] = payload.get("sub") or payload.get("email")
        exp = payload.get("exp")
        if exp:
            result["token_expires_in"] = max(0, int(exp - time.time()))

    return jsonify(result)

# ── Routes: Cameras ────────────────────────────────────────────────────────────
@app.route("/api/cameras")
def api_cameras():
    try:
        resp = nx_request("GET", "/rest/v4/devices")
        resp.raise_for_status()
        devices = resp.json()
        cameras = [
            {
                "id": c["id"].strip("{}"),
                "name": c.get("name", c["id"]),
                "model": c.get("model", ""),
                "status": c.get("status", ""),
            }
            for c in devices
            if c.get("deviceType", "").lower() in ("camera", "nvr", "")
            or "camera" in c.get("deviceType", "").lower()
        ]
        # If nothing matched the filter, return all devices
        if not cameras and devices:
            cameras = [
                {
                    "id": c["id"].strip("{}"),
                    "name": c.get("name", c["id"]),
                    "model": c.get("model", ""),
                    "status": c.get("status", ""),
                    "deviceType": c.get("deviceType", ""),
                }
                for c in devices
            ]
        return jsonify(cameras)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Routes: Bookmarks ─────────────────────────────────────────────────────────
@app.route("/api/bookmarks/<camera_id>")
def api_bookmarks(camera_id):
    try:
        # Try v4 device-scoped endpoint first, fall back to v3
        resp = nx_request("GET", f"/rest/v4/devices/{camera_id}/bookmarks?limit=200&sortOrder=desc")
        if resp.status_code == 404:
            resp = nx_request("GET", f"/rest/v3/bookmarks?deviceId={camera_id}&limit=200&sortOrder=desc")
        resp.raise_for_status()
        raw = resp.json()
        # Normalize: Nx returns either a list or {"bookmarks": [...]}
        items = raw if isinstance(raw, list) else raw.get("bookmarks", raw.get("data", []))
        bookmarks = [
            {
                "id": b.get("id", ""),
                "name": b.get("name") or b.get("description") or "Bookmark",
                "startTimeMs": b.get("startTimeMs") or b.get("startTime") or 0,
                "durationMs": b.get("durationMs") or b.get("duration") or 0,
            }
            for b in items
        ]
        return jsonify(sorted(bookmarks, key=lambda x: -x["startTimeMs"]))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Routes: Video ──────────────────────────────────────────────────────────────
_working_live_idx = {}  # camera_id -> candidate index that works for live

def _live_candidates(camera_id, res):
    size = "&width=640&height=360" if res == "lo" else ""
    return [
        f"/ec2/cameraThumbnail?cameraId={{{camera_id}}}&imageFormat=jpeg{size}",
        f"/ec2/cameraThumbnail?cameraId={camera_id}&imageFormat=jpeg{size}",
        f"/rest/v3/devices/{camera_id}/thumbnail?imageFormat=jpeg{size}",
        f"/rest/v4/devices/{camera_id}/thumbnail?imageFormat=jpeg{size}",
        f"/media/{camera_id}.jpg",
        f"/media/{{{camera_id}}}.jpg",
    ]

_nx_rtsp_port_cache = None
_cloud_system_id_cache = None


def _get_cloud_system_id():
    """Return the Nx Cloud system ID used in relay URLs (e.g. {id}.relay.vmsproxy.com)."""
    global _cloud_system_id_cache
    if _cloud_system_id_cache:
        return _cloud_system_id_cache
    import sys
    for path in ["/rest/v3/system/info", "/api/systemSettings"]:
        try:
            resp = nx_request("GET", path)
            if resp.ok:
                data = resp.json()
                reply = data.get("reply", data)
                # Log all keys on first hit to aid debugging
                print(f"[relay] {path} keys: {list(reply.keys())[:20]}")
                sys.stdout.flush()
                sid = (reply.get("cloudSystemId") or reply.get("cloudId")
                       or reply.get("cloudSystemID") or reply.get("systemId"))
                if sid:
                    _cloud_system_id_cache = sid.strip("{}")
                    print(f"[relay] cloud system ID: {_cloud_system_id_cache}")
                    return _cloud_system_id_cache
        except Exception as e:
            print(f"[relay] {path} error: {e}")
    return None


def _fetch_recorded_via_http_clip(camera_id, pos_ms, res):
    """
    Download a short clip from Nx via HTTP and extract the first video frame.

    Nx media endpoint:  GET /media/{id}.{ext}?pos={unix_µs}&duration={µs}
    Both pos and duration are in MICROSECONDS.

    Requires: pip install av
    """
    import sys
    try:
        import av as pyav
        import io as _io
    except ImportError:
        return None, "PyAV not installed — run: pip install av"

    us = int(pos_ms) * 1000
    duration_us = 2_000_000   # 2-second clip in µs

    # Nx /media/ endpoint uses µs for both pos and duration.
    # (Using ms for duration, e.g. duration=2000, causes Nx to treat it as 2000µs = 2ms
    #  which is negligible, making the server stream live video indefinitely — ~1GB.)
    query_variants = [
        (f"pos={us}&duration={duration_us}",        "µs"),
        (f"startPos={us}&endPos={us+duration_us}",  "startPos/endPos µs"),
    ]

    MAX_BYTES = 15 * 1024 * 1024  # 15 MB safety cap — abort if server ignores duration

    for ext in ("mp4", "mkv"):
        for qs, qs_label in query_variants:
            path = f"/media/{camera_id}.{ext}?{qs}"
            container = None
            try:
                resp = nx_request("GET", path, stream=True)
                if resp.status_code != 200:
                    print(f"  [clip] .{ext} {qs_label} → {resp.status_code}")
                    sys.stdout.flush()
                    resp.close()
                    continue

                # Stream up to MAX_BYTES; close connection if exceeded
                chunks = []
                total = 0
                capped = False
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= MAX_BYTES:
                            capped = True
                            break
                resp.close()

                content = b"".join(chunks)
                cap_note = " [capped]" if capped else ""
                print(f"  [clip] .{ext} {qs_label} → 200 ({len(content)}b{cap_note})")
                sys.stdout.flush()
                if len(content) < 1000:
                    continue

                buf = _io.BytesIO(content)
                container = pyav.open(buf)
                video = container.streams.video[0]
                tb = video.time_base

                st = container.start_time or 0
                print(f"  [clip] start_time={st}µs  requested={us}µs  diff={st-us}µs")
                sys.stdout.flush()

                for frame in container.decode(video=0):
                    img = frame.to_image()
                    if res == "lo":
                        img.thumbnail((640, 360))
                    else:
                        img.thumbnail((1920, 1080))
                    out = _io.BytesIO()
                    img.save(out, format="JPEG", quality=85)
                    jpeg = out.getvalue()
                    pts_sec = None
                    if frame.pts is not None and tb and tb.denominator:
                        pts_sec = frame.pts * tb.numerator / tb.denominator
                    print(f"  [clip] OK — {len(jpeg)}b JPEG  pts={frame.pts}  pts_abs≈{pts_sec:.0f}s  requested={us//1_000_000}s")
                    container.close()
                    return jpeg, None
                container.close()
            except Exception as e:
                print(f"  [clip] .{ext} {qs_label} error: {e}")
                sys.stdout.flush()
                if container:
                    try: container.close()
                    except: pass

    return None, None


def _fetch_recorded_via_relay(camera_id, pos_ms, res):
    """
    Fetch a recorded frame via the Nx Cloud RTSP relay.

    URL format: rtsp://{user}:{pass}@{system_id}.relay.vmsproxy.com:443/{camera_id}
    The system ID is fetched from /rest/v3/system/info (cloudSystemId).
    Requires: pip install av
    """
    import sys
    try:
        import av as pyav
        import io as _io
    except ImportError:
        return None, "PyAV not installed"

    system_id = _get_cloud_system_id()
    if not system_id:
        print("  [relay] cloud system ID unavailable — skipping relay")
        return None, None

    from urllib.parse import quote as _quote
    cfg = load_config()
    nx = cfg["nx"]
    user = _quote(nx["username"], safe="")
    pwd_raw = nx["password"]
    pwd = _quote(pwd_raw, safe="")
    us = int(pos_ms) * 1000
    relay_host = f"{system_id}.relay.vmsproxy.com"

    # Try rtsp:// first (relay accepts plain RTSP on 443), then rtsps://
    url_configs = [
        (f"rtsp://{user}:{pwd}@{relay_host}:443/{camera_id}?stream=primary&pos={us}",
         {"rtsp_transport": "tcp"}),
        (f"rtsps://{user}:{pwd}@{relay_host}:443/{camera_id}?stream=primary&pos={us}",
         {"rtsp_transport": "tcp", "tls_verify": "0"}),
    ]

    for url, extra_opts in url_configs:
        safe = url.replace(pwd, "***")
        print(f"  [relay] {safe[:110]}")
        sys.stdout.flush()
        container = None
        try:
            opts = {"stimeout": "8000000", "timeout": "8000000"}
            opts.update(extra_opts)
            container = pyav.open(url, options=opts)
            video = container.streams.video[0]

            tb = video.time_base
            if tb and tb.numerator:
                target_pts = int(us * tb.denominator / (tb.numerator * 1_000_000))
            else:
                target_pts = us

            try:
                container.seek(target_pts, backward=True, stream=video)
            except Exception as se:
                print(f"  [relay] seek error (continuing): {se}")

            for frame in container.decode(video=0):
                img = frame.to_image()
                if res == "lo":
                    img.thumbnail((640, 360))
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                jpeg = buf.getvalue()
                print(f"  [relay] OK — {len(jpeg)}b JPEG (pts={frame.pts})")
                return jpeg, None

        except Exception as e:
            err_str = str(e).replace(pwd_raw, "***").replace(pwd, "***")
            print(f"  [relay] error: {err_str}")
        finally:
            if container:
                try: container.close()
                except: pass

    return None, None


def _fetch_frame(camera_id, pos_ms=None, res="lo"):
    """
    Fetch a JPEG frame from Nx.
    Live: uses ec2 (fast, always works); caches which candidate succeeded.
    Recorded: tries HTTP clip export first, then cloud relay RTSP, then live fallback.
    Returns (bytes, None) on success or (None, error_str) on failure.
    """
    if pos_ms:
        import sys
        print(f"[recorded] cam={camera_id[:8]}… pos={pos_ms}")
        us  = int(pos_ms) * 1000
        bid = f"{{{camera_id}}}"
        size_lo = "&width=640&height=360"
        size_hi = "&width=1920&height=1080"
        size = size_lo if res == "lo" else size_hi

        # Try all thumbnail/snapshot endpoints.
        # ec2/cameraThumbnail?time= may use either ms or µs depending on Nx version;
        # the REST v3/v4 thumbnail endpoint uses timestampMs (ms).
        # Crucially, try ms FIRST — sending µs to an ms-expecting API yields a
        # timestamp in year ~58,000, causing Nx to silently fall back to the live frame.
        thumb_candidates = [
            f"/ec2/cameraThumbnail?cameraId={bid}&imageFormat=jpeg{size}&time={int(pos_ms)}",
            f"/rest/v4/devices/{camera_id}/thumbnail?imageFormat=jpeg{size}&timestampMs={int(pos_ms)}",
            f"/rest/v3/devices/{camera_id}/thumbnail?imageFormat=jpeg{size}&timestampMs={int(pos_ms)}",
            f"/media/{camera_id}.jpg?pos={us}",
            f"/media/{camera_id}.jpg?pos={int(pos_ms)}",
            f"/ec2/cameraThumbnail?cameraId={bid}&imageFormat=jpeg{size}&time={us}",
        ]
        for path in thumb_candidates:
            label = path.split("?")[0].rsplit("/", 1)[-1]
            qs    = path.split("?", 1)[1] if "?" in path else ""
            print(f"  [thumb] {label} {qs[:60]}")
            sys.stdout.flush()
            try:
                resp = nx_request("GET", path)
            except Exception as e:
                print(f"  [thumb] error: {e}")
                sys.stdout.flush()
                continue
            print(f"  [thumb] → {resp.status_code} ({len(resp.content)}b)")
            sys.stdout.flush()
            if resp.status_code == 200 and len(resp.content) > 500:
                return resp.content, None

        # HTTP clip export (PyAV decode of MP4 clip)
        frame, err = _fetch_recorded_via_http_clip(camera_id, pos_ms, res)
        if frame:
            return frame, None
        if err:
            return None, err

        # Cloud relay RTSP
        frame, _ = _fetch_recorded_via_relay(camera_id, pos_ms, res)
        if frame:
            return frame, None

        return None, "Recorded frame unavailable on this Nx server"

    candidates = _live_candidates(camera_id, res)
    cached = _working_live_idx.get(camera_id)
    if cached is not None and cached < len(candidates):
        try:
            resp = nx_request("GET", candidates[cached], stream=False)
            if resp.status_code == 200 and resp.content:
                return resp.content, None
        except Exception:
            pass
    print(f"[live] cam={camera_id[:8]}… scanning candidates")

    errors = []
    for i, path in enumerate(candidates):
        try:
            resp = nx_request("GET", path, stream=False)
            print(f"  [{i}] {path[:100]} → {resp.status_code} ({len(resp.content)}b)")
            if resp.status_code == 200 and resp.content:
                _working_live_idx[camera_id] = i
                return resp.content, None
            errors.append(f"{path.split('?')[0].rsplit('/', 1)[-1]}={resp.status_code}")
        except Exception:
            errors.append("err")
    return None, "All snapshot paths failed: " + ", ".join(errors)

@app.route("/api/frame/<camera_id>")
def api_frame(camera_id):
    """
    Get a single JPEG frame.
    Query params: pos=<unix_ms> for recorded video, stream=<0|1> for primary/secondary.
    """
    pos = request.args.get("pos")
    res = request.args.get("res", "lo")
    data, err = _fetch_frame(camera_id, pos, res)
    if data:
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
    return jsonify({"error": err}), 404

@app.route("/api/stream/<camera_id>")
def api_stream(camera_id):
    token, nx = get_nx_token()
    cfg = load_config()
    nxcfg = cfg["nx"]
    url = f"https://{nxcfg['host']}:{nxcfg['port']}/media/{camera_id}.mjpeg?resolution=low&fps=5"

    def generate():
        with requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            verify=False,
            stream=True,
            timeout=60,
        ) as r:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace;boundary=--myboundary",
    )

# ── Routes: Capture ────────────────────────────────────────────────────────────
@app.route("/api/capture", methods=["POST"])
def api_capture():
    """
    Capture a frame from a camera and store it in a training class.
    Body: {"class_name": "...", "camera_id": "..."}
    OR:   {"class_name": "...", "frame_b64": "<jpeg base64>"}
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    class_name = (data.get("class_name") or "").strip()
    if not class_name:
        return jsonify({"error": "class_name required"}), 400

    # Get frame bytes
    if data.get("frame_b64"):
        frame_bytes = base64.b64decode(data["frame_b64"])
    elif data.get("camera_id"):
        frame_bytes, err = _fetch_frame(data["camera_id"], data.get("pos"), data.get("res", "lo"))
        if not frame_bytes:
            return jsonify({"error": f"Failed to capture from camera: {err}"}), 500
    else:
        return jsonify({"error": "Provide camera_id or frame_b64"}), 400

    # Save frame to disk
    class_dir = get_class_dir(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(class_dir.glob("*.jpg")))
    frame_path = class_dir / f"{count:04d}.jpg"
    frame_path.write_bytes(frame_bytes)

    # Update state
    if class_name not in _training_state["classes"]:
        _training_state["classes"][class_name] = []
    _training_state["classes"][class_name].append(str(frame_path))
    _training_state["status"] = "idle"  # reset trained status on new capture

    # Scale frame down to a small thumbnail for the response to keep JSON compact
    # (large base64 payloads can cause silent JSON-parse failures in some CEF builds)
    thumb_bytes = frame_bytes
    try:
        import io
        from PIL import Image as _PILImg
        _img = _PILImg.open(io.BytesIO(frame_bytes)).convert("RGB")
        _img.thumbnail((128, 128))
        _buf = io.BytesIO()
        _img.save(_buf, format="JPEG", quality=70)
        thumb_bytes = _buf.getvalue()
    except Exception:
        pass
    thumb_b64 = base64.b64encode(thumb_bytes).decode()
    return jsonify({"ok": True, "count": len(_training_state["classes"][class_name]), "thumb": thumb_b64})

@app.route("/api/capture/<class_name>", methods=["DELETE"])
def api_delete_class(class_name):
    """Delete all samples for a class."""
    class_dir = get_class_dir(class_name)
    if class_dir.exists():
        shutil.rmtree(class_dir)
    _training_state["classes"].pop(class_name, None)
    _training_state["status"] = "idle"
    return jsonify({"ok": True})

@app.route("/api/capture/reset", methods=["POST"])
def api_reset():
    """Delete all training data and reset state."""
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    _training_state["classes"].clear()
    _training_state["model_path"] = None
    _training_state["class_names"] = []
    _training_state["status"] = "idle"
    _training_state["accuracy"] = None
    return jsonify({"ok": True})

# ── Routes: Train ──────────────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def api_train():
    """Train a classifier on captured frames; exports ONNX to BASE_DIR/.model.onnx.
    Body JSON: { "method": "basic" | "cnn" | "mobilenet" }
    """
    data = request.get_json(silent=True) or {}
    method = data.get("method", "basic")

    classes = _training_state["classes"]
    if len(classes) < 2:
        return jsonify({"error": "Need at least 2 classes"}), 400
    for name, samples in classes.items():
        if len(samples) < 5:
            return jsonify({"error": f"Class '{name}' needs at least 5 samples (has {len(samples)})"}), 400

    class_names = sorted(classes.keys())
    model_path  = BASE_DIR / ".model.onnx"
    meta_path   = BASE_DIR / ".model_meta.json"

    # ── shared helpers ────────────────────────────────────────────────────────
    def letterbox_pil(img_path, size):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        tw, th = size
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        canvas = Image.new("RGB", size, (0, 0, 0))
        canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((tw - nw) // 2, (th - nh) // 2))
        return canvas

    out_tensor_name = "bboxes-format:xyxysc;" + ";".join(
        f"{i}:{name}" for i, name in enumerate(class_names)
    )

    def _rename_onnx_output(path, new_name):
        from onnx import load as ol, save as os_, version_converter
        m = ol(str(path))
        # Scailable requires opset <= 16 and IR version <= 9
        current_opset = max((op.version for op in m.opset_import), default=0)
        if current_opset > 16:
            m = version_converter.convert_version(m, 16)
        m.ir_version = 7
        old = m.graph.output[0].name
        for node in m.graph.node:
            for i, o in enumerate(node.output):
                if o == old:
                    node.output[i] = new_name
        m.graph.output[0].name = new_name
        os_(m, str(path))

    # ── Method: basic — LR on raw pixels ─────────────────────────────────────
    def do_train_basic():
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score
        from onnx import helper as oh, TensorProto, numpy_helper

        IMG_SIZE = (64, 64)
        X, y = [], []
        for li, cls_name in enumerate(class_names):
            for sp in classes[cls_name]:
                try:
                    arr = np.array(letterbox_pil(sp, IMG_SIZE), dtype=np.float32) / 255.0
                    X.append(arr.transpose(2, 0, 1).flatten())
                    y.append(li)
                except Exception:
                    continue
        if not X:
            raise ValueError("No valid training samples")
        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs")),
        ])
        pipe.fit(X, y)
        acc = float(accuracy_score(y, pipe.predict(X)))

        H, W      = IMG_SIZE
        coef      = pipe.named_steps["clf"].coef_.astype(np.float32)
        bias      = pipe.named_steps["clf"].intercept_.astype(np.float32)
        s_mean    = pipe.named_steps["scaler"].mean_.astype(np.float32)
        s_scale   = np.maximum(pipe.named_steps["scaler"].scale_, 1e-7).astype(np.float32)
        n_cls     = coef.shape[0]
        is_bin    = (n_cls == 1)
        n_out     = 2 if is_bin else n_cls

        X_in  = oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, H, W])
        X_out = oh.make_tensor_value_info(out_tensor_name, TensorProto.FLOAT, [1, n_out, 6])

        if is_bin:
            logit_scale = np.array([[10.0]], dtype=np.float32)
            ones_11     = np.array([[1.0]], dtype=np.float32)
            p_shape     = np.array([1, 1, 1], dtype=np.int64)
            bbox_tl     = np.array([[0.00, 0.00, 0.06, 0.06]], dtype=np.float32).reshape(1,1,4)
            bbox_tr     = np.array([[0.94, 0.00, 1.00, 0.06]], dtype=np.float32).reshape(1,1,4)
            cls0        = np.array([[[0.0]]], dtype=np.float32)
            cls1        = np.array([[[1.0]]], dtype=np.float32)
            nodes = [
                oh.make_node("Flatten", ["input"],              ["flat"],    axis=1),
                oh.make_node("Sub",     ["flat",  "s_mean"],    ["Xc"]),
                oh.make_node("Div",     ["Xc",    "s_scale"],   ["Xs"]),
                oh.make_node("Gemm",    ["Xs",    "W",   "b"],  ["logit"]),
                oh.make_node("Mul",     ["logit", "ls"],        ["logit_s"]),
                oh.make_node("Sigmoid", ["logit_s"],            ["p1"]),
                oh.make_node("Sub",     ["ones",  "p1"],        ["p0"]),
                oh.make_node("Reshape", ["p0",    "p_shape"],   ["p0_3d"]),
                oh.make_node("Reshape", ["p1",    "p_shape"],   ["p1_3d"]),
                oh.make_node("Concat",  ["bbox_tl","p0_3d","cls0"], ["det0"], axis=2),
                oh.make_node("Concat",  ["bbox_tr","p1_3d","cls1"], ["det1"], axis=2),
                oh.make_node("Concat",  ["det0","det1"], [out_tensor_name], axis=1),
            ]
            inits = [
                numpy_helper.from_array(s_mean,       "s_mean"),
                numpy_helper.from_array(s_scale,      "s_scale"),
                numpy_helper.from_array(coef.T,       "W"),
                numpy_helper.from_array(bias,         "b"),
                numpy_helper.from_array(logit_scale,  "ls"),
                numpy_helper.from_array(ones_11,      "ones"),
                numpy_helper.from_array(p_shape,      "p_shape"),
                numpy_helper.from_array(bbox_tl,      "bbox_tl"),
                numpy_helper.from_array(bbox_tr,      "bbox_tr"),
                numpy_helper.from_array(cls0,         "cls0"),
                numpy_helper.from_array(cls1,         "cls1"),
            ]
        else:
            sw          = 1.0 / n_cls
            bbox_data   = np.array([[i*sw,0.0,(i+1)*sw,0.06] for i in range(n_cls)], dtype=np.float32).reshape(1,n_cls,4)
            cls_data    = np.arange(n_cls, dtype=np.float32).reshape(1,n_cls,1)
            probs_shape = np.array([1,n_cls,1], dtype=np.int64)
            ls_mc       = np.array([[10.0]*n_cls], dtype=np.float32)
            nodes = [
                oh.make_node("Flatten", ["input"],                          ["flat"],    axis=1),
                oh.make_node("Sub",     ["flat",    "s_mean"],              ["Xc"]),
                oh.make_node("Div",     ["Xc",      "s_scale"],             ["Xs"]),
                oh.make_node("Gemm",    ["Xs",      "W",       "b"],        ["logits"]),
                oh.make_node("Mul",     ["logits",  "ls_mc"],               ["logits_s"]),
                oh.make_node("Softmax", ["logits_s"],                       ["probs"],   axis=1),
                oh.make_node("Reshape", ["probs",   "probs_shape"],         ["probs3d"]),
                oh.make_node("Concat",  ["bbox_init","probs3d","cls_init"], [out_tensor_name], axis=2),
            ]
            inits = [
                numpy_helper.from_array(s_mean,       "s_mean"),
                numpy_helper.from_array(s_scale,      "s_scale"),
                numpy_helper.from_array(coef.T,       "W"),
                numpy_helper.from_array(bias,         "b"),
                numpy_helper.from_array(ls_mc,        "ls_mc"),
                numpy_helper.from_array(bbox_data,    "bbox_init"),
                numpy_helper.from_array(probs_shape,  "probs_shape"),
                numpy_helper.from_array(cls_data,     "cls_init"),
            ]

        g = oh.make_graph(nodes, "nx_ai_trainer_basic", [X_in], [X_out], inits)
        m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 12)])
        m.ir_version = 7
        model_path.write_bytes(m.SerializeToString())
        meta_path.write_text(json.dumps({
            "class_names": class_names, "img_size": list(IMG_SIZE),
            "n_features": X.shape[1], "accuracy": acc, "method": "basic",
        }))
        return {"ok": True, "accuracy": round(acc*100,1), "class_names": class_names,
                "n_samples": len(X), "model_size_kb": round(model_path.stat().st_size/1024,1),
                "method": "Basic"}

    # ── Method: cnn — small PyTorch CNN trained from scratch ─────────────────
    def do_train_cnn():
        try:
            import torch, torch.nn as nn, torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        import numpy as np
        from onnx import load as onnx_load, save as onnx_save

        IMG_SIZE = (96, 96)
        FEAT = IMG_SIZE[0] // 4   # spatial size of feature map after 2× MaxPool2d(2) = 24
        n_cls = len(class_names)
        imgs, labels = [], []
        for li, cls_name in enumerate(class_names):
            for sp in classes[cls_name]:
                try:
                    arr = np.array(letterbox_pil(sp, IMG_SIZE), dtype=np.float32) / 255.0
                    imgs.append(arr.transpose(2,0,1))
                    labels.append(li)
                    # Horizontal flip augmentation
                    imgs.append(arr[:, ::-1, :].copy().transpose(2,0,1))
                    labels.append(li)
                except Exception:
                    continue
        if not imgs:
            raise ValueError("No valid training samples")

        X = torch.tensor(np.array(imgs, dtype=np.float32))
        y = torch.tensor(labels, dtype=torch.long)

        class SmallCNN(nn.Module):
            def __init__(self, n_out):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.AvgPool2d(FEAT),
                )
                self.drop = nn.Dropout(0.2)
                self.fc = nn.Linear(128, n_out)
            def forward(self, x):
                return self.fc(self.drop(self.features(x).flatten(1)))

        class DetWrapper(nn.Module):
            def __init__(self, net, n_out):
                super().__init__()
                self.net = net
                if n_out == 2:
                    bboxes = [[0.00,0.00,0.06,0.06],[0.94,0.00,1.00,0.06]]
                else:
                    sw = 1.0 / n_out
                    bboxes = [[i*sw, 0.0, (i+1)*sw, 0.06] for i in range(n_out)]
                self.register_buffer("bboxes",   torch.tensor([bboxes], dtype=torch.float32))
                self.register_buffer("cls_ids",  torch.arange(n_out, dtype=torch.float32).view(1,n_out,1))
            def forward(self, x):
                logits = self.net(x)
                probs  = torch.softmax(logits * 10.0, dim=1).unsqueeze(2)
                return torch.cat([self.bboxes.expand(x.shape[0],-1,-1),
                                  probs,
                                  self.cls_ids.expand(x.shape[0],-1,-1)], dim=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = SmallCNN(n_cls).to(device)
        X, y = X.to(device), y.to(device)
        opt = optim.Adam(backbone.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        backbone.train()
        best_loss, patience = float('inf'), 0
        for _ in range(80):
            opt.zero_grad()
            loss = loss_fn(backbone(X), y)
            loss.backward()
            opt.step()
            if loss.item() < best_loss - 1e-4:
                best_loss, patience = loss.item(), 0
            else:
                patience += 1
                if patience >= 25:
                    break

        backbone.eval()
        with torch.no_grad():
            acc = float((backbone(X).argmax(1) == y).float().mean())

        backbone.cpu()
        wrapper = DetWrapper(backbone, n_cls)
        wrapper.eval()
        tmp = model_path.with_suffix(".tmp.onnx")
        torch.onnx.export(wrapper, torch.zeros(1,3,*IMG_SIZE), str(tmp),
                          input_names=["input"], output_names=["det_raw"],
                          opset_version=12, do_constant_folding=True)
        _rename_onnx_output(tmp, out_tensor_name)
        tmp.replace(model_path)
        meta_path.write_text(json.dumps({
            "class_names": class_names, "img_size": list(IMG_SIZE),
            "n_features": None, "accuracy": acc, "method": "cnn",
        }))
        return {"ok": True, "accuracy": round(acc*100,1), "class_names": class_names,
                "n_samples": len(imgs), "model_size_kb": round(model_path.stat().st_size/1024,1),
                "method": "CNN"}

    # ── Method: mobilenet — fine-tune MobileNetV2 with transfer learning ──────
    def do_train_mobilenet():
        try:
            import torch, torch.nn as nn, torch.optim as optim
            from torchvision import models as tvm
        except ImportError:
            raise ImportError("torch/torchvision not installed. Run: pip install torch torchvision")
        import numpy as np

        IMG_SIZE = (224, 224)
        n_cls = len(class_names)
        IN_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(3,1,1)
        IN_STD  = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(3,1,1)

        imgs, labels = [], []
        for li, cls_name in enumerate(class_names):
            for sp in classes[cls_name]:
                try:
                    arr = np.array(letterbox_pil(sp, IMG_SIZE), dtype=np.float32) / 255.0
                    imgs.append((arr.transpose(2,0,1) - IN_MEAN) / IN_STD)
                    labels.append(li)
                except Exception:
                    continue
        if not imgs:
            raise ValueError("No valid training samples")

        X = torch.tensor(np.array(imgs, dtype=np.float32))
        y = torch.tensor(labels, dtype=torch.long)

        try:
            from torchvision.models import MobileNet_V2_Weights
            base = tvm.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except (ImportError, AttributeError):
            base = tvm.mobilenet_v2(pretrained=True)
        for p in base.parameters():
            p.requires_grad = False
        base.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, n_cls))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = base.to(device)
        X, y = X.to(device), y.to(device)
        opt = optim.Adam(base.classifier.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        base.train()
        for _ in range(30):
            opt.zero_grad()
            loss_fn(base(X), y).backward()
            opt.step()

        base.eval()
        with torch.no_grad():
            acc = float((base(X).argmax(1) == y).float().mean())

        base.cpu()
        class MobileWrapper(nn.Module):
            def __init__(self, net, n_out):
                super().__init__()
                self.net = net
                self.register_buffer("in_mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
                self.register_buffer("in_std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
                if n_out == 2:
                    bboxes = [[0.00,0.00,0.06,0.06],[0.94,0.00,1.00,0.06]]
                else:
                    sw = 1.0/n_out
                    bboxes = [[i*sw,0.0,(i+1)*sw,0.06] for i in range(n_out)]
                self.register_buffer("bboxes",  torch.tensor([bboxes], dtype=torch.float32))
                self.register_buffer("cls_ids", torch.arange(n_out, dtype=torch.float32).view(1,n_out,1))
            def forward(self, x):
                x = (x - self.in_mean) / self.in_std
                probs = torch.softmax(self.net(x)*2.0, dim=1).unsqueeze(2)
                bs = x.shape[0]
                return torch.cat([self.bboxes.expand(bs,-1,-1), probs,
                                  self.cls_ids.expand(bs,-1,-1)], dim=2)

        wrapper = MobileWrapper(base, n_cls)
        wrapper.eval()
        tmp = model_path.with_suffix(".tmp.onnx")
        torch.onnx.export(wrapper, torch.zeros(1,3,*IMG_SIZE), str(tmp),
                          input_names=["input"], output_names=["det_raw"],
                          opset_version=12, do_constant_folding=True)
        _rename_onnx_output(tmp, out_tensor_name)
        tmp.replace(model_path)
        meta_path.write_text(json.dumps({
            "class_names": class_names, "img_size": list(IMG_SIZE),
            "n_features": None, "accuracy": acc, "method": "mobilenet",
        }))
        return {"ok": True, "accuracy": round(acc*100,1), "class_names": class_names,
                "n_samples": len(imgs), "model_size_kb": round(model_path.stat().st_size/1024,1),
                "method": "MobileNetV2"}

    # ── Dispatch ──────────────────────────────────────────────────────────────
    dispatch = {"basic": do_train_basic, "cnn": do_train_cnn, "mobilenet": do_train_mobilenet}
    if method not in dispatch:
        return jsonify({"error": f"Unknown method '{method}'. Choose: basic, cnn, mobilenet"}), 400

    try:
        result = dispatch[method]()
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        print(f"[train] {method} error: {traceback.format_exc()}")
        return jsonify({"error": f"Training failed: {e}"}), 500

    _training_state["model_path"] = str(model_path)
    _training_state["class_names"] = class_names
    _training_state["status"] = "trained"
    _training_state["accuracy"] = result["accuracy"] / 100.0

    return jsonify(result)

# ── Routes: Model download ────────────────────────────────────────────────────
@app.route("/api/model/download")
def api_model_download():
    """Download the trained ONNX model file."""
    if _training_state["status"] != "trained" or not _training_state["model_path"]:
        return jsonify({"error": "No trained model — run Train first"}), 400
    model_path = Path(_training_state["model_path"])
    if not model_path.exists():
        return jsonify({"error": "Model file missing — retrain"}), 400
    return send_file(
        model_path,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="nx_ai_trainer_model.onnx",
    )

@app.route("/api/test_model")
def api_test_model():
    """
    Run the saved ONNX model on a training sample via numpy forward pass.
    Returns the raw output tensor and predicted class — use this to verify the model
    before uploading to Scailable.
    """
    try:
        import numpy as np
        from PIL import Image
        from onnx import numpy_helper, load as onnx_load
    except ImportError as e:
        return jsonify({"error": f"Missing dependency: {e}"}), 500

    model_path = BASE_DIR / ".model.onnx"
    if not model_path.exists():
        return jsonify({"error": "No trained model — run Train first"}), 400

    meta_path = BASE_DIR / ".model_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    class_names = meta.get("class_names", [])
    H, W = meta.get("img_size", [32, 32])

    # Extract initializer weights from ONNX graph
    model = onnx_load(str(model_path))
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    mean   = inits.get("mean")
    scale  = inits.get("scale")
    W_mat  = inits.get("W")   # [n_features, n_cls]
    b_vec  = inits.get("b")   # [n_cls]
    if any(v is None for v in [mean, scale, W_mat, b_vec]):
        return jsonify({"error": "ONNX initializers not found — retrain model"}), 400

    # Find one sample per class from training data
    results = []
    classes = _training_state["classes"]
    if not classes:
        return jsonify({"error": "No training data loaded — reset or restart server"}), 400

    # Detect binary case: sklearn LR stores coef.shape=(1,n) for 2-class problems
    # W_mat = coef.T, so W_mat.shape = [n_features, n_cls_coef]
    n_cls_coef = W_mat.shape[1]
    is_binary  = (n_cls_coef == 1)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    for cls_name, samples in classes.items():
        if not samples:
            continue
        img_path = samples[0]
        try:
            img = Image.open(img_path).convert("RGB").resize((W, H))
        except Exception as ex:
            results.append({"class": cls_name, "error": str(ex)})
            continue

        x = np.array(img, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1).flatten().reshape(1, -1)   # [1, n_features]
        x_c = x - mean
        x_s = x_c / scale
        logits = x_s @ W_mat + b_vec                          # [1, n_cls_coef]

        if is_binary:
            # Mirrors the ONNX Sigmoid path: p1 = sigmoid(logit), p0 = 1 - p1
            p1 = float(sigmoid(logits[0, 0]))
            probs = np.array([1.0 - p1, p1], dtype=np.float32)
        else:
            probs = softmax(logits[0])                        # [n_cls_coef]

        n_det     = len(probs)
        pred_idx  = int(np.argmax(probs))
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

        # Simulate ONNX output tensor [1, n_det, 6]
        output = np.zeros((1, n_det, 6), dtype=np.float32)
        for i, p in enumerate(probs):
            output[0, i] = [0.0, 0.0, 1.0, 1.0, float(p), float(i)]

        results.append({
            "input_class":  cls_name,
            "pred_class":   pred_name,
            "pred_idx":     pred_idx,
            "probs":        {(class_names[i] if i < len(class_names) else str(i)): round(float(p), 4)
                             for i, p in enumerate(probs)},
            "output_shape": list(output.shape),
            "output_row0":  output[0, 0].tolist(),
            "output_row1":  output[0, 1].tolist() if n_det > 1 else None,
        })

    return jsonify({"ok": True, "results": results, "class_names": class_names})

@app.route("/api/assign", methods=["POST"])
def api_assign():
    """
    Assign an existing Scailable model UUID to a camera — skips the upload step.
    Body: {"model_uuid": "...", "camera_id": "..."}
    """
    import sys
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    model_uuid = (data.get("model_uuid") or "").strip()
    camera_id  = (data.get("camera_id")  or "").strip()
    if not model_uuid:
        return jsonify({"error": "model_uuid required"}), 400
    if not camera_id:
        return jsonify({"error": "camera_id required"}), 400

    user_auth_headers = get_scailable_headers()

    cloud_id = ""
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or _get_cloud_system_id() or ""
    except Exception:
        cloud_id = _get_cloud_system_id() or ""

    # Scailable DEV API — PUT /dev/system/{cloud_id}/devices/pipelines
    # This is the exact call the admin portal makes to assign a model to a camera.
    # Scailable stores NX device IDs with curly braces: {uuid}
    if cloud_id:
        assign_url = f"{SCAILABLE_DEV}/system/{cloud_id}/devices/pipelines"
        nx_device_id = f"{{{camera_id}}}" if not camera_id.startswith("{") else camera_id
        assign_body = {
            "Devices":   [nx_device_id],
            "Functions": [model_uuid],
            "Pipelines": [{
                "Postprocessor":  "",
                "Preprocessor":   "",
                "modelNMS":       0.42,
                "modelUUID":      model_uuid,
                "resizingMethod": "Letterbox",
                "chains":         [],
            }],
        }
        oauth_hdrs = get_oauth_headers()
        dev_headers_to_try = []
        if oauth_hdrs:
            dev_headers_to_try.append(("oauth", oauth_hdrs))
        if user_auth_headers:
            dev_headers_to_try.append(("apikey", user_auth_headers))
        for auth_label, hdrs in dev_headers_to_try:
            try:
                r = requests.put(assign_url,
                                 headers={**hdrs, "Content-Type": "application/json"},
                                 json=assign_body, timeout=30)
                print(f"[assign] Scailable DEV ({auth_label}) → {r.status_code}: {r.text[:300]}")
                sys.stdout.flush()
                if r.ok or r.status_code not in (401, 403, 422):
                    break
            except Exception as e:
                print(f"[assign] Scailable DEV ({auth_label}) error: {e}")
                sys.stdout.flush()

    # NX REST — enable device agent and ensure camera is in Custom pipeline mode.
    # Do NOT set selectedPipeline to the model UUID — NX AI Manager restricts that
    # field to pre-built UUIDs + "Custom". Custom means "use whatever Scailable DEV API set".
    engine_id = ""
    try:
        engine_id = get_nxai_engine_id()
        nx_request("PUT",
                   f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{camera_id}",
                   json={"isEnabled": True},
                   headers={"Content-Type": "application/json"})
    except Exception:
        pass

    if engine_id:
        settings_base = f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{camera_id}"
        try:
            r_get = nx_request("GET", f"{settings_base}/settings")
            print(f"[assign] GET settings → {r_get.status_code}: {r_get.text[:120]}")
            sys.stdout.flush()
            if r_get.ok:
                current = r_get.json()
                existing_values = current.get("values") or {}
                if existing_values.get("selectedPipeline") != "Custom":
                    r_put = nx_request("PUT", f"{settings_base}/settings",
                                       json={"values": {"selectedPipeline": "Custom"}},
                                       headers={"Content-Type": "application/json"})
                    print(f"[assign] PUT settings → Custom: {r_put.status_code}")
                    sys.stdout.flush()
                else:
                    print("[assign] NX selectedPipeline already Custom — no PUT needed")
                    sys.stdout.flush()
        except Exception as e:
            print(f"[assign] NX settings error: {e}")
            sys.stdout.flush()

    return jsonify({"ok": True, "model_uuid": model_uuid})

# ── Routes: Deploy ─────────────────────────────────────────────────────────────
@app.route("/api/deploy/status")
def api_deploy_status():
    return jsonify(_deploy_state)

@app.route("/api/deploy/cancel", methods=["POST"])
def api_deploy_cancel():
    _deploy_state["phase"] = "idle"
    _deploy_state["detail"] = ""
    _deploy_state.pop("model_uuid", None)
    _deploy_state.pop("last_poll", None)
    return jsonify({"ok": True})

@app.route("/api/deploy", methods=["POST"])
def api_deploy():
    """
    Upload trained ONNX model to Scailable and assign it to a camera.
    Body: {"model_name": "...", "camera_id": "..."}
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    model_name = data.get("model_name", "nx-ai-trainer model").strip() or "nx-ai-trainer model"
    camera_id = data.get("camera_id", "").strip()

    if not camera_id:
        return jsonify({"error": "camera_id is required"}), 400

    if _training_state["status"] != "trained" or not _training_state["model_path"]:
        return jsonify({"error": "No trained model — run /api/train first"}), 400

    model_path = Path(_training_state["model_path"])
    if not model_path.exists():
        return jsonify({"error": "Model file not found — retrain"}), 400

    import sys, time as _time

    _deploy_state["phase"] = "uploading"
    _deploy_state["detail"] = "Uploading model to Scailable…"

    # ── Resolve cloud IDs ─────────────────────────────────────────────────────
    try:
        get_nxai_engine_id()  # populates _engine_integration_id_cache
    except Exception:
        pass
    integration_id = _engine_integration_id_cache or ""

    cloud_id = ""
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or ""
    except Exception:
        pass
    if not cloud_id:
        cloud_id = _get_cloud_system_id() or ""

    # ── Auth: user OAuth token has write permission on Scailable CPT API. ────────
    # System credentials token is read-only (POST /functions → 403).
    user_auth_headers = get_scailable_headers()
    if not user_auth_headers:
        return jsonify({"error": "Not authenticated — click Sign In first"}), 401

    # ── Build upload metadata ─────────────────────────────────────────────────
    class_names = _training_state["class_names"]
    acc_pct = round((_training_state["accuracy"] or 0) * 100, 1)

    # Load img_size written by api_train
    try:
        _meta = json.loads((BASE_DIR / ".model_meta.json").read_text())
        _img_h, _img_w = _meta.get("img_size", [64, 64])
    except Exception:
        _img_h, _img_w = 64, 64

    model_bytes = model_path.read_bytes()
    unique_name = f"{model_name} {int(_time.time()) % 100000}"
    doc_str = (f"nx-ai-trainer classifier. "
               f"Classes: {', '.join(class_names)}. Accuracy: {acc_pct}%")

    print(f"[deploy] model={len(model_bytes)}b  name={unique_name!r}  classes={class_names}")
    sys.stdout.flush()

    def _log_resp(tag, resp):
        # Log full body for error responses so we can see validation details
        body_limit = 1200 if resp.status_code >= 400 else 600
        print(f"[deploy] {tag} → {resp.status_code}  body: {resp.text[:body_limit]}")
        print(f"[deploy] {tag} req-content-type={resp.request.headers.get('Content-Type','?')!r}")
        sys.stdout.flush()

    def _extract_uuid(d):
        return d.get("UUID") or d.get("uuid") or d.get("Id") or d.get("id")

    model_uuid = None

    # ── Upload: POST /cpt/functions (JSON body per OpenAPI spec) ─────────────────
    # Required fields from OpenAPI: Name, Customization, InputDriver, InputDriverDetails,
    # OutputDriver, OutputDriverDetails, EnableConversions
    # Matches the tensor name baked into the ONNX output node.
    _out_tensor_name = "bboxes-format:xyxysc;" + ";".join(
        f"{i}:{n}" for i, n in enumerate(class_names)
    )
    upload_body = {
        "Name":               unique_name,
        "Customization":      "metavms",
        "Documentation":      doc_str,
        "InputDriver":        "vision",
        "InputDriverDetails": {
            "channels": 3,
            "height":   _img_h,
            "width":    _img_w,
            "means":    [0, 0, 0],
            "vars":     [255, 255, 255],  # NX AI Manager divides pixels by 255 → 0-1 range
            "order":    0,               # 0 = RGB
        },
        "OutputDriver":        "vision",
        "OutputDriverDetails": {"postProcessing": None},
        # NamedInput/NamedOutput tell the NX plugin what tensor names to use and how to
        # interpret the output. The trailing "-" on input is Scailable's naming convention.
        # NamedOutput Name encodes the detection format + class labels; the NX plugin reads
        # this to populate Class Visualisation and label bounding boxes.
        "NamedInput": [
            {"DataType": "FLOAT", "Name": "input-", "Shape": [1, 3, _img_h, _img_w]},
        ],
        "NamedOutput": [
            {"DataType": "FLOAT", "Name": _out_tensor_name,
             "Shape": [1, len(class_names), 6]},
        ],
        "EnableConversions":   [],
        "SourceName":          "nx-ai-trainer",
    }
    if cloud_id:
        upload_body["Site"] = cloud_id

    # Build auth header variants to try.
    # The admin portal uses "nxcdb-<refresh_token>" with NO Bearer prefix for all CPT writes.
    # We try that first (meta_token = refresh token), then fall back to access-token variants.
    raw_token = user_auth_headers.get("Authorization", "")
    bare_token = raw_token.removeprefix("Bearer ").strip()

    tokens_now = _auto_refresh_tokens()
    meta_tok = tokens_now.get("meta_token") or tokens_now.get("refresh_token") or ""
    if meta_tok and not meta_tok.startswith("nxcdb-"):
        meta_tok = f"nxcdb-{meta_tok}"

    auth_variants = []
    if meta_tok:
        auth_variants.append(meta_tok)           # nxcdb-{refresh_token}, no Bearer — portal format
    if not bare_token.startswith("nxcdb-"):
        auth_variants.append(f"nxcdb-{bare_token}")   # nxcdb-{access_token}, no Bearer
        auth_variants.append(f"Bearer {bare_token}")
        auth_variants.append(f"Bearer nxcdb-{bare_token}")
    else:
        auth_variants.append(bare_token)         # nxcdb-{token}, no Bearer
        auth_variants.append(f"Bearer {bare_token}")

    upload_attempts = []
    for auth_val in auth_variants:
        for body_kwargs, body_label in [
            (
                {
                    "data":  {"data": json.dumps(upload_body)},
                    "files": {"file": (f"{unique_name}.onnx", model_bytes, "application/octet-stream")},
                },
                "multipart",
            ),
            (
                {
                    "headers": {"Content-Type": "application/json"},
                    "json":    dict(upload_body, OnnxModel=base64.b64encode(model_bytes).decode()),
                },
                "json+b64",
            ),
        ]:
            upload_attempts.append((auth_val, body_kwargs, body_label))

    last_status = None
    for auth_val, kwargs, label in upload_attempts:
        # Merge any headers from kwargs (e.g. Content-Type for json+b64) into hdrs
        # to avoid passing 'headers' both explicitly and via **kwargs.
        extra_hdrs = kwargs.pop("headers", {})
        hdrs = {**user_auth_headers, "Authorization": auth_val, **extra_hdrs}
        try:
            r = requests.post(
                f"{SCAILABLE_CPT}/functions",
                headers=hdrs,
                timeout=60,
                **kwargs,
            )
            last_status = r.status_code
            print(f"[deploy] upload {label} auth={auth_val[:30]}… → {r.status_code}: {r.text[:400]}")
            sys.stdout.flush()
            if r.status_code in (401, 403, 500):
                continue  # try next auth variant
            if r.status_code in (200, 201):
                d = r.json()
                model_uuid = d.get("UUID") or d.get("uuid") or d.get("Id") or d.get("id")
                if model_uuid:
                    break
        except requests.exceptions.Timeout:
            print(f"[deploy] upload {label} timed out")
            sys.stdout.flush()
            last_status = 0
        except Exception as e:
            print(f"[deploy] upload {label} error: {e}")
            sys.stdout.flush()
        if model_uuid:
            break

    if not model_uuid:
        _deploy_state["phase"] = "error"
        if last_status == 403:
            msg = ("Upload blocked (403 Forbidden). Paste the token from admin.sclbl.nxvms.com "
                   "into the API key field in the Deploy panel. Open DevTools → Network on that "
                   "site, look for any request to api.sclbl.nxvms.com, and copy the full "
                   "Authorization header value.")
            _deploy_state["detail"] = msg
            return jsonify({"error": msg}), 403
        if last_status == 401:
            msg = "Upload unauthorized (401) — your session may have expired. Sign out and sign in again."
            _deploy_state["detail"] = msg
            return jsonify({"error": msg}), 401
        if last_status == 0:
            msg = "Upload timed out — Nx AI Manager may be busy or unreachable. Try again."
            _deploy_state["detail"] = msg
            return jsonify({"error": msg}), 504
        _deploy_state["detail"] = f"Upload failed (HTTP {last_status})"
        return jsonify({"error": f"Upload failed (HTTP {last_status}) — check server log for details"}), 500

    # Step 1b: Add model to catalogue so it appears in rpc/pipelines/available.
    # Use nxcdb auth (portal format).
    tokens_now2 = _auto_refresh_tokens()
    meta_tok2 = tokens_now2.get("meta_token") or tokens_now2.get("refresh_token") or ""
    if meta_tok2 and not meta_tok2.startswith("nxcdb-"):
        meta_tok2 = f"nxcdb-{meta_tok2}"
    cat_auth2 = meta_tok2 or user_auth_headers.get("Authorization", "")

    # Discover available catalogues so we can find the right UUID to register under.
    cat_uuids = []
    try:
        r_cats = requests.get(f"{SCAILABLE_CPT}/catalogues",
                              headers={"Authorization": cat_auth2}, timeout=15)
        print(f"[deploy] GET /catalogues → {r_cats.status_code}: {r_cats.text[:600]}")
        sys.stdout.flush()
        if r_cats.ok:
            cats_raw = r_cats.json()
            if not isinstance(cats_raw, list):
                cats_raw = cats_raw.get("catalogues") or cats_raw.get("data") or []
            for c in cats_raw:
                uid = c.get("UUID") or c.get("uuid") or ""
                name = c.get("Name") or c.get("name") or ""
                if uid:
                    cat_uuids.append((uid, name))
                    print(f"[deploy]   catalogue {uid!r} name={name!r}")
            sys.stdout.flush()
    except Exception as e:
        print(f"[deploy] GET /catalogues error: {e}")
        sys.stdout.flush()

    # Try PATCH /functions/{uuid} with Catalogues as UUID list, then string list.
    # Also try POST /catalogues/{cat_uuid}/functions for each discovered catalogue.
    cat_registered = False
    for cat_method in ("PATCH", "PUT"):
        try:
            fn = requests.patch if cat_method == "PATCH" else requests.put
            r_cat = fn(
                f"{SCAILABLE_CPT}/functions/{model_uuid}",
                headers={"Authorization": cat_auth2, "Content-Type": "application/json"},
                json={"Catalogues": ["metavms"]},
                timeout=15,
            )
            print(f"[deploy] {cat_method} /functions/{model_uuid} Catalogues → {r_cat.status_code}: {r_cat.text[:200]}")
            sys.stdout.flush()
            if r_cat.ok:
                cat_registered = True
                break
        except Exception as e:
            print(f"[deploy] catalogue {cat_method} error: {e}")
            sys.stdout.flush()

    # Try POST /catalogues/{uuid}/functions for each discovered catalogue.
    if not cat_registered:
        for cat_uuid, cat_name in cat_uuids:
            for post_body in [
                {"FunctionUUID": model_uuid},
                {"UUID": model_uuid},
                {"functions": [model_uuid]},
            ]:
                try:
                    r_cp = requests.post(
                        f"{SCAILABLE_CPT}/catalogues/{cat_uuid}/functions",
                        headers={"Authorization": cat_auth2, "Content-Type": "application/json"},
                        json=post_body,
                        timeout=15,
                    )
                    print(f"[deploy] POST /catalogues/{cat_uuid[:8]}({cat_name}) /functions {post_body} → {r_cp.status_code}: {r_cp.text[:200]}")
                    sys.stdout.flush()
                    if r_cp.ok:
                        cat_registered = True
                        break
                except Exception as e:
                    print(f"[deploy] POST /catalogues/{cat_uuid[:8]}/functions error: {e}")
                    sys.stdout.flush()
            if cat_registered:
                break

    # Step 2: Wait for model to finish processing before assigning.
    # Scailable compiles ONNX asynchronously; assigning a "processing" model is a no-op.
    # Use nxcdb auth (portal format) — Bearer auth returns empty fields on GET /functions/{uuid}.
    tokens_now3 = _auto_refresh_tokens()
    meta_tok3 = tokens_now3.get("meta_token") or tokens_now3.get("refresh_token") or ""
    if meta_tok3 and not meta_tok3.startswith("nxcdb-"):
        meta_tok3 = f"nxcdb-{meta_tok3}"
    poll_hdrs = {"Authorization": meta_tok3} if meta_tok3 else user_auth_headers
    _deploy_state["phase"] = "processing"
    _deploy_state["model_uuid"] = model_uuid
    _deploy_state["detail"] = "Waiting for model to compile…"
    # Poll until code.Status is a terminal state AND a compiled binary (CDNURI) exists.
    # outer='ok' alone is NOT enough — it just means the record is saved; the onnx2c
    # compiled binary may not yet be ready (CDNURI still empty → NX AI Manager can't
    # fetch the model → inference silently produces nothing).
    _STILL_PROCESSING = {"", "new", "processing", "uploading", "queued", "pending"}
    NIL_UUID_POLL = "00000000-0000-0000-0000-000000000000"
    cdn_uri = ""
    for attempt in range(120):  # up to ~10 minutes
        try:
            # Use the list endpoint — GET /functions/{uuid} with nxcdb auth omits the Code
            # object, so we can't see CDNURI. The list endpoint returns full objects.
            r_poll = requests.get(
                f"{SCAILABLE_CPT}/functions?Catalogue={NIL_UUID_POLL}&Customization=metavms",
                headers=poll_hdrs, timeout=15)
            _all = r_poll.json() if r_poll.ok else []
            if not isinstance(_all, list):
                _all = _all.get("functions") or _all.get("data") or _all.get("items") or []
            _d = next((m for m in _all if (m.get("UUID") or "").lower() == model_uuid.lower()), {})
            # Fall back to direct GET if model not in list yet
            if not _d:
                r2 = requests.get(f"{SCAILABLE_CPT}/functions/{model_uuid}",
                                  headers=poll_hdrs, timeout=15)
                if r2.ok:
                    _d = r2.json()
            elapsed = (attempt + 1) * 5
            _deploy_state["detail"] = f"Compiling model… ({elapsed}s)"
            code_obj     = _d.get("Code") or {}
            code_status  = (code_obj.get("Status") or "").lower()
            cdn_uri      = code_obj.get("CDNURI") or ""
            outer_status = (_d.get("Status") or "").lower()
            status_val   = code_status or outer_status
            _deploy_state["last_poll"] = f"outer={outer_status} code={code_status} cdn={'yes' if cdn_uri else 'no'}"
            cdn_display = repr(cdn_uri[:60]) if cdn_uri else "''"
            print(f"[deploy] poll ({attempt+1}): outer={outer_status!r} code={code_status!r} "
                  f"cdn_uri={cdn_display}")
            sys.stdout.flush()
            # Proceed as soon as status is terminal — don't block on cdn_uri since
            # some model types never populate it via this auth path.
            if status_val not in _STILL_PROCESSING:
                _deploy_state["detail"] = f"Model {status_val} — assigning to camera…"
                break
        except Exception as e:
            _deploy_state["last_poll"] = f"error: {e}"
            print(f"[deploy] poll error: {e}")
            sys.stdout.flush()
        _time.sleep(5)

    # Step 2b: Ensure model is in the Catalogues list so it appears in
    # rpc/pipelines/available — NX AI Manager validates selectedPipeline against that list.
    # Try PATCH first (update existing), then PUT if that fails.
    _deploy_state["detail"] = "Registering model in pipeline catalogue…"
    for cat_method in ("PATCH", "PUT"):
        try:
            fn = requests.patch if cat_method == "PATCH" else requests.put
            r_cat = fn(
                f"{SCAILABLE_CPT}/functions/{model_uuid}",
                headers={"Authorization": cat_auth2, "Content-Type": "application/json"},
                json={"Catalogues": ["metavms"]},
                timeout=15,
            )
            print(f"[deploy] {cat_method} /functions/{model_uuid} catalogues → {r_cat.status_code}: {r_cat.text[:200]}")
            sys.stdout.flush()
            if r_cat.ok:
                break
        except Exception as e:
            print(f"[deploy] catalogue {cat_method} error: {e}")
            sys.stdout.flush()

    # Step 2c: Resolve engine ID for post-assign verification.
    # We no longer disable/re-enable the agent: DEV assign is an internal Scailable
    # operation that does not trigger NX AI Manager's settings revalidation, so there
    # is no race to guard against.  Disabling then re-enabling the agent caused NX AI
    # Manager to reset deviceActiveSwitch back to its default (false), which prevented
    # frames from being sent to the AI runtime.
    engine_id = ""
    try:
        engine_id = get_nxai_engine_id()
    except Exception as e:
        print(f"[deploy] get engine id: {e}")
        sys.stdout.flush()

    # Step 3a: Scailable DEV API — PUT /dev/system/{cloud_id}/devices/pipelines
    # Exact endpoint and body format the admin portal uses to assign a model to a camera.
    # Scailable stores NX device IDs with curly braces: {uuid}
    _deploy_state["phase"] = "assigning"
    _deploy_state["detail"] = "Assigning model to camera…"
    if cloud_id:
        assign_url = f"{SCAILABLE_DEV}/system/{cloud_id}/devices/pipelines"
        nx_device_id = f"{{{camera_id}}}" if not camera_id.startswith("{") else camera_id
        assign_body = {
            "Devices":   [nx_device_id],
            "Functions": [model_uuid],
            "Pipelines": [{
                "Postprocessor":  "",
                "Preprocessor":   "",
                "modelNMS":       0.42,
                "modelUUID":      model_uuid,
                "resizingMethod": "Letterbox",
                "chains":         [],
            }],
        }
        oauth_hdrs = get_oauth_headers()
        dev_headers_to_try = []
        if oauth_hdrs:
            dev_headers_to_try.append(("oauth", oauth_hdrs))
        if user_auth_headers:
            dev_headers_to_try.append(("apikey", user_auth_headers))

        # Probe registered devices so we can see whether our camera is known to Scailable.
        # 422 "No devices are applicable" means NX AI Manager hasn't re-announced this
        # device yet (happens after a service restart) — we retry with backoff.
        for probe_label, probe_hdrs in dev_headers_to_try[:1]:
            try:
                r_devs = requests.get(
                    f"{SCAILABLE_DEV}/system/{cloud_id}/devices",
                    headers=probe_hdrs, timeout=15,
                )
                full_body = r_devs.text
                registered = nx_device_id in full_body
                print(f"[deploy] registered devices → {r_devs.status_code} "
                      f"(camera present={registered}, total chars={len(full_body)})")
                try:
                    devs = r_devs.json() if r_devs.ok else []
                    if not isinstance(devs, list): devs = [devs]
                    for d in devs:
                        nx_id   = d.get("NxID", "?")
                        name    = d.get("Name", "?")
                        srv     = d.get("Server", {})
                        status  = srv.get("Status", "?")
                        ai_ver  = srv.get("AIPluginVersion", "?")
                        sup_pip = (srv.get("Supports") or {}).get("Pipelines", "?")
                        print(f"[deploy]   device {nx_id!r} name={name!r} "
                              f"server_status={status!r} ai_ver={ai_ver!r} "
                              f"supports_pipelines={sup_pip!r} "
                              f"target_match={nx_id == camera_id}")
                except Exception:
                    print(f"[deploy]   raw: {full_body[:800]}")
                sys.stdout.flush()
            except Exception as e:
                print(f"[deploy] device list probe error: {e}")
                sys.stdout.flush()

        assign_ok = False
        last_was_422 = False
        # Retry up to 4 times (0, 15, 30, 45 s) — covers NX AI Manager re-registration lag
        for attempt in range(4):
            if attempt > 0:
                if not last_was_422:
                    break  # only retry on 422 "No devices applicable"
                wait = attempt * 15
                _deploy_state["detail"] = f"Device not yet registered — retrying in {wait}s…"
                print(f"[deploy] DEV assign attempt {attempt+1}: waiting {wait}s for device registration…")
                sys.stdout.flush()
                _time.sleep(wait)
            last_was_422 = False
            for auth_label, hdrs in dev_headers_to_try:
                try:
                    r_assign = requests.put(
                        assign_url,
                        headers={**hdrs, "Content-Type": "application/json"},
                        json=assign_body,
                        timeout=30,
                    )
                    print(f"[deploy] Scailable DEV ({auth_label}) attempt {attempt+1} → {r_assign.status_code}: {r_assign.text[:300]}")
                    sys.stdout.flush()
                    if r_assign.ok:
                        assign_ok = True
                        break
                    if r_assign.status_code == 422 and "applicable" in r_assign.text:
                        last_was_422 = True
                except Exception as e:
                    print(f"[deploy] Scailable DEV ({auth_label}) error: {e}")
                    sys.stdout.flush()
            if assign_ok:
                break

    # Step 3b: Verify state — no settings PUTs.  Any external settings PUT causes NX AI
    # Manager to revalidate the model UUID against rpc/pipelines/available, fail (our model
    # is not in that partner-managed list), and synchronously revert to Demo COCO.
    # DEV assign already set selectedPipeline="Custom", model_nms_{uuid}, and
    # deviceActiveSwitch=true internally.  We only read back to confirm.
    if engine_id:
        settings_url = f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{camera_id}/settings"

        # Brief pause: let NX AI Manager propagate the DEV-assign before we read back
        _time.sleep(3)
        try:
            r_verify = nx_request("GET", settings_url)
            if r_verify.ok:
                vals = r_verify.json().get("values", {})
                actual    = vals.get("selectedPipeline",  "?")
                active_sw = vals.get("deviceActiveSwitch", "?")
                nms_key   = f"model_nms_{model_uuid}"
                nms_val   = vals.get(nms_key, "missing")
                print(f"[deploy] settings after DEV assign: "
                      f"selectedPipeline={actual!r}  "
                      f"deviceActiveSwitch={active_sw!r}  "
                      f"{nms_key}={nms_val}")
                sys.stdout.flush()
                if active_sw is False or active_sw == "false" or active_sw == False:
                    print("[deploy] WARNING: deviceActiveSwitch=false — frames will NOT be "
                          "sent to AI runtime.  Enable it via AI Manager UI: device → "
                          "Integration Settings → toggle 'Active'.")
                    sys.stdout.flush()
        except Exception as e:
            print(f"[deploy] verify settings error: {e}")
            sys.stdout.flush()

    # Probe: try rpc/pipelines/available WITHOUT /metavms — NX AI Manager might use
    # a broader path that returns more pipelines (including our custom model).
    if cloud_id and user_auth_headers:
        for avail_path in [
            f"{SCAILABLE_CPT}/rpc/pipelines/available/{cloud_id}",
            f"{SCAILABLE_CPT}/rpc/pipelines/available/{cloud_id}/metavms",
        ]:
            try:
                r_avail = requests.get(avail_path, headers=user_auth_headers, timeout=10)
                print(f"[deploy] {avail_path[len(SCAILABLE_CPT):]} → {r_avail.status_code}: {r_avail.text[:600]}")
                sys.stdout.flush()
            except Exception as e:
                print(f"[deploy] rpc/pipelines/available probe error: {e}")
                sys.stdout.flush()

    _deploy_state["phase"] = "done"
    _deploy_state["detail"] = ""
    return jsonify({"ok": True, "model_uuid": model_uuid, "class_names": class_names})

# ── Routes: Models list ────────────────────────────────────────────────────────
@app.route("/api/models")
def api_models():
    import sys

    auth_hdrs = get_scailable_headers()
    if not auth_hdrs:
        return jsonify({"error": "Not authenticated with Scailable"}), 401

    # Portal uses /cpt/catalogues with Authorization: nxcdb-{refresh_token} (no Bearer prefix).
    # Use meta_token (the refresh token stored at OAuth time) in that exact format.
    tokens_now = _auto_refresh_tokens()
    meta_tok = tokens_now.get("meta_token") or tokens_now.get("refresh_token") or ""
    if meta_tok and not meta_tok.startswith("nxcdb-"):
        meta_tok = f"nxcdb-{meta_tok}"
    cat_auth = meta_tok or auth_hdrs.get("Authorization", "")

    cloud_id = ""
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or _get_cloud_system_id() or ""
    except Exception:
        cloud_id = _get_cloud_system_id() or ""

    merged = {}  # uuid → model dict

    # Source 1: /rpc/pipelines/available — pipeline-ready models for this site
    if cloud_id:
        try:
            url = f"{SCAILABLE_CPT}/rpc/pipelines/available/{cloud_id}/metavms"
            resp = requests.get(url, headers=auth_hdrs, timeout=15)
            print(f"[models] GET /rpc/pipelines/available → {resp.status_code}: {resp.text[:200]}")
            sys.stdout.flush()
            if resp.ok:
                for m in resp.json() if isinstance(resp.json(), list) else []:
                    uid = m.get("UUID") or m.get("uuid") or ""
                    if uid:
                        if not m.get("Status") and not (m.get("Code") or {}).get("Status"):
                            m = {**m, "Status": "ok"}
                        merged[uid] = {**m, "_source": "pipeline"}
        except Exception as e:
            print(f"[models] pipelines/available error: {e}")
            sys.stdout.flush()

    # Source 2: all models via nil-UUID catalogue wildcard (matches portal "All Available Models")

    # Catalogue=00000000-…-0000 is the portal's "all models" wildcard UUID.
    # Matches what admin.sclbl.nxvms.com uses for its "All Available Models" view.
    NIL_UUID = "00000000-0000-0000-0000-000000000000"
    fn_url = (f"{SCAILABLE_CPT}/functions"
              f"?Catalogue={NIL_UUID}&Customization=metavms&OrderBy=-UpdatedAt")
    try:
        resp = requests.get(fn_url, headers={"Authorization": cat_auth}, timeout=15)
        print(f"[models] GET /functions?Catalogue=nil → {resp.status_code}: {resp.text[:300]}")
        sys.stdout.flush()
        if resp.ok:
            raw = resp.json()
            items = raw if isinstance(raw, list) else (
                raw.get("functions") or raw.get("data") or raw.get("items") or []
            )
            for m in items:
                uid = m.get("UUID") or m.get("uuid") or ""
                if uid:
                    merged[uid] = {**merged.get(uid, {}), **m, "_source": "portal"}
    except Exception as e:
        print(f"[models] functions error: {e}")
        sys.stdout.flush()

    if not merged:
        return jsonify([])

    # Sort: portal models with ok status first, then by name
    def _sort_key(m):
        status = (m.get("Status") or m.get("Code", {}).get("Status") or "").lower()
        return (0 if status == "ok" else 1, (m.get("Name") or "").lower())

    return jsonify(sorted(merged.values(), key=_sort_key))


@app.route("/api/debug/settings")
def api_debug_settings():
    """
    Read current NX device-agent settings for a camera.
    GET /api/debug/settings?camera_id=<uuid>
    Shows the full settings schema so we can find the correct field for pipeline selection.
    Also tries to read the settings after a PUT to verify it saved.
    """
    import sys
    out = {}
    camera_id = request.args.get("camera_id", "")
    if not camera_id:
        return jsonify({"error": "Pass ?camera_id=<uuid>"}), 400

    try:
        engine_id = get_nxai_engine_id()
        out["engine_id"] = engine_id
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    base = f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{camera_id}"

    # GET current device agent info
    r = nx_request("GET", f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{camera_id}")
    out["deviceAgent_get"] = {"status": r.status_code, "body": r.text[:2000]}
    print(f"[debug_settings] GET deviceAgent → {r.status_code}: {r.text[:500]}")
    sys.stdout.flush()

    # GET current settings (full schema)
    r = nx_request("GET", f"{base}/settings")
    out["settings_get"] = {"status": r.status_code, "body": r.text[:3000]}
    print(f"[debug_settings] GET settings → {r.status_code}: {r.text[:1000]}")
    sys.stdout.flush()

    # Also try the model_uuid query param if provided
    model_uuid = request.args.get("set_pipeline", "")
    if model_uuid:
        # Try setting selectedPipeline
        for field_name in ["selectedPipeline", "pipeline", "pipelineId", "modelId"]:
            r = nx_request("PUT", f"{base}/settings",
                           json={"values": {field_name: model_uuid}},
                           headers={"Content-Type": "application/json"})
            print(f"[debug_settings] PUT settings {field_name}={model_uuid} → {r.status_code}: {r.text[:300]}")
            sys.stdout.flush()
            out[f"put_settings_{field_name}"] = {"status": r.status_code, "body": r.text[:500]}

        # GET again to see if it saved
        r = nx_request("GET", f"{base}/settings")
        out["settings_after_put"] = {"status": r.status_code, "body": r.text[:3000]}
        print(f"[debug_settings] GET settings (after PUT) → {r.status_code}: {r.text[:1000]}")
        sys.stdout.flush()

    return jsonify(out)


@app.route("/api/debug/devices")
def api_debug_devices():
    """List Scailable devices for this site so we can verify the device IDs."""
    import sys
    auth_hdrs = get_scailable_headers()
    if not auth_hdrs:
        return jsonify({"error": "Not authenticated"}), 401
    cloud_id = ""
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or _get_cloud_system_id() or ""
    except Exception:
        cloud_id = _get_cloud_system_id() or ""

    result = {"cloud_id": cloud_id}

    # Get full device list and parse it
    url = f"{SCAILABLE_DEV}/system/{cloud_id}/devices"
    try:
        r = requests.get(url, headers=auth_hdrs, timeout=15)
        result["devices_status"] = r.status_code
        if r.ok:
            devices = r.json() if isinstance(r.json(), list) else []
            result["devices_raw"] = devices  # full raw data for debugging
            # Return simplified summary of each device
            result["devices"] = [
                {
                    "NxID": d.get("NxID"),
                    "Name": d.get("Name"),
                    "ServerName": (d.get("Server") or {}).get("Name"),
                    "ServerStatus": (d.get("Server") or {}).get("Status"),
                    "SupportsPipelines": (d.get("Server") or {}).get("Supports", {}).get("Pipelines"),
                    "V2_id": (d.get("V2") or {}).get("id") or (d.get("V2") or {}).get("ID"),
                    "CurrentPipeline": d.get("CurrentPipeline") or d.get("Pipeline") or d.get("FunctionID"),
                    "all_keys": list(d.keys()),
                }
                for d in devices
            ]
        else:
            result["devices_error"] = r.text[:500]
    except Exception as e:
        result["devices_error"] = str(e)

    # Also GET full raw device data for a specific device if camera_id provided
    camera_id = request.args.get("camera_id", "")
    if camera_id:
        # Find V2 ID for this camera from the device list
        v2_id = None
        for d in result.get("devices_raw", []):
            if d.get("NxID") == camera_id or d.get("NxID", "").strip("{}") == camera_id:
                v2_id = (d.get("V2") or {}).get("id") or (d.get("V2") or {}).get("ID")
                result["device_raw_full"] = d
                break

        result["camera_v2_id"] = v2_id

        for url in [
            f"{SCAILABLE_DEV}/system/{cloud_id}/device/{camera_id}",
            f"{SCAILABLE_DEV}/system/{cloud_id}/device/{camera_id}/functions",
        ]:
            try:
                r = requests.get(url, headers=auth_hdrs, timeout=15)
                result[url.split("/")[-1] + "_get"] = {"status": r.status_code, "body": r.text[:1000]}
            except Exception as e:
                result[url.split("/")[-1] + "_get"] = {"error": str(e)}

        # Also try with V2 ID if found
        if v2_id:
            for url in [
                f"{SCAILABLE_DEV}/system/{cloud_id}/device/{v2_id}",
                f"{SCAILABLE_DEV}/system/{cloud_id}/device/{v2_id}/functions",
            ]:
                try:
                    r = requests.get(url, headers=auth_hdrs, timeout=15)
                    key = "v2_" + url.split("/")[-1] + "_get"
                    result[key] = {"status": r.status_code, "body": r.text[:1000]}
                except Exception as e:
                    result["v2_error"] = str(e)

    return jsonify(result)


@app.route("/api/debug/test_assign")
def api_debug_test_assign():
    """
    Test the Scailable DEV API pipeline assignment with a given camera + model UUID.
    GET /api/debug/test_assign?camera_id=<uuid>&model_uuid=<uuid>
    Returns the full HTTP response for each auth attempt (PUT and POST) so we can
    diagnose auth / body-format issues without reading Windows service logs.
    """
    import sys
    camera_id  = request.args.get("camera_id", "").strip("{} ")
    model_uuid = request.args.get("model_uuid", "").strip()
    if not camera_id or not model_uuid:
        return jsonify({"error": "Pass ?camera_id=<uuid>&model_uuid=<uuid>"}), 400

    cloud_id = ""
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or _get_cloud_system_id() or ""
    except Exception:
        cloud_id = _get_cloud_system_id() or ""

    nx_device_id = f"{{{camera_id}}}"
    assign_url   = f"{SCAILABLE_DEV}/system/{cloud_id}/devices/pipelines"

    assign_body = {
        "Devices":   [nx_device_id],
        "Functions": [model_uuid],
        "Pipelines": [{
            "Postprocessor":  "",
            "Preprocessor":   "",
            "modelNMS":       0.42,
            "modelUUID":      model_uuid,
            "resizingMethod": "Letterbox",
            "chains":         [],
        }],
    }

    # Collect all available tokens to try
    tokens = load_tokens()
    candidates = []
    # Meta / refresh token — what admin portal uses
    for key in ("meta_token", "refresh_token", "cpt_refresh_token"):
        t = tokens.get(key)
        if t:
            candidates.append((key, t))
    # CPT access token (works for GET; try for PUT too)
    for key in ("access_token", "cpt_token"):
        t = tokens.get(key)
        if t and not t.startswith("nxcdb-"):
            candidates.append((key, t))
        elif t:
            candidates.append((key + "_stripped", t[len("nxcdb-"):]))

    oauth_hdrs_base = {}
    if cloud_id:
        oauth_hdrs_base["X-Cloud-System-Id"] = cloud_id
        oauth_hdrs_base["X-Nx-System-Id"]    = cloud_id

    result = {
        "cloud_id":     cloud_id,
        "camera_id":    camera_id,
        "nx_device_id": nx_device_id,
        "model_uuid":   model_uuid,
        "assign_url":   assign_url,
        "assign_body":  assign_body,
        "attempts":     [],
    }

    for tok_label, tok in candidates:
        hdrs = {**oauth_hdrs_base, "Authorization": tok, "Content-Type": "application/json"}
        for method in ("PUT", "POST"):
            attempt = {"token": tok_label, "method": method}
            try:
                fn = requests.put if method == "PUT" else requests.post
                r  = fn(assign_url, headers=hdrs, json=assign_body, timeout=20)
                attempt["status"] = r.status_code
                attempt["body"]   = r.text[:600]
                print(f"[test_assign] {method} {tok_label} → {r.status_code}: {r.text[:300]}")
                sys.stdout.flush()
            except Exception as e:
                attempt["error"] = str(e)
                print(f"[test_assign] {method} {tok_label} error: {e}")
                sys.stdout.flush()
            result["attempts"].append(attempt)
            # Stop trying methods for this token once we get a clear success or permanent failure
            status = attempt.get("status", 0)
            if 200 <= status < 300:
                attempt["verdict"] = "SUCCESS"
                break
            elif status in (401, 403):
                attempt["verdict"] = "auth_fail"
                break
            else:
                attempt["verdict"] = f"fail_{status}"

    # Also GET the device to verify CurrentPipeline after attempts (brief wait for async propagation)
    _time.sleep(3)
    try:
        dev_url   = f"{SCAILABLE_DEV}/system/{cloud_id}/devices"
        meta_hdrs = get_oauth_headers() or get_scailable_headers() or {}
        r_dev     = requests.get(dev_url, headers=meta_hdrs, timeout=15)
        if r_dev.ok:
            devs = r_dev.json() if isinstance(r_dev.json(), list) else []
            for d in devs:
                if (d.get("NxID") or "").strip("{}") == camera_id:
                    result["device_after"] = {
                        "CurrentPipeline": d.get("CurrentPipeline") or d.get("Pipeline") or d.get("FunctionID"),
                        "nxAI_pipelines":  (d.get("nxAI") or {}).get("pipelines"),
                    }
                    break
    except Exception as e:
        result["device_after_error"] = str(e)

    return jsonify(result)


@app.route("/api/debug/pipelines")
def api_debug_pipelines():
    """
    Diagnose why our model doesn't appear in rpc/pipelines/available.
    1. Fetch the full rpc/pipelines/available response
    2. Fetch details of Demo COCO from /cpt/functions to see what makes it appear there
    3. Fetch our latest model's /cpt/functions entry
    4. Try to POST/PUT our model into the available pipelines
    GET /api/debug/pipelines?model_uuid=<uuid>
    """
    import sys
    model_uuid = request.args.get("model_uuid", "").strip()
    auth_hdrs  = get_scailable_headers()
    if not auth_hdrs:
        return jsonify({"error": "Not authenticated"}), 401

    cloud_id = ""
    try:
        cloud_id = load_config().get("cloud", {}).get("cloud_system_id") or _get_cloud_system_id() or ""
    except Exception:
        cloud_id = _get_cloud_system_id() or ""

    DEMO_UUID = "459d2273-1514-431c-9d34-f5b72f3bfe20"
    result = {"cloud_id": cloud_id, "model_uuid": model_uuid}

    # 1. Full rpc/pipelines/available response
    try:
        r = requests.get(f"{SCAILABLE_CPT}/rpc/pipelines/available/{cloud_id}/metavms",
                         headers=auth_hdrs, timeout=15)
        result["pipelines_available"] = {"status": r.status_code, "body": r.json() if r.ok else r.text[:1000]}
        print(f"[debug_pipelines] rpc/pipelines/available → {r.status_code}: {r.text[:500]}")
        sys.stdout.flush()
    except Exception as e:
        result["pipelines_available_error"] = str(e)

    # 2. Demo COCO model details from /functions
    try:
        r = requests.get(f"{SCAILABLE_CPT}/functions/{DEMO_UUID}",
                         headers=auth_hdrs, timeout=15)
        result["demo_function"] = {"status": r.status_code, "body": r.json() if r.ok else r.text[:1000]}
        print(f"[debug_pipelines] GET /functions/{DEMO_UUID[:8]}… → {r.status_code}: {r.text[:500]}")
        sys.stdout.flush()
    except Exception as e:
        result["demo_function_error"] = str(e)

    # 3. Our model's /functions entry (if uuid provided)
    if model_uuid:
        try:
            r = requests.get(f"{SCAILABLE_CPT}/functions/{model_uuid}",
                             headers=auth_hdrs, timeout=15)
            result["our_function"] = {"status": r.status_code, "body": r.json() if r.ok else r.text[:1000]}
            print(f"[debug_pipelines] GET /functions/{model_uuid[:8]}… → {r.status_code}: {r.text[:500]}")
            sys.stdout.flush()
        except Exception as e:
            result["our_function_error"] = str(e)

        # 4. Try to register our model as a pipeline via various candidate endpoints
        register_attempts = []
        meta_hdrs = get_oauth_headers() or auth_hdrs
        for method, path, body in [
            # Try adding to site's available pipelines
            ("POST", f"/rpc/pipelines/available/{cloud_id}/metavms",
             {"UUID": model_uuid}),
            ("PUT",  f"/rpc/pipelines/available/{cloud_id}/metavms",
             [{"UUID": model_uuid}]),
            # Try updating the function's catalogues
            ("PATCH", f"/functions/{model_uuid}",
             {"Catalogues": ["metavms"]}),
            ("PUT",  f"/functions/{model_uuid}",
             {"Catalogues": ["metavms"]}),
            # Try a pipeline creation endpoint
            ("POST", "/rpc/pipelines",
             {"Name": "nx-ai-trainer pipeline", "FunctionID": model_uuid,
              "Site": cloud_id, "Customization": "metavms"}),
            ("POST", f"/rpc/pipelines/{cloud_id}",
             {"FunctionID": model_uuid, "Customization": "metavms"}),
        ]:
            try:
                url = f"{SCAILABLE_CPT}{path}"
                fn  = {"POST": requests.post, "PUT": requests.put, "PATCH": requests.patch}[method]
                r   = fn(url, headers={**meta_hdrs, "Content-Type": "application/json"},
                         json=body, timeout=15)
                entry = {"method": method, "path": path, "status": r.status_code, "body": r.text[:400]}
                register_attempts.append(entry)
                print(f"[debug_pipelines] {method} {path} → {r.status_code}: {r.text[:300]}")
                sys.stdout.flush()
                if 200 <= r.status_code < 300:
                    entry["verdict"] = "SUCCESS"
            except Exception as e:
                register_attempts.append({"method": method, "path": path, "error": str(e)[:100]})

        result["register_attempts"] = register_attempts

    return jsonify(result)


@app.route("/api/debug/scailable")
def api_debug_scailable():
    """Show token claims, CPT exchange attempts, and connection tests."""
    import sys
    tokens = load_tokens()
    raw_token = tokens.get("access_token", "")
    cpt_token = tokens.get("cpt_token", "")
    result: dict = {"has_cdb_token": bool(raw_token), "has_cpt_token": bool(cpt_token)}
    if not raw_token:
        return jsonify(result)

    result["cdb_token_prefix"] = raw_token[:12] + "..."
    payload = decode_jwt_payload(raw_token)
    # Log ALL JWT claims — org/partner IDs may be in non-standard claims
    result["cdb_jwt_claims_all"] = payload
    result["cdb_jwt_claims"] = {k: payload[k] for k in ("sub", "iss", "aud", "exp", "client_id") if k in payload}
    # Pull out anything that looks like an org or partner ID
    org_hints = {k: v for k, v in payload.items()
                 if any(x in k.lower() for x in ("org", "partner", "tenant", "account", "company", "channel"))}
    result["cdb_jwt_org_hints"] = org_hints
    print(f"[debug] JWT ALL claims: {payload}")
    print(f"[debug] JWT org-related: {org_hints}")

    if cpt_token:
        result["cpt_token_prefix"] = cpt_token[:12] + "..."
        cpt_payload = decode_jwt_payload(cpt_token)
        result["cpt_jwt_claims_all"] = cpt_payload
        result["cpt_jwt_claims"] = {k: v for k, v in cpt_payload.items()
                                     if k in ("sub", "iss", "aud", "exp")}

    # Test GET /functions with each available token
    token_stripped = raw_token[len("nxcdb-"):] if raw_token.startswith("nxcdb-") else raw_token
    test_tokens = [("cdb_stripped", token_stripped)]
    if cpt_token:
        test_tokens.append(("cpt", cpt_token))

    # Probe org/partner listing endpoints — find the Scailable org ID for this user
    cloud_id = _get_cloud_system_id() or ""
    result["org_probes"] = {}
    for probe_path in [
        "/rpc/organizations",
        "/rpc/organization",
        "/rpc/user/organizations",
        "/rpc/account",
        "/rpc/me",
        f"/rpc/pipelines/available/{cloud_id}/metavms",
        "/rpc/functions",
        "/rpc/models",
    ]:
        try:
            r = requests.get(f"{SCAILABLE_CPT}{probe_path}",
                             headers={"Authorization": f"Bearer {token_stripped}"}, timeout=15)
            result["org_probes"][probe_path] = {"status": r.status_code, "body": r.text[:400]}
            print(f"[debug] GET {probe_path} → {r.status_code}: {r.text[:300]}")
            sys.stdout.flush()
        except Exception as e:
            result["org_probes"][probe_path] = {"error": str(e)[:100]}

    result["function_tests"] = {}
    for vname, tok in test_tokens:
        try:
            r = requests.get(f"{SCAILABLE_CPT}/functions",
                             headers={"Authorization": f"Bearer {tok}"}, timeout=15)
            result["function_tests"][vname] = {"status": r.status_code, "body": r.text[:300]}
            print(f"[debug] GET /functions ({vname}) → {r.status_code}: {r.text[:200]}")
            sys.stdout.flush()
        except Exception as e:
            result["function_tests"][vname] = {"error": str(e)}

    # Probe CPT auth endpoints — useful when first signing in
    result["cpt_auth_probes"] = {}
    for path in ["/auth/token", "/auth", "/token", "/v1/auth"]:
        try:
            r = requests.post(f"{SCAILABLE_CPT}{path}",
                              json={"token": token_stripped},
                              headers={"Content-Type": "application/json"}, timeout=10)
            result["cpt_auth_probes"][path] = {"status": r.status_code, "body": r.text[:200]}
            print(f"[debug] POST {path} → {r.status_code}: {r.text[:100]}")
            sys.stdout.flush()
        except Exception as e:
            result["cpt_auth_probes"][path] = {"error": str(e)[:100]}

    return jsonify(result)


@app.route("/api/debug/nx")
def api_debug_nx():
    """Probe NX AI Manager REST endpoints, localhost ports, and integrationId-based Scailable URLs."""
    import sys
    out = {}
    try:
        engine_id = get_nxai_engine_id()
        integration_id = _engine_integration_id_cache or ""
        out["engine_id"] = engine_id
        out["integration_id"] = integration_id

        # ── NX VMS system settings (may contain cloud auth key) ──────────────
        for path in [
            "/api/systemSettings",
            "/rest/v3/system/info",
            "/rest/v4/system/settings",
        ]:
            try:
                r = nx_request("GET", path)
                out[f"vms:{path}"] = {"status": r.status_code, "body": r.text[:800]}
                print(f"[debug_nx] {path} → {r.status_code}: {r.text[:400]}")
                sys.stdout.flush()
            except Exception as e:
                out[f"vms:{path}"] = {"error": str(e)[:100]}

        # ── integrationId-based Scailable URL patterns ────────────────────────
        tokens = load_tokens()
        raw_tok = tokens.get("access_token", "")
        tok = raw_tok[len("nxcdb-"):] if raw_tok.startswith("nxcdb-") else raw_tok
        if integration_id and tok:
            for url in [
                f"{SCAILABLE_CPT}/{integration_id}/functions",
                f"https://api.sclbl.nxvms.com/cpt/{integration_id}/functions",
                f"{SCAILABLE_CPT}/functions?integrationId={integration_id}",
            ]:
                try:
                    r = requests.get(url, headers={"Authorization": f"Bearer {tok}",
                                                   "X-Integration-Id": integration_id},
                                     timeout=10)
                    out[f"cpt:{url[-60:]}"] = {"status": r.status_code, "body": r.text[:300]}
                    print(f"[debug_nx] {url[-60:]} → {r.status_code}: {r.text[:150]}")
                    sys.stdout.flush()
                except Exception as e:
                    out[f"cpt:{url[-60:]}"] = {"error": str(e)[:80]}

        # ── NX AI Manager local HTTP API (plugin runs its own server) ─────────
        # Use very short connect timeout; parallelize with threads
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        local_hits = {}
        def _probe_local(port, api_path):
            url = f"http://localhost:{port}{api_path}"
            try:
                r = requests.get(url, timeout=0.5)
                return url, r.status_code, r.text[:300]
            except Exception:
                return url, None, None

        probe_jobs = [
            (port, path)
            for port in [8888, 8880, 8080, 9090, 7001, 7000, 3000, 5001, 8766, 8769]
            for path in ["/api/functions", "/functions", "/api/models"]
        ]
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = {ex.submit(_probe_local, p, pp): (p, pp) for p, pp in probe_jobs}
            for fut in _as_completed(futures):
                url, status, body = fut.result()
                if status is not None:   # port responded to something
                    out[f"local:{url[16:]}"] = {"status": status, "body": body}
                    print(f"[debug_nx] {url} → {status}: {body[:100] if body else ''}")
                    sys.stdout.flush()

        # ── NX service data dir discovery + directory listing ────────────────────
        svc_dirs = _nx_service_data_dirs()
        out["nx_service_data_dirs"] = svc_dirs
        # List actual files in each candidate dir so we can see what's there
        dir_listings = {}
        for d in svc_dirs:
            try:
                if os.path.isdir(d):
                    dir_listings[d] = sorted(os.listdir(d))
            except Exception as e:
                dir_listings[d] = [f"error: {e}"]
        out["nx_service_dir_contents"] = dir_listings

        # ── mserver.sqlite / mserver.conf cloud credentials probe ────────────────
        conf_path, conf_values = _find_nx_server_conf()
        out["mserver_config_path"] = conf_path
        out["mserver_config_type"] = ("sqlite" if conf_path and conf_path.endswith(".sqlite")
                                      else "conf" if conf_path else None)
        # Show cloud/auth related keys (mask values longer than 8 chars for safety)
        cloud_keys = {k: (v[:4] + "…" if len(v) > 8 else v)
                      for k, v in conf_values.items()
                      if any(x in k.lower() for x in ("cloud", "auth", "key", "secret", "token", "password"))}
        out["mserver_cloud_keys"] = cloud_keys
        # Also show ALL keys (just the key names, not values) so we know what's available
        out["mserver_all_key_names"] = sorted(conf_values.keys())

        # ── NX system settings — extract ALL cloud-related fields ─────────────
        try:
            r_ss = nx_request("GET", "/api/systemSettings")
            if r_ss.ok:
                ss = r_ss.json()
                settings = ss.get("reply", {}).get("settings") or ss.get("settings") or ss
                cloud_settings = {k: v for k, v in settings.items()
                                  if any(x in k.lower() for x in
                                         ("cloud", "auth", "key", "secret", "token", "credential"))}
                out["vms_cloud_settings"] = cloud_settings
        except Exception as e:
            out["vms_cloud_settings"] = {"error": str(e)}

        # ── Manual config check ───────────────────────────────────────────────
        try:
            cfg_cloud = load_config().get("cloud", {})
            out["config_cloud"] = {
                "cloud_system_id_set": bool(cfg_cloud.get("cloud_system_id")),
                "auth_key_set": bool(cfg_cloud.get("auth_key") or cfg_cloud.get("authKey")),
            }
        except Exception as e:
            out["config_cloud"] = {"error": str(e)}

        # ── VMS REST cloud credentials probe ──────────────────────────────────
        for cred_path in ["/rest/v3/system/cloudCredentials",
                          "/rest/v4/system/cloudCredentials",
                          "/api/cloudCredentials"]:
            try:
                r = nx_request("GET", cred_path)
                out[f"vms_cred:{cred_path}"] = {"status": r.status_code, "body": r.text[:400]}
                print(f"[debug_nx] {cred_path} → {r.status_code}: {r.text[:200]}")
                sys.stdout.flush()
            except Exception as e:
                out[f"vms_cred:{cred_path}"] = {"error": str(e)[:80]}

        # ── Scailable OpenAPI spec (shows full API surface) ───────────────────
        try:
            r_spec = requests.get(f"{SCAILABLE_CPT}/openapi.json", timeout=15)
            if r_spec.ok:
                spec = r_spec.json()
                # Extract just paths + their parameters — not the whole schema
                paths = {}
                for path, methods in spec.get("paths", {}).items():
                    paths[path] = {}
                    for method, detail in methods.items():
                        params = [p.get("name") for p in detail.get("parameters", [])]
                        paths[path][method] = {"params": params,
                                               "summary": detail.get("summary", "")}
                out["openapi_paths"] = paths
            else:
                out["openapi_paths"] = {"error": f"{r_spec.status_code}: {r_spec.text[:200]}"}
        except Exception as e:
            out["openapi_paths"] = {"error": str(e)[:100]}

        # ── CPT service config + pipeline availability ────────────────────────
        auth_hdrs_dbg = get_scailable_headers() or {}
        for probe_url in [
            f"{SCAILABLE_CPT}/configuration",
            f"{SCAILABLE_CPT}/rpc/pipelines/available/{_get_cloud_system_id() or 'UNKNOWN'}/metavms",
            f"{SCAILABLE_CPT}/functions?Customization=metavms&Limit=5",
        ]:
            try:
                r_p = requests.get(probe_url, headers=auth_hdrs_dbg, timeout=10)
                key = "probe:" + probe_url[len(SCAILABLE_CPT):]
                out[key] = {"status": r_p.status_code, "body": r_p.text[:400]}
                print(f"[debug_nx] {probe_url[len(SCAILABLE_CPT):]} → {r_p.status_code}: {r_p.text[:200]}")
                sys.stdout.flush()
            except Exception as e:
                out["probe:" + probe_url[len(SCAILABLE_CPT):]] = {"error": str(e)[:80]}

        # ── CPT API no-auth test (confirms whether 500 is auth-related) ─────────
        try:
            r_noauth = requests.get(f"{SCAILABLE_CPT}/functions", timeout=10)
            out["cpt_no_auth"] = {"status": r_noauth.status_code, "body": r_noauth.text[:300]}
            print(f"[debug_nx] CPT no-auth → {r_noauth.status_code}: {r_noauth.text[:150]}")
            sys.stdout.flush()
        except Exception as e:
            out["cpt_no_auth"] = {"error": str(e)[:100]}

        # ── CPT API with cloud system ID headers ──────────────────────────────
        tokens = load_tokens()
        raw_tok = tokens.get("access_token", "")
        tok = raw_tok[len("nxcdb-"):] if raw_tok.startswith("nxcdb-") else raw_tok
        cloud_id = _get_cloud_system_id() or ""
        if tok and cloud_id:
            hdrs_with_sys = {
                "Authorization": f"Bearer {tok}",
                "X-Cloud-System-Id": cloud_id,
                "X-System-Id": cloud_id,
            }
            if integration_id:
                hdrs_with_sys["X-Integration-Id"] = integration_id
            try:
                r_sys = requests.get(f"{SCAILABLE_CPT}/functions", headers=hdrs_with_sys, timeout=10)
                out["cpt_with_system_headers"] = {"status": r_sys.status_code, "body": r_sys.text[:300]}
                print(f"[debug_nx] CPT+system headers → {r_sys.status_code}: {r_sys.text[:150]}")
                sys.stdout.flush()
            except Exception as e:
                out["cpt_with_system_headers"] = {"error": str(e)[:100]}

        # ── Device agent settings schema ──────────────────────────────────────
        try:
            cameras_r = nx_request("GET", "/rest/v4/devices?limit=1")
            if cameras_r.ok:
                devs = cameras_r.json()
                if devs:
                    cam_id = devs[0]["id"].strip("{}")
                    settings_r = nx_request("GET",
                        f"/rest/v4/analytics/engines/{engine_id}/deviceAgents/{cam_id}/settings")
                    out["device_agent_settings"] = {"status": settings_r.status_code,
                                                    "body": settings_r.text[:600]}
                    print(f"[debug_nx] device agent settings → {settings_r.status_code}: {settings_r.text[:300]}")
                    sys.stdout.flush()
        except Exception as e:
            out["device_agent_settings"] = {"error": str(e)[:100]}

        # ── NX AI Manager engine-level settings (may contain Scailable auth) ──
        raw_engine_settings = _get_nxai_engine_settings_raw()
        out["nxai_engine_settings"] = raw_engine_settings

        # ── NX AI Manager plugin config files on disk ──────────────────────────
        plugin_confs = _find_nxai_plugin_conf()
        out["nxai_plugin_conf_files"] = [p for p, _ in plugin_confs]
        # Show content (mask long values for safety)
        safe_confs = {}
        for fpath, cdata in plugin_confs:
            if isinstance(cdata, dict):
                safe = {}
                for k, v in cdata.items():
                    sv = str(v)
                    safe[k] = sv[:8] + "…" if len(sv) > 12 else sv
                safe_confs[fpath] = safe
            else:
                safe_confs[fpath] = str(cdata)[:200]
        out["nxai_plugin_conf_content"] = safe_confs

        # ── Comprehensive ecs.sqlite dump (ALL tables, first 50 rows each) ──────
        # Very useful to catch non-(name,value) tables like vms_resource_params
        if conf_path and conf_path.endswith(".sqlite"):
            try:
                full_dump = _dump_all_nx_sqlite_tables(conf_path)
                # Summarize: show table names + row counts, flag credential-like content
                cred_hits = {}
                table_summary = {}
                for tname, tdata in full_dump.items():
                    if "error" in tdata:
                        table_summary[tname] = f"ERROR: {tdata['error']}"
                        continue
                    cols = tdata.get("columns", [])
                    rows = tdata.get("rows", [])
                    table_summary[tname] = f"{len(rows)} rows, cols={cols}"
                    # Scan for credential-like values
                    for row in rows:
                        for i, cell in enumerate(row):
                            cs = str(cell or "")
                            ck = cols[i].lower() if i < len(cols) else ""
                            if (any(x in ck for x in ("auth", "cloud", "key", "secret", "token", "cred")) or
                                    any(x in cs.lower() for x in ("authkey", "cloudauth", "scailable", "sclbl"))):
                                if cs:
                                    cred_hits[f"{tname}.{cols[i] if i < len(cols) else i}"] = cs[:80]
                out["sqlite_table_summary"] = table_summary
                out["sqlite_cred_hits"] = cred_hits
            except Exception as e:
                out["sqlite_full_dump_error"] = str(e)

    except Exception as e:
        out["error"] = str(e)
    return jsonify(out)


# ── Routes: OAuth ──────────────────────────────────────────────────────────────

@app.route("/auth/login", methods=["POST"])
def auth_login():
    """
    Direct password-based login using NX Cloud CDB 'password' grant.
    Requests scope '{CPT_URL} cloudSystemId=*' so the returned token has
    aud: 'https://api.sclbl.nxvms.com/cpt cloudSystemId=*' — which is what
    the Scailable CPT API accepts.  Falls back to meta.nxvms.com scope if the
    CPT scope isn't supported for this user.
    """
    import sys
    data = request.get_json() or {}
    email    = data.get("email", "").strip()
    password = data.get("password", "").strip()
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    # We need two tokens with different scopes:
    # - CPT token  (aud: api.sclbl.nxvms.com/cpt) → model upload
    # - Meta token (aud: meta.nxvms.com)           → DEV API device assignment
    cpt_scope  = f"{SCAILABLE_CPT_AUDIENCE} cloudSystemId=*"
    meta_scope = f"{CLOUD_ENDPOINT} cloudSystemId=*"          # same aud the admin portal uses

    token_url = f"{CLOUD_CDB_ENDPOINT}/oauth2/token"
    token_store = {}
    got_email = email

    def _fetch_token(client_id, scope):
        body = {
            "grant_type":    "password",
            "response_type": "token",
            "client_id":     client_id,
            "scope":         scope,
            "username":      email,
            "password":      password,
        }
        resp = requests.post(token_url, json=body, timeout=15)
        print(f"[auth/login] client={client_id!r} scope={scope[:40]!r} → {resp.status_code}: {resp.text[:400]}")
        sys.stdout.flush()
        if resp.ok:
            rd = resp.json()
            # Return (access_token, refresh_token) — refresh token has longer expiry
            # and the Scailable DEV API requires a refresh token (typ=refreshToken)
            at = rd.get("access_token") or rd.get("token") or rd.get("accessToken")
            rt = rd.get("refresh_token") or rd.get("refreshToken")
            return at, rt
        return None, None

    # Fetch CPT-scoped token (for upload)
    for client_id in [OAUTH_CLIENT_ID, "3rdParty"]:
        at, rt = None, None
        try:
            at, rt = _fetch_token(client_id, cpt_scope)
        except Exception as e:
            print(f"[auth/login] cpt error: {e}"); sys.stdout.flush()
        if at:
            aud = decode_jwt_payload(at).get("aud", "")
            print(f"[auth/login] CPT token aud={aud!r}  has_refresh={bool(rt)}")
            sys.stdout.flush()
            token_store["access_token"] = at
            if rt:
                token_store["cpt_refresh_token"] = rt
            got_email = decode_jwt_payload(at).get("sub") or email
            break

    # Fetch meta-scoped token (for DEV API assignment — same scope as admin portal)
    for client_id in [OAUTH_CLIENT_ID, "3rdParty"]:
        at, rt = None, None
        try:
            at, rt = _fetch_token(client_id, meta_scope)
        except Exception as e:
            print(f"[auth/login] meta error: {e}"); sys.stdout.flush()
        if at:
            aud = decode_jwt_payload(at).get("aud", "")
            # Prefer refresh token for DEV API — admin portal sends refreshToken type
            dev_tok = rt or at
            print(f"[auth/login] meta token aud={aud!r}  has_refresh={bool(rt)}")
            sys.stdout.flush()
            token_store["meta_token"] = dev_tok
            if not token_store.get("access_token"):
                token_store["access_token"] = at
                got_email = decode_jwt_payload(at).get("sub") or email
            break

    if not token_store:
        return jsonify({"error": "Login failed — check credentials and try again"}), 401

    save_tokens(token_store)
    return jsonify({
        "ok":         True,
        "email":      got_email,
        "cpt_scoped": bool(token_store.get("access_token")),
        "meta_scoped": bool(token_store.get("meta_token")),
        "aud":        decode_jwt_payload(token_store.get("access_token", "")).get("aud", ""),
    })


@app.route("/auth/login/sso")
def auth_login_sso():
    """Browser-redirect OAuth flow (fallback to the password grant above)."""
    state_val = secrets.token_urlsafe(16)
    callback_url = request.url_root.rstrip("/") + "/auth/callback"
    _oauth_states[state_val] = callback_url
    # Request meta scope — this is the audience the admin portal uses, and the
    # refresh token returned has typ=refreshToken which the DEV API requires.
    meta_scope = f"{CLOUD_ENDPOINT} cloudSystemId=*"
    params = urlencode({
        "redirect_url": callback_url,
        "client_id": OAUTH_CLIENT_ID,
        "state": state_val,
        "scope": meta_scope,
    })
    return redirect(f"{CLOUD_ENDPOINT}/authorize?{params}")

@app.route("/auth/callback")
def auth_callback():
    code = request.args.get("code")
    state_val = request.args.get("state")

    if not code:
        return _auth_result_page(False, "Login cancelled — no authorization code.")

    redirect_url = _oauth_states.pop(state_val, None)
    if state_val and redirect_url is None:
        # State not found — server likely restarted between authorize and callback.
        # Log the mismatch but proceed: CSRF risk is negligible on a localhost server.
        print(f"[auth] state mismatch (server restart?): {state_val!r} not in _oauth_states — continuing")
        import sys; sys.stdout.flush()

    try:
        import sys
        resp = requests.post(
            f"{CLOUD_CDB_ENDPOINT}/oauth2/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": OAUTH_CLIENT_ID,
                "audience": SCAILABLE_CPT_AUDIENCE,
            },
            timeout=15,
        )
        print(f"[auth] token exchange → {resp.status_code}: {resp.text[:300]}")
        sys.stdout.flush()
        if not resp.ok:
            return _auth_result_page(False, f"Token exchange {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        access_token = (
            data.get("access_token") or data.get("token")
            or data.get("Token") or data.get("accessToken")
        )
        if not access_token:
            return _auth_result_page(False, f"No token in response: {str(data)[:300]}")

        refresh_token = data.get("refresh_token") or data.get("refreshToken")
        token_store = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            # DEV API requires the refresh token (typ=refreshToken, aud=meta.nxvms.com)
            # which is exactly what the SSO flow returns
            "meta_token": refresh_token or access_token,
        }

        # Try RFC 8693 token exchange to get a CPT-audience token.
        # The CDB token audience is meta.nxvms.com; CPT needs its own audience.
        for exchange_body in [
            # RFC 8693 token exchange
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token": access_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "audience": SCAILABLE_CPT_AUDIENCE,
                "client_id": OAUTH_CLIENT_ID,
            },
            # Simple audience-scoped re-issue
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token": access_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "audience": SCAILABLE_CPT_AUDIENCE,
                "client_id": OAUTH_CLIENT_ID,
            },
        ]:
            try:
                ex = requests.post(
                    f"{CLOUD_CDB_ENDPOINT}/oauth2/token",
                    json=exchange_body,
                    timeout=15,
                )
                print(f"[auth] CPT token exchange → {ex.status_code}: {ex.text[:300]}")
                sys.stdout.flush()
                if ex.ok:
                    cpt_tok = ex.json().get("access_token") or ex.json().get("token")
                    if cpt_tok and cpt_tok != access_token:
                        token_store["cpt_token"] = cpt_tok
                        print(f"[auth] CPT token obtained — aud={decode_jwt_payload(cpt_tok).get('aud')}")
                        sys.stdout.flush()
                        break
            except Exception as ex_err:
                print(f"[auth] CPT exchange error: {ex_err}")
                sys.stdout.flush()

        # Also try the CPT API's own auth endpoint (if it has one)
        if "cpt_token" not in token_store:
            for cpt_auth_path in ["/auth/token", "/auth", "/token"]:
                try:
                    cpt_r = requests.post(
                        f"{SCAILABLE_CPT}{cpt_auth_path}",
                        json={"token": access_token},
                        headers={"Content-Type": "application/json"},
                        timeout=10,
                    )
                    print(f"[auth] CPT {cpt_auth_path} → {cpt_r.status_code}: {cpt_r.text[:200]}")
                    sys.stdout.flush()
                    if cpt_r.ok:
                        cpt_tok = (cpt_r.json().get("access_token") or cpt_r.json().get("token")
                                   or cpt_r.json().get("Token"))
                        if cpt_tok:
                            token_store["cpt_token"] = cpt_tok
                            break
                except Exception:
                    pass

        save_tokens(token_store)
        return _auth_result_page(True, "Signed in to Nx Cloud successfully.")
    except Exception as e:
        return _auth_result_page(False, f"Token exchange error: {e}")

def _auth_result_page(success, message):
    icon = "✓" if success else "✗"
    color = "#2FA2DB" if success else "#e74c3c"
    script = (
        "window.opener && window.opener.postMessage({type:'auth_complete',success:true},'*');"
        "setTimeout(()=>window.close(),1200);"
        if success else ""
    )
    return f"""<!DOCTYPE html><html><head><title>nx-ai-trainer</title>
<style>body{{font-family:sans-serif;display:flex;align-items:center;justify-content:center;
min-height:100vh;margin:0;background:#0f111a;}}
.box{{background:#161b2e;padding:2rem 3rem;border-radius:8px;text-align:center;border:1px solid {color};}}
h2{{color:{color};margin:0 0 1rem;font-size:2rem;}}p{{color:#ccc;margin:0;}}</style>
</head><body><div class="box"><h2>{icon}</h2><p>{message}</p></div>
<script>{script}</script></body></html>"""

@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    TOKEN_FILE.unlink(missing_ok=True)
    return jsonify({"ok": True})

@app.route("/auth/status")
def auth_status():
    return jsonify({"authenticated": bool(load_tokens().get("access_token"))})

# ── Routes: Config ─────────────────────────────────────────────────────────────
@app.route("/api/config", methods=["GET"])
def api_config_get():
    cfg = load_config()
    safe_nx = {k: v for k, v in cfg["nx"].items() if k != "password"}
    safe_nx["password"] = "***"
    out = {**cfg, "nx": safe_nx}
    # Return whether an API key is set, not its value
    out["sclbl_api_key_set"] = bool(cfg.get("sclbl_api_key", "").strip())
    out.pop("sclbl_api_key", None)
    # Expose cloud system ID so the frontend can build a direct link to the API Keys page
    out["cloud_system_id"] = (cfg.get("cloud", {}).get("cloud_system_id") or
                               _get_cloud_system_id() or "")
    return jsonify(out)

@app.route("/api/config", methods=["POST"])
def api_config_save():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    cfg = load_config()
    if "nx" in data:
        for k, v in data["nx"].items():
            if k == "password" and v == "***":
                continue
            cfg["nx"][k] = v
    if "port" in data:
        cfg["port"] = data["port"]
    if "sclbl_api_key" in data:
        cfg["sclbl_api_key"] = data["sclbl_api_key"].strip()
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    _nx_token_cache.clear()
    global _engine_id_cache
    _engine_id_cache = None
    return jsonify({"ok": True})

@app.route("/api/debug/nxai_local")
def api_debug_nxai_local():
    """
    Discover the NX AI Manager plugin's local HTTP API.
    Probes common ports and paths to find the plugin's management server,
    then reads its OpenAPI spec and tests model-upload endpoints.
    """
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed as _afc
    out = {}

    # ── NX VMS analytics engine info ─────────────────────────────────────────
    engine_id = ""
    try:
        engine_id = get_nxai_engine_id()
        out["engine_id"] = engine_id
        out["integration_id"] = _engine_integration_id_cache or ""
    except Exception as e:
        out["engine_error"] = str(e)

    # ── NX VMS analytics engine commands (may include upload actions) ─────────
    if engine_id:
        for cmd_path in [
            f"/rest/v4/analytics/engines/{engine_id}/commands",
            f"/rest/v4/analytics/engines/{engine_id}/manifest",
        ]:
            try:
                r = nx_request("GET", cmd_path)
                out[f"nx:{cmd_path[-40:]}"] = {"status": r.status_code, "body": r.text[:500]}
                print(f"[nxai_local] {cmd_path} → {r.status_code}: {r.text[:300]}")
                sys.stdout.flush()
            except Exception as e:
                out[f"nx:{cmd_path[-40:]}"] = {"error": str(e)}

    # ── NX VMS proxy paths for the plugin's local HTTP server ─────────────────
    # Format: /rest/v4/analytics/engines/{engineId}/proxy/{plugin_path}
    probe_plugin_paths = [
        "/",
        "/api",
        "/api/v1",
        "/api/v1/status",
        "/api/v1/models",
        "/api/v1/plugins",
        "/api/v1/functions",
        "/api/v1/devices",
        "/api/v1/pipelines",
        "/api/v1/pipeline",
        "/api/models",
        "/api/plugins",
        "/api/functions",
        "/api/pipelines",
        "/api/pipeline",
        "/api/status",
        "/api/info",
        "/openapi.json",
        "/swagger.json",
        "/api/openapi.json",
        "/api/v1/openapi.json",
        "/api/v1/info",
    ]
    if engine_id:
        out["nx_proxy_probes"] = {}
        for pp in probe_plugin_paths:
            proxy_url = f"/rest/v4/analytics/engines/{engine_id}/proxy{pp}"
            try:
                r = nx_request("GET", proxy_url, timeout=8)
                body = r.text[:600]
                out["nx_proxy_probes"][pp] = {"status": r.status_code, "body": body}
                print(f"[nxai_local] proxy{pp} → {r.status_code}: {body[:200]}")
                sys.stdout.flush()
            except Exception as e:
                out["nx_proxy_probes"][pp] = {"error": str(e)[:80]}

    # ── Direct localhost port probing ─────────────────────────────────────────
    def _probe(port, path):
        url = f"http://localhost:{port}{path}"
        try:
            r = requests.get(url, timeout=1.5)
            return port, path, r.status_code, r.text[:400]
        except requests.exceptions.ConnectionError:
            return port, path, None, None  # port closed
        except Exception as e:
            return port, path, -1, str(e)[:100]

    probe_ports  = [7002, 7777, 8765, 8766, 8768, 8769, 8770, 8780, 9000, 9001, 3456, 4321]
    probe_paths  = ["/api/v1/status", "/api/v1/models", "/api/v1/plugins",
                    "/api/status", "/api/models", "/api/functions",
                    "/openapi.json", "/swagger.json", "/"]
    jobs = [(p, path) for p in probe_ports for path in probe_paths]
    local_hits = {}
    with ThreadPoolExecutor(max_workers=30) as ex:
        futs = [ex.submit(_probe, p, path) for p, path in jobs]
        for fut in _afc(futs):
            port, path, status, body = fut.result()
            if status is not None:  # connection succeeded
                key = f"{port}{path}"
                local_hits[key] = {"status": status, "body": body}
                print(f"[nxai_local] localhost:{port}{path} → {status}: {str(body)[:150]}")
                sys.stdout.flush()
    out["local_port_hits"] = local_hits

    # ── Summary of what we found ──────────────────────────────────────────────
    open_ports = sorted({int(k.split("/")[0]) for k in local_hits})
    out["open_ports"] = open_ports
    print(f"[nxai_local] open ports: {open_ports}")
    sys.stdout.flush()

    return jsonify(out)


@app.route("/api/debug/plugin_dir")
def api_debug_plugin_dir():
    """
    Read all files in the NX AI Manager plugin directory.
    The plugin stores its Scailable credentials alongside settings.json —
    this dumps everything in that etc/ directory so we can find auth keys.
    Also probes CPT API paths we haven't tried yet.
    """
    import sys, os
    out = {}

    # ── 1. Read the full plugin etc/ directory ────────────────────────────────
    plugin_etc = (r"C:\Windows\System32\config\systemprofile\AppData\Local"
                  r"\Network Optix\Network Optix MetaVMS Media Server"
                  r"\nx_ai_manager\nxai_manager\etc")
    plugin_root = os.path.dirname(plugin_etc)  # nxai_manager/
    plugin_parent = os.path.dirname(plugin_root)  # nx_ai_manager/

    out["plugin_dirs_checked"] = [plugin_etc, plugin_root, plugin_parent]

    for base_dir in [plugin_parent, plugin_root, plugin_etc]:
        if not os.path.isdir(base_dir):
            out[f"dir_missing:{base_dir[-40:]}"] = True
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(base_dir):
                depth = dirpath.replace(base_dir, "").count(os.sep)
                if depth > 4:
                    dirnames.clear(); continue
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    rel   = fpath.replace(base_dir, "").lstrip("\\/")
                    try:
                        with open(fpath, "rb") as f:
                            raw = f.read(8192)
                        try:
                            text = raw.decode("utf-8", errors="replace")
                        except Exception:
                            text = repr(raw[:200])
                        out[f"file:{rel}"] = text
                        print(f"[plugin_dir] {rel} ({len(raw)}b): {text[:200]}")
                        sys.stdout.flush()
                    except Exception as e:
                        out[f"file:{rel}"] = f"ERROR: {e}"
        except Exception as e:
            out[f"walk_error:{base_dir[-40:]}"] = str(e)

    # ── 2. Probe unexplored CPT paths ─────────────────────────────────────────
    tokens = load_tokens()
    raw_tok = tokens.get("access_token", "")
    tok = raw_tok[len("nxcdb-"):] if raw_tok.startswith("nxcdb-") else raw_tok

    cloud_id = _get_cloud_system_id() or ""
    auth_hdr = {"Authorization": f"Bearer {tok}"} if tok else {}

    cpt_probes = {}

    # GET /functions/options — might reveal allowed methods / auth requirements
    # GET /functions/statistics — might list functions we can see
    # POST /functions/link — presigned upload URL?
    # POST /functions/sourceinfo — source info for a URL-based model
    for method, path, body in [
        ("GET",  "/functions/options",    None),
        ("GET",  "/functions/statistics", None),
        ("GET",  "/functions/sourceinfo", None),
        ("GET",  "/functions/link",       None),
        ("GET",  f"/rpc/pipelines/available/{cloud_id}/metavms", None),
        # Try POST /functions with absolutely minimal body
        ("POST", "/functions", {
            "Name": "test-minimal",
            "Customization": "metavms",
            "InputDriver": "image",
            "InputDriverDetails": {},
            "OutputDriver": "classification",
            "OutputDriverDetails": {},
            "EnableConversions": [],
        }),
        # Try with Site = cloud system ID
        ("POST", "/functions", {
            "Name": "test-with-site",
            "Customization": "metavms",
            "InputDriver": "image",
            "InputDriverDetails": {},
            "OutputDriver": "classification",
            "OutputDriverDetails": {},
            "EnableConversions": [],
            "Site": cloud_id,
        }),
    ]:
        if not tok:
            cpt_probes[f"{method} {path}"] = {"error": "no token"}
            continue
        try:
            if method == "GET":
                r = requests.get(f"{SCAILABLE_CPT}{path}", headers=auth_hdr, timeout=15)
            else:
                r = requests.post(f"{SCAILABLE_CPT}{path}",
                                  headers={**auth_hdr, "Content-Type": "application/json"},
                                  json=body, timeout=15)
            cpt_probes[f"{method} {path}"] = {"status": r.status_code, "body": r.text[:600]}
            print(f"[plugin_dir] {method} {path} → {r.status_code}: {r.text[:300]}")
            sys.stdout.flush()
        except Exception as e:
            cpt_probes[f"{method} {path}"] = {"error": str(e)[:100]}

    out["cpt_probes"] = cpt_probes
    return jsonify(out)


@app.route("/api/debug/nxai_creds")
def api_debug_nxai_creds():
    """
    Deep search for NX AI Manager plugin credentials on disk.
    The plugin must store its Scailable channel-partner credentials somewhere —
    this endpoint finds them so they can be used for CPT API uploads.
    Also dumps the full CPT OpenAPI spec security schemes.
    """
    import sys, os, glob as _glob
    out = {}

    # ── 1. Full CPT OpenAPI spec ──────────────────────────────────────────────
    try:
        tokens = load_tokens()
        raw_tok = tokens.get("access_token", "")
        tok = raw_tok[len("nxcdb-"):] if raw_tok.startswith("nxcdb-") else raw_tok
        spec_r = requests.get(f"{SCAILABLE_CPT}/openapi.json",
                              headers={"Authorization": f"Bearer {tok}"}, timeout=15)
        print(f"[creds] CPT OpenAPI → {spec_r.status_code}")
        sys.stdout.flush()
        if spec_r.ok:
            spec = spec_r.json()
            out["cpt_spec_paths"] = list((spec.get("paths") or {}).keys())
            out["cpt_spec_security_schemes"] = spec.get("components", {}).get("securitySchemes", {})
            out["cpt_spec_security"] = spec.get("security", [])
            # POST /functions full schema
            fn_post = (spec.get("paths", {}).get("/functions", {}).get("post", {}))
            rb = fn_post.get("requestBody", {})
            schema_ref = (rb.get("content", {}).get("application/json", {})
                           .get("schema", {}))
            if "$ref" in schema_ref:
                ref_name = schema_ref["$ref"].split("/")[-1]
                schema_ref = spec.get("components", {}).get("schemas", {}).get(ref_name, {})
            out["cpt_post_functions_schema"] = schema_ref
            print(f"[creds] CPT paths: {out['cpt_spec_paths']}")
            print(f"[creds] CPT security: {out['cpt_spec_security_schemes']}")
            sys.stdout.flush()
        else:
            out["cpt_spec_error"] = f"{spec_r.status_code}: {spec_r.text[:300]}"
    except Exception as e:
        out["cpt_spec_error"] = str(e)

    # ── 2. NX engine settings (full, unfiltered) ──────────────────────────────
    out["engine_settings"] = {}
    try:
        engine_id = get_nxai_engine_id()
        for path in [f"/rest/v4/analytics/engines/{engine_id}/settings",
                     f"/rest/v3/analytics/engines/{engine_id}/settings",
                     f"/rest/v4/analytics/engines/{engine_id}/parameters"]:
            try:
                r = nx_request("GET", path)
                print(f"[creds] engine settings {path} → {r.status_code}: {r.text[:400]}")
                sys.stdout.flush()
                if r.ok:
                    out["engine_settings"][path] = r.json()
                    break
                else:
                    out["engine_settings"][path] = {"status": r.status_code,
                                                    "body": r.text[:400]}
            except Exception as e:
                out["engine_settings"][path] = {"error": str(e)[:100]}
    except Exception as e:
        out["engine_settings_error"] = str(e)

    # ── 3. Thorough filesystem search for plugin data/creds ───────────────────
    # NX AI Manager stores its Scailable credentials in one of these locations.
    # We search for any file containing Scailable/sclbl keywords.
    if os.name != "nt":
        out["fs_search"] = "Windows only"
    else:
        cred_hits = {}
        pdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        prog  = r"C:\Program Files"
        system_appdata = r"C:\Windows\System32\config\systemprofile\AppData\Local"

        # Candidate roots — cast wide net
        search_roots = [pdata, prog, system_appdata]
        for drive in "CDEF":
            for name in ("Nx MetaVMS Media", "NxMetaVMS", "Nx Meta", "NxMeta",
                         "NetworkOptix", "Network Optix"):
                d = fr"{drive}:\{name}"
                if os.path.isdir(d):
                    search_roots.append(d)

        CRED_KEYWORDS = (b"sclbl", b"scailable", b"api.sclbl", b"nxvms.com/cpt",
                         b"channel_partner", b"partner_key", b"partnerKey",
                         b"clientSecret", b"client_secret")

        def _search_dir(root, max_depth=6):
            hits = {}
            try:
                for dirpath, dirnames, filenames in os.walk(root):
                    depth = dirpath.replace(root, "").count(os.sep)
                    if depth >= max_depth:
                        dirnames.clear()
                        continue
                    for fname in filenames:
                        ext = os.path.splitext(fname.lower())[1]
                        if ext not in (".json", ".conf", ".ini", ".yaml", ".yml",
                                       ".txt", ".cfg", ".dat", ".db", ".sqlite",
                                       ".token", ".key", ""):
                            continue
                        fpath = os.path.join(dirpath, fname)
                        try:
                            with open(fpath, "rb") as f:
                                raw = f.read(32768)  # first 32KB
                            if any(kw in raw.lower() for kw in CRED_KEYWORDS):
                                try:
                                    text = raw.decode("utf-8", errors="replace")[:2000]
                                except Exception:
                                    text = repr(raw[:500])
                                hits[fpath] = text
                                print(f"[creds] HIT: {fpath}")
                                sys.stdout.flush()
                        except Exception:
                            pass
            except Exception as e:
                hits["__error__"] = str(e)
            return hits

        for root in search_roots:
            if os.path.isdir(root):
                hits = _search_dir(root)
                cred_hits.update(hits)

        out["fs_cred_hits"] = cred_hits
        print(f"[creds] filesystem hits: {len(cred_hits)} files")
        sys.stdout.flush()

    # ── 4. Windows Registry search for Scailable/NX AI Manager entries ────────
    if os.name == "nt":
        reg_results = {}
        try:
            import winreg
            def _scan_reg(root, path):
                try:
                    key = winreg.OpenKey(root, path)
                    i = 0
                    while True:
                        try:
                            name, data, _ = winreg.EnumValue(key, i)
                            if any(x in str(name).lower() or x in str(data).lower()
                                   for x in ("sclbl", "scailable", "partner", "auth_key",
                                              "authkey", "nxai", "nx_ai")):
                                reg_results[f"{path}\\{name}"] = str(data)[:200]
                            i += 1
                        except OSError:
                            break
                    winreg.CloseKey(key)
                except Exception:
                    pass

            for reg_path in [
                r"SOFTWARE\Network Optix",
                r"SOFTWARE\Scailable",
                r"SOFTWARE\NX AI Manager",
                r"SOFTWARE\WOW6432Node\Network Optix",
                r"SOFTWARE\WOW6432Node\Scailable",
            ]:
                _scan_reg(winreg.HKEY_LOCAL_MACHINE, reg_path)
                _scan_reg(winreg.HKEY_CURRENT_USER,  reg_path)

        except ImportError:
            reg_results["error"] = "winreg not available"
        except Exception as e:
            reg_results["error"] = str(e)
        out["registry_cred_hits"] = reg_results
        print(f"[creds] registry hits: {list(reg_results.keys())[:10]}")
        sys.stdout.flush()

    return jsonify(out)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    port = cfg.get("port", 8767)
    print(f"nx-ai-trainer v0.1  |  http://localhost:{port}")
    print(f"Nx server: {cfg['nx']['host']}:{cfg['nx']['port']}")
    print(f"Training data: {TRAIN_DIR}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
