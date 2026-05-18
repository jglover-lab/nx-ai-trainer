/**
 * nx-ai-trainer — frontend
 * Camera selection, frame capture (server-side), training trigger, deploy.
 */

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  cameras: [],
  selectedCamera: null,
  classes: [],          // [{name, count, thumbs: [dataURL]}]
  captureClass: null,   // index in state.classes
  modelReady: false,
  training: false,
  deploying: false,
  trainedWithCurrentData: false,  // true after successful train; reset on new captures
};

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', async () => {
  setStream('lo'); // default to low-res stream
  await checkStatus();
  await loadCameras();
  setInterval(checkStatus, 15_000);
});

// ── Status ────────────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const data = await apiFetch('/api/status');
    setIndicator('status-nx', data.nx, data.nx ? data.nx_version : (data.nx_error || 'unreachable'));
    let sclblLabel = data.scailable ? 'signed in' : 'not signed in';
    if (data.scailable && data.token_expires_in !== undefined) {
      const mins = Math.floor(data.token_expires_in / 60);
      if (data.token_expires_in < 300) sclblLabel = `token expires in ${mins}m — re-sign in`;
    }
    setIndicator('status-sclbl', data.scailable, sclblLabel);
    document.getElementById('sclbl-user').textContent = data.scailable_user || 'AI Manager';
    document.getElementById('btn-login').style.display = data.scailable ? 'none' : '';
    document.getElementById('btn-logout').style.display = data.scailable ? '' : 'none';
    if (data.scailable && state.classes.length === 0) loadModels();

    // Sync class counts from server
    if (data.class_counts) {
      for (const cls of state.classes) {
        if (data.class_counts[cls.name] !== undefined) {
          cls.count = data.class_counts[cls.name];
        }
      }
      renderClasses();
    }
    if (data.training_status === 'trained' && !state.modelReady) {
      state.modelReady = true;
      state.trainedWithCurrentData = true;
      updateDeployBtn();
      if (data.trained_class_names && data.trained_class_names.length) {
        const el = document.getElementById('train-result');
        el.style.display = '';
        const parts = [`Classes: ${data.trained_class_names.join(', ')}`];
        if (data.trained_accuracy) parts.unshift(`Accuracy: ${data.trained_accuracy}%`);
        if (data.trained_model_kb) parts.push(`${data.trained_model_kb}KB`);
        el.textContent = `✓ ${parts.join('  |  ')}`;
        setTrainInfo('Training complete — ready to deploy.', 'ok');
      }
      checkTrainReady();
    }
  } catch {
    setIndicator('status-nx', false, 'unreachable');
    setIndicator('status-sclbl', false, 'unknown');
  }
}

function setIndicator(id, ok, label) {
  const el = document.getElementById(id);
  el.className = `status-item ${ok ? 'ok' : 'error'}`;
  el.querySelector('.status-label').textContent = label;
}

// ── Cameras ───────────────────────────────────────────────────────────────────
function _setCamLabel(text) {
  document.getElementById('camera-select-label').textContent = text;
}

let _deployCamera = null; // tracks selected deploy-camera value

function toggleDeployDropdown() {
  const list = document.getElementById('deploy-camera-list');
  if (list.classList.contains('open')) {
    closeDeployDropdown();
  } else {
    list.classList.add('open');
    document.getElementById('deploy-camera-btn').classList.add('open');
    document.getElementById('deploy-dropdown-overlay').style.display = 'block';
  }
}

function closeDeployDropdown() {
  document.getElementById('deploy-camera-list').classList.remove('open');
  document.getElementById('deploy-camera-btn').classList.remove('open');
  document.getElementById('deploy-dropdown-overlay').style.display = 'none';
}

function _selectDeployCamera(id, name) {
  _deployCamera = id || null;
  document.getElementById('deploy-camera-label').textContent = name;
  document.getElementById('deploy-camera-list').querySelectorAll('.cam-select-opt')
    .forEach(o => o.classList.toggle('selected', o.dataset.id === id));
  closeDeployDropdown();
  updateDeployBtn();
}

function toggleCamDropdown() {
  const list = document.getElementById('camera-select-list');
  if (list.classList.contains('open')) {
    closeCamDropdown();
  } else {
    list.classList.add('open');
    document.getElementById('camera-select-btn').classList.add('open');
    document.getElementById('cam-dropdown-overlay').style.display = 'block';
  }
}

function closeCamDropdown() {
  document.getElementById('camera-select-list').classList.remove('open');
  document.getElementById('camera-select-btn').classList.remove('open');
  document.getElementById('cam-dropdown-overlay').style.display = 'none';
}

function _selectCamera(id, name) {
  document.getElementById('camera-select-list').querySelectorAll('.cam-select-opt')
    .forEach(o => o.classList.toggle('selected', o.dataset.id === id));
  _setCamLabel(name || '— select camera —');
  closeCamDropdown();
  state.selectedCamera = id || null;
  if (id) {
    document.getElementById('stream-img').alt = '';
    startStream();
    if (_videoMode === 'recorded') loadBookmarks();
  } else {
    stopStream();
    const img = document.getElementById('stream-img');
    img.removeAttribute('src');
    img.style.display = 'none';
    document.getElementById('stream-hint').style.display = '';
    document.getElementById('bookmark-select').innerHTML = '<option value="">— select a bookmark —</option>';
  }
  updateCaptureButton();
  updateDeployBtn();
}

async function loadCameras() {
  _setCamLabel('Loading…');
  try {
    const cameras = await apiFetch('/api/cameras');
    state.cameras = cameras;
    const list = document.getElementById('camera-select-list');
    list.innerHTML = '';
    const placeholder = document.createElement('div');
    placeholder.className = 'cam-select-opt';
    placeholder.textContent = '— select camera —';
    placeholder.onclick = () => { _selectCamera('', '— select camera —'); };
    list.appendChild(placeholder);
    cameras.forEach(c => {
      const o = document.createElement('div');
      o.className = 'cam-select-opt';
      o.dataset.id = c.id;
      o.textContent = c.name || c.id;
      o.onclick = () => { _selectCamera(c.id, c.name || c.id); };
      list.appendChild(o);
    });
    _setCamLabel('— select camera —');
    const deployList = document.getElementById('deploy-camera-list');
    deployList.innerHTML = '';
    const dp = document.createElement('div');
    dp.className = 'cam-select-opt';
    dp.textContent = 'Same as selected camera';
    dp.onclick = () => _selectDeployCamera('', 'Same as selected camera');
    deployList.appendChild(dp);
    cameras.forEach(c => {
      const o = document.createElement('div');
      o.className = 'cam-select-opt';
      o.dataset.id = c.id;
      o.textContent = c.name || c.id;
      o.onclick = () => _selectDeployCamera(c.id, c.name || c.id);
      deployList.appendChild(o);
    });
  } catch (e) {
    _setCamLabel(`Error: ${e.message}`);
  }
}

// ── Video mode ─────────────────────────────────────────────────────────────────
let _pollTimer       = null;
let _liveActive      = false;
let _videoMode       = 'live';   // 'live' | 'recorded' | 'bookmark'
let _streamRes       = 'lo';     // 'lo' | 'hi'
let _recordedBlobUrl = null;     // current blob URL for recorded frame (revoked on next fetch)
let _lastLiveImg     = null;     // last fully-loaded live frame Image (for thumbnail extraction)
let _playbackActive  = false;    // true while recorded/bookmark playback is running
let _playbackPosMs   = null;     // current playback wall-clock position (ms)
let _playbackGen     = 0;        // incremented on each startPlayback(); stale chains check this

/** Format a ms timestamp as YYYY-MM-DDTHH:MM in LOCAL time (for datetime-local input) */
function fmtLocal(ms) {
  const d = new Date(ms);
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

function setMode(mode) {
  stopPlayback();
  _videoMode = mode;
  document.getElementById('mode-live').classList.toggle('active', mode === 'live');
  document.getElementById('mode-recorded').classList.toggle('active', mode === 'recorded');
  document.getElementById('mode-bookmark').classList.toggle('active', mode === 'bookmark');
  document.getElementById('recorded-controls').style.display = mode === 'recorded' ? '' : 'none';
  document.getElementById('bookmark-controls').style.display = mode === 'bookmark' ? '' : 'none';
  document.getElementById('playback-controls').style.display = mode === 'live' ? 'none' : '';
  if (mode === 'recorded') {
    if (!document.getElementById('recorded-time').value) {
      // Default to 1 hour ago — much more likely to have recorded footage than "now"
      document.getElementById('recorded-time').value = fmtLocal(Date.now() - 3_600_000);
    }
  }
  if (mode === 'bookmark') {
    loadBookmarks();
  }
  startStream();
}

function setStream(res) {
  _streamRes = res;
  document.getElementById('stream-lo').classList.toggle('active', res === 'lo');
  document.getElementById('stream-hi').classList.toggle('active', res === 'hi');
  refreshFrame();
}

async function loadBookmarks() {
  if (!state.selectedCamera) return;
  const sel = document.getElementById('bookmark-select');
  sel.innerHTML = '<option value="">Loading bookmarks...</option>';
  try {
    const bookmarks = await apiFetch(`/api/bookmarks/${state.selectedCamera}`);
    sel.innerHTML = '<option value="">— select bookmark or set time below —</option>';
    if (bookmarks.error) {
      sel.innerHTML = `<option value="">No bookmarks (${bookmarks.error})</option>`;
      return;
    }
    if (bookmarks.length === 0) {
      sel.innerHTML = '<option value="">No bookmarks for this camera</option>';
      return;
    }
    bookmarks.forEach(b => {
      const o = document.createElement('option');
      o.value = b.startTimeMs;
      const d = new Date(b.startTimeMs);
      o.textContent = `${escHtml(b.name)} — ${d.toLocaleString()}`;
      sel.appendChild(o);
    });
  } catch (e) {
    sel.innerHTML = `<option value="">Error loading bookmarks</option>`;
  }
}

function onBookmarkChange() {
  if (!document.getElementById('bookmark-select').value) return;
  stopPlayback();
  startPlayback(); // auto-start playback from selected bookmark
}

function stepTime(deltaMs) {
  const input = document.getElementById('recorded-time');
  if (!input.value) return;
  const t = new Date(input.value);   // parsed as local time — correct
  document.getElementById('recorded-time').value = fmtLocal(t.getTime() + deltaMs);
  refreshFrame();
}

function getFrameUrl() {
  if (!state.selectedCamera) return null;
  const params = [`res=${_streamRes}`];
  if (_videoMode === 'recorded') {
    const val = document.getElementById('recorded-time').value;
    if (val) params.push(`pos=${new Date(val).getTime()}`);
  } else if (_videoMode === 'bookmark') {
    const val = document.getElementById('bookmark-select').value;
    if (val) params.push(`pos=${val}`); // bookmark value is already a ms timestamp
  }
  return `/api/frame/${state.selectedCamera}?${params.join('&')}`;
}

// Sequence counter so rapid bookmark changes never show a stale frame.
let _refreshSeq = 0;

function refreshFrame() {
  if (_videoMode === 'live' && _liveActive) return; // live handled by chain loader
  // Pause current playback chain while we manually seek; it restarts after the frame loads
  if (_videoMode !== 'live') stopPlayback();
  const url = getFrameUrl();
  if (!url) return;

  const img  = document.getElementById('stream-img');
  const hint = document.getElementById('stream-hint');
  img.style.display = 'block';

  if (_videoMode === 'live') {
    hint.style.display = 'none';
    img.style.opacity = '1';
    img.src = url + '&_t=' + Date.now();
    return;
  }

  // Recorded / bookmark: use fetch() so the bytes are in memory before we
  // assign img.src. The Image() tmp pattern re-requests the URL when
  // img.src = tmp.src is set (Cache-Control: no-store prohibits reuse),
  // causing the stale live frame to remain visible during that second fetch.
  const seq = ++_refreshSeq;
  img.style.opacity = '0.3';
  hint.className = 'stream-loading-label';
  hint.textContent = 'Fetching recorded frame…';
  hint.style.display = '';

  const newUrl = url + '&_t=' + Date.now();

  fetch(newUrl)
    .then(r => {
      if (!r.ok) return r.json().catch(() => ({})).then(d => Promise.reject(new Error(d.error || `HTTP ${r.status}`)));
      return r.blob();
    })
    .then(blob => {
      if (seq !== _refreshSeq) return;   // a newer request superseded this one
      // Revoke previous blob URL to free memory
      if (_recordedBlobUrl) { URL.revokeObjectURL(_recordedBlobUrl); _recordedBlobUrl = null; }
      _recordedBlobUrl = URL.createObjectURL(blob);
      img.src = _recordedBlobUrl;        // already in memory — no second HTTP request
      img.style.opacity = '1';
      hint.style.display = 'none';
      hint.className = '';
      // Update timestamp overlay with the recorded frame's wall-clock time
      const ts = document.getElementById('frame-ts');
      if (ts) {
        const posParam = new URLSearchParams(newUrl.split('?')[1] || '').get('pos');
        ts.style.display = '';
        ts.className = 'recorded';
        ts.textContent = posParam
          ? '◆ REC  ' + new Date(parseInt(posParam)).toLocaleString()
          : '◆ REC';
      }
      // Auto-restart playback from the newly seeked position
      startPlayback();
    })
    .catch(err => {
      if (seq !== _refreshSeq) return;
      img.style.opacity = '1';
      hint.className = '';
      hint.textContent = err.message || 'Frame request failed';
      hint.style.display = '';
    });
}

// ── Live frame chain-loader ────────────────────────────────────────────────────
// Waits for each frame to load before requesting the next one — no queuing.
function _chainLoad() {
  if (!_liveActive || !state.selectedCamera || _videoMode !== 'live') return;
  const url = getFrameUrl() + '&_t=' + Date.now();
  const tmp = new Image();
  const t0 = Date.now();
  tmp.onload = () => {
    if (!_liveActive) return;
    _lastLiveImg = tmp;
    const img = document.getElementById('stream-img');
    img.src = tmp.src;
    img.style.opacity = '1';
    img.style.display = 'block';
    document.getElementById('stream-hint').style.display = 'none';
    // Update timestamp overlay with current wall-clock time (no flash)
    const ts = document.getElementById('frame-ts');
    if (ts) {
      ts.style.display = '';
      ts.textContent = '● LIVE  ' + new Date().toLocaleTimeString();
      ts.className = 'live';
    }
    // Maintain ~5fps max; if frame arrived quickly, wait the remainder
    _pollTimer = setTimeout(_chainLoad, Math.max(0, 200 - (Date.now() - t0)));
  };
  tmp.onerror = () => {
    if (!_liveActive) return;
    _pollTimer = setTimeout(_chainLoad, 1000); // back off on error
  };
  tmp.src = url;
}

// Debounced handler for manual datetime typing
let _recordedTimeDebounce = null;
function onRecordedTimeInput() {
  clearTimeout(_recordedTimeDebounce);
  _recordedTimeDebounce = setTimeout(refreshFrame, 400);
}

// Show server error text inside stream box if a live frame fails to load
document.addEventListener('DOMContentLoaded', () => {
  setMethod('basic');
  document.getElementById('stream-img').addEventListener('error', async () => {
    // Only handle live-mode errors here; recorded errors are handled in refreshFrame()
    if (_videoMode !== 'live') return;
    const url = getFrameUrl();
    if (!url || !state.selectedCamera) return;
    try {
      const r = await fetch(url.split('?')[0] + '?_t=' + Date.now());
      const data = await r.json().catch(() => ({}));
      document.getElementById('stream-hint').textContent = data.error || `Frame error (HTTP ${r.status})`;
    } catch {
      document.getElementById('stream-hint').textContent = 'Frame request failed';
    }
    document.getElementById('stream-hint').style.display = '';
  });
});

function startStream() {
  stopStream();
  // Reset any loading state left over from a previous recorded frame fetch
  const img = document.getElementById('stream-img');
  img.style.opacity = '1';
  document.getElementById('stream-hint').className = '';
  if (!state.selectedCamera) return;
  if (_videoMode === 'live') {
    _liveActive = true;
    _chainLoad();
  } else {
    // Auto-start playback; for bookmark mode startPlayback() returns early if
    // no bookmark is selected yet — onBookmarkChange() will start it instead.
    startPlayback();
  }
}

function stopStream() {
  stopPlayback();
  _liveActive = false;
  if (_pollTimer) { clearTimeout(_pollTimer); _pollTimer = null; }
  if (_recordedBlobUrl) { URL.revokeObjectURL(_recordedBlobUrl); _recordedBlobUrl = null; }
  const ts = document.getElementById('frame-ts');
  if (ts) ts.style.display = 'none';
}

// ── Playback (recorded / bookmark auto-advance) ────────────────────────────────
function togglePlayback() {
  _playbackActive ? stopPlayback() : startPlayback();
}

function startPlayback() {
  let posMs = null;
  if (_videoMode === 'recorded') {
    const val = document.getElementById('recorded-time').value;
    if (val) posMs = new Date(val).getTime();
  } else if (_videoMode === 'bookmark') {
    const val = document.getElementById('bookmark-select').value;
    if (val) posMs = parseInt(val);
  }
  if (!posMs || !state.selectedCamera) return;
  _playbackPosMs = posMs;
  _playbackActive = true;
  const gen = ++_playbackGen;   // new generation — any in-flight chains from prior gens die
  document.getElementById('btn-play').textContent = '⏸ Pause';
  _playbackAdvance(gen);
}

function stopPlayback() {
  _playbackActive = false;
  const btn = document.getElementById('btn-play');
  if (btn) btn.textContent = '▶ Play';
}

function _playbackAdvance(gen) {
  // Stale chain guard: stop if playback was cancelled or a newer chain started
  if (!_playbackActive || gen !== _playbackGen || !state.selectedCamera) {
    if (gen === _playbackGen) stopPlayback();
    return;
  }
  const img    = document.getElementById('stream-img');
  const hint   = document.getElementById('stream-hint');
  const stepMs = parseInt(document.getElementById('playback-step').value) || 5000;
  const posMs  = _playbackPosMs;
  const t0     = Date.now();

  fetch(`/api/frame/${state.selectedCamera}?res=${_streamRes}&pos=${posMs}&_t=${t0}`)
    .then(r => r.ok ? r.blob() : Promise.reject(new Error(`HTTP ${r.status}`)))
    .then(blob => {
      if (!_playbackActive || gen !== _playbackGen) return; // stale — discard
      if (_recordedBlobUrl) { URL.revokeObjectURL(_recordedBlobUrl); _recordedBlobUrl = null; }
      _recordedBlobUrl = URL.createObjectURL(blob);
      img.src = _recordedBlobUrl;
      img.style.opacity = '1';
      img.style.display = 'block';
      hint.style.display = 'none';
      hint.className = '';
      // Update timestamp overlay
      const ts = document.getElementById('frame-ts');
      if (ts) {
        ts.style.display = '';
        ts.className = 'recorded';
        ts.textContent = '◆ REC  ' + new Date(posMs).toLocaleString();
      }
      // Advance position and update the input for recorded mode
      _playbackPosMs += stepMs;
      if (_videoMode === 'recorded') {
        document.getElementById('recorded-time').value = fmtLocal(_playbackPosMs);
      }
      // Rate-limit to ~1 fps: wait whatever is left of 1 second before next fetch
      const elapsed = Date.now() - t0;
      setTimeout(() => _playbackAdvance(gen), Math.max(0, 1000 - elapsed));
    })
    .catch(() => {
      if (_playbackActive && gen === _playbackGen) setTimeout(() => _playbackAdvance(gen), 2000);
    });
}


function updateDeployBtn() {
  const cameraId = _deployCamera || state.selectedCamera;
  const ready = state.modelReady && !state.deploying && !!cameraId;
  document.getElementById('btn-deploy').disabled = !ready;
  document.getElementById('btn-download').disabled = !state.modelReady;
  document.querySelectorAll('.assign-model-btn').forEach(b => { b.disabled = !cameraId; });
}


// ── Classes ───────────────────────────────────────────────────────────────────
function addClass() {
  const input = document.getElementById('new-class-input');
  const name = input.value.trim().replace(/\b\w/g, ch => ch.toUpperCase());
  if (!name) return;
  if (state.classes.find(c => c.name.toLowerCase() === name.toLowerCase())) { input.select(); return; }
  state.classes.push({ name, count: 0, thumbs: [] });
  input.value = '';
  renderClasses();
  setActiveClass(state.classes.length - 1);
  state.modelReady = false;
  document.getElementById('btn-deploy').disabled = true;
  checkTrainReady();
}

async function removeClass(idx) {
  const cls = state.classes[idx];
  if (!cls) return;
  try {
    await apiFetch(`/api/capture/${encodeURIComponent(cls.name)}`, { method: 'DELETE' });
  } catch {}
  state.classes.splice(idx, 1);
  if (state.captureClass >= state.classes.length) state.captureClass = state.classes.length - 1;
  state.modelReady = false;
  document.getElementById('btn-deploy').disabled = true;
  renderClasses();
  updateCaptureButton();
  checkTrainReady();
}

function setActiveClass(idx) {
  state.captureClass = idx;
  renderClasses();
  updateCaptureButton();
}

function renderClasses() {
  const container = document.getElementById('class-list');
  container.innerHTML = '';
  state.classes.forEach((cls, i) => {
    const isActive = state.captureClass === i;
    const card = document.createElement('div');
    card.className = `class-card${isActive ? ' active' : ''}`;
    card.innerHTML = `
      <div class="class-card-header">
        <span class="class-name">${escHtml(cls.name)}</span>
        <span>
          <span class="class-count">${cls.count} frames</span>
          <button class="btn-small" onclick="removeClass(${i})" style="margin-left:0.4rem" title="Delete class">✕</button>
        </span>
      </div>
      <div class="thumb-strip" id="thumbs-${i}">
        ${cls.thumbs.slice(-12).map(s => `<img class="thumb" src="${s}">`).join('')}
        ${cls.count > 12 ? `<span class="class-count" style="line-height:32px;padding-left:3px">+${cls.count - 12} more</span>` : ''}
      </div>
      <button class="btn-small" onclick="setActiveClass(${i})" style="width:100%;text-align:center">
        ${isActive ? '● Capturing to this class' : '▷ Capture here'}
      </button>
    `;
    container.appendChild(card);
  });

  // Class tabs in camera panel
  document.getElementById('class-tabs').innerHTML = state.classes.map((cls, i) => `
    <button class="class-tab${state.captureClass === i ? ' active' : ''}" onclick="setActiveClass(${i})">
      ${escHtml(cls.name)}
    </button>
  `).join('');

  const nameEl = document.getElementById('capture-class-name');
  nameEl.textContent = (state.captureClass !== null && state.classes[state.captureClass])
    ? state.classes[state.captureClass].name
    : '—';
}

function updateCaptureButton() {
  document.getElementById('btn-capture').disabled = !(
    state.selectedCamera && state.captureClass !== null && state.classes.length > 0 && !state.training
  );
}

// ── Capture ───────────────────────────────────────────────────────────────────
let _captureHoldTimer = null;
let _capturing = false;
let _captureHeld = false;

function startCaptureHold(e) {
  if (e) e.preventDefault();
  _captureHeld = true;
  captureFrame(); // one immediate capture on press
  // Only start repeating after 400ms — prevents double-fire on a quick click
  _captureHoldTimer = setTimeout(function loop() {
    if (!_captureHeld) return;
    if (!_capturing) captureFrame();
    _captureHoldTimer = setTimeout(loop, 300);
  }, 400);
}

function stopCaptureHold() {
  _captureHeld = false;
  if (_captureHoldTimer) { clearTimeout(_captureHoldTimer); _captureHoldTimer = null; }
}

async function captureFrame() {
  if (_capturing || state.captureClass === null || !state.selectedCamera) return;
  _capturing = true;

  try {
    const cls = state.classes[state.captureClass];

    // Always grab the displayed frame via canvas for the thumbnail strip.
    // In live mode the canvas data also becomes the payload sent to the server
    // (avoids a round-trip). In recorded/bookmark mode we still send camera_id
    // + pos so the server fetches the exact historical timestamp — but we use
    // the already-displayed blob URL for the local thumbnail.
    let frame_b64 = null;
    // In live mode try canvas extraction to avoid a server round-trip for the frame payload.
    // Use _lastLiveImg (guaranteed complete) rather than stream-img (mid-swap is incomplete).
    const srcImg = (_videoMode === 'live' && _lastLiveImg) ? _lastLiveImg : document.getElementById('stream-img');
    if (_videoMode === 'live' && srcImg.complete && srcImg.naturalWidth > 0) {
      try {
        const canvas = document.createElement('canvas');
        canvas.width  = srcImg.naturalWidth;
        canvas.height = srcImg.naturalHeight;
        canvas.getContext('2d').drawImage(srcImg, 0, 0);
        frame_b64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
      } catch {}
    }

    const body = { class_name: cls.name };
    if (frame_b64) {
      body.frame_b64 = frame_b64;
    } else {
      // Server fetches directly from Nx at the requested timestamp
      body.camera_id = state.selectedCamera;
      body.res = _streamRes;
      if (_videoMode === 'recorded') {
        const val = document.getElementById('recorded-time').value;
        if (val) body.pos = new Date(val).getTime();
      } else if (_videoMode === 'bookmark') {
        const val = document.getElementById('bookmark-select').value;
        if (val) body.pos = parseInt(val);
      }
    }

    const result = await apiFetch('/api/capture', { method: 'POST', body: JSON.stringify(body) });
    cls.count = result.count;
    if (result.thumb) cls.thumbs.push(`data:image/jpeg;base64,${result.thumb}`);

    state.trainedWithCurrentData = false;
    state.modelReady = false;
    document.getElementById('btn-deploy').disabled = true;
    renderClasses();
    checkTrainReady();
  } catch (e) {
    setTrainInfo(`Capture failed: ${friendlyError(e.message)}`, 'error');
    stopCaptureHold();
  } finally {
    _capturing = false;
    updateCaptureButton();
  }
}


// ── Training ──────────────────────────────────────────────────────────────────
const METHOD_HINTS = {
  basic:     'Fast. Logistic regression on raw pixels. Best for large scene changes (door open/closed, lights on/off). No extra dependencies.',
  cnn:       'Slower (3-5 min on CPU). Small convolutional neural net trained from scratch. Better for moderate visual differences. Requires: torch.',
  mobilenet: 'Slowest (~1-2 min). Fine-tunes MobileNetV2 pretrained on ImageNet. Best accuracy for subtle differences. Requires: torch torchvision.',
};

let _trainMethod = 'basic';
const _methodLabels = { basic: 'Basic — Scene Change', cnn: 'CNN — Small Neural Net', mobilenet: 'MobileNetV2 — Transfer Learning' };

function toggleMethodDropdown() {
  const list = document.getElementById('method-select-list');
  if (list.classList.contains('open')) {
    closeMethodDropdown();
  } else {
    list.classList.add('open');
    document.getElementById('method-select-btn').classList.add('open');
    document.getElementById('method-dropdown-overlay').style.display = 'block';
  }
}

function closeMethodDropdown() {
  document.getElementById('method-select-list').classList.remove('open');
  document.getElementById('method-select-btn').classList.remove('open');
  document.getElementById('method-dropdown-overlay').style.display = 'none';
}

function setMethod(m) {
  _trainMethod = m;
  document.getElementById('method-select-label').textContent = _methodLabels[m] || m;
  document.getElementById('method-select-list').querySelectorAll('.cam-select-opt')
    .forEach((o, i) => o.classList.toggle('selected', ['basic','cnn','mobilenet'][i] === m));
  closeMethodDropdown();
  document.getElementById('method-hint').textContent = METHOD_HINTS[m] || '';
  state.trainedWithCurrentData = false;
  checkTrainReady();
}

function onMethodChange() { setMethod(_trainMethod); }

function checkTrainReady() {
  const minSamples = 5;
  const ready = state.classes.length >= 2 && state.classes.every(c => c.count >= minSamples);
  const alreadyTrained = ready && state.trainedWithCurrentData;
  document.getElementById('btn-train').disabled = !ready || state.training || alreadyTrained;

  if (state.training) return;

  const info = document.getElementById('train-info');
  if (alreadyTrained) {
    info.textContent = 'Model trained — capture new frames to retrain.';
  } else if (state.classes.length < 2) {
    info.textContent = 'Add at least 2 classes with 5+ frames each.';
  } else {
    const needs = state.classes.filter(c => c.count < minSamples).map(c => `"${c.name}" (${minSamples - c.count} more)`);
    if (needs.length > 0) {
      info.textContent = `Need more frames: ${needs.join(', ')}.`;
    } else {
      const total = state.classes.reduce((s, c) => s + c.count, 0);
      info.textContent = `${state.classes.length} classes, ${total} frames total — ready to train.`;
    }
  }
}

async function trainModel() {
  state.training = true;
  const btn = document.getElementById('btn-train');
  const trainMethod = _trainMethod;
  const slowMethod = trainMethod === 'cnn' || trainMethod === 'mobilenet';
  btn.disabled = true;
  btn.textContent = 'Training...';
  document.getElementById('train-progress').style.display = '';
  setTrainFill(10);
  setTrainInfo(slowMethod ? 'Training — CNN typically takes 3-5 min on CPU…' : 'Sending training request...');
  document.getElementById('train-result').style.display = 'none';
  state.modelReady = false;
  document.getElementById('btn-deploy').disabled = true;
  updateCaptureButton();

  // Animate progress while waiting (server-side training takes a few seconds)
  let pct = 10;
  const timer = setInterval(() => {
    if (pct < 85) { pct += 5; setTrainFill(pct); }
  }, 400);

  try {
    const result = await apiFetch('/api/train', {
      method: 'POST',
      body: JSON.stringify({ method: trainMethod }),
    });
    clearInterval(timer);
    setTrainFill(100);
    state.modelReady = true;
    state.trainedWithCurrentData = true;
    updateDeployBtn();
    const el = document.getElementById('train-result');
    el.style.display = '';
    el.textContent = `✓ [${result.method}] Accuracy: ${result.accuracy}%  |  ${result.n_samples} samples  |  ${result.model_size_kb}KB  |  Classes: ${result.class_names.join(', ')}`;
    setTrainInfo('Training complete — ready to deploy.', 'ok');
  } catch (e) {
    clearInterval(timer);
    setTrainFill(0);
    const el = document.getElementById('train-result');
    el.style.display = '';
    el.textContent = `✗ ${friendlyError(e.message)}`;
    setTrainInfo('Training failed.', 'error');
  } finally {
    state.training = false;
    btn.disabled = false;
    btn.textContent = '▶ Train Model';
    updateCaptureButton();
    checkTrainReady();
  }
}

function setTrainFill(pct) {
  document.getElementById('train-fill').style.width = `${pct}%`;
}

function setTrainInfo(msg, type) {
  const el = document.getElementById('train-info');
  el.textContent = msg;
  el.style.color = type === 'ok' ? 'var(--green)' : type === 'error' ? 'var(--red)' : 'var(--muted)';
}

// ── Deploy ────────────────────────────────────────────────────────────────────
async function deployModel() {
  if (!state.modelReady || state.deploying) return;
  state.deploying = true;
  const btn = document.getElementById('btn-deploy');
  btn.disabled = true;
  showDeployStatus('info', 'Uploading model to Nx AI Manager…');

  const modelName = document.getElementById('model-name').value.trim() || 'nx-ai-trainer model';
  const cameraId = _deployCamera || state.selectedCamera;

  if (!cameraId) {
    showDeployStatus('error', '✗ No camera selected');
    btn.disabled = false;
    btn.textContent = '↑ Upload & Assign';
    state.deploying = false;
    return;
  }

  // Poll /api/deploy/status while the POST is in-flight so the user sees progress
  const phaseLabels = {
    uploading:  'Uploading model to Nx AI Manager…',
    processing: null,  // use detail from server
    assigning:  'Assigning model to camera…',
  };
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'btn-small';
  cancelBtn.textContent = '✕ Cancel';
  cancelBtn.style.marginLeft = '0.5rem';
  cancelBtn.onclick = async () => {
    await apiFetch('/api/deploy/cancel', { method: 'POST' });
    clearInterval(statusPoller);
    state.deploying = false;
    btn.textContent = '↑ Upload & Assign';
    cancelBtn.remove();
    updateDeployBtn();
    showDeployStatus('info', 'Deploy cancelled.');
  };
  btn.parentElement.appendChild(cancelBtn);

  const statusPoller = setInterval(async () => {
    try {
      const s = await apiFetch('/api/deploy/status');
      if (s.phase === 'processing' || s.phase === 'uploading' || s.phase === 'assigning') {
        const label = phaseLabels[s.phase] || s.detail || s.phase;
        btn.textContent = s.phase === 'processing' ? '⏳ Compiling…' : '↑ Upload & Assign';
        showDeployStatus('info', s.detail || label);
      }
    } catch {}
  }, 2000);

  try {
    const result = await apiFetch('/api/deploy', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName, camera_id: cameraId }),
    });

    clearInterval(statusPoller);
    cancelBtn.remove();
    if (result.ok) {
      showDeployStatus('ok', `✓ Deployed! Model: ${result.model_uuid.slice(0, 8)}… | Classes: ${result.class_names.join(', ')}`);
      await loadModels();
    } else if (result.warning) {
      showDeployStatus('error', `⚠ ${result.warning} — ${result.assignment_error || ''}`);
    }
  } catch (e) {
    clearInterval(statusPoller);
    cancelBtn.remove();
    showDeployStatus('error', `✗ ${friendlyError(e.message)}`);
  } finally {
    state.deploying = false;
    btn.textContent = '↑ Upload & Assign';
    updateDeployBtn();
  }
}

function showDeployStatus(type, msg) {
  const el = document.getElementById('deploy-status');
  el.style.display = '';
  el.className = `deploy-status ${type}`;
  el.textContent = msg;
}

function downloadModel() {
  window.location.href = '/api/model/download';
}

async function manualAssign() {
  const uuid = (document.getElementById('manual-uuid').value || '').trim();
  const cameraId = _deployCamera || state.selectedCamera;
  const statusEl = document.getElementById('assign-status');

  if (!uuid) {
    statusEl.style.display = '';
    statusEl.className = 'deploy-status error';
    statusEl.textContent = '✗ Enter a model UUID';
    return;
  }
  if (!cameraId) {
    statusEl.style.display = '';
    statusEl.className = 'deploy-status error';
    statusEl.textContent = '✗ No camera selected';
    return;
  }

  statusEl.style.display = '';
  statusEl.className = 'deploy-status info';
  statusEl.textContent = 'Assigning…';

  try {
    const result = await apiFetch('/api/assign', {
      method: 'POST',
      body: JSON.stringify({ model_uuid: uuid, camera_id: cameraId }),
    });
    if (result.ok) {
      statusEl.className = 'deploy-status ok';
      statusEl.textContent = `✓ Assigned ${uuid.slice(0, 8)}… to camera`;
    } else {
      statusEl.className = 'deploy-status error';
      statusEl.textContent = `⚠ ${result.warning || result.error || 'Assignment failed'}`;
    }
  } catch (e) {
    statusEl.className = 'deploy-status error';
    statusEl.textContent = `✗ ${e.message}`;
  }
}

// ── Reset ─────────────────────────────────────────────────────────────────────
async function resetAll() {
  if (!confirm('Delete all training data and start over?')) return;
  try {
    await apiFetch('/api/capture/reset', { method: 'POST' });
    state.classes = [];
    state.captureClass = null;
    state.modelReady = false;
    state.trainedWithCurrentData = false;
    renderClasses();
    updateCaptureButton();
    checkTrainReady();
    document.getElementById('btn-deploy').disabled = true;
    document.getElementById('train-result').style.display = 'none';
    document.getElementById('deploy-status').style.display = 'none';
    setTrainFill(0);
  } catch (e) {
    setTrainInfo(`Reset failed: ${e.message}`, 'error');
  }
}

// ── Models list ───────────────────────────────────────────────────────────────
async function loadModels() {
  const container = document.getElementById('models-list');
  try {
    const models = await apiFetch('/api/models');
    if (!Array.isArray(models) || models.length === 0) {
      container.innerHTML = '<span style="color:var(--muted)">No models yet.</span>';
      return;
    }
    const hasCam = !!(_deployCamera || state.selectedCamera);
    container.innerHTML = models.map(m => {
      const uuid = m.UUID || m.uuid || '';
      const name = m.Name || m.name || 'Unnamed';
      const status = m.Status || (m.Code && m.Code.Status) || '';
      const statusColor = status === 'ok' ? 'var(--green)' : status === 'processing' ? 'var(--muted)' : status ? '#e8a020' : 'var(--muted)';
      return `<div class="model-item">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1" title="${escHtml(uuid)}">${escHtml(name)}</span>
        <span style="display:flex;gap:0.4rem;align-items:center;flex-shrink:0">
          ${status ? `<span style="font-size:0.68rem;color:${statusColor}">${escHtml(status)}</span>` : ''}
          <span class="model-uuid">${uuid.slice(0, 8)}…</span>
          <button class="btn-small btn-primary assign-model-btn" ${hasCam ? '' : 'disabled'} onclick="assignModelToCamera('${escHtml(uuid)}')" title="Assign this model to the selected camera">Assign</button>
        </span>
      </div>`;
    }).join('');
    updateDeployBtn();
  } catch (e) {
    container.innerHTML = `<span style="color:var(--muted)">${e.message}</span>`;
  }
}

function useModel(uuid) {
  document.getElementById('manual-uuid').value = uuid;
  document.getElementById('manual-uuid').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function assignModelToCamera(uuid) {
  const cameraId = _deployCamera || state.selectedCamera;
  if (!cameraId) return;
  const statusEl = document.getElementById('assign-status');
  statusEl.style.display = '';
  statusEl.className = 'deploy-status info';
  statusEl.textContent = 'Assigning…';
  try {
    const result = await apiFetch('/api/assign', {
      method: 'POST',
      body: JSON.stringify({ model_uuid: uuid, camera_id: cameraId }),
    });
    statusEl.className = 'deploy-status ok';
    statusEl.textContent = `✓ Assigned ${uuid.slice(0, 8)}… to camera`;
  } catch (e) {
    statusEl.className = 'deploy-status error';
    statusEl.textContent = `✗ ${e.message}`;
  }
}

// ── Auth ──────────────────────────────────────────────────────────────────────
async function signOut() {
  try { await apiFetch('/auth/logout', { method: 'POST' }); } catch {}
  await checkStatus();
}

function signIn() {
  window.open('/auth/login/sso', '_blank', 'width=600,height=700');
}

// ── Utils ─────────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const resp = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw Object.assign(new Error(data.error || `HTTP ${resp.status}`), data);
  return data;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function friendlyError(msg) {
  if (!msg) return 'Unknown error';
  const m = String(msg);
  if (m.includes('Cannot reach Nx server') || m.includes('check host/port'))
    return m;  // already friendly from server
  if (m.includes('ConnectionError') || m.includes('Connection refused') || m.includes('ECONNREFUSED'))
    return 'Cannot reach Nx server — check host/port in Settings (⚙)';
  if (m.includes('timed out') || m.includes('Timeout') || m.includes('ETIMEDOUT'))
    return 'Request timed out — server may be busy or unreachable';
  if (m.includes('401') || m.includes('Unauthorized'))
    return 'Session expired — sign out and sign in again';
  if (m.includes('403') || m.includes('Forbidden'))
    return 'Access denied (403) — check your Nx AI Manager permissions';
  if (m.includes('PyTorch') || m.includes('pip install torch'))
    return m + ' — run the pip install command in the server terminal';
  if (m.includes('No trained model'))
    return 'No trained model — complete training in Step 3 first';
  if (m.includes('Not authenticated'))
    return 'Not signed in to Nx AI Manager — click Sign In';
  return m;
}

// ── Settings modal ────────────────────────────────────────────────────────────
async function openSettings() {
  const overlay = document.getElementById('settings-overlay');
  overlay.style.display = 'flex';
  setCfgStatus('');
  try {
    const cfg = await apiFetch('/api/config');
    document.getElementById('cfg-host').value = cfg.nx?.host || '';
    document.getElementById('cfg-port').value = cfg.nx?.port || '';
    document.getElementById('cfg-user').value = cfg.nx?.username || '';
    document.getElementById('cfg-pass').value = '';
    document.getElementById('cfg-pass').placeholder = '(unchanged)';
  } catch(e) {
    setCfgStatus('Could not load config: ' + e.message, true);
  }
}

function closeSettings() {
  document.getElementById('settings-overlay').style.display = 'none';
}

async function saveSettings() {
  const pass = document.getElementById('cfg-pass').value;
  const nx = {
    host:     document.getElementById('cfg-host').value.trim(),
    port:     parseInt(document.getElementById('cfg-port').value) || 7001,
    username: document.getElementById('cfg-user').value.trim(),
  };
  if (pass) nx.password = pass;
  const body = { nx };
  setCfgStatus('Saving…');
  try {
    await apiFetch('/api/config', { method: 'POST', body: JSON.stringify(body) });
    setCfgStatus('Saved.', false);
    checkStatus();
  } catch(e) {
    setCfgStatus('Save failed: ' + e.message, true);
  }
}

async function testConnection() {
  setCfgStatus('Testing…');
  try {
    const s = await apiFetch('/api/status');
    if (s.nx) {
      setCfgStatus('✓ Connected to Nx server.', false);
    } else {
      setCfgStatus('✗ Could not reach Nx server.', true);
    }
  } catch(e) {
    setCfgStatus('✗ ' + e.message, true);
  }
}

function setCfgStatus(msg, isError) {
  const el = document.getElementById('cfg-status');
  el.textContent = msg;
  el.style.color = isError ? 'var(--red)' : isError === false ? 'var(--green)' : 'var(--muted)';
}
