// ── Config ──────────────────────────────────────────────────────────────────
// When deployed: replace with your Render backend URL
// e.g. 'https://emotion-recognition-api.onrender.com'
const API_URL = 'http://localhost:8000';

const CAPTURE_INTERVAL_MS = 300;  // ~3 fps sent to backend
const JPEG_QUALITY = 0.75;

// ── Emotion metadata ─────────────────────────────────────────────────────────
const EMOTIONS = {
  angry:    { emoji: '😠', color: '#ff4757' },
  disgust:  { emoji: '🤢', color: '#2ed573' },
  fear:     { emoji: '😨', color: '#a29bfe' },
  happy:    { emoji: '😊', color: '#ffd32a' },
  neutral:  { emoji: '😐', color: '#95afc0' },
  sad:      { emoji: '😢', color: '#74b9ff' },
  surprise: { emoji: '😲', color: '#fd9644' },
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const video         = document.getElementById('video');
const overlay       = document.getElementById('overlay');
const captureCanvas = document.getElementById('capture');
const startBtn      = document.getElementById('startBtn');
const stopBtn       = document.getElementById('stopBtn');
const statusBadge   = document.getElementById('statusBadge');
const faceCountEl   = document.getElementById('faceCount');
const emotionDisplay= document.getElementById('emotionDisplay');
const fpsDisplay    = document.getElementById('fpsDisplay');

const overlayCtx  = overlay.getContext('2d');
const captureCtx  = captureCanvas.getContext('2d');

// ── State ────────────────────────────────────────────────────────────────────
let stream       = null;
let captureTimer = null;
let lastFrameTs  = null;
let inflight     = false;   // prevent concurrent requests

// ── Camera lifecycle ─────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();

    video.addEventListener('loadedmetadata', syncCanvasSizes, { once: true });

    startBtn.disabled = true;
    stopBtn.disabled  = false;
    setStatus('detecting', '● Detecting…');

    captureTimer = setInterval(processFrame, CAPTURE_INTERVAL_MS);
  } catch (err) {
    setStatus('error', '● Camera error');
    console.error('Camera error:', err);
  }
}

function stopCamera() {
  clearInterval(captureTimer);
  captureTimer = null;

  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  setStatus('idle', '● Idle');
  faceCountEl.textContent = '—';
  fpsDisplay.textContent  = '';
  showPlaceholder();

  startBtn.disabled = false;
  stopBtn.disabled  = true;
}

function syncCanvasSizes() {
  // Match canvas pixel dimensions to the camera's natural resolution
  overlay.width        = video.videoWidth;
  overlay.height       = video.videoHeight;
  captureCanvas.width  = video.videoWidth;
  captureCanvas.height = video.videoHeight;
}

// ── Frame processing ─────────────────────────────────────────────────────────
async function processFrame() {
  if (!stream || video.readyState < 2 || inflight) return;

  captureCtx.drawImage(video, 0, 0);
  const base64 = captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1];

  inflight = true;
  try {
    const res  = await fetch(`${API_URL}/predict`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ image: base64 }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { faces } = await res.json();

    renderResults(faces);
    tickFPS();
    setStatus('detecting', '● Detecting…');
  } catch (err) {
    setStatus('error', '● API error');
    console.error('Predict error:', err);
  } finally {
    inflight = false;
  }
}

function tickFPS() {
  const now = Date.now();
  if (lastFrameTs) {
    fpsDisplay.textContent = `${Math.round(1000 / (now - lastFrameTs))} fps`;
  }
  lastFrameTs = now;
}

// ── Rendering ────────────────────────────────────────────────────────────────
function renderResults(faces) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (!faces || faces.length === 0) {
    faceCountEl.textContent = 'No faces';
    showPlaceholder();
    return;
  }

  faceCountEl.textContent = faces.length === 1 ? '1 face' : `${faces.length} faces`;

  faces.forEach(face => drawFaceBox(face));
  updateEmotionPanel(faces[0]);
}

function drawFaceBox(face) {
  const [x1, y1, x2, y2] = face.box;
  const meta  = EMOTIONS[face.emotion] || { emoji: '', color: '#7c6ef4' };
  const pct   = Math.round(face.confidence * 100);
  const label = `${meta.emoji} ${face.emotion} ${pct}%`;

  // Glowing bounding box
  overlayCtx.shadowColor = meta.color;
  overlayCtx.shadowBlur  = 10;
  overlayCtx.strokeStyle = meta.color;
  overlayCtx.lineWidth   = 2.5;
  overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  overlayCtx.shadowBlur  = 0;

  // Label pill
  overlayCtx.font = 'bold 14px Inter, sans-serif';
  const textW   = overlayCtx.measureText(label).width;
  const padX    = 8;
  const padY    = 5;
  const labelH  = 24;
  const lx      = x1;
  const ly      = Math.max(0, y1 - labelH - 4);

  overlayCtx.fillStyle = meta.color;
  overlayCtx.fillRect(lx, ly, textW + padX * 2, labelH);

  overlayCtx.fillStyle = '#000';
  overlayCtx.fillText(label, lx + padX, ly + labelH - padY);
}

function updateEmotionPanel(face) {
  const meta   = EMOTIONS[face.emotion] || { emoji: '❓', color: '#7c6ef4' };
  const sorted = Object.entries(face.all_emotions).sort((a, b) => b[1] - a[1]);

  emotionDisplay.innerHTML = `
    <div class="dominant">
      <span class="dominant-emoji">${meta.emoji}</span>
      <div class="dominant-info">
        <span class="dominant-name">${face.emotion}</span>
        <span class="dominant-conf">${Math.round(face.confidence * 100)}% confident</span>
      </div>
    </div>

    <div class="emotion-bars">
      ${sorted.map(([emotion, prob]) => {
        const m = EMOTIONS[emotion] || { emoji: '', color: '#7c6ef4' };
        const pct = Math.round(prob * 100);
        return `
          <div class="emotion-bar ${emotion === face.emotion ? 'active' : ''}">
            <div class="bar-header">
              <span class="bar-emoji">${m.emoji}</span>
              <span class="bar-label">${emotion}</span>
              <span class="bar-pct">${pct}%</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct}%; background:${m.color}"></div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function showPlaceholder() {
  emotionDisplay.innerHTML = `
    <div class="placeholder">
      <div class="placeholder-icon">👤</div>
      <p>Point your camera at a face<br />to begin detection</p>
    </div>
  `;
}

// ── Status helper ─────────────────────────────────────────────────────────────
function setStatus(type, text) {
  statusBadge.textContent = text;
  statusBadge.className   = `status-badge status-${type}`;
}

// ── Wire up buttons ───────────────────────────────────────────────────────────
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
