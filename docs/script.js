// ── Config ──────────────────────────────────────────────────────────────────
const API_URL = 'https://emotion-recognition-7qal.onrender.com';

const CAPTURE_INTERVAL_MS = 300;
const JPEG_QUALITY  = 0.6;
const CAPTURE_W     = 320;   // send small frames to keep latency low
const CAPTURE_H     = 240;
const FETCH_TIMEOUT = 8000;  // abort if backend takes > 8s

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
const video          = document.getElementById('video');
const overlay        = document.getElementById('overlay');
const captureCanvas  = document.getElementById('capture');
const startBtn       = document.getElementById('startBtn');
const stopBtn        = document.getElementById('stopBtn');
const statusBadge    = document.getElementById('statusBadge');
const faceCountEl    = document.getElementById('faceCount');
const emotionDisplay = document.getElementById('emotionDisplay');
const fpsDisplay     = document.getElementById('fpsDisplay');

const overlayCtx = overlay.getContext('2d');
const captureCtx = captureCanvas.getContext('2d');

// Capture canvas is always small (fast to encode + send)
captureCanvas.width  = CAPTURE_W;
captureCanvas.height = CAPTURE_H;

// ── State ────────────────────────────────────────────────────────────────────
let stream       = null;
let captureTimer = null;
let lastFrameTs  = null;
let inflight     = false;

// ── Camera lifecycle ─────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();

    video.addEventListener('loadedmetadata', () => {
      // Overlay matches the camera's natural resolution for sharp box drawing
      overlay.width  = video.videoWidth;
      overlay.height = video.videoHeight;
    }, { once: true });

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

// ── Frame processing ─────────────────────────────────────────────────────────
async function processFrame() {
  if (!stream || video.readyState < 2 || inflight) return;

  // Draw video into small capture canvas (320×240) to reduce payload size
  captureCtx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
  const base64 = captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1];

  inflight = true;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ image: base64 }),
      signal : controller.signal,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { faces } = await res.json();

    renderResults(faces);
    tickFPS();
    setStatus('detecting', '● Detecting…');
  } catch (err) {
    if (err.name === 'AbortError') {
      setStatus('error', '● Timeout');
    } else {
      setStatus('error', '● API error');
      console.error('Predict error:', err);
    }
  } finally {
    clearTimeout(timer);
    inflight = false;
  }
}

function tickFPS() {
  const now = Date.now();
  if (lastFrameTs) {
    const fps = (1000 / (now - lastFrameTs)).toFixed(1);
    fpsDisplay.textContent = `${fps} fps`;
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

  // Scale boxes from capture resolution (320×240) → overlay resolution
  const scaleX = overlay.width  / CAPTURE_W;
  const scaleY = overlay.height / CAPTURE_H;

  faces.forEach(face => drawFaceBox(face, scaleX, scaleY));
  updateEmotionPanel(faces[0]);
}

function drawFaceBox(face, scaleX, scaleY) {
  const [bx1, by1, bx2, by2] = face.box;
  const x1 = bx1 * scaleX;
  const y1 = by1 * scaleY;
  const x2 = bx2 * scaleX;
  const y2 = by2 * scaleY;

  const meta  = EMOTIONS[face.emotion] || { emoji: '', color: '#7c6ef4' };
  const pct   = Math.round(face.confidence * 100);
  const label = `${meta.emoji} ${face.emotion} ${pct}%`;

  overlayCtx.shadowColor = meta.color;
  overlayCtx.shadowBlur  = 10;
  overlayCtx.strokeStyle = meta.color;
  overlayCtx.lineWidth   = 2.5;
  overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  overlayCtx.shadowBlur  = 0;

  overlayCtx.font = 'bold 14px Inter, sans-serif';
  const textW  = overlayCtx.measureText(label).width;
  const padX   = 8;
  const labelH = 24;
  const lx     = x1;
  const ly     = Math.max(0, y1 - labelH - 4);

  overlayCtx.fillStyle = meta.color;
  overlayCtx.fillRect(lx, ly, textW + padX * 2, labelH);

  overlayCtx.fillStyle = '#000';
  overlayCtx.fillText(label, lx + padX, ly + labelH - 5);
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
