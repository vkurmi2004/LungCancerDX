/* ==========================================================================
   LungCancerDX – Main Application JavaScript
   Full-featured diagnostic interface
   ========================================================================== */

'use strict';

// ─── Configuration ────────────────────────────────────────────────────────────
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : window.location.origin;

const MODEL_INFO = {
  ResNet50:       { desc: 'Deep residual learning, 50 layers. Excellent at capturing hierarchical CT features.',      icon: '🏗️', color: '#6366f1', acc: '94.2%', params: '25.6M' },
  EfficientNetB0: { desc: 'Scaled-efficiency CNN. Best accuracy-to-computation ratio for resource-limited systems.',  icon: '⚡', color: '#10b981', acc: '95.1%', params: '5.3M'  },
  DenseNet121:    { desc: 'Densely connected layers maximize feature reuse and gradient flow.',                        icon: '🕸️', color: '#f59e0b', acc: '93.8%', params: '7.9M'  },
  MobileNetV3:    { desc: 'Lightweight mobile-optimized CNN for fast real-time inference.',                            icon: '📱', color: '#8b5cf6', acc: '91.6%', params: '2.5M'  },
  VGG16:          { desc: 'Classic deep CNN with very uniform architecture, strong baseline performance.',             icon: '🏛️', color: '#ec4899', acc: '92.4%', params: '138M'  },
};

// ─── DOM References ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const el = {
  dropZone:        $('dropZone'),
  fileInput:       $('fileInput'),
  previewBox:      $('previewBox'),
  previewImg:      $('previewImg'),
  previewMeta:     $('previewMeta'),
  clearBtn:        $('clearBtn'),
  analyzeBtn:      $('analyzeBtn'),
  analyzeBtnText:  $('analyzeBtnText'),
  analyzeBtnSpinner: $('analyzeBtnSpinner'),
  webcamBtn:       $('webcamBtn'),
  webcamVideo:     $('webcamVideo'),
  captureBtn:      $('captureBtn'),
  sampleBtn:       $('sampleBtn'),
  placeholderState: $('placeholderState'),
  resultContent:   $('resultContent'),
  resultCard:      $('resultCard'),
  predLabel:       $('predLabel'),
  predConf:        $('predConf'),
  riskBadge:       $('riskBadge'),
  probBars:        $('probBars'),
  modelBreakdownBody: $('modelBreakdownBody'),
  recommendationText: $('recommendationText'),
  metaRow:         $('metaRow'),
  resetBtn:        $('resetBtn'),
  downloadReportBtn: $('downloadReportBtn'),
  modelCards:      $('modelCards'),
  statusDot:       $('statusDot'),
  statusText:      $('statusText'),
  toast:           $('toast'),
  navToggle:       $('navToggle'),
  navLinks:        $('navLinks'),
  statTotal:       $('statTotal'),
  benignCount:     $('benignCount'),
  malignantCount:  $('malignantCount'),
  normalCount:     $('normalCount'),
};

// ─── State ────────────────────────────────────────────────────────────────────
let currentFile = null;
let lastResult  = null;
let webcamStream = null;
let healthData  = null;
let toastTimer  = null;

// ─── Toast Notifications ──────────────────────────────────────────────────────
function showToast(msg, type = 'info', duration = 3500) {
  if (toastTimer) clearTimeout(toastTimer);
  const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
  el.toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span> ${msg}`;
  el.toast.className = `toast toast--${type} toast--show`;
  toastTimer = setTimeout(() => el.toast.classList.remove('toast--show'), duration);
}

// ─── API Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    healthData = await res.json();
    el.statusDot.classList.add('status-dot--online');
    el.statusDot.classList.remove('status-dot--offline');
    el.statusText.textContent = 'API Online';
    if (!healthData.any_model_ready) {
      showToast('⚠️ No trained models found – running in demo mode (untrained predictions)', 'warning', 7000);
    }
    return true;
  } catch {
    el.statusDot.classList.add('status-dot--offline');
    el.statusDot.classList.remove('status-dot--online');
    el.statusText.textContent = 'API Offline';
    showToast('Backend API is offline. Please start the server → uvicorn backend.main:app', 'error', 8000);
    return false;
  }
}

// ─── Dataset Stats ────────────────────────────────────────────────────────────
async function loadDatasetStats() {
  try {
    const res  = await fetch(`${API_BASE}/api/dataset/stats`);
    const data = await res.json();
    const classes = data.classes;
    const total   = data.total;

    // Map folder names to display IDs
    const map = {
      'Bengin cases':    el.benignCount,
      'Malignant cases': el.malignantCount,
      'Normal cases':    el.normalCount,
    };
    Object.entries(classes).forEach(([cls, cnt]) => {
      if (map[cls]) map[cls].textContent = cnt.toLocaleString();
    });
    if (el.statTotal) el.statTotal.textContent = total.toLocaleString();

    // Animate count-up
    animateCountup();
  } catch {
    // Use defaults already in HTML
  }
}

function animateCountup() {
  document.querySelectorAll('.hero__stat-value').forEach(el => {
    const target = parseInt(el.textContent.replace(/[^0-9]/g, ''));
    if (isNaN(target)) return;
    let start = 0;
    const step = Math.ceil(target / 60);
    const timer = setInterval(() => {
      start = Math.min(start + step, target);
      el.textContent = start.toLocaleString() + (el.textContent.includes('%') ? '%' : '');
      if (start >= target) clearInterval(timer);
    }, 20);
  });
}

// ─── Model Cards ──────────────────────────────────────────────────────────────
function renderModelCards(loadedStatus = {}) {
  el.modelCards.innerHTML = Object.entries(MODEL_INFO).map(([name, info]) => {
    const loaded = loadedStatus[name];
    const badge  = loaded === true  ? '<span class="model-card__status model-card__status--loaded">✓ Loaded</span>'
                 : loaded === false ? '<span class="model-card__status model-card__status--missing">○ Not trained</span>'
                 : '';
    return `
      <div class="model-card" style="--accent:${info.color}">
        <div class="model-card__icon">${info.icon}</div>
        <div class="model-card__name">${name}</div>
        ${badge}
        <div class="model-card__desc">${info.desc}</div>
        <div class="model-card__meta">
          <span class="model-card__chip">Acc ~${info.acc}</span>
          <span class="model-card__chip">${info.params} params</span>
          <span class="model-card__chip" style="background:${info.color}22;color:${info.color}">
            W: ${getEnsembleWeight(name)}
          </span>
        </div>
      </div>`;
  }).join('');
}

function getEnsembleWeight(name) {
  const w = { ResNet50: 1.3, EfficientNetB0: 1.2, DenseNet121: 1.2, MobileNetV3: 0.9, VGG16: 1.0 };
  return w[name] ?? 1.0;
}

async function loadModels() {
  try {
    const res  = await fetch(`${API_BASE}/api/models`);
    const data = await res.json();
    renderModelCards(data.loaded || {});
  } catch {
    renderModelCards({});
  }
}

// ─── File Handling ────────────────────────────────────────────────────────────
function setFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('Please select an image file (JPG, PNG, JPEG)', 'error');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast('File too large (max 20 MB)', 'error');
    return;
  }

  currentFile = file;
  const url = URL.createObjectURL(file);
  el.previewImg.src = url;
  el.previewMeta.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB  ·  ${file.type}`;
  el.dropZone.hidden   = true;
  el.previewBox.hidden = false;
  el.analyzeBtn.disabled = false;
  resetResults();
}

el.fileInput.addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

el.dropZone.addEventListener('click', () => el.fileInput.click());
el.dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') el.fileInput.click(); });

el.dropZone.addEventListener('dragover', e => { e.preventDefault(); el.dropZone.classList.add('drop-zone--over'); });
el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('drop-zone--over'));
el.dropZone.addEventListener('drop', e => {
  e.preventDefault();
  el.dropZone.classList.remove('drop-zone--over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

el.clearBtn.addEventListener('click', clearAll);
el.resetBtn.addEventListener('click', clearAll);

function clearAll() {
  currentFile = null;
  lastResult  = null;
  el.fileInput.value = '';
  el.previewImg.src  = '';
  el.previewBox.hidden = true;
  el.dropZone.hidden   = false;
  el.analyzeBtn.disabled = true;
  resetResults();
  stopWebcam();
}

function resetResults() {
  el.placeholderState.hidden = false;
  el.resultContent.hidden    = true;
}

// ─── Webcam ───────────────────────────────────────────────────────────────────
el.webcamBtn.addEventListener('click', async () => {
  if (webcamStream) { stopWebcam(); return; }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    el.webcamVideo.srcObject = webcamStream;
    el.webcamVideo.hidden    = false;
    el.captureBtn.hidden     = false;
    el.webcamBtn.textContent = '🛑 Stop Camera';
  } catch {
    showToast('Camera access denied or unavailable', 'error');
  }
});

el.captureBtn.addEventListener('click', () => {
  const video  = el.webcamVideo;
  const canvas = document.createElement('canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    const f = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
    setFile(f);
    stopWebcam();
  }, 'image/jpeg', 0.92);
});

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  el.webcamVideo.hidden = true;
  el.captureBtn.hidden  = true;
  el.webcamBtn.textContent = '📸 Camera Capture';
}

// ─── Sample Image ─────────────────────────────────────────────────────────────
el.sampleBtn.addEventListener('click', async () => {
  // Create a synthetic grayscale lung-like test image using canvas
  showToast('Loading demo CT scan image…', 'info', 2000);
  const canvas = document.createElement('canvas');
  canvas.width  = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  // Draw a convincing grayscale lung CT pattern
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, 224, 224);

  // Lung outline - left
  ctx.beginPath();
  ctx.ellipse(70, 112, 50, 80, -0.2, 0, Math.PI * 2);
  ctx.fillStyle = '#3a3a3a';
  ctx.fill();

  // Lung outline - right
  ctx.beginPath();
  ctx.ellipse(154, 112, 50, 80, 0.2, 0, Math.PI * 2);
  ctx.fillStyle = '#3a3a3a';
  ctx.fill();

  // Trachea / bronchi
  ctx.fillStyle = '#555';
  ctx.fillRect(107, 20, 10, 40);
  ctx.fillRect(90, 58, 27, 8);

  // Add gradient for 3D effect
  const grad = ctx.createRadialGradient(70, 95, 5, 70, 95, 55);
  grad.addColorStop(0, 'rgba(200,200,200,0.3)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 224, 224);

  // Some noise for realism
  const imgData = ctx.getImageData(0, 0, 224, 224);
  for (let i = 0; i < imgData.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 20;
    imgData.data[i]     = Math.max(0, Math.min(255, imgData.data[i] + n));
    imgData.data[i + 1] = imgData.data[i];
    imgData.data[i + 2] = imgData.data[i];
  }
  ctx.putImageData(imgData, 0, 0);

  canvas.toBlob(blob => {
    const f = new File([blob], 'sample_ct_scan.jpg', { type: 'image/jpeg' });
    setFile(f);
  }, 'image/jpeg', 0.9);
});

// ─── Analysis / Prediction ─────────────────────────────────────────────────────
el.analyzeBtn.addEventListener('click', analyze);

async function analyze() {
  if (!currentFile) { showToast('Please select a CT scan image first', 'warning'); return; }

  // Set loading state
  el.analyzeBtn.disabled       = true;
  el.analyzeBtnSpinner.classList.remove('hidden');
  el.analyzeBtnText.textContent = 'Analyzing…';

  try {
    const formData = new FormData();
    formData.append('file', currentFile);

    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(30000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    lastResult = await res.json();
    renderResults(lastResult);
    showToast('Analysis complete!', 'success');

    // Smooth scroll to results
    document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (e) {
    showToast(`Analysis failed: ${e.message}`, 'error', 6000);
  } finally {
    el.analyzeBtn.disabled        = false;
    el.analyzeBtnSpinner.classList.add('hidden');
    el.analyzeBtnText.textContent = '🔍 Analyze Scan';
  }
}

// ─── Result Rendering ──────────────────────────────────────────────────────────
function renderResults(data) {
  el.placeholderState.hidden = true;
  el.resultContent.hidden    = false;

  // Card colours
  const riskColors = { HIGH: '#ef4444', MODERATE: '#f59e0b', LOW: '#22c55e' };
  const riskGradients = {
    HIGH:     'linear-gradient(135deg,#ef444422,#ef444411)',
    MODERATE: 'linear-gradient(135deg,#f59e0b22,#f59e0b11)',
    LOW:      'linear-gradient(135deg,#22c55e22,#22c55e11)',
  };

  el.resultCard.style.background = riskGradients[data.risk_level];
  el.resultCard.style.borderColor = riskColors[data.risk_level] + '66';
  el.predLabel.textContent = data.prediction;
  el.predLabel.style.color = riskColors[data.risk_level];
  el.predConf.textContent  = `${data.confidence}% confidence`;
  el.riskBadge.textContent = `${data.risk_level} RISK`;
  el.riskBadge.style.background = riskColors[data.risk_level] + '22';
  el.riskBadge.style.color      = riskColors[data.risk_level];

  // Probability bars
  const classColors = { Benign: '#f59e0b', Malignant: '#ef4444', Normal: '#22c55e' };
  el.probBars.innerHTML = Object.entries(data.class_probabilities).map(([cls, prob]) => `
    <div class="prob-bar-row">
      <span class="prob-bar-label">${cls}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:0%;background:${classColors[cls] || '#6366f1'}"
             data-target="${prob}"></div>
      </div>
      <span class="prob-bar-value">${prob}%</span>
    </div>`).join('');

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll('.prob-bar-fill').forEach(bar => {
      bar.style.transition = 'width 0.8s cubic-bezier(.4,0,.2,1)';
      bar.style.width = bar.dataset.target + '%';
    });
  }, 50);

  // Per-model breakdown
  const models = data.per_model_predictions || {};
  if (Object.keys(models).length) {
    el.modelBreakdownBody.innerHTML = Object.entries(models).map(([name, info]) => {
      const mi  = MODEL_INFO[name] || {};
      const col = mi.color || '#6366f1';
      return `
        <div class="breakdown-row">
          <span class="breakdown-icon" style="color:${col}">${mi.icon || '🧠'}</span>
          <span class="breakdown-name">${name}</span>
          <span class="breakdown-pred" style="color:${col}">${info.prediction}</span>
          <span class="breakdown-conf">${info.confidence}%</span>
          <span class="breakdown-weight" title="Ensemble weight">w=${info.weight}</span>
        </div>`;
    }).join('');
  } else {
    el.modelBreakdownBody.innerHTML = '<p class="dim">No per-model data (demo mode)</p>';
  }

  // Recommendation
  el.recommendationText.textContent = data.recommendation || '—';

  // Meta row
  const modelsUsed = (data.models_used || []).join(', ') || 'Demo';
  el.metaRow.innerHTML = `
    <div class="meta-chip">⏱ ${data.inference_time_ms} ms</div>
    <div class="meta-chip">🧠 ${(data.models_used || []).length || 1} model(s)</div>
    <div class="meta-chip">🕐 ${new Date(data.timestamp).toLocaleTimeString()}</div>
  `;
}

// ─── Download Report ──────────────────────────────────────────────────────────
el.downloadReportBtn.addEventListener('click', generateReport);

function generateReport() {
  if (!lastResult) { showToast('No results to export', 'warning'); return; }

  const now   = new Date();
  const lines = [
    'LUNGCANCERDX — DIAGNOSTIC REPORT',
    '='.repeat(50),
    `Date       : ${now.toLocaleDateString()}`,
    `Time       : ${now.toLocaleTimeString()}`,
    '',
    'PREDICTION SUMMARY',
    '-'.repeat(30),
    `Diagnosis  : ${lastResult.prediction}`,
    `Confidence : ${lastResult.confidence}%`,
    `Risk Level : ${lastResult.risk_level}`,
    '',
    'CLASS PROBABILITIES',
    '-'.repeat(30),
    ...Object.entries(lastResult.class_probabilities).map(([c, p]) => `  ${c.padEnd(12)}: ${p}%`),
    '',
    'PER-MODEL BREAKDOWN',
    '-'.repeat(30),
    ...Object.entries(lastResult.per_model_predictions || {}).map(([m, i]) =>
      `  ${m.padEnd(16)}: ${i.prediction} (${i.confidence}%)  weight=${i.weight}`),
    '',
    'CLINICAL RECOMMENDATION',
    '-'.repeat(30),
    lastResult.recommendation,
    '',
    'TECHNICAL DETAILS',
    '-'.repeat(30),
    `Inference Time : ${lastResult.inference_time_ms} ms`,
    `Models Used    : ${(lastResult.models_used || []).join(', ') || 'Demo'}`,
    `Ensemble       : Weighted Soft-Voting`,
    '',
    '='.repeat(50),
    '⚠️  FOR RESEARCH/EDUCATIONAL USE ONLY.',
    'This output is NOT a substitute for professional medical diagnosis.',
  ];

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `LungCancerDX_Report_${now.toISOString().slice(0, 10)}.txt`;
  a.click();
  showToast('Report downloaded!', 'success');
}

// ─── Navigation ───────────────────────────────────────────────────────────────
el.navToggle.addEventListener('click', () => {
  el.navLinks.classList.toggle('nav__links--open');
});

// Active link on scroll
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav__link');

const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + e.target.id));
    }
  });
}, { threshold: 0.4 });

sections.forEach(s => observer.observe(s));

// Smooth scroll for nav links
navLinks.forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth' });
    el.navLinks.classList.remove('nav__links--open');
  });
});

// Hero scan button
const heroScanBtn = $('heroScanBtn');
if (heroScanBtn) {
  heroScanBtn.addEventListener('click', e => {
    e.preventDefault();
    document.getElementById('scanner').scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => el.fileInput.click(), 600);
  });
}

// ─── Animated Scan Line ───────────────────────────────────────────────────────
function startScanAnimation() {
  const scanLine = document.querySelector('.scan-line');
  if (!scanLine) return;
  let pos = 0, dir = 1;
  setInterval(() => {
    pos += dir * 1.5;
    if (pos >= 100) { pos = 100; dir = -1; }
    if (pos <= 0)   { pos = 0;   dir = 1;  }
    scanLine.style.top = pos + '%';
  }, 16);
}

// ─── Particle / Orb Parallax ──────────────────────────────────────────────────
document.addEventListener('mousemove', e => {
  const { innerWidth: W, innerHeight: H } = window;
  const xPct = (e.clientX / W - 0.5) * 2;
  const yPct = (e.clientY / H - 0.5) * 2;
  document.querySelectorAll('.hero__orb').forEach((orb, i) => {
    const depth = (i + 1) * 12;
    orb.style.transform = `translate(${xPct * depth}px, ${yPct * depth}px)`;
  });
});

// ─── Initialise ───────────────────────────────────────────────────────────────
(async () => {
  renderModelCards({});
  await checkHealth();

  // Load model status & dataset stats in parallel
  Promise.all([loadModels(), loadDatasetStats()]);

  startScanAnimation();

  // Pulse health check every 30s
  setInterval(checkHealth, 30_000);
})();
