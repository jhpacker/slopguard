// Re-injection guard. background.js injects this file via chrome.scripting on
// every toolbar/context-menu invocation (no static content_scripts auto-inject
// anymore). The first injection
// registers the message + contextmenu listeners and observers; later injections
// must NOT re-register them. The listeners from the first load persist and still
// receive scan-now / check-one, so a repeat injection just no-ops here.
if (window.__slopguardLoaded) {
  // already injected on this page
} else {
  window.__slopguardLoaded = true;

const MIN_AREA = 40000; // ~200 × 200 rendered px
const TAG = '[SlopGuard]';

let scanActive = false;
let processed = new WeakSet();
let debugMode = false;

// Progress tracking. Counts only the initial click-time batch — once it
// finishes, infinite-scroll images get scanned silently in the background.
let totalToCheck = 0;
let doneCount = 0;
let initialScanComplete = false;
let progressEl = null;
let progressHideTimer = null;

// Load + watch debug setting
try {
  chrome.storage?.local?.get({ debugMode: false }, (s) => {
    debugMode = !!(s && s.debugMode);
    if (debugMode) console.log(`${TAG} content script loaded on`, location.href);
  });
  chrome.storage?.onChanged?.addListener((changes) => {
    if (changes.debugMode) debugMode = !!changes.debugMode.newValue;
  });
} catch (e) {
  console.warn(`${TAG} chrome.storage unavailable — reload the extension to pick up new permissions`, e);
}
const dlog = (...args) => debugMode && console.log(...args);
const dwarn = (...args) => debugMode && console.warn(...args);

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === 'scan-now') {
    startScan();
    return;
  }
  if (msg?.type === 'check-one') {
    checkOne(msg.srcUrl);
    return;
  }
});

// Track the most recently right-clicked element so a "Check this image"
// context-menu click can resolve the exact <img> the user targeted — more
// reliable than matching on srcUrl alone when a page has duplicate sources.
// Capture phase so we see it even if the page stops propagation.
let lastContextTarget = null;
document.addEventListener(
  'contextmenu',
  (e) => {
    lastContextTarget = e.target;
  },
  true,
);

function startScan() {
  dlog(`${TAG} scan triggered`);
  scanActive = true;
  processed = new WeakSet();
  doneCount = 0;
  initialScanComplete = false;
  clearAllMarks();

  const all = Array.from(document.querySelectorAll('img'));
  const eligible = all.filter((img) => isEligible(img));
  totalToCheck = eligible.length;

  showProgress();

  if (eligible.length === 0) {
    updateProgressText('No images on this page big enough to check');
    initialScanComplete = true;
    scheduleHideProgress(2500);
    return;
  }

  // Run checks in parallel — background serializes via offscreen anyway,
  // but parallel sends keep the progress UI lively and don't waste time.
  for (const img of eligible) check(img);
}

function isEligible(img) {
  if (processed.has(img)) return false;
  if (!img.src) return false;
  if (img.src.startsWith('data:') || img.src.startsWith('blob:')) return false;
  if (!img.complete || !img.naturalWidth) return false;
  // Rendered size — what the user actually sees. Skips images with large
  // intrinsic size but small CSS-rendered size (thumbnails, avatars, etc.)
  // and also skips display:none / 0-sized elements.
  const rect = img.getBoundingClientRect();
  if (rect.width * rect.height < MIN_AREA) return false;
  return true;
}

function isContextValid() {
  try {
    return !!chrome.runtime?.id;
  } catch {
    return false;
  }
}

function handleContextInvalidation() {
  if (!scanActive) return;
  scanActive = false;
  showProgress();
  updateProgressText('SlopGuard was reloaded — refresh the page to scan again');
  scheduleHideProgress(6000);
}

async function check(img) {
  if (!scanActive) return;
  if (processed.has(img)) return;
  if (!isContextValid()) {
    handleContextInvalidation();
    return;
  }
  processed.add(img);

  dlog(`${TAG} checking ${img.naturalWidth}x${img.naturalHeight}:`, img.src);

  let result;
  try {
    result = await chrome.runtime.sendMessage({ type: 'detect', url: img.src });
  } catch (e) {
    if (String(e?.message || e).includes('Extension context invalidated')) {
      handleContextInvalidation();
    } else {
      dwarn(`${TAG} sendMessage failed for`, img.src, e);
    }
    bumpProgress();
    return;
  }

  if (result?.ai) {
    dlog(`%c${TAG} AI detected`, 'color:#ff3b30;font-weight:bold', result, img.src);
    mark(img, result);
  } else {
    dlog(`${TAG} not AI:`, result?.reason || 'no-match', result?.detail ? `— ${result.detail}` : '', img.src);
    markClean(img, result);
  }
  bumpProgress();
}

// Resolve the <img> for a context-menu "Check this image" click. Prefer the
// element the user actually right-clicked; fall back to matching the menu's
// srcUrl against img.src / img.currentSrc (covers srcset-resolved sources).
function findImageForCheck(srcUrl) {
  const t = lastContextTarget;
  if (t && t.tagName === 'IMG') return t;
  if (srcUrl) {
    for (const img of document.querySelectorAll('img')) {
      if (img.src === srcUrl || img.currentSrc === srcUrl) return img;
    }
  }
  return null;
}

// On-demand single-image check (right-click menu). Independent of the page
// scan: it doesn't require scanActive, ignores the MIN_AREA size gate (the
// user explicitly chose this image), and re-evaluates even if a prior scan
// already marked it.
async function checkOne(srcUrl) {
  if (!isContextValid()) {
    handleContextInvalidation();
    return;
  }
  const img = findImageForCheck(srcUrl);
  if (!img) {
    showProgress();
    updateProgressText("Couldn't find that image to check");
    scheduleHideProgress(2500);
    return;
  }

  resetMarkFor(img);
  processed.add(img);
  showProgress();
  updateProgressText('Checking image…');
  dlog(`${TAG} check-one ${img.naturalWidth}x${img.naturalHeight}:`, img.src);

  let result;
  try {
    result = await chrome.runtime.sendMessage({ type: 'detect', url: img.src });
  } catch (e) {
    if (String(e?.message || e).includes('Extension context invalidated')) {
      handleContextInvalidation();
    } else {
      dwarn(`${TAG} check-one sendMessage failed for`, img.src, e);
      updateProgressText("Couldn't check that image");
      scheduleHideProgress(2500);
    }
    return;
  }

  if (result?.ai) {
    dlog(`%c${TAG} AI detected (single)`, 'color:#ff3b30;font-weight:bold', result, img.src);
    mark(img, result);
  } else {
    dlog(`${TAG} not AI (single):`, result?.reason || 'no-match', img.src);
    markClean(img, result);
  }
  updateProgressText(result?.ai ? 'Flagged as AI-generated' : 'Checked — no AI markers found');
  scheduleHideProgress(2000);
}

// Reasons that indicate the check itself failed (network error, model failure,
// etc.) rather than the image actually being non-AI. These get a grey border.
const FAILURE_REASONS = new Set([
  'visual-call-failed',
  'visual-fetch-failed',
  'visual-fetch-error',
  'visual-error',
  'visual-all-models-errored',
  'synthid-call-failed',
  'synthid-fetch-failed',
  'synthid-fetch-error',
  'synthid-error',
  'c2pa-call-failed',
  'fetch-failed',
  'error',
]);

function markClean(img, result) {
  if (img.classList.contains('slopguard-img')) return;
  if (img.classList.contains('slopguard-clean')) return;
  if (img.classList.contains('slopguard-failed')) return;

  const reason = result?.reason;
  const isFailure = !result || FAILURE_REASONS.has(reason);
  const cls = isFailure ? 'slopguard-failed' : 'slopguard-clean';
  img.classList.add(cls);
  if (isFailure && reason) {
    img.title = `SlopGuard couldn't evaluate this image (${reason})`;
  }

  // In debug mode, render the per-tier metadata readout on clean/failed images
  // too — not just flagged ones — so every scanned image is inspectable.
  let debug = null;
  let anchorName = null;
  if (debugMode) ({ debug, anchorName } = attachDebug(img, result, null));

  const m = { img, label: null, debug, anchorName };
  activeMarks.push(m);

  if (debug && !supportsAnchor) {
    positionMark(m);
    ensureRepositionListeners();
  }
}

function classifyConfidence(result) {
  // Tier 5 visual classifier — softer label since pixel reasoning isn't as
  // authoritative as declared metadata. `visualTier` is set by the offscreen
  // handler based on confidentThreshold crossings.
  if (result.reason === 'visual-classifier') {
    if (result.visualTier === 'high') {
      return { text: 'Probably AI', tier: 'visual-high' };
    }
    return { text: 'Maybe AI', tier: 'visual-low' };
  }
  // Everything else (c2pa, exif, byte-scan) — declared / explicit signals
  // from the AI tool itself. Highest confidence.
  return { text: 'AI', tier: 'metadata' };
}

// Body-appended overlays anchored to the image. We don't wrap the <img>
// because some sites (Twitter, Reddit, sites with aspect-ratio containers)
// position the img as absolute inside its parent — a wrapper would collapse
// to 0 height and break centering.
//
// Preferred: CSS anchor positioning (Chrome 125+) — the browser tracks the
// img and repositions automatically through any kind of scroll/layout change.
// Fallback: position:fixed with a rAF loop for browsers without anchor support.
const supportsAnchor =
  typeof CSS !== 'undefined' && CSS.supports?.('anchor-name', '--x');
let anchorCounter = 0;
const activeMarks = [];

// Maps a tier status to a row class. Non-clean rows are colored so they stand
// out: 'ai' → red, 'fail' → grey (matching the grey "failed" image outline).
function debugStatusClass(status) {
  if (status === 'ai') return 'slopguard-debug-ai';
  if (status === 'fail') return 'slopguard-debug-failrow';
  return '';
}

// Status for the verdict row: red if the image was flagged AI, grey if the
// check failed, otherwise clean.
function verdictStatus(result) {
  if (!result) return 'fail';
  if (result.ai) return 'ai';
  if (FAILURE_REASONS.has(result.reason)) return 'fail';
  return 'clean';
}

// Debug readout shown under the image in debug mode, rendered as a table for
// readability: a verdict row spanning both columns, then one { tier, text,
// status } row per tier from the metadata summary collected by background.js.
// textContent on every cell keeps it XSS-safe even though the strings come
// from our own background.
function buildDebugTable(result) {
  const table = document.createElement('table');
  table.className = 'slopguard-debug-table';

  const vRow = table.insertRow();
  vRow.className = debugStatusClass(verdictStatus(result));
  const vCell = vRow.insertCell();
  vCell.colSpan = 2;
  vCell.className = 'slopguard-debug-verdict';
  vCell.textContent = result?.reason
    ? `${result.reason}${result.detail ? ': ' + result.detail : ''}`
    : '(no result)';

  if (Array.isArray(result?.metaChecks)) {
    for (const item of result.metaChecks) {
      const row = table.insertRow();
      row.className = debugStatusClass(item.status);
      const key = row.insertCell();
      key.className = 'slopguard-debug-key';
      key.textContent = item.tier ?? '';
      row.insertCell().textContent = item.text ?? '';
    }
  }
  return table;
}

// Create + anchor the debug overlay element. Reuses an existing anchorName
// (minted for an AI label) or mints one for images with no label (clean/failed).
function attachDebug(img, result, anchorName) {
  const debug = document.createElement('div');
  debug.className = 'slopguard-debug';
  debug.appendChild(buildDebugTable(result));
  document.body.appendChild(debug);
  if (supportsAnchor) {
    if (!anchorName) {
      anchorName = `--slopguard-${++anchorCounter}`;
      img.style.setProperty('anchor-name', anchorName);
    }
    debug.classList.add('slopguard-debug--anchored');
    debug.style.setProperty('position-anchor', anchorName);
  }
  return { debug, anchorName };
}

function mark(img, result) {
  if (img.classList.contains('slopguard-img')) return;
  const conf = classifyConfidence(result);

  img.classList.add('slopguard-img');

  const label = document.createElement('div');
  label.className = `slopguard-label slopguard-label--${conf.tier}`;
  label.textContent = conf.text;
  if (result.detail) label.title = `${result.reason}: ${result.detail}`;
  document.body.appendChild(label);

  let anchorName = null;
  if (supportsAnchor) {
    anchorName = `--slopguard-${++anchorCounter}`;
    img.style.setProperty('anchor-name', anchorName);
    label.classList.add('slopguard-label--anchored');
    label.style.setProperty('position-anchor', anchorName);
  }

  let debug = null;
  if (debugMode) ({ debug, anchorName } = attachDebug(img, result, anchorName));

  const m = { img, label, debug, anchorName };
  activeMarks.push(m);

  // Only run JS positioning for the fallback path; anchor positioning is pure CSS.
  if (!supportsAnchor) {
    positionMark(m);
    ensureRepositionListeners();
  }
}

function positionMark(m) {
  const rect = m.img.getBoundingClientRect();
  const hidden = rect.width === 0 || rect.height === 0;
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  // m.label is null for clean/failed images (debug overlay only) — guard it.
  if (m.label) {
    if (hidden) {
      m.label.style.display = 'none';
    } else {
      m.label.style.display = '';
      m.label.style.left = `${cx}px`;
      m.label.style.top = `${cy}px`;
    }
  }
  if (m.debug) {
    if (hidden) {
      m.debug.style.display = 'none';
    } else {
      m.debug.style.display = '';
      m.debug.style.left = `${cx}px`;
      m.debug.style.top = `${cy + 26}px`;
    }
  }
}

function repositionAll() {
  for (const m of activeMarks) positionMark(m);
}

// Sites like Twitter virtualize scroll via transforms or non-bubbling inner
// scroll, so a window scroll listener misses position changes. A rAF loop
// while there are marks is more expensive but bulletproof. ~1ms/frame for
// 20 marks since getBoundingClientRect is cheap when layout isn't dirty.
let repositionLoopActive = false;
function ensureRepositionListeners() {
  if (repositionLoopActive) return;
  repositionLoopActive = true;
  const tick = () => {
    if (activeMarks.length === 0) {
      repositionLoopActive = false;
      return;
    }
    repositionAll();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function clearAllMarks() {
  for (const m of activeMarks) {
    m.label?.remove();
    m.debug?.remove();
    m.img.classList.remove('slopguard-img');
    m.img.classList.remove('slopguard-clean');
    m.img.classList.remove('slopguard-failed');
    if (m.anchorName) m.img.style.removeProperty('anchor-name');
  }
  activeMarks.length = 0;
}

// ---- Progress UI ----

function showProgress() {
  if (progressHideTimer) {
    clearTimeout(progressHideTimer);
    progressHideTimer = null;
  }
  if (!progressEl) {
    progressEl = document.createElement('div');
    progressEl.className = 'slopguard-progress';
    progressEl.innerHTML = `
      <div class="slopguard-progress-text">Scanning…</div>
      <div class="slopguard-progress-bar-bg"><div class="slopguard-progress-bar"></div></div>
    `;
    document.documentElement.appendChild(progressEl);
  }
  progressEl.style.opacity = '1';
  updateProgress();
}

function updateProgress() {
  if (!progressEl) return;
  const pct = totalToCheck === 0 ? 0 : Math.round((doneCount / totalToCheck) * 100);
  progressEl.querySelector('.slopguard-progress-text').textContent =
    `Scanning ${doneCount} / ${totalToCheck} images`;
  progressEl.querySelector('.slopguard-progress-bar').style.width = `${pct}%`;
}

function updateProgressText(text) {
  if (!progressEl) return;
  progressEl.querySelector('.slopguard-progress-text').textContent = text;
  progressEl.querySelector('.slopguard-progress-bar').style.width = '100%';
}

function bumpProgress() {
  // Once initial scan is done, infinite-scroll items are processed silently.
  if (initialScanComplete) return;
  doneCount++;
  if (doneCount >= totalToCheck) {
    initialScanComplete = true;
    updateProgressText(
      `Scanned ${doneCount} image${doneCount === 1 ? '' : 's'} — watching for more`,
    );
    scheduleHideProgress(3000);
  } else {
    updateProgress();
  }
}

function scheduleHideProgress(delay) {
  progressHideTimer = setTimeout(() => {
    if (progressEl) progressEl.style.opacity = '0';
    setTimeout(() => {
      if (progressEl?.parentNode) progressEl.parentNode.removeChild(progressEl);
      progressEl = null;
    }, 400);
  }, delay);
}

// Track URL so we can reset scan state on SPA navigation. The user wants a
// fresh button click required for each "page", but SPAs (Gencraft, Reddit,
// Twitter, etc.) change URL without reloading the document — so scanActive
// would otherwise persist across routes.
let lastUrl = location.href;
function onNavigation() {
  scanActive = false;
  processed = new WeakSet();
  totalToCheck = 0;
  doneCount = 0;
  clearAllMarks();
  if (progressEl) {
    if (progressHideTimer) clearTimeout(progressHideTimer);
    progressEl.style.opacity = '0';
    setTimeout(() => {
      if (progressEl?.parentNode) progressEl.parentNode.removeChild(progressEl);
      progressEl = null;
    }, 200);
  }
  dlog(`${TAG} URL changed to`, location.href, '— scan deactivated. Click toolbar to scan again.');
}

function watchImageForInfinite(img) {
  if (img.complete && img.naturalWidth) {
    if (isEligible(img)) check(img);
  } else {
    img.addEventListener(
      'load',
      () => {
        if (scanActive && isEligible(img)) check(img);
      },
      { once: true },
    );
  }
}

function resetMarkFor(img) {
  // Image's src changed — clear our state so we re-evaluate the new image.
  processed.delete(img);
  img.classList.remove('slopguard-img');
  img.classList.remove('slopguard-clean');
  img.classList.remove('slopguard-failed');
  for (let i = activeMarks.length - 1; i >= 0; i--) {
    if (activeMarks[i].img === img) {
      activeMarks[i].label?.remove();
      activeMarks[i].debug?.remove();
      activeMarks.splice(i, 1);
    }
  }
}

// Watch DOM for:
//   - SPA navigations (URL change → reset)
//   - newly-added <img> elements (infinite scroll)
//   - src changes on existing <img>s (lazy-load swaps)
const observer = new MutationObserver((mutations) => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    onNavigation();
    return;
  }
  if (!scanActive) return;

  for (const m of mutations) {
    if (m.type === 'attributes' && m.target.tagName === 'IMG') {
      const img = m.target;
      resetMarkFor(img);
      watchImageForInfinite(img);
      continue;
    }
    for (const node of m.addedNodes || []) {
      if (node.nodeType !== 1) continue;
      const imgs =
        node.tagName === 'IMG' ? [node] : node.querySelectorAll?.('img') || [];
      for (const img of imgs) watchImageForInfinite(img);
    }
  }
});
observer.observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ['src'],
});

} // end re-injection guard
