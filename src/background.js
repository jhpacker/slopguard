import * as exifr from 'exifr';

const TAG = '[SlopGuard bg]';
const cache = new Map();
const MAX_CACHE = 1000;
const MAX_FETCH = 8 * 1024 * 1024;

let debugMode = false;
// Fast debug: still show the per-tier debug table, but short-circuit on the
// first detecting tier exactly like the non-debug cascade (don't force-run the
// expensive ONNX tiers just to fill the table). Only meaningful when debugMode
// is on. See doDetect's `forceAllTiers`.
let fastDebug = false;
try {
  chrome.storage?.local?.get({ debugMode: false, fastDebug: false }, (s) => {
    debugMode = !!(s && s.debugMode);
    fastDebug = !!(s && s.fastDebug);
  });
  chrome.storage?.onChanged?.addListener((changes) => {
    if (changes.debugMode) debugMode = !!changes.debugMode.newValue;
    if (changes.fastDebug) fastDebug = !!changes.fastDebug.newValue;
  });
} catch (e) {
  console.warn(`${TAG} chrome.storage unavailable — reload the extension to pick up new permissions`, e);
}
const dlog = (...args) => debugMode && console.log(...args);
const dwarn = (...args) => debugMode && console.warn(...args);

console.log(`${TAG} service worker started`);

// Toolbar icon is the static six-finger-hand mark declared in manifest.json
// ("icons" + action "default_icon", built from assets/logo/icon-*.png).

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type !== 'detect') return false;
  detect(msg.url).then((res) => {
    dlog(`${TAG} result for`, msg.url, '→', res);
    sendResponse(res);
  });
  return true;
});

// The content script is no longer auto-injected on every page (the static
// content_scripts block was removed to shrink the attack surface CWS scrutinizes).
// Instead we inject it on demand via `chrome.scripting` into the tab the user
// invoked. `<all_urls>` host_permissions is retained — it's required for the
// cross-origin credentialed image-byte fetches in fetchBytes/offscreen, which
// activeTab cannot cover — and it also authorizes these injections.
// content.js guards against double-injection, so re-injecting on a repeat click
// is harmless.
async function ensureInjected(tabId, frameId) {
  const target = frameId != null ? { tabId, frameIds: [frameId] } : { tabId };
  // Probe first: if content.js is already present, skip re-injection. Re-running
  // executeScript would no-op via content.js's __slopguardLoaded guard, but
  // insertCSS has no such guard and would stack a duplicate (identical)
  // stylesheet on every repeat click.
  const [probe] = await chrome.scripting.executeScript({
    target,
    func: () => !!window.__slopguardLoaded,
  });
  if (probe?.result) return;
  await chrome.scripting.insertCSS({ target, files: ['dist/overlay.css'] });
  await chrome.scripting.executeScript({ target, files: ['dist/content.js'] });
}

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab?.id) return;
  try {
    await ensureInjected(tab.id);
    await chrome.tabs.sendMessage(tab.id, { type: 'scan-now' });
    dlog(`${TAG} sent scan-now to tab`, tab.id);
  } catch (e) {
    console.warn(`${TAG} cannot scan tab ${tab.id} (probably a chrome:// or restricted page):`, e?.message);
  }
});

// Right-click → "Check this image for AI" — scans a single image on demand
// without scanning the whole page. The menu registration persists across
// service-worker restarts; onInstalled is the documented place to create it.
const CHECK_ONE_MENU_ID = 'slopguard-check-image';
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: CHECK_ONE_MENU_ID,
    title: 'Check this image for AI',
    contexts: ['image'],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== CHECK_ONE_MENU_ID || !tab?.id) return;
  try {
    // frameId targets the exact frame the image lives in (handles iframes).
    // Inject into that frame first so content.js is present to receive check-one.
    await ensureInjected(tab.id, info.frameId);
    await chrome.tabs.sendMessage(
      tab.id,
      { type: 'check-one', srcUrl: info.srcUrl },
      info.frameId != null ? { frameId: info.frameId } : undefined,
    );
    dlog(`${TAG} sent check-one to tab`, tab.id, 'frame', info.frameId, info.srcUrl);
  } catch (e) {
    console.warn(`${TAG} cannot check image in tab ${tab.id} (probably a restricted page):`, e?.message);
  }
});

async function detect(url) {
  if (cache.has(url)) return cache.get(url);
  const promise = doDetect(url).catch((err) => ({
    ai: false,
    reason: 'error',
    detail: String(err?.message || err),
  }));
  cache.set(url, promise);
  if (cache.size > MAX_CACHE) {
    cache.delete(cache.keys().next().value);
  }
  return promise;
}

async function doDetect(url) {
  // Tier 1: c2pa-js via offscreen document — authoritative for C2PA-signed images.
  // We ignore validation status; the manifest contents are what we trust.
  const c2paResult = await c2paDetect(url);

  // In *full* debug mode we run every tier so the table is complete even when an
  // earlier tier already flagged AI. In *fast* debug mode we behave like the
  // non-debug cascade (short-circuit on the first hit) and only show scores for
  // the tiers that actually ran. forceAllTiers distinguishes the two.
  const forceAllTiers = debugMode && !fastDebug;

  // Normal/fast flow short-circuits on a C2PA AI hit without fetching bytes.
  // Only full debug fetches the bytes anyway, to populate the metadata tiers.
  const bytes = (!c2paResult.ai || forceAllTiers) ? await fetchBytes(url) : null;

  // Tier-4 SynthID watermark result, memoized. In debug mode we want its row
  // in the table for EVERY image — even ones an earlier tier already flagged —
  // so finalize() forces the detection; the normal tier-4 dispatch reuses the
  // same memoized value, so it never runs twice.
  let synthidResult;
  let synthidRan = false;
  const getSynthid = async () => {
    if (!synthidRan) {
      synthidRan = true;
      synthidResult = await synthidDetect(url);
    }
    return synthidResult;
  };

  // Tier-5 visual (OpenFake) result, memoized the same way. In debug mode
  // finalize forces it so its EP + timing row shows for EVERY image, even one
  // an earlier tier already flagged; the normal tier-5 dispatch reuses this.
  let visualResult;
  let visualRan = false;
  const getVisual = async () => {
    if (!visualRan) {
      visualRan = true;
      visualResult = await visualDetect(url);
    }
    return visualResult;
  };

  // In debug mode, every return path carries a compact per-tier summary
  // (metaChecks) that the content script renders on the image.
  const finalize = async (result) => {
    if (debugMode) {
      // Full debug forces the ONNX tiers so their rows always appear. Fast debug
      // uses only whatever the cascade actually ran (memoized) — unreached tiers
      // stay undefined and render as "not run".
      const sid = forceAllTiers ? await getSynthid() : synthidResult;
      const vis = forceAllTiers ? await getVisual() : visualResult;
      result.metaChecks = await buildMetaChecks(bytes, c2paResult, sid, vis);
    }
    return result;
  };

  if (c2paResult.ai) return finalize(c2paResult);
  if (!bytes) {
    return finalize({ ai: false, reason: c2paResult.reason || 'fetch-failed' });
  }

  // Tier 2: structured EXIF/IPTC/XMP parse — catches AI attribution in
  // Artist/Author/Credit fields where the value is meaningful only in context.
  const attribResult = await exifrAttributionDetect(bytes);
  if (attribResult?.ai) return finalize(attribResult);

  // Tier 3: byte scan — XMP DigitalSourceType, agent names, prompt params.
  dlog(`${TAG} byte-scanning ${bytes.length} bytes from`, url);
  const byteResult = byteScanDetect(bytes);
  if (byteResult.ai) return finalize(byteResult);

  // Tier 4: SynthID watermark surrogate (offscreen). High-precision provenance
  // signal for Google-generated images (Imagen/Gemini/Nano Banana) — survives
  // metadata stripping, which tiers 1–3 don't. Short-circuit on a hit.
  const synthidHit = await getSynthid();
  if (synthidHit?.ai) return finalize(synthidHit);

  // Tier 5: ONNX visual classifier ensemble. The original three pixel
  // classifiers (Organika SDXL, Siglip2 Deepfake, AI-vs-Human SigLIP) proved
  // unreliable in the wild (false positives on real photos) and are commented
  // out of VISUAL_MODELS. Currently re-enabled to TRIAL a single new model —
  // OpenFake SwinV2 (ComplexDataLab/OpenFakeDemo) — in isolation. If it
  // regresses to the same FP behavior, re-comment this dispatch.
  const tier5 = await getVisual();
  if (tier5?.ai) return finalize(tier5);
  return finalize(tier5 || byteResult);

  // Tier 6 (LLM vision judge, Chrome built-in Gemini Nano) was removed: its
  // verdict was driven by prompt framing rather than the pixels, so it never
  // did reliable vision-based AI detection. See CLAUDE.md "Things tried and
  // rejected" for the full write-up.
}

async function synthidDetect(url) {
  const t0 = performance.now();
  dlog(`${TAG} synthid → start`, url);
  try {
    await ensureOffscreen();
    const result = await chrome.runtime.sendMessage({
      target: 'offscreen',
      type: 'synthid-detect',
      url,
    });
    dlog(
      `${TAG} synthid ← done in ${(performance.now() - t0).toFixed(0)}ms`,
      result?.reason,
      result?.detail || '',
    );
    return result;
  } catch (e) {
    dwarn(`${TAG} synthid ✗ offscreen call failed in ${(performance.now() - t0).toFixed(0)}ms for`, url, e);
    return { ai: false, reason: 'synthid-call-failed', detail: String(e?.message || e) };
  }
}

async function visualDetect(url) {
  const t0 = performance.now();
  dlog(`${TAG} tier5 → start`, url);
  try {
    await ensureOffscreen();
    const result = await chrome.runtime.sendMessage({
      target: 'offscreen',
      type: 'visual-detect',
      url,
    });
    dlog(
      `${TAG} tier5 ← done in ${(performance.now() - t0).toFixed(0)}ms`,
      result?.reason,
      result?.detail || '',
    );
    return result;
  } catch (e) {
    dwarn(`${TAG} tier5 ✗ offscreen call failed in ${(performance.now() - t0).toFixed(0)}ms for`, url, e);
    return { ai: false, reason: 'visual-call-failed', detail: String(e?.message || e) };
  }
}

// Attribution fields walked by tier 2, and the value pattern that marks one as
// AI-declared. Hoisted to module scope so the debug summary (summarizeExif)
// reuses the exact same definitions.
const ATTRIB_FIELDS = [
  'Artist',
  'Author',
  'By',
  'Byline',
  'By-line',
  'Credit',
  'Source',
  'Creator',
  'Copyright',
  'Rights',
];
const AI_ATTRIB_VALUE =
  /^(?:a\.?\s*i\.?|ai[\s_-]+(?:generated|created|artist|art)|generated[\s_-]+by[\s_-]+ai|artificial[\s_-]+intelligence|ai[\s_-]+image)$/i;

const EXIFR_OPTS = {
  iptc: true,
  xmp: true,
  exif: true,
  jfif: false,
  icc: false,
  ihdr: false,
  mergeOutput: true,
  reviveValues: false,
};

async function exifrAttributionDetect(bytes) {
  let meta;
  try {
    meta = await exifr.parse(bytes, EXIFR_OPTS);
  } catch {
    return null;
  }
  if (!meta) return null;

  for (const field of ATTRIB_FIELDS) {
    const val = meta[field];
    const str = Array.isArray(val) ? val[0] : val;
    if (typeof str === 'string' && AI_ATTRIB_VALUE.test(str.trim())) {
      return {
        ai: true,
        reason: 'exif-attribution',
        detail: `${field}: ${str.trim()}`,
      };
    }
  }
  return null;
}

async function c2paDetect(url) {
  try {
    await ensureOffscreen();
    return await chrome.runtime.sendMessage({
      target: 'offscreen',
      type: 'c2pa-detect',
      url,
    });
  } catch (e) {
    dwarn(`${TAG} c2pa offscreen call failed for`, url, e);
    return { ai: false, reason: 'c2pa-call-failed', parsed: false };
  }
}

let creatingOffscreen;
async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
  });
  if (contexts.length > 0) return;
  if (creatingOffscreen) return creatingOffscreen;
  creatingOffscreen = chrome.offscreen.createDocument({
    url: chrome.runtime.getURL('dist/offscreen.html'),
    reasons: ['WORKERS'],
    justification: 'Run c2pa-js (which uses a Web Worker) for image AI provenance.',
  });
  try {
    await creatingOffscreen;
  } finally {
    creatingOffscreen = null;
  }
}

// Decode bytes as latin1 and strip nulls so we also match UTF-16-encoded EXIF
// UserComment text (chars become "S\0a\0m\0p..." in the raw byte view).
function decodeBytes(bytes) {
  return new TextDecoder('latin1').decode(bytes).replace(/\x00/g, '');
}

function byteScanDetect(bytes) {
  const text = decodeBytes(bytes);

  // Tier A: IPTC / C2PA digital source type — strongest signal
  const dst = text.match(
    /(?:trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|algorithmicMedia)/,
  );
  if (dst) {
    return { ai: true, reason: 'byte-scan-digital-source-type', detail: dst[0] };
  }

  // Tier B: known AI claim generators / software agents
  // Patterns kept here are "specific enough not to false-positive in random
  // binary." Rejected for being too short / too common:
  //   - 'Gemini' / 'Imagen' alone (English/Spanish words). Caught via 'Google C2PA' / 'Google AI'.
  //   - 'FLUX' alone (4 chars, common word). Caught via 'black-forest-labs'.
  //   - 'xAI' / 'Grok' (Grok strips all metadata, so byte-scan can never legitimately catch it).
  const agents = [
    { name: 'Midjourney', re: /Midjourney/ },
    { name: 'OpenAI / ChatGPT / DALL·E', re: /\b(?:ChatGPT|OpenAI|DALL[-·]?E)\b/ },
    { name: 'Adobe Firefly', re: /Adobe[_ ]?Firefly|Firefly/ },
    { name: 'Microsoft Image Creator', re: /Microsoft[_ ]Responsible[_ ]AI|Image Creator from Designer/ },
    { name: 'Google Gemini / Imagen', re: /Google C2PA|Google AI/ },
    { name: 'Stable Diffusion', re: /Stable[_ ]?Diffusion|Automatic1111|ComfyUI|InvokeAI/ },
    { name: 'Leonardo.ai', re: /Leonardo\.ai/ },
    { name: 'NovelAI', re: /NovelAI/ },
    { name: 'Runway', re: /Runway[_ ]?ML|RunwayML/ },
    { name: 'Ideogram', re: /Ideogram/ },
    { name: 'FLUX', re: /black-forest-labs/i },
  ];
  for (const { name, re } of agents) {
    const m = text.match(re);
    if (m) {
      return { ai: true, reason: 'byte-scan-agent-name', detail: `${name} (matched "${m[0]}")` };
    }
  }

  // Tier C: AI generation parameters in EXIF UserComment, PNG tEXt, XMP, etc.
  // These cover Civitai / A1111 / ComfyUI / OpenAI workflow dumps.
  const promptPatterns = [
    { name: 'A1111 generation params', re: /Steps:\s*\d+\s*,\s*Sampler:/i },
    { name: 'CFG scale parameter', re: /CFG\s+scale:\s*-?\d/i },
    { name: 'Sampler parameter', re: /Sampler:\s*[A-Za-z0-9._-]{3,}/ },
    { name: 'negative prompt', re: /\bnegative[_\s-]prompt\b/i },
    { name: 'Civitai metadata', re: /\bcivitai\b/i },
    { name: 'OpenAI variant tag', re: /openaiVariant/i },
    { name: 'GPT-Image model', re: /\bgpt[-_]image[-_]?\d/i },
    { name: 'txt2img / img2img workflow', re: /\b(?:txt2img|img2img)\b/i },
    { name: 'model hash', re: /\bmodel[_-]hash\s*[:=]/i },
    { name: 'prompt hash', re: /\bpromptHash\b/ },
    { name: 'ComfyUI workflow', re: /ComfyUI/ },
    { name: 'JSON prompt field', re: /"prompt"\s*:\s*"[^"]{8,}/ },
  ];
  for (const { name, re } of promptPatterns) {
    const m = text.match(re);
    if (m) {
      const snippet = m[0].length > 60 ? m[0].slice(0, 60) + '…' : m[0];
      return { ai: true, reason: 'byte-scan-prompt-params', detail: `${name} (matched "${snippet}")` };
    }
  }

  return { ai: false, reason: 'no-ai-markers' };
}

// ---- Debug-only per-tier metadata summary ------------------------------------
// In debug mode, every image gets a compact readout of what each metadata tier
// found — regardless of the final verdict — rendered on the image by content.js.

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + '…' : s;
}

// Each summarizer returns { text, status } where status is one of
// 'clean' | 'ai' | 'fail'. content.js colors the row by status (ai → red,
// fail → grey), so non-clean tiers stand out in the debug table.

function summarizeC2pa(r) {
  switch (r?.reason) {
    case 'c2pa':
      return { text: `AI manifest — ${r.detail}`, status: 'ai' };
    case 'c2pa-no-ai-marker':
      return { text: 'signed, no AI assertion', status: 'clean' };
    case 'no-c2pa-manifest':
      return { text: 'no manifest', status: 'clean' };
    case 'c2pa-error':
      return { text: `parse error (${r.detail || '?'})`, status: 'fail' };
    case 'fetch-failed':
    case 'fetch-error':
    case 'c2pa-call-failed':
      return { text: `unavailable (${r.reason})`, status: 'fail' };
    default:
      return { text: r?.reason || 'unknown', status: 'clean' };
  }
}

async function summarizeExif(bytes) {
  let meta;
  try {
    meta = await exifr.parse(bytes, EXIFR_OPTS);
  } catch {
    return { text: 'parse error', status: 'fail' };
  }
  if (!meta) return { text: 'no EXIF/IPTC/XMP', status: 'clean' };

  const present = [];
  let aiField = null;
  for (const field of ATTRIB_FIELDS) {
    const val = meta[field];
    const str = Array.isArray(val) ? val[0] : val;
    if (typeof str === 'string' && str.trim()) {
      present.push(`${field}="${truncate(str.trim(), 24)}"`);
      if (!aiField && AI_ATTRIB_VALUE.test(str.trim())) aiField = field;
    }
  }
  if (aiField) return { text: `AI value in ${aiField} ✓ — ${present.join(', ')}`, status: 'ai' };
  if (present.length) return { text: `no AI value — ${present.join(', ')}`, status: 'clean' };
  return { text: 'no attribution fields', status: 'clean' };
}

function summarizeByteScan(bytes) {
  const r = byteScanDetect(bytes);
  if (r.ai) return { text: `${r.reason.replace(/^byte-scan-/, '')} — ${r.detail}`, status: 'ai' };
  return { text: 'clean', status: 'clean' };
}

// Declared Google/SynthID provenance found in *metadata*. This is the invisible
// watermark's metadata cousin — labeled "(metadata only)" so it's never confused
// with the actual tier-4 watermark detection (summarizeSynthIdWatermark below).
function summarizeSynthId(text, c2paResult) {
  const hits = [];
  if (/synthid/i.test(text)) hits.push('"SynthID" string');
  if (/Google C2PA|Made with Google|Google AI/i.test(text)) hits.push('Google AI marker');
  if (c2paResult?.reason === 'c2pa' && /google/i.test(c2paResult.detail || '')) {
    hits.push('Google C2PA manifest');
  }
  return hits.length
    ? { text: `${hits.join(', ')} (metadata only)`, status: 'ai' }
    : { text: 'no Google/SynthID markers in metadata', status: 'clean' };
}

// Tier-4 SynthID watermark surrogate result (the actual ONNX detection, not
// metadata). `r` is the synthidDetect() result, or null if the tier didn't run.
// Compact " · prep Nms · infer Nms[ · load Nms]" suffix for debug-table timing
// rows. Shared by the SynthID-watermark and Visual summarizers so both read the
// same. `load` (one-time session-load cost) only shows when non-trivial.
function fmtTiming(t) {
  if (!t || t.prepMs == null || t.inferMs == null) return '';
  const load = (t.loadMs != null && t.loadMs >= 1) ? ` · load ${t.loadMs.toFixed(0)}ms` : '';
  return ` · prep ${t.prepMs.toFixed(0)}ms · infer ${t.inferMs.toFixed(0)}ms${load}`;
}

function summarizeSynthIdWatermark(r) {
  if (!r) return { text: 'not run', status: 'clean' };
  switch (r.reason) {
    case 'synthid-watermark':
      return { text: `${r.detail || 'watermark detected'}${fmtTiming(r.timing)}`, status: 'ai' };
    case 'synthid-below-threshold':
      return { text: `${r.detail || 'below threshold'}${fmtTiming(r.timing)}`, status: 'clean' };
    case 'synthid-fetch-failed':
    case 'synthid-fetch-error':
    case 'synthid-error':
    case 'synthid-call-failed':
      return { text: `unavailable (${r.reason})`, status: 'fail' };
    default:
      return { text: r.detail ? `${r.reason}: ${r.detail}` : r.reason || 'unknown', status: 'clean' };
  }
}

// Returns an array of { tier, text, status } rows for the debug table.
async function buildMetaChecks(bytes, c2paResult, synthidResult, visualResult) {
  const lines = [{ tier: 'C2PA', ...summarizeC2pa(c2paResult) }];
  if (!bytes) {
    // In fast debug a C2PA AI hit short-circuits before bytes are fetched, so the
    // byte-backed tiers were never reached (not failed). Bytes are otherwise only
    // null when the fetch genuinely failed.
    const skipped = !!c2paResult.ai;
    const text = skipped ? '(not reached)' : '(fetch failed)';
    const status = skipped ? 'clean' : 'fail';
    lines.push({ tier: 'EXIF', text, status });
    lines.push({ tier: 'Bytes', text, status });
    lines.push({ tier: 'SynthID(meta)', ...summarizeSynthId('', c2paResult) });
    lines.push({ tier: 'SynthID(watermark)', ...summarizeSynthIdWatermark(synthidResult) });
    lines.push({ tier: 'Visual', ...summarizeVisual(visualResult) });
    return lines;
  }
  lines.push({ tier: 'EXIF', ...(await summarizeExif(bytes)) });
  lines.push({ tier: 'Bytes', ...summarizeByteScan(bytes) });
  lines.push({ tier: 'SynthID(meta)', ...summarizeSynthId(decodeBytes(bytes), c2paResult) });
  lines.push({ tier: 'SynthID(watermark)', ...summarizeSynthIdWatermark(synthidResult) });
  lines.push({ tier: 'Visual', ...summarizeVisual(visualResult) });
  return lines;
}

// Tier-5 (OpenFake SwinV2) debug row: P(AI), which EP backed the session, and
// per-image prep/infer/load timing (see fmtTiming). Falls back gracefully if the
// offscreen result predates timing.
function summarizeVisual(visualResult) {
  // null/undefined → the cascade never reached this tier (fast debug short-circuit),
  // which is not a failure. An actual error is carried on the result object below.
  if (!visualResult) return { text: '(not run)', status: 'clean' };
  const m = visualResult.models?.[0];
  if (!m || m.error) {
    return { text: m?.error ? `error: ${m.error}` : '(no result)', status: 'fail' };
  }
  const ep = m.ep ? `[${m.ep}] ` : '';
  const p = `P=${(m.probAI * 100).toFixed(1)}%`;
  return {
    text: `${m.name} ${ep}${p}${fmtTiming(m)}`.replace(/\s+/g, ' ').trim(),
    status: visualResult.ai ? 'ai' : 'clean',
  };
}

async function fetchBytes(url) {
  try {
    const resp = await fetch(url, { credentials: 'include' });
    if (!resp.ok) return null;
    const buf = await resp.arrayBuffer();
    if (buf.byteLength > MAX_FETCH) {
      return new Uint8Array(buf, 0, MAX_FETCH);
    }
    return new Uint8Array(buf);
  } catch {
    return null;
  }
}
