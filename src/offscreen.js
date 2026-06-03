import { createC2pa } from 'c2pa';
import * as ort from 'onnxruntime-web';
import { buildSynthIdInput, SYNTHID_SIZE } from './synthid-preprocess.js';

const TAG = '[SlopGuard off]';

let debugMode = false;
try {
  chrome.storage?.local?.get({ debugMode: false }, (s) => {
    debugMode = !!(s && s.debugMode);
    if (debugMode) console.log(`${TAG} offscreen document loaded`);
  });
  chrome.storage?.onChanged?.addListener((changes) => {
    if (changes.debugMode) debugMode = !!changes.debugMode.newValue;
  });
} catch (e) {
  console.warn(`${TAG} chrome.storage unavailable — has the extension been reloaded since "storage" permission was added?`, e);
}
const dlog = (...args) => debugMode && console.log(...args);
const dwarn = (...args) => debugMode && console.warn(...args);

// Point ONNX runtime at our locally-bundled WASM files (avoids CSP issues
// with cross-origin script/worker loading).
ort.env.wasm.wasmPaths = chrome.runtime.getURL('dist/');

// Visual classifiers — three-model ensemble, all SigLIP/Swin classifiers
// taking a single 224×224 image. OR-aggregated: any model crossing its
// threshold flags the image.
//
//   Organika SDXL    → Swin Transformer, ImageNet norm. Softmax over
//                      [artificial, human]. Strong on Bing / ChatGPT / Firefly.
//   Siglip2 Deepfake → SigLIP2-base fine-tune. Softmax over [Fake, Real].
//   AI-vs-Human SigLIP → SigLIP-base fine-tune. Softmax over [ai, hum].
//
// All three use AI index 0 and HF's `logits` output. SigLIP models use
// normalization mean/std = [0.5, 0.5, 0.5]; Organika uses ImageNet stats.
//
// `threshold`        — minimum P(AI) to flag the image
// `confidentThreshold` — above this, "Probably AI" (red); below, "Maybe AI" (yellow)
function softmaxBinary(tensor, aiIndex) {
  const a = tensor.data[aiIndex];
  const b = tensor.data[1 - aiIndex];
  const m = Math.max(a, b);
  return Math.exp(a - m) / (Math.exp(a - m) + Math.exp(b - m));
}

const SIGLIP_MEAN = [0.5, 0.5, 0.5];
const SIGLIP_STD = [0.5, 0.5, 0.5];

const VISUAL_MODELS = [
  // The original three classifiers are DISABLED — they produced too many
  // real-photo false positives in the wild. Currently trialing OpenFake
  // SwinV2 alone (below). To restore the full OR-aggregated ensemble,
  // uncomment these three. Their .onnx files are still copied to dist/.
  /*
  {
    name: 'Organika SDXL',
    url: chrome.runtime.getURL('dist/models/Organika/model.onnx'),
    inputSize: 224,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    inputName: 'pixel_values',
    outputName: 'logits',
    interpret: (tensor) => softmaxBinary(tensor, 0),
    threshold: 0.7,
    confidentThreshold: 0.85,
  },
  {
    name: 'Siglip2 Deepfake',
    url: chrome.runtime.getURL('dist/models/Siglip2-Deepfake/model.onnx'),
    inputSize: 224,
    mean: SIGLIP_MEAN,
    std: SIGLIP_STD,
    inputName: 'pixel_values',
    outputName: 'logits',
    interpret: (tensor) => softmaxBinary(tensor, 0),
    threshold: 0.7,
    confidentThreshold: 0.85,
  },
  {
    name: 'AI-vs-Human SigLIP',
    url: chrome.runtime.getURL('dist/models/AIvHuman/model.onnx'),
    inputSize: 224,
    mean: SIGLIP_MEAN,
    std: SIGLIP_STD,
    inputName: 'pixel_values',
    outputName: 'logits',
    interpret: (tensor) => softmaxBinary(tensor, 0),
    threshold: 0.7,
    confidentThreshold: 0.85,
  },
  */
  {
    // ComplexDataLab/OpenFakeDemo — SwinV2-base fine-tune (real/fake). Unlike
    // the others this is a 256² input with ImageNet norm, and the AI class is
    // index 1 (labels [real, fake]). Logits only; JS softmaxes. The Space
    // applies a softmax temperature of 2.0 before reporting p_fake — not baked
    // in here, so raw P(AI) reads more confident than their demo number.
    name: 'OpenFake SwinV2',
    url: chrome.runtime.getURL('dist/models/OpenFake/model.onnx'),
    inputSize: 256,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    inputName: 'pixel_values',
    outputName: 'logits',
    interpret: (tensor) => softmaxBinary(tensor, 1),
    threshold: 0.7,
    confidentThreshold: 0.85,
  },
];

// SynthID watermark surrogate (fyxme/opensynthid-detect-0.1). Distinct from the
// VISUAL_MODELS ensemble: it doesn't ask "does this look AI" — it looks for
// Google's SynthID watermark specifically (present in Imagen/Gemini/Nano Banana
// output, and designed to survive moderate editing). Single (1,6,512,512) input
// built by buildSynthIdInput; sigmoid is fused into the graph so the output is
// P(watermark) directly. Threshold raised to 0.95 (up from the 0.5 model-card
// default) to suppress real-photo false positives seen in the wild; separation
// is wide in practice (Gemini ~0.99, real photos ~0.00) so true positives are
// unaffected. Note threshold > confidentThreshold means every hit is 'high'.
const SYNTHID_MODEL = {
  name: 'OpenSynthID',
  url: chrome.runtime.getURL('dist/models/OpenSynthID/model.onnx'),
  inputName: 'input',
  outputName: 'prob_ai',
  threshold: 0.95,
  confidentThreshold: 0.85,
};

const sessions = new Map();

function getSession(model) {
  if (!sessions.has(model.name)) {
    dlog(`${TAG} loading model ${model.name}`);
    const promise = ort.InferenceSession.create(model.url, {
      executionProviders: ['wasm'],
    })
      .then((s) => {
        dlog(`${TAG} ${model.name} ready`);
        return s;
      })
      .catch((e) => {
        console.error(`${TAG} ${model.name} failed to load:`, e);
        sessions.delete(model.name); // allow retry on next image
        throw e;
      });
    sessions.set(model.name, promise);
  }
  return sessions.get(model.name);
}

function preprocessGlobal(bitmap, model) {
  const canvas = new OffscreenCanvas(model.inputSize, model.inputSize);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, model.inputSize, model.inputSize);
  const imageData = ctx.getImageData(0, 0, model.inputSize, model.inputSize);

  const N = model.inputSize * model.inputSize;
  const out = new Float32Array(3 * N);
  const px = imageData.data;
  for (let i = 0; i < N; i++) {
    const r = px[i * 4 + 0] / 255;
    const g = px[i * 4 + 1] / 255;
    const b = px[i * 4 + 2] / 255;
    out[i] = (r - model.mean[0]) / model.std[0];
    out[N + i] = (g - model.mean[1]) / model.std[1];
    out[2 * N + i] = (b - model.mean[2]) / model.std[2];
  }
  return new ort.Tensor('float32', out, [1, 3, model.inputSize, model.inputSize]);
}

async function runSingleModel(blob, model) {
  let bitmap;
  try {
    const session = await getSession(model);
    bitmap = await createImageBitmap(blob);
    const tensor = preprocessGlobal(bitmap, model);
    const outputs = await session.run({ [model.inputName]: tensor });
    const probAI = model.interpret(outputs[model.outputName]);
    return {
      name: model.name,
      probAI,
      detected: probAI >= model.threshold,
      confident: probAI >= model.confidentThreshold,
    };
  } catch (e) {
    return { name: model.name, error: String(e?.message || e) };
  } finally {
    bitmap?.close();
  }
}

// Resize to 512×512 and split into R/G/B byte planes, then build the model's
// 6-channel input (RGB + wavelet residual + FFT log-mag + carrier mask) in JS.
function preprocessSynthId(bitmap) {
  const size = SYNTHID_SIZE;
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;

  const N = size * size;
  const r = new Uint8ClampedArray(N);
  const g = new Uint8ClampedArray(N);
  const b = new Uint8ClampedArray(N);
  for (let i = 0; i < N; i++) {
    r[i] = px[i * 4];
    g[i] = px[i * 4 + 1];
    b[i] = px[i * 4 + 2];
  }
  const data = buildSynthIdInput(r, g, b, size);
  return new ort.Tensor('float32', data, [1, 6, size, size]);
}

async function handleSynthIdDetect(url) {
  const t0 = performance.now();
  dlog(`${TAG} synthid received`, url);
  let blob;
  try {
    const resp = await fetch(url, { credentials: 'include' });
    if (!resp.ok) return { ai: false, reason: 'synthid-fetch-failed' };
    blob = await resp.blob();
  } catch (e) {
    return { ai: false, reason: 'synthid-fetch-error', detail: String(e?.message || e) };
  }

  let bitmap;
  try {
    const session = await getSession(SYNTHID_MODEL);
    bitmap = await createImageBitmap(blob);
    const tPre = performance.now();
    const tensor = preprocessSynthId(bitmap);
    const tInfer = performance.now();
    const outputs = await session.run({ [SYNTHID_MODEL.inputName]: tensor });
    const probAI = outputs[SYNTHID_MODEL.outputName].data[0];
    dlog(
      `${TAG} synthid P(watermark)=${(probAI * 100).toFixed(1)}% ` +
      `(prep ${(tInfer - tPre).toFixed(0)}ms, infer ${(performance.now() - tInfer).toFixed(0)}ms, ` +
      `total ${(performance.now() - t0).toFixed(0)}ms)`,
    );
    const detail = `SynthID watermark P=${(probAI * 100).toFixed(1)}%`;
    if (probAI >= SYNTHID_MODEL.threshold) {
      return {
        ai: true,
        reason: 'synthid-watermark',
        detail,
        visualTier: probAI >= SYNTHID_MODEL.confidentThreshold ? 'high' : 'low',
      };
    }
    return { ai: false, reason: 'synthid-below-threshold', detail };
  } catch (e) {
    dwarn(`${TAG} synthid inference failed for`, url, e);
    return { ai: false, reason: 'synthid-error', detail: String(e?.message || e) };
  } finally {
    bitmap?.close();
  }
}

let c2paPromise;
function getC2pa() {
  if (!c2paPromise) {
    c2paPromise = createC2pa({
      wasmSrc: chrome.runtime.getURL('dist/toolkit_bg.wasm'),
      workerSrc: chrome.runtime.getURL('dist/c2pa.worker.min.js'),
      // Local-only: never let image content trigger a network call. By default
      // c2pa-js fetches cloud-stored (remote) manifests referenced inside an
      // asset — a crafted image could point that URL anywhere (SSRF / "user
      // viewed this" beacon). We only read manifests embedded in the bytes we
      // already fetched, so disable remote fetch. ocspFetch is likewise off:
      // we ignore cert validity entirely, so no revocation network call either.
      fetchRemoteManifests: false,
      settings: { verify: { ocspFetch: false, remoteManifestFetch: false } },
    });
  }
  return c2paPromise;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target !== 'offscreen') return false;
  if (msg.type === 'c2pa-detect') {
    handleDetect(msg.url).then(sendResponse);
    return true;
  }
  if (msg.type === 'visual-detect') {
    handleVisualDetect(msg.url).then(sendResponse);
    return true;
  }
  if (msg.type === 'synthid-detect') {
    handleSynthIdDetect(msg.url).then(sendResponse);
    return true;
  }
  return false;
});

async function handleVisualDetect(url) {
  const t0 = performance.now();
  dlog(`${TAG} tier5 [visual] received`, url);
  let blob;
  try {
    const resp = await fetch(url, { credentials: 'include' });
    if (!resp.ok) return { ai: false, reason: 'visual-fetch-failed' };
    blob = await resp.blob();
    dlog(`${TAG} tier5 [visual] fetched ${blob.size}B in ${(performance.now() - t0).toFixed(0)}ms`);
  } catch (e) {
    return { ai: false, reason: 'visual-fetch-error', detail: String(e?.message || e) };
  }

  // Run all models in parallel. OR-aggregated: any one crossing its
  // detect threshold flags the image.
  const tInfer = performance.now();
  const results = await Promise.all(VISUAL_MODELS.map((m) => runSingleModel(blob, m)));
  dlog(`${TAG} tier5 [visual] inference done in ${(performance.now() - tInfer).toFixed(0)}ms (total ${(performance.now() - t0).toFixed(0)}ms)`);

  const summary = results
    .map((r) => (r.error ? `${r.name}=err` : `${r.name}=${(r.probAI * 100).toFixed(1)}%`))
    .join('\n');

  if (results.every((r) => r.error)) {
    return { ai: false, reason: 'visual-all-models-errored', detail: summary };
  }

  if (!results.some((r) => r.detected)) {
    return { ai: false, reason: 'visual-below-threshold', detail: summary };
  }

  // "Probably AI" if any model crossed its confidentThreshold; "Maybe AI" otherwise.
  return {
    ai: true,
    reason: 'visual-classifier',
    detail: summary,
    visualTier: results.some((r) => r.confident) ? 'high' : 'low',
  };
}

async function handleDetect(url) {
  let blob;
  try {
    const resp = await fetch(url, { credentials: 'include' });
    if (!resp.ok) return { ai: false, reason: 'fetch-failed', parsed: false };
    blob = await resp.blob();
  } catch (e) {
    return { ai: false, reason: 'fetch-error', detail: String(e?.message || e), parsed: false };
  }

  let result;
  try {
    const c2pa = await getC2pa();
    result = await c2pa.read(blob);
  } catch (e) {
    dwarn(`${TAG} c2pa.read threw for`, url, e);
    return { ai: false, reason: 'c2pa-error', detail: String(e?.message || e), parsed: false };
  }

  const store = result?.manifestStore;
  if (!store?.activeManifest) {
    return { ai: false, reason: 'no-c2pa-manifest', parsed: false };
  }

  // Scan EVERY manifest in the store, not just the active one. Google's
  // edit-chain images (e.g. created-by-AI → resized → visible-watermark →
  // converted-to-png) carry the trainedAlgorithmicMedia marker on an
  // *ingredient* manifest; the active manifest's actions only describe the
  // last edit (c2pa.edited/composite, c2pa.converted) and have no AI source
  // type. So the AI provenance is provable, just not on activeManifest.
  // Active manifest first so its generator wins for the simple single-manifest case.
  const active = store.activeManifest;
  const manifests = [active, ...Object.values(store.manifests || {}).filter((m) => m !== active)];

  for (const manifest of manifests) {
    const generator = manifest.claimGenerator || 'unknown';
    // C2PA actions live under either the v1 label ('c2pa.actions') or the newer
    // v2 label ('c2pa.actions.v2'). .get() is an exact-label match, so we must
    // query both — Google Generative AI / SynthID images use the .v2 label.
    const actionsList = [
      ...(manifest.assertions?.get('c2pa.actions.v2') || []),
      ...(manifest.assertions?.get('c2pa.actions') || []),
    ];

    for (const assertion of actionsList) {
      const actions = assertion?.data?.actions || [];
      for (const action of actions) {
        const dst = action.digitalSourceType || '';
        if (/trainedAlgorithm|algorithmicMedia/i.test(dst)) {
          const dstShort = String(dst).split('/').pop();
          const agent = action.softwareAgent
            ? (typeof action.softwareAgent === 'string'
                ? action.softwareAgent
                : action.softwareAgent.name)
            : null;
          return {
            ai: true,
            reason: 'c2pa',
            detail: `${generator}${agent ? ` / ${agent}` : ''} (${dstShort})`,
            parsed: true,
            validation: store.validationStatus,
          };
        }
      }
    }
  }

  return { ai: false, reason: 'c2pa-no-ai-marker', parsed: true };
}
