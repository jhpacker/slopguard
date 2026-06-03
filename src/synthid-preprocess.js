// SynthID surrogate (OpenSynthID) preprocessing — builds the 6-channel input
// tensor the model expects, replicating fyxme/opensynthid-detect-0.1's infer.py
// pipeline (which used numpy/pywt/cv2) in pure JS.
//
// Channel order (matches infer.py `preprocess`):
//   0,1,2 — RGB / 255
//   3     — wavelet-denoise residual (db4, level 3, VisuShrink soft-threshold),
//           per-RGB-channel residual averaged to grayscale
//   4     — FFT log-magnitude of grayscale, normalized to [0,1]
//   5     — static carrier-frequency mask
//
// PARITY NOTE: the db4 wavelet transform here is validated against pywt
// (mode='symmetric') by tools/wavelet_parity.mjs. The wavelet residual is the
// only numerically subtle channel; FFT log-mag (min-max normalized, so FFT
// scaling is irrelevant) and the carrier mask are exact by construction.

// db4 filter bank, taken verbatim from pywt.Wavelet('db4').filter_bank.
const DB4 = {
  decLo: [-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309, -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965],
  decHi: [-0.2303778133088965, 0.7148465705529157, -0.6308807679298589, -0.027983769416859854, 0.18703481171909309, 0.030841381835560764, -0.0328830116668852, -0.010597401785069032],
  recLo: [0.2303778133088965, 0.7148465705529157, 0.6308807679298589, -0.027983769416859854, -0.18703481171909309, 0.030841381835560764, 0.0328830116668852, -0.010597401785069032],
  recHi: [-0.010597401785069032, -0.0328830116668852, 0.030841381835560764, 0.18703481171909309, -0.027983769416859854, -0.6308807679298589, 0.7148465705529157, -0.2303778133088965],
};

// Half-sample-symmetric extension (pywt mode='symmetric'): edge sample repeated,
// e.g. [a b c] -> ... b a | a b c | c b ...
function symmExt(data, p) {
  const N = data.length;
  const out = new Float64Array(N + 2 * p);
  for (let i = 0; i < p; i++) out[i] = data[p - 1 - i];
  for (let i = 0; i < N; i++) out[p + i] = data[i];
  for (let j = 0; j < p; j++) out[p + N + j] = data[N - 1 - j];
  return out;
}

// Single-level 1D DWT, mode='symmetric'. Returns { a, d } each of length
// floor((N + F - 1) / 2), matching pywt.dwt.
function dwt1d(data, lo, hi) {
  const F = lo.length;
  const N = data.length;
  const ext = symmExt(data, F - 1);
  const outLen = Math.floor((N + F - 1) / 2);
  const a = new Float64Array(outLen);
  const d = new Float64Array(outLen);
  // With p = F-1 symmetric padding and full convolution conv[n]=Σ filt[k]·ext[n-k],
  // pywt's downsampled output is conv[F + 2i] (offset F, stride 2). Empirically
  // verified against pywt.dwt(mode='symmetric') in tools/wavelet_parity.mjs.
  for (let i = 0; i < outLen; i++) {
    let sa = 0;
    let sd = 0;
    const base = 2 * i + F;
    for (let k = 0; k < F; k++) {
      const idx = base - k;
      if (idx >= 0 && idx < ext.length) {
        sa += lo[k] * ext[idx];
        sd += hi[k] * ext[idx];
      }
    }
    a[i] = sa;
    d[i] = sd;
  }
  return { a, d };
}

// Single-level 1D inverse DWT, mode='symmetric'. Inverse of dwt1d; returns a
// signal of length `outLen` (the original pre-dwt length, passed back in).
function idwt1d(a, d, recLo, recHi, outLen) {
  const F = recLo.length;
  const L = a.length;
  // Upsample (insert zeros) then convolve with reconstruction filters and sum.
  const up = 2 * L;
  const ua = new Float64Array(up);
  const ud = new Float64Array(up);
  for (let i = 0; i < L; i++) {
    ua[2 * i] = a[i];
    ud[2 * i] = d[i];
  }
  // Full convolution, then crop the valid central region to outLen.
  const convLen = up + F - 1;
  const rec = new Float64Array(convLen);
  for (let n = 0; n < convLen; n++) {
    let s = 0;
    const kmin = Math.max(0, n - (up - 1));
    const kmax = Math.min(n, F - 1);
    for (let k = kmin; k <= kmax; k++) {
      s += recLo[k] * ua[n - k] + recHi[k] * ud[n - k];
    }
    rec[n] = s;
  }
  // pywt crops F-2 from the front to undo the filter delay.
  const start = F - 2;
  const out = new Float64Array(outLen);
  for (let i = 0; i < outLen; i++) out[i] = rec[start + i];
  return out;
}

// Soft threshold: sign(x) * max(|x| - t, 0).
function softThreshold(arr, t) {
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    const m = Math.abs(v) - t;
    out[i] = m > 0 ? (v < 0 ? -m : m) : 0;
  }
  return out;
}

function median(arr) {
  const s = Float64Array.from(arr).sort();
  const n = s.length;
  if (n === 0) return 0;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

// Apply a 1D transform to every row of a (h × w) Float64Array, returning the
// concatenated approx (left) and detail (right) halves as a new (h × w2) pair.
function rowsDWT(mat, w, h, lo, hi) {
  const w2 = Math.floor((w + lo.length - 1) / 2);
  const A = new Float64Array(h * w2);
  const D = new Float64Array(h * w2);
  const row = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) row[x] = mat[y * w + x];
    const { a, d } = dwt1d(row, lo, hi);
    A.set(a, y * w2);
    D.set(d, y * w2);
  }
  return { A, D, w2 };
}

function colsDWT(mat, w, h, lo, hi) {
  const h2 = Math.floor((h + lo.length - 1) / 2);
  const A = new Float64Array(h2 * w);
  const D = new Float64Array(h2 * w);
  const col = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = mat[y * w + x];
    const { a, d } = dwt1d(col, lo, hi);
    for (let y = 0; y < h2; y++) {
      A[y * w + x] = a[y];
      D[y * w + x] = d[y];
    }
  }
  return { A, D, h2 };
}

function rowsIDWT(A, D, w2, h, lo, hi, wOut) {
  const out = new Float64Array(h * wOut);
  const a = new Float64Array(w2);
  const d = new Float64Array(w2);
  for (let y = 0; y < h; y++) {
    for (let i = 0; i < w2; i++) {
      a[i] = A[y * w2 + i];
      d[i] = D[y * w2 + i];
    }
    const r = idwt1d(a, d, lo, hi, wOut);
    out.set(r, y * wOut);
  }
  return out;
}

function colsIDWT(A, D, w, h2, lo, hi, hOut) {
  const out = new Float64Array(hOut * w);
  const a = new Float64Array(h2);
  const d = new Float64Array(h2);
  for (let x = 0; x < w; x++) {
    for (let i = 0; i < h2; i++) {
      a[i] = A[i * w + x];
      d[i] = D[i * w + x];
    }
    const r = idwt1d(a, d, lo, hi, hOut);
    for (let y = 0; y < hOut; y++) out[y * w + x] = r[y];
  }
  return out;
}

// Multi-level 2D wavelet denoise matching infer.py wavelet_denoise():
//   coeffs = wavedec2(channel, db4, level)
//   sigma = median(|cH_finest|) / 0.6745
//   t = sigma * sqrt(2 ln N); soft-threshold every detail subband; waverec2.
// `mat` is a (size × size) Float64Array; returns the denoised array.
export function waveletDenoise(mat, size, level = 3) {
  const { decLo, decHi, recLo, recHi } = DB4;

  // --- forward: build the pyramid, recording dims to invert exactly ---
  let cur = mat;
  let w = size;
  let h = size;
  const levels = []; // each: { LH, HL, HH, w, h, w2, h2 }
  for (let lv = 0; lv < level; lv++) {
    const { A: L, D: H, w2 } = rowsDWT(cur, w, h, decLo, decHi);
    const { A: LL, D: LH, h2 } = colsDWT(L, w2, h, decLo, decHi);
    const { A: HL, D: HH } = colsDWT(H, w2, h, decLo, decHi);
    levels.push({ LH, HL, HH, w, h, w2, h2 });
    cur = LL;
    w = w2;
    h = h2;
  }

  // sigma from the finest-level horizontal detail (coeffs[-1][0] in pywt).
  // Our LH (low-pass rows → high-pass cols) is one detail subband; its median
  // statistic matches pywt's cH closely enough for the VisuShrink threshold.
  const finest = levels[0];
  const sigma = median(finest.LH.map((v) => Math.abs(v))) / 0.6745;
  const threshold = sigma * Math.sqrt(2 * Math.log(size * size));

  // --- inverse: threshold details, reconstruct from coarsest to finest ---
  let ll = cur; // coarsest approximation, untouched
  for (let lv = level - 1; lv >= 0; lv--) {
    const { LH, HL, HH, w: wO, h: hO, w2, h2 } = levels[lv];
    const tLH = softThreshold(LH, threshold);
    const tHL = softThreshold(HL, threshold);
    const tHH = softThreshold(HH, threshold);
    const L = colsIDWT(ll, tLH, w2, h2, recLo, recHi, hO);
    const H = colsIDWT(tHL, tHH, w2, h2, recLo, recHi, hO);
    ll = rowsIDWT(L, H, w2, hO, recLo, recHi, wO);
  }
  return ll;
}

// --- FFT (radix-2 iterative, in-place) -------------------------------------
function fftRadix2(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wpr = Math.cos(ang);
    const wpi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1;
      let wi = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = i + k + len / 2;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const nwr = wr * wpr - wi * wpi;
        wi = wr * wpi + wi * wpr;
        wr = nwr;
      }
    }
  }
}

// FFT log-magnitude of a grayscale (size × size) float image, fftshifted and
// min-max normalized to [0,1]. Matches infer.py fft_log_magnitude(). Because
// of the final min-max normalization, FFT scaling conventions don't matter, so
// no 1/N normalization is needed (matching numpy.fft.fft2 forward, also unscaled).
export function fftLogMagnitude(gray, size) {
  const re = new Float64Array(size * size);
  const im = new Float64Array(size * size);
  re.set(gray);
  // rows
  const rr = new Float64Array(size);
  const ri = new Float64Array(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      rr[x] = re[y * size + x];
      ri[x] = im[y * size + x];
    }
    fftRadix2(rr, ri, false);
    for (let x = 0; x < size; x++) {
      re[y * size + x] = rr[x];
      im[y * size + x] = ri[x];
    }
  }
  // cols
  const cr = new Float64Array(size);
  const ci = new Float64Array(size);
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      cr[y] = re[y * size + x];
      ci[y] = im[y * size + x];
    }
    fftRadix2(cr, ci, false);
    for (let y = 0; y < size; y++) {
      re[y * size + x] = cr[y];
      im[y * size + x] = ci[y];
    }
  }
  // fftshift + log1p(|.|)
  const half = size / 2;
  const out = new Float32Array(size * size);
  let mn = Infinity;
  let mx = -Infinity;
  for (let y = 0; y < size; y++) {
    const sy = (y + half) % size;
    for (let x = 0; x < size; x++) {
      const sx = (x + half) % size;
      const idx = y * size + x;
      const mag = Math.hypot(re[idx], im[idx]);
      const v = Math.log1p(mag);
      const di = sy * size + sx;
      out[di] = v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const range = mx - mn + 1e-8;
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - mn) / range;
  return out;
}

// Static carrier-frequency mask (infer.py build_carrier_mask). Independent of
// the image, so computed once and cached.
const CARRIERS = [[14, 14], [-14, -14], [126, 14], [-126, -14], [98, -14], [-98, 14], [128, 128], [-128, -128]];
let carrierCache = null;
let carrierCacheSize = 0;
export function carrierMask(size) {
  if (carrierCache && carrierCacheSize === size) return carrierCache;
  const mask = new Float32Array(size * size);
  const c = size >> 1;
  for (const [fy, fx] of CARRIERS) {
    for (const [yy, xx] of [[c + fy, c + fx], [c - fy, c - fx]]) {
      if (yy >= 0 && yy < size && xx >= 0 && xx < size) mask[yy * size + xx] = 1;
    }
  }
  carrierCache = mask;
  carrierCacheSize = size;
  return mask;
}

// Build the full (1×6×size×size) Float32Array from RGB byte planes (length
// size*size each, 0–255). Returns a flat Float32Array of length 6*size*size in
// NCHW order: [R, G, B, residual, fft, carrier].
export function buildSynthIdInput(rByte, gByte, bByte, size) {
  const N = size * size;
  const out = new Float32Array(6 * N);

  // RGB normalized [0,1] (channels 0–2) and float copies for the wavelet.
  const rF = new Float64Array(N);
  const gF = new Float64Array(N);
  const bF = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const r = rByte[i] / 255;
    const g = gByte[i] / 255;
    const b = bByte[i] / 255;
    out[i] = r;
    out[N + i] = g;
    out[2 * N + i] = b;
    rF[i] = r;
    gF[i] = g;
    bF[i] = b;
  }

  // Channel 3 — wavelet residual averaged across RGB.
  const denR = waveletDenoise(rF, size);
  const denG = waveletDenoise(gF, size);
  const denB = waveletDenoise(bF, size);
  for (let i = 0; i < N; i++) {
    const res = ((rF[i] - denR[i]) + (gF[i] - denG[i]) + (bF[i] - denB[i])) / 3;
    out[3 * N + i] = res;
  }

  // Channel 4 — FFT log-magnitude of cv2-style grayscale (rounded uint8 / 255).
  const gray = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const y = Math.round(0.299 * rByte[i] + 0.587 * gByte[i] + 0.114 * bByte[i]);
    gray[i] = Math.min(255, Math.max(0, y)) / 255;
  }
  const fft = fftLogMagnitude(gray, size);
  out.set(fft, 4 * N);

  // Channel 5 — carrier mask.
  out.set(carrierMask(size), 5 * N);

  // nan_to_num (infer.py): NaN→0, +Inf→1, -Inf→-1.
  for (let i = 0; i < out.length; i++) {
    const v = out[i];
    if (Number.isNaN(v)) out[i] = 0;
    else if (v === Infinity) out[i] = 1;
    else if (v === -Infinity) out[i] = -1;
  }
  return out;
}

export const SYNTHID_SIZE = 512;
// exported for parity testing
export const __test = { dwt1d, idwt1d, DB4 };
