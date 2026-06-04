# SlopGuard

An extension that detects most AI-generated images on web pages and visually flags them.

When you click the toolbar icon (the six-finger hand icon) on any page, SlopGuard runs every eligible image (rendered area ≥ ~200×200 px, images that are loaded on-screen) against a series of checks including:
- Metadata extraction
- Check for Google's SynthID invisible watermark
- A visual classifier (i.e. a local model that says an image looks like AI)

You can also right-click on any image to check just that one image.

AI-flagged images get a red **AI** / **Probably AI** / yellow **Maybe AI** label and are dimmed to grayscale; checked-clean images get a thin green outline; failed checks get a grey outline.

**Everything runs locally** — no remote API calls, no user-supplied keys. This means it's fast and private.

**Visual Identification of AI images is a best guess, there will be misses and false positives**

The two on-device detectors are based upon others' work, so thanks are due to:
**Victor Livernoche @ ComplexDataLab**'s [OpenFake](https://github.com/vicliv/OpenFake) project for the general classifier.

**fyxme**'s [`fyxme/opensynthid-detect-0.1`](https://huggingface.co/fyxme/opensynthid-detect-0.1) local SynthID surrogate. 

## HOW IT WORKS

| # | Tier | What it checks |
|---|---|---|
| 1 | **C2PA manifest** (c2pa-js, ignores cert validation) | `c2pa.actions(.v2)` assertions with a `digitalSourceType` of `trainedAlgorithmicMedia` etc., walking the **whole** manifest chain (the AI marker usually lives on an ingredient/parent manifest, not the active one)|
| 2 | **EXIF/IPTC/XMP attribution** (exifr) | Artist / Author / Credit / Source values matching AI patterns (`"ai"`, `"AI Generated"`, etc.)
| 3 | **Byte scan** (regex over latin1-decoded bytes) | DigitalSourceType URL strings, known AI agent names (Midjourney, ChatGPT, Adobe_Firefly, Microsoft Responsible AI, Google C2PA, ...), generation params (`Sampler:`, `CFG scale:`, `civitai`, `ComfyUI`, `txt2img`, ...)|
| 4 | **SynthID watermark surrogate** (OpenSynthID ONNX, onnxruntime-web) | The Google **SynthID** watermark directly in the pixels — a reverse-engineered surrogate ([`fyxme/opensynthid-detect-0.1`](https://huggingface.co/fyxme/opensynthid-detect-0.1)).|
| 5 | **Visual classifier** (OpenFake SwinV2 ONNX, onnxruntime-web) | General "does this look AI-generated" — a SwinV2 real/fake detector ([`ComplexDataLab/OpenFakeDemo`](https://huggingface.co/spaces/ComplexDataLab/OpenFakeDemo)) at 256×256.|


**Tiers 1–4 short-circuit on hit.** Tiers 1 & 2 are declared signals (the file literally claims it's AI); tier 4 reads the SynthID watermark, treated as strong provenance. All four produce the red **AI** label.

**Tier 5 fires when OpenFake crosses its detect threshold** → red **Probably AI** if it also crosses the confident threshold, otherwise yellow **Maybe AI**. A failure to evaluate gets a grey outline.

## Model weights

The ONNX classifier weights are **not committed to the repo** — they're large
(hundreds of MB) and fully reproducible from their upstream sources, so they're
git-ignored along with `models/`. `npm run build` expects them to exist at:

- `models/OpenSynthID/model.onnx` — tier 4, SynthID watermark surrogate
  (from [`fyxme/opensynthid-detect-0.1`](https://huggingface.co/fyxme/opensynthid-detect-0.1))
- `models/OpenFake/model.onnx` — tier 5, OpenFake SwinV2 real/fake detector
  (from the [`ComplexDataLab/OpenFakeDemo`](https://huggingface.co/spaces/ComplexDataLab/OpenFakeDemo) Space)

To fetch the upstream weights and convert them to ONNX, run:

```bash
./build-models.sh            # fresh clone: download sources + convert to ONNX
./build-models.sh --refresh  # also re-pull upstream weights (after a maintainer update)
```

The script creates a self-contained Python venv (`.venv-convert/`, git-ignored),
installs the conversion deps, downloads the source checkpoints, and runs
`convert_model.py` for each model. It needs a Python with `torch` wheels
available (prefers `python3.13`; override with `PYTHON=…`). Run it once before
the first `npm run build`.

## Configuration

The Options page (right-click toolbar icon → **Options**) has one setting:

- **Debug mode** — when on, all three contexts (page, service worker, offscreen document) log per-image detection results to their consoles, and AI-flagged overlays show a small monospace line under the label with the rule + per-model score.

## Limitations (be aware)

- **SynthID detection uses an unvalidated community surrogate.** Google's official SynthID Detector is not publicly available. Tier 4 uses `fyxme/opensynthid-detect-0.1`, a reverse-engineered surrogate (v0.1, author discloses it's unvalidated).

> **SynthID** is a watermarking technology developed by [Google DeepMind](https://deepmind.google/models/synthid/). SlopGuard is not affiliated with, endorsed by, or connected to Google or DeepMind. Tier 4 does not use Google's SynthID software; it relies on `fyxme/opensynthid-detect-0.1`, an independent third-party model that attempts to *detect* the watermark's statistical signature. "SynthID" is used here for identification and descriptive purposes only.
- **Grok metadata is stripped.** Real Grok images carry no metadata, so tiers 1–3 can't catch them. The tier-5 classifier catches some.

- **No background-image detection.** Only `<img>` elements are scanned. Sites that render images via CSS `background-image` (some Twitter cards) won't be caught.
- **C2PA signature validation is not performed** — we trust the manifest contents. False-positive risk is low because the strings we look for (e.g. `trainedAlgorithmicMedia`) are AI-specific.

## Development

```bash
npm run watch    # esbuild watch mode
```

Click the **Reload** button on the extension's card after rebuilding to pick up changes. Page tabs need a refresh too — content scripts don't auto-update on extension reload.

### Build outputs

- `dist/background.js` — service worker (~99KB, includes exifr)
- `dist/content.js` — content script (page DOM, scan UI)
- `dist/offscreen.js` — offscreen document (~1.1MB, c2pa-js + onnxruntime-web)
- `dist/options.{html,js}` — options page
- `dist/models/` — ONNX classifier weights (OpenSynthID ~82MB, OpenFake ~350MB), produced by `./build-models.sh` — see [Model weights](#model-weights)
- `dist/ort-wasm-*` — ONNX runtime WASM
- `dist/c2pa.worker.min.js`, `dist/toolkit_bg.wasm` — c2pa-js assets

### Adding a visual classifier

1. Convert PyTorch / safetensors → ONNX. For HuggingFace `*ForImageClassification` models, add a `convert_hf_image_classifier(...)` call in `convert_model.py` and run `python3.13 convert_model.py` — it will read the model's processor for input size + normalization and verify on `firefly-sample.jpg`.
2. Place the `.onnx` in `models/<YourModel>/`
3. Add a copy entry in `build.js`
4. Add a config to `VISUAL_MODELS` in `src/offscreen.js` (input size, mean/std, output interpretation, thresholds)
5. Rebuild

## File layout

```
.
├── manifest.json           # MV3 manifest
├── package.json            # esbuild + dependencies
├── build.js                # esbuild build script
├── build-models.sh         # fetch upstream weights + convert to ONNX (see "Model weights")
├── convert_model.py        # PyTorch/safetensors→ONNX converter (--opensynthid, --openfake, and others)
├── src/
│   ├── background.js       # service worker, fetches bytes, runs tiers 2/3, dispatches to offscreen
│   ├── content.js          # DOM scan, overlays, progress bar, SPA nav handling
│   ├── offscreen.js        # c2pa-js + onnxruntime-web inference (tier 4 SynthID + tier 5 classifier)
│   ├── synthid-preprocess.js  # pure-JS db4 wavelet + FFT + carrier mask for the tier-4 input
│   ├── offscreen.html      # offscreen doc host page
│   ├── options.html, options.js  # extension options
│   └── overlay.css         # label / progress bar / outline styles
├── models/                 # ONNX weights (git-ignored, built by build-models.sh): OpenSynthID + OpenFake copied to dist/
├── templates/              # watermark PNGs (Gemini, Grok) — for the disabled tier 4a
├── test-images/            # 7 AI + 3 real images for broader testing
├── test.html               # local sample page (six AI generators)
├── test-images.html        # broader test page using test-images/
└── *-sample.{jpg,png}      # AI image samples for testing
```

## Licensing

SlopGuard's own source code is licensed **GPL-3.0-or-later** — see [`LICENSE`](LICENSE).

It bundles third-party models and libraries under their own terms; the full
inventory and required attributions are in
[`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md) and [`NOTICE`](NOTICE):

- **OpenSynthID** (tier 4) — Apache-2.0
- **OpenFake** (tier 5) — **CC BY-NC 4.0**
- c2pa-js, onnxruntime-web, exifr — MIT

> ⚠️ **The bundled distribution is NonCommercial.** OpenFake's CC BY-NC 4.0 term
> applies to any build that includes the OpenFake model, so the bundle as shipped
> **may not be used commercially**.
