# TODO

Tracking items ahead of a possible Chrome Web Store (CWS) release. See `CLAUDE.md`
for the deeper architectural context.

## Security / CWS readiness

### ✅ Done
- **Disable c2pa remote manifest fetching.** `createC2pa(...)` in `src/offscreen.js`
  now passes `fetchRemoteManifests: false` + `settings.verify.{ocspFetch,
  remoteManifestFetch} = false`. Previously a crafted image referencing a
  cloud-stored manifest could make the offscreen doc fetch an attacker-controlled
  URL (SSRF / "user viewed this" beacon) — and it broke the local-only goal.
  Embedded-manifest detection (the common case, incl. Google edit chains) is
  unaffected; only rare cloud-manifest images are now skipped.
- **Clean CWS package build.** `./package.sh` builds via `build.js` and zips ONLY
  `manifest.json` + `dist/`, stripping source maps, `sourceMappingURL` comments,
  and `.DS_Store`. Output: `build/slopguard-<version>.zip`.

### ✅ Done (partial) — On-demand injection; `<all_urls>` host_permissions retained by necessity

**What shipped:** removed the static `content_scripts` auto-injection (the biggest
CWS scrutiny point — code running on every page at `document_idle`) and switched to
**on-demand injection via `chrome.scripting`**. The content script now loads only
into the tab the user explicitly invokes (toolbar click or right-click), not every
page in the background.

- **manifest.json:** removed the `content_scripts` block; added `"scripting"` to
  `permissions`. **Kept `host_permissions: ["<all_urls>"]`** (see below).
- **background.js:** new `ensureInjected(tabId, frameId)` helper does
  `insertCSS` + `executeScript` of `dist/overlay.css` / `dist/content.js`. Both the
  `action.onClicked` (toolbar → `scan-now`) and `contextMenus.onClicked`
  (right-click → `check-one`, into `info.frameId`) paths inject before messaging.
- **content.js:** top-level `window.__slopguardLoaded` guard wraps the whole IIFE so
  a repeat injection no-ops (listeners/observers from the first load persist and
  still receive `scan-now`/`check-one`).

**Why `<all_urls>` host_permissions was NOT dropped (plan revised):** the original
plan was `<all_urls>` → `activeTab`. That doesn't work. The core detection fetches
image bytes **cross-origin from the service worker** (`fetchBytes`) and **offscreen
doc** (×3), to arbitrary image CDNs (pbs.twimg.com, i.redd.it, …), with
`credentials: 'include'`. In MV3 those cross-origin fetches are CORS-gated and
`host_permissions` is what grants the bypass. `activeTab` only grants temporary
access to the **active tab's own origin** — not the third-party CDN origins where
the images live — so `activeTab`-only would break byte-fetching for most real-world
images. Decision (2026-06-03): keep `<all_urls>` host_permissions (also authorizes
the on-demand `scripting` injections, making `activeTab` redundant). The remaining
CWS justification is "needs to fetch image bytes from any origin to analyze them,"
which is honest and defensible.

**In-browser re-testing: ✅ confirmed working (2026-06-03)** — toolbar scan, SPA
nav + infinite scroll (content.js now arrives post-click instead of at
`document_idle`), first-right-click single-image check (resolves via the `srcUrl`
fallback in `findImageForCheck` that first time, since the capture-phase
`contextmenu` listener isn't present until after the first menu click injects
content.js), and auth-gated image fetches (Twitter/X, Reddit — unchanged,
`<all_urls>` retained).

### ✅ Done — Removed `web_accessible_resources`
`manifest.json` previously exposed `toolkit_bg.wasm` + `c2pa.worker.min.js` to
`<all_urls>`, which let any site fingerprint the extension via its (post-publish
stable) ID. The whole WAR block is now **deleted**. c2pa runs in the offscreen
document (extension origin) and spawns same-origin workers, so it loads its WASM +
worker via `chrome.runtime.getURL` without needing WAR — same access pattern as the
ORT WASM files, which were never in WAR. Confirmed c2pa detection still works
in-browser after removal.

## Package size (heavy, but under CWS's 2GB cap)

The zip is dominated by:
- **OpenFake model — 352M** (fp32 SwinV2, the tier-5 model). **Trial passed
  (2026-06-03)** — real-world FP rate held up (unlike the original three classifiers
  that were disabled), so it's staying as the active tier-5 detector and is no
  longer at risk of removal. **Staying fp32 by decision** — quantizing was
  considered and rejected: fp16 gives no runtime win on the `wasm` EP (no native
  fp16 kernels → Cast-to-fp32 at inference, possibly *slower*) and int8 risks
  accuracy drift on a SwinV2 transformer. Runtime speed matters more to users than
  package size here. This is the single biggest contributor to zip size, but it's a
  load-bearing model now, not trimmable.
- ~~**4 ORT WASM variants — ~75M**~~ ✅ **Done.** `build.js` now copies only the
  plain `ort-wasm-simd-threaded.{wasm,mjs}` (the one EP actually instantiated,
  `executionProviders: ['wasm']`). Dropped the `jsep`, `asyncify`, and `jspi`
  variants, reclaiming ~64M. Re-add `.jsep.{wasm,mjs}` if ever switching to the
  webgpu EP.
  - **REQUIRED paired change (regression fixed 2026-06-03):** the default
    `import ... from 'onnxruntime-web'` resolves to the unified **jsep** bundle,
    which loads `ort-wasm-simd-threaded.jsep.wasm` at runtime *even under the
    `wasm` EP* — so trimming the jsep wasm made `InferenceSession.create` 404 and
    every visual/SynthID model errored (`*=err`) in-browser. (Node didn't catch
    it: it resolves wasm from `node_modules/`, where all variants still exist.)
    Fix: `src/offscreen.js` now imports from **`onnxruntime-web/wasm`** (the
    wasm-only bundle, which requests the plain artifact we ship). If you ever
    re-add the jsep wasm for WebGPU, switch this import back to the default entry.

## Misc cleanup

### ✅ Done
- **Removed dead tier-6 LLM code.** Deleted `handleLlmJudgeDetect` + the
  `LanguageModel` session/downscale helpers and the `llm-judge-detect` dispatch
  from `src/offscreen.js`, the `llmJudgeDetect` helper + disabled dispatch block
  from `src/background.js`, and the now-unreachable `llm-*` failure reasons +
  `llm-judge` label branch from `src/content.js`. Re-enable instructions live in
  CLAUDE.md ("Things tried and rejected") if ever revisited.
- **Deleted orphaned `src/empty.js`** (old OpenCV stub, unreferenced).
