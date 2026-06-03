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

### ⬜ Narrow host permissions: `<all_urls>` → `activeTab` + `scripting` (RECOMMENDED before submit)

**Why:** Today `manifest.json` declares `host_permissions: ["<all_urls>"]` *and* a
content script auto-injected into every page (`matches: ["<all_urls>"]`,
`run_at: document_idle`). Combined with credentialed cross-origin image fetches
(`credentials: 'include'`), this is the broadest possible privilege and the thing
CWS review scrutinizes most. The extension only ever acts on an explicit user
gesture (toolbar click or right-click menu), so the standing all-pages access
isn't actually needed.

**Target model:** inject the content script on demand, only into the tab the user
invoked, using `activeTab` + `chrome.scripting`.

**What the refactor requires:**
1. **manifest.json**
   - Replace `host_permissions: ["<all_urls>"]` with `"activeTab"` and
     `"scripting"` in `permissions`.
   - Remove the static `content_scripts` block entirely (no more auto-injection).
   - Keep `offscreen`, `storage`, `contextMenus`.
   - Likely drop `web_accessible_resources` too (see next item) — c2pa runs in the
     offscreen doc, not in page context.
2. **background.js — toolbar path:** in `chrome.action.onClicked`, instead of
   `chrome.tabs.sendMessage(tab.id, {type:'scan-now'})`, first
   `chrome.scripting.executeScript({ target: { tabId }, files: ['dist/content.js'] })`
   and `chrome.scripting.insertCSS({ target: { tabId }, files: ['dist/overlay.css'] })`,
   then send `scan-now`. Guard against double-injection (executeScript is
   idempotent enough, or set a `window.__slopguard` sentinel in content.js).
3. **background.js — context-menu path:** same thing — the `contextMenus.onClicked`
   handler must `executeScript`/`insertCSS` into `tab.id`/`info.frameId` before
   sending `check-one`. `activeTab` grants access because the right-click is a
   user gesture on that tab.
4. **content.js:** must tolerate being injected after `document_idle` / multiple
   times. It already guards most state; add a top-level "already loaded" check so a
   second injection doesn't re-register listeners/observers. The `MutationObserver`
   for SPA nav + infinite scroll still works once injected.
5. **Credentialed fetches:** `credentials: 'include'` on image fetches
   (`background.js` `fetchBytes`, `offscreen.js` ×3) stays functionally the same,
   but under `activeTab` the host access is scoped to the active tab's origin at
   invocation time. Re-test auth-gated images (Twitter/X, Reddit) after the change.

**Trade-off:** loses passive/automatic scanning (already not a feature — scanning
is click-triggered), in exchange for a far easier permission justification and much
smaller blast radius. Re-test SPA navigation + infinite scroll, since the content
script now arrives later in the page lifecycle.

### ⬜ Remove (or narrow) `web_accessible_resources`
`manifest.json` exposes `toolkit_bg.wasm` + `c2pa.worker.min.js` to `<all_urls>`,
which lets any site fingerprint the extension via its (post-publish stable) ID.
c2pa now runs in the offscreen document (extension origin) and spawns same-origin
workers, so WAR is probably unnecessary. Test removing the block entirely; if c2pa
breaks, narrow `matches` rather than reverting to `<all_urls>`.

## Package size (heavy, but under CWS's 2GB cap)

The zip is dominated by:
- **OpenFake model — 352M** (fp32 SwinV2, the tier-5 *trial* model). If the trial
  doesn't pan out (same real-world FP problem that killed the original three),
  dropping it reclaims most of the package. **Staying fp32 by decision** —
  quantizing was considered and rejected: fp16 gives no runtime win on the `wasm`
  EP (no native fp16 kernels → Cast-to-fp32 at inference, possibly *slower*) and
  int8 risks accuracy drift on a SwinV2 transformer, which would confound the
  trial's whole purpose (measuring real-world FP rate). Runtime speed matters
  more to users than package size here.
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
