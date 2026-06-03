# SlopGuard Privacy Policy Summary

SlopGuard is a Chrome extension that detects AI-generated images on web pages. All
detection runs entirely on your device — there is no backend, no analytics, no
telemetry, and the developer receives zero user data.

## Key Privacy Features

**On-Device Detection**: Every detection tier runs locally in your browser. C2PA
provenance reading (WebAssembly), metadata/EXIF parsing, byte scanning, the SynthID
watermark detector, and the visual classifier (ONNX) all execute on your machine.
The detection models are bundled inside the extension, so nothing is downloaded from
a model server at scan time. No remote inference API is called, and no API keys are
required or used.

**No Developer Access**: There is no backend infrastructure, analytics, or
telemetry. The creator (Quantable LLC) receives zero user data.

**Local Storage Only**: The only persisted setting is a `debugMode` on/off toggle,
stored in `chrome.storage.local`. Scan results are held in a small in-memory cache
that is discarded when the browser's service worker restarts. No browsing history,
image content, page content, or personal data is ever saved.

## Network Activity

SlopGuard is **not fully network-silent**, and we want to be clear about why. To
inspect an image, the extension fetches that image's bytes from the URL where it is
already hosted on the page you are viewing. These requests go to the **same servers
that already served the image to the page** — SlopGuard introduces no new third
parties, no developer-controlled servers, and sends no data anywhere about you, the
pages you visit, or the results. Image bytes are read in memory, analyzed, and
discarded; they are never uploaded, logged, or transmitted to the developer or any
external service.

These image fetches include your browser's existing cookies for that image host
(the request is made with credentials), exactly as the page's own image load does.
This is what lets SlopGuard retrieve images that are only visible while you are
logged in (for example on social networks). The extension never reads, stores, or
transmits the contents of those cookies — the browser simply attaches them to the
request to the image's own host, the same way it does for any normal page load.

## Page Actions

SlopGuard only acts when you explicitly ask it to. It does **not** scan pages
automatically in the background. Detection runs when you:

- **Click the toolbar icon** — scans the `<img>` elements on the current page that
  are larger than roughly 200×200 px.
- **Right-click an image → "Check this image for AI"** — scans only that one image.

For each scanned image, SlopGuard reads:
- The image URL (used to fetch the bytes and to cache the result in memory)
- The image bytes (up to an 8 MB cap), for metadata, provenance, and pixel analysis

This content is processed locally and never leaves your device.

## Permissions Rationale

The extension requests only what it needs to scan images on the page you are on:

- `host_permissions: <all_urls>` — to fetch image bytes from whatever site hosts
  the images on the page you choose to scan. Images are served from many different
  domains, so this access cannot be limited to a fixed list. It is used only to
  retrieve images for analysis, only in response to your action.
- `scripting` — to inject the scanning logic into a tab **only** when you invoke the
  extension on that tab (toolbar click or right-click). The extension does not run a
  content script on pages automatically; nothing is injected until you act.
- `storage` — to remember the single Debug-mode setting.
- `contextMenus` — for the right-click "Check this image for AI" option.
- `offscreen` — to run the WebAssembly (C2PA) and ONNX inference off the service
  worker.

Notably absent: analytics, telemetry, remote/developer-controlled servers, any
reading or storing of your cookies or browsing history, and any user-supplied or
developer-supplied API keys.

**Contact**: jason@quantable.com or the GitHub repository issue tracker.
