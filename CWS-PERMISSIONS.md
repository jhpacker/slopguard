# Chrome Web Store — permission justifications

Copy-paste source for the "Privacy practices" tab of the CWS developer
dashboard. Each justification must explain why the permission is needed for the
extension's single purpose. Keep these in sync with manifest.json.

## Single purpose

SlopGuard detects AI-generated images on web pages and visually flags them. When
the user clicks the toolbar button or right-clicks an image, it analyzes the
relevant images on that page — locally, on the user's device — and labels the
ones it judges to be AI-generated.

## Permission justifications

### scripting
Used to inject the image-scanning script into a tab only when the user invokes
the extension on that tab (by clicking the toolbar button or choosing the
right-click "Check this image for AI" menu item). The extension intentionally
has no automatically-injected content script; nothing runs on a page until the
user acts on it. `scripting` is what allows this on-demand injection.

### offscreen
Used to run the image analysis off the background service worker. The detection
work — WebAssembly C2PA provenance parsing and ONNX machine-learning inference —
must run in a DOM/worker-capable context, which the service worker is not. The
offscreen document hosts this local processing.

### contextMenus
Used to add the "Check this image for AI" item to the right-click menu on
images, so the user can check a single image on demand without scanning the
whole page.

### storage
Used to persist a single user setting — an on/off "Debug mode" toggle — via
chrome.storage.local. No personal or browsing data is stored.

## Host permission justification

### host_permissions: <all_urls>
The extension's core function is checking images on whatever page the user is
viewing. To analyze an image it must fetch that image's bytes from the server
that hosts it, and images are served from a very large and unpredictable set of
domains (CDNs, social networks, image hosts), so access cannot be restricted to
a fixed list of sites. This access is used only to retrieve images for on-device
analysis, only in response to an explicit user action (toolbar click or
right-click). No data about the user, the pages visited, or the results is
transmitted anywhere — all analysis is local.

## Remote code

The extension does NOT use remote code. All code, including the WebAssembly and
machine-learning model files, is bundled in the package and loaded from the
extension's own origin. No scripts or code are fetched or executed from remote
servers. (Answer "No, I am not using remote code" in the dashboard.)

## Data usage declarations

SlopGuard does not collect or transmit any user data. Suggested answers for the
data-use checklist:

- Does NOT collect or use: personally identifiable information, health
  information, financial/payment information, authentication information,
  personal communications, location, web history, or user activity.
- Does NOT sell or transfer user data to third parties.
- Does NOT use or transfer user data for purposes unrelated to the single
  purpose.
- Does NOT use or transfer user data to determine creditworthiness or for
  lending.

All processing is on-device; nothing is sent to the developer or any third
party. See PRIVACY.md (hosted publicly) for the full policy.
