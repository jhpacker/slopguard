# Chrome Web Store listing copy

Source text for the SlopGuard Chrome Web Store listing. CWS does not render
Markdown in the description field — paste the plain text below as-is.

## Short description (max 132 chars)

Detects AI-generated images on web pages via metadata, C2PA provenance, a SynthID watermark detector, and a visual classifier.

## Detailed description

SlopGuard flags AI-generated images on the web pages you visit, and runs entirely on your own device. No accounts, no API keys, no servers, nothing uploaded.

HOW TO USE IT

• Click the SlopGuard toolbar icon on any page to scan every eligible
  image on it (images rendered at roughly 200×200 px or larger).
• Or right-click any single image and choose "Check this image for AI"
  to check just that one.

WHAT YOU SEE

• A red "AI" or "Probably AI" label, or a yellow "Maybe AI" label, on
  images judged to be AI-generated — and the image is dimmed to grayscale.
• A thin green outline on images that were checked and came back clean.
• A grey outline when a check couldn't be completed.

HOW IT DECIDES

SlopGuard runs several independent checks and stops at the first solid hit:

1. C2PA provenance: reads the cryptographic "content credentials" many
   generators embed (though are frequently stripped). We don't validate this certificate (i.e. we assume if something says it's AI it's AI).

2. Metadata attribution: EXIF / IPTC / XMP fields that name an AI tool
   or "AI" as the creator.

3. Byte signatures: fingerprints of known generators (Midjourney,
   ChatGPT/DALL·E, Adobe Firefly, Stable Diffusion, ComfyUI, and more)
   and tell-tale generation parameters. This is to catch cases where metadata did exist but got incompletely stripped.

4. SynthID watermark: looks for Google's invisible SynthID watermark
   directly in the pixels.

5. Visual classifier: a general "does this look AI-generated?" model for
   images that carry no metadata at all.

PRIVATE BY DESIGN

Every check runs locally in your browser. The detection models are bundled inside the extension, so nothing is downloaded at scan time and no image, URL, or result is ever sent to the developer or any third party. The only thing stored is a single on/off "Debug mode" setting. SlopGuard acts only when you click.

GOOD TO KNOW (HONEST LIMITATIONS)

• Visual AI detection is a best guess — expect occasional misses and false positives.
• The SynthID check uses an independent community surrogate, not Google's official detector, and detects only Google's invisible watermark.
• Images with all metadata completely stripped can only be caught by the
  visual classifier, if at all.
• Only <img> elements are scanned; CSS background images are not.

CREDITS

On-device detection builds on the work of others:
• OpenFake (visual classifier) — Victor Livernoche @ ComplexDataLab.
• opensynthid-detect (SynthID surrogate) — fyxme.

SlopGuard is not affiliated with, endorsed by, or connected to Google or
DeepMind. "SynthID" is a trademark of Google DeepMind, used here only to describe what the watermark check looks for.

OPEN SOURCE & LICENSING

SlopGuard is free and open source, licensed GPL-3.0-or-later. Source code, build instructions, and the issue tracker are on GitHub:
https://github.com/jhpacker/slopguard

It bundles third-party models and libraries under their own licenses:
• OpenFake (visual classifier) — CC BY-NC 4.0 (NonCommercial)
• OpenSynthID (SynthID watermark surrogate) — Apache-2.0
• c2pa-js, onnxruntime-web, exifr — MIT

Because the bundled OpenFake model is licensed CC BY-NC 4.0, this build is provided for NonCommercial use. Full license texts and attributions are in the repository (THIRD-PARTY-LICENSES.md and NOTICE).
