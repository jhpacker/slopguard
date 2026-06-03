# Third-Party Licenses

SlopGuard bundles third-party models and libraries. Their licenses are listed
below. SlopGuard's own source code is licensed separately (see `LICENSE`).

> ⚠️ **Distribution is NonCommercial.** One bundled model (OpenFake) is licensed
> **CC BY-NC 4.0**. Its NonCommercial term travels with the file regardless of
> SlopGuard's own code license, so any distribution that includes
> `models/OpenFake/` (and the `dist/models/OpenFake/model.onnx` produced from it)
> **may not be used for commercial purposes**. To distribute SlopGuard for
> commercial use, remove the OpenFake model (disable tier 5) or replace it with a
> permissively-licensed detector.
>
> CC BY-NC 4.0 has **no ShareAlike/copyleft clause** — it does *not* require
> SlopGuard's code to adopt the same license.

## Bundled models

### OpenFake (tier 5 — SwinV2 AI-image detector)

- **License:** Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0) — https://creativecommons.org/licenses/by-nc/4.0/
- **Upstream project:** OpenFake — https://github.com/vicliv/OpenFake
- **Source weights:** `ComplexDataLab/OpenFakeDemo` (Hugging Face Space), a
  `microsoft/swinv2-base-patch4-window16-256` fine-tune.
- **Modifications:** rebuilt the base SwinV2 model + 2-class classification head
  and loaded the Space's `model.safetensors`, then exported to ONNX
  (`convert_model.py --openfake`). Distributed here as
  `models/OpenFake/model.onnx`.
- **Attribution:** "OpenFake" by vicliv et al., licensed CC BY-NC 4.0, modified
  (ONNX conversion).

### OpenSynthID (tier 4 — SynthID watermark surrogate)

- **License:** Apache License 2.0 —
  https://www.apache.org/licenses/LICENSE-2.0
- **Source:** `fyxme/opensynthid-detect-0.1` (Hugging Face) —
  https://huggingface.co/fyxme/opensynthid-detect-0.1
- **Modifications:** exported the source `.pt` checkpoint to ONNX with the
  sigmoid fused into the graph (`convert_model.py --opensynthid`). Distributed
  here as `models/OpenSynthID/model.onnx`.
- **Attribution / NOTICE:** retain the upstream copyright and license notice as
  required by Apache-2.0 §4. No NonCommercial restriction.

## Bundled JavaScript libraries

| Library | License | Use |
|---|---|---|
| [c2pa](https://github.com/contentauth/c2pa-js) | MIT | Tier 1 — C2PA manifest reading (WASM) |
| [onnxruntime-web](https://github.com/microsoft/onnxruntime) | MIT | Tiers 4 & 5 — ONNX inference |
| [exifr](https://github.com/MikeKovarik/exifr) | MIT | Tier 2 — EXIF/IPTC/XMP parsing |

(Build-time-only dependencies such as esbuild are not redistributed and are not
listed here.)

## Removed models (no longer bundled)

The following classifiers were part of earlier tier-5 ensembles and have been
removed; they are **not** distributed with SlopGuard. If reintroduced, verify and
add their licenses here:

- `Organika/sdxl-detector`
- `prithivMLmods/Deepfake-Detect-Siglip2`
- `Ateeqq/ai-vs-human-image-detector`
