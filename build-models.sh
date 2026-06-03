#!/usr/bin/env bash
#
# build-models.sh — fetch + convert the ONNX model weights SlopGuard needs.
#
# The .onnx weights are NOT committed (they're large and reproducible — see
# .gitignore). This script recreates them from a fresh clone, and also picks up
# any upstream weight updates published by the model maintainers.
#
# It produces the two files build.js copies into dist/:
#     models/OpenSynthID/model.onnx   (tier 4 — SynthID watermark surrogate)
#     models/OpenFake/model.onnx      (tier 5 — OpenFake SwinV2 real/fake)
#
# Sources:
#   OpenSynthID — fyxme/opensynthid-detect-0.1   (a .pt checkpoint we download,
#                 then convert via convert_model.py --opensynthid)
#   OpenFake    — ComplexDataLab/OpenFakeDemo     (a HF *Space*; its safetensors
#                 are fetched automatically by convert_model.py --openfake)
#
# Usage:
#     ./build-models.sh              # create venv if needed, fetch + convert
#     ./build-models.sh --refresh    # also re-download upstream source weights
#                                    # (use after maintainers update a model)
#
# Env overrides:
#     PYTHON=python3.13   # interpreter to build the venv from (needs torch wheels)
#
set -euo pipefail

cd "$(dirname "$0")"

REFRESH=0
for arg in "$@"; do
  case "$arg" in
    --refresh) REFRESH=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# --- pick a Python that can install torch -----------------------------------
# torch has no wheels for very new Pythons (e.g. 3.14), so prefer 3.13.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
fi
if [ -z "$PYTHON" ] || ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "No suitable Python found. Set PYTHON=pythonX.Y (needs torch wheels)." >&2
  exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version 2>&1))"

# --- venv -------------------------------------------------------------------
VENV=".venv-convert"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "Creating venv at $VENV …"
  "$PYTHON" -m venv "$VENV"
fi

echo "Installing / updating Python deps (first run downloads torch — slow) …"
"$PY" -m pip install --quiet --upgrade pip
# Deps used by the --opensynthid and --openfake conversion paths only.
"$PY" -m pip install --quiet --upgrade \
  torch torchvision \
  transformers safetensors huggingface_hub \
  onnx onnxscript onnxruntime \
  pywavelets numpy pillow

# --- OpenSynthID: download the source .pt, then convert ---------------------
OSID_DIR="models/OpenSynthID"
OSID_PT="$OSID_DIR/model.pt"
OSID_URL="https://huggingface.co/fyxme/opensynthid-detect-0.1/resolve/main/model.pt"
mkdir -p "$OSID_DIR"
if [ "$REFRESH" = "1" ] || [ ! -f "$OSID_PT" ]; then
  echo "Downloading OpenSynthID checkpoint → $OSID_PT …"
  curl -L --fail -o "$OSID_PT" "$OSID_URL"
else
  echo "OpenSynthID checkpoint present ($OSID_PT) — pass --refresh to re-download."
fi
echo "Converting OpenSynthID → $OSID_DIR/model.onnx …"
"$PY" convert_model.py --opensynthid

# --- OpenFake: convert_model.py fetches the Space weights itself -------------
# With --refresh, drop the cached HF download so the newest safetensors is pulled.
if [ "$REFRESH" = "1" ]; then
  echo "Clearing cached OpenFakeDemo download (forcing re-fetch) …"
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}"/hub/spaces--ComplexDataLab--OpenFakeDemo 2>/dev/null || true
fi
echo "Converting OpenFake → models/OpenFake/model.onnx …"
"$PY" convert_model.py --openfake

echo
echo "Done. Built:"
for f in models/OpenSynthID/model.onnx models/OpenFake/model.onnx; do
  if [ -f "$f" ]; then
    printf '  %-32s %s\n' "$f" "$(du -h "$f" | cut -f1)"
  else
    echo "  MISSING: $f" >&2
  fi
done
echo "Now run: npm run build"
