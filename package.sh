#!/usr/bin/env bash
#
# package.sh — produce a clean Chrome Web Store upload zip.
#
# Builds the extension via build.js, then stages ONLY the files the published
# extension needs (manifest.json + dist/), stripping development artifacts:
#   - source maps (*.map) and their //# sourceMappingURL comments
#   - .DS_Store and other macOS cruft
#
# Everything else in the repo (node_modules/, src/, models/ source weights,
# attic/, convert_model.py, test pages, etc.) is intentionally excluded — the
# zip root is just manifest.json and dist/, matching the paths in manifest.json.
#
# Usage:
#   ./package.sh
#
# Output:
#   build/slopguard-<version>.zip

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STAGE="$ROOT/build/pkg"

# Read version from manifest.json (no jq dependency).
VERSION="$(grep -oE '"version"[[:space:]]*:[[:space:]]*"[^"]+"' manifest.json | head -1 | sed -E 's/.*"([^"]+)"$/\1/')"
if [[ -z "$VERSION" ]]; then
  echo "ERROR: could not read version from manifest.json" >&2
  exit 1
fi
ZIP="$ROOT/build/slopguard-${VERSION}.zip"

echo "==> Building (build.js)…"
node build.js

echo "==> Staging clean package in $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/dist"

# manifest + the whole dist tree, then prune dev artifacts from the copy.
cp manifest.json "$STAGE/manifest.json"
cp -R dist/. "$STAGE/dist/"

# Drop source maps and any OS cruft from the staged copy only.
find "$STAGE" -type f \( -name '*.map' -o -name '.DS_Store' \) -delete

# Remove dangling sourceMappingURL comments so devtools doesn't 404 looking
# for the maps we just stripped. (perl is portable across macOS/Linux.)
find "$STAGE/dist" -type f -name '*.js' -exec \
  perl -i -ne 'print unless m{^//# sourceMappingURL=}' {} +

echo "==> Zipping → $ZIP"
rm -f "$ZIP"
( cd "$STAGE" && zip -rq "$ZIP" manifest.json dist )

echo
echo "==> Done. Package contents:"
unzip -l "$ZIP" | sed -n '1,4p;$p'
echo
echo "Size: $(du -h "$ZIP" | cut -f1)"
echo "Zip:  $ZIP"
echo
echo "Sanity checks:"
# These should all print 0 — no maps, no source, no node_modules in the zip.
echo "  *.map entries:        $(unzip -l "$ZIP" | grep -c '\.map$' || true)"
echo "  src/ entries:         $(unzip -l "$ZIP" | grep -c ' src/' || true)"
echo "  node_modules entries: $(unzip -l "$ZIP" | grep -c 'node_modules/' || true)"
