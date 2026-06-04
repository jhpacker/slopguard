import * as esbuild from 'esbuild';
import { copyFile, mkdir, rm } from 'node:fs/promises';

const watch = process.argv.includes('--watch');

// Wipe dist/ first so stale outputs from a previous build (e.g. ORT WASM
// variants we no longer copy) can't linger and get re-zipped by package.sh.
// Everything in dist/ is generated, so this is always safe.
await rm('dist', { recursive: true, force: true });
await mkdir('dist', { recursive: true });

const configs = [
  {
    entryPoints: ['src/background.js'],
    outfile: 'dist/background.js',
    bundle: true,
    format: 'esm',
    target: 'chrome120',
    sourcemap: true,
    logLevel: 'info',
  },
  {
    entryPoints: ['src/content.js'],
    outfile: 'dist/content.js',
    bundle: true,
    format: 'iife',
    target: 'chrome120',
    sourcemap: true,
    logLevel: 'info',
  },
  {
    entryPoints: ['src/offscreen.js'],
    outfile: 'dist/offscreen.js',
    bundle: true,
    format: 'iife',
    target: 'chrome120',
    sourcemap: true,
    logLevel: 'info',
  },
  {
    entryPoints: ['src/options.js'],
    outfile: 'dist/options.js',
    bundle: true,
    format: 'iife',
    target: 'chrome120',
    sourcemap: true,
    logLevel: 'info',
  },
];

if (watch) {
  const ctxs = await Promise.all(configs.map(esbuild.context));
  await Promise.all(ctxs.map((c) => c.watch()));
  console.log('Watching for changes...');
} else {
  await Promise.all(configs.map(esbuild.build));
}

// Only the plain SIMD+threaded WASM build is used at runtime — offscreen.js
// imports `onnxruntime-web/wasm` and instantiates with `executionProviders:
// ['wasm']`. The jsep (WebGPU/WebNN), asyncify, and jspi variants are ~64M of
// dead weight, so they're not copied. (A WebGPU EP trial that needed the
// asyncify variant was reverted 2026-06-03 — it slowed the CPU path; see the
// offscreen.js header note. If you re-trial webgpu, re-add the asyncify .{wasm,
// mjs} and switch the import back to `onnxruntime-web/webgpu`.)
const ortFiles = [
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
];

// The three tier-5 visual classifiers (Organika, Siglip2-Deepfake, AIvHuman)
// are intentionally NOT bundled — they're disabled in VISUAL_MODELS and only
// OpenFake is being trialed. Tier 4 (OpenSynthID) stays. Re-add their mkdir +
// copyFile lines to restore the full ensemble.
// await mkdir('dist/models/Organika', { recursive: true });
// await mkdir('dist/models/Siglip2-Deepfake', { recursive: true });
// await mkdir('dist/models/AIvHuman', { recursive: true });
await mkdir('dist/models/OpenSynthID', { recursive: true });
await mkdir('dist/models/OpenFake', { recursive: true });
await mkdir('dist/icons', { recursive: true });

const iconSizes = [16, 32, 48, 128];

await Promise.all([
  copyFile('src/overlay.css', 'dist/overlay.css'),
  copyFile('src/offscreen.html', 'dist/offscreen.html'),
  copyFile('src/options.html', 'dist/options.html'),
  ...iconSizes.map((s) =>
    copyFile(`assets/logo/icon-${s}.png`, `dist/icons/icon-${s}.png`),
  ),
  copyFile('node_modules/c2pa/dist/c2pa.worker.min.js', 'dist/c2pa.worker.min.js'),
  copyFile('node_modules/c2pa/dist/assets/wasm/toolkit_bg.wasm', 'dist/toolkit_bg.wasm'),
  // Tier-5 classifiers excluded (see note above); tier-4 OpenSynthID kept.
  // copyFile('models/Organika/model.onnx', 'dist/models/Organika/model.onnx'),
  // copyFile('models/Siglip2-Deepfake/model.onnx', 'dist/models/Siglip2-Deepfake/model.onnx'),
  // copyFile('models/AIvHuman/model.onnx', 'dist/models/AIvHuman/model.onnx'),
  copyFile('models/OpenSynthID/model.onnx', 'dist/models/OpenSynthID/model.onnx'),
  copyFile('models/OpenFake/model.onnx', 'dist/models/OpenFake/model.onnx'),
  ...ortFiles.map((f) =>
    copyFile(`node_modules/onnxruntime-web/dist/${f}`, `dist/${f}`),
  ),
]);
