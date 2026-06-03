"""
Convert HuggingFace image classifiers to ONNX for use in the browser via
onnxruntime-web.

This is a one-time operation. Outputs are preserved in the repo so the
extension build doesn't depend on Python. Re-run only if you want to
re-fetch / regenerate.

Install once:

    pip install --user torch transformers timm onnx onnxscript huggingface_hub pillow

If your system Python doesn't have torch wheels yet (e.g. Python 3.14),
use Python 3.13 explicitly:

    python3.13 -m pip install --user torch transformers ...
    python3.13 convert_model.py

By default this script converts the two SigLIP-based tier-5 models. Pass
`--cf` to also convert the (currently-unused) CommunityForensics ViT.

Outputs:
    models/Siglip2-Deepfake/model.onnx        — prithivMLmods/Deepfake-Detect-Siglip2
    models/AIvHuman/model.onnx                — Ateeqq/ai-vs-human-image-detector
    models/CommunityForensics/model.onnx      — buildborderless/CommunityForensics-DeepfakeDet-ViT (only with --cf)
"""
import os
import sys
import torch
import torch.nn as nn
import onnx
from PIL import Image
# huggingface_hub is imported lazily inside the converters that need it
# (convert_communityforensics / convert_bombek1) so the opensynthid path,
# which downloads nothing, doesn't require it installed.


def _merge_external_data(out_path, work_dir):
    """Inline any external weight tensors so the ONNX file is self-contained."""
    m = onnx.load(out_path)
    for tensor in m.graph.initializer:
        if (
            tensor.HasField('data_location')
            and tensor.data_location == onnx.TensorProto.EXTERNAL
        ):
            onnx.external_data_helper.load_external_data_for_tensor(tensor, work_dir)
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
    onnx.save_model(m, out_path, save_as_external_data=False)
    data_path = out_path + '.data'
    if os.path.exists(data_path):
        os.remove(data_path)


class _LogitsOnly(nn.Module):
    """Wrap an HF image classifier so it returns just the logits tensor,
    suitable for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def convert_hf_image_classifier(
    repo_id,
    out_dir,
    sample_image,
    ai_label_name,
    expected_min_p_ai=0.5,
):
    """Convert a HF SiglipForImageClassification (or any AutoModelForImageClassification)
    to ONNX. Reads input size + normalization from the model's processor.

    Sanity-checks against `sample_image` (must be an AI image) — fails the
    export if P(AI) < `expected_min_p_ai`.
    """
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    import torch.nn.functional as F

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {repo_id} ===")
    print("Loading model + processor…")
    model = AutoModelForImageClassification.from_pretrained(repo_id)
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model.eval()

    # Pull preprocessing details for the JS side. transformers ≥5 wraps
    # this in a SizeDict; older versions use a plain dict; older still, an
    # int. Cover all three.
    size = processor.size
    if hasattr(size, 'height') and getattr(size, 'height') is not None:
        h, w = size.height, size.width
    elif isinstance(size, dict):
        h = size.get('height') or size.get('shortest_edge')
        w = size.get('width') or size.get('shortest_edge')
    else:
        h = w = int(size)
    assert h and w and h == w, f"Non-square or unknown input size: {size}"
    input_size = int(h)
    mean = list(map(float, processor.image_mean))
    std = list(map(float, processor.image_std))

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels = [id2label[i] for i in sorted(id2label)]
    matches = [i for i, lbl in enumerate(labels) if lbl.lower() == ai_label_name.lower()]
    assert matches, f"Couldn't find AI label '{ai_label_name}' in {labels}"
    ai_index = matches[0]

    print(f"  input: {input_size}×{input_size}, mean={mean}, std={std}")
    print(f"  labels: {labels} (AI index = {ai_index})")

    # Sanity check
    img = Image.open(sample_image).convert('RGB')
    inputs = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
    p_ai = float(probs[ai_index].item())
    print(f"  sanity ({os.path.basename(sample_image)}): P(AI)={p_ai:.3f}")
    assert p_ai >= expected_min_p_ai, (
        f"Sanity check failed: P(AI)={p_ai:.3f} < {expected_min_p_ai}"
    )

    # ONNX export
    out_path = os.path.join(out_dir, 'model.onnx')
    print(f"Exporting to {out_path}…")
    wrapped = _LogitsOnly(model)
    wrapped.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        wrapped,
        dummy,
        out_path,
        opset_version=17,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={'pixel_values': {0: 'batch'}, 'logits': {0: 'batch'}},
        do_constant_folding=True,
    )

    print("Inlining external weights…")
    _merge_external_data(out_path, out_dir)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB at {out_path}")
    print(f"  JS config: inputSize={input_size}, mean={mean}, std={std}, aiIndex={ai_index}")


def convert_communityforensics():
    """The original conversion path — kept for reference. Currently unused
    in the extension because the model produced too many false positives."""
    import timm
    from torchvision import transforms
    from huggingface_hub import hf_hub_download

    out_dir = "models/CommunityForensics"
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== buildborderless/CommunityForensics-DeepfakeDet-ViT ===")
    print("Downloading trained checkpoint…")
    ckpt_path = hf_hub_download(
        'buildborderless/CommunityForensics-DeepfakeDet-ViT',
        'pretrained_weights/model_v11_ViT_224_base_ckpt.pt',
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
    inner = {k[len('vit.'):]: v for k, v in state.items() if k.startswith('vit.')}

    print("Building timm ViT-Small (224, patch16) with 1-output head…")
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=1)
    missing, unexpected = model.load_state_dict(inner, strict=True)
    assert not missing and not unexpected, f"State dict mismatch: {missing} / {unexpected}"
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    img = Image.open('firefly-sample.jpg').convert('RGB')
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].item()
    print(f"Sanity (firefly sample): P(AI)={p:.3f}")
    assert p > 0.5

    out_path = os.path.join(out_dir, 'model.onnx')
    print(f"Exporting to {out_path}…")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=17,
        input_names=['pixel_values'],
        output_names=['logit'],
        dynamic_axes={'pixel_values': {0: 'batch'}, 'logit': {0: 'batch'}},
        do_constant_folding=True,
    )
    _merge_external_data(out_path, out_dir)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB at {out_path}")


def convert_bombek1():
    """Convert Bombek1/ai-image-detector-siglip-dinov2 to ONNX.

    The repo isn't a standard HF AutoModelForImageClassification — it ships
    a custom `EnsembleAIDetector` (SigLIP2-SO400M + DINOv2-Large with LoRA)
    plus its own model.py. We dynamically import that file, build the model,
    load weights, fuse to fp32, and trace.

    Forward takes TWO inputs at the same 392×392 resolution but with
    different normalizations:
      siglip_pixels — SigLIP norm  (mean/std = [0.5, 0.5, 0.5])
      dinov2_pixels — ImageNet norm (mean/std = [0.485..., 0.229...])
    Output is a single sigmoid'd P(AI) — we fuse the sigmoid into the graph
    so the JS side just reads it directly.
    """
    import importlib.util
    import json
    from huggingface_hub import snapshot_download

    out_dir = "models/Bombek1-AIDetector"
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Bombek1/ai-image-detector-siglip-dinov2 ===")
    print("Snapshotting repo (model.py + pytorch_model.pt + config.json, ~2.1GB)…")
    repo_dir = snapshot_download('Bombek1/ai-image-detector-siglip-dinov2')
    print(f"  cached at {repo_dir}")

    # Dynamically import model.py — it's not packaged, just a flat file.
    spec = importlib.util.spec_from_file_location('bombek1_model', f"{repo_dir}/model.py")
    bombek1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bombek1)

    with open(f"{repo_dir}/config.json") as f:
        cfg = json.load(f)

    print("Building model (downloads SigLIP2-SO400M + DINOv2-Large bases on first run, ~3GB)…")
    model = bombek1.create_model_with_lora(
        siglip_model_name=cfg.get('backbone_siglip', 'google/siglip2-so400m-patch14-384'),
        dinov2_model_name=cfg.get('backbone_dinov2', 'vit_large_patch14_dinov2.lvd142m'),
        image_size=cfg.get('image_size', 392),
        lora_rank=cfg['lora']['rank'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
    )

    print("Loading trained weights…")
    ckpt = torch.load(f"{repo_dir}/pytorch_model.pt", map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']

    # transformers ≥5 flattened SiglipVisionModel: it no longer has a
    # `.vision_model` submodule. The Bombek1 checkpoint was saved with the
    # older structure, so we strip the obsolete prefix from the SigLIP keys.
    # DINOv2 and classifier keys are unaffected.
    remapped = {}
    siglip_prefix_old = 'siglip.base_model.model.vision_model.'
    siglip_prefix_new = 'siglip.base_model.model.'
    for k, v in state.items():
        nk = k.replace(siglip_prefix_old, siglip_prefix_new) if k.startswith(siglip_prefix_old) else k
        remapped[nk] = v
    model.load_state_dict(remapped)

    # SigLIP was loaded as bf16 — force everything fp32 for ONNX export.
    model = model.float().eval()

    # Wrap to return sigmoid(logits) only. Original forward returns
    # (logits, siglip_features, dinov2_features) — features are unused.
    class _ProbOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, siglip_pixels, dinov2_pixels):
            logits, _, _ = self.m(siglip_pixels, dinov2_pixels)
            return torch.sigmoid(logits)

    wrapped = _ProbOnly(model).eval()

    # Sanity check on a known AI image.
    from transformers import AutoProcessor
    siglip_processor = AutoProcessor.from_pretrained(cfg.get('backbone_siglip', 'google/siglip2-so400m-patch14-384'))
    dinov2_transform = bombek1.create_transforms(cfg.get('image_size', 392))

    img = Image.open('firefly-sample.jpg').convert('RGB')
    siglip_pixels = siglip_processor(images=img, return_tensors='pt')['pixel_values']
    dinov2_pixels = dinov2_transform(img).unsqueeze(0)
    print(f"  siglip input: {tuple(siglip_pixels.shape)}, dtype={siglip_pixels.dtype}")
    print(f"  dinov2 input: {tuple(dinov2_pixels.shape)}, dtype={dinov2_pixels.dtype}")

    with torch.no_grad():
        prob = wrapped(siglip_pixels, dinov2_pixels).item()
    print(f"  sanity (firefly-sample.jpg): P(AI) = {prob:.3f}")
    assert prob > 0.5, f"Sanity check failed: P(AI)={prob:.3f}"

    out_path = os.path.join(out_dir, 'model.onnx')
    print(f"Exporting to {out_path} (with external data — model > 2GB protobuf limit)…")
    torch.onnx.export(
        wrapped,
        (siglip_pixels, dinov2_pixels),
        out_path,
        opset_version=17,
        input_names=['siglip_pixels', 'dinov2_pixels'],
        output_names=['prob_ai'],
        dynamic_axes={
            'siglip_pixels': {0: 'batch'},
            'dinov2_pixels': {0: 'batch'},
            'prob_ai': {0: 'batch'},
        },
        do_constant_folding=True,
        external_data=True,
    )

    # Keep external data sidecar — total weights exceed protobuf's 2GB inline
    # limit, so we ship `model.onnx` + `model.onnx.data` together. ORT-web
    # auto-loads the sidecar from the same directory.
    onnx_size = os.path.getsize(out_path) / (1024 * 1024)
    data_path = out_path + '.data'
    data_size = os.path.getsize(data_path) / (1024 * 1024) if os.path.exists(data_path) else 0
    print(f"Done. {out_path} = {onnx_size:.1f} MB; .data sidecar = {data_size:.1f} MB")
    print("  JS config: two inputs at 392×392 — siglip(mean=[0.5]*3,std=[0.5]*3), dinov2(ImageNet); single output prob_ai")


def convert_opensynthid():
    """Convert fyxme/opensynthid-detect-0.1 (.pt) to ONNX.

    A reverse-engineered SynthID watermark *surrogate* (not official Google).
    Architecture: DualStreamWatermarkNet — a ResNet34 spatial branch over
    RGB+wavelet-residual (4ch) plus a lightweight freq-CNN over FFT-log-mag +
    carrier-frequency-mask (2ch), fused to a single logit.

    The model takes ONE (1,6,512,512) tensor; the 6-channel construction
    (wavelet denoise residual, FFT log-magnitude, carrier mask) is replicated
    in JS in src/offscreen.js — see synthidPreprocess(). We fuse sigmoid into
    the graph so the JS side reads P(watermark) directly.

    Source for the architecture + preprocessing: the repo's infer.py.
    """
    import numpy as np
    import pywt
    from torchvision import models as tv_models

    out_dir = "models/OpenSynthID"
    ckpt_path = os.path.join(out_dir, 'model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. Download it first:\n"
            "  curl -L -o models/OpenSynthID/model.pt "
            "https://huggingface.co/fyxme/opensynthid-detect-0.1/resolve/main/model.pt"
        )

    # --- architecture (verbatim from infer.py) ---
    class DualStreamWatermarkNet(nn.Module):
        def __init__(self, spatial_in=4, freq_in=1, hidden_dim=256, pretrained=False, backbone="resnet18"):
            super().__init__()
            if backbone == "resnet34":
                self.spatial = tv_models.resnet34(weights=None)
            else:
                self.spatial = tv_models.resnet18(weights=None)
            self.spatial.conv1 = nn.Conv2d(spatial_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.spatial.fc = nn.Identity()
            self.freq = nn.Sequential(
                nn.Conv2d(freq_in, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Linear(512 + 128, hidden_dim),
                nn.ReLU(inplace=True), nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            spatial = x[:, :4, :, :]
            freq = x[:, 4:, :, :]
            s_feat = self.spatial(spatial)
            f_feat = self.freq(freq).flatten(1)
            return self.classifier(torch.cat([s_feat, f_feat], dim=1)).squeeze(1)

    # --- preprocessing (verbatim from infer.py, cv2 swapped for PIL+numpy so
    #     the converter has no OpenCV dependency — only used for the sanity /
    #     parity check, never shipped) ---
    def wavelet_denoise(channel, wavelet="db4", level=3):
        channel = np.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=0.0)
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        detail = coeffs[-1][0]
        sigma = np.median(np.abs(detail)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(channel.size))
        new_coeffs = [coeffs[0]]
        for details in coeffs[1:]:
            new_coeffs.append(tuple(pywt.threshold(d, threshold, mode="soft") for d in details))
        denoised = pywt.waverec2(new_coeffs, wavelet)
        return denoised[: channel.shape[0], : channel.shape[1]]

    def carrier_mask(size):
        carriers = [(14, 14), (-14, -14), (126, 14), (-126, -14), (98, -14), (-98, 14), (128, 128), (-128, -128)]
        mask = np.zeros((size, size), dtype=np.float32)
        c = size // 2
        for fy, fx in carriers:
            for yy, xx in ((c + fy, c + fx), (c - fy, c - fx)):
                if 0 <= yy < size and 0 <= xx < size:
                    mask[yy, xx] = 1.0
        return mask

    def fft_log_mag(gray):
        f = np.fft.fftshift(np.fft.fft2(gray))
        log_mag = np.log1p(np.abs(f))
        return ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)).astype(np.float32)

    def osid_preprocess(path, size=512):
        img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3) RGB
        residual = np.zeros((size, size, 3), dtype=np.float32)
        for c in range(3):
            residual[:, :, c] = arr[:, :, c] - wavelet_denoise(arr[:, :, c])
        residual_gray = residual.mean(axis=2)
        # cv2 RGB2GRAY weights, rounded to uint8 then /255
        gray = np.round(arr.dot([0.299, 0.587, 0.114]) * 255.0).astype(np.uint8).astype(np.float32) / 255.0
        chans = [arr.transpose(2, 0, 1), residual_gray[None], fft_log_mag(gray)[None], carrier_mask(size)[None]]
        x = np.concatenate(chans, axis=0)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.from_numpy(x).unsqueeze(0).float()

    print("\n=== fyxme/opensynthid-detect-0.1 ===")
    print(f"Loading checkpoint {ckpt_path}…")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    total_channels = int(ckpt.get('channels', 6))
    backbone = ckpt.get('args', {}).get('backbone', 'resnet34')
    freq_in = max(1, total_channels - 4)
    print(f"  channels={total_channels}, backbone={backbone}")

    model = DualStreamWatermarkNet(spatial_in=4, freq_in=freq_in, backbone=backbone)
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected (strict=False)")
    model.eval()

    class _ProbOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(self.m(x))

    wrapped = _ProbOnly(model).eval()

    # Sanity / early separation signal. We do NOT assert a verdict (this is an
    # unvalidated surrogate). SynthID is a Google watermark, so a Gemini image
    # *should* score higher than a real photo if the model works at all.
    probe = [
        ('test-images/ai-gemini-sample.png', 'Google/Gemini — expect HIGH'),
        ('test-images/ai-firefly-sample.jpg', 'Adobe Firefly — non-Google AI'),
        ('test-images/real-1.jpeg', 'real photo — expect LOW'),
    ]
    x = None
    prob = None
    for path, note in probe:
        if not os.path.exists(path):
            continue
        xi = osid_preprocess(path)
        with torch.no_grad():
            p = wrapped(xi).item()
        if x is None:
            x, prob = xi, p  # reuse the first for the ONNX parity check below
        print(f"  P(watermark)={p:.4f}  {os.path.basename(path)}  ({note})")
    if x is None:
        raise FileNotFoundError("No probe images found under test-images/")

    out_path = os.path.join(out_dir, 'model.onnx')
    print(f"Exporting to {out_path}…")
    dummy = torch.randn(1, total_channels, 512, 512)
    torch.onnx.export(
        wrapped, dummy, out_path,
        opset_version=17,
        input_names=['input'], output_names=['prob_ai'],
        dynamic_axes={'input': {0: 'batch'}, 'prob_ai': {0: 'batch'}},
        do_constant_folding=True,
    )
    _merge_external_data(out_path, out_dir)

    # Verify the ONNX graph reproduces the torch output on the same input.
    try:
        import onnxruntime as ortrt
        sess = ortrt.InferenceSession(out_path, providers=['CPUExecutionProvider'])
        onnx_prob = float(sess.run(['prob_ai'], {'input': x.numpy()})[0].reshape(-1)[0])
        print(f"  ONNX parity: torch={prob:.5f} onnx={onnx_prob:.5f} (Δ={abs(prob - onnx_prob):.2e})")
        assert abs(prob - onnx_prob) < 1e-3, "ONNX output diverges from torch"
    except ImportError:
        print("  (onnxruntime not installed — skipped ONNX parity check)")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB at {out_path}")
    print("  JS config: single input (1,6,512,512); channels [R,G,B,wavelet-residual,fft-logmag,carrier]; output prob_ai")


def convert_openfake():
    """Convert ComplexDataLab/OpenFakeDemo's detector (.safetensors) to ONNX.

    The model lives in a HF *Space*, not a model repo: it's a
    `microsoft/swinv2-base-patch4-window16-256` (SwinV2) backbone with the
    classification head replaced by a fresh 2-class `nn.Linear`, fine-tuned on
    real vs. synthetic images. Labels: 0 = real, 1 = fake. We replicate the
    Space's `app/model.py` loading exactly (build base model, swap head, strip
    DDP/Lightning key prefixes, load_state_dict strict=False), read the
    processor's input size + normalization, sanity-check, and export logits.

    Output is raw 2-class logits (like the other VISUAL_MODELS); the JS side
    applies softmax and reads index 1 (fake) as P(AI). Note: the Space applies a
    softmax *temperature* of 2.0 before reporting p_fake — we do NOT bake that in
    (it only rescales confidence and the JS threshold absorbs it), but it's why
    raw P(AI) here will look more confident than the demo's number.
    """
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import torch.nn.functional as F

    HF_NAME = 'microsoft/swinv2-base-patch4-window16-256'
    NUM_LABELS = 2
    out_dir = 'models/OpenFake'
    os.makedirs(out_dir, exist_ok=True)

    print('\n=== ComplexDataLab/OpenFakeDemo (SwinV2 real/fake) ===')
    print('Fetching fine-tuned weights from the Space…')
    weights_path = hf_hub_download(
        repo_id='ComplexDataLab/OpenFakeDemo',
        filename='model.safetensors',
        repo_type='space',
    )

    print(f'Building base {HF_NAME} + 2-class head…')
    model = AutoModelForImageClassification.from_pretrained(HF_NAME)
    processor = AutoImageProcessor.from_pretrained(HF_NAME)
    model.num_labels = NUM_LABELS
    model.config.num_labels = NUM_LABELS
    model.config.id2label = {0: 'real', 1: 'fake'}
    model.config.label2id = {'real': 0, 'fake': 1}
    model.classifier = nn.Linear(model.swinv2.num_features, NUM_LABELS)

    # Strip common wrapper prefixes (DDP "module.", Lightning "model.") —
    # verbatim from the Space's app/model.py _strip_prefixes.
    raw = load_file(weights_path)
    state_dict = {}
    for k, v in raw.items():
        nk = k
        for p in ('module.', 'model.'):
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        state_dict[nk] = v
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  missing keys ({len(missing)}): {missing[:6]}')
    if unexpected:
        print(f'  unexpected keys ({len(unexpected)}): {unexpected[:6]}')
    # The only legitimately-missing keys should be the few buffers SwinV2
    # recomputes; a swapped-in head with NO loaded weights would be a silent
    # disaster, so assert the classifier actually got weights.
    assert not any(k.startswith('classifier') for k in missing), (
        f'classifier head did not load from the checkpoint: {missing}'
    )
    model.eval()

    # Input size + normalization straight from the processor (256×256,
    # ImageNet norm for this SwinV2). transformers ≥5 SizeDict / dict / int.
    size = processor.size
    if hasattr(size, 'height') and getattr(size, 'height') is not None:
        h, w = size.height, size.width
    elif isinstance(size, dict):
        h = size.get('height') or size.get('shortest_edge')
        w = size.get('width') or size.get('shortest_edge')
    else:
        h = w = int(size)
    assert h and w and h == w, f'Non-square or unknown input size: {size}'
    input_size = int(h)
    mean = list(map(float, processor.image_mean))
    std = list(map(float, processor.image_std))
    print(f'  input: {input_size}×{input_size}, mean={mean}, std={std}, aiIndex=1 (fake)')

    # Sanity / separation probe. AI images should score high P(fake), real low.
    probe = [
        ('test-images/ai-firefly-sample.jpg', 'Adobe Firefly — expect HIGH'),
        ('test-images/ai-gemini-sample.png', 'Google/Gemini — expect HIGH'),
        ('test-images/ai-midjourney-sample.png', 'Midjourney — expect HIGH'),
        ('test-images/real-1.jpeg', 'real photo — expect LOW'),
        ('test-images/real-2.webp', 'real photo — expect LOW'),
    ]
    sane_x = None
    sane_logits = None
    for path, note in probe:
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert('RGB')
        inputs = processor(images=img, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
            p_fake = float(F.softmax(logits, dim=-1)[0, 1].item())
        if sane_x is None:
            sane_x, sane_logits = inputs['pixel_values'], logits
        print(f'  P(fake)={p_fake:.3f}  {os.path.basename(path)}  ({note})')
    if sane_x is None:
        raise FileNotFoundError('No probe images found under test-images/')

    out_path = os.path.join(out_dir, 'model.onnx')
    print(f'Exporting to {out_path}…')
    wrapped = _LogitsOnly(model).eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        wrapped, dummy, out_path,
        opset_version=17,
        input_names=['pixel_values'], output_names=['logits'],
        dynamic_axes={'pixel_values': {0: 'batch'}, 'logits': {0: 'batch'}},
        do_constant_folding=True,
    )
    _merge_external_data(out_path, out_dir)

    # Verify the ONNX graph matches torch on the same input.
    try:
        import onnxruntime as ortrt
        sess = ortrt.InferenceSession(out_path, providers=['CPUExecutionProvider'])
        onnx_logits = sess.run(['logits'], {'pixel_values': sane_x.numpy()})[0]
        delta = float(abs(sane_logits.numpy() - onnx_logits).max())
        print(f'  ONNX parity: max|Δlogit|={delta:.2e}')
        assert delta < 1e-3, 'ONNX output diverges from torch'
    except ImportError:
        print('  (onnxruntime not installed — skipped ONNX parity check)')

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'Done. {size_mb:.1f} MB at {out_path}')
    print(f'  JS config: inputSize={input_size}, mean={mean}, std={std}, '
          'aiIndex=1, interpret=softmaxBinary(tensor,1)')


def quantize_bombek1_fp16():
    """Convert the existing Bombek1 fp32 ONNX to fp16, halving the weight
    sidecar (~2.8GB → ~1.4GB) so it can actually load inside a Chrome
    extension. fp32 crashes the renderer on load — Chrome ArrayBuffer
    allocation can't reliably hold 2.8GB even before WebGPU sees it.

    Uses `keep_io_types=True` so the model's inputs/outputs stay fp32 (Cast
    nodes auto-inserted at the boundaries). This means the JS preprocessing
    can stay unchanged — only the internal weights become fp16.
    """
    from onnxconverter_common import float16

    out_dir = 'models/Bombek1-AIDetector'
    src = os.path.join(out_dir, 'model.onnx')
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"{src} not found. Run `python3.13 convert_model.py --bombek1-only` first."
        )

    print(f"\n=== Bombek1 fp16 quantization ===")
    print(f"Loading fp32 ONNX from {src} (and {src}.data)…")
    m = onnx.load(src)  # also loads referenced external data into memory

    print("Converting weights to fp16 (keep_io_types=True, disable_shape_infer=True)…")
    # disable_shape_infer is required: the default shape-inference pass
    # serializes the whole model proto inline, which blows past protobuf's
    # 2GB cap for a model this large.
    m_fp16 = float16.convert_float_to_float16(
        m, keep_io_types=True, disable_shape_infer=True
    )

    # Save into a tmp dir then atomically replace, so a failed write doesn't
    # leave a half-broken model in place.
    tmp_dir = out_dir + '_fp16_tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, 'model.onnx')

    print(f"Writing fp16 model to {tmp_path}…")
    onnx.save_model(
        m_fp16,
        tmp_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location='model.onnx.data',
    )

    print("Replacing originals…")
    os.replace(tmp_path, src)
    os.replace(os.path.join(tmp_dir, 'model.onnx.data'), src + '.data')
    os.rmdir(tmp_dir)

    onnx_size = os.path.getsize(src) / (1024 * 1024)
    data_size = os.path.getsize(src + '.data') / (1024 * 1024)
    print(f"Done. {src} = {onnx_size:.1f} MB; .data sidecar = {data_size:.1f} MB")


if __name__ == '__main__':
    SAMPLE = 'firefly-sample.jpg'  # known AI image, both models should call it AI

    if '--bombek1-fp16' in sys.argv:
        quantize_bombek1_fp16()
    elif '--bombek1-only' in sys.argv:
        convert_bombek1()
    elif '--opensynthid' in sys.argv:
        convert_opensynthid()
    elif '--openfake' in sys.argv:
        convert_openfake()
    else:
        convert_hf_image_classifier(
            repo_id='prithivMLmods/Deepfake-Detect-Siglip2',
            out_dir='models/Siglip2-Deepfake',
            sample_image=SAMPLE,
            ai_label_name='Fake',
        )
        convert_hf_image_classifier(
            repo_id='Ateeqq/ai-vs-human-image-detector',
            out_dir='models/AIvHuman',
            sample_image=SAMPLE,
            ai_label_name='ai',
        )
        if '--bombek1' in sys.argv:
            convert_bombek1()

        if '--cf' in sys.argv:
            convert_communityforensics()
