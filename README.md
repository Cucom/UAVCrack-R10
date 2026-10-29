# UAVCrack-R10

Final six-model solution for UAV-Crack binary crack segmentation.
2026-10-29 leaderboard snapshot: **mIoU 82.50%, rank 5/82**. The score refers
to the original CUDA AMP submission.

ResNet34 U-Net, MiT-B2 SegFormer and ConvNeXt-Tiny U-Net EMA are each trained
with seeds 42 and 43. Average family probabilities and fuse with weights
**0.65 / 0.25 / 0.10**, then threshold at **0.40**. ResNet/ConvNeXt use four
flips; MiT uses one view. See [MODEL_CARD.md](MODEL_CARD.md).

## Downloads

- [Version v1.0.0 release](https://github.com/Cucom/UAVCrack-R10/releases/tag/v1.0.0)
- [Six PyTorch state dictionaries](https://github.com/Cucom/UAVCrack-R10/releases/download/v1.0.0/uavcrack_r10_weights.zip)
- [Six FP32 ONNX graphs](https://github.com/Cucom/UAVCrack-R10/releases/download/v1.0.0/uavcrack_r10_onnx.zip)
- [SHA256 checksums](https://github.com/Cucom/UAVCrack-R10/releases/download/v1.0.0/SHA256SUMS.txt)

Retain the release for at least 90 days from publication; see
[availability policy](AVAILABILITY.md). The optional `python tools/download_models.py --format both` command downloads
model archives with built-in SHA256 verification.

Source, weights and ONNX are separate
downloads. Training data, report, environments and optimizer states are excluded.

## Inference

Python 3.12 was used for validation. From the repository root:

```bash
uv venv --seed --python python3.12 .venv-ort
uv pip install --python .venv-ort/bin/python -r deploy/requirements-onnx-cpu-lock.txt
unzip uavcrack_r10_onnx.zip -d artifacts
.venv-ort/bin/python deploy/predict_onnx.py \
  --input-dir /path/to/official_images \
  --model-dir artifacts/uavcrack_r10_final/models/onnx \
  --output outputs/onnx --provider cpu --expected-count 300
```

For original-precision CUDA AMP inference:

```bash
uv venv --seed --python python3.12 .venv-torch
uv pip install --python .venv-torch/bin/python --index-strategy unsafe-best-match -r deploy/requirements-torch-export-lock.txt
unzip uavcrack_r10_weights.zip -d artifacts
.venv-torch/bin/python deploy/predict_torch.py \
  --input-dir /path/to/official_images \
  --model-dir artifacts/uavcrack_r10_final/models/weights \
  --output outputs/torch --device cuda --amp --expected-count 300
```

Outputs contain same-stem PNG files under result/ and a result.zip. Masks are
672x378, grayscale L, background 0 / crack 1. Output directories must be new.
ONNX CPU inference does not require Torch, Lightning, MMCV or MMSegmentation.
CUDA ORT and ONNX export instructions are in [deploy/README.md](deploy/README.md).

## Training and reproduction

[TRAINING.md](TRAINING.md) contains full-data and source-group-fold training
commands for all three model families. Original split manifests and the
competition data are not part of this repository; exact training reproduction
requires the authorized original manifests. Initial pretrained encoder files
are separate from the six final segmentation weights. Direct training
requirements and the MMCV build patch are included.

Same-host fresh-environment Torch AMP inference reproduced all 300 original
PNG payloads exactly. All six re-exports retained identical ONNX SHA256.
ONNX is FP32: its CUDA result differed by 925 pixels from the original AMP
submission and is not assigned the original online score. CPU validation
covered 3 images; CUDA ORT covered 300. Full training and cross-machine
reproduction were not repeated for this release.

## License

Project contributions and the released final model weights: **Apache-2.0**.
See [LICENSE](LICENSE), [NOTICE](NOTICE), [weights license](WEIGHTS_LICENSE.md)
and [third-party notices](deploy/THIRD_PARTY_NOTICES.txt). Third-party code
retains its original license. Competition data are not licensed or distributed
by this repository.
