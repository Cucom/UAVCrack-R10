# UAVCrack-R10 model card

Task: binary UAV image crack segmentation, RGB 672 x 378 pixels.
The final system contains six independently trained models:

| Family | Seeds | Inference | Fusion weight |
|---|---|---|---|
| ResNet34 U-Net | 42,43 | four flips | 0.65 |
| MiT-B2 SegFormer | 42,43 | single view | 0.25 |
| ConvNeXt-Tiny U-Net EMA | 42,43 | four flips | 0.10 |

Average probabilities within each family, combine with the fixed weights,
and mark crack where probability >= 0.40. No morphological postprocessing.
Convolutional training: 42 epochs on 1,200 labeled images. MiT training:
24,000 batch-2 iterations. Encoders start from ImageNet initialization.
The ConvNeXt EMA decay is 0.999. See TRAINING.md for the full settings.

Official leaderboard snapshot supplied on 2026-10-29: mIoU 82.50%, crack F1
80.53%, crack IoU 67.40%, precision 80.41%, recall 80.65%, aAcc 97.71%;
rank 5 of 82. These scores describe the original CUDA AMP submission only.

A fresh same-host Torch environment reproduced all 300 submitted PNG payloads
byte for byte. Six ONNX FP32 exports were reproduced with identical SHA256.
Pure CUDA ORT over 300 images differed by 2 thresholded pixels from matching
FP32 Torch and 925 pixels from the original AMP masks. CPU ORT was checked on
3 images. The ONNX output has not been scored as a new official submission.
No full retraining or cross-machine reproduction was performed during release
validation. Results outside this dataset, including different capture settings
or image geometry, have not been established by these measurements.

License: Apache-2.0 for project contributions and the final released weights,
with retained third-party notices. Competition data are not redistributed.
