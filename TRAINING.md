# UAVCrack-R10 训练与核验

从本仓库根运行。三个模型族各seed42/43；最终模型使用1200张授权训练图。来源组主、次折分别984/216，训练组与验证组隔离。此轮整理只核验初始化、数据读取、前向和loss，没有重新训练42轮。

## 数据和初始化

比赛数据不随源码分享。取得使用授权后，将原始目录置于data/raw/UAV-Crack-dataset，包含leftImg8bit/train与gtFine/train。全量清单manifests/full_training_p59.csv和两个来源组清单manifests/split.csv、manifests/secondary_group_split_v2.csv由项目持有人另行提供；本地已保留，公开源码ZIP不含清单或图片。CSV列包含image、mask、source_group、split；入口校验原清单SHA、1200总数及984/216隔离。缺少原清单时不能声称精确重现原训练划分。

checkpoints目录需放置resnet34-b627a593.pth、mit_b2_20220624-66e8bf70.pth、convnext_tiny_1k_224_ema.pth。入口校验完整SHA。它们是ImageNet初始化，区别于分享的六份最终分割权重。ConvNeXt通过timm原加载流程读取本地文件；与历史URL缓存加载后所有参数逐张量一致。

## 环境

Python3.12。训练设置经过验证的直接依赖：Torch2.14.1+cu132、torchvision0.29.1、SMP0.5.0、timm1.0.15、Lightning2.5.1、MMEngine0.10.7、MMSeg1.2.2、MMCV2.1.0。requirements-training.txt固定直接依赖，不是全新训练环境的完整安装验收记录。

```bash
uv venv --seed --python python3.12 .venv-train
uv pip install --python .venv-train/bin/python --index-strategy unsafe-best-match -r requirements-training.txt
```

MiT训练还需要含ops的MMCV2.1.0，不能用mmcv-lite替代。为对应Torch/CUDA安装匹配的二进制包，或从官方v2.1.0源码编译。当前RTX5070Ti/CUDA13.2环境需要先对官方MMCV源码应用本仓库patches/mmcv-2.1.0-torch214-cxx20.patch，再以MMCV_WITH_OPS=1、对应CUDA_HOME和GPU架构编译安装。完整上游源码不随本仓库重复分发。使用其他硬件时按MMCV官方安装说明匹配工具链。训练环境从零安装和异机重训尚未单独验收。

## 入口

```bash
.venv-train/bin/python scripts/train_resunet_description.py --seed 42 --full-data --work-dir runs/resnet34_full_s42 --check-only
.venv-train/bin/python scripts/train_mit.py --seed 42 --full-data --work-dir runs/mit_full_s42 --check-only
.venv-train/bin/python scripts/train_convnext_ema.py --seed 42 --full-data --work-dir runs/convnext_full_s42 --check-only
```

检查通过后移除--check-only开始训练；将42改为43运行另一种子。留出训练用--fold primary或--fold secondary替换--full-data。ResNet默认主折；MiT与ConvNeXt必须显式选择。work-dir必须尚不存在，检查模式不写入模型或执行优化器更新。

ResNet34：42轮，batch8，独立H/V翻转各0.5，BCE+Dice，AdamW lr3e-4/wd1e-4，cosine42/eta_min1e-6，AMP。全量末轮42为交付状态。主折保留best仅作诊断，次折固定末轮比较。

MiT-B2：batch2、无增强、二分类CE，AdamW lr6e-5/wd.01、head倍率10；来源折19680步、warmup1500，全量24000步、warmup1829，其后线性PolyLR至0。使用AmpOptimWrapper动态loss scale。主折检查点cadence492，全量2400；交付使用固定末步。

ConvNeXt：42轮，batch8，BCE+Dice和相同翻转；AdamW编码器2e-5/解码器与头3e-4/wd1e-4，cosine42/eta_min1e-6、AMP。EMA decay .999，每训练batch结束更新参数和浮点buffer，整型buffer直接复制；EMA保持eval。最终保存ema_last.pth，live last.ckpt另保留恢复训练状态。

## 重新核验与打包

推理和ONNX导出见deploy/README.md。tools/cache_reference.py接受--weights、--input-dir、--output、--device与--limit，生成关闭TF32的独立Torch FP32参考。tools/validate_onnx.py在不含Torch的ORT环境运行，接受--models、--input-dir、--reference、--provider、--output和可选--original-zip；同时检查六模型单视图logits、融合概率和阈值掩码。参考缓存按需生成，不常驻仓库。

tools/audit_release.py核对六份权重、六个ONNX的SHA，300张原提交格式和分享ZIP CRC。tools/prepare_weights.py从显式、带SHA的可信训练状态清单提取state_dict。tools/package_release.py按显式JSON文件白名单打包并复核全部成员SHA。当前已验证的分享ZIP不自动重建覆盖。
