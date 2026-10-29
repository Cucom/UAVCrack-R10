# UAVCrack-R10 最终六模型推理与ONNX

2026 VLP 挑战赛参赛作品。本目录为最终方案的推理/导出入口；完整训练入口见仓库根TRAINING.md，比赛数据不随源码提供。与平台方法名统一为UAVCrack-R10，对应原始线上mIoU82.50%的冻结六份权重。

## 固定方法

- ResNet34 U-Net seed42/43：原图、水平、垂直、双翻转，按seed42四视图后seed43四视图顺序累加Sigmoid概率并除8。
- MiT-B2 SegFormer seed42/43：原图单视图，二分类Softmax裂缝概率均值。
- ConvNeXt-Tiny U-Net EMA seed42/43：各自四视图Sigmoid概率均值，再取双seed均值。
- 最终概率0.65×ResNet34+0.25×MiT-B2+0.10×ConvNeXt-EMA；阈值≥0.40。不增加后处理。
- 输入RGB原尺寸672×378。卷积分支输入float32[0,1]，图内ImageNet归一化后底部补6行再裁回；MiT图输入float32[0,255]，按原始255量纲归一化后补齐，先将头输出双线性恢复至384×672再裁回378行。predict_onnx.py负责按分支选择量纲、视图逆变换与概率融合。

## 整理后的目录

从deploy目录执行下文命令。按仓库根README解压模型后，运行用模型在../artifacts/uavcrack_r10_final/models/weights和../artifacts/uavcrack_r10_final/models/onnx；将分享权重包解压后，也可通过--model-dir指定实际位置。模型ZIP保留原验证目录布局。

只运行ONNX不需要Torch、Lightning、MMCV或MMSeg。入口启动时校验模型SHA，原训练检查点另行私有保存。

## 环境与命令

已在Linux x86_64、Python3.12.3、RTX5070Ti/驱动595.99.02上使用全新venv、无缓存下载安装验证。以下相对路径从source目录执行。用uv创建环境可避免系统Python缺少ensurepip：

```
uv venv --seed --python python3.12 .venv-onnx
uv pip install --python .venv-onnx/bin/python -r requirements-onnx-cpu-lock.txt
.venv-onnx/bin/python predict_onnx.py --input-dir /path/to/official_images --model-dir ../artifacts/uavcrack_r10_final/models/onnx --output /path/to/new_output --provider cpu --expected-count 300
```

CUDA ONNX使用独立环境，禁止同时安装CPU版和GPU版onnxruntime：

```
uv venv --seed --python python3.12 .venv-ort-cuda
uv pip install --python .venv-ort-cuda/bin/python -r requirements-onnx-cuda-lock.txt
.venv-ort-cuda/bin/python predict_onnx.py --input-dir /path/to/official_images --model-dir ../artifacts/uavcrack_r10_final/models/onnx --output /path/to/new_cuda_output --provider cuda --expected-count 300
```

CUDA锁文件提供CUDA13/cuDNN运行库，宿主机须具备兼容NVIDIA驱动。程序使用ONNX Runtime CUDAExecutionProvider、关闭TF32，CUDA不可用时直接报错；部分形状操作使用CPU是ORT正常图划分。不要据此把CUDA命令当作CPU兼容环境。

PyTorch原提交精度推理（CUDA AMP）及ONNX重导出：

```
uv venv --seed --python python3.12 .venv-torch
uv pip install --python .venv-torch/bin/python --index-strategy unsafe-best-match -r requirements-torch-export-lock.txt
.venv-torch/bin/python predict_torch.py --input-dir /path/to/official_images --model-dir ../artifacts/uavcrack_r10_final/models/weights --output /path/to/new_torch_output --device cuda --amp --expected-count 300
.venv-torch/bin/python export_onnx.py --weights-dir ../artifacts/uavcrack_r10_final/models/weights --output-dir /path/to/new_onnx_export
```

所有输出目录必须尚不存在。输出包含result/下同名PNG以及result.zip。PNG为原尺寸、L模式、背景0/裂缝1；程序不输出0/255可视化掩码。export_onnx.py使用opset17、固定batch1，每文件包含权重，不依赖外部.data文件。

## 复现范围

验证覆盖同机独立环境的最终六模型推理、CPU/CUDA ONNX数值检查、六模型重导出及交付包解压运行；不等同另一台机器验证，也没有重做训练。PyTorch CUDA AMP是原始提交的数值配方；ONNX为FP32，阈值附近可能出现少量掩码差异。详见仓库根MODEL_CARD.md中的实测结果，不能用原82.50%回执替代转换版本的新线上成绩。

原始最佳result.zip独立保留；转换后的输出不替换原包，不会自动上传平台。
