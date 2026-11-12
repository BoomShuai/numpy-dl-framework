# NumPy 深度学习框架

[![CI](https://github.com/BoomShuai/numpy-dl-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/BoomShuai/numpy-dl-framework/actions/workflows/ci.yml)

用 NumPy 手写前向计算、反向传播与参数更新的教学项目。可以逐层检查梯度、训练小型卷积网络、保存参数，并对照朴素卷积实现观察 im2col 的计算方式。

| 模块 | 已实现内容 |
| --- | --- |
| 网络层 | Conv2D、BatchNorm2d、Linear、ReLU、Flatten、Sequential |
| 数值计算 | 稳定 Softmax 交叉熵、中心差分梯度检查 |
| 优化器 | SGD（可选动量）、Adam；包括 BatchNorm 的 γ/β |
| 工程能力 | CLI、合成训练示例、参数检查点、测试、隐私扫描、可校验发布包 |

## 快速开始

Python 3.9+。建议新建虚拟环境，使用 Python 3.12。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m numpy_dl check
python -m numpy_dl train --epochs 15 --seed 42
python -m numpy_dl benchmark
```

也可使用 `numpy-dl` 命令。所有示例离线运行，训练数据在内存中生成，不下载数据集。

```python
import numpy as np
from numpy_dl import Linear, ReLU, Sequential, SoftmaxCrossEntropyLoss, Adam

rng = np.random.default_rng(42)
model = Sequential([Linear(4, 8, rng), ReLU(), Linear(8, 2, rng)])
criterion, optimizer = SoftmaxCrossEntropyLoss(), Adam(lr=0.001)
x = rng.normal(size=(6, 4))
y = np.array([0, 1, 0, 1, 0, 1])
model.zero_grad()
loss = criterion.forward(model.forward(x), y)
model.backward(criterion.backward())
optimizer.step(model)
```

## 保存与评估

```bash
python -m numpy_dl train --checkpoint runs/demo.npz
```

检查点仅包含浮点参数和 BatchNorm 运行统计，使用 `allow_pickle=False` 加载；不覆盖已有文件。加载前需构造相同网络，例如 `from numpy_dl.demo import demo_model; model = demo_model(); model.load("runs/demo.npz"); model.eval()`。优化器、随机数与训练进度未保存，不能据此精确恢复中断训练。

## 验证

```bash
python -m unittest discover -s tests -v
python scripts/privacy_check.py --history
python scripts/build_release.py --output dist/source.zip --source-date-epoch 1788134400
python scripts/verify_release.py dist/source.zip
```

测试涵盖矩形卷积核、步长/填充、非默认 γ 的 BatchNorm、极端 logits、SGD/Adam、参数往返以及合成训练。安装 PyTorch 后会额外执行卷积前向与反向数值对照。参考环境中的层梯度最大绝对误差小于 `1e-9`；这是小尺寸数值验证，不是全输入空间的正确性证明。合成任务的准确率不代表真实图像任务性能。

这是显式反向传播教学库：CPU、NCHW、浮点 NumPy 数组；不包含自动求导图、GPU、分布式训练或生产推理服务。每层只保存最近一次前向缓存；反向梯度覆盖而非累加，更新前不要穿插其他前向调用。

[计算约定与 API](docs/API.md) · [验证记录](docs/VALIDATION.md) · [隐私边界](PRIVACY.md) · [贡献说明](CONTRIBUTING.md)

许可证暂未指定；公开可见不等同于授予任意再分发许可。NumPy/PyTorch 为独立第三方依赖，遵循各自许可证。
