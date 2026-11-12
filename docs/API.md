# 计算约定与 API

## 网络和张量

`Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, rng=None)` 接收 `(N,C,H,W)`。核、步长和填充支持整数或二元组。输出空间尺寸为 `floor((H+2P-K)/S)+1`。实现的是常用深度学习卷积互相关；不翻转核，不支持 dilation/groups。im2col 会额外分配内存，适合小规模教学。

`Linear(in_features, out_features, rng=None)` 接收二维 `(N,D)`，权重排列 `(D,O)`。`Flatten` 保留批次维。`ReLU` 在零点采用导数 0。

`BatchNorm2d(C, eps=1e-5, momentum=0.1)` 沿 N/H/W 统计每通道均值及总体方差。训练更新 `running = (1-momentum)*running + momentum*batch`；推理使用冻结的运行统计。运行方差也使用总体方差，与部分框架保存无偏运行方差的约定不同。γ/β 是普通可训练参数。推理前向之后不能执行 BatchNorm 训练反向。

`SoftmaxCrossEntropyLoss.forward(logits, labels)` 返回批次均值；`labels` 为 `[0,C)` 的一维整数数组。反向返回 `(prob-one_hot)/N`。log-sum-exp 避免先 softmax 再取对数在大数值下的问题，但任何浮点系统仍有有限的可表示范围。

## 手动训练顺序

1. `model.train(); model.zero_grad()`。
2. `logits = model.forward(x)`，计算损失。
3. `model.backward(criterion.backward())`。
4. `optimizer.step(model)`。
5. 评估时 `model.eval()`，完成后恢复 `model.train()`。

`Sequential.parameters()` 返回名称到 `Parameter` 的映射；`.data` 和 `.grad` 可检查。`buffers()` 为不可训练运行统计。`zero_grad()` 将梯度设置为 `None`。SGD/Adam 跳过 `None`，并在修改参数前检查所有活动梯度的形状与有限性。该版本没有权重衰减、梯度累积或混合精度；不要在 Sequential 中复用同一个层对象。

`save/load` 使用命名 NumPy 数组；检查架构键、形状和有限性，加载压缩前后大小均限制为 64 MiB。加载成功后数据就地更新；不执行 Python 对象反序列化。仅加载你信任的本地文件。

## 数值和复现

默认参数为 float64，建议整个教学实验使用 float64。随机性通过 `np.random.default_rng(seed)` 显式传入。不同 BLAS、NumPy 版本和 CPU 会产生细微浮点差异。

`python -m numpy_dl check` 检查输入、权重及偏置/γ/β。`benchmark` 先验证独立循环卷积的相等性，再在同机同输入下计时；速度比不应跨硬件比较。
