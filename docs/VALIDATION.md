# 验证范围

参考环境为 Python 3.12.13，NumPy 2.5.4。基础依赖的参考版本记录在 `requirements-core.txt`；正常安装按 `pyproject.toml` 兼容范围选择。参考锁定命令：`python -m pip install -r requirements-core.txt -e .`。锁定文件不包含本机路径，不保证所有平台的传递依赖逐位一致。

数值测试验证 Linear、Conv2D、非默认 γ 的 BatchNorm、Softmax CE 的输入及参数梯度；与独立循环卷积比较前向结果。安装 PyTorch 时额外比较卷积前向与反向。float64 中心差分检查最大绝对误差低于 1e-9。15轮合成训练的损失下降验证训练链路，不作为真实任务基准。

CI在 Linux/Windows 上运行 Python 3.9/3.12；PyTorch对照单独在Linux运行。项目不要求安装PyTorch。

发布工具测试覆盖可复现打包、源码白名单、拒绝覆盖、校验和及篡改检测。安装后的CLI还会在仓库之外的临时目录运行；发布ZIP检查逐文件大小和SHA-256。隐私扫描和人工检查覆盖公开源码与Git历史，不代表对所有潜在语义信息的绝对保证。
