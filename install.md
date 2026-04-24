# FoundationPose 安装指南（新版 conda 流程）

本文档用于 FoundationPose 的新版 conda 安装流程整理，仅覆盖当前已验证的环境组合：

- Linux
- conda
- Python 3.9
- CUDA 12.1
- PyTorch 2.4.1

如果你准备使用仓库当前这套 `requirements_new.txt`、`C++17` 扩展编译配置和 `build_all_conda.sh`，请以本文档为准。

> 说明：`readme.md` 中的 conda 安装段仍然是旧流程，不适用于这里这套新版环境。

## 1. 环境前提

开始前请先确认本机已经具备以下基础环境：

- 已正确安装 conda
- 已正确安装 CUDA 12.1，并且 `nvcc` 可用
- 系统编译器支持 C++17，例如较新的 `g++`
- 已安装 `cmake`

可先执行以下命令做快速检查：

```bash
python --version
nvcc --version
g++ --version
cmake --version
```

## 2. 创建 conda 环境

建议单独创建一个新的 conda 环境：

```bash
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
python -m pip install --upgrade pip
```

## 3. 安装 PyTorch 2.4.1（CUDA 12.1）

本文档只保证下面这组版本已经验证通过：

- `torch==2.4.1`
- `torchvision==0.19.1`
- `torchaudio==2.4.1`
- CUDA 12.1

安装命令如下：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

## 4. 安装 Python 依赖

仓库针对新版环境提供了新的依赖文件 `requirements_new.txt`，请不要再使用旧的 `requirements.txt`。

```bash
python -m pip install -r requirements_new.txt --no-build-isolation
```

然后安装 Eigen 3.4.0：

```bash
conda install -c conda-forge eigen=3.4.0 -y
```

## 5. 安装需要源码编译的依赖

先安装 `nvdiffrast`：

```bash
python -m pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

再安装 `pytorch3d`：

```bash
python -m pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
```

## 6. 编译仓库内扩展

当前仓库中的编译脚本已经按新版环境做过调整：

- `bundlesdf/mycuda/setup.py` 使用 `C++17`
- Eigen 会优先从当前 conda 环境中查找
- `build_all_conda.sh` 会编译 `mycpp` 和 `bundlesdf/mycuda`

执行下面的命令完成编译：

```bash
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

## 7. 安装完成验证

安装结束后，建议至少执行下面两组检查。

先确认 PyTorch 版本与 CUDA 版本：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

预期输出应包含：

```text
2.4.1
12.1
```

再确认关键依赖和扩展可以正常导入：

```bash
python -c "import pytorch3d; import nvdiffrast.torch as dr; import common; import gridencoder; print('install ok')"
```

如果最后输出 `install ok`，说明这套安装流程已经基本可用。

## 8. 常见说明

1. 本文档只覆盖 `CUDA 12.1 + torch 2.4.1` 这一组已验证组合，其他版本不保证兼容。
2. 命令中的 `--no-build-isolation` 不要去掉。这里的源码包需要复用当前环境中的 `torch`、`pybind11` 等依赖。
3. 如果你修改了 Python 版本，`CMAKE_PREFIX_PATH` 中的 `python3.9` 路径也需要对应调整。
4. 如果扩展编译失败，优先检查 `nvcc`、`g++`、`cmake`、`CONDA_PREFIX` 和 `eigen=3.4.0` 是否都正确可用。
