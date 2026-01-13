# RailSeg-Mamba

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>


基于 **Mamba 状态空间模型** 的铁路场景三维点云语义分割网络,实现高效准确的铁路点云分割。


---

## 💻 环境要求

| 软件    | 版本          |
| ------- | ------------- |
| Ubuntu  | 20.04 / 22.04 |
| Python  | ≥ 3.8         |
| CUDA    | ≥ 11.6        |
| cuDNN   | ≥ 8.6         |
| PyTorch | ≥ 1.12.0      |

---

## 🔧 安装步骤

### 方式一：使用 Conda 环境（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/RailSeg-Mamba.git
cd RailSeg-Mamba

# 2. 创建 Conda 虚拟环境
conda create -n railseg-mamba python=3.10 -y
conda activate railseg-mamba

# 3. 安装 PyTorch (根据您的 CUDA 版本选择)
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装 Mamba 依赖
pip install causal-conv1d>=1.2.0
pip install mamba-ssm>=2.0.0

# 5. 安装 torchsparse (稀疏卷积库)
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0

# 6. 安装其他依赖
pip install -r requirements.txt
```

### 方式三：手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 升级 pip
pip install --upgrade pip setuptools wheel

# 3. 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装核心依赖
pip install causal-conv1d>=1.2.0
pip install mamba-ssm>=2.0.0

# 5. 安装 torchsparse
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0

# 6. 安装项目依赖
pip install -r requirements.txt
```

---

## 📁 数据准备

### 数据集结构

```
Railway360/
├── train/
│   ├── 0000000.bin
│   ├── 0000001.bin
│   ├── 0000002.bin
│   └── ...
└── test/
    ├── 0000100.bin
    ├── 0000101.bin
    └── ...
```

### 数据格式

每个 `.bin` 文件包含点云数据，格式为：

```
[x, y, z, intensity, category, timestamp]  # float32, shape: (N, 6)
```

### 类别定义

| ID   | 类别名称 | 描述        |
| ---- | -------- | ----------- |
| 0    | void     | 未标注/噪点 |
| 1    | tree     | 树木        |
| 2    | line     | 线缆        |
| 3    | pole     | 杆塔        |
| 4    | building | 建筑物      |
| 5    | rail     | 铁轨        |
| 6    | pathway  | 轨道床      |
| 7    | barrier  | 护栏        |
| 8    | hillside | 边坡        |
| 9    | tunnel   | 隧道        |

---

## 🙏 致谢

本项目的实现参考了以下优秀工作：

- [Mamba](https://github.com/state-spaces/mamba) - State Space Models
- [TorchSparse](https://github.com/mit-han-lab/torchsparse) - Sparse Convolution Library
- [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)
- [numpy-hilbert-curve](https://github.com/PrincetonLIPS/numpy-hilbert-curve)

---

