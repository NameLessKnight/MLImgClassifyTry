## 项目简介

这是一个基于 PyTorch 的图片分类小项目，使用 ResNet-18 对图片进行训练并对未分类图片进行批量分类/整理。

项目初衷：iphone 图片和网络下载的图片混在一起，整理起来很麻烦；近些年机器学习比较火，就顺便玩了玩，感觉还不错。整个项目都是 AI 生成的，AI 真是牛逼！！！

主要脚本：
- `train.py`：训练脚本，按 `dataset/train` / `dataset/val` 下的文件夹结构读取数据并训练模型，训练完成后会保存为 `best_classifier.pth`。
- `run.py`：推理/整理脚本，会把 `unsorted_images` 下的图片按预测类别复制到 `sorted_images` 下对应文件夹（默认为 `anime`, `camera`, `other`）。

## 目录结构

- `train.py`    # 训练主脚本
- `run.py`      # 推理/批量分类脚本
- `dataset/`    # 数据集目录，需包含 `train/` 与 `val/`，每个子文件夹为一个类别
- `unsorted_images/`  # 待分类图片（示例）
- `sorted_images/`    # 分类结果输出目录（运行 `run.py` 后生成）

## 环境要求与安装

- Python 3.8+
- PyTorch + torchvision
- Pillow
- 建议（可选）：NVIDIA GPU + 对应的 CUDA 驱动

建议使用虚拟环境（本仓库已有 `pytorch-env/`，也可以新建）：

```powershell
# 新建并激活一个虚拟环境（示例）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装 PyTorch 时请根据你的 CUDA 版本选择合适的二进制包。以下示例在 PowerShell 中使用 pip：

（1）CPU-only（无 CUDA）：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

（2）常见 CUDA 版本示例：

```powershell
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```


安装完成后，可运行下面的简单命令检查 PyTorch 与 CUDA 是否可用：

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

然后安装其它依赖：

```powershell
pip install -r requirements.txt
```

## 目录与文件说明

下面对仓库中常用路径和文件做详细说明，帮助你把数据和权重放到合适的位置：

- `train.py`
	- 训练主脚本。读取 `dataset/train` 与 `dataset/val` 下每个子文件夹作为一个类别。训练时会保存权重为 `best_classifier.pth`（默认）。

- `run.py`
	- 推理与批量整理脚本。默认会从 `unsorted_images/` 读取要分类的图片，并把预测结果复制到 `sorted_images/<class>/` 下。
	- 加载 `best_classifier.pth`（默认）。
	- 脚本中有 `classes` 列表，决定输出子文件夹的顺序与名称；确保训练标签的顺序与这里一致。

- `dataset/train/<class_name>/`
	- 训练图片目录。每个子文件夹名称即为一个类别的标签（例如 `anime`, `camera`, `other`）。
	- 推荐图片格式：`.jpg`, `.jpeg`, `.png`。名称可以任意，但尽量无特殊字符。建议每类至少几十张图，数据越多效果越好。
	- 图片尺寸：训练代码会做缩放/裁剪（通常为 224x224），你无需事先统一尺寸，但保证图片不是极端小（例如 < 64px）。

- `dataset/val/<class_name>/`
	- 验证/评估集，结构与 `train/` 相同。用于在训练过程中评估模型性能。

- `unsorted_images/`
	- 待分类的图片集合。把所有需要整理的图片放在这个文件夹下（脚本默认只扫描该目录下的文件，不递归子目录——如果你有子目录，需要先汇总或修改 `run.py`）。
	- 支持的文件格式：`.jpg`, `.jpeg`, `.png`。其他格式可能无法读取。

- `sorted_images/`
	- 运行 `run.py` 后生成的输出目录。脚本会为每个 `class` 创建子文件夹并把图片复制进去（不会删除原始文件）。

- `best_classifier.pth`
	- 训练脚本默认保存的权重文件。

- `requirements.txt`
	- 列出项目依赖，可用 `pip install -r requirements.txt` 安装。

- `pytorch-env/` 或其他虚拟环境文件夹
	- 可选的虚拟环境目录，包含 Python 可执行与 site-packages，方便隔离依赖。

- `README*.md`
	- 文档文件（当前文件为中文 README，另有英文/日文翻译）。

常见注意事项：

- 类别顺序非常重要：训练时标签到索引的映射必须和 `run.py` 中的 `classes` 保持一致，否则分类结果会对不上。
- 图片格式与权限：确保脚本运行用户对图片文件具有读取权限，且文件名不包含导致路径解析错误的特殊字符。
- 批量处理：`run.py` 默认会把图片复制到 `sorted_images` 下，如果你希望移动而非复制，请在运行前修改脚本并做好备份。

## 类别说明

本项目中默认使用的类别为 `anime`, `camera`, `other`。简要说明如下：

- `anime`：以插画、动漫风格或手绘风格为主的图片（例如二次元角色、CG 插画、漫画截取）。
- `camera`：真实相片，通常由手机或相机拍摄，包含自然场景、人像、生活照等真实世界照片。
- `other`：不属于上述两类的图像，例如截屏、网页抓图、图标、表格、广告图片、混合风格或难以判别的图片。

注意：如果你修改或扩展了类别，请同时更新 `run.py` 中的 `classes` 列表，并保证训练时使用的标签顺序与推理时一致。

## 训练

```powershell
python .\train.py
```

训练结束会生成 `best_classifier.pth`。

## 推理 / 批量整理

```powershell
python .\run.py
```

结束后会自动复制分类图片从`unsorted_images/`到 `sorted_images/`。