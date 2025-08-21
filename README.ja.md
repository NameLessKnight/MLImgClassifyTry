## Languages

- [English](README.md)
- [中文](README.zh.md)
- [日本語](README.ja.md)

## プロジェクト概要

これは PyTorch を使った小さな画像分類プロジェクトです。ResNet-18 を訓練して画像を分類し、未分類の画像を一括で整理できます。

プロジェクトの目的：iPhone の写真とネットからダウンロードした画像が混ざっていて整理が面倒；最近は機械学習が流行っているので遊んでみたら楽しかった。プロジェクトはすべて AI によって生成されました — AI は本当にすごい！！！

主なスクリプト：
- `train.py`：訓練スクリプト。`dataset/train` と `dataset/val` のサブフォルダをクラスとして読み込み、訓練後に `best_classifier.pth` を保存します（デフォルト）。
- `run.py`：推論／一括整理スクリプト。`unsorted_images` の画像を予測ラベルに応じて `sorted_images/<class>/` にコピーします（デフォルトのクラスは `anime`, `camera`, `other`）。

## ディレクトリ構成

- `train.py`    # 訓練エントリ
- `run.py`      # 推論／バッチ分類
- `dataset/`    # `train/` と `val/` を含む必要があり、各サブフォルダがクラスになる
- `unsorted_images/`  # 分類したい画像を置く
- `sorted_images/`    # `run.py` によって生成される出力フォルダ

## 動作環境とインストール

- Python 3.8+
- PyTorch + torchvision
- Pillow
- 任意：NVIDIA GPU と対応する CUDA ドライバ

仮想環境の利用を推奨します（リポジトリに `pytorch-env/` フォルダが含まれていますが、新しく作っても構いません）：

```powershell
# 仮想環境の作成と有効化（例）
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

PyTorch は CUDA バージョンに合わせてインストールしてください。PowerShell の例：

CPU のみ：
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

CUDA の例：
```powershell
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

インストール後、PyTorch と CUDA の確認：

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

その他の依存関係をインストール：

```powershell
pip install -r requirements.txt
```

## ファイルと使い方の説明

- `train.py`
  - 訓練スクリプト。`dataset/train` と `dataset/val` のサブフォルダをクラスとして使用し、デフォルトで `best_classifier.pth` を保存します。

- `run.py`
  - 推論と一括整理用スクリプト。`unsorted_images/` を読み、`sorted_images/<class>/` に画像をコピーします。
  - デフォルトで `best_classifier.pth` を読み込みます。
  - スクリプト内の `classes` リストが出力フォルダ名を決めるので、訓練時のラベル順と一致させてください。

- `dataset/train/<class_name>/` と `dataset/val/<class_name>/`
  - 各クラス名のサブフォルダに訓練／検証画像を入れてください（例：`anime`, `camera`, `other`）。
  - 推奨フォーマット：`.jpg`, `.jpeg`, `.png`。
  - 画像サイズは訓練コードがリサイズ／クロップします（通常 224x224）。

- `unsorted_images/`
  - 整理したい画像をここに置いてください。デフォルトではこのフォルダ直下のファイルのみをスキャンします（再帰処理はされません）。

- `sorted_images/`
  - `run.py` によって生成される出力フォルダ。画像はコピーされ、元のファイルは削除されません。

- `best_classifier.pth`
  - `train.py` によって保存されるモデルの重みファイル。

注意事項：

- クラスの順序が重要です：訓練時のラベル→インデックスのマッピングが `run.py` 内の `classes` と一致している必要があります。
- ファイル形式とアクセス権：画像ファイルが読み取り可能であること、パスで問題を起こす特殊文字を避けてください。

## クラスの説明

デフォルトのクラス：`anime`, `camera`, `other`。

- `anime`：イラスト、アニメ風や手描き風の画像（2D キャラ、CG、漫画など）。
- `camera`：スマホやカメラで撮影された実写真（人物、風景、日常写真など）。
- `other`：スクリーンショット、ウェブキャプチャ、アイコン、表、広告、判別しにくい画像など。

クラスを変更・追加する場合は、`run.py` の `classes` リストを更新し、訓練時のラベル順を保持してください。

## 訓練

```powershell
python .\\train.py
```

訓練が終わると `best_classifier.pth` が生成されます。

## 推論 / 一括整理

```powershell
python .\\run.py
```

実行後、`unsorted_images/` の画像が予測クラスごとに `sorted_images/` にコピーされます。
