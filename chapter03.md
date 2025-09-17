# 第3章 開発環境を整える―ツールとセットアップ
### 3.1 必要なソフトウェア
ONNXの学習環境はPythonを中心に整えます。Python 3.8以降であれば主要ライブラリがサポートしているため安心です。Pythonが複数バージョンインストールされている場合は、`pyenv`や`conda`などのバージョン管理ツールを使うと切り替えが簡単です。

依存パッケージは用途によって異なりますが、最低限インストールしたいのは次のとおりです。

- `onnx`: ONNXファイルを読み書きするための公式ライブラリ。
- `onnxruntime`（必要であれば`onnxruntime-gpu`）: 推論を実行するエンジン。
- `protobuf`: ONNXファイルが内部で利用するGoogle製のデータフォーマットを扱うためのライブラリ。

学習用のフレームワーク（PyTorchやTensorFlow）もあわせてインストールします。GPUを利用する場合は、CUDAのバージョンとフレームワークの対応関係を公式ドキュメントで確認しておくとトラブルを避けられます。

### 3.2 環境構築手順（例：PyTorch + ONNX）
ここでは、PyTorchとONNX Runtimeを同じ仮想環境に入れる手順を紹介します。Windows、macOS、Linuxいずれでも概ね同じ流れです。

1. 作業用ディレクトリを決めて、Pythonの仮想環境を作成します。

   ```bash
   python -m venv onnx-env
   source onnx-env/bin/activate  # Windowsでは Scripts\activate.bat
   ```

2. 仮想環境がアクティブになったことを確認したら、必要なライブラリをまとめてインストールします。

   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio onnx onnxruntime onnxruntime-gpu protobuf
   ```

   GPUを使用しない場合は`onnxruntime-gpu`を省略して構いません。

3. 開発にはJupyter NotebookやVS Codeがあると便利です。Jupyterを使う場合は、次のコマンドでNotebook環境を準備できます。

   ```bash
   pip install notebook
   jupyter notebook
   ```

   ブラウザが開いたら、新しいノートブックを作成して動作確認を行いましょう。

### 3.3 サンプルモデルの準備
環境が整ったら、まずはシンプルなモデルでONNX変換の流れを体験します。PyTorchに付属する学習済みモデル（`torchvision.models`）を利用すると手軽です。次のコードは、学習済みのResNet18モデルを読み込み、ダミーの入力データを使ってONNX形式に書き出す例です。

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
)
```

このコードを実行すると作業ディレクトリに`resnet18.onnx`が作成されます。最初のうちは層の構造がシンプルなモデルで試し、ONNXファイルの中身を確認する習慣をつけましょう。Chapter 4ではこのONNXファイルの読み方を詳しく紹介します。

## 図のアイデア
- images/ch03_env_setup.png — 推奨開発環境の構成図（仮想環境・ツールチェーン）

## 演習
1. 仮想環境を実際に構築し、インストールしたパッケージとバージョンをリスト化する
2. PyTorchのサンプルモデルをONNXへ変換し、出力ファイルのメタデータを確認するスクリプトを書く
