# 第5章 代表的フレームワークからのモデル変換実践
### 5.1 PyTorchからONNXへ
PyTorchはONNXへの変換が非常にシンプルです。`torch.onnx.export`関数にモデル、ダミー入力、出力ファイル名を渡すだけで基本形が完成します。先ほどのResNet18を例にすると以下のようになります。

```python
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=17,
)
```

`dynamic_axes`を指定すると、バッチサイズを可変にしたONNXモデルを生成できます。`opset_version`はONNXの演算子セットのバージョンで、一般的には最新の安定版（2024年時点では17や18）を選べば問題ありません。

変換後は`onnx.checker.check_model`でモデルの整合性を確認し、`onnxruntime.InferenceSession`で推論テストをすると安心です。

### 5.2 TensorFlowからONNXへ
TensorFlowでは`tf2onnx`を利用する方法が一般的です。以下はTensorFlow 2.xで作成したKerasモデルをONNXへ変換する例です。

```bash
pip install tf2onnx
python -m tf2onnx.convert \
  --saved-model ./saved_model \
  --output model.onnx \
  --opset 17
```

すでにKerasの`model`オブジェクトがある場合は、Pythonスクリプト内で`tf2onnx.convert.from_keras`を呼び出して変換できます。TensorFlowは入力形状を厳密に指定しないとエラーになることがあるため、SavedModelをエクスポートする前に`model.build(input_shape=(None, 224, 224, 3))`のように形を確定させておくとトラブルを避けられます。

### 5.3 変換時のトラブルシューティング
変換時に「未対応の演算子がある」と言われることがあります。これはONNXの仕様にまだ追加されていない特殊な演算を使っている場合に発生します。対処方法としては、該当箇所を標準的なレイヤーに置き換える、あるいはカスタムオペレーターを実装する方法があります。PyTorchの場合、`torch.onnx.register_custom_op_symbolic`で独自の演算子を登録することが可能です。

変換後に推論結果が元モデルと異なる場合は、入力を同じにして両者の中間出力を順番に比較します。ONNX Runtimeでは`session.run`に`RunOptions`を渡し、中間テンソルをファイルに出力できるため、差分のある層を特定しやすくなります。少し手間はかかりますが、一つずつ比較していくと原因が見えてきます。

## 図のアイデア
- images/ch05_conversion_matrix.png — 各フレームワークと変換ツールの対応表

## 演習
1. PyTorchモデルとTensorFlowモデルをそれぞれONNXに変換し、`onnx.checker.check_model`で検証する
2. 未対応演算子が出たときの代替策を3案考え、メリット・デメリットを比較する
