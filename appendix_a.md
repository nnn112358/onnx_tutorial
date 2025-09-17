# 付録A 主要ツールとAPIリファレンス
ONNXとONNX Runtimeを使いこなすうえで、よく利用するAPIとCLIツールをまとめました。リファレンスとして活用してください。

### A.1 Python APIの要点
- `onnx.helper.make_tensor_value_info(name, elem_type, shape)`: 入力や出力の定義を作成します。カスタムモデルを生成するときに便利です。
- `onnx.helper.make_node(op_type, inputs, outputs, **attributes)`: 新しいノードを定義するときに使用します。
- `onnx.save(model, path)`, `onnx.load(path)`: ONNXファイルの読み書きに利用します。
- `onnx.checker.check_model(model)`: モデルの整合性を検証します。変換後は必ず実行しましょう。

ONNX Runtimeでは、`InferenceSession`のほかに`SessionOptions`や`RunOptions`が重要です。`SessionOptions`でグラフ最適化レベルやスレッド数を設定し、`RunOptions`でログ出力の詳細度やプロファイルを制御できます。

### A.2 CLIツール
- `onnxruntime_perf_test`: ONNXモデルの推論性能を測定するための公式ツール。入力データの形状を指定してベンチマークを実行できます。
- `onnxruntime_tools.optimizer_cli`: Transformer系モデルを中心に、最適化を適用して最終的なONNXファイルを出力します。
- `onnxsim`: ONNX Simplifierと呼ばれるサードパーティツールで、不要な演算を削除してモデルを単純化します。導入後の推論が安定しやすくなることがあります。

### A.3 ONNX Model Zooの活用
ONNX Model Zoo（https://github.com/onnx/models）は、ONNX形式で公開されている学習済みモデルのコレクションです。画像分類、自然言語処理、音声認識などカテゴリーごとに整理されており、すぐに試せるサンプルコードが付属しています。利用する際はモデルごとにライセンスが異なるので、プロジェクトの用途に合致しているか必ず確認してください。
