# 第4章 ONNXモデルの内部構造を読み解く
### 4.1 ONNXモデルファイルの基本
ONNXファイルはGoogleのProtocol Buffers形式で表現されています。内部には`ModelProto`という構造体があり、そこにモデルのグラフ、重み、メタデータが格納されています。PythonのONNXライブラリを使うと、以下のようにしてファイルを読み込み、構造を確認できます。

```python
import onnx

model = onnx.load("resnet18.onnx")
print(onnx.helper.printable_graph(model.graph))
```

`printable_graph`は人間が読みやすい形式でノードの情報を出力してくれます。出力されたテキストにはノード名、使用している演算子、入出力テンソルの名前などが含まれます。最初は情報量が多く感じるかもしれませんが、目立つキーワードを眺めるだけでもモデルの構造が見えてきます。

### 4.2 グラフ構造の理解
ONNXモデルは有向非巡回グラフ（DAG）で表現されます。ノードが演算（Operator）に対応し、エッジはテンソルの流れ、つまりデータの受け渡しを表します。例えばResNet18では、畳み込み（`Conv`）や活性化関数（`Relu`）、バッチ正規化（`BatchNormalization`）といったノードが連続しています。

各ノードには入力テンソルと出力テンソルが定義されます。テンソルには形状（Dimensions）とデータ型（float32、int64など）が設定されており、推論時にはこの形状に合わせてデータを渡す必要があります。形状が合わないと`InvalidGraph`や`InvalidArgument`といったエラーが発生するため、推論前に`model.graph.input`や`model.graph.output`を確認しておきましょう。

```python
for input_tensor in model.graph.input:
    tensor_type = input_tensor.type.tensor_type
    shape = [dim.dim_value for dim in tensor_type.shape.dim]
    print(f"Input {input_tensor.name}: shape={shape}")
```

このスニペットで入力テンソルの形状がわかります。バッチサイズが可変の場合は`dim_value`が0または空欄になることがあり、その場合は推論時に任意のバッチサイズを指定できます。

### 4.3 実用的な可視化
テキストだけでグラフ構造を追うのは大変なので、可視化ツールを使うと理解が進みます。`Netron`はONNXモデルを読み込んで、ノードの接続やテンソル形状をブラウザで確認できる人気ツールです。公式サイト（https://netron.app）にアクセスしてファイルをドラッグ＆ドロップするだけで可視化が始まります。

Netronでは各ノードをクリックすると、対応する演算子の属性（例えば畳み込みのフィルターサイズやストライド）を確認でき、中間テンソルの形状やデータ型も表示されます。複雑なモデルでも全体の流れを俯瞰できるため、オンボーディングやドキュメント作成に役立ちます。ローカル版のアプリも提供されているので、オフライン環境でも利用可能です。

## 図のアイデア
- images/ch04_graph_overview.png — 典型的なONNXグラフ構造の可視化イメージ

## 演習
1. Netronで任意のONNXモデルを開き、主要なノードとテンソル形状をスクリーンショット付きで記録する
2. Python APIで`model.graph.input`と`model.graph.output`を列挙するスクリプトを作成し、結果を解説する
