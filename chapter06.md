# 第6章 ONNX Runtimeによる推論とパフォーマンスチューニング
### 6.1 ONNX Runtimeの基本操作
ONNX RuntimeはPython、C++、C#などさまざまな言語バインディングを提供しています。ここではPython APIを使った最小の推論例を紹介します。

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "resnet18.onnx",
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: dummy_input})
print(result[0].shape)
```

`providers`引数で使用したい実行プロバイダを指定します。GPUを利用する場合は`"CUDAExecutionProvider"`や`"TensorrtExecutionProvider"`を並べると、利用可能なものが自動的に選択されます。実行プロバイダは優先順位の高い順に並べるのがコツです。

### 6.2 パフォーマンス計測
推論性能を確かめるには、実際に複数回推論して平均時間を測定するのが確実です。以下はPythonの`time`モジュールを使った簡易ベンチマークの例です。

```python
import time

runs = 50
start = time.time()
for _ in range(runs):
    session.run([output_name], {input_name: dummy_input})
end = time.time()

print(f"Average latency: {(end - start) / runs * 1000:.2f} ms")
```

バッチサイズを増やすとスループット（1秒あたりの処理件数）は向上する一方、1件あたりのレイテンシが上がることがあります。対象アプリケーションがリアルタイム性を重視するのか、バッチ処理を重視するのかを明確にし、測定条件を合わせることが重要です。

### 6.3 高度な最適化手法
ONNX Runtimeはグラフ最適化のレベルを`session_options.graph_optimization_level`で設定できます。`ORT_ENABLE_ALL`にすると、演算子の融合や不要なノードの削除など幅広い最適化が適用され、速度向上が期待できます。

```python
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    "resnet18.onnx",
    sess_options=session_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

さらに高速化が必要な場合は、ONNX RuntimeとTensorRTやOpenVINOを組み合わせる方法があります。TensorRTはNVIDIA GPU向けの最適化エンジンで、精度の高いFP32のままでも高速化できますし、INT8やFP16に量子化することで一段と速くなります。OpenVINOはIntel製CPUやVPUに最適化されており、クラウドからエッジまで幅広いデバイスで利用できます。

## 図のアイデア
- images/ch06_runtime_flow.png — ONNX Runtimeによる推論処理のフローチャート

## 演習
1. CPUとGPUそれぞれで同じONNXモデルを実行し、平均レイテンシを計測して比較レポートを作成する
2. `SessionOptions`を変更しながらGraph Optimization Levelの影響を検証する実験計画を立てる
