# 第7章 モデル最適化と量子化の基礎
### 7.1 なぜ最適化が必要か
推論環境は学習環境に比べてリソースが限られていることが多く、特にエッジデバイスではメモリ容量や電力が制約になります。モデルが重いままだと推論に時間がかかり、ユーザー体験が悪化したり、バッテリー消費が激しくなったりします。ONNX経由でモデルを扱っていると、変換後に最適化ツールをかけるだけで速度向上やメモリ削減が見込めることが多いため、最適化は非常に重要です。

### 7.2 ONNX Runtimeによる最適化
ONNX Runtimeには`onnxruntime-tools`という付属ツールがあり、グラフ最適化を自動で適用できます。例として、BERTなどのTransformerモデル用に特化した`optimizer.py`を使うと、Layer NormalizationやGELUの演算を統合して高速化できます。

```bash
pip install onnxruntime-tools
python -m onnxruntime_tools.optimizer_cli \
  --input bert.onnx \
  --output bert_optimized.onnx \
  --model_type bert
```

`onnxoptimizer`という別のツールセットもあり、共通の最適化パス（不要ノード削除、定数伝播など）を適用できます。最適化後は必ず`onnx.checker.check_model`や`onnxruntime.InferenceSession`で正しく動作するか確認しましょう。

### 7.3 量子化（Quantization）の基本
量子化とは、モデル内部の重みやアクティベーションを低精度（例：float32からint8）に変換することで軽量化し、計算速度を上げる手法です。ONNX RuntimeはPost Training Quantization（PTQ）をサポートしており、学習済みモデルに少量の代表データを流して統計情報を取得し、量子化を実行します。

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="bert_optimized.onnx",
    model_output="bert_int8.onnx",
    op_types_to_quantize=["MatMul", "Attention"],
    weight_type=QuantType.QInt8,
)
```

量子化後は精度が落ちていないか確認することが大切です。PTQで精度が許容範囲に収まらない場合は、学習時から量子化を意識したQuantization Aware Training（QAT）に取り組む必要があります。QATでは訓練段階で量子化の影響をシミュレーションしながら学習するため、精度の低下を最小限に抑えられます。

## 図のアイデア
- images/ch07_optimization_pipeline.png — 最適化と量子化のパイプライン図

## 演習
1. ONNX Runtime Toolsを使ってTransformerモデルを最適化し、ノード数とモデルサイズの変化を記録する
2. 動的量子化と静的量子化の違いを表形式で整理し、適用時の注意点を解説する
