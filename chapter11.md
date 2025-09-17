# 第11章 ケーススタディ2：自然言語処理モデルを活用する
### 11.1 課題設定
自然言語処理の代表的なタスクである質問応答（Question Answering）を題材に、BERTベースのモデルをONNXで活用する方法を学びます。Hugging Face Transformersから`bert-base-uncased`をダウンロードし、ONNXに変換して推論を行います。TransformersライブラリはONNXエクスポート用のユーティリティを提供しているため、比較的簡単に変換できます。

```bash
pip install transformers onnxruntime onnx
python -m transformers.onnx --model=bert-base-uncased onnx-model/
```

このコマンドを実行すると`onnx-model/model.onnx`が生成されます。変換されるモデルは質問応答だけでなく、多くの自然言語処理タスクに対応できる汎用的な構造を持ちます。

### 11.2 推論パイプライン
Hugging Faceのトークナイザーを使って入力テキストをトークン化し、ONNX Runtimeで推論を行うコードは次の通りです。

```python
from transformers import BertTokenizer
import onnxruntime as ort
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
session = ort.InferenceSession("onnx-model/model.onnx")

def answer(question: str, context: str) -> str:
    encoded = tokenizer(question, context, return_tensors="np")
    outputs = session.run(None, dict(encoded))
    start_logits, end_logits = outputs
    start = int(np.argmax(start_logits))
    end = int(np.argmax(end_logits)) + 1
    tokens = encoded["input_ids"][0][start:end]
    return tokenizer.decode(tokens)

print(answer("Where do penguins live?", "Penguins live in the Southern Hemisphere."))
```

ここではNumPy形式に変換されたトークン化結果をONNX Runtimeにそのまま渡しています。大規模なアプリケーションでは、バッチ処理に対応したり、複数モデルを切り替えたりする仕組みを用意すると拡張性が高まります。入力データの長さが上限を超えないように、事前に切り詰める処理を加えることも忘れないでください。

### 11.3 本番運用の工夫
質問応答モデルは計算量が多いため、レイテンシを抑える工夫が欠かせません。前処理後のトークンをキャッシュしておく、複数リクエストをまとめて推論するバッチングを導入する、といった方法が効果的です。ONNX Runtimeには`run_with_iobinding`を使って入出力を事前に確保し、メモリアロケーションのオーバーヘッドを減らすテクニックもあります。

モデルサイズが大きすぎる場合は、蒸留（Distillation）済みのBERT（DistilBERTなど）を使うか、Chapter 7で紹介した量子化を適用するとよいでしょう。サーバーサイドではGunicornやUvicornのワーカー数を調整し、サーバーインスタンスを水平スケールさせることで高負荷にも対応できます。

## 図のアイデア
- images/ch11_nlp_pipeline.png — BERT質問応答パイプラインのデータフロー

## 演習
1. DistilBERTなど軽量モデルで同様のパイプラインを構築し、レイテンシと精度の差をまとめる
2. キャッシュ戦略やバッチング手法を検討し、推論基盤の設計メモを作成する
