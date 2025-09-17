# 第10章 ケーススタディ1：画像分類アプリを作る
### 10.1 課題設定
ここでは、手書き数字を分類するMNISTデータセットを題材にします。MNISTは28×28ピクセルのモノクロ画像と正解ラベル（0～9）が対になったデータで、初心者がディープラーニングを学ぶ定番教材です。まずPyTorchで簡単なCNN（畳み込みニューラルネットワーク）を学習し、そのモデルをONNXに変換するところまでを実践します。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

学習が終わったら検証用データで精度を確認し、`torch.onnx.export`でモデルをONNX形式に変換します。

```python
model.eval()
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy_input, "mnist.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}
)
```

### 10.2 推論アプリの実装
変換した`mnist.onnx`を使って、Web APIを構築します。ここではFlaskを例に取り、画像データを受け取って推論結果を返すAPIを作ります。

```python
from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
session = ort.InferenceSession("mnist.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.post("/predict")
def predict():
    payload = request.get_json()
    pixels = np.array(payload["pixels"], dtype=np.float32).reshape(1, 1, 28, 28)
    result = session.run([output_name], {input_name: pixels})
    probabilities = np.exp(result[0]) / np.sum(np.exp(result[0]))
    label = int(np.argmax(probabilities))
    return jsonify({"label": label, "probabilities": probabilities.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

このAPIに対して、フロントエンドから28×28ピクセルの配列を送信すると予測結果が返ってきます。ブラウザで手書き数字を描画するCanvasを用意し、JavaScriptでピクセル値を取得してAPIに送ると、リアルタイムで推論結果を表示するWebアプリが完成します。

### 10.3 エッジ展開と最適化
完成したアプリをRaspberry Piにデプロイしてみましょう。PythonとONNX RuntimeのARM版をインストールしたら、先ほど作成したFlaskアプリをそのまま動かせます。Piのような小型デバイスではCPU性能が限られるため、Chapter 7で紹介した動的量子化を適用した`mnist_int8.onnx`を使うとレスポンスが向上します。

推論時間とバッテリー消費を比較し、どの程度効果が出るかを測定すると、最適化の重要性を実感できるでしょう。必要に応じてスレッド数や優先度を設定し、他のアプリと共存できるように調整します。

## 図のアイデア
- images/ch10_app_architecture.png — 画像分類アプリのシステム構成図

## 演習
1. Flaskアプリに入力検証とログ出力を追加し、運用時の監視ポイントを整理する
2. Raspberry Piで量子化モデルを実行した結果を計測し、非量子化モデルとの比較表を作成する
