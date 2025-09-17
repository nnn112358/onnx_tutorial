# 第8章 エッジ・クラウド・ブラウザでの展開
### 8.1 エッジデバイス展開
Raspberry PiやNVIDIA Jetsonのような小型デバイスでもONNX Runtimeは動作します。公式ドキュメントではARM向けのビルド手順が公開されており、事前に最適化されたホイール（`.whl`）をダウンロードしてインストールするだけで利用できます。エッジデバイスではメモリが限られるため、Chapter 7で紹介した最適化や量子化を組み合わせると安定して動作します。

推論プログラムは、小さなPythonスクリプトやC++アプリとしてデバイス上で動かすことが多いです。センサー入力を受け取り、必要な前処理を行い、ONNX Runtimeで推論して結果を表示するという流れをシンプルに保つことがポイントです。また、電源が限られる現場では、推論間隔やバッチサイズを調整して消費電力を抑える工夫も重要になります。

### 8.2 クラウド環境での活用
クラウドではスケーラブルな推論サービスを構築できます。Azure Machine Learning、AWS SageMaker、GCP Vertex AIなど、多くのクラウドプラットフォームがONNXモデルのデプロイに対応しています。一般的な手順は、DockerイメージにONNXモデルと推論コードを含め、クラウド側でスケール設定を行うという流れです。

たとえば、FastAPIで推論APIを作り、Docker化してAzure Container Appsにデプロイすると、アクセス量に応じて自動的にインスタンスが増減します。クラウド環境ではログ収集や監視、バージョン管理が重要になるため、モデルの更新履歴をONNXファイル名やメタデータに埋め込んでおくと運用が楽になります。

### 8.3 ブラウザでの推論
ONNX RuntimeにはWebAssembly版（onnxruntime-web）が提供されており、ブラウザ内でONNXモデルを直接実行できます。サーバーにリクエストを送る必要がないため、プライバシーを重視するアプリやオフライン対応が必要なアプリに適しています。インストールはnpm経由で行います。

```bash
npm install onnxruntime-web
```

JavaScriptでは次のように記述します。

```javascript
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('model.onnx');
const feeds = { input: new ort.Tensor('float32', inputData, [1, 3, 224, 224]) };
const results = await session.run(feeds);
```

ブラウザ推論では、ファイルサイズやユーザー端末の性能がボトルネックになります。モデルを軽量化したり、Service Workerを使ってキャッシュしたりといった工夫が必要です。WebGPU対応が進めばさらに高速なブラウザ推論が可能になると期待されています。

## 図のアイデア
- images/ch08_deployment_targets.png — エッジ/クラウド/ブラウザの展開先マトリクス

## 演習
1. Raspberry Pi向けのONNX Runtimeインストール手順を調べ、セットアップ手順書をまとめる
2. クラウドサービスを1つ選び、ONNX推論APIをデプロイする際の構成図を描く
