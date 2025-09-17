# 図版プレースホルダー

各章の「図のアイデア」で参照している図版のラフ案をまとめています。実際の画像を作成する際は、本プレースホルダーを差し替えてください。

| ファイル名 | 説明 |
| --- | --- |
| ch01_timeline.png | ONNX誕生までの主要な出来事を年表形式で整理する図。MicrosoftやFacebookによる発表、主要フレームワークの登場を時系列で示す。 |
| ch02_training_inference_flow.png | 学習環境と推論環境の切り替えフロー。トレーニングと推論の分離、ONNXによる橋渡しを矢印とアイコンで表現。 |
| ch03_env_setup.png | 推奨開発環境の構成図。Python仮想環境、主要ライブラリ、ツールチェーンの関係をブロック図で示す。 |
| ch04_graph_overview.png | 典型的なONNXグラフ構造（ノード・テンソル）の概念図。ノードとエッジ、入力・出力テンソルの概念を視覚化。 |
| ch05_conversion_matrix.png | フレームワーク別のONNX変換ツール対応表。縦軸にフレームワーク、横軸に変換ツール、対応状況を○/△/×で表示。 |
| ch06_runtime_flow.png | ONNX Runtimeによる推論処理のフローチャート。入力→前処理→InferenceSession→出力の流れとExecution Provider切り替えを示す。 |
| ch07_optimization_pipeline.png | 最適化と量子化のパイプライン。元モデル→最適化→量子化→デプロイの段階を矢印で表現。 |
| ch08_deployment_targets.png | エッジ／クラウド／ブラウザ各展開先の特徴比較。デバイス例と要件をマトリクス化。 |
| ch09_debug_flow.png | トラブルシューティングの意思決定フロー。エラー発生→チェック項目→原因切り分け→解決策の分岐を示す。 |
| ch10_app_architecture.png | 画像分類アプリの構成図。フロントエンド、APIサーバー、ONNX Runtime、デバイスを含む全体像。 |
| ch11_nlp_pipeline.png | BERTを使った質問応答パイプライン。トークナイザー、ONNX Runtime、後処理までのデータフローを矢印で表現。 |

> 画像生成ツール例: draw.io、Figma、PowerPointなど。PNG以外の形式を利用する場合は、記載を更新してください。
